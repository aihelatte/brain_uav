"""Short CPU integration tests for the independent V2 TD3 interaction loop."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.models import V2ANNCritic, V2ANNPolicyActor, ZoneSetEncoderConfig
from brain_uav.observations import V2ObservationScales
from brain_uav.trainers import V2ReplayBuffer, V2TD3UpdateEngine, V2TD3UpdateMetrics


def _phase4_api():
    from brain_uav.envs import (
        V2_ENV_SCENARIO_FORMAT,
        V2_ENV_SCENARIO_VERSION,
        V2StaticNoFlyTrajectoryEnv,
    )
    from brain_uav.trainers import V2TD3TrainingLoop, V2TrainingLoopMetrics

    return (
        V2StaticNoFlyTrajectoryEnv,
        V2TD3TrainingLoop,
        V2TrainingLoopMetrics,
        V2_ENV_SCENARIO_FORMAT,
        V2_ENV_SCENARIO_VERSION,
    )


def _config(**overrides):
    values = {
        'world_xy': 100.0,
        'world_z_min': 0.1,
        'world_z_max': 50.0,
        'speed': 1.0,
        'dt': 1.0,
        'max_steps': 100,
        'goal_radius': 0.25,
    }
    values.update(overrides)
    return ScenarioConfig(**values)


def _scenario(zones, *, state=None, goal=None, level='test'):
    _, _, _, scenario_format, scenario_version = _phase4_api()
    return {
        'format': scenario_format,
        'format_version': scenario_version,
        'state': list(state or [0.0, 0.0, 10.0, 0.0, 0.0]),
        'goal': list(goal or [80.0, 0.0, 10.0]),
        'zones': [zone.to_dict() for zone in zones],
        'curriculum_level': level,
    }


def _zones(count):
    return [
        NoFlyZone(f'zone-{index}', Sphere([20.0 + index, 20.0, 10.0], 0.5))
        for index in range(count)
    ]


class TestV2TD3TrainingLoop(unittest.TestCase):
    def make_env(self, zone_count=0, *, config=None, payload=None, seed=11):
        env_type, _, _, _, _ = _phase4_api()
        scenario_config = config or _config()
        return env_type(
            scenario=scenario_config,
            rewards=RewardConfig(),
            seed=seed,
            fixed_scenarios=[payload or _scenario(_zones(zone_count))],
        )

    def make_engine(self, env, *, batch_size=2, replay_capacity=64):
        scales = V2ObservationScales(
            env.scenario.world_xy,
            env.scenario.world_z_min,
            env.scenario.world_z_max,
            env.scenario.gamma_max,
        )
        encoder_config = ZoneSetEncoderConfig(
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            ffn_dim=16,
        )
        action_limit = torch.as_tensor(env.action_space.high, dtype=torch.float32)
        actor = V2ANNPolicyActor(
            scales,
            action_dim=2,
            hidden_dim=8,
            action_limit=action_limit,
            encoder_config=encoder_config,
        )
        critic1 = V2ANNCritic(
            scales,
            action_dim=2,
            hidden_dim=8,
            encoder_config=encoder_config,
        )
        critic2 = V2ANNCritic(
            scales,
            action_dim=2,
            hidden_dim=8,
            encoder_config=encoder_config,
        )
        replay = V2ReplayBuffer(
            replay_capacity,
            action_dim=2,
            zone_storage_capacity=10,
            success_sample_bias=4.0,
            near_goal_sample_bias=2.0,
            success_replay_fraction=0.25,
            success_batch_fraction=0.25,
        )
        return V2TD3UpdateEngine(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            replay=replay,
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.015,
            noise_clip=0.03,
            policy_delay=2,
            batch_size=batch_size,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            actor_freeze_steps=1000,
            terminal_geo_regularization_enabled=True,
            terminal_geo_radius=250.0,
            terminal_geo_lambda=3000.0,
            device='cpu',
        )

    def make_loop(self, env, engine, *, warmup_steps=0, seed=23, exploration_noise=0.0):
        _, loop_type, _, _, _ = _phase4_api()
        return loop_type(
            env,
            engine,
            warmup_steps=warmup_steps,
            exploration_noise=exploration_noise,
            near_goal_radius=250.0,
            terminal_geo_safe_clearance=40.0,
            seed=seed,
        )

    def test_cpu_twenty_step_smoke_populates_replay_and_updates_critic(self):
        torch.manual_seed(101)
        env = self.make_env(0)
        engine = self.make_engine(env, batch_size=4)
        loop = self.make_loop(env, engine, warmup_steps=3)

        metrics = loop.run(20)

        self.assertEqual(metrics.total_steps, 20)
        self.assertEqual(len(engine.replay), 20)
        self.assertGreaterEqual(metrics.update_count, 1)
        self.assertGreaterEqual(engine.critic_update_count, 1)
        self.assertTrue(math.isfinite(metrics.last_update_metrics.critic_loss))
        self.assertTrue(np.isfinite(engine.replay.reward[: len(engine.replay)]).all())

    def test_zero_two_five_six_and_ten_zones_complete_actor_and_update_path(self):
        for count in (0, 2, 5, 6, 10):
            with self.subTest(count=count):
                torch.manual_seed(200 + count)
                env = self.make_env(count)
                engine = self.make_engine(env, batch_size=1)
                loop = self.make_loop(env, engine, warmup_steps=0)
                metrics = loop.run(2)
                self.assertEqual(metrics.update_count, 2)
                self.assertEqual(engine.replay.zone_count[0], count)
                self.assertEqual(engine.replay.next_zone_count[0], count)
                self.assertTrue(math.isfinite(metrics.last_update_metrics.critic_loss))
                self.assertFalse(hasattr(engine.actor, 'max_zones'))

    def test_line_to_goal_safety_is_computed_before_action_and_stored(self):
        blocking = NoFlyZone('blocking', Sphere([5.0, 0.0, 10.0], 1.0))
        env = self.make_env(
            payload=_scenario([blocking], goal=[20.0, 0.0, 10.0])
        )
        engine = self.make_engine(env, batch_size=8)
        loop = self.make_loop(env, engine, warmup_steps=1)
        initial_position = np.array([0.0, 0.0, 10.0], dtype=np.float32)

        with mock.patch.object(
            env,
            'line_to_goal_is_safe',
            wraps=env.line_to_goal_is_safe,
        ) as line_safe:
            loop.run(1)

        pre_action_calls = [
            call
            for call in line_safe.call_args_list
            if call.kwargs.get('clearance') == 40.0
        ]
        self.assertEqual(len(pre_action_calls), 1)
        np.testing.assert_array_equal(pre_action_calls[0].args[0], initial_position)
        self.assertFalse(engine.replay.line_to_goal_safe[0])
        self.assertNotEqual(float(env.state[0]), float(initial_position[0]))

    def test_goal_episode_marks_primary_and_copies_success_replay(self):
        config = _config(speed=25.0, goal_radius=5.0, max_steps=5)
        payload = _scenario(
            [],
            state=[0.0, 0.0, 10.0, 0.0, 0.0],
            goal=[10.0, 0.0, 10.0],
        )
        env = self.make_env(config=config, payload=payload)
        engine = self.make_engine(env, batch_size=8)
        loop = self.make_loop(env, engine, warmup_steps=1)

        metrics = loop.run(1)

        self.assertEqual(metrics.outcomes['goal'], 1)
        self.assertEqual(engine.replay.success_count, 1)
        self.assertEqual(engine.replay.success_size, 1)
        self.assertTrue(engine.replay.success[0])
        self.assertEqual(engine.replay.success_batch_fraction, 0.25)
        self.assertEqual(metrics.replay_success_fraction, 1.0)

    def test_timeout_is_stored_as_done(self):
        env = self.make_env(0, config=_config(max_steps=1))
        engine = self.make_engine(env, batch_size=8)
        loop = self.make_loop(env, engine, warmup_steps=1)

        metrics = loop.run(1)

        self.assertEqual(metrics.outcomes['timeout'], 1)
        self.assertEqual(float(engine.replay.done[0, 0]), 1.0)

    def test_warmup_skips_actor_then_policy_path_calls_actor(self):
        env = self.make_env(0)
        engine = self.make_engine(env, batch_size=2)
        loop = self.make_loop(env, engine, warmup_steps=1)

        with (
            mock.patch.object(
                engine,
                'select_action',
                return_value=np.zeros(2, dtype=np.float32),
            ) as select_action,
            mock.patch.object(
                engine,
                'update_once',
                return_value=V2TD3UpdateMetrics(critic_loss=1.0, critic_updated=True),
            ),
        ):
            loop.run(2)

        self.assertEqual(select_action.call_count, 1)

    def test_same_seed_reproduces_warmup_actions_and_environment_trajectory(self):
        results = []
        for _ in range(2):
            torch.manual_seed(999)
            env = self.make_env(0, seed=31)
            engine = self.make_engine(env, batch_size=16)
            loop = self.make_loop(env, engine, warmup_steps=5, seed=77)
            loop.run(5)
            results.append(
                (
                    engine.replay.action[:5].copy(),
                    np.asarray(env.trajectory).copy(),
                )
            )

        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_near_goal_uses_endpoint_segment_and_goal_hit_fields(self):
        config = _config(speed=1.0, max_steps=5)
        payload = _scenario([], goal=[10.0, 0.0, 10.0])
        env = self.make_env(config=config, payload=payload)
        engine = self.make_engine(env, batch_size=8)
        _, loop_type, _, _, _ = _phase4_api()
        loop = loop_type(
            env,
            engine,
            warmup_steps=1,
            near_goal_radius=20.0,
            terminal_geo_safe_clearance=40.0,
            seed=2,
        )

        loop.run(1)

        self.assertTrue(engine.replay.near_goal[0])


if __name__ == '__main__':
    unittest.main()
