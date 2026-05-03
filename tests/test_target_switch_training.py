"""Tests for target-switch curriculum training."""

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from brain_uav.config import ExperimentConfig, RewardConfig, ScenarioConfig
from brain_uav.envs import TargetSwitchTrajectoryEnv
from brain_uav.scripts.train_target_switch_td3 import (
    _best_window_score,
    _is_better_window,
    _save_target_switch_checkpoint,
    _target_switch_checkpoint_paths,
    build_parser,
)
from brain_uav.target_switch import TargetSwitchConfig, sample_valid_new_goal
from brain_uav.trainers import TD3Trainer
from brain_uav.utils.io import load_checkpoint


def _small_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        target_distance=30.0,
        world_xy=100.0,
        world_z_max=100.0,
        max_steps=3,
        no_fly_radius_range=(5.0, 8.0),
        no_fly_radius_curriculum={
            'easy': (5.0, 8.0),
            'easy_two_zone': (5.0, 8.0),
            'medium': (5.0, 8.0),
            'hard': (5.0, 8.0),
            'benchmark': (5.0, 8.0),
        },
        start_zone_clearance=2.0,
    )


def _switch_config() -> TargetSwitchConfig:
    return TargetSwitchConfig(
        level='target_switch_easy',
        switch_step_ratio_range=(0.34, 0.34),
        switch_angle_deg_range=(0.0, 10.0),
        new_goal_distance_ratio_range=(0.50, 0.60),
        new_goal_z_ratio_range=(0.20, 0.25),
        max_height_gap_ratio=0.20,
        lateral_offset_ratio=0.02,
    )


def _fixed_payload() -> dict:
    return {
        'state': [0.0, 0.0, 20.0, 0.0, 0.0],
        'goal': [80.0, 0.0, 20.0],
        'zones': [],
        'curriculum_level': 'hard',
    }


class TestTargetSwitchTraining(unittest.TestCase):
    def make_env(self) -> TargetSwitchTrajectoryEnv:
        return TargetSwitchTrajectoryEnv(
            scenario=_small_scenario(),
            rewards=RewardConfig(),
            target_switch=_switch_config(),
            seed=5,
            fixed_scenarios=[_fixed_payload()],
        )

    def make_trainer(self) -> TD3Trainer:
        env = self.make_env()
        obs, _ = env.reset(seed=5)
        actor = _TinyActor(obs.shape[0], env.action_space.shape[0])
        critic1 = _TinyCritic(obs.shape[0], env.action_space.shape[0])
        critic2 = _TinyCritic(obs.shape[0], env.action_space.shape[0])
        return TD3Trainer(
            env=env,
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.01,
            noise_clip=0.02,
            policy_delay=2,
            replay_size=32,
            batch_size=2,
            warmup_steps=10,
            exploration_noise=0.01,
            success_sample_bias=1.0,
        )

    def test_reset_starts_with_old_goal(self):
        env = self.make_env()
        env.reset(seed=5)

        np.testing.assert_allclose(env.goal, np.asarray(_fixed_payload()['goal'], dtype=np.float32))
        np.testing.assert_allclose(env.old_goal, env.goal)
        self.assertFalse(env.switched)

    def test_goal_switches_at_switch_step_and_resets_leg_timer(self):
        env = self.make_env()
        env.reset(seed=5)

        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(env.switched)
        self.assertEqual(env.steps, 0)
        self.assertTrue(info['switch_transition'])
        self.assertIsNotNone(env.new_goal)
        np.testing.assert_allclose(env.goal, env.new_goal)

    def test_post_switch_max_steps_refreshes_timeout_budget(self):
        env = self.make_env()
        env.reset(seed=5)
        env.step(np.zeros(2, dtype=np.float32))

        outcome = 'running'
        for _ in range(env.scenario.max_steps):
            _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
            outcome = info['outcome']
            if terminated or truncated:
                break

        self.assertEqual(outcome, 'timeout')
        self.assertEqual(env.steps, env.scenario.max_steps)

    def test_switch_resets_progress_baseline_to_new_goal(self):
        env = self.make_env()
        env.reset(seed=5)
        env.step(np.zeros(2, dtype=np.float32))
        assert env.new_goal is not None

        expected_distance = float(np.linalg.norm(env.state[:3] - env.new_goal))
        self.assertEqual(env.recent_progress, [])
        self.assertAlmostEqual(env.best_goal_distance_so_far, expected_distance)
        self.assertAlmostEqual(env.last_segment_goal_distance, expected_distance)

    def test_training_new_goal_sampling_is_valid(self):
        env = self.make_env()
        env.reset(seed=5)
        goal = sample_valid_new_goal(env, np.random.default_rng(5), _switch_config())

        self.assertLessEqual(abs(float(goal[0])), env.scenario.world_xy)
        self.assertLessEqual(abs(float(goal[1])), env.scenario.world_xy)
        self.assertGreater(float(goal[2]), env.scenario.world_z_min)
        self.assertLess(float(goal[2]), env.scenario.world_z_max)

    def test_ceiling_penalty_is_negative_near_ceiling_when_climbing(self):
        env = self.make_env()
        env.reset(seed=5)
        env.state[2] = env.scenario.world_z_max - 5.0
        env.state[3] = 0.2

        self.assertLess(env._ceiling_safety_penalty(), 0.0)

    def test_no_old_goal_inertia_penalty_exists(self):
        cfg = _switch_config()

        self.assertFalse(hasattr(cfg, 'old_goal_inertia_penalty'))
        self.assertFalse(hasattr(cfg, 'old_goal_inertia_penalty_weight'))

    def test_trainer_skips_switch_transition_replay_write(self):
        trainer = self.make_trainer()

        trainer.train(total_timesteps=1, verbose=False, summary_every_episodes=0)

        self.assertEqual(len(trainer.replay), 0)

    def test_training_parser_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.timesteps, 500000)
        self.assertEqual(args.summary_every_episodes, 50)

    def test_best_window_score_prefers_goal_rate_and_safety(self):
        current = {'goal_rate': 0.8, 'goal_count': 40, 'ground_count': 2, 'boundary_count': 0, 'collision_count': 0}
        better_rate = {'goal_rate': 0.9, 'goal_count': 39, 'ground_count': 20}
        better_ground = dict(current)
        better_ground['ground_count'] = 1

        self.assertTrue(_is_better_window(better_rate, current))
        self.assertGreater(_best_window_score(better_ground), _best_window_score(current))

    def test_best_window_score_prefers_smaller_final_distance(self):
        current = {
            'goal_rate': 0.8,
            'goal_count': 40,
            'ground_count': 0,
            'boundary_count': 0,
            'collision_count': 0,
            'timeout_count': 0,
            'pre_switch_done_count': 0,
            'avg_final_to_new_distance': 120.0,
            'avg_return': 10.0,
        }
        candidate = dict(current)
        candidate['avg_final_to_new_distance'] = 80.0

        self.assertTrue(_is_better_window(candidate, current))

    def test_checkpoint_paths_keep_best_final_and_compat_names(self):
        trainer = self.make_trainer()
        cfg = ExperimentConfig(scenario=_small_scenario(), rewards=RewardConfig())
        args = argparse.Namespace(model='snn', target_switch_level='target_switch_hard', checkpoint=Path('source.pt'))
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            paths = _target_switch_checkpoint_paths(run_dir, 'snn', 'target_switch_hard')
            best_window = {'episode_start': 1, 'episode_end': 50, 'goal_rate': 0.5}
            for path, kind in ((paths['best'], 'best'), (paths['final'], 'final'), (paths['compat'], 'final')):
                _save_target_switch_checkpoint(
                    path,
                    args=args,
                    trainer=trainer,
                    cfg=cfg,
                    target_switch_cfg=_switch_config(),
                    metrics_dict={'kind': kind},
                    summary={'best_checkpoint': str(paths['best'])},
                    timestamp='20260503_000000',
                    console_log=run_dir / 'console.log',
                    best_window=best_window,
                    checkpoint_kind=kind,
                )

            self.assertTrue(paths['best'].exists())
            self.assertTrue(paths['final'].exists())
            self.assertTrue(paths['compat'].exists())
            self.assertEqual(paths['final'].name, 'td3_snn_target_switch_hard_final.pt')
            self.assertEqual(paths['compat'].name, 'td3_snn_target_switch_hard.pt')
            self.assertEqual(load_checkpoint(paths['best'])['best_window']['goal_rate'], 0.5)


class _TinyActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        del obs_dim
        self.bias = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.bias).expand(obs.shape[0], -1)


class _TinyCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        del obs_dim, action_dim
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return self.weight.expand(obs.shape[0], 1)


if __name__ == '__main__':
    unittest.main()
