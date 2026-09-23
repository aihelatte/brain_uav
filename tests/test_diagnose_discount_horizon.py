"""Smoke test for the offline discount-horizon diagnostic script (B1)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import (
    V2_ENV_SCENARIO_FORMAT,
    V2_ENV_SCENARIO_VERSION,
    V2StaticNoFlyTrajectoryEnv,
)
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.scripts.diagnose_discount_horizon import (
    _nearest_zone_avoidance_action,
    load_actor_and_critics,
    run_discount_horizon_diagnostic,
)
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig,
    build_v2_periodic_snapshot,
    save_v2_periodic_snapshot,
)
from test_v2_formal_training import _engine, _scenario_payload


def _scenario_with_zone(y_offset: float) -> dict:
    """A UAV at the origin heading toward +x, with one zone ahead at y_offset."""

    zone = NoFlyZone('z0', Sphere([500.0, y_offset, 100.0], 1.0)).to_dict()
    return {
        'format': V2_ENV_SCENARIO_FORMAT,
        'format_version': V2_ENV_SCENARIO_VERSION,
        'state': [0.0, 0.0, 100.0, 0.0, 0.0],
        'goal': [1000.0, 0.0, 100.0],
        'zones': [zone],
        'curriculum_level': 'easy',
    }


def _scenario_without_zone() -> dict:
    return {
        'format': V2_ENV_SCENARIO_FORMAT,
        'format_version': V2_ENV_SCENARIO_VERSION,
        'state': [0.0, 0.0, 100.0, 0.0, 0.0],
        'goal': [1000.0, 0.0, 100.0],
        'zones': [],
        'curriculum_level': 'easy',
    }


class TestDiagnoseDiscountHorizon(unittest.TestCase):
    def _write_snapshot(self, directory: Path, scenario: ScenarioConfig) -> Path:
        engine = _engine(scenario)
        config = V2FormalTrainingConfig(stage='easy', max_steps=1)
        payload = build_v2_periodic_snapshot(
            engine,
            config,
            stage_steps=1,
            scenario=scenario,
            rewards=RewardConfig(),
            uav_collision_radius=0.0,
            seed_manifest={'base_seed': 7},
            initialization_source={'kind': 'v2_bc_best', 'path': 'bc.pt'},
        )
        checkpoint_path = directory / 'snapshot.pt'
        save_v2_periodic_snapshot(checkpoint_path, payload)
        return checkpoint_path

    def test_load_actor_and_critics_reconstructs_online_and_target_networks(self) -> None:
        scenario = ScenarioConfig(max_steps=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = self._write_snapshot(Path(directory), scenario)
            models = load_actor_and_critics(
                checkpoint_path, model_type='ann', device='cpu',
            )
        self.assertEqual(
            set(models),
            {
                'actor', 'actor_target', 'critic1', 'critic2',
                'critic1_target', 'critic2_target', 'scenario', 'rewards',
                'uav_collision_radius', 'gamma', 'action_high',
            },
        )
        self.assertFalse(models['actor'].training)
        self.assertEqual(models['action_high'].shape, (2,))

    def test_runs_end_to_end_and_reports_crossover_and_proxy_replay_stats(self) -> None:
        scenario = ScenarioConfig(max_steps=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = self._write_snapshot(Path(directory), scenario)
            result = run_discount_horizon_diagnostic(
                checkpoint=checkpoint_path,
                scenario_payload=_scenario_payload(2),
                model_type='ann',
                device='cpu',
            )
        self.assertIn(
            result['outcome'],
            ('goal', 'ground', 'boundary', 'collision', 'timeout'),
        )
        self.assertGreater(result['episode_length'], 0)
        self.assertEqual(len(result['per_step']), result['episode_length'])
        for step in result['per_step']:
            self.assertAlmostEqual(
                step['q_avoid_minus_q_policy'],
                step['q_avoid'] - step['q_policy'],
                places=5,
            )
        self.assertIn('proxy_batch_occupancy_fraction', result)
        self.assertGreaterEqual(result['proxy_batch_occupancy_fraction'], 0.0)
        self.assertLessEqual(result['proxy_batch_occupancy_fraction'], 1.0)
        self.assertEqual(
            result['td_error_near_collision']['count'],
            result['near_collision_transition_count'],
        )
        self.assertIn('measurement_note', result)
        self.assertIn('replay buffer', result['measurement_note'])

    def _assert_avoidance_turns_away_from_zone(self, y_offset: float) -> None:
        # Regression guard for the sign convention in
        # _nearest_zone_avoidance_action: a wrong sign would silently steer
        # the "avoidance" action toward the zone instead of away from it,
        # inverting every H2 conclusion the script produces without ever
        # raising an error.
        scenario = ScenarioConfig()
        rewards = RewardConfig()
        action_high = np.array(
            [scenario.delta_gamma_max, scenario.delta_psi_max], dtype=np.float32,
        )
        payload = _scenario_with_zone(y_offset)
        zone_center_xy = np.array([500.0, y_offset], dtype=np.float64)

        avoid_env = V2StaticNoFlyTrajectoryEnv(scenario, rewards)
        avoid_env.reset(options={'scenario': payload})
        avoid_action = _nearest_zone_avoidance_action(avoid_env, action_high)
        self.assertNotEqual(float(avoid_action[1]), 0.0)

        straight_env = V2StaticNoFlyTrajectoryEnv(scenario, rewards)
        straight_env.reset(options={'scenario': payload})

        for _ in range(5):
            avoid_env.step(avoid_action)
            straight_env.step(np.zeros(2, dtype=np.float32))

        avoid_distance = float(
            np.linalg.norm(avoid_env.state[:2].astype(np.float64) - zone_center_xy)
        )
        straight_distance = float(
            np.linalg.norm(straight_env.state[:2].astype(np.float64) - zone_center_xy)
        )
        self.assertGreater(avoid_distance, straight_distance)

    def test_avoidance_action_turns_away_from_zone_ahead_and_left(self) -> None:
        self._assert_avoidance_turns_away_from_zone(50.0)

    def test_avoidance_action_turns_away_from_zone_ahead_and_right(self) -> None:
        self._assert_avoidance_turns_away_from_zone(-50.0)

    def test_avoidance_action_is_zero_when_no_zones_are_present(self) -> None:
        scenario = ScenarioConfig()
        rewards = RewardConfig()
        action_high = np.array(
            [scenario.delta_gamma_max, scenario.delta_psi_max], dtype=np.float32,
        )
        env = V2StaticNoFlyTrajectoryEnv(scenario, rewards)
        env.reset(options={'scenario': _scenario_without_zone()})

        avoid_action = _nearest_zone_avoidance_action(env, action_high)

        self.assertTrue(np.array_equal(avoid_action, np.zeros_like(action_high)))


if __name__ == '__main__':
    unittest.main()
