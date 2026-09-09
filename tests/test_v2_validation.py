"""Tests for deterministic, strict V2 fixed-validation pools."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import (
    V2_ENV_SCENARIO_FORMAT,
    V2_ENV_SCENARIO_VERSION,
)
from brain_uav.envs.v2_scenario_generator import (
    V2_SCENARIO_GENERATOR_NAME,
    V2_SCENARIO_GENERATOR_VERSION,
)
from brain_uav.models import V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.trainers.v2_validation import (
    V2_VALIDATION_POOL_FORMAT,
    V2_VALIDATION_POOL_VERSION,
    V2ValidationPool,
    derive_validation_stage_seed,
    evaluate_v2_fixed_validation,
    generate_v2_validation_pool,
    load_v2_validation_pool,
    save_v2_validation_pool,
    validation_passes,
)


def _metadata(stage: str, scenario_seed: int, zone_count: int = 0) -> dict:
    return {
        'generator': V2_SCENARIO_GENERATOR_NAME,
        'generator_version': V2_SCENARIO_GENERATOR_VERSION,
        'scenario_seed': scenario_seed,
        'requested_curriculum_level': stage,
        'effective_curriculum_level': stage,
        'requested_zone_count': zone_count,
        'effective_zone_count': zone_count,
        'shape_counts': {
            'sphere': 0,
            'ellipsoid': 0,
            'box': 0,
            'triangular_pyramid': 0,
            'quadrangular_pyramid': 0,
        },
        'requested_shape_types': [],
        'requested_ground_contact': [],
        'requested_reference_scales': [],
        'overlap_allowed': stage == 'hard',
        'aabb_overlap_pair_count': 0,
        'direct_path_blocker_count': 0,
        'feasibility_check': 'direct_safe_corridor',
        'feasibility_passed': True,
        'feasibility_examined_nodes': 0,
        'feasibility_edge_checks': 0,
        'generation_attempts': 1,
        'rejection_counts': {},
    }


def _payload(stage: str, scenario_seed: int, *, goal_x: float = 1.0) -> dict:
    return {
        'format': V2_ENV_SCENARIO_FORMAT,
        'format_version': V2_ENV_SCENARIO_VERSION,
        'state': [0.0, 0.0, 100.0, 0.0, 0.0],
        'goal': [goal_x, 0.0, 100.0],
        'zones': [],
        'curriculum_level': stage,
        'metadata': _metadata(stage, scenario_seed),
    }


def _pool(stage: str, scenario: ScenarioConfig, goals: tuple[float, ...]) -> V2ValidationPool:
    scenarios = tuple(
        {
            'scenario_id': f'{stage}_{index:04d}',
            'sequence_index': index,
            'scenario_seed': 1000 + index,
            'payload': _payload(stage, 1000 + index, goal_x=goal_x),
        }
        for index, goal_x in enumerate(goals)
    )
    return V2ValidationPool(
        curriculum_level=stage,
        master_seed=20260904,
        stage_seed=derive_validation_stage_seed(20260904, stage),
        scenario_config=asdict(scenario),
        uav_collision_radius=0.0,
        scenarios=scenarios,
    )


class TestV2Validation(unittest.TestCase):
    def setUp(self):
        self.scenario = ScenarioConfig(max_steps=1)

    def test_pool_round_trip_is_strict_and_binds_stage_and_config(self):
        pool = _pool('easy', self.scenario, (1.0, 100.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'easy.json'
            save_v2_validation_pool(path, pool)
            raw = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(raw['format'], V2_VALIDATION_POOL_FORMAT)
            self.assertEqual(raw['format_version'], V2_VALIDATION_POOL_VERSION)
            loaded = load_v2_validation_pool(
                path,
                expected_level='easy',
                expected_scenario=self.scenario,
                expected_count=2,
                expected_uav_collision_radius=0.0,
            )
        self.assertEqual(loaded.to_dict(), pool.to_dict())
        self.assertEqual(loaded.content_digest, pool.content_digest)

    def test_generation_is_deterministic_and_stage_pools_are_distinct(self):
        scenario = ScenarioConfig(scenario_max_sampling_attempts=80)
        first = generate_v2_validation_pool(
            scenario,
            'easy',
            scenario_count=2,
            master_seed=20260904,
        )
        second = generate_v2_validation_pool(
            scenario,
            'easy',
            scenario_count=2,
            master_seed=20260904,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(
            first.stage_seed,
            derive_validation_stage_seed(20260904, 'medium'),
        )
        self.assertTrue(all(
            item['payload']['curriculum_level'] == 'easy'
            for item in first.scenarios
        ))

    def test_pool_requires_derived_seed_and_loader_checks_expected_seeds(self):
        expected_master_seed = 20260904
        expected_stage_seed = derive_validation_stage_seed(expected_master_seed, 'easy')
        pool = _pool('easy', self.scenario, (1.0, 100.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'easy.json'
            save_v2_validation_pool(path, pool)
            loaded = load_v2_validation_pool(
                path,
                expected_level='easy',
                expected_master_seed=expected_master_seed,
                expected_stage_seed=expected_stage_seed,
            )
            self.assertEqual(loaded.master_seed, expected_master_seed)
            self.assertEqual(loaded.stage_seed, expected_stage_seed)

            with self.assertRaisesRegex(ValueError, 'master_seed'):
                load_v2_validation_pool(
                    path,
                    expected_level='easy',
                    expected_master_seed=expected_master_seed + 1,
                )
            with self.assertRaisesRegex(ValueError, 'stage_seed'):
                load_v2_validation_pool(
                    path,
                    expected_level='easy',
                    expected_stage_seed=expected_stage_seed + 1,
                )

            raw = json.loads(path.read_text(encoding='utf-8'))
            raw['stage_seed'] = expected_stage_seed + 1
            path.write_text(json.dumps(raw, allow_nan=False), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'derived'):
                load_v2_validation_pool(path, expected_level='easy')

        with self.assertRaisesRegex(ValueError, 'derived'):
            V2ValidationPool(
                curriculum_level='easy',
                master_seed=expected_master_seed,
                stage_seed=expected_stage_seed + 1,
                scenario_config=asdict(self.scenario),
                uav_collision_radius=0.0,
                scenarios=pool.scenarios,
            )

    def test_pool_rejects_wrong_stage_config_unknown_fields_nan_and_duplicate_ids(self):
        pool = _pool('medium', self.scenario, (1.0, 100.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'medium.json'
            save_v2_validation_pool(path, pool)
            with self.assertRaisesRegex(ValueError, 'curriculum'):
                load_v2_validation_pool(path, expected_level='hard')
            with self.assertRaisesRegex(ValueError, 'ScenarioConfig'):
                load_v2_validation_pool(
                    path,
                    expected_level='medium',
                    expected_scenario=ScenarioConfig(max_steps=2),
                )

            raw = json.loads(path.read_text(encoding='utf-8'))
            raw['unknown'] = 1
            path.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fields'):
                load_v2_validation_pool(path, expected_level='medium')

            raw.pop('unknown')
            raw['scenarios'][1]['scenario_id'] = raw['scenarios'][0]['scenario_id']
            path.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_v2_validation_pool(path, expected_level='medium')

            path.write_text('{"format": NaN}', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_v2_validation_pool(path, expected_level='medium')

    def test_expert_easy_pool_format_cannot_impersonate_validation_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pool.json'
            path.write_text(
                json.dumps({'format': 'v2_easy_expert_scenario_pool', 'format_version': 3}),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'format'):
                load_v2_validation_pool(path, expected_level='easy')

    def test_validation_threshold_is_only_failure_count(self):
        self.assertTrue(validation_passes(6, max_failures=6))
        self.assertFalse(validation_passes(7, max_failures=6))
        with self.assertRaises(ValueError):
            validation_passes(-1, max_failures=6)

    def test_fixed_validation_is_noiseless_restores_actor_mode_and_reports_outcomes(self):
        scales = V2ObservationScales(
            self.scenario.world_xy,
            self.scenario.world_z_min,
            self.scenario.world_z_max,
            self.scenario.gamma_max,
        )
        torch.manual_seed(17)
        actor = V2ANNPolicyActor(
            scales,
            2,
            8,
            torch.tensor(
                [self.scenario.delta_gamma_max, self.scenario.delta_psi_max],
                dtype=torch.float32,
            ),
            uav_radius=0.0,
        )
        for parameter in actor.parameters():
            parameter.data.zero_()
        actor.train()
        state_before = {
            name: value.detach().clone()
            for name, value in actor.state_dict().items()
        }
        pool = _pool('easy', self.scenario, (1.0, 100.0))

        result = evaluate_v2_fixed_validation(
            actor,
            pool,
            RewardConfig(),
            max_failures=1,
            device='cpu',
        )

        self.assertTrue(actor.training)
        for name, value in actor.state_dict().items():
            self.assertTrue(torch.equal(value, state_before[name]))
        self.assertTrue(result.passed)
        self.assertEqual(result.goal_count, 1)
        self.assertEqual(result.failure_count, 1)
        self.assertEqual(result.outcome_counts['timeout'], 1)
        self.assertEqual(tuple(item['outcome'] for item in result.scenarios), ('goal', 'timeout'))
        self.assertTrue(math.isfinite(result.scenarios[0]['episode_return']))

    def test_unindexed_cuda_request_uses_current_nonzero_device(self):
        scales = V2ObservationScales(
            self.scenario.world_xy,
            self.scenario.world_z_min,
            self.scenario.world_z_max,
            self.scenario.gamma_max,
        )
        actor = V2ANNPolicyActor(
            scales,
            2,
            8,
            torch.tensor(
                [self.scenario.delta_gamma_max, self.scenario.delta_psi_max],
                dtype=torch.float32,
            ),
        )
        pool = _pool('easy', self.scenario, (1.0,))
        reached_environment = RuntimeError('device check reached environment')
        with mock.patch.object(
            actor,
            'parameters',
            return_value=iter([SimpleNamespace(device=torch.device('cuda:3'))]),
        ), mock.patch(
            'brain_uav.trainers.v2_validation.torch.cuda.is_available',
            return_value=True,
        ), mock.patch(
            'brain_uav.trainers.v2_validation.torch.cuda.current_device',
            return_value=3,
        ) as current_device, mock.patch(
            'brain_uav.trainers.v2_validation.V2StaticNoFlyTrajectoryEnv',
            side_effect=reached_environment,
        ) as environment:
            with self.assertRaisesRegex(RuntimeError, 'device check reached environment'):
                evaluate_v2_fixed_validation(
                    actor,
                    pool,
                    RewardConfig(),
                    device='cuda',
                )
        current_device.assert_called_once_with()
        environment.assert_called_once()

    def test_explicit_cuda_index_is_not_replaced_by_current_device(self):
        scales = V2ObservationScales(
            self.scenario.world_xy,
            self.scenario.world_z_min,
            self.scenario.world_z_max,
            self.scenario.gamma_max,
        )
        actor = V2ANNPolicyActor(
            scales,
            2,
            8,
            torch.tensor(
                [self.scenario.delta_gamma_max, self.scenario.delta_psi_max],
                dtype=torch.float32,
            ),
        )
        pool = _pool('easy', self.scenario, (1.0,))
        reached_environment = RuntimeError('device check reached environment')
        with mock.patch.object(
            actor,
            'parameters',
            return_value=iter([SimpleNamespace(device=torch.device('cuda:2'))]),
        ), mock.patch(
            'brain_uav.trainers.v2_validation.torch.cuda.current_device',
            return_value=3,
        ) as current_device, mock.patch(
            'brain_uav.trainers.v2_validation.V2StaticNoFlyTrajectoryEnv',
            side_effect=reached_environment,
        ) as environment:
            with self.assertRaisesRegex(RuntimeError, 'device check reached environment'):
                evaluate_v2_fixed_validation(
                    actor,
                    pool,
                    RewardConfig(),
                    device='cuda:2',
                )
        current_device.assert_not_called()
        environment.assert_called_once()

    def test_explicit_cuda_index_mismatch_is_still_rejected(self):
        scales = V2ObservationScales(
            self.scenario.world_xy,
            self.scenario.world_z_min,
            self.scenario.world_z_max,
            self.scenario.gamma_max,
        )
        actor = V2ANNPolicyActor(
            scales,
            2,
            8,
            torch.tensor(
                [self.scenario.delta_gamma_max, self.scenario.delta_psi_max],
                dtype=torch.float32,
            ),
        )
        pool = _pool('easy', self.scenario, (1.0,))
        with mock.patch.object(
            actor,
            'parameters',
            return_value=iter([SimpleNamespace(device=torch.device('cuda:2'))]),
        ), mock.patch(
            'brain_uav.trainers.v2_validation.V2StaticNoFlyTrajectoryEnv',
        ) as environment:
            with self.assertRaisesRegex(ValueError, 'requested validation device'):
                evaluate_v2_fixed_validation(
                    actor,
                    pool,
                    RewardConfig(),
                    device='cuda:3',
                )
        environment.assert_not_called()

    def test_fixed_validation_accepts_snn_and_leaves_no_lif_state(self):
        pool = _pool('easy', self.scenario, (1.0,))
        scales = V2ObservationScales(
            self.scenario.world_xy,
            self.scenario.world_z_min,
            self.scenario.world_z_max,
            self.scenario.gamma_max,
        )
        actor = V2SNNPolicyActor(
            scales,
            2,
            8,
            torch.tensor(
                [self.scenario.delta_gamma_max, self.scenario.delta_psi_max],
                dtype=torch.float32,
            ),
            time_window=2,
        )
        result = evaluate_v2_fixed_validation(
            actor,
            pool,
            RewardConfig(),
            max_failures=0,
            device='cpu',
        )
        self.assertTrue(result.passed)
        self.assertEqual(actor.snn_head.lif1.v, 0.0)
        self.assertEqual(actor.snn_head.lif2.v, 0.0)


if __name__ == '__main__':
    unittest.main()
