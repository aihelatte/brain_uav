"""Targeted reward-option, geometry reuse and artifact compatibility checks."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.envs.static_no_fly_env_runtime import StaticNoFlyTrajectoryEnv
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig, V2FormalTrainingResult, build_v2_formal_checkpoint,
    build_v2_periodic_snapshot, load_v2_formal_checkpoint, load_v2_periodic_snapshot,
    prepare_v2_stage_initialization,
)
from test_diagnose_v2_policy import write_inputs
from test_run_v2_td3_curriculum import _prepared_initialization
from test_v2_formal_training import _engine
from test_v2_static_no_fly_env import _config, _scenario


NEW_FIELDS = ('breakthrough_clearance_gate_enabled', 'descent_three_band_enabled',
              'descent_low_height', 'descent_high_height', 'descent_transition_factor')


class TestV2RewardOptions(unittest.TestCase):
    def env(self, **options):
        env = V2StaticNoFlyTrajectoryEnv(_config(max_steps=30), RewardConfig(**options),
                                       uav_collision_radius=0.5)
        env.enable_reward_diagnostics()
        return env

    def test_disabled_options_preserve_captured_baseline_parts(self):
        env = V2StaticNoFlyTrajectoryEnv(_config(max_steps=30), RewardConfig())
        env.enable_reward_diagnostics()
        env.reset(options={'scenario': _scenario([0, 0, 10, 0, 0], [60, 0, 10],
            [NoFlyZone('nearby', Sphere([20, 10, 10], 2))])})
        for _ in range(10):
            _, reward, _, _, info = env.step([0, 0])
        self.assertEqual(info['reward_components'], {
            'distance_progress': 20.0, 'breakthrough': 10.0, 'terminal_direction': 45.0,
            'terminal_radial_tangential': 45.0, 'step': -2.5, 'action_magnitude': -0.0,
            'action_change': -0.0, 'zone_warning': -347.35519497414987,
            'boundary_warning': -0.0, 'ground_warning': -0.0, 'descent_trend': -0.0,
            'inefficiency': -0.0, 'termination': 0.0,
        })
        self.assertEqual(reward, -229.85519497414987)
        v1 = StaticNoFlyTrajectoryEnv(_config(), RewardConfig())
        previous = np.array([0, 0, 51, -0.6, 0.])
        current = np.array([0, 0, 50, -0.6, 0.])
        self.assertAlmostEqual(v1._descent_trend_penalty(previous, current), 42.)

    def test_clearance_gate_direct_reward_calls_and_original_conditions(self):
        env = self.env(breakthrough_clearance_gate_enabled=True)
        zone = NoFlyZone('zone', Sphere([0, 10, 10], 2), safety_margin=1.)
        env.reset(options={'scenario': _scenario([0, 0, 10, 0, 0], [60, 0, 10], [zone])})
        env.recent_progress = [1.] * 10
        for previous, current, expected in ((0., 2., 10.), (2., 0., 0.),
                                             (-2., 2., 0.), (2., 2.000001, 0.)):
            with self.subTest(previous=previous, current=current):
                env.state = np.array([current, 0, 10, 0, 0], dtype=np.float32)
                before = np.array([previous, 0, 10, 0, 0], dtype=np.float32)
                env._compute_reward(before, np.zeros(2), 60, 59, np.zeros(2), 'running', 60)
                self.assertEqual(env.last_reward_components['breakthrough'], expected)
                self.assertEqual(env.last_reward_components['distance_progress'], 20.)
        for outcome, best, progress in (('collision', 60, [1.] * 10),
                ('ground', 60, [1.] * 10), ('boundary', 60, [1.] * 10),
                ('running', 58, [1.] * 10), ('running', 60, [1.]),
                ('running', 60, [0.] * 10)):
            env.state = np.array([2, 0, 10, 0, 0], dtype=np.float32)
            env.recent_progress = progress
            env._compute_reward(np.array([0, 0, 10, 0, 0]), np.zeros(2), 60, 59,
                                np.zeros(2), outcome, best)
            self.assertEqual(env.last_reward_components['breakthrough'], 0.)
        env.reset(options={'scenario': _scenario([0, 0, 10, 0, 0], [60, 0, 10], [])})
        env.recent_progress = [1.] * 10
        env._compute_reward(env.state.copy(), np.zeros(2), 60, 59, np.zeros(2), 'running', 60)
        self.assertEqual(env.last_reward_components['breakthrough'], 0.)

    def test_gate_rejects_far_zones_and_invalidates_previous_geometry(self):
        env = self.env(breakthrough_clearance_gate_enabled=True)
        payload = _scenario([0, 0, 10, 0, 0], [60, 0, 10],
                            [NoFlyZone('zone', Sphere([0, 10, 10], 2), safety_margin=1.)])
        env.reset(options={'scenario': payload})
        self.assertAlmostEqual(env.min_zone_clearance(), 6.5)  # shape - margin - radius
        env.recent_progress = [1.] * 10
        previous = env.state.copy()
        env.state[0] = 2
        # Changing radius invalidates a reset's clearance even at the same previous position.
        env.uav_collision_radius = 1.5
        with mock.patch.object(env, '_point_clearances', wraps=env._point_clearances) as query:
            env._compute_reward(previous, np.zeros(2), 60, 59, np.zeros(2), 'running', 60)
            self.assertEqual(query.call_count, 2)  # current + uncached previous
        self.assertEqual(env.last_reward_components['breakthrough'], 10.)
        env.zones = [NoFlyZone('replaced', Sphere([1000, 10, 10], 2))]
        env._compute_reward(previous, np.zeros(2), 60, 59, np.zeros(2), 'running', 60)
        self.assertEqual(env.last_reward_components['breakthrough'], 0.)
        self.assertFalse(hasattr(env, '_breakthrough_context'))

    def test_step_geometry_reuse_reset_and_four_independent_combinations(self):
        parts = {}
        for gate in (False, True):
            for descent in (False, True):
                env = self.env(breakthrough_clearance_gate_enabled=gate,
                               descent_three_band_enabled=descent)
                zone = NoFlyZone('zone', Sphere([20, 10, 10], 2))
                env.reset(options={'scenario': _scenario([0, 0, 40, -.6, 0], [60, 0, 40], [zone])})
                env.recent_progress = [1.] * 10
                with mock.patch.object(env, '_point_clearances', wraps=env._point_clearances) as queries:
                    _, reward, _, _, info = env.step([0, 0])
                    self.assertEqual(queries.call_count, 0 if gate else 1)
                self.assertAlmostEqual(math.fsum(info['reward_components'].values()), reward)
                parts[gate, descent] = info['reward_components']
                # A reset replaces the stored position/zone geometry, including empty scenes.
                for zones in ([], [NoFlyZone('new', Sphere([-20, 10, 10], 2))]):
                    env.reset(options={'scenario': _scenario([0, 0, 10, 0, 0], [60, 0, 10], zones)})
                    env.recent_progress = [1.] * 10
                    _, _, _, _, info = env.step([0, 0])
                    if gate:
                        self.assertEqual(info['reward_components']['breakthrough'], 10. if zones else 0.)
        for name in parts[False, False]:
            if name == 'breakthrough':
                self.assertEqual(parts[True, False][name], parts[True, True][name])
                self.assertEqual(parts[False, False][name], parts[False, True][name])
            elif name == 'descent_trend':
                self.assertEqual(parts[False, True][name], parts[True, True][name])
                self.assertEqual(parts[False, False][name], parts[True, False][name])
            else:
                self.assertEqual(len({value[name] for value in parts.values()}), 1)
        self.assertNotEqual(parts[False, False]['descent_trend'], parts[False, True]['descent_trend'])
        self.assertEqual(parts[False, False]['breakthrough'], 10.)
        self.assertEqual(parts[True, False]['breakthrough'], 0.)

    def test_three_band_values_continuity_angles_and_non_descent(self):
        env = self.env(descent_three_band_enabled=True)
        for z, expected in ((60, 0), (61, 0), (50, 10.5), (40, 21), (30, 31.5),
                            (20, 42), (12, 73.3567839196), (6, 96.8743718593), (.1, 120), (0, 120)):
            before = np.array([0, 0, z + 1, -.6, 0.])
            after = np.array([0, 0, z, -.6, 0.])
            self.assertAlmostEqual(env._descent_trend_penalty(before, after), expected, places=7)
        for z in (20, 60):
            values = [env._descent_trend_penalty(np.array([0, 0, h+1, -.6, 0]),
                        np.array([0, 0, h, -.6, 0])) for h in (z-1e-7, z, z+1e-7)]
            self.assertLess(max(values) - min(values), 1e-5)
        for gamma in (-.08, 0., .2):
            self.assertEqual(env._descent_trend_penalty(np.array([0, 0, 21, gamma, 0]),
                             np.array([0, 0, 20, gamma, 0])), 0.)
        self.assertAlmostEqual(env._descent_trend_penalty(np.array([0, 0, 21, -.34, 0]),
                               np.array([0, 0, 20, -.34, 0])), 21.)
        self.assertEqual(env._descent_trend_penalty(np.array([0, 0, 19, -.6, 0]),
                         np.array([0, 0, 20, -.6, 0])), 0.)

    def test_new_field_validation_and_scenario_contract(self):
        for field, value in (('breakthrough_clearance_gate_enabled', 1),
                ('descent_three_band_enabled', 'true'), ('descent_low_height', True),
                ('descent_low_height', '20'), ('descent_low_height', float('nan')),
                ('descent_high_height', float('inf')), ('descent_low_height', -1),
                ('descent_low_height', 10**1000),
                ('descent_transition_factor', -.1), ('descent_transition_factor', 1.1)):
            with self.subTest(field=field, value=value), self.assertRaises((ValueError, TypeError)):
                RewardConfig(**{field: value})
        with self.assertRaises(ValueError):
            RewardConfig(descent_low_height=60, descent_high_height=20)
        with self.assertRaises(ValueError):
            V2StaticNoFlyTrajectoryEnv(_config(world_z_min=20), RewardConfig(descent_three_band_enabled=True))
        # Disabled rules preserve legacy scenarios even if their ground is above X.
        V2StaticNoFlyTrajectoryEnv(_config(world_z_min=20), RewardConfig())


class TestV2RewardArtifacts(unittest.TestCase):
    def test_old_formal_and_periodic_snapshots_normalize_only_added_fields(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint, pool, payload = write_inputs(Path(root))
            scenario = ScenarioConfig(max_steps=2)
            periodic = build_v2_periodic_snapshot(
                _engine(scenario), V2FormalTrainingConfig(stage='easy'), stage_steps=1,
                scenario=scenario, rewards=RewardConfig(),
                uav_collision_radius=0., seed_manifest={'model': 7},
                initialization_source=payload['initialization_source'],
            )
            for source, loader in ((payload, load_v2_formal_checkpoint), (periodic, load_v2_periodic_snapshot)):
                old = deepcopy(source)
                for name in NEW_FIELDS:
                    old['reward_config'].pop(name, None)
                normalized = loader(old)
                self.assertEqual(normalized['reward_config'], asdict(RewardConfig()))
                self.assertNotIn('descent_three_band_enabled', old['reward_config'])
                changed = deepcopy(source)
                changed['reward_config'] = asdict(RewardConfig(
                    breakthrough_clearance_gate_enabled=True, descent_three_band_enabled=True))
                self.assertEqual(loader(changed)['reward_config'], changed['reward_config'])
                mixed = deepcopy(old)
                mixed['reward_config']['breakthrough_clearance_gate_enabled'] = True
                self.assertTrue(loader(mixed)['reward_config']['breakthrough_clearance_gate_enabled'])
                for name, value in (('progress_weight', None), ('unknown', 1),
                                    ('descent_three_band_enabled', 1), ('descent_low_height', float('nan'))):
                    bad = deepcopy(old)
                    if value is None:
                        bad['reward_config'].pop(name)
                    else:
                        bad['reward_config'][name] = value
                    with self.assertRaises((ValueError, TypeError)):
                        loader(bad)
            from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_inputs
            # Old checkpoints remain unchanged on disk; the loader exposes effective defaults.
            import torch
            torch.save({
                **payload, 'reward_config': {k: v for k, v in payload['reward_config'].items() if k not in NEW_FIELDS}
            }, checkpoint)
            inputs = load_diagnostic_inputs(model='ann', checkpoint=checkpoint, validation_pool=pool, device='cpu')
            self.assertFalse(inputs['checkpoint_payload']['reward_config']['descent_three_band_enabled'])

    def test_formal_predecessor_inheritance_mismatch_and_failed_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, payload = write_inputs(Path(root))
            from test_v2_formal_training import _validation_result
            scenario = ScenarioConfig(max_steps=2)
            rewards = RewardConfig(breakthrough_clearance_gate_enabled=True, descent_three_band_enabled=True)
            result = V2FormalTrainingResult.empty('easy', global_steps_start=0)
            result.status, result.passed_validation, result.stop_reason = 'passed', True, 'validation_passed'
            result.validation_records = [_validation_result(True).to_dict()]
            passed = build_v2_formal_checkpoint(_engine(scenario), result,
                V2FormalTrainingConfig(stage='easy'), scenario=scenario, rewards=rewards,
                uav_collision_radius=0., seed_manifest={'model': 7},
                validation_pool_metadata=payload['validation_pool'], initialization_source=payload['initialization_source'])
            prepared = prepare_v2_stage_initialization(V2FormalTrainingConfig(stage='medium'), init_checkpoint=passed)
            self.assertEqual(prepared.reward_config, rewards)
            with self.assertRaisesRegex(ValueError, 'RewardConfig'):
                prepare_v2_stage_initialization(V2FormalTrainingConfig(stage='medium'),
                                                init_checkpoint=passed, rewards=RewardConfig())
            with self.assertRaises(ValueError):
                prepare_v2_stage_initialization(V2FormalTrainingConfig(stage='medium'), init_checkpoint=payload)

    def test_curriculum_uses_same_rewards_for_easy_preparation_and_stage_calls(self):
        from brain_uav.scripts import run_v2_td3_curriculum as curriculum
        from brain_uav.trainers import v2_formal_training as formal
        from test_v2_formal_training import _validation_result
        import torch
        for gate, descent in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(gate=gate, descent=descent), tempfile.TemporaryDirectory() as root:
                directory = Path(root)
                bc = directory / 'bc.pt'
                bc.write_bytes(b'fixture')
                rewards = RewardConfig(breakthrough_clearance_gate_enabled=gate, descent_three_band_enabled=descent)
                prepared = replace(_prepared_initialization(ScenarioConfig(), source=bc), reward_config=rewards)
                calls = []
                def stage(**kwargs):
                    calls.append(kwargs)
                    self.assertEqual(kwargs['rewards'], rewards if kwargs['stage'] == 'easy' else None)
                    config = V2FormalTrainingConfig(stage=kwargs['stage'])
                    if kwargs['stage'] == 'easy':
                        effective = kwargs['prepared_initialization']
                        formal.validate_v2_prepared_stage_initialization(effective, config,
                            init_checkpoint=bc, rewards=kwargs['rewards'])
                    else:
                        self.assertEqual(kwargs['init_checkpoint'], calls[-2]['output'])
                        effective = formal.prepare_v2_stage_initialization(config,
                            init_checkpoint=kwargs['init_checkpoint'], rewards=kwargs['rewards'])
                    self.assertEqual(effective.reward_config, rewards)
                    result = V2FormalTrainingResult.empty(kwargs['stage'], global_steps_start=0)
                    result.status, result.passed_validation, result.stop_reason = 'passed', True, 'validation_passed'
                    result.validation_records = [_validation_result(True).to_dict()]
                    previous_stage = {'medium': 'easy', 'hard': 'medium'}.get(kwargs['stage'])
                    source = {'kind': 'validated_v2_td3_stage' if previous_stage else 'v2_bc_best',
                              'path': str(kwargs['init_checkpoint'])}
                    if previous_stage:
                        source['previous_stage'] = previous_stage
                    artifact = build_v2_formal_checkpoint(_engine(effective.scenario_config), result, config,
                        scenario=effective.scenario_config, rewards=effective.reward_config,
                        uav_collision_radius=0., seed_manifest={'model': 7},
                        validation_pool_metadata={'path': str(kwargs['validation_pool']), 'format_version': 1,
                            'curriculum_level': kwargs['stage'], 'master_seed': 20260904, 'stage_seed': 7,
                            'scenario_count': 100, 'content_digest': 'fixture'}, initialization_source=source)
                    torch.save(artifact, kwargs['output'])
                    return {'stage': kwargs['stage'], 'checkpoint': str(kwargs['output']),
                            'passed': True, 'steps': 1, 'global_steps_end': len(calls)}
                with mock.patch.object(formal, 'load_v2_bc_formal_initialization',
                        return_value=prepared.bc_initialization), \
                        mock.patch.object(curriculum, 'prepare_v2_validation_pools',
                            return_value={s: directory / f'{s}.json' for s in ('easy', 'medium', 'hard')}), \
                        mock.patch.object(curriculum, 'load_v2_validation_pool',
                            return_value=SimpleNamespace(master_seed=20260904, stage_seed=7,
                                                         scenario_count=100, content_digest='fixture')):
                    summary = curriculum.run_v2_curriculum(bc_checkpoint=bc, output_root=directory / 'out',
                        validation_pool_dir=directory / 'pools', rewards=rewards, device='cpu', stage_runner=stage)
                self.assertEqual(len(calls), 3)
                self.assertEqual(summary['reward_config'], asdict(rewards))

    def test_report_and_old_diagnostic_manifest_record_effective_rewards(self):
        from brain_uav.trainers.v2_reporting import V2ExperimentReporter
        from brain_uav.scripts.diagnose_v2_policy import run_policy_diagnostic
        import torch
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            rewards = RewardConfig(breakthrough_clearance_gate_enabled=True, descent_three_band_enabled=True)
            reporter = V2ExperimentReporter(directory / 'report', stage='easy', model_type='ann',
                scenario=ScenarioConfig(), rewards=rewards, uav_collision_radius=0., max_steps=1)
            try:
                reporter.start_stage({})
            finally:
                reporter.close()
            self.assertEqual(json.loads((directory / 'report' / 'stage_start.json').read_text())['reward_config'],
                             asdict(rewards))
            checkpoint, pool, payload = write_inputs(directory)
            payload['reward_config'] = {k: v for k, v in payload['reward_config'].items() if k not in NEW_FIELDS}
            torch.save(payload, checkpoint)
            run_policy_diagnostic(model='ann', checkpoint=checkpoint, validation_pool=pool,
                                  output_dir=directory / 'diagnostic', max_scenes=1)
            manifest = json.loads((directory / 'diagnostic' / 'manifest.json').read_text())
            self.assertEqual(manifest['reward_config'], asdict(RewardConfig()))


if __name__ == '__main__':
    unittest.main()
