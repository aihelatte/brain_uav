"""Focused P3 reward, provenance and terminal-record regression tests."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig, reward_config_from_snapshot
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.trainers.v2_reporting import V2ExperimentReporter
from test_v2_static_no_fly_env import _scenario


P3_FIELDS = ('boundary_directional_warning_enabled',
             'boundary_directional_warning_distance', 'boundary_directional_penalty_weight')


class TestBoundaryDirectional(unittest.TestCase):
    def env(self, enabled=True, **options):
        rewards = RewardConfig(boundary_directional_warning_enabled=enabled, **options)
        env = V2StaticNoFlyTrajectoryEnv(
            ScenarioConfig(world_xy=1312.5, world_z_max=600), rewards)
        env.reset(options={'scenario': _scenario([0, 0, 100, 0, 0], [500, 0, 100], [])})
        return env

    def penalty(self, env, xyz, gamma=0.6, psi=0):
        env.state = np.array([*xyz, gamma, psi], dtype=np.float64)
        return env._boundary_warning_penalty(env.state[:3])

    def test_upper_height_values_and_actual_velocity(self):
        env = self.env()
        for z, expected in ((540, 0), (550, 2.5), (570, 22.5), (580, 40),
                            (590, 62.5), (595, 120.625)):
            with self.subTest(z=z):
                self.assertAlmostEqual(self.penalty(env, [0, 0, z]), expected)
                legacy = 45 if z == 595 else 0
                for gamma in (0, -0.6):
                    self.assertAlmostEqual(self.penalty(env, [0, 0, z], gamma), legacy)
        env.enable_reward_diagnostics()
        env.state = np.array([0., 0., 595., .6, 0.])
        reward = env._compute_reward(env.state.copy(), np.zeros(2), 500, 500,
                                    np.zeros(2), 'running', 500)
        self.assertEqual(env.last_reward_components['boundary_warning'], -45)
        self.assertAlmostEqual(env.last_reward_components['boundary_directional'], -75.625)
        self.assertAlmostEqual(math.fsum(env.last_reward_components.values()), reward)

    def test_horizontal_symmetry_and_same_face_maximum(self):
        env = self.env()
        for xyz, outward, inward in (([1282.5, 0, 100], 0, math.pi),
                ([-1282.5, 0, 100], math.pi, 0),
                ([0, 1282.5, 100], math.pi / 2, -math.pi / 2),
                ([0, -1282.5, 100], -math.pi / 2, math.pi / 2)):
            self.assertAlmostEqual(self.penalty(env, xyz, 0, outward), 22.5)
            self.assertAlmostEqual(self.penalty(env, xyz, 0, inward), 0)
        # +x owns the legacy risk (45), +y owns directional risk (22.5).
        # They must compete, not add to 67.5.
        self.assertAlmostEqual(self.penalty(env, [1307.5, 1282.5, 100], 0, math.pi / 2), 45)
        env.enable_reward_diagnostics()
        reward = env._compute_reward(env.state.copy(), np.zeros(2), 500, 500,
                                    np.zeros(2), 'running', 500)
        self.assertAlmostEqual(env.last_reward_components['boundary_directional'], 0, delta=1e-12)
        self.assertAlmostEqual(math.fsum(env.last_reward_components.values()), reward)
        self.assertAlmostEqual(self.penalty(env, [1400, 0, 100], 0, 0), 240)

    def test_disabled_parity_and_diagnostics_do_not_leak(self):
        env = self.env(False)
        for z, expected in ((540, 0), (590, 0), (595, 45), (600, 180), (605, 180)):
            self.assertAlmostEqual(self.penalty(env, [0, 0, z]), expected)
        for p2, p3 in ((False, False), (False, True), (True, False), (True, True)):
            env = self.env(p3, descent_three_band_enabled=p2)
            env.enable_reward_diagnostics()
            for xyz, gamma in (([0, 0, 595], .6), ([0, 0, 30], -.6), ([0, 0, 100], 0)):
                env.state = np.array([*xyz, gamma, 0.])
                prev = env.state.copy()
                prev[2] -= env.scenario.speed * math.sin(gamma)
                reward = env._compute_reward(prev, np.zeros(2), 500, 500,
                                            np.zeros(2), 'running', 500)
                parts = env.last_reward_components
                self.assertEqual('boundary_directional' in parts, p3)
                if p3:
                    expected = -75.625 if xyz[2] == 595 else 0
                    self.assertAlmostEqual(parts['boundary_directional'], expected)
                if xyz[2] == 30:
                    self.assertAlmostEqual(parts['descent_trend'], -31.5 if p2 else -42)
                self.assertAlmostEqual(math.fsum(parts.values()), reward)
            env.reset(options={'scenario': _scenario([0, 0, 100, 0, 0], [500, 0, 100], [])})
            self.assertIsNone(env.last_reward_components)
        env = self.env(True)
        env.enable_reward_diagnostics()
        env.state = np.array([1400., 0., 100., 0., 0.])
        reward = env._compute_reward(env.state.copy(), np.zeros(2), 500, 500,
                                    np.zeros(2), 'boundary', 500)
        self.assertEqual(env.last_reward_components['boundary_warning'], -180)
        self.assertEqual(env.last_reward_components['boundary_directional'], -60)
        self.assertEqual(env.last_reward_components['termination'], -24000)
        self.assertAlmostEqual(math.fsum(env.last_reward_components.values()), reward)

    def test_config_validation_and_legacy_defaults(self):
        for value in (0, 1, None, 'true'):
            with self.assertRaises(TypeError):
                RewardConfig(boundary_directional_warning_enabled=value)
        for name in P3_FIELDS[1:]:
            for value in (True, '60', math.nan, math.inf, -1, 10**1000):
                with self.subTest(name=name, value=repr(value)[:25]), self.assertRaises(ValueError):
                    RewardConfig(**{name: value})
        with self.assertRaises(ValueError):
            RewardConfig(boundary_directional_warning_distance=0)
        RewardConfig(boundary_directional_penalty_weight=0)
        for distance in (5, 10):
            with self.assertRaises(ValueError):
                RewardConfig(boundary_directional_warning_enabled=True,
                    boundary_directional_warning_distance=distance).validate_scenario(ScenarioConfig())
        legacy = asdict(RewardConfig())
        for name in P3_FIELDS:
            legacy.pop(name)
        self.assertEqual(reward_config_from_snapshot(legacy), RewardConfig())
        from test_v2_reward_options import NEW_FIELDS
        for name in NEW_FIELDS:
            legacy.pop(name)
        self.assertEqual(reward_config_from_snapshot(legacy), RewardConfig())
        current = asdict(RewardConfig(boundary_directional_warning_enabled=True,
                                     boundary_directional_penalty_weight=75))
        self.assertEqual(asdict(reward_config_from_snapshot(current)), current)
        for broken in ({**current, 'unknown': 0}, {**current, P3_FIELDS[0]: 1},
                       {k: v for k, v in current.items() if k != 'progress_weight'}):
            with self.assertRaises((ValueError, TypeError)):
                reward_config_from_snapshot(broken)

    def test_terminal_info_all_axes_ground_and_reset(self):
        env = self.env()
        for xyz, outcome, expected in (([1313, -1313, 601], 'boundary',
                [('x', 'positive', 1312.5), ('y', 'negative', -1312.5), ('z', 'positive', 600)]),
                ([0, 0, 0], 'ground', [('z', 'negative', .1)]),
                ([0, 0, 100], 'timeout', []),
                ([1312.5, -1312.5, .1], 'ground', [])):
            env.state = np.array([*xyz, 0., 0.])
            info = env._info(progress=0, outcome=outcome)
            self.assertEqual(info['outcome'], outcome)
            self.assertEqual(info['terminal_position'], xyz)
            self.assertEqual([(x['axis'], x['direction'], x['limit']) for x in info['boundary_violations']], expected)
            self.assertEqual([x['value'] for x in info['boundary_violations']],
                             [xyz[('x', 'y', 'z').index(x['axis'])] for x in info['boundary_violations']])
        _, info = env.reset(options={'scenario': _scenario([0, 0, 100, 0, 0], [500, 0, 100], [])})
        self.assertNotIn('terminal_position', info)

    def test_zero_action_keeps_upward_heading_and_diagnostic_is_observational(self):
        payload = _scenario([0, 0, 593.5, .6, 0], [500, 0, 100], [])
        plain, diagnostic = self.env(), self.env()
        diagnostic.enable_reward_diagnostics()
        for env in (plain, diagnostic):
            env.reset(options={'scenario': payload})
        for _ in range(2):
            first = plain.step([0, 0])
            second = diagnostic.step([0, 0])
            np.testing.assert_array_equal(plain.state, diagnostic.state)
            self.assertEqual(first[1:4], second[1:4])
            self.assertGreater(plain.state[3], 0)
            self.assertLess(second[4]['reward_components']['boundary_directional'], 0)
            self.assertAlmostEqual(math.fsum(second[4]['reward_components'].values()), second[1])


class TestBoundaryRecordsAndArtifacts(unittest.TestCase):
    def test_all_complete_training_and_validation_records_reach_disk(self):
        from test_v2_formal_training import _engine, _Source, _validation_result
        from test_v2_validation import _pool
        from brain_uav.trainers.v2_formal_training import V2FormalTrainingConfig, V2FormalStageTrainer
        from brain_uav.trainers.v2_validation import evaluate_v2_fixed_validation
        scenario = ScenarioConfig(world_xy=1312.5, world_z_max=600, max_steps=1)
        rewards = RewardConfig(boundary_directional_warning_enabled=True)
        payloads = [
            _scenario([1312.3, -1312.3, 599.8, .6, -math.pi / 4], [0, 0, 100], [], level='easy'),
            _scenario([0, 0, .2, -.6, 0], [500, 0, 100], [], level='easy'),
            _scenario([0, 0, 100, 0, 0], [500, 0, 100], [], level='easy'),
        ]
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            reporter = V2ExperimentReporter(directory, stage='easy', model_type='ann',
                scenario=scenario, rewards=rewards, uav_collision_radius=0., max_steps=1)
            try:
                engine = _engine(scenario)
                trainer = V2FormalStageTrainer(scenario, rewards,
                    V2FormalTrainingConfig(stage='easy', max_steps=1, batch_size=2, replay_capacity=8),
                    engine, scenario_sources={'easy': _Source(payloads[0])},
                    validation_runner=lambda actor: _validation_result(False), reporter=reporter)
                with mock.patch.object(engine, 'select_action', return_value=np.zeros(2, dtype=np.float32)), \
                     mock.patch.object(engine, 'update_once') as update, \
                     mock.patch('brain_uav.trainers.v2_reporting.export_v2_trajectory_views'), \
                     mock.patch('brain_uav.trainers.v2_reporting._plot_training_windows'):
                    result = trainer.run()
                    update.assert_not_called()
                    reporter.finish_stage(result.to_dict())
                episode = result.episodes[0]
                self.assertEqual(episode['outcome'], 'boundary')
                self.assertEqual(len(episode['boundary_violations']), 3)
                self.assertEqual(episode['terminal_position'], trainer.env.state[:3].tolist())
                persisted = json.loads((directory / 'episodes.jsonl').read_text().strip())
                for field in ('terminal_position', 'boundary_violations'):
                    self.assertEqual(persisted[field], episode[field])
                    self.assertEqual(json.loads((directory / 'stage_end.json').read_text())['episodes'][0][field], episode[field])
                pool = _pool('easy', scenario, (500, 500, 500))
                records = deepcopy(pool.scenarios)
                for record, payload in zip(records, payloads):
                    record['payload']['state'] = payload['state']
                    record['payload']['goal'] = payload['goal']
                pool = replace(pool, scenarios=records)
                for parameter in engine.actor.parameters():
                    parameter.data.zero_()
                reporter.prepare_validation_candidate(1, global_steps=1)
                with mock.patch('brain_uav.trainers.v2_reporting.export_v2_trajectory_views'):
                    validation = evaluate_v2_fixed_validation(engine.actor, pool, rewards, reporter=reporter)
                candidate = directory / 'validation' / 'candidate_0001'
                rows = [json.loads(line) for line in (candidate / 'scenarios.jsonl').read_text().splitlines()]
                summary = json.loads((candidate / 'summary.json').read_text())
                self.assertEqual([r['outcome'] for r in rows], ['boundary', 'ground', 'timeout'])
                self.assertEqual([len(r['boundary_violations']) for r in rows], [3, 1, 0])
                for row, detail, saved in zip(rows, validation.scenarios, summary['scenarios']):
                    for field in ('terminal_position', 'boundary_violations'):
                        self.assertEqual(row[field], detail[field])
                        self.assertEqual(saved[field], detail[field])
            finally:
                reporter.close()

    def test_checkpoint_snapshot_diagnostic_and_curriculum_inheritance(self):
        from types import SimpleNamespace
        from test_diagnose_v2_policy import write_inputs
        from test_run_v2_td3_curriculum import _prepared_initialization
        from test_v2_formal_training import _engine, _validation_result
        from brain_uav.trainers import v2_formal_training as formal
        from brain_uav.scripts import run_v2_td3_curriculum as curriculum
        from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_inputs
        rewards = RewardConfig(descent_three_band_enabled=True, boundary_directional_warning_enabled=True)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            checkpoint, pool, original = write_inputs(directory)
            scenario = ScenarioConfig(max_steps=2)
            periodic = formal.build_v2_periodic_snapshot(_engine(scenario),
                formal.V2FormalTrainingConfig(stage='easy'), stage_steps=1,
                scenario=scenario, rewards=rewards, uav_collision_radius=0., seed_manifest={'model': 7},
                initialization_source=original['initialization_source'])
            for source, loader in ((original, formal.load_v2_formal_checkpoint),
                                    (periodic, formal.load_v2_periodic_snapshot)):
                old = deepcopy(source)
                for field in P3_FIELDS:
                    old['reward_config'].pop(field)
                self.assertFalse(loader(old)['reward_config'][P3_FIELDS[0]])
                from test_v2_reward_options import NEW_FIELDS
                older = deepcopy(old)
                for field in NEW_FIELDS:
                    older['reward_config'].pop(field)
                self.assertEqual(loader(older)['reward_config'], asdict(RewardConfig()))
                current = {**source, 'reward_config': asdict(rewards)}
                self.assertEqual(loader(current)['reward_config'], asdict(rewards))
                for changes in ({P3_FIELDS[0]: 1}, {P3_FIELDS[1]: math.nan}, {'unknown': 0}):
                    bad = {**current, 'reward_config': {**asdict(rewards), **changes}}
                    with self.assertRaises((ValueError, TypeError)):
                        loader(bad)
            legacy = {**original, 'reward_config': {k: v for k, v in original['reward_config'].items() if k not in P3_FIELDS}}
            torch.save(legacy, checkpoint)
            inputs = load_diagnostic_inputs(model='ann', checkpoint=checkpoint, validation_pool=pool, device='cpu')
            self.assertFalse(inputs['checkpoint_payload']['reward_config'][P3_FIELDS[0]])
            torch.save({**original, 'reward_config': asdict(rewards)}, checkpoint)
            inputs = load_diagnostic_inputs(model='ann', checkpoint=checkpoint, validation_pool=pool, device='cpu')
            self.assertEqual(inputs['checkpoint_payload']['reward_config'], asdict(rewards))
            with self.assertRaises(ValueError):
                formal.prepare_v2_stage_initialization(formal.V2FormalTrainingConfig(stage='medium'), init_checkpoint=checkpoint)
            bc = directory / 'bc.pt'
            bc.write_bytes(b'fixture')
            prepared = _prepared_initialization(ScenarioConfig(), source=bc)
            calls = []
            def stage(**kwargs):
                calls.append(kwargs)
                config = formal.V2FormalTrainingConfig(stage=kwargs['stage'])
                if kwargs['stage'] == 'easy':
                    effective = kwargs['prepared_initialization']
                    formal.validate_v2_prepared_stage_initialization(effective, config,
                        init_checkpoint=bc, rewards=kwargs['rewards'])
                else:
                    self.assertEqual(kwargs['init_checkpoint'], calls[-2]['output'])
                    effective = formal.prepare_v2_stage_initialization(config, init_checkpoint=kwargs['init_checkpoint'])
                    with self.assertRaisesRegex(ValueError, 'RewardConfig'):
                        formal.prepare_v2_stage_initialization(config, init_checkpoint=kwargs['init_checkpoint'], rewards=RewardConfig())
                self.assertEqual(effective.reward_config, rewards)
                result = formal.V2FormalTrainingResult.empty(kwargs['stage'], global_steps_start=0)
                result.status, result.passed_validation, result.stop_reason = 'passed', True, 'validation_passed'
                result.validation_records = [_validation_result(True).to_dict()]
                previous = {'medium': 'easy', 'hard': 'medium'}.get(kwargs['stage'])
                source = {'kind': 'validated_v2_td3_stage' if previous else 'v2_bc_best', 'path': str(kwargs['init_checkpoint'])}
                if previous:
                    source['previous_stage'] = previous
                artifact = formal.build_v2_formal_checkpoint(_engine(effective.scenario_config), result, config,
                    scenario=effective.scenario_config, rewards=effective.reward_config,
                    uav_collision_radius=0., seed_manifest={'model': 7},
                    validation_pool_metadata={'path': str(kwargs['validation_pool']), 'format_version': 1,
                        'curriculum_level': kwargs['stage'], 'master_seed': 20260904, 'stage_seed': 7,
                        'scenario_count': 100, 'content_digest': 'fixture'}, initialization_source=source)
                torch.save(artifact, kwargs['output'])
                return {'stage': kwargs['stage'], 'checkpoint': str(kwargs['output']),
                        'passed': True, 'steps': 1, 'global_steps_end': len(calls)}
            with mock.patch.object(formal, 'load_v2_bc_formal_initialization', return_value=prepared.bc_initialization), \
                 mock.patch.object(curriculum, 'prepare_v2_validation_pools', return_value={s: directory / f'{s}.json' for s in ('easy', 'medium', 'hard')}), \
                 mock.patch.object(curriculum, 'load_v2_validation_pool', return_value=SimpleNamespace(
                     master_seed=20260904, stage_seed=7, scenario_count=100, content_digest='fixture')):
                summary = curriculum.run_v2_curriculum(bc_checkpoint=bc, output_root=directory / 'course',
                    validation_pool_dir=directory / 'pools', rewards=rewards, device='cpu', stage_runner=stage)
            self.assertEqual(len(calls), 3)
            self.assertEqual(summary['reward_config'], asdict(rewards))


if __name__ == '__main__':
    unittest.main()
