"""Small deterministic fixtures for diagnostic-only policy evaluation."""

import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig, V2FormalTrainingResult, build_v2_formal_checkpoint,
    save_v2_formal_checkpoint,
)
from brain_uav.trainers.v2_validation import save_v2_validation_pool
from test_v2_formal_training import _engine
from test_v2_static_no_fly_env import _config, _scenario
from test_v2_validation import _payload, _pool


def write_inputs(directory):
    scenario = ScenarioConfig(max_steps=2)
    pool = _pool('easy', scenario, (1.0, 100.0, 100.0))
    records = list(pool.scenarios)
    records[-1]['payload'] = _payload('easy', 1002, goal_x=100.0, zones=((40, 100, 100),))
    pool = replace(pool, scenarios=records)
    pool_path = directory / 'pool.json'
    save_v2_validation_pool(pool_path, pool)
    engine = _engine(scenario)
    for parameter in engine.actor.parameters():
        parameter.data.zero_()
    result = V2FormalTrainingResult.empty('easy', global_steps_start=0)
    result.status = 'failed'
    result.stop_reason = 'max_steps_without_validation'
    result.stage_steps = result.global_steps_end = 500000
    config = V2FormalTrainingConfig(stage='easy', gamma=0.99)
    payload = build_v2_formal_checkpoint(
        engine, result, config, scenario=scenario, rewards=RewardConfig(),
        uav_collision_radius=0.0, seed_manifest={'model': 7},
        validation_pool_metadata={
            'path': str(pool_path), 'format_version': 1, 'curriculum_level': 'easy',
            'master_seed': pool.master_seed, 'stage_seed': pool.stage_seed,
            'scenario_count': pool.scenario_count, 'content_digest': pool.content_digest,
        }, initialization_source={'kind': 'v2_bc_best', 'path': 'fixture_bc.pt'},
    )
    checkpoint = directory / 'failed.pt'
    save_v2_formal_checkpoint(checkpoint, payload)
    return checkpoint, pool_path, payload


class TestRewardDiagnostics(unittest.TestCase):
    def test_collection_is_observational_for_running_and_all_terminal_branches(self):
        cases = [
            ('running', [0, 0, 10, 0, 0], [60, 0, 10], [], {}),
            ('goal', [0, 0, 10, 0, 0], [1, 0, 10], [], {}),
            ('ground', [0, 0, 0.2, -0.6, 0], [60, 0, 10], [], {}),
            ('boundary', [99.8, 99.8, 49.8, 0.6, math.pi / 4], [0, 0, 10], [], {}),
            ('collision', [0, 0, 10, 0, 0], [60, 0, 10],
             [NoFlyZone('solid', Sphere([1, 0, 10], 0.3))], {}),
            ('timeout', [0, 0, 10, 0, 0], [60, 0, 10], [], {'max_steps': 1}),
        ]
        for outcome, state, goal, zones, overrides in cases:
            with self.subTest(outcome=outcome):
                payload = _scenario(state, goal, zones)
                plain = V2StaticNoFlyTrajectoryEnv(_config(**overrides), RewardConfig())
                collected = V2StaticNoFlyTrajectoryEnv(_config(**overrides), RewardConfig())
                self.assertTrue(hasattr(collected, 'enable_reward_diagnostics'))
                collected.enable_reward_diagnostics()
                plain.reset(options={'scenario': payload})
                collected.reset(options={'scenario': payload})
                for action in [np.zeros(2), np.array([0.01, -0.02])]:
                    obs_a, reward_a, done_a, trunc_a, info_a = plain.step(action)
                    obs_b, reward_b, done_b, trunc_b, info_b = collected.step(action)
                    np.testing.assert_array_equal(plain.state, collected.state)
                    np.testing.assert_array_equal(obs_a.zone_features, obs_b.zone_features)
                    self.assertEqual((reward_a, done_a, trunc_a), (reward_b, done_b, trunc_b))
                    self.assertNotIn('reward_components', info_a)
                    self.assertIsNone(plain.last_reward_components)
                    parts = info_b['reward_components']
                    self.assertEqual(len(parts), 13)
                    self.assertAlmostEqual(math.fsum(parts.values()), reward_b, places=8)
                    self.assertLessEqual(info_b['reward_component_error'], 1e-8)
                    self.assertEqual(info_b['outcome'], outcome)
                    expected = {'goal': 5000, 'ground': -24000, 'collision': -24000,
                                'boundary': -24000, 'timeout': -5000}.get(outcome, 0)
                    self.assertEqual(parts['termination'], expected)
                    if done_b or trunc_b:
                        break
                collected.reset(options={'scenario': payload})
                self.assertIsNone(collected.last_reward_components)
                self.assertFalse(collected.last_terminal_los_active)
                collected.enable_reward_diagnostics(False)
                self.assertNotIn('reward_components', collected.step(np.zeros(2))[-1])

    def test_terminal_activation_counts_actual_gating_even_with_zero_weights(self):
        env = V2StaticNoFlyTrajectoryEnv(_config(), RewardConfig(
            terminal_los_weight=0, terminal_los_penalty_weight=0,
            terminal_radial_weight=0, terminal_tangential_penalty_weight=0,
        ))
        self.assertTrue(hasattr(env, 'enable_reward_diagnostics'))
        env.enable_reward_diagnostics()
        env.reset(options={'scenario': _scenario([0, 0, 10, 0, 0], [60, 0, 10], [])})
        info = env.step(np.zeros(2))[-1]
        self.assertTrue(info['terminal_los_active'])
        self.assertTrue(info['terminal_radial_tangential_active'])
        self.assertEqual(info['reward_components']['terminal_direction'], 0)

    def test_window_terms_and_occluded_terminal_guidance_use_real_reward_helpers(self):
        zone = NoFlyZone('nearby', Sphere([20, 10, 10], 2))
        env = V2StaticNoFlyTrajectoryEnv(_config(max_steps=30), RewardConfig())
        env.enable_reward_diagnostics()
        env.reset(options={'scenario': _scenario([0, 0, 10, 0, 0], [60, 0, 10], [zone])})
        for _ in range(10):
            info = env.step(np.zeros(2))[-1]
        self.assertGreater(info['reward_components']['breakthrough'], 0)
        self.assertLessEqual(info['reward_components']['breakthrough'], env.rewards.breakthrough_reward_cap)
        self.assertLess(info['reward_components']['zone_warning'], 0)
        env.reset(options={'scenario': _scenario([0, 0, 10, 0, math.pi], [60, 0, 10], [])})
        for _ in range(10):
            info = env.step(np.zeros(2))[-1]
        self.assertLess(info['reward_components']['inefficiency'], 0)
        blocker = NoFlyZone('blocker', Sphere([30, 0, 10], 2))
        env.reset(options={'scenario': _scenario([0, 0, 10, 0, 0], [60, 0, 10], [blocker])})
        info = env.step(np.zeros(2))[-1]
        self.assertFalse(info['terminal_los_active'])
        self.assertFalse(info['terminal_radial_tangential_active'])
        self.assertEqual(info['reward_components']['terminal_direction'], 0)


class TestPolicyDiagnostic(unittest.TestCase):
    def test_operation_counting_preserves_rollout_and_exports_every_decision(self):
        from brain_uav.scripts import diagnose_v2_policy as diagnostic
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            checkpoint, pool, _ = write_inputs(directory)
            summaries, all_steps = [], []
            for enabled in (False, True):
                output = directory / str(enabled)
                summary = diagnostic.run_policy_diagnostic(
                    model='ann', checkpoint=checkpoint, validation_pool=pool,
                    output_dir=output, device='cpu', count_operations=enabled,
                )
                summaries.append(summary)
                rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
                steps = [json.loads(line) for row in rows
                         for line in (output / row['steps_path']).read_text().splitlines()]
                if enabled:
                    report = json.loads((output / 'operation_counts.json').read_text())
                    self.assertEqual(report['rollout']['decision_count'], len(steps))
                    self.assertEqual(report['initial_observations']['decision_count'], 3)
                    self.assertEqual(report['rollout']['totals']['macs'],
                                     sum(step['operation_counts']['macs'] for step in steps))
                    self.assertEqual(report['rollout']['totals']['acs'], 0)
                    self.assertEqual(report['by_zone_count']['0']['decision_count'], 3)
                    for row in rows:
                        self.assertEqual(row['operation_counts']['decision_count'], row['episode_length'])
                    for step in steps:
                        step.pop('operation_counts')
                else:
                    self.assertNotIn('operation_counts', summary)
                    self.assertFalse((output / 'operation_counts.json').exists())
                all_steps.append(steps)
            self.assertEqual(summaries[0]['overall'], summaries[1]['overall'])
            self.assertEqual(all_steps[0], all_steps[1])
            self.assertEqual(torch.load(checkpoint, weights_only=False)['status'], 'failed')

    def test_cache_is_explicitly_enabled_and_manifest_records_actual_state(self):
        from brain_uav.scripts import diagnose_v2_policy as diagnostic
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            checkpoint, pool, _ = write_inputs(directory)
            for actual_override in (None, False):
                with self.subTest(actual_override=actual_override):
                    environments, requested_flags = [], []

                    def default_off_env(*args, cache_ellipsoid_segment_clearance=False, **kwargs):
                        requested_flags.append(cache_ellipsoid_segment_clearance)
                        env = V2StaticNoFlyTrajectoryEnv(
                            *args, cache_ellipsoid_segment_clearance=cache_ellipsoid_segment_clearance,
                            **kwargs,
                        )
                        if actual_override is not None:
                            env.cache_ellipsoid_segment_clearance = actual_override
                        environments.append(env)
                        return env

                    output = directory / f'cache_{actual_override}'
                    with mock.patch.object(diagnostic, 'V2StaticNoFlyTrajectoryEnv', default_off_env):
                        diagnostic.run_policy_diagnostic(
                            model='ann', checkpoint=checkpoint, validation_pool=pool,
                            output_dir=output, device='cpu', max_scenes=1,
                        )
                    self.assertEqual(requested_flags, [True])
                    manifest = json.loads((output / 'manifest.json').read_text())
                    actual = environments[0].cache_ellipsoid_segment_clearance
                    self.assertEqual(actual, True if actual_override is None else actual_override)
                    self.assertEqual(manifest['runtime']['cache_ellipsoid_segment_clearance'], actual)

    def test_artifacts_totals_tail_groups_and_failed_status_without_training(self):
        from brain_uav.scripts import diagnose_v2_policy as diagnostic
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            checkpoint, pool, payload = write_inputs(directory)
            output = directory / 'diagnostic'
            with mock.patch('torch.optim.Adam', side_effect=AssertionError('no optimizer')):
                summary = diagnostic.run_policy_diagnostic(
                    model='ann', checkpoint=checkpoint, validation_pool=pool,
                    output_dir=output, device='cpu',
                )
            self.assertTrue(summary['diagnostic_only'])
            self.assertEqual(summary['overall']['scenario_count'], 3)
            self.assertEqual(summary['overall']['success_rate'], 1 / 3)
            self.assertEqual(summary['by_outcome']['timeout']['scenario_count'], 2)
            self.assertEqual(summary['by_zone_count']['0']['scenario_count'], 2)
            self.assertEqual(summary['by_zone_count']['1']['scenario_count'], 1)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['checkpoint']['status'], 'failed')
            self.assertFalse(manifest['checkpoint']['passed_validation'])
            self.assertEqual(manifest['checkpoint']['stage_steps'], 500000)
            self.assertEqual(manifest['pool']['content_digest'], payload['validation_pool']['content_digest'])
            self.assertEqual(manifest['runtime']['device'], 'cpu')
            rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
            with (output / 'episodes.csv').open(newline='', encoding='utf-8') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            for row in rows:
                records = [json.loads(line) for line in (output / row['steps_path']).read_text().splitlines()]
                tail = [json.loads(line) for line in (output / row['tail_50_path']).read_text().splitlines()]
                self.assertEqual(tail, records[-50:])
                self.assertAlmostEqual(sum(step['reward'] for step in records), row['episode_return'])
                self.assertAlmostEqual(sum(.99**i * step['reward'] for i, step in enumerate(records)), row['discounted_return'])
                self.assertEqual(row['last_reward_components'], records[-1]['reward_components'])
                self.assertTrue(row['entered_250_range'])
                if row['zone_count'] == 0:
                    self.assertIsNone(records[0]['min_zone_clearance'])
                else:
                    self.assertIsInstance(records[0]['min_zone_clearance'], float)
                for name, total in row['cumulative_reward_components'].items():
                    self.assertAlmostEqual(sum(step['reward_components'][name] for step in records), total)
                self.assertTrue((output / row['trajectory_views']['png']).is_file())
            self.assertEqual(diagnostic.main([
                '--model', 'ann', '--checkpoint', str(checkpoint), '--validation-pool', str(pool),
                '--output-dir', str(directory / 'smoke'), '--device', 'cpu', '--max-scenes', '1',
            ]), 0)
            with self.assertRaises(FileExistsError):
                diagnostic.run_policy_diagnostic(model='ann', checkpoint=checkpoint,
                    validation_pool=pool, output_dir=output, device='cpu')
            self.assertEqual(torch.load(checkpoint, weights_only=False)['status'], 'failed')

    def test_contract_mismatches_and_cuda_unavailability_are_rejected(self):
        from brain_uav.scripts import diagnose_v2_policy as diagnostic
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            checkpoint, pool, payload = write_inputs(directory)
            bad_values = [
                ('digest', lambda p: p['validation_pool'].update(content_digest='wrong')),
                ('bounds', lambda p: p['engine_checkpoint'].update(action_low=[-1, -1])),
                ('radius', lambda p: p.update(uav_collision_radius=1)),
                ('gamma', lambda p: p['engine_checkpoint']['algorithm_config'].update(gamma=.995)),
                ('weights', lambda p: p['engine_checkpoint']['actor_state_dict'].pop('action_limit')),
                ('observation', lambda p: p['engine_checkpoint']['observation_contract'].update(id='legacy')),
                ('buffer', lambda p: p['engine_checkpoint']['actor_state_dict'].update({
                    'zone_set_encoder.pair_relation_builder.world_diagonal': torch.tensor(1.0)})),
                ('dtype', lambda p: p['engine_checkpoint']['actor_state_dict'].update({
                    'action_limit': p['engine_checkpoint']['actor_state_dict']['action_limit'].double()})),
            ]
            from copy import deepcopy
            for name, mutate in bad_values:
                with self.subTest(name=name):
                    changed = deepcopy(payload)
                    mutate(changed)
                    bad = directory / f'{name}.pt'
                    torch.save(changed, bad)
                    with self.assertRaises((ValueError, RuntimeError)):
                        diagnostic.load_diagnostic_inputs(model='ann', checkpoint=bad,
                            validation_pool=pool, device='cpu')
            with mock.patch('torch.cuda.is_available', return_value=False):
                with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                    diagnostic.load_diagnostic_inputs(model='ann', checkpoint=checkpoint,
                        validation_pool=pool, device='cuda')

    def test_boundary_axes_reports_multiple_actual_directions(self):
        from brain_uav.scripts import diagnose_v2_policy as diagnostic
        self.assertEqual(diagnostic.boundary_violations([101, -102, 51, 0, 0], _config()),
                         [{'axis': 'x', 'direction': 'positive', 'value': 101, 'limit': 100},
                          {'axis': 'y', 'direction': 'negative', 'value': -102, 'limit': -100},
                          {'axis': 'z', 'direction': 'positive', 'value': 51, 'limit': 50}])


if __name__ == '__main__':
    unittest.main()
