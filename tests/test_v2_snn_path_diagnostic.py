"""CPU diagnostic contracts only; these tests never establish CUDA Graph correctness."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from brain_uav.config import ScenarioConfig, RewardConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2SNNPolicyActor
from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig
from brain_uav.scripts import profile_v2_td3 as profile
from brain_uav.scripts import v2_snn_path_diagnostic as diagnostic
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig, V2FormalTrainingResult, build_v2_formal_checkpoint,
    save_v2_formal_checkpoint,
)
from brain_uav.trainers.v2_validation import save_v2_validation_pool
from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
from test_v2_formal_training import _engine
from test_v2_validation import _pool, _payload


def inputs_fixture(directory):
    scenario = ScenarioConfig()
    pool = _pool('easy', scenario, (1000., 1000., 1000.))
    records = deepcopy(list(pool.scenarios))
    for index, centers in ((1, ((400., 100., 100.),)), (2, ((400., 100., 100.), (600., -100., 100.)))):
        records[index]['payload'] = _payload('easy', 1000 + index, goal_x=1000., zones=centers)
    pool = replace(pool, scenarios=tuple(records))
    pool_path = directory / 'pool.json'
    save_v2_validation_pool(pool_path, pool)
    old = _engine(scenario)
    actor = V2SNNPolicyActor(old.actor.scales, 2, 192, old.actor.action_limit, projection_dim=32)
    spike = V2SpikeRegularizationConfig(1.)
    engine = V2TD3UpdateEngine(actor, old.critic1, old.critic2, old.replay,
        .001, .001, .99, .005, .015, .03, 2, 2, old.action_low, old.action_high,
        actor_freeze_steps=25000, actor_grad_clip_norm=1., critic_grad_clip_norm=1.,
        bc_reference_actor=deepcopy(actor), spike_regularization=spike)
    # Populate real Adam moments and nonzero counters, with a single synthetic update.
    env = V2StaticNoFlyTrajectoryEnv(scenario, RewardConfig())
    observation = env.reset(options={'scenario': records[0]['payload']})[0]
    obs = diagnostic.collate_v2_observations([observation] * 2)
    batch = profile._fixed_numeric_replay_batch(obs, action_dim=2, device=torch.device('cpu'))
    engine.replay.sample = lambda size: batch
    engine.update_once(total_steps=75000, bc_lambda=1.)
    result = V2FormalTrainingResult.empty('easy', global_steps_start=0)
    result.status = 'failed'
    result.stop_reason = 'max_steps_without_validation'
    result.stage_steps = result.global_steps_end = 500000
    config = V2FormalTrainingConfig(stage='easy', gamma=.99, spike_regularization=spike)
    payload = build_v2_formal_checkpoint(engine, result, config, scenario=scenario,
        rewards=RewardConfig(), uav_collision_radius=0., seed_manifest={'model': 7},
        validation_pool_metadata={'path': str(pool_path), 'format_version': 1,
            'curriculum_level': 'easy', 'master_seed': pool.master_seed, 'stage_seed': pool.stage_seed,
            'scenario_count': pool.scenario_count, 'content_digest': pool.content_digest},
        initialization_source={'kind': 'v2_bc_best', 'path': 'unused_fixture_bc.pt', 'model_type': 'snn'})
    checkpoint = directory / 'failed.pt'
    save_v2_formal_checkpoint(checkpoint, payload)
    return checkpoint, pool_path


class TestSNNPathDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_difference_binary_and_none_semantics(self):
        cases = ({'action': torch.tensor([.1])}, {'actor.spikes.l1': torch.tensor([0.])},
                 {'gradient.pre_clip.x.none': torch.tensor(True)})
        for expected in cases:
            actual = {key: value.clone() for key, value in expected.items()}
            results = []
            diagnostic.compare_tensors(expected, actual, stage='same', results=results)
            self.assertTrue(results[-1]['passed'])
            actual = {key: torch.ones_like(value) for key, value in expected.items()}
            if next(iter(expected)).endswith('.none'):
                actual = {'gradient.pre_clip.x': torch.tensor([1.])}
            with self.assertRaises(AssertionError):
                diagnostic.compare_tensors(expected, actual, stage='different', results=results)
            self.assertFalse(results[-1]['passed'])

    def test_cpu_smoke_restores_projected_checkpoint_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint, pool = inputs_fixture(directory)
            hashes = [diagnostic.file_sha256(path) for path in (checkpoint, pool)]
            inputs = diagnostic.load_diagnostic_inputs(model='snn', checkpoint=checkpoint,
                validation_pool=pool, device='cpu')
            a, b = (diagnostic.restore_engine(inputs, fused=True) for _ in range(2))
            diagnostic.compare_tensors(diagnostic.engine_snapshot(a), diagnostic.engine_snapshot(b),
                                      stage='restored', results=[])
            self.assertEqual((a.actor.projection_dim, a.actor.hidden_dim, a.actor.time_window), (32, 192, 4))
            self.assertEqual(a.spike_regularization, V2SpikeRegularizationConfig(1.))
            self.assertEqual(a.actor_update_count, 1)
            self.assertTrue(a.actor_optimizer.state)
            self.assertTrue(a.critic_optimizer.state)
            a.actor.projection.weight.data.add_(1.)
            self.assertFalse(torch.equal(a.actor.projection.weight, b.actor.projection.weight))
            result = diagnostic.run_snn_path_diagnostic(checkpoint=checkpoint, validation_pool=pool,
                output_dir=directory / 'out', device='cpu', cpu_smoke=True,
                expected_digest=inputs['pool'].content_digest)
            self.assertTrue(result['passed'])
            self.assertEqual(result['cuda_status'], 'not_run')
            self.assertEqual([c['beta'] for c in result['cases'] if c['actor_backward']], [.00004, .5, .50004, 1., 1.])
            manifest = json.loads((directory / 'out' / 'manifest.json').read_text())
            self.assertFalse(manifest['passed_validation'])
            self.assertEqual(manifest['architecture']['policy_head']['projection_dim'], 32)
            self.assertEqual(manifest['algorithm']['spike_regularization']['beta_max'], 1.)
            self.assertTrue(any(s.get('label') == 'near_obstacle' for s in manifest['observation_sources']))
            self.assertTrue(any(s.startswith('reset_repeat.') for s in result['stages']))
            self.assertEqual(hashes, [diagnostic.file_sha256(path) for path in (checkpoint, pool)])
            self.assertFalse(list((directory / 'out').glob('*.pt')))
            for filename in ('summary.json', 'comparisons.json', 'manifest.json'):
                json.loads((directory / 'out' / filename).read_text())
            with self.assertRaises(FileExistsError):
                diagnostic.run_snn_path_diagnostic(checkpoint=checkpoint, validation_pool=pool,
                    output_dir=directory / 'out', device='cpu', cpu_smoke=True)

    def test_recording_is_observational_and_copies_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, pool = inputs_fixture(Path(tmp))
            inputs = diagnostic.load_diagnostic_inputs(model='snn', checkpoint=checkpoint, validation_pool=pool, device='cpu')
            a, b = (diagnostic.restore_engine(inputs, fused=True) for _ in range(2))
            observations, _ = diagnostic.observation_cases(inputs)
            obs = diagnostic.collate_v2_observations(observations[:2])
            batch = profile._fixed_numeric_replay_batch(obs, action_dim=2, device=torch.device('cpu'))
            for engine in (a, b):
                engine.replay.sample = lambda size: batch
                engine.last_total_steps = 49999  # Explicit independent diagnostic step position.
            torch.manual_seed(7)
            rng = diagnostic.rng_state()
            metric_a = a.update_once(total_steps=50000, bc_lambda=1.)
            diagnostic.restore_rng(rng)
            tensors = {}
            with diagnostic.capture_gradients(b, tensors), diagnostic.capture_forward(b.actor, tensors, 'actor_forward'):
                metric_b = b.update_once(total_steps=50000, bc_lambda=1.,
                    diagnostic_tensor_recorder=lambda name, tensor: diagnostic.flatten(tensor, name, tensors))
            self.assertEqual(metric_a, metric_b)
            diagnostic.compare_tensors(diagnostic.engine_snapshot(a), diagnostic.engine_snapshot(b), stage='observational', results=[])
            self.assertIn('target.td', tensors)
            self.assertIn('actor.spike_loss', tensors)
            self.assertIn('gradient.pre_clip.actor.projection.weight', tensors)
            self.assertIn('gradient.post_clip.actor.projection.weight', tensors)
            self.assertEqual(tensors['actor.spike_beta'].item(), .5)
            torch.testing.assert_close(tensors['actor.spike_loss'], torch.relu(tensors['actor.spike_rate'] - .17).square())
            gradient = tensors['gradient.post_clip.actor.projection.weight'].clone()
            b.actor_optimizer.zero_grad(set_to_none=True)
            self.assertIsNone(b.actor.projection.weight.grad)
            torch.testing.assert_close(gradient, tensors['gradient.post_clip.actor.projection.weight'])
            self.assertNotIn('_validate_and_clip_gradients', b.__dict__)
            existing = b._validate_and_clip_gradients
            b._validate_and_clip_gradients = existing
            with diagnostic.capture_gradients(b, {}):
                pass
            self.assertIs(b.__dict__['_validate_and_clip_gradients'], existing)

    def test_cli_dispatch_is_bounded_and_existing_mode_requires_bc(self):
        with mock.patch.object(diagnostic, 'run_snn_path_diagnostic', return_value={'cuda_status': 'not_run'}) as run:
            profile.main(['--model', 'snn', '--snn-path-consistency', '--checkpoint', 'failed.pt',
                '--validation-pool', 'pool.json', '--output-dir', 'out', '--device', 'cpu', '--snn-path-cpu-smoke'])
            self.assertEqual(run.call_args.kwargs['checkpoint'], Path('failed.pt'))
            self.assertTrue(run.call_args.kwargs['cpu_smoke'])
        with self.assertRaisesRegex(ValueError, '--bc-checkpoint'):
            profile.main(['--model', 'ann', '--output-dir', 'out'])
        with self.assertRaises(ValueError):
            diagnostic.run_snn_path_diagnostic(checkpoint='unused', validation_pool='unused', output_dir='unused', device='cpu')
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            profile.main(['--model', 'snn', '--snn-path-consistency', '--checkpoint', 'failed.pt',
                '--validation-pool', 'pool.json', '--output-dir', 'out', '--compile-actor-objective'])


if __name__ == '__main__':
    unittest.main()
