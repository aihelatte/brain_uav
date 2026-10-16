"""Execution defaults, never CUDA performance evidence."""
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

from brain_uav.scripts import train_v2_td3 as stage
from brain_uav.scripts import run_v2_td3_curriculum as curriculum


class TestAccelerationDefaults(TestCase):
    def test_failed_checkpoint_names(self):
        for value, expected in [('model', 'model_failed.pt'), ('model.pt', 'model_failed.pt'),
                                ('model.pth', 'model_failed.pth')]:
            self.assertEqual(stage._failed_checkpoint_path(Path(value)), Path(expected))

    def test_formal_main_omitted_defaults_reach_real_construction(self):
        from contextlib import ExitStack, redirect_stdout
        from io import StringIO
        from brain_uav.config import ScenarioConfig, RewardConfig
        class ReachedTrainer(Exception): pass
        for device, extra, compute, cpu in (
            ('cuda', [], True, True),
            ('cuda', ['--no-gpu-td3-compute','--no-cpu-deduplication'], False, False),
            ('cpu', [], False, True),
        ):
            with self.subTest(device=device, extra=extra), TemporaryDirectory() as directory, ExitStack() as stack:
                scenario = ScenarioConfig()
                prepared = SimpleNamespace(scenario_config=scenario, reward_config=RewardConfig(),
                    uav_collision_radius=0., bc_initialization=SimpleNamespace(actor=object()))
                components = SimpleNamespace(engine=object(), scenario_generators={}, selector=None,
                                             exploration_rng=None)
                stack.enter_context(mock.patch.object(stage, 'resolve_training_device', return_value=device))
                stack.enter_context(mock.patch('brain_uav.trainers.v2_gpu_update.require_cuda_abi'))
                stack.enter_context(mock.patch.object(stage, 'prepare_v2_stage_initialization', return_value=prepared))
                stack.enter_context(mock.patch.object(stage, 'validate_v2_prepared_stage_initialization'))
                stack.enter_context(mock.patch.object(stage, 'validate_policy_request'))
                stack.enter_context(mock.patch.object(stage, 'load_v2_validation_pool', return_value=SimpleNamespace(scenario_count=100)))
                builder = stack.enter_context(mock.patch.object(stage, 'build_v2_stage_engine', return_value=components))
                stack.enter_context(mock.patch.object(stage, '_configure_stage_compilation', return_value={}))
                trainer = stack.enter_context(mock.patch.object(stage, 'V2FormalStageTrainer', side_effect=ReachedTrainer))
                real_run = stage.run_v2_td3_stage
                def without_reports(**kwargs):
                    self.assertIsNone(kwargs['gpu_td3_compute'] if not extra else None)
                    return real_run(**kwargs, reporting=False)
                stack.enter_context(mock.patch.object(stage, 'run_v2_td3_stage', side_effect=without_reports))
                stack.enter_context(redirect_stdout(StringIO()))
                with self.assertRaises(ReachedTrainer):
                    stage.main(['--stage','easy','--init-checkpoint','bc.pt','--validation-pool','pool.json',
                        '--output',str(Path(directory)/'out.pt'),'--metrics-out',str(Path(directory)/'metrics.json'),
                        '--device',device, *extra])
                self.assertIs(builder.call_args.kwargs['gpu_td3_compute'], compute)
                self.assertIs(trainer.call_args.kwargs['cpu_deduplication'], cpu)

    def test_cpu_gate_ignores_policy_drift_but_rejects_environment_difference(self):
        from brain_uav.scripts import v2_cpu_dedup_diagnostic as d
        row = dict.fromkeys(('step','measured','action_bits','input','output','state','outcome'), 'same')
        a = dict(row, rng='before', counts=[1])
        b = dict(row, rng='after', counts=[2])
        self.assertTrue(d.compare_environment_evidence([a],[b])['passed'])
        self.assertFalse(d.compare_evidence([a],[b])['passed'])
        b['state'] = 'changed'
        self.assertFalse(d.compare_environment_evidence([a],[b])['passed'])

    def test_known_engineering_scope_is_not_an_automatic_waiver(self):
        from brain_uav.scripts import v2_cpu_dedup_diagnostic as d, profile_v2_td3 as p
        for failure in ({'passed':False,'known_ann_gradient':True}, {'passed':False,'new_failure':True}):
            self.assertFalse(d.require_strict(failure)['strict_passed'])
            verdict = d.require_strict(failure, basis='ann-existing-engineering')
            self.assertFalse(verdict['engineering_accepted']); self.assertIn('gap',verdict)
        self.assertTrue(d.require_strict({'passed':True})['strict_passed'])
        with TemporaryDirectory() as directory:
            args = p.build_parser().parse_args(['--model','ann','--output-dir',str(Path(directory)/'run')])
            args.acceptance_basis = 'ann-existing-engineering'
            with mock.patch.object(d.subprocess, 'run') as worker:
                with self.assertRaisesRegex(RuntimeError, 'Timing blocked'): d.run(args)
            worker.assert_not_called()

    def test_same_input_action_and_rng_failures_are_not_waived(self):
        import numpy as np
        import torch
        from brain_uav.scripts.v2_cpu_dedup_diagnostic import checked_action
        for kind in ('action', 'rng'):
            def select(obs, *, exploration_rng, cpu_deduplication, **kwargs):
                if cpu_deduplication:
                    if kind == 'rng': exploration_rng.random()
                    if kind == 'action': return np.array([1.], dtype=np.float32)
                return np.array([0.], dtype=np.float32)
            engine = SimpleNamespace(device=torch.device('cpu'), select_action=select)
            with self.subTest(kind=kind), self.assertRaisesRegex(AssertionError, 'timing skipped'):
                checked_action(engine, object(), np.random.default_rng(7))

    def test_snapshot_wrapper_uses_existing_owned_snapshots_and_restores_hook(self):
        import torch
        from brain_uav.scripts import profile_v2_td3 as p, v2_gpu_td3_diagnostic as old
        from brain_uav.scripts.v2_cpu_dedup_diagnostic import capture_controlled_worker
        source = torch.tensor([1.])
        def snapshot(*args):
            return {'actor.weight':source.detach().cpu().clone()}, {}
        def worker(argv):
            self.assertEqual(argv, ['--model','snn'])
            old.snapshot(object())
            source.fill_(2.)
            old.snapshot(object())
        with TemporaryDirectory() as directory, mock.patch.object(old, 'snapshot', side_effect=snapshot) as original, \
                mock.patch.object(p, 'main', side_effect=worker):
            output = Path(directory)/'other'/'snapshots.pt'
            capture_controlled_worker(['--model','snn'], output)
            self.assertIs(old.snapshot, original)
            rows = torch.load(output, weights_only=False)
            self.assertEqual([row['actor.weight'].item() for row in rows], [1.,2.])

    def test_adopted_cross_version_comparison_rejects_new_difference(self):
        import torch
        from brain_uav.scripts.v2_cpu_dedup_diagnostic import compare_adopted_snapshots
        a = [{'actor.weight':torch.tensor([1.])}]
        self.assertTrue(compare_adopted_snapshots(a,a)['passed'])
        self.assertFalse(compare_adopted_snapshots(a,[{'actor.weight':torch.tensor([1.01])}])['passed'])
        self.assertFalse(compare_adopted_snapshots(a,[])['passed'])

    def test_strict_failure_stops_acceptance_before_any_timing(self):
        from brain_uav.scripts import profile_v2_td3 as profile, v2_cpu_dedup_diagnostic as diagnostic
        for failure in ('exit_code', 'missing_report', 'failed_report'):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                output = Path(directory)/'run'
                args = profile.build_parser().parse_args([
                    '--model','ann','--device','cuda','--gpu-td3-compute',
                    '--bc-checkpoint','bc.pt','--validation-pool','pool.json',
                    '--output-dir',str(output)])
                args.acceptance_comparison = 'B'
                args.baseline_source = None
                def worker(command, **kwargs):
                    self.assertEqual(command[command.index('--gpu-compute-worker')+1], 'correctness')
                    self.assertNotIn('--gpu-td3-compute', command)
                    if failure == 'failed_report':
                        path = Path(command[command.index('--output-dir')+1])
                        path.mkdir(parents=True)
                        (path/'correctness.json').write_text('{"passed":false}')
                    return SimpleNamespace(returncode=1 if failure=='exit_code' else 0)
                with mock.patch.object(diagnostic.subprocess, 'run', side_effect=worker) as run:
                    with self.assertRaisesRegex(RuntimeError, 'timing skipped'):
                        diagnostic.run(args)
                self.assertEqual(run.call_count, 1)
                report = json.loads((output/'review/summary.json').read_text())
                self.assertFalse(report['passed'])
                self.assertFalse(report['strict_td3_passed'])
                self.assertIn('error', report)

    def test_formal_defaults_and_explicit_off(self):
        for module, argv, api in (
            (stage, ['--stage','easy','--init-checkpoint','bc.pt','--output','out.pt',
                     '--metrics-out','out.json','--validation-pool','pool.json'], stage.run_v2_td3_stage),
            (curriculum, ['--bc-checkpoint','bc.pt','--output-root','run',
                          '--validation-pool-dir','pools'], curriculum.run_v2_curriculum),
        ):
            args = module.build_parser().parse_args(argv)
            self.assertTrue(args.cpu_deduplication)
            self.assertIsNone(args.gpu_td3_compute)
            args = module.build_parser().parse_args(argv + ['--no-cpu-deduplication','--no-gpu-td3-compute'])
            self.assertFalse(args.cpu_deduplication)
            self.assertFalse(args.gpu_td3_compute)
            self.assertTrue(inspect.signature(api).parameters['cpu_deduplication'].default)
            self.assertIsNone(inspect.signature(api).parameters['gpu_td3_compute'].default)

    def test_resolution_cpu_unsupported_cuda_and_explicit_requests(self):
        resolve = stage.resolve_v2_gpu_compute
        self.assertFalse(resolve(None, device='cpu', fused_adam=True)['effective'])
        self.assertFalse(resolve(False, device='cpu', fused_adam=True)['effective'])
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            resolve(True, device='cpu', fused_adam=True)
        with mock.patch('brain_uav.trainers.v2_gpu_update.require_cuda_abi', side_effect=RuntimeError('ABI unsupported')):
            result = resolve(None, device='cuda', fused_adam=True)
            self.assertFalse(result['effective']); self.assertIn('ABI', result['reason'])
            with self.assertRaisesRegex(RuntimeError, 'ABI'):
                resolve(True, device='cuda', fused_adam=True)
        with mock.patch('brain_uav.trainers.v2_gpu_update.require_cuda_abi'):
            self.assertTrue(resolve(None, device='cuda', fused_adam=True)['effective'])

    def test_retired_parameters_are_removed_from_engine(self):
        from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
        parameters = inspect.signature(V2TD3UpdateEngine).parameters
        for name in ('pinned_batch_transfer','aggregate_relation_values_first',
                     'cuda_graph_critic_tail','compile_critic_update_front','compile_actor_objective'):
            self.assertNotIn(name, parameters)
