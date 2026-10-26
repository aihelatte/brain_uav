"""CPU/mock transfer orchestration only; not CUDA numerical or speed evidence."""
import gc
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock
import torch
from brain_uav.scripts import profile_v2_td3 as p, v2_concurrent_diagnostic as c
from brain_uav.scripts import v2_gpu_compute_diagnostic as d
import test_v2_gpu_compute as fixtures
from test_v2_concurrent_diagnostic import arguments


class TestGPUComputeTransfer(TestCase):
    def test_reference_requires_real_compute_blocking_provenance_and_contract(self):
        with TemporaryDirectory() as root:
            args = SimpleNamespace(gpu_compute_reference=Path(root)/'reference.pt', non_blocking_batch_transfer=True)
            contract = {'model': 'snn', 'schedule': 'smooth-mid'}
            for provenance in (None, {'gpu_td3_compute': False, 'non_blocking_batch_transfer': False},
                               {'gpu_td3_compute': True, 'non_blocking_batch_transfer': True}):
                torch.save({'contract': contract, 'transfer_reference': provenance}, args.gpu_compute_reference)
                with self.assertRaisesRegex(ValueError, 'blocking reference'):
                    d._load_compute_reference(args, contract)
            torch.save({'contract': contract, 'transfer_reference': {
                'gpu_td3_compute': True, 'non_blocking_batch_transfer': False}}, args.gpu_compute_reference)
            self.assertEqual(d._load_compute_reference(args, contract)['contract'], contract)
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                d._load_compute_reference(args, {'schedule': 'wrong'})

    def test_source_lifetime_order_flags_and_exception_drain(self):
        for enabled in (False, True):
            for failure in (None, 'upload', 'compute', 'drain'):
                with self.subTest(enabled=enabled, failure=failure):
                    engine = fixtures.TestGPUCompute().engine()
                    output = engine.replay.sample(2)
                    engine.gpu_td3_compute = True; engine._gpu_update_formal_configured = True
                    engine.non_blocking_batch_transfer = enabled
                    calls = []
                    class Source:
                        def to(self, device, **kwargs):
                            calls.append(('upload', kwargs.get('non_blocking', False)))
                            if failure == 'upload': raise ValueError('upload')
                            return output
                    source = Source(); ref = weakref.ref(source)
                    pending = [source]; del source
                    def sample(size): calls.append('sample'); return pending.pop()
                    engine.replay.sample = sample
                    def noise(value): calls.append('noise'); return torch.zeros_like(value)
                    def compute(*args, **kwargs):
                        calls.append('compute'); gc.collect()
                        self.assertIsNotNone(ref())
                        if failure in ('compute', 'drain'): raise ValueError('compute')
                        return SimpleNamespace(actor_updated=False)
                    engine._gpu_update_executor = SimpleNamespace(diagnostic=False, step_batch=compute, metadata=lambda: {})
                    stream = mock.Mock()
                    def drain():
                        self.assertIsNotNone(ref())
                        if failure == 'drain': raise RuntimeError('drain failed')
                    stream.synchronize.side_effect = drain
                    with mock.patch('torch.cuda.current_stream', return_value=stream), \
                            mock.patch('torch.randn_like', side_effect=noise):
                        if failure:
                            with self.assertRaises((ValueError, RuntimeError)):
                                engine.update_once(total_steps=1)
                        else:
                            engine.update_once(total_steps=1)
                            self.assertEqual(calls, ['sample', ('upload', enabled), 'noise', 'compute'])
                        if enabled and failure == 'drain':
                            self.assertIsNotNone(ref())
                            with self.assertRaisesRegex(RuntimeError, 'source retained'):
                                engine.update_once(total_steps=2)
                    self.assertEqual(stream.synchronize.call_count, int(enabled and bool(failure)))
                    if not failure:
                        meta = engine.local_optimization_metadata()['non_blocking_batch_transfer']
                        self.assertEqual(meta['requested'], enabled)
                        self.assertEqual(meta['effective'], enabled)
                        self.assertEqual(meta['transfer_calls'], 1)
                        self.assertIn('non_blocking=True' if enabled else 'blocking', meta['actual_path'])

    def test_concurrent_cli_is_explicit_default_off_and_forwards_flag(self):
        with TemporaryDirectory() as root:
            args = arguments(Path(root))
            self.assertFalse(args.non_blocking_batch_transfer)
            for enabled in (False, True):
                args.non_blocking_batch_transfer = enabled
                parsed = p.build_parser().parse_args(c.worker_command(args, 'snn', 'snn_0', root)[4:])
                self.assertEqual(parsed.non_blocking_batch_transfer, enabled)
                self.assertTrue(parsed.gpu_td3_compute)
            args.batch_preparation_updates = 2
            with self.assertRaisesRegex(ValueError, 'separately'):
                c.build_plan(args)

    def test_worker_passes_nonblocking_to_real_loop_configuration(self):
        from test_v2_bc import make_scenario_config
        class ReachedLoop(Exception): pass
        with TemporaryDirectory() as root:
            args = arguments(Path(root)); args.non_blocking_batch_transfer = True
            parsed = p.build_parser().parse_args(c.worker_command(args, 'snn', 'snn_0', root)[4:])
            prepared = SimpleNamespace(scenario_config=make_scenario_config(),
                uav_collision_radius=0., reward_config=SimpleNamespace())
            pool = SimpleNamespace(content_digest='digest', scenario_count=3, master_seed=20260904,
                stage_seed=1, scenario_config=prepared.scenario_config, uav_collision_radius=0., scenarios=[{}]*3)
            with mock.patch.object(p, 'prepare_v2_stage_initialization', return_value=prepared), \
                    mock.patch.object(p, 'resolve_training_device', return_value='cuda'), \
                    mock.patch.object(p, 'load_v2_validation_pool', return_value=pool), \
                    mock.patch.object(p, 'V2ValidationPool', side_effect=lambda **kw: SimpleNamespace(**kw)), \
                    mock.patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')), \
                    mock.patch.object(p, '_run_diagnostic_level', side_effect=ReachedLoop) as loop:
                with self.assertRaises(ReachedLoop): p.run_local_optimization_comparison(parsed)
            for key in ('non_blocking_batch_transfer', 'gpu_td3_compute', 'cpu_deduplication',
                        'fused_adam', 'cuda_graph_action_inference'):
                self.assertTrue(loop.call_args.kwargs[key], key)

    def test_correctness_reference_selection_and_no_silent_eager_fallback(self):
        base = ['--model', 'ann', '--device', 'cuda', '--output-dir', 'unused',
                '--bc-checkpoint', 'bc.pt', '--validation-pool', 'pool.json']
        for flags in (['--gpu-compute-write-reference'],
                      ['--gpu-compute-worker', 'correctness', '--gpu-td3-compute',
                       '--gpu-compute-write-reference', '--non-blocking-batch-transfer'],
                      ['--gpu-compute-comparison', '--non-blocking-batch-transfer']):
            with self.assertRaises(ValueError): d.run(p.build_parser().parse_args(base + flags))
        for flags in (['--gpu-compute-write-reference'], ['--non-blocking-batch-transfer', '--gpu-compute-reference', 'reference.pt']):
            parsed = p.build_parser().parse_args(base + ['--gpu-compute-worker', 'correctness', '--gpu-td3-compute'] + flags)
            with mock.patch.object(p, 'run_local_optimization_comparison', return_value={}) as run:
                d.run(parsed)
            self.assertIs(run.call_args.args[0], parsed)
