"""CPU/mock contracts only: no CUDA, real workers or performance evidence."""
from contextlib import nullcontext, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import torch

from brain_uav.scripts import v2_batch_transfer_diagnostic as d
from brain_uav.scripts import v2_concurrent_diagnostic as c
from brain_uav.scripts import profile_v2_td3 as p
from test_v2_concurrent_diagnostic import arguments, worker_summary
import test_v2_td3 as td3_fixtures
import test_profile_v2_td3 as loop_fixtures


class TestBatchPreparationDiagnostic(TestCase):
    def test_real_cpu_profiler_stops_before_outside_allocations(self):
        with TemporaryDirectory() as root:
            recorder = self.recorder(root, trace=True)
            _, batch = self.batch()
            # Real CPU profiler; only the CUDA-only startup is replaced.
            def start_cpu():
                recorder.activities = [torch.profiler.ProfilerActivity.CPU]
                recorder.profiler = torch.profiler.profile(activities=recorder.activities, profile_memory=True)
                recorder.profiler.__enter__()
            with mock.patch.object(recorder, '_start_trace', side_effect=start_cpu):
                for index in (1, 2):
                    def update(callback):
                        callback(batch, 'cpu')
                        allocation = torch.empty(1024)
                        return SimpleNamespace(actor_updated=index == 2)
                    recorder.run(update, update_index=index)
            profiler = recorder.profiler
            before_memory = [e for e in profiler.profiler.kineto_results.events() if e.name() == '[memory]']
            self.assertTrue(before_memory)
            with torch.profiler.record_function('outside_batch_sample'):
                outside = [torch.empty(2048 + i) for i in range(10)]
            del outside
            recorder.finish()
            after_events = json.loads((Path(root) / 'other/batch_preparation_trace.json').read_text())['traceEvents']
            self.assertEqual(len([e for e in profiler.profiler.kineto_results.events() if e.name() == '[memory]']), len(before_memory))
            self.assertEqual(len([e for e in after_events if e.get('name') == '[memory]']), len(before_memory))
            self.assertFalse(any(e.get('name') == 'outside_batch_sample' for e in after_events))

    def test_sample_deadline_rejects_late_start_and_crossing_completion(self):
        with TemporaryDirectory() as root:
            recorder = self.recorder(root)
            _, batch = self.batch()
            clock = [8.]
            window = {'start': 7., 'end': 10.}
            callbacks = []
            def update(callback):
                callbacks.append(callback)
                if callback: callback(batch, 'cpu')
                clock[0] += 1.5
                return SimpleNamespace(actor_updated=len(callbacks) == 2)
            with mock.patch.object(d, 'perf_counter', side_effect=lambda: clock[0]):
                for index in (1, 2, 3):
                    recorder.run(update, update_index=index, window=window)
            self.assertEqual([cb is not None for cb in callbacks], [True, True, False])
            self.assertEqual([r['within_common_window'] for r in recorder.records], [True, False])
            self.assertEqual([(r['sample_started'], r['sample_completed']) for r in recorder.records], [(8., 9.5), (9.5, 11.)])
            report = recorder.finish()
            self.assertEqual(report['valid_samples_in_window'], 1)
            self.assertFalse(report['diagnostic_valid'])

    def test_loop_action_environment_tail_cannot_fill_last_sample(self):
        class Window:
            def ready(self):
                self.window = {'start': p.perf_counter(), 'end': p.perf_counter() + 1.5}
                return self.window
            def finished(self):
                return p.perf_counter() >= self.window['end']
        with TemporaryDirectory() as root, \
                mock.patch.object(c, 'inductor_capture_snapshot', return_value={'available': False}), \
                mock.patch.object(d, 'perf_counter', side_effect=lambda: p.perf_counter()):
            recorder = self.recorder(root)
            result, _, syncs = loop_fixtures.TestProfileV2TD3().run_small_level(
                concurrent_window=Window(), gpu_td3_compute=True, batch_preparation_diagnostic=recorder)
            # Second step starts at 8 < 8.5; its action/env advance the clock
            # to 9 before update_once. The update still runs, uninstrumented.
            self.assertEqual([r['update_index'] for r in recorder.records], [8])
            self.assertEqual(result['measured_steps'], 2)
            self.assertEqual(result['concurrent_window']['completed_steps_in_window'], 1)
            self.assertEqual(sum(s['completed_steps_in_window'] for s in result['scenario_coverage']), 1)
            self.assertFalse(result['batch_preparation_diagnostic']['diagnostic_valid'])
            self.assertEqual(result['batch_preparation_diagnostic']['valid_samples_in_window'], 1)
            self.assertEqual(syncs, [])

    def test_trace_exception_and_partial_finish_stop_exactly_once(self):
        for fail in (False, True):
            with self.subTest(fail=fail), TemporaryDirectory() as root:
                recorder = self.recorder(root, trace=True)
                _, batch = self.batch()
                profiler = mock.MagicMock()
                profiler.export_chrome_trace.side_effect = lambda path: Path(path).write_text('{"traceEvents": []}')
                with mock.patch('torch.profiler.supported_activities', return_value={torch.profiler.ProfilerActivity.CUDA}), \
                        mock.patch('torch.profiler.profile', return_value=profiler), \
                        mock.patch('torch.profiler.record_function', side_effect=lambda _: nullcontext()):
                    def update(callback):
                        callback(batch, 'cpu')
                        if fail: raise ValueError('update failed')
                        return SimpleNamespace(actor_updated=False)
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'update failed'):
                            recorder.run(update, update_index=1)
                        profiler.__exit__.assert_called_once()
                    else:
                        recorder.run(update, update_index=1)
                        profiler.__exit__.assert_not_called()
                    recorder.finish()
                    recorder.finish()
                    profiler.__exit__.assert_called_once()

    def batch(self, count=2, next_count=None):
        fixture = td3_fixtures.TestV2TD3(); fixture.setUp()
        engine = fixture.make_engine()
        fixture.fill_replay(engine, counts=(0, count), next_counts=(count if next_count is None else next_count, 0))
        return engine, engine.replay.sample(2)

    def recorder(self, root, *, trace=False):
        return d.BatchPreparationDiagnostic(updates=2, trace=trace,
            output_dir=root, worker_id='ann_0', model='ann')

    def test_fourteen_order_values_metadata_and_constructor_excludes_moves(self):
        expected = ['obs.' + n for n in ('ego_features', 'goal_features', 'zone_features', 'presence_mask')]
        expected += ['action', 'reward']
        expected += ['next_obs.' + n for n in ('ego_features', 'goal_features', 'zone_features', 'presence_mask')]
        expected += ['done', 'success', 'near_goal', 'line_to_goal_safe']
        for count in (0, 1, 2):
            with self.subTest(zones=count), TemporaryDirectory() as root:
                _, batch = self.batch(count, next_count=(count + 1) % 3)
                recorder = self.recorder(root)
                outputs = []
                def update(transfer):
                    outputs.append(transfer(batch, 'cpu') if transfer else batch.to('cpu'))
                    return SimpleNamespace(actor_updated=len(outputs) % 2 == 0)
                # Each measured operation takes one clock tick. A constructor
                # containing child .to calls would instead take several ticks.
                with mock.patch.object(d, 'perf_counter', side_effect=iter(range(1000))), \
                        mock.patch('torch.profiler.profile', side_effect=AssertionError('profiler enabled')):
                    for index in (1, 2, 3, 4):
                        recorder.run(update, update_index=index)
                self.assertEqual(len(recorder.records), 2)
                for record in recorder.records:
                    self.assertEqual([r['name'] for r in record['tensor_calls']], expected)
                    self.assertEqual([r['name'] for r in record['constructors']],
                        ['obs.V2ObservationBatch', 'next_obs.V2ObservationBatch', 'V2ReplayBatch'])
                    self.assertTrue(all(r['cpu_wall_seconds'] == 1 for r in record['constructors']))
                    self.assertEqual(record['batch_preparation_wall_seconds'], 35)
                    self.assertEqual(record['uncovered_wall_seconds'], 18)
                    self.assertEqual(record['obs_shape'][1], count)
                    self.assertEqual(record['next_obs_shape'][1], (count + 1) % 3)
                    self.assertEqual(record['bytes'], sum(t.numel() * t.element_size() for t in d.fields(batch).values()))
                baseline = batch.to('cpu')
                for output in outputs:
                    self.assertEqual(output.sample_failure_fraction, baseline.sample_failure_fraction)
                    for name, tensor in d.fields(output).items():
                        source = d.fields(baseline)[name]
                        self.assertEqual(tensor.dtype, source.dtype)
                        self.assertEqual(tensor.stride(), source.stride())
                        self.assertEqual(tensor.shape, source.shape)
                        self.assertTrue(torch.equal(tensor, source))
                report = recorder.finish()
                self.assertTrue(report['diagnostic_valid'], report)
                self.assertFalse(report['valid_for_speed_comparison'])
                self.assertEqual(len(json.loads((Path(root) / report['records']).read_text())['updates']), 2)

    def test_engine_callback_only_wraps_upload_and_keeps_rng_order(self):
        engine, batch = self.batch()
        engine.gpu_td3_compute = True
        engine._gpu_update_formal_configured = True
        calls = []
        def sample(size):
            calls.append('sample'); return batch
        engine.replay.sample = sample
        def execute(actual, noise, total, bc, **kwargs):
            calls.append('compute')
            self.assertTrue(torch.equal(actual.done, batch.done))
            return SimpleNamespace(actor_updated=total % 2 == 0)
        engine._gpu_update_executor = SimpleNamespace(diagnostic=False, step_batch=execute)
        def noise(tensor):
            calls.append('noise'); return torch.zeros_like(tensor)
        with TemporaryDirectory() as root:
            recorder = self.recorder(root)
            original_transfer = recorder.transfer
            def transfer(*args):
                calls.append('upload'); return original_transfer(*args)
            with mock.patch('torch.randn_like', side_effect=noise), \
                    mock.patch.object(recorder, 'transfer', side_effect=transfer), \
                    mock.patch('torch.profiler.profile', side_effect=AssertionError('profiler enabled')):
                engine.update_once(total_steps=1)  # default does NOT call diagnostic
                self.assertEqual(calls, ['sample', 'noise', 'compute'])
                calls.clear()
                for total in (1, 2, 3):
                    recorder.run(lambda callback: engine.update_once(total_steps=total,
                        **({'batch_transfer_diagnostic': callback} if callback else {})), update_index=total)
            self.assertEqual(calls, ['sample', 'upload', 'noise', 'compute'] * 2 + ['sample', 'noise', 'compute'])
            self.assertTrue(recorder.finish()['diagnostic_valid'])

    def test_real_short_loop_excludes_warmup_caps_collection_without_extra_sync(self):
        class Window:
            def ready(self):
                self.window = {'start': p.perf_counter(), 'end': p.perf_counter() + 5.5}
                return self.window
            def finished(self):
                return p.perf_counter() >= self.window['end']
        with TemporaryDirectory() as root, \
                mock.patch.object(c, 'inductor_capture_snapshot', return_value={'available': False}), \
                mock.patch.object(d, 'perf_counter', side_effect=lambda: p.perf_counter()), \
                mock.patch('torch.profiler.profile', side_effect=AssertionError('profiler enabled')):
            recorder = self.recorder(root)
            result, _, syncs = loop_fixtures.TestProfileV2TD3().run_small_level(concurrent_window=Window(),
                gpu_td3_compute=True, batch_preparation_diagnostic=recorder)
            self.assertEqual([r['update_index'] for r in recorder.records], [8, 9])
            self.assertEqual(result['warmup_steps'], 7)
            self.assertEqual(result['measured_steps'], 6)
            self.assertTrue(all(r['batch_preparation_wall_seconds'] == .02 for r in recorder.records))
            self.assertTrue(all(r['total_source'] == 'existing_engine_batch_preparation_timer' for r in recorder.records))
            self.assertTrue(result['batch_preparation_diagnostic']['diagnostic_valid'])
            self.assertIsNone(result['timing']['throughput_environment_steps_per_second'])
            self.assertEqual(syncs, [])

    def test_worker_selection_forwarding_and_default_off(self):
        with TemporaryDirectory() as root:
            args = arguments(Path(root))
            command = c.worker_command(args, 'ann', 'ann_0', root)
            parsed = p.build_parser().parse_args(command[4:])
            self.assertEqual((parsed.batch_preparation_updates, parsed.batch_preparation_trace_updates), (0, 0))
            args.batch_preparation_updates = 32
            for name in ('ann_0', 'snn_0', 'ann_1'):
                parsed = p.build_parser().parse_args(c.worker_command(args, name[:3], name, root)[4:])
                self.assertEqual(parsed.batch_preparation_updates, 32 if name.endswith('_0') else 0)
            args.batch_preparation_updates = 0
            args.batch_preparation_trace_updates = 4
            parsed = p.build_parser().parse_args(c.worker_command(args, 'snn', 'snn_0', root)[4:])
            self.assertEqual(parsed.batch_preparation_trace_updates, 4)
            with mock.patch.object(c, 'build_parser') as parser, \
                    mock.patch.object(c, 'build_plan', return_value={}), \
                    mock.patch.object(c, 'run_batch') as batch, \
                    mock.patch.object(c.subprocess, 'Popen') as spawn, redirect_stdout(StringIO()):
                parser.return_value.parse_args.return_value = args
                self.assertEqual(c.main([]), 0)
                batch.assert_not_called(); spawn.assert_not_called()
            self.assertFalse(args.output_dir.exists())
            args.batch_preparation_workers = ['ann_1']
            with self.assertRaisesRegex(ValueError, 'worker ids'):
                c.build_plan(args)
            with self.assertRaisesRegex(ValueError, 'concurrent'):
                p.main(['--model', 'ann', '--output-dir', root, '--batch-preparation-updates', '2'])

    def test_worker_configuration_reaches_recorder_with_formal_compilation(self):
        from test_v2_bc import make_scenario_config
        class ReachedLoop(Exception): pass
        with TemporaryDirectory() as root:
            args = arguments(Path(root)); args.batch_preparation_updates = 32
            parsed = p.build_parser().parse_args(c.worker_command(args, 'snn', 'snn_0', root)[4:])
            prepared = SimpleNamespace(scenario_config=make_scenario_config(),
                uav_collision_radius=0., reward_config=SimpleNamespace())
            pool = SimpleNamespace(content_digest='digest', scenario_count=3, master_seed=20260904,
                stage_seed=1, scenario_config=prepared.scenario_config, uav_collision_radius=0., scenarios=[{}] * 3)
            with mock.patch.object(p, 'prepare_v2_stage_initialization', return_value=prepared), \
                    mock.patch.object(p, 'resolve_training_device', return_value='cuda'), \
                    mock.patch.object(p, 'load_v2_validation_pool', return_value=pool), \
                    mock.patch.object(p, 'V2ValidationPool', side_effect=lambda **kw: SimpleNamespace(**kw)), \
                    mock.patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')), \
                    mock.patch.object(p, '_run_diagnostic_level', side_effect=ReachedLoop) as loop, \
                    mock.patch('torch.profiler.profile', side_effect=AssertionError('profiler enabled')):
                with self.assertRaises(ReachedLoop): p.run_local_optimization_comparison(parsed)
            values = loop.call_args.kwargs
            recorder = values['batch_preparation_diagnostic']
            self.assertEqual((recorder.limit, recorder.trace, recorder.model, recorder.worker_id), (32, False, 'snn', 'snn_0'))
            for name in ('gpu_td3_compute', 'cpu_deduplication', 'cuda_graph_action_inference',
                         'fused_adam', 'reduce_update_stat_syncs'):
                self.assertTrue(values[name])
            self.assertEqual(values['compiled_path_profiler_updates'], 0)

    def test_trace_unavailable_and_update_exception_preserve_invalid_evidence(self):
        with TemporaryDirectory() as root:
            recorder = self.recorder(root, trace=True)
            with mock.patch('torch.profiler.supported_activities', return_value={torch.profiler.ProfilerActivity.CPU}):
                with self.assertRaisesRegex(RuntimeError, 'CUDA activity'):
                    recorder.run(lambda callback: None, update_index=1)
            report = recorder.finish()
            self.assertFalse(report['diagnostic_valid'])
            self.assertIsNone(report['trace'])
            self.assertIn('CUDA activity', report['error']['message'])
        with TemporaryDirectory() as root:
            recorder = self.recorder(root)
            with self.assertRaisesRegex(ValueError, 'broken update'):
                recorder.run(mock.Mock(side_effect=ValueError('broken update')), update_index=1)
            self.assertFalse(recorder.finish()['diagnostic_valid'])
            saved = json.loads((Path(root) / 'review/batch_preparation.json').read_text())
            self.assertEqual(saved['error']['message'], 'broken update')

    def test_bounded_trace_cuda_evidence_and_missing_activity_is_unknown(self):
        for events, valid, export_error in (([{'cat': 'gpu_memcpy', 'name': 'Memcpy HtoD', 'args': {'bytes': 128}},
                                {'cat': 'cuda_runtime', 'name': 'cudaStreamSynchronize'}], True, False),
                                           ([], False, False), ([], False, True)):
            with self.subTest(valid=valid), TemporaryDirectory() as root:
                _, batch = self.batch()
                recorder = self.recorder(root, trace=True)
                profiler = mock.MagicMock()
                profiler.export_chrome_trace.side_effect = lambda path: Path(path).write_text(json.dumps({'traceEvents': events}))
                if export_error:
                    profiler.export_chrome_trace.side_effect = RuntimeError('export unavailable')
                with mock.patch('torch.profiler.supported_activities', return_value={torch.profiler.ProfilerActivity.CUDA}), \
                        mock.patch('torch.profiler.profile', return_value=profiler) as factory, \
                        mock.patch('torch.profiler.record_function', side_effect=lambda _: nullcontext()):
                    for index in (1, 2, 3):
                        def update(callback):
                            if callback: callback(batch, 'cpu')
                            return SimpleNamespace(actor_updated=index % 2 == 0)
                        recorder.run(update, update_index=index)
                    report = recorder.finish()
                factory.assert_called_once()
                self.assertEqual(profiler.step.call_count, 1)
                profiler.__exit__.assert_called_once()
                profiler.export_chrome_trace.assert_called_once()
                self.assertEqual(report['diagnostic_valid'], valid)
                if export_error:
                    self.assertIsNone(report['trace'])
                    self.assertIn('export unavailable', report['error']['message'])
                else:
                    self.assertEqual(report['trace']['copy_bytes'], 128 if valid else None)
                    self.assertEqual(report['trace']['driver_queueing'], 'unknown')

    def test_instrumented_validity_continues_two_then_eight_but_failures_stop(self):
        window = {'start': 100., 'end': 110.}
        summary = worker_summary(window)
        run = summary['runs']['compute'][0]
        run['batch_preparation_diagnostic'] = {'diagnostic_valid': True, 'invalid_reasons': []}
        result = c.summarize_worker(summary, window)
        self.assertTrue(result['diagnostic_valid'])
        self.assertFalse(result['valid_for_speed_comparison'])
        run['local_optimizations']['gpu_td3_compute']['measurement_new_captures'] = 1
        self.assertFalse(c.summarize_worker(summary, window)['diagnostic_valid'])
        for fails in (False, True):
            with self.subTest(fails=fails), TemporaryDirectory() as root:
                args = arguments(Path(root)); args.launch = True; args.repeats = 1
                args.batch_preparation_updates = 32
                counts = []
                def batch(args, count, repeat, root):
                    counts.append(count)
                    return {'concurrency': count, 'repeat': repeat, 'instrumented': True,
                        'diagnostic_valid': not fails, 'valid_for_speed_comparison': False}
                with mock.patch.object(c, 'build_parser') as parser, \
                        mock.patch.object(c, 'build_plan', return_value={}), \
                        mock.patch.object(c, 'run_batch', side_effect=batch), redirect_stdout(StringIO()):
                    parser.return_value.parse_args.return_value = args
                    self.assertEqual(c.main([]), 1 if fails else 0)
                self.assertEqual(counts, [2] if fails else [2, 8])
                saved = c.read_json(args.output_dir / 'review/summary.json')
                self.assertEqual(saved['diagnostic_valid'], not fails)
                self.assertFalse(saved['valid_for_speed_comparison'])

    def test_mock_dispatch_two_and_eight_keeps_unselected_workers_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); args = arguments(root)
            args.batch_preparation_updates = 2
            class Process:
                def __init__(self, command, **kwargs):
                    self.command, self.returncode = command, None
                    self.control = Path(command[command.index('--concurrent-window-dir') + 1])
                    self.name = command[command.index('--concurrent-worker-id') + 1]
                    c.write_json(self.control / (self.name + '.ready.json'), {})
                def poll(self):
                    if self.returncode is None and (self.control / 'window.json').exists():
                        summary = worker_summary(c.read_json(self.control / 'window.json'))
                        if '--batch-preparation-updates' in self.command:
                            summary['runs']['compute'][0]['batch_preparation_diagnostic'] = {
                                'diagnostic_valid': True, 'invalid_reasons': []}
                        target = Path(self.command[self.command.index('--output-dir') + 1]) / 'other'
                        target.mkdir(parents=True)
                        c.write_json(target / 'local_optimization_summary.json', summary)
                        self.returncode = 0
                    return self.returncode
                def terminate(self): self.returncode = -15
                def wait(self, timeout): return self.returncode
            with mock.patch.object(c.subprocess, 'Popen', side_effect=Process):
                for count in (2, 8):
                    report = c.run_batch(args, count, 0, root / str(count))
                    self.assertTrue(report['diagnostic_valid'], report)
                    self.assertTrue(report['completed'])
                    self.assertFalse(report['valid_for_speed_comparison'])
                    self.assertEqual(len(report['workers']), count)
                    self.assertEqual(sum(w['instrumented'] for w in report['workers'].values()), 2)
                    self.assertEqual(report['batch_steps_per_second'], count * 3.)
