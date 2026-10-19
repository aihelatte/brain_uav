"""CPU/mock orchestration checks only; no training processes or CUDA evidence."""
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from brain_uav.scripts import v2_concurrent_diagnostic as c
from brain_uav.scripts import profile_v2_td3 as p
import test_profile_v2_td3 as fixtures


def arguments(root):
    return c.build_parser().parse_args([
        '--output-dir', str(root / 'output'), '--ann-bc-checkpoint', str(root / 'ann.pt'),
        '--snn-bc-checkpoint', str(root / 'snn.pt'), '--validation-pool', str(root / 'pool.json'),
        '--expected-pool-digest', 'digest', '--ann-actor-lr', '0.000155',
        '--snn-actor-lr', '0.000825', '--bc-schedule', 'smooth-mid',
        '--replay-capacity', '123456', '--critic-lr', '.0003', '--alpha', '2.7'])


def worker_summary(window):
    run = {'local_optimizations': {'gpu_td3_compute': {'effective': True,
        'measurement_new_captures': 0, 'capture_seconds': .5, 'initialization_seconds': .2}},
        'concurrent_window': dict(window, actual_start=window['start'] + .01,
            actual_end=window['end'] + .02, completed_steps_in_window=90),
        'critic_encoder_compile': {'measurement_new_graph_count': 0,
            'cuda_graph_action_inference_evidence': {'verified': True},
            'registration_wall_seconds': .1, 'warmup_wall_seconds': .2},
        'concurrent_inductor_captures': {'before': {'available': True, 'manager_id': 1, 'recordings': 2},
            'after': {'available': True, 'manager_id': 1, 'recordings': 2}},
        'measurement_graph_replays': 91, 'cpu_deduplication': True,
        'environment_execution': {'cpu_deduplication': True, 'cache_ellipsoid_segment_clearance': True},
        'scenario_coverage': [{'scenario_id': 'test', 'measured_steps': 91,
            'completed_steps_in_window': 90}],
        'measured_steps': 91, 'loop_warmup_wall_seconds': 1.,
        'timing': {'total_wall_seconds': 10., 'replay_sample_wall_seconds': .4,
            'td3_update_breakdown': {'child': 'not additive'},
            **{key: 1. for key in ('scenario_reset_wall_seconds', 'pre_action_geometry_wall_seconds',
                'action_inference_wall_seconds', 'environment_step_wall_seconds',
                'replay_write_wall_seconds', 'td3_update_wall_seconds')}},
        'batch_transfer_window_completion': {'wait_seconds': .01},
        'replay_occupancy': {'capacity': 123456, 'size': 300},
        'server_load_before': {}, 'server_load_after': {}}
    return {'config': {}, 'runs': {'compute': [run]}}


class TestConcurrentDiagnostic(unittest.TestCase):
    def test_plan_default_does_not_spawn_or_write_and_validates_digest(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            args = arguments(root)
            for model in ('ann', 'snn'):
                (root / f'{model}.pt').write_bytes(b'mock checkpoint')
            pool = SimpleNamespace(content_digest='digest', scenario_count=3, scenarios=[
                {'scenario_id': str(i), 'payload': {'zones': []}} for i in range(3)])
            with mock.patch('brain_uav.trainers.v2_validation.load_v2_validation_pool', return_value=pool) as load:
                plan = c.build_plan(args)
                load.assert_called_once_with(args.validation_pool, expected_level='easy', expected_master_seed=20260904)
                self.assertEqual(plan['models']['snn']['config']['actor_lr'], .000825)
                self.assertEqual(plan['models']['ann']['config']['replay_capacity'], 123456)
                self.assertEqual(plan['bc_schedule']['name'], 'smooth-mid')
                pool.content_digest = 'wrong'
                with self.assertRaisesRegex(ValueError, 'digest'):
                    c.build_plan(args)
            with mock.patch.object(c, 'build_plan', return_value=plan), mock.patch.object(c, 'run_batch') as batch, \
                    mock.patch.object(c.subprocess, 'Popen') as spawn, redirect_stdout(StringIO()):
                argv = c.worker_command(args, 'ann', 'ann_0', root)  # command construction has no effects
                self.assertIn('--gpu-td3-compute', argv)
                cli = ['--output-dir', str(args.output_dir), '--ann-bc-checkpoint', 'ann',
                    '--snn-bc-checkpoint', 'snn', '--validation-pool', 'pool', '--expected-pool-digest', 'digest',
                    '--ann-actor-lr', '.000155', '--snn-actor-lr', '.000825', '--bc-schedule', 'smooth-mid']
                self.assertEqual(c.main(cli), 0)
                spawn.assert_not_called()
                batch.assert_not_called()
                self.assertFalse(args.output_dir.exists())

    def test_worker_command_reaches_real_formal_config_and_timing_callable(self):
        class ReachedLoop(Exception): pass
        with TemporaryDirectory() as d:
            root = Path(d)
            args = arguments(root)
            for model in ('ann', 'snn'):
                for schedule in ('smooth-mid', 'smooth-mid-strong'):
                    args.bc_schedule = schedule
                    command = c.worker_command(args, model, model + '_0', root / schedule)
                    parsed = p.build_parser().parse_args(command[4:])
                    from test_v2_bc import make_scenario_config
                    prepared = SimpleNamespace(scenario_config=make_scenario_config(),
                        uav_collision_radius=0., reward_config=SimpleNamespace())
                    pool = SimpleNamespace(content_digest='digest', scenario_count=3,
                        master_seed=20260904, stage_seed=1, scenario_config=prepared.scenario_config,
                        uav_collision_radius=0., scenarios=[{}] * 3)
                    with mock.patch.object(p, 'prepare_v2_stage_initialization', return_value=prepared), \
                            mock.patch.object(p, 'resolve_training_device', return_value='cuda'), \
                            mock.patch.object(p, 'load_v2_validation_pool', return_value=pool), \
                            mock.patch.object(p, 'V2ValidationPool', side_effect=lambda **kw: SimpleNamespace(**kw)), \
                            mock.patch('subprocess.run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')), \
                            mock.patch.object(p, '_run_diagnostic_level', side_effect=ReachedLoop) as loop:
                        with self.assertRaises(ReachedLoop):
                            p.run_local_optimization_comparison(parsed)
                    config = loop.call_args.kwargs
                    self.assertEqual(config['formal_config'].actor_lr, getattr(args, model + '_actor_lr'))
                    self.assertEqual(config['formal_config'].critic_lr, .0003)
                    self.assertEqual(config['formal_config'].actor_rl_scale_alpha, 2.7)
                    self.assertEqual(config['formal_config'].replay_capacity, 123456)
                    self.assertEqual(config['bc_schedule'], schedule)
                    for key in ('gpu_td3_compute', 'cpu_deduplication', 'cache_ellipsoid_segment_clearance',
                                'fused_adam', 'cuda_graph_action_inference', 'reduce_update_stat_syncs'):
                        self.assertTrue(config[key], key)
                    self.assertEqual(config['concurrent_window'].worker_id, model + '_0')
                    self.assertFalse(config['environment_performance_diagnostic'])
                    self.assertIsNone(config['deduplication_evidence'])

    def test_serial_gpu_diagnostic_forwards_explicit_config_to_both_phases(self):
        from brain_uav.scripts import v2_gpu_compute_diagnostic as diagnostic
        with TemporaryDirectory() as directory:
            args = p.build_parser().parse_args(['--model', 'snn', '--device', 'cuda',
                '--gpu-compute-comparison', '--bc-checkpoint', 'bc.pt', '--validation-pool', 'pool.json',
                '--output-dir', directory, '--comparison-actor-lr', '.000825',
                '--comparison-critic-lr', '.0003', '--comparison-alpha', '2.7',
                '--comparison-bc-schedule', 'smooth-mid-strong'])
            calls = []
            def child(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0 if len(calls) < 3 else 1)
            with mock.patch.object(diagnostic.subprocess, 'run', side_effect=child):
                with self.assertRaisesRegex(RuntimeError, 'Timing worker failed'):
                    diagnostic.run(args)
            self.assertEqual(len(calls), 3)
            for command in calls:
                parsed = p.build_parser().parse_args(command[4:])
                self.assertEqual(parsed.comparison_actor_lr, .000825)
                self.assertEqual(parsed.comparison_critic_lr, .0003)
                self.assertEqual(parsed.comparison_alpha, 2.7)
                self.assertEqual(parsed.comparison_bc_schedule, 'smooth-mid-strong')

    def test_existing_loop_uses_deadline_not_step_cap_and_cycles_scenes(self):
        class Window:
            def ready(self):
                self.window = {'start': p.perf_counter(), 'end': p.perf_counter() + 9.5}
                return self.window
            def finished(self):
                return p.perf_counter() >= self.window['end']
        with mock.patch.object(c, 'inductor_capture_snapshot', return_value={'available': False}):
            result, _, _ = fixtures.TestProfileV2TD3().run_small_level(steps=5, concurrent_window=Window())
        self.assertEqual(result['measured_steps'], 10)
        self.assertEqual(result['concurrent_window']['completed_steps_in_window'], 9)
        self.assertEqual([r['measured_steps'] for r in result['scenario_coverage']], [4, 4, 2])
        self.assertEqual([r['completed_steps_in_window'] for r in result['scenario_coverage']], [4, 4, 1])

    def test_scene_with_only_outside_window_tail_is_not_covered(self):
        for duration, expected_inside, expected_valid in ((4.5, [2, 2, 0], False),
                                                          (5.5, [2, 2, 1], True)):
            with self.subTest(duration=duration):
                class Window:
                    def ready(self):
                        self.window = {'start': p.perf_counter(), 'end': p.perf_counter() + duration}
                        return self.window
                    def finished(self):
                        return p.perf_counter() >= self.window['end']
                with mock.patch.object(c, 'inductor_capture_snapshot', return_value={'available': False}):
                    result, _, _ = fixtures.TestProfileV2TD3().run_small_level(steps=5, concurrent_window=Window())
                self.assertEqual([s['completed_steps_in_window'] for s in result['scenario_coverage']], expected_inside)
                self.assertEqual(result['concurrent_window']['completed_steps_in_window'], sum(expected_inside))
                self.assertEqual(result['measured_steps'], sum(expected_inside) + 1)
                # Use the real loop's counts/window; other validity evidence remains a mock fixture.
                summary = worker_summary(result['concurrent_window'])
                run = summary['runs']['compute'][0]
                for key in ('concurrent_window', 'scenario_coverage', 'measured_steps'):
                    run[key] = result[key]
                report = c.summarize_worker(summary, result['concurrent_window'])
                self.assertEqual(report['valid_for_speed_comparison'], expected_valid)
                self.assertEqual(report['window_steps_per_second'], sum(expected_inside) / duration)
                if not expected_valid:
                    self.assertEqual([s['measured_steps'] for s in result['scenario_coverage']], [2, 2, 1])
                    self.assertEqual(result['concurrent_window']['actual_start'], 7.)
                    self.assertEqual(result['concurrent_window']['actual_end'], 12.)
                    self.assertTrue(any('window' in reason.lower() and 'scene' in reason.lower()
                        for reason in report['invalid_reasons']))

    def test_worker_barrier_wait_timeout_abort_and_shared_times(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            c.write_json(root / 'window.json', {'start': 5., 'end': 10.})
            worker = c.WorkerWindow(root, 'ann_0', timeout=20.)
            # Both workers consume the exact published window, never a local duration.
            with mock.patch.object(c.time, 'perf_counter', return_value=5.):
                self.assertEqual(worker.ready(), {'start': 5., 'end': 10.})
                self.assertFalse(worker.finished())
            with mock.patch.object(c.time, 'perf_counter', return_value=10.):
                self.assertTrue(worker.finished())
            self.assertTrue((root / 'ann_0.ready.json').exists())
            (root / 'window.json').unlink()
            with mock.patch.object(c.time, 'perf_counter', side_effect=[0., 0., 21.]):
                with self.assertRaises(TimeoutError): worker.ready()
            c.write_json(root / 'abort.json', {'error': 'peer failed'})
            with self.assertRaisesRegex(RuntimeError, 'peer failed'): worker.ready()
            process = mock.Mock()
            process.poll.return_value = None
            with mock.patch.object(c.time, 'perf_counter', side_effect=[0., 2.]):
                with self.assertRaises(TimeoutError):
                    c.wait_workers({'owned': process}, lambda _: False, timeout=1.)
            process.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, 'owned.*1'):
                c.wait_workers({'owned': process}, lambda _: False, timeout=1.)

    def test_summary_parent_child_and_invalid_measurements(self):
        window = {'start': 100., 'end': 110.}
        summary = worker_summary(window)
        result = c.summarize_worker(summary, window)
        self.assertTrue(result['valid_for_speed_comparison'])
        self.assertEqual(result['other_wall_seconds'], 4.)  # child sample .4 NOT subtracted again
        self.assertEqual(result['window_steps_per_second'], 9.)
        self.assertEqual(result['steps'], 91)  # tail step not counted in window throughput
        changes = [
            lambda r: r['critic_encoder_compile'].update(measurement_new_graph_count=1),
            lambda r: r['local_optimizations']['gpu_td3_compute'].update(measurement_new_captures=1),
            lambda r: r['concurrent_inductor_captures']['after'].update(recordings=3),
            lambda r: r['concurrent_inductor_captures']['after'].update(available=False),
            lambda r: r.update(measurement_graph_replays=0),
            lambda r: r['concurrent_window'].update(actual_start=101.),
            lambda r: r['concurrent_window'].update(actual_end=109.),
            lambda r: r['scenario_coverage'][0].update(completed_steps_in_window=0)]
        for change in changes:
            modified = deepcopy(summary)
            change(modified['runs']['compute'][0])
            invalid = c.summarize_worker(modified, window)
            self.assertFalse(invalid['valid_for_speed_comparison'])
            self.assertTrue(invalid['invalid_reasons'])

    def test_mock_batches_count_throughput_and_failure_cleanup(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            args = arguments(root)
            spawned = []
            class Process:
                def __init__(self, command, **kwargs):
                    self.command = command
                    self.returncode = None
                    self.directory = Path(command[command.index('--concurrent-window-dir') + 1])
                    self.name = command[command.index('--concurrent-worker-id') + 1]
                    spawned.append(self)
                    c.write_json(self.directory / (self.name + '.ready.json'), {})
                def poll(self):
                    if self.returncode is None and (self.directory / 'window.json').exists():
                        target = Path(self.command[self.command.index('--output-dir') + 1]) / 'other'
                        target.mkdir(parents=True)
                        c.write_json(target / 'local_optimization_summary.json', worker_summary(c.read_json(self.directory / 'window.json')))
                        self.returncode = 0
                    return self.returncode
                def terminate(self): self.returncode = -15
                def wait(self, timeout): return self.returncode
            with mock.patch.object(c.subprocess, 'Popen', side_effect=Process):
                reports = []
                for count in (2, 6, 8):
                    result = c.run_batch(args, count, 0, root / str(count))
                    reports.append(result)
                    self.assertTrue(result['valid_for_speed_comparison'], result)
                    self.assertEqual(len(result['workers']), count)
                    self.assertAlmostEqual(result['batch_steps_per_second'], count * 90 / args.measurement_seconds)
                    self.assertEqual(sum(r['model'] == 'ann' for r in result['workers'].values()), count // 2)
                combined = c.summarize_batches(reports, 3)
                self.assertTrue(combined['valid_for_speed_comparison'])
                self.assertEqual(combined['conditions']['8']['models']['snn']['mean_task_window_steps_per_second'], 3.)
                self.assertFalse(c.summarize_batches(reports, 4)['valid_for_speed_comparison'])
            # A peer error aborts this owned batch only, with a persistent error report.
            bad = mock.Mock(returncode=1)
            bad.poll.return_value = 1
            with mock.patch.object(c.subprocess, 'Popen', return_value=bad):
                result = c.run_batch(args, 2, 0, root / 'failed')
            self.assertFalse(result['completed'])
            self.assertIn('error', c.read_json(root / 'failed' / 'review' / 'summary.json'))
            self.assertTrue((root / 'failed' / 'other' / 'control' / 'abort.json').exists())
            bad.terminate.assert_not_called()

    def test_partial_dispatch_and_timeout_only_stop_owned_processes(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            args = arguments(root)
            owned = mock.Mock(returncode=-15)
            owned.poll.return_value = None
            with mock.patch.object(c.subprocess, 'Popen', side_effect=[owned, OSError('dispatch failed')]):
                result = c.run_batch(args, 8, 0, root / 'partial')
            self.assertIn('dispatch failed', result['error']['message'])
            self.assertEqual(list(result['exit_codes']), ['ann_0'])
            owned.terminate.assert_called_once()
            owned.wait.assert_called_once_with(timeout=10)
            owned.reset_mock()
            with mock.patch.object(c.subprocess, 'Popen', return_value=owned), \
                    mock.patch.object(c, 'wait_workers', side_effect=TimeoutError('ready timeout')):
                result = c.run_batch(args, 2, 0, root / 'timeout')
            self.assertFalse(result['valid_for_speed_comparison'])
            self.assertEqual(result['error']['type'], 'TimeoutError')
            self.assertTrue((root / 'timeout' / 'review' / 'summary.json').is_file())

    def test_driver_invalid_batch_returns_nonzero_and_stops_following_conditions(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            args = arguments(root)
            args.launch = True  # mock dispatcher only; no subprocess or CUDA work
            invalid = {'concurrency': 2, 'repeat': 0, 'valid_for_speed_comparison': False}
            with mock.patch.object(c, 'build_parser') as parser, \
                    mock.patch.object(c, 'build_plan', return_value={}), \
                    mock.patch.object(c, 'run_batch', return_value=invalid) as batch:
                parser.return_value.parse_args.return_value = args
                self.assertEqual(c.main([]), 1)
            batch.assert_called_once()
            self.assertFalse(c.read_json(args.output_dir / 'review' / 'summary.json')['valid_for_speed_comparison'])


if __name__ == '__main__':
    unittest.main()
