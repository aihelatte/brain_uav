import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from brain_uav.scripts import run_v2_formal_easy_baseline as formal
from brain_uav.v2_curriculum import derive_v2_component_seed


PATH = formal.REPO / 'diagnostics' / 'diagnose_v2_medium_ann_cuda.py'
SPEC = importlib.util.spec_from_file_location('medium_ann_cuda_diagnostic', PATH)
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class ShortDiagnosticTests(unittest.TestCase):
    def test_cli_limits_seed_sets_and_uses_a_separate_output_root(self):
        root = formal.REPO / 'course'
        common = ['--course-root', str(root), '--out', str(root.with_name('diagnostic'))]
        args = diag.parse_args(common)
        self.assertEqual(args.seeds, (41,))
        self.assertFalse(args.run)
        self.assertFalse(args.sync_window)
        for argv in (['--seeds', '41', '23'], ['--seeds', '7', '11', '23', '41'],
                     ['--out', str(root / 'diagnostic')]):
            with self.assertRaises(SystemExit):
                diag.parse_args([*common, *argv])
        args = diag.parse_args([*common, '--seeds', '7', '11', '23', '41', '--single-result', 'single'])
        self.assertEqual(args.seeds, (7, 11, 23, 41))

    def test_stop_after_step_1000_before_next_rollout_or_reset(self):
        for episode_ends_at_limit in (False, True):
            result = SimpleNamespace(stage_steps=0)
            config = SimpleNamespace(max_steps=750000)
            step = Mock()
            reset = Mock()
            trainer = SimpleNamespace(result=result, config=config, _new_episode=reset,
                env=SimpleNamespace(last_curriculum_level='medium', zones=[]),
                engine=SimpleNamespace(device='cpu', select_action=step, update_once=Mock(),
                                       _action_inference_tensor=Mock()))
            trace = diag.Trace(io.StringIO(), sync_window=False)

            def run(trainer):
                for index in range(1001):
                    trainer.engine.select_action(None)
                    result.stage_steps = index + 1
                    if episode_ends_at_limit and result.stage_steps == 1000:
                        trainer._new_episode()
                self.fail('Diagnostic performed a step after its limit')

            with self.assertRaises(diag.DiagnosticComplete):
                diag.run_bounded(run, trainer, trace)
            self.assertEqual(result.stage_steps, 1000)
            self.assertEqual(config.max_steps, 750000)
            self.assertEqual(step.call_count, 1000)
            reset.assert_not_called()

    def test_window_logs_host_boundaries_and_explicit_sync_only_when_enabled(self):
        for synchronize in (False, True):
            stream = io.StringIO()
            trace = diag.Trace(stream, sync_window=synchronize)
            trace.synchronize = Mock()
            for step in (674, 675, 700, 701):
                trace.step = step
                with trace.scope('graph_replay_submit'):
                    pass
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertEqual({row['step'] for row in rows}, {675, 700})
            self.assertEqual(trace.synchronize.call_count, 2 if synchronize else 0)
            self.assertEqual(sum(row['event'] == 'sync_done' for row in rows), 2 if synchronize else 0)

    def test_real_worker_resolves_medium_and_marks_bounded_exit_without_artifact(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = diag.parse_args(['--course-root', str(root / 'course'),
                                    '--out', str(root / 'run'), '--run', '--worker', '41'])
            job = formal.jobs(formal.COURSE_EXPERIMENT, 'medium-ann')[-1]
            spec, config = formal.resolved(job, formal.COURSE_EXPERIMENT)
            paths = {name: str(root / name) for name in ('pool', 'parent', 'ann_bc', 'snn_bc', 'source_manifest')}
            original = dict(plan=dict(initialization_timeout=1800, experiment=formal.COURSE_EXPERIMENT,
                jobs=[spec], batch='medium-ann', gate_only=False), batch_id='test',
                inputs=dict(paths=paths, hashes=dict(pool='pool_hash', parent='parent_hash'), global_steps_start=278374))
            formal.atomic_json(args.out / 'manifest.json', original)
            formal.atomic_json(args.out / 'diagnostic.json', diag.describe(args))

            trainer = SimpleNamespace(config=config, result=SimpleNamespace(stage_steps=999), _new_episode=Mock(),
                env=SimpleNamespace(last_curriculum_level='medium', zones=[]),
                engine=SimpleNamespace(device='cpu', select_action=Mock(), update_once=Mock(), _action_inference_tensor=Mock()))

            def simulated_run(trainer):
                trainer.engine.select_action(None)
                trainer.result.stage_steps += 1
                trainer.engine.select_action(None)

            def controlled_stage(**kwargs):
                self.assertEqual((kwargs['stage'], kwargs['model'], kwargs['seed']), ('medium', 'ann', 41))
                self.assertEqual(kwargs['max_stage_steps'], 750000)
                self.assertEqual(kwargs['formal_config'].to_dict(), spec['formal_config'])
                self.assertEqual(str(kwargs['init_checkpoint']), paths['parent'])
                self.assertEqual(str(kwargs['validation_pool']), paths['pool'])
                self.assertEqual(kwargs['global_steps_start'], 278374)
                self.assertTrue(kwargs['gpu_td3_compute'] and kwargs['non_blocking_batch_transfer'])
                self.assertTrue(kwargs['cuda_graph_updates'] and kwargs['cuda_graph_action_inference'])
                try:
                    diag.V2FormalStageTrainer.run(trainer)
                except diag.DiagnosticComplete as exc:
                    diag.stage._write_strict_json(kwargs['metrics_out'].with_name('metrics_execution_error.json'),
                                                  dict(error_type=type(exc).__name__))
                    raise

            with (patch.object(formal, 'check_inputs'),
                  patch.object(diag.V2FormalStageTrainer, 'run', simulated_run),
                  patch.object(diag.stage, 'run_v2_td3_stage', side_effect=controlled_stage)):
                self.assertEqual(diag.worker(args), 0)
            output = args.out / job['id']
            status = diag.read_json(output / 'worker_summary.json')
            self.assertIsNone(status['formal_passed'])
            self.assertIs(status['diagnostic_completed'], True)
            self.assertEqual(status['stage_steps'], 1000)
            self.assertFalse((output / 'metrics_execution_error.json').exists())
            self.assertFalse(list(args.out.rglob('*.pt')))

    def test_four_workers_require_completed_single_run_with_matching_inputs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            binding = dict(batch_id='same', pool_sha256='real-pool')
            summary = dict(original=binding, diagnostic_only=True, execution_completed=True, seeds=[41],
                           mps_cleanup=dict(stopped=True), sync_window=False,
                           processes=dict(ann_seed41=dict(exit_code=0, pid=41)))
            formal.atomic_json(root / 'ann_seed41/worker_summary.json',
                               dict(diagnostic_completed=True, stage_steps=1000, pid=41))
            formal.atomic_json(root / 'summary.json', summary)
            diag.require_single_result(root, binding)
            with self.assertRaises(ValueError):
                diag.require_single_result(root, dict(binding, pool_sha256='wrong'))
            for bad in (dict(sync_window=True), dict(execution_completed=False), dict(mps_cleanup=dict(stopped=False))):
                formal.atomic_json(root / 'summary.json', dict(summary, **bad))
                with self.assertRaises(ValueError):
                    diag.require_single_result(root, binding)

    def test_release_single_seed_41(self):
        self.check_launch_release((41,))

    def test_release_four_medium_ann_seeds(self):
        self.check_launch_release((7, 11, 23, 41))

    def check_launch_release(self, seeds):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ['--course-root', str(root / 'course'), '--out', str(root / 'diagnostic'),
                    '--run', '--seeds', *map(str, seeds)]
            if len(seeds) == 4:
                argv.extend(['--single-result', str(root / 'single')])
            args = diag.parse_args(argv)
            wanted = formal.jobs(formal.COURSE_EXPERIMENT, 'medium-ann')
            jobs = {job['seed']: job for job in wanted}
            specs = {seed: formal.resolved(job, formal.COURSE_EXPERIMENT)[0] for seed, job in jobs.items()}
            old = args.course_root / diag.ATTEMPT / 'train'
            parent = args.course_root / 'batches/easy-ann/winner.pt'
            parent.parent.mkdir(parents=True)
            parent.write_bytes(b'fixture parent weight')
            parent_hash = formal.file_hash(parent)
            evidence = dict(experiment=formal.COURSE_EXPERIMENT, paths=dict(parent=str(parent)),
                            hashes=dict(parent=parent_hash), global_steps_start=278374,
                            source=formal.identity(formal.COURSE_EXPERIMENT))
            original = dict(plan=dict(initialization_timeout=1, root=str(old), jobs=list(specs.values()),
                            batch='medium-ann', experiment=formal.COURSE_EXPERIMENT, gate_only=False),
                            inputs=evidence)
            expected_pids = [22000 + seed for seed in seeds]
            ready_records = {}
            for seed in seeds:
                seed_manifest = {'base_seed': seed, **{
                    f'{name}_seed': derive_v2_component_seed(seed, 'medium', name)
                    for name in ('model', 'replay', 'curriculum', 'exploration',
                                 'medium_generator', 'easy_generator')}}
                ready = dict(pid=22000 + seed, job=jobs[seed], resolved=specs[seed],
                             synchronized=True, seed_manifest=seed_manifest,
                             initialization=dict(path=str(parent), sha256=parent_hash, global_steps_start=278374),
                             initial_state_sha256={'actor': f'fixture-init-{seed}'})
                ready_records[seed] = ready
                formal.atomic_json(old / jobs[seed]['id'] / 'ready.json', ready)

            def spawn(command, **kwargs):
                seed = int(command[-1])
                ready = ready_records[seed]
                directory = args.out / jobs[seed]['id']
                formal.atomic_json(directory / 'ready.json', ready)
                batch_id = diag.read_json(args.out / 'manifest.json')['batch_id']
                formal.atomic_json(directory / 'worker_summary.json', dict(pid=ready['pid'], batch_id=batch_id,
                    diagnostic_completed=True, stage_steps=1000))
                process = Mock(pid=ready['pid'])
                process.poll.side_effect = lambda: 0 if (args.out / 'release.json').exists() else None
                return process

            def verify(processes):
                self.assertEqual([process.pid for process in processes.values()], expected_pids)
                return {'client_server_pid': {str(pid): 21931 for pid in expected_pids}}

            session = Mock(root=args.out / 'mps', pipe=root / 'private-pipe', env={})
            session.verify.side_effect = verify
            session.finish.return_value = dict(stopped=True)
            process_api = SimpleNamespace(Popen=spawn, STDOUT=diag.subprocess.STDOUT)
            with (patch.object(diag, 'Session', return_value=session),
                  patch.object(diag, 'capability'),
                  patch.object(diag, 'subprocess', process_api)):
                self.assertEqual(diag.launch(args, original, wanted, {'batch_id': 'fixture'}), 0)
            session.verify.assert_called_once()
            self.assertEqual(diag.read_json(args.out / 'release.json')['pids'], expected_pids)
