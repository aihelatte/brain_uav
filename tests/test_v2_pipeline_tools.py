"""Integration checks for V2 lightweight outputs and offline tools."""

from contextlib import redirect_stdout
import gzip
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.trainers.v2_reporting import V2ExperimentReporter
from test_v2_reporting import _episode, _scenario_payload


class PipelineToolsTest(unittest.TestCase):
    def test_build_report_reads_both_layouts(self):
        from brain_uav.scripts.build_v2_report import main

        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                batch = 'medium-ann'
                seed = (root / 'batches' / batch / 'attempt-000001/train/ann_seed7'
                        if legacy else root / 'runs' / batch / 'seed7')
                log = seed / 'metrics_reports' if legacy else seed
                log.mkdir(parents=True)
                selection = (root / 'batches' / batch / 'selection.json'
                             if legacy else root / 'winners' / f'{batch}.selection.json')
                selection.parent.mkdir(parents=True, exist_ok=True)
                selection.write_text(json.dumps(dict(winner={'seed': 7})), encoding='utf-8')
                (root / ('course.json' if legacy else 'manifest.json')).write_text('{}')
                (log / 'windows.jsonl').write_text('{"episode_start": 1, "episode_end": 3, "goal_count": 1}\n')
                (log / 'stage_start.json').write_text('{"max_steps": 500000}')
                (log / 'stage_end.json').write_text('{"stage_elapsed_seconds": 2.5}')
                (seed / 'terminal_evaluation.json').write_text(json.dumps(dict(
                    formal_passed=True, actual_steps=125001,
                    evaluation=dict(scenarios=[
                        dict(outcome='goal'), dict(outcome='boundary', boundary_violations=['z_max']),
                        dict(outcome='boundary', boundary_violations=[dict(axis='x', direction='positive')]),
                    ]),
                )))
                main(['--run-dir', str(root)])
                report = root / 'report'
                for path in ('overview_all_stages.png', f'{batch}/batch_seeds_summary.png',
                             f'{batch}/seed7_training_curves.png'):
                    self.assertGreater((report / path).stat().st_size, 0)
                markdown = (report / 'README.md').read_text(encoding='utf-8')
                self.assertIn('| medium | ANN | 7 | yes | yes | 1 | 1 | 1 | 0 |', markdown)
                self.assertIn('500000', markdown)
                self.assertIn(f'({batch}/seed7_training_curves.png)', markdown)

    def test_training_inspection_filters_and_renders_saved_trajectory(self):
        from brain_uav.scripts.inspect_v2_run import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed, out = root / 'run' / 'runs/easy-ann/seed7', root / 'inspection'
            seed.mkdir(parents=True)
            reporter = V2ExperimentReporter(
                seed, stage='easy', model_type='ann', scenario=ScenarioConfig(),
                rewards=RewardConfig(), uav_collision_radius=0., max_steps=500_000,
                render_trajectories=False, trajectory_archive_mode='full',
            )
            reporter.start_stage({})
            for number, outcome in [(1, 'ground'), (2, 'goal')]:
                reporter.record_episode(
                    _episode(number, outcome, 30 + number), scenario_payload=_scenario_payload([]),
                    trajectory=[[-10., -10., 20.], [-9., -10., 21.]],
                    actions=[[0., 0.]], terminal_state=[-9., -10., 21., 0., 0.],
                )
            reporter.close()
            archive = seed / 'trajectories.jsonl.gz'
            original = archive.read_bytes()
            main(['training', '--seed-dir', str(seed), '--windows', '1,4-5',
                  '--episodes', '1-2', '--step-range', '31-32', '--outcomes', 'ground',
                  '--out', str(out)])
            self.assertEqual(archive.read_bytes(), original)
            result = json.loads((out / 'summary.json').read_text())
            self.assertEqual([row['episode'] for row in result['rows']], [1])
            self.assertEqual(result['rows'][0]['mean_vz_last_10'], 1.)
            self.assertIn('ground', (out / 'summary.md').read_text(encoding='utf-8'))
            self.assertEqual(len(list((out / 'views').glob('*.png'))), 1)
            with self.assertRaises(ValueError):
                main(['training', '--seed-dir', str(seed), '--windows', '1',
                      '--out', str(seed / 'unsafe')])
            internal = root / 'run' / '.batches/easy-ann/part-000001/train/ann_seed7'
            internal.mkdir(parents=True)
            with self.assertRaises(ValueError):
                main(['training', '--seed-dir', str(internal), '--windows', '1',
                      '--out', str(root / 'run' / 'unsafe')])

    def test_eval_inspection_replays_selected_scene_deterministically(self):
        from brain_uav.scripts.inspect_v2_run import main
        from brain_uav.trainers.v2_formal_training import (
            V2_PERIODIC_SNAPSHOT_FORMAT, V2_PERIODIC_SNAPSHOT_VERSION, save_v2_periodic_snapshot,
        )
        from brain_uav.trainers.v2_validation import load_v2_validation_pool
        from test_diagnose_v2_policy import write_inputs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'inputs'
            inputs.mkdir()
            checkpoint, pool, terminal = write_inputs(inputs)
            scenario = load_v2_validation_pool(pool, expected_level='easy').scenarios[0]['scenario_id']
            reports = []
            for i in (1, 2):
                out = root / f'inspection{i}'
                main(['eval', '--checkpoint', str(checkpoint), '--pool', str(pool),
                      '--scenarios', scenario, '--device', 'cpu', '--out', str(out)])
                reports.append(json.loads((out / 'summary.json').read_text()))
                self.assertIn(scenario, (out / 'summary.md').read_text(encoding='utf-8'))
                self.assertEqual(len(list((out / 'views').glob('*.png'))), 1)
                self.assertEqual(reports[-1]['rows'][0]['outcome'], 'goal')
            self.assertEqual(reports[0], reports[1])
            periodic = inputs / 'model_periodic' / 'step_000150000.pt'
            periodic.parent.mkdir()
            snapshot = {key: value for key, value in terminal.items() if key not in (
                'status', 'passed_validation', 'training_result', 'validation_pool')}
            snapshot.update(format=V2_PERIODIC_SNAPSHOT_FORMAT, format_version=V2_PERIODIC_SNAPSHOT_VERSION,
                            stage_steps=150003, scheduled_stage_steps=150000)
            save_v2_periodic_snapshot(periodic, snapshot)
            out = root / 'periodic_inspection'
            main(['eval', '--checkpoint', str(periodic), '--pool', str(pool),
                  '--outcomes', 'goal', '--limit', '1', '--out', str(out)])
            result = json.loads((out / 'summary.json').read_text())
            self.assertEqual([(row['scenario_id'], row['stage_steps']) for row in result['rows']], [(scenario, 150003)])
            self.assertEqual(len(list((out / 'views').glob('*.png'))), 1)

    def test_pipeline_plan_and_promotion(self):
        from brain_uav.scripts import run_v2_formal_course as course
        from brain_uav.scripts import run_v2_pipeline as pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'trial'
            with redirect_stdout(StringIO()) as output:
                pipeline.main(['--output-root', str(root), '--models', 'snn,ann', '--stages', 'medium,easy',
                               '--seeds', '7,11,23,41,53', '--plan'])
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['stages'], ['easy', 'medium'])
            self.assertEqual(plan['models'], ['ann', 'snn'])
            specs = plan['batches']['easy-snn']
            self.assertEqual([[job['job']['seed'] for job in chunk] for chunk in specs], [[7, 11, 23, 41], [53]])
            self.assertEqual(specs[1][0]['formal_config']['acc_eta'], .01)
            self.assertEqual(specs[1][0]['formal_config']['max_steps'], 500_000)
            self.assertEqual(specs[1][0]['trajectory_archive_mode'], 'full')
            self.assertFalse(root.exists())
            with redirect_stdout(StringIO()) as output:
                pipeline.main(['--output-root', str(root), '--models', 'ann', '--stages', 'hard',
                               '--seeds', '53', '--max-steps', '12', '--device', 'cpu', '--plan',
                               '--parent-ann', str(Path(directory) / 'mid.pt')])
            single = json.loads(output.getvalue())
            self.assertEqual(single['batches']['hard-ann'][0][0]['formal_config']['max_steps'], 12)
            self.assertFalse(single['batches']['hard-ann'][0][0]['gpu_td3_compute'])
        candidate = dict(seed=53, goal=95, collision=1, execution_completed=True, formal_passed=True,
                         last_50k=dict(successes=1, count=2))
        self.assertEqual(course.select_winner([candidate], seeds=[53]), candidate)
        other = dict(candidate, seed=79, collision=0)
        self.assertEqual(course.select_winner([candidate, other], seeds=[53, 79]), other)
        with self.assertRaises(ValueError):
            course.select_winner([candidate], seeds=[53, 79])

    def test_pipeline_plan_only_changes_vertical_rewards_and_hard_bc(self):
        from brain_uav.scripts import run_v2_pipeline as pipeline

        course = pipeline.course
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()) as output:
            root = Path(directory) / 'new'
            pipeline.main(['--models', 'ann,snn', '--stages', 'easy,medium,hard', '--seeds', '7,11,23,41',
                           '--output-root', str(root), '--plan'])
            self.assertFalse(root.exists())
        plan = json.loads(output.getvalue())
        vertical = dict(
            ground_directional_warning_enabled=True, ground_directional_warning_height=60.0,
            ground_directional_penalty_weight=180.0, ground_soft_penalty_cap=300.0,
            boundary_directional_warning_enabled=True, boundary_directional_warning_distance=60.0,
            boundary_directional_penalty_weight=90.0, upper_boundary_directional_warning_distance=150.0,
            upper_boundary_directional_penalty_weight=200.0, boundary_soft_penalty_cap=300.0,
        )
        self.assertEqual(len(plan['batches']), 6)
        reward_configs = []
        for batch, (jobs,) in plan['batches'].items():
            stage, model = batch.split('-')
            self.assertEqual([spec['job']['seed'] for spec in jobs], [7, 11, 23, 41])
            for spec in jobs:
                with self.subTest(batch=batch, seed=spec['job']['seed']):
                    old, _ = course.resolve_course_job(spec['job'], course_profile='500k')
                    self.assertEqual(old['reward_config']['upper_boundary_directional_warning_distance'], 120.0)
                    self.assertEqual(old['reward_config']['boundary_soft_penalty_cap'], 240.0)
                    self.assertFalse({'ground_directional_warning_enabled', 'upper_boundary_directional_penalty_weight'}
                                     & old['reward_config'].keys())
                    self.assertEqual(old['bc_schedule'], 'smooth-mid-strong' if stage == 'easy' else 'default')
                    self.assertEqual(spec['reward_config'], {**old['reward_config'], **vertical})
                    reward_configs.append(spec['reward_config'])
                    self.assertEqual(spec['bc_schedule'], 'default' if stage == 'medium' else 'smooth-mid-strong')
                    self.assertEqual(spec['formal_config'], old['formal_config'])
                    self.assertEqual(spec['job']['method'], 'min' if model == 'ann' else 'acc')
                    self.assertEqual(spec['formal_config'].get('acc_eta'), None if model == 'ann' else .01)
        self.assertTrue(all(rewards == reward_configs[0] for rewards in reward_configs))

    def test_pipeline_vertical_rewards_survive_real_stage_checkpoint_handoffs(self):
        import torch

        from brain_uav.config import reward_config_from_snapshot
        from brain_uav.scripts import run_v2_formal_course as course
        from brain_uav.scripts.train_v2_bc import train_v2_behavior_cloning
        from brain_uav.trainers import v2_formal_training as formal
        from brain_uav.trainers.v2_validation import scenario_config_from_snapshot
        from test_v2_bc import make_scenario_payload, write_cluster

        def unblocked_scene(index, zone_count):
            payload = make_scenario_payload(index, zone_count)
            payload['metadata']['requested_direct_path_blocker'] = False
            return payload

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        options = dict(course_profile='500k',
                       pipeline=dict(seeds=[7], device='cpu', max_steps=500_000, trajectory_archive='full'))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            batch = course.batch_runner
            for model in ('ann', 'snn'):
                parent = None
                for stage in ('easy', 'medium', 'hard'):
                    job, = batch.jobs(course.EXPERIMENT, f'{stage}-{model}', **options)
                    spec, config = batch.resolved(job, course.EXPERIMENT, **options)
                    scenario = scenario_config_from_snapshot(spec['scenario_config'])
                    rewards = reward_config_from_snapshot(spec['reward_config'])
                    if stage == 'easy':
                        with patch('test_v2_bc.make_scenario_payload', side_effect=unblocked_scene):
                            cluster = write_cluster(root / f'{model}-cluster', scenario_config=scenario, zone_counts=(0,) * 4)
                        bc_dir = root / model
                        train_v2_behavior_cloning(trajectory_cluster=cluster, output_dir=bc_dir, epochs=1,
                            batch_size=2, validation_fraction=.25, device='cpu', model=model,
                            actor_hidden_dim=192, actor_projection_dim=32)
                        parent = bc_dir / f'bc_v2_{model}_best.pt'
                    components = formal.build_v2_stage_engine(scenario, config, init_checkpoint=parent,
                        rewards=rewards, model_type=model, device='cpu')
                    self.assertEqual(components.engine.target_q_aggregation, 'min')
                    acc = getattr(components.engine, 'formal_acc', None)
                    self.assertEqual(None if acc is None else acc.controller.eta, None if model == 'ann' else .01)
                    if stage == 'hard':
                        self.assertEqual(formal.v2_bc_lambda(300_000, schedule=spec['bc_schedule']), 20.0)
                        self.assertEqual(formal.v2_bc_lambda(499_999, schedule=spec['bc_schedule']), 10.0)
                        periodic = formal.build_v2_periodic_snapshot(components.engine, config, stage_steps=150_000,
                            scenario=scenario, rewards=rewards, uav_collision_radius=0.0, seed_manifest={'base_seed': 7},
                            initialization_source=components.initialization_source, bc_schedule=spec['bc_schedule'])
                        self.assertEqual(formal.load_v2_periodic_snapshot(periodic)['reward_config'], spec['reward_config'])
                        with self.assertRaisesRegex(ValueError, 'Previous stage RewardConfig is incompatible'):
                            formal.prepare_v2_stage_initialization(config, init_checkpoint=parent,
                                scenario=scenario, rewards=RewardConfig(), model_type=model)
                    result = formal.V2FormalTrainingResult.empty(stage, global_steps_start=0)
                    result.status, result.passed_validation = 'passed', True
                    result.validation_records.append({'passed': True})
                    payload = formal.build_v2_formal_checkpoint(components.engine, result, config,
                        scenario=scenario, rewards=rewards, uav_collision_radius=0.0, seed_manifest={'base_seed': 7},
                        validation_pool_metadata=dict(path='fixture', format_version=1, curriculum_level=stage,
                            master_seed=123, stage_seed=234, scenario_count=100, content_digest='fixture'),
                        initialization_source=components.initialization_source, bc_schedule=spec['bc_schedule'])
                    parent = root / model / f'{stage}.pt'
                    formal.save_v2_formal_checkpoint(parent, payload)
                    restored = formal.load_v2_formal_checkpoint(parent, require_passed=True, expected_model_type=model)
                    self.assertEqual(restored['reward_config'], spec['reward_config'])
                    self.assertEqual(restored['bc_schedule'].get('name', 'default'), spec['bc_schedule'])

    def test_pipeline_cpu_runs_real_worker_and_publishes_flat_reports(self):
        import torch

        from brain_uav.scripts import run_v2_pipeline as pipeline
        from brain_uav.scripts import train_v2_td3
        from brain_uav.scripts.train_v2_bc import train_v2_behavior_cloning
        from brain_uav.trainers.v2_validation import generate_v2_validation_pool, save_v2_validation_pool
        from test_v2_bc import make_scenario_payload, write_cluster

        course, batch = pipeline.course, pipeline.batch_runner
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            training = root / 'new'
            scenario = ScenarioConfig(max_steps=2)

            def unblocked_scene(index, count):
                payload = make_scenario_payload(index, count)
                payload['metadata']['requested_direct_path_blocker'] = False
                return payload

            with patch('test_v2_bc.make_scenario_payload', side_effect=unblocked_scene):
                cluster = write_cluster(root / 'cluster', scenario_config=scenario, zone_counts=(0, 0, 0, 0))
            train_v2_behavior_cloning(trajectory_cluster=cluster, output_dir=root / 'bc', epochs=1,
                batch_size=2, validation_fraction=.25, device='cpu', model='ann',
                actor_hidden_dim=192, actor_projection_dim=32)
            bc = root / 'bc' / 'bc_v2_ann_best.pt'
            pool = generate_v2_validation_pool(scenario, 'easy', master_seed=20260904)
            source_manifest = root / 'source.json'
            identity = batch.identity(course.EXPERIMENT)
            course.atomic_json(source_manifest, identity)

            def pools(output, _):
                path = output / 'pools' / 'validation' / 'easy.json'
                path.parent.mkdir(parents=True)
                save_v2_validation_pool(path, pool)
                return dict(validation=dict(easy=dict(path=str(path), sha256=batch.file_hash(path)))), dict(
                    validation=dict(easy=pool))

            def inputs(args):
                paths = dict(ann_bc=str(bc), snn_bc=str(bc), pool=str(args.validation_pool))
                args.ann_bc = args.snn_bc = bc
                return dict(paths=paths, hashes={name: batch.file_hash(path) for name, path in paths.items()},
                            source=identity, experiment=args.experiment, global_steps_start=0,
                            pool_digest=pool.content_digest)

            original_resolve = course.resolve_course_job

            def resolve(job, **options):
                spec, config = original_resolve(job, **options)
                return dict(spec, scenario_config=pool.scenario_config), config

            with (patch.object(course, 'prepare_pools', side_effect=pools),
                  patch.object(batch, 'inputs', side_effect=inputs),
                  patch.object(course, 'resolve_course_job', side_effect=resolve),
                  patch.object(train_v2_td3, 'run_v2_td3_stage', wraps=train_v2_td3.run_v2_td3_stage) as stage):
                pipeline.main(['--output-root', str(training), '--models', 'ann', '--stages', 'easy',
                               '--seeds', '53', '--device', 'cpu', '--max-steps', '3',
                               '--source-manifest', str(source_manifest)])
            self.assertIs(stage.call_args.kwargs['render_trajectories'], False)
            self.assertEqual(stage.call_args.kwargs['trajectory_archive_mode'], 'full')
            self.assertTrue(stage.call_args.kwargs['rewards'].ground_directional_warning_enabled)
            self.assertEqual(stage.call_args.kwargs['rewards'].upper_boundary_directional_penalty_weight, 200.0)
            seed = training / 'runs' / 'easy-ann' / 'seed53'
            for name in ('model_failed.pt', 'trajectories.jsonl.gz', 'episodes.jsonl', 'windows.jsonl',
                         'stage_start.json', 'stage_end.json', 'terminal_evaluation.json', 'metrics.json', 'ready.json'):
                self.assertTrue((seed / name).is_file(), name)
            self.assertFalse(list(seed.glob('trajectories/**/*.png')))
            with gzip.open(seed / 'trajectories.jsonl.gz', 'rt', encoding='utf-8') as stream:
                self.assertEqual([json.loads(line)['episode'] for line in stream], [1])
            selection = json.loads((training / 'winners' / 'easy-ann.selection.json').read_text())
            self.assertEqual(selection['status'], 'no_formal_pass')
            self.assertIsNone(selection['winner'])
            self.assertEqual(selection['candidates'][0]['seed'], 53)
            manifest = json.loads((training / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['artifact_sha256']['runs/easy-ann/seed53/model_failed.pt'],
                             batch.file_hash(seed / 'model_failed.pt'))
            self.assertTrue((training / 'report' / 'README.md').is_file())

    def test_pipeline_cuda_release_checks_actual_configured_candidates(self):
        from brain_uav.scripts import run_v2_formal_course as course
        from brain_uav.v2_curriculum import derive_v2_component_seed

        batch = course.batch_runner
        options = dict(course_profile='500k',
                       pipeline=dict(seeds=[53, 79], device='cuda', max_steps=500_000, trajectory_archive='full'))
        jobs = batch.jobs(course.EXPERIMENT, 'medium-ann', **options)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = dict(paths={'parent': 'medium_parent.pt'}, hashes={'parent': 'sha'}, global_steps_start=150_000)
            batch.atomic_json(root / 'manifest.json', dict(plan=options, inputs=evidence))
            processes = {}
            for pid, job in enumerate(jobs, 33001):
                spec, config = batch.resolved(job, course.EXPERIMENT, **options)
                components = ('model', 'replay', 'curriculum', 'exploration',
                              *(f'{name}_generator' for name in config.curriculum_mix))
                seeds = dict(base_seed=config.seed, **{
                    f'{name}_seed': derive_v2_component_seed(config.seed, config.stage, name) for name in components})
                processes[job['id']] = SimpleNamespace(pid=pid, poll=lambda: None)
                batch.atomic_json(root / job['id'] / 'ready.json', dict(pid=pid, job=job, synchronized=True,
                    resolved=spec, seed_manifest=seeds, initialization=batch.initialization_binding(job, evidence)))
            session = Mock()
            session.verify.return_value = {'status': 'verified'}
            with self.assertRaisesRegex(ValueError, 'Expected batch workers'):
                batch.release({'ann_seed53': processes['ann_seed53']}, root, session,
                              course.EXPERIMENT, batch='medium-ann')
            self.assertFalse((root / 'release.json').exists())
            batch.release(processes, root, session, course.EXPERIMENT, batch='medium-ann')
            self.assertEqual(course.read_json(root / 'release.json')['pids'], [33001, 33002])

    def test_archive_and_rendering_controls(self):
        for mode, expected in [('full', [1, 2, 3, 4, 5]), ('failures', [1, 2, 3, 5]),
                               ('none', [])]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with redirect_stdout(StringIO()), patch(
                    'brain_uav.trainers.v2_reporting.export_v2_trajectory_views',
                ) as render:
                    reporter = V2ExperimentReporter(
                        root, stage='easy', model_type='ann', scenario=ScenarioConfig(),
                        rewards=RewardConfig(), uav_collision_radius=0., max_steps=500_000,
                        render_trajectories=False, trajectory_archive_mode=mode,
                    )
                    reporter.start_stage({})
                    events = [(1, 'goal', 49999), (2, 'collision', 50000),
                              (3, 'goal', 50001), (4, 'goal', 70000), (5, 'goal', 100002)]
                    for episode, outcome, steps in events:
                        if episode == 4:
                            reporter.record_window(dict(
                                window_index=1, episode_start=1, episode_end=3, stage_steps=50001,
                                episode_count=3, consecutive_qualified_windows=0, average_return=1.,
                                average_length=1., average_actor_loss=0., average_critic_loss=0.,
                            ))
                        reporter.record_episode(
                            _episode(episode, outcome, steps), scenario_payload=_scenario_payload([]),
                            trajectory=[[-10., -10., 20.], [-9.543212, -10., 20.123456]],
                            actions=[[0.123456, -0.002345]], terminal_state=[-9.543212, -10., 20.123456, 0., 0.],
                        )
                    reporter.prepare_validation_candidate(1, global_steps=100002)
                    reporter.begin_validation(curriculum_level='easy', scenario_count=1)
                    reporter.record_validation_scenario(
                        dict(scenario_id='easy_00001', outcome='timeout'),
                        scenario_payload={}, trajectory=[], actions=[], terminal_state=[],
                    )
                    reporter.close()
                    render.assert_not_called()
                archive = root / 'trajectories.jsonl.gz'
                self.assertEqual(archive.exists(), bool(expected))
                if expected:
                    with gzip.open(archive, 'rt', encoding='utf-8') as stream:
                        rows = [json.loads(line) for line in stream]
                    self.assertEqual([row['episode'] for row in rows], expected)
                    self.assertEqual([row['window_index'] for row in rows],
                                     [1 if episode <= 3 else 2 for episode in expected])
                    self.assertEqual(rows[0]['trajectory'][1], [-9.5432, -10., 20.1235])
                    self.assertEqual(rows[0]['actions'], [[0.1235, -0.0023]])
                    self.assertEqual(rows[0]['terminal_state'][0], -9.5432)
                    self.assertNotIn('reward_config', rows[0])
                    self.assertNotIn('scenario_config', rows[0])
                self.assertFalse(list(root.rglob('*.png')))
                self.assertTrue({'stage', 'model_type', 'scenario_config', 'reward_config',
                                 'uav_collision_radius'} <= json.loads((root / 'stage_start.json').read_text()).keys())
