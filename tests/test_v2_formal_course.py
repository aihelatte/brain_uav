from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from hashlib import sha256
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brain_uav.config import reward_config_from_snapshot
from brain_uav.scripts import run_v2_formal_course as course
from brain_uav.scripts import run_v2_formal_easy_baseline as batch
from brain_uav.scripts import train_v2_td3 as single_stage
from brain_uav.scripts.train_v2_bc import train_v2_behavior_cloning
from brain_uav.trainers import v2_formal_training as formal
from brain_uav.trainers.v2_formal_acc import load_state
from brain_uav.trainers.v2_validation import (
    generate_v2_validation_pool, save_v2_validation_pool,
    scenario_config_from_snapshot,
)
from test_v2_bc import make_scenario_payload, write_cluster
from test_v2_static_no_fly_env import _scenario


class FormalCourseTests(unittest.TestCase):
    def test_course_source_manifest_covers_current_sources(self):
        self.assertEqual(batch.identity(course.EXPERIMENT)['sha256'],
                         course.read_json(course.RESUME_SOURCE_MANIFEST)['sha256'])
        self.assertEqual(batch.source_hash(course.SOURCE_MANIFEST),
                         '0065294ddba0e8593a025eaf6819fbf5e691f774e184529d560a8c597ffebce1')

    def test_resume_plan_keeps_easy_in_course_and_starts_training_at_medium_ann(self):
        with TemporaryDirectory() as tmp:
            old = Path(tmp) / 'historical'
            root = Path(tmp) / 'continuation'
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(course.main(['--root', str(root), '--resume-from', str(old)]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['resume_from'], str(old.resolve()))
            self.assertEqual(plan['reuse_batches'], ['easy-ann', 'easy-snn'])
            self.assertEqual(plan['first_training_batch'], 'medium-ann')
            self.assertEqual(len(plan['batches']), 6)
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(course.main(['--root', str(root), '--resume-from', str(old),
                                             '--resume-after-medium-ann']), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan['reuse_batches'], ['easy-ann', 'easy-snn', 'medium-ann'])
            self.assertEqual(plan['first_training_batch'], 'medium-snn')
            self.assertFalse(root.exists())
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                course.main(['--root', str(old / 'nested'), '--resume-from', str(old)])
            self.assertFalse(old.exists())

    def test_historical_easy_import_and_restart_use_formal_batch_path(self):
        with TemporaryDirectory() as tmp:
            fixture = self.historical_course_fixture(Path(tmp).resolve())
            args, archive, archive_pools, old_bc = fixture
            archived_files = {path: batch.file_hash(path) for path in archive.rglob('*') if path.is_file()}
            launched = []
            interrupted = False
            child_batches = ('medium-ann', 'medium-snn', 'hard-ann', 'hard-snn')

            def launch(namespace):
                nonlocal interrupted
                launched.append((namespace.batch, namespace.root.parent.name, namespace.root.name))
                stage, model = namespace.batch.split('-')
                parent_stage = course.PREDECESSOR[stage]
                self.assertEqual(namespace.parent, args.root / 'batches' / f'{parent_stage}-{model}' / 'winner.pt')
                self.assertEqual(namespace.parent_sha256, batch.file_hash(namespace.parent))
                self.assertEqual(namespace.source_manifest, course.RESUME_SOURCE_MANIFEST)
                self.assertEqual([job['seed'] for job in batch.batch_jobs(namespace)], list(course.SEEDS))
                pool_info = archive_pools['validation'][stage]
                self.assertEqual(namespace.validation_pool.parent, args.root / 'pools' / 'validation')
                self.assertEqual(batch.file_hash(namespace.validation_pool), pool_info['sha256'])
                failure = namespace.batch == 'medium-snn' and not namespace.gate_only and not interrupted
                batch.atomic_json(namespace.root / 'batch_summary.json', dict(
                    execution_completed=not failure, mps_cleanup=dict(stopped=True)))
                if failure:
                    interrupted = True
                    raise RuntimeError('simulated interruption')

            actual_collect = course.collect_batch

            def collect(namespace, pool, evidence, **kwargs):
                if namespace.batch.startswith('easy'):
                    return actual_collect(namespace, pool, evidence, **kwargs)
                stage, model = namespace.batch.split('-')
                rows = []
                for seed in course.SEEDS:
                    path = namespace.root / f'{seed}.pt'
                    path.write_text(f'{stage} {model} {seed}', encoding='utf-8')
                    rows.append(dict(stage=stage, model=model, seed=seed, execution_completed=True,
                        formal_passed=True, checkpoint=str(path), checkpoint_sha256=batch.file_hash(path),
                        goal=100, collision=0, initialization=dict(sha256=namespace.parent_sha256),
                        last_50k=dict(successes=1, count=1, success_rate=1.)))
                return rows

            with (self.mock_historical_inputs(archive_pools, old_bc),
                  patch.object(batch, 'launch', side_effect=launch),
                  patch.object(batch, 'inputs', return_value={}),
                  patch.object(course, 'collect_batch', side_effect=collect),
                  patch.object(course, 'evaluate_winners', return_value=[]) as evaluate):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    course._run_locked(args)
                evaluate.assert_not_called()
                self.assertEqual(course._run_locked(args), 0)
                evaluate.assert_called_once()
                self.assertEqual(course._run_locked(args), 0)
            expected_batches = [(child_batches[0], 'attempt-000001'), (child_batches[1], 'attempt-000001'),
                (child_batches[1], 'attempt-000002'), (child_batches[2], 'attempt-000001'),
                (child_batches[3], 'attempt-000001')]
            self.assertEqual(launched, [(name, attempt, phase) for name, attempt in expected_batches
                                        for phase in ('gate', 'train')])
            receipt = course.read_json(args.root / 'resume_source.json')
            self.assertEqual(receipt['root'], str(archive))
            self.assertEqual(receipt['HEAD'], course.HISTORICAL_HEAD)
            for model, seed in (('ann', 23), ('snn', 11)):
                imported = args.root / 'batches' / f'easy-{model}'
                selection = course.read_json(imported / 'selection.json')
                self.assertEqual(selection['winner']['seed'], seed)
                self.assertEqual((imported / 'winner.pt').read_text(), f'easy {model} {seed}')
                self.assertEqual(batch.file_hash(imported / 'winner.pt'), receipt['easy'][model]['winner_sha256'])
                self.assertFalse(list(imported.glob('attempt-*')))
            with (args.root / 'training.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 24)
            self.assertEqual([(row['model'], int(row['seed']), int(row['goal'])) for row in rows
                              if row['stage'] == 'easy' and row['winner'] == 'True'],
                             [('ann', 23, 100), ('snn', 11, 100)])
            self.assertEqual({path: batch.file_hash(path) for path in archive.rglob('*') if path.is_file()},
                             archived_files)

    def test_historical_tampering_blocks_before_any_new_batch(self):
        for corruption, message in (
                ('winner', 'Historical easy winner SHA256'), ('pool', 'Locked pool changed'),
                ('selection', 'Historical easy promotion differs')):
            with self.subTest(corruption=corruption), TemporaryDirectory() as tmp:
                args, archive, pools, old_bc = self.historical_course_fixture(Path(tmp).resolve())
                if corruption == 'winner':
                    (archive / 'batches' / 'easy-snn' / 'winner.pt').write_bytes(b'changed winner')
                elif corruption == 'pool':
                    Path(pools['independent']['hard']['path']).write_bytes(b'changed pool')
                else:
                    path = archive / 'batches' / 'easy-ann' / 'selection.json'
                    selection = course.read_json(path)
                    selection['candidates'][0]['goal'] -= 1
                    batch.atomic_json(path, selection)
                with (self.mock_historical_inputs(pools, old_bc), patch.object(batch, 'launch') as launch):
                    with self.assertRaisesRegex(ValueError, message):
                        course._run_locked(args)
                    launch.assert_not_called()
                self.assertFalse((args.root / 'batches').exists())

    def test_resume_preserves_medium_ann_selection_and_starts_fresh_snn_gate(self):
        for medium_status in ('promoted', 'no_formal_pass', 'tampered'):
            with self.subTest(status=medium_status), TemporaryDirectory() as tmp:
                workspace = Path(tmp).resolve()
                args, origin, pools, bc = self.historical_course_fixture(workspace)
                prior = args.root
                old_source = course.MEDIUM_ANN_SOURCE_MANIFEST
                bc_hashes = {model: entry[1] for model, entry in bc.items()}
                with self.mock_historical_inputs(pools, bc):
                    course.import_completed(args, origin, bc_hashes, after_medium_ann=False)
                batch.atomic_json(prior / 'course.json', dict(experiment=course.EXPERIMENT,
                    code=course.read_json(old_source)['sha256'], bc=bc_hashes, plan=course.course_plan(prior, origin)))
                parent = course.read_json(prior / 'batches/easy-ann/selection.json')['winner']
                prior_pools = course.read_json(prior / 'pools.json')
                winner_seed = None if medium_status == 'no_formal_pass' else 41
                self.write_historical_batch_fixture(args, prior, prior_pools, bc, 'medium-ann', winner_seed,
                                                    old_source, course.MEDIUM_ANN_HEAD, parent=parent)
                selection_path = prior / 'batches/medium-ann/selection.json'
                if medium_status == 'tampered':
                    selection = course.read_json(selection_path)
                    selection['winner']['seed'] = 7
                    batch.atomic_json(selection_path, selection)
                before = {path: batch.file_hash(path) for path in prior.rglob('*') if path.is_file()}
                args.root = workspace / 'fix'
                args.resume_from = prior
                args.resume_after_medium_ann = True
                launched = []
                real_collect = course.collect_batch

                def launch(namespace):
                    launched.append((namespace.batch, namespace.gate_only))
                    stage, model = namespace.batch.split('-')
                    parent_name = f'{course.PREDECESSOR[stage]}-{model}'
                    self.assertEqual(namespace.parent, args.root / 'batches' / parent_name / 'winner.pt')
                    self.assertEqual(namespace.parent_sha256, batch.file_hash(namespace.parent))
                    self.assertEqual(namespace.source_manifest, course.RESUME_SOURCE_MANIFEST)
                    self.assertEqual([job['seed'] for job in batch.batch_jobs(namespace)], list(course.SEEDS))
                    batch.atomic_json(namespace.root / 'batch_summary.json', dict(execution_completed=True,
                                                                                 mps_cleanup=dict(stopped=True)))

                def collect(namespace, pool, evidence, **kwargs):
                    if kwargs:
                        return real_collect(namespace, pool, evidence, **kwargs)
                    stage, model = namespace.batch.split('-')
                    records = []
                    for seed in course.SEEDS:
                        path = namespace.root / f'{seed}.pt'
                        path.write_text(f'{stage} {model} {seed}', encoding='utf-8')
                        records.append(dict(model=model, stage=stage, seed=seed, steps=160000, goal=100,
                            collision=0, execution_completed=True, formal_passed=True, checkpoint=str(path),
                            checkpoint_sha256=batch.file_hash(path), initialization=dict(sha256=namespace.parent_sha256),
                            last_50k=dict(successes=1, count=1, success_rate=1.)))
                    return records

                def historical_parent_steps(stage, model, path, expected_hash, hashes):
                    if stage == 'easy':
                        return 0
                    self.assertEqual((stage, model, path), ('medium', 'ann', Path(parent['checkpoint'])))
                    self.assertEqual(expected_hash, hashes['parent'])
                    return parent['steps']

                with (self.mock_historical_inputs(pools, bc),
                      patch.object(batch, 'launch', side_effect=launch),
                      patch.object(batch, 'inputs', return_value={}),
                      patch.object(course, 'collect_batch', side_effect=collect),
                      patch.object(course, 'parent_steps', side_effect=historical_parent_steps),
                      patch.object(course, 'evaluate_winners', return_value=[]) as evaluate):
                    if medium_status == 'tampered':
                        with self.assertRaisesRegex(ValueError, 'Historical medium-ann promotion differs'):
                            course._run_locked(args)
                        self.assertEqual(launched, [])
                        evaluate.assert_not_called()
                        continue
                    self.assertEqual(course._run_locked(args), 0)
                    if medium_status == 'promoted':
                        evaluate.assert_called_once()
                    else:
                        evaluate.assert_not_called()
                expected = ['medium-snn', 'hard-ann', 'hard-snn'] if winner_seed else ['medium-snn', 'hard-snn']
                self.assertEqual(launched, [(name, gate_only) for name in expected for gate_only in (True, False)])
                self.assertEqual(course.course_status(args.root)['medium-ann']['status'], medium_status)
                receipt = course.read_json(args.root / 'resume_source.json')['medium-ann']
                self.assertEqual(receipt['selection_sha256'], batch.file_hash(selection_path))
                self.assertEqual(receipt['status'], medium_status)
                self.assertEqual({path: batch.file_hash(path) for path in prior.rglob('*') if path.is_file()}, before)
                with (args.root / 'training.csv').open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 24)
                self.assertEqual(sum(row['status'] == 'blocked_by_no_formal_pass' for row in rows),
                                 0 if winner_seed else 4)

    def historical_course_fixture(self, workspace):
        archive = workspace / 'historical'
        args = SimpleNamespace(root=workspace / 'continued', resume_from=archive,
            input_root=workspace / 'inputs', source_manifest=course.RESUME_SOURCE_MANIFEST,
            timeout=30., gate_timeout=10.)
        bc = {}
        for model in course.MODELS:
            relative = Path(f'{model}.pt')
            path = args.input_root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text(f'BC {model}', encoding='utf-8')
            bc[model] = (relative, batch.file_hash(path))
        source = dict(HEAD=course.HISTORICAL_HEAD, sha256=course.read_json(course.SOURCE_MANIFEST)['sha256'])
        batch.atomic_json(archive / 'course.json', dict(experiment=course.EXPERIMENT, code=source['sha256'],
            bc={model: value[1] for model, value in bc.items()}, plan=course.course_plan(archive)))
        pools = {}
        for partition, master_seed in (('validation', course.VALIDATION_SEED), ('independent', course.INDEPENDENT_SEED)):
            pools[partition] = {}
            for stage in course.V2_TD3_STAGES:
                path = archive / 'pools' / partition / f'v2_validation_{stage}.json'
                payload = [dict(scenario_id=str(i), payload=dict(state=[0, 0, 10, 0, 0],
                    goal=[master_seed, (1 + list(course.V2_TD3_STAGES).index(stage)) * 100 + i, 10], zones=[]))
                    for i in range(100)]
                batch.atomic_json(path, dict(scenarios=payload))
                digest = f'{partition}-{stage}'
                pools[partition][stage] = dict(path=str(path), sha256=batch.file_hash(path),
                    content_digest=digest, master_seed=master_seed, stage_seed=1)
        batch.atomic_json(archive / 'pools.json', pools)
        for model, winner_seed in (('ann', 23), ('snn', 11)):
            self.write_historical_batch_fixture(args, archive, pools, bc, f'easy-{model}', winner_seed,
                                                course.SOURCE_MANIFEST, course.HISTORICAL_HEAD)
        return args, archive, pools, bc

    def write_historical_batch_fixture(self, args, archive, pools, bc, name, winner_seed,
                                       source_manifest, head, parent=None):
        stage, model = name.split('-')
        baseline = course.read_json(batch.CONFIG)
        source = dict(HEAD=head, sha256=course.read_json(source_manifest)['sha256'])
        directory = archive / 'batches' / name
        attempt = directory / 'attempt-000001'
        info = pools['validation'][stage]
        gate, train = attempt / 'gate', attempt / 'train'
        paths = dict(pool=Path(info['path']), config=batch.CONFIG, experiment_config=batch.GUIDANCE_CONFIG,
                     source_manifest=source_manifest)
        hashes = {key: batch.source_hash(path) for key, path in paths.items() if key != 'pool'}
        hashes['pool'] = info['sha256']
        for kind, (relative, digest) in bc.items():
            paths[kind + '_bc'] = args.input_root / relative
            hashes[kind + '_bc'] = digest
            paths[kind + '_stage_start'] = Path(baseline[kind]['stage_start_server']).resolve()
            hashes[kind + '_stage_start'] = baseline[kind]['stage_start_sha256']
        if parent is not None:
            paths['parent'] = Path(parent['checkpoint'])
            hashes['parent'] = parent['checkpoint_sha256']
        start = 0 if parent is None else parent['steps']
        evidence = dict(source=source, paths={key: str(path) for key, path in paths.items()}, hashes=hashes,
            experiment=course.EXPERIMENT, global_steps_start=start, pool_digest=info['content_digest'])
        with patch.dict(batch.provenance.BC, bc):
            gate_args = course.batch_args(args, name, gate, info, parent, gate_only=True)
            train_args = course.batch_args(args, name, train, info, parent, gate_report=gate)
        gate_manifest = gate / 'manifest.json'
        gate_summary = gate / 'batch_summary.json'
        batch.atomic_json(gate_manifest, dict(plan=batch.plan(gate_args), inputs=evidence, batch_id='gate'))
        batch.atomic_json(gate_summary, dict(execution_completed=True, mps_cleanup=dict(stopped=True),
                                             gate_bindings=[]))
        gate_evidence = dict(root=str(gate), batch_id='gate', bindings=[],
            manifest_sha256=batch.file_hash(gate_manifest), summary_sha256=batch.file_hash(gate_summary))
        batch.atomic_json(train / 'manifest.json', dict(plan=batch.plan(train_args), inputs=evidence,
                                                       gate_evidence=gate_evidence))
        records = []
        pids = []
        for job in batch.batch_jobs(train_args):
            seed = job['seed']
            worker = train / job['id']
            init = (batch.initialization_binding(job, evidence) if parent is not None else dict(sha256=bc[model][1]))
            batch.atomic_json(worker / 'ready.json', dict(pid=seed, initialization=init))
            pids.append(seed)
            passed = winner_seed is not None
            checkpoint = worker / ('model.pt' if passed else 'model_failed.pt')
            checkpoint.write_text(f'{stage} {model} {seed}', encoding='utf-8')
            scenarios = [dict(scenario_id=str(i), outcome='collision' if i == 0 and seed != winner_seed
                              else 'goal') for i in range(100)]
            goal = 100 if seed == winner_seed else 99
            counts = {outcome: 0 for outcome in course.OUTCOMES}
            counts.update(goal=goal, collision=100-goal)
            terminal = worker / 'terminal_evaluation.json'
            batch.atomic_json(terminal, dict(actual_steps=160000, formal_passed=passed, checkpoint=str(checkpoint),
                checkpoint_sha256=batch.file_hash(checkpoint), evaluation=dict(scenarios=scenarios, goal_count=goal)))
            episodes = worker / 'metrics_reports' / 'episodes.jsonl'
            episodes.parent.mkdir()
            episodes.write_text(json.dumps(dict(episode=1, stage_steps=160000, episode_length=160000,
                                                outcome='goal')) + '\n', encoding='utf-8')
            records.append(dict(model=model, stage=stage, seed=seed, steps=160000, execution_completed=True,
                formal_passed=passed, checkpoint=str(checkpoint), checkpoint_sha256=batch.file_hash(checkpoint),
                terminal_evaluation=str(terminal), initialization=init, **counts,
                last_50k=dict(from_step_exclusive=110000, through_step=160000, count=1, successes=1, success_rate=1.)))
        batch.atomic_json(train / 'batch_summary.json', dict(execution_completed=True, mps_cleanup=dict(stopped=True),
              formal_passed={job['id']: winner_seed is not None for job in batch.batch_jobs(train_args)}))
        batch.atomic_json(train / 'release.json', dict(pids=pids))
        winner = course.select_winner(records)
        if winner is not None:
            (directory / 'winner.pt').write_bytes(Path(winner['checkpoint']).read_bytes())
        batch.atomic_json(directory / 'selection.json', course.promoted_selection(directory, records, winner))

    @staticmethod
    @contextmanager
    def mock_historical_inputs(pools, bc):
        baseline = course.read_json(batch.CONFIG)
        stage_starts = {Path(value['stage_start_server']): value['stage_start_sha256']
                        for value in (baseline[model] for model in course.MODELS)}
        file_hash = batch.file_hash

        def load_pool(path, **kwargs):
            # Restoring the published copies still uses the original scenario identities.
            info = pools[path.parent.name][kwargs['expected_level']]
            return SimpleNamespace(scenarios=course.read_json(path)['scenarios'],
                master_seed=info['master_seed'], stage_seed=info['stage_seed'], content_digest=info['content_digest'])

        def hash_input(path):
            path = Path(path)
            return stage_starts[path] if path in stage_starts else file_hash(path)

        def terminal(root, manifest, job, pool):
            path = root / job['id'] / 'terminal_evaluation.json'
            result = course.read_json(path)
            if batch.file_hash(result['checkpoint']) != result['checkpoint_sha256']:
                raise ValueError('Candidate checkpoint SHA256 differs')
            return result

        easy = pools['validation']['easy']
        with (patch.dict(batch.provenance.BC, bc), patch.object(batch.provenance, 'DIGEST', easy['content_digest']),
              patch.object(batch, 'POOL_SHA', easy['sha256']), patch.object(batch, 'file_hash', side_effect=hash_input),
              patch.object(course, 'load_v2_validation_pool', side_effect=load_pool),
              patch.object(batch, 'validate_terminal', side_effect=terminal),
              patch.object(batch, 'gate_bindings', return_value=[]),
              patch.object(course, 'generate_v2_validation_pool') as generate):
            yield
            generate.assert_not_called()

    def test_explicit_acc_is_available_across_formal_stages(self):
        for stage in ('easy', 'medium', 'hard'):
            with self.subTest(stage=stage):
                self.assertEqual(formal.V2FormalTrainingConfig(stage=stage, acc_eta=.01).acc_eta, .01)
                self.assertIsNone(formal.V2FormalTrainingConfig(stage=stage).acc_eta)
                with self.assertRaises(ValueError):
                    formal.V2FormalTrainingConfig(stage=stage, acc_eta=.01, target_q_aggregation='mean')

    def test_last_50k_uses_episode_end_and_excludes_boundary(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'episodes.jsonl'
            path.write_text('\n'.join(json.dumps(dict(episode=i, stage_steps=end,
                episode_length=length, outcome=outcome)) for i, (end, length, outcome) in enumerate(
                    ((100, 100, 'timeout'), (110, 10, 'goal'), (50100, 49990, 'collision')), 1)),
                encoding='utf-8')
            self.assertEqual(course.episode_window(path, 50105), dict(from_step_exclusive=105,
                through_step=50105, count=2, successes=1, success_rate=.5))
            self.assertEqual(course.episode_window(path, 50110)['count'], 1)

    def test_plan_uses_a1fceda_easy_settings_and_formal_later_defaults(self):
        plan = course.course_plan(Path('unused'))
        self.assertEqual(sum(map(len, plan['batches'].values())), 24)
        for model in ('ann', 'snn'):
            old = {job['seed']: batch.resolved(job, batch.GUIDANCE_EXPERIMENT)[0]
                   for job in batch.jobs(batch.GUIDANCE_EXPERIMENT) if job['model'] == model}
            for spec in plan['batches'][f'easy-{model}']:
                seed = spec['job']['seed']
                previous = old[seed if seed != 41 else 7]
                for name in previous:
                    if name not in ('job', 'formal_config', 'experiment', 'changes'):
                        self.assertEqual(spec[name], previous[name])
                self.assertEqual(spec['formal_config'], dict(previous['formal_config'], seed=seed))
            for stage, budget, actor_lr, critic_lr, mix in (
                    ('medium', 750000, 1.55e-4 if model == 'ann' else 9.5e-4, 2.5e-4, {'easy': .2, 'medium': .8}),
                    ('hard', 1000000, 1.125e-4, 2.125e-4, {'easy': .1, 'medium': .2, 'hard': .7})):
                for spec in plan['batches'][f'{stage}-{model}']:
                    config = spec['formal_config']
                    self.assertEqual((config['max_steps'], config['actor_lr'], config['critic_lr']), (budget, actor_lr, critic_lr))
                    self.assertEqual(config['curriculum_mix'], mix)
                    self.assertEqual(config['failure_sample_bias'], 1.)
                    self.assertEqual(spec['bc_schedule'], 'default')

    def test_rank_ignores_failed_candidates_and_resolves_each_tie(self):
        def candidate(seed, goal=97, collision=1, success=3, count=4, passed=True):
            return dict(seed=seed, goal=goal, collision=collision, formal_passed=passed,
                        last_50k=dict(successes=success, count=count), execution_completed=True)

        for seed7, seed11, seed23, seed41, expected in (
                (candidate(7, goal=98), candidate(11), candidate(23), candidate(41, goal=100, passed=False), 7),
                (candidate(7), candidate(11, collision=0), candidate(23), candidate(41), 11),
                (candidate(7), candidate(11), candidate(23, success=4), candidate(41), 23),
                (candidate(7, success=6, count=8), candidate(11), candidate(23), candidate(41), 7)):
            self.assertEqual(course.select_winner([seed41, seed23, seed7, seed11])['seed'], expected)
        failed = [candidate(seed, passed=False) for seed in (7, 11, 23, 41)]
        self.assertIsNone(course.select_winner(failed))
        with self.assertRaisesRegex(ValueError, 'four completed'):
            course.select_winner(failed[:-1])
        failed[0]['execution_completed'] = False
        with self.assertRaisesRegex(ValueError, 'four completed'):
            course.select_winner(failed)

    def test_duplicate_physical_scene_ignores_ids_and_metadata(self):
        payload = dict(state=[0, 0, 10, 0, 0], goal=[10, 0, 10], zones=[], metadata={'scenario_seed': 1})
        validation = {'easy': SimpleNamespace(scenarios=[dict(scenario_id='locked', payload=payload)])}
        def test_pool(goal):
            return {'hard': SimpleNamespace(scenarios=[dict(scenario_id='unrelated',
                payload=dict(payload, goal=goal, metadata={'scenario_seed': 42}))])}

        course.require_disjoint_pools(validation, test_pool([11, 0, 10]))
        with self.assertRaisesRegex(ValueError, 'Physical scenario duplicated'):
            course.require_disjoint_pools(validation, test_pool([10., 0., 10.]))

    def test_retry_keeps_failed_attempt_and_reuses_completed_batch(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(root=root, input_root=root, source_manifest=root / 'source.json',
                                   timeout=30, gate_timeout=10)
            kind = 'easy-ann'
            attempt = root / 'batches' / kind
            pool_info = dict(path=str(root / 'pool.json'), sha256='pool')
            launched = []

            def simulate_launch(launch):
                launched.append(launch.root)
                failed = launch.root == attempt / 'attempt-000001' / 'train'
                batch.atomic_json(launch.root / 'batch_summary.json', dict(
                    execution_completed=not failed, mps_cleanup=dict(stopped=True)))
                if failed:
                    raise RuntimeError('simulated worker exit')

            candidates = [dict(seed=seed, goal=95, collision=5, formal_passed=False,
                               execution_completed=True) for seed in (7, 11, 23, 41)]
            with (patch.object(batch, 'launch', side_effect=simulate_launch),
                  patch.object(batch, 'inputs', return_value={}),
                  patch.object(course, 'collect_batch', return_value=candidates)):
                self.assertEqual(course.course_status(root)[kind]['status'], 'missing')
                with self.assertRaisesRegex(RuntimeError, 'simulated worker exit'):
                    course.run_batch(args, kind, pool_info, None, None)
                self.assertEqual(course.course_status(root)[kind]['status'], 'process_failed')
                self.assertEqual(course.run_batch(args, kind, pool_info, None, None), (candidates, None))
                self.assertEqual(course.run_batch(args, kind, pool_info, None, None), (candidates, None))
            self.assertEqual(launched, [attempt / f'attempt-{i:06d}' / phase for i in (1, 2)
                                       for phase in ('gate', 'train')])
            self.assertEqual(course.course_status(root)[kind]['status'], 'no_formal_pass')
            self.assertFalse((attempt / 'winner.pt').exists())

    def test_retry_checks_both_phases_and_waits_for_owned_processes_to_exit(self):
        for phase_name in ('train', 'gate'):
            with self.subTest(phase=phase_name), TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = SimpleNamespace(root=root, input_root=root, source_manifest=root / 'source.json',
                                       timeout=30, gate_timeout=10)
                kind = 'easy-ann'
                attempt = root / 'batches' / kind / 'attempt-000001'
                phase = attempt / phase_name
                pipe = root / 'private-pipe'
                pipe.mkdir(mode=0o700)
                logs = phase / 'mps' / 'logs'
                logs.mkdir(parents=True)
                pid = 9021
                cleanup = dict(stopped=False, daemon_pid=9020, pipe_directory=str(pipe), log_directory=str(logs))
                if phase_name == 'train':
                    batch.atomic_json(phase / 'batch_summary.json', dict(execution_completed=False, mps_cleanup=cleanup))
                    batch.atomic_json(phase / 'ann_seed7' / 'process.json', dict(pid=pid, exit_code=None))
                else:
                    # Interrupted before the daemon/worker could write a completion or ready record.
                    batch.atomic_json(phase / 'manifest.json', dict(batch_id='aborted-gate'))
                    batch.atomic_json(phase / 'mps' / 'namespace.json',
                                      dict(batch_id='aborted-gate', pipe_directory=str(pipe), log_directory=str(logs)))
                proc = root / 'proc'
                live = proc / str(pid)
                live.mkdir(parents=True)
                (live / 'stat').write_text(f'{pid} (python) S 1 {pid} {pid}\n', encoding='utf-8')
                (live / 'environ').write_bytes(f'CUDA_MPS_PIPE_DIRECTORY={pipe}\0CUDA_MPS_LOG_DIRECTORY={logs}\0'.encode())
                with (patch.object(course, 'PROC_ROOT', proc, create=True),
                      patch.object(batch, 'launch', side_effect=RuntimeError('next launch')) as launch,
                      patch.object(batch, 'inputs', return_value={})):
                    with self.assertRaisesRegex(RuntimeError, f'{kind}.*{pid}') as caught:
                        course.run_batch(args, kind, dict(path='pool', sha256='test'), None, None)
                    self.assertIn(str(pipe), str(caught.exception))
                    self.assertIn(str(logs), str(caught.exception))
                    launch.assert_not_called()
                    self.assertFalse(attempt.with_name('attempt-000002').exists())

                    (live / 'stat').unlink()
                    (live / 'environ').unlink()
                    live.rmdir()
                    with self.assertRaisesRegex(RuntimeError, 'next launch'):
                        course.run_batch(args, kind, dict(path='pool', sha256='test'), None, None)
                    self.assertTrue(attempt.with_name('attempt-000002').is_dir())
                    self.assertEqual(launch.call_count, 1)

    def test_batch_names_are_scoped_by_experiment_in_the_real_cli(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'absent'
            common = ['--root', str(root), '--ann-bc', 'a', '--snn-bc', 's', '--validation-pool', 'pool']
            command = [sys.executable, '-B', '-m', batch.__name__]
            env = dict(os.environ, PYTHONPATH=str(batch.REPO / 'src'))

            def cli(mode, experiment, *extra):
                return subprocess.run([*command, mode, *common, '--experiment', experiment, *extra],
                                      env=env, cwd=batch.REPO, capture_output=True, text=True)

            for mode in ('plan', 'launch', 'worker', 'status', 'collect'):
                with self.subTest(mode=mode):
                    args = ['--batch', 'hard-snn']
                    if mode in ('launch', 'worker', 'collect'):
                        args.append('--launch')
                    if mode == 'worker':
                        args.extend(['--job', 'snn_seed7'])
                    invalid = cli(mode, batch.GUIDANCE_EXPERIMENT, *args)
                    self.assertNotEqual(invalid.returncode, 0)
                    self.assertIn('hard-snn', invalid.stderr)
                    self.assertIn(batch.GUIDANCE_EXPERIMENT, invalid.stderr)
                    self.assertFalse(root.exists())
            for experiment, name, seeds in (
                    (batch.GUIDANCE_EXPERIMENT, None, {7, 11, 23}),
                    (batch.GUIDANCE_EXPERIMENT, 'main', {7, 23}),
                    (batch.GUIDANCE_EXPERIMENT, 'seed11', {11}),
                    (batch.GUIDANCE_CONTROL, None, {11}),
                    (batch.GUIDANCE_CONTROL, 'seed11', {11}),
                    (batch.COURSE_EXPERIMENT, 'hard-snn', {7, 11, 23, 41})):
                with self.subTest(experiment=experiment, name=name):
                    result = cli('plan', experiment, *(['--batch', name] if name else []))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    jobs = [spec['job'] for spec in json.loads(result.stdout)['jobs']]
                    self.assertEqual({job['seed'] for job in jobs}, seeds)
                    if experiment == batch.COURSE_EXPERIMENT:
                        self.assertEqual([(j['model'], j['stage'], j['seed']) for j in jobs],
                                         [('snn', 'hard', seed) for seed in (7, 11, 23, 41)])
                    else:
                        self.assertEqual(len(jobs), 2 * len(seeds))
            for experiment, name in ((batch.GUIDANCE_EXPERIMENT, 'hard-snn'),
                                     (batch.GUIDANCE_CONTROL, 'main'), (batch.COURSE_EXPERIMENT, 'main')):
                with self.assertRaises(ValueError):
                    batch.jobs(experiment, name)

    def test_retry_blocks_when_an_interrupted_mps_namespace_cannot_be_identified(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(root=root, input_root=root, source_manifest=root / 'source.json',
                                   timeout=30, gate_timeout=10)
            attempt = root / 'batches' / 'hard-snn' / 'attempt-000001'
            (attempt / 'gate' / 'mps').mkdir(parents=True)
            with (patch.object(batch, 'inputs', return_value={}), patch.object(batch, 'launch') as launch):
                with self.assertRaisesRegex(RuntimeError, 'hard-snn.*pipe identity'):
                    course.run_batch(args, 'hard-snn', dict(path='pool', sha256='test'), None, None)
                launch.assert_not_called()
            self.assertFalse(attempt.with_name('attempt-000002').exists())

    def test_independent_eval_uses_both_frozen_policies_and_the_same_three_pools(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = {name: {} for name in ('validation', 'independent')}
            pools = {name: {} for name in evidence}
            winners = {}
            for model in ('ann', 'snn'):
                path = root / f'{model}.pt'
                path.write_text(model, encoding='utf-8')
                winners[model] = dict(checkpoint=str(path), checkpoint_sha256=batch.file_hash(path), seed=23)
            for name in evidence:
                for stage in ('easy', 'medium', 'hard'):
                    path = root / f'{name}-{stage}.json'
                    path.write_text(f'{name}-{stage}', encoding='utf-8')
                    evidence[name][stage] = dict(path=str(path), sha256=batch.file_hash(path))
                    start = (1000 if name == 'independent' else 0) + 100 * len(pools[name])
                    pools[name][stage] = SimpleNamespace(scenarios=[dict(scenario_id=f'{stage}_{index}',
                        payload=dict(state=[0, 0, 10, 0, 0], goal=[20 + start + index, 0, 10], zones=[]))
                        for index in range(100)])

            def frozen_checkpoint(*, model, checkpoint, validation_pool, device):
                return dict(actor=model, rewards=None, device='cpu', checkpoint_hash=batch.file_hash(checkpoint),
                            pool_hash=batch.file_hash(validation_pool), checkpoint_payload=dict(passed_validation=True))

            seen = []
            def fake_rollout(actor, pool, rewards, **kwargs):
                seen.append((actor, pool))
                scenarios = [dict(scenario_id=row['scenario_id'], outcome='goal') for row in pool.scenarios]
                return SimpleNamespace(to_dict=lambda: dict(goal_count=100, failure_count=0, scenario_count=100,
                    outcome_counts=dict(goal=100, ground=0, boundary=0, collision=0, timeout=0), scenarios=scenarios))

            with (patch('brain_uav.scripts.diagnose_v2_policy.load_diagnostic_inputs', side_effect=frozen_checkpoint),
                  patch.object(course, 'evaluate_v2_fixed_validation', side_effect=fake_rollout)):
                rows = course.evaluate_winners(root, winners, evidence, pools)
                self.assertEqual(course.evaluate_winners(root, winners, evidence, pools), rows)
                self.assertEqual(seen, [(model, pools['independent'][stage]) for model in ('ann', 'snn')
                                        for stage in ('easy', 'medium', 'hard')])
                self.assertEqual(sum(row['goal'] for row in rows), 600)
                output = root / 'independent' / 'ann-hard.json'
                record = course.read_json(output)
                record['evaluation']['scenarios'][0]['outcome'] = 'collision'
                record['evaluation'].update(goal_count=99, failure_count=1,
                                            outcome_counts=dict(goal=99, ground=0, boundary=0, collision=1, timeout=0))
                batch.atomic_json(output, record)
                tampered = output.read_bytes()
                with self.assertRaisesRegex(ValueError, 'ann-hard.json'):
                    course.evaluate_winners(root, winners, evidence, pools)
                self.assertEqual(output.read_bytes(), tampered)
                self.assertEqual(len(seen), 6)

                output.with_suffix('.sha256.json').unlink()
                self.assertEqual(course.evaluate_winners(root, winners, evidence, pools), rows)
                self.assertEqual(seen[6:], [('ann', pools['independent']['hard'])])
                good_raw = output.read_bytes()
                digest_path = output.with_suffix('.sha256.json')
                good_digest = digest_path.read_bytes()
                record = json.loads(good_raw)
                record['evaluation']['outcome_counts']['collision'] = 1
                batch.atomic_json(output, record)
                batch.atomic_json(digest_path, dict(identity=record['identity'], sha256=batch.file_hash(output)))
                with self.assertRaisesRegex(ValueError, 'ann-hard.json.*counts disagree'):
                    course.evaluate_winners(root, winners, evidence, pools)
                output.write_bytes(good_raw)
                digest_path.write_bytes(good_digest)
                pools['independent']['hard'] = pools['validation']['easy']
                with self.assertRaisesRegex(ValueError, 'Physical scenario duplicated'):
                    course.evaluate_winners(root, winners, evidence, pools)

    def test_course_resume_reaches_only_independent_evaluation_after_hard_winners_freeze(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(root=root, input_root=root, source_manifest=course.SOURCE_MANIFEST,
                                   validation_pool_dir=root)
            training = root / 'training.csv'
            training.write_text('sealed training table', encoding='utf-8')
            winners = {model: dict(seed=23, checkpoint=f'{model}.pt') for model in ('ann', 'snn')}
            selection_hashes = {}
            for model, winner in winners.items():
                selection = root / 'batches' / f'hard-{model}' / 'selection.json'
                batch.atomic_json(selection, dict(winner=winner))
                selection_hashes[model] = sha256(selection.read_bytes()).hexdigest()
            batch.atomic_json(root / 'independent_inputs.json', dict(hard_winners=winners, training_rows=24,
                training_sha256=batch.file_hash(training), hard_selection_sha256=selection_hashes))

            def input_hash(path):
                for relative, expected in batch.provenance.BC.values():
                    if path == root / relative:
                        return expected
                return sha256(path.read_bytes()).hexdigest()

            with (patch.object(batch, 'identity', return_value=course.read_json(course.SOURCE_MANIFEST)),
                  patch.object(batch, 'file_hash', side_effect=input_hash),
                  patch.object(course, 'prepare_pools', return_value=({}, {})),
                  patch.object(course, 'run_batch') as run_batch,
                  patch.object(course, 'select_winner') as select_winner,
                  patch.object(course, 'evaluate_winners', return_value=[]) as evaluate):
                self.assertEqual(course._run_locked(args), 0)
                evaluate.assert_called_once_with(root, winners, {}, {})
                run_batch.assert_not_called()
                select_winner.assert_not_called()
                selection.write_text('changed selection', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Frozen winner selection changed'):
                    course._run_locked(args)

    def test_periodic_snapshot_and_evaluation_use_actual_acc_episode_end(self):
        from brain_uav.config import ScenarioConfig
        from brain_uav.trainers.v2_formal_training import V2FormalStageTrainer, load_v2_periodic_snapshot

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            scenario = ScenarioConfig(max_steps=2)

            def unblocked_bc_scene(index, zone_count):
                payload = make_scenario_payload(index, zone_count)
                payload['metadata']['requested_direct_path_blocker'] = False
                return payload

            with patch('test_v2_bc.make_scenario_payload', side_effect=unblocked_bc_scene):
                cluster = write_cluster(root / 'cluster', scenario_config=scenario, zone_counts=(0, 0, 0, 0))
            train_v2_behavior_cloning(trajectory_cluster=cluster, output_dir=root / 'bc', epochs=1,
                batch_size=2, validation_fraction=.25, model='snn', device='cpu')
            pool = root / 'pool.json'
            save_v2_validation_pool(pool, generate_v2_validation_pool(scenario, 'easy',
                scenario_count=1, master_seed=123))
            run = V2FormalStageTrainer.run

            def run_near_snapshot(trainer):
                trainer.result.stage_steps = 24999
                return run(trainer)

            with patch.object(V2FormalStageTrainer, 'run', run_near_snapshot):
                single_stage.run_v2_td3_stage(stage='easy', model='snn', acc_eta=.01, device='cpu',
                    init_checkpoint=root / 'bc' / 'bc_v2_snn_best.pt', output=root / 'model.pt',
                    metrics_out=root / 'metrics.json', validation_pool=pool, validation_scenario_count=1,
                    validation_max_failures=0, max_stage_steps=25001, global_steps_start=100, periodic_snapshot_interval_steps=25000)
            saved = root / 'model_periodic' / 'step_000025000.pt'
            payload = load_v2_periodic_snapshot(saved, expected_stage='easy', expected_model_type='snn')
            self.assertEqual((payload['scheduled_stage_steps'], payload['stage_steps']), (25000, 25001))
            controller = payload['acc_state']['controller']
            self.assertEqual([row['stage_steps'] for row in controller['controls']], [25001])
            self.assertEqual([row['step'] for row in controller['episodes'][0]], [25000, 25001])
            record = json.loads((root / 'metrics_reports' / 'periodic_validation.jsonl').read_text(encoding='utf-8'))
            self.assertEqual((record['scheduled_stage_steps'], record['stage_steps'], record['global_steps']),
                             (25000, 25001, 25101))
            old = payload.copy()
            del old['scheduled_stage_steps']
            legacy_file = root / 'legacy_periodic.pt'
            torch.save(old, legacy_file)
            self.assertEqual(load_v2_periodic_snapshot(legacy_file)['stage_steps'], 25001)
            with self.assertRaisesRegex(ValueError, 'scheduled step exceeds'):
                load_v2_periodic_snapshot(dict(payload, scheduled_stage_steps=25002))
            controller['controls'][0]['stage_steps'] = 25002
            controller['last_adjust'] = 25002
            with self.assertRaisesRegex(ValueError, 'Invalid ACC update interval'):
                load_v2_periodic_snapshot(payload)

    def test_real_workers_build_all_child_seeds_from_the_selected_actor(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        real_stage = single_stage.run_v2_td3_stage

        class TrainerReached(Exception):
            pass

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for model in ('ann', 'snn'):
                model_root = root / model
                easy_job = batch.jobs(course.EXPERIMENT, f'easy-{model}')[-2]
                spec, config = course.resolve_course_job(easy_job)
                scenario = scenario_config_from_snapshot(spec['scenario_config'])
                rewards = reward_config_from_snapshot(spec['reward_config'])
                def unblocked_bc_scene(index, zone_count):
                    payload = make_scenario_payload(index, zone_count)
                    payload['metadata']['requested_direct_path_blocker'] = False
                    return payload

                with patch('test_v2_bc.make_scenario_payload', side_effect=unblocked_bc_scene):
                    cluster = write_cluster(model_root / 'cluster', scenario_config=scenario, zone_counts=(0, 0, 0, 0))
                bc_dir = model_root / 'bc'
                train_v2_behavior_cloning(trajectory_cluster=cluster, output_dir=bc_dir, epochs=1,
                    batch_size=2, validation_fraction=.25, device='cpu', model=model,
                    actor_hidden_dim=192, actor_projection_dim=32)
                bc = bc_dir / f'bc_v2_{model}_best.pt'
                components = formal.build_v2_stage_engine(scenario, config, init_checkpoint=bc,
                    model_type=model, rewards=rewards, device='cpu')
                loaded_actor = {key: value.clone() for key, value in components.engine.actor.state_dict().items()}
                parent = None
                start = 0

                for stage in ('easy', 'medium', 'hard'):
                    pool_path = model_root / f'{stage}.json'
                    save_v2_validation_pool(pool_path, generate_v2_validation_pool(scenario, stage,
                        scenario_count=6, master_seed=123))
                    expected_parent = bc if stage == 'easy' else parent
                    parent_hash = batch.file_hash(expected_parent)
                    hashes = {f'{model}_bc' if stage == 'easy' else 'parent': parent_hash,
                              'pool': batch.file_hash(pool_path)}
                    self.assertEqual(course.parent_steps(stage, model, parent, parent_hash if parent else None, hashes), start)
                    if parent:
                        with self.assertRaisesRegex(ValueError, 'parent checkpoint SHA256'):
                            course.parent_steps(stage, model, parent, '0' * 64, hashes)

                    batch_id = f'{stage}-{model}'
                    jobs = batch.jobs(course.EXPERIMENT, batch_id)
                    attempt = model_root / stage
                    attempt.mkdir()
                    paths = dict(ann_bc=str(bc), snn_bc=str(bc), pool=str(pool_path))
                    if parent:
                        paths['parent'] = str(parent)
                    evidence = dict(paths=paths, hashes=hashes, global_steps_start=start)
                    seen = {}
                    actor_lr, critic_lr = {
                        'easy': (1.55e-4, 2e-4) if model == 'ann' else (8.25e-4, 1e-3),
                        'medium': (1.55e-4 if model == 'ann' else 9.5e-4, 2.5e-4),
                        'hard': (1.125e-4, 2.125e-4),
                    }[stage]

                    def inspect_trainer(trainer):
                        engine = trainer.engine
                        self.assertEqual(trainer.config.actor_lr, actor_lr)
                        self.assertEqual(trainer.config.critic_lr, critic_lr)
                        self.assertEqual(trainer.result.global_steps_start, start)
                        self.assertEqual(trainer.rewards.upper_boundary_directional_warning_distance, 120.)
                        self.assertEqual(trainer.rewards.boundary_directional_warning_distance, 60.)
                        self.assertEqual((engine.terminal_geo_radius, engine.terminal_geo_lambda,
                                          engine.terminal_geo_far_weight), (400., 3000., .5))
                        self.assertEqual(trainer.config.terminal_geo_safe_clearance, 40.)
                        self.assertEqual(engine.target_q_aggregation, 'min')
                        for key, value in engine.actor.state_dict().items():
                            torch.testing.assert_close(value, loaded_actor[key], rtol=0, atol=0)
                        for key, value in engine.bc_reference_actor.state_dict().items():
                            torch.testing.assert_close(value, loaded_actor[key], rtol=0, atol=0)
                        if model == 'snn':
                            self.assertEqual(engine.research_beta, .5)
                            self.assertEqual(engine.formal_acc.controller.eta, .01)
                            self.assertFalse(engine.formal_acc.controller.episodes)
                            self.assertFalse(engine.formal_acc.controller.controls)
                        else:
                            self.assertIsNone(getattr(engine, 'research_target_config', None))
                            self.assertIsNone(getattr(engine, 'formal_acc', None))
                        seen[trainer.config.seed] = trainer
                        raise TrainerReached

                    def cpu_stage(**kwargs):
                        self.assertEqual(kwargs['init_checkpoint'], expected_parent)
                        self.assertTrue(kwargs['gpu_td3_compute'])
                        self.assertTrue(kwargs['cuda_graph_updates'])
                        gate = kwargs['initialization_gate']

                        def emulate_cuda_ready(engine, *args):
                            requested = (engine.gpu_td3_compute, engine.non_blocking_batch_transfer)
                            engine.gpu_td3_compute = engine.non_blocking_batch_transfer = True
                            try:
                                gate(engine, *args)
                            finally:
                                engine.gpu_td3_compute, engine.non_blocking_batch_transfer = requested

                        cuda_flags = spec['compilation']
                        kwargs.update({key: 'eager' if key == 'frozen_critic_strategy' else False for key in cuda_flags})
                        kwargs.update(device='cpu', gpu_td3_compute=False, non_blocking_batch_transfer=False,
                                      validation_scenario_count=6, reporting=False, initialization_gate=emulate_cuda_ready)
                        return real_stage(**kwargs)

                    batch.atomic_json(attempt / 'release.json', dict(pids=[os.getpid()]))
                    for job in jobs:
                        with self.subTest(model=model, stage=stage, seed=job['seed']):
                            argv = ['worker', '--launch', '--root', str(attempt),
                                '--experiment', course.EXPERIMENT, '--batch', batch_id, '--job', job['id'],
                                '--ann-bc', str(bc), '--snn-bc', str(bc), '--validation-pool', str(pool_path),
                                '--pool-sha256', hashes['pool']]
                            if parent:
                                argv.extend(['--parent', str(parent), '--parent-sha256', parent_hash])
                            args = batch.parser().parse_args(argv)
                            batch.atomic_json(attempt / 'manifest.json', dict(
                                plan=batch.plan(args), inputs=evidence, batch_id='course-test'))
                            with (patch.object(batch, 'check_inputs'),
                                  patch.object(torch.cuda, 'synchronize'),
                                  patch.object(single_stage, 'run_v2_td3_stage', side_effect=cpu_stage),
                                  patch.object(formal.V2FormalStageTrainer, 'run', new=inspect_trainer)):
                                if parent:
                                    with self.assertRaisesRegex(ValueError, 'pool/parent'):
                                        batch.worker(SimpleNamespace(**{**vars(args), 'parent_sha256': 'x'}))
                                with self.assertRaises(TrainerReached):
                                    batch.worker(args)
                            ready = course.read_json(attempt / job['id'] / 'ready.json')
                            self.assertEqual(ready['initialization']['sha256'], parent_hash)
                            self.assertEqual(ready['seed_manifest']['base_seed'], job['seed'])
                            _, config_job = course.resolve_course_job(job)
                            batch.provenance.validate_seed_manifest(config_job, ready['seed_manifest'])
                    self.assertEqual(set(seen), {7, 11, 23, 41})
                    trainer = seen[23]
                    if stage != 'hard':
                        with torch.no_grad():
                            next(trainer.engine.actor.parameters()).add_(.015)
                        loaded_actor = {key: value.clone() for key, value in trainer.engine.actor.state_dict().items()}
                    ready = course.read_json(attempt / f'{model}_seed23' / 'ready.json')
                    source = (components.initialization_source if stage == 'easy' else
                        dict(kind='validated_v2_td3_stage', model_type=model,
                             previous_stage=course.PREDECESSOR[stage], path=str(expected_parent)))
                    parent = self.make_parent(model_root, trainer.config, trainer.engine, scenario, rewards,
                                             ready['seed_manifest'], source, start)
                    start += 26000
                    if model == 'snn':
                        saved = formal.load_v2_formal_checkpoint(parent)
                        self.assertNotEqual(saved['acc_state']['controller']['beta'], .5)
                        bank = saved['acc_state']['controller']['episodes']
                        self.assertEqual(len(bank), 1)
                        restored = load_state(trainer.engine, trainer.config, saved)
                        self.assertEqual(restored.controller.beta, saved['acc_state']['controller']['beta'])
                        self.assertEqual(len(restored.controller.episodes), 1)

    @staticmethod
    def make_parent(root, config, engine, scenario, rewards, seeds, source, start):
        result = formal.V2FormalTrainingResult.empty(config.stage, global_steps_start=start)
        result.stage_steps = 26000
        result.global_steps_end = start + result.stage_steps
        result.status = 'passed'
        result.passed_validation = True
        result.stop_reason = 'fixed_validation_passed'
        result.validation_records.append({'passed': True})
        if engine.model_type == 'snn':
            from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
            env = V2StaticNoFlyTrajectoryEnv(scenario, rewards, fixed_scenarios=[
                _scenario((0, 0, 200, 0, 0), (401, 0, 200), (), level=config.stage)])
            observation = env.reset()[0]
            row = SimpleNamespace(observation=observation, reward=100., done=True,
                                  action=np.zeros(2, dtype=np.float32), collected_stage_step=26000)
            engine.formal_acc.episode(SimpleNamespace(engine=engine, result=result), [row])
        payload = formal.build_v2_formal_checkpoint(engine, result, config,
            scenario=scenario, rewards=rewards, uav_collision_radius=0., seed_manifest=seeds,
            validation_pool_metadata=dict(path='fixture', format_version=1, curriculum_level=config.stage,
                master_seed=123, stage_seed=234, scenario_count=100, content_digest='fixture'),
            initialization_source=source, bc_schedule='smooth-mid-strong' if config.stage == 'easy' else 'default')
        path = root / f'{config.stage}_winner.pt'
        formal.save_v2_formal_checkpoint(path, payload)
        return path
