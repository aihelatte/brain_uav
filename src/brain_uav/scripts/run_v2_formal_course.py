"""Run the four-seed formal V2 course and evaluate its frozen hard winners."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import sys

from brain_uav.scripts import run_v2_formal_easy_baseline as batch_runner
from brain_uav.scripts import run_v2_online_wd3_acc as provenance
from brain_uav.scripts.diagnose_v2_wd3_acc import scene_key
from brain_uav.scripts.v2_formal_easy_guidance import OUTCOMES, last_complete_episodes, outcome, read_json
from brain_uav.scripts.train_v2_td3 import _resolve_v2_actor_lr
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig, load_v2_formal_checkpoint, v2_bc_schedule_metadata,
)
from brain_uav.trainers.v2_online_research import atomic_json
from brain_uav.trainers.v2_validation import (
    evaluate_v2_fixed_validation, generate_v2_validation_pool, load_v2_validation_pool,
    scenario_config_from_snapshot,
)
from brain_uav.v2_curriculum import V2_TD3_STAGES


EXPERIMENT = batch_runner.COURSE_EXPERIMENT
MODELS = ('ann', 'snn')
SEEDS = batch_runner.COURSE_SEEDS
VALIDATION_SEED = 20260904
INDEPENDENT_SEED = 20261121
SOURCE_MANIFEST = batch_runner.REPO / 'scripts/v2_formal_course_source.json'
RESUME_SOURCE_MANIFEST = batch_runner.REPO / 'scripts/v2_formal_course_resume_source.json'
HISTORICAL_HEAD = 'e6bc9be6b1cd95ac5c23b370f373b7eb3050c6df'
PREDECESSOR = dict(medium='easy', hard='medium')
PROC_ROOT = Path('/proc')


def resolve_course_job(job):
    model, stage, seed = job['model'], job['stage'], job['seed']
    easy_seed = 7 if seed == 41 else seed
    reference = next(j for j in batch_runner.jobs(batch_runner.GUIDANCE_EXPERIMENT)
                     if j['model'] == model and j['seed'] == easy_seed)
    spec, config = batch_runner.resolved(reference, batch_runner.GUIDANCE_EXPERIMENT)
    periodic = None
    if stage == 'easy':
        config = replace(config, seed=seed)
        spec['changes']['seed'][1] = seed
        source = batch_runner.GUIDANCE_EXPERIMENT
    else:
        config = V2FormalTrainingConfig(
            stage=stage, seed=seed, actor_lr=_resolve_v2_actor_lr(stage, model, None),
            spike_regularization=config.spike_regularization, acc_eta=config.acc_eta,
            terminal_geo_radius=config.terminal_geo_radius,
            terminal_geo_far_weight=config.terminal_geo_far_weight,
        )
        spec['bc_schedule'] = 'default'
        spec.pop('changes')
        source = 'formal curriculum defaults with safe-direct rewards/guidance and inherited model policy settings'
        periodic = 150_000  # Observational cadence of the formal curriculum CLI.
    spec.update(job=job, experiment=EXPERIMENT, formal_config=config.to_dict(),
                training_config_source=source, periodic_snapshot_interval_steps=periodic)
    return spec, config


def parent_steps(stage, model, path, expected_hash, hashes):
    if stage == 'easy':
        return 0
    if hashes['parent'] != expected_hash:
        raise ValueError('Selected parent checkpoint SHA256 mismatch.')
    raw = load_v2_formal_checkpoint(path, expected_model_type=model, next_stage=stage,
                                    require_passed=True)
    parent_seed = raw['formal_config']['seed']
    candidates = batch_runner.jobs(EXPERIMENT, f'{PREDECESSOR[stage]}-{model}')
    job = next((candidate for candidate in candidates if candidate['seed'] == parent_seed), None)
    if job is None:
        raise ValueError('Selected parent seed was not scheduled in this course.')
    spec, config = resolve_course_job(job)
    if (raw['formal_config'] != config.to_dict() or raw['reward_config'] != spec['reward_config']
            or raw['scenario_config'] != spec['scenario_config']
            or raw['uav_collision_radius'] != spec['uav_collision_radius']
            or raw['bc_schedule'] != v2_bc_schedule_metadata(config.stage, schedule=spec['bc_schedule'])):
        raise ValueError('Selected parent differs from the formal course configuration.')
    return raw['training_result']['global_steps_end']


def episode_window(path, terminal_steps):
    history = last_complete_episodes(path, terminal_steps, by_end=True)
    history['from_step_exclusive'] = history.pop('from_step')
    return history


def select_winner(records):
    if len(records) != len(SEEDS) or {row['seed'] for row in records} != set(SEEDS) or any(
            row['execution_completed'] is not True or type(row['formal_passed']) is not bool
            for row in records):
        raise ValueError('Promotion requires four completed candidates.')

    def rank(row):
        window = row['last_50k']
        rate = Fraction(window['successes'], window['count']) if window['count'] else -1
        return -row['goal'], row['collision'], -rate, row['seed']

    return min((row for row in records if row['formal_passed']), key=rank, default=None)


def require_disjoint_pools(validation, independent):
    def keys(pools):
        return [scene_key(row['payload']) for pool in pools.values() for row in pool.scenarios]

    promotion_keys, independent_keys = keys(validation), keys(independent)
    if (len(set(promotion_keys)) != len(promotion_keys)
            or len(set(independent_keys)) != len(independent_keys)
            or set(promotion_keys).intersection(independent_keys)):
        raise ValueError('Physical scenario duplicated within or across promotion and independent pools.')


def seal_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Sealed record differs from current evidence: {path}')
    else:
        atomic_json(path, value)


def seal_copy(source, target, expected_hash):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.tmp')
        shutil.copyfile(source, temporary)
        if batch_runner.file_hash(temporary) != expected_hash:
            raise ValueError(f'Input changed during copy: {source}')
        temporary.replace(target)
    if batch_runner.file_hash(target) != expected_hash:
        raise ValueError(f'Sealed copy differs from expected input: {target}')


def prepare_pools(root, source_dir, *, read_only=False):
    """Seal both partitions before the first CUDA gate."""
    specs = [resolve_course_job(batch_runner.jobs(EXPERIMENT, f'easy-{m}')[0])[0] for m in MODELS]
    if any(specs[0][key] != specs[1][key] for key in ('scenario_config', 'uav_collision_radius')):
        raise ValueError('Models disagree on the shared course pool contract.')
    scenario = scenario_config_from_snapshot(specs[0]['scenario_config'])
    radius = specs[0]['uav_collision_radius']
    lock = root / 'pools.json'
    saved = read_json(lock) if read_only or lock.exists() else None
    evidence, pools = {}, {}
    for partition, seed in (('validation', VALIDATION_SEED), ('independent', INDEPENDENT_SEED)):
        evidence[partition], pools[partition] = {}, {}
        for stage in V2_TD3_STAGES:
            name = f'v2_validation_{stage}.json'
            path = root / 'pools' / partition / name
            source = source_dir / name if not read_only else None
            if saved is not None and batch_runner.file_hash(path) != saved[partition][stage]['sha256']:
                raise ValueError(f'Locked pool changed: {path}')
            if saved is None and partition == 'validation' and (stage == 'easy' or source.exists()):
                expected = batch_runner.POOL_SHA if stage == 'easy' else batch_runner.file_hash(source)
                seal_copy(source, path, expected)
            if not read_only and not path.exists():
                pool = generate_v2_validation_pool(scenario, stage, master_seed=seed,
                                                   uav_collision_radius=radius)
                atomic_json(path, pool.to_dict())
            pool = load_v2_validation_pool(path, expected_level=stage, expected_scenario=scenario,
                                           expected_count=100, expected_uav_collision_radius=radius,
                                           expected_master_seed=seed)
            pools[partition][stage] = pool
            evidence[partition][stage] = dict(path=str(path), sha256=batch_runner.file_hash(path),
                content_digest=pool.content_digest, master_seed=pool.master_seed, stage_seed=pool.stage_seed)
    original = evidence['validation']['easy']
    if original['sha256'] != batch_runner.POOL_SHA or original['content_digest'] != provenance.DIGEST:
        raise ValueError('Course easy requires the original safe-direct promotion pool.')
    require_disjoint_pools(pools['validation'], pools['independent'])
    if read_only:
        if saved != evidence:
            raise ValueError(f'Historical pool seal differs from the verified pools: {lock}')
    else:
        seal_json(lock, evidence)
    return evidence, pools


def batch_args(args, batch, root, pool_info, parent, *, gate_only=False, gate_report=None):
    ann_bc, snn_bc = [args.input_root / provenance.BC[model][0] for model in MODELS]
    return argparse.Namespace(mode='launch', root=root, experiment=EXPERIMENT, batch=batch,
        ann_bc=ann_bc, snn_bc=snn_bc, validation_pool=Path(pool_info['path']),
        source_manifest=args.source_manifest, timeout=args.timeout, gate_timeout=args.gate_timeout,
        launch=True, gate_only=gate_only, gate_report=gate_report, pool_sha256=pool_info['sha256'],
        parent=Path(parent['checkpoint']) if parent else None,
        parent_sha256=parent['checkpoint_sha256'] if parent else None)


def counts_for_pool(evaluation, pool):
    scenes = evaluation['scenarios']
    labels = Counter(outcome(row) for row in scenes)
    if ([row['scenario_id'] for row in scenes] != [row['scenario_id'] for row in pool.scenarios]
            or evaluation['goal_count'] != labels['goal']):
        raise ValueError('Evaluation differs from the locked scene sequence.')
    return {label: labels[label] for label in OUTCOMES}


def collect_batch(args, pool, evidence, *, historical_source=None):
    root = args.root
    manifest = read_json(root / 'manifest.json')
    status = read_json(root / 'batch_summary.json')
    if historical_source is not None:
        check_historical_easy_inputs(args, evidence, historical_source)
    expected_gate = batch_runner.validate_gate(args, evidence)
    expected_plan = batch_runner.plan(args)
    expected_plan['initialization_timeout'] = manifest['plan']['initialization_timeout']
    if (status['execution_completed'] is not True or status['mps_cleanup']['stopped'] is not True
            or manifest['plan'] != expected_plan
            or manifest['gate_evidence'] != expected_gate
            or manifest['inputs']['paths'] != evidence['paths']
            or manifest['inputs']['hashes'] != evidence['hashes']
            or manifest['inputs']['source']['sha256'] != evidence['source']['sha256']
            or manifest['inputs']['global_steps_start'] != evidence['global_steps_start']
            or manifest['inputs']['pool_digest'] != pool.content_digest):
        raise ValueError('Training batch inputs, completion or gate evidence differ.')
    records = []
    pids = set()
    for job in batch_runner.batch_jobs(args):
        validated = batch_runner.validate_terminal(root, manifest, job, pool)
        directory = root / job['id']
        ready = read_json(directory / 'ready.json')
        pids.add(ready['pid'])
        steps = validated['actual_steps']
        record = dict(model=job['model'], stage=job['stage'], seed=job['seed'], steps=steps,
            execution_completed=True, formal_passed=validated['formal_passed'],
            checkpoint=validated['checkpoint'], checkpoint_sha256=validated['checkpoint_sha256'],
            initialization=ready['initialization'], terminal_evaluation=str(directory / 'terminal_evaluation.json'),
            last_50k=episode_window(directory / 'metrics_reports' / 'episodes.jsonl', steps),
            **counts_for_pool(validated['evaluation'], pool))
        if status['formal_passed'][job['id']] != record['formal_passed']:
            raise ValueError('Worker acceptance differs from batch summary.')
        records.append(record)
    if pids != set(read_json(root / 'release.json')['pids']) or len(pids) != len(SEEDS):
        raise ValueError('Training release PIDs differ from the completed workers.')
    if historical_source is None:
        batch_runner.check_inputs(evidence)
    return records


def check_historical_easy_inputs(args, evidence, source):
    """Verify e6bc9be inputs against the historical seal, without using live source identity."""
    paths = evidence['paths']
    hashes = evidence['hashes']
    sealed_files = dict(config=batch_runner.CONFIG, experiment_config=batch_runner.GUIDANCE_CONFIG,
                        source_manifest=SOURCE_MANIFEST)
    expected_hashes = {key: batch_runner.source_hash(path) for key, path in sealed_files.items()}
    baseline = read_json(batch_runner.CONFIG)
    expected_paths = dict(pool=args.validation_pool)
    expected_hashes['pool'] = args.pool_sha256
    for model in MODELS:
        expected_paths[model + '_bc'] = getattr(args, model + '_bc')
        expected_hashes[model + '_bc'] = provenance.BC[model][1]
        expected_paths[model + '_stage_start'] = Path(baseline[model]['stage_start_server'])
        expected_hashes[model + '_stage_start'] = baseline[model]['stage_start_sha256']
    gate_source = read_json(args.gate_report / 'manifest.json')['inputs']['source']
    if (evidence['experiment'] != EXPERIMENT or evidence['global_steps_start'] != 0
            or evidence['source']['HEAD'] != HISTORICAL_HEAD or gate_source['HEAD'] != HISTORICAL_HEAD
            or evidence['source']['sha256'] != source or hashes != expected_hashes
            or set(paths) != set(expected_hashes)):
        raise ValueError(f'Historical easy source or input seal differs: {args.root / "manifest.json"}')
    for key, path in expected_paths.items():
        if Path(paths[key]) != path.resolve() or batch_runner.file_hash(path) != hashes[key]:
            raise ValueError(f'Historical easy input path or SHA256 differs: {path}')


def promoted_selection(directory, candidates, winner):
    if winner is not None:
        winner = {**winner, 'candidate_checkpoint': winner['checkpoint'],
                  'checkpoint': str(directory / 'winner.pt')}
    return dict(candidates=candidates, winner=winner,
                status='promoted' if winner is not None else 'no_formal_pass')


def completed_attempts(directory):
    return [path for path in sorted(directory.glob('attempt-*'))
            if (path / 'train/batch_summary.json').exists()
            and read_json(path / 'train/batch_summary.json')['execution_completed'] is True]


def import_easy(args, archive, bc):
    source = read_json(SOURCE_MANIFEST)['sha256']
    archive_course = archive / 'course.json'
    if read_json(archive_course) != dict(experiment=EXPERIMENT, code=source, bc=bc,
                                         plan=course_plan(archive)):
        raise ValueError(f'Historical course differs from the e6bc9be seal: {archive_course}')
    evidence, pools = prepare_pools(archive, None, read_only=True)
    imported = {}
    originals = {}
    for model in MODELS:
        batch = f'easy-{model}'
        directory = archive / 'batches' / batch
        completed = completed_attempts(directory)
        if len(completed) != 1:
            raise ValueError(f'Expected one completed historical training attempt: {directory}')
        attempt = completed[0]
        training = batch_args(args, batch, attempt / 'train', evidence['validation']['easy'],
                              None, gate_report=attempt / 'gate')
        manifest = training.root / 'manifest.json'
        old_inputs = read_json(manifest)['inputs']
        candidates = collect_batch(training, pools['validation']['easy'], old_inputs,
                                   historical_source=source)
        winner = select_winner(candidates)
        selection = directory / 'selection.json'
        old_winner = directory / 'winner.pt'
        if winner is None or read_json(selection) != promoted_selection(directory, candidates, winner):
            raise ValueError(f'Historical easy promotion differs from terminal candidate evidence: {selection}')
        if batch_runner.file_hash(old_winner) != winner['checkpoint_sha256']:
            raise ValueError(f'Historical easy winner SHA256 differs: {old_winner}')
        imported[model] = promoted_selection(args.root / 'batches' / batch, candidates, winner)
        originals[model] = dict(attempt=str(attempt), selection=str(selection),
            selection_sha256=batch_runner.file_hash(selection), training_manifest_sha256=batch_runner.file_hash(manifest),
            gate_manifest_sha256=batch_runner.file_hash(training.gate_report / 'manifest.json'),
            winner=str(old_winner), winner_sha256=winner['checkpoint_sha256'],
            imported_winner=imported[model]['winner']['checkpoint'])
    seal_json(args.root / 'resume_source.json', dict(root=str(archive), HEAD=HISTORICAL_HEAD,
        course_sha256=batch_runner.file_hash(archive_course),
        pool_seal_sha256=batch_runner.file_hash(archive / 'pools.json'), pools=evidence, easy=originals))
    local_evidence = {}
    for partition, levels in evidence.items():
        local_evidence[partition] = {}
        for stage, info in levels.items():
            source_path = Path(info['path'])
            target = args.root / 'pools' / partition / source_path.name
            seal_copy(source_path, target, info['sha256'])
            local_evidence[partition][stage] = {**info, 'path': str(target)}
    seal_json(args.root / 'pools.json', local_evidence)
    for model, selection in imported.items():
        target = Path(selection['winner']['checkpoint'])
        seal_copy(Path(originals[model]['winner']), target, originals[model]['winner_sha256'])
        seal_json(target.parent / 'selection.json', selection)
    return local_evidence, pools, imported


def require_stopped_phase(phase, batch):
    """Read this phase's evidence and the local process table without contacting MPS."""
    if not phase.exists():
        return
    logs = phase / 'mps' / 'logs'
    pipes, worker_pids, service_pids = set(), set(), set()
    live = {}

    def optional(path):
        return read_json(path) if path.exists() else {}

    def record_session(record):
        if record.get('pipe_directory'):
            pipes.add(record['pipe_directory'])
        if record.get('log_directory') not in (None, str(logs)):
            raise ValueError('Unexpected MPS log directory')
        if record.get('daemon_pid') is not None:
            service_pids.add(record['daemon_pid'])
        for client, server in record.get('client_server_pid', {}).items():
            worker_pids.add(int(client))
            service_pids.add(server)

    try:
        summary = optional(phase / 'batch_summary.json')
        cleanups = [record for record in (summary.get('mps_cleanup'), optional(phase / 'mps/session.json')) if record]
        processes = [read_json(path) for path in phase.glob('*/process.json')]
        if (cleanups and all(record.get('stopped') is True for record in cleanups)
                and all(type(record.get('exit_code')) is int for record in processes)):
            return

        namespace = optional(phase / 'mps/namespace.json')
        for record in (namespace, *cleanups, optional(phase / 'mps_access.json')):
            record_session(record)
        if namespace and (not namespace.get('batch_id')
                or namespace['batch_id'] != optional(phase / 'manifest.json').get('batch_id')):
            raise ValueError('MPS namespace does not match the phase manifest')
        for folder in ('capability', 'termination_capability'):
            record_session(optional(phase / folder / 'summary.json').get('access', {}))
        worker_pids.update(optional(phase / 'release.json').get('pids', []))
        for path in (*phase.glob('*/ready.json'), *phase.glob('*/worker_summary.json')):
            worker_pids.add(read_json(path)['pid'])
        worker_pids.update(record['pid'] for record in processes)

        if not pipes and not (phase / 'mps').exists() and not worker_pids and not cleanups:
            # The launcher writes its manifest before creating a private MPS directory.
            return
        if len(pipes) != 1:
            raise ValueError('Private MPS pipe identity is missing or conflicting')
        pipe = Path(next(iter(pipes)))
        owner = phase.stat().st_uid
        if not pipe.is_absolute() or pipe.is_symlink() or (pipe.exists() and (
                not pipe.is_dir() or (os.name == 'posix' and (
                    pipe.stat().st_uid != owner or pipe.stat().st_mode & 0o077)))):
            raise ValueError('MPS pipe is not a private directory owned by the phase user')
        pid_file = pipe / 'nvidia-cuda-mps-control.pid'
        if pid_file.exists():
            service_pids.add(int(pid_file.read_text().strip()))
        known_pids = worker_pids | service_pids
        if any(type(pid) is not int or pid <= 0 for pid in known_pids):
            raise ValueError('Invalid PID in phase evidence')
        bindings = {os.fsencode(f'CUDA_MPS_PIPE_DIRECTORY={pipe}'),
                    os.fsencode(f'CUDA_MPS_LOG_DIRECTORY={logs}')}
        for process in PROC_ROOT.iterdir():
            if not process.name.isdecimal():
                continue
            pid = int(process.name)
            try:
                state, _, _, sid, *_ = (process / 'stat').read_bytes().rsplit(b')', 1)[1].split()
                if state in (b'Z', b'X'):
                    continue
                # Launchers give each worker/probe a dedicated session. The private
                # environment also identifies children started before a PID was saved.
                sid = int(sid)
                if pid in known_pids or sid in worker_pids:
                    live[pid] = f'recorded PID/session {sid}'
                elif process.stat().st_uid == owner:
                    env = (process / 'environ').read_bytes().split(b'\0')
                    if bindings.intersection(env):
                        live[pid] = f'private MPS environment, session {sid}'
            except FileNotFoundError:
                if process.exists():
                    raise
        if live:
            raise ValueError(f'Live or reused PID/session: {live}')
    except (OSError, ValueError, TypeError, IndexError, KeyError, AttributeError) as exc:
        pipe_paths = ', '.join(sorted(map(str, pipes))) or 'unknown'
        raise RuntimeError(f'Cannot retry {batch} at {phase}: {exc}; '
            f'PIDs={sorted(worker_pids | service_pids | live.keys(), key=str)}, '
            f'MPS pipe={pipe_paths}, MPS logs={logs}, worker logs={phase}/*.log') from exc


def run_batch(args, batch, pool_info, pool, parent):
    directory = args.root / 'batches' / batch
    directory.mkdir(parents=True, exist_ok=True)
    attempts = sorted(directory.glob('attempt-*'))
    completed = completed_attempts(directory)
    if len(completed) > 1:
        raise ValueError(f'Multiple completed training attempts for {batch}.')
    attempt = completed[0] if completed else directory / f'attempt-{len(attempts)+1:06d}'
    training = batch_args(args, batch, attempt / 'train', pool_info, parent, gate_report=attempt / 'gate')
    evidence = batch_runner.inputs(training)
    if not completed:
        for previous in reversed(attempts):
            for phase in ('train', 'gate'):
                require_stopped_phase(previous / phase, batch)
        attempt.mkdir(exist_ok=False)
        print(f'{batch}: CUDA gate {attempt / "gate"}', flush=True)
        batch_runner.launch(batch_args(args, batch, attempt / 'gate', pool_info, parent, gate_only=True))
        print(f'{batch}: formal training {training.root}', flush=True)
        batch_runner.launch(training)
    else:
        print(f'{batch}: verifying completed batch {training.root}', flush=True)
    candidates = collect_batch(training, pool, evidence)
    winner = select_winner(candidates)
    selection = promoted_selection(directory, candidates, winner)
    if winner is not None:
        seal_copy(Path(winner['checkpoint']), directory / 'winner.pt', winner['checkpoint_sha256'])
    seal_json(directory / 'selection.json', selection)
    return candidates, selection['winner']


def evaluate_winners(root, winners, evidence, pools):
    from brain_uav.scripts.diagnose_v2_policy import load_diagnostic_inputs
    require_disjoint_pools(pools['validation'], pools['independent'])
    rows = []
    for model in MODELS:
        winner = winners[model]
        path = Path(winner['checkpoint'])
        if batch_runner.file_hash(path) != winner['checkpoint_sha256']:
            raise ValueError(f'{model} hard winner changed before independent evaluation.')
        hard_pool = evidence['validation']['hard']
        if batch_runner.file_hash(hard_pool['path']) != hard_pool['sha256']:
            raise ValueError('Hard promotion pool changed after locking.')
        frozen = None
        for stage in V2_TD3_STAGES:
            info = evidence['independent'][stage]
            if batch_runner.file_hash(info['path']) != info['sha256']:
                raise ValueError(f'Locked independent pool changed: {stage}')
            output = root / 'independent' / f'{model}-{stage}.json'
            digest_path = output.with_suffix('.sha256.json')
            identity = dict(model=model, stage=stage, winner_seed=winner['seed'],
                            checkpoint_sha256=winner['checkpoint_sha256'], pool=info)
            pool = pools['independent'][stage]
            if digest_path.exists():
                try:
                    digest = read_json(digest_path)
                    raw = output.read_bytes()
                    if digest != dict(identity=identity, sha256=sha256(raw).hexdigest()):
                        raise ValueError('SHA256 or identity mismatch')
                    record = json.loads(raw)
                    if record['identity'] != identity:
                        raise ValueError('Evaluation identity mismatch')
                    counts = independent_counts(record['evaluation'], pool)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f'Independent evaluation cache invalid: {output} (seal {digest_path}): {exc}') from exc
            else:
                if frozen is None:
                    frozen = load_diagnostic_inputs(model=model, checkpoint=path, validation_pool=hard_pool['path'],
                                                    device='cuda')
                    if (frozen['checkpoint_hash'] != winner['checkpoint_sha256']
                            or frozen['pool_hash'] != hard_pool['sha256']
                            or frozen['checkpoint_payload']['passed_validation'] is not True):
                        raise ValueError('Independent evaluation did not load the selected hard policy.')
                value = evaluate_v2_fixed_validation(frozen['actor'], pool, frozen['rewards'],
                    device=frozen['device'], cpu_deduplication=True).to_dict()
                counts = independent_counts(value, pool)
                if (batch_runner.file_hash(path) != winner['checkpoint_sha256']
                        or batch_runner.file_hash(info['path']) != info['sha256']):
                    raise ValueError('Frozen checkpoint or independent pool changed during evaluation.')
                raw = json.dumps(dict(identity=identity, evaluation=value), indent=2, allow_nan=False).encode('utf-8')
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix('.json.tmp')
                temporary.write_bytes(raw)
                temporary.replace(output)
                atomic_json(digest_path, dict(identity=identity, sha256=sha256(raw).hexdigest()))
            rows.append(dict(model=model, stage=stage, winner_seed=winner['seed'],
                **counts, scenarios=str(output)))
    return rows


def independent_counts(evaluation, pool):
    counts = counts_for_pool(evaluation, pool)
    aggregate = {key: counts[key] for key in ('goal', 'collision', 'ground', 'timeout')}
    aggregate['boundary'] = counts['upper_boundary'] + counts['other_boundary']
    if (evaluation['scenario_count'] != 100 or sum(counts.values()) != 100
            or evaluation['failure_count'] != 100 - counts['goal']
            or evaluation['outcome_counts'] != aggregate):
        raise ValueError('Independent evaluation counts disagree with the 100 scene outcomes.')
    return counts


def write_table(path, columns, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    if path.exists():
        if batch_runner.file_hash(path) != batch_runner.file_hash(temporary):
            raise ValueError(f'Existing table differs from validated evidence: {path}')
        temporary.unlink()
        return
    temporary.replace(path)


def resume_root(root, archive):
    if archive is not None:
        root, archive = root.resolve(), archive.resolve()
        if root.is_relative_to(archive) or archive.is_relative_to(root):
            raise ValueError('The continuation and historical result directories must be separate.')
    return archive


def run_course(args):
    resume_root(args.root, getattr(args, 'resume_from', None))
    # The server-only lock also keeps interrupted attempts and published results immutable.
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / '.course.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(args)


def _run_locked(args):
    archive = resume_root(args.root, getattr(args, 'resume_from', None))
    identity = batch_runner.identity(EXPERIMENT)
    reviewed = read_json(args.source_manifest)['sha256']
    if archive is not None and reviewed != read_json(RESUME_SOURCE_MANIFEST)['sha256']:
        raise ValueError('Continuation requires the reviewed post-fix course source manifest.')
    if reviewed != identity['sha256']:
        raise ValueError('Actual source files differ from the reviewed course manifest.')
    bc = {m: batch_runner.file_hash(args.input_root / provenance.BC[m][0]) for m in MODELS}
    if bc != {m: provenance.BC[m][1] for m in MODELS}:
        raise ValueError('Original BC best SHA256 mismatch.')
    seal_json(args.root / 'course.json', dict(experiment=EXPERIMENT,
        code=identity['sha256'], bc=bc, plan=course_plan(args.root, archive)))
    if archive is None:
        evidence, pools = prepare_pools(args.root, args.validation_pool_dir)
        imported = {}
    else:
        evidence, pools, imported = import_easy(args, archive, bc)
    frozen_path = args.root / 'independent_inputs.json'
    if frozen_path.exists() or (args.root / 'independent').exists():
        return finish_independent(args.root, evidence, pools)
    winners, stopped, rows = {}, {}, []
    for stage in V2_TD3_STAGES:
        for model in MODELS:
            if model in stopped:
                rows.extend(dict(model=model, stage=stage, seed=seed, status='blocked_by_no_formal_pass')
                            for seed in SEEDS)
                continue
            if stage == 'easy' and model in imported:
                records, winner = imported[model]['candidates'], imported[model]['winner']
            else:
                records, winner = run_batch(args, f'{stage}-{model}', evidence['validation'][stage],
                                            pools['validation'][stage], winners.get(model))
            if winner is None:
                stopped[model] = stage
                winners.pop(model, None)
            else:
                winners[model] = winner
            for record in records:
                rows.append(dict(record, status='passed' if record['formal_passed'] else 'formal_failed',
                    winner=winner is not None and record['seed'] == winner['seed'],
                    episode_goals=record['last_50k']['successes'], episode_count=record['last_50k']['count'],
                    episode_rate=record['last_50k']['success_rate'],
                    parent_sha256=record['initialization']['sha256']))
    write_table(args.root / 'training.csv',
        ('model', 'stage', 'seed', 'status', 'formal_passed', 'winner', 'steps', *OUTCOMES,
         'episode_goals', 'episode_count', 'episode_rate', 'checkpoint', 'checkpoint_sha256', 'parent_sha256'), rows)
    if not stopped:
        seal_json(frozen_path, dict(hard_winners=winners, training_rows=len(rows),
            training_sha256=batch_runner.file_hash(args.root / 'training.csv'),
            hard_selection_sha256={m: batch_runner.file_hash(args.root / 'batches' / f'hard-{m}' / 'selection.json')
                                   for m in MODELS}))
        return finish_independent(args.root, evidence, pools)
    seal_json(args.root / 'course_summary.json', dict(stopped_at=stopped, hard_winners=winners,
        training_rows=len(rows), independent_evaluations=0))
    return 0


def finish_independent(root, evidence, pools):
    frozen = read_json(root / 'independent_inputs.json')
    if batch_runner.file_hash(root / 'training.csv') != frozen['training_sha256']:
        raise ValueError('Frozen training table changed before independent evaluation.')
    winners = frozen['hard_winners']
    for model in MODELS:
        selection = root / 'batches' / f'hard-{model}' / 'selection.json'
        raw = selection.read_bytes()
        if (sha256(raw).hexdigest() != frozen['hard_selection_sha256'][model]
                or json.loads(raw)['winner'] != winners[model]):
            raise ValueError(f'Frozen winner selection changed: {selection}')
    final = evaluate_winners(root, winners, evidence, pools)
    write_table(root / 'independent.csv', ('model', 'stage', 'winner_seed', *OUTCOMES, 'scenarios'), final)
    seal_json(root / 'course_summary.json', dict(stopped_at={}, hard_winners=winners,
        training_rows=frozen['training_rows'], independent_evaluations=600))
    return 0


def course_plan(root, archive=None):
    batches = {}
    for batch in batch_runner.COURSE_BATCHES:
        entries = []
        for job in batch_runner.jobs(EXPERIMENT, batch):
            spec, config = resolve_course_job(job)
            stage = config.stage
            entries.append(dict(spec, initialization=(f'{job["model"]} original BC best' if stage == 'easy'
                else f'{job["model"]} {PREDECESSOR[stage]} winner'),
                effective_bc_schedule=v2_bc_schedule_metadata(stage, schedule=spec['bc_schedule'])))
        batches[batch] = entries
    plan = dict(root=str(root), experiment=EXPERIMENT, batches=batches,
                validation_seed=VALIDATION_SEED, independent_seed=INDEPENDENT_SEED,
                scene_count_per_level=100, concurrent_workers=4)
    if archive is not None:
        plan.update(resume_from=str(archive), reuse_batches=[f'easy-{model}' for model in MODELS],
                    first_training_batch='medium-ann')
    return plan


def course_status(root):
    result = {}
    for batch in batch_runner.COURSE_BATCHES:
        path = root / 'batches' / batch
        selection = path / 'selection.json'
        if selection.exists():
            value = read_json(selection)
            result[batch] = dict(status=value['status'], winner=value['winner'],
                candidates={row['seed']: 'passed' if row['formal_passed'] else 'formal_failed'
                            for row in value['candidates']})
            continue
        attempts = sorted(path.glob('attempt-*'))
        workers = {}
        if attempts:
            attempt = attempts[-1]
            for phase in ('gate', 'train'):
                summary = attempt / phase / 'batch_summary.json'
                if summary.exists():
                    workers[phase] = read_json(summary)
        result[batch] = dict(status='missing' if not attempts else
            'process_failed' if any(not value['execution_completed'] for value in workers.values())
            else 'uncollected', attempts=len(attempts), batches=workers)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--status', action='store_true')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, default=Path('/home/zj/projects/my_project'))
    parser.add_argument('--validation-pool-dir', type=Path)
    parser.add_argument('--resume-from', type=Path, help='Read completed e6bc9be easy results and pools here.')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--timeout', type=float, default=1800.)
    parser.add_argument('--gate-timeout', type=float, default=300.)
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    try:
        args.resume_from = resume_root(args.root, args.resume_from)
    except ValueError as error:
        parser.error(str(error))
    if args.resume_from is not None and args.validation_pool_dir is not None:
        parser.error('--resume-from uses the six pools sealed by the historical course.')
    args.source_manifest = args.source_manifest or RESUME_SOURCE_MANIFEST
    if args.status:
        print(json.dumps(course_status(args.root), indent=2))
    elif not args.run:
        print(json.dumps(course_plan(args.root, args.resume_from), indent=2))
    else:
        if Path(sys.executable).resolve() != Path(provenance.PYTHON).resolve():
            parser.error('Use the declared server Python.')
        if any(not math.isfinite(value) or value <= 0 for value in (args.timeout, args.gate_timeout)):
            parser.error('Timeouts must be finite and positive.')
        args.input_root = args.input_root.resolve()
        args.source_manifest = args.source_manifest.resolve()
        args.validation_pool_dir = (args.validation_pool_dir or args.input_root / provenance.POOL.parent).resolve()
        print(f'Total result directory: {args.root}', flush=True)
        return run_course(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
