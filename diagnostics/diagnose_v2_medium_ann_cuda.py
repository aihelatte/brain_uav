"""Bounded ANN medium CUDA reproduction using the sealed e6bc9be course."""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4
from unittest.mock import patch

from brain_uav.observations.v2_batch import V2ObservationBatch
from brain_uav.scripts import run_v2_formal_easy_baseline as formal
from brain_uav.scripts import train_v2_td3 as stage
from brain_uav.scripts.v2_formal_easy_guidance import read_json
from brain_uav.scripts.v2_mps_diagnostic import Session, capability
from brain_uav.trainers.v2_formal_training import V2FormalStageTrainer
from brain_uav.trainers.v2_gpu_compute import GPUCompute, clone_batch, tensors


BASE = 'e6bc9be6b1cd95ac5c23b370f373b7eb3050c6df'
COURSE_ROOT = Path('/home/zj/diagnostics/outputs/v2_course_e6bc9be')
BATCH = 'medium-ann'
ALL_SEEDS = (7, 11, 23, 41)
STOP_STEP = 1000
WINDOW = (675, 700)
ATTEMPT = Path('batches/medium-ann/attempt-000001')
COMPARE_STEP = 686
COMPARE_KEY = (2, 2, 'frozen_targets', 0, 0)
FIRST_SAMPLE_ARM = 'first-sample-graph'
RECAPTURE_ARMS = ('fresh-graph', FIRST_SAMPLE_ARM)
# The graph and staged server references predate the fresh-graph code.
HISTORICAL_DIAGNOSTIC_SHA256 = 'f36bedd5c0d84f9ccc41215aef2988d8fbd6794f92c72a3faeec018635524f73'


class DiagnosticComplete(Exception):
    """Planned exit after the last step, before formal checkpoint publication."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--course-root', type=Path, default=COURSE_ROOT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41])
    parser.add_argument('--single-result', type=Path, help='Completed async seed-41 output required for the four-seed run.')
    parser.add_argument('--sync-window', action='store_true', help='Synchronize between marked CUDA sections only in steps 675–700.')
    parser.add_argument('--compare-686', choices=('graph', 'staged', *RECAPTURE_ARMS),
                        help='Stop at step 686 after cached replay, staged computation, or recapture and replay.')
    parser.add_argument('--graph-reference', type=Path, help='Cached graph output for staged or recapture modes.')
    parser.add_argument('--staged-reference', type=Path, help='Successful staged output for recapture modes.')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', type=int, choices=ALL_SEEDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.course_root, args.out = args.course_root.resolve(), args.out.resolve()
    args.seeds = tuple(args.seeds)
    if args.seeds not in ((41,), ALL_SEEDS):
        parser.error('Choose seed 41 alone or ordered seeds 7 11 23 41.')
    if (args.seeds == ALL_SEEDS) != (args.single_result is not None):
        parser.error('The four-seed run requires --single-result from the async seed-41 run.')
    if args.single_result is not None:
        args.single_result = args.single_result.resolve()
    if args.compare_686 and (args.seeds != (41,) or args.sync_window):
        parser.error('--compare-686 uses seed 41 and adds no synchronization before step 686.')
    for name, required in (('graph_reference', args.compare_686 in ('staged', *RECAPTURE_ARMS)),
                           ('staged_reference', args.compare_686 in RECAPTURE_ARMS)):
        path = getattr(args, name)
        option = '--' + name.replace('_', '-')
        if required != (path is not None):
            parser.error(f'{option} is required for its comparison mode and forbidden in other modes.')
        if path is not None:
            path = path.resolve()
            setattr(args, name, path)
            if args.out.is_relative_to(path) or path.is_relative_to(args.out):
                parser.error(f'--out must be separate from {option}.')
    if args.out.is_relative_to(args.course_root) or args.course_root.is_relative_to(args.out):
        parser.error('--out must be separate from the original course.')
    if args.worker is not None and (not args.run or args.worker not in args.seeds):
        parser.error('Worker requires a selected seed and --run.')
    return args


def describe(args):
    comparison = (dict(compare_686=args.compare_686,
                       diagnostic_sha256=formal.file_hash(Path(__file__)),
                       graph_reference=str(args.graph_reference) if args.graph_reference else None)
                  if args.compare_686 else {})
    if args.staged_reference is not None:
        comparison['staged_reference'] = str(args.staged_reference)
    return dict(diagnostic_only=True, base=BASE, batch=BATCH, seeds=list(args.seeds), formal_max_steps=750000,
                stop_step=COMPARE_STEP if comparison else STOP_STEP,
                trace_window=[COMPARE_STEP, COMPARE_STEP] if comparison else list(WINDOW),
                sync_window=args.sync_window,
                course_root=str(args.course_root), output=str(args.out), **comparison)


def source_args(root, source, *, job=None):
    paths, hashes = source['inputs']['paths'], source['inputs']['hashes']
    argv = [('--root', root), ('--ann-bc', paths['ann_bc']), ('--snn-bc', paths['snn_bc']),
            ('--validation-pool', paths['pool']), ('--parent', paths['parent']),
            ('--source-manifest', paths['source_manifest']), ('--timeout', source['plan']['initialization_timeout']),
            ('--pool-sha256', hashes['pool']), ('--parent-sha256', hashes['parent']),
            ('--experiment', formal.COURSE_EXPERIMENT), ('--batch', BATCH)]
    if job is not None:
        argv.append(('--job', job))
    return formal.parser().parse_args(['worker' if job else 'launch', '--launch',
                                        *(str(item) for pair in argv for item in pair)])


def load_original(args):
    from brain_uav.trainers.v2_validation import load_v2_validation_pool

    train = args.course_root / ATTEMPT / 'train'
    manifest_path = train / 'manifest.json'
    source = read_json(manifest_path)
    identity = source['inputs']
    wanted = formal.jobs(formal.COURSE_EXPERIMENT, BATCH)
    resolved = [formal.resolved(job, formal.COURSE_EXPERIMENT)[0] for job in wanted]
    selection = read_json(args.course_root / 'batches/easy-ann/selection.json')['winner']
    pool = read_json(args.course_root / 'pools.json')['validation']['medium']
    expected_paths = dict(parent=args.course_root / 'batches/easy-ann/winner.pt',
                          pool=args.course_root / 'pools/validation/v2_validation_medium.json')
    if (identity['source']['HEAD'] != BASE or source['plan']['root'] != str(train)
            or source['plan']['batch'] != BATCH or source['plan']['gate_only'] is not False
            or source['plan']['experiment'] != formal.COURSE_EXPERIMENT
            or source['plan']['jobs'] != resolved
            or any(Path(identity['paths'][key]) != path for key, path in expected_paths.items())
            or selection['checkpoint'] != identity['paths']['parent']
            or selection['checkpoint_sha256'] != identity['hashes']['parent']
            or pool['path'] != identity['paths']['pool'] or pool['sha256'] != identity['hashes']['pool']
            or pool['content_digest'] != identity['pool_digest']):
        raise ValueError('Original committed medium ANN batch, parent, or pool differs.')
    formal.check_inputs(identity)
    parsed = source_args(train, source)
    parsed.gate_report = train.parent / 'gate'
    if formal.validate_gate(parsed, identity) != source['gate_evidence']:
        raise ValueError('Original medium ANN gate evidence differs.')
    # Recheck the existing input contract without opening a trainer or writing to the course.
    actual = formal.inputs(parsed)
    if any(actual[key] != identity[key] for key in ('paths', 'hashes', 'pool_digest', 'global_steps_start')):
        raise ValueError('Original formal initialization evidence differs.')
    load_v2_validation_pool(parsed.validation_pool, expected_level='medium', expected_count=100,
                            expected_master_seed=pool['master_seed'], expected_stage_seed=pool['stage_seed'])
    binding = dict(manifest=str(manifest_path), sha256=formal.file_hash(manifest_path),
                   batch_id=source['batch_id'], parent_sha256=identity['hashes']['parent'],
                   pool_sha256=identity['hashes']['pool'])
    return source, wanted, binding


def require_single_result(path, binding):
    summary = read_json(path / 'summary.json')
    worker = read_json(path / 'ann_seed41/worker_summary.json')
    process = summary.get('processes', {}).get('ann_seed41', {})
    if (summary.get('original') != binding or summary.get('diagnostic_only') is not True
            or summary.get('execution_completed') is not True or summary.get('seeds') != [41]
            or summary.get('mps_cleanup', {}).get('stopped') is not True
            or process.get('exit_code') != 0 or process.get('pid') != worker.get('pid')
            or summary.get('sync_window') is not False or worker.get('diagnostic_completed') is not True
            or worker.get('stage_steps') != STOP_STEP):
        raise ValueError('Four-seed run requires the completed asynchronous seed-41 diagnostic on these inputs.')


def shapes(observation):
    return {name: list(getattr(observation, name).shape) for name in
            ('ego_features', 'goal_features', 'zone_features', 'presence_mask')}


def observation_evidence(batch):
    return dict(obs_shapes=shapes(batch.obs), next_obs_shapes=shapes(batch.next_obs),
                obs_mask=batch.obs.presence_mask.detach().cpu().tolist(),
                next_obs_mask=batch.next_obs.presence_mask.detach().cpu().tolist())


def graph_addresses(executor, key, entry):
    """Read CPU metadata from the real cache entry and its bound tensors."""
    _, fixed, fraction, output, presence, _ = entry
    names = tuple(shapes(fixed.obs))
    inputs = [f'{side}.{name}' for side in ('obs', 'next_obs') for name in names]
    inputs += ['action', 'reward', 'done', 'success', 'near_goal', 'line_to_goal_safe']
    bound = dict(zip(inputs, tensors(fixed), strict=True), fraction=fraction, output=output,
                 noise=executor.noise, coefficients=executor.coefficients, injections=executor.injections)
    for group, bank in enumerate(key[3:]):
        bound.update((f'grad[{group},{bank},{i}]', tensor)
                     for i, tensor in enumerate(executor.banks[group][bank]) if presence[group][i])
    bound.update((f'persistent[{i}]', tensor) for i, tensor in enumerate(executor.persistent()))
    return {name: dict(address=hex(tensor.data_ptr()), storage=hex(tensor.untyped_storage().data_ptr()),
                       shape=list(tensor.shape), stride=list(tensor.stride()),
                       offset=tensor.storage_offset(), dtype=str(tensor.dtype))
            for name, tensor in bound.items()}


def staged_compute(executor, key, entry, trace):
    """Run GPUCompute.compute itself, with fences at its existing call boundaries."""
    import torch
    from brain_uav.trainers import v2_gpu_update

    _, fixed, fraction, _, presence, _ = entry
    engine = executor.e
    with ExitStack() as hooks:
        def fence(owner, attribute, label):
            original = getattr(owner, attribute)

            def call(*args, **kwargs):
                with trace.scope(label, sync=True):
                    return original(*args, **kwargs)

            hooks.enter_context(patch.object(owner, attribute, call))

        fence(executor, 'arguments', 'relation_inputs')
        fence(executor, 'actor_forward', 'target_actor_forward')
        fence(engine, '_compute_target_critics_td_tensors', 'target_values')
        fence(engine.critic1, '_compute_full_forward_tensors', 'critic1_forward')
        fence(engine.critic2, '_compute_full_forward_tensors', 'critic2_forward')
        fence(engine, '_twin_td_loss', 'critic_loss')
        fence(torch.autograd, 'backward', 'critic_backward')
        fence(executor, 'clip_gradients', 'critic_gradient_clip')
        fence(executor.adams[1], 'step', 'critic_optimizer')
        fence(v2_gpu_update, 'gated_targets', 'critic_target_update')
        hooks.enter_context(patch.object(executor, 'failure_fraction', fraction))
        # The graph writes its selected gradient banks directly. Bind those same
        # buffers for eager autograd, and let the original step publish them.
        for group, bank in enumerate(key[3:]):
            for parameter, gradient, active in zip(executor.params[group],
                    executor.banks[group][bank], presence[group], strict=True):
                parameter.grad = gradient if active else None
        with trace.scope('staged_compute_result', sync=True):
            return executor.compute(fixed, key[2])


def read_comparison_reference(path, original, arm, diagnostic_sha256):
    summary = read_json(path / 'summary.json')
    folder = path / 'ann_seed41'
    raw = (folder / 'step686_before.json').read_bytes()
    before = json.loads(raw)
    result = read_json(folder / 'step686_result.json')
    if (summary.get('compare_686') != arm or summary.get('seeds') != [41]
            or summary.get('diagnostic_sha256') != diagnostic_sha256
            or summary.get('original') != original or before['original'] != original
            or summary.get('mps_cleanup', {}).get('stopped') is not True
            or summary['processes']['ann_seed41']['pid'] != before['pid']
            or result['pid'] != before['pid'] or before['arm'] != arm or result['arm'] != arm
            or result['before_sha256'] != sha256(raw).hexdigest()
            or result['last_step'] != COMPARE_STEP or not result['submitted'] or not before['addresses_match']
            or before['identity']['key'] != list(COMPARE_KEY)):
        raise ValueError(f'{arm} reference is incomplete or belongs to a different run: {path}')
    if arm == 'staged' and (summary.get('execution_completed') is not True
            or summary['processes']['ann_seed41']['exit_code'] != 0
            or result['completed'] is not True or result['first_failure'] is not None
            or result['reference_matches'] is not True or before['reference_matches'] is not True
            or not {'metrics', 'post_state', 'result_sha256'} <= result.keys()):
        raise ValueError(f'Staged reference did not complete a matching update: {path}')
    return before['identity'], result


def comparison_references(args, original):
    if args.graph_reference is None:
        return {}
    if args.compare_686 == 'staged':
        return {'graph': read_comparison_reference(args.graph_reference, original, 'graph',
                                                    formal.file_hash(Path(__file__)))}
    graph = read_comparison_reference(args.graph_reference, original, 'graph', HISTORICAL_DIAGNOSTIC_SHA256)
    staged = read_comparison_reference(args.staged_reference, original, 'staged', HISTORICAL_DIAGNOSTIC_SHA256)
    failure = graph[1]['first_failure'] or {}
    if (graph[1]['completed'] is not False or failure.get('section') != 'graph_replay_submit'
            or failure.get('phase') != 'synchronize' or graph[0] != staged[0]):
        raise ValueError('Recapture requires a cached replay synchronization failure and successful staged '
                         f'update with identical inputs and state: {args.graph_reference}, {args.staged_reference}')
    return {'graph': graph, 'staged': staged}


class StepComparison:
    def __init__(self, args, output, original):
        self.arm, self.output, self.original = args.compare_686, output, original
        self.references = comparison_references(args, original)
        self.captured = None
        self.capture_step = None
        self.selected = None
        self.before_sha256 = None
        self.matched = None
        self.result = None
        self.fresh_entry = None
        self.failure = None
        self.submitted = False
        self.first_sample = None
        self.first_sample_evidence = None
        self.cache_before = None
        self.fresh_addresses = None

    def capture(self, executor, key, step, cpu_sample=None):
        if key == COMPARE_KEY and step < COMPARE_STEP:
            self.captured = graph_addresses(executor, key, executor.graphs[key])
            self.capture_step = step
            if self.arm == FIRST_SAMPLE_ARM and self.first_sample is None:
                from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash

                if cpu_sample is None or any(t.device.type != 'cpu' for t in tensors(cpu_sample)):
                    raise ValueError('First capture requires the original CPU replay sample.')
                self.first_sample = clone_batch(cpu_sample)
                self.first_sample_evidence = dict(
                    key=list(key), capture_step=step, sample_sha256=state_hash(tensors(self.first_sample)),
                    sample_failure_fraction=self.first_sample.sample_failure_fraction,
                    **observation_evidence(self.first_sample))
                formal.atomic_json(self.output / 'first_capture_sample.json', self.first_sample_evidence)

    @staticmethod
    def cache_identity(executor):
        return dict(capture_count=executor.captures, prior_replays=executor.replays,
                    graph_keys=[list(k) for k in executor.graphs])

    def select(self, executor, key, batch, trace):
        if key != COMPARE_KEY or key not in executor.graphs or self.captured is None:
            raise ValueError(f'Step 686 requires a pre-existing capture of {COMPARE_KEY}; found {key}.')
        if executor.e.last_total_steps != COMPARE_STEP - 1 or executor.e.window_update_statistics:
            raise ValueError('The comparison requires the original updates through step 685.')
        entry = executor.graphs[key]
        self.selected = (executor, key, batch, entry)
        if self.arm == FIRST_SAMPLE_ARM:
            if (self.first_sample_evidence is None or self.first_sample_evidence['capture_step'] != self.capture_step
                    or graph_addresses(executor, key, entry) != self.captured):
                raise ValueError('Cannot verify the original cached entry and its first capture sample.')
            self.cache_before = self.cache_identity(executor)
            # Keep the original entry alive in self.selected. A cache miss here
            # lets the formal update capture, copy and replay the replacement.
            del executor.graphs[key]
        elif self.arm != 'graph':
            comparison = self

            class ComparisonReplay:
                def replay(self):
                    if comparison.arm == 'fresh-graph':
                        comparison.fresh_entry[0].replay()
                        comparison.submitted = True
                        comparison.result = comparison.fresh_entry[3]
                    else:
                        comparison.result = staged_compute(executor, key, entry, trace)

                def cpu(self):
                    return comparison.result.cpu()

            replay = ComparisonReplay()
            executor.graphs[key] = (replay, entry[1], entry[2], replay, *entry[4:])

    def restore_cache(self):
        if self.selected is not None:
            executor, key, _, entry = self.selected
            executor.graphs[key] = entry

    @staticmethod
    def state(executor, trainer):
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
        from brain_uav.trainers.v2_online_research import training_identity
        return dict(training=training_identity(executor.e, trainer.exploration_rng, trainer.selector),
                    device_adam=state_hash([opt.state_tensors() for opt in executor.adams]),
                    gradient_banks=state_hash(executor.banks), published=list(executor.published))

    @staticmethod
    def fixed_inputs(executor, entry):
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash

        return dict(fixed_sha256=state_hash(tensors(entry[1])), fraction_sha256=state_hash(entry[2]),
                    noise_sha256=state_hash(executor.noise), coefficients_sha256=state_hash(executor.coefficients),
                    injection_sha256=state_hash(executor.injections))

    def before_replay(self, trace):
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash

        executor, key, batch, entry = self.selected
        first_sample_arm = self.arm == FIRST_SAMPLE_ARM
        captured = self.fresh_addresses if first_sample_arm else self.captured
        cache = self.cache_before if first_sample_arm else self.cache_identity(executor)
        if first_sample_arm:
            entry = self.fresh_entry
        else:
            # The first-sample mode already fenced the formal fixed input copy.
            with trace.scope('fixed_input_completion', sync=True):
                pass
        _, fixed, _, _, presence, _ = entry
        with trace.scope('comparison_before_replay'):
            addresses = graph_addresses(executor, key, entry)
            identity = dict(key=list(key), capture_step=self.capture_step, **cache,
                participation=[list(group) for group in presence], scene=trace.scene,
                **observation_evidence(fixed),
                sample_sha256=state_hash(tensors(batch)), **self.fixed_inputs(executor, entry),
                state=self.state(executor, trace.trainer))
            mismatches = [name for name in identity
                          if any(identity[name] != reference.get(name) for reference, _ in self.references.values())]
            self.matched = not mismatches if self.references else None
            before = self.output / 'step686_before.json'
            formal.atomic_json(before, dict(arm=self.arm, pid=os.getpid(), original=self.original,
                identity=identity, capture_addresses=captured, replay_addresses=addresses,
                addresses_match=(addresses == captured),
                reference_matches=self.matched, reference_mismatches=mismatches))
            self.before_sha256 = formal.file_hash(before)
            if addresses != captured:
                raise ValueError('Captured buffer addresses differ from the cached replay addresses.')
            if identity['sample_sha256'] != identity['fixed_sha256']:
                raise ValueError('Copied graph inputs differ from the sampled batch.')
            if mismatches:
                raise ValueError(f'Step 686 reference mismatch: {", ".join(mismatches)}.')
        if self.arm == 'fresh-graph':
            self.recapture(trace, identity)
        elif not first_sample_arm:
            self.submitted = True

    def recapture_first_sample(self, trace, capture):
        import torch
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash

        executor, key, _, old_entry = self.selected
        with trace.scope('first_sample_recapture_precheck'):
            state = self.state(executor, trace.trainer)
            reference = self.references['staged'][0]
            if state != reference['state']:
                raise ValueError('Pre-capture training state differs from staged.')
        with trace.scope('first_sample_recapture', sync=True):
            batch = self.first_sample.to(executor.e.device)
            with patch.object(executor, 'failure_fraction', executor.failure_fraction):
                capture(executor, key, batch)
            self.fresh_entry = executor.graphs[key]
        with trace.scope('first_sample_recapture_verify'):
            entry = self.fresh_entry
            self.fresh_addresses = graph_addresses(executor, key, entry)
            after = self.state(executor, trace.trainer)
            captured_sha256 = state_hash(tensors(entry[1]))
            fraction_sha256 = state_hash(entry[2])
            fraction_matches = fraction_sha256 == state_hash(torch.tensor(
                self.first_sample.sample_failure_fraction, dtype=entry[2].dtype, device='cpu'))
            checks = dict(state_matches=(after == state),
                sample_matches=(captured_sha256 == self.first_sample_evidence['sample_sha256'] and fraction_matches),
                new_capture=(entry[0] is not old_entry[0]
                             and executor.captures == self.cache_before['capture_count'] + 1),
                replay_count_matches=(executor.replays == self.cache_before['prior_replays']),
                participation_matches=(entry[4] == old_entry[4]))
            formal.atomic_json(self.output / 'step686_recapture.json', dict(key=list(key),
                capture_step=self.capture_step, capture_count_before=self.cache_before['capture_count'],
                capture_count_after=executor.captures, addresses=self.fresh_addresses,
                capture_sample_sha256=captured_sha256, capture_fraction_sha256=fraction_sha256,
                capture_sample_differs_from_replay=(captured_sha256 != reference['sample_sha256']
                                                    or fraction_sha256 != reference['fraction_sha256']),
                state_after_capture=after, **checks))
            if not all(checks.values()):
                raise ValueError('First-sample recapture failed verification.')

    def recapture(self, trace, identity):
        executor, key, batch, old_entry = self.selected
        with trace.scope('fresh_graph_recapture', sync=True):
            # Hold the old entry alive while capturing into the same cache slot.
            del executor.graphs[key]
            with patch.object(executor, 'failure_fraction', executor.failure_fraction):
                executor.capture_shape(key, batch)
            self.fresh_entry = executor.graphs[key]
        with trace.scope('fresh_graph_verify'):
            after = self.state(executor, trace.trainer)
            inputs = self.fixed_inputs(executor, self.fresh_entry)
            state_matches = after == identity['state']
            input_matches = {name: value == identity[name] for name, value in inputs.items()}
            fresh_graph = self.fresh_entry[0] is not old_entry[0]
            delta = executor.captures - identity['capture_count']
            replay_count_matches = executor.replays == identity['prior_replays']
            participation_matches = self.fresh_entry[4] == old_entry[4]
            formal.atomic_json(self.output / 'step686_recapture.json', dict(
                key=list(key), capture_count_before=identity['capture_count'], capture_count_after=executor.captures,
                addresses=graph_addresses(executor, key, self.fresh_entry), fresh_graph=fresh_graph,
                state_after_capture=after, state_matches=state_matches, input_matches=input_matches,
                replay_count_matches=replay_count_matches, participation_matches=participation_matches))
            if not (fresh_graph and delta == 1 and replay_count_matches and participation_matches
                    and all(input_matches.values()) and state_matches):
                raise ValueError('Fresh capture changed step 686 inputs or training state, or did not build one new graph.')

    def record(self, trace, *, metric=None):
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
        result = dict(arm=self.arm, pid=os.getpid(), before_sha256=self.before_sha256,
                      reference_matches=self.matched, submitted=self.submitted,
                      completed=metric is not None, first_failure=self.failure, last_step=trace.step)
        if metric is not None:
            executor, _, _, entry = self.selected
            if self.arm == FIRST_SAMPLE_ARM:
                output = self.fresh_entry[3]
            else:
                output = entry[3] if self.arm == 'graph' else self.result
            result.update(metrics=asdict(metric), post_state=self.state(executor, trace.trainer),
                          result_sha256=state_hash(output))
        if self.references and self.matched:
            other = self.references['graph'][1]
            result['graph_completed'] = other['completed']
            result['graph_first_failure'] = other['first_failure']
            reference_arm = 'staged' if self.arm in RECAPTURE_ARMS else 'graph'
            other = self.references[reference_arm][1]
            if metric is not None and other['completed']:
                result['comparison'] = {key + '_exact_match': result[key] == other[key]
                                        for key in ('metrics', 'post_state', 'result_sha256')}
                result[reference_arm + '_metrics'] = other['metrics']
                if self.arm in RECAPTURE_ARMS:
                    result['comparison_details'] = {
                        group: {name: name in other[group] and value == other[group][name]
                                for name, value in result[group].items()}
                        for group in ('metrics', 'post_state')}
        formal.atomic_json(self.output / 'step686_result.json', result)


class Trace:
    def __init__(self, stream, *, sync_window, comparison=None):
        self.stream = stream
        self.sync_window = sync_window
        self.step = 0
        self.device = None
        self.phase = None
        self.scene = {}
        self.comparison = comparison
        self.trainer = None

    @property
    def comparing(self):
        return self.comparison is not None and self.step == COMPARE_STEP

    @property
    def active(self):
        if self.comparison is not None:
            return self.comparing
        return WINDOW[0] <= self.step <= WINDOW[1]

    def emit(self, event, section, **details):
        if self.active:
            print(json.dumps(dict(pid=os.getpid(), step=self.step, event=event, section=section,
                                  monotonic_ns=time.monotonic_ns(), **self.scene, **details)),
                  file=self.stream, flush=True)

    def synchronize(self):
        import torch
        torch.cuda.synchronize(self.device)

    @contextmanager
    def scope(self, name, *, sync=False):
        phase = 'before'
        comparison_replay = self.comparing and name == 'graph_replay_submit'
        first_sample_arm = self.comparing and self.comparison.arm == FIRST_SAMPLE_ARM
        sync_copy = first_sample_arm and name == 'graph_fixed_input_copy'
        try:
            if comparison_replay:
                self.comparison.before_replay(self)
            self.emit('begin', name)
            phase = 'host_call'
            yield
            if first_sample_arm and comparison_replay:
                self.comparison.submitted = True
            self.emit('host_return', name)
            if self.active and (self.sync_window or sync or comparison_replay or sync_copy):
                phase = 'synchronize'
                self.emit('sync_begin', name)
                self.synchronize()
                self.emit('sync_done', name)
        except BaseException as exc:
            detail = dict(section=name, phase=phase, error_type=type(exc).__name__, error=str(exc))
            if self.comparing and self.comparison.failure is None:
                self.comparison.failure = detail
            self.emit('error', name, **{key: value for key, value in detail.items() if key != 'section'})
            raise

    def observe_batch(self, batch):
        self.emit('shape', 'batch_upload', obs=shapes(batch.obs), next_obs=shapes(batch.next_obs))


def run_bounded(run, trainer, trace):
    engine = trainer.engine
    if trainer.config.max_steps != 750000:
        raise ValueError('The diagnostic requires the original 750000-step schedule.')
    trace.device = engine.device
    trace.trainer = trainer
    stop_step = COMPARE_STEP if trace.comparison is not None else STOP_STEP
    new_episode, select_action = trainer._new_episode, engine.select_action
    action_tensor, update = engine._action_inference_tensor, engine.update_once
    action_to, graph_key, capture = V2ObservationBatch.to, GPUCompute.key, GPUCompute.capture_shape
    first_sample_arm = trace.comparison is not None and trace.comparison.arm == FIRST_SAMPLE_ARM
    cpu_sample = None

    def stop_if_done():
        if trainer.result.stage_steps == stop_step:
            raise DiagnosticComplete(f'Completed {stop_step} diagnostic steps.')
        if trainer.result.stage_steps > stop_step:
            raise RuntimeError('Diagnostic exceeded the stop step.')

    def bounded_reset(*args, **kwargs):
        stop_if_done()
        return new_episode(*args, **kwargs)

    def traced_select(*args, **kwargs):
        stop_if_done()
        trace.step = trainer.result.stage_steps + 1
        if not trace.active:
            return select_action(*args, **kwargs)
        trace.scene = dict(level=trainer.env.last_curriculum_level, obstacles=len(trainer.env.zones))
        if trace.sync_window and trace.step == WINDOW[0]:
            trace.emit('sync_begin', 'window_entry')
            trace.synchronize()
            trace.emit('sync_done', 'window_entry')
        trace.phase = 'action'
        try:
            with trace.scope('action_inference_and_readback'):
                result = select_action(*args, **kwargs)
                trace.emit('host_return', 'action_result_readback')
                return result
        finally:
            trace.phase = None

    def traced_action_to(observation, *args, **kwargs):
        if trace.phase != 'action':
            return action_to(observation, *args, **kwargs)
        trace.emit('shape', 'action_upload', obs=shapes(observation))
        with trace.scope('action_upload'):
            return action_to(observation, *args, **kwargs)

    def traced_tensor(*args, **kwargs):
        if trace.phase != 'action':
            return action_tensor(*args, **kwargs)
        with trace.scope('action_forward'):
            result = action_tensor(*args, **kwargs)
        trace.emit('begin', 'action_result_readback')
        return result

    def traced_update(**kwargs):
        nonlocal cpu_sample
        if not trace.active:
            try:
                return update(**kwargs)
            finally:
                cpu_sample = None
        if kwargs['total_steps'] != trace.step:
            raise RuntimeError('Trace step differs from the actual update step.')
        try:
            with trace.scope('update'):
                metric = update(**kwargs, gpu_compute_submission_diagnostic=trace)
            if trace.comparing:
                trace.comparison.record(trace, metric=metric)
            return metric
        finally:
            if trace.comparing:
                trace.comparison.restore_cache()

    def traced_key(executor, batch, mode):
        key = graph_key(executor, batch, mode)
        if trace.active:
            trace.emit('lookup', 'update_graph', key=list(key), cache='reuse' if key in executor.graphs else 'new')
        if trace.comparing:
            trace.comparison.select(executor, key, batch, trace)
        return key

    def traced_capture(executor, key, batch):
        if trace.comparing and first_sample_arm:
            return trace.comparison.recapture_first_sample(trace, capture)
        if (not trace.active and trace.comparison is None
                or trace.comparing and trace.comparison.arm == 'fresh-graph'):
            return capture(executor, key, batch)
        with trace.scope('update_graph_capture'):
            result = capture(executor, key, batch)
        if trace.comparison is not None:
            trace.comparison.capture(executor, key, trace.step, cpu_sample)
        return result

    with (ExitStack() as hooks,
          patch.object(trainer, '_new_episode', bounded_reset),
          patch.object(engine, 'select_action', traced_select),
          patch.object(engine, '_action_inference_tensor', traced_tensor),
          patch.object(engine, 'update_once', traced_update),
          patch.object(V2ObservationBatch, 'to', traced_action_to),
          patch.object(GPUCompute, 'key', traced_key),
          patch.object(GPUCompute, 'capture_shape', traced_capture)):
        if first_sample_arm:
            sample_batch = engine._sample_training_batch

            def traced_sample(*args, **kwargs):
                nonlocal cpu_sample
                batch = sample_batch(*args, **kwargs)
                if trace.comparison.first_sample is None and trace.step < COMPARE_STEP:
                    cpu_sample = batch
                return batch

            hooks.enter_context(patch.object(engine, '_sample_training_batch', traced_sample))
        run(trainer)
    raise RuntimeError('Formal trainer returned before the diagnostic stop hook.')


def worker(args):
    scratch = read_json(args.out / 'manifest.json')
    description = read_json(args.out / 'diagnostic.json')
    if any(description[key] != value for key, value in describe(args).items()):
        raise ValueError('Diagnostic worker arguments differ from the launcher.')
    spec = next(spec for spec in scratch['plan']['jobs'] if spec['job']['seed'] == args.worker)
    job = spec['job']['id']
    output = args.out / job
    comparison = StepComparison(args, output, description['original']) if args.compare_686 else None
    stop_step = description['stop_step']
    original_run, original_write = V2FormalStageTrainer.run, stage._write_strict_json
    write_json = formal.atomic_json
    completed = False

    def run(trainer):
        nonlocal completed
        with (output / 'cuda_trace.jsonl').open('x', encoding='utf-8', buffering=1) as events:
            trace = Trace(events, sync_window=args.sync_window, comparison=comparison)
            try:
                return run_bounded(original_run, trainer, trace)
            except DiagnosticComplete:
                completed = True
                raise
            except BaseException as exc:
                if comparison is not None:
                    if comparison.failure is None:
                        comparison.failure = dict(section='worker', phase='host_call',
                                                  error_type=type(exc).__name__, error=str(exc))
                    try:
                        comparison.record(trace)
                    except Exception as report_error:
                        print(f'Could not write step686_result.json: {report_error}', file=sys.stderr, flush=True)
                raise

    def stage_report(path, value):
        if (completed and path == output / 'metrics_execution_error.json'
                and value['error_type'] == DiagnosticComplete.__name__):
            return
        return original_write(path, value)

    def worker_report(path, value):
        if completed and path == output / 'worker_summary.json':
            if value.get('error', '').startswith(DiagnosticComplete.__name__ + ':'):
                value = {key: item for key, item in value.items() if key != 'error'}
                value.update(diagnostic_only=True, diagnostic_completed=True,
                             execution_completed=True, stage_steps=stop_step, formal_passed=None)
        return write_json(path, value)

    def reject_checkpoint(*args, **kwargs):
        raise RuntimeError('The short diagnostic cannot publish a checkpoint.')

    with (patch.object(formal, 'batch_jobs', return_value=[spec['job'] for spec in scratch['plan']['jobs']]),
          patch.object(V2FormalStageTrainer, 'run', run),
          patch.object(stage, '_write_strict_json', stage_report),
          patch.object(stage, 'save_v2_formal_checkpoint', reject_checkpoint),
          patch.object(stage, 'save_v2_periodic_snapshot', reject_checkpoint),
          patch.object(formal, 'atomic_json', worker_report)):
        try:
            formal.worker(source_args(args.out, scratch, job=job))
        except DiagnosticComplete:
            report = read_json(output / 'worker_summary.json')
            if report.get('diagnostic_completed') is True and report['stage_steps'] == stop_step:
                return 0
            raise
    raise RuntimeError('Worker exited without the bounded diagnostic stop.')


def launch(args, original, wanted, binding):
    chosen = [job for job in wanted if job['seed'] in args.seeds]
    evidence = original['inputs']
    args.out.mkdir(parents=True, exist_ok=False)
    description = dict(describe(args), original=binding)
    scratch = dict(original, batch_id=uuid4().hex,
                   plan=dict(original['plan'], root=str(args.out),
                             jobs=[formal.resolved(job, formal.COURSE_EXPERIMENT)[0] for job in chosen]))
    formal.atomic_json(args.out / 'diagnostic.json', description)
    formal.atomic_json(args.out / 'manifest.json', scratch)
    argv = [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '--run',
            '--course-root', str(args.course_root), '--out', str(args.out),
            '--seeds', *map(str, args.seeds)]
    if args.single_result is not None:
        argv.extend(('--single-result', str(args.single_result)))
    if args.sync_window:
        argv.append('--sync-window')
    if args.compare_686:
        argv.extend(('--compare-686', args.compare_686))
    if args.graph_reference is not None:
        argv.extend(('--graph-reference', str(args.graph_reference)))
    if args.staged_reference is not None:
        argv.extend(('--staged-reference', str(args.staged_reference)))
    status = dict(description, execution_completed=False)
    processes, session = {}, None
    failure = None
    started = time.monotonic()
    first_signal = True

    def terminate(signum, frame):
        nonlocal first_signal
        if first_signal:
            first_signal = False
            raise InterruptedError('SIGTERM received by diagnostic launcher.')

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        with ExitStack() as logs:
            session = Session(args.out / 'mps')
            formal.atomic_json(args.out / 'mps/namespace.json', dict(batch_id=scratch['batch_id'],
                pipe_directory=str(session.pipe), log_directory=str(session.root / 'logs')))
            session.start()
            capability(args.out / 'capability', session)
            capability(args.out / 'termination_capability', session, terminate_client=True)
            for job in chosen:
                name = job['id']
                log = logs.enter_context((args.out / f'{name}.log').open('x', encoding='utf-8'))
                process = subprocess.Popen([*argv, '--worker', str(job['seed'])], cwd=formal.REPO,
                    env=dict(session.env, PYTHONPATH=str(formal.REPO / 'src')),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes[name] = process
                session.register({name: process})
                formal.atomic_json(args.out / 'pids.json', {key: p.pid for key, p in processes.items()})
            formal.wait_ready(processes, args.out, original['plan']['initialization_timeout'])
            old = args.course_root / ATTEMPT / 'train'
            for name in processes:
                ready, reference = [read_json(root / name / 'ready.json') for root in (args.out, old)]
                for key in ('seed_manifest', 'initial_state_sha256'):
                    if ready[key] != reference[key]:
                        raise ValueError(f'Diagnostic initialization differs from original {name}: {key}')
            original_jobs = formal.jobs

            def release_jobs(experiment='baseline', batch=None):
                if (experiment, batch) == (formal.COURSE_EXPERIMENT, BATCH):
                    return chosen
                return original_jobs(experiment, batch)

            with patch.object(formal, 'jobs', new=release_jobs):
                formal.release(processes, args.out, session, formal.COURSE_EXPERIMENT, batch=BATCH)
            while True:
                returncodes = {name: process.poll() for name, process in processes.items()}
                if any(code not in (None, 0) for code in returncodes.values()):
                    raise RuntimeError(f'Diagnostic workers returned {returncodes}. See worker and MPS logs.')
                if all(code == 0 for code in returncodes.values()):
                    break
                time.sleep(.2)
            for name, process in processes.items():
                report = read_json(args.out / name / 'worker_summary.json')
                if (report.get('diagnostic_completed') is not True or report.get('stage_steps') != description['stop_step']
                        or report.get('pid') != process.pid or report.get('batch_id') != scratch['batch_id']):
                    raise ValueError(f'Incomplete diagnostic worker {name}.')
            formal.check_inputs(evidence)
            status['execution_completed'] = True
    except BaseException as exc:
        failure = exc
        status['error'] = f'{type(exc).__name__}: {exc}'
        try:
            formal.atomic_json(args.out / 'abort.json', status)
        finally:
            if session is not None:
                try:
                    formal.stop_owned(processes, session)
                except BaseException as cleanup:
                    status['worker_cleanup_error'] = str(cleanup)
    finally:
        try:
            if session is not None:
                try:
                    status['mps_cleanup'] = session.finish()
                except BaseException as cleanup:
                    status['mps_cleanup'] = dict(stopped=False, error=str(cleanup))
            status['execution_completed'] &= status.get('mps_cleanup', {}).get('stopped') is True
            status['processes'] = {name: dict(pid=p.pid, exit_code=p.poll()) for name, p in processes.items()}
            status['process_wall_seconds'] = time.monotonic() - started
            formal.atomic_json(args.out / 'summary.json', status)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    if failure is not None:
        raise failure
    return 0 if status['execution_completed'] else 1


def main(argv=None):
    args = parse_args(argv)
    if not args.run:
        print(json.dumps(describe(args), indent=2))
        return 0
    if os.name != 'posix' or Path(sys.executable).resolve() != Path(formal.provenance.PYTHON).resolve():
        raise ValueError('Run the diagnostic in the declared Linux server Python.')
    if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') != '0':
        raise ValueError('Keep CUDA_LAUNCH_BLOCKING unset or 0; explicit synchronization is window-scoped.')
    if args.worker is not None:
        return worker(args)
    print(f'Diagnostic output: {args.out}', flush=True)
    original, jobs, binding = load_original(args)
    if args.single_result is not None:
        require_single_result(args.single_result, binding)
    comparison_references(args, binding)
    return launch(args, original, jobs, binding)


if __name__ == '__main__':
    raise SystemExit(main())
