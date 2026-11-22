from contextlib import ExitStack, nullcontext
from collections import namedtuple
from dataclasses import make_dataclass
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

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

    def test_fresh_graph_replays_new_entry_only_after_verification(self):
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
        from brain_uav.trainers.v2_gpu_compute import clone_batch
        from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch

        obs = diag.V2ObservationBatch(torch.zeros(1, 6), torch.zeros(1, 4),
                                     torch.zeros(1, 2, 19), torch.tensor([[True, False]]))
        sample = V2ReplayBatch(obs, torch.zeros(1, 3), torch.zeros(1, 1), obs,
                               *(torch.zeros(1, 1) for _ in range(4)),
                               sample_failure_fraction=0.25)
        key = diag.COMPARE_KEY
        original = dict(batch_id='source')
        for change in (None, 'state', 'reference', 'replay'):
            with self.subTest(change=change), TemporaryDirectory() as tmp:
                root = Path(tmp)
                old, new = Mock(), Mock()
                stable = clone_batch(sample)
                fraction, result = torch.tensor(0.25), torch.tensor([4.0])
                entry = (old, stable, fraction, torch.tensor([-1.0]), ((), ()), {})
                fresh = (new, clone_batch(sample), fraction.clone(), result, ((), ()), {})
                executor = SimpleNamespace(e=SimpleNamespace(last_total_steps=685, window_update_statistics=[]),
                    graphs={key: entry}, captures=3, replays=42, banks=[[[]], [[]]],
                    failure_fraction=fraction, noise=torch.zeros(1), coefficients=torch.zeros(4),
                    injections=torch.zeros(4), persistent=lambda: [])
                state = dict(training='before')
                trace = diag.Trace(io.StringIO(), sync_window=False)
                trace.synchronize = Mock()
                trace.step = 685

                def capture(actual_key, batch):
                    self.assertEqual(actual_key, key)
                    self.assertIs(batch, sample)
                    self.assertNotIn(key, executor.graphs)
                    executor.graphs[key] = fresh
                    executor.captures += 1
                    if change == 'state':
                        state['training'] = 'advanced during capture'

                executor.capture_shape = Mock(side_effect=capture)

                def replay():
                    state['training'] = 'after'
                    if change == 'replay':
                        trace.synchronize.side_effect = RuntimeError('illegal access')

                new.replay.side_effect = replay
                prior_failure = dict(section='graph_replay_submit', phase='synchronize')
                output_hash = state_hash(result)
                with patch.object(diag.StepComparison, 'state', side_effect=lambda *_: dict(state)):
                    graph_args = diag.parse_args(['--out', str(root / 'graph'), '--compare-686', 'graph'])
                    graph = diag.StepComparison(graph_args, graph_args.out, original)
                    trace.comparison = graph
                    graph.capture(executor, key, 42)
                    trace.step = 686
                    graph.select(executor, key, sample, trace)
                    graph.before_replay(trace)
                    graph.failure = prior_failure
                    graph.record(trace)
                    before = diag.read_json(graph_args.out / 'step686_before.json')
                    staged_before = dict(before, arm='staged', reference_matches=True)
                    staged_output = root / 'staged' / 'ann_seed41'
                    formal.atomic_json(staged_output / 'step686_before.json', staged_before)
                    formal.atomic_json(staged_output / 'step686_result.json', dict(
                        arm='staged', pid=before['pid'], before_sha256=formal.file_hash(
                            staged_output / 'step686_before.json'), completed=True, submitted=True, last_step=686,
                        reference_matches=True, first_failure=None, metrics={}, post_state=dict(training='after'),
                        result_sha256=output_hash))
                    (graph_args.out / 'ann_seed41').mkdir()
                    for path in graph_args.out.glob('step686_*.json'):
                        path.rename(graph_args.out / 'ann_seed41' / path.name)
                    for arm in ('graph', 'staged'):
                        formal.atomic_json(root / arm / 'summary.json', dict(compare_686=arm,
                            diagnostic_sha256='old-script', original=original, seeds=[41],
                            mps_cleanup=dict(stopped=True), execution_completed=arm == 'staged',
                            processes=dict(ann_seed41=dict(pid=before['pid'], exit_code=int(arm == 'graph')))))
                    args = diag.parse_args(['--out', str(root / 'fresh'), '--compare-686', 'fresh-graph',
                        '--graph-reference', str(root / 'graph'), '--staged-reference', str(root / 'staged')])
                    with patch.object(diag, 'HISTORICAL_DIAGNOSTIC_SHA256', 'old-script'):
                        comparison = diag.StepComparison(args, args.out, original)
                    trace.comparison = comparison
                    comparison.capture(executor, key, 42)
                    comparison.select(executor, key, sample, trace)
                    replay_proxy, _, _, readback, _, _ = executor.graphs[key]
                    if change == 'reference':
                        executor.noise.fill_(1)

                    expectation = nullcontext()
                    if change == 'replay':
                        expectation = self.assertRaisesRegex(RuntimeError, 'illegal access')
                    elif change:
                        expectation = self.assertRaisesRegex(ValueError, 'changed|mismatch')
                    with expectation:
                        with trace.scope('graph_replay_submit'):
                            replay_proxy.replay()
                    old.replay.assert_not_called()
                    self.assertEqual(executor.capture_shape.call_count, int(change != 'reference'))
                    self.assertEqual(new.replay.call_count, int(change in (None, 'replay')))
                    if change == 'replay':
                        self.assertEqual((comparison.failure['section'], comparison.failure['phase']),
                                         ('graph_replay_submit', 'synchronize'))
                    if change is None:
                        torch.testing.assert_close(readback.cpu(), result)
                        comparison.record(trace, metric=make_dataclass('Metric', [])())
                        report = diag.read_json(args.out / 'step686_result.json')
                        self.assertTrue(all(report['comparison'].values()))
                        recapture = diag.read_json(args.out / 'step686_recapture.json')
                        self.assertEqual((recapture['capture_count_before'], recapture['capture_count_after']), (3, 4))
                        self.assertTrue(recapture['state_matches'])
                        self.assertNotEqual(recapture['addresses']['output'], before['replay_addresses']['output'])
                        events = [json.loads(line) for line in trace.stream.getvalue().splitlines()]
                        markers = [(row['section'], row['event']) for row in events]
                        ordered = [markers.index(pair) for pair in (
                            ('fresh_graph_recapture', 'begin'), ('fresh_graph_recapture', 'sync_done'),
                            ('graph_replay_submit', 'host_return'), ('graph_replay_submit', 'sync_done'))]
                        self.assertEqual(ordered, sorted(ordered))
                    comparison.restore_cache()
                    self.assertIs(executor.graphs[key], entry)

    def test_first_sample_graph_uses_formal_copy_between_capture_and_replay(self):
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
        from brain_uav.trainers.v2_gpu_compute import GPUCompute, clone_batch, tensors
        from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch

        def sample(mask, value, fraction):
            obs = diag.V2ObservationBatch(torch.full((1, 6), value), torch.zeros(1, 4),
                                         torch.zeros(1, 2, 19), torch.tensor([mask]))
            next_obs = diag.V2ObservationBatch(obs.ego_features, obs.goal_features, obs.zone_features,
                                              torch.tensor([mask[::-1]]))
            return V2ReplayBatch(obs, torch.full((1, 3), value), torch.zeros(1, 1), next_obs,
                                 *(torch.zeros(1, 1) for _ in range(4)), sample_failure_fraction=fraction)

        key = diag.COMPARE_KEY
        metric_type = make_dataclass('Metric', [('sample_failure_fraction', float)])
        for failure in (None, 'capture', 'copy', 'replay'):
            with self.subTest(failure=failure), TemporaryDirectory() as tmp:
                root = Path(tmp)
                args = diag.parse_args(['--out', str(root / 'out'), '--compare-686', 'first-sample-graph',
                    '--graph-reference', str(root / 'graph'), '--staged-reference', str(root / 'staged')])
                first, real = sample([True, False], 1.0, 0.125), sample([False, True], 2.0, 0.75)
                first_hash, real_hash = state_hash(tensors(first)), state_hash(tensors(real))
                noise, result = torch.ones(1, 3), torch.tensor([0., 0., 0.])
                e = SimpleNamespace(device=torch.device('cpu'), batch_size=1, policy_delay=2, actor_freeze_steps=900,
                    last_total_steps=683, window_update_statistics=[], mc_return_weight=0, update_count=0,
                    critic_update_count=0, critic_target_update_count=0, actor_update_count=0,
                    gpu_compute_lightweight_timing=False, gpu_compute_non_blocking_coefficients=False,
                    bc_reference_actor=None, spike_regularization=SimpleNamespace(beta=lambda _: 0.),
                    policy_noise=0.2, noise_clip=0.5, _new_episode=Mock(), select_action=Mock(),
                    _action_inference_tensor=Mock(), _sample_training_batch=Mock(side_effect=[first, real, real]))
                gpu = object.__new__(GPUCompute)
                vars(gpu).update(e=e, graphs={}, captures=0, replays=0, handoffs=0, copy_seconds=0.,
                    compute_seconds=0., ready_to_result_seconds=0., mode_replays={'frozen_targets': 0, 'critic_only': 0},
                    params=([], []), adams=(None, SimpleNamespace(params=[], states=[])), banks=[[[]], [[]]],
                    published=[None, None], bound_device=e.device, zone_limit=2, noise=torch.zeros_like(noise),
                    coefficients=torch.zeros(4), injections=torch.zeros(4), persistent=lambda: [],
                    validate_binding=Mock(), metrics=lambda *_: metric_type(0.0))
                trainer = SimpleNamespace(engine=e, config=SimpleNamespace(max_steps=750000),
                    result=SimpleNamespace(stage_steps=683), _new_episode=e._new_episode,
                    env=SimpleNamespace(last_curriculum_level='medium', zones=[None, None]))
                trace = diag.Trace(io.StringIO(), sync_window=False)
                trace.synchronize = Mock()
                recaptured = Mock()

                def capture(executor, actual_key, batch):
                    def replay():
                        if trace.step == 686:
                            recaptured()
                            self.assertEqual(state_hash(tensors(fixed)), real_hash)
                            self.assertEqual(fraction.item(), 0.75)
                            if failure == 'replay':
                                trace.synchronize.side_effect = RuntimeError('illegal replay')

                    if trace.step == 686:
                        self.assertEqual(state_hash(tensors(batch)), first_hash)
                        self.assertEqual(batch.sample_failure_fraction, 0.125)
                        if failure == 'capture':
                            raise RuntimeError('illegal capture')
                        if failure == 'copy':
                            trace.synchronize.side_effect = [None, RuntimeError('illegal copy')]
                    fixed = clone_batch(batch)
                    fraction = torch.tensor(batch.sample_failure_fraction)
                    executor.failure_fraction = fraction
                    executor.graphs[actual_key] = (SimpleNamespace(replay=replay), fixed, fraction, result, ((), ()), {})
                    executor.captures += 1

                def update(*, total_steps, gpu_compute_submission_diagnostic=None):
                    batch = e._sample_training_batch(total_steps).to(e.device)
                    return gpu._step_batch(batch, noise, total_steps, 0., owned_sources=[],
                                           submission_diagnostic=gpu_compute_submission_diagnostic)

                e.update_once = update
                identity = dict(key=list(key), capture_step=684, capture_count=2, prior_replays=2,
                    graph_keys=[list(key), [2, 2, 'critic_only', 0, 0]], participation=[[], []],
                    scene=dict(level='medium', obstacles=2), obs_shapes=diag.shapes(real.obs),
                    next_obs_shapes=diag.shapes(real.next_obs), obs_mask=[[False, True]],
                    next_obs_mask=[[True, False]], sample_sha256=real_hash, fixed_sha256=real_hash,
                    noise_sha256=state_hash(noise), fraction_sha256=state_hash(torch.tensor(0.75)),
                    coefficients_sha256=state_hash(torch.tensor([0., 0., 0.2, 0.5])),
                    injection_sha256=state_hash(torch.zeros(4)), state=dict(steps=685))
                staged = dict(completed=True, first_failure=None, metrics=dict(sample_failure_fraction=0.75),
                              post_state=dict(steps=686), result_sha256=state_hash(result))
                refs = dict(graph=(identity, dict(completed=False, first_failure={})), staged=(identity, staged))

                def run(trainer):
                    for step in (684, 685, 686):
                        e.select_action()
                        e.update_once(total_steps=step)
                        trainer.result.stage_steps = step
                        if step == 684:
                            first.obs.presence_mask.fill_(False)
                            first.action.zero_()
                    e.select_action()

                expected = (self.assertRaisesRegex(RuntimeError, f'illegal {failure}') if failure else
                            self.assertRaises(diag.DiagnosticComplete))
                with (patch.object(diag, 'comparison_references', return_value=refs),
                      patch.object(diag.StepComparison, 'state', side_effect=lambda *_: dict(steps=e.last_total_steps)),
                      patch.object(GPUCompute, 'capture_shape', capture)):
                    trace.comparison = diag.StepComparison(args, args.out, {})
                    with expected:
                        diag.run_bounded(run, trainer, trace)
                self.assertEqual(recaptured.call_count, int(failure in (None, 'replay')))
                initial = diag.read_json(args.out / 'first_capture_sample.json')
                self.assertEqual((initial['capture_step'], initial['obs_mask'], initial['next_obs_mask']),
                                 (684, [[True, False]], [[False, True]]))
                if failure:
                    first_failure = trace.comparison.failure
                    locations = dict(capture=('first_sample_recapture', 'host_call'),
                                     copy=('graph_fixed_input_copy', 'synchronize'),
                                     replay=('graph_replay_submit', 'synchronize'))
                    self.assertEqual((first_failure['section'], first_failure['phase']), locations[failure])
                else:
                    report = diag.read_json(args.out / 'step686_result.json')
                    self.assertTrue(all(report['comparison'].values()))
                    self.assertTrue(diag.read_json(args.out / 'step686_recapture.json')[
                        'capture_sample_differs_from_replay'])
                    markers = [(event['section'], event['event']) for event in
                               map(json.loads, trace.stream.getvalue().splitlines())]
                    ordered = [markers.index((section, 'sync_done')) for section in
                               ('first_sample_recapture', 'graph_fixed_input_copy', 'graph_replay_submit')]
                    self.assertEqual(ordered, sorted(ordered))

    def test_gap_only_selects_first_warmup_when_second_has_no_clear(self):
        self.check_gap_warmup_bypass(clears_on=1, function_ids=(4, 5))

    def test_gap_bypasses_second_call_to_same_function(self):
        self.check_gap_warmup_bypass(clears_on=2, function_ids=(4, 4))

    def check_gap_warmup_bypass(self, clears_on, function_ids):
        function_id_type = namedtuple('FunctionID', 'id')
        ids = [function_id_type(value) for value in function_ids]
        clear = Mock()
        warmup = Mock()
        compiled = Mock()
        torch_stub = SimpleNamespace(__version__='2.5.1+cu124', version=SimpleNamespace(cuda='12.4',
            git_version='stub'), _C=SimpleNamespace(_cuda_clearCublasWorkspaces=clear),
            cuda=SimpleNamespace(CUDAGraph=type('Graph', (), dict(
                capture_begin=lambda self: None, capture_end=lambda self: None))))

        class Manager:
            ids_to_funcs = {function_id: SimpleNamespace(model=compiled) for function_id in ids}

            def __init__(self):
                self.next_warmup_id = iter(range(81, 83))

            def new_graph_id(self):
                raise AssertionError('Expected warmup node ID')

            def new_warmup_node_id(self):
                return SimpleNamespace(id=next(self.next_warmup_id))

            def run_eager(self, inputs, function_id):
                self.new_warmup_node_id()
                warmup(inputs[0])
                if inputs[0] == clears_on:
                    torch_stub._C._cuda_clearCublasWorkspaces()

            def record_function(self, inputs, function_id):
                raise AssertionError('Expected warmup')

            def execute_node(self, node, inputs):
                raise AssertionError('Expected warmup')

        def run(gap):
            trace = SimpleNamespace(step=685, phase='action')
            with ExitStack() as hooks:
                gap.install(trace, hooks, torch_stub, Manager)
                manager = Manager()
                for call, function_id in enumerate(ids, 1):
                    manager.run_eager([call], function_id)
            return gap.report()

        observation = run(diag.GraphGap())
        self.assertEqual(clear.call_count, 1)
        warmup.reset_mock()
        bypass = run(diag.GraphGap(({}, observation)))
        compiled.assert_called_once_with([clears_on])
        warmup.assert_called_once_with(3 - clears_on)
        self.assertEqual(clear.call_count, 1)
        self.assertEqual(observation['action_candidates'], [dict(step=685, call_index=clears_on, branch='warmup',
            function_id=ids[clears_on - 1].id, graph_id=80 + clears_on)])
        clear_event, = [row for row in observation['events'] if row['event'] == 'cublas_workspace_clear']
        self.assertEqual(clear_event['call_index'], clears_on)
        self.assertTrue(bypass['bypassed'])
        bypass_event, = [row for row in bypass['events'] if row['event'] == 'action_graph_bypass']
        self.assertEqual((bypass_event['call_index'], bypass_event['graph_id']), (clears_on, None))

    def test_gap_observation_and_single_action_bypass_keep_old_training_graph(self):
        from brain_uav.trainers.v2_gpu_compute import clone_batch
        from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = dict(batch_id='course')
            observe_args = diag.parse_args(['--out', str(root / 'observe'), '--compare-686', 'gap-observe'])
            bypass_args = diag.parse_args(['--out', str(root / 'bypass'), '--compare-686', 'gap-bypass',
                                          '--gap-reference', str(observe_args.out)])
            obs = diag.V2ObservationBatch(torch.zeros(1, 6), torch.zeros(1, 4), torch.zeros(1, 2, 19),
                                         torch.tensor([[True, False]]))
            batch = V2ReplayBatch(obs, torch.zeros(1, 3), torch.zeros(1, 1), obs,
                                  *(torch.zeros(1, 1) for _ in range(4)))
            fixed = clone_batch(batch)
            unchanged = Mock()
            graph = Mock()
            key = diag.COMPARE_KEY
            entry = (graph, fixed, torch.zeros(()), torch.tensor([1.]), ((), ()), {})
            executor = SimpleNamespace(e=SimpleNamespace(last_total_steps=685, window_update_statistics=[]),
                graphs={key: entry}, captures=1, replays=9, banks=[[[]], [[]]], persistent=lambda: [],
                noise=torch.zeros(1), coefficients=torch.zeros(4), injections=torch.zeros(4))
            graph_id = SimpleNamespace(id=8)
            function_id = namedtuple('FunctionID', 'id')(4)

            class Graph:
                capture_begin = lambda self: None
                capture_end = lambda self: None

            class Manager:
                def new_graph_id(self):
                    return graph_id

                def new_warmup_node_id(self):
                    return SimpleNamespace(id=-1)

                def run_eager(self, inputs, function_id):
                    return inputs[0]

                def record_function(self, inputs, function_id):
                    self.new_graph_id()
                    torch_stub._C._cuda_clearCublasWorkspaces()
                    gpu = Graph()
                    gpu.capture_begin()
                    gpu.capture_end()
                    return inputs[0]

                def execute_node(self, node, inputs):
                    return inputs[0]

            manager = Manager()
            manager.ids_to_funcs = {function_id: SimpleNamespace(model=unchanged)}
            unchanged.side_effect = lambda inputs: inputs[0]
            clear = Mock()
            torch_stub = SimpleNamespace(__version__='2.5.1+cu124', version=SimpleNamespace(cuda='12.4',
                git_version='stub'), _C=SimpleNamespace(_cuda_clearCublasWorkspaces=clear),
                cuda=SimpleNamespace(CUDAGraph=Graph))
            model_state = {'models': {'actor': 'same'}, 'optimizers': {'critic': 'adam'}, 'gradients': {'critic': 'grad'}}

            for args, mismatch in ((observe_args, False), (bypass_args, False), (bypass_args, True)):
                with self.subTest(arm=args.compare_686, mismatch=mismatch), ExitStack() as hooks:
                    output = args.out / 'ann_seed41'
                    comparison = diag.StepComparison(args, output, original)
                    trace = diag.Trace(io.StringIO(), sync_window=False, comparison=comparison)
                    trace.synchronize = Mock()
                    gap = trace.gap
                    gap.install(trace, hooks, torch_stub, Manager)
                    comparison.capture(executor, key, 2)
                    gap.last_key = list(key)
                    gap.training_completed(684, 2)
                    gap.event(685, 'scenario_reset', level='medium', obstacles=2)
                    trace.step, trace.phase = 685, 'action'
                    gap.event(685, 'action_input', shapes=diag.shapes(obs))
                    action = manager.record_function([0.5], function_id)
                    gap.event(685, 'action_result', value=[action + int(mismatch)])
                    trace.step = 686
                    manager.execute_node(SimpleNamespace(id=graph_id,
                        wrapped_function=SimpleNamespace(id=function_id)), [action])
                    trace.phase = 'training_update'
                    torch_stub._C._cuda_clearCublasWorkspaces()
                    comparison.select(executor, key, batch, trace)
                    self.assertIs(executor.graphs[key], entry)
                    observed = args is observe_args
                    if observed:
                        trace.synchronize.side_effect = [None, RuntimeError('CUDA illegal memory access')]
                    expected = (self.assertRaisesRegex(RuntimeError, 'illegal memory') if observed else
                                self.assertRaisesRegex(ValueError, 'action_result.685') if mismatch else nullcontext())
                    prior_replays = graph.replay.call_count
                    with (patch.object(comparison, 'state', return_value={'training': 'unchanged'}),
                          patch.object(comparison, 'gap_state', return_value=model_state), expected):
                        with trace.scope('graph_replay_submit'):
                            executor.graphs[key][0].replay()
                    self.assertEqual(graph.replay.call_count - prior_replays, int(not mismatch))
                    with patch.object(comparison, 'state', return_value={'training': 'after'}):
                        comparison.record(trace, metric=None if observed or mismatch else make_dataclass('Metric', [])())
                    evidence = diag.read_json(output / 'graph_gap.json')
                    result = diag.read_json(output / 'step686_result.json')
                    self.assertEqual(evidence['last_successful_replay_step'], 684)
                    self.assertEqual(evidence['runtime']['torch'], '2.5.1+cu124')
                    self.assertEqual(evidence['bypassed'], not observed)
                    reused, = [row for row in evidence['events']
                               if row.get('branch') == 'reuse' and row['event'] == 'action_graph_begin']
                    self.assertEqual((reused['call_index'], reused['function_id'], reused['graph_id']), (1, 4, 8))
                    if observed:
                        self.assertEqual(evidence['action_candidates'], [dict(step=685, call_index=1,
                            branch='record', function_id=4, graph_id=8)])
                        capture, = [row for row in evidence['events'] if row['event'] == 'action_cuda_capture_end']
                        self.assertEqual((capture['call_index'], capture['function_id'], capture['graph_id']), (1, 4, 8))
                        formal.atomic_json(args.out / 'summary.json', dict(compare_686=args.compare_686,
                            diagnostic_sha256=formal.file_hash(PATH), original=original, seeds=[41],
                            mps_cleanup=dict(stopped=True), processes=dict(ann_seed41=dict(pid=result['pid']))))
                    else:
                        self.assertEqual(result['completed'], not mismatch)
                        self.assertEqual(result['reference_matches'], not mismatch)
                        self.assertEqual(evidence['action_candidates'], [])
            self.assertEqual(clear.call_count, 4)
            self.assertEqual(unchanged.call_count, 2)
            baseline = observe_args.out / 'ann_seed41'
            path = baseline / 'graph_gap.json'
            evidence = diag.read_json(path)
            evidence['action_candidates'] = []
            formal.atomic_json(path, evidence)
            path = baseline / 'step686_result.json'
            formal.atomic_json(path, dict(diag.read_json(path), gap_sha256=formal.file_hash(baseline / 'graph_gap.json')))
            with self.assertRaisesRegex(ValueError, 'No action capture or action workspace-clear candidate'):
                diag.gap_reference(observe_args.out, original)

    def test_warmup_diagnostic_verifies_startup_reuse_and_continues_to_1000(self):
        from brain_uav.scripts.diagnose_v2_critic_calibration import state_hash
        from brain_uav.trainers.v2_gpu_compute import clone_batch
        from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            staged = root / 'staged' / 'ann_seed41' / 'step686_result.json'
            raw_result = torch.tensor([1.])
            metric = make_dataclass('Metric', [('critic_updated', bool)])(True)
            after = {'training': 'after'}
            formal.atomic_json(staged, dict(completed=True, metrics=dict(critic_updated=True),
                result_sha256=state_hash(raw_result), post_state=after))
            args = diag.parse_args(['--out', str(root / 'run'), '--warmup-staged-reference', str(root / 'staged')])
            self.assertEqual(diag.describe(args)['stop_step'], 1000)
            output = args.out / 'ann_seed41'
            formal.atomic_json(output / 'ready.json', dict(compilation=dict(cuda_graph_action_inference_evidence={
                'shapes': [{'batch_shape': [1, 1], 'verified': True, 'cuda_graph_launch_count': 1}]})))
            obs = diag.V2ObservationBatch(torch.zeros(1, 6), torch.zeros(1, 4), torch.zeros(1, 2, 19),
                                         torch.tensor([[True, False]]))
            batch = V2ReplayBatch(obs, torch.zeros(1, 3), torch.zeros(1, 1), obs,
                                  *(torch.zeros(1, 1) for _ in range(4)))
            key = diag.COMPARE_KEY
            graph = Mock()
            entry = (graph, clone_batch(batch), torch.zeros(()), raw_result, ((), ()), {})
            executor = SimpleNamespace(e=SimpleNamespace(last_total_steps=685, window_update_statistics=[]),
                graphs={key: entry}, captures=1, replays=9, banks=[[[]], [[]]], persistent=lambda: [],
                noise=torch.zeros(1), coefficients=torch.zeros(4), injections=torch.zeros(4))
            engine = SimpleNamespace(device='cpu', select_action=lambda: torch.zeros(2), update_once=Mock(),
                                     _action_inference_tensor=Mock())
            trainer = SimpleNamespace(engine=engine, config=SimpleNamespace(max_steps=750000),
                result=SimpleNamespace(stage_steps=685), _new_episode=Mock(),
                env=SimpleNamespace(last_curriculum_level='easy', zones=[None]))
            trace = diag.Trace(io.StringIO(), sync_window=False)
            trace.synchronize = Mock()
            state = {'training': 'before'}
            legacy = diag.StepComparison(diag.parse_args(['--out', str(root / 'legacy'),
                '--compare-686', 'graph']), root / 'legacy', {})
            trace.step, trace.scene = 686, dict(level='easy', obstacles=1)
            identity_path = root / 'legacy' / 'step686_before.json'
            with patch.object(diag.StepComparison, 'state', side_effect=lambda *_: dict(state)):
                legacy.capture(executor, key, 5)
                legacy.select(executor, key, batch, trace)
                legacy.before_replay(trace)
                staged_identity = diag.read_json(identity_path)['identity']
                with patch.object(diag, 'warmup_reference', return_value=(staged_identity, diag.read_json(staged))):
                    comparison = diag.StepComparison(args, output, {})
                trace.comparison = comparison
                comparison.capture(executor, key, 5)
                trace.gap.last_key = list(key)
                trace.gap.training_completed(684, 5)
                trace.gap.event(686, 'action_input', shapes={'zone_features': [1, 1, 19]})
                trace.gap.event(686, 'action_graph_complete', branch='reuse', graph_id=4, function_id=2)

                def update(*, total_steps, gpu_compute_submission_diagnostic):
                    if total_steps == 686:
                        comparison.select(executor, key, batch, trace)
                        with trace.scope('graph_replay_submit'):
                            executor.graphs[key][0].replay()
                        state.update(after)
                    return metric

                def run(trainer):
                    for step in range(686, 1001):
                        engine.select_action()
                        engine.update_once(total_steps=step)
                        trainer.result.stage_steps = step
                    engine.select_action()

                engine.update_once = update
                with (patch.object(diag.GraphGap, 'install'),
                      patch.object(comparison, 'gap_state', return_value={}),
                      self.assertRaises(diag.DiagnosticComplete)):
                    diag.run_bounded(run, trainer, trace)
            self.assertEqual(trainer.result.stage_steps, 1000)
            graph.replay.assert_called_once_with()
            self.assertIs(executor.graphs[key], entry)
            result = diag.read_json(output / 'step686_result.json')
            self.assertTrue(result['startup_action_graph_reused'])
            self.assertTrue(all(result['comparison'].values()))
            trace.gap.event(685, 'action_cuda_capture_end', graph_id=4, function_id=2)
            with self.assertRaisesRegex(ValueError, 'did not reuse a startup-verified'):
                comparison.verify_startup_graph_reuse(686)

    def test_warmup_diagnostic_only_accepts_the_stage_entry_source_difference(self):
        with TemporaryDirectory() as tmp:
            seal = Path(tmp) / 'seal.json'
            expected = {diag.WARMUP_SOURCE: 'old', 'other': 'original'}
            formal.atomic_json(seal, {'sha256': expected})
            historical = dict(paths={'source_manifest': str(seal)}, source={'sha256': expected})
            amended = dict(expected, **{diag.WARMUP_SOURCE: 'new'})
            for files in (amended, dict(amended, other='changed'), expected):
                with self.subTest(files=files), patch.object(formal, 'identity', return_value={'sha256': files}):
                    if files == amended:
                        self.assertEqual(diag.fixed_warmup_inputs(historical)['source']['sha256'], amended)
                        self.assertEqual(historical['source']['sha256'], expected)
                    else:
                        with self.assertRaisesRegex(ValueError, 'original source seal plus the stage-entry fix'):
                            diag.fixed_warmup_inputs(historical)

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
            staged = root / 'staged' / 'ann_seed41/step686_result.json'
            formal.atomic_json(staged, {'completed': True})
            warmup = dict(warmup_staged_reference=str(root / 'staged'), warmup_source_sha256='code',
                          diagnostic_sha256='script')
            formal.atomic_json(root / 'summary.json', dict(summary, **warmup))
            result = dict(arm=diag.WARMUP_ARM, pid=41, completed=True, submitted=True, last_step=686,
                          reference_matches=True, startup_action_graph_reused=True,
                          staged_result_sha256=formal.file_hash(staged),
                          comparison=dict(metrics_exact_match=True, post_state_exact_match=True,
                                          result_sha256_exact_match=True))
            result_path = root / 'ann_seed41/step686_result.json'
            formal.atomic_json(result_path, result)
            diag.require_single_result(root, binding, warmup=warmup)
            with self.assertRaisesRegex(ValueError, 'without --warmup-staged-reference'):
                diag.require_single_result(root, binding)
            formal.atomic_json(result_path, dict(result, startup_action_graph_reused=False))
            with self.assertRaisesRegex(ValueError, 'requires verified seed-41 replay'):
                diag.require_single_result(root, binding, warmup=warmup)

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
