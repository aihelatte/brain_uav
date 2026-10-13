"""CUDA capability gate, NOT a training executor or a performance benchmark.

Run on the target 2.5.1 CUDA build before implementing gpu_td3_update.
Outer manual graphs own capture; all model tensor functions are eager, with
no nested Inductor graphs. Inputs/relations are prepared before capture.
Synthetic losses exercise backward, clipping, gated Adam and target order;
this does not prove full TD3 loss, replay or engine-counter equivalence.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

import torch
from torch import nn

from brain_uav.models import V2ANNPolicyActor, V2ANNCritic, V2SNNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.observations.v2_contract import EGO_FEATURE_DIM, GOAL_FEATURE_DIM, ZONE_FEATURE_DIM


def require_capability():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA capability NOT validated: CUDA is unavailable; no CPU fallback.')
    if torch.__version__.split('+')[0] != '2.5.1':
        raise RuntimeError('Probe binds the internal fused-Adam ABI to PyTorch 2.5.1 only.')
    if torch.version.cuda != '12.4' or not hasattr(torch, '_fused_adam_'):
        raise RuntimeError('Expected target PyTorch 2.5.1+cu124 fused-Adam capability.')


def replay_budget(capacity=500_000, zones=6, success_fraction=.25):
    """Exact payload sizes for current CPU replay layout, not a VRAM fit claim."""
    if type(capacity) is not int or capacity <= 0 or type(zones) is not int or zones < 0:
        raise ValueError('Invalid replay capacity or zone capacity.')
    if isinstance(success_fraction, bool) or not 0 <= success_fraction <= 1:
        raise ValueError('Invalid success fraction.')
    obs_bytes = 4 * (EGO_FEATURE_DIM + GOAL_FEATURE_DIM + zones * ZONE_FEATURE_DIM)
    # Two observations/counts, action[2], reward/done, four bool markers,
    # double sampling weight, double Fenwick node and int64 slot generation.
    primary_stride = 2 * obs_bytes + 8 + 16 + 4 + 8 + 8 + 8
    success_stride = 2 * obs_bytes + 8 + 16 + 2
    primary = capacity * primary_stride + 8  # Fenwick's extra node
    success = int(capacity * success_fraction) * success_stride
    return dict(capacity=capacity, zones=zones, success_capacity=int(capacity * success_fraction),
                primary_bytes=primary, success_bytes=success, payload_bytes=primary + success,
                payload_gib=(primary + success) / 2**30,
                excludes=['GPU sampler workspace/metadata', 'fixed batch and gradients',
                          'model/Adam tensors', 'graph-private allocations', 'temporary kernels'],
                fit_validated=False)


class GatedAdam:
    """Version-bound probe of PyTorch fused Adam found_inf (no AMP/scaling).

    Match native fused Adam: increment step, fused kernel conditionally commits,
    subtract found_inf. Missing grads are absent from ALL lists, never zeroed.
    Static gradient-presence contract is checked outside the captured step.
    """
    def __init__(self, parameters, lr):
        self.params = list(parameters)
        self.lr = lr
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.steps = [torch.zeros((), device=p.device) for p in self.params]

    @torch.no_grad()
    def step(self, found_inf):
        active = [i for i, p in enumerate(self.params) if p.grad is not None]
        if not active:
            return
        params = [self.params[i] for i in active]
        steps = [self.steps[i] for i in active]
        torch._foreach_add_(steps, 1)
        torch._fused_adam_(params, [p.grad for p in params],
                           [self.m[i] for i in active], [self.v[i] for i in active], [], steps,
                           amsgrad=False, lr=self.lr, beta1=.9, beta2=.999,
                           weight_decay=0., eps=1e-8, maximize=False,
                           grad_scale=None, found_inf=found_inf)
        torch._foreach_sub_(steps, [found_inf] * len(steps))

    def state_tensors(self):
        return self.m + self.v + self.steps


def first_error(previous, failed, code):
    return torch.where((previous == 0) & failed, code, previous)


def checked_clip(parameters, loss, error, loss_code, gradient_code):
    """Device-only error reductions; no .item(), no CPU barrier in this probe."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    error = first_error(error, ~torch.isfinite(loss), loss_code)
    # Same norm/clip arithmetic as clip_grad_norm_; error flag BEFORE Adam.
    norms = torch._foreach_norm(gradients, 2)
    norm = torch.linalg.vector_norm(torch.stack(norms), 2)
    error = first_error(error, ~torch.isfinite(norm), gradient_code)
    coefficient = torch.clamp(1. / (norm + 1e-6), max=1.)
    torch._foreach_mul_(gradients, coefficient)
    return error


@torch.no_grad()
def gated_targets(online, targets, error):
    candidate = torch._foreach_mul(targets, .995)
    torch._foreach_add_(candidate, online, alpha=.005)
    # Select rather than NaN*0 or lr=0: failed branch never writes candidate.
    for target, value in zip(targets, candidate):
        target.copy_(torch.where(error == 0, value, target))


class UpdateProbe:
    def __init__(self, model):
        scales = V2ObservationScales(1312.5, .1, 600., .6)
        limits = torch.tensor([.1, .1])
        cls = V2ANNPolicyActor if model == 'ann' else V2SNNPolicyActor
        self.actor = cls(scales, 2, 192, limits, projection_dim=32).cuda()
        self.critics = [V2ANNCritic(scales, 2, 128).cuda() for _ in range(2)]
        self.actor_target = deepcopy(self.actor)
        self.critic_targets = deepcopy(self.critics)
        self.bc = deepcopy(self.actor)
        self.model = model
        self.actor_params = list(self.actor.parameters())
        self.critic_params = [p for c in self.critics for p in c.parameters()]
        self.actor_adam = GatedAdam(self.actor_params, 9.5e-4 if model == 'snn' else 1.55e-4)
        self.critic_adam = GatedAdam(self.critic_params, 2.5e-4)
        ego = torch.randn(64, EGO_FEATURE_DIM, device='cuda') * .1
        goal = torch.randn(64, GOAL_FEATURE_DIM, device='cuda') * .1
        zones = torch.zeros(64, 2, ZONE_FEATURE_DIM, device='cuda')
        zones[:, :, 0] = 1  # synthetic valid shape token
        mask = torch.arange(2, device='cuda')[None] < (torch.arange(64, device='cuda') % 3)[:, None]
        self.args = self.actor.zone_set_encoder.prepare_tensor_forward_arguments(ego, goal, zones, mask)
        self.next_args = tuple(x.clone() for x in self.args)
        self.action = torch.randn(64, 2, device='cuda') * .01
        self.noise = torch.randn_like(self.action) * .01  # fixed explicitly, not RNG in graph
        self.injections = torch.zeros(4, device='cuda')  # loss NaN / gradient Inf flags
        self.beta = torch.tensor(.5, device='cuda')
        self.bc_weight = torch.tensor(.2, device='cuda')
        self.error = torch.zeros((), device='cuda', dtype=torch.int64)
        self.outputs = None

    def actor_forward(self, actor, arguments):
        if self.model == 'ann':
            return actor._compute_full_forward_tensors(*arguments), self.beta.new_zeros(())
        with actor.reset_state_context():
            context = actor.zone_set_encoder._compute_policy_context_tensors(*arguments)
            raw, first, _ = actor.snn_head._forward_impl(actor.projection(context))
            return torch.tanh(raw) * actor.action_limit, first.mean()

    def operation(self, mode):
        # Graph-owned stable gradients for this capability probe only. Production
        # zero_grad(set_to_none=True) semantics are NOT claimed by this probe.
        for p in self.critic_params:
            if p.grad is not None:
                p.grad.zero_()
        with torch.no_grad():
            next_action, _ = self.actor_forward(self.actor_target, self.next_args)
            next_action = (next_action + self.noise.clamp(-.02, .02)).clamp(-.1, .1)
            td = .99 * torch.minimum(*[c._compute_full_forward_tensors(*self.next_args, next_action)
                                      for c in self.critic_targets])
        q = [c._compute_full_forward_tensors(*self.args, self.action) for c in self.critics]
        loss = sum((value - td).square().mean() for value in q)
        loss = torch.where(self.injections[0] != 0, torch.full_like(loss, float('nan')), loss)
        loss.backward()
        g = self.critic_params[0].grad
        g.copy_(torch.where(self.injections[1] != 0, torch.full_like(g, float('inf')), g))
        error = checked_clip(self.critic_params, loss, self.error.new_zeros(()), 1, 2)
        self.critic_adam.step((error != 0).float())
        if mode != 'critic_only':
            gated_targets(self.critic_params, [p for c in self.critic_targets for p in c.parameters()], error)
        rate = loss.new_zeros(())
        actor_loss = loss.new_zeros(())
        if mode == 'actor':
            for p in self.actor_params:
                if p.grad is not None:
                    p.grad.zero_()
            action, rate = self.actor_forward(self.actor, self.args)
            with torch.no_grad():
                reference, _ = self.actor_forward(self.bc, self.args)
                context = self.critics[0].zone_set_encoder._compute_policy_context_tensors(*self.args)
            # Frozen head parameters, Q retains derivative w.r.t. actor action.
            head = self.critics[0].head
            q_actor = torch.func.functional_call(head, {n: p.detach() for n, p in head.named_parameters()},
                                                 (torch.cat((context, action), dim=-1),))
            actor_loss = -q_actor.mean() + self.bc_weight * (action - reference).square().mean()
            actor_loss = actor_loss + self.beta * torch.relu(rate - .17).square()
            actor_loss = torch.where(self.injections[2] != 0, torch.full_like(actor_loss, float('nan')), actor_loss)
            actor_loss.backward()
            g = self.actor_params[0].grad
            g.copy_(torch.where(self.injections[3] != 0, torch.full_like(g, float('inf')), g))
            error = checked_clip(self.actor_params, actor_loss, error, 3, 4)
            self.actor_adam.step((error != 0).float())
            gated_targets(self.actor_params, list(self.actor_target.parameters()), error)
        self.error.copy_(error)
        self.outputs = torch.stack((loss.detach(), actor_loss.detach(), rate.detach()))

    def persistent(self):
        modules = [self.actor, *self.critics, self.actor_target, *self.critic_targets, self.bc]
        return [p for m in modules for p in m.parameters()] + self.critic_adam.state_tensors() + self.actor_adam.state_tensors()


def compare(a, b):
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.


def numeric_snapshot(tensor):
    """Independent diagnostic values, never part of the autograd graph."""
    return tensor.detach().clone()


def _location(progress, stage, *, model=None, mode=None, replay_index=None, injection=None):
    if progress is not None:
        progress.update(stage=stage, model=model, mode=mode,
                        replay_index=replay_index, injection=injection)


def probe_snn_reset(probe, stream, *, progress=None, evidence=None):
    """A -> A and A -> B -> A through one captured SNN callable."""
    evidence = {} if evidence is None else evidence
    evidence.update(passed=False, completed_samples=[])
    _location(progress, 'snn_reset_prepare', model='snn')
    initial_ego = numeric_snapshot(probe.args[0])
    def forward():
        with torch.no_grad():
            return probe.actor_forward(probe.actor, probe.args)
    stream.wait_stream(torch.cuda.current_stream())
    _location(progress, 'snn_reset_warmup_capture', model='snn')
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = forward()
    torch.cuda.current_stream().wait_stream(stream)
    samples = []
    for index, offset in enumerate((0., 0., .2, 0.)):
        _location(progress, 'snn_reset_replay', model='snn', replay_index=index)
        probe.args[0].copy_(initial_ego + offset)
        expected = forward()
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(output, expected):
            compare(actual, reference)
        samples.append(tuple(numeric_snapshot(x) for x in output))
        evidence['completed_samples'].append(index)
    _location(progress, 'snn_reset_repeat_compare', model='snn')
    for index in (1, 3):
        for actual, reference in zip(samples[index], samples[0]):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    probe.args[0].copy_(initial_ego)
    _location(progress, 'snn_reset_membrane_check', model='snn')
    for name in ('lif1', 'lif2'):
        membrane = getattr(probe.actor.snn_head, name).v
        if isinstance(membrane, torch.Tensor):
            assert torch.count_nonzero(membrane) == 0
        else:
            assert membrane == 0
    evidence.update(passed=True, patterns=['A,A', 'A,B,A'], lif1_reset=True, lif2_reset=True)
    return evidence


def probe_adam_gate(*, progress=None, evidence=None):
    evidence = {} if evidence is None else evidence
    evidence.update(passed=False, skipped_nonfinite=[])
    _location(progress, 'adam_gate_prepare')
    p = nn.Parameter(torch.tensor([1., -2.], device='cuda'))
    missing = nn.Parameter(torch.tensor([3.], device='cuda'))
    p.grad = torch.ones_like(p)
    adam = GatedAdam([p, missing], .001)
    flag = torch.ones((), device='cuda')
    # Nonzero preexisting moments/step: skipping must preserve all three.
    for value in adam.state_tensors():
        value.fill_(7.)
    initial = [numeric_snapshot(x) for x in [p, missing, *adam.state_tensors()]]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _location(progress, 'adam_gate_warmup_capture')
    with torch.cuda.stream(stream):
        for _ in range(3):
            adam.step(flag)
        with torch.cuda.graph(graph, stream=stream):
            adam.step(flag)
    torch.cuda.current_stream().wait_stream(stream)
    for bad in (float('inf'), float('nan')):
        label = 'Inf' if bad == float('inf') else 'NaN'
        _location(progress, 'adam_gate_skip_replay', injection=label)
        p.grad.fill_(bad)
        graph.replay()
        torch.cuda.synchronize()
        for x, y in zip([p, missing, *adam.state_tensors()], initial):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        evidence['skipped_nonfinite'].append(label)
    # Direct native fused Adam parity on a successful commit from identical state.
    _location(progress, 'adam_gate_native_parity')
    reference = nn.Parameter(p.detach().clone())
    optimizer = torch.optim.Adam([reference], lr=.001, fused=True)
    optimizer.state[reference] = dict(step=numeric_snapshot(adam.steps[0]), exp_avg=numeric_snapshot(adam.m[0]), exp_avg_sq=numeric_snapshot(adam.v[0]))
    reference.grad = torch.ones_like(reference)
    p.grad.fill_(1.)
    flag.zero_()
    graph.replay()
    optimizer.step()
    torch.cuda.synchronize()
    for actual, expected in zip([p, adam.steps[0], adam.m[0], adam.v[0]],
                                [reference, *[optimizer.state[reference][k] for k in ('step', 'exp_avg', 'exp_avg_sq')]]):
        compare(actual, expected)
    evidence['native_fused_parity'] = True
    _location(progress, 'adam_gate_missing_gradient')
    assert missing.grad is None
    for x, y in zip([missing, adam.m[1], adam.v[1], adam.steps[1]],
                    [initial[1], initial[3], initial[5], initial[7]]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    evidence.update(passed=True, missing_gradient_preserved=True)
    return evidence


def probe_update(model, *, progress=None, evidence=None):
    evidence = {} if evidence is None else evidence
    rows, failures = [], []
    evidence.update(passed=False, continuous_replays=rows, error_barriers=failures,
                    completed_captures=[], completed_reference_steps=[])
    _location(progress, 'update_prepare', model=model)
    torch.manual_seed(7)
    probe = UpdateProbe(model)
    saved = [numeric_snapshot(x) for x in probe.persistent()]
    def restore():
        with torch.no_grad():
            for x, value in zip(probe.persistent(), saved):
                x.copy_(value)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    reset_evidence = None
    if model == 'snn':
        reset_evidence = evidence['snn_reset'] = {}
        probe_snn_reset(probe, stream, progress=progress, evidence=reset_evidence)
    # reset probe's final input restoration/sample clones run on the current
    # stream AFTER its earlier wait. Order them before update warmup, once only.
    _location(progress, 'update_warmup_dependency', model=model)
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outputs = {}, {}
    start = time.perf_counter()
    with torch.cuda.stream(stream):
        for mode in ('critic_only', 'frozen_targets', 'actor'):
            _location(progress, 'update_warmup', model=model, mode=mode)
            for _ in range(3):
                probe.operation(mode)
            restore()
            graphs[mode] = torch.cuda.CUDAGraph()
            _location(progress, 'update_capture', model=model, mode=mode)
            with torch.cuda.graph(graphs[mode], stream=stream):
                probe.operation(mode)
            outputs[mode] = probe.outputs
            restore()
            evidence['completed_captures'].append(mode)
    _location(progress, 'update_capture_restore_check', model=model)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - start
    torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)
    for x, y in zip(probe.persistent(), saved):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    evidence.update(capture_seconds=capture_seconds, warmup_capture_rng_unchanged=True)
    # Each sequence begins from the exact same independent persistent snapshots.
    sequence = ['critic_only', 'frozen_targets', 'actor'] * 2
    expected = []
    for index, mode in enumerate(sequence):
        _location(progress, 'update_reference', model=model, mode=mode, replay_index=index)
        probe.operation(mode)
        gradients = [None if p.grad is None else numeric_snapshot(p.grad)
                     for p in probe.critic_params + probe.actor_params]
        expected.append(([numeric_snapshot(x) for x in probe.persistent()], numeric_snapshot(probe.outputs), gradients))
        evidence['completed_reference_steps'].append(index)
    restore()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for index, mode in enumerate(sequence):
            _location(progress, 'update_replay_compare', model=model, mode=mode, replay_index=index)
            graphs[mode].replay()
            torch.cuda.synchronize()
            errors = [compare(x, y) for x, y in zip(probe.persistent(), expected[index][0])]
            # Adam step is discrete. Do not accept a fractional tolerance here.
            for x in probe.actor_adam.steps + probe.critic_adam.steps:
                position = next(i for i, value in enumerate(probe.persistent()) if value is x)
                torch.testing.assert_close(x, expected[index][0][position], rtol=0, atol=0)
            gradient_errors = []
            for parameter_index, (parameter, reference) in enumerate(zip(
                    probe.critic_params + probe.actor_params, expected[index][2])):
                if parameter_index >= len(probe.critic_params) and mode != 'actor':
                    continue  # not consumed in this step, per existing diagnostic contract
                if reference is None:
                    assert parameter.grad is None, 'Active gradient presence changed.'
                    continue
                gradient_errors.append(compare(parameter.grad, reference))
            compare(outputs[mode], expected[index][1])
            rows.append(dict(index=index, mode=mode, max_abs_error=max(errors),
                             post_clip_gradient_max_abs_error=max(gradient_errors, default=0.)))
    launches = sum(e.count for e in profiler.key_averages() if 'cudaGraphLaunch' in e.key)
    evidence['cuda_graph_launches'] = launches
    _location(progress, 'update_replay_evidence', model=model)
    if launches < len(sequence):
        raise RuntimeError('Requested manual graph replay evidence missing.')
    for injection in range(4):
        _location(progress, 'update_error_barrier', model=model, mode='actor', injection=injection)
        restore()
        probe.injections.zero_()
        probe.injections[injection] = 1
        graphs['actor'].replay()
        torch.cuda.synchronize()
        assert int(probe.error) == injection + 1
        # Critic failures protect ALL commits; actor failures keep critic commits.
        protected = probe.actor_params + list(probe.actor_target.parameters()) + probe.actor_adam.state_tensors()
        saved_by_id = {id(x): y for x, y in zip(probe.persistent(), saved)}
        if injection < 2:
            protected = probe.persistent()
        else:
            assert any(not torch.equal(p, saved_by_id[id(p)]) for p in probe.critic_params)
            assert any(not torch.equal(p, saved_by_id[id(p)])
                       for c in probe.critic_targets for p in c.parameters())
            assert all(float(step) == 1 for step in probe.critic_adam.steps)
        for p in protected:
            torch.testing.assert_close(p, saved_by_id[id(p)], rtol=0, atol=0)
        failures.append(dict(stage=injection + 1, commits_protected=True))
    probe.injections.zero_()
    _location(progress, 'update_memory_report', model=model)
    evidence.update(passed=True, capture_seconds=capture_seconds, cuda_graph_launches=launches,
                continuous_replays=rows, error_barriers=failures,
                snn_reset=reset_evidence,
                warmup_capture_rng_unchanged=True,
                model_adam_persistent_bytes=sum(x.numel() * x.element_size() for x in probe.persistent()),
                allocated_bytes=torch.cuda.memory_allocated(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                compilation_owner='outer_manual_graph; uncompiled model tensor functions',
                limitations=['Synthetic losses, no terminal geometry or real TD3 loss parity',
                             'Prepared relations; no device replay/gather validation',
                             'Static graph-owned gradients; empty-token/mask participation not certified',
                             'No engine counters, checkpoint rebinding or performance claims'])
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refuse nonempty probe output directory.')
    output.mkdir(parents=True, exist_ok=True)
    result = dict(diagnostic_only=True, gpu_td3_update_implemented=False, passed=False,
                  status='waiting_cuda_capability_validation', torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, replay_budget=replay_budget())
    result['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    result['git_status'] = subprocess.check_output(['git', 'status', '--short'], text=True).splitlines()
    progress = {}
    try:
        _location(progress, 'require_capability')
        require_capability()
        _location(progress, 'gpu_information')
        result['gpu'] = torch.cuda.get_device_name()
        result['adam_gate'] = {}
        probe_adam_gate(progress=progress, evidence=result['adam_gate'])
        result['updates'] = {}
        for model in ('ann', 'snn'):
            result['updates'][model] = {}
            probe_update(model, progress=progress, evidence=result['updates'][model])
        result.update(passed=True, status='capability_probes_passed_not_executor_validated')
        return 0
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['failure_context'] = dict(progress)
        result['traceback'] = traceback.format_exc()
        if torch.cuda.is_available():
            result['status'] = 'capability_probe_failed'
        return 1
    finally:
        (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(dict(passed=result['passed'], status=result['status'])))


if __name__ == '__main__':
    raise SystemExit(main())
