"""Removable CUDA feasibility probe; no TD3 engine, replay or training wiring.

Manual graphs own capture. A graph is bound to specific work-gradient banks;
changing Python .grad never redirects its writes. Only the terminal status
read publishes a bank. Synthetic losses are NOT real TD3 correctness evidence.
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import argparse
import json
import traceback

import torch
from torch import nn

from brain_uav.scripts.probe_v2_gpu_td3 import (
    GatedAdam, first_error, gated_targets, numeric_snapshot, require_capability,
)


def preserving_clip(parameters, loss, error, loss_code, gradient_code):
    """Select coefficient BEFORE multiplication; never perform Inf * 0."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    error = first_error(error, ~torch.isfinite(loss), loss_code)
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients, 2)), 2)
    error = first_error(error, ~torch.isfinite(norm), gradient_code)
    coefficient = torch.where(error == 0, (1. / (norm + 1e-6)).clamp(max=1.),
                              torch.ones_like(norm))
    torch._foreach_mul_(gradients, coefficient)
    return error


class ToyActor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.optional = nn.Parameter(torch.zeros(2))
        self.projection = nn.Linear(4, 4)
        self.model = model
        if model == 'snn':
            from brain_uav.models.v2_snn import V2SNNPolicyHead
            self.head = V2SNNPolicyHead(4, 8, 2, time_window=4, tau=2.)
        else:
            self.head = nn.Sequential(nn.Linear(4, 8), nn.LeakyReLU(), nn.Linear(8, 2))

    def forward(self, x, optional):
        x = self.projection(x)
        if self.model == 'snn':
            from spikingjelly.activation_based import functional
            functional.reset_net(self.head)
            try:
                action, first, _ = self.head._forward_impl(x)
                rate = first.mean()
            finally:
                functional.reset_net(self.head)
        else:
            action, rate = self.head(x), x.new_zeros(())
        return (action + self.optional if optional else action).tanh(), rate


class ToyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.optional = nn.Parameter(torch.zeros(1))
        self.head = nn.Sequential(nn.Linear(6, 8), nn.LeakyReLU(), nn.Linear(8, 1))

    def forward(self, x, action, optional):
        q = self.head(torch.cat((x, action), -1))
        return q + self.optional if optional else q


class ProbeState:
    def __init__(self, model, device):
        self.actor = ToyActor(model).to(device)
        self.critic = ToyCritic().to(device)
        self.actor_target, self.critic_target = deepcopy(self.actor), deepcopy(self.critic)
        self.params = [list(self.critic.parameters()), list(self.actor.parameters())]
        self.optimizers = [torch.optim.Adam(p, lr=.001, fused=True) for p in self.params]
        self.adams = [GatedAdam(p, .001) for p in self.params]
        self.counts = dict(updates=0, critic_targets=0, actors=0)
        self.model = model

    def persistent(self):
        return [p for m in (self.actor, self.critic, self.actor_target, self.critic_target)
                for p in m.parameters()] + [t for a in self.adams for t in a.state_tensors()]

    def active(self, group, optional):
        return [p for p in self.params[group] if optional or p is not self.params[group][0]]

    def publish_adam(self, group, optional):
        # Exclude absent parameters from optimizer.state, as native Adam does.
        a = self.adams[group]
        for i, p in enumerate(a.params):
            if optional or i:
                self.optimizers[group].state[p] = dict(step=a.steps[i], exp_avg=a.m[i], exp_avg_sq=a.v[i])

    def losses(self, x, optional):
        with torch.no_grad():
            action, _ = self.actor_target(x, optional)
            td = self.critic_target(x, action, optional) * .99
        return (self.critic(x, x[:, :2].tanh(), optional) - td).square().mean()

    def actor_loss(self, x, optional):
        action, rate = self.actor(x, optional)
        q = torch.func.functional_call(self.critic,
            {n: p.detach() for n, p in self.critic.named_parameters()}, (x, action, optional))
        return -q.mean() + .2 * action.square().mean() + torch.relu(rate - .17).square()

    def commit_counts(self, mode, error):
        if error in (0, 3, 4) and mode != 'critic_only':
            self.counts['critic_targets'] += 1
        if error == 0:
            self.counts['updates'] += 1
            self.counts['actors'] += int(mode == 'actor')

    def eager(self, x, optional, mode, injection):
        from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine as Engine
        loss = self.losses(x, optional)
        error = 0
        for group in (0, 1) if mode == 'actor' else (0,):
            if group:
                loss = self.actor_loss(x, optional)
            loss_code, grad_code = (1, 2) if group == 0 else (3, 4)
            try:
                if injection[2*group]:
                    loss = loss * float('nan')
                Engine._require_finite_loss(loss, component=str(group), total_steps=0)
            except FloatingPointError:
                error = loss_code
                break
            self.optimizers[group].zero_grad(set_to_none=True)
            loss.backward()
            if injection[2*group+1]:
                    self.active(group, optional)[0].grad.fill_(float('nan') if injection[2*group+1] == 2 else float('inf'))
            try:
                Engine._validate_and_clip_gradients(self.params[group], max_norm=1., component=str(group), total_steps=0)
            except FloatingPointError:
                error = grad_code
                break
            self.optimizers[group].step()
            # Mirror native state into the independently allocated probe Adam
            # storage for a complete comparison including not-yet-active params.
            for i, p in enumerate(self.params[group]):
                if p.grad is not None:
                    native = self.optimizers[group].state[p]
                    with torch.no_grad():
                        for dst, key in ((self.adams[group].steps[i], 'step'),
                                         (self.adams[group].m[i], 'exp_avg'), (self.adams[group].v[i], 'exp_avg_sq')):
                            dst.copy_(native[key])
            if group or mode != 'critic_only':
                target = self.actor_target if group else self.critic_target
                gated_targets(self.params[group], list(target.parameters()), loss.new_zeros((), dtype=torch.long))
        self.commit_counts(mode, error)
        return error


class BankProbe:
    def __init__(self, state, max_graphs=16, progress=None):
        self.s, self.max_graphs = state, max_graphs
        self.banks = [[[torch.zeros_like(p) for p in params] for _ in range(2)] for params in state.params]
        self.published = [None, None]
        self.graphs = {}
        self.capture_seconds = 0.
        self.capture_evidence = []
        self.stream = torch.cuda.Stream()
        self.progress = progress

    def stage(self, name, key):
        if getattr(self, 'progress', None) is not None:
            self.progress.update(stage=name, graph_key=list(key))

    def operation(self, x, flags, optional, mode):
        s = self.s
        loss = s.losses(x, optional)
        loss = torch.where(flags[0] != 0, torch.full_like(loss, float('nan')), loss)
        finite_critic = torch.isfinite(loss)
        loss.backward()
        g = s.active(0, optional)[0].grad
        bad = torch.where(flags[1] == 2, torch.full_like(g, float('nan')), torch.full_like(g, float('inf')))
        g.copy_(torch.where(flags[1] != 0, bad, g))
        error = preserving_clip(s.params[0], loss, loss.new_zeros((), dtype=torch.long), 1, 2)
        s.adams[0].step((error != 0).float())
        if mode != 'critic_only':
            gated_targets(s.params[0], list(s.critic_target.parameters()), error)
        publish_actor = torch.zeros_like(finite_critic)
        if mode == 'actor':
            actor_loss = s.actor_loss(x, optional)
            actor_loss = torch.where(flags[2] != 0, torch.full_like(actor_loss, float('nan')), actor_loss)
            publish_actor = (error == 0) & torch.isfinite(actor_loss)
            actor_loss.backward()
            g = s.active(1, optional)[0].grad
            bad = torch.where(flags[3] == 2, torch.full_like(g, float('nan')), torch.full_like(g, float('inf')))
            g.copy_(torch.where(flags[3] != 0, bad, g))
            error = preserving_clip(s.params[1], actor_loss, error, 3, 4)
            s.adams[1].step((error != 0).float())
            gated_targets(s.params[1], list(s.actor_target.parameters()), error)
        return torch.stack((error, finite_critic.long(), publish_actor.long()))

    def capture(self, key, x):
        if len(self.graphs) >= self.max_graphs:
            raise RuntimeError('Bounded graph cache exhausted; no eviction or fallback.')
        _, optional, mode, cb, ab = key
        self.stage('capture_prepare', key)
        s = self.s
        persistent = s.persistent()
        saved = [numeric_snapshot(t) for t in persistent]
        # Capture may zero even the PUBLIC bank (other optimizer/non-actor mode).
        # Preserve values for BOTH banks, including externally bound references.
        bank_tensors = [g for group in self.banks for bank in group for g in bank]
        bank_values = [numeric_snapshot(g) for g in bank_tensors]
        refs = [p.grad for params in s.params for p in params]
        grad_values = [None if g is None else numeric_snapshot(g) for g in refs]
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        counts = s.counts.copy()
        flags = x.new_zeros(4)
        output = None
        start = perf_counter()
        attempt = dict(key=list(key), restored=False, completed=False)
        self.capture_evidence.append(attempt)
        self.stream.wait_stream(torch.cuda.current_stream())

        def bind_and_zero():
            for group, bank in enumerate((cb, ab)):
                for i, p in enumerate(s.params[group]):
                    p.grad = self.banks[group][bank][i] if (optional or i) and (group == 0 or mode == 'actor') else None
                    if p.grad is not None:
                        p.grad.zero_()

        def restore():
            with torch.no_grad():
                for dst, src in zip(persistent + bank_tensors, saved + bank_values):
                    dst.copy_(src)
                for p, ref, value in zip([p for params in s.params for p in params], refs, grad_values):
                    p.grad = ref
                    if ref is not None:
                        ref.copy_(value)
            s.counts = counts.copy()
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)

        failure_location = None
        try:
            with torch.cuda.stream(self.stream):
                self.stage('capture_warmup', key)
                for _ in range(3):
                    bind_and_zero()
                    self.operation(x, flags, optional, mode)
                graph = torch.cuda.CUDAGraph()
                # The ZERO operations are recorded at the selected bank addresses.
                self.stage('capture_graph', key)
                with torch.cuda.graph(graph, stream=self.stream):
                    bind_and_zero()
                    output = self.operation(x, flags, optional, mode)
            torch.cuda.current_stream().wait_stream(self.stream)
            torch.cuda.synchronize()
            # Reset contexts must already have reset both LIFs, not mask leaks.
            assert_reset(s)
        except BaseException:
            failure_location = dict(getattr(self, 'progress', None) or {})
            attempt['failure_location'] = failure_location
            raise
        finally:
            self.stage('capture_restore', key)
            torch.cuda.current_stream().wait_stream(self.stream)
            torch.cuda.synchronize()
            restore()
            if failure_location is not None and getattr(self, 'progress', None) is not None:
                self.progress.clear()
                self.progress.update(failure_location)
        assert_state_values(persistent, saved)
        assert_state_values(bank_tensors, bank_values)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        assert s.counts == counts
        for p, ref, value in zip([p for params in s.params for p in params], refs, grad_values):
            assert p.grad is ref
            if ref is not None:
                torch.testing.assert_close(ref, value, rtol=0, atol=0, equal_nan=True)
        self.capture_seconds += perf_counter() - start
        attempt.update(restored=True, completed=True, seconds=perf_counter()-start)
        self.graphs[key] = (graph, x, flags, output)

    def step(self, x, optional, mode, injection):
        work = [0 if bank is None else 1-bank for bank in self.published]
        key = (x.shape[0], optional, mode, *work)
        if key not in self.graphs:
            self.capture(key, numeric_snapshot(x))
        graph, stable_x, flags, output = self.graphs[key]
        self.stage('replay_input', key)
        stable_x.copy_(x)
        flags.copy_(torch.tensor(injection, device=x.device, dtype=x.dtype))
        self.stage('replay', key)
        graph.replay()
        # Only ONE terminal status read. No intermediate loss/gradient barrier.
        self.stage('terminal_status', key)
        error, pub_critic, pub_actor = output.cpu().tolist()
        for group, publish in enumerate((pub_critic, pub_actor)):
            if publish:
                self.published[group] = work[group]
                for i, p in enumerate(self.s.params[group]):
                    p.grad = self.banks[group][work[group]][i] if optional or i else None
        if error in (0, 3, 4):
            self.s.publish_adam(0, optional)
        if error == 0 and mode == 'actor':
            self.s.publish_adam(1, optional)
        self.s.commit_counts(mode, error)
        return error, key


def assert_reset(s):
    if s.model == 'snn':
        for actor in (s.actor, s.actor_target):
            for node in (actor.head.lif1, actor.head.lif2):
                if isinstance(node.v, torch.Tensor):
                    assert torch.count_nonzero(node.v) == 0
                else:
                    assert node.v == 0


def assert_state_values(actual, expected):
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def compare_states(actual, expected):
    maxima = []
    for index, (a, b) in enumerate(zip(actual.persistent(), expected.persistent())):
        try:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        except AssertionError as exc:
            raise AssertionError(f'persistent[{index}]: {exc}') from exc
        maxima.append(float((a.double()-b.double()).abs().max()))
    assert actual.counts == expected.counts
    names = [list(actual.critic.named_parameters()), list(actual.actor.named_parameters())]
    for group, (ap, bp) in enumerate(zip(actual.params, expected.params)):
        for i, (a, b) in enumerate(zip(ap, bp)):
            name = ('critic.' if group == 0 else 'actor.') + names[group][i][0]
            assert (a.grad is None) == (b.grad is None), (name, 'None mismatch')
            if a.grad is not None:
                # Inf/NaN positions must match exactly; finite entries use original tolerance.
                for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
                    assert torch.equal(predicate(a.grad), predicate(b.grad)), (name, 'nonfinite mismatch')
                mask = torch.isfinite(b.grad)
                torch.testing.assert_close(a.grad[mask], b.grad[mask], rtol=1e-4, atol=1e-5)
            assert set(actual.optimizers[group].state.get(a, {})) == set(expected.optimizers[group].state.get(b, {}))
            if a in actual.optimizers[group].state:
                torch.testing.assert_close(actual.optimizers[group].state[a]['step'],
                                           expected.optimizers[group].state[b]['step'], rtol=0, atol=0)
    assert_reset(actual)
    assert_reset(expected)
    return max(maxima, default=0.)


def protected_values(s, group):
    target = s.actor_target if group else s.critic_target
    return s.params[group] + s.adams[group].state_tensors() + list(target.parameters())


def run_model(model, progress, evidence):
    progress.update(stage='model_initialization', model=model)
    torch.manual_seed(7)
    reference = ProbeState(model, 'cuda')
    candidate = ProbeState(model, 'cuda')
    for a, b in zip(candidate.persistent(), reference.persistent()):
        with torch.no_grad():
            a.copy_(b)
    executor = BankProbe(candidate, progress=progress)
    # Repeated shapes/participation, all three modes, both banks, loss/Inf paths.
    sequence = [
        (4, False, 'critic_only', [0, 0, 0, 0]),
        (4, False, 'frozen_targets', [0, 0, 0, 0]),
        (4, False, 'actor', [0, 0, 0, 0]),
        (4, False, 'actor', [1, 1, 0, 0]),  # loss wins over gradient
        (4, False, 'actor', [0, 1, 0, 0]),
        (4, False, 'actor', [0, 0, 1, 1]),
        (4, False, 'actor', [0, 0, 0, 1]),
        (4, False, 'actor', [0, 2, 0, 0]),
        (2, True, 'actor', [0, 0, 0, 0]),
        (4, False, 'actor', [0, 0, 0, 0]),
        (4, False, 'actor', [0, 0, 0, 0]),
        (2, True, 'actor', [0, 0, 0, 0]),
        (4, False, 'actor', [0, 0, 0, 0]),
    ]
    evidence.update(passed=False, steps=[], captures=executor.capture_evidence)
    old_keys = set()
    for index, (size, optional, mode, flags) in enumerate(sequence):
        progress.update(model=model, stage='step', index=index, shape=[size, 4], optional=optional, mode=mode, injection=flags)
        x = torch.arange(size*4, device='cuda', dtype=torch.float32).reshape(size, 4)/20.
        # Diagnosis only, not the executor mechanism: exact protected-state
        # assertions must not merely accept small unexpected commits by tolerance.
        protected = []
        for state in (reference, candidate):
            group = 0 if flags[0] or flags[1] else 1 if flags[2] or flags[3] else None
            if group is not None:
                groups = (0, 1) if group == 0 else (1,)
                values = [v for g in groups for v in protected_values(state, g)]
                protected.append((values, [numeric_snapshot(v) for v in values]))
        progress['stage'] = 'eager_reference'
        expected_error = reference.eager(x, optional, mode, flags)
        error, key = executor.step(x, optional, mode, flags)
        assert error == expected_error, (error, expected_error)
        progress['stage'] = 'comparison'
        for values, saved in protected:
            assert_state_values(values, saved)
        maximum = compare_states(candidate, reference)
        evidence['steps'].append(dict(index=index, error=error, max_abs_error=maximum,
            key=list(key), reused_graph=key in old_keys))
        old_keys.add(key)
    assert any(row['reused_graph'] and row['index'] > 7 for row in evidence['steps']), 'Old graph was not replayed after new shape capture.'
    evidence.update(passed=True, graph_count=len(executor.graphs), graph_limit=executor.max_graphs,
        capture_seconds=executor.capture_seconds,
        gradient_bank_bytes=sum(g.numel()*g.element_size() for group in executor.banks for bank in group for g in bank),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        normal_gradient_copies=0, terminal_status_reads=len(sequence))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Refuse nonempty output directory.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, completed=False, models={}, location={},
        scope='synthetic gradient-bank feasibility only; no real TD3 or performance conclusion',
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        costs='two gradient banks; up to 16 lazy graphs/model (48 theoretical keys); one terminal status read/update; capture-only snapshots',
        limitation='failed loss may execute unused backward in private bank; public gradients/Adam/targets/counts compared; full TD3 and arbitrary external rebinding not supported')
    try:
        report['location']['stage'] = 'capability'
        require_capability()
        report['gpu'] = torch.cuda.get_device_name()
        for model in ('ann', 'snn'):
            torch.cuda.reset_peak_memory_stats()
            evidence = report['models'][model] = {}
            run_model(model, report['location'], evidence)
        report.update(passed=True, completed=True)
    except BaseException as exc:
        report['failure'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
