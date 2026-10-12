"""Opt-in CUDA Graph for the checked critic commit tail (no backward/checks)."""
from time import perf_counter
import torch


class CriticUpdateTail:
    def __init__(self, engine):
        self.engine = engine
        self.graphs = {}
        self.buffers = ()
        self.signature = None
        self.capture_seconds = 0.0
        self.replays = 0
        self.reset_internal_timing()

    def reset_internal_timing(self, *, enabled=False):
        """Diagnostic-only measurement window; capture/warmup remain separate."""
        if type(enabled) is not bool:
            raise TypeError('enabled must be a bool.')
        self.internal_timing_enabled = enabled
        self._timing_calls = 0
        self._timing_seconds = dict.fromkeys(
            ('checks', 'gradient_copy_submission', 'replay_submission', 'total'), 0.0)

    def internal_timing_report(self):
        values = dict(self._timing_seconds)
        values['uncovered'] = values['total'] - sum(
            values[key] for key in ('checks', 'gradient_copy_submission', 'replay_submission'))
        return {
            'enabled': self.internal_timing_enabled,
            'calls': self._timing_calls,
            'total_seconds': values,
            'average_seconds': {key: value / self._timing_calls if self._timing_calls else None
                                for key, value in values.items()},
            'capture_and_warmup_seconds': self.capture_seconds,
            'note': 'CPU wall intervals may include submission and waits, not pure GPU compute. '
                    'Internal instrumentation has overhead; not a final speed acceptance run. '
                    'Capture calls are excluded; counters reset at measurement start.',
        }

    def _signature(self):
        e = self.engine
        optimizer = e.critic_optimizer
        models = (e.critic1, e.critic2, e.critic1_target, e.critic2_target)
        parameters = tuple(p for model in models for p in model.parameters())
        group_signature = tuple(tuple((k, repr(v)) for k, v in g.items() if k != 'params')
                                for g in optimizer.param_groups)
        state = tuple((id(p), tuple((k, id(v), v.data_ptr(), v.device, v.dtype, tuple(v.shape))
                     for k, v in optimizer.state.get(p, {}).items()))
                     for g in optimizer.param_groups for p in g['params'])
        return (id(optimizer), tuple((id(p), p.data_ptr(), p.device, p.dtype, tuple(p.shape))
                for p in parameters), tuple(tuple(id(p) for p in g['params'])
                for g in optimizer.param_groups), group_signature, state, e.tau)

    def _operations(self, targets):
        e = self.engine
        e.critic_optimizer.step()
        if targets:
            e._soft_update(e.critic1, e.critic1_target)
            e._soft_update(e.critic2, e.critic2_target)

    def _capture(self, parameters):
        e = self.engine
        optimizer = e.critic_optimizer
        if e.device.type != 'cuda' or not all(g.get('fused') and g.get('capturable') for g in optimizer.param_groups):
            raise ValueError('Critic tail capture requires CUDA fused Adam with capturable=True.')
        original_grads = tuple(p.grad for p in parameters)
        self.buffers = tuple(None if g is None else g.detach().clone() for g in original_grads)
        tensors = tuple(p for model in (e.critic1, e.critic2, e.critic1_target, e.critic2_target)
                        for p in model.parameters())
        saved_parameters = tuple(p.detach().clone() for p in tensors)
        original_states = {p: {k: v.clone() for k, v in s.items()}
                           for p, s in optimizer.state.items()}
        saved_states = None
        start = perf_counter()
        stream = torch.cuda.Stream(device=e.device)
        stream.wait_stream(torch.cuda.current_stream(e.device))

        def restore():
            with torch.no_grad():
                for p, value in zip(tensors, saved_parameters):
                    p.copy_(value)
                for p, state in saved_states.items():
                    for k, value in state.items():
                        optimizer.state[p][k].copy_(value)

        try:
            with torch.cuda.stream(stream):
                for p, g in zip(parameters, self.buffers):
                    p.grad = g
                # Initialize lazy fused-Adam state, then restore its initial value in place.
                self._operations(False)
                for p, state in optimizer.state.items():
                    for k, value in state.items():
                        if p in original_states:
                            value.copy_(original_states[p][k])
                        else:
                            value.zero_()
                saved_states = {p: {k: v.clone() for k, v in s.items()}
                                for p, s in optimizer.state.items()}
                restore()
                for targets in (False, True):
                    for _ in range(3):
                        self._operations(targets)
                    restore()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        self._operations(targets)
                    restore()
                    self.graphs[targets] = graph
            torch.cuda.current_stream(e.device).wait_stream(stream)
            torch.cuda.synchronize(e.device)
            self.signature = self._signature()
        except Exception:
            # Capture failure must not commit even the lazy-state warmup.
            torch.cuda.current_stream(e.device).wait_stream(stream)
            with torch.no_grad():
                for p, value in zip(tensors, saved_parameters):
                    p.copy_(value)
                for p in list(optimizer.state):
                    if p not in original_states:
                        del optimizer.state[p]
                    else:
                        for k, value in original_states[p].items():
                            optimizer.state[p][k].copy_(value)
            self.graphs.clear()
            self.signature = None
            raise
        finally:
            for p, g in zip(parameters, original_grads):
                p.grad = g
            self.capture_seconds += perf_counter() - start

    def step(self, targets):
        # A first-call capture is not a steady-state sample. No new clocks when disabled.
        timed = self.internal_timing_enabled and bool(self.graphs)
        started = perf_counter() if timed else None
        e = self.engine
        parameters = tuple(p for g in e.critic_optimizer.param_groups for p in g['params'])
        if not self.graphs:
            self._capture(parameters)
        if self.signature != self._signature():
            raise RuntimeError('Critic tail addresses/configuration changed; invalidate before recapture.')
        if any((p.grad is None) != (g is None) for p, g in zip(parameters, self.buffers)):
            raise RuntimeError('Critic tail gradient presence changed before parameter update.')
        with torch.no_grad():
            for p, g in zip(parameters, self.buffers):
                if g is not None:
                    if p.grad.shape != g.shape or p.grad.device != g.device or p.grad.dtype != g.dtype:
                        raise RuntimeError('Critic tail gradient layout changed before parameter update.')
            if timed:
                checked = perf_counter()
            for p, g in zip(parameters, self.buffers):
                if g is not None:
                    g.copy_(p.grad)
            if timed:
                copied = perf_counter()
        # Copies and replay use the current stream, with FIFO dependency.
        if timed:
            replay_started = perf_counter()
        self.graphs[targets].replay()
        self.replays += 1
        if timed:
            finished = perf_counter()
            self._timing_calls += 1
            self._timing_seconds['checks'] += checked - started
            self._timing_seconds['gradient_copy_submission'] += copied - checked
            self._timing_seconds['replay_submission'] += finished - replay_started
            self._timing_seconds['total'] += finished - started
