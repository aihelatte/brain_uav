"""Bounded current-stream Event spans; not pure GPU compute or speed acceptance."""
from time import perf_counter
import math
from pathlib import Path
import torch

from brain_uav.scripts.diagnose_v2_policy import _write_json


class GPUComputeEventDiagnostic:
    def __init__(self, *, updates, output_dir, worker_id, model, scene_ids):
        if not 2 <= updates <= 32: raise ValueError('Event sample limit must be 2..32.')
        self.limit = updates
        self.output = Path(output_dir)/'other'/'gpu_compute_event_samples.json'
        self.worker_id, self.model, self.scene_ids = worker_id, model, set(scene_ids)
        self.events = None
        self.context = None
        self.window = None
        self.pending = None
        self.rows = []
        self.seen = set()
        self.errors = []
        self.closed = False
        self.failed = False

    def prime(self, device):
        if self.events is not None: raise RuntimeError('Events already primed.')
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        stream = torch.cuda.current_stream(device)
        for event in self.events: event.record(stream)
        # Caller performs the EXISTING warmup synchronization, not a new wait.

    def arm(self, window):
        if self.events is None or not all(e.query() for e in self.events):
            raise RuntimeError('Event warmup not complete at existing synchronization.')
        self.window = window

    def set_context(self, scenario_id):
        self.context = scenario_id

    def begin(self, *, device, mode, total_steps, batch, update_started):
        now = perf_counter()
        if (self.closed or self.errors or self.window is None or self.context is None
                or self.pending is not None or len(self.rows) >= self.limit
                or not self.window['start'] <= now < self.window['end']):
            return False
        fraction = (now-self.window['start'])/(self.window['end']-self.window['start'])
        keys = {('scene',self.context,mode), ('time',min(7,int(fraction*8)),mode)}
        if keys <= self.seen: return False
        self.seen.update(keys)
        self.pending = dict(worker=self.worker_id,model=self.model,mode=mode,total_steps=total_steps,
            scenario_id=self.context, current_shape=list(batch.obs.zone_features.shape),
            next_shape=list(batch.next_obs.zone_features.shape),
            update_started=update_started,sample_started=now,valid=False)
        self.stream = torch.cuda.current_stream(device)
        self.events[0].record(self.stream)
        return True

    def mark(self, index):
        self.events[index].record(self.stream)

    def complete(self, cpu_times):
        row = self.pending
        row['sample_completed'] = perf_counter()
        row['cpu_wall_seconds'] = dict(zip(('coefficients','graph_replay_call','result_completion'),cpu_times))
        ready = all(e.query() for e in self.events)
        row['events_ready'] = ready
        if ready:
            spans = [self.events[0].elapsed_time(self.events[1]),self.events[1].elapsed_time(self.events[2])]
            if not all(math.isfinite(x) and x >= 0 for x in spans):
                raise ValueError('Invalid CUDA Event span.')
            row['gpu_timeline_span_ms'] = dict(fixed_input=spans[0],graph_replay=spans[1])
        else:
            row['invalid_reason'] = 'Events not ready after original result barrier; no extra wait.'
            self.errors.append(row['invalid_reason'])  # Do not overwrite still-in-flight events.
        row['within_window'] = (self.window['start'] <= row['sample_started']
                                and row['sample_completed'] <= self.window['end'])
        row['valid'] = ready and row['within_window']
        if not row['within_window']: row['invalid_reason'] = 'Sample crosses common deadline.'
        self.rows.append(row)
        self.pending = None

    def step_completed(self, in_window):
        # Same whole-step completion predicate as throughput and scene coverage.
        if self.rows and 'step_completed_in_window' not in self.rows[-1]:
            row = self.rows[-1]
            row['step_completed_in_window'] = in_window
            if not in_window:
                row.update(valid=False,invalid_reason='Whole-step tail outside common window.')

    def fail(self, exc):
        self.failed = True
        self.errors.append(f'Update aborted: {type(exc).__name__}: {exc}')
        if self.pending is not None:
            self.pending.update(valid=False,invalid_reason=self.errors[-1])
            self.rows.append(self.pending)
            self.pending = None
        elif self.rows and 'step_completed_in_window' not in self.rows[-1]:
            self.rows[-1].update(valid=False,invalid_reason=self.errors[-1])
        # Errors prevent any subsequent reuse; do not query/wait on error paths.

    def metadata(self):
        valid = [r for r in self.rows if r['valid'] and r.get('step_completed_in_window',False)]
        by_mode = {}
        for mode in sorted({r['mode'] for r in valid}):
            samples = [r for r in valid if r['mode'] == mode]
            by_mode[mode] = dict(count=len(samples),
                mean_gpu_timeline_span_ms={key:sum(r['gpu_timeline_span_ms'][key] for r in samples)/len(samples)
                    for key in ('fixed_input','graph_replay')},
                mean_cpu_wall_seconds={key:sum(r['cpu_wall_seconds'][key] for r in samples)/len(samples)
                    for key in ('coefficients','graph_replay_call','result_completion')})
        reasons = list(self.errors)
        if not {'actor','critic_only'} <= {r['mode'] for r in valid}:
            reasons.append('Event samples do not cover actor and critic_only.')
        if not self.scene_ids <= {r['scenario_id'] for r in valid}:
            reasons.append('Valid Event samples do not cover all selected scenes.')
        return dict(requested=True,effective=bool(self.rows),instrumented=True,
            valid_for_speed_comparison=False,diagnostic_valid=not reasons,invalid_reasons=reasons,
            sample_limit=self.limit,sampled_updates=len(self.rows),valid_samples=len(valid),
            by_mode=by_mode,
            event_count=3 if self.events is not None else 0,detail=str(self.output),
            meaning='GPU timeline spans may include CPU submission gaps/scheduling/contention; exclude queue before E0 executes. '
                'Not pure compute, not subtractable from CPU wall or additive to overlapping parents; no process/lock attribution.')

    def finish(self):
        if not self.closed:
            self.closed = True
            if self.pending is not None: self.fail(RuntimeError('Unfinished Event sample.'))
            self.output.parent.mkdir(parents=True,exist_ok=True)
            _write_json(self.output,dict(summary=self.metadata(),samples=self.rows))
        return self.metadata()
