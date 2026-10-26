"""Small isolated A/B driver around existing formal correctness/timing workers."""
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
import json
import subprocess
import sys
import numpy as np
import torch
from contextlib import nullcontext
from brain_uav.observations import V2ObservationBatch
from brain_uav.trainers.v2_replay_buffer import V2ReplayBatch
from brain_uav.observations import V2Observation
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer


class BatchPreparationDiagnostic:
    """Bounded, opt-in observation of the existing blocking transfer sequence.

    No tensor contents or tensor references survive an update. Wall intervals
    include allocation/copy/waits; only the optional trace can separate visible
    runtime activity. This is never a throughput acceptance measurement.
    """
    def __init__(self, *, updates, trace, output_dir, worker_id, model):
        if not 2 <= updates <= (8 if trace else 64):
            raise ValueError('Batch diagnosis needs 2..64 updates (trace: 2..8).')
        self.limit, self.trace = updates, trace
        self.output = Path(output_dir)
        self.worker_id, self.model = worker_id, model
        self.records, self.error = [], None
        self.profiler = None
        self.trace_stopped = False
        self.current = None

    def _start_trace(self):
        activities = torch.profiler.supported_activities()
        if torch.profiler.ProfilerActivity.CUDA not in activities:
            raise RuntimeError('Batch trace unavailable: torch.profiler has no CUDA activity support.')
        self.activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        profiler = torch.profiler.profile(activities=self.activities,
            record_shapes=False, profile_memory=True, with_stack=False)
        profiler.__enter__()
        self.profiler = profiler

    def _stop_trace(self):
        if self.profiler is not None and not self.trace_stopped:
            # Mark before stopping: neither finish nor error cleanup retries exit.
            self.trace_stopped = True
            try:
                self.profiler.__exit__(None, None, None)
            except Exception as exc:
                self.error = {'type': type(exc).__name__, 'message': 'Trace stop: ' + str(exc)}
                raise

    def run(self, operation, *, update_index, window=None):
        if len(self.records) >= self.limit:
            return operation(None)
        started = perf_counter()
        if window is not None and not window['start'] <= started < window['end']:
            self._stop_trace()
            return operation(None)
        record = {'worker': self.worker_id, 'model': self.model, 'update_index': update_index,
                  'sample_index': len(self.records), 'tensor_calls': [], 'constructors': [],
                  'sample_started': started}
        self.current = record
        try:
            if self.trace:
                if self.profiler is None:
                    self._start_trace()
                else:
                    self.profiler.toggle_collection_dynamic(True, self.activities)
            with (torch.profiler.record_function(f'batch_diagnostic.update.{update_index}')
                  if self.trace else nullcontext()):
                result = operation(self.transfer)
            if 'batch_preparation_wall_seconds' not in record:
                raise RuntimeError('Batch diagnostic callback was not called by GPU compute.')
            record['actor_updated'] = bool(result.actor_updated)
            record['sample_completed'] = perf_counter()
            record['within_common_window'] = (window is None or
                window['start'] <= started < window['end'] and record['sample_completed'] <= window['end'])
            self.records.append(record)
            return result
        except Exception as exc:
            self.error = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            self.current = None
            if self.profiler is not None:
                if (self.error or len(self.records) >= self.limit
                        or self.records and not self.records[-1]['within_common_window']):
                    self._stop_trace()
                else:
                    try:
                        self.profiler.toggle_collection_dynamic(False, self.activities)
                        self.profiler.step()
                    except Exception as exc:
                        self.error = {'type': type(exc).__name__, 'message': str(exc)}
                        self._stop_trace()
                        raise

    def _call(self, name, operation, *, tensor=None):
        # Metadata/scopes are outside the individual call interval, inside total.
        metadata = {'name': name}
        if tensor is not None:
            metadata.update(dtype=str(tensor.dtype), shape=list(tensor.shape),
                stride=list(tensor.stride()), bytes=tensor.numel() * tensor.element_size(),
                source_device=str(tensor.device), pinned=tensor.is_pinned(), layout=str(tensor.layout))
        with (torch.profiler.record_function('batch_preparation.' + name)
              if self.trace else nullcontext()):
            start = perf_counter()
            value = operation()
            metadata['cpu_wall_seconds'] = perf_counter() - start
        self.current['tensor_calls' if tensor is not None else 'constructors'].append(metadata)
        return value

    def transfer(self, batch, device):
        """Same evaluation order as V2ReplayBatch.to; constructors timed alone."""
        start = perf_counter()
        target = torch.device(device)
        self.current.update(target_device=str(target), non_blocking=False,
            obs_shape=list(batch.obs.zone_features.shape),
            next_obs_shape=list(batch.next_obs.zone_features.shape))

        def move(name, tensor):
            return self._call(name, lambda: tensor.to(target, non_blocking=False), tensor=tensor)

        def observation(prefix, source):
            # Evaluate all .to arguments BEFORE starting the constructor timer.
            values = {name: move(prefix + '.' + name, getattr(source, name))
                for name in ('ego_features', 'goal_features', 'zone_features', 'presence_mask')}
            return self._call(prefix + '.V2ObservationBatch', lambda: V2ObservationBatch(**values))

        obs = observation('obs', batch.obs)
        action = move('action', batch.action)
        reward = move('reward', batch.reward)
        next_obs = observation('next_obs', batch.next_obs)
        remaining = {name: move(name, getattr(batch, name))
            for name in ('done', 'success', 'near_goal', 'line_to_goal_safe')}
        result = self._call('V2ReplayBatch', lambda: V2ReplayBatch(
            obs=obs, action=action, reward=reward, next_obs=next_obs,
            **remaining, sample_failure_fraction=batch.sample_failure_fraction))
        total = perf_counter() - start
        covered = sum(c['cpu_wall_seconds'] for key in ('tensor_calls', 'constructors')
                      for c in self.current[key])
        self.current.update(batch_preparation_wall_seconds=total,
            total_source='transfer_callback',
            uncovered_wall_seconds=total - covered,
            bytes=sum(c['bytes'] for c in self.current['tensor_calls']))
        return result

    def record_outer_total(self, sections):
        if self.current is not None and 'batch_preparation_wall_seconds' in self.current:
            # The existing engine timer also includes our final bookkeeping.
            total = sections['batch_preparation']
            covered = sum(c['cpu_wall_seconds'] for key in ('tensor_calls', 'constructors')
                          for c in self.current[key])
            self.current.update(batch_preparation_wall_seconds=total,
                total_source='existing_engine_batch_preparation_timer', uncovered_wall_seconds=total - covered)

    def finish(self):
        """Called after the common window, also on failure; no per-step file I/O."""
        from brain_uav.scripts.v2_concurrent_diagnostic import write_json
        other, review = self.output / 'other', self.output / 'review'
        other.mkdir(parents=True, exist_ok=True)
        review.mkdir(parents=True, exist_ok=True)
        trace_evidence = None
        if self.profiler is not None:
            try:
                self._stop_trace()
                path = other / 'batch_preparation_trace.json'
                self.profiler.export_chrome_trace(str(path))
                trace_evidence = self._trace_evidence(path)
            except Exception as exc:
                self.error = {'type': type(exc).__name__, 'message': 'Trace finalization: ' + str(exc)}
            finally:
                self.profiler = None
        reasons = []
        if self.error:
            reasons.append(self.error['message'])
        valid_records = [r for r in self.records if r['within_common_window']]
        if len(valid_records) != self.limit:
            reasons.append('Bounded batch sample incomplete within common measurement window.')
        roles = {r['actor_updated'] for r in valid_records}
        if roles != {False, True}:
            reasons.append('Batch samples must include critic-only and actor updates.')
        if self.trace and (trace_evidence is None or not trace_evidence['cuda_activity_observed']):
            reasons.append('CUDA trace activity unavailable; copy/allocation/wait attribution unknown.')
        names = [c['name'] for r in self.records for c in r['tensor_calls'] + r['constructors']]
        from statistics import mean
        summary = {'instrumented': True, 'mode': 'trace' if self.trace else 'tensor_wall',
            'diagnostic_valid': not reasons, 'invalid_reasons': reasons,
            'valid_for_speed_comparison': False, 'error': self.error,
            'requested_updates': self.limit, 'captured_updates': len(self.records),
            'valid_samples_in_window': len(valid_records),
            'worker': self.worker_id, 'model': self.model,
            'actor_updates': sum(r['actor_updated'] for r in self.records),
            'calls': {name: {'count': names.count(name), 'mean_cpu_wall_seconds': mean(
                c['cpu_wall_seconds'] for r in self.records
                for c in r['tensor_calls'] + r['constructors'] if c['name'] == name)}
                for name in dict.fromkeys(names)},
            'mean_batch_preparation_wall_seconds': (mean(r['batch_preparation_wall_seconds'] for r in self.records)
                if self.records else None),
            'mean_uncovered_wall_seconds': (mean(r['uncovered_wall_seconds'] for r in self.records)
                if self.records else None),
            'trace': trace_evidence, 'records': 'other/batch_preparation_records.json',
            'note': 'Instrumented CPU wall includes allocation/copy/submission/waits. Constructors exclude child .to calls. '
                'Residual includes metadata, scopes and timer overhead; parent/child times are not additive. '
                'No added CUDA synchronization or tensor-content readback. Driver queueing and other GPU processes are unknown. '
                'Trace/profiler may synchronize and perturb all concurrent workers; never a pure throughput result.'}
        write_json(other / 'batch_preparation_records.json', {'updates': self.records, 'error': self.error})
        write_json(review / 'batch_preparation.json', summary)
        return summary

    @staticmethod
    def _trace_evidence(path):
        events = json.loads(path.read_text(encoding='utf-8'))['traceEvents']
        copies = [e for e in events if 'gpu_memcpy' in e.get('cat', '').lower()]
        kernels = [e for e in events if e.get('cat') == 'kernel']
        htod = [e for e in copies if 'htod' in e.get('name', '').lower()]
        byte_values = [e.get('args', {}).get('bytes') for e in copies]
        htod_bytes = [e.get('args', {}).get('bytes') for e in htod]
        return {'path': str(path), 'cuda_activity_observed': bool(copies or kernels),
            'gpu_copy_events': len(copies) if copies else None,
            'htod_copy_events': len(htod) if htod else None,
            'copy_event_note': 'null means no copy activity observed; transfer attribution is unknown, not zero.',
            'copy_bytes': sum(byte_values) if copies and all(isinstance(n, (int, float)) for n in byte_values) else None,
            'htod_bytes': sum(htod_bytes) if htod and all(isinstance(n, (int, float)) for n in htod_bytes) else None,
            'copy_bytes_note': 'Visible trace events only; null means bytes unavailable, not zero transfer.',
            'runtime_calls': {name: {'count': sum(e.get('name') == name for e in events),
                'cpu_wall_seconds': sum(e.get('dur', 0) for e in events if e.get('name') == name) / 1e6}
                for name in sorted({e['name'] for e in events if e.get('cat') == 'cuda_runtime'})},
            'memory_events_observed': sum(e.get('name') == '[memory]' for e in events) or None,
            'runtime_memory_note': 'Observed profiler events only; allocator reuse, synchronization ownership and unseen activity require timeline inspection.',
            'driver_queueing': 'unknown', 'other_gpu_process_activity': 'unknown'}


def cpu_batches(mixed, nonempty, action_dim, seed):
    """Actual CPU replay sampling, including empty and dynamically cropped batches."""
    result=[]
    for case,observation in enumerate((mixed,nonempty,mixed)):
        observation=observation.to('cpu')
        b=observation.batch_size
        replay=V2ReplayBuffer(b,action_dim,observation.max_zone_count,failure_sample_bias=2.,seed=seed+case)
        for i in range(b):
            mask=observation.presence_mask[i].numpy()
            zones=observation.zone_features[i].numpy()[mask] if case else observation.zone_features[i].numpy()[:0]
            obs=V2Observation(observation.ego_features[i].numpy(),observation.goal_features[i].numpy(),
                zones,np.ones(len(zones),np.bool_))
            replay.add(obs,np.zeros(action_dim,np.float32),float(i+case*b),obs,False,near_goal=i%2==0)
        replay.mark_failure_slots([(i,int(replay.write_id[i])) for i in range(0,b,3)])
        result.append(replay.sample(b))
    return result


def fields(batch):
    result={}
    for prefix in ('obs','next_obs'):
        obs=getattr(batch,prefix)
        for name in ('ego_features','goal_features','zone_features','presence_mask'):
            result[prefix+'.'+name]=getattr(obs,name)
    for name in ('action','reward','done','success','near_goal','line_to_goal_safe'):
        result[name]=getattr(batch,name)
    return result


def check_transfers(batches,device,repeats):
    """Completion wall and submission wall are distinct; no network timing here."""
    if torch.device(device).type!='cuda': raise ValueError('Transfer validation requires real CUDA.')
    if repeats<1: raise ValueError('Transfer repeat count must be positive.')
    report={'scope':'pageable CPU batch -> CUDA, completion includes one end-window synchronization',
        'groups':{},'exact_fields':True,'consecutive_batches_checked':len(batches)}
    for enabled in (False,True):
        runs=[]
        # Untimed preparation avoids attributing initial device allocation to steady copies.
        warm=[b.to(device,non_blocking=enabled) for b in batches]
        torch.cuda.synchronize(device); del warm
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start=perf_counter()
            outputs=[b.to(device,non_blocking=enabled) for b in batches]
            submitted=perf_counter()-start
            torch.cuda.synchronize(device); completed=perf_counter()-start
            for source,actual in zip(batches,outputs):
                if source.sample_failure_fraction!=actual.sample_failure_fraction:
                    raise AssertionError('sample_failure_fraction changed during transfer.')
                for name,value in fields(source).items():
                    got=fields(actual)[name]
                    target=torch.device(device)
                    target=torch.device('cuda',torch.cuda.current_device()) if target.index is None else target
                    if value.shape!=got.shape or value.dtype!=got.dtype or got.device!=target:
                        raise AssertionError('Transfer metadata changed: '+name)
                    torch.testing.assert_close(value,got.cpu(),rtol=0,atol=0)
            runs.append({'batch_count':len(batches),'submission_seconds':submitted,'completion_seconds':completed})
        report['groups']['nonblocking' if enabled else 'baseline']={'requested':enabled,'effective':enabled,'runs':runs}
    return report


def run(args):
    from brain_uav.scripts import profile_v2_td3 as p
    conflicts=('gpu_td3_update','gpu_td3_suite','gpu_td3_worker','gpu_td3_sampling_cost',

        'window_update_statistics','reuse_update_buffers',

        'local_optimization_correctness','local_optimization_comparison',
        'snn_path_consistency','detailed_profiler_updates','compiled_performance_diagnostic_updates')
    if any(getattr(args,n) for n in conflicts) or args.device!='cuda' or args.comparison_eager_cpu:
        raise ValueError('Batch transfer comparison requires CUDA and original formal path, without other experiments/profiler.')
    if args.bc_checkpoint is None or args.validation_pool is None:
        raise ValueError('Matching BC and easy fixed pool are required.')
    if args.batch_transfer_worker:
        args.correctness_updates=max(8,args.correctness_updates)
        return p.run_local_optimization_comparison(args)
    if args.non_blocking_batch_transfer:
        raise ValueError('The serial driver owns baseline/nonblocking selection.')
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Refusing nonempty comparison directory.')
    output.mkdir(parents=True,exist_ok=True)
    report={'diagnostic_only':True,'passed':False,'model':args.model,'groups':{}}
    report['timing_scope']='batch_preparation is CPU submission/allocation/validation wall, not transfer completion. Independent transfer check synchronizes matching window boundaries. TD3 totals below include the original timing window completion wait; full environment totals already include it.'
    forwarded=('model','bc_checkpoint','validation_pool','validation_seed','seed','batch_size','replay_capacity',
        'scenario_count','warmup_steps','steps_per_level','repeats','controlled_stage_step_offset',
        'comparison_bc_schedule','snn_time_window','correctness_updates')
    baseline_ok=False
    try:
        # Both correctness processes complete before either throughput worker starts.
        for enabled in (False,True):
            group='nonblocking' if enabled else 'baseline'
            item=report['groups'][group]={'requested':enabled,'correctness_passed':False}
            if enabled and not baseline_ok: item['skipped']='baseline correctness failed'; continue
            command=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--device','cuda']
            for name in forwarded: command+=['--'+name.replace('_','-'),str(getattr(args,name))]
            if enabled: command+=['--non-blocking-batch-transfer','--td3-block-reference',str(output/'baseline_correctness'/'reference.pt')]
            with (output/(group+'_correctness.log')).open('w',encoding='utf-8') as log:
                rc=subprocess.run(command+['--batch-transfer-worker','correctness','--output-dir',str(output/(group+'_correctness'))],
                    stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            item['correctness_exit_code']=rc; item['correctness_passed']=rc==0
            if not enabled: baseline_ok=rc==0
        if not all(v['correctness_passed'] for v in report['groups'].values()):
            raise RuntimeError('Transfer/update correctness failed; this model timing skipped.')
        for enabled in (False,True):
            group='nonblocking' if enabled else 'baseline'
            command=[sys.executable,'-u','-m','brain_uav.scripts.profile_v2_td3','--device','cuda']
            for name in forwarded: command+=['--'+name.replace('_','-'),str(getattr(args,name))]
            if enabled: command+=['--non-blocking-batch-transfer']
            with (output/(group+'_timing.log')).open('w',encoding='utf-8') as log:
                rc=subprocess.run(command+['--batch-transfer-worker','timing','--output-dir',str(output/(group+'_timing'))],
                    stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            report['groups'][group]['timing_exit_code']=rc
            if rc: raise RuntimeError('Timing worker failed: '+group)
            data=json.loads((output/(group+'_timing')/'local_optimization_summary.json').read_text())
            if not data['valid_for_speed_comparison']: raise RuntimeError('Measurement compiled new graphs: '+group)
            report['groups'][group]['effective']=data['runs'][group][0]['local_optimizations']['non_blocking_batch_transfer']
            runs=data['runs'][group]
            report['groups'][group]['raw_runs']=[dict(environment_seconds=r['timing']['total_wall_seconds'],
                td3_seconds=r['timing']['td3_update_wall_seconds'],
                completion=r['batch_transfer_window_completion'],
                batch_submission=r['timing']['td3_update_breakdown']['overall_weighted']['sections']['batch_preparation'],
                measured_steps=r['measured_steps'],compilation=r['critic_encoder_compile']) for r in runs]
            report['groups'][group]['mean_step_seconds']=mean(r['timing']['total_wall_seconds']/r['measured_steps'] for r in runs)
            report['groups'][group]['stdev_step_seconds']=stdev(r['timing']['total_wall_seconds']/r['measured_steps'] for r in runs) if len(runs)>1 else 0.
            report['groups'][group]['mean_td3_seconds']=mean(r['batch_transfer_window_completion']['td3_wall_including_final_window_wait_seconds'] for r in runs)
        a,b=report['groups']['baseline'],report['groups']['nonblocking']
        report['time_saved_percent']=100*(1-b['mean_step_seconds']/a['mean_step_seconds'])
        report['overall_speedup_observed']=b['mean_step_seconds']<a['mean_step_seconds'] and b['mean_td3_seconds']<a['mean_td3_seconds']
        report['verdict']='short comparison improved both totals; stability/adoption not established' if report['overall_speedup_observed'] else '未取得整体加速'
        report['passed']=True
    finally: p._strict_json_write(output/'summary.json',report)
    return report
