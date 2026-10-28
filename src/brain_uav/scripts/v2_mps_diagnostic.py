"""Plan-only MPS capability + ordinary/MPS eight-worker diagnostic. No training changes."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from brain_uav.scripts import v2_concurrent_diagnostic as concurrent

CONTROL = '/usr/bin/nvidia-cuda-mps-control'
SERVER = '/usr/bin/nvidia-cuda-mps-server'


def command(argv, *, env=None, text=None):
    result = subprocess.run(argv, input=text, env=env, capture_output=True, text=True, timeout=30)
    if result.returncode or re.search(r'\b(error|failed|invalid|unknown command|not found)\b',
                                     result.stdout+'\n'+result.stderr, re.I):
        raise RuntimeError(f'Command failed: {argv}: rc={result.returncode}: {result.stdout} {result.stderr}')
    return result.stdout.strip()


def pid_list(text):
    if not text: return set()
    if any(not line.strip().isdigit() for line in text.splitlines()):
        raise RuntimeError(f'Unrecognized MPS PID list: {text!r}')
    return {int(line.strip()) for line in text.splitlines()}


def mps_processes():
    text = command(['ps','-eo','pid=,comm='])
    return [line.strip() for line in text.splitlines()
            if len(line.split()) == 2 and line.split()[1].startswith('nvidia-cuda-mps')]


def gpu_processes():
    text = command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    return pid_list(text)


class Session:
    """One private daemon namespace; refuses ambiguous ownership or unsafe kill."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=False)
        # Short Unix socket path, private permissions; preserve directory for audit.
        self.pipe = Path(tempfile.mkdtemp(prefix='uav_mps_'))
        self.env = dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=str(self.pipe),
                        CUDA_MPS_LOG_DIRECTORY=str(self.root/'logs'))
        (self.root/'logs').mkdir()
        self.started = False
        self.owned = {}
        self.actions = []
        self.daemon_pid = None

    def control(self, text):
        output = command([CONTROL],env=self.env,text=text+'\n')
        self.actions.append(dict(command=text,response=output))
        return output

    def start(self):
        if mps_processes(): raise RuntimeError('MPS already running; will not manage external services.')
        if not Path(CONTROL).is_file() or not Path(SERVER).is_file(): raise RuntimeError('MPS binaries unavailable.')
        command([CONTROL,'-d'],env=self.env)
        self.started = True
        self.daemon_pid = int((self.pipe/'nvidia-cuda-mps-control.pid').read_text().strip())
        # Legacy drivers need not implement interactive help. Exercise the interface.
        pid_list(self.control('get_server_list'))

    def clients(self):
        result = {}
        for server in sorted(pid_list(self.control('get_server_list'))):
            for client in pid_list(self.control(f'get_client_list {server}')):
                if client in result: raise RuntimeError('Ambiguous MPS client ownership.')
                result[client] = server
        return result

    def register(self, processes):
        for process in processes.values(): self.owned[process.pid] = process

    def verify(self, processes):
        self.register(processes)
        clients = self.clients()
        expected = {p.pid for p in processes.values()}
        if set(clients) != expected:
            raise RuntimeError(f'MPS actual clients {sorted(clients)} != owned ready PIDs {sorted(expected)}')
        return dict(requested='mps',effective='mps',daemon_pid=self.daemon_pid,
            pipe_directory=str(self.pipe),client_server_pid={str(k):v for k,v in clients.items()},
            verified_before_window=True,reason='Exact server-side get_client_list membership, not environment variables')

    def before_terminate(self, process):
        if process.pid not in self.owned: raise RuntimeError('Refusing foreign process cleanup.')
        clients = self.clients()
        if set(clients)-set(self.owned): raise RuntimeError('Foreign MPS client present; cleanup blocked.')
        if process.pid not in clients:
            raise RuntimeError('Live owned process not listed by MPS; CUDA context status unknown; no OS kill.')
        # Volta+ command completes GPU work/context termination before OS signals.
        response = self.control(f'terminate_client {clients[process.pid]} {process.pid}')
        # Legacy terminate_client returns a status, unlike the PID-list commands.
        if response != '0':
            raise RuntimeError(f'terminate_client did not confirm success (expected 0): {response!r}; no OS kill.')

    def finish(self):
        report = dict(pipe_directory=str(self.pipe),log_directory=str(self.root/'logs'),
                      daemon_pid=self.daemon_pid,stopped=False)
        try:
            if self.started:
                if any(p.poll() is None for p in self.owned.values()):
                    raise RuntimeError('Owned clients still alive; leave service intact, manual review required.')
                if self.clients(): raise RuntimeError('Clients still connected; do not stop service.')
                self.control('quit')
                report['stopped'] = True
        except Exception as exc: report['cleanup_error'] = str(exc)
        report['control_actions'] = self.actions
        concurrent.write_json(self.root/'session.json',report)
        return report


def ordinary_access(processes):
    if mps_processes(): raise RuntimeError('Unexpected MPS process during ordinary CUDA condition.')
    actual = gpu_processes()
    expected = {p.pid for p in processes.values()}
    if actual != expected: raise RuntimeError('Direct CUDA GPU PID evidence differs from ready worker PIDs.')
    return dict(requested='ordinary',effective='ordinary',compute_pids=sorted(actual),
                verified_before_window=True,reason='No MPS processes; exact nvidia-smi compute PID membership')


def compact_batch(result, root):
    """Keep existing detailed report on disk; review contains only measured evidence."""
    keep = ('concurrency','repeat','completed','diagnostic_valid','valid_for_speed_comparison',
            'instrumented','common_window','batch_steps_per_second','actual_concurrent_overlap',
            'cuda_access_evidence','exit_codes','cleanup_errors','error')
    compact = {key:result[key] for key in keep if key in result}
    compact['detail'] = str(Path(root)/'review'/'summary.json')
    compact['workers'] = {}
    for name,worker in result.get('workers',{}).items():
        compact['workers'][name] = dict(
            seconds_per_step_including_final_drain=worker['seconds_per_step_including_final_drain'],
            window_steps_per_second=worker['window_steps_per_second'],steps=worker['steps'],
            parent_wall_seconds=worker['parent_wall_seconds'],final_completion=worker['final_completion'],
            td3_sections=worker['children_within_td3']['update_breakdown']['overall_weighted'],
            cpu_lightweight_timing=worker['execution'].get('gpu_compute_lightweight_timing'),
            diagnostic_valid=worker['diagnostic_valid'],invalid_reasons=worker['invalid_reasons'])
    return compact


def stop_probe(process, session):
    if process.poll() is None:
        session.before_terminate(process)
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=10)
        return True
    return False


def capability(root, session, timeout=60, *, terminate_client=False):
    root = Path(root); root.mkdir()
    log_path = root/'probe.log'
    result = dict(passed=False,termination_probe=terminate_client)
    termination_attempted = False
    with log_path.open('w',encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable,'-u','-m','brain_uav.scripts.v2_mps_diagnostic','--probe',str(root),str(timeout)],
            cwd=Path(__file__).resolve().parents[3],env=session.env,stdout=log,stderr=subprocess.STDOUT)
        session.register({'probe':process})
        try:
            concurrent.wait_workers({'probe':process},lambda _: (root/'ready.json').is_file(),timeout=timeout)
            if process.poll() is not None: raise RuntimeError('Probe exited before membership verification.')
            result['access'] = session.verify({'probe':process})
            ready = concurrent.read_json(root/'ready.json')
            if ready != dict(pid=process.pid,graph_replays=8,synchronized=True):
                raise RuntimeError('Probe CUDA Graph evidence mismatch.')
            if terminate_client:
                # Separate synchronized, PID-verified child remains waiting for us.
                # Do not retry an ambiguous/failed termination in finally.
                termination_attempted = True
                if not stop_probe(process,session):
                    raise RuntimeError('Probe exited before terminate_client could be tested.')
                result['terminate_client_response'] = '0'
            else:
                concurrent.write_json(root/'release.json',dict(verified=True))
                rc = process.wait(timeout=timeout)
                result['exit_code'] = rc
                if rc: raise RuntimeError('MPS CUDA Graph capability failed.')
            result['passed'] = True
        except BaseException as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            try:
                if terminate_client:
                    if not termination_attempted:
                        raise RuntimeError('Termination probe not synchronized and verified; no OS kill; leave service intact.')
                else:
                    stop_probe(process,session)
            except Exception as exc:
                result.update(passed=False,cleanup_error=str(exc))
            result['exit_code'] = process.poll()
            concurrent.write_json(root/'summary.json',result)
    if not result['passed']: raise RuntimeError('MPS capability cleanup failed.')
    return result


def probe(root, timeout):
    # Only this explicitly launched child imports torch/creates a CUDA context.
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no fallback.')
    value = torch.ones(32,device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): value.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): value.add_(1)
    for _ in range(8): graph.replay()
    torch.cuda.synchronize()
    concurrent.write_json(root/'ready.json',dict(pid=os.getpid(),graph_replays=8,synchronized=True))
    deadline = time.monotonic()+timeout
    while not (root/'release.json').is_file():
        if time.monotonic() >= deadline: raise TimeoutError('Probe membership release timeout.')
        time.sleep(.05)
    torch.cuda.synchronize()


def build_parser():
    parser = concurrent.build_parser()
    parser.description = __doc__
    parser.add_argument('--capability-only',action='store_true',
        help='With --launch, run only isolated MPS Graph capability/membership check; no eight-worker batches.')
    return parser


def validate(args):
    if (args.concurrency != [8] or args.repeats != 2 or args.measurement_seconds != 30
            or not args.non_blocking_batch_transfer or args.gpu_compute_non_blocking_coefficients
            or args.gpu_compute_lightweight_timing or args.gpu_compute_event_updates
            or args.batch_preparation_updates or args.batch_preparation_trace_updates
            or args.gpu_compute_submission_trace_updates):
        raise ValueError('MPS driver requires 8 workers, 2 repeats, 30s, A-config nonblocking batch; instrumentation selected internally per phase.')
    if any(k.startswith('CUDA_MPS_') for k in os.environ):
        raise ValueError('Inherited CUDA_MPS_* overrides are not allowed in controlled comparison.')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--probe':
        probe(Path(argv[1]),float(argv[2])); return 0
    args = build_parser().parse_args(argv)
    validate(args)
    plan = concurrent.build_plan(args)
    plan['mps_comparison'] = dict(conditions=['ordinary','mps'],phases=['lightweight','throughput'],
        capability_first=True,termination_probe=True,control=CONTROL,server=SERVER,thread_environment='unchanged',
        cleanup='terminate_client server_pid client_pid before OS termination; fail closed',
        caveat='No runtime capability, client attachment or speed claim until explicit launch succeeds.')
    if not args.launch:
        print(json.dumps(dict(mode='plan_only',plan=plan),indent=2)); return 0
    root = args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=False)
    (root/'review').mkdir(); (root/'other').mkdir()
    concurrent.write_json(root/'review'/'plan.json',plan)
    report = dict(completed=False,capability_passed=False,batches=[],sessions=[])
    session = None
    try:
        if sys.platform != 'linux': raise RuntimeError('MPS requires Linux; no fallback.')
        if mps_processes() or gpu_processes(): raise RuntimeError('GPU/MPS already in use; refuse to disturb external tasks.')
        report['gpu'] = command(['nvidia-smi','--query-gpu=name,uuid,driver_version,compute_mode','--format=csv'])
        session = Session(root/'other'/'capability_mps')
        session.start()
        report['capability'] = capability(root/'other'/'capability',session)
        report['termination_capability'] = capability(root/'other'/'termination_capability',session,terminate_client=True)
        stopped = session.finish(); report['sessions'].append(stopped); session = None
        if not stopped['stopped']: raise RuntimeError('Capability service did not stop safely.')
        report['capability_passed'] = True
        if args.capability_only:
            report['completed'] = True
            return 0
        for phase in ('lightweight','throughput'):
            for condition in ('ordinary','mps'):
                if mps_processes() or gpu_processes(): raise RuntimeError('Unexpected live GPU/MPS workload between conditions.')
                environment = dict(os.environ)
                # A private, empty namespace prevents accidental default-pipe attach.
                environment['CUDA_MPS_PIPE_DIRECTORY'] = tempfile.mkdtemp(prefix='uav_direct_')
                if condition == 'mps':
                    session = Session(root/'other'/f'{phase}_mps')
                    session.start(); environment = session.env
                for repeat in range(2):
                    run_args = deepcopy(args)
                    run_args.gpu_compute_lightweight_timing = phase == 'lightweight'
                    batch_root = root/'other'/f'{phase}_{condition}_{repeat}'
                    # Register at creation, even when a worker fails before ready.
                    def access(processes): return session.verify(processes) if session else ordinary_access(processes)
                    result = concurrent.run_batch(run_args,8,repeat,batch_root,worker_env=environment,
                        access_check=access,before_terminate=session.before_terminate if session else None,
                        **({'register_process':session.register} if session else {}))
                    compact = compact_batch(result,batch_root)
                    compact.update(phase=phase,condition=condition)
                    report['batches'].append(compact)
                    if not result['diagnostic_valid'] or (phase=='throughput' and not result['valid_for_speed_comparison']):
                        raise RuntimeError('Batch invalid; stop remaining measurements, retain evidence.')
                if session:
                    stopped = session.finish(); report['sessions'].append(stopped); session = None
                    if not stopped['stopped']: raise RuntimeError('MPS service cleanup incomplete.')
        report['completed'] = True
    except (Exception,KeyboardInterrupt) as exc:
        report['error'] = dict(type=type(exc).__name__,message=str(exc))
    finally:
        if session:
            stopped = session.finish(); report['sessions'].append(stopped)
            if not stopped['stopped']: report['completed'] = False
        if 'error' in report:
            print(f"MPS diagnostic failed: {report['error']['type']}: {report['error']['message']}",file=sys.stderr,flush=True)
        for stopped in report['sessions']:
            if 'cleanup_error' in stopped:
                print(f"MPS cleanup failed: {stopped['cleanup_error']}",file=sys.stderr,flush=True)
        concurrent.write_json(root/'review'/'summary.json',report)
    return 0 if report['completed'] else 1


if __name__ == '__main__': raise SystemExit(main())
