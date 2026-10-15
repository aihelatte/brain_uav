"""Removable CUDA inference/manual-graph interaction probe; no training wiring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

import torch

CASES = ('plain', 'compiled_inference', 'registered_then_compiled_inference')


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def empty_capture():
    """Keep capture outside inference mode; its body touches no user tensors."""
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Empty capture must run outside InferenceMode.')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        pass
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


def worker(case, output):
    report = dict(case=case, diagnostic_only=True, stage='require_cuda',
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        cpp_stacktraces=os.environ.get('TORCH_SHOW_CPP_STACKTRACES'),
        empty_capture_succeeded=False, error=None)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU is not capture evidence.')
        report['gpu'] = torch.cuda.get_device_name()
        torch.cuda.init()
        retained_graph = None
        if case == CASES[2]:
            report['stage'] = 'early_registration'
            before = torch.cuda.get_rng_state().clone()
            retained_graph = empty_capture()
            report['registration_rng_unchanged'] = torch.equal(before, torch.cuda.get_rng_state())
            if not report['registration_rng_unchanged']:
                raise AssertionError('Early empty capture changed RNG state.')
        if case != 'plain':
            report['stage'] = 'compiled_inference_warmup'
            torch.manual_seed(7)
            model = torch.nn.Sequential(torch.nn.Linear(32, 192), torch.nn.LeakyReLU(),
                torch.nn.Linear(192, 2), torch.nn.Tanh()).cuda().eval()
            inputs = torch.ones(1, 32, device='cuda')
            report['user_input_is_inference'] = inputs.is_inference()
            report['parameter_inference_count'] = sum(p.is_inference() for p in model.parameters())
            options = dict(backend='inductor', mode=None, fullgraph=True, dynamic=True,
                options={'triton.cudagraphs': True})
            report['compilation'] = options
            compiled = torch.compile(model, **options)
            def invoke():
                torch.compiler.cudagraph_mark_step_begin()
                result = compiled(inputs)
                report['compiled_output_is_inference'] = result.is_inference()
            with torch.inference_mode():
                for _ in range(6):
                    invoke()
                report['stage'] = 'compiled_replay_evidence'
                # Same existing CUDA Graph launch evidence mechanism, no new profiler.
                from brain_uav.trainers.v2_td3 import V2TD3UpdateEngine
                report['graph_evidence'] = V2TD3UpdateEngine._profile_cuda_graph_replay(
                    SimpleNamespace(device=torch.device('cuda')), invoke)
            report['compiled_graph_verified'] = report['graph_evidence']['cuda_graph_launch_count'] > 0
        report['stage'] = 'normal_empty_capture'
        report['inference_mode_at_capture'] = torch.is_inference_mode_enabled()
        final_graph = empty_capture()
        report['empty_capture_succeeded'] = True
        report['early_graph_retained_through_final_capture'] = retained_graph is not None
        # Keep both graph objects alive until after the final capture and checks.
        assert final_graph is not None
        report['stage'] = 'complete'
        return 0
    except BaseException as exc:
        report['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        return 1
    finally:
        write(output/'result.json', report)


def verdict(results):
    plain, compiled, registered = (results[name] for name in CASES)
    if not plain.get('empty_capture_succeeded'):
        return 'control_failed'
    if not all(r.get('compiled_graph_verified') for r in (compiled, registered)):
        return 'insufficient_cuda_graph_evidence'
    if compiled.get('empty_capture_succeeded'):
        return 'not_reproduced_on_simplified_model'
    failure = compiled.get('error') or {}
    reproduced = compiled.get('stage') == 'normal_empty_capture' and (
        'Inplace update to inference tensor outside InferenceMode' in failure.get('message', ''))
    if not reproduced:
        return 'different_failure_requires_review'
    if (registered.get('empty_capture_succeeded') and registered.get('registration_rng_unchanged')
            and registered.get('early_graph_retained_through_final_capture')):
        return 'supports_early_registration_candidate_not_real_td3_validation'
    return 'reproduced_candidate_not_validated'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing nonempty probe directory.')
    output.mkdir(parents=True, exist_ok=True)
    if args.case:
        return worker(args.case, output)
    env = dict(os.environ, TORCH_SHOW_CPP_STACKTRACES='1')
    results = {}
    for case in CASES:
        directory = output/case
        try:
            with (output/(case+'.log')).open('w', encoding='utf-8') as log:
                process = subprocess.run([sys.executable, '-u', '-m', __spec__.name,
                    '--case', case, '--output-dir', str(directory)], env=env,
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
            result['exit_code'] = process.returncode
        except Exception as exc:
            result = dict(case=case, stage='child_process_or_report', error=dict(
                type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        results[case] = result
        write(output/'summary.json', dict(diagnostic_only=True, status='collecting', results=results))
    status = verdict(results)
    write(output/'summary.json', dict(diagnostic_only=True, status=status, results=results,
        limitation='Synthetic inference/empty graphs only; no TD3 update or performance evidence.'))
    print(status)
    return 0 if status in ('not_reproduced_on_simplified_model',
        'supports_early_registration_candidate_not_real_td3_validation') else 1


if __name__ == '__main__':
    raise SystemExit(main())
