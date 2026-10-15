"""Removable ANN index-26 investigation. No training, timing, or SNN worker."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import argparse
import pickle
import random
import subprocess
import sys
import traceback

import numpy as np
import torch

from brain_uav.scripts import v2_gpu_compute_diagnostic as d
from brain_uav.scripts.v2_gpu_td3_diagnostic import flatten

VERSION = 1
TARGET = 'critic2.zone_set_encoder.empty_scene_token'
MODELS = ('actor', 'critic1', 'critic2', 'actor_target', 'critic1_target',
          'critic2_target', 'bc_reference_actor')
COUNTS = ('update_count', 'critic_update_count', 'critic_target_update_count',
          'actor_update_count', 'last_total_steps')


def owned(value):
    """Numeric snapshots never share storage or retain autograd edges."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: owned(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(owned(v) for v in value)
    return deepcopy(value)


def rng_state(e):
    return dict(cpu=owned(torch.get_rng_state()),
        cuda=owned(torch.cuda.get_rng_state(e.device)) if e.device.type == 'cuda' else None,
        python=random.getstate(), numpy=deepcopy(np.random.get_state()),
        replay=deepcopy(e.replay.rng.bit_generator.state))


def restore_rng(e, state):
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], e.device)
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    e.replay.rng.bit_generator.state = deepcopy(state['replay'])


def values(e):
    result = d.error_state(e)['values']
    rng = rng_state(e)
    for name in ('cpu', 'cuda'):
        if rng[name] is not None:
            result['rng.'+name] = rng.pop(name)
        else:
            rng.pop(name)
    result['rng.other'] = torch.tensor(list(pickle.dumps(rng)), dtype=torch.uint8)
    return result


def precision():
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        matmul_precision=torch.get_float32_matmul_precision(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        fp16_reduced_precision=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        bf16_reduced_precision=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic)


def install_gradients(e, saved):
    """Only initial diagnostic state installation, before any update graph."""
    for module in MODELS:
        model = getattr(e, module)
        if model is not None:
            for name, p in model.named_parameters():
                value = saved['gradient.'+module+'.'+name]
                p.grad = None if value is None else value.to(p.device).clone()


@contextmanager
def restored_warmup(e):
    """Capture-only pattern: restore values/grad references in place, not modules.

    Graphs and compiler caches stay alive. No load_state_dict after capture.
    This is diagnostic-only, not an additional steady-state gradient copy.
    """
    models = [getattr(e, n) for n in MODELS if getattr(e, n) is not None]
    model_handles = [(m, dict(m.state_dict(keep_vars=True))) for m in models]
    bindings = [t for _, state in model_handles for t in state.values()]
    optimizers = (e.actor_optimizer, e.critic_optimizer)
    states = [(o, p, o.state.get(p)) for o in optimizers for g in o.param_groups for p in g['params']]
    state_handles = [(s, dict(s)) for _, _, s in states if s is not None]
    bindings += [t for _, _, s in states if s for t in s.values() if isinstance(t, torch.Tensor)]
    executor = e._gpu_update_executor
    published = native_states = None
    if executor is not None:
        bindings += executor.persistent()
        bindings += [t for group in executor.banks for bank in group for t in bank]
        published = list(executor.published)
        native_states = [list(group) for group in executor.native_states]
    # Deduplicate aliases while retaining each captured tensor's original object.
    bindings = list({id(t): t for t in bindings}.values())
    addresses = [t.data_ptr() for t in bindings]
    saved = [owned(t) for t in bindings]
    grads = [(p, p.grad, None if p.grad is None else owned(p.grad))
             for m in models for p in m.parameters()]
    counts, rng = {n: getattr(e, n) for n in COUNTS}, rng_state(e)
    try:
        yield
    finally:
        if e.device.type == 'cuda':
            torch.cuda.synchronize(e.device)
        for model, handles in model_handles:
            live = model.state_dict(keep_vars=True)
            if live.keys() != handles.keys() or any(live[n] is not t for n,t in handles.items()):
                raise RuntimeError('Warmup replaced model tensor handles; cannot restore captured bindings.')
        if addresses != [t.data_ptr() for t in bindings]:
            raise RuntimeError('Warmup changed tensor addresses; cannot restore captured bindings.')
        with torch.no_grad():
            for t, v in zip(bindings, saved):
                t.copy_(v)
            for p, ref, v in grads:
                p.grad = ref
                if ref is not None:
                    ref.copy_(v)
        for o, p, state in states:
            if state is None:
                o.state.pop(p, None)
            elif o.state.get(p) is not state:
                raise RuntimeError('Warmup replaced an existing Adam state; cannot preserve graph bindings.')
        for state, handles in state_handles:
            if state.keys() != handles.keys() or any(state[n] is not t for n,t in handles.items()):
                raise RuntimeError('Warmup replaced Adam tensor handles; cannot preserve graph bindings.')
        if executor is not None:
            executor.published[:] = published
            for group, saved_group in zip(executor.native_states, native_states):
                group[:] = saved_group
        for n, v in counts.items():
            setattr(e, n, v)
        restore_rng(e, rng)


class HistoryReady(Exception):
    pass


class Focus:
    """Small hook into the existing v8 history and normal update recorder."""
    def __init__(self, worker, root, source):
        self.worker, self.root, self.source = worker, root, source
        self.replaying = worker != 'history'
        self.bundle = None
        self.noises = []

    def initial(self, engine, initial, contract):
        self.contract = owned(contract)
        if not self.replaying:
            return initial
        self.bundle = torch.load(self.source, map_location='cpu', weights_only=False)
        if self.bundle['version'] != VERSION or self.bundle['contract'] != contract:
            raise ValueError('Incompatible ANN common-state snapshot; regenerate history.')
        if self.bundle['precision'] != precision():
            raise ValueError('Common-state precision/software settings differ.')
        algorithm = self.bundle['engine']['algorithm_config']
        engine.set_target_noise(policy_noise=algorithm['policy_noise'], noise_clip=algorithm['noise_clip'])
        return self.bundle['engine']

    def before_step(self, e, index, total, batch, noise, failure):
        from brain_uav.scripts.profile_v2_td3 import v2_bc_lambda
        from brain_uav.trainers.v2_gpu_compute import clone_batch
        if self.replaying or index != 26:
            return
        if total != 75028 or batch is None or noise is not None or failure is not None:
            raise ValueError('Index-26 history/input contract changed; refusing approximate reproduction.')
        self.bundle = dict(version=VERSION, diagnostic_only=True, contract=self.contract, index=index, total=total,
            engine=owned(e.checkpoint_state_dict()), before=values(e), rng=rng_state(e),
            gradients={'gradient.'+module+'.'+name: None if p.grad is None else owned(p.grad)
                for module in MODELS if getattr(e,module) is not None
                for name,p in getattr(e,module).named_parameters()},
            batch=clone_batch(batch.to('cpu')), precision=precision(),
            post_clip_key='grad.critic2.'+str(next(i for i,(name,_) in
                enumerate(e.critic2.named_parameters()) if name == TARGET.removeprefix('critic2.'))),
            schedule=dict(bc_lambda=v2_bc_lambda(total, schedule=self.contract['schedule']),
                spike_beta=e.spike_regularization.beta(total),
                policy_noise=e.policy_noise, noise_clip=e.noise_clip))

    @contextmanager
    def observe_noise(self, index):
        if not self.replaying and index != 26:
            yield
            return
        original = torch.randn_like
        self.noises = []
        def observed(*a, **kw):
            result = original(*a, **kw)  # Preserve the real CUDA RNG call/shape/order.
            self.noises.append(owned(result))
            if self.replaying:
                torch.testing.assert_close(self.noises[-1], self.bundle['noise'], rtol=0, atol=0)
            return result
        with patch('torch.randn_like', new=observed):
            yield
        if len(self.noises) != 1:
            raise AssertionError('Expected exactly one real target-noise draw.')

    def after_step(self, e, index, snap):
        if self.replaying or index != 26:
            return
        self.bundle.update(noise=self.noises[0], continuous_after=owned(snap))
        torch.save(self.bundle, self.source)
        raise HistoryReady()

    def history_complete(self, exc):
        return isinstance(exc, HistoryReady)

    def replay(self, e, run_step):
        from brain_uav.trainers.v2_gpu_compute import GPUCompute
        saved = self.bundle
        install_gradients(e, saved['gradients'])
        restore_rng(e, saved['rng'])
        # Compute construction is AFTER the normal action-inference registration
        # and compilation setup. Its existing capture_shape owns warmup/restore.
        if e.gpu_td3_compute:
            e._gpu_update_executor = GPUCompute(e, diagnostic=True)
        before_warmup = values(e)
        if not d.state_difference(saved['before'], before_warmup, exact=True)['passed']:
            raise AssertionError('Loaded common state differs before same-callable warmup.')
        with restored_warmup(e):
            run_step(saved['total'], saved['batch'])
        restored = values(e)
        alignment = d.state_difference(saved['before'], restored, exact=True)
        torch.save(dict(expected=saved['before'], actual=restored), self.root/'other'/f'{self.worker}_alignment.pt')
        if not alignment['passed']:
            raise AssertionError('Warmup state/RNG restoration failed: '+str(alignment['first_difference']))
        from brain_uav.scripts.profile_v2_td3 import v2_bc_lambda
        actual_schedule = dict(bc_lambda=v2_bc_lambda(saved['total'], schedule=self.contract['schedule']),
            spike_beta=e.spike_regularization.beta(saved['total']),
            policy_noise=e.policy_noise, noise_clip=e.noise_clip)
        if actual_schedule != saved['schedule']:
            raise AssertionError('Restored diagnostic schedule differs.')
        # Real graph execution evidence on this single update, not warmup only.
        result = {}
        def update():
            result.update(run_step(saved['total'], saved['batch']))
        evidence = e._profile_cuda_graph_replay(update)
        if evidence['cuda_graph_launch_count'] < 1:
            raise AssertionError('No actual CUDA Graph replay observed on aligned update.')
        delta = {name: result[name]-restored[name] for name in restored
                 if name.startswith(tuple(module+'.' for module in MODELS))
                 and name in result and restored[name].is_floating_point()}
        torch.save(dict(snapshot=owned(result), deltas=owned(delta),
            alignment=alignment, precision=precision(), graph_evidence=evidence),
            self.root/'other'/f'{self.worker}_result.pt')


def target_summary(a, b):
    """All 64 elements plus explicit index 19; original tolerance, no exemption."""
    a, b = a.flatten().double(), b.flatten().double()
    tolerance = 1e-5 + 1e-4*a.abs()
    delta = (a-b).abs()
    return dict(elements=a.numel(), mismatches=int((~torch.isclose(a,b,rtol=1e-4,atol=1e-5)).sum()),
        baseline=a.tolist(), compute=b.tolist(), absolute_error=delta.tolist(),
        allowed_error=tolerance.tolist(), index19=dict(baseline=float(a[19]), compute=float(b[19]),
            absolute_error=float(delta[19]), allowed_error=float(tolerance[19])))


def compare(root):
    a, b = [torch.load(root/'other'/f'{worker}_result.pt', map_location='cpu', weights_only=False)
            for worker in ('baseline','compute')]
    if a['precision'] != b['precision']:
        raise ValueError('Worker precision settings differ.')
    evidence = {}
    comparison = d.state_difference(a['snapshot'], b['snapshot'], evidence=evidence, label='same_state')
    deltas = d.state_difference(a['deltas'], b['deltas'], evidence=evidence, label='update_delta')
    torch.save(evidence, root/'other/differences.pt')
    target = {}
    bundle = torch.load(root/'other/common_state.pt', map_location='cpu', weights_only=False)
    for stage in ('pre_clip','post_clip'):
        if stage == 'pre_clip':
            key = 'tensor.gradient.pre_clip.'+TARGET
        else:
            # Indexed public post-clip gradients use the actual parameter name
            # stored by the common-state history, never a hardcoded ordinal.
            # Use explicit mapping saved below rather than state_dict indices.
            key = bundle['post_clip_key']
        target[stage] = target_summary(a['snapshot'][key], b['snapshot'][key])
    passed = comparison['passed'] and deltas['passed']
    continuous = (d.state_difference(bundle['continuous_after'],a['snapshot'])
        if 'continuous_after' in bundle else None)
    if continuous is None or not continuous['passed']:
        interpretation = '连续基准未复现，原因定位证据不足；同状态对照结果单独保留。'
    elif passed:
        interpretation = '同状态通过，支持历史状态差异参与原失败；不代表ANN完整验收通过。'
    else:
        interpretation = '该共同状态下仍有路径差异；查看首次差异，尚不能据此确定根因。'
    report = dict(passed=passed, completed=True, diagnostic_only=True, precision=a['precision'],
        snapshot_index=bundle.get('index',26), stage_steps=bundle.get('total',75028),
        rtol=1e-4, atol=1e-5, target=target, comparison=comparison, parameter_deltas=deltas,
        alignment={k:v['alignment'] for k,v in (('baseline',a),('compute',b))},
        graph_evidence={k:v['graph_evidence'] for k,v in (('baseline',a),('compute',b))},
        continuous_baseline_vs_aligned=continuous,
        interpretation=interpretation,
        evidence='other/differences.pt', original_continuous_result='other/common_state.pt:continuous_after')
    (root/'review/summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return passed


def main(argv=None):
    from brain_uav.scripts import profile_v2_td3 as p
    parser = p.build_parser()
    parser.description = __doc__
    parser.add_argument('--ann-common-worker', choices=('history','baseline','compute'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args.model != 'ann' or args.device != 'cuda' or args.gpu_td3_compute
            or args.correctness_updates > 8 or args.seed != 7 or args.batch_size != 64
            or args.gpu_compute_first_critic_loss_only or args.gpu_compute_trace_retained_gradient):
        raise ValueError('ANN only, CUDA only, original 8-step prefix; driver selects execution path.')
    root = Path(args.output_dir)
    if args.ann_common_worker:
        focus = Focus(args.ann_common_worker, root, root/'other/common_state.pt')
        args._ann_common_state = focus
        args.gpu_compute_worker = 'correctness'
        args.gpu_td3_compute = args.ann_common_worker == 'compute'
        args.correctness_updates = 8
        args.output_dir = root/'other'/args.ann_common_worker
        d.run(args)
        return
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Refusing nonempty output directory.')
    (root/'other').mkdir(parents=True)
    (root/'review').mkdir()
    result = dict(passed=False, completed=False, workers={})
    try:
        # Only ANN; three new processes, no timing or serial SNN suite.
        for worker in ('history','baseline','compute'):
            command = [sys.executable,'-u','-m',__spec__.name, *(sys.argv[1:] if argv is None else argv),
                       '--ann-common-worker',worker]
            with (root/'review'/f'{worker}.log').open('w',encoding='utf-8') as log:
                rc = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            result['workers'][worker] = rc
            if rc:
                raise RuntimeError('ANN common-state worker failed: '+worker)
        result.update(passed=compare(root), completed=True)
    except BaseException as exc:
        result['error'] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (root/'review/driver.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
