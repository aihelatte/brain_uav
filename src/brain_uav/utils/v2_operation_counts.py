"""Offline dense and event-driven contraction counts for eager V2 actors.

These are logical operation counts, not measured CUDA instruction counts.
Only the known binary-input SNN fc2 projection is reclassified as spike ACs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from brain_uav.models import V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.observations import V2ObservationBatch


COUNT_KEYS = ('decision_count', 'dense_macs', 'macs', 'acs', 'encoder_macs',
              'head_dense_macs', 'head_macs', 'spike_count_l1', 'spike_slots_l1')

OPERATION_COUNT_METHOD = {
    'format': 'v2_actor_dense_and_spike_contractions', 'version': 1,
    'included': [
        'All eager actor linear/mm/addmm/bmm/matmul contractions, including fused functional projections.',
        'Attention QK, attention-value and relation-value contractions.',
        'Both task-pooling dot-product reductions (one logical MAC per product).',
        'Masked/padded tokens still count when dense operations execute.',
    ],
    'macs': 'Continuous-input contractions; one multiply-accumulate counts as one MAC.',
    'dense_macs': 'All contractions as dense MACs, including SNN fc2 over the entire time window.',
    'acs': 'SNN fc2 nonzero binary input spikes over all time steps times its out_features.',
    'encoder_macs': 'Entire continuous ZoneSetEncoder, including learned relation embeddings and attention.',
    'head_macs': 'Continuous policy-head projections; SNN fc1 and membrane-mixed action readout remain MACs.',
    'excluded': [
        'Bias and residual additions, LIF membrane updates/resets, nonlinearities and normalization.',
        'Other elementwise arithmetic, parameter-free geometric relation construction, env/observation building.',
        'Critics, optimizers, training, backward passes and measured latency/energy.',
    ],
    'flops_note': '2*MACs+ACs is the included contraction arithmetic only, not full-model FLOPs.',
    'execution_note': 'Actual PyTorch inference remains dense; event-driven ACs are an accounting estimate.',
}


class _ContractionCounter(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.section = 'head'
        self.macs = {'encoder': 0, 'head': 0}
        self.spike_dense_macs = self.spike_count = self.spike_slots = self.spike_acs = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.addmm.default:
            self.macs[self.section] += result.numel() * args[1].shape[-1]
        elif func == torch.ops.aten.linear.default:
            self.macs[self.section] += result.numel() * args[1].shape[-1]
        elif func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default,
                      torch.ops.aten.matmul.default):
            self.macs[self.section] += result.numel() * args[0].shape[-1]
        elif func == torch.ops.aten.einsum.default:
            if args[0] != 'bhij,bhijd->bhid':
                raise RuntimeError(f'Unsupported actor einsum contraction: {args[0]}')
            self.macs[self.section] += result.numel() * args[1][0].shape[-1]
        return result


def count_v2_actor_operations(
    actor: V2ANNPolicyActor | V2SNNPolicyActor, batch: V2ObservationBatch,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Return the unchanged action and counts from that exact single forward."""
    if not isinstance(actor, (V2ANNPolicyActor, V2SNNPolicyActor)):
        raise TypeError('Operation counting supports only V2 ANN/SNN policy actors.')
    encoder = actor.zone_set_encoder
    if actor.training or encoder.compiled_tensor_forward_enabled or encoder.compiled_shared_relations_enabled:
        raise ValueError('Operation counting requires an eval-mode eager actor.')
    if isinstance(actor, V2ANNPolicyActor) and actor.compiled_full_forward_enabled:
        raise ValueError('Operation counting requires an eager ANN actor.')
    if not isinstance(batch, V2ObservationBatch) or batch.ego_features.shape[0] != 1:
        raise ValueError('Operation counting requires one online observation per decision.')
    counter = _ContractionCounter()
    handles = []

    def enter_encoder(module, inputs):
        counter.section = 'encoder'

    def leave_encoder(module, inputs, output):
        counter.section = 'head'

    def pool_products(module, inputs):
        # Pooling uses multiply+sum rather than mm: score and weighted summary.
        counter.macs['encoder'] += 2 * inputs[1].numel()

    def binary_projection(module, inputs):
        spikes = inputs[0]
        # This exact projection follows LIF1, whose outputs are binary spikes.
        counter.spike_count = int(torch.count_nonzero(spikes).item())
        counter.spike_slots = spikes.numel()
        counter.spike_dense_macs = spikes.numel() * module.out_features
        counter.spike_acs = counter.spike_count * module.out_features

    try:
        handles.append(encoder.register_forward_pre_hook(enter_encoder))
        handles.append(encoder.register_forward_hook(leave_encoder, always_call=True))
        handles.append(encoder.pooling.register_forward_pre_hook(pool_products))
        if isinstance(actor, V2SNNPolicyActor):
            handles.append(actor.snn_head.fc2.register_forward_pre_hook(binary_projection))
        with torch.inference_mode(), counter:
            action = actor(batch)
    finally:
        for handle in handles:
            handle.remove()
    encoder_macs, head_dense = counter.macs['encoder'], counter.macs['head']
    if head_dense < counter.spike_dense_macs:
        raise RuntimeError('Binary projection was not captured by the dense contraction counter.')
    head_macs = head_dense - counter.spike_dense_macs
    return action, {
        'decision_count': 1, 'dense_macs': encoder_macs + head_dense,
        'macs': encoder_macs + head_macs, 'acs': counter.spike_acs,
        'encoder_macs': encoder_macs, 'head_dense_macs': head_dense, 'head_macs': head_macs,
        'spike_count_l1': counter.spike_count, 'spike_slots_l1': counter.spike_slots,
    }


def summarize_operation_counts(counts: Iterable[dict[str, int]]) -> dict[str, Any]:
    """Combine per-decision counts or already summed episode totals."""
    totals = {key: 0 for key in COUNT_KEYS}
    for item in counts:
        for key in COUNT_KEYS:
            totals[key] += item[key]
    decisions = totals['decision_count']
    mean = {key: value / decisions if decisions else None
            for key, value in totals.items() if key != 'decision_count'}
    slots = totals['spike_slots_l1']
    return {'decision_count': decisions, 'totals': totals, 'mean_per_decision': mean,
            'spike_rate_l1': totals['spike_count_l1'] / slots if slots else None}
