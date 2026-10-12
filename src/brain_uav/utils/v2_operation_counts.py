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
              'head_dense_macs', 'head_macs', 'spike_count_l1', 'spike_slots_l1',
              'projection_macs', 'policy_head_dense_macs', 'policy_head_macs',
              'policy_head_ops', 'actor_contraction_ops', 'spike_count_l2', 'spike_slots_l2')

OPERATION_COUNT_METHOD = {
    'format': 'v2_actor_dense_and_spike_contractions', 'version': 2,
    'legacy_head_fields': 'head_macs/head_dense_macs retain all post-encoder contractions, including projection.',
    'policy_head_fields': 'policy_head_* exclude the separately recorded continuous projection.',
    'activity_note': 'Unobserved neurons over this finite evaluation set, not permanently dead neurons.',
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
        self.macs = {'encoder': 0, 'projection': 0, 'head': 0}
        self.spike_dense_macs = self.spike_count = self.spike_slots = self.spike_acs = 0
        self.spike_count_l2 = self.spike_slots_l2 = 0
        self.active_neurons = {1: [], 2: []}

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
) -> tuple[torch.Tensor, dict[str, Any]]:
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

    def enter_projection(module, inputs):
        counter.section = 'projection'

    def leave_projection(module, inputs, output):
        counter.section = 'head'

    def record_activity(layer):
        def hook(module, inputs, spikes):
            counter.active_neurons[layer] = spikes.detach().bool().any(dim=0).any(dim=0).cpu().tolist()
            if layer == 2:
                counter.spike_count_l2 = int(torch.count_nonzero(spikes).item())
                counter.spike_slots_l2 = spikes.numel()
        return hook

    try:
        handles.append(encoder.register_forward_pre_hook(enter_encoder))
        handles.append(encoder.register_forward_hook(leave_encoder, always_call=True))
        handles.append(encoder.pooling.register_forward_pre_hook(pool_products))
        handles.append(actor.projection.register_forward_pre_hook(enter_projection))
        handles.append(actor.projection.register_forward_hook(leave_projection, always_call=True))
        if isinstance(actor, V2SNNPolicyActor):
            handles.append(actor.snn_head.fc2.register_forward_pre_hook(binary_projection))
            handles.append(actor.snn_head.lif1.register_forward_hook(record_activity(1)))
            handles.append(actor.snn_head.lif2.register_forward_hook(record_activity(2)))
        with torch.inference_mode(), counter:
            action = actor(batch)
    finally:
        for handle in handles:
            handle.remove()
    encoder_macs, policy_dense, projection = counter.macs['encoder'], counter.macs['head'], counter.macs['projection']
    head_dense = policy_dense + projection
    if head_dense < counter.spike_dense_macs:
        raise RuntimeError('Binary projection was not captured by the dense contraction counter.')
    head_macs = head_dense - counter.spike_dense_macs
    return action, {
        'decision_count': 1, 'dense_macs': encoder_macs + head_dense,
        'macs': encoder_macs + head_macs, 'acs': counter.spike_acs,
        'encoder_macs': encoder_macs, 'head_dense_macs': head_dense, 'head_macs': head_macs,
        'spike_count_l1': counter.spike_count, 'spike_slots_l1': counter.spike_slots,
        'projection_macs': projection, 'policy_head_dense_macs': policy_dense,
        'policy_head_macs': head_macs - projection,
        'policy_head_ops': 2 * (head_macs - projection) + counter.spike_acs,
        'actor_contraction_ops': 2 * (encoder_macs + head_macs) + counter.spike_acs,
        'spike_count_l2': counter.spike_count_l2, 'spike_slots_l2': counter.spike_slots_l2,
        'active_neurons_l1': counter.active_neurons[1], 'active_neurons_l2': counter.active_neurons[2],
    }


def summarize_operation_counts(counts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-decision counts or already summed episode totals."""
    totals = {key: 0 for key in COUNT_KEYS}
    active = {1: [], 2: []}
    for item in counts:
        for key in COUNT_KEYS:
            totals[key] += item.get(key, 0)
        for layer in (1, 2):
            observed = item.get(f'active_neurons_l{layer}', [])
            if observed:
                if active[layer] and len(active[layer]) != len(observed):
                    raise ValueError('Cannot aggregate different neuron widths.')
                active[layer] = [a or b for a, b in zip(active[layer] or [False] * len(observed), observed)]
    decisions = totals['decision_count']
    mean = {key: value / decisions if decisions else None
            for key, value in totals.items() if key != 'decision_count'}
    slots = totals['spike_slots_l1']
    for layer in (1, 2):
        totals[f'active_neurons_l{layer}'] = active[layer]
    return {'decision_count': decisions, 'totals': totals, 'mean_per_decision': mean,
            'spike_rate_l1': totals['spike_count_l1'] / slots if slots else None,
            'spike_rate_l2': totals['spike_count_l2'] / totals['spike_slots_l2'] if totals['spike_slots_l2'] else None,
            'evaluation_unobserved_neuron_fraction': {
                f'l{layer}': 1 - sum(active[layer]) / len(active[layer]) if active[layer] else None
                for layer in (1, 2)}}


def summarize_spike_activity(counts):
    summary = summarize_operation_counts(counts)
    return {'decision_count': summary['decision_count'],
            'totals': {key: value for key, value in summary['totals'].items()
                       if key == 'decision_count' or key.startswith(('spike_', 'active_neurons_'))},
            'measured_spike_rate_l1': summary['spike_rate_l1'],
            'measured_spike_rate_l2': summary['spike_rate_l2'],
            'evaluation_unobserved_neuron_fraction': summary['evaluation_unobserved_neuron_fraction']}
