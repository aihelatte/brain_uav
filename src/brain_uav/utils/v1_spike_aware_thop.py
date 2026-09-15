"""V1-only, event-driven *synaptic* operation estimates using THOP hooks.

This is not a count of GPU instructions or a complete FLOP count. Bias adds
are reported separately; LIF dynamics, scaling, activations, and readout
arithmetic are excluded identically and explicitly from the synaptic scope.
"""

from copy import deepcopy
from importlib.metadata import version

import numpy as np
import torch
from torch import nn

from ..models.ann import ANNPolicyActor
from ..models.snn import SNNPolicyActor


def validate_spike_aware_thop_model(model):
    """Fail before a long benchmark if the requested counter cannot be used."""
    from thop import profile  # noqa: F401 -- validate the dependency, without fallback.
    if type(model) not in (ANNPolicyActor, SNNPolicyActor):
        raise ValueError('Spike-aware THOP supports only the V1 ANN/SNN actors.')
    is_snn = isinstance(model, SNNPolicyActor)
    if is_snn:
        from spikingjelly.activation_based.neuron import LIFNode
        if not isinstance(model.lif1, LIFNode) or not isinstance(model.lif2, LIFNode):
            raise ValueError('Spike-aware THOP requires the real SpikingJelly V1 actor.')


def profile_spike_aware_thop(model, observations):
    """Average single-decision counts over supplied observations, without mutation.

    Only the known V1 SNN fc2 receives binary spikes. Other linear layers
    always count dense MACs, even if a particular input happens to be binary.
    The time dimension is included naturally in fc2's input/output tensors.
    """
    from thop import profile
    validate_spike_aware_thop_model(model)
    is_snn = isinstance(model, SNNPolicyActor)
    if len(observations) == 0:
        raise ValueError('Spike-aware THOP requires actual observation samples.')

    actor = deepcopy(model).eval()
    names = {m: name for name, m in actor.named_modules() if isinstance(m, nn.Linear)}
    rows = {name: {'macs': 0, 'acs': 0, 'bias_additions': 0} for name in names.values()}
    parameter = next(actor.parameters())
    input_dim = actor.fc1.in_features if is_snn else actor.net[0].in_features

    def count_linear(module, inputs, output):
        x = inputs[0]
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite values in spike-aware THOP profiling.')
        row = rows[names[module]]
        if is_snn and module is actor.fc2:
            if not bool(((x == 0) | (x == 1)).all()):
                raise ValueError('V1 fc2 input is not binary; cannot classify it as AC.')
            acs = int(torch.count_nonzero(x).item()) * module.out_features
            macs = 0
        else:
            macs = output.numel() * module.in_features
            acs = 0
        row['macs'] += macs
        row['acs'] += acs
        row['bias_additions'] += output.numel() if module.bias is not None else 0
        # THOP's returned ops field is MAC-only in this custom rule. ACs are
        # collected separately so that neither category is silently conflated.
        module.total_ops += macs

    thop_macs = 0.0
    for observation in observations:
        obs = np.asarray(observation)
        if obs.shape != (input_dim,) or not np.isfinite(obs).all():
            raise ValueError('Expected one finite V1 observation per sample.')
        tensor = torch.as_tensor(obs.copy(), dtype=parameter.dtype,
                                 device=parameter.device).unsqueeze(0)
        macs, _ = profile(actor, inputs=(tensor,), custom_ops={nn.Linear: count_linear},
                          verbose=False)
        thop_macs += float(macs)
    if thop_macs != sum(row['macs'] for row in rows.values()):
        raise ValueError('THOP counted unexpected operators outside the linear scope.')
    n = len(observations)
    layers = {name: {f'mean_{k}': v / n for k, v in row.items()}
              for name, row in rows.items()}
    for module, name in names.items():
        layers[name].update(in_features=module.in_features, out_features=module.out_features,
                            input_kind='binary_spikes' if is_snn and module is actor.fc2
                            else 'continuous')
    return {
        'format': 'v1_thop_spike_aware_synaptic_counts', 'version': 1,
        'method': 'thop.profile(custom_ops)', 'thop_version': version('thop'),
        'model_type': 'snn' if is_snn else 'ann',
        'time_window': actor.time_window if is_snn else 1,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'sample_count': n, 'unit': 'operations_per_decision',
        'mean_macs': thop_macs / n,
        'mean_acs': sum(row['acs'] for row in rows.values()) / n,
        'mean_bias_additions': sum(row['bias_additions'] for row in rows.values()) / n,
        'layers': layers,
        'scope': 'Linear synaptic operations only; fc2 binary spikes counted as AC, '
                 'all other linear inputs counted as MAC. Full SNN time window included.',
        'excluded': ['bias additions (reported separately)', 'LIF membrane updates',
                     'observation scaling', 'activations', 'temporal readout arithmetic',
                     'action scaling'],
        'interpretation': 'Event-driven operation estimate, not measured GPU instructions '
                          'or total model FLOPs. Not directly comparable to SyOps totals '
                          'unless their operator scopes and inputs are aligned.',
    }
