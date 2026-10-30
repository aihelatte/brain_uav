"""Opt-in supervision from completed exploratory training episodes, not policy Q labels."""
from math import isfinite
from numbers import Integral, Real

import torch
from torch.nn import functional as F


def validate_mc_return_config(weight, max_age_steps, aggregation):
    if isinstance(weight, bool) or not isinstance(weight, Real) or not isfinite(weight) or weight < 0:
        raise ValueError('mc_return_weight must be a finite nonnegative real number.')
    if isinstance(max_age_steps, bool) or not isinstance(max_age_steps, Integral) or max_age_steps <= 0:
        raise ValueError('mc_return_max_age_steps must be a positive integer.')
    if weight > 0 and aggregation != 'mean':
        raise ValueError('MC return supervision requires explicit target_q_aggregation=mean.')
    return float(weight), int(max_age_steps)


def discounted_episode_returns(rewards, gamma):
    if isinstance(gamma, bool) or not isinstance(gamma, Real) or not isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('gamma must be finite and in [0, 1].')
    result = [0.] * len(rewards)
    value = 0.
    fp32_limit = torch.finfo(torch.float32).max
    for i in range(len(rewards) - 1, -1, -1):
        value = float(rewards[i]) + gamma * value
        if not isfinite(value) or abs(value) > fp32_limit:
            raise ValueError('MC returns must be finite FP32 values.')
        result[i] = value
    return result


def critic_loss_components(q1, q2, target, weight, returns=None, mask=None):
    """Same twin-Q forwards and backward; count-normalized MC on that TD batch."""
    td = F.mse_loss(q1, target.detach()) + F.mse_loss(q2, target.detach())
    if weight == 0:
        return td, td, None, None, None
    if returns is None or mask is None:
        raise ValueError('MC-enabled updates require mc_return and mc_return_mask.')
    mask = mask.detach()
    # Inactive elements are replaced BEFORE squaring, not NaN/Inf multiplied by 0.
    g = returns.detach()
    d1 = torch.where(mask > 0, q1 - g, torch.zeros_like(q1))
    d2 = torch.where(mask > 0, q2 - g, torch.zeros_like(q2))
    count = mask.sum()
    mc = (d1.square() + d2.square()).sum() / count.clamp_min(1.)
    return td + weight * mc, td, mc, count, count / mask.numel()
