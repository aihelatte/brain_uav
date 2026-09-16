"""Standalone TD3 update validation for structured V2 observations."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite, pi
import random
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import record_function

from brain_uav.models.v2_ann import V2ANNCritic, V2ANNPolicyActor
from brain_uav.models.v2_snn import V2SNNPolicyActor
from brain_uav.models.zone_set_encoder import ZoneSetSharedRelations
from brain_uav.observations import (
    EGO_FEATURE_DIM,
    EGO_FEATURE_INDEX,
    GOAL_FEATURE_DIM,
    GOAL_FEATURE_INDEX,
    ZONE_FEATURE_DIM,
    V2Observation,
    V2ObservationBatch,
    collate_v2_observations,
)

from .v2_replay_buffer import V2ReplayBuffer

_GOAL_FORWARD_NORM_INDEX = int(GOAL_FEATURE_INDEX['goal_forward_norm'])
_GOAL_RIGHT_NORM_INDEX = int(GOAL_FEATURE_INDEX['goal_right_norm'])
_GOAL_UP_NORM_INDEX = int(GOAL_FEATURE_INDEX['goal_up_norm'])
_GAMMA_FRACTION_INDEX = int(EGO_FEATURE_INDEX['gamma_fraction'])


def _action_inference_ann_tensors(
    actor: V2ANNPolicyActor,
    ego_features: torch.Tensor,
    goal_features: torch.Tensor,
    clean_zone_features: torch.Tensor,
    presence_mask: torch.Tensor,
    valid_token_mask: torch.Tensor,
    token_pair_relations: torch.Tensor,
    relation_pair_mask: torch.Tensor,
) -> torch.Tensor:
    return actor._compute_full_forward_tensors(
        ego_features, goal_features, clean_zone_features, presence_mask,
        valid_token_mask, token_pair_relations, relation_pair_mask,
    )


def _action_inference_snn_encoder_tensors(
    encoder: nn.Module,
    ego_features: torch.Tensor,
    goal_features: torch.Tensor,
    clean_zone_features: torch.Tensor,
    presence_mask: torch.Tensor,
    valid_token_mask: torch.Tensor,
    token_pair_relations: torch.Tensor,
    relation_pair_mask: torch.Tensor,
) -> torch.Tensor:
    return encoder._compute_policy_context_tensors(
        ego_features, goal_features, clean_zone_features, presence_mask,
        valid_token_mask, token_pair_relations, relation_pair_mask,
    )


def _actor_loss_tensor_block(
    actor_actions: torch.Tensor,
    q_values: torch.Tensor,
    reference_actions: torch.Tensor | None,
    ego_features: torch.Tensor,
    goal_features: torch.Tensor,
    line_to_goal_safe: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    alpha: torch.Tensor,
    bc_lambda: torch.Tensor,
    terminal_lambda: torch.Tensor,
    horizontal_span: float,
    vertical_span: float,
    gamma_max: float,
    terminal_radius: float,
    terminal_enabled: bool,
) -> tuple[torch.Tensor, ...]:
    rl_actor_loss = -q_values.mean()
    q_scale = q_values.detach().abs().mean().clamp(min=1.0)
    actor_rl_scale = alpha / q_scale
    scaled_rl_actor_loss = rl_actor_loss * actor_rl_scale
    bc_loss = (
        actor_actions.sum() * 0.0 if reference_actions is None
        else F.mse_loss(actor_actions, reference_actions)
    )
    if terminal_enabled:
        goal_forward = goal_features[:, _GOAL_FORWARD_NORM_INDEX] * horizontal_span
        goal_right = goal_features[:, _GOAL_RIGHT_NORM_INDEX] * horizontal_span
        goal_up = goal_features[:, _GOAL_UP_NORM_INDEX] * vertical_span
        goal_distance = torch.sqrt(
            goal_forward.square() + goal_right.square() + goal_up.square()
        )
        gamma = ego_features[:, _GAMMA_FRACTION_INDEX] * gamma_max
        target_gamma = torch.atan2(
            goal_up, torch.sqrt(goal_forward.square() + goal_right.square()),
        )
        relative_target_psi = torch.atan2(goal_right, goal_forward)
        delta_gamma = torch.maximum(
            torch.minimum(target_gamma - gamma, action_high[0]), action_low[0],
        )
        delta_psi = torch.maximum(
            torch.minimum(
                torch.remainder(relative_target_psi + pi, 2.0 * pi) - pi,
                action_high[1],
            ),
            action_low[1],
        )
        target_action = torch.stack((delta_gamma, delta_psi), dim=-1)
        squared_error = (actor_actions - target_action).square().mean(dim=-1)
        eligible = (
            (line_to_goal_safe.squeeze(-1) > 0.5)
            & (goal_distance <= terminal_radius)
        ).to(squared_error.dtype)
        terminal_geo_loss = (
            (squared_error * eligible).sum() / eligible.sum().clamp_min(1.0)
        )
    else:
        terminal_geo_loss = actor_actions.sum() * 0.0
    actor_loss = (
        scaled_rl_actor_loss + bc_lambda * bc_loss
        + terminal_lambda * terminal_geo_loss
    )
    return (
        actor_loss, rl_actor_loss, scaled_rl_actor_loss,
        actor_rl_scale, bc_loss, terminal_geo_loss,
    )


V2_TD3_CHECKPOINT_FORMAT = 'v2_td3_dynamic_set'
V2_TD3_CHECKPOINT_VERSION = 3
V2_SNN_TD3_CHECKPOINT_FORMAT = 'v2_snn_td3_dynamic_set'
V2_SNN_TD3_CHECKPOINT_VERSION = 1
V2_OBSERVATION_CONTRACT_ID = 'v2_dynamic_zone_set_observation_v1'
V2_TD3_UPDATE_TIMING_SECTIONS = (
    'replay_sample',
    'batch_preparation',
    'target_forward_and_td_target',
    'online_critic_forward_and_loss',
    'critic_backward',
    'critic_gradient_check_and_clip',
    'critic_optimizer_step',
    'actor_update',
    'target_soft_update',
)
V2_TD3_UPDATE_DETAIL_SECTIONS = (
    'critic_zero_grad',
    'critic_loss_backward',
    'actor_guidance_context',
    'actor_forward',
    'actor_q_rl_loss_and_scale',
    'actor_bc_reference_forward_and_loss',
    'actor_terminal_geometry_loss',
    'actor_loss_composition_and_finite_check',
    'actor_zero_grad',
    'actor_backward',
    'actor_gradient_check_and_clip',
    'actor_optimizer_step',
)

V2PolicyActor = V2ANNPolicyActor | V2SNNPolicyActor


def _actor_model_type(actor: Any, *, name: str = 'actor') -> str:
    if isinstance(actor, V2SNNPolicyActor):
        return 'snn'
    if isinstance(actor, V2ANNPolicyActor):
        return 'ann'
    raise TypeError(f'{name} must be a V2ANNPolicyActor or V2SNNPolicyActor.')


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be finite.') from exc
    if not isfinite(result):
        raise ValueError(f'{name} must be finite.')
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer.')
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a positive integer.') from exc
    if result <= 0 or result != value:
        raise ValueError(f'{name} must be a positive integer.')
    return result


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a non-negative integer.')
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a non-negative integer.') from exc
    if result < 0 or result != value:
        raise ValueError(f'{name} must be a non-negative integer.')
    return result


@dataclass(frozen=True, slots=True)
class V2TD3UpdateMetrics:
    """Scalar results from one independent V2 TD3 update."""

    critic_loss: float = 0.0
    actor_loss: float = 0.0
    rl_actor_loss: float = 0.0
    scaled_rl_actor_loss: float = 0.0
    actor_rl_scale: float = 1.0
    bc_loss: float = 0.0
    bc_lambda: float = 0.0
    terminal_geo_loss: float = 0.0
    terminal_geo_lambda: float = 0.0
    sample_success_fraction: float = 0.0
    sample_near_goal_fraction: float = 0.0
    critic_updated: bool = False
    critic_targets_updated: bool = False
    actor_updated: bool = False


@dataclass(frozen=True, slots=True)
class _ActorLossTerms:
    actor_loss: torch.Tensor
    rl_actor_loss: torch.Tensor
    scaled_rl_actor_loss: torch.Tensor
    actor_rl_scale: torch.Tensor
    bc_loss: torch.Tensor
    terminal_geo_loss: torch.Tensor


class _OptionalUpdateWallTimer:
    """Collect diagnostic-only wall intervals without synchronizing CUDA."""

    def __init__(
        self,
        recorder: Callable[[dict[str, float]], None] | None,
    ) -> None:
        if recorder is not None and not callable(recorder):
            raise TypeError('timing_recorder must be callable when provided.')
        self.recorder = recorder
        self.values = {
            name: 0.0 for name in V2_TD3_UPDATE_TIMING_SECTIONS
        }

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if self.recorder is None:
            yield
            return
        started = perf_counter()
        try:
            yield
        finally:
            self.values[name] += perf_counter() - started

    def finish(self) -> None:
        if self.recorder is not None:
            self.recorder(dict(self.values))


class _OptionalUpdateDetailTimer:
    """Optional nested wall/profiler regions that never synchronize CUDA."""

    def __init__(
        self,
        recorder: Callable[[dict[str, Any]], None] | None,
        profile_sections: bool,
    ) -> None:
        if recorder is not None and not callable(recorder):
            raise TypeError(
                'diagnostic_timing_recorder must be callable when provided.'
            )
        if type(profile_sections) is not bool:
            raise TypeError('diagnostic_profile_sections must be a bool.')
        self.recorder = recorder
        self.profile_sections = profile_sections
        self.wall_seconds = {
            name: 0.0 for name in V2_TD3_UPDATE_DETAIL_SECTIONS
        }
        self.calls = {name: 0 for name in V2_TD3_UPDATE_DETAIL_SECTIONS}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if name not in self.wall_seconds:
            raise KeyError(f'Unknown TD3 detail timing section {name!r}.')
        started = perf_counter() if self.recorder is not None else None
        try:
            if self.profile_sections:
                with record_function(f'v2_td3.detail.{name}'):
                    yield
            else:
                yield
        finally:
            if started is not None:
                self.wall_seconds[name] += perf_counter() - started
            if self.recorder is not None or self.profile_sections:
                self.calls[name] += 1

    def finish(self) -> None:
        if self.recorder is not None:
            self.recorder({
                'wall_seconds': dict(self.wall_seconds),
                'calls': dict(self.calls),
            })


class V2TD3UpdateEngine:
    """Independent V2 structured-observation update and validation engine.

    This is not a formal environment trainer and is not connected to the
    project's production training pipeline.
    """

    def __init__(
        self,
        actor: V2PolicyActor,
        critic1: V2ANNCritic,
        critic2: V2ANNCritic,
        replay: V2ReplayBuffer,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
        batch_size: int,
        action_low: np.ndarray | torch.Tensor,
        action_high: np.ndarray | torch.Tensor,
        *,
        actor_freeze_steps: int = 0,
        actor_grad_clip_norm: float | None = None,
        critic_grad_clip_norm: float | None = None,
        actor_rl_scale_alpha: float = 2.5,
        terminal_geo_regularization_enabled: bool = True,
        terminal_geo_radius: float = 250.0,
        terminal_geo_lambda: float = 3000.0,
        bc_reference_actor: V2PolicyActor | None = None,
        device: str | torch.device = 'cpu',
        fused_adam: bool = False,
        aggregate_relation_values_first: bool = False,
    ) -> None:
        actor_model_type = _actor_model_type(actor)
        if not isinstance(critic1, V2ANNCritic) or not isinstance(critic2, V2ANNCritic):
            raise TypeError('critic1 and critic2 must be V2ANNCritic instances.')
        if not isinstance(replay, V2ReplayBuffer):
            raise TypeError('replay must be a V2ReplayBuffer.')
        parameter_sets = (
            {id(parameter) for parameter in actor.parameters()},
            {id(parameter) for parameter in critic1.parameters()},
            {id(parameter) for parameter in critic2.parameters()},
        )
        if (
            not parameter_sets[0].isdisjoint(parameter_sets[1])
            or not parameter_sets[0].isdisjoint(parameter_sets[2])
            or not parameter_sets[1].isdisjoint(parameter_sets[2])
        ):
            raise ValueError('Actor, critic1, and critic2 must not share Parameters.')
        if actor.action_dim != critic1.action_dim or actor.action_dim != critic2.action_dim:
            raise ValueError('Actor and critics must use the same action_dim.')
        if replay.action_dim != actor.action_dim:
            raise ValueError('Replay and networks must use the same action_dim.')
        self._validate_network_architecture(actor, critic1, name='critic1')
        self._validate_network_architecture(actor, critic2, name='critic2')

        self.device = torch.device(device)
        if type(fused_adam) is not bool:
            raise TypeError('fused_adam must be a bool.')
        if type(aggregate_relation_values_first) is not bool:
            raise TypeError('aggregate_relation_values_first must be a bool.')
        if fused_adam and self.device.type not in ('cpu', 'cuda'):
            raise ValueError(f'fused Adam is unsupported on {self.device.type}.')
        self.fused_adam = fused_adam
        self.aggregate_relation_values_first = aggregate_relation_values_first
        self.model_type = actor_model_type
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.replay = replay
        self.action_dim = actor.action_dim
        self.action_low = self._action_bound(
            action_low, name='action_low', device=self.device
        )
        self.action_high = self._action_bound(
            action_high, name='action_high', device=self.device
        )
        if not bool((self.action_low < self.action_high).all()):
            raise ValueError('Every action_low value must be below action_high.')
        if not torch.equal(self.action_high, self.actor.action_limit):
            raise ValueError(
                'action_high must exactly match actor.action_limit.'
            )
        if not torch.equal(self.action_low, -self.actor.action_limit):
            raise ValueError(
                'action_low must exactly match -actor.action_limit.'
            )
        self._action_low_cpu = self.action_low.detach().cpu().numpy().copy()
        self._action_high_cpu = self.action_high.detach().cpu().numpy().copy()
        self._action_low_cpu.setflags(write=False)
        self._action_high_cpu.setflags(write=False)

        actor_lr_value = _finite_float(actor_lr, name='actor_lr')
        critic_lr_value = _finite_float(critic_lr, name='critic_lr')
        if actor_lr_value <= 0.0 or critic_lr_value <= 0.0:
            raise ValueError('Learning rates must be greater than zero.')
        self.gamma = _finite_float(gamma, name='gamma')
        self.tau = _finite_float(tau, name='tau')
        if self.gamma < 0.0 or self.gamma > 1.0:
            raise ValueError('gamma must be in [0, 1].')
        if self.tau <= 0.0 or self.tau > 1.0:
            raise ValueError('tau must be in (0, 1].')
        self.policy_noise = 0.0
        self.noise_clip = 0.0
        self.set_target_noise(
            policy_noise=policy_noise,
            noise_clip=noise_clip,
        )
        self.policy_delay = _positive_int(policy_delay, name='policy_delay')
        self.batch_size = _positive_int(batch_size, name='batch_size')
        self.actor_freeze_steps = _nonnegative_int(
            actor_freeze_steps, name='actor_freeze_steps'
        )
        self.actor_grad_clip_norm = self._optional_positive_float(
            actor_grad_clip_norm, name='actor_grad_clip_norm'
        )
        self.critic_grad_clip_norm = self._optional_positive_float(
            critic_grad_clip_norm, name='critic_grad_clip_norm'
        )
        self.actor_rl_scale_alpha = _finite_float(
            actor_rl_scale_alpha, name='actor_rl_scale_alpha'
        )
        if self.actor_rl_scale_alpha <= 0.0:
            raise ValueError('actor_rl_scale_alpha must be greater than zero.')
        self.terminal_geo_regularization_enabled = bool(
            terminal_geo_regularization_enabled
        )
        self.terminal_geo_radius = _finite_float(
            terminal_geo_radius, name='terminal_geo_radius'
        )
        self.terminal_geo_lambda = _finite_float(
            terminal_geo_lambda, name='terminal_geo_lambda'
        )
        if self.terminal_geo_radius < 0.0 or self.terminal_geo_lambda < 0.0:
            raise ValueError('Terminal geometry radius and lambda must be non-negative.')
        if self.terminal_geo_regularization_enabled and self.action_dim != 2:
            raise ValueError(
                'terminal geometric regularization requires action_dim == 2.'
            )

        self.actor_target = self._frozen_target(self.actor)
        self.critic1_target = self._frozen_target(self.critic1)
        self.critic2_target = self._frozen_target(self.critic2)
        self._validate_all_live_fixed_buffers()
        self._soft_update_parameter_pairs = self._bind_soft_update_parameter_pairs()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr_value,
            fused=True if fused_adam else None,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr_value,
            fused=True if fused_adam else None,
        )
        self.bc_reference_actor: V2PolicyActor | None = None
        if bc_reference_actor is not None:
            reference_model_type = _actor_model_type(
                bc_reference_actor, name='bc_reference_actor'
            )
            if reference_model_type != self.model_type:
                raise ValueError(
                    'actor and bc_reference_actor must use the same model type.'
                )
            self._validate_actor_architecture(
                self.actor, bc_reference_actor, name='bc_reference_actor'
            )
            self.bc_reference_actor = self._frozen_target(
                bc_reference_actor,
                device=self.device,
            )
            self._validate_model_fixed_buffers(
                self.bc_reference_actor,
                name='bc_reference_actor',
            )
        if aggregate_relation_values_first:
            for model in (
                self.actor, self.critic1, self.critic2,
                self.actor_target, self.critic1_target, self.critic2_target,
                self.bc_reference_actor,
            ):
                if model is not None:
                    for layer in model.zone_set_encoder.layers:
                        layer.attention.enable_relation_value_aggregation()

        self.update_count = 0
        self.critic_update_count = 0
        self.critic_target_update_count = 0
        self.actor_update_count = 0
        self.last_total_steps = 0
        self.frozen_critic_strategy = 'eager'
        self._compiled_critic_loss: Callable[..., tuple[torch.Tensor, ...]] | None = None
        self._compiled_target_block: Callable[..., tuple[torch.Tensor, ...]] | None = None
        self._compiled_target_critic_td: Callable[..., tuple[torch.Tensor, ...]] | None = None
        self._compiled_actor_loss: Callable[..., tuple[torch.Tensor, ...]] | None = None
        self.cache_actor_loss_coefficients = False
        self._actor_loss_coefficient_cache: dict[
            str, tuple[tuple[float, torch.device, torch.dtype], torch.Tensor]
        ] = {}
        self._compiled_action_inference: Callable[..., torch.Tensor] | None = None
        self._compiled_action_inference_config: dict[str, object] | None = None

    def enable_actor_loss_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str, ...]:
        if self._compiled_actor_loss is not None:
            raise RuntimeError('Actor loss tensor block is already compiled.')
        self._compiled_actor_loss = torch.compile(
            _actor_loss_tensor_block,
            backend=backend, mode=mode, fullgraph=fullgraph, dynamic=dynamic,
        )
        return ('actor_loss.tensor_block',)

    def warmup_actor_loss_compile(
        self, batches: Sequence[V2ObservationBatch],
    ) -> None:
        if self._compiled_actor_loss is None:
            raise RuntimeError('The actor loss tensor block must be compiled first.')
        warmup_batches = tuple(batches)
        if not warmup_batches:
            raise ValueError('At least one actor loss warmup batch is required.')
        torch_rng = torch.random.get_rng_state()
        numpy_rng = np.random.get_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        )
        counts = (
            self.update_count, self.critic_update_count,
            self.critic_target_update_count, self.actor_update_count,
            self.last_total_steps,
        )
        try:
            for batch in warmup_batches:
                device_batch = batch.to(self.device)
                for coefficient in (0.0, 1.5):
                    actions = torch.zeros(
                        (device_batch.batch_size, self.action_dim),
                        device=self.device, dtype=torch.float32,
                        requires_grad=True,
                    )
                    q_values = torch.ones(
                        (device_batch.batch_size, 1), device=self.device,
                        dtype=torch.float32, requires_grad=True,
                    )
                    reference = (
                        torch.zeros_like(actions)
                        if self.bc_reference_actor is not None else None
                    )
                    total, *_ = self._compiled_actor_loss(
                        actions, q_values, reference,
                        device_batch.ego_features, device_batch.goal_features,
                        torch.ones_like(q_values), self.action_low, self.action_high,
                        torch.as_tensor(self.actor_rl_scale_alpha, device=self.device),
                        torch.as_tensor(coefficient, device=self.device),
                        torch.as_tensor(
                            self.terminal_geo_lambda
                            if self.terminal_geo_regularization_enabled else 0.0,
                            device=self.device,
                        ),
                        self.actor.scales.horizontal_span,
                        self.actor.scales.vertical_span,
                        self.actor.scales.gamma_max,
                        self.terminal_geo_radius,
                        self.terminal_geo_regularization_enabled,
                    )
                    total.backward()
                    if actions.grad is None or q_values.grad is None:
                        raise RuntimeError('Actor loss warmup lost action or Q gradients.')
        finally:
            torch.random.set_rng_state(torch_rng)
            np.random.set_state(numpy_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        if counts != (
            self.update_count, self.critic_update_count,
            self.critic_target_update_count, self.actor_update_count,
            self.last_total_steps,
        ):
            raise RuntimeError('Actor loss warmup must not change TD3 counters.')

    @staticmethod
    def _optional_positive_float(value: float | None, *, name: str) -> float | None:
        if value is None:
            return None
        result = _finite_float(value, name=name)
        if result <= 0.0:
            raise ValueError(f'{name} must be greater than zero when provided.')
        return result

    def _action_bound(
        self,
        value: np.ndarray | torch.Tensor,
        *,
        name: str,
        device: torch.device,
    ) -> torch.Tensor:
        try:
            bound = torch.as_tensor(value, dtype=torch.float32, device=device).clone()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f'{name} must be a finite action vector.') from exc
        if bound.shape != (self.action_dim,):
            raise ValueError(
                f'{name} must have shape ({self.action_dim},); got {tuple(bound.shape)}.'
            )
        if not bool(torch.isfinite(bound).all()):
            raise ValueError(f'{name} must contain only finite values.')
        return bound

    def _expected_fixed_buffers(self, model: nn.Module) -> dict[str, torch.Tensor]:
        pair_prefix = 'zone_set_encoder.pair_relation_builder.'
        expected_values: dict[str, float | np.ndarray] = {
            f'{pair_prefix}horizontal_span': float(model.scales.horizontal_span),
            f'{pair_prefix}vertical_span': float(model.scales.vertical_span),
            f'{pair_prefix}world_diagonal': float(model.scales.world_diagonal),
            f'{pair_prefix}uav_radius': float(model.uav_radius),
        }
        if isinstance(model, (V2ANNPolicyActor, V2SNNPolicyActor)):
            expected_values['action_limit'] = self._action_high_cpu
        elif not isinstance(model, V2ANNCritic):
            raise TypeError('Fixed-buffer validation requires a V2 actor or critic.')

        buffers = dict(model.named_buffers())
        if buffers.keys() != expected_values.keys():
            raise ValueError(
                'Model fixed buffer names do not match the declared V2 architecture.'
            )
        expected: dict[str, torch.Tensor] = {}
        for name, value in expected_values.items():
            buffer = buffers[name]
            source = value.copy() if isinstance(value, np.ndarray) else value
            expected[name] = torch.as_tensor(
                source,
                dtype=buffer.dtype,
                device=buffer.device,
            )
        return expected

    def _validate_model_fixed_buffers(self, model: nn.Module, *, name: str) -> None:
        buffers = dict(model.named_buffers())
        expected = self._expected_fixed_buffers(model)
        for buffer_name, expected_value in expected.items():
            candidate = buffers[buffer_name]
            if (
                candidate.shape != expected_value.shape
                or candidate.dtype != expected_value.dtype
                or not torch.equal(candidate, expected_value)
            ):
                raise ValueError(
                    f'{name} fixed buffer {buffer_name!r} does not match '
                    'the declared V2 architecture.'
                )

    def _validate_state_dict_fixed_buffers(
        self,
        model: nn.Module,
        state_dict: dict[str, torch.Tensor],
        *,
        name: str,
    ) -> None:
        expected = self._expected_fixed_buffers(model)
        for buffer_name, expected_value in expected.items():
            candidate = state_dict[buffer_name]
            if (
                candidate.shape != expected_value.shape
                or candidate.dtype != expected_value.dtype
                or not torch.equal(
                    candidate.detach().cpu(),
                    expected_value.detach().cpu(),
                )
            ):
                raise ValueError(
                    f'{name} fixed buffer {buffer_name!r} does not match '
                    'the declared V2 architecture.'
                )

    def _validate_all_live_fixed_buffers(self) -> None:
        for name, model in (
            ('actor', self.actor),
            ('critic1', self.critic1),
            ('critic2', self.critic2),
            ('actor_target', self.actor_target),
            ('critic1_target', self.critic1_target),
            ('critic2_target', self.critic2_target),
        ):
            self._validate_model_fixed_buffers(model, name=name)

    @staticmethod
    def _validate_actor_architecture(
        expected: V2PolicyActor,
        candidate: V2PolicyActor,
        *,
        name: str,
    ) -> None:
        if _actor_model_type(expected) != _actor_model_type(candidate, name=name):
            raise ValueError(f'{name} model type does not match actor.')
        if not torch.equal(
            candidate.action_limit.detach().cpu(),
            expected.action_limit.detach().cpu(),
        ):
            raise ValueError(f'{name} action_limit does not match actor.')
        if (
            candidate.action_dim != expected.action_dim
            or candidate.hidden_dim != expected.hidden_dim
            or candidate.encoder_config != expected.encoder_config
            or candidate.scales != expected.scales
            or candidate.uav_radius != expected.uav_radius
        ):
            raise ValueError(f'{name} architecture does not match actor.')
        if isinstance(expected, V2SNNPolicyActor) and (
            not isinstance(candidate, V2SNNPolicyActor)
            or candidate.time_window != expected.time_window
            or candidate.tau != expected.tau
            or candidate.surrogate_name != expected.surrogate_name
            or candidate.backend != expected.backend
        ):
            raise ValueError(f'{name} SNN architecture does not match actor.')

    @staticmethod
    def _validate_network_architecture(
        actor: V2PolicyActor,
        critic: V2ANNCritic,
        *,
        name: str,
    ) -> None:
        if (
            critic.encoder_config != actor.encoder_config
            or critic.scales != actor.scales
            or critic.uav_radius != actor.uav_radius
        ):
            raise ValueError(f'{name} encoder architecture does not match actor.')

    def _bind_soft_update_parameter_pairs(
        self,
    ) -> dict[tuple[int, int], tuple[list[nn.Parameter], list[nn.Parameter]]]:
        bound: dict[
            tuple[int, int], tuple[list[nn.Parameter], list[nn.Parameter]]
        ] = {}
        for online, target in (
            (self.actor, self.actor_target),
            (self.critic1, self.critic1_target),
            (self.critic2, self.critic2_target),
        ):
            online_named = tuple(online.named_parameters())
            target_named = tuple(target.named_parameters())
            if tuple(name for name, _ in online_named) != tuple(
                name for name, _ in target_named
            ):
                raise ValueError('Online and target parameter names do not match.')
            online_parameters: list[nn.Parameter] = []
            target_parameters: list[nn.Parameter] = []
            for (name, source), (_, destination) in zip(
                online_named, target_named
            ):
                if source is destination:
                    raise ValueError(
                        f'Online and target parameter {name!r} must be independent.'
                    )
                if source.shape != destination.shape:
                    raise ValueError(
                        f'Online and target parameter {name!r} shapes do not match.'
                    )
                if source.dtype != destination.dtype:
                    raise ValueError(
                        f'Online and target parameter {name!r} dtypes do not match.'
                    )
                if source.device != destination.device:
                    raise ValueError(
                        f'Online and target parameter {name!r} devices do not match.'
                    )
                online_parameters.append(source)
                target_parameters.append(destination)
            bound[(id(online), id(target))] = (
                online_parameters,
                target_parameters,
            )
        return bound

    @staticmethod
    def _frozen_target(
        model: nn.Module,
        *,
        device: torch.device | None = None,
    ):
        target = deepcopy(model)
        if device is not None:
            target = target.to(device)
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        return target

    def set_target_noise(
        self,
        *,
        policy_noise: float,
        noise_clip: float,
    ) -> None:
        """Set scheduled TD3 target noise after strict validation."""

        policy_noise_value = _finite_float(
            policy_noise,
            name='policy_noise',
        )
        noise_clip_value = _finite_float(noise_clip, name='noise_clip')
        if policy_noise_value < 0.0 or noise_clip_value < 0.0:
            raise ValueError('policy_noise and noise_clip must be non-negative.')
        self.policy_noise = policy_noise_value
        self.noise_clip = noise_clip_value

    def enable_online_critic_encoder_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str, str]:
        enabled: list[str] = []
        for name, critic in (
            ('critic1.zone_set_encoder', self.critic1),
            ('critic2.zone_set_encoder', self.critic2),
        ):
            critic.zone_set_encoder.enable_compiled_tensor_forward(
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            enabled.append(name)
        return enabled[0], enabled[1]

    def enable_shared_relations_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str]:
        self.actor.zone_set_encoder.enable_compiled_shared_relations(
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return ('shared_relations.tensor_build',)

    def set_frozen_critic_strategy(self, strategy: str) -> None:
        if strategy not in ('eager', 'compiled_no_grad_context'):
            raise ValueError(
                'frozen critic strategy must be eager or '
                'compiled_no_grad_context.'
            )
        if (
            strategy == 'compiled_no_grad_context'
            and not self.critic1.zone_set_encoder.compiled_tensor_forward_enabled
        ):
            raise RuntimeError(
                'compiled_no_grad_context requires a compiled critic1 encoder.'
            )
        self.frozen_critic_strategy = strategy

    def configure_compilation(
        self,
        *,
        compile_critic_encoder: bool = False,
        compile_target_encoders: bool = False,
        compile_actors: bool = False,
        frozen_critic_strategy: str = 'eager',
        compile_critic_block: bool = False,
        compile_target_block: bool = False,
        compile_shared_relations: bool = False,
        compile_snn_target_encoder: bool = False,
        compile_actor_loss: bool = False,
        cache_actor_loss_coefficients: bool = False,
        compile_action_inference: bool = False,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> dict[str, object]:
        """Register one explicit, non-fallback TD3 compilation configuration."""

        if compile_critic_encoder and compile_critic_block:
            raise ValueError(
                'compile_critic_encoder and compile_critic_block are mutually '
                'exclusive compilation granularities.'
            )
        if compile_target_encoders and compile_target_block:
            raise ValueError(
                'compile_target_encoders and compile_target_block are mutually '
                'exclusive compilation granularities.'
            )
        if compile_target_encoders and compile_snn_target_encoder:
            raise ValueError(
                'compile_target_encoders and compile_snn_target_encoder are '
                'mutually exclusive target encoder scopes.'
            )
        if type(cache_actor_loss_coefficients) is not bool:
            raise TypeError('cache_actor_loss_coefficients must be a bool.')
        if cache_actor_loss_coefficients and not compile_actor_loss:
            raise ValueError(
                'cache_actor_loss_coefficients requires compile_actor_loss.'
            )
        if compile_snn_target_encoder and not isinstance(
            self.actor_target, V2SNNPolicyActor
        ):
            raise ValueError(
                'compile_snn_target_encoder requires an SNN actor.'
            )
        if compile_target_encoders and not compile_critic_encoder:
            raise ValueError(
                'compile_target_encoders requires compile_critic_encoder.'
            )
        if compile_target_block and not compile_critic_block:
            raise ValueError('compile_target_block requires compile_critic_block.')
        if frozen_critic_strategy == 'compiled_no_grad_context' and not (
            compile_critic_encoder or compile_critic_block
        ):
            raise ValueError(
                'compiled_no_grad_context requires a compiled critic path.'
            )

        options = {
            'backend': backend,
            'mode': mode,
            'fullgraph': fullgraph,
            'dynamic': dynamic,
        }
        enabled: list[str] = []
        if compile_shared_relations:
            enabled.extend(self.enable_shared_relations_compile(**options))
        if compile_actor_loss:
            enabled.extend(self.enable_actor_loss_compile(**options))
        if compile_action_inference:
            enabled.extend(self.enable_action_inference_compile(**options))
        if compile_critic_block:
            enabled.extend(self.enable_critic_loss_compile(**options))
        elif compile_critic_encoder:
            enabled.extend(self.enable_online_critic_encoder_compile(**options))
        if (
            frozen_critic_strategy == 'compiled_no_grad_context'
            and compile_critic_block
        ):
            enabled.extend(self.enable_actor_guidance_context_compile(**options))
        if compile_snn_target_encoder:
            enabled.extend(self.enable_snn_target_encoder_compile(**options))
        if compile_target_block:
            enabled.extend(self.enable_target_block_compile(**options))
        elif compile_target_encoders:
            enabled.extend(self.enable_target_encoder_compile(**options))
        if compile_actors:
            enabled.extend(self.enable_actor_compile(**options))
        self.set_frozen_critic_strategy(frozen_critic_strategy)
        self.cache_actor_loss_coefficients = cache_actor_loss_coefficients
        return {
            'enabled_objects': enabled,
            'critic_granularity': (
                'full_forward_and_loss' if compile_critic_block
                else 'encoder' if compile_critic_encoder else 'eager'
            ),
            'target_granularity': (
                'full_tensor_block' if compile_target_block
                else 'encoder' if compile_target_encoders else 'eager'
            ),
            'shared_relations_granularity': (
                'compiled_tensor_build' if compile_shared_relations else 'eager'
            ),
            'snn_target_actor_granularity': (
                'encoder' if compile_snn_target_encoder else 'eager'
            ),
            'actor_granularity': (
                'ann_full_forward_or_snn_encoder' if compile_actors else 'eager'
            ),
            'actor_loss_granularity': (
                'tensor_block' if compile_actor_loss else 'eager'
            ),
            'actor_loss_coefficient_execution': (
                'cached' if self.cache_actor_loss_coefficients else 'per_update'
            ),
            'action_inference_granularity': (
                ('ann_full_forward' if isinstance(self.actor, V2ANNPolicyActor)
                 else 'snn_encoder')
                if compile_action_inference else 'eager'
            ),
            'optimizer_execution': (
                'fused_adam' if all(
                    group.get('fused') is True
                    for optimizer in (self.actor_optimizer, self.critic_optimizer)
                    for group in optimizer.param_groups
                ) else 'adam'
            ),
            'relation_value_execution': (
                'aggregate_then_project' if self.aggregate_relation_values_first
                else 'project_then_aggregate'
            ),
            'frozen_critic_strategy': frozen_critic_strategy,
            'select_action_execution': (
                'compiled' if compile_action_inference else 'eager'
            ),
            'backend': backend,
            'mode': mode,
            'fullgraph': fullgraph,
            'dynamic': dynamic,
            'cuda_graph': False,
        }

    def _compute_twin_critic_loss_tensors(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        clean_zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        valid_token_mask: torch.Tensor,
        token_pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_arguments = (
            ego_features,
            goal_features,
            clean_zone_features,
            presence_mask,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
        )
        current_q1 = self.critic1._compute_full_forward_tensors(
            *encoder_arguments, action
        )
        current_q2 = self.critic2._compute_full_forward_tensors(
            *encoder_arguments, action
        )
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        return current_q1, current_q2, critic_loss

    def enable_critic_loss_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str, str, str]:
        if self._compiled_critic_loss is not None:
            raise RuntimeError('Twin critic loss block is already compiled.')
        if (
            self.critic1.zone_set_encoder.compiled_tensor_forward_enabled
            or self.critic2.zone_set_encoder.compiled_tensor_forward_enabled
        ):
            raise RuntimeError(
                'Full critic block forbids nested encoder compilation.'
            )
        self._compiled_critic_loss = torch.compile(
            self._compute_twin_critic_loss_tensors,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return (
            'critic1.full_forward',
            'critic2.full_forward',
            'twin_critic_loss',
        )

    def enable_actor_guidance_context_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str]:
        self.critic1.zone_set_encoder.enable_compiled_tensor_forward(
            role='critic_guidance',
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return ('critic1.actor_guidance_context',)

    def _compute_ann_target_block_tensors(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        clean_zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        valid_token_mask: torch.Tensor,
        token_pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
        noise: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_arguments = (
            ego_features,
            goal_features,
            clean_zone_features,
            presence_mask,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
        )
        next_action = self.actor_target._compute_full_forward_tensors(
            *encoder_arguments
        ) + noise
        next_action = torch.maximum(
            torch.minimum(next_action, self.action_high),
            self.action_low,
        )
        return self._compute_target_critics_td_tensors(
            *encoder_arguments,
            next_action,
            reward,
            done,
        )

    def _compute_target_critics_td_tensors(
        self,
        ego_features: torch.Tensor,
        goal_features: torch.Tensor,
        clean_zone_features: torch.Tensor,
        presence_mask: torch.Tensor,
        valid_token_mask: torch.Tensor,
        token_pair_relations: torch.Tensor,
        relation_pair_mask: torch.Tensor,
        next_action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_arguments = (
            ego_features,
            goal_features,
            clean_zone_features,
            presence_mask,
            valid_token_mask,
            token_pair_relations,
            relation_pair_mask,
        )
        target_q1 = self.critic1_target._compute_full_forward_tensors(
            *encoder_arguments, next_action
        )
        target_q2 = self.critic2_target._compute_full_forward_tensors(
            *encoder_arguments, next_action
        )
        target_q = reward + (
            (1.0 - done) * self.gamma * torch.minimum(target_q1, target_q2)
        )
        return next_action, target_q1, target_q2, target_q

    def enable_target_block_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str, str, str, str]:
        if (
            self.critic1_target.zone_set_encoder.compiled_tensor_forward_enabled
            or self.critic2_target.zone_set_encoder.compiled_tensor_forward_enabled
            or (
                isinstance(self.actor_target, V2ANNPolicyActor)
                and self.actor_target.zone_set_encoder.compiled_tensor_forward_enabled
            )
        ):
            raise RuntimeError(
                'Full target block forbids nested encoder compilation.'
            )
        if isinstance(self.actor_target, V2ANNPolicyActor):
            self._compiled_target_block = torch.compile(
                self._compute_ann_target_block_tensors,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            return (
                'actor_target.full_forward',
                'critic1_target.full_forward',
                'critic2_target.full_forward',
                'td_target',
            )
        self._compiled_target_critic_td = torch.compile(
            self._compute_target_critics_td_tensors,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return (
            (
                'actor_target.compiled_snn_encoder'
                if self.actor_target.zone_set_encoder.compiled_tensor_forward_enabled
                else 'actor_target.eager_snn'
            ),
            'critic1_target.full_forward',
            'critic2_target.full_forward',
            'td_target',
        )

    @contextmanager
    def _actor_critic_guidance(
        self,
        observation: V2ObservationBatch,
        *,
        shared_relations: ZoneSetSharedRelations | None,
        profile_sections: bool,
    ):
        if self.frozen_critic_strategy == 'eager':
            parameters = tuple(self.critic1.parameters())
            original_requires_grad = tuple(
                parameter.requires_grad for parameter in parameters
            )
            try:
                for parameter in parameters:
                    parameter.requires_grad_(False)
                with self.critic1.zone_set_encoder.eager_tensor_forward():
                    yield None
            finally:
                for parameter, required in zip(parameters, original_requires_grad):
                    parameter.requires_grad_(required)
            return

        with torch.no_grad():
            context = self.critic1.encode_context(
                observation,
                shared_relations=shared_relations,
                profile_sections=profile_sections,
            )
        head_parameters = tuple(self.critic1.head.parameters())
        original_requires_grad = tuple(
            parameter.requires_grad for parameter in head_parameters
        )
        try:
            for parameter in head_parameters:
                parameter.requires_grad_(False)
            yield context
        finally:
            for parameter, required in zip(head_parameters, original_requires_grad):
                parameter.requires_grad_(required)

    def enable_actor_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str, ...]:
        enabled: list[str] = []
        for name, actor in (
            ('actor', self.actor),
            ('bc_reference_actor', self.bc_reference_actor),
        ):
            if actor is None:
                continue
            if isinstance(actor, V2ANNPolicyActor):
                actor.enable_compiled_full_forward(
                    backend=backend,
                    mode=mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
                enabled.append(f'{name}.full_forward')
            else:
                actor.zone_set_encoder.enable_compiled_tensor_forward(
                    role='online_actor' if name == 'actor' else 'bc_reference',
                    backend=backend,
                    mode=mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic,
                )
                enabled.append(f'{name}.zone_set_encoder')
        return tuple(enabled)

    def enable_action_inference_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str, ...]:
        if self._compiled_action_inference is not None:
            raise RuntimeError('Action inference is already compiled.')
        if isinstance(self.actor, V2ANNPolicyActor):
            tensor_forward = _action_inference_ann_tensors
            enabled = 'actor.action_inference_full_forward'
        else:
            tensor_forward = _action_inference_snn_encoder_tensors
            enabled = 'actor.action_inference_encoder'
        self._compiled_action_inference = torch.compile(
            tensor_forward,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        self._compiled_action_inference_config = {
            'backend': backend,
            'mode': mode,
            'fullgraph': bool(fullgraph),
            'dynamic': bool(dynamic),
        }
        return (enabled,)

    def _action_inference_tensor(self, batch: V2ObservationBatch) -> torch.Tensor:
        if self._compiled_action_inference is None:
            actor_eager_context = (
                self.actor.eager_full_forward()
                if isinstance(self.actor, V2ANNPolicyActor)
                else self.actor.zone_set_encoder.eager_tensor_forward()
            )
            with actor_eager_context:
                return self.actor(batch)
        if isinstance(self.actor, V2ANNPolicyActor):
            with self.actor.zone_set_encoder.eager_tensor_forward():
                arguments = self.actor.zone_set_encoder.prepare_tensor_forward_arguments(
                    batch.ego_features,
                    batch.goal_features,
                    batch.zone_features,
                    batch.presence_mask,
                )
            return self._compiled_action_inference(self.actor, *arguments)
        with self.actor.reset_state_context():
            with self.actor.zone_set_encoder.eager_tensor_forward():
                arguments = self.actor.zone_set_encoder.prepare_tensor_forward_arguments(
                    batch.ego_features,
                    batch.goal_features,
                    batch.zone_features,
                    batch.presence_mask,
                )
            context = self._compiled_action_inference(
                self.actor.zone_set_encoder, *arguments,
            )
            return self.actor.action_from_context(context)

    def warmup_action_inference_compile(
        self,
        batches: Sequence[V2ObservationBatch],
    ) -> None:
        if self._compiled_action_inference is None:
            raise RuntimeError('Action inference must be compiled first.')
        warmup_batches = tuple(batches)
        if not warmup_batches:
            raise ValueError('At least one action inference warmup batch is required.')
        if any(not isinstance(batch, V2ObservationBatch) for batch in warmup_batches):
            raise TypeError('Action inference warmup batches must be V2ObservationBatch values.')
        if any(batch.batch_size != 1 for batch in warmup_batches):
            raise ValueError('Action inference warmup requires batch size 1.')
        python_rng = random.getstate()
        torch_rng = torch.random.get_rng_state()
        numpy_rng = np.random.get_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        )
        snn_memories = (
            tuple(
                (module, name, value.detach().clone() if isinstance(value, torch.Tensor)
                 else deepcopy(value))
                for module in self.actor.snn_head.modules()
                if hasattr(module, 'named_memories')
                for name, value in module.named_memories()
            )
            if isinstance(self.actor, V2SNNPolicyActor) else ()
        )
        counts_before = (
            self.update_count, self.critic_update_count,
            self.critic_target_update_count, self.actor_update_count,
            self.last_total_steps,
        )
        try:
            with torch.inference_mode():
                for batch in warmup_batches:
                    self._action_inference_tensor(batch.to(self.device))
        finally:
            random.setstate(python_rng)
            torch.random.set_rng_state(torch_rng)
            np.random.set_state(numpy_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            for module, name, value in snn_memories:
                setattr(module, name, value)
        counts_after = (
            self.update_count, self.critic_update_count,
            self.critic_target_update_count, self.actor_update_count,
            self.last_total_steps,
        )
        if counts_after != counts_before:
            raise RuntimeError('Action inference warmup must not change TD3 counters.')

    def warmup_actor_compile(
        self,
        batches: Sequence[V2ObservationBatch],
    ) -> None:
        warmup_batches = tuple(batches)
        if not warmup_batches:
            raise ValueError('At least one actor compile warmup batch is required.')
        actors = tuple(
            actor for actor in (self.actor, self.bc_reference_actor)
            if actor is not None
        )
        actor_parameters = tuple(
            parameter for actor in actors for parameter in actor.parameters()
        )
        requires_grad = tuple(
            parameter.requires_grad for parameter in actor_parameters
        )
        gradient_state = tuple(
            (
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in actor_parameters
        )
        training_modes = tuple(actor.training for actor in actors)
        torch_rng_state = torch.random.get_rng_state()
        numpy_rng_state = np.random.get_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        )
        counts_before = (
            self.update_count,
            self.critic_update_count,
            self.critic_target_update_count,
            self.actor_update_count,
            self.last_total_steps,
        )
        try:
            self.actor.train()
            for batch in warmup_batches:
                device_batch = batch.to(self.device)
                shared_relations = self._build_shared_relations(device_batch)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.actor(
                    device_batch,
                    shared_relations=shared_relations,
                ).sum().backward()
                self.actor_optimizer.zero_grad(set_to_none=True)
                if self.bc_reference_actor is not None:
                    self.bc_reference_actor.eval()
                    with torch.no_grad():
                        self.bc_reference_actor(
                            device_batch,
                            shared_relations=shared_relations,
                        )
        finally:
            for actor, training in zip(actors, training_modes):
                actor.train(training)
            for parameter, required, (original_grad, saved_grad) in zip(
                actor_parameters,
                requires_grad,
                gradient_state,
            ):
                parameter.requires_grad_(required)
                if original_grad is None:
                    parameter.grad = None
                else:
                    original_grad.copy_(saved_grad)
                    parameter.grad = original_grad
            torch.random.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        counts_after = (
            self.update_count,
            self.critic_update_count,
            self.critic_target_update_count,
            self.actor_update_count,
            self.last_total_steps,
        )
        if counts_after != counts_before:
            raise RuntimeError('Actor compile warmup must not change TD3 counters.')

    def warmup_full_compile(
        self,
        batches: Sequence[V2ObservationBatch],
    ) -> None:
        warmup_batches = tuple(batches)
        if not warmup_batches:
            raise ValueError('At least one full compile warmup batch is required.')
        if self._compiled_critic_loss is None:
            raise RuntimeError('The full critic loss block must be compiled first.')
        target_block_enabled = (
            self._compiled_target_block is not None
            or self._compiled_target_critic_td is not None
        )
        critic_parameters = tuple(self.critic1.parameters()) + tuple(
            self.critic2.parameters()
        )
        requires_grad = tuple(
            parameter.requires_grad for parameter in critic_parameters
        )
        gradient_state = tuple(
            (
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in critic_parameters
        )
        torch_rng_state = torch.random.get_rng_state()
        numpy_rng_state = np.random.get_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        )
        counts_before = (
            self.update_count,
            self.critic_update_count,
            self.critic_target_update_count,
            self.actor_update_count,
            self.last_total_steps,
        )
        try:
            for batch in warmup_batches:
                device_batch = batch.to(self.device)
                shared_relations = self._build_shared_relations(device_batch)
                encoder_arguments = (
                    self.critic1.zone_set_encoder.prepare_tensor_forward_arguments(
                        device_batch.ego_features,
                        device_batch.goal_features,
                        device_batch.zone_features,
                        device_batch.presence_mask,
                        shared_relations=shared_relations,
                    )
                )
                action = torch.zeros(
                    (device_batch.batch_size, self.action_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
                target_q = torch.zeros(
                    (device_batch.batch_size, 1),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                _, _, critic_loss = self._compiled_critic_loss(
                    *encoder_arguments,
                    action,
                    target_q,
                )
                critic_loss.backward()
                self.critic_optimizer.zero_grad(set_to_none=True)

                if target_block_enabled:
                    with torch.no_grad():
                        noise = torch.zeros_like(action)
                        reward = torch.zeros_like(target_q)
                        done = torch.zeros_like(target_q)
                        if self._compiled_target_block is not None:
                            self._compiled_target_block(
                                *encoder_arguments,
                                noise,
                                reward,
                                done,
                            )
                        else:
                            next_action = self.actor_target(
                                device_batch,
                                shared_relations=shared_relations,
                            )
                            self._compiled_target_critic_td(
                                *encoder_arguments,
                                next_action,
                                reward,
                                done,
                            )

                if self.frozen_critic_strategy == 'compiled_no_grad_context':
                    guidance_action = action.detach().clone().requires_grad_(True)
                    with self._actor_critic_guidance(
                        device_batch,
                        shared_relations=shared_relations,
                        profile_sections=False,
                    ) as context:
                        guidance_q = self.critic1.forward_from_context(
                            context,
                            guidance_action,
                        )
                    guidance_q.sum().backward()
                    if guidance_action.grad is None:
                        raise RuntimeError(
                            'Full compile warmup did not preserve action gradients.'
                        )
        finally:
            for parameter, required, (original_grad, saved_grad) in zip(
                critic_parameters,
                requires_grad,
                gradient_state,
            ):
                parameter.requires_grad_(required)
                if original_grad is None:
                    parameter.grad = None
                else:
                    original_grad.copy_(saved_grad)
                    parameter.grad = original_grad
            torch.random.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        counts_after = (
            self.update_count,
            self.critic_update_count,
            self.critic_target_update_count,
            self.actor_update_count,
            self.last_total_steps,
        )
        if counts_after != counts_before:
            raise RuntimeError('Full compile warmup must not change TD3 counters.')

    def warmup_online_critic_encoder_compile(
        self,
        batches: Sequence[V2ObservationBatch],
    ) -> None:
        warmup_batches = tuple(batches)
        if not warmup_batches:
            raise ValueError('At least one compile warmup batch is required.')
        if not (
            self.critic1.zone_set_encoder.compiled_tensor_forward_enabled
            and self.critic2.zone_set_encoder.compiled_tensor_forward_enabled
        ):
            raise RuntimeError('Both online critic encoders must be compiled first.')
        if any(not isinstance(batch, V2ObservationBatch) for batch in warmup_batches):
            raise TypeError('Compile warmup batches must be V2ObservationBatch values.')

        critic_parameters = tuple(self.critic1.parameters()) + tuple(
            self.critic2.parameters()
        )
        requires_grad = tuple(
            parameter.requires_grad for parameter in critic_parameters
        )
        gradient_state = tuple(
            (
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in critic_parameters
        )
        torch_rng_state = torch.random.get_rng_state()
        numpy_rng_state = np.random.get_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        )
        counts_before = (
            self.update_count,
            self.critic_update_count,
            self.critic_target_update_count,
            self.actor_update_count,
            self.last_total_steps,
        )
        try:
            for batch in warmup_batches:
                device_batch = batch.to(self.device)
                shared_relations = self._build_shared_relations(device_batch)
                action = torch.zeros(
                    (device_batch.batch_size, self.action_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_sum = self.critic1(
                    device_batch,
                    action,
                    shared_relations=shared_relations,
                ).sum() + self.critic2(
                    device_batch,
                    action,
                    shared_relations=shared_relations,
                ).sum()
                critic_sum.backward()
                self.critic_optimizer.zero_grad(set_to_none=True)

                frozen_action = action.detach().clone().requires_grad_(True)
                with self._actor_critic_guidance(
                    device_batch,
                    shared_relations=shared_relations,
                    profile_sections=False,
                ) as context:
                    if context is None:
                        frozen_q = self.critic1(
                            device_batch,
                            frozen_action,
                            shared_relations=shared_relations,
                        )
                    else:
                        frozen_q = self.critic1.forward_from_context(
                            context,
                            frozen_action,
                        )
                    frozen_q.sum().backward()
                if frozen_action.grad is None:
                    raise RuntimeError(
                        'Compiled frozen critic warmup did not preserve action gradients.'
                    )
        finally:
            for parameter, required, (original_grad, saved_grad) in zip(
                critic_parameters,
                requires_grad,
                gradient_state,
            ):
                parameter.requires_grad_(required)
                if original_grad is None:
                    parameter.grad = None
                else:
                    original_grad.copy_(saved_grad)
                    parameter.grad = original_grad
            torch.random.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        counts_after = (
            self.update_count,
            self.critic_update_count,
            self.critic_target_update_count,
            self.actor_update_count,
            self.last_total_steps,
        )
        if counts_after != counts_before:
            raise RuntimeError('Compile warmup must not change TD3 update counters.')

    def enable_target_encoder_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str, str, str]:
        enabled: list[str] = []
        for name, target in (
            ('actor_target.zone_set_encoder', self.actor_target),
            ('critic1_target.zone_set_encoder', self.critic1_target),
            ('critic2_target.zone_set_encoder', self.critic2_target),
        ):
            target.zone_set_encoder.enable_compiled_tensor_forward(
                role='target_actor' if target is self.actor_target else None,
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            enabled.append(name)
        return enabled[0], enabled[1], enabled[2]

    def enable_snn_target_encoder_compile(
        self,
        *,
        backend: str = 'inductor',
        mode: str = 'default',
        fullgraph: bool = True,
        dynamic: bool = True,
    ) -> tuple[str]:
        if not isinstance(self.actor_target, V2SNNPolicyActor):
            raise ValueError('SNN target encoder compilation requires an SNN actor.')
        self.actor_target.zone_set_encoder.enable_compiled_tensor_forward(
            role='target_actor',
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        return ('actor_target.zone_set_encoder',)

    def warmup_shared_relations_compile(
        self,
        batches: Sequence[V2ObservationBatch],
    ) -> None:
        warmup_batches = tuple(batches)
        if not warmup_batches:
            raise ValueError('At least one compile warmup batch is required.')
        if not self.actor.zone_set_encoder.compiled_shared_relations_enabled:
            raise RuntimeError('Shared relations must be compiled first.')
        if any(not isinstance(batch, V2ObservationBatch) for batch in warmup_batches):
            raise TypeError('Compile warmup batches must be V2ObservationBatch values.')
        torch_rng_state = torch.random.get_rng_state()
        numpy_rng_state = np.random.get_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        )
        counts_before = (
            self.update_count, self.critic_update_count,
            self.critic_target_update_count, self.actor_update_count,
            self.last_total_steps,
        )
        try:
            for batch in warmup_batches:
                self._build_shared_relations(batch.to(self.device))
        finally:
            torch.random.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        counts_after = (
            self.update_count, self.critic_update_count,
            self.critic_target_update_count, self.actor_update_count,
            self.last_total_steps,
        )
        if counts_after != counts_before:
            raise RuntimeError('Compile warmup must not change TD3 update counters.')

    def warmup_snn_target_encoder_compile(
        self,
        batches: Sequence[V2ObservationBatch],
    ) -> None:
        warmup_batches = tuple(batches)
        if not warmup_batches:
            raise ValueError('At least one compile warmup batch is required.')
        target = self.actor_target
        if not isinstance(target, V2SNNPolicyActor):
            raise ValueError('SNN target encoder warmup requires an SNN actor.')
        if not target.zone_set_encoder.compiled_tensor_forward_enabled:
            raise RuntimeError('SNN target encoder must be compiled first.')
        if any(not isinstance(batch, V2ObservationBatch) for batch in warmup_batches):
            raise TypeError('Compile warmup batches must be V2ObservationBatch values.')
        parameters = tuple(target.parameters())
        requires_grad = tuple(parameter.requires_grad for parameter in parameters)
        gradient_state = tuple(
            (parameter.grad, None if parameter.grad is None else parameter.grad.detach().clone())
            for parameter in parameters
        )
        training = target.training
        torch_rng_state = torch.random.get_rng_state()
        numpy_rng_state = np.random.get_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        )
        counts_before = (
            self.update_count, self.critic_update_count,
            self.critic_target_update_count, self.actor_update_count,
            self.last_total_steps,
        )
        try:
            target.eval()
            with torch.no_grad():
                for batch in warmup_batches:
                    device_batch = batch.to(self.device)
                    target(
                        device_batch,
                        shared_relations=self._build_shared_relations(device_batch),
                    )
        finally:
            target.train(training)
            for parameter, required, (original_grad, saved_grad) in zip(
                parameters, requires_grad, gradient_state,
            ):
                parameter.requires_grad_(required)
                if original_grad is None:
                    parameter.grad = None
                else:
                    original_grad.copy_(saved_grad)
                    parameter.grad = original_grad
            torch.random.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        counts_after = (
            self.update_count, self.critic_update_count,
            self.critic_target_update_count, self.actor_update_count,
            self.last_total_steps,
        )
        if counts_after != counts_before:
            raise RuntimeError('Compile warmup must not change TD3 update counters.')

    def warmup_target_encoder_compile(
        self,
        batches: Sequence[V2ObservationBatch],
    ) -> None:
        warmup_batches = tuple(batches)
        if not warmup_batches:
            raise ValueError('At least one compile warmup batch is required.')
        targets = (
            self.actor_target,
            self.critic1_target,
            self.critic2_target,
        )
        if not all(
            target.zone_set_encoder.compiled_tensor_forward_enabled
            for target in targets
        ):
            raise RuntimeError('All target encoders must be compiled first.')
        if any(not isinstance(batch, V2ObservationBatch) for batch in warmup_batches):
            raise TypeError('Compile warmup batches must be V2ObservationBatch values.')

        target_parameters = tuple(
            parameter for target in targets for parameter in target.parameters()
        )
        requires_grad = tuple(
            parameter.requires_grad for parameter in target_parameters
        )
        gradient_state = tuple(
            (
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in target_parameters
        )
        training_modes = tuple(target.training for target in targets)
        torch_rng_state = torch.random.get_rng_state()
        numpy_rng_state = np.random.get_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else None
        )
        counts_before = (
            self.update_count,
            self.critic_update_count,
            self.critic_target_update_count,
            self.actor_update_count,
            self.last_total_steps,
        )
        try:
            for target in targets:
                target.eval()
            with torch.no_grad():
                for batch in warmup_batches:
                    device_batch = batch.to(self.device)
                    shared_relations = self._build_shared_relations(device_batch)
                    action = self.actor_target(
                        device_batch,
                        shared_relations=shared_relations,
                    )
                    self.critic1_target(
                        device_batch,
                        action,
                        shared_relations=shared_relations,
                    )
                    self.critic2_target(
                        device_batch,
                        action,
                        shared_relations=shared_relations,
                    )
        finally:
            for target, training in zip(targets, training_modes):
                target.train(training)
            for parameter, required, (original_grad, saved_grad) in zip(
                target_parameters,
                requires_grad,
                gradient_state,
            ):
                parameter.requires_grad_(required)
                if original_grad is None:
                    parameter.grad = None
                else:
                    original_grad.copy_(saved_grad)
                    parameter.grad = original_grad
            torch.random.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        counts_after = (
            self.update_count,
            self.critic_update_count,
            self.critic_target_update_count,
            self.actor_update_count,
            self.last_total_steps,
        )
        if counts_after != counts_before:
            raise RuntimeError('Compile warmup must not change TD3 update counters.')

    @staticmethod
    def _require_finite_loss(
        loss: torch.Tensor,
        *,
        component: str,
        total_steps: int,
    ) -> None:
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(
                f'Non-finite {component} loss at total_steps={total_steps}.'
            )

    @staticmethod
    def _validate_and_clip_gradients(
        parameters: list[nn.Parameter],
        *,
        max_norm: float | None,
        component: str,
        total_steps: int,
    ) -> None:
        gradients = [
            parameter.grad
            for parameter in parameters
            if parameter.grad is not None
        ]
        if max_norm is not None:
            try:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=max_norm,
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                raise FloatingPointError(
                    f'Non-finite {component} gradient at '
                    f'total_steps={total_steps}.'
                ) from exc
            return
        if gradients:
            finite = torch.stack(
                [torch.isfinite(gradient).all() for gradient in gradients]
            ).all()
            if not bool(finite):
                raise FloatingPointError(
                    f'Non-finite {component} gradient at '
                    f'total_steps={total_steps}.'
                )

    def update_once(
        self,
        *,
        total_steps: int,
        bc_lambda: float = 0.0,
        timing_recorder: Callable[[dict[str, float]], None] | None = None,
        reuse_shared_relations: bool = True,
        profile_sections: bool = False,
        diagnostic_timing_recorder: (
            Callable[[dict[str, Any]], None] | None
        ) = None,
        diagnostic_profile_sections: bool = False,
        compiled_execution_recorder: Callable[[str], None] | None = None,
    ) -> V2TD3UpdateMetrics:
        update_timing = _OptionalUpdateWallTimer(timing_recorder)
        detail_timing = _OptionalUpdateDetailTimer(
            diagnostic_timing_recorder,
            diagnostic_profile_sections,
        )
        if (
            compiled_execution_recorder is not None
            and not callable(compiled_execution_recorder)
        ):
            raise TypeError('compiled_execution_recorder must be callable.')

        def record_compiled_execution(name: str) -> None:
            if compiled_execution_recorder is not None:
                compiled_execution_recorder(name)
        total_steps_value = _nonnegative_int(total_steps, name='total_steps')
        bc_lambda_value = _finite_float(bc_lambda, name='bc_lambda')
        if bc_lambda_value < 0.0:
            raise ValueError('bc_lambda must be non-negative.')
        if self.bc_reference_actor is None and bc_lambda_value != 0.0:
            raise ValueError('bc_lambda must be 0 when no V2 BC reference actor exists.')

        with update_timing.section('replay_sample'):
            batch = self.replay.sample(self.batch_size)
        with update_timing.section('batch_preparation'):
            batch = batch.to(self.device)
        with update_timing.section('target_forward_and_td_target'):
            with torch.no_grad():
                if (
                    reuse_shared_relations
                    and self.actor.zone_set_encoder.compiled_shared_relations_enabled
                ):
                    record_compiled_execution('shared_relations')
                next_shared_relations = (
                    self._build_shared_relations(
                        batch.next_obs,
                        profile_sections=profile_sections,
                    )
                    if reuse_shared_relations
                    else None
                )
                noise = (torch.randn_like(batch.action) * self.policy_noise).clamp(
                    -self.noise_clip,
                    self.noise_clip,
                )
                if (
                    self._compiled_target_block is not None
                    or self._compiled_target_critic_td is not None
                ):
                    target_arguments = (
                        self.actor_target.zone_set_encoder.
                        prepare_tensor_forward_arguments(
                            batch.next_obs.ego_features,
                            batch.next_obs.goal_features,
                            batch.next_obs.zone_features,
                            batch.next_obs.presence_mask,
                            shared_relations=next_shared_relations,
                        )
                    )
                    if self._compiled_target_block is not None:
                        record_compiled_execution('target_block')
                        with (
                            record_function('v2_td3.compiled.target_block')
                            if diagnostic_profile_sections
                            else nullcontext()
                        ):
                            next_action, target_q1, target_q2, target_q = (
                                self._compiled_target_block(
                                    *target_arguments,
                                    noise,
                                    batch.reward,
                                    batch.done,
                                )
                            )
                    else:
                        if self.actor_target.zone_set_encoder.compiled_tensor_forward_enabled:
                            record_compiled_execution('snn_target_encoder')
                        next_action = self.actor_target(
                            batch.next_obs,
                            shared_relations=next_shared_relations,
                            profile_sections=profile_sections,
                        ) + noise
                        next_action = torch.maximum(
                            torch.minimum(next_action, self.action_high),
                            self.action_low,
                        )
                        record_compiled_execution('target_critic_td_block')
                        with (
                            record_function(
                                'v2_td3.compiled.target_critic_td_block'
                            )
                            if diagnostic_profile_sections
                            else nullcontext()
                        ):
                            next_action, target_q1, target_q2, target_q = (
                                self._compiled_target_critic_td(
                                    *target_arguments,
                                    next_action,
                                    batch.reward,
                                    batch.done,
                                )
                            )
                else:
                    if (
                        isinstance(self.actor_target, V2SNNPolicyActor)
                        and self.actor_target.zone_set_encoder.compiled_tensor_forward_enabled
                    ):
                        record_compiled_execution('snn_target_encoder')
                    next_action = self.actor_target(
                        batch.next_obs,
                        shared_relations=next_shared_relations,
                        profile_sections=profile_sections,
                    ) + noise
                    next_action = torch.maximum(
                        torch.minimum(next_action, self.action_high),
                        self.action_low,
                    )
                    target_q1 = self.critic1_target(
                        batch.next_obs,
                        next_action,
                        shared_relations=next_shared_relations,
                        profile_sections=profile_sections,
                    )
                    target_q2 = self.critic2_target(
                        batch.next_obs,
                        next_action,
                        shared_relations=next_shared_relations,
                        profile_sections=profile_sections,
                    )
                    target_q = batch.reward + (
                        (1.0 - batch.done)
                        * self.gamma
                        * torch.minimum(target_q1, target_q2)
                    )

        with update_timing.section('online_critic_forward_and_loss'):
            if (
                reuse_shared_relations
                and self.actor.zone_set_encoder.compiled_shared_relations_enabled
            ):
                record_compiled_execution('shared_relations')
            current_shared_relations = (
                self._build_shared_relations(
                    batch.obs,
                    profile_sections=profile_sections,
                )
                if reuse_shared_relations
                else None
            )
            if self._compiled_critic_loss is not None:
                critic_arguments = (
                    self.critic1.zone_set_encoder.prepare_tensor_forward_arguments(
                        batch.obs.ego_features,
                        batch.obs.goal_features,
                        batch.obs.zone_features,
                        batch.obs.presence_mask,
                        shared_relations=current_shared_relations,
                    )
                )
                record_compiled_execution('critic_block')
                with (
                    record_function('v2_td3.compiled.critic_block')
                    if diagnostic_profile_sections
                    else nullcontext()
                ):
                    current_q1, current_q2, critic_loss = (
                        self._compiled_critic_loss(
                            *critic_arguments,
                            batch.action,
                            target_q,
                        )
                    )
            else:
                current_q1 = self.critic1(
                    batch.obs,
                    batch.action,
                    shared_relations=current_shared_relations,
                    profile_sections=profile_sections,
                )
                current_q2 = self.critic2(
                    batch.obs,
                    batch.action,
                    shared_relations=current_shared_relations,
                    profile_sections=profile_sections,
                )
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                    current_q2, target_q
                )
            self._require_finite_loss(
                critic_loss,
                component='critic',
                total_steps=total_steps_value,
            )
        with update_timing.section('critic_backward'):
            with (
                record_function('v2_td3.critic_backward')
                if profile_sections or diagnostic_profile_sections
                else nullcontext()
            ):
                with detail_timing.section('critic_zero_grad'):
                    if profile_sections:
                        with record_function('v2_td3.critic_zero_grad'):
                            self.critic_optimizer.zero_grad(set_to_none=True)
                    else:
                        self.critic_optimizer.zero_grad(set_to_none=True)
                with detail_timing.section('critic_loss_backward'):
                    if profile_sections:
                        with record_function('v2_td3.critic_loss_backward'):
                            critic_loss.backward()
                    else:
                        critic_loss.backward()
        with update_timing.section('critic_gradient_check_and_clip'):
            critic_parameters = list(self.critic1.parameters()) + list(
                self.critic2.parameters()
            )
            if profile_sections or diagnostic_profile_sections:
                with record_function('v2_td3.critic_gradient_check_and_clip'):
                    self._validate_and_clip_gradients(
                        critic_parameters,
                        max_norm=self.critic_grad_clip_norm,
                        component='critic',
                        total_steps=total_steps_value,
                    )
            else:
                self._validate_and_clip_gradients(
                    critic_parameters,
                    max_norm=self.critic_grad_clip_norm,
                    component='critic',
                    total_steps=total_steps_value,
                )
        with update_timing.section('critic_optimizer_step'):
            if profile_sections or diagnostic_profile_sections:
                with record_function('v2_td3.critic_optimizer_step'):
                    self.critic_optimizer.step()
            else:
                self.critic_optimizer.step()

        critic_targets_updated = total_steps_value % self.policy_delay == 0
        actor_updated = (
            critic_targets_updated
            and total_steps_value > self.actor_freeze_steps
        )
        if critic_targets_updated:
            with update_timing.section('target_soft_update'):
                self._soft_update(self.critic1, self.critic1_target)
                self._soft_update(self.critic2, self.critic2_target)
                self.critic_target_update_count += 1

        actor_terms: _ActorLossTerms | None = None
        if actor_updated:
            with update_timing.section('actor_update'):
                guidance = self._actor_critic_guidance(
                    batch.obs,
                    shared_relations=current_shared_relations,
                    profile_sections=profile_sections,
                )
                guidance_entered = False
                try:
                    with detail_timing.section('actor_guidance_context'):
                        critic_context = guidance.__enter__()
                        guidance_entered = True
                        if self.frozen_critic_strategy == 'compiled_no_grad_context':
                            record_compiled_execution('frozen_critic_context')
                    actor_terms = self._compute_actor_loss_terms(
                        batch.obs,
                        batch.line_to_goal_safe,
                        bc_lambda=bc_lambda_value,
                        shared_relations=current_shared_relations,
                        profile_sections=profile_sections,
                        critic_context=critic_context,
                        diagnostic_timing=detail_timing,
                        compiled_execution_recorder=compiled_execution_recorder,
                    )
                finally:
                    if guidance_entered:
                        with detail_timing.section('actor_guidance_context'):
                            guidance.__exit__(None, None, None)
                with detail_timing.section(
                    'actor_loss_composition_and_finite_check'
                ):
                    self._require_finite_loss(
                        actor_terms.actor_loss,
                        component='actor',
                        total_steps=total_steps_value,
                    )
                with detail_timing.section('actor_zero_grad'):
                    self.actor_optimizer.zero_grad(set_to_none=True)
                with detail_timing.section('actor_backward'):
                    actor_terms.actor_loss.backward()
                with detail_timing.section('actor_gradient_check_and_clip'):
                    actor_parameters = list(self.actor.parameters())
                    self._validate_and_clip_gradients(
                        actor_parameters,
                        max_norm=self.actor_grad_clip_norm,
                        component='actor',
                        total_steps=total_steps_value,
                    )
                with detail_timing.section('actor_optimizer_step'):
                    self.actor_optimizer.step()
            with update_timing.section('target_soft_update'):
                self._soft_update(self.actor, self.actor_target)
                self.actor_update_count += 1

        self.update_count += 1
        self.critic_update_count += 1
        self.last_total_steps = total_steps_value
        if actor_terms is None:
            metrics = V2TD3UpdateMetrics(
                critic_loss=float(critic_loss.item()),
                sample_success_fraction=float(batch.success.mean().item()),
                sample_near_goal_fraction=float(batch.near_goal.mean().item()),
                critic_updated=True,
                critic_targets_updated=critic_targets_updated,
                actor_updated=False,
            )
        else:
            metrics = V2TD3UpdateMetrics(
                critic_loss=float(critic_loss.item()),
                actor_loss=float(actor_terms.actor_loss.item()),
                rl_actor_loss=float(actor_terms.rl_actor_loss.item()),
                scaled_rl_actor_loss=float(
                    actor_terms.scaled_rl_actor_loss.item()
                ),
                actor_rl_scale=float(actor_terms.actor_rl_scale.item()),
                bc_loss=float(actor_terms.bc_loss.item()),
                bc_lambda=bc_lambda_value,
                terminal_geo_loss=float(actor_terms.terminal_geo_loss.item()),
                terminal_geo_lambda=(
                    self.terminal_geo_lambda
                    if self.terminal_geo_regularization_enabled
                    else 0.0
                ),
                sample_success_fraction=float(batch.success.mean().item()),
                sample_near_goal_fraction=float(batch.near_goal.mean().item()),
                critic_updated=True,
                critic_targets_updated=critic_targets_updated,
                actor_updated=True,
            )
        detail_timing.finish()
        update_timing.finish()
        return metrics

    def _actor_loss_coefficient_tensor(
        self, name: str, value: float, like: torch.Tensor,
    ) -> torch.Tensor:
        key = (value, like.device, like.dtype)
        cached = self._actor_loss_coefficient_cache.get(name)
        if cached is None or cached[0] != key:
            tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
            cached = (key, tensor)
            self._actor_loss_coefficient_cache[name] = cached
        return cached[1]

    def _compute_actor_loss_terms(
        self,
        observation: V2ObservationBatch,
        line_to_goal_safe: torch.Tensor,
        *,
        bc_lambda: float,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
        critic_context: torch.Tensor | None = None,
        diagnostic_timing: _OptionalUpdateDetailTimer | None = None,
        compiled_execution_recorder: Callable[[str], None] | None = None,
    ) -> _ActorLossTerms:
        detail = diagnostic_timing or _OptionalUpdateDetailTimer(None, False)
        with detail.section('actor_forward'):
            actor_uses_compiled = (
                not profile_sections
                and (
                    (
                        isinstance(self.actor, V2ANNPolicyActor)
                        and self.actor.compiled_full_forward_enabled
                    )
                    or (
                        isinstance(self.actor, V2SNNPolicyActor)
                        and self.actor.zone_set_encoder.compiled_tensor_forward_enabled
                    )
                )
            )
            if actor_uses_compiled and compiled_execution_recorder is not None:
                compiled_execution_recorder('actor')
            actor_actions = self.actor(
                observation,
                shared_relations=shared_relations,
                profile_sections=profile_sections,
            )
        with detail.section('actor_q_rl_loss_and_scale'):
            q_values = (
                self.critic1(
                    observation,
                    actor_actions,
                    shared_relations=shared_relations,
                    profile_sections=profile_sections,
                )
                if critic_context is None
                else self.critic1.forward_from_context(
                    critic_context,
                    actor_actions,
                )
            )
            if self._compiled_actor_loss is None:
                rl_actor_loss = -q_values.mean()
                q_scale = q_values.detach().abs().mean().clamp(min=1.0)
                actor_rl_scale = torch.as_tensor(
                    self.actor_rl_scale_alpha,
                    dtype=q_values.dtype,
                    device=q_values.device,
                ) / q_scale
                scaled_rl_actor_loss = rl_actor_loss * actor_rl_scale
        with detail.section('actor_bc_reference_forward_and_loss'):
            reference_actions = None
            if self._compiled_actor_loss is None:
                bc_loss = actor_actions.sum() * 0.0
            if self.bc_reference_actor is not None:
                with torch.no_grad():
                    reference_uses_compiled = (
                        not profile_sections
                        and (
                            (
                                isinstance(
                                    self.bc_reference_actor,
                                    V2ANNPolicyActor,
                                )
                                and self.bc_reference_actor.
                                compiled_full_forward_enabled
                            )
                            or (
                                isinstance(
                                    self.bc_reference_actor,
                                    V2SNNPolicyActor,
                                )
                                and self.bc_reference_actor.zone_set_encoder.
                                compiled_tensor_forward_enabled
                            )
                        )
                    )
                    if (
                        reference_uses_compiled
                        and compiled_execution_recorder is not None
                    ):
                        compiled_execution_recorder('bc_reference_actor')
                    reference_actions = self.bc_reference_actor(
                        observation,
                        shared_relations=shared_relations,
                        profile_sections=profile_sections,
                    )
                if self._compiled_actor_loss is None:
                    bc_loss = F.mse_loss(actor_actions, reference_actions)
        with detail.section('actor_terminal_geometry_loss'):
            if self._compiled_actor_loss is None:
                terminal_geo_loss = self._terminal_geo_loss(
                    observation, actor_actions, line_to_goal_safe,
                )
        with detail.section('actor_loss_composition_and_finite_check'):
            terminal_lambda = (
                self.terminal_geo_lambda
                if self.terminal_geo_regularization_enabled
                else 0.0
            )
            if self._compiled_actor_loss is None:
                actor_loss = (
                    scaled_rl_actor_loss
                    + bc_lambda * bc_loss
                    + terminal_lambda * terminal_geo_loss
                )
            else:
                if compiled_execution_recorder is not None:
                    compiled_execution_recorder('actor_loss')
                if self.cache_actor_loss_coefficients:
                    coefficients = (
                        self._actor_loss_coefficient_tensor(
                            'alpha', self.actor_rl_scale_alpha, q_values,
                        ),
                        self._actor_loss_coefficient_tensor('bc', bc_lambda, q_values),
                        self._actor_loss_coefficient_tensor(
                            'terminal', terminal_lambda, q_values,
                        ),
                    )
                else:
                    coefficients = (
                        torch.as_tensor(
                            self.actor_rl_scale_alpha,
                            dtype=q_values.dtype, device=q_values.device,
                        ),
                        torch.as_tensor(
                            bc_lambda, dtype=q_values.dtype, device=q_values.device,
                        ),
                        torch.as_tensor(
                            terminal_lambda, dtype=q_values.dtype,
                            device=q_values.device,
                        ),
                    )
                (
                    actor_loss, rl_actor_loss, scaled_rl_actor_loss,
                    actor_rl_scale, bc_loss, terminal_geo_loss,
                ) = self._compiled_actor_loss(
                    actor_actions, q_values, reference_actions,
                    observation.ego_features, observation.goal_features,
                    line_to_goal_safe, self.action_low, self.action_high,
                    *coefficients,
                    self.actor.scales.horizontal_span,
                    self.actor.scales.vertical_span,
                    self.actor.scales.gamma_max,
                    self.terminal_geo_radius,
                    self.terminal_geo_regularization_enabled,
                )
        return _ActorLossTerms(
            actor_loss=actor_loss,
            rl_actor_loss=rl_actor_loss,
            scaled_rl_actor_loss=scaled_rl_actor_loss,
            actor_rl_scale=actor_rl_scale,
            bc_loss=bc_loss,
            terminal_geo_loss=terminal_geo_loss,
        )

    def _build_shared_relations(
        self,
        observation: V2ObservationBatch,
        *,
        profile_sections: bool = False,
    ) -> ZoneSetSharedRelations:
        return self.actor.zone_set_encoder.build_shared_relations(
            observation.ego_features,
            observation.goal_features,
            observation.zone_features,
            observation.presence_mask,
            profile_sections=profile_sections,
        )

    def _terminal_geo_loss(
        self,
        observation: V2ObservationBatch,
        actor_actions: torch.Tensor,
        line_to_goal_safe: torch.Tensor,
    ) -> torch.Tensor:
        if not self.terminal_geo_regularization_enabled:
            return actor_actions.sum() * 0.0
        goal_forward = (
            observation.goal_features[
                :, GOAL_FEATURE_INDEX['goal_forward_norm']
            ]
            * self.actor.scales.horizontal_span
        )
        goal_right = (
            observation.goal_features[
                :, GOAL_FEATURE_INDEX['goal_right_norm']
            ]
            * self.actor.scales.horizontal_span
        )
        goal_up = (
            observation.goal_features[:, GOAL_FEATURE_INDEX['goal_up_norm']]
            * self.actor.scales.vertical_span
        )
        goal_distance = torch.sqrt(
            goal_forward.square() + goal_right.square() + goal_up.square()
        )
        gamma = (
            observation.ego_features[:, EGO_FEATURE_INDEX['gamma_fraction']]
            * self.actor.scales.gamma_max
        )
        target_gamma = torch.atan2(
            goal_up,
            torch.sqrt(goal_forward.square() + goal_right.square()),
        )
        relative_target_psi = torch.atan2(goal_right, goal_forward)
        delta_gamma = torch.maximum(
            torch.minimum(target_gamma - gamma, self.action_high[0]),
            self.action_low[0],
        )
        delta_psi = torch.maximum(
            torch.minimum(
                self._wrap_angle_tensor(relative_target_psi),
                self.action_high[1],
            ),
            self.action_low[1],
        )
        target_action = torch.stack((delta_gamma, delta_psi), dim=-1)
        squared_error = (actor_actions - target_action).square().mean(dim=-1)
        eligible = (
            (line_to_goal_safe.squeeze(-1) > 0.5)
            & (goal_distance <= self.terminal_geo_radius)
        ).to(squared_error.dtype)
        return (squared_error * eligible).sum() / eligible.sum().clamp_min(1.0)

    @staticmethod
    def _wrap_angle_tensor(value: torch.Tensor) -> torch.Tensor:
        return torch.remainder(value + pi, 2.0 * pi) - pi

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        try:
            online_parameters, target_parameters = (
                self._soft_update_parameter_pairs[(id(online), id(target))]
            )
        except KeyError as exc:
            raise ValueError(
                'Online and target modules are not a bound soft-update pair.'
            ) from exc
        with torch.no_grad():
            torch._foreach_mul_(target_parameters, 1.0 - self.tau)
            torch._foreach_add_(target_parameters, online_parameters, alpha=self.tau)

    def select_action(
        self,
        observation: V2Observation,
        *,
        exploration_noise: float = 0.0,
        exploration_rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if not isinstance(observation, V2Observation):
            raise TypeError('observation must be a V2Observation.')
        noise_scale = _finite_float(
            exploration_noise, name='exploration_noise'
        )
        if noise_scale < 0.0:
            raise ValueError('exploration_noise must be non-negative.')
        if exploration_rng is not None and not isinstance(
            exploration_rng,
            np.random.Generator,
        ):
            raise TypeError('exploration_rng must be a numpy.random.Generator.')
        batch = collate_v2_observations([observation]).to(self.device)
        with torch.inference_mode():
            action = self._action_inference_tensor(batch).detach().cpu().numpy()[0]
        if noise_scale > 0.0:
            noise_source = exploration_rng if exploration_rng is not None else np.random
            action = action + noise_source.normal(
                0.0,
                noise_scale,
                size=action.shape,
            )
        return np.clip(
            action,
            self._action_low_cpu,
            self._action_high_cpu,
        ).astype(np.float32)

    def _architecture_metadata(self) -> dict[str, Any]:
        metadata = {
            'scales': asdict(self.actor.scales),
            'encoder_config': asdict(self.actor.encoder_config),
            'uav_radius': self.actor.uav_radius,
            'actor_hidden_dim': self.actor.hidden_dim,
            'critic1_hidden_dim': self.critic1.hidden_dim,
            'critic2_hidden_dim': self.critic2.hidden_dim,
        }
        if isinstance(self.actor, V2SNNPolicyActor):
            metadata.update({
                'model_type': 'snn',
                'time_window': self.actor.time_window,
                'tau': self.actor.tau,
                'surrogate': self.actor.surrogate_name,
                'backend': self.actor.backend,
            })
        return metadata

    def _checkpoint_format(self) -> tuple[str, int]:
        if self.model_type == 'snn':
            return V2_SNN_TD3_CHECKPOINT_FORMAT, V2_SNN_TD3_CHECKPOINT_VERSION
        return V2_TD3_CHECKPOINT_FORMAT, V2_TD3_CHECKPOINT_VERSION

    def _algorithm_config(self) -> dict[str, Any]:
        return {
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'policy_delay': self.policy_delay,
            'batch_size': self.batch_size,
            'actor_freeze_steps': self.actor_freeze_steps,
            'actor_grad_clip_norm': self.actor_grad_clip_norm,
            'critic_grad_clip_norm': self.critic_grad_clip_norm,
            'actor_rl_scale_alpha': self.actor_rl_scale_alpha,
            'terminal_geo_regularization_enabled': (
                self.terminal_geo_regularization_enabled
            ),
            'terminal_geo_radius': self.terminal_geo_radius,
            'terminal_geo_lambda': self.terminal_geo_lambda,
        }

    @staticmethod
    def _observation_contract() -> dict[str, Any]:
        return {
            'id': V2_OBSERVATION_CONTRACT_ID,
            'ego_dim': EGO_FEATURE_DIM,
            'goal_dim': GOAL_FEATURE_DIM,
            'zone_dim': ZONE_FEATURE_DIM,
            'presence_mask': True,
        }

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Return the strict in-memory V2 checkpoint payload."""

        bc_reference_present = self.bc_reference_actor is not None
        checkpoint_format, checkpoint_version = self._checkpoint_format()
        payload = {
            'format': checkpoint_format,
            'format_version': checkpoint_version,
            'observation_contract': self._observation_contract(),
            'architecture': self._architecture_metadata(),
            'algorithm_config': self._algorithm_config(),
            'action_dim': self.action_dim,
            'action_low': self._action_low_cpu.tolist(),
            'action_high': self._action_high_cpu.tolist(),
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'bc_reference_present': bc_reference_present,
            'bc_reference_actor_state_dict': (
                self.bc_reference_actor.state_dict()
                if bc_reference_present
                else None
            ),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'update_counts': {
                'update_count': self.update_count,
                'critic_update_count': self.critic_update_count,
                'critic_target_update_count': self.critic_target_update_count,
                'actor_update_count': self.actor_update_count,
                'last_total_steps': self.last_total_steps,
            },
        }
        if self.model_type == 'snn':
            payload['model_type'] = 'snn'
        return payload

    @staticmethod
    def _validate_state_dict(
        model: nn.Module,
        state_dict: Any,
        *,
        name: str,
    ) -> None:
        if not isinstance(state_dict, dict):
            raise ValueError(f'{name} must be a state_dict mapping.')
        expected = model.state_dict()
        if expected.keys() != state_dict.keys():
            raise ValueError(f'{name} keys do not match the V2 architecture.')
        for key, expected_value in expected.items():
            candidate = state_dict[key]
            if not isinstance(candidate, torch.Tensor):
                raise ValueError(f'{name}[{key!r}] must be a tensor.')
            if (
                candidate.shape != expected_value.shape
                or candidate.dtype != expected_value.dtype
            ):
                raise ValueError(
                    f'{name}[{key!r}] is incompatible with the V2 architecture.'
                )

    def _validated_network_state_targets(
        self,
        payload: dict[str, Any],
    ) -> tuple[tuple[str, nn.Module], ...]:
        if not isinstance(payload, dict):
            raise TypeError('checkpoint payload must be a dict.')
        expected_format, expected_version = self._checkpoint_format()
        if payload.get('format') != expected_format:
            raise ValueError(
                f'checkpoint format must be {expected_format!r}; '
                'ANN, SNN, and old flat-observation checkpoints cannot be mixed.'
            )
        if payload.get('format_version') != expected_version:
            raise ValueError('Unsupported V2 TD3 checkpoint format_version.')
        if self.model_type == 'snn' and payload.get('model_type') != 'snn':
            raise ValueError('V2 SNN TD3 checkpoint model_type is incompatible.')
        if payload.get('observation_contract') != self._observation_contract():
            raise ValueError('checkpoint observation_contract is incompatible.')
        if payload.get('action_dim') != self.action_dim:
            raise ValueError('checkpoint action_dim is incompatible.')
        if payload.get('architecture') != self._architecture_metadata():
            raise ValueError('checkpoint encoder or head architecture is incompatible.')
        try:
            checkpoint_low = np.asarray(payload['action_low'], dtype=np.float32)
            checkpoint_high = np.asarray(payload['action_high'], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('checkpoint action range is missing or invalid.') from exc
        if not np.array_equal(
            checkpoint_low, self._action_low_cpu
        ) or not np.array_equal(
            checkpoint_high, self._action_high_cpu
        ):
            raise ValueError('checkpoint action range is incompatible.')
        state_targets = (
            ('actor_state_dict', self.actor),
            ('critic1_state_dict', self.critic1),
            ('critic2_state_dict', self.critic2),
            ('actor_target_state_dict', self.actor_target),
            ('critic1_target_state_dict', self.critic1_target),
            ('critic2_target_state_dict', self.critic2_target),
        )
        for key, model in state_targets:
            if key not in payload:
                raise ValueError(f'checkpoint is missing {key}.')
            self._validate_state_dict(model, payload[key], name=key)
            self._validate_state_dict_fixed_buffers(
                model,
                payload[key],
                name=key,
            )
        return state_targets

    def load_network_state_dicts(self, payload: dict[str, Any]) -> None:
        """Load only the six online/target networks for a new curriculum stage.

        Optimizer state, update counters, replay contents, algorithm settings,
        and the destination stage's BC reference are deliberately preserved.
        """

        state_targets = self._validated_network_state_targets(payload)
        for key, model in state_targets:
            model.load_state_dict(payload[key], strict=True)
        self._validate_all_live_fixed_buffers()
        for target in (
            self.actor_target,
            self.critic1_target,
            self.critic2_target,
        ):
            target.eval()
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        self._actor_loss_coefficient_cache.clear()

    def load_checkpoint_state_dict(self, payload: dict[str, Any]) -> None:
        """Strictly restore a V2 dynamic-set TD3 checkpoint payload."""

        state_targets = self._validated_network_state_targets(payload)
        if 'algorithm_config' not in payload:
            raise ValueError('checkpoint is missing algorithm_config.')
        if payload['algorithm_config'] != self._algorithm_config():
            raise ValueError('checkpoint algorithm_config is incompatible.')

        if 'bc_reference_present' not in payload:
            raise ValueError('checkpoint is missing bc_reference_present.')
        bc_reference_present = payload['bc_reference_present']
        if type(bc_reference_present) is not bool:
            raise ValueError('checkpoint bc_reference_present must be a bool.')
        if 'bc_reference_actor_state_dict' not in payload:
            raise ValueError(
                'checkpoint is missing bc_reference_actor_state_dict.'
            )
        bc_reference_state = payload['bc_reference_actor_state_dict']
        bc_reference_candidate: V2PolicyActor | None = None
        if bc_reference_present:
            if bc_reference_state is None:
                raise ValueError(
                    'checkpoint bc_reference_actor_state_dict must be present '
                    'when bc_reference_present is True.'
                )
            bc_reference_candidate = self._frozen_target(
                self.actor,
                device=self.device,
            )
            self._validate_state_dict(
                bc_reference_candidate,
                bc_reference_state,
                name='bc_reference_actor_state_dict',
            )
            self._validate_state_dict_fixed_buffers(
                bc_reference_candidate,
                bc_reference_state,
                name='bc_reference_actor_state_dict',
            )
            bc_reference_candidate.load_state_dict(
                bc_reference_state,
                strict=True,
            )
            self._validate_actor_architecture(
                self.actor,
                bc_reference_candidate,
                name='checkpoint bc_reference_actor',
            )
        elif bc_reference_state is not None:
            raise ValueError(
                'checkpoint bc_reference_actor_state_dict must be None '
                'when bc_reference_present is False.'
            )

        if 'actor_optimizer_state_dict' not in payload:
            raise ValueError('checkpoint is missing actor_optimizer_state_dict.')
        if 'critic_optimizer_state_dict' not in payload:
            raise ValueError('checkpoint is missing critic_optimizer_state_dict.')
        counts = payload.get('update_counts')
        if not isinstance(counts, dict):
            raise ValueError('checkpoint is missing update_counts.')
        required_counts = (
            'update_count',
            'critic_update_count',
            'critic_target_update_count',
            'actor_update_count',
            'last_total_steps',
        )
        validated_counts = {
            name: _nonnegative_int(counts.get(name), name=name)
            for name in required_counts
        }

        for key, model in state_targets:
            model.load_state_dict(payload[key], strict=True)
        for optimizer, key in (
            (self.actor_optimizer, 'actor_optimizer_state_dict'),
            (self.critic_optimizer, 'critic_optimizer_state_dict'),
        ):
            optimizer_state = deepcopy(payload[key])
            for saved_group, target_group in zip(
                optimizer_state['param_groups'], optimizer.param_groups,
            ):
                saved_group['fused'] = target_group.get('fused')
                saved_group['capturable'] = target_group.get('capturable', False)
            if not self.fused_adam:
                for state in optimizer_state['state'].values():
                    step = state.get('step')
                    if isinstance(step, torch.Tensor):
                        state['step'] = step.to(device='cpu', dtype=torch.float32)
            optimizer.load_state_dict(optimizer_state)
        self.update_count = validated_counts['update_count']
        self.critic_update_count = validated_counts['critic_update_count']
        self.critic_target_update_count = validated_counts[
            'critic_target_update_count'
        ]
        self.actor_update_count = validated_counts['actor_update_count']
        self.last_total_steps = validated_counts['last_total_steps']
        self.bc_reference_actor = bc_reference_candidate
        self._validate_all_live_fixed_buffers()
        if self.bc_reference_actor is not None:
            self._validate_model_fixed_buffers(
                self.bc_reference_actor,
                name='bc_reference_actor',
            )
        for target in (
            self.actor_target,
            self.critic1_target,
            self.critic2_target,
        ):
            target.eval()
            for parameter in target.parameters():
                parameter.requires_grad_(False)
        self._actor_loss_coefficient_cache.clear()
