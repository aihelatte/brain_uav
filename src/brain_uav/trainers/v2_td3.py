"""Standalone TD3 update validation for structured V2 observations."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import isfinite, pi
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
            self.actor.parameters(), lr=actor_lr_value
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr_value,
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

        self.update_count = 0
        self.critic_update_count = 0
        self.critic_target_update_count = 0
        self.actor_update_count = 0
        self.last_total_steps = 0

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

                critic1_parameters = tuple(self.critic1.parameters())
                for parameter in critic1_parameters:
                    parameter.requires_grad_(False)
                frozen_action = action.detach().clone().requires_grad_(True)
                try:
                    frozen_q = self.critic1(
                        device_batch,
                        frozen_action,
                        shared_relations=shared_relations,
                    )
                    frozen_q.sum().backward()
                    if frozen_action.grad is None:
                        raise RuntimeError(
                            'Compiled frozen critic warmup did not preserve action gradients.'
                        )
                finally:
                    for parameter in critic1_parameters:
                        parameter.requires_grad_(True)
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
                backend=backend,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            enabled.append(name)
        return enabled[0], enabled[1], enabled[2]

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
    ) -> V2TD3UpdateMetrics:
        update_timing = _OptionalUpdateWallTimer(timing_recorder)
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
            current_shared_relations = (
                self._build_shared_relations(
                    batch.obs,
                    profile_sections=profile_sections,
                )
                if reuse_shared_relations
                else None
            )
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
            if profile_sections:
                with record_function('v2_td3.critic_backward'):
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
            else:
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
        with update_timing.section('critic_gradient_check_and_clip'):
            critic_parameters = list(self.critic1.parameters()) + list(
                self.critic2.parameters()
            )
            if profile_sections:
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
            if profile_sections:
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
                critic1_parameters = list(self.critic1.parameters())
                critic1_requires_grad = [
                    parameter.requires_grad for parameter in critic1_parameters
                ]
            try:
                with update_timing.section('actor_update'):
                    for parameter in critic1_parameters:
                        parameter.requires_grad_(False)
                    actor_terms = self._compute_actor_loss_terms(
                        batch.obs,
                        batch.line_to_goal_safe,
                        bc_lambda=bc_lambda_value,
                        shared_relations=current_shared_relations,
                        profile_sections=profile_sections,
                    )
                    self._require_finite_loss(
                        actor_terms.actor_loss,
                        component='actor',
                        total_steps=total_steps_value,
                    )
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_terms.actor_loss.backward()
                    actor_parameters = list(self.actor.parameters())
                    self._validate_and_clip_gradients(
                        actor_parameters,
                        max_norm=self.actor_grad_clip_norm,
                        component='actor',
                        total_steps=total_steps_value,
                    )
                    self.actor_optimizer.step()
                with update_timing.section('target_soft_update'):
                    self._soft_update(self.actor, self.actor_target)
                    self.actor_update_count += 1
            finally:
                with update_timing.section('actor_update'):
                    for parameter, requires_grad in zip(
                        critic1_parameters,
                        critic1_requires_grad,
                    ):
                        parameter.requires_grad_(requires_grad)

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
        update_timing.finish()
        return metrics

    def _compute_actor_loss_terms(
        self,
        observation: V2ObservationBatch,
        line_to_goal_safe: torch.Tensor,
        *,
        bc_lambda: float,
        shared_relations: ZoneSetSharedRelations | None = None,
        profile_sections: bool = False,
    ) -> _ActorLossTerms:
        actor_actions = self.actor(
            observation,
            shared_relations=shared_relations,
            profile_sections=profile_sections,
        )
        q_values = self.critic1(
            observation,
            actor_actions,
            shared_relations=shared_relations,
            profile_sections=profile_sections,
        )
        rl_actor_loss = -q_values.mean()
        q_scale = q_values.detach().abs().mean().clamp(min=1.0)
        actor_rl_scale = torch.as_tensor(
            self.actor_rl_scale_alpha,
            dtype=q_values.dtype,
            device=q_values.device,
        ) / q_scale
        scaled_rl_actor_loss = rl_actor_loss * actor_rl_scale
        bc_loss = actor_actions.sum() * 0.0
        if self.bc_reference_actor is not None:
            with torch.no_grad():
                reference_actions = self.bc_reference_actor(
                    observation,
                    shared_relations=shared_relations,
                    profile_sections=profile_sections,
                )
            bc_loss = F.mse_loss(actor_actions, reference_actions)
        terminal_geo_loss = self._terminal_geo_loss(
            observation,
            actor_actions,
            line_to_goal_safe,
        )
        terminal_lambda = (
            self.terminal_geo_lambda
            if self.terminal_geo_regularization_enabled
            else 0.0
        )
        actor_loss = (
            scaled_rl_actor_loss
            + bc_lambda * bc_loss
            + terminal_lambda * terminal_geo_loss
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
            action = self.actor(batch).detach().cpu().numpy()[0]
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
        self.actor_optimizer.load_state_dict(payload['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(payload['critic_optimizer_state_dict'])
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
