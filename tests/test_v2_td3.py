"""Tests for the standalone structured-observation V2 TD3 update engine."""

from __future__ import annotations

from contextlib import ExitStack
from contextlib import contextmanager
import math
import random
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from unittest import mock

import numpy as np
import torch

from brain_uav.models import V2ANNCritic, V2ANNPolicyActor
from brain_uav.observations import (
    EGO_FEATURE_DIM,
    EGO_FEATURE_INDEX,
    GOAL_FEATURE_DIM,
    GOAL_FEATURE_INDEX,
    ZONE_FEATURE_DIM,
    V2Observation,
    V2ObservationScales,
    collate_v2_observations,
)
from brain_uav.trainers import (
    V2ReplayBuffer,
    V2TD3UpdateEngine,
    V2TD3UpdateMetrics,
)


def _observation(
    zone_count: int,
    *,
    offset: float = 0.0,
    goal_forward: float = 20.0,
    goal_right: float = 5.0,
    goal_up: float = 3.0,
    gamma: float = 0.1,
    scales: V2ObservationScales,
) -> V2Observation:
    ego = np.zeros(EGO_FEATURE_DIM, dtype=np.float32)
    ego[EGO_FEATURE_INDEX['uav_x_norm']] = 0.01 * offset
    ego[EGO_FEATURE_INDEX['uav_z_fraction']] = 0.4
    ego[EGO_FEATURE_INDEX['gamma_fraction']] = gamma / scales.gamma_max
    ego[EGO_FEATURE_INDEX['sin_psi']] = math.sin(0.6)
    ego[EGO_FEATURE_INDEX['cos_psi']] = math.cos(0.6)
    distance = math.sqrt(goal_forward**2 + goal_right**2 + goal_up**2)
    goal = np.zeros(GOAL_FEATURE_DIM, dtype=np.float32)
    goal[GOAL_FEATURE_INDEX['goal_forward_norm']] = goal_forward / scales.horizontal_span
    goal[GOAL_FEATURE_INDEX['goal_right_norm']] = goal_right / scales.horizontal_span
    goal[GOAL_FEATURE_INDEX['goal_up_norm']] = goal_up / scales.vertical_span
    goal[GOAL_FEATURE_INDEX['goal_distance_norm']] = distance / scales.world_diagonal
    zones = np.zeros((zone_count, ZONE_FEATURE_DIM), dtype=np.float32)
    for index in range(zone_count):
        zones[index, 0] = 1.0
        zones[index, 5] = -0.2 + 0.05 * index
        zones[index, 6] = 0.03 * ((-1.0) ** index)
        zones[index, 7] = 0.02 * index
        zones[index, 8:11] = np.array([0.05, 0.04, 0.08], dtype=np.float32)
        zones[index, 11] = 0.01
        zones[index, 12] = 0.2
        zones[index, 13] = -1.0
        zones[index, 16] = 0.5
        zones[index, 17] = float(index % 2 == 0)
        zones[index, 18] = 0.2 + 0.01 * index
    return V2Observation(ego, goal, zones, np.ones(zone_count, dtype=np.bool_))


class _OldFlatActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 2)

    def forward(self, observation):
        return self.linear(observation)


class _DeviceTrackingV2Actor(V2ANNPolicyActor):
    """V2 actor that records device-migration calls on each object copy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_call_count = 0

    def to(self, *args, **kwargs):
        self.to_call_count += 1
        return super().to(*args, **kwargs)


class TestV2TD3(unittest.TestCase):
    def setUp(self):
        self.scales = V2ObservationScales(100.0, 0.0, 50.0, math.pi / 4.0)

    def make_engine(
        self,
        *,
        policy_delay=2,
        actor_freeze_steps=0,
        gamma=0.99,
        tau=0.25,
        terminal_enabled=False,
        bc_reference_actor=None,
        batch_size=2,
        action_limit=None,
        action_low=None,
        action_high=None,
        actor_grad_clip_norm=1.0,
        critic_grad_clip_norm=1.0,
        fused_adam=False,
        aggregate_relation_values_first=False,
        device='cpu',
    ):
        if action_limit is None:
            action_limit = torch.tensor([0.2, 0.3], dtype=torch.float32)
        if action_low is None:
            action_low = np.array([-0.2, -0.3], dtype=np.float32)
        if action_high is None:
            action_high = np.array([0.2, 0.3], dtype=np.float32)
        actor = V2ANNPolicyActor(
            self.scales,
            2,
            16,
            action_limit,
        )
        critic1 = V2ANNCritic(self.scales, 2, 16)
        critic2 = V2ANNCritic(self.scales, 2, 16)
        replay = V2ReplayBuffer(32, 2, 10, success_replay_fraction=0.25)
        return V2TD3UpdateEngine(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            replay=replay,
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=gamma,
            tau=tau,
            policy_noise=0.1,
            noise_clip=0.05,
            policy_delay=policy_delay,
            batch_size=batch_size,
            action_low=action_low,
            action_high=action_high,
            actor_freeze_steps=actor_freeze_steps,
            actor_grad_clip_norm=actor_grad_clip_norm,
            critic_grad_clip_norm=critic_grad_clip_norm,
            terminal_geo_regularization_enabled=terminal_enabled,
            bc_reference_actor=bc_reference_actor,
            device=device,
            fused_adam=fused_adam,
            aggregate_relation_values_first=aggregate_relation_values_first,
        )

    def fill_replay(self, engine, counts=(2, 3), next_counts=None):
        if next_counts is None:
            next_counts = counts
        for index, (count, next_count) in enumerate(zip(counts, next_counts)):
            engine.replay.add(
                _observation(count, offset=float(index), scales=self.scales),
                np.array([0.01 * index, -0.02 * index], dtype=np.float32),
                1.0 + index,
                _observation(next_count, offset=float(index) + 0.5, scales=self.scales),
                index == len(counts) - 1,
                success=index == 0,
                near_goal=True,
                line_to_goal_safe=index % 2 == 0,
            )

    def make_bc_reference(self, value):
        reference = V2ANNPolicyActor(
            self.scales,
            2,
            16,
            torch.tensor([0.2, 0.3], dtype=torch.float32),
        )
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.fill_(value)
        return reference

    def assert_state_dict_equal(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_fused_adam_preserves_zero_none_and_momentum_state_against_adam(self):
        torch.manual_seed(743)
        eager = self.make_engine()
        torch.manual_seed(743)
        fused = self.make_engine(fused_adam=True)
        for eager_optimizer, fused_optimizer in (
            (eager.actor_optimizer, fused.actor_optimizer),
            (eager.critic_optimizer, fused.critic_optimizer),
        ):
            eager_parameters = tuple(eager_optimizer.param_groups[0]['params'])
            fused_parameters = tuple(fused_optimizer.param_groups[0]['params'])
            self.assertEqual(len(eager_parameters), len(fused_parameters))
            self.assertIs(eager_optimizer.param_groups[0].get('fused'), None)
            self.assertIs(fused_optimizer.param_groups[0]['fused'], True)
            for step in range(3):
                for left, right in zip(eager_parameters, fused_parameters):
                    left.grad = None
                    right.grad = None
                for index, gradient in (
                    (0, 0.2 if step < 2 else None),
                    (1, 0.0),
                    (2, None if step < 2 else 0.3),
                ):
                    if gradient is not None:
                        eager_parameters[index].grad = torch.full_like(
                            eager_parameters[index], gradient,
                        )
                        fused_parameters[index].grad = torch.full_like(
                            fused_parameters[index], gradient,
                        )
                eager_optimizer.step()
                fused_optimizer.step()
                for left, right in zip(eager_parameters, fused_parameters):
                    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
                    self.assertEqual(left in eager_optimizer.state, right in fused_optimizer.state)
                    if left in eager_optimizer.state:
                        for name in ('step', 'exp_avg', 'exp_avg_sq'):
                            torch.testing.assert_close(
                                eager_optimizer.state[left][name],
                                fused_optimizer.state[right][name],
                                atol=2e-6, rtol=2e-6,
                            )
                self.assertEqual(
                    int(eager_optimizer.state[eager_parameters[0]]['step']),
                    min(step + 1, 2),
                )
                self.assertEqual(
                    int(eager_optimizer.state[eager_parameters[1]]['step']),
                    step + 1,
                )

    def test_checkpoint_restore_keeps_selected_fused_adam_execution(self):
        eager = self.make_engine(policy_delay=1)
        self.fill_replay(eager)
        eager.update_once(total_steps=1)
        fused = self.make_engine(policy_delay=1, fused_adam=True)
        fused.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        self.assertTrue(fused.actor_optimizer.param_groups[0]['fused'])
        self.assertTrue(fused.critic_optimizer.param_groups[0]['fused'])
        self.assertTrue(fused.actor_optimizer.state)
        self.assertTrue(fused.critic_optimizer.state)

    def _assert_checkpoint_adam_execution_switch(self, device):
        for source_fused, target_fused in ((False, True), (True, False)):
            with self.subTest(device=device, source_fused=source_fused):
                torch.manual_seed(751)
                source = self.make_engine(fused_adam=source_fused, device=device)
                for optimizer in (source.actor_optimizer, source.critic_optimizer):
                    parameters = optimizer.param_groups[0]['params']
                    parameters[0].grad = torch.full_like(parameters[0], 0.2)
                    parameters[1].grad = torch.zeros_like(parameters[1])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                payload = source.checkpoint_state_dict()
                optimizer_keys = (
                    'actor_optimizer_state_dict', 'critic_optimizer_state_dict',
                )
                if device == 'cpu' and target_fused:
                    # Also exercise the fused loader's float32 step conversion on CPU.
                    for key in optimizer_keys:
                        for state in payload[key]['state'].values():
                            state['step'] = state['step'].to(dtype=torch.float64)
                saved_optimizers = {
                    key: deepcopy(payload[key]) for key in optimizer_keys
                }
                reference = self.make_engine(
                    fused_adam=source_fused, device=device,
                )
                reference.load_checkpoint_state_dict(
                    deepcopy(source.checkpoint_state_dict())
                )
                restored = self.make_engine(
                    fused_adam=target_fused, device=device,
                )
                restored.load_checkpoint_state_dict(payload)

                for key, left, right in zip(
                    optimizer_keys,
                    (source.actor_optimizer, source.critic_optimizer),
                    (restored.actor_optimizer, restored.critic_optimizer),
                ):
                    group = right.param_groups[0]
                    self.assertIs(group['fused'], True if target_fused else None)
                    self.assertIs(group['capturable'], False)
                    for name in ('lr', 'betas', 'eps', 'weight_decay'):
                        self.assertEqual(group[name], left.param_groups[0][name])
                    for source_parameter, restored_parameter in zip(
                        left.param_groups[0]['params'], group['params'],
                    ):
                        torch.testing.assert_close(
                            restored_parameter, source_parameter,
                            atol=0, rtol=0,
                        )
                        self.assertEqual(
                            source_parameter in left.state,
                            restored_parameter in right.state,
                        )
                        if source_parameter in left.state:
                            source_state = left.state[source_parameter]
                            restored_state = right.state[restored_parameter]
                            self.assertEqual(
                                restored_state['step'].device,
                                restored_parameter.device if target_fused
                                else torch.device('cpu'),
                            )
                            self.assertEqual(
                                restored_state['step'].dtype, torch.float32,
                            )
                            self.assertEqual(
                                int(restored_state['step']),
                                int(source_state['step']),
                            )
                            for name in ('exp_avg', 'exp_avg_sq'):
                                torch.testing.assert_close(
                                    restored_state[name], source_state[name],
                                    atol=0, rtol=0,
                                )
                    self.assertEqual(
                        payload[key]['param_groups'],
                        saved_optimizers[key]['param_groups'],
                    )
                    for index, state in payload[key]['state'].items():
                        for name in ('step', 'exp_avg', 'exp_avg_sq'):
                            torch.testing.assert_close(
                                state[name], saved_optimizers[key]['state'][index][name],
                                atol=0, rtol=0,
                            )

                    reference_optimizer = (
                        reference.actor_optimizer if key == optimizer_keys[0]
                        else reference.critic_optimizer
                    )
                    left_parameters = reference_optimizer.param_groups[0]['params']
                    right_parameters = group['params']
                    left_parameters[0].grad = torch.full_like(left_parameters[0], 0.1)
                    right_parameters[0].grad = torch.full_like(right_parameters[0], 0.1)
                    reference_optimizer.step()
                    right.step()
                    for source_parameter, restored_parameter in zip(
                        left_parameters, right_parameters,
                    ):
                        torch.testing.assert_close(
                            restored_parameter, source_parameter,
                            atol=2e-6, rtol=2e-6,
                        )
                        if source_parameter in reference_optimizer.state:
                            self.assertEqual(
                                int(right.state[restored_parameter]['step']),
                                int(reference_optimizer.state[source_parameter]['step']),
                            )
                            for name in ('exp_avg', 'exp_avg_sq'):
                                torch.testing.assert_close(
                                    right.state[restored_parameter][name],
                                    reference_optimizer.state[source_parameter][name],
                                    atol=2e-6, rtol=2e-6,
                                )
                    self.assertEqual(int(right.state[right_parameters[0]]['step']), 2)
                    self.assertEqual(int(right.state[right_parameters[1]]['step']), 1)
                    self.assertEqual(
                        payload[key]['param_groups'],
                        saved_optimizers[key]['param_groups'],
                    )
                    for index, state in payload[key]['state'].items():
                        for name in ('step', 'exp_avg', 'exp_avg_sq'):
                            torch.testing.assert_close(
                                state[name], saved_optimizers[key]['state'][index][name],
                                atol=0, rtol=0,
                            )

    def test_checkpoint_switches_adam_execution_with_nonempty_cpu_state(self):
        self._assert_checkpoint_adam_execution_switch('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_checkpoint_switches_adam_execution_with_nonempty_cuda_state(self):
        self._assert_checkpoint_adam_execution_switch('cuda')

    def test_compiled_actor_loss_matches_old_eager_terms_action_q_and_parameter_grads(self):
        for bc_reference, bc_lambda, terminal, safe_value in (
            (False, 0.0, False, 0.0),
            (True, 0.0, False, 0.0),
            (True, 1.5, True, 0.0),
            (True, 1.5, True, 1.0),
        ):
            with self.subTest(
                bc_reference=bc_reference, bc_lambda=bc_lambda,
                terminal=terminal, safe_value=safe_value,
            ):
                torch.manual_seed(757)
                reference_actor = self.make_bc_reference(0.01) if bc_reference else None
                reference = self.make_engine(
                    terminal_enabled=terminal, bc_reference_actor=reference_actor,
                )
                torch.manual_seed(757)
                compiled_actor = self.make_bc_reference(0.01) if bc_reference else None
                compiled = self.make_engine(
                    terminal_enabled=terminal, bc_reference_actor=compiled_actor,
                )
                compiled.load_checkpoint_state_dict(reference.checkpoint_state_dict())
                compiled.enable_actor_loss_compile(backend='eager', fullgraph=True)
                observation = collate_v2_observations([
                    _observation(0, scales=self.scales),
                    _observation(7, scales=self.scales),
                ])
                safe = torch.full((2, 1), safe_value)
                captures = []
                for engine in (reference, compiled):
                    recorded = {}
                    actor_forward = engine.actor.forward
                    critic_forward = engine.critic1.forward

                    def capture_actor(*args, _forward=actor_forward, _recorded=recorded, **kwargs):
                        result = _forward(*args, **kwargs)
                        result.retain_grad()
                        _recorded['action'] = result
                        return result

                    def capture_q(*args, _forward=critic_forward, _recorded=recorded, **kwargs):
                        result = _forward(*args, **kwargs)
                        result.retain_grad()
                        _recorded['q'] = result
                        return result

                    with mock.patch.object(engine.actor, 'forward', side_effect=capture_actor), mock.patch.object(
                        engine.critic1, 'forward', side_effect=capture_q,
                    ):
                        terms = engine._compute_actor_loss_terms(
                            observation, safe, bc_lambda=bc_lambda,
                        )
                        terms.actor_loss.backward()
                    recorded['terms'] = terms
                    recorded['actor_grads'] = {
                        name: None if p.grad is None else p.grad.detach().clone()
                        for name, p in engine.actor.named_parameters()
                    }
                    recorded['critic_grads'] = {
                        name: None if p.grad is None else p.grad.detach().clone()
                        for name, p in engine.critic1.named_parameters()
                    }
                    captures.append(recorded)
                old, new = captures
                for name in (
                    'actor_loss', 'rl_actor_loss', 'scaled_rl_actor_loss',
                    'actor_rl_scale', 'bc_loss', 'terminal_geo_loss',
                ):
                    torch.testing.assert_close(
                        getattr(old['terms'], name), getattr(new['terms'], name),
                        atol=2e-6, rtol=2e-6,
                    )
                for name in ('action', 'q'):
                    torch.testing.assert_close(old[name], new[name], atol=2e-6, rtol=2e-6)
                    torch.testing.assert_close(old[name].grad, new[name].grad, atol=2e-6, rtol=2e-6)
                for key in ('actor_grads', 'critic_grads'):
                    self.assertEqual(old[key].keys(), new[key].keys())
                    for name in old[key]:
                        self.assertEqual(old[key][name] is None, new[key][name] is None, name)
                        if old[key][name] is not None:
                            torch.testing.assert_close(
                                old[key][name], new[key][name], atol=2e-6, rtol=2e-6,
                            )

    def test_three_optimization_flags_report_actual_scope_and_preserve_bindings(self):
        engine = self.make_engine(
            fused_adam=True,
            aggregate_relation_values_first=True,
            bc_reference_actor=self.make_bc_reference(0.01),
        )
        model_names = (
            'actor', 'critic1', 'critic2', 'actor_target',
            'critic1_target', 'critic2_target', 'bc_reference_actor',
        )
        ids_before = {
            name: {key: id(p) for key, p in getattr(engine, name).named_parameters()}
            for name in model_names
        }
        keys_before = {
            name: tuple(getattr(engine, name).state_dict()) for name in model_names
        }
        metadata = engine.configure_compilation(compile_actor_loss=True, backend='eager')
        self.assertEqual(metadata['actor_loss_granularity'], 'tensor_block')
        self.assertEqual(metadata['optimizer_execution'], 'fused_adam')
        self.assertEqual(metadata['relation_value_execution'], 'aggregate_then_project')
        self.assertIn('actor_loss.tensor_block', metadata['enabled_objects'])
        for name in model_names:
            model = getattr(engine, name)
            self.assertTrue(all(
                layer.attention.aggregate_relation_values_first
                for layer in model.zone_set_encoder.layers
            ))
            self.assertEqual(
                {key: id(p) for key, p in model.named_parameters()}, ids_before[name],
            )
            self.assertEqual(tuple(model.state_dict()), keys_before[name])

    def test_actor_loss_compile_warmup_preserves_rng_models_optimizers_and_counts(self):
        engine = self.make_engine(
            bc_reference_actor=self.make_bc_reference(0.01), terminal_enabled=True,
        )
        engine.configure_compilation(compile_actor_loss=True, backend='eager')
        batches = tuple(collate_v2_observations([
            _observation(count, scales=self.scales),
            _observation(7, scales=self.scales),
        ]) for count in (0, 3, 7))
        model_names = (
            'actor', 'critic1', 'critic2', 'actor_target',
            'critic1_target', 'critic2_target', 'bc_reference_actor',
        )
        before = {name: deepcopy(getattr(engine, name).state_dict()) for name in model_names}
        optimizer_before = (
            deepcopy(engine.actor_optimizer.state_dict()),
            deepcopy(engine.critic_optimizer.state_dict()),
        )
        counters = (
            engine.update_count, engine.critic_update_count,
            engine.critic_target_update_count, engine.actor_update_count,
            engine.last_total_steps,
        )
        torch_rng = torch.random.get_rng_state().clone()
        numpy_rng = np.random.get_state()
        engine.warmup_actor_loss_compile(batches)
        for name in model_names:
            self.assert_state_dict_equal(getattr(engine, name).state_dict(), before[name])
        self.assertEqual(engine.actor_optimizer.state_dict(), optimizer_before[0])
        self.assertEqual(engine.critic_optimizer.state_dict(), optimizer_before[1])
        self.assertEqual(counters, (
            engine.update_count, engine.critic_update_count,
            engine.critic_target_update_count, engine.actor_update_count,
            engine.last_total_steps,
        ))
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_rng[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
        self.assertEqual(numpy_after[2:], numpy_rng[2:])

    def test_fused_adam_and_real_dynamo_eager_actor_loss_match_short_td3_updates(self):
        torch.manual_seed(761)
        reference = self.make_engine(
            policy_delay=2, terminal_enabled=True,
            bc_reference_actor=self.make_bc_reference(0.01),
        )
        torch.manual_seed(761)
        optimized = self.make_engine(
            policy_delay=2, terminal_enabled=True,
            bc_reference_actor=self.make_bc_reference(0.01), fused_adam=True,
        )
        optimized.load_checkpoint_state_dict(reference.checkpoint_state_dict())
        self.fill_replay(reference, counts=(0, 7), next_counts=(7, 0))
        fixed_batch = reference.replay.sample(2)
        reference.replay.sample = lambda batch_size: fixed_batch
        optimized.replay.sample = lambda batch_size: fixed_batch
        optimized.configure_compilation(compile_actor_loss=True, backend='eager')
        for total_steps, coefficient in ((1, 0.0), (2, 0.0), (3, 0.0), (4, 1.5)):
            update_rng = torch.random.get_rng_state()
            expected = reference.update_once(total_steps=total_steps, bc_lambda=coefficient)
            torch.random.set_rng_state(update_rng)
            actual = optimized.update_once(total_steps=total_steps, bc_lambda=coefficient)
            for name, value in asdict(expected).items():
                observed = getattr(actual, name)
                if type(value) is bool:
                    self.assertEqual(value, observed)
                elif type(value) is float:
                    torch.testing.assert_close(
                        torch.tensor(observed), torch.tensor(value),
                        atol=2e-6, rtol=2e-6,
                    )
                else:
                    self.assertEqual(value, observed)
            for model_name in (
                'actor', 'critic1', 'critic2',
                'actor_target', 'critic1_target', 'critic2_target',
            ):
                self.assert_state_dict_equal(
                    getattr(optimized, model_name).state_dict(),
                    getattr(reference, model_name).state_dict(),
                )
            for expected_optimizer, actual_optimizer in (
                (reference.actor_optimizer, optimized.actor_optimizer),
                (reference.critic_optimizer, optimized.critic_optimizer),
            ):
                expected_parameters = expected_optimizer.param_groups[0]['params']
                actual_parameters = actual_optimizer.param_groups[0]['params']
                for left, right in zip(expected_parameters, actual_parameters):
                    left_state = expected_optimizer.state.get(left)
                    right_state = actual_optimizer.state.get(right)
                    self.assertEqual(left_state is None, right_state is None)
                    if left_state is not None:
                        for name in ('step', 'exp_avg', 'exp_avg_sq'):
                            torch.testing.assert_close(
                                left_state[name], right_state[name],
                                atol=2e-6, rtol=2e-6,
                            )

    def test_actor_loss_stage_bc_coefficient_changes_without_recompilation(self):
        torch._dynamo.reset()
        graphs = []

        def counting_backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        engine = self.make_engine(
            terminal_enabled=True,
            bc_reference_actor=self.make_bc_reference(0.01),
        )
        engine.enable_actor_loss_compile(backend=counting_backend, fullgraph=True)
        observation = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        safe = torch.ones((2, 1), dtype=torch.float32)
        zero = engine._compute_actor_loss_terms(observation, safe, bc_lambda=0.0)
        nonzero = engine._compute_actor_loss_terms(observation, safe, bc_lambda=1.5)
        self.assertEqual(len(graphs), 1)
        self.assertGreater(nonzero.actor_loss.item(), zero.actor_loss.item())

    def test_compiled_actor_loss_uses_contract_derived_indices_without_runtime_lookup(self):
        class ForbiddenLookup:
            def __getitem__(self, key):
                raise AssertionError(f'contract lookup during compiled loss: {key}')

        engine = self.make_engine(terminal_enabled=True)
        engine.enable_actor_loss_compile(backend='eager', fullgraph=True)
        observation = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        actions = torch.zeros((2, 2), requires_grad=True)
        q_values = torch.ones((2, 1), requires_grad=True)
        safe = torch.ones((2, 1))
        with mock.patch(
            'brain_uav.trainers.v2_td3.GOAL_FEATURE_INDEX', ForbiddenLookup(),
        ), mock.patch(
            'brain_uav.trainers.v2_td3.EGO_FEATURE_INDEX', ForbiddenLookup(),
        ):
            total, *_ = engine._compiled_actor_loss(
                actions, q_values, None,
                observation.ego_features, observation.goal_features,
                safe, engine.action_low, engine.action_high,
                torch.tensor(engine.actor_rl_scale_alpha), torch.tensor(0.0),
                torch.tensor(engine.terminal_geo_lambda),
                engine.actor.scales.horizontal_span,
                engine.actor.scales.vertical_span,
                engine.actor.scales.gamma_max,
                engine.terminal_geo_radius, True,
            )
            total.backward()
        self.assertIsNotNone(actions.grad)
        self.assertIsNotNone(q_values.grad)

    def test_targets_are_equal_but_independent_frozen_and_not_optimized(self):
        engine = self.make_engine()
        pairs = (
            (engine.actor, engine.actor_target),
            (engine.critic1, engine.critic1_target),
            (engine.critic2, engine.critic2_target),
        )
        for online, target in pairs:
            online_parameters = dict(online.named_parameters())
            target_parameters = dict(target.named_parameters())
            self.assertEqual(online_parameters.keys(), target_parameters.keys())
            for name in online_parameters:
                self.assertIsNot(online_parameters[name], target_parameters[name])
                torch.testing.assert_close(online_parameters[name], target_parameters[name])
                self.assertFalse(target_parameters[name].requires_grad)

        actor_optimizer_ids = {
            id(parameter)
            for group in engine.actor_optimizer.param_groups
            for parameter in group['params']
        }
        critic_optimizer_ids = {
            id(parameter)
            for group in engine.critic_optimizer.param_groups
            for parameter in group['params']
        }
        self.assertEqual(actor_optimizer_ids, {id(p) for p in engine.actor.parameters()})
        self.assertEqual(
            critic_optimizer_ids,
            {id(p) for p in engine.critic1.parameters()} | {id(p) for p in engine.critic2.parameters()},
        )
        target_ids = {
            id(p)
            for model in (engine.actor_target, engine.critic1_target, engine.critic2_target)
            for p in model.parameters()
        }
        self.assertTrue(actor_optimizer_ids.isdisjoint(target_ids))
        self.assertTrue(critic_optimizer_ids.isdisjoint(target_ids))

    def test_constructor_rejects_shared_parameters_and_old_bc_actor(self):
        actor = V2ANNPolicyActor(
            self.scales, 2, 16, torch.tensor([0.2, 0.3], dtype=torch.float32)
        )
        critic = V2ANNCritic(self.scales, 2, 16)
        replay = V2ReplayBuffer(8, 2, 10)
        common = dict(
            actor=actor,
            critic1=critic,
            critic2=critic,
            replay=replay,
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.1,
            policy_noise=0.1,
            noise_clip=0.05,
            policy_delay=2,
            batch_size=2,
            action_low=np.array([-0.2, -0.3], dtype=np.float32),
            action_high=np.array([0.2, 0.3], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, 'share'):
            V2TD3UpdateEngine(**common)

        common['critic2'] = V2ANNCritic(self.scales, 2, 16)
        common['bc_reference_actor'] = _OldFlatActor()
        with self.assertRaisesRegex(TypeError, 'V2'):
            V2TD3UpdateEngine(**common)

    def test_constructor_requires_action_bounds_to_match_actor_limit(self):
        with self.assertRaisesRegex(ValueError, 'action_high.*action_limit'):
            self.make_engine(
                action_high=np.array([0.2, 0.25], dtype=np.float32)
            )
        with self.assertRaisesRegex(ValueError, 'action_low.*action_limit'):
            self.make_engine(
                action_low=np.array([-0.2, -0.25], dtype=np.float32)
            )

        engine = self.make_engine()
        torch.testing.assert_close(engine.action_high, engine.actor.action_limit)
        torch.testing.assert_close(engine.action_low, -engine.actor.action_limit)

    def test_constructor_rejects_bc_reference_with_different_action_limit(self):
        reference = V2ANNPolicyActor(
            self.scales,
            2,
            16,
            torch.tensor([0.2, 0.4], dtype=torch.float32),
        )
        with self.assertRaisesRegex(ValueError, 'bc_reference_actor.*action_limit'):
            self.make_engine(bc_reference_actor=reference)

    def test_critic_updates_on_non_delay_step_while_actor_stays_fixed(self):
        engine = self.make_engine(policy_delay=2)
        self.fill_replay(engine)
        actor_before = {name: p.detach().clone() for name, p in engine.actor.named_parameters()}
        critic_before = {name: p.detach().clone() for name, p in engine.critic1.named_parameters()}

        metrics = engine.update_once(total_steps=1)

        self.assertIsInstance(metrics, V2TD3UpdateMetrics)
        self.assertTrue(math.isfinite(metrics.critic_loss))
        self.assertTrue(metrics.critic_updated)
        self.assertFalse(metrics.actor_updated)
        self.assertFalse(metrics.critic_targets_updated)
        self.assertTrue(any(not torch.equal(critic_before[name], p) for name, p in engine.critic1.named_parameters()))
        self.assertTrue(all(torch.equal(actor_before[name], p) for name, p in engine.actor.named_parameters()))
        self.assertTrue(any(p.grad is not None for p in engine.critic1.parameters()))
        self.assertTrue(any(p.grad is not None for p in engine.critic2.parameters()))

    def test_actor_freeze_blocks_actor_but_not_delayed_critic_targets(self):
        engine = self.make_engine(policy_delay=1, actor_freeze_steps=10)
        self.fill_replay(engine)
        actor_before = {name: p.detach().clone() for name, p in engine.actor.named_parameters()}
        target_before = {
            name: p.detach().clone() for name, p in engine.critic1_target.named_parameters()
        }

        metrics = engine.update_once(total_steps=5)

        self.assertFalse(metrics.actor_updated)
        self.assertTrue(metrics.critic_targets_updated)
        self.assertTrue(all(torch.equal(actor_before[name], p) for name, p in engine.actor.named_parameters()))
        self.assertTrue(any(not torch.equal(target_before[name], p) for name, p in engine.critic1_target.named_parameters()))

    def test_formal_actor_freeze_boundary_includes_step_25000(self):
        frozen = self.make_engine(policy_delay=1, actor_freeze_steps=25_000)
        self.fill_replay(frozen)
        self.assertFalse(frozen.update_once(total_steps=25_000, bc_lambda=0.0).actor_updated)

        released = self.make_engine(policy_delay=1, actor_freeze_steps=25_000)
        self.fill_replay(released)
        self.assertTrue(released.update_once(total_steps=25_001, bc_lambda=0.0).actor_updated)

    def test_eligible_actor_update_changes_actor_encoder_and_target(self):
        engine = self.make_engine(policy_delay=1, actor_freeze_steps=0)
        self.fill_replay(engine, counts=(3, 4))
        encoder_before = {
            name: p.detach().clone()
            for name, p in engine.actor.zone_set_encoder.named_parameters()
        }
        target_before = {
            name: p.detach().clone() for name, p in engine.actor_target.named_parameters()
        }

        metrics = engine.update_once(total_steps=1)

        self.assertTrue(metrics.actor_updated)
        self.assertTrue(math.isfinite(metrics.actor_loss))
        self.assertTrue(
            any(
                not torch.equal(encoder_before[name], parameter)
                for name, parameter in engine.actor.zone_set_encoder.named_parameters()
            )
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in engine.actor.zone_set_encoder.parameters())
        )
        self.assertTrue(any(not torch.equal(target_before[name], p) for name, p in engine.actor_target.named_parameters()))

    def test_nonfinite_critic_loss_fails_before_backward_and_optimizer_step(self):
        for clip_norm in (1.0, None):
            with self.subTest(critic_grad_clip_norm=clip_norm):
                engine = self.make_engine(
                    policy_delay=2,
                    critic_grad_clip_norm=clip_norm,
                )
                self.fill_replay(engine)

                def nonfinite_loss(current, target):
                    del target
                    return current.sum() * torch.as_tensor(float('nan'))

                with mock.patch(
                    'brain_uav.trainers.v2_td3.F.mse_loss',
                    side_effect=nonfinite_loss,
                ), mock.patch.object(
                    engine.critic_optimizer, 'step', wraps=engine.critic_optimizer.step
                ) as optimizer_step:
                    with self.assertRaisesRegex(
                        FloatingPointError, 'critic loss.*total_steps=1'
                    ):
                        engine.update_once(total_steps=1)
                optimizer_step.assert_not_called()
                self.assertTrue(
                    all(parameter.grad is None for parameter in engine.critic1.parameters())
                )

    def test_nonfinite_critic_gradient_fails_before_step_with_and_without_clipping(self):
        for clip_norm in (1.0, None):
            with self.subTest(critic_grad_clip_norm=clip_norm):
                engine = self.make_engine(
                    policy_delay=2,
                    critic_grad_clip_norm=clip_norm,
                )
                self.fill_replay(engine)
                parameter = next(engine.critic1.parameters())
                handle = parameter.register_hook(
                    lambda gradient: torch.full_like(gradient, float('inf'))
                )
                try:
                    with mock.patch.object(
                        engine.critic_optimizer,
                        'step',
                        wraps=engine.critic_optimizer.step,
                    ) as optimizer_step:
                        with self.assertRaisesRegex(
                            FloatingPointError, 'critic gradient.*total_steps=1'
                        ):
                            engine.update_once(total_steps=1)
                    optimizer_step.assert_not_called()
                finally:
                    handle.remove()

    def test_nonfinite_actor_loss_and_gradient_fail_before_actor_step(self):
        for failure_kind in ('loss', 'gradient'):
            for clip_norm in (1.0, None):
                with self.subTest(
                    failure_kind=failure_kind,
                    actor_grad_clip_norm=clip_norm,
                ):
                    engine = self.make_engine(
                        policy_delay=1,
                        actor_grad_clip_norm=clip_norm,
                    )
                    self.fill_replay(engine)
                    original_compute = engine._compute_actor_loss_terms
                    context = mock.patch.object(
                        engine.actor_optimizer,
                        'step',
                        wraps=engine.actor_optimizer.step,
                    )
                    gradient_handle = None
                    if failure_kind == 'loss':
                        def nonfinite_actor_terms(*args, **kwargs):
                            terms = original_compute(*args, **kwargs)
                            return replace(
                                terms,
                                actor_loss=(
                                    terms.actor_loss
                                    * torch.as_tensor(float('nan'))
                                ),
                            )

                        failure_context = mock.patch.object(
                            engine,
                            '_compute_actor_loss_terms',
                            side_effect=nonfinite_actor_terms,
                        )
                        expected = 'actor loss.*total_steps=1'
                    else:
                        parameter = next(engine.actor.parameters())
                        gradient_handle = parameter.register_hook(
                            lambda gradient: torch.full_like(
                                gradient, float('inf')
                            )
                        )
                        failure_context = mock.patch.object(
                            engine,
                            '_compute_actor_loss_terms',
                            wraps=original_compute,
                        )
                        expected = 'actor gradient.*total_steps=1'
                    try:
                        with failure_context, context as optimizer_step:
                            with self.assertRaisesRegex(
                                FloatingPointError, expected
                            ):
                                engine.update_once(total_steps=1)
                        optimizer_step.assert_not_called()
                    finally:
                        if gradient_handle is not None:
                            gradient_handle.remove()

    def test_actor_loss_preserves_action_gradient_without_critic_parameter_grads(self):
        engine = self.make_engine(policy_delay=1)
        observation = collate_v2_observations([
            _observation(2, scales=self.scales),
            _observation(3, offset=1.0, scales=self.scales),
        ])
        safe = torch.ones((2, 1), dtype=torch.float32)

        engine.actor_optimizer.zero_grad(set_to_none=True)
        engine.critic_optimizer.zero_grad(set_to_none=True)
        baseline = engine._compute_actor_loss_terms(
            observation, safe, bc_lambda=0.0
        )
        baseline.actor_loss.backward()
        expected_actor_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in engine.actor.named_parameters()
            if parameter.grad is not None
        }
        self.assertTrue(expected_actor_grads)

        engine.actor_optimizer.zero_grad(set_to_none=True)
        engine.critic_optimizer.zero_grad(set_to_none=True)
        original_states = [
            parameter.requires_grad for parameter in engine.critic1.parameters()
        ]
        try:
            for parameter in engine.critic1.parameters():
                parameter.requires_grad_(False)
            frozen = engine._compute_actor_loss_terms(
                observation, safe, bc_lambda=0.0
            )
            frozen.actor_loss.backward()
        finally:
            for parameter, original in zip(
                engine.critic1.parameters(), original_states
            ):
                parameter.requires_grad_(original)

        for name, expected in expected_actor_grads.items():
            torch.testing.assert_close(
                dict(engine.actor.named_parameters())[name].grad,
                expected,
                rtol=1e-6,
                atol=1e-7,
            )
        self.assertTrue(
            all(parameter.grad is None for parameter in engine.critic1.parameters())
        )

    def test_actor_update_restores_critic_requires_grad_on_success_and_failure(self):
        successful = self.make_engine(policy_delay=1)
        self.fill_replay(successful)
        success_states = [
            parameter.requires_grad for parameter in successful.critic1.parameters()
        ]
        successful.update_once(total_steps=1)
        self.assertEqual(
            [parameter.requires_grad for parameter in successful.critic1.parameters()],
            success_states,
        )

        failing = self.make_engine(policy_delay=1)
        self.fill_replay(failing)
        failure_states = [
            parameter.requires_grad for parameter in failing.critic1.parameters()
        ]
        with mock.patch.object(
            failing,
            '_compute_actor_loss_terms',
            side_effect=RuntimeError('controlled actor failure'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'controlled actor failure'):
                failing.update_once(total_steps=1)
        self.assertEqual(
            [parameter.requires_grad for parameter in failing.critic1.parameters()],
            failure_states,
        )

    def test_named_soft_update_covers_parameters_without_rechecking_fixed_buffers(self):
        engine = self.make_engine(tau=0.25)
        names = (
            'zone_set_encoder.empty_scene_token',
            'zone_set_encoder.layers.0.attention.relation_bias.weight',
            'zone_set_encoder.layers.0.attention.relation_value.weight',
            'head.0.weight',
        )
        online = dict(engine.actor.named_parameters())
        target = dict(engine.actor_target.named_parameters())
        before = {name: target[name].detach().clone() for name in names}
        with torch.no_grad():
            for name in names:
                online[name].add_(2.0)

        with mock.patch.object(
            torch,
            'equal',
            side_effect=AssertionError('soft update must not compare fixed buffers'),
        ):
            engine._soft_update(engine.actor, engine.actor_target)

        for name in names:
            expected = before[name] * 0.75 + online[name] * 0.25
            torch.testing.assert_close(target[name], expected)

    def test_soft_update_batches_cached_parameter_pairs_and_survives_checkpoint_load(self):
        engine = self.make_engine(tau=0.25)

        def expected_after_update(online, target):
            before = {
                name: parameter.detach().clone()
                for name, parameter in target.named_parameters()
            }
            with torch.no_grad():
                for parameter in online.parameters():
                    parameter.add_(0.5)
            return {
                name: before[name] * 0.75 + parameter.detach() * 0.25
                for name, parameter in online.named_parameters()
            }

        expected = []
        pairs = (
            (engine.actor, engine.actor_target),
            (engine.critic1, engine.critic1_target),
            (engine.critic2, engine.critic2_target),
        )
        for online, target in pairs:
            expected.append(expected_after_update(online, target))
        with mock.patch.object(
            torch, '_foreach_mul_', wraps=torch._foreach_mul_,
        ) as multiply, mock.patch.object(
            torch, '_foreach_add_', wraps=torch._foreach_add_,
        ) as add:
            for online, target in pairs:
                engine._soft_update(online, target)
        self.assertEqual(multiply.call_count, 3)
        self.assertEqual(add.call_count, 3)
        for (_, target), reference in zip(pairs, expected):
            for name, parameter in target.named_parameters():
                torch.testing.assert_close(parameter, reference[name])

        restored = self.make_engine(tau=0.25)
        restored.load_checkpoint_state_dict(engine.checkpoint_state_dict())
        reference = expected_after_update(restored.actor, restored.actor_target)
        restored._soft_update(restored.actor, restored.actor_target)
        for name, parameter in restored.actor_target.named_parameters():
            torch.testing.assert_close(parameter, reference[name])

    def test_update_reuses_current_and_next_relations_once_each(self):
        engine = self.make_engine(policy_delay=1)
        self.fill_replay(engine)
        encoders = (
            engine.actor.zone_set_encoder,
            engine.actor_target.zone_set_encoder,
            engine.critic1.zone_set_encoder,
            engine.critic2.zone_set_encoder,
            engine.critic1_target.zone_set_encoder,
            engine.critic2_target.zone_set_encoder,
        )
        with ExitStack() as stack:
            shared_builder = stack.enter_context(mock.patch.object(
                encoders[0].pair_relation_builder,
                'compute_relations',
                wraps=encoders[0].pair_relation_builder.compute_relations,
            ))
            other_builders = [
                stack.enter_context(mock.patch.object(
                    encoder.pair_relation_builder,
                    'compute_relations',
                    wraps=encoder.pair_relation_builder.compute_relations,
                ))
                for encoder in encoders[1:]
            ]
            engine.update_once(total_steps=1)

        self.assertEqual(shared_builder.call_count, 2)
        self.assertEqual(sum(spy.call_count for spy in other_builders), 0)

    def test_ann_select_action_bypasses_compiled_shared_relations(self):
        engine = self.make_engine()
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_shared_relations_compile()
        encoder = engine.actor.zone_set_encoder
        compiled = mock.Mock(wraps=encoder._compiled_shared_relations)
        encoder._compiled_shared_relations = compiled

        engine.select_action(_observation(7, scales=self.scales))
        self.assertEqual(compiled.call_count, 0)
        batch = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        engine._build_shared_relations(batch)
        self.assertEqual(compiled.call_count, 1)

    def test_compile_enables_only_online_critic_encoders_and_preserves_training_bindings(self):
        engine = self.make_engine(policy_delay=1)
        self.fill_replay(engine)
        modules = {
            name: getattr(engine, name)
            for name in (
                'actor', 'actor_target', 'critic1', 'critic2',
                'critic1_target', 'critic2_target',
            )
        }
        parameter_ids = {
            name: tuple(id(parameter) for parameter in module.parameters())
            for name, module in modules.items()
        }
        state_keys = {
            name: tuple(module.state_dict()) for name, module in modules.items()
        }
        soft_update_binding_ids = {
            key: (
                tuple(id(parameter) for parameter in online),
                tuple(id(parameter) for parameter in target),
            )
            for key, (online, target) in engine._soft_update_parameter_pairs.items()
        }
        batch = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        counts_before = (
            engine.update_count,
            engine.critic_update_count,
            engine.critic_target_update_count,
            engine.actor_update_count,
        )

        enabled = engine.enable_online_critic_encoder_compile(backend='eager')
        engine.warmup_online_critic_encoder_compile((batch,))

        self.assertEqual(enabled, (
            'critic1.zone_set_encoder',
            'critic2.zone_set_encoder',
        ))
        self.assertTrue(engine.critic1.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertTrue(engine.critic2.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertFalse(engine.actor.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertFalse(engine.actor_target.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertFalse(engine.critic1_target.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertFalse(engine.critic2_target.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertEqual(counts_before, (
            engine.update_count,
            engine.critic_update_count,
            engine.critic_target_update_count,
            engine.actor_update_count,
        ))
        for name, module in modules.items():
            self.assertEqual(
                tuple(id(parameter) for parameter in module.parameters()),
                parameter_ids[name],
            )
            self.assertEqual(tuple(module.state_dict()), state_keys[name])
        self.assertEqual({
            key: (
                tuple(id(parameter) for parameter in online),
                tuple(id(parameter) for parameter in target),
            )
            for key, (online, target) in engine._soft_update_parameter_pairs.items()
        }, soft_update_binding_ids)
        self.assertTrue(all(parameter.requires_grad for parameter in engine.critic1.parameters()))

        metrics = engine.update_once(total_steps=1)
        self.assertTrue(metrics.actor_updated)
        self.assertTrue(all(parameter.requires_grad for parameter in engine.critic1.parameters()))
        engine.update_once(total_steps=2)

    def test_compiled_critic_uses_eager_only_while_frozen_for_actor(self):
        engine = self.make_engine(policy_delay=1)
        self.fill_replay(engine)
        observation = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(3, offset=1.0, scales=self.scales),
        ])
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_online_critic_encoder_compile(backend='eager')
        encoder = engine.critic1.zone_set_encoder
        compiled = mock.Mock(wraps=encoder._compiled_tensor_forward)
        encoder._compiled_tensor_forward = compiled
        original_requires_grad = tuple(
            parameter.requires_grad for parameter in engine.critic1.parameters()
        )

        with mock.patch.object(
            encoder,
            '_compute_policy_context_tensors',
            wraps=encoder._compute_policy_context_tensors,
        ) as eager:
            engine.warmup_online_critic_encoder_compile((observation,))
            self.assertGreater(compiled.call_count, 0)
            self.assertGreater(eager.call_count, 0)
            compiled.reset_mock()
            eager.reset_mock()

            metrics = engine.update_once(total_steps=1, bc_lambda=0.0)
            self.assertTrue(metrics.actor_updated)
            self.assertEqual(compiled.call_count, 1)
            self.assertEqual(eager.call_count, 1)
            self.assertTrue(any(
                parameter.grad is not None
                and bool(torch.count_nonzero(parameter.grad))
                for parameter in engine.actor.parameters()
            ))
            self.assertEqual(
                tuple(parameter.requires_grad for parameter in engine.critic1.parameters()),
                original_requires_grad,
            )

            with mock.patch.object(
                engine,
                '_compute_actor_loss_terms',
                side_effect=RuntimeError('controlled actor failure'),
            ):
                with self.assertRaisesRegex(RuntimeError, 'controlled actor failure'):
                    engine.update_once(total_steps=2, bc_lambda=0.0)
            self.assertEqual(
                tuple(parameter.requires_grad for parameter in engine.critic1.parameters()),
                original_requires_grad,
            )
            before = compiled.call_count
            engine.critic1(
                observation,
                torch.zeros((observation.batch_size, engine.action_dim)),
            )
            self.assertEqual(compiled.call_count, before + 1)

    def test_ann_online_and_bc_actor_compile_while_select_action_stays_eager(self):
        engine = self.make_engine(
            policy_delay=1,
            bc_reference_actor=self.make_bc_reference(0.01),
        )
        self.fill_replay(engine, counts=(0, 7))
        batch = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        with mock.patch(
            'brain_uav.models.v2_ann.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            enabled = engine.enable_actor_compile()
        self.assertEqual(enabled, (
            'actor.full_forward',
            'bc_reference_actor.full_forward',
        ))
        self.assertTrue(engine.actor.compiled_full_forward_enabled)
        self.assertTrue(engine.bc_reference_actor.compiled_full_forward_enabled)
        self.assertFalse(engine.actor_target.compiled_full_forward_enabled)
        engine.warmup_actor_compile((batch,))

        compiled = mock.Mock(wraps=engine.actor._compiled_full_forward)
        engine.actor._compiled_full_forward = compiled
        engine.select_action(_observation(7, scales=self.scales))
        self.assertEqual(compiled.call_count, 0)
        with torch.no_grad():
            engine.actor(batch)
        self.assertEqual(compiled.call_count, 0)
        metrics = engine.update_once(total_steps=1, bc_lambda=1.5)
        self.assertTrue(metrics.actor_updated)
        self.assertEqual(compiled.call_count, 1)
        self.assertTrue(all(
            not parameter.requires_grad
            and parameter.grad is None
            for parameter in engine.bc_reference_actor.parameters()
        ))

    def test_compiled_no_grad_context_strategy_preserves_actor_rl_gradient(self):
        engine = self.make_engine(policy_delay=1)
        self.fill_replay(engine, counts=(0, 7))
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_online_critic_encoder_compile(backend='eager')
        engine.set_frozen_critic_strategy('compiled_no_grad_context')
        encoder_observations = []
        head_observations = []

        def observe_encoder(_module, _inputs):
            encoder_observations.append((
                torch.is_grad_enabled(),
                tuple(
                    parameter.requires_grad
                    for parameter in engine.critic1.zone_set_encoder.parameters()
                ),
            ))

        def observe_head(_module, inputs):
            head_observations.append((
                bool(inputs[0].requires_grad),
                tuple(parameter.requires_grad for parameter in engine.critic1.head.parameters()),
            ))

        encoder_hook = engine.critic1.zone_set_encoder.register_forward_pre_hook(
            observe_encoder
        )
        head_hook = engine.critic1.head.register_forward_pre_hook(observe_head)
        critic_hook_counts = [0 for _ in engine.critic1.parameters()]
        gradient_hooks = []
        for index, parameter in enumerate(engine.critic1.parameters()):
            def count_gradient(gradient, slot=index):
                critic_hook_counts[slot] += 1
                return gradient
            gradient_hooks.append(parameter.register_hook(count_gradient))
        try:
            metrics = engine.update_once(total_steps=1, bc_lambda=0.0)
        finally:
            encoder_hook.remove()
            head_hook.remove()
            for handle in gradient_hooks:
                handle.remove()

        self.assertTrue(metrics.actor_updated)
        self.assertTrue(any(
            not grad_enabled and all(requires)
            for grad_enabled, requires in encoder_observations
        ))
        self.assertTrue(any(
            input_requires_grad and not any(head_requires_grad)
            for input_requires_grad, head_requires_grad in head_observations
        ))
        self.assertTrue(all(count <= 1 for count in critic_hook_counts))
        self.assertTrue(any(
            parameter.grad is not None
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in engine.actor.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad for parameter in engine.critic1.parameters()
        ))

        with mock.patch.object(
            engine,
            '_compute_actor_loss_terms',
            side_effect=RuntimeError('controlled guidance failure'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'controlled guidance failure'):
                engine.update_once(total_steps=2, bc_lambda=0.0)
        self.assertTrue(all(
            parameter.requires_grad for parameter in engine.critic1.parameters()
        ))

    def test_compiled_no_grad_context_strategy_requires_compiled_encoder(self):
        engine = self.make_engine()
        with self.assertRaisesRegex(RuntimeError, 'compiled critic1 encoder'):
            engine.set_frozen_critic_strategy('compiled_no_grad_context')
        with self.assertRaisesRegex(ValueError, 'frozen critic strategy'):
            engine.set_frozen_critic_strategy('unknown')

    def test_full_compiled_blocks_match_eager_three_update_chain(self):
        def assert_optimizer_equal(actual, expected):
            self.assertEqual(actual['param_groups'], expected['param_groups'])
            self.assertEqual(actual['state'].keys(), expected['state'].keys())
            for parameter_id, expected_state in expected['state'].items():
                self.assertEqual(
                    actual['state'][parameter_id].keys(),
                    expected_state.keys(),
                )
                for name, expected_value in expected_state.items():
                    torch.testing.assert_close(
                        actual['state'][parameter_id][name], expected_value
                    )

        torch.manual_seed(2468)
        eager = self.make_engine(
            terminal_enabled=True,
            bc_reference_actor=self.make_bc_reference(0.02),
        )
        torch.manual_seed(9753)
        compiled = self.make_engine(
            terminal_enabled=True,
            bc_reference_actor=self.make_bc_reference(-0.03),
        )
        compiled.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        self.fill_replay(eager, counts=(0, 7), next_counts=(7, 0))
        batch = eager.replay.sample(2)
        eager.replay.sample = lambda batch_size: batch
        compiled.replay.sample = lambda batch_size: batch
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.v2_ann.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            self.assertEqual(compiled.enable_critic_loss_compile(), (
                'critic1.full_forward',
                'critic2.full_forward',
                'twin_critic_loss',
            ))
            self.assertEqual(compiled.enable_target_block_compile(), (
                'actor_target.full_forward',
                'critic1_target.full_forward',
                'critic2_target.full_forward',
                'td_target',
            ))
            compiled.enable_actor_compile()
            compiled.enable_actor_guidance_context_compile()
        compiled.set_frozen_critic_strategy('compiled_no_grad_context')

        actor_metrics = None
        for total_steps, bc_lambda in ((1, 0.0), (2, 1.5), (3, 0.0)):
            update_rng = torch.random.get_rng_state()
            eager_metrics = eager.update_once(
                total_steps=total_steps,
                bc_lambda=bc_lambda,
            )
            torch.random.set_rng_state(update_rng)
            compiled_metrics = compiled.update_once(
                total_steps=total_steps,
                bc_lambda=bc_lambda,
            )
            self.assertEqual(eager_metrics, compiled_metrics)
            if eager_metrics.actor_updated:
                actor_metrics = eager_metrics
            for model_name in (
                'actor', 'critic1', 'critic2',
                'actor_target', 'critic1_target', 'critic2_target',
            ):
                self.assert_state_dict_equal(
                    getattr(compiled, model_name).state_dict(),
                    getattr(eager, model_name).state_dict(),
                )
            assert_optimizer_equal(
                compiled.actor_optimizer.state_dict(),
                eager.actor_optimizer.state_dict(),
            )
            assert_optimizer_equal(
                compiled.critic_optimizer.state_dict(),
                eager.critic_optimizer.state_dict(),
            )
        self.assertIsNotNone(actor_metrics)
        self.assertEqual(actor_metrics.bc_lambda, 1.5)
        self.assertGreater(actor_metrics.terminal_geo_loss, 0.0)

    def test_full_blocks_reject_nested_encoder_compile(self):
        engine = self.make_engine()
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_online_critic_encoder_compile(backend='eager')
        with self.assertRaisesRegex(RuntimeError, 'nested encoder compilation'):
            engine.enable_critic_loss_compile(backend='eager')

        target_engine = self.make_engine()
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            target_engine.enable_target_encoder_compile(backend='eager')
        with self.assertRaisesRegex(RuntimeError, 'nested encoder compilation'):
            target_engine.enable_target_block_compile(backend='eager')

    def test_full_compiled_blocks_prepare_shared_tensor_arguments_once(self):
        engine = self.make_engine(policy_delay=2)
        self.fill_replay(engine)
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_critic_loss_compile(backend='eager')
            engine.enable_target_block_compile(backend='eager')

        encoders = {
            'critic1': engine.critic1.zone_set_encoder,
            'critic2': engine.critic2.zone_set_encoder,
            'actor_target': engine.actor_target.zone_set_encoder,
            'critic1_target': engine.critic1_target.zone_set_encoder,
            'critic2_target': engine.critic2_target.zone_set_encoder,
        }
        with ExitStack() as stack:
            prepared = {
                name: stack.enter_context(mock.patch.object(
                    encoder,
                    'prepare_tensor_forward_arguments',
                    wraps=encoder.prepare_tensor_forward_arguments,
                ))
                for name, encoder in encoders.items()
            }
            engine.update_once(total_steps=1)

        self.assertEqual(prepared['critic1'].call_count, 1)
        self.assertEqual(prepared['critic2'].call_count, 0)
        self.assertEqual(prepared['actor_target'].call_count, 1)
        self.assertEqual(prepared['critic1_target'].call_count, 0)
        self.assertEqual(prepared['critic2_target'].call_count, 0)

    def test_full_compile_warmup_preserves_parameters_rng_optimizers_and_counts(self):
        engine = self.make_engine(
            bc_reference_actor=self.make_bc_reference(0.02),
        )
        batch = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.v2_ann.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_critic_loss_compile()
            engine.enable_target_block_compile()
            engine.enable_actor_compile()
            engine.enable_actor_guidance_context_compile()
        engine.set_frozen_critic_strategy('compiled_no_grad_context')
        compiled_critic = mock.Mock(wraps=engine._compiled_critic_loss)
        compiled_target = mock.Mock(wraps=engine._compiled_target_block)
        engine._compiled_critic_loss = compiled_critic
        engine._compiled_target_block = compiled_target
        state_before = {
            name: deepcopy(getattr(engine, name).state_dict())
            for name in (
                'actor', 'critic1', 'critic2',
                'actor_target', 'critic1_target', 'critic2_target',
                'bc_reference_actor',
            )
        }
        counts_before = (
            engine.update_count,
            engine.critic_update_count,
            engine.critic_target_update_count,
            engine.actor_update_count,
            engine.last_total_steps,
        )
        torch.manual_seed(24680)
        np.random.seed(13579)
        torch_rng = torch.random.get_rng_state().clone()
        numpy_rng = np.random.get_state()

        engine.warmup_actor_compile((batch,))
        engine.warmup_full_compile((batch,))

        self.assertGreater(compiled_critic.call_count, 0)
        self.assertGreater(compiled_target.call_count, 0)

        for name, expected in state_before.items():
            self.assert_state_dict_equal(getattr(engine, name).state_dict(), expected)
        self.assertEqual(counts_before, (
            engine.update_count,
            engine.critic_update_count,
            engine.critic_target_update_count,
            engine.actor_update_count,
            engine.last_total_steps,
        ))
        self.assertFalse(engine.actor_optimizer.state_dict()['state'])
        self.assertFalse(engine.critic_optimizer.state_dict()['state'])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_rng[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
        self.assertEqual(numpy_after[2:], numpy_rng[2:])

    def test_full_critic_only_compile_warmup_does_not_require_target_block(self):
        engine = self.make_engine()
        batch = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            metadata = engine.configure_compilation(compile_critic_block=True)
        state_before = {
            name: deepcopy(getattr(engine, name).state_dict())
            for name in (
                'actor', 'critic1', 'critic2',
                'actor_target', 'critic1_target', 'critic2_target',
            )
        }
        with mock.patch.object(
            engine.actor_target,
            'forward',
            side_effect=AssertionError('target warmup must not run'),
        ):
            engine.warmup_full_compile((batch,))
        for name, expected in state_before.items():
            self.assert_state_dict_equal(getattr(engine, name).state_dict(), expected)
        self.assertIsNone(engine._compiled_target_block)
        self.assertIsNone(engine._compiled_target_critic_td)
        self.assertEqual(metadata['target_granularity'], 'eager')
        self.assertNotIn('td_target', metadata['enabled_objects'])

    def test_compilation_configuration_reports_scope_without_changing_identity(self):
        engine = self.make_engine(
            bc_reference_actor=self.make_bc_reference(0.02),
        )
        parameter_ids = {
            name: tuple(id(parameter) for parameter in model.parameters())
            for name, model in (
                ('actor', engine.actor), ('critic1', engine.critic1),
                ('critic2', engine.critic2), ('actor_target', engine.actor_target),
                ('critic1_target', engine.critic1_target),
                ('critic2_target', engine.critic2_target),
                ('bc_reference_actor', engine.bc_reference_actor),
            )
        }
        state_keys = {
            name: tuple(model.state_dict())
            for name, model in (
                ('actor', engine.actor), ('critic1', engine.critic1),
                ('critic2', engine.critic2), ('actor_target', engine.actor_target),
                ('critic1_target', engine.critic1_target),
                ('critic2_target', engine.critic2_target),
                ('bc_reference_actor', engine.bc_reference_actor),
            )
        }
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.v2_ann.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            metadata = engine.configure_compilation(
                compile_actors=True,
                compile_action_inference=True,
                frozen_critic_strategy='compiled_no_grad_context',
                compile_critic_block=True,
                compile_target_block=True,
            )
        self.assertEqual(metadata['critic_granularity'], 'full_forward_and_loss')
        self.assertEqual(metadata['target_granularity'], 'full_tensor_block')
        self.assertEqual(metadata['actor_granularity'], 'ann_full_forward_or_snn_encoder')
        self.assertEqual(metadata['frozen_critic_strategy'], 'compiled_no_grad_context')
        self.assertEqual(metadata['select_action_execution'], 'compiled')
        self.assertIn(
            'actor.action_inference_full_forward', metadata['enabled_objects']
        )
        self.assertFalse(metadata['cuda_graph'])
        for name, expected in parameter_ids.items():
            model = getattr(engine, name)
            self.assertEqual(tuple(id(parameter) for parameter in model.parameters()), expected)
            self.assertEqual(tuple(model.state_dict()), state_keys[name])

        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            self.make_engine().configure_compilation(
                compile_critic_encoder=True,
                compile_critic_block=True,
            )

    def test_shared_relation_compile_warmup_and_update_build_each_observation_once(self):
        engine = self.make_engine(policy_delay=2)
        self.fill_replay(engine, counts=(0, 7), next_counts=(7, 0))
        batch = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        state_before = deepcopy(engine.actor.state_dict())
        counts_before = (
            engine.update_count, engine.critic_update_count,
            engine.critic_target_update_count, engine.actor_update_count,
            engine.last_total_steps,
        )
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            metadata = engine.configure_compilation(compile_shared_relations=True)
        engine.warmup_shared_relations_compile((batch,))
        self.assertEqual(metadata['shared_relations_granularity'], 'compiled_tensor_build')
        self.assert_state_dict_equal(engine.actor.state_dict(), state_before)
        self.assertEqual(counts_before, (
            engine.update_count, engine.critic_update_count,
            engine.critic_target_update_count, engine.actor_update_count,
            engine.last_total_steps,
        ))
        with mock.patch.object(
            engine.actor.zone_set_encoder,
            'build_shared_relations',
            wraps=engine.actor.zone_set_encoder.build_shared_relations,
        ) as build:
            engine.update_once(total_steps=1)
        self.assertEqual(build.call_count, 2)

    def test_compiled_shared_relations_match_two_updates_and_adam_state(self):
        torch.manual_seed(31415)
        eager = self.make_engine(policy_delay=1)
        compiled = self.make_engine(policy_delay=1)
        compiled.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        self.fill_replay(eager, counts=(0, 7), next_counts=(7, 0))
        batch = eager.replay.sample(2)
        eager.replay.sample = lambda batch_size: batch
        compiled.replay.sample = lambda batch_size: batch
        compiled.enable_shared_relations_compile(backend='eager')
        compiled.warmup_shared_relations_compile((batch.obs, batch.next_obs))

        for total_steps in (1, 2):
            rng_state = torch.random.get_rng_state()
            expected_metrics = eager.update_once(total_steps=total_steps)
            torch.random.set_rng_state(rng_state)
            actual_metrics = compiled.update_once(total_steps=total_steps)
            self.assertEqual(actual_metrics, expected_metrics)
        for name in (
            'actor', 'critic1', 'critic2',
            'actor_target', 'critic1_target', 'critic2_target',
        ):
            self.assert_state_dict_equal(
                getattr(compiled, name).state_dict(),
                getattr(eager, name).state_dict(),
            )
        for actual, expected in (
            (compiled.actor_optimizer.state_dict(), eager.actor_optimizer.state_dict()),
            (compiled.critic_optimizer.state_dict(), eager.critic_optimizer.state_dict()),
        ):
            self.assertEqual(actual['param_groups'], expected['param_groups'])
            self.assertEqual(actual['state'].keys(), expected['state'].keys())
            for parameter_id, expected_state in expected['state'].items():
                self.assertEqual(actual['state'][parameter_id].keys(), expected_state.keys())
                for state_name, expected_value in expected_state.items():
                    torch.testing.assert_close(
                        actual['state'][parameter_id][state_name], expected_value
                    )

    def test_compile_error_propagates_without_enabling_fallback(self):
        engine = self.make_engine()
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=RuntimeError('compile failed'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'compile failed'):
                engine.enable_online_critic_encoder_compile()
        self.assertFalse(engine.critic1.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertFalse(engine.critic2.zone_set_encoder.compiled_tensor_forward_enabled)

    def test_target_encoder_compile_scope_and_warmup_preserve_engine_state(self):
        engine = self.make_engine(
            bc_reference_actor=self.make_bc_reference(0.01),
        )
        batch = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        modules = {
            name: getattr(engine, name)
            for name in (
                'actor', 'actor_target', 'critic1', 'critic2',
                'critic1_target', 'critic2_target', 'bc_reference_actor',
            )
        }
        state_before = {
            name: deepcopy(module.state_dict()) for name, module in modules.items()
        }
        parameter_ids = {
            name: tuple(id(parameter) for parameter in module.parameters())
            for name, module in modules.items()
        }
        state_keys = {
            name: tuple(module.state_dict()) for name, module in modules.items()
        }
        modes_before = {name: module.training for name, module in modules.items()}
        requires_grad_before = {
            name: tuple(parameter.requires_grad for parameter in module.parameters())
            for name, module in modules.items()
        }
        counts_before = (
            engine.update_count,
            engine.critic_update_count,
            engine.critic_target_update_count,
            engine.actor_update_count,
            engine.last_total_steps,
        )
        torch.manual_seed(1234)
        np.random.seed(5678)
        torch_rng_before = torch.random.get_rng_state().clone()
        numpy_rng_before = np.random.get_state()
        target_grad_modes = []
        target_hooks = [
            target.register_forward_pre_hook(
                lambda _module, _args: target_grad_modes.append(
                    torch.is_grad_enabled()
                )
            )
            for target in (
                engine.actor_target, engine.critic1_target, engine.critic2_target,
            )
        ]

        try:
            engine.enable_online_critic_encoder_compile(backend='eager')
            enabled = engine.enable_target_encoder_compile(backend='eager')
            engine.warmup_online_critic_encoder_compile((batch,))
            engine.warmup_target_encoder_compile((batch,))
        finally:
            for hook in target_hooks:
                hook.remove()

        self.assertEqual(enabled, (
            'actor_target.zone_set_encoder',
            'critic1_target.zone_set_encoder',
            'critic2_target.zone_set_encoder',
        ))
        for name in ('critic1', 'critic2', 'actor_target',
                     'critic1_target', 'critic2_target'):
            self.assertTrue(
                modules[name].zone_set_encoder.compiled_tensor_forward_enabled
            )
        self.assertFalse(engine.actor.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertFalse(
            engine.bc_reference_actor.zone_set_encoder.compiled_tensor_forward_enabled
        )
        for name, module in modules.items():
            self.assertEqual(tuple(module.state_dict()), state_keys[name])
            self.assertEqual(
                tuple(id(parameter) for parameter in module.parameters()),
                parameter_ids[name],
            )
            self.assertEqual(module.training, modes_before[name])
            self.assertEqual(
                tuple(parameter.requires_grad for parameter in module.parameters()),
                requires_grad_before[name],
            )
            self.assert_state_dict_equal(module.state_dict(), state_before[name])
        self.assertFalse(engine.actor_target.training)
        self.assertTrue(target_grad_modes)
        self.assertFalse(any(target_grad_modes))
        self.assertTrue(all(
            not parameter.requires_grad
            for target in (
                engine.actor_target, engine.critic1_target, engine.critic2_target,
            )
            for parameter in target.parameters()
        ))
        self.assertEqual(counts_before, (
            engine.update_count,
            engine.critic_update_count,
            engine.critic_target_update_count,
            engine.actor_update_count,
            engine.last_total_steps,
        ))
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng_before)
        numpy_rng_after = np.random.get_state()
        self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
        np.testing.assert_array_equal(numpy_rng_after[1], numpy_rng_before[1])
        self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])

    def test_compiled_target_encoder_reads_parameters_after_soft_update(self):
        torch.manual_seed(2468)
        eager = self.make_engine(tau=0.5)
        compiled = self.make_engine(tau=0.5)
        compiled.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        compiled.enable_online_critic_encoder_compile(backend='eager')
        compiled.enable_target_encoder_compile(backend='eager')
        observation = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        compiled.warmup_online_critic_encoder_compile((observation,))
        compiled.warmup_target_encoder_compile((observation,))
        with torch.no_grad():
            before = compiled.actor_target(observation)
            for eager_parameter, compiled_parameter in zip(
                eager.actor.parameters(), compiled.actor.parameters()
            ):
                eager_parameter.add_(0.125)
                compiled_parameter.add_(0.125)
            eager._soft_update(eager.actor, eager.actor_target)
            compiled._soft_update(compiled.actor, compiled.actor_target)
            expected = eager.actor_target(observation)
            actual = compiled.actor_target(observation)

        self.assertFalse(torch.equal(actual, before))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_reused_relations_match_original_td3_update_for_both_delay_paths(self):
        for total_steps in (1, 2):
            with self.subTest(total_steps=total_steps):
                def make_seeded_engine():
                    torch.manual_seed(2468)
                    engine = self.make_engine(policy_delay=2)
                    self.fill_replay(engine, counts=(0, 7), next_counts=(7, 0))
                    engine.replay.rng = np.random.default_rng(1357)
                    return engine

                original = make_seeded_engine()
                reused = make_seeded_engine()
                torch.manual_seed(9753)
                original_metrics = original.update_once(
                    total_steps=total_steps,
                    reuse_shared_relations=False,
                )
                torch.manual_seed(9753)
                reused_metrics = reused.update_once(total_steps=total_steps)

                self.assertEqual(reused_metrics, original_metrics)
                for reused_model, original_model in (
                    (reused.actor, original.actor),
                    (reused.critic1, original.critic1),
                    (reused.critic2, original.critic2),
                    (reused.actor_target, original.actor_target),
                    (reused.critic1_target, original.critic1_target),
                    (reused.critic2_target, original.critic2_target),
                ):
                    self.assert_state_dict_equal(
                        reused_model.state_dict(), original_model.state_dict()
                    )
                self.assertEqual(reused.update_count, original.update_count)
                self.assertEqual(
                    reused.actor_update_count, original.actor_update_count
                )

    def test_profiled_update_marks_critic_backward_gradient_and_optimizer(self):
        engine = self.make_engine(policy_delay=2)
        self.fill_replay(engine)
        labels = []

        @contextmanager
        def record(label):
            labels.append(label)
            yield

        with mock.patch(
            'brain_uav.trainers.v2_td3.record_function',
            side_effect=record,
        ):
            engine.update_once(total_steps=1, profile_sections=True)

        self.assertEqual(labels, [
            'v2_td3.critic_backward',
            'v2_td3.critic_zero_grad',
            'v2_td3.critic_loss_backward',
            'v2_td3.critic_gradient_check_and_clip',
            'v2_td3.critic_optimizer_step',
        ])

    def test_compiled_path_diagnostic_marks_nested_update_without_profile_sections(self):
        engine = self.make_engine(policy_delay=1)
        self.fill_replay(engine)
        labels = []
        detail_records = []
        actor_profile_flags = []
        original_actor_forward = engine.actor.forward

        @contextmanager
        def record(label):
            labels.append(label)
            yield

        def actor_forward(*args, **kwargs):
            actor_profile_flags.append(kwargs.get('profile_sections', False))
            return original_actor_forward(*args, **kwargs)

        with mock.patch(
            'brain_uav.trainers.v2_td3.record_function',
            side_effect=record,
        ), mock.patch.object(
            engine.actor,
            'forward',
            side_effect=actor_forward,
        ):
            metrics = engine.update_once(
                total_steps=1,
                diagnostic_profile_sections=True,
                diagnostic_timing_recorder=detail_records.append,
            )

        self.assertTrue(metrics.actor_updated)
        self.assertTrue(actor_profile_flags)
        self.assertFalse(any(actor_profile_flags))
        expected = {
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
        }
        self.assertEqual(
            {
                label.removeprefix('v2_td3.detail.')
                for label in labels
                if label.startswith('v2_td3.detail.')
            },
            expected,
        )
        self.assertEqual(len(detail_records), 1)
        self.assertEqual(set(detail_records[0]['wall_seconds']), expected)
        self.assertEqual(
            detail_records[0]['calls']['actor_loss_composition_and_finite_check'],
            2,
        )

    def test_compiled_execution_recorder_observes_real_full_entries(self):
        engine = self.make_engine(
            policy_delay=1,
            bc_reference_actor=self.make_bc_reference(0.01),
        )
        self.fill_replay(engine)
        observation = collate_v2_observations([
            _observation(0, scales=self.scales),
            _observation(7, scales=self.scales),
        ])
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.v2_ann.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.configure_compilation(
                compile_actors=True,
                frozen_critic_strategy='compiled_no_grad_context',
                compile_critic_block=True,
                compile_target_block=True,
                backend='eager',
            )
            engine.warmup_actor_compile((observation,))
            engine.warmup_full_compile((observation,))
            entries = []
            metrics = engine.update_once(
                total_steps=1,
                bc_lambda=1.0,
                diagnostic_profile_sections=True,
                compiled_execution_recorder=entries.append,
            )

        self.assertTrue(metrics.actor_updated)
        self.assertEqual(set(entries), {
            'target_block',
            'critic_block',
            'frozen_critic_context',
            'actor',
            'bc_reference_actor',
        })

    def test_fixed_buffers_are_validated_at_engine_and_checkpoint_boundaries(self):
        action_limit = torch.tensor([0.2, 0.3], dtype=torch.float32)
        actor = V2ANNPolicyActor(self.scales, 2, 16, action_limit)
        critic1 = V2ANNCritic(self.scales, 2, 16)
        critic2 = V2ANNCritic(self.scales, 2, 16)
        with torch.no_grad():
            actor.zone_set_encoder.pair_relation_builder.horizontal_span.add_(1.0)
        with self.assertRaisesRegex(ValueError, 'fixed buffer.*horizontal_span'):
            V2TD3UpdateEngine(
                actor,
                critic1,
                critic2,
                V2ReplayBuffer(8, 2, 2),
                actor_lr=1e-3,
                critic_lr=1e-3,
                gamma=0.99,
                tau=0.25,
                policy_noise=0.1,
                noise_clip=0.05,
                policy_delay=2,
                batch_size=2,
                action_low=np.array([-0.2, -0.3], dtype=np.float32),
                action_high=np.array([0.2, 0.3], dtype=np.float32),
                terminal_geo_regularization_enabled=False,
            )

        engine = self.make_engine()
        payload = deepcopy(engine.checkpoint_state_dict())
        for state_name in ('actor_state_dict', 'actor_target_state_dict'):
            payload[state_name]['action_limit'] = torch.tensor(
                [0.25, 0.35], dtype=torch.float32
            )
        with self.assertRaisesRegex(ValueError, 'fixed buffer.*action_limit'):
            engine.load_network_state_dicts(payload)

    def test_target_noise_and_target_action_are_both_clipped(self):
        engine = self.make_engine(policy_delay=2)
        self.fill_replay(engine)
        captured_actions = []
        original_forward = engine.critic1_target.forward

        def capture(observation, action, **kwargs):
            captured_actions.append(action.detach().clone())
            return original_forward(observation, action, **kwargs)

        actor_output = torch.tensor(
            [[0.19, -0.29], [0.19, -0.29]], dtype=torch.float32
        )
        raw_noise = torch.tensor(
            [[10.0, -10.0], [10.0, -10.0]], dtype=torch.float32
        )
        with (
            mock.patch.object(engine.actor_target, 'forward', return_value=actor_output),
            mock.patch.object(engine.critic1_target, 'forward', side_effect=capture),
            mock.patch('torch.randn_like', return_value=raw_noise),
        ):
            engine.update_once(total_steps=1)

        self.assertEqual(len(captured_actions), 1)
        torch.testing.assert_close(
            captured_actions[0],
            torch.tensor([[0.2, -0.3], [0.2, -0.3]], dtype=torch.float32),
        )

    def test_updates_support_empty_large_and_different_next_zone_axes(self):
        cases = (
            ((0, 0), (0, 0)),
            ((1, 0), (6, 5)),
            ((6, 10), (10, 6)),
        )
        for counts, next_counts in cases:
            with self.subTest(counts=counts, next_counts=next_counts):
                engine = self.make_engine(policy_delay=2)
                self.fill_replay(engine, counts=counts, next_counts=next_counts)
                metrics = engine.update_once(total_steps=1)
                self.assertTrue(math.isfinite(metrics.critic_loss))

    def test_actor_rl_scale_uses_detached_old_formula(self):
        engine = self.make_engine(terminal_enabled=False)
        observation = collate_v2_observations([
            _observation(1, scales=self.scales),
            _observation(2, scales=self.scales),
        ])
        safe = torch.ones((2, 1), dtype=torch.float32)

        def constant_q(obs, action, **_kwargs):
            del obs
            return action[:, :1] * 0.0 + 10.0

        with mock.patch.object(engine.critic1, 'forward', side_effect=constant_q):
            terms = engine._compute_actor_loss_terms(observation, safe, bc_lambda=0.0)

        self.assertAlmostEqual(terms.rl_actor_loss.item(), -10.0)
        self.assertAlmostEqual(terms.actor_rl_scale.item(), 0.25)
        self.assertAlmostEqual(terms.scaled_rl_actor_loss.item(), -2.5)
        self.assertFalse(terms.actor_rl_scale.requires_grad)

    def test_terminal_geometry_local_formula_matches_world_reference(self):
        engine = self.make_engine(terminal_enabled=True)
        forward, right, up, gamma, psi = 20.0, 5.0, 3.0, 0.1, 0.6
        observation = _observation(
            0,
            goal_forward=forward,
            goal_right=right,
            goal_up=up,
            gamma=gamma,
            scales=self.scales,
        )
        batch = collate_v2_observations([observation])
        actor_actions = torch.tensor([[0.02, -0.04]], dtype=torch.float32, requires_grad=True)
        safe = torch.ones((1, 1), dtype=torch.float32)

        loss = engine._terminal_geo_loss(batch, actor_actions, safe)

        dx = forward * math.cos(psi) - right * math.sin(psi)
        dy = forward * math.sin(psi) + right * math.cos(psi)
        target_gamma = math.atan2(up, math.sqrt(dx * dx + dy * dy))
        target_psi = math.atan2(dy, dx)
        delta_gamma = float(np.clip(target_gamma - gamma, -0.2, 0.2))
        delta_psi = (target_psi - psi + math.pi) % (2.0 * math.pi) - math.pi
        delta_psi = float(np.clip(delta_psi, -0.3, 0.3))
        expected = ((0.02 - delta_gamma) ** 2 + (-0.04 - delta_psi) ** 2) / 2.0
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertTrue(engine.terminal_geo_regularization_enabled)
        self.assertEqual(engine.terminal_geo_radius, 250.0)
        self.assertEqual(engine.terminal_geo_lambda, 3000.0)

    def test_no_terminal_eligible_sample_returns_graph_connected_zero(self):
        engine = self.make_engine(terminal_enabled=True)
        batch = collate_v2_observations([
            _observation(0, goal_forward=400.0, scales=self.scales),
            _observation(1, goal_forward=20.0, scales=self.scales),
        ])
        actions = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
        safe = torch.zeros((2, 1), dtype=torch.float32)

        loss = engine._terminal_geo_loss(batch, actions, safe)
        loss.backward()

        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(actions.grad)
        torch.testing.assert_close(actions.grad, torch.zeros_like(actions.grad))

    def test_bc_reference_is_copied_frozen_and_contributes_only_when_requested(self):
        reference = _DeviceTrackingV2Actor(
            self.scales, 2, 16, torch.tensor([0.2, 0.3], dtype=torch.float32)
        )
        reference.train()
        original_devices = {parameter.device for parameter in reference.parameters()}
        original_requires_grad = [
            parameter.requires_grad for parameter in reference.parameters()
        ]
        engine = self.make_engine(
            policy_delay=1,
            bc_reference_actor=reference,
            terminal_enabled=False,
        )
        self.fill_replay(engine)

        self.assertIsNot(engine.bc_reference_actor, reference)
        self.assertEqual(reference.to_call_count, 0)
        self.assertTrue(reference.training)
        self.assertEqual(
            {parameter.device for parameter in reference.parameters()},
            original_devices,
        )
        self.assertEqual(
            [parameter.requires_grad for parameter in reference.parameters()],
            original_requires_grad,
        )
        self.assertFalse(engine.bc_reference_actor.training)
        self.assertTrue(all(not p.requires_grad for p in engine.bc_reference_actor.parameters()))
        metrics = engine.update_once(total_steps=1, bc_lambda=2.0)
        self.assertTrue(metrics.actor_updated)
        self.assertEqual(metrics.bc_lambda, 2.0)
        self.assertGreaterEqual(metrics.bc_loss, 0.0)

        no_reference = self.make_engine(policy_delay=1)
        self.fill_replay(no_reference)
        with self.assertRaisesRegex(ValueError, 'bc_lambda'):
            no_reference.update_once(total_steps=1, bc_lambda=1.0)

    def test_checkpoint_round_trip_restores_networks_optimizers_and_counts(self):
        engine = self.make_engine(policy_delay=1)
        self.fill_replay(engine)
        engine.update_once(total_steps=1)
        observation = collate_v2_observations([_observation(3, scales=self.scales)])
        action = torch.tensor([[0.01, -0.02]], dtype=torch.float32)
        actor_before = engine.actor(observation).detach().clone()
        critic_before = engine.critic1(observation, action).detach().clone()
        target_before = engine.actor_target(observation).detach().clone()

        payload = engine.checkpoint_state_dict()
        self.assertEqual(payload['format'], 'v2_td3_dynamic_set')
        self.assertEqual(payload['format_version'], 3)
        self.assertIn('observation_contract', payload)
        self.assertIs(payload['bc_reference_present'], False)
        self.assertIsNone(payload['bc_reference_actor_state_dict'])
        self.assertEqual(
            payload['algorithm_config'],
            {
                'gamma': 0.99,
                'tau': 0.25,
                'policy_noise': 0.1,
                'noise_clip': 0.05,
                'policy_delay': 1,
                'batch_size': 2,
                'actor_freeze_steps': 0,
                'actor_grad_clip_norm': 1.0,
                'critic_grad_clip_norm': 1.0,
                'actor_rl_scale_alpha': 2.5,
                'terminal_geo_regularization_enabled': False,
                'terminal_geo_radius': 250.0,
                'terminal_geo_lambda': 3000.0,
            },
        )

        torch.manual_seed(999)
        restored = self.make_engine(policy_delay=1)
        restored.load_checkpoint_state_dict(payload)

        torch.testing.assert_close(restored.actor(observation), actor_before)
        torch.testing.assert_close(restored.critic1(observation, action), critic_before)
        torch.testing.assert_close(restored.actor_target(observation), target_before)
        self.assertEqual(restored.update_count, engine.update_count)
        self.assertEqual(restored.actor_update_count, engine.actor_update_count)
        self.assertTrue(restored.actor_optimizer.state_dict()['state'])
        self.assertTrue(restored.critic_optimizer.state_dict()['state'])

        with self.assertRaisesRegex(ValueError, 'format'):
            restored.load_checkpoint_state_dict({'state_dict': engine.actor.state_dict()})
        incompatible = deepcopy(payload)
        incompatible['action_dim'] = 3
        with self.assertRaisesRegex(ValueError, 'action_dim'):
            restored.load_checkpoint_state_dict(incompatible)

    def test_checkpoint_rejects_algorithm_mismatch_missing_config_and_version_two(self):
        source = self.make_engine(policy_delay=2, tau=0.25, gamma=0.99)
        payload = source.checkpoint_state_dict()

        for name, restored in (
            ('tau', self.make_engine(policy_delay=2, tau=0.1, gamma=0.99)),
            ('gamma', self.make_engine(policy_delay=2, tau=0.25, gamma=0.95)),
            ('policy_delay', self.make_engine(policy_delay=1, tau=0.25, gamma=0.99)),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'algorithm_config'):
                    restored.load_checkpoint_state_dict(payload)

        missing = deepcopy(payload)
        missing.pop('algorithm_config', None)
        with self.assertRaisesRegex(ValueError, 'algorithm_config'):
            source.load_checkpoint_state_dict(missing)

        version_two = deepcopy(payload)
        version_two['format_version'] = 2
        with self.assertRaisesRegex(ValueError, 'format_version'):
            source.load_checkpoint_state_dict(version_two)

    def test_checkpoint_restores_bc_reference_into_engine_without_one(self):
        reference_a = self.make_bc_reference(0.125)
        source = self.make_engine(bc_reference_actor=reference_a)
        payload = source.checkpoint_state_dict()
        expected_state = {
            name: value.detach().clone()
            for name, value in source.bc_reference_actor.state_dict().items()
        }
        self.assertIs(payload.get('bc_reference_present'), True)
        self.assertIsInstance(payload['bc_reference_actor_state_dict'], dict)

        restored = self.make_engine()
        self.assertIsNone(restored.bc_reference_actor)
        restored.load_checkpoint_state_dict(payload)

        self.assertIsNotNone(restored.bc_reference_actor)
        self.assert_state_dict_equal(
            restored.bc_reference_actor.state_dict(),
            expected_state,
        )
        self.assertFalse(restored.bc_reference_actor.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in restored.bc_reference_actor.parameters()
            )
        )
        bc_ids = {id(parameter) for parameter in restored.bc_reference_actor.parameters()}
        online_and_target_ids = {
            id(parameter)
            for model in (restored.actor, restored.actor_target)
            for parameter in model.parameters()
        }
        self.assertTrue(bc_ids.isdisjoint(online_and_target_ids))
        self.assertEqual(
            {parameter.device for parameter in restored.bc_reference_actor.parameters()},
            {restored.device},
        )

    def test_checkpoint_bc_reference_overwrites_existing_reference(self):
        reference_a = self.make_bc_reference(0.125)
        reference_b = self.make_bc_reference(-0.25)
        source = self.make_engine(bc_reference_actor=reference_a)
        restored = self.make_engine(bc_reference_actor=reference_b)
        state_b_before = {
            name: value.detach().clone()
            for name, value in restored.bc_reference_actor.state_dict().items()
        }

        restored.load_checkpoint_state_dict(source.checkpoint_state_dict())

        self.assert_state_dict_equal(
            restored.bc_reference_actor.state_dict(),
            source.bc_reference_actor.state_dict(),
        )
        self.assertTrue(
            any(
                not torch.equal(state_b_before[name], value)
                for name, value in restored.bc_reference_actor.state_dict().items()
            )
        )

    def test_checkpoint_without_bc_reference_clears_existing_reference(self):
        source = self.make_engine()
        restored = self.make_engine(
            bc_reference_actor=self.make_bc_reference(-0.25)
        )
        self.assertIsNotNone(restored.bc_reference_actor)

        restored.load_checkpoint_state_dict(source.checkpoint_state_dict())

        self.assertIsNone(restored.bc_reference_actor)

    def test_checkpoint_rejects_invalid_bc_fields_before_mutating_networks(self):
        source = self.make_engine(
            bc_reference_actor=self.make_bc_reference(0.125)
        )
        payload = source.checkpoint_state_dict()
        restored = self.make_engine()

        missing_present = deepcopy(payload)
        missing_present.pop('bc_reference_present', None)
        with self.assertRaisesRegex(ValueError, 'bc_reference_present'):
            restored.load_checkpoint_state_dict(missing_present)

        missing_state = deepcopy(payload)
        missing_state.pop('bc_reference_actor_state_dict', None)
        with self.assertRaisesRegex(ValueError, 'bc_reference_actor_state_dict'):
            restored.load_checkpoint_state_dict(missing_state)

        malformed = deepcopy(payload)
        state_dict = malformed['bc_reference_actor_state_dict']
        first_name = next(iter(state_dict))
        state_dict[first_name] = torch.zeros((1,), dtype=state_dict[first_name].dtype)
        actor_before = {
            name: parameter.detach().clone()
            for name, parameter in restored.actor.named_parameters()
        }
        with self.assertRaisesRegex(ValueError, 'bc_reference_actor_state_dict'):
            restored.load_checkpoint_state_dict(malformed)
        for name, parameter in restored.actor.named_parameters():
            torch.testing.assert_close(parameter, actor_before[name])

        version_two = deepcopy(payload)
        version_two['format_version'] = 2
        with self.assertRaisesRegex(ValueError, 'format_version'):
            restored.load_checkpoint_state_dict(version_two)

    def test_checkpoint_rejects_inconsistent_bc_presence_contract(self):
        source = self.make_engine(
            bc_reference_actor=self.make_bc_reference(0.125)
        )
        payload = source.checkpoint_state_dict()
        cases = []

        non_boolean = deepcopy(payload)
        non_boolean['bc_reference_present'] = 1
        cases.append(('non_boolean', non_boolean))

        present_without_state = deepcopy(payload)
        present_without_state['bc_reference_present'] = True
        present_without_state['bc_reference_actor_state_dict'] = None
        cases.append(('present_without_state', present_without_state))

        absent_with_state = deepcopy(payload)
        absent_with_state['bc_reference_present'] = False
        cases.append(('absent_with_state', absent_with_state))

        for name, invalid in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, 'bc_reference'):
                    self.make_engine().load_checkpoint_state_dict(invalid)

    def test_select_action_supports_zero_six_and_ten_zones(self):
        engine = self.make_engine()
        for count in (0, 6, 10):
            with self.subTest(count=count):
                action = engine.select_action(
                    _observation(count, scales=self.scales),
                    exploration_noise=0.0,
                )
                self.assertEqual(action.shape, (2,))
                self.assertEqual(action.dtype, np.float32)
                self.assertTrue(np.all(action >= np.array([-0.2, -0.3], dtype=np.float32)))
                self.assertTrue(np.all(action <= np.array([0.2, 0.3], dtype=np.float32)))

    def test_ann_action_inference_compile_is_independent_and_uses_current_parameters(self):
        torch.manual_seed(1201)
        eager = self.make_engine()
        compiled = self.make_engine()
        compiled.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ) as compile_call:
            enabled = compiled.enable_action_inference_compile(backend='eager')
        self.assertEqual(enabled, ('actor.action_inference_full_forward',))
        self.assertEqual(compile_call.call_count, 1)
        self.assertIsNot(
            compiled._compiled_action_inference,
            compiled.actor._compiled_full_forward,
        )
        inference = mock.Mock(wraps=compiled._compiled_action_inference)
        compiled._compiled_action_inference = inference

        for count in (0, 6, 10):
            observation = _observation(count, scales=self.scales)
            np.testing.assert_allclose(
                compiled.select_action(observation),
                eager.select_action(observation),
                rtol=1e-4, atol=1e-5,
            )
        self.assertEqual(inference.call_count, 3)

        with torch.no_grad():
            eager.actor.head[4].bias.add_(0.05)
            compiled.actor.head[4].bias.add_(0.05)
        observation = _observation(3, scales=self.scales)
        np.testing.assert_allclose(
            compiled.select_action(observation),
            eager.select_action(observation),
            rtol=1e-4, atol=1e-5,
        )

    def test_configure_compilation_reports_independent_action_inference_scope(self):
        engine = self.make_engine()
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            metadata = engine.configure_compilation(
                compile_action_inference=True,
                backend='eager', fullgraph=True, dynamic=True,
            )
        self.assertEqual(
            metadata['enabled_objects'],
            ['actor.action_inference_full_forward'],
        )
        self.assertEqual(metadata['select_action_execution'], 'compiled')
        self.assertEqual(
            metadata['action_inference_granularity'],
            'ann_full_forward',
        )
        self.assertEqual(metadata['backend'], 'eager')

    def test_action_inference_compile_preserves_exploration_rng_progression(self):
        torch.manual_seed(1211)
        eager = self.make_engine()
        compiled = self.make_engine()
        compiled.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            compiled.enable_action_inference_compile(backend='eager')
        left_rng = np.random.default_rng(412)
        right_rng = np.random.default_rng(412)
        observation = _observation(6, scales=self.scales)
        for _ in range(2):
            np.testing.assert_array_equal(
                compiled.select_action(
                    observation, exploration_noise=0.02,
                    exploration_rng=left_rng,
                ),
                eager.select_action(
                    observation, exploration_noise=0.02,
                    exploration_rng=right_rng,
                ),
            )
        self.assertEqual(left_rng.bit_generator.state, right_rng.bit_generator.state)

    def test_action_inference_compile_coexists_with_critic_only_and_actor_updates(self):
        engine = self.make_engine(policy_delay=2)
        self.fill_replay(engine)
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_action_inference_compile(backend='eager')
        observation = _observation(6, scales=self.scales)
        engine.select_action(observation)
        critic_only = engine.update_once(total_steps=1, bc_lambda=0.0)
        self.assertFalse(critic_only.actor_updated)
        engine.select_action(observation)
        actor_update = engine.update_once(total_steps=2, bc_lambda=0.0)
        self.assertTrue(actor_update.actor_updated)
        engine.select_action(observation)
        self.assertTrue(any(
            parameter.grad is not None for parameter in engine.actor.parameters()
        ))

    def test_ann_action_inference_real_dynamo_eager_matches_eager(self):
        engine = self.make_engine()
        observations = tuple(
            _observation(count, scales=self.scales) for count in (0, 6, 10)
        )
        expected = tuple(engine.select_action(value) for value in observations)
        engine.enable_action_inference_compile(
            backend='eager', fullgraph=True, dynamic=True,
        )
        batches = tuple(
            collate_v2_observations([value]) for value in observations
        )
        engine.warmup_action_inference_compile(batches)
        for observation, reference in zip(observations, expected):
            np.testing.assert_allclose(
                engine.select_action(observation), reference,
                rtol=1e-4, atol=1e-5,
            )
        with torch.no_grad():
            engine.actor.head[4].bias.add_(0.04)
        observation = observations[1]
        compiled_action = engine.select_action(observation)
        compiled_forward = engine._compiled_action_inference
        engine._compiled_action_inference = None
        try:
            eager_action = engine.select_action(observation)
        finally:
            engine._compiled_action_inference = compiled_forward
        np.testing.assert_allclose(
            compiled_action, eager_action, rtol=1e-4, atol=1e-5,
        )

    def test_action_inference_warmup_is_batch_one_and_has_no_engine_side_effects(self):
        engine = self.make_engine()
        batches = tuple(collate_v2_observations([
            _observation(count, scales=self.scales)
        ]) for count in (0, 6, 10))
        state_before = deepcopy(engine.actor.state_dict())
        other_states_before = {
            name: deepcopy(getattr(engine, name).state_dict())
            for name in (
                'actor_target', 'critic1', 'critic2',
                'critic1_target', 'critic2_target',
            )
        }
        optimizer_before = deepcopy(engine.actor_optimizer.state_dict())
        critic_optimizer_before = deepcopy(engine.critic_optimizer.state_dict())
        replay_before = (
            engine.replay.size,
            engine.replay.position,
            engine.replay.success_size,
            engine.replay.success_position,
            deepcopy(engine.replay.rng.bit_generator.state),
        )
        counts_before = (
            engine.update_count, engine.critic_update_count,
            engine.critic_target_update_count, engine.actor_update_count,
            engine.last_total_steps,
        )
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        engine.actor.train()
        engine.actor.head.eval()
        training_modes = tuple(
            module.training for module in engine.actor.modules()
        )
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_action_inference_compile(backend='eager')
        engine.warmup_action_inference_compile(batches)
        self.assert_state_dict_equal(engine.actor.state_dict(), state_before)
        for name, state in other_states_before.items():
            self.assert_state_dict_equal(getattr(engine, name).state_dict(), state)
        self.assertEqual(engine.actor_optimizer.state_dict(), optimizer_before)
        self.assertEqual(engine.critic_optimizer.state_dict(), critic_optimizer_before)
        self.assertEqual(replay_before, (
            engine.replay.size,
            engine.replay.position,
            engine.replay.success_size,
            engine.replay.success_position,
            engine.replay.rng.bit_generator.state,
        ))
        self.assertEqual(counts_before, (
            engine.update_count, engine.critic_update_count,
            engine.critic_target_update_count, engine.actor_update_count,
            engine.last_total_steps,
        ))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), python_state)
        self.assertEqual(np.random.get_state()[0], numpy_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(
            tuple(module.training for module in engine.actor.modules()),
            training_modes,
        )
        with self.assertRaisesRegex(ValueError, 'batch size 1'):
            engine.warmup_action_inference_compile((collate_v2_observations([
                _observation(0, scales=self.scales),
                _observation(1, scales=self.scales),
            ]),))

    def test_select_action_uses_independent_precomputed_cpu_action_bounds(self):
        action_low = np.array([-0.2, -0.3], dtype=np.float32)
        action_high = np.array([0.2, 0.3], dtype=np.float32)
        engine = self.make_engine(action_low=action_low, action_high=action_high)
        action_low[:] = -9.0
        action_high[:] = 9.0
        bound_storage = {
            engine.action_low.untyped_storage().data_ptr(),
            engine.action_high.untyped_storage().data_ptr(),
        }
        cpu_bound_transfers = []
        original_cpu = torch.Tensor.cpu

        def track_cpu(tensor, *args, **kwargs):
            if tensor.untyped_storage().data_ptr() in bound_storage:
                cpu_bound_transfers.append(tensor)
            return original_cpu(tensor, *args, **kwargs)

        with mock.patch.object(
            engine.actor,
            'forward',
            return_value=torch.tensor([[1.0, -1.0]], dtype=torch.float32),
        ), mock.patch.object(torch.Tensor, 'cpu', new=track_cpu):
            action = engine.select_action(
                _observation(1, scales=self.scales),
                exploration_noise=0.0,
            )

        np.testing.assert_array_equal(
            action,
            np.array([0.2, -0.3], dtype=np.float32),
        )
        self.assertEqual(cpu_bound_transfers, [])

    def test_target_noise_has_strict_runtime_schedule_interface(self):
        engine = self.make_engine()
        engine.set_target_noise(policy_noise=0.015, noise_clip=0.03)
        self.assertEqual(engine.policy_noise, 0.015)
        self.assertEqual(engine.noise_clip, 0.03)
        for policy_noise, noise_clip in (
            (-0.1, 0.1),
            (0.1, -0.1),
            (float('nan'), 0.1),
            (0.1, float('inf')),
        ):
            with self.subTest(policy_noise=policy_noise, noise_clip=noise_clip):
                with self.assertRaises(ValueError):
                    engine.set_target_noise(
                        policy_noise=policy_noise,
                        noise_clip=noise_clip,
                    )

    def test_select_action_uses_explicit_exploration_rng_when_supplied(self):
        engine = self.make_engine()
        observation = _observation(1, scales=self.scales)
        np.random.seed(1)
        first = engine.select_action(
            observation,
            exploration_noise=0.02,
            exploration_rng=np.random.default_rng(314),
        )
        np.random.seed(999)
        second = engine.select_action(
            observation,
            exploration_noise=0.02,
            exploration_rng=np.random.default_rng(314),
        )
        self.assertTrue(np.array_equal(first, second))
        with self.assertRaisesRegex(TypeError, 'exploration_rng'):
            engine.select_action(
                observation,
                exploration_noise=0.02,
                exploration_rng=object(),
            )

    def test_network_only_stage_handoff_preserves_fresh_optimizer_and_reference(self):
        source = self.make_engine(policy_delay=1, tau=0.25)
        self.fill_replay(source)
        source.update_once(total_steps=1)
        payload = source.checkpoint_state_dict()

        new_reference = self.make_bc_reference(-0.125)
        target = self.make_engine(
            policy_delay=2,
            tau=0.1,
            bc_reference_actor=new_reference,
        )
        self.assertFalse(target.actor_optimizer.state_dict()['state'])
        self.assertFalse(target.critic_optimizer.state_dict()['state'])
        reference_before = {
            name: value.detach().clone()
            for name, value in target.bc_reference_actor.state_dict().items()
        }

        target.load_network_state_dicts(payload)

        for name in (
            'actor',
            'critic1',
            'critic2',
            'actor_target',
            'critic1_target',
            'critic2_target',
        ):
            self.assert_state_dict_equal(
                getattr(target, name).state_dict(),
                getattr(source, name).state_dict(),
            )
        self.assertFalse(target.actor_optimizer.state_dict()['state'])
        self.assertFalse(target.critic_optimizer.state_dict()['state'])
        self.assert_state_dict_equal(
            target.bc_reference_actor.state_dict(),
            reference_before,
        )
        self.assertEqual(target.update_count, 0)

        malformed = deepcopy(payload)
        malformed['actor_state_dict'].pop(next(iter(malformed['actor_state_dict'])))
        actor_before = {
            name: value.detach().clone()
            for name, value in target.actor.state_dict().items()
        }
        with self.assertRaisesRegex(ValueError, 'actor_state_dict'):
            target.load_network_state_dicts(malformed)
        self.assert_state_dict_equal(target.actor.state_dict(), actor_before)

    def test_engine_construction_does_not_reset_torch_rng(self):
        actor = V2ANNPolicyActor(
            self.scales, 2, 16, torch.tensor([0.2, 0.3], dtype=torch.float32)
        )
        critic1 = V2ANNCritic(self.scales, 2, 16)
        critic2 = V2ANNCritic(self.scales, 2, 16)
        replay = V2ReplayBuffer(8, 2, 10)
        torch.manual_seed(12345)
        expected = torch.rand(4)
        torch.manual_seed(12345)

        V2TD3UpdateEngine(
            actor,
            critic1,
            critic2,
            replay,
            1e-3,
            1e-3,
            0.99,
            0.1,
            0.1,
            0.05,
            2,
            2,
            np.array([-0.2, -0.3], dtype=np.float32),
            np.array([0.2, 0.3], dtype=np.float32),
            terminal_geo_regularization_enabled=False,
        )
        actual = torch.rand(4)

        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
