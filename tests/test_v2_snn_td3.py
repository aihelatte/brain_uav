"""TD3 integration tests for the V2 SpikingJelly actor with ANN critics."""

from __future__ import annotations

from copy import deepcopy
import inspect
import math
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.models import V2ANNCritic, V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
    V2ObservationScales,
    collate_v2_observations,
)
from brain_uav.trainers import V2ReplayBuffer, V2TD3UpdateEngine
from brain_uav.trainers.v2_td3 import V2_SNN_TD3_CHECKPOINT_FORMAT


def _observation(count: int, scales: V2ObservationScales, offset: float = 0.0):
    ego = np.zeros(EGO_FEATURE_DIM, dtype=np.float32)
    ego[0] = 0.01 * offset
    ego[2] = 0.4
    ego[4] = math.sin(0.2)
    ego[5] = math.cos(0.2)
    goal = np.array([0.1, 0.02, 0.03, 0.2], dtype=np.float32)
    zones = np.zeros((count, ZONE_FEATURE_DIM), dtype=np.float32)
    for index in range(count):
        zones[index, 0] = 1.0
        zones[index, 5] = -0.2 + index * 0.05
        zones[index, 6] = 0.02
        zones[index, 8:11] = (0.05, 0.04, 0.08)
        zones[index, 12] = 0.2
        zones[index, 13] = -1.0
        zones[index, 16] = 0.5
        zones[index, 18] = 1.0
    return V2Observation(ego, goal, zones, np.ones(count, dtype=np.bool_))


class TestV2SNNTD3(unittest.TestCase):
    def setUp(self) -> None:
        self.scales = V2ObservationScales(100.0, 0.0, 50.0, math.pi / 4.0)

    def make_actor(self, *, time_window: int = 2) -> V2SNNPolicyActor:
        return V2SNNPolicyActor(
            self.scales,
            2,
            8,
            torch.tensor([0.2, 0.3], dtype=torch.float32),
            time_window=time_window,
        )

    def make_engine(
        self, *, freeze: int = 0, bc=None,
        fused_adam: bool = False,
        aggregate_relation_values_first: bool = False,
        time_window: int = 2,
    ) -> V2TD3UpdateEngine:
        actor = self.make_actor(time_window=time_window)
        critic1 = V2ANNCritic(self.scales, 2, 8)
        critic2 = V2ANNCritic(self.scales, 2, 8)
        replay = V2ReplayBuffer(16, 2, 10, seed=31)
        engine = V2TD3UpdateEngine(
            actor,
            critic1,
            critic2,
            replay,
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.25,
            policy_noise=0.01,
            noise_clip=0.02,
            policy_delay=1,
            batch_size=2,
            action_low=np.array([-0.2, -0.3], dtype=np.float32),
            action_high=np.array([0.2, 0.3], dtype=np.float32),
            actor_freeze_steps=freeze,
            terminal_geo_regularization_enabled=False,
            bc_reference_actor=bc,
            fused_adam=fused_adam,
            aggregate_relation_values_first=aggregate_relation_values_first,
        )
        for index, count in enumerate((0, 6, 10, 2)):
            engine.replay.add(
                _observation(count, self.scales, float(index)),
                np.array([0.01, -0.02], dtype=np.float32),
                1.0,
                _observation((count + 1) % 11, self.scales, float(index) + 0.5),
                False,
                line_to_goal_safe=True,
            )
        return engine

    def test_snn_actor_ann_critics_update_and_state_isolation(self) -> None:
        engine = self.make_engine()
        actor_before = deepcopy(engine.actor.state_dict())
        metrics = engine.update_once(total_steps=1, bc_lambda=0.0)
        self.assertTrue(metrics.critic_updated)
        self.assertTrue(metrics.actor_updated)
        self.assertIsInstance(engine.actor_target, V2SNNPolicyActor)
        self.assertTrue(any(
            not torch.equal(actor_before[name], value)
            for name, value in engine.actor.state_dict().items()
            if value.is_floating_point()
        ))
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor_target.snn_head.lif2.v, 0.0)

    def test_snn_three_optimizations_keep_lif_reset_and_target_frozen(self) -> None:
        engine = self.make_engine(
            bc=self.make_actor(), fused_adam=True,
            aggregate_relation_values_first=True,
        )
        metadata = engine.configure_compilation(
            compile_actor_loss=True, backend='eager', fullgraph=True,
        )
        self.assertEqual(metadata['actor_loss_granularity'], 'tensor_block')
        self.assertEqual(metadata['optimizer_execution'], 'fused_adam')
        self.assertTrue(all(
            layer.attention.aggregate_relation_values_first
            for layer in engine.actor.zone_set_encoder.layers
        ))
        self.assertTrue(all(not p.requires_grad for p in engine.actor_target.parameters()))
        metrics = engine.update_once(total_steps=1, bc_lambda=1.0)
        self.assertTrue(metrics.actor_updated)
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor_target.snn_head.lif2.v, 0.0)
        engine.select_action(_observation(10, self.scales))
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)

    def test_snn_actor_compile_scope_is_encoder_only_and_preserves_reset(self) -> None:
        engine = self.make_engine(bc=self.make_actor())
        batch = collate_v2_observations([
            _observation(0, self.scales),
            _observation(10, self.scales),
        ])
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            enabled = engine.enable_actor_compile()
        self.assertEqual(enabled, (
            'actor.zone_set_encoder',
            'bc_reference_actor.zone_set_encoder',
        ))
        self.assertFalse(hasattr(engine.actor, 'compiled_full_forward_enabled'))
        engine.warmup_actor_compile((batch,))
        with mock.patch.object(
            engine.actor.zone_set_encoder,
            'forward',
            wraps=engine.actor.zone_set_encoder.forward,
        ) as encoder_forward:
            output = engine.actor(batch)
        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(encoder_forward.call_count, 1)
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor.snn_head.lif2.v, 0.0)

    def test_snn_select_action_bypasses_compiled_shared_relations(self) -> None:
        engine = self.make_engine()
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_shared_relations_compile()
        encoder = engine.actor.zone_set_encoder
        compiled = mock.Mock(wraps=encoder._compiled_shared_relations)
        encoder._compiled_shared_relations = compiled

        engine.select_action(_observation(10, self.scales))
        self.assertEqual(compiled.call_count, 0)
        batch = collate_v2_observations([
            _observation(0, self.scales),
            _observation(10, self.scales),
        ])
        engine._build_shared_relations(batch)
        self.assertEqual(compiled.call_count, 1)

    def test_snn_action_inference_compile_is_encoder_only_and_matches_eager(self) -> None:
        torch.manual_seed(1301)
        eager = self.make_engine(time_window=4)
        compiled = self.make_engine(time_window=4)
        compiled.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ) as compile_call:
            enabled = compiled.enable_action_inference_compile(backend='eager')
        self.assertEqual(enabled, ('actor.action_inference_encoder',))
        self.assertEqual(compile_call.call_count, 1)
        inference = mock.Mock(wraps=compiled._compiled_action_inference)
        compiled._compiled_action_inference = inference
        for count in (0, 6, 10):
            observation = _observation(count, self.scales)
            np.testing.assert_allclose(
                compiled.select_action(observation),
                eager.select_action(observation),
                rtol=1e-4, atol=1e-5,
            )
            self.assertEqual(compiled.actor.snn_head.lif1.v, 0.0)
            self.assertEqual(compiled.actor.snn_head.lif2.v, 0.0)
        self.assertEqual(inference.call_count, 3)

        before = compiled.select_action(_observation(2, self.scales))
        with torch.no_grad():
            compiled.actor.snn_head.fc2.bias.add_(0.05)
        after = compiled.select_action(_observation(2, self.scales))
        self.assertFalse(np.array_equal(before, after))

    def test_snn_action_inference_compile_resets_state_on_failure_and_after_update(self) -> None:
        engine = self.make_engine(time_window=4)
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_action_inference_compile(backend='eager')
        original = engine._compiled_action_inference
        engine._compiled_action_inference = mock.Mock(
            side_effect=RuntimeError('controlled inference failure')
        )
        engine.actor.snn_head.lif1.v = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, 'controlled inference failure'):
            engine.select_action(_observation(6, self.scales))
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        engine.actor.snn_head.lif1.v = torch.ones(1)
        with mock.patch.object(
            engine.actor.zone_set_encoder,
            'prepare_tensor_forward_arguments',
            side_effect=RuntimeError('controlled preparation failure'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'controlled preparation failure'):
                engine.select_action(_observation(6, self.scales))
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        engine._compiled_action_inference = original

        engine.select_action(_observation(0, self.scales))
        metrics = engine.update_once(total_steps=1, bc_lambda=0.0)
        self.assertTrue(metrics.actor_updated)
        engine.select_action(_observation(10, self.scales))
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor.snn_head.lif2.v, 0.0)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in engine.actor.parameters()
        ))

    def test_snn_action_inference_real_dynamo_eager_matches_eager(self) -> None:
        engine = self.make_engine(time_window=4)
        observations = tuple(
            _observation(count, self.scales) for count in (0, 6, 10)
        )
        expected = tuple(engine.select_action(value) for value in observations)
        engine.enable_action_inference_compile(
            backend='eager', fullgraph=True, dynamic=True,
        )
        engine.warmup_action_inference_compile(tuple(
            collate_v2_observations([value]) for value in observations
        ))
        for observation, reference in zip(observations, expected):
            np.testing.assert_allclose(
                engine.select_action(observation), reference,
                rtol=1e-4, atol=1e-5,
            )
            self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
            self.assertEqual(engine.actor.snn_head.lif2.v, 0.0)
        with torch.no_grad():
            engine.actor.zone_set_encoder.empty_scene_token.add_(0.03)
        observation = observations[0]
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

    def test_snn_training_and_action_inference_have_separate_dynamo_code(self) -> None:
        engine = self.make_engine(time_window=4)
        engine.configure_compilation(
            compile_actors=True, compile_action_inference=True,
            backend='eager', fullgraph=True, dynamic=True,
        )
        encoder = engine.actor.zone_set_encoder
        training_code = inspect.unwrap(encoder._compiled_tensor_forward).__code__
        inference_code = inspect.unwrap(engine._compiled_action_inference).__code__
        self.assertIs(training_code, encoder._online_actor_context_tensors.__code__)
        self.assertIsNot(inference_code, training_code)

        for count in (0, 6):
            observation = _observation(count, self.scales)
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
            self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
            self.assertEqual(engine.actor.snn_head.lif2.v, 0.0)

        observation = _observation(6, self.scales)
        actor_before = deepcopy(engine.actor.state_dict())
        metrics = engine.update_once(total_steps=1, bc_lambda=0.0)
        self.assertTrue(metrics.actor_updated)
        self.assertTrue(any(
            not torch.equal(actor_before[name], value)
            for name, value in engine.actor.state_dict().items()
            if value.is_floating_point()
        ))
        self.assertTrue(any(parameter.grad is not None for parameter in engine.actor.parameters()))
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
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor.snn_head.lif2.v, 0.0)

    def test_snn_action_inference_warmup_restores_persistent_state(self) -> None:
        engine = self.make_engine(time_window=4)
        batches = tuple(collate_v2_observations([
            _observation(count, self.scales)
        ]) for count in (0, 6, 10))
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            engine.enable_action_inference_compile(backend='eager')
        engine.actor.snn_head.lif1.v = torch.tensor([3.0])
        engine.actor.snn_head.lif2.v = 4.0
        engine.warmup_action_inference_compile(batches)
        torch.testing.assert_close(
            engine.actor.snn_head.lif1.v, torch.tensor([3.0])
        )
        self.assertEqual(engine.actor.snn_head.lif2.v, 4.0)

    def test_snn_formal_compile_roles_have_separate_code_and_warmup(self) -> None:
        engine = self.make_engine(bc=self.make_actor(time_window=4), time_window=4)
        engine.configure_compilation(
            compile_actors=True, compile_critic_block=True,
            compile_target_block=True, compile_snn_target_encoder=True,
            frozen_critic_strategy='compiled_no_grad_context', backend='eager',
        )
        encoders = (
            engine.actor.zone_set_encoder, engine.bc_reference_actor.zone_set_encoder,
            engine.actor_target.zone_set_encoder, engine.critic1.zone_set_encoder,
        )
        codes = tuple(inspect.unwrap(e._compiled_tensor_forward).__code__ for e in encoders)
        self.assertEqual(len(set(codes)), 4)
        self.assertTrue(all(c is not encoders[0]._compute_policy_context_tensors.__code__ for c in codes))
        batches = tuple(collate_v2_observations([
            _observation(count, self.scales), _observation(count, self.scales),
        ]) for count in (0, 6, 10))
        before = deepcopy(engine.actor.state_dict())
        engine.warmup_actor_compile(batches)
        engine.warmup_full_compile(batches)
        for name, value in engine.actor.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertEqual(engine.update_count, 0)
        self.assertEqual(engine.actor_optimizer.state, {})
        observation = batches[1]
        actions = torch.zeros((2, 2), requires_grad=True)
        with engine._actor_critic_guidance(
            observation, shared_relations=engine._build_shared_relations(observation),
            profile_sections=False,
        ) as context:
            engine.critic1.forward_from_context(context, actions).sum().backward()
        self.assertIsNotNone(actions.grad)
        self.assertTrue(torch.isfinite(actions.grad).all())
        metrics = engine.update_once(total_steps=1, bc_lambda=1.75)
        self.assertTrue(metrics.actor_updated)
        self.assertTrue(any(p.grad is not None for p in engine.actor.parameters()))
        self.assertTrue(any(p.grad is not None for p in engine.critic1.parameters()))
        for frozen in (engine.actor_target, engine.bc_reference_actor):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in frozen.parameters()))
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor_target.snn_head.lif2.v, 0.0)

    def test_snn_cuda_graph_update_registration_keeps_lif_outside_graph(self) -> None:
        engine = self.make_engine()
        engine.device = torch.device('cuda')
        compile_calls = []

        def capture(function, **kwargs):
            compile_calls.append((function, kwargs))
            return function

        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile', side_effect=capture,
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile', side_effect=capture,
        ):
            metadata = engine.configure_compilation(
                compile_actors=True,
                compile_critic_block=True,
                compile_target_block=True,
                compile_actor_loss=True,
                cuda_graph_updates=True,
            )

        graph_functions = {
            function.__name__ for function, kwargs in compile_calls
            if kwargs.get('options') == {'triton.cudagraphs': True}
        }
        self.assertEqual(graph_functions, {
            '_actor_loss_tensor_block',
            '_compute_twin_critic_loss_tensors',
            '_compute_target_critics_td_tensors',
        })
        self.assertNotIn('actor_target.eager_snn', metadata['cuda_graph_scope'])
        self.assertEqual(metadata['actor_granularity'], 'ann_full_forward_or_snn_encoder')

    def test_snn_full_target_scope_keeps_lif_actor_eager(self) -> None:
        engine = self.make_engine(bc=self.make_actor())
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            metadata = engine.configure_compilation(
                compile_actors=True,
                frozen_critic_strategy='compiled_no_grad_context',
                compile_critic_block=True,
                compile_target_block=True,
            )
        self.assertIn('actor_target.eager_snn', metadata['enabled_objects'])
        self.assertIn('critic1_target.full_forward', metadata['enabled_objects'])
        self.assertFalse(hasattr(engine.actor_target, 'compiled_full_forward_enabled'))
        self.assertTrue(engine.actor.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertTrue(
            engine.bc_reference_actor.zone_set_encoder.compiled_tensor_forward_enabled
        )
        metrics = engine.update_once(total_steps=1, bc_lambda=0.0)
        self.assertTrue(metrics.actor_updated)
        self.assertEqual(engine.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor_target.snn_head.lif2.v, 0.0)

    def test_snn_target_encoder_compile_is_frozen_encoder_only_and_warmup_is_clean(self) -> None:
        engine = self.make_engine()
        batch = collate_v2_observations([
            _observation(0, self.scales),
            _observation(10, self.scales),
        ])
        state_before = deepcopy(engine.actor_target.state_dict())
        counts_before = (
            engine.update_count, engine.critic_update_count,
            engine.critic_target_update_count, engine.actor_update_count,
            engine.last_total_steps,
        )
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            metadata = engine.configure_compilation(
                compile_snn_target_encoder=True,
            )
        self.assertEqual(metadata['snn_target_actor_granularity'], 'encoder')
        self.assertTrue(engine.actor_target.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertFalse(engine.actor.zone_set_encoder.compiled_tensor_forward_enabled)
        self.assertTrue(all(not parameter.requires_grad for parameter in engine.actor_target.parameters()))
        engine.warmup_snn_target_encoder_compile((batch,))
        self.assertEqual(engine.actor_target.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor_target.snn_head.lif2.v, 0.0)
        self.assertEqual(counts_before, (
            engine.update_count, engine.critic_update_count,
            engine.critic_target_update_count, engine.actor_update_count,
            engine.last_total_steps,
        ))
        for name, expected in state_before.items():
            torch.testing.assert_close(engine.actor_target.state_dict()[name], expected)
        with torch.no_grad():
            output = engine.actor_target(batch)
        self.assertFalse(output.requires_grad)

    def test_snn_target_encoder_compile_reads_soft_updated_weights(self) -> None:
        torch.manual_seed(777)
        eager = self.make_engine()
        compiled = self.make_engine()
        compiled.load_checkpoint_state_dict(eager.checkpoint_state_dict())
        batch = collate_v2_observations([
            _observation(0, self.scales),
            _observation(10, self.scales),
        ])
        compiled.enable_snn_target_encoder_compile(backend='eager')
        compiled.warmup_snn_target_encoder_compile((batch,))
        with torch.no_grad():
            before = compiled.actor_target(batch)
            eager_parameter = eager.actor.zone_set_encoder.pooling.value.weight
            compiled_parameter = compiled.actor.zone_set_encoder.pooling.value.weight
            eager_parameter.add_(0.25)
            compiled_parameter.add_(0.25)
            eager._soft_update(eager.actor, eager.actor_target)
            compiled._soft_update(compiled.actor, compiled.actor_target)
            expected = eager.actor_target(batch)
            actual = compiled.actor_target(batch)
        self.assertFalse(torch.equal(actual, before))
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_snn_target_encoder_compile_combinations_are_precisely_checked(self) -> None:
        ann_actor = V2ANNPolicyActor(
            self.scales, 2, 8, torch.tensor([0.2, 0.3], dtype=torch.float32)
        )
        ann_engine = V2TD3UpdateEngine(
            ann_actor, V2ANNCritic(self.scales, 2, 8),
            V2ANNCritic(self.scales, 2, 8), V2ReplayBuffer(8, 2, 10),
            actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.25,
            policy_noise=0.01, noise_clip=0.02, policy_delay=1,
            batch_size=2, action_low=np.array([-0.2, -0.3], dtype=np.float32),
            action_high=np.array([0.2, 0.3], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, 'requires an SNN actor'):
            ann_engine.configure_compilation(compile_snn_target_encoder=True)
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            self.make_engine().configure_compilation(
                compile_critic_encoder=True,
                compile_target_encoders=True,
                compile_snn_target_encoder=True,
            )
        engine = self.make_engine()
        with mock.patch(
            'brain_uav.trainers.v2_td3.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ), mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            metadata = engine.configure_compilation(
                compile_critic_block=True,
                compile_target_block=True,
                compile_snn_target_encoder=True,
            )
        self.assertIn('actor_target.compiled_snn_encoder', metadata['enabled_objects'])
        metrics = engine.update_once(total_steps=1)
        self.assertTrue(metrics.critic_updated)
        self.assertEqual(engine.actor_target.snn_head.lif1.v, 0.0)
        self.assertEqual(engine.actor_target.snn_head.lif2.v, 0.0)

    def test_snn_actor_target_uses_batched_soft_update_reference_formula(self) -> None:
        engine = self.make_engine()
        before = {
            name: parameter.detach().clone()
            for name, parameter in engine.actor_target.named_parameters()
        }
        with torch.no_grad():
            for parameter in engine.actor.parameters():
                parameter.add_(0.25)
        expected = {
            name: before[name] * 0.75 + parameter.detach() * 0.25
            for name, parameter in engine.actor.named_parameters()
        }

        engine._soft_update(engine.actor, engine.actor_target)

        for name, parameter in engine.actor_target.named_parameters():
            torch.testing.assert_close(parameter, expected[name])

    def test_snn_update_with_reused_relations_matches_original_path(self) -> None:
        torch.manual_seed(8642)
        original = self.make_engine()
        torch.manual_seed(8642)
        reused = self.make_engine()
        torch.manual_seed(7531)
        original_metrics = original.update_once(
            total_steps=1,
            reuse_shared_relations=False,
        )
        torch.manual_seed(7531)
        reused_metrics = reused.update_once(total_steps=1)

        self.assertEqual(reused_metrics, original_metrics)
        for reused_model, original_model in (
            (reused.actor, original.actor),
            (reused.critic1, original.critic1),
            (reused.critic2, original.critic2),
            (reused.actor_target, original.actor_target),
            (reused.critic1_target, original.critic1_target),
            (reused.critic2_target, original.critic2_target),
        ):
            for name, value in reused_model.state_dict().items():
                torch.testing.assert_close(
                    value, original_model.state_dict()[name]
                )

    def test_snn_fixed_buffers_are_validated_before_training(self) -> None:
        actor = self.make_actor()
        with torch.no_grad():
            actor.zone_set_encoder.pair_relation_builder.uav_radius.add_(0.5)
        with self.assertRaisesRegex(ValueError, 'fixed buffer.*uav_radius'):
            V2TD3UpdateEngine(
                actor,
                V2ANNCritic(self.scales, 2, 8),
                V2ANNCritic(self.scales, 2, 8),
                V2ReplayBuffer(8, 2, 10, seed=31),
                actor_lr=1e-3,
                critic_lr=1e-3,
                gamma=0.99,
                tau=0.25,
                policy_noise=0.01,
                noise_clip=0.02,
                policy_delay=1,
                batch_size=2,
                action_low=np.array([-0.2, -0.3], dtype=np.float32),
                action_high=np.array([0.2, 0.3], dtype=np.float32),
                terminal_geo_regularization_enabled=False,
            )

    def test_snn_actor_update_does_not_accumulate_critic_parameter_gradients(self) -> None:
        engine = self.make_engine()
        hook_counts = [0 for _ in engine.critic1.parameters()]
        handles = []
        for index, parameter in enumerate(engine.critic1.parameters()):
            def count_hook(gradient, *, slot=index):
                hook_counts[slot] += 1
                return gradient

            handles.append(parameter.register_hook(count_hook))
        original_states = [
            parameter.requires_grad for parameter in engine.critic1.parameters()
        ]
        try:
            metrics = engine.update_once(total_steps=1, bc_lambda=0.0)
        finally:
            for handle in handles:
                handle.remove()

        self.assertTrue(metrics.actor_updated)
        self.assertTrue(any(count == 1 for count in hook_counts))
        self.assertTrue(all(count <= 1 for count in hook_counts))
        self.assertTrue(any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad) > 0
            for parameter in engine.actor.parameters()
        ))
        self.assertEqual(
            [parameter.requires_grad for parameter in engine.critic1.parameters()],
            original_states,
        )

    def test_snn_bc_reference_is_frozen_independent_and_receives_no_gradient(self) -> None:
        reference = self.make_actor()
        engine = self.make_engine(bc=reference)
        metrics = engine.update_once(total_steps=1, bc_lambda=2.0)
        self.assertGreaterEqual(metrics.bc_loss, 0.0)
        self.assertIsInstance(engine.bc_reference_actor, V2SNNPolicyActor)
        self.assertTrue(all(not p.requires_grad for p in engine.bc_reference_actor.parameters()))
        self.assertTrue(all(p.grad is None for p in engine.bc_reference_actor.parameters()))
        actor_ids = {id(p) for p in engine.actor.parameters()}
        reference_ids = {id(p) for p in engine.bc_reference_actor.parameters()}
        self.assertTrue(actor_ids.isdisjoint(reference_ids))

    def test_actor_and_reference_model_types_must_match(self) -> None:
        ann = V2ANNPolicyActor(
            self.scales,
            2,
            8,
            torch.tensor([0.2, 0.3], dtype=torch.float32),
        )
        with self.assertRaisesRegex((TypeError, ValueError), 'same|model'):
            self.make_engine(bc=ann)

    def test_snn_checkpoint_round_trip_is_distinct_and_has_no_membrane_state(self) -> None:
        engine = self.make_engine(bc=self.make_actor())
        engine.update_once(total_steps=1, bc_lambda=1.0)
        payload = engine.checkpoint_state_dict()
        self.assertEqual(payload['format'], V2_SNN_TD3_CHECKPOINT_FORMAT)
        self.assertEqual(payload['model_type'], 'snn')
        self.assertEqual(payload['architecture']['time_window'], 2)
        all_keys = set(payload['actor_state_dict']) | set(
            payload['actor_target_state_dict']
        )
        self.assertFalse(any(name.endswith('.v') or 'membrane' in name for name in all_keys))

        restored = self.make_engine()
        restored.load_checkpoint_state_dict(payload)
        self.assertIsInstance(restored.bc_reference_actor, V2SNNPolicyActor)
        for name, value in engine.actor.state_dict().items():
            self.assertTrue(torch.equal(value, restored.actor.state_dict()[name]))

        ann_actor = V2ANNPolicyActor(
            self.scales, 2, 8, torch.tensor([0.2, 0.3], dtype=torch.float32)
        )
        ann_engine = V2TD3UpdateEngine(
            ann_actor,
            V2ANNCritic(self.scales, 2, 8),
            V2ANNCritic(self.scales, 2, 8),
            V2ReplayBuffer(8, 2, 10),
            1e-3,
            1e-3,
            0.99,
            0.1,
            0.01,
            0.02,
            1,
            2,
            np.array([-0.2, -0.3], dtype=np.float32),
            np.array([0.2, 0.3], dtype=np.float32),
            terminal_geo_regularization_enabled=False,
        )
        with self.assertRaisesRegex(ValueError, 'format'):
            ann_engine.load_checkpoint_state_dict(payload)
        with self.assertRaisesRegex(ValueError, 'format'):
            restored.load_checkpoint_state_dict(ann_engine.checkpoint_state_dict())


if __name__ == '__main__':
    unittest.main()
