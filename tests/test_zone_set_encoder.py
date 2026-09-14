"""Tests for the standalone relation-aware dynamic zone-set encoder."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
from types import MethodType
import unittest
from unittest import mock

import torch

from brain_uav.models import (
    RelationAttentionBlock,
    RelationAwareSelfAttention,
    TaskConditionedPooling,
    TaskEncoder,
    ZoneEncoder,
    ZoneSetEncoder,
    ZoneSetEncoderConfig,
    ZoneSetEncoderDiagnostics,
)
from brain_uav.observations import (
    EGO_FEATURE_DIM,
    EGO_FEATURE_INDEX,
    GOAL_FEATURE_DIM,
    PAIR_RELATION_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    ZONE_FEATURE_INDEX,
    V2ObservationScales,
)


def _reference_attention_forward(
    attention,
    tokens,
    valid_token_mask,
    pair_relations,
    relation_pair_mask,
    *,
    return_attention_weights=False,
):
    """Pre-optimization attention path with three independent Q/K/V linears."""

    batch_size, token_count, _ = tokens.shape
    clean_relations = torch.where(
        relation_pair_mask.unsqueeze(-1),
        pair_relations,
        torch.zeros_like(pair_relations),
    )
    queries = attention._split_heads(attention.query(tokens))
    keys = attention._split_heads(attention.key(tokens))
    values = attention._split_heads(attention.value(tokens))
    relation_bias = attention.relation_bias(clean_relations).permute(0, 3, 1, 2)
    relation_bias = relation_bias * relation_pair_mask.unsqueeze(1).to(
        relation_bias.dtype
    )
    relation_values = attention.relation_value(clean_relations).reshape(
        batch_size,
        token_count,
        token_count,
        attention.num_heads,
        attention.head_dim,
    ).permute(0, 3, 1, 2, 4)
    relation_values = relation_values * relation_pair_mask[
        :, None, :, :, None
    ].to(relation_values.dtype)

    scores = torch.matmul(queries, keys.transpose(-2, -1))
    scores = scores * attention.score_scale + relation_bias
    key_mask = valid_token_mask[:, None, None, :]
    scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    attention_weights = torch.softmax(scores, dim=-1)
    valid_attention = (
        valid_token_mask[:, None, :, None]
        & valid_token_mask[:, None, None, :]
    )
    attention_weights = torch.where(
        valid_attention,
        attention_weights,
        torch.zeros_like(attention_weights),
    )
    normalizer = attention_weights.sum(dim=-1, keepdim=True)
    attention_weights = torch.where(
        valid_token_mask[:, None, :, None],
        attention_weights / normalizer.clamp_min(
            torch.finfo(attention_weights.dtype).tiny
        ),
        torch.zeros_like(attention_weights),
    )

    standard_messages = torch.matmul(attention_weights, values)
    relation_messages = torch.einsum(
        'bhij,bhijd->bhid',
        attention_weights,
        relation_values,
    )
    contextual = standard_messages + relation_messages
    contextual = contextual.transpose(1, 2).contiguous().reshape(
        batch_size,
        token_count,
        attention.hidden_dim,
    )
    contextual = attention.output(contextual)
    contextual = contextual * valid_token_mask.unsqueeze(-1).to(contextual.dtype)
    if return_attention_weights:
        return contextual, attention_weights
    return contextual


def _use_reference_attention(model):
    for layer in model.layers:
        layer.attention.forward = MethodType(
            _reference_attention_forward,
            layer.attention,
        )


class TestZoneSetEncoder(unittest.TestCase):
    def setUp(self):
        self.scales = V2ObservationScales(100.0, 0.0, 50.0, math.pi / 4.0)
        torch.manual_seed(20260903)
        self.model = ZoneSetEncoder(self.scales)
        self.model.eval()

    def inputs(self, counts):
        batch_size = len(counts)
        max_count = max(counts, default=0)
        ego = torch.zeros((batch_size, EGO_FEATURE_DIM), dtype=torch.float32)
        goal = torch.zeros((batch_size, GOAL_FEATURE_DIM), dtype=torch.float32)
        zones = torch.zeros(
            (batch_size, max_count, ZONE_FEATURE_DIM),
            dtype=torch.float32,
        )
        mask = torch.zeros((batch_size, max_count), dtype=torch.bool)
        for batch_index, count in enumerate(counts):
            angle = 0.2 * (batch_index + 1)
            ego[batch_index, EGO_FEATURE_INDEX['uav_x_norm']] = 0.1 * batch_index
            ego[batch_index, EGO_FEATURE_INDEX['uav_z_fraction']] = 0.3
            ego[batch_index, EGO_FEATURE_INDEX['sin_psi']] = math.sin(angle)
            ego[batch_index, EGO_FEATURE_INDEX['cos_psi']] = math.cos(angle)
            goal[batch_index] = torch.tensor(
                [0.2 + batch_index, -0.1, 0.05, 0.4],
                dtype=torch.float32,
            )
            for zone_index in range(count):
                row = zones[batch_index, zone_index]
                row[ZONE_FEATURE_INDEX['shape_sphere']] = 1.0
                row[ZONE_FEATURE_INDEX['zone_forward_norm']] = (
                    -0.3 + 0.11 * zone_index
                )
                row[ZONE_FEATURE_INDEX['zone_right_norm']] = (
                    (-1.0) ** zone_index * (0.04 + 0.02 * zone_index)
                )
                row[ZONE_FEATURE_INDEX['zone_up_norm']] = (
                    -0.1 + 0.05 * zone_index
                )
                row[ZONE_FEATURE_INDEX['extent_x_norm']] = 0.05 + 0.005 * zone_index
                row[ZONE_FEATURE_INDEX['extent_y_norm']] = 0.04 + 0.003 * zone_index
                row[ZONE_FEATURE_INDEX['extent_z_norm']] = 0.08 + 0.004 * zone_index
                row[ZONE_FEATURE_INDEX['safety_margin_norm']] = 0.01
                row[ZONE_FEATURE_INDEX['point_clearance_norm']] = 0.2
                row[ZONE_FEATURE_INDEX['surface_normal_forward']] = -1.0
                row[ZONE_FEATURE_INDEX['approach_cosine']] = 0.5
                row[ZONE_FEATURE_INDEX['raw_goal_path_intersects']] = float(
                    zone_index % 2 == 0
                )
                row[ZONE_FEATURE_INDEX['raw_first_intersection_fraction']] = (
                    0.25 + 0.02 * zone_index
                )
            mask[batch_index, :count] = True
        return ego, goal, zones, mask

    def test_config_defaults_validation_and_component_interfaces(self):
        config = ZoneSetEncoderConfig()
        self.assertEqual(config.ego_dim, EGO_FEATURE_DIM)
        self.assertEqual(config.goal_dim, GOAL_FEATURE_DIM)
        self.assertEqual(config.zone_dim, ZONE_FEATURE_DIM)
        self.assertEqual(config.relation_dim, PAIR_RELATION_FEATURE_DIM)
        self.assertEqual(config.hidden_dim, 64)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.ffn_dim, 128)
        self.assertEqual(config.dropout, 0.0)
        self.assertFalse(hasattr(config, 'max_zones'))

        invalid = (
            {'ego_dim': 0},
            {'goal_dim': 0},
            {'zone_dim': 0},
            {'relation_dim': 0},
            {'hidden_dim': 0},
            {'hidden_dim': 63, 'num_heads': 4},
            {'num_heads': 0},
            {'num_layers': 0},
            {'ffn_dim': 0},
            {'dropout': 0.1},
            {'ego_dim': EGO_FEATURE_DIM + 1},
            {'goal_dim': GOAL_FEATURE_DIM + 1},
            {'zone_dim': ZONE_FEATURE_DIM + 1},
            {'relation_dim': PAIR_RELATION_FEATURE_DIM + 1},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                ZoneSetEncoderConfig(**values)

        self.assertIsInstance(self.model.zone_encoder, ZoneEncoder)
        self.assertIsInstance(self.model.task_encoder, TaskEncoder)
        self.assertIsInstance(self.model.layers[0], RelationAttentionBlock)
        self.assertIsInstance(
            self.model.layers[0].attention,
            RelationAwareSelfAttention,
        )
        self.assertIsInstance(self.model.pooling, TaskConditionedPooling)
        self.assertEqual(self.model.output_dim, 128)
        self.assertEqual(self.model.zone_encoder.net[0].in_features, ZONE_FEATURE_DIM)
        self.assertEqual(self.model.zone_encoder.net[0].out_features, 64)
        self.assertEqual(
            sum(parameter.numel() for parameter in self.model.parameters()),
            91_816,
        )
        self.assertFalse(
            any(
                'slot' in name.lower() or 'position' in name.lower()
                for name, _ in self.model.named_parameters()
            )
        )

    def test_default_forward_and_counts_zero_one_five_six_ten(self):
        initial_shapes = {
            name: tuple(value.shape)
            for name, value in self.model.state_dict().items()
        }
        for count in (0, 1, 5, 6, 10):
            with self.subTest(count=count):
                output = self.model(*self.inputs([count]))
                self.assertIsInstance(output, torch.Tensor)
                self.assertEqual(output.shape, (1, 128))
                self.assertTrue(torch.isfinite(output).all())
        final_shapes = {
            name: tuple(value.shape)
            for name, value in self.model.state_dict().items()
        }
        self.assertEqual(final_shapes, initial_shapes)

    def test_fast_forward_and_diagnostics_do_not_convert_tensor_contents(self):
        inputs = self.inputs([0, 2, 6])

        with (
            mock.patch.object(
                torch.Tensor,
                '__bool__',
                side_effect=AssertionError('Tensor.__bool__ must not be called.'),
            ),
            mock.patch.object(
                torch.Tensor,
                'item',
                side_effect=AssertionError('Tensor.item must not be called.'),
            ),
        ):
            output = self.model(*inputs)
            diagnostics = self.model.forward_with_diagnostics(*inputs)

        self.assertEqual(output.shape, (3, 128))
        self.assertEqual(diagnostics.policy_context.shape, (3, 128))

    def test_attention_all_false_mask_is_finite_without_scalar_conversion(self):
        config = ZoneSetEncoderConfig()
        attention = RelationAwareSelfAttention(config).eval()
        tokens = torch.randn((2, 3, config.hidden_dim), dtype=torch.float32)
        valid_mask = torch.zeros((2, 3), dtype=torch.bool)
        pair_relations = torch.zeros(
            (2, 3, 3, config.relation_dim),
            dtype=torch.float32,
        )
        relation_mask = torch.zeros((2, 3, 3), dtype=torch.bool)

        with (
            mock.patch.object(
                torch.Tensor,
                '__bool__',
                side_effect=AssertionError('Tensor.__bool__ must not be called.'),
            ),
            mock.patch.object(
                torch.Tensor,
                'item',
                side_effect=AssertionError('Tensor.item must not be called.'),
            ),
        ):
            output, weights = attention(
                tokens,
                valid_mask,
                pair_relations,
                relation_mask,
                return_attention_weights=True,
            )

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(output, torch.zeros_like(output))
        torch.testing.assert_close(weights, torch.zeros_like(weights))

    def test_attention_uses_one_merged_qkv_linear(self):
        config = ZoneSetEncoderConfig()
        attention = RelationAwareSelfAttention(config).eval()
        tokens = torch.randn((2, 4, config.hidden_dim), dtype=torch.float32)
        valid_mask = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )
        pair_relations = torch.randn(
            (2, 4, 4, config.relation_dim),
            dtype=torch.float32,
        )
        relation_mask = valid_mask[:, :, None] & valid_mask[:, None, :]

        original_linear = torch.nn.functional.linear
        with mock.patch(
            'brain_uav.models.zone_set_encoder.F.linear',
            wraps=original_linear,
        ) as linear:
            attention(tokens, valid_mask, pair_relations, relation_mask)

        self.assertEqual(linear.call_count, 4)

    def test_merged_qkv_matches_independent_reference_outputs_and_gradients(self):
        for counts in ([0], [0, 3, 7]):
            with self.subTest(counts=counts):
                torch.manual_seed(731)
                model = ZoneSetEncoder(self.scales).train()
                reference = deepcopy(model)
                _use_reference_attention(reference)
                actual_inputs = tuple(
                    value.clone().requires_grad_(value.dtype == torch.float32)
                    for value in self.inputs(counts)
                )
                reference_inputs = tuple(
                    value.detach().clone().requires_grad_(value.dtype == torch.float32)
                    for value in actual_inputs
                )

                actual_output = model(*actual_inputs)
                reference_output = reference(*reference_inputs)
                actual_output.square().sum().backward()
                reference_output.square().sum().backward()

                torch.testing.assert_close(
                    actual_output,
                    reference_output,
                    atol=2e-6,
                    rtol=2e-6,
                )
                for actual, expected in zip(actual_inputs[:3], reference_inputs[:3]):
                    torch.testing.assert_close(
                        actual.grad,
                        expected.grad,
                        atol=2e-6,
                        rtol=2e-6,
                    )
                for layer_index, (actual_layer, reference_layer) in enumerate(
                    zip(model.layers, reference.layers)
                ):
                    for projection_name in ('query', 'key', 'value'):
                        actual_projection = getattr(
                            actual_layer.attention,
                            projection_name,
                        )
                        reference_projection = getattr(
                            reference_layer.attention,
                            projection_name,
                        )
                        for parameter_name in ('weight', 'bias'):
                            with self.subTest(
                                layer=layer_index,
                                projection=projection_name,
                                parameter=parameter_name,
                            ):
                                torch.testing.assert_close(
                                    getattr(actual_projection, parameter_name).grad,
                                    getattr(reference_projection, parameter_name).grad,
                                    atol=2e-6,
                                    rtol=2e-6,
                                )

    def test_merged_qkv_preserves_parameters_checkpoint_and_rng_contract(self):
        torch.manual_seed(90210)
        attention = RelationAwareSelfAttention(ZoneSetEncoderConfig())
        random_state = torch.random.get_rng_state().clone()
        state = deepcopy(attention.state_dict())
        parameter_ids = {
            name: id(parameter)
            for name, parameter in attention.named_parameters()
        }

        torch.manual_seed(90210)
        restored = RelationAwareSelfAttention(ZoneSetEncoderConfig())
        torch.testing.assert_close(torch.random.get_rng_state(), random_state)
        restored.load_state_dict(state, strict=True)
        self.assertEqual(tuple(restored.state_dict()), tuple(state))

        tokens = torch.randn((1, 2, attention.hidden_dim), dtype=torch.float32)
        valid_mask = torch.ones((1, 2), dtype=torch.bool)
        pair_relations = torch.zeros(
            (1, 2, 2, attention.relation_bias.in_features),
            dtype=torch.float32,
        )
        relation_mask = torch.ones((1, 2, 2), dtype=torch.bool)
        attention(tokens, valid_mask, pair_relations, relation_mask)
        self.assertEqual(
            {
                name: id(parameter)
                for name, parameter in attention.named_parameters()
            },
            parameter_ids,
        )

    def test_merged_qkv_two_adam_updates_match_independent_reference(self):
        torch.manual_seed(481)
        attention = RelationAwareSelfAttention(ZoneSetEncoderConfig()).train()
        reference = deepcopy(attention)
        reference.forward = MethodType(_reference_attention_forward, reference)
        optimizer = torch.optim.Adam(attention.parameters(), lr=1e-2)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-2)
        tokens = torch.randn((2, 8, attention.hidden_dim), dtype=torch.float32)
        valid_mask = torch.tensor(
            [
                [True, False, False, False, False, False, False, False],
                [True, True, True, True, True, True, True, True],
            ]
        )
        pair_relations = torch.randn(
            (2, 8, 8, attention.relation_bias.in_features),
            dtype=torch.float32,
        )
        relation_mask = valid_mask[:, :, None] & valid_mask[:, None, :]
        previous_output = None

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            actual_output = attention(
                tokens,
                valid_mask,
                pair_relations,
                relation_mask,
            )
            reference_output = reference(
                tokens,
                valid_mask,
                pair_relations,
                relation_mask,
            )
            torch.testing.assert_close(
                actual_output,
                reference_output,
                atol=2e-6,
                rtol=2e-6,
            )
            if previous_output is not None:
                self.assertFalse(torch.equal(actual_output, previous_output))
            actual_output.square().mean().backward()
            reference_output.square().mean().backward()
            optimizer.step()
            reference_optimizer.step()
            previous_output = actual_output.detach().clone()

            for (actual_name, actual_parameter), (
                reference_name,
                reference_parameter,
            ) in zip(attention.named_parameters(), reference.named_parameters()):
                self.assertEqual(actual_name, reference_name)
                torch.testing.assert_close(
                    actual_parameter,
                    reference_parameter,
                    atol=2e-6,
                    rtol=2e-6,
                )
            torch.testing.assert_close(
                optimizer.state_dict(),
                reference_optimizer.state_dict(),
                atol=2e-6,
                rtol=2e-6,
            )

    def test_checked_forward_matches_fast_for_all_supported_counts(self):
        for count in (0, 1, 5, 6, 10):
            inputs = self.inputs([count])
            with self.subTest(count=count):
                fast = self.model(*inputs)
                checked = self.model.forward_checked(*inputs)
                torch.testing.assert_close(checked, fast)

    def test_precomputed_relations_match_original_outputs_and_gradients(self):
        for counts in ([0], [0, 3, 7]):
            with self.subTest(counts=counts):
                model = ZoneSetEncoder(self.scales).train()
                baseline_inputs = tuple(
                    value.clone().requires_grad_(value.dtype == torch.float32)
                    for value in self.inputs(counts)
                )
                baseline = model(*baseline_inputs)
                baseline.square().sum().backward()
                baseline_parameter_grads = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                }
                baseline_input_grads = tuple(
                    value.grad.detach().clone()
                    for value in baseline_inputs[:3]
                )

                model.zero_grad(set_to_none=True)
                shared_inputs = tuple(
                    value.detach().clone().requires_grad_(value.dtype == torch.float32)
                    for value in baseline_inputs
                )
                shared = model.build_shared_relations(*shared_inputs)
                reused = model(*shared_inputs, shared_relations=shared)
                reused.square().sum().backward()

                torch.testing.assert_close(reused, baseline)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad,
                        baseline_parameter_grads[name],
                    )
                for actual, expected in zip(shared_inputs[:3], baseline_input_grads):
                    torch.testing.assert_close(actual.grad, expected)

    def test_compiled_tensor_forward_matches_eager_outputs_and_gradients(self):
        for counts in ([0], [0, 3, 7]):
            with self.subTest(counts=counts):
                model = ZoneSetEncoder(self.scales).train()
                self.assertFalse(model.compiled_tensor_forward_enabled)
                parameter_ids = tuple(id(parameter) for parameter in model.parameters())
                state_keys = tuple(model.state_dict())
                eager_inputs = tuple(
                    value.clone().requires_grad_(value.dtype == torch.float32)
                    for value in self.inputs(counts)
                )
                eager_shared = model.build_shared_relations(*eager_inputs)
                eager = model(*eager_inputs, shared_relations=eager_shared)
                eager.square().sum().backward()
                eager_parameter_grads = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                }
                eager_input_grads = tuple(
                    value.grad.detach().clone() for value in eager_inputs[:3]
                )

                model.zero_grad(set_to_none=True)
                model.enable_compiled_tensor_forward(backend='eager')
                compiled_inputs = tuple(
                    value.detach().clone().requires_grad_(value.dtype == torch.float32)
                    for value in eager_inputs
                )
                compiled_shared = model.build_shared_relations(*compiled_inputs)
                compiled = model(
                    *compiled_inputs,
                    shared_relations=compiled_shared,
                )
                compiled.square().sum().backward()

                torch.testing.assert_close(compiled, eager)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(
                        parameter.grad,
                        eager_parameter_grads[name],
                    )
                for actual, expected in zip(compiled_inputs[:3], eager_input_grads):
                    torch.testing.assert_close(actual.grad, expected)

                self.assertEqual(
                    tuple(id(parameter) for parameter in model.parameters()),
                    parameter_ids,
                )
                self.assertEqual(tuple(model.state_dict()), state_keys)

    def test_eager_tensor_forward_context_bypasses_compile_and_restores(self):
        ego, goal, zones, mask = self.inputs([0, 3])
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            self.model.enable_compiled_tensor_forward(backend='eager')
        compiled = mock.Mock(wraps=self.model._compiled_tensor_forward)
        self.model._compiled_tensor_forward = compiled
        with mock.patch.object(
            self.model,
            '_compute_policy_context_tensors',
            wraps=self.model._compute_policy_context_tensors,
        ) as eager:
            self.model(ego, goal, zones, mask)
            self.assertEqual(compiled.call_count, 1)
            self.assertEqual(eager.call_count, 0)

            with self.model.eager_tensor_forward():
                output = self.model(ego, goal, zones, mask)
                output.sum().backward()
            self.assertEqual(compiled.call_count, 1)
            self.assertEqual(eager.call_count, 1)
            self.assertIsNotNone(self.model.empty_scene_token.grad)

            with self.assertRaisesRegex(RuntimeError, 'controlled eager failure'):
                with self.model.eager_tensor_forward():
                    raise RuntimeError('controlled eager failure')
            self.model(ego, goal, zones, mask)
            self.assertEqual(compiled.call_count, 2)
            self.assertEqual(eager.call_count, 1)

    def test_profiled_forward_marks_only_major_encoder_sections(self):
        inputs = self.inputs([0, 3, 7])
        expected = self.model(*inputs)
        labels = []

        @contextmanager
        def record(label):
            labels.append(label)
            yield

        with mock.patch(
            'brain_uav.models.zone_set_encoder.record_function',
            side_effect=record,
        ):
            shared = self.model.build_shared_relations(
                *inputs,
                profile_sections=True,
            )
            actual = self.model(
                *inputs,
                shared_relations=shared,
                profile_sections=True,
            )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(labels, [
            'v2_encoder.shared_relation_build',
            'v2_encoder.zone_task_encoding',
            'v2_encoder.relation_attention',
            'v2_encoder.task_conditioned_pooling',
        ])

    def test_checked_forward_rejects_nonfinite_inputs_and_output(self):
        ego, goal, zones, mask = self.inputs([2])
        invalid_inputs = (
            (ego.clone().fill_(float('nan')), goal, zones, mask, 'ego_features'),
            (ego.clone().fill_(float('inf')), goal, zones, mask, 'ego_features'),
            (ego, goal.clone().fill_(float('nan')), zones, mask, 'goal_features'),
            (ego, goal.clone().fill_(float('-inf')), zones, mask, 'goal_features'),
            (ego, goal, zones.clone().fill_(float('nan')), mask, 'zone_features'),
            (ego, goal, zones.clone().fill_(float('inf')), mask, 'zone_features'),
        )
        for invalid_ego, invalid_goal, invalid_zones, invalid_mask, name in invalid_inputs:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.model.forward_checked(
                    invalid_ego,
                    invalid_goal,
                    invalid_zones,
                    invalid_mask,
                )

        for value in (float('nan'), float('inf')):
            invalid_output = torch.full((1, 128), value, dtype=torch.float32)
            with (
                self.subTest(output_value=value),
                mock.patch.object(self.model, 'forward', return_value=invalid_output),
                self.assertRaisesRegex(ValueError, 'output'),
            ):
                self.model.forward_checked(ego, goal, zones, mask)

    def test_mixed_counts_diagnostics_empty_token_and_attention_masks(self):
        diagnostics = self.model.forward_with_diagnostics(*self.inputs([0, 2, 5]))

        self.assertIsInstance(diagnostics, ZoneSetEncoderDiagnostics)
        self.assertEqual(diagnostics.policy_context.shape, (3, 128))
        self.assertEqual(
            diagnostics.pair_relations.shape,
            (3, 5, 5, PAIR_RELATION_FEATURE_DIM),
        )
        self.assertEqual(diagnostics.contextual_zone_tokens.shape, (3, 5, 64))
        self.assertEqual(len(diagnostics.self_attention_weights), 2)
        self.assertEqual(diagnostics.pooling_weights.shape, (3, 6))
        self.assertEqual(diagnostics.valid_token_mask.shape, (3, 6))
        expected_mask = torch.tensor(
            [
                [True, False, False, False, False, False],
                [False, True, True, False, False, False],
                [False, True, True, True, True, True],
            ]
        )
        torch.testing.assert_close(diagnostics.valid_token_mask, expected_mask)
        self.assertEqual(float(diagnostics.pooling_weights[0, 0]), 1.0)
        self.assertEqual(
            int(torch.count_nonzero(diagnostics.pooling_weights[0, 1:])),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(diagnostics.pooling_weights[1:, 0])),
            0,
        )
        torch.testing.assert_close(
            diagnostics.pooling_weights.sum(dim=-1),
            torch.ones(3),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(
            int(torch.count_nonzero(diagnostics.contextual_zone_tokens[0])),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(diagnostics.contextual_zone_tokens[1, 2:])),
            0,
        )

        for weights in diagnostics.self_attention_weights:
            self.assertEqual(weights.shape, (3, 4, 6, 6))
            query_mask = diagnostics.valid_token_mask[:, None, :, None].expand_as(
                weights
            )
            key_mask = diagnostics.valid_token_mask[:, None, None, :].expand_as(
                weights
            )
            self.assertEqual(
                int(torch.count_nonzero(weights.masked_select(~query_mask))),
                0,
            )
            self.assertEqual(
                int(torch.count_nonzero(weights.masked_select(~key_mask))),
                0,
            )
            sums = weights.sum(dim=-1)
            expanded_valid = diagnostics.valid_token_mask[:, None, :].expand_as(sums)
            torch.testing.assert_close(
                sums.masked_select(expanded_valid),
                torch.ones_like(sums.masked_select(expanded_valid)),
                atol=1e-6,
                rtol=1e-6,
            )
            self.assertEqual(
                int(torch.count_nonzero(sums.masked_select(~expanded_valid))),
                0,
            )

    def test_zone_permutation_preserves_context_and_permutes_diagnostics(self):
        inputs = self.inputs([6])
        original = self.model.forward_with_diagnostics(*inputs)
        permutation = torch.tensor([3, 0, 5, 1, 4, 2])
        permuted_inputs = (
            inputs[0],
            inputs[1],
            inputs[2][:, permutation],
            inputs[3][:, permutation],
        )
        permuted = self.model.forward_with_diagnostics(*permuted_inputs)

        torch.testing.assert_close(
            permuted.policy_context,
            original.policy_context,
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            permuted.contextual_zone_tokens,
            original.contextual_zone_tokens[:, permutation],
            atol=2e-6,
            rtol=2e-6,
        )
        expected_relations = original.pair_relations[:, permutation][:, :, permutation]
        torch.testing.assert_close(
            permuted.pair_relations,
            expected_relations,
            atol=1e-6,
            rtol=1e-6,
        )
        token_permutation = torch.cat((torch.tensor([0]), permutation + 1))
        torch.testing.assert_close(
            permuted.pooling_weights,
            original.pooling_weights[:, token_permutation],
            atol=2e-6,
            rtol=2e-6,
        )

    def test_masked_garbage_cannot_change_real_outputs_or_weights(self):
        ego, goal, zones, mask = self.inputs([2, 5])
        baseline = self.model.forward_with_diagnostics(ego, goal, zones, mask)
        garbage = zones.clone()
        generator = torch.Generator().manual_seed(444)
        garbage[~mask] = torch.randn(
            (int((~mask).sum()), ZONE_FEATURE_DIM),
            generator=generator,
        ) * 1000.0
        changed = self.model.forward_with_diagnostics(ego, goal, garbage, mask)

        torch.testing.assert_close(
            changed.policy_context,
            baseline.policy_context,
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            changed.contextual_zone_tokens[0, :2],
            baseline.contextual_zone_tokens[0, :2],
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            changed.pooling_weights,
            baseline.pooling_weights,
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertEqual(
            int(torch.count_nonzero(changed.contextual_zone_tokens[0, 2:])),
            0,
        )
        self.assertEqual(
            int(torch.count_nonzero(changed.pooling_weights[0, 3:])),
            0,
        )

    def test_scene_is_invariant_to_larger_padding_from_batch_composition(self):
        single_inputs = self.inputs([2])
        single = self.model.forward_with_diagnostics(*single_inputs)
        mixed_inputs = self.inputs([2, 10])
        mixed = self.model.forward_with_diagnostics(*mixed_inputs)

        torch.testing.assert_close(
            mixed.policy_context[0],
            single.policy_context[0],
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            mixed.contextual_zone_tokens[0, :2],
            single.contextual_zone_tokens[0],
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            mixed.pair_relations[0, :2, :2],
            single.pair_relations[0],
        )
        torch.testing.assert_close(
            mixed.pooling_weights[0, :3],
            single.pooling_weights[0],
            atol=2e-6,
            rtol=2e-6,
        )
        self.assertEqual(
            int(torch.count_nonzero(mixed.pooling_weights[0, 3:])),
            0,
        )

    def test_backward_reaches_all_required_trainable_components(self):
        model = ZoneSetEncoder(self.scales)
        model.train()
        ego, goal, zones, mask = self.inputs([3, 4])
        ego.requires_grad_()
        goal.requires_grad_()
        zones.requires_grad_()

        output = model(ego, goal, zones, mask)
        loss = output.square().mean() + output.mean()
        loss.backward()

        required_modules = {
            'zone_encoder': model.zone_encoder,
            'task_encoder': model.task_encoder,
            'self_attention': model.layers[0].attention,
            'relation_bias': model.layers[0].attention.relation_bias,
            'relation_value': model.layers[0].attention.relation_value,
            'pooling': model.pooling,
        }
        for name, module in required_modules.items():
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            with self.subTest(module=name):
                self.assertTrue(gradients)
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
                )
        self.assertIsNotNone(ego.grad)
        self.assertIsNotNone(goal.grad)
        self.assertIsNotNone(zones.grad)
        self.assertTrue(torch.isfinite(ego.grad).all())
        self.assertTrue(torch.isfinite(goal.grad).all())
        self.assertTrue(torch.isfinite(zones.grad).all())

    def test_initialization_obeys_external_seed_and_forward_preserves_rng(self):
        torch.manual_seed(12345)
        first = ZoneSetEncoder(self.scales)
        first_state = {
            name: value.detach().clone()
            for name, value in first.state_dict().items()
        }
        torch.manual_seed(12345)
        second = ZoneSetEncoder(self.scales)
        second_state = second.state_dict()
        self.assertEqual(first_state.keys(), second_state.keys())
        for name in first_state:
            torch.testing.assert_close(first_state[name], second_state[name])

        torch.manual_seed(54321)
        third = ZoneSetEncoder(self.scales)
        self.assertTrue(
            any(
                not torch.equal(first_state[name], third.state_dict()[name])
                for name in first_state
                if first_state[name].is_floating_point()
                and first_state[name].numel() > 1
            )
        )

        random_state = torch.random.get_rng_state().clone()
        first.eval()
        first(*self.inputs([5]))
        torch.testing.assert_close(torch.random.get_rng_state(), random_state)

    def test_input_validation_rejects_bad_shapes_dtypes_and_devices(self):
        ego, goal, zones, mask = self.inputs([2, 3])
        invalid = (
            (ego.double(), goal, zones, mask),
            (ego, goal.double(), zones, mask),
            (ego, goal, zones.double(), mask),
            (ego, goal, zones, mask.to(torch.uint8)),
            (ego[:, :-1], goal, zones, mask),
            (ego, goal[:, :-1], zones, mask),
            (ego, goal, zones[:, :, :-1], mask),
            (ego, goal, zones, mask[:, :-1]),
            (
                torch.zeros((0, EGO_FEATURE_DIM)),
                torch.zeros((0, GOAL_FEATURE_DIM)),
                torch.zeros((0, 0, ZONE_FEATURE_DIM)),
                torch.zeros((0, 0), dtype=torch.bool),
            ),
        )
        for values in invalid:
            with self.subTest(
                shapes=tuple(tuple(value.shape) for value in values),
                dtypes=tuple(value.dtype for value in values),
            ):
                with self.assertRaises((TypeError, ValueError)):
                    self.model(*values)

        meta_goal = torch.empty((2, GOAL_FEATURE_DIM), device='meta')
        with self.assertRaises(ValueError):
            self.model(ego, meta_goal, zones, mask)


if __name__ == '__main__':
    unittest.main()
