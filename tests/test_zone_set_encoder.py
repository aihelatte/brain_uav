"""Tests for the standalone relation-aware dynamic zone-set encoder."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
from types import MethodType
import unittest
from unittest import mock

import torch
import brain_uav.observations.v2_relations as v2_relations

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
    merge_qkv=False,
    merge_relations=False,
):
    """Independent attention reference with unfused relation projections."""

    batch_size, token_count, _ = tokens.shape
    clean_relations = torch.where(
        relation_pair_mask.unsqueeze(-1),
        pair_relations,
        torch.zeros_like(pair_relations),
    )
    if merge_qkv:
        qkv_weight = torch.cat(
            (attention.query.weight, attention.key.weight, attention.value.weight),
            dim=0,
        )
        qkv_bias = torch.cat(
            (attention.query.bias, attention.key.bias, attention.value.bias),
            dim=0,
        )
        queries, keys, values = torch.nn.functional.linear(
            tokens,
            qkv_weight,
            qkv_bias,
        ).split(attention.hidden_dim, dim=-1)
        queries = attention._split_heads(queries)
        keys = attention._split_heads(keys)
        values = attention._split_heads(values)
    else:
        queries = attention._split_heads(attention.query(tokens))
        keys = attention._split_heads(attention.key(tokens))
        values = attention._split_heads(attention.value(tokens))
    if merge_relations:
        relation_bias, relation_values = torch.nn.functional.linear(
            clean_relations,
            torch.cat((attention.relation_bias.weight, attention.relation_value.weight)),
            torch.cat((attention.relation_bias.bias, attention.relation_value.bias)),
        ).split((attention.relation_bias.out_features, attention.relation_value.out_features), dim=-1)
    else:
        relation_bias = attention.relation_bias(clean_relations)
        relation_values = attention.relation_value(clean_relations)
    relation_bias = relation_bias.permute(0, 3, 1, 2)
    relation_bias = relation_bias * relation_pair_mask.unsqueeze(1).to(
        relation_bias.dtype
    )
    relation_values = relation_values.reshape(
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


def _reference_attention_with_independent_relations(attention, *args, **kwargs):
    return _reference_attention_forward(
        attention,
        *args,
        **kwargs,
        merge_qkv=True,
    )


def _reference_attention_with_merged_relations(attention, *args, **kwargs):
    return _reference_attention_forward(
        attention, *args, **kwargs, merge_qkv=True, merge_relations=True,
    )


def _use_independent_relation_reference(model):
    for layer in model.layers:
        layer.attention.forward = MethodType(
            _reference_attention_with_independent_relations,
            layer.attention,
        )


def _reference_pooling_forward(
    pooling,
    task_embedding,
    contextual_tokens,
    valid_token_mask,
    *,
    return_attention_weights=False,
):
    query = pooling.query(task_embedding).unsqueeze(1)
    keys = pooling.key(contextual_tokens)
    values = pooling.value(contextual_tokens)
    scores = torch.sum(query * keys, dim=-1) * pooling.score_scale
    scores = scores.masked_fill(
        ~valid_token_mask,
        torch.finfo(scores.dtype).min,
    )
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(
        valid_token_mask,
        weights,
        torch.zeros_like(weights),
    )
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    summary = torch.sum(weights.unsqueeze(-1) * values, dim=1)
    if return_attention_weights:
        return summary, weights
    return summary


class _ForbiddenIndexLookup:
    def __getitem__(self, key):
        raise AssertionError(f'compiled relation path looked up {key!r}')


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

    def test_attention_uses_one_merged_linear_per_projection_group(self):
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

        self.assertEqual(linear.call_count, 3)

    def test_relation_value_aggregation_matches_old_merged_order_and_gradients(self):
        for valid_counts in ((0,), (1, 4, 8)):
            with self.subTest(valid_counts=valid_counts):
                torch.manual_seed(641)
                attention = RelationAwareSelfAttention(ZoneSetEncoderConfig()).double().train()
                reference = deepcopy(attention)
                reference.forward = MethodType(
                    _reference_attention_with_merged_relations, reference,
                )
                attention.enable_relation_value_aggregation()
                with torch.no_grad():
                    attention.relation_value.bias.fill_(1.7)
                    reference.load_state_dict(attention.state_dict(), strict=True)
                count = max(max(valid_counts), 1)
                tokens = torch.randn(
                    (len(valid_counts), count, attention.hidden_dim),
                    dtype=torch.float64, requires_grad=True,
                )
                relations = torch.randn(
                    (len(valid_counts), count, count, attention.relation_bias.in_features),
                    dtype=torch.float64, requires_grad=True,
                )
                reference_tokens = tokens.detach().clone().requires_grad_(True)
                reference_relations = relations.detach().clone().requires_grad_(True)
                valid = torch.arange(count)[None, :] < torch.tensor(valid_counts)[:, None]
                relation_mask = valid[:, :, None] & valid[:, None, :]
                relation_mask = relation_mask & ~torch.eye(count, dtype=torch.bool)[None]
                actual, actual_weights = attention(
                    tokens, valid, relations, relation_mask,
                    return_attention_weights=True,
                )
                expected, expected_weights = reference(
                    reference_tokens, valid, reference_relations, relation_mask,
                    return_attention_weights=True,
                )
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(actual_weights, expected_weights, atol=2e-6, rtol=2e-6)
                if count > 1:
                    masked_weight_sum = (
                        actual_weights * relation_mask[:, None]
                    ).sum(dim=-1)
                    self.assertTrue(bool((masked_weight_sum[2, :, 0] < 1.0).all()))
                    self.assertTrue(bool((masked_weight_sum[2, :, 0] > 0.0).all()))
                actual.square().sum().backward()
                expected.square().sum().backward()
                for left, right in ((tokens, reference_tokens), (relations, reference_relations)):
                    self.assertEqual(left.grad is None, right.grad is None)
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-6)
                for (name, left), (expected_name, right) in zip(
                    attention.named_parameters(), reference.named_parameters(),
                ):
                    self.assertEqual(name, expected_name)
                    self.assertEqual(left.grad is None, right.grad is None, name)
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-6)

    def test_relation_value_aggregation_two_adam_updates_use_new_parameters(self):
        torch.manual_seed(643)
        attention = RelationAwareSelfAttention(ZoneSetEncoderConfig()).double().train()
        reference = deepcopy(attention)
        reference.forward = MethodType(_reference_attention_with_merged_relations, reference)
        attention.enable_relation_value_aggregation()
        with torch.no_grad():
            attention.relation_value.bias.fill_(1.7)
            reference.load_state_dict(attention.state_dict(), strict=True)
        parameter_ids = {name: id(p) for name, p in attention.named_parameters()}
        original_keys = tuple(attention.state_dict())
        restored = RelationAwareSelfAttention(ZoneSetEncoderConfig()).double()
        after_initialization = torch.random.get_rng_state().clone()
        restored.enable_relation_value_aggregation()
        torch.testing.assert_close(torch.random.get_rng_state(), after_initialization)
        restored.load_state_dict(reference.state_dict(), strict=True)
        self.assertEqual(tuple(restored.state_dict()), original_keys)
        actual_optimizer = torch.optim.Adam(attention.parameters(), lr=1e-2)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-2)
        tokens = torch.randn((2, 8, attention.hidden_dim), dtype=torch.float64)
        relations = torch.randn((2, 8, 8, attention.relation_bias.in_features), dtype=torch.float64)
        valid = torch.tensor([[True] + [False] * 7, [True] * 8])
        relation_mask = valid[:, :, None] & valid[:, None, :]
        relation_mask &= ~torch.eye(8, dtype=torch.bool)[None]
        previous = None
        for _ in range(2):
            actual_optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            actual = attention(tokens, valid, relations, relation_mask)
            expected = reference(tokens, valid, relations, relation_mask)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            if previous is not None:
                self.assertFalse(torch.equal(actual, previous))
            actual.square().mean().backward()
            expected.square().mean().backward()
            actual_optimizer.step()
            reference_optimizer.step()
            previous = actual.detach().clone()
            for (_, left), (_, right) in zip(
                attention.named_parameters(), reference.named_parameters(),
            ):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(
                actual_optimizer.state_dict(), reference_optimizer.state_dict(),
                atol=2e-6, rtol=2e-6,
            )
        self.assertEqual(tuple(attention.state_dict()), original_keys)
        self.assertEqual(
            {name: id(p) for name, p in attention.named_parameters()}, parameter_ids,
        )

    def test_relation_value_aggregation_tracks_with_real_fullgraph_dynamo_eager(self):
        torch._dynamo.reset()
        torch.manual_seed(647)
        attention = RelationAwareSelfAttention(ZoneSetEncoderConfig()).double().train()
        reference = deepcopy(attention)
        reference.forward = MethodType(_reference_attention_with_merged_relations, reference)
        attention.enable_relation_value_aggregation()
        with torch.no_grad():
            attention.relation_value.bias.fill_(1.7)
            reference.load_state_dict(attention.state_dict(), strict=True)
        compiled = torch.compile(attention, backend='eager', fullgraph=True, dynamic=True)
        tokens = torch.randn((2, 8, attention.hidden_dim), dtype=torch.float64, requires_grad=True)
        relations = torch.randn(
            (2, 8, 8, attention.relation_bias.in_features),
            dtype=torch.float64, requires_grad=True,
        )
        expected_tokens = tokens.detach().clone().requires_grad_(True)
        expected_relations = relations.detach().clone().requires_grad_(True)
        valid = torch.tensor([[True] + [False] * 7, [True] * 8])
        relation_mask = valid[:, :, None] & valid[:, None, :]
        relation_mask &= ~torch.eye(8, dtype=torch.bool)[None]
        actual, actual_weights = compiled(
            tokens, valid, relations, relation_mask, return_attention_weights=True,
        )
        expected, expected_weights = reference(
            expected_tokens, valid, expected_relations, relation_mask,
            return_attention_weights=True,
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual_weights, expected_weights, atol=2e-6, rtol=2e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for left, right in ((tokens, expected_tokens), (relations, expected_relations)):
            self.assertEqual(left.grad is None, right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-6)
        for (_, left), (_, right) in zip(
            attention.named_parameters(), reference.named_parameters(),
        ):
            self.assertEqual(left.grad is None, right.grad is None)
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-6)

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

    def test_merged_relations_match_independent_reference_outputs_and_gradients(self):
        for counts in ([0], [0, 3, 7]):
            with self.subTest(counts=counts):
                torch.manual_seed(733)
                model = ZoneSetEncoder(self.scales).train()
                reference = deepcopy(model)
                _use_independent_relation_reference(reference)
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
                for actual_layer, reference_layer in zip(
                    model.layers,
                    reference.layers,
                ):
                    for projection_name in ('relation_bias', 'relation_value'):
                        actual_projection = getattr(
                            actual_layer.attention,
                            projection_name,
                        )
                        reference_projection = getattr(
                            reference_layer.attention,
                            projection_name,
                        )
                        for parameter_name in ('weight', 'bias'):
                            torch.testing.assert_close(
                                getattr(actual_projection, parameter_name).grad,
                                getattr(reference_projection, parameter_name).grad,
                                atol=2e-6,
                                rtol=2e-6,
                            )

    def test_false_relation_mask_blocks_nonzero_relation_biases(self):
        torch.manual_seed(739)
        attention = RelationAwareSelfAttention(ZoneSetEncoderConfig()).train()
        reference = deepcopy(attention)
        reference.forward = MethodType(
            _reference_attention_with_independent_relations,
            reference,
        )
        with torch.no_grad():
            attention.relation_bias.bias.fill_(3.0)
            attention.relation_value.bias.fill_(-2.0)
            reference.load_state_dict(attention.state_dict(), strict=True)
        tokens = torch.randn((2, 8, attention.hidden_dim), dtype=torch.float32)
        valid_mask = torch.ones((2, 8), dtype=torch.bool)
        pair_relations = torch.randn(
            (2, 8, 8, attention.relation_bias.in_features),
            dtype=torch.float32,
        )
        relation_mask = torch.zeros((2, 8, 8), dtype=torch.bool)

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
        actual_output.square().sum().backward()
        reference_output.square().sum().backward()

        torch.testing.assert_close(
            actual_output,
            reference_output,
            atol=2e-6,
            rtol=2e-6,
        )
        for projection_name in ('relation_bias', 'relation_value'):
            actual_projection = getattr(attention, projection_name)
            reference_projection = getattr(reference, projection_name)
            for parameter_name in ('weight', 'bias'):
                actual_gradient = getattr(actual_projection, parameter_name).grad
                reference_gradient = getattr(reference_projection, parameter_name).grad
                self.assertIsNotNone(actual_gradient)
                self.assertIsNotNone(reference_gradient)
                torch.testing.assert_close(actual_gradient, reference_gradient)
                torch.testing.assert_close(
                    actual_gradient,
                    torch.zeros_like(actual_gradient),
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

    def test_merged_relations_two_adam_updates_match_independent_reference(self):
        torch.manual_seed(487)
        attention = RelationAwareSelfAttention(ZoneSetEncoderConfig()).train()
        reference = deepcopy(attention)
        reference.forward = MethodType(
            _reference_attention_with_independent_relations,
            reference,
        )
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

    def test_pooling_uses_one_merged_key_value_linear(self):
        config = ZoneSetEncoderConfig()
        pooling = TaskConditionedPooling(config).eval()
        task = torch.randn((2, config.hidden_dim), dtype=torch.float32)
        tokens = torch.randn((2, 8, config.hidden_dim), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, False, False, False, False, False, False, False],
                [True, True, True, True, True, True, True, True],
            ]
        )

        original_linear = torch.nn.functional.linear
        with mock.patch(
            'brain_uav.models.zone_set_encoder.F.linear',
            wraps=original_linear,
        ) as linear:
            pooling(task, tokens, mask)

        self.assertEqual(linear.call_count, 2)

    def test_pooling_merged_key_value_matches_independent_reference(self):
        config = ZoneSetEncoderConfig()
        for token_count, mask in (
            (1, torch.tensor([[True]], dtype=torch.bool)),
            (
                8,
                torch.tensor(
                    [
                        [True, False, False, False, False, False, False, False],
                        [True, True, True, True, True, True, True, True],
                    ],
                    dtype=torch.bool,
                ),
            ),
        ):
            with self.subTest(token_count=token_count):
                torch.manual_seed(743)
                pooling = TaskConditionedPooling(config).train()
                reference = deepcopy(pooling)
                reference.forward = MethodType(_reference_pooling_forward, reference)
                batch_size = int(mask.shape[0])
                task = torch.randn(
                    (batch_size, config.hidden_dim),
                    dtype=torch.float32,
                    requires_grad=True,
                )
                tokens = torch.randn(
                    (batch_size, token_count, config.hidden_dim),
                    dtype=torch.float32,
                    requires_grad=True,
                )
                reference_task = task.detach().clone().requires_grad_(True)
                reference_tokens = tokens.detach().clone().requires_grad_(True)
                state = deepcopy(pooling.state_dict())
                parameter_ids = tuple(id(value) for value in pooling.parameters())

                actual = pooling(task, tokens, mask)
                expected = reference(reference_task, reference_tokens, mask)
                actual.square().sum().backward()
                expected.square().sum().backward()

                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(task.grad, reference_task.grad, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(tokens.grad, reference_tokens.grad, atol=2e-6, rtol=2e-6)
                for projection_name in ('key', 'value'):
                    actual_projection = getattr(pooling, projection_name)
                    expected_projection = getattr(reference, projection_name)
                    for parameter_name in ('weight', 'bias'):
                        torch.testing.assert_close(
                            getattr(actual_projection, parameter_name).grad,
                            getattr(expected_projection, parameter_name).grad,
                            atol=2e-6,
                            rtol=2e-6,
                        )
                self.assertEqual(
                    tuple(id(value) for value in pooling.parameters()),
                    parameter_ids,
                )
                restored = TaskConditionedPooling(config)
                restored.load_state_dict(state, strict=True)
                self.assertEqual(tuple(restored.state_dict()), tuple(state))

    def test_pooling_merged_key_value_tracks_two_adam_updates(self):
        torch.manual_seed(751)
        config = ZoneSetEncoderConfig()
        pooling = TaskConditionedPooling(config).train()
        reference = deepcopy(pooling)
        reference.forward = MethodType(_reference_pooling_forward, reference)
        optimizer = torch.optim.Adam(pooling.parameters(), lr=1e-2)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-2)
        task = torch.randn((2, config.hidden_dim), dtype=torch.float32)
        tokens = torch.randn((2, 8, config.hidden_dim), dtype=torch.float32)
        mask = torch.tensor(
            [
                [True, False, False, False, False, False, False, False],
                [True, True, True, True, True, True, True, True],
            ]
        )
        previous_output = None

        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            actual = pooling(task, tokens, mask)
            expected = reference(task, tokens, mask)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            if previous_output is not None:
                self.assertFalse(torch.equal(actual, previous_output))
            actual.square().mean().backward()
            expected.square().mean().backward()
            optimizer.step()
            reference_optimizer.step()
            previous_output = actual.detach().clone()
            for (actual_name, actual_parameter), (
                expected_name,
                expected_parameter,
            ) in zip(pooling.named_parameters(), reference.named_parameters()):
                self.assertEqual(actual_name, expected_name)
                torch.testing.assert_close(
                    actual_parameter,
                    expected_parameter,
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

    def test_eager_context_bypasses_compiled_shared_relations_and_restores_on_error(self):
        inputs = self.inputs([0, 3])
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            self.model.enable_compiled_shared_relations()
        compiled = mock.Mock(wraps=self.model._compiled_shared_relations)
        self.model._compiled_shared_relations = compiled

        self.model.build_shared_relations(*inputs)
        self.assertEqual(compiled.call_count, 1)
        with self.model.eager_tensor_forward():
            self.model.build_shared_relations(*inputs)
        self.assertEqual(compiled.call_count, 1)

        with self.assertRaisesRegex(RuntimeError, 'controlled eager failure'):
            with self.model.eager_tensor_forward():
                self.model.build_shared_relations(*inputs)
                raise RuntimeError('controlled eager failure')
        self.assertEqual(compiled.call_count, 1)
        self.model.build_shared_relations(*inputs)
        self.assertEqual(compiled.call_count, 2)

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

    def test_compiled_shared_relations_match_eager_and_preserve_gradients(self):
        eager = ZoneSetEncoder(self.scales)
        compiled = ZoneSetEncoder(self.scales)
        compiled.load_state_dict(eager.state_dict(), strict=True)
        parameter_ids = tuple(id(parameter) for parameter in compiled.parameters())
        state_keys = tuple(compiled.state_dict())
        compiled.enable_compiled_shared_relations(backend='eager')
        self.assertTrue(compiled.compiled_shared_relations_enabled)
        self.assertEqual(compiled.compiled_shared_relations_config['backend'], 'eager')
        self.assertEqual(tuple(id(parameter) for parameter in compiled.parameters()), parameter_ids)
        self.assertEqual(tuple(compiled.state_dict()), state_keys)

        for counts in ([0], [0, 3, 7]):
            with self.subTest(counts=counts):
                eager_inputs = tuple(value.clone() for value in self.inputs(counts))
                compiled_inputs = tuple(value.clone() for value in self.inputs(counts))
                for index in (0, 2):
                    eager_inputs[index].requires_grad_()
                    compiled_inputs[index].requires_grad_()
                expected = eager.build_shared_relations(*eager_inputs)
                actual = compiled.build_shared_relations(*compiled_inputs)
                for name in (
                    'clean_zone_features', 'pair_relations', 'valid_token_mask',
                    'token_pair_relations', 'relation_pair_mask',
                ):
                    torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
                self.assertIs(actual.ego_features, compiled_inputs[0])
                self.assertIs(actual.goal_features, compiled_inputs[1])
                self.assertIs(actual.zone_features, compiled_inputs[2])
                self.assertIs(actual.presence_mask, compiled_inputs[3])
                if max(counts) > 0:
                    expected_loss = expected.clean_zone_features.sum() + expected.pair_relations.sum()
                    actual_loss = actual.clean_zone_features.sum() + actual.pair_relations.sum()
                    expected_loss.backward()
                    actual_loss.backward()
                    torch.testing.assert_close(compiled_inputs[0].grad, eager_inputs[0].grad)
                    torch.testing.assert_close(compiled_inputs[2].grad, eager_inputs[2].grad)

    def test_compiled_shared_relations_do_not_access_contract_mappings_during_trace(self):
        inputs = self.inputs([0, 3, 7])
        torch._dynamo.reset()
        self.model.enable_compiled_shared_relations(
            backend='eager', fullgraph=True, dynamic=True
        )
        try:
            with mock.patch.object(
                v2_relations, 'ZONE_FEATURE_INDEX', _ForbiddenIndexLookup()
            ), mock.patch.object(
                v2_relations, 'EGO_FEATURE_INDEX', _ForbiddenIndexLookup()
            ):
                shared = self.model.build_shared_relations(*inputs)
        except Exception as exc:
            self.fail(f'compiled relation tracing accessed a contract mapping: {exc}')
        self.assertEqual(shared.pair_relations.shape, (3, 7, 7, 12))

    def test_compiled_shared_relations_do_not_reuse_a_previous_batch(self):
        with mock.patch(
            'brain_uav.models.zone_set_encoder.torch.compile',
            side_effect=lambda function, **kwargs: function,
        ):
            self.model.enable_compiled_shared_relations()
        first_inputs = self.inputs([2, 7])
        second_inputs = self.inputs([0, 3])
        second_inputs[0][1, EGO_FEATURE_INDEX['sin_psi']] = 0.91
        first = self.model.build_shared_relations(*first_inputs)
        second = self.model.build_shared_relations(*second_inputs)
        eager_second = self.model._compute_shared_relation_tensors(
            second_inputs[0], second_inputs[2], second_inputs[3]
        )
        self.assertIsNot(first.clean_zone_features, second.clean_zone_features)
        for actual, expected in zip((
            second.clean_zone_features, second.pair_relations,
            second.valid_token_mask, second.token_pair_relations,
            second.relation_pair_mask,
        ), eager_second):
            torch.testing.assert_close(actual, expected)

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
