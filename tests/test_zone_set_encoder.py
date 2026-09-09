"""Tests for the standalone relation-aware dynamic zone-set encoder."""

from __future__ import annotations

import math
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

    def test_checked_forward_matches_fast_for_all_supported_counts(self):
        for count in (0, 1, 5, 6, 10):
            inputs = self.inputs([count])
            with self.subTest(count=count):
                fast = self.model(*inputs)
                checked = self.model.forward_checked(*inputs)
                torch.testing.assert_close(checked, fast)

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
