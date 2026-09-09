"""Tests for vectorized pairwise V2 no-fly-zone relations."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import torch

from brain_uav.observations import (
    EGO_FEATURE_DIM,
    EGO_FEATURE_INDEX,
    PAIR_RELATION_FEATURE_DIM,
    PAIR_RELATION_FEATURE_INDEX,
    PAIR_RELATION_FEATURE_NAMES,
    ZONE_FEATURE_DIM,
    ZONE_FEATURE_INDEX,
    PairRelationBuilder,
    V2ObservationScales,
)


class TestPairRelationBuilder(unittest.TestCase):
    def setUp(self):
        self.scales = V2ObservationScales(100.0, 0.0, 50.0, math.pi / 4.0)
        self.builder = PairRelationBuilder(self.scales)

    def ego(self, *, sin_psi=0.0, cos_psi=1.0, batch_size=1):
        value = torch.zeros((batch_size, EGO_FEATURE_DIM), dtype=torch.float32)
        value[:, EGO_FEATURE_INDEX['sin_psi']] = sin_psi
        value[:, EGO_FEATURE_INDEX['cos_psi']] = cos_psi
        return value

    def zone(
        self,
        *,
        forward,
        right,
        up,
        extent_x,
        extent_y,
        extent_z,
        margin=0.0,
        blocks=False,
    ):
        value = torch.zeros(ZONE_FEATURE_DIM, dtype=torch.float32)
        value[ZONE_FEATURE_INDEX['zone_forward_norm']] = (
            forward / self.scales.horizontal_span
        )
        value[ZONE_FEATURE_INDEX['zone_right_norm']] = (
            right / self.scales.horizontal_span
        )
        value[ZONE_FEATURE_INDEX['zone_up_norm']] = up / self.scales.vertical_span
        value[ZONE_FEATURE_INDEX['extent_x_norm']] = (
            extent_x / self.scales.horizontal_span
        )
        value[ZONE_FEATURE_INDEX['extent_y_norm']] = (
            extent_y / self.scales.horizontal_span
        )
        value[ZONE_FEATURE_INDEX['extent_z_norm']] = (
            extent_z / self.scales.vertical_span
        )
        value[ZONE_FEATURE_INDEX['safety_margin_norm']] = (
            margin / self.scales.world_diagonal
        )
        value[ZONE_FEATURE_INDEX['raw_goal_path_intersects']] = float(blocks)
        return value

    def test_relation_names_order_dimensions_and_buffers_are_fixed(self):
        self.assertEqual(
            PAIR_RELATION_FEATURE_NAMES,
            (
                'delta_forward_norm',
                'delta_right_norm',
                'delta_up_norm',
                'center_distance_norm',
                'gap_forward_norm',
                'gap_right_norm',
                'gap_up_norm',
                'expanded_aabb_clearance_norm',
                'expanded_aabb_overlap',
                'straddles_uav_right_axis',
                'straddles_uav_up_axis',
                'both_block_raw_goal_path',
            ),
        )
        self.assertEqual(PAIR_RELATION_FEATURE_DIM, 12)
        self.assertEqual(
            dict(PAIR_RELATION_FEATURE_INDEX),
            {
                name: index
                for index, name in enumerate(PAIR_RELATION_FEATURE_NAMES)
            },
        )
        with self.assertRaises(TypeError):
            PAIR_RELATION_FEATURE_INDEX['delta_forward_norm'] = 99
        self.assertEqual(sum(parameter.numel() for parameter in self.builder.parameters()), 0)
        buffers = dict(self.builder.named_buffers())
        self.assertIn('horizontal_span', buffers)
        self.assertIn('vertical_span', buffers)
        self.assertIn('world_diagonal', buffers)
        self.assertIn('uav_radius', buffers)

    def test_two_zone_relation_matches_hand_calculation_at_yaw_zero(self):
        zones = torch.stack(
            [
                self.zone(
                    forward=-10.0,
                    right=-5.0,
                    up=-5.0,
                    extent_x=10.0,
                    extent_y=20.0,
                    extent_z=4.0,
                    margin=2.0,
                    blocks=True,
                ),
                self.zone(
                    forward=20.0,
                    right=15.0,
                    up=10.0,
                    extent_x=6.0,
                    extent_y=8.0,
                    extent_z=10.0,
                    margin=1.0,
                    blocks=True,
                ),
            ]
        ).unsqueeze(0)
        relations = self.builder(
            self.ego(),
            zones,
            torch.tensor([[True, True]]),
        )
        relation = relations[0, 0, 1]
        expected = {
            'delta_forward_norm': 30.0 / self.scales.horizontal_span,
            'delta_right_norm': 20.0 / self.scales.horizontal_span,
            'delta_up_norm': 15.0 / self.scales.vertical_span,
            'center_distance_norm': math.sqrt(30.0**2 + 20.0**2 + 15.0**2)
            / self.scales.world_diagonal,
            'gap_forward_norm': 19.0 / self.scales.horizontal_span,
            'gap_right_norm': 3.0 / self.scales.horizontal_span,
            'gap_up_norm': 5.0 / self.scales.vertical_span,
            'expanded_aabb_clearance_norm': math.sqrt(19.0**2 + 3.0**2 + 5.0**2)
            / self.scales.world_diagonal,
            'expanded_aabb_overlap': 0.0,
            'straddles_uav_right_axis': 1.0,
            'straddles_uav_up_axis': 1.0,
            'both_block_raw_goal_path': 1.0,
        }
        for name, value in expected.items():
            self.assertAlmostEqual(
                float(relation[PAIR_RELATION_FEATURE_INDEX[name]]),
                value,
                places=6,
                msg=name,
            )

    def test_yaw_half_pi_swaps_world_aabb_horizontal_projection(self):
        zones = torch.stack(
            [
                self.zone(
                    forward=-15.0,
                    right=-15.0,
                    up=0.0,
                    extent_x=10.0,
                    extent_y=20.0,
                    extent_z=2.0,
                ),
                self.zone(
                    forward=15.0,
                    right=15.0,
                    up=0.0,
                    extent_x=10.0,
                    extent_y=20.0,
                    extent_z=2.0,
                ),
            ]
        ).unsqueeze(0)
        relations = self.builder(
            self.ego(sin_psi=1.0, cos_psi=0.0),
            zones,
            torch.tensor([[True, True]]),
        )
        relation = relations[0, 0, 1]
        self.assertAlmostEqual(
            float(relation[PAIR_RELATION_FEATURE_INDEX['gap_forward_norm']]),
            10.0 / self.scales.horizontal_span,
            places=6,
        )
        self.assertAlmostEqual(
            float(relation[PAIR_RELATION_FEATURE_INDEX['gap_right_norm']]),
            20.0 / self.scales.horizontal_span,
            places=6,
        )

    def test_uav_radius_reduces_each_pair_gap_by_twice_the_radius(self):
        zones = torch.stack(
            [
                self.zone(
                    forward=-20.0,
                    right=-20.0,
                    up=-10.0,
                    extent_x=4.0,
                    extent_y=4.0,
                    extent_z=4.0,
                    margin=1.0,
                ),
                self.zone(
                    forward=20.0,
                    right=20.0,
                    up=10.0,
                    extent_x=4.0,
                    extent_y=4.0,
                    extent_z=4.0,
                    margin=2.0,
                ),
            ]
        ).unsqueeze(0)
        mask = torch.tensor([[True, True]])
        without_radius = self.builder(self.ego(), zones, mask)[0, 0, 1]
        with_radius = PairRelationBuilder(self.scales, uav_radius=3.0)(
            self.ego(), zones, mask
        )[0, 0, 1]
        for name, scale in (
            ('gap_forward_norm', self.scales.horizontal_span),
            ('gap_right_norm', self.scales.horizontal_span),
            ('gap_up_norm', self.scales.vertical_span),
        ):
            reduction = float(
                (without_radius[PAIR_RELATION_FEATURE_INDEX[name]]
                - with_radius[PAIR_RELATION_FEATURE_INDEX[name]])
                * scale
            )
            self.assertAlmostEqual(reduction, 6.0, places=5, msg=name)

    def test_delta_is_antisymmetric_and_other_pair_features_are_symmetric(self):
        zones = torch.stack(
            [
                self.zone(
                    forward=-10.0,
                    right=-5.0,
                    up=-2.0,
                    extent_x=6.0,
                    extent_y=8.0,
                    extent_z=4.0,
                    blocks=True,
                ),
                self.zone(
                    forward=20.0,
                    right=7.0,
                    up=3.0,
                    extent_x=10.0,
                    extent_y=4.0,
                    extent_z=6.0,
                    blocks=True,
                ),
            ]
        ).unsqueeze(0)
        relations = self.builder(self.ego(), zones, torch.tensor([[True, True]]))
        delta_indices = [
            PAIR_RELATION_FEATURE_INDEX[name]
            for name in (
                'delta_forward_norm',
                'delta_right_norm',
                'delta_up_norm',
            )
        ]
        symmetric_indices = [
            PAIR_RELATION_FEATURE_INDEX[name]
            for name in PAIR_RELATION_FEATURE_NAMES[3:]
        ]
        torch.testing.assert_close(
            relations[0, 0, 1, delta_indices],
            -relations[0, 1, 0, delta_indices],
        )
        torch.testing.assert_close(
            relations[0, 0, 1, symmetric_indices],
            relations[0, 1, 0, symmetric_indices],
        )

    def test_overlap_diagonal_padding_zero_and_small_zone_counts(self):
        overlapping = torch.stack(
            [
                self.zone(
                    forward=-1.0,
                    right=-1.0,
                    up=-1.0,
                    extent_x=10.0,
                    extent_y=10.0,
                    extent_z=10.0,
                ),
                self.zone(
                    forward=1.0,
                    right=1.0,
                    up=1.0,
                    extent_x=10.0,
                    extent_y=10.0,
                    extent_z=10.0,
                ),
                torch.full((ZONE_FEATURE_DIM,), 123.0),
            ]
        ).unsqueeze(0)
        relations = self.builder(
            self.ego(),
            overlapping,
            torch.tensor([[True, True, False]]),
        )
        self.assertEqual(relations.shape, (1, 3, 3, PAIR_RELATION_FEATURE_DIM))
        self.assertEqual(
            float(
                relations[
                    0, 0, 1, PAIR_RELATION_FEATURE_INDEX['expanded_aabb_overlap']
                ]
            ),
            1.0,
        )
        self.assertEqual(
            float(
                relations[
                    0,
                    0,
                    1,
                    PAIR_RELATION_FEATURE_INDEX['expanded_aabb_clearance_norm'],
                ]
            ),
            0.0,
        )
        torch.testing.assert_close(
            torch.diagonal(relations, dim1=1, dim2=2),
            torch.zeros((1, PAIR_RELATION_FEATURE_DIM, 3)),
        )
        torch.testing.assert_close(relations[:, 2], torch.zeros_like(relations[:, 2]))
        torch.testing.assert_close(relations[:, :, 2], torch.zeros_like(relations[:, :, 2]))
        self.assertTrue(torch.isfinite(relations).all())

        empty = self.builder(
            self.ego(batch_size=2),
            torch.zeros((2, 0, ZONE_FEATURE_DIM)),
            torch.zeros((2, 0), dtype=torch.bool),
        )
        self.assertEqual(empty.shape, (2, 0, 0, PAIR_RELATION_FEATURE_DIM))
        one = self.builder(
            self.ego(),
            overlapping[:, :1],
            torch.tensor([[True]]),
        )
        self.assertEqual(one.shape, (1, 1, 1, PAIR_RELATION_FEATURE_DIM))
        self.assertEqual(int(torch.count_nonzero(one)), 0)

    def test_fast_forward_does_not_convert_tensor_contents_to_python_scalars(self):
        ego = self.ego()
        zones = torch.stack(
            [
                self.zone(
                    forward=-5.0,
                    right=0.0,
                    up=0.0,
                    extent_x=2.0,
                    extent_y=2.0,
                    extent_z=2.0,
                ),
                self.zone(
                    forward=5.0,
                    right=0.0,
                    up=0.0,
                    extent_x=2.0,
                    extent_y=2.0,
                    extent_z=2.0,
                ),
            ]
        ).unsqueeze(0)
        mask = torch.tensor([[True, True]])

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
            relations = self.builder(ego, zones, mask)

        self.assertEqual(relations.shape, (1, 2, 2, PAIR_RELATION_FEATURE_DIM))

    def test_checked_forward_rejects_nonfinite_inputs_and_output(self):
        ego = self.ego()
        zones = torch.zeros((1, 2, ZONE_FEATURE_DIM), dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.bool)
        invalid_inputs = (
            (ego.clone().fill_(float('nan')), zones, mask, 'ego_features'),
            (ego.clone().fill_(float('inf')), zones, mask, 'ego_features'),
            (ego, zones.clone().fill_(float('nan')), mask, 'zone_features'),
            (ego, zones.clone().fill_(float('-inf')), mask, 'zone_features'),
        )
        for invalid_ego, invalid_zones, invalid_mask, name in invalid_inputs:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.builder.forward_checked(
                    invalid_ego,
                    invalid_zones,
                    invalid_mask,
                )

        invalid_output = torch.full(
            (1, 2, 2, PAIR_RELATION_FEATURE_DIM),
            float('nan'),
        )
        with (
            mock.patch.object(self.builder, 'forward', return_value=invalid_output),
            self.assertRaisesRegex(ValueError, 'output'),
        ):
            self.builder.forward_checked(ego, zones, mask)

    def test_checked_forward_matches_fast_forward_for_legal_inputs(self):
        ego = self.ego()
        zones = torch.stack(
            [
                self.zone(
                    forward=-5.0,
                    right=-1.0,
                    up=0.0,
                    extent_x=2.0,
                    extent_y=4.0,
                    extent_z=3.0,
                ),
                self.zone(
                    forward=7.0,
                    right=2.0,
                    up=1.0,
                    extent_x=3.0,
                    extent_y=2.0,
                    extent_z=5.0,
                ),
            ]
        ).unsqueeze(0)
        mask = torch.tensor([[True, True]])

        fast = self.builder(ego, zones, mask)
        checked = self.builder.forward_checked(ego, zones, mask)

        torch.testing.assert_close(checked, fast)

    def test_validation_and_forward_do_not_change_random_state(self):
        with self.assertRaises(TypeError):
            PairRelationBuilder(object())
        for radius in (-1.0, float('nan'), float('inf')):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                PairRelationBuilder(self.scales, uav_radius=radius)

        ego = self.ego()
        zones = torch.zeros((1, 2, ZONE_FEATURE_DIM), dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.bool)
        invalid = (
            (ego.double(), zones, mask),
            (ego, zones.double(), mask),
            (ego, zones, mask.to(torch.uint8)),
            (ego[:, :-1], zones, mask),
            (ego, zones[:, :, :-1], mask),
            (ego, zones, mask[:, :-1]),
            (
                torch.zeros((0, EGO_FEATURE_DIM)),
                torch.zeros((0, 2, ZONE_FEATURE_DIM)),
                torch.zeros((0, 2), dtype=torch.bool),
            ),
        )
        for values in invalid:
            with self.subTest(
                shapes=tuple(tuple(value.shape) for value in values),
                dtypes=tuple(value.dtype for value in values),
            ):
                with self.assertRaises((TypeError, ValueError)):
                    self.builder(*values)

        random_state = torch.random.get_rng_state().clone()
        self.builder(ego, zones, mask)
        torch.testing.assert_close(torch.random.get_rng_state(), random_state)


if __name__ == '__main__':
    unittest.main()
