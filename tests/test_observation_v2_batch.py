"""Tests for dynamic collation of structured V2 observations."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
    V2ObservationBatch,
    collate_v2_observations,
)


def _observation(zone_count: int, *, offset: float = 0.0) -> V2Observation:
    ego = np.arange(EGO_FEATURE_DIM, dtype=np.float32) + offset
    goal = np.arange(GOAL_FEATURE_DIM, dtype=np.float32) + 10.0 + offset
    zones = np.zeros((zone_count, ZONE_FEATURE_DIM), dtype=np.float32)
    for index in range(zone_count):
        zones[index] = offset + 100.0 + index
    return V2Observation(
        ego_features=ego,
        goal_features=goal,
        zone_features=zones,
        presence_mask=np.ones(zone_count, dtype=np.bool_),
    )


class TestV2ObservationBatch(unittest.TestCase):
    def test_single_and_mixed_collation_uses_current_batch_maximum(self):
        first = _observation(2, offset=1.0)
        second = _observation(5, offset=2.0)

        batch = collate_v2_observations([first, second])

        self.assertIsInstance(batch, V2ObservationBatch)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.max_zone_count, 5)
        self.assertEqual(batch.ego_features.shape, (2, EGO_FEATURE_DIM))
        self.assertEqual(batch.goal_features.shape, (2, GOAL_FEATURE_DIM))
        self.assertEqual(batch.zone_features.shape, (2, 5, ZONE_FEATURE_DIM))
        self.assertEqual(batch.presence_mask.shape, (2, 5))
        torch.testing.assert_close(
            batch.zone_features[0, :2],
            torch.from_numpy(first.zone_features.copy()),
        )
        torch.testing.assert_close(
            batch.zone_features[1],
            torch.from_numpy(second.zone_features.copy()),
        )
        torch.testing.assert_close(
            batch.zone_features[0, 2:],
            torch.zeros((3, ZONE_FEATURE_DIM), dtype=torch.float32),
        )
        torch.testing.assert_close(
            batch.presence_mask,
            torch.tensor(
                [
                    [True, True, False, False, False],
                    [True, True, True, True, True],
                ]
            ),
        )

    def test_counts_zero_one_five_six_and_ten_are_dynamic(self):
        for count in (0, 1, 5, 6, 10):
            observation = _observation(count, offset=float(count))
            with self.subTest(count=count):
                batch = collate_v2_observations([observation])
                self.assertEqual(batch.zone_features.shape, (1, count, ZONE_FEATURE_DIM))
                self.assertEqual(batch.presence_mask.shape, (1, count))
                self.assertEqual(batch.max_zone_count, count)
                torch.testing.assert_close(
                    batch.zone_features[0],
                    torch.from_numpy(observation.zone_features.copy()),
                )
                self.assertTrue(batch.presence_mask.all())

    def test_all_empty_batch_keeps_zero_length_zone_axis(self):
        batch = collate_v2_observations(
            [_observation(0, offset=1.0), _observation(0, offset=2.0)]
        )

        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.max_zone_count, 0)
        self.assertEqual(batch.zone_features.shape, (2, 0, ZONE_FEATURE_DIM))
        self.assertEqual(batch.presence_mask.shape, (2, 0))
        self.assertTrue(torch.isfinite(batch.zone_features).all())

    def test_collation_preserves_each_scene_zone_order(self):
        observation = _observation(6)
        reversed_zones = observation.zone_features[::-1].copy()
        reversed_observation = V2Observation(
            observation.ego_features,
            observation.goal_features,
            reversed_zones,
            np.ones(6, dtype=np.bool_),
        )

        batch = collate_v2_observations([observation, reversed_observation])

        torch.testing.assert_close(
            batch.zone_features[0],
            torch.from_numpy(observation.zone_features.copy()),
        )
        torch.testing.assert_close(
            batch.zone_features[1],
            torch.from_numpy(reversed_zones),
        )

    def test_dtypes_device_finite_values_and_to_returns_new_batch(self):
        batch = collate_v2_observations([_observation(1)], device='cpu')

        self.assertEqual(batch.ego_features.dtype, torch.float32)
        self.assertEqual(batch.goal_features.dtype, torch.float32)
        self.assertEqual(batch.zone_features.dtype, torch.float32)
        self.assertEqual(batch.presence_mask.dtype, torch.bool)
        self.assertEqual(batch.ego_features.device.type, 'cpu')
        for tensor in (
            batch.ego_features,
            batch.goal_features,
            batch.zone_features,
        ):
            self.assertTrue(torch.isfinite(tensor).all())

        moved = batch.to(torch.device('cpu'))
        self.assertIsNot(moved, batch)
        self.assertEqual(moved.ego_features.device.type, 'cpu')
        torch.testing.assert_close(moved.ego_features, batch.ego_features)
        torch.testing.assert_close(moved.zone_features, batch.zone_features)
        torch.testing.assert_close(moved.presence_mask, batch.presence_mask)

    def test_collator_builds_complete_cpu_batch_before_one_batch_transfer(self):
        observations = [
            _observation(0, offset=1.0),
            _observation(2, offset=2.0),
            _observation(10, offset=3.0),
        ]
        original_to = V2ObservationBatch.to
        observed_devices = []

        def checked_transfer(batch, device):
            observed_devices.append(
                (
                    batch.ego_features.device.type,
                    batch.goal_features.device.type,
                    batch.zone_features.device.type,
                    batch.presence_mask.device.type,
                )
            )
            return original_to(batch, device)

        with mock.patch.object(
            V2ObservationBatch,
            'to',
            autospec=True,
            side_effect=checked_transfer,
        ) as transfer:
            batch = collate_v2_observations(observations, device='cpu')

        self.assertEqual(transfer.call_count, 1)
        self.assertEqual(observed_devices, [('cpu', 'cpu', 'cpu', 'cpu')])
        self.assertEqual(batch.zone_features.shape, (3, 10, ZONE_FEATURE_DIM))
        self.assertEqual(
            batch.presence_mask.sum(dim=1).tolist(),
            [0, 2, 10],
        )

    def test_to_does_not_convert_tensor_contents_to_python_scalars(self):
        batch = collate_v2_observations([_observation(2)])

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
            moved = batch.to('cpu')

        self.assertIsNot(moved, batch)
        self.assertEqual(moved.zone_features.shape, batch.zone_features.shape)

    def test_validate_finite_explicitly_rejects_nan_and_infinity(self):
        ego = torch.zeros((1, EGO_FEATURE_DIM), dtype=torch.float32)
        goal = torch.zeros((1, GOAL_FEATURE_DIM), dtype=torch.float32)
        zones = torch.zeros((1, 1, ZONE_FEATURE_DIM), dtype=torch.float32)
        mask = torch.ones((1, 1), dtype=torch.bool)
        cases = (
            ('ego_features', float('nan')),
            ('goal_features', float('inf')),
            ('zone_features', float('-inf')),
        )

        for name, invalid_value in cases:
            tensors = {
                'ego_features': ego.clone(),
                'goal_features': goal.clone(),
                'zone_features': zones.clone(),
            }
            tensors[name].reshape(-1)[0] = invalid_value
            with self.subTest(name=name):
                batch = V2ObservationBatch(
                    tensors['ego_features'],
                    tensors['goal_features'],
                    tensors['zone_features'],
                    mask,
                )
                with self.assertRaisesRegex(ValueError, name):
                    batch.validate_finite()

    def test_collator_rejects_empty_nonsequence_and_nonobservations(self):
        with self.assertRaises(ValueError):
            collate_v2_observations([])
        for value in (
            None,
            'observation',
            (_observation(1) for _ in range(1)),
        ):
            with self.subTest(value_type=type(value).__name__), self.assertRaises(TypeError):
                collate_v2_observations(value)
        with self.assertRaises(TypeError):
            collate_v2_observations([_observation(1), object()])

    def test_batch_contract_rejects_invalid_shapes_dtypes_and_devices(self):
        ego = torch.zeros((2, EGO_FEATURE_DIM), dtype=torch.float32)
        goal = torch.zeros((2, GOAL_FEATURE_DIM), dtype=torch.float32)
        zones = torch.zeros((2, 3, ZONE_FEATURE_DIM), dtype=torch.float32)
        mask = torch.ones((2, 3), dtype=torch.bool)
        invalid = (
            (torch.zeros((0, EGO_FEATURE_DIM)), goal[:0], zones[:0], mask[:0]),
            (torch.zeros((2, EGO_FEATURE_DIM + 1)), goal, zones, mask),
            (ego, torch.zeros((2, GOAL_FEATURE_DIM + 1)), zones, mask),
            (ego, goal, torch.zeros((2, 3, ZONE_FEATURE_DIM + 1)), mask),
            (ego, goal, zones, torch.ones((2, 2), dtype=torch.bool)),
            (ego.double(), goal, zones, mask),
            (ego, goal.double(), zones, mask),
            (ego, goal, zones.double(), mask),
            (ego, goal, zones, mask.to(torch.uint8)),
        )
        for tensors in invalid:
            with self.subTest(
                shapes=tuple(tuple(tensor.shape) for tensor in tensors),
                dtypes=tuple(tensor.dtype for tensor in tensors),
            ):
                with self.assertRaises((TypeError, ValueError)):
                    V2ObservationBatch(*tensors)

        meta_goal = torch.empty((2, GOAL_FEATURE_DIM), device='meta')
        with self.assertRaises(ValueError):
            V2ObservationBatch(ego, meta_goal, zones, mask)


if __name__ == '__main__':
    unittest.main()
