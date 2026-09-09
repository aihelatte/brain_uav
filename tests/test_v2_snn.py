"""Contract tests for the structured-observation V2 SNN actor."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.models import V2SNNPolicyActor
from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
    V2ObservationBatch,
    V2ObservationScales,
    collate_v2_observations,
)


def _observation(zone_count: int, *, offset: float = 0.0) -> V2Observation:
    ego = np.zeros(EGO_FEATURE_DIM, dtype=np.float32)
    ego[0] = offset * 0.01
    ego[2] = 0.4
    ego[4] = math.sin(0.3)
    ego[5] = math.cos(0.3)
    goal = np.array([0.2, -0.1, 0.05, 0.3], dtype=np.float32)
    zones = np.zeros((zone_count, ZONE_FEATURE_DIM), dtype=np.float32)
    for index in range(zone_count):
        zones[index, 0] = 1.0
        zones[index, 5] = -0.2 + 0.04 * index
        zones[index, 6] = 0.03 * ((-1.0) ** index)
        zones[index, 7] = 0.02 * index
        zones[index, 8:11] = (0.05, 0.04, 0.08)
        zones[index, 11] = 0.01
        zones[index, 12] = 0.2
        zones[index, 13] = -1.0
        zones[index, 16] = 0.5
        zones[index, 17] = float(index % 2 == 0)
        zones[index, 18] = 0.2
    return V2Observation(ego, goal, zones, np.ones(zone_count, dtype=np.bool_))


class TestV2SNNPolicyActor(unittest.TestCase):
    def setUp(self) -> None:
        self.scales = V2ObservationScales(100.0, 0.0, 50.0, math.pi / 4.0)
        torch.manual_seed(20260908)
        self.actor = V2SNNPolicyActor(
            self.scales,
            action_dim=2,
            hidden_dim=16,
            action_limit=torch.tensor([0.2, 0.3], dtype=torch.float32),
            time_window=4,
            tau=2.0,
        )

    def test_dynamic_counts_mixed_batch_shapes_and_action_bounds(self) -> None:
        for count in (0, 1, 5, 6, 10):
            with self.subTest(count=count):
                output = self.actor(collate_v2_observations([_observation(count)]))
                self.assertEqual(output.shape, (1, 2))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.all(output.abs() <= self.actor.action_limit))

        mixed = collate_v2_observations(
            [_observation(count, offset=float(count)) for count in (0, 1, 6, 10)]
        )
        self.assertEqual(self.actor(mixed).shape, (4, 2))

    def test_padding_garbage_and_masked_positions_do_not_change_output(self) -> None:
        batch = collate_v2_observations([_observation(1), _observation(6)])
        garbage = batch.zone_features.clone()
        garbage[~batch.presence_mask] = 9182.0
        garbage_batch = V2ObservationBatch(
            batch.ego_features,
            batch.goal_features,
            garbage,
            batch.presence_mask,
        )
        torch.testing.assert_close(
            self.actor(batch), self.actor(garbage_batch), atol=1e-6, rtol=1e-6
        )

    def test_context_encoder_runs_once_and_lif_state_never_leaks(self) -> None:
        batch_a = collate_v2_observations([_observation(2)])
        batch_b = collate_v2_observations([_observation(5, offset=3.0)])
        with mock.patch.object(
            self.actor.zone_set_encoder,
            'forward',
            wraps=self.actor.zone_set_encoder.forward,
        ) as encoder_forward:
            first = self.actor(batch_a)
            self.assertEqual(encoder_forward.call_count, 1)
        self.actor(batch_b)
        second = self.actor(batch_a)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(self.actor.snn_head.lif1.v, 0.0)
        self.assertEqual(self.actor.snn_head.lif2.v, 0.0)

    def test_backward_reaches_encoder_and_spiking_head_with_finite_gradients(self) -> None:
        batch = collate_v2_observations([_observation(2), _observation(4)])
        self.actor(batch).square().mean().backward()
        encoder_gradients = [
            parameter.grad
            for parameter in self.actor.zone_set_encoder.parameters()
            if parameter.requires_grad
        ]
        head_gradients = [
            parameter.grad
            for parameter in self.actor.snn_head.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(value is not None for value in encoder_gradients))
        self.assertTrue(any(value is not None for value in head_gradients))
        for gradient in encoder_gradients + head_gradients:
            if gradient is not None:
                self.assertTrue(torch.isfinite(gradient).all())

    def test_constructor_and_forward_validate_contract(self) -> None:
        limit = torch.tensor([0.2, 0.3], dtype=torch.float32)
        for time_window in (0, -1, 1.5, True):
            with self.subTest(time_window=time_window):
                with self.assertRaises(ValueError):
                    V2SNNPolicyActor(
                        self.scales, 2, 16, limit, time_window=time_window
                    )
        for tau in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(tau=tau):
                with self.assertRaises(ValueError):
                    V2SNNPolicyActor(self.scales, 2, 16, limit, tau=tau)
        for invalid_limit in (
            torch.tensor([0.2]),
            torch.tensor([0.2, -0.3]),
            torch.tensor([0.2, float('nan')]),
        ):
            with self.assertRaises((TypeError, ValueError)):
                V2SNNPolicyActor(self.scales, 2, 16, invalid_limit)
        with self.assertRaises(TypeError):
            self.actor(torch.zeros((1, 10), dtype=torch.float32))

    def test_missing_spikingjelly_is_an_explicit_error_without_fallback(self) -> None:
        import brain_uav.models.v2_snn as v2_snn

        with mock.patch.object(v2_snn, '_SPIKINGJELLY_AVAILABLE', False), mock.patch.object(
            v2_snn, '_SPIKINGJELLY_IMPORT_ERROR', ImportError('missing')
        ):
            with self.assertRaisesRegex(RuntimeError, 'SpikingJelly'):
                v2_snn.V2SNNPolicyActor(
                    self.scales,
                    2,
                    16,
                    torch.tensor([0.2, 0.3], dtype=torch.float32),
                )


if __name__ == '__main__':
    unittest.main()
