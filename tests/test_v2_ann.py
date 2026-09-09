"""Tests for ANN actor and critic heads over structured V2 observations."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.models.scaling import FixedObsScaler
from brain_uav.models import V2ANNCritic, V2ANNPolicyActor
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
    ego[0] = 0.05 * offset
    ego[2] = 0.4
    ego[4] = math.sin(0.2 + offset * 0.01)
    ego[5] = math.cos(0.2 + offset * 0.01)
    goal = np.array([0.2 + offset * 0.01, -0.1, 0.05, 0.3], dtype=np.float32)
    zones = np.zeros((zone_count, ZONE_FEATURE_DIM), dtype=np.float32)
    for index in range(zone_count):
        zones[index, 0] = 1.0
        zones[index, 5] = -0.3 + 0.07 * index
        zones[index, 6] = 0.02 * ((-1.0) ** index) * (index + 1)
        zones[index, 7] = -0.1 + 0.03 * index
        zones[index, 8:11] = np.array([0.05, 0.04, 0.08], dtype=np.float32)
        zones[index, 11] = 0.01
        zones[index, 12] = 0.2
        zones[index, 13] = -1.0
        zones[index, 16] = 0.5
        zones[index, 17] = float(index % 2 == 0)
        zones[index, 18] = 0.2 + 0.01 * index
    return V2Observation(
        ego_features=ego,
        goal_features=goal,
        zone_features=zones,
        presence_mask=np.ones(zone_count, dtype=np.bool_),
    )


class TestV2ANN(unittest.TestCase):
    def setUp(self):
        self.scales = V2ObservationScales(100.0, 0.0, 50.0, math.pi / 4.0)
        torch.manual_seed(20260903)
        self.actor = V2ANNPolicyActor(
            self.scales,
            action_dim=2,
            hidden_dim=32,
            action_limit=torch.tensor([0.2, 0.3], dtype=torch.float32),
        ).eval()
        self.critic = V2ANNCritic(
            self.scales,
            action_dim=2,
            hidden_dim=32,
        ).eval()

    def test_actor_supports_zero_one_five_six_and_ten_without_hard_limit(self):
        for count in (0, 1, 5, 6, 10):
            with self.subTest(count=count):
                batch = collate_v2_observations([_observation(count)])
                action = self.actor(batch)
                self.assertEqual(action.shape, (1, 2))
                self.assertTrue(torch.isfinite(action).all())
                self.assertTrue(torch.all(action.abs() <= self.actor.action_limit))

        self.assertFalse(hasattr(self.actor, 'max_zones'))
        self.assertFalse(
            any(
                'slot' in name.lower() or 'position' in name.lower()
                for name, _ in self.actor.named_parameters()
            )
        )

    def test_critic_accepts_mixed_dynamic_batch_and_returns_one_q_per_row(self):
        observations = [_observation(count, offset=float(index)) for index, count in enumerate((0, 1, 5, 6, 10))]
        batch = collate_v2_observations(observations)
        action = torch.zeros((5, 2), dtype=torch.float32)

        q_values = self.critic(batch, action)

        self.assertEqual(q_values.shape, (5, 1))
        self.assertTrue(torch.isfinite(q_values).all())

    def test_zone_permutation_does_not_change_actor_or_critic_outputs(self):
        observation = _observation(6)
        permutation = np.array([3, 0, 5, 1, 4, 2])
        permuted = V2Observation(
            observation.ego_features,
            observation.goal_features,
            observation.zone_features[permutation],
            np.ones(6, dtype=np.bool_),
        )
        original_batch = collate_v2_observations([observation])
        permuted_batch = collate_v2_observations([permuted])
        action = torch.tensor([[0.05, -0.03]], dtype=torch.float32)

        torch.testing.assert_close(
            self.actor(permuted_batch),
            self.actor(original_batch),
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            self.critic(permuted_batch, action),
            self.critic(original_batch, action),
            atol=2e-6,
            rtol=2e-6,
        )

    def test_padding_garbage_does_not_change_actor_or_critic_outputs(self):
        batch = collate_v2_observations([_observation(2), _observation(6)])
        garbage_zones = batch.zone_features.clone()
        garbage_zones[~batch.presence_mask] = 9876.0
        garbage_batch = V2ObservationBatch(
            batch.ego_features,
            batch.goal_features,
            garbage_zones,
            batch.presence_mask,
        )
        action = torch.zeros((2, 2), dtype=torch.float32)

        torch.testing.assert_close(
            self.actor(garbage_batch),
            self.actor(batch),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            self.critic(garbage_batch, action),
            self.critic(batch, action),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_actor_and_two_critics_have_disjoint_parameters_and_encoders(self):
        critic2 = V2ANNCritic(self.scales, action_dim=2, hidden_dim=32)
        modules = (self.actor, self.critic, critic2)
        parameter_ids = [
            {id(parameter) for parameter in module.parameters()}
            for module in modules
        ]
        self.assertTrue(parameter_ids[0].isdisjoint(parameter_ids[1]))
        self.assertTrue(parameter_ids[0].isdisjoint(parameter_ids[2]))
        self.assertTrue(parameter_ids[1].isdisjoint(parameter_ids[2]))
        self.assertIsNot(self.actor.zone_set_encoder, self.critic.zone_set_encoder)
        self.assertIsNot(self.critic.zone_set_encoder, critic2.zone_set_encoder)

    def test_gradients_reach_each_model_encoder_and_head(self):
        actor = V2ANNPolicyActor(
            self.scales,
            2,
            32,
            torch.tensor([0.2, 0.3], dtype=torch.float32),
        )
        critic = V2ANNCritic(self.scales, 2, 32)
        batch = collate_v2_observations([_observation(3), _observation(4)])

        actor(batch).square().mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in actor.zone_set_encoder.parameters())
        )
        self.assertTrue(any(parameter.grad is not None for parameter in actor.head.parameters()))

        critic(batch, torch.zeros((2, 2), dtype=torch.float32)).mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in critic.zone_set_encoder.parameters())
        )
        self.assertTrue(any(parameter.grad is not None for parameter in critic.head.parameters()))

    def test_fast_forwards_do_not_convert_tensor_contents_to_python_scalars(self):
        batch = collate_v2_observations([_observation(0), _observation(6)])
        action = torch.zeros((2, 2), dtype=torch.float32)

        with (
            mock.patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('__bool__')),
            mock.patch.object(torch.Tensor, 'item', side_effect=AssertionError('item')),
            mock.patch.object(torch.Tensor, '__float__', side_effect=AssertionError('__float__')),
            mock.patch.object(torch.Tensor, 'tolist', side_effect=AssertionError('tolist')),
        ):
            actor_output = self.actor(batch)
            critic_output = self.critic(batch, action)

        self.assertEqual(actor_output.shape, (2, 2))
        self.assertEqual(critic_output.shape, (2, 1))

    def test_external_seed_reproduces_initialization(self):
        def build_actor():
            return V2ANNPolicyActor(
                self.scales,
                2,
                32,
                torch.tensor([0.2, 0.3], dtype=torch.float32),
            )

        torch.manual_seed(12345)
        first = build_actor().state_dict()
        torch.manual_seed(12345)
        second = build_actor().state_dict()
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            torch.testing.assert_close(first[name], second[name])

    def test_head_initialization_and_model_structure_follow_ann_contract(self):
        final_linear = self.actor.head[4]
        self.assertLessEqual(float(final_linear.weight.abs().max()), 1.0e-3)
        self.assertLessEqual(float(final_linear.bias.abs().max()), 1.0e-3)
        self.assertIn('action_limit', dict(self.actor.named_buffers()))
        self.assertFalse(
            any(isinstance(module, FixedObsScaler) for module in self.actor.modules())
        )
        self.assertFalse(
            any(isinstance(module, FixedObsScaler) for module in self.critic.modules())
        )
        self.assertEqual(self.actor.head[0].in_features, 128)
        self.assertEqual(self.critic.head[0].in_features, 130)

    def test_forward_validation_rejects_wrong_contracts(self):
        batch = collate_v2_observations([_observation(1), _observation(2)])
        with self.assertRaises(TypeError):
            self.actor(torch.zeros((2, 24), dtype=torch.float32))
        for action in (
            torch.zeros((2, 3), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float64),
        ):
            with self.subTest(shape=tuple(action.shape), dtype=action.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    self.critic(batch, action)


if __name__ == '__main__':
    unittest.main()
