"""TD3 integration tests for the V2 SpikingJelly actor with ANN critics."""

from __future__ import annotations

from copy import deepcopy
import math
import unittest

import numpy as np
import torch

from brain_uav.models import V2ANNCritic, V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
    V2ObservationScales,
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

    def make_engine(self, *, freeze: int = 0, bc=None) -> V2TD3UpdateEngine:
        actor = self.make_actor()
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
