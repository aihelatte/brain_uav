"""Tests for replay buffer counters and sampling cache."""

import unittest
from unittest import mock

import numpy as np

from brain_uav.trainers.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_add_returns_slot_ref_and_mark_success_slots_updates_weights(self):
        buffer = ReplayBuffer(capacity=4, success_sample_bias=3.0, near_goal_sample_bias=2.0)
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)

        slot_ref = buffer.add(obs, action, 0.0, obs + 1.0, False, success=False, near_goal=True)

        self.assertEqual(slot_ref, (0, 0))
        self.assertEqual(buffer.success_count, 0)
        self.assertAlmostEqual(buffer.sample_weight[0], 2.0)
        self.assertAlmostEqual(buffer.total_sample_weight, 2.0)

        updated = buffer.mark_success_slots([slot_ref], success=True)

        self.assertEqual(updated, 1)
        self.assertTrue(buffer.success[0])
        self.assertEqual(buffer.success_count, 1)
        self.assertAlmostEqual(buffer.sample_weight[0], 6.0)
        self.assertAlmostEqual(buffer.total_sample_weight, 6.0)

    def test_mark_success_slots_ignores_overwritten_slot_refs(self):
        buffer = ReplayBuffer(capacity=1, success_sample_bias=3.0)
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)

        stale_ref = buffer.add(obs, action, 0.0, obs + 1.0, False)
        live_ref = buffer.add(obs + 2.0, action, 1.0, obs + 3.0, False)

        self.assertEqual(stale_ref, (0, 0))
        self.assertEqual(live_ref, (0, 1))
        self.assertEqual(buffer.mark_success_slots([stale_ref], success=True), 0)
        self.assertEqual(buffer.success_count, 0)
        self.assertFalse(buffer.success[0])

        self.assertEqual(buffer.mark_success_slots([live_ref], success=True), 1)
        self.assertEqual(buffer.success_count, 1)
        self.assertTrue(buffer.success[0])

    def test_counts_and_incremental_sampling_weights_update(self):
        buffer = ReplayBuffer(capacity=3, success_sample_bias=2.0, near_goal_sample_bias=3.0)
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)

        buffer.add(obs + 0.0, action, 0.0, obs + 1.0, False, success=False, near_goal=False)
        buffer.add(obs + 1.0, action, 1.0, obs + 2.0, False, success=True, near_goal=False)
        buffer.add(obs + 2.0, action, 2.0, obs + 3.0, True, success=False, near_goal=True)

        self.assertAlmostEqual(buffer.success_fraction(), 1.0 / 3.0)
        self.assertAlmostEqual(buffer.near_goal_fraction(), 1.0 / 3.0)
        np.testing.assert_allclose(buffer.sample_weight[:3], np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(buffer.total_sample_weight, 6.0)

        with mock.patch('numpy.random.choice', return_value=np.array([0, 2], dtype=np.int64)) as choice_mock:
            batch = buffer.sample(2)
        probs = choice_mock.call_args.kwargs['p']
        np.testing.assert_allclose(probs, np.array([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]))
        self.assertEqual(batch['obs'].shape, (2, 4))

        buffer.add(obs + 3.0, action, 3.0, obs + 4.0, True, success=True, near_goal=True)
        self.assertAlmostEqual(buffer.success_fraction(), 2.0 / 3.0)
        self.assertAlmostEqual(buffer.near_goal_fraction(), 2.0 / 3.0)
        np.testing.assert_allclose(buffer.sample_weight[:3], np.array([6.0, 2.0, 3.0]))
        self.assertAlmostEqual(buffer.total_sample_weight, 11.0)

        with mock.patch('numpy.random.choice', return_value=np.array([1, 2], dtype=np.int64)) as choice_mock:
            batch = buffer.sample(2)
        probs = choice_mock.call_args.kwargs['p']
        np.testing.assert_allclose(probs, np.array([6.0 / 11.0, 2.0 / 11.0, 3.0 / 11.0]))
        self.assertEqual(batch['action'].shape, (2, 2))
        self.assertIn('near_goal', batch)

    def test_sample_falls_back_to_primary_when_success_replay_empty(self):
        buffer = ReplayBuffer(capacity=8, success_replay_fraction=0.25, success_batch_fraction=0.25)
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        for idx in range(4):
            buffer.add(obs + idx, action, float(idx), obs + idx + 1, False)

        with mock.patch('numpy.random.choice', return_value=np.array([0, 1, 2, 3], dtype=np.int64)) as choice_mock:
            batch = buffer.sample(4)

        self.assertEqual(choice_mock.call_count, 1)
        self.assertEqual(batch['obs'].shape[0], 4)
        self.assertTrue((batch['success'] == 0).all())

    def test_sample_mixes_success_batch_fraction(self):
        buffer = ReplayBuffer(capacity=16, success_replay_fraction=0.25, success_batch_fraction=0.25)
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        for idx in range(8):
            buffer.add(obs + idx, action, float(idx), obs + idx + 1, False, success=(idx == 7))
        for idx in range(3):
            buffer.add_success_transition(obs + 10 + idx, action, 1.0, obs + 11 + idx, True, near_goal=True)

        with mock.patch(
            'numpy.random.choice',
            side_effect=[
                np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
                np.array([0, 1], dtype=np.int64),
            ],
        ) as choice_mock:
            batch = buffer.sample(8)

        self.assertEqual(choice_mock.call_count, 2)
        self.assertEqual(batch['obs'].shape[0], 8)
        self.assertEqual(int(batch['success'].sum().item()), 2)

    def test_success_replay_shortage_is_backfilled_from_primary(self):
        buffer = ReplayBuffer(capacity=16, success_replay_fraction=0.25, success_batch_fraction=0.25)
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        for idx in range(8):
            buffer.add(obs + idx, action, float(idx), obs + idx + 1, False, success=(idx == 7))
        buffer.add_success_transition(obs + 99, action, 1.0, obs + 100, True, near_goal=True)

        with mock.patch(
            'numpy.random.choice',
            side_effect=[
                np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
                np.array([0], dtype=np.int64),
            ],
        ) as choice_mock:
            batch = buffer.sample(8)

        self.assertEqual(choice_mock.call_count, 2)
        self.assertEqual(batch['obs'].shape[0], 8)
        self.assertEqual(int(batch['success'].sum().item()), 1)


if __name__ == '__main__':
    unittest.main()
