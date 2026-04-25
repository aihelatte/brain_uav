"""Tests for replay buffer counters and sampling cache."""

import unittest
from unittest import mock

import numpy as np

from brain_uav.trainers.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
