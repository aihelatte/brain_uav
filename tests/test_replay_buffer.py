"""Tests for replay buffer counters and sampling cache."""

import unittest

import numpy as np

from brain_uav.trainers.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_counts_and_probability_cache_update(self):
        buffer = ReplayBuffer(capacity=3, success_sample_bias=2.0, near_goal_sample_bias=3.0)
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)

        buffer.add(obs + 0.0, action, 0.0, obs + 1.0, False, success=False, near_goal=False)
        buffer.add(obs + 1.0, action, 1.0, obs + 2.0, False, success=True, near_goal=False)
        buffer.add(obs + 2.0, action, 2.0, obs + 3.0, True, success=False, near_goal=True)

        self.assertAlmostEqual(buffer.success_fraction(), 1.0 / 3.0)
        self.assertAlmostEqual(buffer.near_goal_fraction(), 1.0 / 3.0)

        buffer.sample(2)
        first_cache = buffer._probabilities_cache
        first_token = buffer._probabilities_cache_token
        self.assertIsNotNone(first_cache)
        self.assertIsNotNone(first_token)

        buffer.sample(2)
        self.assertIs(first_cache, buffer._probabilities_cache)
        self.assertEqual(first_token, buffer._probabilities_cache_token)

        buffer.add(obs + 3.0, action, 3.0, obs + 4.0, True, success=True, near_goal=True)
        self.assertIsNone(buffer._probabilities_cache)
        self.assertIsNone(buffer._probabilities_cache_token)
        self.assertAlmostEqual(buffer.success_fraction(), 2.0 / 3.0)
        self.assertAlmostEqual(buffer.near_goal_fraction(), 2.0 / 3.0)

        batch = buffer.sample(2)
        self.assertEqual(batch['obs'].shape, (2, 4))
        self.assertEqual(batch['action'].shape, (2, 2))
        self.assertIn('near_goal', batch)


if __name__ == '__main__':
    unittest.main()
