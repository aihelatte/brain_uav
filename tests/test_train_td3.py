"""Tests for TD3 script helpers."""

import unittest

from brain_uav.scripts.train_td3 import build_parser, make_early_stop_callback


class TestTrainTD3Helpers(unittest.TestCase):
    def test_parser_accepts_snn_backend(self):
        parser = build_parser()
        args = parser.parse_args(['--curriculum-level', 'easy', '--device', 'cuda', '--snn-backend', 'cupy'])
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.snn_backend, 'cupy')
        self.assertEqual(args.summary_every_episodes, 15)
        self.assertEqual(args.early_stop_windows, 4)
        self.assertEqual(args.early_stop_max_failures_per_window, 1)
        self.assertEqual(args.early_stop_goal_rate, 0.95)
        self.assertEqual(args.early_stop_min_steps, 12000)

    def test_early_stop_callback_uses_new_rule(self):
        callback = make_early_stop_callback(
            enabled=True,
            goal_rate_threshold=0.95,
            consecutive_windows=4,
            min_steps=12000,
            max_failures_per_window=1,
        )
        window = {
            'episode_count': 15,
            'goal_count': 14,
            'timeout_count': 1,
            'boundary_count': 0,
            'ground_count': 0,
            'collision_count': 0,
            'other_count': 0,
            'total_steps': 12000,
        }
        self.assertIsNone(callback({**window, 'total_steps': 3000}))
        self.assertIsNone(callback({**window, 'total_steps': 12000}))
        self.assertIsNone(callback({**window, 'total_steps': 13000}))
        reason = callback({**window, 'total_steps': 14000})
        self.assertIsInstance(reason, str)
        self.assertIn('qualified_windows=4/4', reason)


if __name__ == '__main__':
    unittest.main()
