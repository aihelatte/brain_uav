"""Tests for TD3 script helpers."""

import unittest

import numpy as np
import torch

from brain_uav.config import ExperimentConfig
from brain_uav.scripts.train_td3 import apply_model_training_overrides, build_parser, make_early_stop_callback
from brain_uav.trainers.td3 import TD3Trainer


class _DummySpace:
    def __init__(self) -> None:
        self.low = np.array([-1.0, -1.0], dtype=np.float32)
        self.high = np.array([1.0, 1.0], dtype=np.float32)

    def sample(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class _OneStepEnv:
    def __init__(self) -> None:
        self.action_space = _DummySpace()
        self._obs = np.zeros(4, dtype=np.float32)
        self.trajectory = [np.zeros(3, dtype=np.float32)]
        self.state = np.zeros(5, dtype=np.float32)

    def reset(self, seed: int | None = None):
        del seed
        self.trajectory = [np.zeros(3, dtype=np.float32)]
        self.state = np.zeros(5, dtype=np.float32)
        return self._obs.copy(), {}

    def step(self, action):
        del action
        self.trajectory.append(np.zeros(3, dtype=np.float32))
        return self._obs.copy(), 1.0, True, False, {
            'outcome': 'goal',
            'goal_distance': 0.0,
            'segment_goal_distance': 0.0,
            'goal_reached_by_segment': True,
            'progress': 1.0,
            'steps': 1,
            'curriculum_level': 'easy',
        }

    def export_scenario(self):
        return {'state': [0, 0, 0, 0, 0], 'goal': [0, 0, 0], 'zones': [], 'curriculum_level': 'easy'}


class _DummyActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


class _DummyCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(6, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([obs, action], dim=-1))


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

    def test_ann_default_actor_freeze_steps_is_5000(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'ann', '--curriculum-level', 'easy'])
        cfg = ExperimentConfig()

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_freeze_steps, 5000)

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

    def test_bc_lambda_schedule_is_500_150_30_5(self):
        trainer = TD3Trainer(
            env=_OneStepEnv(),
            actor=_DummyActor(),
            critic1=_DummyCritic(),
            critic2=_DummyCritic(),
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.01,
            noise_clip=0.02,
            policy_delay=2,
            replay_size=32,
            batch_size=2,
            warmup_steps=0,
            exploration_noise=0.01,
            success_sample_bias=1.0,
        )
        expected = {
            0: 500.0,
            14999: 500.0,
            15000: 150.0,
            29999: 150.0,
            30000: 30.0,
            49999: 30.0,
            50000: 5.0,
            150000: 5.0,
        }
        for steps, value in expected.items():
            trainer.total_steps = steps
            self.assertEqual(trainer._bc_lambda(), value)

    def test_true_early_stop_sets_early_stopped(self):
        trainer = TD3Trainer(
            env=_OneStepEnv(),
            actor=_DummyActor(),
            critic1=_DummyCritic(),
            critic2=_DummyCritic(),
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.01,
            noise_clip=0.02,
            policy_delay=2,
            replay_size=32,
            batch_size=64,
            warmup_steps=10,
            exploration_noise=0.01,
            success_sample_bias=1.0,
        )

        trainer.train(
            total_timesteps=3,
            verbose=False,
            summary_every_episodes=1,
            window_callback=lambda _: 'loop stop',
        )
        self.assertTrue(trainer.early_stopped)
        self.assertEqual(trainer.stop_reason, 'loop stop')

    def test_final_flush_stop_reason_does_not_mark_early_stopped(self):
        trainer = TD3Trainer(
            env=_OneStepEnv(),
            actor=_DummyActor(),
            critic1=_DummyCritic(),
            critic2=_DummyCritic(),
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.01,
            noise_clip=0.02,
            policy_delay=2,
            replay_size=32,
            batch_size=64,
            warmup_steps=10,
            exploration_noise=0.01,
            success_sample_bias=1.0,
        )

        trainer.train(
            total_timesteps=1,
            verbose=False,
            summary_every_episodes=2,
            window_callback=lambda _: 'final flush stop',
        )
        self.assertFalse(trainer.early_stopped)
        self.assertEqual(trainer.stop_reason, 'final flush stop')


if __name__ == '__main__':
    unittest.main()
