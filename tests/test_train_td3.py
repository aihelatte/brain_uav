"""Tests for TD3 script helpers."""

import unittest
from unittest import mock

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


class _ConstantCritic(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return torch.full((obs.shape[0], 1), self.value, dtype=obs.dtype, device=obs.device)


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

    def test_training_config_defaults_raise_timeout_and_success_bias(self):
        cfg = ExperimentConfig()

        self.assertEqual(cfg.training.success_sample_bias, 4.0)
        self.assertEqual(cfg.training.actor_grad_clip_norm, 1.0)
        self.assertGreater(cfg.rewards.timeout_penalty, 1500.0)
        self.assertEqual(cfg.rewards.timeout_penalty, 2500.0)
        self.assertEqual(cfg.training.actor_rl_scale_alpha, 2.5)

    def test_ann_default_actor_freeze_steps_is_5000(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'ann', '--curriculum-level', 'easy'])
        cfg = ExperimentConfig()

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_freeze_steps, 5000)

    def test_ann_default_learning_rates_are_midrange(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'ann', '--curriculum-level', 'easy'])
        cfg = ExperimentConfig()

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_lr, 1.5e-4)
        self.assertEqual(cfg.training.critic_lr, 1.5e-4)

    def test_snn_default_learning_rates_remain_unchanged(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn', '--curriculum-level', 'easy'])
        cfg = ExperimentConfig()
        baseline_actor_lr = cfg.training.actor_lr
        baseline_critic_lr = cfg.training.critic_lr

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_lr, baseline_actor_lr)
        self.assertEqual(cfg.training.critic_lr, baseline_critic_lr)

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

    def test_actor_grad_clip_is_used_in_actor_update(self):
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
            policy_delay=1,
            replay_size=32,
            batch_size=2,
            warmup_steps=0,
            exploration_noise=0.01,
            success_sample_bias=1.0,
            actor_freeze_steps=0,
            actor_grad_clip_norm=1.0,
        )
        obs = np.zeros(4, dtype=np.float32)
        action = np.zeros(2, dtype=np.float32)
        trainer.replay.add(obs, action, 1.0, obs, False)
        trainer.replay.add(obs + 1.0, action, 1.0, obs + 1.0, True)
        trainer.total_steps = 1

        with mock.patch('torch.nn.utils.clip_grad_norm_') as clip_mock:
            trainer._update()

        clip_mock.assert_called_once()
        self.assertEqual(clip_mock.call_args.kwargs['max_norm'], 1.0)

    def test_actor_loss_uses_scaled_rl_term_with_bc_reference(self):
        trainer = TD3Trainer(
            env=_OneStepEnv(),
            actor=_DummyActor(),
            critic1=_ConstantCritic(10.0),
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
            actor_rl_scale_alpha=2.5,
            bc_reference_actor=_DummyActor(),
        )
        trainer.total_steps = 20_000
        obs = torch.zeros((2, 4), dtype=torch.float32)

        actor_loss, rl_actor_loss, scaled_rl_actor_loss, bc_loss, bc_lambda, actor_rl_scale = trainer._compute_actor_loss_terms(obs)

        self.assertAlmostEqual(rl_actor_loss.item(), -10.0)
        self.assertAlmostEqual(actor_rl_scale, 0.25)
        self.assertAlmostEqual(scaled_rl_actor_loss.item(), -2.5)
        self.assertAlmostEqual(actor_loss.item(), scaled_rl_actor_loss.item() + bc_lambda * bc_loss.item(), places=5)

    def test_actor_loss_without_bc_reference_stays_raw_rl_loss(self):
        trainer = TD3Trainer(
            env=_OneStepEnv(),
            actor=_DummyActor(),
            critic1=_ConstantCritic(8.0),
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
            actor_rl_scale_alpha=2.5,
        )
        obs = torch.zeros((2, 4), dtype=torch.float32)

        actor_loss, rl_actor_loss, scaled_rl_actor_loss, bc_loss, bc_lambda, actor_rl_scale = trainer._compute_actor_loss_terms(obs)

        self.assertAlmostEqual(rl_actor_loss.item(), -8.0)
        self.assertAlmostEqual(scaled_rl_actor_loss.item(), rl_actor_loss.item())
        self.assertEqual(actor_rl_scale, 1.0)
        self.assertEqual(bc_lambda, 0.0)
        self.assertAlmostEqual(bc_loss.item(), 0.0)
        self.assertAlmostEqual(actor_loss.item(), rl_actor_loss.item())

    def test_metrics_include_scaled_rl_fields(self):
        metrics = TD3Trainer(
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
        ).metrics.to_dict()

        self.assertIn('scaled_rl_actor_loss', metrics)
        self.assertIn('actor_rl_scale', metrics)

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
