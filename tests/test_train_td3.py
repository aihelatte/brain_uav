"""Tests for TD3 script helpers."""

import unittest
from unittest import mock
from pathlib import Path
import tempfile

import numpy as np
import torch

from brain_uav.config import ExperimentConfig, ScenarioConfig
from brain_uav.scripts.train_td3 import (
    apply_model_training_overrides,
    build_parser,
    export_episode_result,
    make_early_stop_callback,
    make_episode_capture_callback,
    _resolve_active_goal_radius,
    load_training_state,
)
from brain_uav.trainers.td3 import TD3Trainer


class _DummySpace:
    def __init__(self) -> None:
        self.low = np.array([-1.0, -1.0], dtype=np.float32)
        self.high = np.array([1.0, 1.0], dtype=np.float32)

    def sample(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class _SimpleZone:
    def __init__(self, center_xy, radius: float) -> None:
        self.center_xy = np.asarray(center_xy, dtype=np.float32)
        self.radius = float(radius)


class _OneStepEnv:
    def __init__(self) -> None:
        self.action_space = _DummySpace()
        self._obs = np.zeros(24, dtype=np.float32)
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

    def _line_to_goal_is_safe(self, pos, clearance=0.0):
        del pos, clearance
        return True


class _EpisodeSequenceEnv:
    def __init__(
        self,
        outcomes: list[str],
        episode_length: int = 5,
        zones: list[_SimpleZone] | None = None,
        trajectory_point: np.ndarray | None = None,
    ) -> None:
        self.action_space = _DummySpace()
        self._obs = np.zeros(24, dtype=np.float32)
        self._trajectory_point = (
            np.asarray(trajectory_point, dtype=np.float32)
            if trajectory_point is not None
            else np.zeros(3, dtype=np.float32)
        )
        self.trajectory = [self._trajectory_point.copy()]
        self.state = np.zeros(5, dtype=np.float32)
        self.outcomes = outcomes
        self.episode_length = episode_length
        self.episode_idx = -1
        self.step_idx = 0
        self.current_outcome = outcomes[0]
        self.zones = zones or []

    def reset(self, seed: int | None = None):
        del seed
        self.episode_idx += 1
        self.step_idx = 0
        self.current_outcome = self.outcomes[min(self.episode_idx, len(self.outcomes) - 1)]
        self.trajectory = [self._trajectory_point.copy()]
        self.state = np.zeros(5, dtype=np.float32)
        return self._obs.copy(), {}

    def step(self, action):
        del action
        self.step_idx += 1
        self.trajectory.append(self._trajectory_point.copy())
        done = self.step_idx >= self.episode_length
        if done:
            outcome = self.current_outcome
            terminated = outcome != 'timeout'
            truncated = outcome == 'timeout'
        else:
            outcome = 'running'
            terminated = False
            truncated = False
        goal_distance = 60.0 if self.step_idx >= self.episode_length - 1 else 120.0
        segment_goal_distance = 70.0 if self.step_idx >= self.episode_length - 1 else 120.0
        return self._obs.copy(), 1.0, terminated, truncated, {
            'outcome': outcome,
            'goal_distance': goal_distance,
            'segment_goal_distance': segment_goal_distance,
            'goal_reached_by_segment': bool(done and self.current_outcome == 'goal'),
            'progress': 1.0,
            'steps': self.step_idx,
            'curriculum_level': 'easy',
        }

    def export_scenario(self):
        return {'state': [0, 0, 0, 0, 0], 'goal': [0, 0, 0], 'zones': [], 'curriculum_level': 'easy'}

    def _line_to_goal_is_safe(self, pos, clearance=0.0):
        del pos, clearance
        return True


class _DummyActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(24, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


class _DummyCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(26, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([obs, action], dim=-1))


class _ConstantCritic(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.value = float(value)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return torch.ones((obs.shape[0], 1), dtype=obs.dtype, device=obs.device) * (self.anchor + self.value)


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
        self.assertEqual(args.early_stop_min_steps, 125000)

    def test_training_config_defaults_raise_timeout_and_success_bias(self):
        cfg = ExperimentConfig()

        self.assertEqual(cfg.training.success_sample_bias, 2.0)
        self.assertEqual(cfg.training.success_replay_min_zone_clearance, 30.0)
        self.assertEqual(cfg.training.success_primary_min_zone_clearance, 80.0)
        self.assertEqual(cfg.training.success_batch_fraction, 0.25)
        self.assertEqual(cfg.training.near_goal_sample_bias, 2.0)
        self.assertEqual(cfg.training.replay_size, 500_000)
        self.assertEqual(cfg.training.warmup_steps, 1280)
        self.assertEqual(cfg.training.near_goal_radius, 250.0)
        self.assertEqual(cfg.training.terminal_geo_radius, 250.0)
        self.assertEqual(cfg.training.terminal_geo_lambda, 3000.0)
        self.assertEqual(cfg.training.terminal_geo_safe_clearance, 40.0)
        self.assertEqual(cfg.training.actor_grad_clip_norm, 1.0)
        self.assertGreater(cfg.rewards.timeout_penalty, 1500.0)
        self.assertEqual(cfg.rewards.timeout_penalty, 5000.0)
        self.assertEqual(cfg.rewards.goal_reward, 5000.0)
        self.assertEqual(cfg.training.actor_rl_scale_alpha, 2.5)

    def test_ann_default_actor_freeze_steps_is_25000(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'ann', '--curriculum-level', 'easy'])
        cfg = ExperimentConfig()

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_freeze_steps, 25000)

    def test_ann_default_learning_rates_are_midrange(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'ann', '--curriculum-level', 'easy'])
        cfg = ExperimentConfig()

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_lr, 1.5e-4)
        self.assertEqual(cfg.training.critic_lr, 2.5e-4)

    def test_ann_easy_two_zone_uses_conservative_stability_overrides(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'ann', '--curriculum-level', 'easy_two_zone'])
        cfg = ExperimentConfig()

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_lr, 1.5e-4)
        self.assertEqual(cfg.training.critic_lr, 2.0e-4)
        self.assertEqual(cfg.rewards.collision_penalty, 18_000.0)
        self.assertEqual(cfg.rewards.ground_soft_penalty_weight, 180.0)
        self.assertEqual(cfg.rewards.ground_soft_penalty_cap, 300.0)
        self.assertEqual(cfg.rewards.descent_trend_penalty_weight, 120.0)
        self.assertEqual(cfg.rewards.descent_trend_penalty_cap, 260.0)

    def test_ann_easy_medium_hard_do_not_use_easy_two_zone_reward_overrides(self):
        parser = build_parser()
        for level in ('easy', 'medium', 'hard'):
            with self.subTest(level=level):
                args = parser.parse_args(['--model', 'ann', '--curriculum-level', level])
                cfg = ExperimentConfig()

                apply_model_training_overrides(cfg, args)

                self.assertEqual(cfg.rewards.collision_penalty, 12_000.0)
                self.assertEqual(cfg.rewards.ground_soft_penalty_weight, 120.0)
                self.assertEqual(cfg.rewards.ground_soft_penalty_cap, 200.0)
                self.assertEqual(cfg.rewards.descent_trend_penalty_weight, 80.0)
                self.assertEqual(cfg.rewards.descent_trend_penalty_cap, 180.0)

    def test_snn_easy_two_zone_does_not_use_ann_stability_overrides(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn', '--curriculum-level', 'easy_two_zone'])
        cfg = ExperimentConfig()
        baseline_actor_lr = cfg.training.actor_lr
        baseline_critic_lr = cfg.training.critic_lr

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_lr, baseline_actor_lr)
        self.assertEqual(cfg.training.critic_lr, baseline_critic_lr)
        self.assertEqual(cfg.rewards.collision_penalty, 12_000.0)
        self.assertEqual(cfg.rewards.ground_soft_penalty_weight, 120.0)

    def test_snn_default_learning_rates_remain_unchanged(self):
        parser = build_parser()
        args = parser.parse_args(['--model', 'snn', '--curriculum-level', 'easy'])
        cfg = ExperimentConfig()
        baseline_actor_lr = cfg.training.actor_lr
        baseline_critic_lr = cfg.training.critic_lr

        apply_model_training_overrides(cfg, args)

        self.assertEqual(cfg.training.actor_lr, baseline_actor_lr)
        self.assertEqual(cfg.training.critic_lr, baseline_critic_lr)

    def test_td3_default_timesteps_use_1000_step_scale(self):
        source = Path('E:/wurenji/my_project/src/brain_uav/scripts/train_td3.py').read_text(encoding='utf-8')

        self.assertIn(
            "args.timesteps = 500000 if args.curriculum_level == 'hard' else 400000",
            source,
        )

    def test_early_stop_callback_uses_new_rule(self):
        callback = make_early_stop_callback(
            enabled=True,
            goal_rate_threshold=0.95,
            consecutive_windows=4,
            min_steps=125000,
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
            'total_steps': 125000,
        }
        self.assertIsNone(callback({**window, 'total_steps': 3000}))
        self.assertIsNone(callback({**window, 'total_steps': 125000}))
        self.assertIsNone(callback({**window, 'total_steps': 126000}))
        reason = callback({**window, 'total_steps': 127000})
        self.assertIsInstance(reason, str)
        self.assertIn('qualified_windows=4/4', reason)

    def test_goal_examples_are_kept_once_per_five_windows(self):
        callback = make_episode_capture_callback(
            result_root=self._make_dummy_path(),
            summary_every_episodes=15,
            total_timesteps=900000,
            config_payload={'scenario': {'world_xy': ScenarioConfig().world_xy, 'world_z_max': ScenarioConfig().world_z_max, 'goal_radius': 5.0,
                                         'warning_distance': 10.0, 'boundary_warning_distance': 10.0,
                                         'ground_warning_height': 4.0}},
        )
        record = {
            'episode': 1,
            'total_steps': 1,
            'return': 0.0,
            'length': 1,
            'outcome': 'goal',
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'scenario': {'state': [0, 0, 0, 0, 0], 'goal': [1, 1, 1], 'zones': []},
            'trajectory': [[0, 0, 0], [1, 1, 1]],
            'final_state': [1, 1, 1, 0, 0],
            'info': {'goal_distance': 0.0, 'curriculum_level': 'easy'},
        }
        saved_stems: list[str] = []

        with mock.patch('brain_uav.scripts.train_td3.export_episode_result', side_effect=lambda target_dir, stem, record, config_payload: saved_stems.append(stem) or {}):
            for window_idx in range(12):
                episode = window_idx * 15 + 1
                callback({**record, 'episode': episode, 'total_steps': episode})

        goal_stems = [stem for stem in saved_stems if stem.startswith('goal_group_')]
        self.assertEqual(goal_stems, ['goal_group_01_ep00001', 'goal_group_02_ep00076', 'goal_group_03_ep00151'])

    def test_export_episode_result_prefers_active_goal_radius_and_writes_outputs(self):
        import tempfile
        from pathlib import Path

        record = {
            'episode': 1,
            'total_steps': 1,
            'return': 0.0,
            'length': 1,
            'outcome': 'goal',
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'scenario': {'state': [0, 0, 10, 0, 0], 'goal': [20, 0, 10], 'zones': []},
            'trajectory': [[0, 0, 10], [20, 0, 10]],
            'final_state': [20, 0, 10, 0, 0],
            'info': {'goal_distance': 0.0, 'curriculum_level': 'easy', 'active_goal_radius': 10.0},
        }
        config_payload = {
            'scenario': {
                'world_xy': ScenarioConfig().world_xy,
                'world_z_max': ScenarioConfig().world_z_max,
                'goal_radius': 5.0,
                'warning_distance': 10.0,
                'boundary_warning_distance': 10.0,
                'ground_warning_height': 4.0,
            }
        }

        self.assertEqual(_resolve_active_goal_radius(record, config_payload['scenario']), 10.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch('brain_uav.scripts.train_td3._draw_goal_radius_projection') as goal_radius_mock:
                outputs = export_episode_result(Path(tmpdir), 'episode_demo', record, config_payload)
            self.assertTrue(Path(outputs['json']).is_file())
            self.assertTrue(Path(outputs['png']).is_file())
            self.assertEqual(goal_radius_mock.call_count, 3)
            for call in goal_radius_mock.call_args_list:
                self.assertEqual(call.args[2], 10.0)

    @staticmethod
    def _make_dummy_path():
        import tempfile
        from pathlib import Path

        return Path(tempfile.mkdtemp())

    @staticmethod
    def _make_trainer(actor=None, critic1=None, critic2=None) -> TD3Trainer:
        return TD3Trainer(
            env=_OneStepEnv(),
            actor=actor or _DummyActor(),
            critic1=critic1 or _DummyCritic(),
            critic2=critic2 or _DummyCritic(),
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

    def test_actor_only_checkpoint_sets_bc_reference_source(self):
        actor = _DummyActor()
        critic1 = _DummyCritic()
        critic2 = _DummyCritic()
        trainer = self._make_trainer(actor=actor, critic1=critic1, critic2=critic2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / 'bc.pt'
            torch.save({'state_dict': actor.state_dict()}, checkpoint)
            strategy = load_training_state(checkpoint, actor, critic1, critic2, trainer, '[TEST]')

        self.assertEqual(strategy, 'policy')
        self.assertTrue(trainer.metrics.bc_regularization_enabled)
        self.assertEqual(trainer.metrics.reference_source, 'bc')
        self.assertIsNotNone(trainer.bc_reference_actor)

    def test_td3_checkpoint_sets_previous_stage_reference_source(self):
        actor = _DummyActor()
        critic1 = _DummyCritic()
        critic2 = _DummyCritic()
        trainer = self._make_trainer(actor=actor, critic1=critic1, critic2=critic2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / 'td3.pt'
            torch.save(
                {
                    'state_dict': actor.state_dict(),
                    'critic1_state_dict': critic1.state_dict(),
                    'critic2_state_dict': critic2.state_dict(),
                },
                checkpoint,
            )
            strategy = load_training_state(checkpoint, actor, critic1, critic2, trainer, '[TEST]')

        self.assertEqual(strategy, 'policy')
        self.assertTrue(trainer.metrics.bc_regularization_enabled)
        self.assertEqual(trainer.metrics.reference_source, 'previous_stage')
        self.assertIsNotNone(trainer.bc_reference_actor)

    def test_previous_stage_reference_contributes_bc_loss_terms(self):
        trainer = self._make_trainer(actor=_DummyActor())
        trainer.set_bc_reference_actor(_DummyActor(), source='previous_stage')
        trainer.total_steps = 80_000
        obs = torch.zeros((2, 24), dtype=torch.float32)
        line_to_goal_safe = torch.zeros((2, 1), dtype=torch.float32)

        (
            _actor_loss,
            _rl_actor_loss,
            _scaled_rl_actor_loss,
            bc_loss,
            bc_lambda,
            _actor_rl_scale,
            _terminal_geo_loss,
            _terminal_geo_lambda,
        ) = trainer._compute_actor_loss_terms(obs, line_to_goal_safe)

        self.assertEqual(trainer.metrics.reference_source, 'previous_stage')
        self.assertGreater(bc_lambda, 0.0)
        self.assertGreaterEqual(bc_loss.item(), 0.0)

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
            74999: 500.0,
            75000: 150.0,
            149999: 150.0,
            150000: 30.0,
            249999: 30.0,
            250000: 5.0,
            300000: 5.0,
        }
        for steps, value in expected.items():
            trainer.total_steps = steps
            self.assertEqual(trainer._bc_lambda(), value)

    def test_easy_two_zone_bc_lambda_keeps_stronger_late_anchor(self):
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
            curriculum_level='easy_two_zone',
            easy_two_zone_late_bc_lambda=20.0,
        )
        expected = {
            0: 500.0,
            74999: 500.0,
            75000: 150.0,
            149999: 150.0,
            150000: 30.0,
            249999: 30.0,
            250000: 20.0,
            600000: 20.0,
        }
        for steps, value in expected.items():
            trainer.total_steps = steps
            self.assertEqual(trainer._bc_lambda(), value)

    def test_easy_two_zone_bc_lambda_defaults_to_global_schedule_without_override(self):
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
            curriculum_level='easy_two_zone',
        )

        trainer.total_steps = 250000

        self.assertEqual(trainer._bc_lambda(), 5.0)

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
        obs = np.zeros(24, dtype=np.float32)
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
            terminal_geo_regularization_enabled=False,
        )
        trainer.total_steps = 20_000
        obs = torch.zeros((2, 24), dtype=torch.float32)
        line_to_goal_safe = torch.ones((2, 1), dtype=torch.float32)

        (
            actor_loss,
            rl_actor_loss,
            scaled_rl_actor_loss,
            bc_loss,
            bc_lambda,
            actor_rl_scale,
            terminal_geo_loss,
            terminal_geo_lambda,
        ) = trainer._compute_actor_loss_terms(obs, line_to_goal_safe)

        self.assertAlmostEqual(rl_actor_loss.item(), -10.0)
        self.assertAlmostEqual(actor_rl_scale, 0.25)
        self.assertAlmostEqual(scaled_rl_actor_loss.item(), -2.5)
        self.assertAlmostEqual(actor_loss.item(), scaled_rl_actor_loss.item() + bc_lambda * bc_loss.item(), places=5)
        self.assertAlmostEqual(terminal_geo_loss.item(), 0.0)
        self.assertEqual(terminal_geo_lambda, 0.0)

    def test_actor_loss_without_bc_reference_still_uses_q_scale(self):
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
            terminal_geo_regularization_enabled=False,
        )
        obs = torch.zeros((2, 24), dtype=torch.float32)
        line_to_goal_safe = torch.ones((2, 1), dtype=torch.float32)

        (
            actor_loss,
            rl_actor_loss,
            scaled_rl_actor_loss,
            bc_loss,
            bc_lambda,
            actor_rl_scale,
            terminal_geo_loss,
            terminal_geo_lambda,
        ) = trainer._compute_actor_loss_terms(obs, line_to_goal_safe)

        self.assertAlmostEqual(rl_actor_loss.item(), -8.0)
        self.assertAlmostEqual(actor_rl_scale, 0.3125)
        self.assertAlmostEqual(scaled_rl_actor_loss.item(), -2.5)
        self.assertEqual(bc_lambda, 0.0)
        self.assertAlmostEqual(bc_loss.item(), 0.0)
        self.assertAlmostEqual(actor_loss.item(), scaled_rl_actor_loss.item())
        self.assertAlmostEqual(terminal_geo_loss.item(), 0.0)
        self.assertEqual(terminal_geo_lambda, 0.0)

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
        self.assertIn('terminal_geo_loss', metrics)
        self.assertIn('terminal_geo_lambda', metrics)
        self.assertIn('bc_actor_loss_contribution', metrics)
        self.assertIn('terminal_geo_loss_contribution', metrics)
        self.assertIn('reference_source', metrics)
        self.assertIn('success_replay_min_zone_clearance', metrics)
        self.assertIn('success_primary_min_zone_clearance', metrics)
        self.assertIn('last_episode_min_zone_clearance', metrics)
        self.assertIn('success_replay_accept_count', metrics)
        self.assertIn('success_replay_reject_count', metrics)
        self.assertIn('success_primary_accept_count', metrics)
        self.assertIn('success_primary_reject_count', metrics)

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

    def test_success_episode_enters_success_replay(self):
        trainer = TD3Trainer(
            env=_EpisodeSequenceEnv(['goal']),
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

        trainer.train(total_timesteps=5, verbose=False, summary_every_episodes=10)

        self.assertEqual(trainer.replay.success_size, 5)
        self.assertEqual(trainer.replay.success_count, 5)
        self.assertEqual(trainer.metrics.success_episode_accept_count, 1)
        self.assertEqual(trainer.metrics.success_episode_reject_count, 0)
        self.assertEqual(trainer.metrics.success_replay_accept_count, 1)
        self.assertEqual(trainer.metrics.success_replay_reject_count, 0)
        self.assertEqual(trainer.metrics.success_primary_accept_count, 1)
        self.assertEqual(trainer.metrics.success_primary_reject_count, 0)

    def test_failed_episode_does_not_enter_success_replay(self):
        trainer = TD3Trainer(
            env=_EpisodeSequenceEnv(['boundary']),
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

        trainer.train(total_timesteps=5, verbose=False, summary_every_episodes=10)

        self.assertEqual(trainer.replay.success_size, 0)
        self.assertEqual(trainer.replay.success_count, 0)

    def test_low_clearance_goal_does_not_enter_success_replay_or_primary_success_bias(self):
        trainer = TD3Trainer(
            env=_EpisodeSequenceEnv(
                ['goal'],
                zones=[_SimpleZone([0.0, 0.0], radius=10.0)],
                trajectory_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            ),
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
            success_sample_bias=2.0,
            success_replay_min_zone_clearance=30.0,
            success_primary_min_zone_clearance=80.0,
        )

        trainer.train(total_timesteps=5, verbose=False, summary_every_episodes=10)

        self.assertEqual(len(trainer.replay), 5)
        self.assertEqual(trainer.replay.success_size, 0)
        self.assertEqual(trainer.replay.success_count, 0)
        self.assertEqual(trainer.metrics.success_episode_accept_count, 0)
        self.assertEqual(trainer.metrics.success_episode_reject_count, 1)
        self.assertEqual(trainer.metrics.success_episode_reject_reason_zone_clearance, 1)
        self.assertEqual(trainer.metrics.success_replay_accept_count, 0)
        self.assertEqual(trainer.metrics.success_replay_reject_count, 1)
        self.assertEqual(trainer.metrics.success_replay_reject_reason_zone_clearance, 1)
        self.assertEqual(trainer.metrics.success_primary_accept_count, 0)
        self.assertEqual(trainer.metrics.success_primary_reject_count, 1)
        self.assertLess(trainer.metrics.last_episode_min_zone_clearance, 30.0)

    def test_medium_clearance_goal_enters_success_replay_without_primary_success_bias(self):
        trainer = TD3Trainer(
            env=_EpisodeSequenceEnv(
                ['goal'],
                zones=[_SimpleZone([0.0, 0.0], radius=10.0)],
                trajectory_point=np.array([60.0, 0.0, 0.0], dtype=np.float32),
            ),
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
            success_sample_bias=2.0,
            success_replay_min_zone_clearance=30.0,
            success_primary_min_zone_clearance=80.0,
        )

        trainer.train(total_timesteps=5, verbose=False, summary_every_episodes=10)

        self.assertEqual(len(trainer.replay), 5)
        self.assertEqual(trainer.replay.success_size, 5)
        self.assertEqual(trainer.replay.success_count, 0)
        self.assertEqual(trainer.metrics.success_replay_accept_count, 1)
        self.assertEqual(trainer.metrics.success_replay_reject_count, 0)
        self.assertEqual(trainer.metrics.success_primary_accept_count, 0)
        self.assertEqual(trainer.metrics.success_primary_reject_count, 1)
        self.assertGreaterEqual(trainer.metrics.last_episode_min_zone_clearance, 30.0)
        self.assertLess(trainer.metrics.last_episode_min_zone_clearance, 80.0)

    def test_high_clearance_goal_enters_success_replay_and_primary_success_bias(self):
        trainer = TD3Trainer(
            env=_EpisodeSequenceEnv(
                ['goal'],
                zones=[_SimpleZone([0.0, 0.0], radius=10.0)],
                trajectory_point=np.array([100.0, 0.0, 0.0], dtype=np.float32),
            ),
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
            success_sample_bias=2.0,
            success_replay_min_zone_clearance=30.0,
            success_primary_min_zone_clearance=80.0,
        )

        trainer.train(total_timesteps=5, verbose=False, summary_every_episodes=10)

        self.assertEqual(len(trainer.replay), 5)
        self.assertEqual(trainer.replay.success_size, 5)
        self.assertEqual(trainer.replay.success_count, 5)
        self.assertEqual(trainer.metrics.success_replay_accept_count, 1)
        self.assertEqual(trainer.metrics.success_primary_accept_count, 1)
        self.assertEqual(trainer.metrics.success_primary_reject_count, 0)
        self.assertGreaterEqual(trainer.metrics.last_episode_min_zone_clearance, 80.0)

    def test_failure_samples_do_not_overwrite_success_replay(self):
        trainer = TD3Trainer(
            env=_EpisodeSequenceEnv(['goal', 'boundary']),
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

        trainer.train(total_timesteps=10, verbose=False, summary_every_episodes=10)

        self.assertEqual(trainer.replay.success_size, 5)
        self.assertEqual(trainer.replay.success_count, 5)

    def test_near_goal_radius_uses_goal_or_segment_distance(self):
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
            near_goal_radius=250.0,
        )

        self.assertTrue(trainer._is_near_goal({'goal_distance': 249.0, 'segment_goal_distance': 300.0, 'goal_reached_by_segment': False}))
        self.assertTrue(trainer._is_near_goal({'goal_distance': 300.0, 'segment_goal_distance': 249.0, 'goal_reached_by_segment': False}))
        self.assertTrue(trainer._is_near_goal({'goal_distance': 300.0, 'segment_goal_distance': 300.0, 'goal_reached_by_segment': True}))

    def test_noise_decays_in_first_half_then_stays_final(self):
        trainer = TD3Trainer(
            env=_OneStepEnv(),
            actor=_DummyActor(),
            critic1=_DummyCritic(),
            critic2=_DummyCritic(),
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.015,
            noise_clip=0.03,
            policy_delay=2,
            replay_size=32,
            batch_size=2,
            warmup_steps=0,
            exploration_noise=0.02,
            success_sample_bias=1.0,
            noise_decay_fraction=0.5,
            exploration_noise_final=0.005,
            policy_noise_final=0.006,
            noise_clip_final=0.012,
        )
        trainer.current_stage_timesteps = 200

        trainer.total_steps = 0
        self.assertEqual(trainer._current_noise(), (0.02, 0.015, 0.03))

        trainer.total_steps = 50
        exploration_noise, policy_noise, noise_clip = trainer._current_noise()
        self.assertAlmostEqual(exploration_noise, 0.0125)
        self.assertAlmostEqual(policy_noise, 0.0105)
        self.assertAlmostEqual(noise_clip, 0.021)

        trainer.total_steps = 100
        self.assertEqual(trainer._current_noise(), (0.005, 0.006, 0.012))

        trainer.total_steps = 180
        self.assertEqual(trainer._current_noise(), (0.005, 0.006, 0.012))


if __name__ == '__main__':
    unittest.main()
