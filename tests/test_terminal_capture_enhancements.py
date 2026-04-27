"""Tests for terminal capture reward shaping and goal radius curriculum."""

import unittest

import numpy as np
import torch

from brain_uav.config import ExperimentConfig, RewardConfig, ScenarioConfig
from brain_uav.envs import StaticNoFlyTrajectoryEnv
from brain_uav.envs.static_no_fly_env_runtime import Zone
from brain_uav.scripts.common import make_env
from brain_uav.trainers.td3 import TD3Trainer


class _DummySpace:
    def __init__(self) -> None:
        self.low = np.array([-1.0, -1.0], dtype=np.float32)
        self.high = np.array([1.0, 1.0], dtype=np.float32)

    def sample(self) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class _TerminalObsEnv:
    def __init__(self) -> None:
        self.action_space = _DummySpace()
        self._obs = np.zeros(24, dtype=np.float32)
        self.trajectory = [np.zeros(3, dtype=np.float32)]
        self.state = np.zeros(5, dtype=np.float32)


class _LineSafetyTimingEnv:
    def __init__(self) -> None:
        self.action_space = _DummySpace()
        self._obs = np.zeros(24, dtype=np.float32)
        self.state = np.zeros(5, dtype=np.float32)
        self.trajectory = [np.zeros(3, dtype=np.float32)]

    def reset(self):
        self.state = np.zeros(5, dtype=np.float32)
        self.trajectory = [self.state[:3].copy()]
        return self._obs.copy(), {}

    def step(self, action: np.ndarray):
        del action
        self.state = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.trajectory.append(self.state[:3].copy())
        return self._obs.copy(), 0.0, False, False, {'outcome': 'running'}

    def _line_to_goal_is_safe(self, pos: np.ndarray, clearance: float = 0.0) -> bool:
        del clearance
        return bool(pos[0] < 0.5)


class _ZeroActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(24, 2)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(obs)


class _ConstantCritic(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.value = float(value)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return torch.ones((obs.shape[0], 1), dtype=obs.dtype, device=obs.device) * (self.anchor + self.value)


def _fixed_scenario(level: str) -> dict:
    return {
        'state': [-40.0, 0.0, 20.0, 0.0, 0.0],
        'goal': [40.0, 0.0, 20.0],
        'zones': [],
        'curriculum_level': level,
    }


def _terminal_obs(*, dx: float, dy: float, dz: float, gamma: float = 0.0, psi: float = 0.0) -> torch.Tensor:
    obs = torch.zeros((1, 24), dtype=torch.float32)
    obs[0, 3] = gamma
    obs[0, 4] = psi
    obs[0, 5] = dx
    obs[0, 6] = dy
    obs[0, 7] = dz
    return obs


class TestTerminalCaptureEnhancements(unittest.TestCase):
    def test_reward_config_defaults_for_terminal_capture(self):
        rewards = RewardConfig()

        self.assertEqual(rewards.goal_reward, 5000.0)
        self.assertEqual(rewards.timeout_penalty, 4000.0)

    def test_goal_radius_curriculum_per_level(self):
        scenario = ScenarioConfig(goal_radius_curriculum_enabled=True)
        rewards = RewardConfig()
        expected = {
            'easy': 10.0,
            'easy_two_zone': 8.0,
            'medium': 6.5,
            'hard': 5.0,
            'benchmark': 5.0,
        }

        for level, radius in expected.items():
            with self.subTest(level=level):
                env = StaticNoFlyTrajectoryEnv(
                    scenario=scenario,
                    rewards=rewards,
                    fixed_scenarios=[_fixed_scenario(level)],
                )
                env.reset()
                self.assertEqual(env._active_goal_radius(), radius)

    def test_dataset_and_bc_env_keep_goal_radius_fixed_at_five(self):
        cfg = ExperimentConfig()
        env = make_env(
            cfg,
            seed=7,
            curriculum_level='easy',
            curriculum_mix={'easy': 1.0},
            goal_radius_curriculum_enabled=False,
        )
        env.reset(seed=7)

        self.assertEqual(cfg.scenario.goal_radius, 5.0)
        self.assertEqual(env._active_goal_radius(), 5.0)

    def test_terminal_rewards_require_goal_proximity_and_safe_line(self):
        scenario = ScenarioConfig(min_no_fly_zones=0, max_no_fly_zones=0)
        env = StaticNoFlyTrajectoryEnv(scenario=scenario, rewards=RewardConfig(), fixed_scenarios=[_fixed_scenario('hard')])
        env.reset()
        env.state = np.array([0.0, 0.0, 20.0, 0.0, 0.0], dtype=np.float32)
        env.goal = np.array([20.0, 0.0, 20.0], dtype=np.float32)
        safe_los = env._terminal_los_reward(env.state[:3], float(env.state[3]), float(env.state[4]), 'running')
        safe_tangent = env._terminal_radial_tangential_reward(
            env.state[:3],
            float(env.state[3]),
            float(env.state[4]),
            'running',
        )

        self.assertGreater(safe_los, 0.0)
        self.assertGreater(safe_tangent, 0.0)

        env.goal = np.array([120.0, 0.0, 20.0], dtype=np.float32)
        self.assertEqual(env._terminal_los_reward(env.state[:3], float(env.state[3]), float(env.state[4]), 'running'), 0.0)
        self.assertEqual(
            env._terminal_radial_tangential_reward(env.state[:3], float(env.state[3]), float(env.state[4]), 'running'),
            0.0,
        )

        env.goal = np.array([20.0, 0.0, 20.0], dtype=np.float32)

        env.zones = [Zone(center_xy=np.array([10.0, 0.0], dtype=np.float32), radius=25.0)]
        unsafe_los = env._terminal_los_reward(env.state[:3], float(env.state[3]), float(env.state[4]), 'running')
        unsafe_tangent = env._terminal_radial_tangential_reward(
            env.state[:3],
            float(env.state[3]),
            float(env.state[4]),
            'running',
        )

        self.assertEqual(unsafe_los, 0.0)
        self.assertEqual(unsafe_tangent, 0.0)

    def test_actor_loss_includes_terminal_geo_term(self):
        trainer = TD3Trainer(
            env=_TerminalObsEnv(),
            actor=_ZeroActor(),
            critic1=_ConstantCritic(6.0),
            critic2=_ConstantCritic(6.0),
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
            terminal_geo_regularization_enabled=True,
            terminal_geo_radius=80.0,
            terminal_geo_lambda=0.15,
            terminal_geo_safe_clearance=3.0,
        )
        obs = _terminal_obs(dx=20.0, dy=20.0, dz=0.0)
        safe_mask = torch.ones((1, 1), dtype=torch.float32)

        (
            actor_loss,
            rl_actor_loss,
            scaled_rl_actor_loss,
            bc_loss,
            bc_lambda,
            _actor_rl_scale,
            terminal_geo_loss,
            terminal_geo_lambda,
        ) = trainer._compute_actor_loss_terms(obs, safe_mask)

        self.assertGreater(terminal_geo_loss.item(), 0.0)
        self.assertEqual(terminal_geo_lambda, 0.15)
        self.assertAlmostEqual(
            actor_loss.item(),
            scaled_rl_actor_loss.item() + bc_lambda * bc_loss.item() + terminal_geo_lambda * terminal_geo_loss.item(),
            places=5,
        )
        self.assertAlmostEqual(rl_actor_loss.item(), scaled_rl_actor_loss.item(), places=5)

    def test_terminal_geo_loss_zero_without_terminal_samples(self):
        trainer = TD3Trainer(
            env=_TerminalObsEnv(),
            actor=_ZeroActor(),
            critic1=_ConstantCritic(6.0),
            critic2=_ConstantCritic(6.0),
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
            terminal_geo_regularization_enabled=True,
            terminal_geo_radius=80.0,
            terminal_geo_lambda=0.15,
            terminal_geo_safe_clearance=3.0,
        )
        obs = _terminal_obs(dx=120.0, dy=0.0, dz=0.0)
        safe_mask = torch.zeros((1, 1), dtype=torch.float32)

        (
            _actor_loss,
            _rl_actor_loss,
            _scaled_rl_actor_loss,
            _bc_loss,
            _bc_lambda,
            _actor_rl_scale,
            terminal_geo_loss,
            terminal_geo_lambda,
        ) = trainer._compute_actor_loss_terms(obs, safe_mask)

        self.assertEqual(terminal_geo_loss.item(), 0.0)
        self.assertEqual(terminal_geo_lambda, 0.0)

    def test_replay_line_to_goal_safe_matches_transition_obs(self):
        env = _LineSafetyTimingEnv()
        trainer = TD3Trainer(
            env=env,
            actor=_ZeroActor(),
            critic1=_ConstantCritic(6.0),
            critic2=_ConstantCritic(6.0),
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.01,
            noise_clip=0.02,
            policy_delay=2,
            replay_size=8,
            batch_size=4,
            warmup_steps=1,
            exploration_noise=0.01,
            success_sample_bias=1.0,
        )

        trainer.train(total_timesteps=1, verbose=False)

        self.assertTrue(bool(trainer.replay.line_to_goal_safe[0]))


if __name__ == '__main__':
    unittest.main()
