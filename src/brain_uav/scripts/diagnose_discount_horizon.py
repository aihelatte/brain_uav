"""Offline discount-horizon diagnostic (H2 in docs/无法早停排查文档.md).

Not part of the production training pipeline and not imported by it. Given
a trained V2 TD3 checkpoint (a passed/failed formal checkpoint or a C1
periodic snapshot) and one known-collision medium scenario, this replays
the loaded policy step by step in a fixed environment and, at each step,
compares two twin-critic-min Q values:

- ``Q(s, pi(s))``: the actual action the policy would take.
- ``Q(s, a_avoid)``: a manually constructed avoidance action that turns
  away from the nearest no-fly zone, perpendicular to the line from the
  UAV to that zone's center, by the maximum allowed yaw rate for one step.

It reports the step (if any) where ``Q(s, a_avoid) - Q(s, pi(s))`` first
turns positive, and how many steps that is before the episode's terminal
step (the "how far in advance does the critic see the avoidance action as
better" question from P1 in docs/无法早停排查文档.md).

Caveat this script is explicit about in its output: this project does not
persist replay buffer contents in any checkpoint (only actor/critic/
optimizer state -- see build_v2_periodic_snapshot in
trainers/v2_formal_training.py). So the "distance-to-collision transition
occupancy and |TD error| distribution in a replay batch" reading below uses
the diagnostic scenario's own re-simulated transitions as a stand-in for a
replay batch, not an actual training replay buffer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from brain_uav.config import RewardConfig
from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
from brain_uav.models import V2ANNCritic, V2ANNPolicyActor, V2SNNPolicyActor
from brain_uav.observations import collate_v2_observations
from brain_uav.scripts.common import DEVICE_CHOICES, resolve_training_device
from brain_uav.trainers.v2_formal_training import (
    _architecture_from_engine_payload,
    load_v2_formal_checkpoint,
    load_v2_periodic_snapshot,
)
from brain_uav.trainers.v2_validation import scenario_config_from_snapshot


def _load_checkpoint_payload(
    path: str | Path, *, model_type: str,
) -> dict[str, Any]:
    """Accept either a strict formal checkpoint or a C1 periodic snapshot."""

    try:
        return load_v2_formal_checkpoint(path, expected_model_type=model_type)
    except (ValueError, FileNotFoundError) as formal_error:
        try:
            return load_v2_periodic_snapshot(path, expected_model_type=model_type)
        except (ValueError, FileNotFoundError) as periodic_error:
            raise ValueError(
                f'{path} is neither a strict V2 formal checkpoint '
                f'({formal_error}) nor a V2 periodic snapshot '
                f'({periodic_error}).'
            ) from periodic_error


def load_actor_and_critics(
    checkpoint: str | Path, *, model_type: str, device: str = 'cpu',
) -> dict[str, Any]:
    """Reconstruct online actor/critic1/critic2 plus their target copies."""

    if model_type not in ('ann', 'snn'):
        raise ValueError('model_type must be "ann" or "snn".')
    payload = _load_checkpoint_payload(checkpoint, model_type=model_type)
    engine_payload = payload['engine_checkpoint']
    scenario = scenario_config_from_snapshot(payload['scenario_config'])
    rewards = RewardConfig(**payload['reward_config'])
    radius = float(payload['uav_collision_radius'])
    gamma = float(payload['formal_config']['gamma'])
    (
        scales, encoder_config, checkpoint_radius, actor_hidden,
        critic1_hidden, critic2_hidden, action_high, snn_metadata,
    ) = _architecture_from_engine_payload(engine_payload, model_type=model_type)
    if checkpoint_radius != radius:
        raise ValueError('Checkpoint uav_collision_radius is inconsistent.')

    def build_actor() -> V2ANNPolicyActor | V2SNNPolicyActor:
        if model_type == 'ann':
            return V2ANNPolicyActor(
                scales, int(action_high.shape[0]), actor_hidden, action_high,
                uav_radius=radius, encoder_config=encoder_config,
            )
        return V2SNNPolicyActor(
            scales, int(action_high.shape[0]), actor_hidden, action_high,
            time_window=snn_metadata['time_window'], tau=snn_metadata['tau'],
            uav_radius=radius, encoder_config=encoder_config,
        )

    actor = build_actor()
    actor.load_state_dict(engine_payload['actor_state_dict'], strict=True)
    actor_target = build_actor()
    actor_target.load_state_dict(
        engine_payload['actor_target_state_dict'], strict=True,
    )
    critic1 = V2ANNCritic(
        scales, 2, critic1_hidden, uav_radius=radius, encoder_config=encoder_config,
    )
    critic1.load_state_dict(engine_payload['critic1_state_dict'], strict=True)
    critic2 = V2ANNCritic(
        scales, 2, critic2_hidden, uav_radius=radius, encoder_config=encoder_config,
    )
    critic2.load_state_dict(engine_payload['critic2_state_dict'], strict=True)
    critic1_target = V2ANNCritic(
        scales, 2, critic1_hidden, uav_radius=radius, encoder_config=encoder_config,
    )
    critic1_target.load_state_dict(
        engine_payload['critic1_target_state_dict'], strict=True,
    )
    critic2_target = V2ANNCritic(
        scales, 2, critic2_hidden, uav_radius=radius, encoder_config=encoder_config,
    )
    critic2_target.load_state_dict(
        engine_payload['critic2_target_state_dict'], strict=True,
    )
    modules = (actor, actor_target, critic1, critic2, critic1_target, critic2_target)
    for module in modules:
        module.to(device)
        module.eval()
    return {
        'actor': actor, 'actor_target': actor_target,
        'critic1': critic1, 'critic2': critic2,
        'critic1_target': critic1_target, 'critic2_target': critic2_target,
        'scenario': scenario, 'rewards': rewards,
        'uav_collision_radius': radius, 'gamma': gamma,
        'action_high': action_high.detach().cpu().numpy().astype(np.float32),
    }


def _nearest_zone_avoidance_action(
    env: V2StaticNoFlyTrajectoryEnv, action_high: np.ndarray,
) -> np.ndarray:
    """A fixed-magnitude turn perpendicular to the nearest zone's bearing.

    This is a deliberately simple, deterministic heuristic (an example, not
    a claimed-optimal avoidance policy): turn away from whichever side the
    nearest zone's center is on, at the maximum allowed yaw rate, with no
    pitch change.
    """

    if not env.zones:
        return np.zeros_like(action_high)
    uav_xy = np.asarray(env.state[:2], dtype=np.float64)
    psi = float(env.state[4])
    nearest = min(
        env.zones,
        key=lambda zone: float(
            np.linalg.norm(np.asarray(zone.shape.center[:2], dtype=np.float64) - uav_xy)
        ),
    )
    to_zone = np.asarray(nearest.shape.center[:2], dtype=np.float64) - uav_xy
    heading = np.array([np.cos(psi), np.sin(psi)], dtype=np.float64)
    cross_z = heading[0] * to_zone[1] - heading[1] * to_zone[0]
    sign = -1.0 if cross_z >= 0.0 else 1.0
    return np.array([0.0, sign * float(action_high[1])], dtype=np.float32)


def _min_q(
    critic1: V2ANNCritic, critic2: V2ANNCritic,
    observation_batch, action: torch.Tensor,
) -> float:
    with torch.no_grad():
        q1 = critic1(observation_batch, action)
        q2 = critic2(observation_batch, action)
        return float(torch.minimum(q1, q2).item())


def run_discount_horizon_diagnostic(
    *,
    checkpoint: str | Path,
    scenario_payload: dict[str, Any],
    model_type: str = 'ann',
    device: str = 'cpu',
    near_collision_step_range: tuple[int, int] = (150, 300),
) -> dict[str, Any]:
    models = load_actor_and_critics(checkpoint, model_type=model_type, device=device)
    env = V2StaticNoFlyTrajectoryEnv(
        models['scenario'], models['rewards'],
        uav_collision_radius=models['uav_collision_radius'],
    )
    observation, _ = env.reset(options={'scenario': scenario_payload})
    action_high = models['action_high']

    per_step: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    outcome = 'running'
    while outcome == 'running':
        batch = collate_v2_observations([observation], device=device)
        with torch.no_grad():
            policy_action = (
                models['actor'](batch)[0].detach().cpu().numpy().astype(np.float32)
            )
        avoid_action = _nearest_zone_avoidance_action(env, action_high)
        policy_tensor = torch.as_tensor(
            policy_action, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        avoid_tensor = torch.as_tensor(
            avoid_action, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        q_policy = _min_q(models['critic1'], models['critic2'], batch, policy_tensor)
        q_avoid = _min_q(models['critic1'], models['critic2'], batch, avoid_tensor)
        step_index = len(per_step)
        per_step.append({
            'step': step_index,
            'q_policy': q_policy,
            'q_avoid': q_avoid,
            'q_avoid_minus_q_policy': q_avoid - q_policy,
        })
        next_observation, reward, terminated, truncated, info = env.step(policy_action)
        done = bool(terminated or truncated)
        transitions.append({
            'step': step_index,
            'observation': observation,
            'action': policy_action,
            'reward': float(reward),
            'next_observation': next_observation,
            'done': done,
        })
        observation = next_observation
        if done:
            outcome = str(info.get('outcome', 'unknown'))

    terminal_step = len(per_step) - 1
    crossover_step = next(
        (item['step'] for item in per_step if item['q_avoid_minus_q_policy'] > 0.0),
        None,
    )
    lead_steps_before_terminal = (
        None if crossover_step is None else terminal_step - crossover_step
    )

    low, high = near_collision_step_range
    near_collision_transitions = [
        item for item in transitions
        if low <= (terminal_step - item['step']) <= high
    ]
    proxy_batch_occupancy_fraction = (
        len(near_collision_transitions) / len(transitions) if transitions else 0.0
    )
    gamma = models['gamma']
    td_errors: list[float] = []
    for item in near_collision_transitions:
        obs_batch = collate_v2_observations([item['observation']], device=device)
        next_obs_batch = collate_v2_observations(
            [item['next_observation']], device=device,
        )
        action_tensor = torch.as_tensor(
            item['action'], dtype=torch.float32, device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            current_q = _min_q(
                models['critic1'], models['critic2'], obs_batch, action_tensor,
            )
            next_action = models['actor_target'](next_obs_batch)
            target_q = _min_q(
                models['critic1_target'], models['critic2_target'],
                next_obs_batch, next_action,
            )
            bootstrapped = item['reward'] + (0.0 if item['done'] else gamma * target_q)
        td_errors.append(abs(current_q - bootstrapped))

    return {
        'purpose': 'discount_horizon_diagnostic',
        'measurement_note': (
            'proxy_batch_occupancy_fraction/td_error_* below use this '
            "scenario's own re-simulated transitions as a stand-in for a "
            'replay batch; no real replay buffer is persisted by any '
            'checkpoint in this project.'
        ),
        'checkpoint': str(checkpoint),
        'model_type': model_type,
        'gamma': gamma,
        'outcome': outcome,
        'episode_length': len(per_step),
        'terminal_step': terminal_step,
        'per_step': per_step,
        'crossover_step': crossover_step,
        'lead_steps_before_terminal': lead_steps_before_terminal,
        'near_collision_step_range': list(near_collision_step_range),
        'near_collision_transition_count': len(near_collision_transitions),
        'proxy_batch_occupancy_fraction': proxy_batch_occupancy_fraction,
        'td_error_near_collision': {
            'count': len(td_errors),
            'mean': float(np.mean(td_errors)) if td_errors else 0.0,
            'std': float(np.std(td_errors)) if td_errors else 0.0,
            'max': float(np.max(td_errors)) if td_errors else 0.0,
            'values': td_errors,
        },
    }


def _write_strict_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f'Output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Offline discount-horizon diagnostic (H2): compare Q(s, pi(s)) '
            'against Q(s, a_avoid) along a known-collision scenario.'
        )
    )
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--scenario', type=Path, required=True,
                         help='A JSON file containing one V2 scenario payload.')
    parser.add_argument('--model', choices=('ann', 'snn'), default='ann')
    parser.add_argument('--device', choices=DEVICE_CHOICES, default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--near-collision-min-steps', type=int, default=150)
    parser.add_argument('--near-collision-max-steps', type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_device = resolve_training_device(args.device)
    scenario_payload = json.loads(args.scenario.read_text(encoding='utf-8'))
    result = run_discount_horizon_diagnostic(
        checkpoint=args.checkpoint,
        scenario_payload=scenario_payload,
        model_type=args.model,
        device=resolved_device,
        near_collision_step_range=(
            args.near_collision_min_steps, args.near_collision_max_steps,
        ),
    )
    _write_strict_json(args.output, result)
    print(json.dumps({
        key: value for key, value in result.items() if key != 'per_step'
    }, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
