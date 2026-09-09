"""Bounded ANN/SNN TD3 diagnostic; never a formal stage or convergence test.

Run from the project root with PYTHONPATH=src. CUDA is mandatory by default.
An explicit --device cpu is only for checking this script locally.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
import random
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from brain_uav.envs import V2_ENV_SCENARIO_FORMAT, V2_ENV_SCENARIO_VERSION
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig, V2FormalStageTrainer,
    prepare_v2_stage_initialization, build_v2_stage_engine,
)
from brain_uav.trainers.v2_replay_buffer import V2ReplayBuffer
from brain_uav.trainers.v2_validation import (
    generate_v2_validation_pool, save_v2_validation_pool, evaluate_v2_fixed_validation,
)


def check_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert first.dtype == second.dtype and first.shape == second.shape
        assert torch.equal(first.cpu(), second.cpu()), 'Tensor state changed'
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys(), 'State keys changed'
        for key in first:
            check_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            check_equal(a, b)
    else:
        assert first == second, 'State value changed'


def require_finite(value):
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor'
    elif isinstance(value, dict):
        for child in value.values():
            require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            require_finite(child)


def no_candidate(_actor):
    raise AssertionError('A bounded diagnostic must not trigger formal promotion')


def network_snapshot(engine):
    return {name: {key: value.detach().cpu().clone()
                   for key, value in getattr(engine, name).named_parameters()}
            for name in ('actor', 'critic1', 'critic2', 'actor_target',
                         'critic1_target', 'critic2_target')}


def check_snn_reset(engine):
    if engine.model_type == 'snn':
        for actor in (engine.actor, engine.actor_target, engine.bc_reference_actor):
            for lif in (actor.snn_head.lif1, actor.snn_head.lif2):
                assert bool(torch.as_tensor(lif.v).eq(0).all()), 'SNN memory was not reset'


def check_success_episode(engine, prepared, config):
    """Exercise the REAL formal loop with a deterministic three-step fixture.

    Only the action selector is prescribed here. A real environment must report
    goal; this fixture says nothing about the learned policy's success rate.
    """
    scenario = replace(prepared.scenario_config, max_steps=3)
    distance = scenario.goal_radius + 2.5 * scenario.speed * scenario.dt
    assert distance / 2 < scenario.world_xy, 'Three-step fixture does not fit world'
    z = (scenario.world_z_min + scenario.world_z_max) / 2
    payload = {'format': V2_ENV_SCENARIO_FORMAT,
               'format_version': V2_ENV_SCENARIO_VERSION,
               'state': [-distance / 2, 0, z, 0, 0],
               'goal': [distance / 2, 0, z], 'zones': [], 'curriculum_level': 'easy'}

    class Source:
        def generate(self):
            return deepcopy(payload)

    original_replay = engine.replay
    old_noise = (engine.policy_noise, engine.noise_clip)
    updates_before = engine.update_count
    engine.replay = V2ReplayBuffer(256, 2, 6, seed=7)
    try:
        with patch.object(engine, 'select_action', return_value=np.zeros(2, dtype=np.float32)):
            result = V2FormalStageTrainer(
                scenario, prepared.reward_config, replace(config, max_steps=3), engine,
                scenario_sources={'easy': Source()}, validation_runner=no_candidate,
                uav_collision_radius=prepared.uav_collision_radius,
            ).run()
        assert result.outcome_counts['goal'] == 1
        assert result.episodes[0]['episode_length'] == 3
        assert engine.replay.success_size == 3 and engine.replay.success_count == 3
        assert engine.replay.success[:3].all()
        np.testing.assert_array_equal(engine.replay.success_done[:3, 0], [0, 0, 1])
        np.testing.assert_array_equal(engine.replay.success_ego_features[:3], engine.replay.ego_features[:3])
        assert engine.update_count == updates_before
    finally:
        engine.replay = original_replay
        engine.set_target_noise(policy_noise=old_noise[0], noise_clip=old_noise[1])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', choices=('ann', 'snn'), required=True)
    result.add_argument('--bc-checkpoint', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--snn-time-window', type=int, default=4)
    return result


def main(argv=None):
    if not __debug__:
        raise RuntimeError('Run without python -O: diagnostic assertions must remain enabled')
    args = parser().parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    if args.output_dir.exists():
        raise FileExistsError(f'Use a fresh diagnostic directory: {args.output_dir}')
    started = perf_counter()
    config = V2FormalTrainingConfig(stage='easy', seed=7, max_steps=72,
                                   replay_capacity=256, batch_size=64, actor_freeze_steps=0)
    print(f'[1/5] loading {args.model} BC best on {args.device}', flush=True)
    prepared = prepare_v2_stage_initialization(
        config, init_checkpoint=args.bc_checkpoint, device=args.device,
        model_type=args.model, snn_time_window=args.snn_time_window,
    )
    def build():
        return build_v2_stage_engine(
            None, config, init_checkpoint=args.bc_checkpoint, device=args.device,
            model_type=args.model, snn_time_window=args.snn_time_window,
            prepared_initialization=prepared,
        )
    components = build()
    engine = components.engine
    actual_devices = {p.device.type for p in engine.actor.parameters()}
    assert actual_devices == {args.device}
    args.output_dir.mkdir(parents=True, exist_ok=False)

    print('[2/5] checking complete three-step success replay (prescribed-action fixture)', flush=True)
    check_success_episode(engine, prepared, config)
    before = network_snapshot(engine)
    print('[3/5] running 72 real environment steps and CUDA/CPU TD3 updates', flush=True)
    trainer = V2FormalStageTrainer(
        prepared.scenario_config, prepared.reward_config, config, engine,
        scenario_sources=components.scenario_generators, validation_runner=no_candidate,
        selector=components.selector, exploration_rng=components.exploration_rng,
        uav_collision_radius=prepared.uav_collision_radius,
    )
    result = trainer.run()
    assert result.stage_steps == 72 and engine.critic_update_count == 9
    assert engine.actor_update_count > 0 and engine.critic_target_update_count > 0
    after = network_snapshot(engine)
    for name in before:
        assert any(not torch.equal(before[name][key], after[name][key]) for key in before[name]), name + ' did not update'
    payload = deepcopy(engine.checkpoint_state_dict())
    require_finite(payload)
    check_snn_reset(engine)

    print('[4/5] saving and reloading diagnostic engine snapshot', flush=True)
    artifact = args.output_dir / 'diagnostic_engine.pt'
    torch.save({'format': 'uav_td3_precheck_only', 'formal_stage_passed': False,
                'engine': payload}, artifact)
    loaded = torch.load(artifact, map_location=args.device, weights_only=False)
    assert loaded['format'] == 'uav_td3_precheck_only' and loaded['formal_stage_passed'] is False
    restored = build().engine
    restored.set_target_noise(policy_noise=engine.policy_noise, noise_clip=engine.noise_clip)
    restored.load_checkpoint_state_dict(loaded['engine'])
    check_equal(payload, restored.checkpoint_state_dict())
    # Also exercise six-network loading, without asserting formal stage promotion.
    restored.load_network_state_dicts(loaded['engine'])
    check_equal(payload, restored.checkpoint_state_dict())
    check_snn_reset(restored)
    del restored

    print('[5/5] evaluating one independently seeded fixed scene; no success threshold required', flush=True)
    pool = generate_v2_validation_pool(prepared.scenario_config, 'easy', scenario_count=1,
                                      master_seed=20260904,
                                      uav_collision_radius=prepared.uav_collision_radius)
    save_v2_validation_pool(args.output_dir / 'validation_pool.json', pool)
    snapshot = deepcopy(engine.checkpoint_state_dict())
    replay_before = pickle.dumps(engine.replay, protocol=5)
    result_before = deepcopy(result.to_dict())
    mode_before = engine.actor.training
    rng_before = (torch.get_rng_state().clone(),
                  torch.cuda.get_rng_state_all() if args.device == 'cuda' else [],
                  deepcopy(np.random.get_state()), random.getstate(),
                  deepcopy(components.exploration_rng.bit_generator.state))
    validation = evaluate_v2_fixed_validation(engine.actor, pool, prepared.reward_config,
                                             device=args.device, max_failures=0)
    check_equal(snapshot, engine.checkpoint_state_dict())
    assert replay_before == pickle.dumps(engine.replay, protocol=5), 'Validation changed replay'
    check_equal(result_before, result.to_dict())
    assert mode_before == engine.actor.training
    check_equal(rng_before, (torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if args.device == 'cuda' else [],
                np.random.get_state(), random.getstate(), components.exploration_rng.bit_generator.state))
    check_snn_reset(engine)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    summary = {'format': 'uav_td3_precheck_summary', 'status': 'passed',
               'formal_stage_passed': False, 'device': args.device, 'model': args.model,
               'bc_checkpoint': str(args.bc_checkpoint.resolve()), 'config': config.to_dict(),
               'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
               'environment_steps': result.stage_steps, 'critic_updates': engine.critic_update_count,
               'actor_updates': engine.actor_update_count, 'success_fixture_transitions': 3,
               'validation_pool_digest': pool.content_digest,
               'validation': validation.to_dict(), 'elapsed_seconds': perf_counter() - started}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


if __name__ == '__main__':
    main()
