from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.envs import V2_ENV_SCENARIO_FORMAT, V2_ENV_SCENARIO_VERSION
from brain_uav.envs.v2_scenario_generator import (
    V2_SCENARIO_GENERATOR_NAME,
    V2_SCENARIO_GENERATOR_VERSION,
)
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.observations import V2Observation
from brain_uav.scripts.generate_v2_trajectory_clusters import (
    DEFAULT_V2_PLANNER_SPECS,
    V2_SCENARIO_POOL_VERSION,
    V2_TRAJECTORY_CLUSTER_VERSION,
    V2PlannerSpec,
    V2PlannerRolloutResult,
    _build_parser,
    collect_v2_planner_rollout,
    generate_easy_scenario_pool,
    generate_v2_trajectory_clusters,
    load_easy_scenario_pool,
    main as generate_clusters_main,
    save_easy_scenario_pool,
)
from brain_uav.scripts.run_v2_td3_curriculum import prepare_v2_validation_pools
from brain_uav.scripts.v2_trajectory_io import (
    V2_TRAJECTORY_SHARD_VERSION,
    V2TrajectoryShardBuffer,
    build_successful_v2_trajectory,
    load_v2_trajectory_shard,
)
from brain_uav.models import V2ANNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.trainers.v2_bc import (
    V2BCTrainingConfig,
    V2BCTrainingResult,
    build_v2_bc_checkpoint_payload,
    load_v2_bc_actor_checkpoint,
    load_v2_bc_trajectory_cluster,
    split_v2_bc_scenarios,
)
from brain_uav.trainers.v2_validation import (
    generate_v2_validation_pool,
    load_v2_validation_pool,
    save_v2_validation_pool,
)


class _ConstantPlanner:
    def __init__(self, env, action=(0.0, 0.0)) -> None:
        self.env = env
        self.action = np.asarray(action, dtype=np.float32)

    def act(self, observation: V2Observation) -> np.ndarray:
        del observation
        return self.action.copy()


def _config(**overrides) -> ScenarioConfig:
    values = dict(
        speed=1.0,
        dt=1.0,
        world_xy=30.0,
        world_z_min=0.1,
        world_z_max=25.0,
        max_steps=2,
        goal_radius=0.2,
    )
    values.update(overrides)
    return ScenarioConfig(**values)


def _payload(
    *,
    state=(-5.0, 0.0, 10.0, 0.0, 0.0),
    goal=(5.0, 0.0, 10.0),
    zones=(),
    scenario_seed=17,
    requested_direct_path_blocker=False,
    direct_path_blocker_count=None,
    feasibility_check='direct_safe_corridor',
) -> dict[str, object]:
    return {
        'format': V2_ENV_SCENARIO_FORMAT,
        'format_version': V2_ENV_SCENARIO_VERSION,
        'state': list(state),
        'goal': list(goal),
        'zones': [zone.to_dict() for zone in zones],
        'curriculum_level': 'easy',
        'metadata': {
            'generator': V2_SCENARIO_GENERATOR_NAME,
            'generator_version': V2_SCENARIO_GENERATOR_VERSION,
            'scenario_seed': scenario_seed,
            'requested_curriculum_level': 'easy',
            'effective_curriculum_level': 'easy',
            'requested_zone_count': len(zones),
            'effective_zone_count': len(zones),
            'requested_shape_types': [
                zone.shape.to_dict()['shape_type'] for zone in zones
            ],
            'requested_ground_contact': [
                bool(zone.metadata.get('ground_contact', False)) for zone in zones
            ],
            'shape_counts': {},
            'overlap_allowed': False,
            'aabb_overlap_pair_count': 0,
            'requested_direct_path_blocker': requested_direct_path_blocker,
            'direct_path_blocker_count': (
                (1 if requested_direct_path_blocker else 0)
                if direct_path_blocker_count is None
                else direct_path_blocker_count
            ),
            'feasibility_passed': True,
            'feasibility_check': feasibility_check,
        },
    }


def _normalized_config(config: ScenarioConfig) -> dict[str, object]:
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _pool(
    payloads,
    *,
    seed=7,
    scenario: ScenarioConfig | None = None,
    uav_collision_radius=0.0,
) -> dict[str, object]:
    scenario = scenario or _config()
    scenarios = []
    for index, payload in enumerate(payloads):
        scenarios.append(
            {
                'scenario_id': f'scenario_{index + 1:06d}',
                'sequence_index': index,
                'scenario_seed': payload['metadata']['scenario_seed'],
                'payload': payload,
            }
        )
    return {
        'format': 'v2_easy_training_scenario_pool',
        'format_version': V2_SCENARIO_POOL_VERSION,
        'master_seed': seed,
        'scenario_count': len(scenarios),
        'curriculum_level': 'easy',
        'scenario_config': _normalized_config(scenario),
        'uav_collision_radius': uav_collision_radius,
        'scenarios': scenarios,
    }


def _observation(zone_count: int, marker: float) -> V2Observation:
    return V2Observation(
        ego_features=np.full(6, marker, dtype=np.float32),
        goal_features=np.full(4, marker + 1.0, dtype=np.float32),
        zone_features=np.full((zone_count, 19), marker + 2.0, dtype=np.float32),
        presence_mask=np.ones(zone_count, dtype=np.bool_),
    )


class TestEasyScenarioPool(unittest.TestCase):
    def test_same_seed_produces_identical_ordered_easy_pool(self) -> None:
        first = generate_easy_scenario_pool(ScenarioConfig(), 3, seed=4401)
        second = generate_easy_scenario_pool(ScenarioConfig(), 3, seed=4401)
        self.assertEqual(first, second)
        self.assertEqual(
            [item['sequence_index'] for item in first['scenarios']],
            [0, 1, 2],
        )
        for item in first['scenarios']:
            self.assertEqual(item['payload']['curriculum_level'], 'easy')
            self.assertEqual(
                item['scenario_seed'],
                item['payload']['metadata']['scenario_seed'],
            )

    def test_pool_round_trip_is_strict_json_and_refuses_overwrite(self) -> None:
        scenario = _config(
            speed=3.25,
            max_steps=19,
            goal_radius=1.75,
            corridor_blocking_margin=0.5,
        )
        pool = _pool(
            [
                _payload(
                    zones=(
                        NoFlyZone('safe-side-zone', Sphere([0.0, 5.0, 10.0], 1.0)),
                    ),
                )
            ],
            scenario=scenario,
            uav_collision_radius=0.35,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pool.json'
            save_easy_scenario_pool(path, pool)
            loaded = load_easy_scenario_pool(path)
            self.assertEqual(loaded, pool)
            self.assertEqual(loaded['scenario_config'], _normalized_config(scenario))
            self.assertEqual(loaded['uav_collision_radius'], 0.35)
            with self.assertRaises(FileExistsError):
                save_easy_scenario_pool(path, pool)

    def test_version_one_pool_is_rejected(self) -> None:
        pool = _pool([_payload()])
        pool['format_version'] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.json'
            path.write_text(json.dumps(pool), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'version'):
                load_easy_scenario_pool(path)

    def test_latest_easy_generator_metadata_is_strictly_required(self) -> None:
        invalid_values = {
            'generator': 'old-generator',
            'generator_version': V2_SCENARIO_GENERATOR_VERSION + 1,
            'overlap_allowed': True,
            'aabb_overlap_pair_count': 1,
            'direct_path_blocker_count': 1,
            'feasibility_passed': False,
            'feasibility_check': 'unexpected_search',
        }
        for key, value in invalid_values.items():
            with self.subTest(field=key):
                payload = _payload()
                payload['metadata'][key] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'invalid.json'
                    with self.assertRaisesRegex(ValueError, key):
                        save_easy_scenario_pool(path, _pool([payload]))

    def test_generated_pool_captures_complete_scenario_config(self) -> None:
        scenario = replace(
            ScenarioConfig(),
            speed=3.0,
            dt=0.5,
            max_steps=23,
            goal_radius=1.25,
        )
        pool = generate_easy_scenario_pool(
            scenario,
            1,
            seed=4401,
            uav_collision_radius=0.4,
        )
        self.assertEqual(pool['scenario_config'], _normalized_config(scenario))
        self.assertEqual(pool['uav_collision_radius'], 0.4)

    def test_actual_direct_path_blocker_cannot_hide_behind_zero_metadata(self) -> None:
        scenario = _config(corridor_blocking_margin=0.5)
        payload = _payload(
            zones=(NoFlyZone('blocker', Sphere([0.0, 0.0, 10.0], 0.4)),),
        )
        self.assertEqual(payload['metadata']['direct_path_blocker_count'], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forged-blocker.json'
            with self.assertRaisesRegex(ValueError, 'actual direct path blocker'):
                save_easy_scenario_pool(
                    path,
                    _pool([payload], scenario=scenario),
                )
            self.assertFalse(path.exists())

    def test_blocked_easy_scenario_is_accepted_when_metadata_matches_geometry(self) -> None:
        scenario = _config(corridor_blocking_margin=0.5)
        payload = _payload(
            zones=(NoFlyZone('blocker', Sphere([0.0, 0.0, 10.0], 0.4)),),
            requested_direct_path_blocker=True,
            feasibility_check='sparse_visibility_graph',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'blocked.json'
            save_easy_scenario_pool(path, _pool([payload], scenario=scenario))
            loaded = load_easy_scenario_pool(path)
        self.assertEqual(
            loaded['scenarios'][0]['payload']['metadata'][
                'direct_path_blocker_count'
            ],
            1,
        )

    def test_blocked_easy_without_actual_blocker_is_rejected(self) -> None:
        scenario = _config(corridor_blocking_margin=0.5)
        payload = _payload(
            zones=(NoFlyZone('side-zone', Sphere([0.0, 5.0, 10.0], 1.0)),),
            requested_direct_path_blocker=True,
            feasibility_check='sparse_visibility_graph',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forged-blocked.json'
            with self.assertRaisesRegex(ValueError, 'actual direct path blocker'):
                save_easy_scenario_pool(path, _pool([payload], scenario=scenario))
            self.assertFalse(path.exists())

    def test_blocked_easy_cannot_claim_direct_safe_corridor(self) -> None:
        scenario = _config(corridor_blocking_margin=0.5)
        payload = _payload(
            zones=(NoFlyZone('blocker', Sphere([0.0, 0.0, 10.0], 0.4)),),
            requested_direct_path_blocker=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad-feasibility.json'
            with self.assertRaisesRegex(
                ValueError, 'direct_safe_corridor feasibility_check'
            ):
                save_easy_scenario_pool(path, _pool([payload], scenario=scenario))
            self.assertFalse(path.exists())

    def test_actual_aabb_overlap_cannot_hide_behind_zero_metadata(self) -> None:
        scenario = _config(corridor_blocking_margin=0.5)
        payload = _payload(
            zones=(
                NoFlyZone('first', Sphere([0.0, 5.0, 10.0], 1.0)),
                NoFlyZone('second', Sphere([1.0, 5.0, 10.0], 1.0)),
            ),
        )
        self.assertEqual(payload['metadata']['aabb_overlap_pair_count'], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'forged-overlap.json'
            path.write_text(
                json.dumps(_pool([payload], scenario=scenario), allow_nan=False),
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'actual AABB overlap'):
                load_easy_scenario_pool(path)

    def test_pool_rejects_non_easy_payloads(self) -> None:
        payload = _payload()
        payload['curriculum_level'] = 'medium'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(_pool([payload])), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'easy'):
                load_easy_scenario_pool(path)


class TestV2RaggedTrajectoryShard(unittest.TestCase):
    def test_multiple_trajectories_round_trip_with_cross_trajectory_offsets(self) -> None:
        first = build_successful_v2_trajectory(
            trajectory_id='trajectory-1',
            scenario_id='scenario_000001',
            planner_name='first',
            scenario_seed=17,
            observations=[
                _observation(0, 1.0),
                _observation(1, 2.0),
                _observation(2, 3.0),
            ],
            states_before_action=np.arange(15, dtype=np.float32).reshape(3, 5),
            actions=np.zeros((3, 2), dtype=np.float32),
            terminal_state=np.ones(5, dtype=np.float32),
            outcome='goal',
        )
        second = build_successful_v2_trajectory(
            trajectory_id='trajectory-2',
            scenario_id='scenario_000002',
            planner_name='second',
            scenario_seed=18,
            observations=[_observation(2, 4.0), _observation(0, 5.0)],
            states_before_action=np.arange(10, dtype=np.float32).reshape(2, 5),
            actions=np.ones((2, 2), dtype=np.float32),
            terminal_state=np.full(5, 2.0, dtype=np.float32),
            outcome='goal',
        )
        buffer = V2TrajectoryShardBuffer()
        buffer.add(first)
        buffer.add(second)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shard_0001.npz'
            summary = buffer.flush(path)
            arrays = load_v2_trajectory_shard(path)

        np.testing.assert_array_equal(arrays['zone_offsets'], [0, 0, 1, 3, 5, 5])
        np.testing.assert_array_equal(arrays['trajectory_offsets'], [0, 3, 5])
        np.testing.assert_array_equal(arrays['scenario_ids'], ['scenario_000001', 'scenario_000002'])
        np.testing.assert_array_equal(arrays['planner_names'], ['first', 'second'])
        np.testing.assert_array_equal(arrays['step_counts'], [3, 2])
        self.assertEqual(arrays['zone_features'].shape, (5, 19))
        self.assertEqual(arrays['ego_features'].shape, (5, 6))
        self.assertEqual(arrays['goal_features'].shape, (5, 4))
        self.assertEqual(arrays['actions'].shape, (5, 2))
        self.assertNotIn('minimum_segment_clearance_values', arrays)
        self.assertNotIn('minimum_segment_clearance_present', arrays)
        self.assertEqual(int(arrays['format_version']), V2_TRAJECTORY_SHARD_VERSION)
        self.assertEqual(summary['trajectory_count'], 2)
        self.assertEqual(buffer.trajectory_count, 0)
        self.assertEqual(buffer.step_count, 0)

    def test_only_goal_trajectories_can_enter_a_shard(self) -> None:
        with self.assertRaisesRegex(ValueError, 'goal'):
            build_successful_v2_trajectory(
                trajectory_id='failed',
                scenario_id='scenario_000001',
                planner_name='test',
                scenario_seed=17,
                observations=[_observation(0, 1.0)],
                states_before_action=np.zeros((1, 5), dtype=np.float32),
                actions=np.zeros((1, 2), dtype=np.float32),
                terminal_state=np.zeros(5, dtype=np.float32),
                outcome='collision',
            )


class TestV2PlannerRollout(unittest.TestCase):
    def test_same_payload_runs_both_real_v2_experts(self) -> None:
        scenario = _config(goal_radius=3.0, max_steps=1)
        payload = _payload(
            state=(0.0, 0.0, 10.0, 0.0, 0.0),
            goal=(1.0, 0.0, 10.0),
        )
        results = [
            collect_v2_planner_rollout(
                payload,
                scenario_id='scenario_000001',
                trajectory_id=f'trajectory-{index}',
                planner_spec=spec,
                scenario=scenario,
                rewards=RewardConfig(),
            )
            for index, spec in enumerate(DEFAULT_V2_PLANNER_SPECS)
        ]
        self.assertEqual(
            [result.planner_name for result in results],
            ['v2_heuristic', 'v2_apf'],
        )
        self.assertTrue(all(result.trajectory is not None for result in results))

    def test_goal_is_saved_and_executed_action_is_clipped(self) -> None:
        scenario = _config(goal_radius=3.0, max_steps=1)
        payload = _payload(state=(0.0, 0.0, 10.0, 0.0, 0.0), goal=(1.0, 0.0, 10.0))
        spec = V2PlannerSpec('constant', lambda env: _ConstantPlanner(env, (99.0, -99.0)))
        result = collect_v2_planner_rollout(
            payload,
            scenario_id='scenario_000001',
            trajectory_id='trajectory-1',
            planner_spec=spec,
            scenario=scenario,
            rewards=RewardConfig(),
        )
        self.assertEqual(result.outcome, 'goal')
        self.assertIsNotNone(result.trajectory)
        self.assertFalse(hasattr(result, 'minimum_segment_clearance'))
        self.assertFalse(hasattr(result.trajectory, 'minimum_segment_clearance'))
        self.assertNotIn('minimum_segment_clearance', result.summary())
        np.testing.assert_allclose(
            result.trajectory.actions[0],
            [scenario.delta_gamma_max, -scenario.delta_psi_max],
        )

    def test_non_goal_outcomes_only_return_failure_summaries(self) -> None:
        cases = {
            'timeout': (
                _config(max_steps=1),
                _payload(),
            ),
            'collision': (
                _config(
                    speed=2.0,
                    max_steps=2,
                    corridor_blocking_margin=0.5,
                ),
                _payload(
                    state=(0.0, 0.0, 10.0, 0.0, np.pi / 2.0),
                    goal=(10.0, 0.0, 10.0),
                    zones=(NoFlyZone('side-zone', Sphere([0.0, 1.0, 10.0], 0.4)),),
                ),
            ),
            'ground': (
                _config(speed=2.0, max_steps=2),
                _payload(state=(0.0, 0.0, 0.2, -0.6, 0.0), goal=(10.0, 0.0, 5.0)),
            ),
            'boundary': (
                _config(speed=2.0, world_xy=1.0, max_steps=2),
                _payload(state=(0.8, 0.0, 10.0, 0.0, 0.0), goal=(-0.8, 0.0, 10.0)),
            ),
        }
        spec = V2PlannerSpec('constant', lambda env: _ConstantPlanner(env))
        for expected, (scenario, payload) in cases.items():
            with self.subTest(outcome=expected):
                result = collect_v2_planner_rollout(
                    payload,
                    scenario_id='scenario_000001',
                    trajectory_id='trajectory-1',
                    planner_spec=spec,
                    scenario=scenario,
                    rewards=RewardConfig(),
                )
                self.assertEqual(result.outcome, expected)
                self.assertIsNone(result.trajectory)

    def test_same_payload_is_reset_independently_for_both_planners(self) -> None:
        captures = {}

        def factory(name):
            def create(env):
                captures[name] = (
                    env.state.copy(),
                    env.goal.copy(),
                    [zone.to_dict() for zone in env.zones],
                )
                return _ConstantPlanner(env)

            return create

        payload = _payload(zones=(NoFlyZone('far', Sphere([0.0, 8.0, 10.0], 1.0)),))
        scenario = _config(max_steps=1, corridor_blocking_margin=0.5)
        for name in ('first', 'second'):
            collect_v2_planner_rollout(
                payload,
                scenario_id='scenario_000001',
                trajectory_id=f'trajectory-{name}',
                planner_spec=V2PlannerSpec(name, factory(name)),
                scenario=scenario,
                rewards=RewardConfig(),
            )
        np.testing.assert_array_equal(captures['first'][0], captures['second'][0])
        np.testing.assert_array_equal(captures['first'][1], captures['second'][1])
        self.assertEqual(captures['first'][2], captures['second'][2])


class TestClusterOrchestration(unittest.TestCase):
    def test_world_z_max_cli_snapshot_and_downstream_config_validation(self) -> None:
        required_args = ['--output-dir', 'unused', '--scenario-count', '1']
        defaults = _build_parser().parse_args(required_args)
        self.assertIsNone(defaults.world_z_max)
        self.assertEqual(defaults.heuristic_repulsive_gain, 3.0)
        self.assertEqual(defaults.heuristic_influence_margin, 4.0)
        self.assertEqual(defaults.apf_repulsive_gain, 5000.0)
        self.assertEqual(defaults.apf_influence_margin, 6.0)
        self.assertEqual(
            _build_parser().parse_args(required_args + ['--world-z-max', '600']).world_z_max,
            600.0,
        )

        def fake_rollout(
            payload,
            *,
            scenario_id,
            trajectory_id,
            planner_spec,
            **_kwargs,
        ):
            trajectory = build_successful_v2_trajectory(
                trajectory_id=trajectory_id,
                scenario_id=scenario_id,
                planner_name=planner_spec.name,
                scenario_seed=payload['metadata']['scenario_seed'],
                observations=[_observation(len(payload['zones']), 1.0)],
                states_before_action=np.asarray(payload['state'], dtype=np.float32)[None, :],
                actions=np.zeros((1, 2), dtype=np.float32),
                terminal_state=np.asarray(payload['state'], dtype=np.float32),
                outcome='goal',
            )
            return V2PlannerRolloutResult(
                scenario_id=scenario_id,
                planner_name=planner_spec.name,
                outcome='goal',
                step_count=1,
                trajectory=trajectory,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'clusters'
            with mock.patch(
                'brain_uav.scripts.generate_v2_trajectory_clusters.collect_v2_planner_rollout',
                side_effect=fake_rollout,
            ):
                generate_clusters_main(
                    [
                        '--output-dir', str(output),
                        '--scenario-count', '2',
                        '--seed', '717',
                        '--shard-size', '1',
                        '--world-z-max', '600',
                        '--apf-repulsive-gain', '1e7',
                        '--apf-influence-margin', '6',
                    ]
                )

            pool = json.loads((output / 'scenario_pool.json').read_text(encoding='utf-8'))
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(pool['scenario_config']['world_z_max'], 600.0)
            self.assertEqual(manifest['scenario_config'], pool['scenario_config'])
            self.assertEqual(
                manifest['statistics']['planners']['v2_heuristic']['parameters'],
                {'repulsive_gain': 3.0, 'influence_margin': 4.0},
            )
            self.assertEqual(
                manifest['statistics']['planners']['v2_apf']['parameters'],
                {
                    'attractive_gain': 1.0,
                    'repulsive_gain': 1e7,
                    'influence_margin': 6.0,
                },
            )

            cluster = load_v2_bc_trajectory_cluster(output)
            self.assertEqual(cluster.scenario_config.world_z_max, 600.0)

            scenario = cluster.scenario_config
            actor = V2ANNPolicyActor(
                V2ObservationScales(
                    world_xy=float(scenario.world_xy),
                    world_z_min=float(scenario.world_z_min),
                    world_z_max=float(scenario.world_z_max),
                    gamma_max=float(scenario.gamma_max),
                ),
                action_dim=2,
                hidden_dim=8,
                action_limit=torch.tensor(
                    [scenario.delta_gamma_max, scenario.delta_psi_max],
                    dtype=torch.float32,
                ),
            )
            state_dict = actor.state_dict()
            training_config = V2BCTrainingConfig(
                epochs=1,
                batch_size=1,
                learning_rate=1e-3,
                seed=717,
                validation_fraction=0.5,
            )
            checkpoint_payload = build_v2_bc_checkpoint_payload(
                checkpoint_kind='best',
                actor=actor,
                actor_state_dict=state_dict,
                cluster=cluster,
                split=split_v2_bc_scenarios(
                    cluster, validation_fraction=0.5, seed=717
                ),
                config=training_config,
                result=V2BCTrainingResult(
                    train_loss_history=(0.2,),
                    validation_loss_history=(0.1,),
                    best_epoch=1,
                    best_validation_loss=0.1,
                    best_state_dict=state_dict,
                    final_state_dict=state_dict,
                ),
                finished_at='2026-09-25T00:00:00Z',
            )
            self.assertEqual(
                checkpoint_payload['dataset_provenance']['scenario_config'],
                pool['scenario_config'],
            )
            checkpoint_path = root / 'bc_best.pt'
            torch.save(checkpoint_payload, checkpoint_path)
            loaded_actor = load_v2_bc_actor_checkpoint(checkpoint_path)
            self.assertEqual(loaded_actor.scales.world_z_max, 600.0)

            validation_dir = root / 'validation'
            validation_paths = prepare_v2_validation_pools(
                validation_dir,
                cluster.scenario_config,
                validation_seed=901,
                scenario_count=1,
            )
            for stage, path in validation_paths.items():
                loaded = load_v2_validation_pool(
                    path,
                    expected_level=stage,
                    expected_scenario=cluster.scenario_config,
                    expected_count=1,
                    expected_master_seed=901,
                )
                self.assertEqual(loaded.scenario_config['world_z_max'], 600.0)

            legacy_path = root / 'legacy_validation.json'
            legacy_pool = generate_v2_validation_pool(
                ScenarioConfig(),
                'easy',
                scenario_count=1,
                master_seed=902,
            )
            save_v2_validation_pool(legacy_path, legacy_pool)
            with self.assertRaisesRegex(ValueError, 'ScenarioConfig is incompatible'):
                load_v2_validation_pool(
                    legacy_path,
                    expected_level='easy',
                    expected_scenario=cluster.scenario_config,
                    expected_count=1,
                    expected_master_seed=902,
                )

            legacy_scenario_pool_path = root / 'legacy_scenario_pool.json'
            save_easy_scenario_pool(
                legacy_scenario_pool_path,
                generate_easy_scenario_pool(ScenarioConfig(), 1, seed=903),
            )
            mismatched_output = root / 'mismatched_clusters'
            with self.assertRaisesRegex(ValueError, 'ScenarioConfig does not match'):
                generate_clusters_main(
                    [
                        '--output-dir', str(mismatched_output),
                        '--scenario-count', '1',
                        '--scenario-pool', str(legacy_scenario_pool_path),
                        '--world-z-max', '600',
                    ]
                )
            self.assertFalse(mismatched_output.exists())

    def test_loaded_pool_uses_its_saved_scenario_config_and_radius(self) -> None:
        scenario = _config(
            speed=2.0,
            max_steps=1,
            goal_radius=0.5,
            corridor_blocking_margin=0.5,
        )
        payload = _payload(
            state=(0.0, 0.0, 10.0, 0.0, 0.0),
            goal=(2.0, 0.0, 10.0),
        )
        pool = _pool(
            [payload],
            scenario=scenario,
            uav_collision_radius=0.25,
        )
        captured = []

        def capture_environment(env):
            captured.append((env.scenario, env.uav_collision_radius))
            return _ConstantPlanner(env)

        specs = (
            V2PlannerSpec('first', capture_environment),
            V2PlannerSpec('second', capture_environment),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / 'input_pool.json'
            save_easy_scenario_pool(pool_path, pool)
            manifest = generate_v2_trajectory_clusters(
                output_dir=root / 'output',
                scenario_count=1,
                seed=999,
                shard_size=1,
                scenario_pool_path=pool_path,
                planner_specs=specs,
            )
        self.assertEqual(manifest['scenario_config'], _normalized_config(scenario))
        self.assertEqual(manifest['uav_collision_radius'], 0.25)
        self.assertEqual(manifest['format_version'], V2_TRAJECTORY_CLUSTER_VERSION)
        self.assertEqual(len(captured), 2)
        for effective_scenario, effective_radius in captured:
            self.assertEqual(_normalized_config(effective_scenario), _normalized_config(scenario))
            self.assertEqual(effective_radius, 0.25)

    def test_loaded_pool_rejects_explicit_scenario_config_mismatch(self) -> None:
        pool_scenario = _config(speed=2.0, max_steps=3, goal_radius=0.5)
        pool = _pool([_payload()], scenario=pool_scenario)
        mismatches = (
            replace(pool_scenario, speed=3.0),
            replace(pool_scenario, max_steps=4),
            replace(pool_scenario, goal_radius=0.75),
            replace(pool_scenario, world_z_max=600.0),
        )
        for index, mismatch in enumerate(mismatches):
            with self.subTest(config=mismatch):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    pool_path = root / 'input_pool.json'
                    save_easy_scenario_pool(pool_path, pool)
                    with self.assertRaisesRegex(ValueError, 'ScenarioConfig'):
                        generate_v2_trajectory_clusters(
                            output_dir=root / f'output-{index}',
                            scenario_count=1,
                            seed=7,
                            shard_size=1,
                            scenario_pool_path=pool_path,
                            scenario=mismatch,
                        )

    def test_loaded_pool_rejects_explicit_collision_radius_mismatch(self) -> None:
        scenario = _config()
        pool = _pool(
            [_payload()],
            scenario=scenario,
            uav_collision_radius=0.25,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / 'input_pool.json'
            save_easy_scenario_pool(pool_path, pool)
            with self.assertRaisesRegex(ValueError, 'uav_collision_radius'):
                generate_v2_trajectory_clusters(
                    output_dir=root / 'output',
                    scenario_count=1,
                    seed=7,
                    shard_size=1,
                    scenario_pool_path=pool_path,
                    scenario=scenario,
                    uav_collision_radius=0.5,
                )

    def test_zero_success_raises_without_fallback_and_records_failures(self) -> None:
        scenario = _config(max_steps=1)
        pool = _pool([_payload()], scenario=scenario)
        specs = (
            V2PlannerSpec('first', lambda env: _ConstantPlanner(env)),
            V2PlannerSpec('second', lambda env: _ConstantPlanner(env)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / 'input_pool.json'
            output_dir = root / 'output'
            save_easy_scenario_pool(pool_path, pool)
            with self.assertRaisesRegex(RuntimeError, 'zero successful'):
                generate_v2_trajectory_clusters(
                    output_dir=output_dir,
                    scenario_count=1,
                    seed=7,
                    shard_size=1,
                    scenario_pool_path=pool_path,
                    scenario=scenario,
                    rewards=RewardConfig(),
                    planner_specs=specs,
                )
            manifest = json.loads((output_dir / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['statistics']['successful_trajectories'], 0)
            self.assertEqual(manifest['statistics']['planners']['first']['timeout'], 1)
            self.assertEqual(manifest['statistics']['planners']['second']['timeout'], 1)
            self.assertEqual(manifest['shards'], [])

    def test_successes_are_sharded_and_failures_are_only_counted(self) -> None:
        scenario = _config(
            speed=2.0,
            max_steps=1,
            goal_radius=0.5,
            corridor_blocking_margin=0.5,
        )
        goal_payload = _payload(
            state=(0.0, 0.0, 10.0, 0.0, 0.0),
            goal=(2.0, 0.0, 10.0),
            scenario_seed=1,
        )
        collision_payload = _payload(
            state=(0.0, 0.0, 10.0, 0.0, np.pi / 2.0),
            goal=(10.0, 0.0, 10.0),
            zones=(NoFlyZone('side-zone', Sphere([0.0, 1.0, 10.0], 0.4)),),
            scenario_seed=2,
        )
        pool = _pool([goal_payload, collision_payload], scenario=scenario)
        specs = (
            V2PlannerSpec('first', lambda env: _ConstantPlanner(env)),
            V2PlannerSpec('second', lambda env: _ConstantPlanner(env)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool_path = root / 'input_pool.json'
            output_dir = root / 'output'
            save_easy_scenario_pool(pool_path, pool)
            manifest = generate_v2_trajectory_clusters(
                output_dir=output_dir,
                scenario_count=2,
                seed=7,
                shard_size=1,
                scenario_pool_path=pool_path,
                scenario=scenario,
                rewards=RewardConfig(),
                planner_specs=specs,
            )
            self.assertEqual(manifest['statistics']['successful_trajectories'], 2)
            self.assertEqual(manifest['statistics']['planners']['first']['collision'], 1)
            self.assertEqual(manifest['statistics']['planners']['second']['collision'], 1)
            arrays = load_v2_trajectory_shard(output_dir / manifest['shards'][0]['file'])
            self.assertTrue(np.all(arrays['outcomes'] == 'goal'))
            self.assertNotIn('minimum_segment_clearance_values', arrays)
            self.assertNotIn('minimum_segment_clearance_present', arrays)
            serialized = json.dumps(manifest, allow_nan=False)
            self.assertNotIn('minimum_segment_clearance', serialized)


if __name__ == '__main__':
    unittest.main()
