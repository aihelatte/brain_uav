from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brain_uav.config import ScenarioConfig
from brain_uav.envs import V2_ENV_SCENARIO_FORMAT, V2_ENV_SCENARIO_VERSION
from brain_uav.envs.v2_scenario_generator import (
    V2_SCENARIO_GENERATOR_NAME,
    V2_SCENARIO_GENERATOR_VERSION,
)
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.models import V2ANNPolicyActor
from brain_uav.observations import (
    V2Observation,
    V2ObservationBatch,
    V2ObservationScales,
    collate_v2_observations,
)
from brain_uav.scripts.generate_v2_trajectory_clusters import (
    V2_SCENARIO_POOL_FORMAT,
    V2_SCENARIO_POOL_VERSION,
    V2_TRAJECTORY_CLUSTER_FORMAT,
    V2_TRAJECTORY_CLUSTER_VERSION,
    save_easy_scenario_pool,
)
from brain_uav.scripts.v2_trajectory_io import (
    V2TrajectoryShardBuffer,
    build_successful_v2_trajectory,
    load_v2_trajectory_shard,
)
from brain_uav.trainers.v2_bc import (
    V2ShardArrayCache,
    ValidationBestTracker,
    assemble_v2_bc_batch,
    iter_v2_bc_batches,
    load_v2_bc_trajectory_cluster,
    restore_v2_observation,
    split_v2_bc_scenarios,
)


def make_scenario_config() -> ScenarioConfig:
    return ScenarioConfig(
        speed=1.0,
        dt=1.0,
        world_xy=50.0,
        world_z_min=0.1,
        world_z_max=30.0,
        max_steps=10,
        goal_radius=1.0,
        corridor_blocking_margin=0.25,
    )


def normalised_config(config: ScenarioConfig) -> dict[str, object]:
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def make_scenario_payload(index: int, zone_count: int) -> dict[str, object]:
    zones = [
        NoFlyZone(
            f'zone-{index}-{zone_index}',
            Sphere([(-10.0 + 20.0 * zone_index), 10.0, 10.0], 1.0),
            metadata={'ground_contact': False},
        )
        for zone_index in range(zone_count)
    ]
    return {
        'format': V2_ENV_SCENARIO_FORMAT,
        'format_version': V2_ENV_SCENARIO_VERSION,
        'state': [-5.0, 0.0, 10.0, 0.0, 0.0],
        'goal': [5.0, 0.0, 10.0],
        'zones': [zone.to_dict() for zone in zones],
        'curriculum_level': 'easy',
        'metadata': {
            'generator': V2_SCENARIO_GENERATOR_NAME,
            'generator_version': V2_SCENARIO_GENERATOR_VERSION,
            'scenario_seed': 1000 + index,
            'requested_curriculum_level': 'easy',
            'effective_curriculum_level': 'easy',
            'requested_zone_count': zone_count,
            'effective_zone_count': zone_count,
            'requested_shape_types': ['sphere'] * zone_count,
            'requested_ground_contact': [False] * zone_count,
            'shape_counts': {'sphere': zone_count},
            'overlap_allowed': False,
            'aabb_overlap_pair_count': 0,
            'direct_path_blocker_count': 0,
            'feasibility_passed': True,
            'feasibility_check': 'direct_safe_corridor',
        },
    }


def make_observation(zone_count: int, marker: float) -> V2Observation:
    zones = np.empty((zone_count, 19), dtype=np.float32)
    for zone_index in range(zone_count):
        zones[zone_index] = marker + zone_index + np.arange(19, dtype=np.float32) / 100.0
    return V2Observation(
        ego_features=np.full(6, marker, dtype=np.float32),
        goal_features=np.full(4, marker + 0.5, dtype=np.float32),
        zone_features=zones,
        presence_mask=np.ones(zone_count, dtype=np.bool_),
    )


def make_trajectory(
    trajectory_id: str,
    scenario_id: str,
    planner_name: str,
    scenario_seed: int,
    zone_count: int,
    step_count: int,
):
    return build_successful_v2_trajectory(
        trajectory_id=trajectory_id,
        scenario_id=scenario_id,
        planner_name=planner_name,
        scenario_seed=scenario_seed,
        observations=[make_observation(zone_count, float(index + 1)) for index in range(step_count)],
        states_before_action=np.zeros((step_count, 5), dtype=np.float32),
        actions=np.tile(np.asarray([0.01, -0.02], dtype=np.float32), (step_count, 1)),
        terminal_state=np.zeros(5, dtype=np.float32),
        outcome='goal',
    )


def write_cluster(
    root: Path,
    *,
    zone_counts=(0, 1, 2, 0, 1),
    trajectories_per_scenario=2,
    shard_scenario_count=2,
    scenario_config: ScenarioConfig | None = None,
    uav_collision_radius: float = 0.0,
) -> Path:
    config = make_scenario_config() if scenario_config is None else scenario_config
    scenarios = []
    for index, zone_count in enumerate(zone_counts):
        payload = make_scenario_payload(index, zone_count)
        scenarios.append({
            'scenario_id': f'scenario_{index + 1:06d}',
            'sequence_index': index,
            'scenario_seed': payload['metadata']['scenario_seed'],
            'payload': payload,
        })
    pool = {
        'format': V2_SCENARIO_POOL_FORMAT,
        'format_version': V2_SCENARIO_POOL_VERSION,
        'master_seed': 77,
        'scenario_count': len(scenarios),
        'curriculum_level': 'easy',
        'scenario_config': normalised_config(config),
        'uav_collision_radius': uav_collision_radius,
        'scenarios': scenarios,
    }
    save_easy_scenario_pool(root / 'scenario_pool.json', pool)

    shards = []
    trajectory_number = 1
    successful_count = 0
    for shard_number, start in enumerate(range(0, len(scenarios), shard_scenario_count), 1):
        end = min(start + shard_scenario_count, len(scenarios))
        buffer = V2TrajectoryShardBuffer()
        for scenario_index in range(start, end):
            item = scenarios[scenario_index]
            for planner_index in range(trajectories_per_scenario):
                planner = 'v2_heuristic' if planner_index % 2 == 0 else 'v2_apf'
                buffer.add(make_trajectory(
                    f'trajectory_{trajectory_number:08d}',
                    item['scenario_id'],
                    planner,
                    item['scenario_seed'],
                    zone_counts[scenario_index],
                    scenario_index % 3 + 1,
                ))
                trajectory_number += 1
                successful_count += 1
        filename = f'shard_{shard_number:04d}.npz'
        summary = buffer.flush(root / filename)
        shards.append({
            'file': filename,
            'scenario_sequence_start': start,
            'scenario_sequence_end': end - 1,
            **summary,
        })
    manifest = {
        'format': V2_TRAJECTORY_CLUSTER_FORMAT,
        'format_version': V2_TRAJECTORY_CLUSTER_VERSION,
        'status': 'complete',
        'scenario_pool_file': 'scenario_pool.json',
        'master_seed': 77,
        'requested_scenarios': len(scenarios),
        'shard_size': shard_scenario_count,
        'scenario_config': normalised_config(config),
        'uav_collision_radius': uav_collision_radius,
        'shards': shards,
        'statistics': {'successful_trajectories': successful_count},
    }
    (root / 'manifest.json').write_text(
        json.dumps(manifest, allow_nan=False, indent=2), encoding='utf-8'
    )
    return root


def rewrite_shard(path: Path, mutator) -> None:
    arrays = load_v2_trajectory_shard(path)
    mutator(arrays)
    np.savez_compressed(path, **arrays)


class TestV2BCClusterLoading(unittest.TestCase):
    def test_exact_same_scenario_expert_trajectories_are_optionally_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(
                Path(directory), zone_counts=(0, 1, 2, 0),
                trajectories_per_scenario=2, shard_scenario_count=2,
            )
            original = load_v2_bc_trajectory_cluster(root)
            deduplicated = load_v2_bc_trajectory_cluster(
                root,
                deduplicate_identical_trajectories=True,
                shard_cache_mb=16,
            )
            split = split_v2_bc_scenarios(
                deduplicated, validation_fraction=0.25, seed=11
            )
            original_split = split_v2_bc_scenarios(
                original, validation_fraction=0.25, seed=11
            )
            effective_batches = list(iter_v2_bc_batches(
                deduplicated,
                deduplicated.scenario_ids,
                batch_size=3,
                shuffle=False,
                rng=None,
            ))

        self.assertEqual(original.trajectory_count, 8)
        self.assertEqual(deduplicated.source_trajectory_count, 8)
        self.assertEqual(deduplicated.trajectory_count, 4)
        self.assertEqual(
            deduplicated.scenario_ids,
            original.scenario_ids,
        )
        self.assertEqual(
            split.train_scenario_ids,
            original_split.train_scenario_ids,
        )
        self.assertEqual(
            split.validation_scenario_ids,
            original_split.validation_scenario_ids,
        )
        self.assertEqual(len(deduplicated.duplicate_trajectory_mappings), 4)
        self.assertEqual(
            sum(batch.batch_size for batch, _ in effective_batches),
            deduplicated.step_count,
        )
        self.assertTrue(all(
            record.planner_name == 'v2_heuristic'
            for record in deduplicated.trajectories
        ))
        combined = (
            split.train_statistics['deduplication']['removed_trajectory_count']
            + split.validation_statistics['deduplication']['removed_trajectory_count']
        )
        self.assertEqual(combined, 4)

    def test_one_different_field_keeps_both_experts_and_never_deduplicates_across_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(
                Path(directory), zone_counts=(0, 0),
                trajectories_per_scenario=2, shard_scenario_count=2,
            )
            rewrite_shard(
                root / 'shard_0001.npz',
                lambda arrays: arrays['actions'].__setitem__(
                    (int(arrays['trajectory_offsets'][1]), 0), np.float32(0.011)
                ),
            )
            cluster = load_v2_bc_trajectory_cluster(
                root, deduplicate_identical_trajectories=True
            )

            cross_root = Path(directory) / 'cross'
            write_cluster(
                cross_root, zone_counts=(0, 0),
                trajectories_per_scenario=1, shard_scenario_count=2,
            )
            cross = load_v2_bc_trajectory_cluster(
                cross_root, deduplicate_identical_trajectories=True
            )

        self.assertEqual(cluster.source_trajectory_count, 4)
        self.assertEqual(cluster.trajectory_count, 3)
        first_scenario = cluster.scenario_ids[0]
        self.assertEqual(
            sum(record.scenario_id == first_scenario for record in cluster.trajectories),
            2,
        )
        self.assertEqual(cross.source_trajectory_count, 2)
        self.assertEqual(cross.trajectory_count, 2)
        self.assertEqual(cross.duplicate_trajectory_mappings, ())

    def test_shard_cache_hits_respects_budget_and_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(
                Path(directory), zone_counts=(0, 1, 2, 0), shard_scenario_count=2
            )
            paths = (root / 'shard_0001.npz', root / 'shard_0002.npz')
            loaded = [load_v2_trajectory_shard(path) for path in paths]
            sizes = [sum(array.nbytes for array in arrays.values()) for arrays in loaded]

            with mock.patch(
                'brain_uav.trainers.v2_bc.load_v2_trajectory_shard',
                wraps=load_v2_trajectory_shard,
            ) as loader:
                cache = V2ShardArrayCache(max_bytes=max(sizes))
                first = cache.load(paths[0])
                again = cache.load(paths[0])
                self.assertIs(first, again)
                self.assertEqual(loader.call_count, 1)
                cache.load(paths[1])
                self.assertLessEqual(cache.cached_bytes, cache.max_bytes)
                self.assertEqual(cache.entry_count, 1)

            with mock.patch(
                'brain_uav.trainers.v2_bc.load_v2_trajectory_shard',
                wraps=load_v2_trajectory_shard,
            ) as loader:
                disabled = V2ShardArrayCache(max_bytes=0)
                first = disabled.load(paths[0])
                second = disabled.load(paths[0])
                self.assertEqual(loader.call_count, 2)
                self.assertEqual(disabled.entry_count, 0)
                for name in first:
                    np.testing.assert_array_equal(first[name], second[name])

    def test_validated_cache_is_shared_by_index_training_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(
                Path(directory), zone_counts=(0, 1, 2, 0), shard_scenario_count=2
            )
            with mock.patch(
                'brain_uav.trainers.v2_bc.load_v2_trajectory_shard',
                wraps=load_v2_trajectory_shard,
            ) as loader:
                cluster = load_v2_bc_trajectory_cluster(root, shard_cache_mb=16)
                initial_loads = loader.call_count
                split = split_v2_bc_scenarios(
                    cluster, validation_fraction=0.25, seed=3
                )
                list(iter_v2_bc_batches(
                    cluster, split.train_scenario_ids, batch_size=2,
                    shuffle=False, rng=None,
                ))
                list(iter_v2_bc_batches(
                    cluster, split.validation_scenario_ids, batch_size=2,
                    shuffle=False, rng=None,
                ))
        self.assertEqual(initial_loads, len(cluster.shards))
        self.assertEqual(loader.call_count, initial_loads)

    def test_direct_batch_assembly_matches_observation_collator_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(
                Path(directory), zone_counts=(0, 1, 2),
                trajectories_per_scenario=1, shard_scenario_count=3,
            )
            arrays = load_v2_trajectory_shard(root / 'shard_0001.npz')
            indices = np.arange(arrays['ego_features'].shape[0], dtype=np.int64)
            reference = collate_v2_observations([
                restore_v2_observation(arrays, int(index)) for index in indices
            ])
            actual, actions = assemble_v2_bc_batch(arrays, indices, device='cpu')

        self.assertTrue(torch.equal(actual.ego_features, reference.ego_features))
        self.assertTrue(torch.equal(actual.goal_features, reference.goal_features))
        self.assertTrue(torch.equal(actual.zone_features, reference.zone_features))
        self.assertTrue(torch.equal(actual.presence_mask, reference.presence_mask))
        self.assertTrue(torch.equal(actions, torch.from_numpy(arrays['actions'][indices])))

    def test_direct_batch_assembly_supports_large_batches_and_ten_zones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ten_zones.npz'
            buffer = V2TrajectoryShardBuffer()
            buffer.add(make_trajectory(
                'trajectory-ten', 'scenario-ten', 'v2_heuristic', 9, 10, 1
            ))
            buffer.flush(path)
            arrays = load_v2_trajectory_shard(path)
            for batch_size in (128, 256, 512):
                with self.subTest(batch_size=batch_size):
                    indices = np.zeros(batch_size, dtype=np.int64)
                    batch, actions = assemble_v2_bc_batch(
                        arrays, indices, device='cpu'
                    )
                    self.assertEqual(batch.zone_features.shape, (batch_size, 10, 19))
                    self.assertEqual(batch.presence_mask.shape, (batch_size, 10))
                    self.assertTrue(batch.presence_mask.all())
                    self.assertEqual(actions.shape, (batch_size, 2))

    def test_ragged_steps_restore_zero_one_and_two_zones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cluster = load_v2_bc_trajectory_cluster(write_cluster(Path(directory), zone_counts=(0, 1, 2)))
            restored_counts = []
            for shard in cluster.shards:
                arrays = load_v2_trajectory_shard(shard.path)
                for step_index in range(arrays['ego_features'].shape[0]):
                    restored = restore_v2_observation(arrays, step_index)
                    restored_counts.append(restored.zone_features.shape[0])
                    self.assertTrue(np.all(restored.presence_mask))
            self.assertEqual(set(restored_counts), {0, 1, 2})

    def test_mixed_batch_uses_dynamic_padding_and_presence_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cluster = load_v2_bc_trajectory_cluster(write_cluster(
                Path(directory), zone_counts=(0, 1, 2),
                trajectories_per_scenario=1, shard_scenario_count=3,
            ))
            batches = list(iter_v2_bc_batches(
                cluster, cluster.scenario_ids, batch_size=16,
                shuffle=False, rng=None, device='cpu',
            ))
        self.assertEqual(len(batches), 1)
        observations, actions = batches[0]
        self.assertEqual(observations.max_zone_count, 2)
        self.assertEqual(observations.zone_features.shape[2], 19)
        self.assertEqual(actions.dtype, torch.float32)
        counts = observations.presence_mask.sum(dim=1).tolist()
        self.assertTrue({0, 1, 2}.issubset(set(counts)))
        self.assertTrue(torch.equal(
            observations.zone_features[~observations.presence_mask],
            torch.zeros_like(observations.zone_features[~observations.presence_mask]),
        ))

    def test_stream_assembles_on_cpu_before_moving_whole_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cluster = load_v2_bc_trajectory_cluster(write_cluster(
                Path(directory), zone_counts=(0, 1, 2),
                trajectories_per_scenario=1, shard_scenario_count=3,
            ))
            original_to = V2ObservationBatch.to
            transfer_sources = []

            def checked_transfer(batch, device):
                transfer_sources.append((
                    batch.ego_features.device.type,
                    batch.goal_features.device.type,
                    batch.zone_features.device.type,
                    batch.presence_mask.device.type,
                    str(torch.device(device)),
                ))
                return original_to(batch, device)

            with mock.patch.object(
                V2ObservationBatch,
                'to',
                autospec=True,
                side_effect=checked_transfer,
            ):
                batches = list(iter_v2_bc_batches(
                    cluster,
                    cluster.scenario_ids,
                    batch_size=16,
                    shuffle=False,
                    rng=None,
                    device='cpu',
                ))

        self.assertEqual(len(batches), 1)
        self.assertEqual(
            transfer_sources,
            [('cpu', 'cpu', 'cpu', 'cpu', 'cpu')],
        )

    def test_all_empty_zone_batch_runs_through_v2_actor(self) -> None:
        batch = collate_v2_observations([make_observation(0, 1.0), make_observation(0, 2.0)])
        actor = V2ANNPolicyActor(
            V2ObservationScales(50.0, 0.1, 30.0, 0.6), 2, 16,
            torch.tensor([0.14, 0.2], dtype=torch.float32),
        )
        output = actor(batch)
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_padding_garbage_does_not_change_actor_output(self) -> None:
        actor = V2ANNPolicyActor(
            V2ObservationScales(50.0, 0.1, 30.0, 0.6), 2, 16,
            torch.tensor([0.14, 0.2], dtype=torch.float32),
        )
        actor.eval()
        clean = V2ObservationBatch(
            torch.zeros((1, 6), dtype=torch.float32),
            torch.zeros((1, 4), dtype=torch.float32),
            torch.zeros((1, 3, 19), dtype=torch.float32),
            torch.tensor([[True, False, False]]),
        )
        garbage_zones = clean.zone_features.clone()
        garbage_zones[:, 1:] = 1234.0
        garbage = V2ObservationBatch(
            clean.ego_features.clone(), clean.goal_features.clone(),
            garbage_zones, clean.presence_mask.clone(),
        )
        with torch.inference_mode():
            torch.testing.assert_close(actor(clean), actor(garbage), atol=1e-6, rtol=1e-6)

    def test_each_shard_is_loaded_once_per_stream_not_once_per_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(Path(directory), zone_counts=(0, 1, 2, 0))
            with mock.patch(
                'brain_uav.trainers.v2_bc.load_v2_trajectory_shard',
                wraps=load_v2_trajectory_shard,
            ) as loader:
                cluster = load_v2_bc_trajectory_cluster(root)
                index_loads = loader.call_count
                list(iter_v2_bc_batches(
                    cluster, cluster.scenario_ids, batch_size=1,
                    shuffle=False, rng=None,
                ))
                epoch_loads = loader.call_count - index_loads
        self.assertEqual(index_loads, len(cluster.shards))
        self.assertEqual(epoch_loads, len(cluster.shards))
        self.assertGreater(cluster.step_count, epoch_loads)

    def test_missing_shard_wrong_manifest_and_zero_success_are_rejected(self) -> None:
        cases = (
            ('missing shard', lambda manifest: manifest['shards'][0].update(file='missing.npz')),
            ('format', lambda manifest: manifest.update(format='legacy_flat_bc')),
            ('version', lambda manifest: manifest.update(format_version=-1)),
            ('status', lambda manifest: manifest.update(status='failed_zero_success')),
            ('successful', lambda manifest: manifest['statistics'].update(successful_trajectories=0)),
        )
        for expected, mutate in cases:
            with self.subTest(case=expected), tempfile.TemporaryDirectory() as directory:
                root = write_cluster(Path(directory))
                path = root / 'manifest.json'
                manifest = json.loads(path.read_text(encoding='utf-8'))
                mutate(manifest)
                path.write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex((ValueError, FileNotFoundError), expected):
                    load_v2_bc_trajectory_cluster(root)

    def test_out_of_range_action_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(Path(directory))
            rewrite_shard(
                root / 'shard_0001.npz',
                lambda arrays: arrays['actions'].__setitem__((0, 0), np.float32(0.5)),
            )
            with self.assertRaisesRegex(ValueError, 'action'):
                load_v2_bc_trajectory_cluster(root)

    def test_manifest_pool_binding_and_shard_statistics_are_strict(self) -> None:
        mutations = {
            'master_seed': lambda manifest: manifest.__setitem__('master_seed', 999),
            'requested_scenarios': lambda manifest: manifest.__setitem__('requested_scenarios', 999),
            'scenario_config': lambda manifest: manifest['scenario_config'].__setitem__('speed', 9.0),
            'uav_collision_radius': lambda manifest: manifest.__setitem__('uav_collision_radius', 1.0),
            'statistics mismatch': lambda manifest: manifest['shards'][0].__setitem__('step_count', 999),
        }
        for expected, mutate in mutations.items():
            with self.subTest(case=expected), tempfile.TemporaryDirectory() as directory:
                root = write_cluster(Path(directory))
                path = root / 'manifest.json'
                manifest = json.loads(path.read_text(encoding='utf-8'))
                mutate(manifest)
                path.write_text(json.dumps(manifest), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, expected):
                    load_v2_bc_trajectory_cluster(root)

    def test_unknown_planner_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(Path(directory))
            rewrite_shard(
                root / 'shard_0001.npz',
                lambda arrays: arrays['planner_names'].__setitem__(0, 'legacy_planner'),
            )
            with self.assertRaisesRegex(ValueError, 'planner_name'):
                load_v2_bc_trajectory_cluster(root)

    def test_duplicate_trajectory_id_across_shards_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = write_cluster(Path(directory), zone_counts=(0, 1, 2, 0))
            first = load_v2_trajectory_shard(root / 'shard_0001.npz')
            duplicate = str(first['trajectory_ids'][0])
            rewrite_shard(
                root / 'shard_0002.npz',
                lambda arrays: arrays['trajectory_ids'].__setitem__(0, duplicate),
            )
            with self.assertRaisesRegex(ValueError, 'trajectory_id'):
                load_v2_bc_trajectory_cluster(root)

    def test_unknown_scenario_and_mismatched_payload_ref_are_rejected(self) -> None:
        cases = (
            ('scenario_id', 'scenario_ids', 'scenario_999999'),
            ('scenario_payload_ref', 'scenario_payload_refs', 'scenario_999999'),
        )
        for expected, array_name, value in cases:
            with self.subTest(case=expected), tempfile.TemporaryDirectory() as directory:
                root = write_cluster(Path(directory))
                rewrite_shard(
                    root / 'shard_0001.npz',
                    lambda arrays, n=array_name, v=value: arrays[n].__setitem__(0, v),
                )
                with self.assertRaisesRegex(ValueError, expected):
                    load_v2_bc_trajectory_cluster(root)


class TestV2BCScenarioSplit(unittest.TestCase):
    def test_split_is_disjoint_deterministic_and_groups_both_experts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cluster = load_v2_bc_trajectory_cluster(write_cluster(
                Path(directory), zone_counts=(0, 1, 2, 0, 1, 2),
            ))
            first = split_v2_bc_scenarios(cluster, validation_fraction=0.2, seed=123)
            second = split_v2_bc_scenarios(cluster, validation_fraction=0.2, seed=123)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_scenario_ids) & set(first.validation_scenario_ids))
        for record in cluster.trajectories:
            memberships = (
                record.scenario_id in first.train_scenario_ids,
                record.scenario_id in first.validation_scenario_ids,
            )
            self.assertEqual(sum(memberships), 1)
        self.assertEqual(first.train_statistics['scenario_count'], 4)
        self.assertEqual(first.validation_statistics['scenario_count'], 2)
        self.assertIn('v2_heuristic', first.train_statistics['planners'])
        self.assertIn('zone_count_steps', first.validation_statistics)

    def test_invalid_fraction_and_too_few_successful_scenarios_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cluster = load_v2_bc_trajectory_cluster(write_cluster(Path(directory), zone_counts=(0,)))
            for value in (0.0, 1.0, -0.1, float('nan')):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, 'validation_fraction'):
                        split_v2_bc_scenarios(cluster, validation_fraction=value, seed=1)
            with self.assertRaisesRegex(ValueError, 'training.*validation'):
                split_v2_bc_scenarios(cluster, validation_fraction=0.2, seed=1)


class TestValidationBestTracker(unittest.TestCase):
    def test_strictly_lower_validation_loss_selects_best_and_ties_keep_earlier(self) -> None:
        actor = torch.nn.Linear(2, 1)
        tracker = ValidationBestTracker()
        with torch.no_grad():
            actor.weight.fill_(1.0)
        self.assertTrue(tracker.consider(epoch=1, validation_loss=0.5, actor=actor))
        with torch.no_grad():
            actor.weight.fill_(2.0)
        self.assertFalse(tracker.consider(epoch=2, validation_loss=0.6, actor=actor))
        self.assertFalse(tracker.consider(epoch=3, validation_loss=0.5, actor=actor))
        with torch.no_grad():
            actor.weight.fill_(3.0)
        self.assertTrue(tracker.consider(epoch=4, validation_loss=0.4, actor=actor))
        self.assertEqual(tracker.best_epoch, 4)
        self.assertEqual(tracker.best_validation_loss, 0.4)
        torch.testing.assert_close(
            tracker.best_state_dict['weight'],
            torch.full_like(tracker.best_state_dict['weight'], 3.0),
        )


if __name__ == '__main__':
    unittest.main()
