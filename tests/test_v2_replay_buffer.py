"""Tests for structured V2 replay storage and dynamic sampling."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from brain_uav.observations import (
    EGO_FEATURE_DIM,
    GOAL_FEATURE_DIM,
    ZONE_FEATURE_DIM,
    V2Observation,
    collate_v2_observations,
)
from brain_uav.trainers import V2ReplayBatch, V2ReplayBuffer


def _observation(zone_count: int, *, offset: float = 0.0) -> V2Observation:
    ego = np.arange(EGO_FEATURE_DIM, dtype=np.float32) + offset
    goal = np.arange(GOAL_FEATURE_DIM, dtype=np.float32) + 10.0 + offset
    zones = np.zeros((zone_count, ZONE_FEATURE_DIM), dtype=np.float32)
    for zone_index in range(zone_count):
        zones[zone_index] = offset * 100.0 + zone_index + 1.0
    return V2Observation(
        ego_features=ego,
        goal_features=goal,
        zone_features=zones,
        presence_mask=np.ones(zone_count, dtype=np.bool_),
    )


class _ScriptedRNG:
    def __init__(self, *, draws=(), choices=()):
        self._draws = iter(draws)
        self._choices = iter(choices)

    def random(self):
        return next(self._draws)

    def choice(self, *args, **kwargs):
        del args, kwargs
        return next(self._choices)


class _FailingRNG:
    def __init__(self):
        self.calls = 0

    def random(self):
        self.calls += 1
        if self.calls == 1:
            return 0.1
        raise RuntimeError('controlled random failure')


class TestV2ReplayBuffer(unittest.TestCase):
    def test_constructor_strictly_rejects_invalid_biases_fractions_and_seed(self):
        invalid_cases = (
            {'success_sample_bias': 0.99},
            {'success_sample_bias': float('nan')},
            {'near_goal_sample_bias': 0.5},
            {'near_goal_sample_bias': float('inf')},
            {'success_replay_fraction': -0.01},
            {'success_replay_fraction': 1.01},
            {'success_batch_fraction': float('nan')},
            {'success_batch_fraction': float('inf')},
            {'seed': -1},
            {'seed': True},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides), self.assertRaises((TypeError, ValueError)):
                V2ReplayBuffer(8, 2, 6, **overrides)

    def test_owned_rng_reproduces_sampling_without_global_numpy_state(self):
        first = V2ReplayBuffer(16, 2, 6, seed=73)
        second = V2ReplayBuffer(16, 2, 6, seed=73)
        for index in range(10):
            observation = _observation(index % 7, offset=float(index))
            for replay in (first, second):
                replay.add(
                    observation,
                    np.zeros(2, dtype=np.float32),
                    float(index),
                    observation,
                    False,
                    near_goal=index % 3 == 0,
                )

        np.random.seed(1)
        first_batch = first.sample(6)
        np.random.seed(999)
        second_batch = second.sample(6)

        self.assertTrue(torch.equal(first_batch.obs.ego_features, second_batch.obs.ego_features))
        self.assertTrue(torch.equal(first_batch.reward, second_batch.reward))

    def test_formal_six_zone_storage_rejects_seven_without_truncation(self):
        replay = V2ReplayBuffer(4, 2, 6, seed=5)
        valid = _observation(6)
        replay.add(valid, np.zeros(2, dtype=np.float32), 0.0, valid, False)
        before = (len(replay), replay.position, replay.next_write_id)
        with self.assertRaisesRegex(ValueError, 'zone_storage_capacity=6'):
            replay.add(
                _observation(7),
                np.zeros(2, dtype=np.float32),
                0.0,
                valid,
                False,
            )
        self.assertEqual((len(replay), replay.position, replay.next_write_id), before)

    def test_add_and_sample_supports_zero_one_five_six_and_ten_zones(self):
        replay = V2ReplayBuffer(
            capacity=8,
            action_dim=2,
            zone_storage_capacity=10,
            success_replay_fraction=0.0,
        )
        counts = (0, 1, 5, 6, 10)
        for index, count in enumerate(counts):
            replay.add(
                _observation(count, offset=float(index)),
                np.array([index, -index], dtype=np.float32),
                float(index),
                _observation(count, offset=float(index) + 0.5),
                index % 2 == 0,
            )

        replay.rng = _ScriptedRNG(draws=[0.0] * len(counts))
        batch = replay.sample(len(counts))

        self.assertIsInstance(batch, V2ReplayBatch)
        self.assertEqual(batch.obs.zone_features.shape, (5, 10, ZONE_FEATURE_DIM))
        self.assertEqual(batch.obs.presence_mask.shape, (5, 10))
        self.assertEqual(batch.next_obs.zone_features.shape, (5, 10, ZONE_FEATURE_DIM))
        for row, count in enumerate(counts):
            with self.subTest(count=count):
                self.assertEqual(int(batch.obs.presence_mask[row].sum()), count)
                self.assertEqual(int(batch.next_obs.presence_mask[row].sum()), count)
                torch.testing.assert_close(
                    batch.obs.zone_features[row, :count],
                    torch.from_numpy(_observation(count, offset=float(row)).zone_features.copy()),
                )
                torch.testing.assert_close(
                    batch.obs.zone_features[row, count:],
                    torch.zeros((10 - count, ZONE_FEATURE_DIM)),
                )

    def test_obs_and_next_obs_are_cropped_to_independent_batch_maxima(self):
        replay = V2ReplayBuffer(4, 2, 10, success_replay_fraction=0.0)
        replay.add(
            _observation(1, offset=1.0),
            np.zeros(2, dtype=np.float32),
            1.0,
            _observation(6, offset=2.0),
            False,
        )
        replay.add(
            _observation(0, offset=3.0),
            np.ones(2, dtype=np.float32),
            2.0,
            _observation(5, offset=4.0),
            True,
        )

        replay.rng = _ScriptedRNG(draws=[0.0, 0.0])
        batch = replay.sample(2)

        self.assertEqual(batch.obs.zone_features.shape, (2, 1, ZONE_FEATURE_DIM))
        self.assertEqual(batch.obs.presence_mask.tolist(), [[True], [False]])
        self.assertEqual(batch.next_obs.zone_features.shape, (2, 6, ZONE_FEATURE_DIM))
        self.assertEqual(
            batch.next_obs.presence_mask.tolist(),
            [[True, True, True, True, True, True], [True, True, True, True, True, False]],
        )
        torch.testing.assert_close(
            batch.next_obs.zone_features[1, 5:],
            torch.zeros((1, ZONE_FEATURE_DIM)),
        )

    def test_all_empty_sample_keeps_zero_length_zone_axes(self):
        replay = V2ReplayBuffer(3, 2, 10, success_replay_fraction=0.0)
        for index in range(3):
            replay.add(
                _observation(0, offset=float(index)),
                np.zeros(2, dtype=np.float32),
                0.0,
                _observation(0, offset=float(index) + 1.0),
                False,
            )

        replay.rng = _ScriptedRNG(draws=[0.0, 0.75])
        batch = replay.sample(2)

        self.assertEqual(batch.obs.zone_features.shape, (2, 0, ZONE_FEATURE_DIM))
        self.assertEqual(batch.obs.presence_mask.shape, (2, 0))
        self.assertEqual(batch.next_obs.zone_features.shape, (2, 0, ZONE_FEATURE_DIM))
        self.assertEqual(batch.next_obs.presence_mask.shape, (2, 0))

    def test_ring_overwrite_clears_unused_zone_storage_and_copies_action(self):
        replay = V2ReplayBuffer(1, 2, 10, success_replay_fraction=0.0)
        replay.add(
            _observation(10, offset=1.0),
            np.array([1.0, 2.0], dtype=np.float32),
            1.0,
            _observation(10, offset=2.0),
            False,
        )
        action = np.array([3.0, 4.0], dtype=np.float32)
        replay.add(
            _observation(1, offset=3.0),
            action,
            2.0,
            _observation(0, offset=4.0),
            True,
        )
        action[:] = -999.0

        self.assertTrue(np.all(replay.zone_features[0, 1:] == 0.0))
        self.assertTrue(np.all(replay.next_zone_features[0] == 0.0))
        replay.rng = _ScriptedRNG(draws=[0.0])
        batch = replay.sample(1)
        torch.testing.assert_close(batch.action, torch.tensor([[3.0, 4.0]]))
        self.assertEqual(batch.obs.max_zone_count, 1)
        self.assertEqual(batch.next_obs.max_zone_count, 0)

    def test_zone_storage_overflow_is_rejected_without_partial_write(self):
        replay = V2ReplayBuffer(2, 2, 10, success_replay_fraction=0.0)
        valid = _observation(1)
        replay.add(valid, np.zeros(2, dtype=np.float32), 0.0, valid, False)
        before = (len(replay), replay.position, replay.next_write_id)

        with self.assertRaisesRegex(ValueError, 'zone_storage_capacity'):
            replay.add(
                _observation(11),
                np.zeros(2, dtype=np.float32),
                0.0,
                valid,
                False,
            )
        self.assertEqual((len(replay), replay.position, replay.next_write_id), before)

        with self.assertRaisesRegex(ValueError, 'zone_storage_capacity'):
            replay.add_success_transition(
                valid,
                np.zeros(2, dtype=np.float32),
                0.0,
                _observation(11),
                False,
            )

    def test_write_id_protects_overwritten_slots_and_updates_weights(self):
        replay = V2ReplayBuffer(
            1,
            2,
            2,
            success_sample_bias=3.0,
            near_goal_sample_bias=2.0,
            success_replay_fraction=0.0,
        )
        obs = _observation(1)
        stale = replay.add(
            obs,
            np.zeros(2, dtype=np.float32),
            0.0,
            obs,
            False,
            near_goal=True,
        )
        live = replay.add(
            obs,
            np.zeros(2, dtype=np.float32),
            0.0,
            obs,
            False,
        )

        self.assertEqual(stale, (0, 0))
        self.assertEqual(live, (0, 1))
        self.assertEqual(replay.mark_success_slots([stale]), 0)
        self.assertEqual(replay.mark_success_slots([live]), 1)
        self.assertEqual(replay.success_count, 1)
        self.assertEqual(replay.near_goal_count, 0)
        self.assertEqual(replay.sample_weight[0], 3.0)
        self.assertEqual(replay.total_sample_weight, 3.0)
        self.assertEqual(replay.sampling_weight_total, 3.0)

    def test_weighted_primary_sampling_uses_exact_remaining_weight_intervals(self):
        replay = V2ReplayBuffer(
            4,
            2,
            1,
            success_sample_bias=2.0,
            near_goal_sample_bias=3.0,
            success_replay_fraction=0.0,
        )
        obs = _observation(1)
        replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False)
        replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False, success=True)
        replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False, near_goal=True)
        replay.add(
            obs,
            np.zeros(2, dtype=np.float32),
            0.0,
            obs,
            False,
            success=True,
            near_goal=True,
        )

        np.testing.assert_array_equal(replay.sample_weight, [1.0, 2.0, 3.0, 6.0])
        self.assertEqual(replay.total_sample_weight, 12.0)
        self.assertEqual(replay.sampling_weight_total, 12.0)
        replay.rng = _ScriptedRNG(draws=[0.0, 0.39])
        indices = replay._sample_primary_indices(2)

        np.testing.assert_array_equal(indices, [0, 2])
        self.assertEqual(replay.sampling_weight_total, 12.0)
        self.assertEqual(replay.sampling_implementation, 'fenwick_ppswor_v1')
        self.assertAlmostEqual(replay.success_fraction(), 0.5)
        self.assertAlmostEqual(replay.near_goal_fraction(), 0.5)

    def test_ordered_weighted_without_replacement_probability_matches_formula(self):
        replay = V2ReplayBuffer(
            3,
            2,
            1,
            success_sample_bias=2.0,
            near_goal_sample_bias=3.0,
            success_replay_fraction=0.0,
            seed=20260909,
        )
        obs = _observation(1)
        replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False)
        replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False, success=True)
        replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False, near_goal=True)

        trials = 20_000
        ordered_01 = sum(
            tuple(replay._sample_primary_indices(2)) == (0, 1)
            for _ in range(trials)
        )
        observed = ordered_01 / trials
        expected = (1.0 / 6.0) * (2.0 / 5.0)
        self.assertAlmostEqual(observed, expected, delta=0.006)
        self.assertEqual(replay.sampling_weight_total, 6.0)

    def test_primary_sampling_restores_tree_after_success_and_exception(self):
        replay = V2ReplayBuffer(
            4,
            2,
            1,
            success_sample_bias=2.0,
            near_goal_sample_bias=3.0,
            success_replay_fraction=0.0,
        )
        obs = _observation(1)
        for index in range(4):
            replay.add(
                obs,
                np.zeros(2, dtype=np.float32),
                float(index),
                obs,
                False,
                success=index in (1, 3),
                near_goal=index in (2, 3),
            )
        expected_total = replay.total_sample_weight
        replay.rng = _FailingRNG()
        with self.assertRaisesRegex(RuntimeError, 'controlled random failure'):
            replay._sample_primary_indices(3)
        self.assertEqual(replay.sampling_weight_total, expected_total)

        replay.rng = _ScriptedRNG(draws=[0.0] * 4)
        indices = replay._sample_primary_indices(4)
        self.assertEqual(len(set(indices.tolist())), 4)
        self.assertEqual(replay.sampling_weight_total, expected_total)

    def test_primary_sampling_rejects_corrupt_cumulative_state(self):
        replay = V2ReplayBuffer(2, 2, 1, success_replay_fraction=0.0)
        obs = _observation(1)
        replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False)
        replay._sampling_weight_tree.add(0, 1.0)
        replay.rng = _ScriptedRNG(draws=[0.0])
        with self.assertRaisesRegex(RuntimeError, 'cumulative weight state'):
            replay._sample_primary_indices(1)

    def test_success_replay_uses_quarter_batch_and_primary_first_order(self):
        replay = V2ReplayBuffer(16, 2, 10)
        for index in range(8):
            obs = _observation(index % 3, offset=float(index))
            replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False)
        for index in range(3):
            obs = _observation(index + 1, offset=100.0 + index)
            replay.add_success_transition(
                obs,
                np.zeros(2, dtype=np.float32),
                1.0,
                _observation(index + 2, offset=110.0 + index),
                True,
                near_goal=True,
            )

        replay.rng = _ScriptedRNG(
            draws=[0.0] * 6,
            choices=[np.array([1, 2], dtype=np.int64)],
        )
        batch = replay.sample(8)

        torch.testing.assert_close(
            batch.obs.ego_features[:, 0],
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 101.0, 102.0]),
        )
        torch.testing.assert_close(
            batch.success,
            torch.tensor([[0.0]] * 6 + [[1.0], [1.0]]),
        )
        self.assertEqual(batch.obs.max_zone_count, 3)
        self.assertEqual(batch.next_obs.max_zone_count, 4)

    def test_success_shortage_is_backfilled_from_primary(self):
        replay = V2ReplayBuffer(16, 2, 3)
        for index in range(8):
            obs = _observation(1, offset=float(index))
            replay.add(obs, np.zeros(2, dtype=np.float32), 0.0, obs, False)
        replay.add_success_transition(
            _observation(1, offset=99.0),
            np.zeros(2, dtype=np.float32),
            1.0,
            _observation(1, offset=100.0),
            True,
        )

        replay.rng = _ScriptedRNG(
            draws=[0.0] * 7,
            choices=[np.array([0], dtype=np.int64)],
        )
        batch = replay.sample(8)

        self.assertEqual(batch.success[:7].sum().item(), 0.0)
        self.assertEqual(batch.success[7].item(), 1.0)
        self.assertEqual(batch.obs.ego_features[7, 0].item(), 99.0)

    def test_replay_batch_contract_and_to_avoid_python_scalar_conversion(self):
        obs = collate_v2_observations([_observation(1), _observation(0)])
        next_obs = collate_v2_observations([_observation(2), _observation(1)])
        batch = V2ReplayBatch(
            obs=obs,
            action=torch.zeros((2, 2), dtype=torch.float32),
            reward=torch.zeros((2, 1), dtype=torch.float32),
            next_obs=next_obs,
            done=torch.zeros((2, 1), dtype=torch.float32),
            success=torch.zeros((2, 1), dtype=torch.float32),
            near_goal=torch.zeros((2, 1), dtype=torch.float32),
            line_to_goal_safe=torch.ones((2, 1), dtype=torch.float32),
        )

        with (
            mock.patch.object(
                torch.Tensor,
                '__bool__',
                side_effect=AssertionError('Tensor.__bool__ must not be called.'),
            ),
            mock.patch.object(
                torch.Tensor,
                'item',
                side_effect=AssertionError('Tensor.item must not be called.'),
            ),
            mock.patch.object(
                torch.Tensor,
                '__float__',
                side_effect=AssertionError('Tensor.__float__ must not be called.'),
            ),
            mock.patch.object(
                torch.Tensor,
                'tolist',
                side_effect=AssertionError('Tensor.tolist must not be called.'),
            ),
        ):
            moved = batch.to('cpu')

        self.assertIsNot(moved, batch)
        self.assertEqual(moved.batch_size, 2)
        self.assertEqual(moved.obs.max_zone_count, 1)
        self.assertEqual(moved.next_obs.max_zone_count, 2)
        self.assertEqual(moved.action.dtype, torch.float32)
        self.assertEqual(moved.obs.presence_mask.dtype, torch.bool)

        with self.assertRaises((TypeError, ValueError)):
            V2ReplayBatch(
                obs=obs,
                action=torch.zeros((2, 2), dtype=torch.float64),
                reward=batch.reward,
                next_obs=next_obs,
                done=batch.done,
                success=batch.success,
                near_goal=batch.near_goal,
                line_to_goal_safe=batch.line_to_goal_safe,
            )

    def test_constructor_and_add_validate_dimensions(self):
        for args in ((0, 2, 10), (4, 0, 10), (4, 2, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                V2ReplayBuffer(*args)

        replay = V2ReplayBuffer(4, 2, 10)
        obs = _observation(1)
        with self.assertRaisesRegex(ValueError, 'action'):
            replay.add(obs, np.zeros(3, dtype=np.float32), 0.0, obs, False)
        with self.assertRaisesRegex(ValueError, 'batch_size'):
            replay.sample(1)


if __name__ == '__main__':
    unittest.main()
