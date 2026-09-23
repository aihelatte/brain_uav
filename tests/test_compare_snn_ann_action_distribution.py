"""Smoke test for the ANN-vs-SNN action distribution comparison script (B2)."""

from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path

import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.models import V2ANNCritic, V2SNNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.scripts.compare_snn_ann_action_distribution import (
    compare_action_distributions,
)
from brain_uav.trainers import V2ReplayBuffer, V2TD3UpdateEngine
from brain_uav.trainers.v2_formal_training import (
    V2FormalTrainingConfig,
    build_v2_periodic_snapshot,
    save_v2_periodic_snapshot,
)
from brain_uav.trainers.v2_validation import (
    generate_v2_validation_pool,
    save_v2_validation_pool,
)
from test_v2_formal_training import _engine


def _snn_engine(scenario: ScenarioConfig, *, seed: int = 5) -> V2TD3UpdateEngine:
    torch.manual_seed(seed)
    scales = V2ObservationScales(
        scenario.world_xy, scenario.world_z_min,
        scenario.world_z_max, scenario.gamma_max,
    )
    limit = torch.tensor(
        [scenario.delta_gamma_max, scenario.delta_psi_max], dtype=torch.float32,
    )
    actor = V2SNNPolicyActor(scales, 2, 8, limit, time_window=2, tau=2.0)
    critic1 = V2ANNCritic(scales, 2, 8)
    critic2 = V2ANNCritic(scales, 2, 8)
    replay = V2ReplayBuffer(16, 2, 6, seed=19)
    return V2TD3UpdateEngine(
        actor, critic1, critic2, replay,
        1e-3, 1e-3, 0.99, 0.005, 0.015, 0.03, 2, 2,
        -limit.numpy(), limit.numpy(),
        actor_freeze_steps=0, actor_grad_clip_norm=1.0, critic_grad_clip_norm=1.0,
        bc_reference_actor=deepcopy(actor),
    )


class TestCompareSNNANNActionDistribution(unittest.TestCase):
    def test_runs_end_to_end_on_matching_ann_and_snn_snapshots(self) -> None:
        scenario = ScenarioConfig(max_steps=1)
        ann_engine = _engine(scenario)
        snn_engine = _snn_engine(scenario)
        config = V2FormalTrainingConfig(stage='easy', max_steps=1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = generate_v2_validation_pool(
                scenario, 'easy', scenario_count=2, master_seed=20260904,
            )
            pool_path = root / 'v2_validation_easy.json'
            save_v2_validation_pool(pool_path, pool)

            ann_path = root / 'ann_snapshot.pt'
            save_v2_periodic_snapshot(
                ann_path,
                build_v2_periodic_snapshot(
                    ann_engine, config, stage_steps=1, scenario=scenario,
                    rewards=RewardConfig(), uav_collision_radius=0.0,
                    seed_manifest={'base_seed': 7},
                    initialization_source={'kind': 'v2_bc_best', 'path': 'ann_bc.pt'},
                ),
            )
            snn_path = root / 'snn_snapshot.pt'
            save_v2_periodic_snapshot(
                snn_path,
                build_v2_periodic_snapshot(
                    snn_engine, config, stage_steps=1, scenario=scenario,
                    rewards=RewardConfig(), uav_collision_radius=0.0,
                    seed_manifest={'base_seed': 5},
                    initialization_source={'kind': 'v2_bc_best', 'path': 'snn_bc.pt'},
                ),
            )

            result = compare_action_distributions(
                ann_checkpoint=ann_path,
                snn_checkpoint=snn_path,
                validation_pool=pool_path,
                curriculum_level='easy',
                device='cpu',
            )

        self.assertEqual(result['scenario_count'], 2)
        for model_key in ('ann', 'snn'):
            distribution = result[model_key]['action_distribution']
            self.assertEqual(len(distribution['mean']), 2)
            self.assertEqual(len(distribution['std']), 2)
            self.assertIn('mean', result[model_key]['zone_perturbation_sensitivity'])
        self.assertEqual(
            len(result['per_scenario_sensitivity']),
            result['perturbation_scenario_count'],
        )

    def test_rejects_incompatible_scenario_configs(self) -> None:
        scenario = ScenarioConfig(max_steps=1)
        other_scenario = ScenarioConfig(max_steps=2)
        ann_engine = _engine(scenario)
        snn_engine = _snn_engine(other_scenario)
        ann_config = V2FormalTrainingConfig(stage='easy', max_steps=1)
        snn_config = V2FormalTrainingConfig(stage='easy', max_steps=2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = generate_v2_validation_pool(
                scenario, 'easy', scenario_count=1, master_seed=20260904,
            )
            pool_path = root / 'v2_validation_easy.json'
            save_v2_validation_pool(pool_path, pool)

            ann_path = root / 'ann_snapshot.pt'
            save_v2_periodic_snapshot(
                ann_path,
                build_v2_periodic_snapshot(
                    ann_engine, ann_config, stage_steps=1, scenario=scenario,
                    rewards=RewardConfig(), uav_collision_radius=0.0,
                    seed_manifest={'base_seed': 7},
                    initialization_source={'kind': 'v2_bc_best', 'path': 'ann_bc.pt'},
                ),
            )
            snn_path = root / 'snn_snapshot.pt'
            save_v2_periodic_snapshot(
                snn_path,
                build_v2_periodic_snapshot(
                    snn_engine, snn_config, stage_steps=1, scenario=other_scenario,
                    rewards=RewardConfig(), uav_collision_radius=0.0,
                    seed_manifest={'base_seed': 5},
                    initialization_source={'kind': 'v2_bc_best', 'path': 'snn_bc.pt'},
                ),
            )

            with self.assertRaisesRegex(ValueError, 'ScenarioConfig'):
                compare_action_distributions(
                    ann_checkpoint=ann_path,
                    snn_checkpoint=snn_path,
                    validation_pool=pool_path,
                    curriculum_level='easy',
                    device='cpu',
                )


if __name__ == '__main__':
    unittest.main()
