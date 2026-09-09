"""Formal-stage contracts for V2 SNN actors with unchanged ANN critics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import torch

from brain_uav.config import RewardConfig, TrainingConfig
from brain_uav.models import V2SNNPolicyActor
from brain_uav.scripts.train_v2_bc import train_v2_behavior_cloning
from brain_uav.trainers.v2_formal_training import (
    V2_SNN_FORMAL_CHECKPOINT_FORMAT,
    V2FormalTrainingConfig,
    V2FormalTrainingResult,
    V2FormalStageTrainer,
    build_v2_formal_checkpoint,
    build_v2_stage_engine,
    load_v2_formal_checkpoint,
    prepare_v2_stage_initialization,
)
from test_v2_bc import make_scenario_config, make_scenario_payload, write_cluster


class _Source:
    def __init__(self, payload):
        self.payload = payload

    def generate(self):
        return self.payload


class TestV2SNNFormalTraining(unittest.TestCase):
    def make_snn_bc(
        self,
        root: Path,
        *,
        scenario=None,
        uav_collision_radius: float = 0.0,
    ):
        effective_scenario = make_scenario_config() if scenario is None else scenario
        cluster = write_cluster(
            root / 'cluster',
            zone_counts=(0, 1, 2, 0),
            scenario_config=effective_scenario,
            uav_collision_radius=uav_collision_radius,
        )
        train_v2_behavior_cloning(
            trajectory_cluster=cluster,
            output_dir=root / 'bc',
            seed=37,
            validation_fraction=0.25,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            training_config=TrainingConfig(hidden_dim=8, bc_epochs=1, batch_size=2),
            model='snn',
            snn_time_window=2,
            device='cpu',
        )
        return root / 'bc' / 'bc_v2_snn_best.pt', effective_scenario

    def test_snn_easy_derives_nondefault_scenario_and_collision_radius(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scenario = replace(
                make_scenario_config(),
                target_distance=41.0,
                warning_distance=17.0,
            )
            checkpoint, _ = self.make_snn_bc(
                Path(directory),
                scenario=scenario,
                uav_collision_radius=0.75,
            )
            config = V2FormalTrainingConfig(
                stage='easy', replay_capacity=8, batch_size=2
            )
            prepared = prepare_v2_stage_initialization(
                config,
                init_checkpoint=checkpoint,
                scenario=None,
                rewards=None,
                uav_collision_radius=None,
                device='cpu',
                model_type='snn',
                snn_time_window=2,
            )

            self.assertEqual(prepared.scenario_config, scenario)
            self.assertEqual(prepared.uav_collision_radius, 0.75)
            components = build_v2_stage_engine(
                None,
                config,
                init_checkpoint=checkpoint,
                model_type='snn',
                snn_time_window=2,
                device='cpu',
                prepared_initialization=prepared,
            )
            self.assertIsInstance(components.engine.actor, V2SNNPolicyActor)
            self.assertEqual(components.engine.actor.uav_radius, 0.75)

    @staticmethod
    def passed_result(stage: str) -> V2FormalTrainingResult:
        result = V2FormalTrainingResult.empty(stage, global_steps_start=0)
        result.status = 'passed'
        result.passed_validation = True
        result.stop_reason = 'fixed_validation_passed'
        result.validation_records.append({'passed': True})
        return result

    @staticmethod
    def validation_metadata(stage: str) -> dict:
        return {
            'path': f'{stage}.json',
            'format_version': 1,
            'curriculum_level': stage,
            'master_seed': 20260904,
            'stage_seed': 100,
            'scenario_count': 1,
            'content_digest': 'digest',
        }

    def test_easy_requires_snn_best_and_builds_snn_actor_with_ann_critics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, scenario = self.make_snn_bc(Path(directory))
            components = build_v2_stage_engine(
                scenario,
                V2FormalTrainingConfig(
                    stage='easy', replay_capacity=8, batch_size=2
                ),
                init_checkpoint=checkpoint,
                rewards=RewardConfig(),
                model_type='snn',
                snn_time_window=2,
                device='cpu',
            )
            self.assertIsInstance(components.engine.actor, V2SNNPolicyActor)
            self.assertIsInstance(components.engine.actor_target, V2SNNPolicyActor)
            self.assertEqual(components.engine.model_type, 'snn')
            self.assertTrue(all(
                not parameter.requires_grad
                for parameter in components.engine.bc_reference_actor.parameters()
            ))

            with self.assertRaisesRegex(ValueError, 'ANN|format|model'):
                build_v2_stage_engine(
                    scenario,
                    V2FormalTrainingConfig(
                        stage='easy', replay_capacity=8, batch_size=2
                    ),
                    init_checkpoint=checkpoint,
                    model_type='ann',
                )

    def test_medium_handoff_preserves_snn_networks_and_resets_optimizer_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bc_checkpoint, scenario = self.make_snn_bc(root)
            easy_config = V2FormalTrainingConfig(
                stage='easy', replay_capacity=8, batch_size=2
            )
            easy = build_v2_stage_engine(
                scenario,
                easy_config,
                init_checkpoint=bc_checkpoint,
                rewards=RewardConfig(),
                model_type='snn',
                snn_time_window=2,
            )
            with torch.no_grad():
                for parameter in easy.engine.actor.parameters():
                    parameter.add_(0.01)
            payload = build_v2_formal_checkpoint(
                easy.engine,
                self.passed_result('easy'),
                easy_config,
                scenario=scenario,
                rewards=RewardConfig(),
                uav_collision_radius=0.0,
                seed_manifest=easy.seed_manifest,
                validation_pool_metadata=self.validation_metadata('easy'),
                initialization_source=easy.initialization_source,
            )
            self.assertEqual(payload['format'], V2_SNN_FORMAL_CHECKPOINT_FORMAT)
            self.assertEqual(payload['model_type'], 'snn')
            medium = build_v2_stage_engine(
                scenario,
                V2FormalTrainingConfig(
                    stage='medium', replay_capacity=8, batch_size=2
                ),
                init_checkpoint=payload,
                rewards=RewardConfig(),
                model_type='snn',
                snn_time_window=2,
            )
            self.assertEqual(len(medium.engine.replay), 0)
            self.assertEqual(medium.engine.actor_optimizer.state, {})
            self.assertEqual(medium.engine.critic_optimizer.state, {})
            for name, value in easy.engine.actor.state_dict().items():
                self.assertTrue(torch.equal(value, medium.engine.actor.state_dict()[name]))
            self.assertIsInstance(medium.engine.bc_reference_actor, V2SNNPolicyActor)
            for name, value in medium.engine.actor.state_dict().items():
                self.assertTrue(
                    torch.equal(value, medium.engine.bc_reference_actor.state_dict()[name])
                )

            with self.assertRaisesRegex(ValueError, 'model|format'):
                load_v2_formal_checkpoint(payload, expected_model_type='ann')

    def test_same_seed_build_and_extremely_short_structured_loop_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, scenario = self.make_snn_bc(Path(directory))
            config = V2FormalTrainingConfig(
                stage='easy',
                seed=41,
                max_steps=2,
                replay_capacity=8,
                batch_size=1,
                warmup_steps=0,
                actor_freeze_steps=0,
                policy_delay=1,
                early_stop_min_steps=99,
                window_episode_count=2,
                consecutive_qualified_windows=2,
                validation_max_failures=0,
            )
            first = build_v2_stage_engine(
                scenario,
                config,
                init_checkpoint=checkpoint,
                model_type='snn',
                snn_time_window=2,
            )
            second = build_v2_stage_engine(
                scenario,
                config,
                init_checkpoint=checkpoint,
                model_type='snn',
                snn_time_window=2,
            )
            for left, right in (
                (first.engine.actor, second.engine.actor),
                (first.engine.critic1, second.engine.critic1),
                (first.engine.critic2, second.engine.critic2),
            ):
                for name, value in left.state_dict().items():
                    self.assertTrue(torch.equal(value, right.state_dict()[name]))
            self.assertEqual(first.selector.sample(), second.selector.sample())
            self.assertEqual(
                first.exploration_rng.normal(size=4).tolist(),
                second.exploration_rng.normal(size=4).tolist(),
            )
            self.assertEqual(
                first.engine.replay.rng.integers(0, 1000, size=4).tolist(),
                second.engine.replay.rng.integers(0, 1000, size=4).tolist(),
            )

            trainer = V2FormalStageTrainer(
                scenario,
                RewardConfig(),
                config,
                first.engine,
                scenario_sources={'easy': _Source(make_scenario_payload(0, 0))},
                validation_runner=lambda actor: self.fail(
                    'fixed validation must not run in this two-step smoke loop'
                ),
                selector=first.selector,
                exploration_rng=first.exploration_rng,
            )
            result = trainer.run()
            self.assertEqual(result.stage_steps, 2)
            self.assertGreaterEqual(result.update_count, 1)
            self.assertEqual(first.engine.actor.snn_head.lif1.v, 0.0)


if __name__ == '__main__':
    unittest.main()
