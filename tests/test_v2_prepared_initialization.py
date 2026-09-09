"""Focused binding tests for prepared formal V2 stage initialization."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest import mock
import os
import tempfile
import unittest

import torch

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.models import V2ANNPolicyActor
from brain_uav.observations import V2ObservationScales
from brain_uav.scripts.train_v2_td3 import run_v2_td3_stage
from brain_uav.trainers.v2_formal_training import (
    V2BCFormalInitialization,
    V2FormalTrainingConfig,
    V2_FORMAL_CHECKPOINT_FORMAT,
    V2_FORMAL_CHECKPOINT_VERSION,
    build_v2_stage_engine,
    prepare_v2_stage_initialization,
    validate_v2_prepared_stage_initialization,
)
from brain_uav.trainers.v2_validation import scenario_config_snapshot
from brain_uav.v2_curriculum import derive_v2_component_seed


def _config(stage: str = 'easy', *, seed: int = 7) -> V2FormalTrainingConfig:
    return V2FormalTrainingConfig(
        stage=stage,
        seed=seed,
        max_steps=1,
        replay_capacity=8,
        batch_size=2,
    )


def _ann_actor(
    scenario: ScenarioConfig,
    *,
    uav_collision_radius: float,
) -> V2ANNPolicyActor:
    return V2ANNPolicyActor(
        V2ObservationScales(
            scenario.world_xy,
            scenario.world_z_min,
            scenario.world_z_max,
            scenario.gamma_max,
        ),
        action_dim=2,
        hidden_dim=8,
        action_limit=torch.tensor(
            [scenario.delta_gamma_max, scenario.delta_psi_max],
            dtype=torch.float32,
        ),
        uav_radius=uav_collision_radius,
    )


class TestV2PreparedStageInitializationBinding(unittest.TestCase):
    def _prepare_easy(self, source: Path, *, seed: int = 7):
        scenario = ScenarioConfig(target_distance=1_701.0)
        radius = 0.25
        initialization = V2BCFormalInitialization(
            actor=_ann_actor(scenario, uav_collision_radius=radius),
            scenario_config=scenario,
            uav_collision_radius=radius,
            model_type='ann',
        )
        loader = mock.patch(
            'brain_uav.trainers.v2_formal_training.load_v2_bc_formal_initialization',
            return_value=initialization,
        )
        mocked_loader = loader.start()
        self.addCleanup(loader.stop)
        prepared = prepare_v2_stage_initialization(
            _config(seed=seed),
            init_checkpoint=source,
            device='cpu',
        )
        return prepared, scenario, radius, mocked_loader

    def test_same_stage_seed_and_equivalent_path_reuses_verified_source(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / 'bc.pt'
            source.touch()
            prepared, scenario, radius, _ = self._prepare_easy(source)
            relative_source = Path(os.path.relpath(source, Path.cwd()))

            validate_v2_prepared_stage_initialization(
                prepared,
                _config(),
                init_checkpoint=relative_source,
                scenario=scenario,
                rewards=RewardConfig(),
                uav_collision_radius=radius,
                model_type='ann',
            )

        self.assertEqual(prepared.stage, 'easy')
        self.assertEqual(
            prepared.model_seed,
            derive_v2_component_seed(7, 'easy', 'model'),
        )
        self.assertEqual(prepared.verified_initialization_source, source.resolve())

    def test_seed_stage_and_different_source_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / 'a.pt'
            other = root / 'b.pt'
            source.touch()
            other.touch()
            prepared, _, _, _ = self._prepare_easy(source)

            with self.assertRaisesRegex(ValueError, 'model seed'):
                build_v2_stage_engine(
                    None,
                    _config(seed=8),
                    init_checkpoint=source,
                    prepared_initialization=prepared,
                )
            with self.assertRaisesRegex(ValueError, 'stage'):
                build_v2_stage_engine(
                    None,
                    _config('medium'),
                    init_checkpoint=source,
                    prepared_initialization=prepared,
                )
            with self.assertRaisesRegex(ValueError, 'initialization source'):
                build_v2_stage_engine(
                    None,
                    _config(),
                    init_checkpoint=other,
                    prepared_initialization=prepared,
                )

    def test_in_memory_source_is_bound_to_the_same_mapping_object(self):
        scenario = ScenarioConfig()
        rewards = RewardConfig()
        source: dict[str, object] = {'source': 'identity'}
        wrapper = {
            'format': V2_FORMAL_CHECKPOINT_FORMAT,
            'format_version': V2_FORMAL_CHECKPOINT_VERSION,
            'stage': 'easy',
            'status': 'passed',
            'passed_validation': True,
            'scenario_config': scenario_config_snapshot(scenario),
            'reward_config': asdict(rewards),
            'uav_collision_radius': 0.0,
        }
        with mock.patch(
            'brain_uav.trainers.v2_formal_training.load_v2_formal_checkpoint',
            return_value=wrapper,
        ):
            prepared = prepare_v2_stage_initialization(
                _config('medium'),
                init_checkpoint=source,
                device='cpu',
            )

        validate_v2_prepared_stage_initialization(
            prepared,
            _config('medium'),
            init_checkpoint=source,
            model_type='ann',
        )
        with self.assertRaisesRegex(ValueError, 'initialization source'):
            validate_v2_prepared_stage_initialization(
                prepared,
                _config('medium'),
                init_checkpoint=dict(source),
                model_type='ann',
            )

    def test_cached_formal_checkpoint_must_remain_passed_direct_predecessor(self):
        scenario = ScenarioConfig()
        rewards = RewardConfig()
        source: dict[str, object] = {'source': 'identity'}

        def prepare_wrapper():
            wrapper = {
                'format': V2_FORMAL_CHECKPOINT_FORMAT,
                'format_version': V2_FORMAL_CHECKPOINT_VERSION,
                'stage': 'easy',
                'status': 'passed',
                'passed_validation': True,
                'scenario_config': scenario_config_snapshot(scenario),
                'reward_config': asdict(rewards),
                'uav_collision_radius': 0.0,
            }
            with mock.patch(
                'brain_uav.trainers.v2_formal_training.load_v2_formal_checkpoint',
                return_value=wrapper,
            ):
                prepared = prepare_v2_stage_initialization(
                    _config('medium'),
                    init_checkpoint=source,
                    device='cpu',
                )
            return prepared, wrapper

        prepared, wrapper = prepare_wrapper()
        wrapper['status'] = 'failed'
        wrapper['passed_validation'] = False
        with self.assertRaisesRegex(ValueError, 'passed validation'):
            build_v2_stage_engine(
                None,
                _config('medium'),
                init_checkpoint=source,
                prepared_initialization=prepared,
            )

        prepared, wrapper = prepare_wrapper()
        wrapper['stage'] = 'hard'
        with self.assertRaisesRegex(ValueError, 'predecessor'):
            build_v2_stage_engine(
                None,
                _config('medium'),
                init_checkpoint=source,
                prepared_initialization=prepared,
            )

    def test_correct_reuse_does_not_reload_and_reproduces_rng_and_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            source = Path(directory) / 'bc.pt'
            source.touch()
            prepared, scenario, radius, loader = self._prepare_easy(source)
            relative_source = Path(os.path.relpath(source, Path.cwd()))

            first = build_v2_stage_engine(
                scenario,
                _config(),
                init_checkpoint=relative_source,
                rewards=RewardConfig(),
                uav_collision_radius=radius,
                device='cpu',
                prepared_initialization=prepared,
            )
            first_rng_state = torch.get_rng_state().clone()
            second = build_v2_stage_engine(
                scenario,
                _config(),
                init_checkpoint=source,
                rewards=RewardConfig(),
                uav_collision_radius=radius,
                device='cpu',
                prepared_initialization=prepared,
            )

        self.assertEqual(loader.call_count, 1)
        self.assertTrue(torch.equal(first_rng_state, torch.get_rng_state()))
        for first_model, second_model in (
            (first.engine.actor, second.engine.actor),
            (first.engine.critic1, second.engine.critic1),
            (first.engine.critic2, second.engine.critic2),
        ):
            for name, value in first_model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, second_model.state_dict()[name]),
                    msg=name,
                )
        self.assertEqual(first.seed_manifest, second.seed_manifest)
        self.assertEqual(
            first.initialization_source['path'],
            str(source.resolve()),
        )
        self.assertEqual(
            second.initialization_source['path'],
            str(source.resolve()),
        )

    def test_stage_entry_rejects_bad_prepared_seed_before_validation_or_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = root / 'bc.pt'
            source.touch()
            prepared, _, _, _ = self._prepare_easy(source, seed=7)
            output = root / 'run' / 'easy.pt'
            metrics = root / 'run' / 'metrics.json'
            with mock.patch(
                'brain_uav.scripts.train_v2_td3.load_v2_validation_pool'
            ) as pool_loader, mock.patch(
                'brain_uav.scripts.train_v2_td3.build_v2_stage_engine'
            ) as engine_builder:
                with self.assertRaisesRegex(ValueError, 'model seed'):
                    run_v2_td3_stage(
                        stage='easy',
                        init_checkpoint=source,
                        output=output,
                        metrics_out=metrics,
                        validation_pool=root / 'validation.json',
                        seed=8,
                        device='cpu',
                        prepared_initialization=prepared,
                    )

        pool_loader.assert_not_called()
        engine_builder.assert_not_called()
        self.assertFalse(output.parent.exists())


if __name__ == '__main__':
    unittest.main()
