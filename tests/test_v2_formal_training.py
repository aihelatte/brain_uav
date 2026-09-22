"""Tests for the bounded formal V2 ANN TD3 stage trainer."""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brain_uav.config import RewardConfig, ScenarioConfig, TrainingConfig
from brain_uav.envs import V2_ENV_SCENARIO_FORMAT, V2_ENV_SCENARIO_VERSION
from brain_uav.geometry import NoFlyZone, Sphere
from brain_uav.models import V2ANNCritic, V2ANNPolicyActor
from brain_uav.observations import V2ObservationScales, build_v2_observation
from brain_uav.scripts.train_v2_bc import train_v2_behavior_cloning
from brain_uav.trainers import V2ReplayBuffer, V2TD3UpdateEngine
from brain_uav.trainers.v2_formal_training import (
    V2BCFormalInitialization,
    V2PreparedStageInitialization,
    V2_FORMAL_CHECKPOINT_FORMAT,
    V2EarlyStopController,
    V2FormalStageTrainer,
    V2FormalTrainingConfig,
    V2FormalTrainingResult,
    build_v2_formal_checkpoint,
    build_v2_stage_engine,
    load_v2_bc_formal_initialization,
    load_v2_formal_checkpoint,
    prepare_v2_stage_initialization,
    save_v2_formal_checkpoint,
)
from brain_uav.trainers.v2_validation import V2ValidationResult
from brain_uav.trainers.v2_reporting import V2ExperimentReporter
from brain_uav.trainers.v2_bc import (
    V2_BC_CHECKPOINT_FORMAT,
    V2_BC_CHECKPOINT_VERSION,
)
from test_v2_bc import make_scenario_config, write_cluster
from brain_uav.v2_curriculum import derive_v2_component_seed


def _scenario_payload(zone_count: int, *, goal: bool = False) -> dict:
    zones = [
        NoFlyZone(
            f'z{index}',
            Sphere([500.0, -500.0 + 5.0 * index, 100.0], 1.0),
        ).to_dict()
        for index in range(zone_count)
    ]
    return {
        'format': V2_ENV_SCENARIO_FORMAT,
        'format_version': V2_ENV_SCENARIO_VERSION,
        'state': [0.0, 0.0, 100.0, 0.0, 0.0],
        'goal': [1.0 if goal else 100.0, 0.0, 100.0],
        'zones': zones,
        'curriculum_level': 'easy',
        'metadata': {'scenario_seed': 5, 'requested_zone_count': zone_count},
    }


class _Source:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def generate(self):
        self.calls += 1
        return self.payload


def _engine(scenario: ScenarioConfig, *, seed: int = 3, bc_reference=None):
    torch.manual_seed(seed)
    scales = V2ObservationScales(
        scenario.world_xy,
        scenario.world_z_min,
        scenario.world_z_max,
        scenario.gamma_max,
    )
    limit = torch.tensor(
        [scenario.delta_gamma_max, scenario.delta_psi_max], dtype=torch.float32
    )
    actor = V2ANNPolicyActor(scales, 2, 8, limit)
    critic1 = V2ANNCritic(scales, 2, 8)
    critic2 = V2ANNCritic(scales, 2, 8)
    replay = V2ReplayBuffer(16, 2, 6, success_sample_bias=1.0, seed=17)
    if bc_reference is None:
        bc_reference = deepcopy(actor)
    return V2TD3UpdateEngine(
        actor,
        critic1,
        critic2,
        replay,
        1e-3,
        1e-3,
        0.99,
        0.005,
        0.015,
        0.03,
        2,
        2,
        -limit.numpy(),
        limit.numpy(),
        actor_freeze_steps=25_000,
        actor_grad_clip_norm=1.0,
        critic_grad_clip_norm=1.0,
        bc_reference_actor=bc_reference,
    )


def _validation_result(passed: bool) -> V2ValidationResult:
    failures = 0 if passed else 1
    return V2ValidationResult(
        curriculum_level='easy',
        scenario_count=1,
        goal_count=1 - failures,
        failure_count=failures,
        outcome_counts={
            'goal': 1 - failures,
            'ground': 0,
            'boundary': 0,
            'collision': 0,
            'timeout': failures,
        },
        scenarios=({'scenario_id': 'v0', 'outcome': 'goal' if passed else 'timeout', 'episode_length': 1, 'episode_return': 0.0},),
        max_failures=0,
        passed=passed,
    )


class TestV2FormalTraining(unittest.TestCase):
    def test_builder_forwards_fused_and_relation_flags_with_in_memory_initialization(self):
        scenario = make_scenario_config()
        config = V2FormalTrainingConfig(
            stage='easy', seed=83, max_steps=2, replay_capacity=8, batch_size=2,
        )
        source = {}
        actor = _engine(scenario).actor
        prepared = V2PreparedStageInitialization(
            stage='easy',
            model_seed=derive_v2_component_seed(83, 'easy', 'model'),
            verified_initialization_source=source,
            snn_time_window=None,
            scenario_config=scenario,
            reward_config=RewardConfig(),
            uav_collision_radius=0.0,
            model_type='ann',
            bc_initialization=V2BCFormalInitialization(
                actor=actor, scenario_config=scenario,
                uav_collision_radius=0.0, model_type='ann',
            ),
            formal_checkpoint=None,
            torch_rng_state=torch.get_rng_state().clone(),
        )
        eager = build_v2_stage_engine(
            scenario, config, init_checkpoint=source,
            prepared_initialization=prepared, device='cpu',
        ).engine
        optimized = build_v2_stage_engine(
            scenario, config, init_checkpoint=source,
            prepared_initialization=prepared, device='cpu',
            fused_adam=True, aggregate_relation_values_first=True,
            reduce_update_stat_syncs=True,
        ).engine
        self.assertTrue(optimized.actor_optimizer.param_groups[0]['fused'])
        self.assertTrue(optimized.critic_optimizer.param_groups[0]['fused'])
        self.assertTrue(optimized.reduce_update_stat_syncs)
        self.assertTrue(all(
            layer.attention.aggregate_relation_values_first
            for layer in optimized.actor.zone_set_encoder.layers
        ))
        self.assertEqual(
            tuple(optimized.checkpoint_state_dict()),
            tuple(eager.checkpoint_state_dict()),
        )

    @staticmethod
    def _make_v2_bc_checkpoint(
        root: Path,
        *,
        scenario: ScenarioConfig | None = None,
        uav_collision_radius: float = 0.0,
    ) -> tuple[Path, ScenarioConfig]:
        effective_scenario = make_scenario_config() if scenario is None else scenario
        cluster_path = write_cluster(
            root / 'cluster',
            zone_counts=(0, 1, 2, 0),
            scenario_config=effective_scenario,
            uav_collision_radius=uav_collision_radius,
        )
        train_v2_behavior_cloning(
            trajectory_cluster=cluster_path,
            output_dir=root / 'bc',
            seed=19,
            validation_fraction=0.25,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            training_config=TrainingConfig(hidden_dim=8, bc_epochs=1, batch_size=2),
            device='cpu',
        )
        return root / 'bc' / 'bc_v2_ann_best.pt', effective_scenario

    def test_stage_initialization_derives_upstream_configs_and_rejects_explicit_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = replace(
                make_scenario_config(),
                target_distance=41.0,
                warning_distance=17.0,
            )
            checkpoint, _ = self._make_v2_bc_checkpoint(
                root,
                scenario=scenario,
                uav_collision_radius=0.75,
            )
            easy_config = V2FormalTrainingConfig(
                stage='easy', max_steps=1, replay_capacity=8, batch_size=2
            )

            prepared_easy = prepare_v2_stage_initialization(
                easy_config,
                init_checkpoint=checkpoint,
                scenario=None,
                rewards=None,
                uav_collision_radius=None,
                device='cpu',
                model_type='ann',
            )

            self.assertEqual(prepared_easy.scenario_config, scenario)
            self.assertEqual(prepared_easy.reward_config, RewardConfig())
            self.assertEqual(prepared_easy.uav_collision_radius, 0.75)
            directly_derived = build_v2_stage_engine(
                None,
                easy_config,
                init_checkpoint=checkpoint,
                device='cpu',
            )
            self.assertEqual(directly_derived.engine.actor.scales.world_xy, scenario.world_xy)
            self.assertEqual(directly_derived.engine.actor.uav_radius, 0.75)
            with self.assertRaisesRegex(ValueError, 'ScenarioConfig'):
                prepare_v2_stage_initialization(
                    easy_config,
                    init_checkpoint=checkpoint,
                    scenario=replace(scenario, target_distance=42.0),
                    uav_collision_radius=0.75,
                    device='cpu',
                )
            with self.assertRaisesRegex(ValueError, 'uav_collision_radius'):
                prepare_v2_stage_initialization(
                    easy_config,
                    init_checkpoint=checkpoint,
                    scenario=scenario,
                    uav_collision_radius=0.0,
                    device='cpu',
                )

            components = build_v2_stage_engine(
                scenario,
                easy_config,
                init_checkpoint=checkpoint,
                rewards=RewardConfig(),
                uav_collision_radius=0.75,
                device='cpu',
                prepared_initialization=prepared_easy,
            )
            result = V2FormalTrainingResult.empty('easy', global_steps_start=0)
            result.status = 'passed'
            result.passed_validation = True
            result.stop_reason = 'fixed_validation_passed'
            result.validation_records.append(_validation_result(True).to_dict())
            rewards = replace(RewardConfig(), progress_weight=3.25)
            previous = build_v2_formal_checkpoint(
                components.engine,
                result,
                easy_config,
                scenario=scenario,
                rewards=rewards,
                uav_collision_radius=0.75,
                seed_manifest={'base_seed': 7},
                validation_pool_metadata={
                    'path': 'easy.json',
                    'format_version': 1,
                    'curriculum_level': 'easy',
                    'master_seed': 1,
                    'stage_seed': 2,
                    'scenario_count': 100,
                    'content_digest': 'x',
                },
                initialization_source={'kind': 'v2_bc_best', 'path': str(checkpoint)},
            )
            medium_config = V2FormalTrainingConfig(
                stage='medium', max_steps=1, replay_capacity=8, batch_size=2
            )
            prepared_medium = prepare_v2_stage_initialization(
                medium_config,
                init_checkpoint=previous,
                scenario=None,
                rewards=None,
                uav_collision_radius=None,
                device='cpu',
            )

            self.assertEqual(prepared_medium.scenario_config, scenario)
            self.assertEqual(prepared_medium.reward_config, rewards)
            self.assertEqual(prepared_medium.uav_collision_radius, 0.75)
            with self.assertRaisesRegex(ValueError, 'RewardConfig'):
                prepare_v2_stage_initialization(
                    medium_config,
                    init_checkpoint=previous,
                    scenario=scenario,
                    rewards=RewardConfig(),
                    uav_collision_radius=0.75,
                    device='cpu',
                )

    def test_config_defaults_and_strict_validation(self):
        easy = V2FormalTrainingConfig(stage='easy')
        hard = V2FormalTrainingConfig(stage='hard')
        self.assertEqual(easy.max_steps, 750_000)
        self.assertEqual((easy.actor_lr, easy.critic_lr), (1.5e-4, 2.5e-4))
        self.assertEqual(hard.max_steps, 1_000_000)
        self.assertEqual((hard.actor_lr, hard.critic_lr), (1.125e-4, 2.125e-4))
        self.assertEqual(easy.success_sample_bias, 1.0)
        self.assertEqual(easy.near_goal_sample_bias, 2.0)
        self.assertEqual(easy.zone_storage_capacity, 6)
        self.assertEqual(easy.warmup_strategy, 'policy_with_noise')
        for kwargs in (
            {'stage': 'easy_two_zone'},
            {'stage': 'easy', 'success_sample_bias': 0.5},
            {'stage': 'easy', 'success_batch_fraction': 1.1},
            {'stage': 'easy', 'window_episode_count': 0},
            {'stage': 'easy', 'max_failures_per_window': 15},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                V2FormalTrainingConfig(**kwargs)

    def test_early_stop_uses_only_failures_and_validation_gate(self):
        controller = V2EarlyStopController(
            window_episode_count=3,
            max_failures_per_window=1,
            consecutive_qualified_windows=2,
            early_stop_min_steps=5,
        )
        self.assertIsNone(controller.add_episode('goal', stage_steps=1))
        self.assertIsNone(controller.add_episode('collision', stage_steps=2))
        first = controller.add_episode('goal', stage_steps=5)
        self.assertTrue(first['qualified'])
        self.assertFalse(first['candidate'])
        for outcome in ('goal', 'goal'):
            self.assertIsNone(controller.add_episode(outcome, stage_steps=6))
        second = controller.add_episode('timeout', stage_steps=7)
        self.assertTrue(second['candidate'])
        controller.record_validation(False)
        self.assertEqual(controller.consecutive_qualified, 0)
        self.assertEqual(controller.candidate_count, 1)
        self.assertNotIn('goal_rate_threshold', second)

    def test_short_formal_loop_uses_policy_warmup_updates_critic_and_handles_dynamic_counts(self):
        for zone_count in (0, 2, 5, 6):
            with self.subTest(zone_count=zone_count):
                scenario = ScenarioConfig(max_steps=1)
                engine = _engine(scenario)
                config = V2FormalTrainingConfig(
                    stage='easy',
                    max_steps=3,
                    replay_capacity=16,
                    batch_size=2,
                    warmup_steps=3,
                    actor_freeze_steps=25_000,
                    window_episode_count=2,
                    consecutive_qualified_windows=2,
                    early_stop_min_steps=2,
                )
                source = _Source(_scenario_payload(zone_count))
                actor_calls = 0
                original_select = engine.select_action

                def counted(*args, **kwargs):
                    nonlocal actor_calls
                    actor_calls += 1
                    self.assertIn('exploration_rng', kwargs)
                    return original_select(*args, **kwargs)

                engine.select_action = counted
                trainer = V2FormalStageTrainer(
                    scenario,
                    RewardConfig(),
                    config,
                    engine,
                    scenario_sources={'easy': source},
                    validation_runner=lambda actor: _validation_result(False),
                )
                result = trainer.run()
                self.assertEqual(result.stage_steps, 3)
                self.assertEqual(len(engine.replay), 3)
                self.assertGreaterEqual(engine.critic_update_count, 1)
                self.assertEqual(engine.actor_update_count, 0)
                self.assertEqual(actor_calls, 3)
                self.assertEqual(
                    sum(item['policy_warmup_steps'] for item in result.episodes),
                    3,
                )
                self.assertEqual(result.curriculum_sample_counts, {'easy': 3})
                self.assertEqual(result.stop_reason, 'max_steps_without_validation')

    def test_formal_trainer_adds_stage_context_to_numerical_failure(self):
        scenario = ScenarioConfig(max_steps=1)
        engine = _engine(scenario)
        config = V2FormalTrainingConfig(
            stage='easy',
            max_steps=2,
            replay_capacity=8,
            batch_size=2,
            window_episode_count=2,
            consecutive_qualified_windows=2,
            early_stop_min_steps=99,
        )
        trainer = V2FormalStageTrainer(
            scenario,
            RewardConfig(),
            config,
            engine,
            scenario_sources={'easy': _Source(_scenario_payload(0))},
            validation_runner=lambda actor: _validation_result(False),
        )
        with mock.patch.object(
            engine,
            'update_once',
            side_effect=FloatingPointError(
                'Non-finite critic gradient at total_steps=2.'
            ),
        ) as update:
            with self.assertRaisesRegex(
                FloatingPointError,
                'stage=easy.*stage_steps=2.*critic gradient',
            ):
                trainer.run()
        update.assert_called_once()

    def test_goal_is_only_success_and_goal_episode_populates_both_replays(self):
        scenario = ScenarioConfig(max_steps=1)
        engine = _engine(scenario)
        config = V2FormalTrainingConfig(
            stage='easy',
            max_steps=1,
            replay_capacity=8,
            batch_size=2,
            window_episode_count=1,
            max_failures_per_window=0,
            consecutive_qualified_windows=1,
            early_stop_min_steps=99,
        )
        trainer = V2FormalStageTrainer(
            scenario,
            RewardConfig(),
            config,
            engine,
            scenario_sources={'easy': _Source(_scenario_payload(0, goal=True))},
            validation_runner=lambda actor: _validation_result(True),
        )
        result = trainer.run()
        self.assertEqual(result.outcome_counts['goal'], 1)
        self.assertEqual(engine.replay.success_count, 1)
        self.assertEqual(engine.replay.success_size, 1)

    def test_line_to_goal_safety_is_measured_before_the_action(self):
        scenario = ScenarioConfig(max_steps=1)
        engine = _engine(scenario)
        config = V2FormalTrainingConfig(
            stage='easy',
            max_steps=1,
            replay_capacity=8,
            batch_size=2,
            window_episode_count=1,
            max_failures_per_window=0,
            consecutive_qualified_windows=1,
            early_stop_min_steps=99,
        )
        trainer = V2FormalStageTrainer(
            scenario,
            RewardConfig(),
            config,
            engine,
            scenario_sources={'easy': _Source(_scenario_payload(0))},
            validation_runner=lambda actor: _validation_result(False),
        )
        events = []
        original_safe = trainer.env.line_to_goal_is_safe
        original_step = trainer.env.step

        def safe(*args, **kwargs):
            events.append('safe')
            return original_safe(*args, **kwargs)

        def step(action):
            events.append('step')
            return original_step(action)

        trainer.env.line_to_goal_is_safe = safe
        trainer.env.step = step
        trainer.run()
        first_step = events.index('step')
        self.assertGreater(first_step, 0)
        self.assertEqual(events[first_step - 1:first_step + 1], ['safe', 'step'])
        self.assertEqual(engine.replay.line_to_goal_safe[0], True)

    def test_same_seed_reproduces_short_cpu_training_state(self):
        scenario = ScenarioConfig(max_steps=1)
        config = V2FormalTrainingConfig(
            stage='easy',
            seed=41,
            max_steps=3,
            replay_capacity=16,
            batch_size=2,
            window_episode_count=2,
            consecutive_qualified_windows=2,
            early_stop_min_steps=99,
        )

        def run_once():
            engine = _engine(scenario, seed=41)
            trainer = V2FormalStageTrainer(
                scenario,
                RewardConfig(),
                config,
                engine,
                scenario_sources={'easy': _Source(_scenario_payload(2))},
                validation_runner=lambda actor: _validation_result(False),
            )
            result = trainer.run()
            return result, {
                name: value.detach().clone()
                for name, value in engine.actor.state_dict().items()
            }, {
                name: value.detach().clone()
                for name, value in engine.critic1.state_dict().items()
            }

        first_result, first_actor, first_critic = run_once()
        second_result, second_actor, second_critic = run_once()
        self.assertEqual(first_result.to_dict(), second_result.to_dict())
        for name in first_actor:
            self.assertTrue(torch.equal(first_actor[name], second_actor[name]))
        for name in first_critic:
            self.assertTrue(torch.equal(first_critic[name], second_critic[name]))

    def test_reporting_does_not_change_training_or_owned_random_state(self):
        scenario = ScenarioConfig(max_steps=1)
        config = V2FormalTrainingConfig(
            stage='easy', seed=43, max_steps=3, replay_capacity=16,
            batch_size=2, window_episode_count=2,
            consecutive_qualified_windows=2, early_stop_min_steps=99,
        )

        def run_once(report_directory: Path | None):
            engine = _engine(scenario, seed=43)
            reporter = None
            if report_directory is not None:
                reporter = V2ExperimentReporter(
                    report_directory,
                    stage='easy',
                    model_type='ann',
                    scenario=scenario,
                    rewards=RewardConfig(),
                    uav_collision_radius=0.0,
                    max_steps=config.max_steps,
                )
            trainer = V2FormalStageTrainer(
                scenario,
                RewardConfig(),
                config,
                engine,
                scenario_sources={'easy': _Source(_scenario_payload(2))},
                validation_runner=lambda actor: _validation_result(False),
                reporter=reporter,
            )
            try:
                result = trainer.run()
                if reporter is not None:
                    reporter.finish_stage(result.to_dict())
            finally:
                if reporter is not None:
                    reporter.close()
            return (
                result.to_dict(),
                {name: value.detach().clone() for name, value in engine.actor.state_dict().items()},
                {name: value.detach().clone() for name, value in engine.critic1.state_dict().items()},
                engine.replay.action.copy(),
                deepcopy(trainer.exploration_rng.bit_generator.state),
                deepcopy(engine.replay.rng.bit_generator.state),
            )

        with tempfile.TemporaryDirectory() as directory:
            without = run_once(None)
            with_report = run_once(Path(directory) / 'report')
        self.assertEqual(without[0], with_report[0])
        for first, second in zip(without[1:3], with_report[1:3]):
            for name in first:
                self.assertTrue(torch.equal(first[name], second[name]), name)
        np.testing.assert_array_equal(without[3], with_report[3])
        self.assertEqual(without[4], with_report[4])
        self.assertEqual(without[5], with_report[5])

    def test_reported_training_time_excludes_plotting_and_validation_but_stage_time_includes_them(self):
        class Clock:
            value = 0.0

            def __call__(self):
                return self.value

        class TimedSource(_Source):
            def __init__(self, payload, clock):
                super().__init__(payload)
                self.clock = clock

            def generate(self):
                self.clock.value += 2.0
                return super().generate()

        clock = Clock()
        scenario = ScenarioConfig(max_steps=1)
        config = V2FormalTrainingConfig(
            stage='easy',
            seed=47,
            max_steps=2,
            replay_capacity=16,
            batch_size=2,
            window_episode_count=1,
            max_failures_per_window=0,
            consecutive_qualified_windows=1,
            early_stop_min_steps=1,
        )
        engine = _engine(scenario, seed=47)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = V2ExperimentReporter(
                root / 'report',
                stage='easy',
                model_type='ann',
                scenario=scenario,
                rewards=RewardConfig(),
                uav_collision_radius=0.0,
                max_steps=config.max_steps,
                clock=clock,
            )

            def validation(actor):
                reporter.begin_validation(curriculum_level='easy', scenario_count=1)
                clock.value += 20.0
                result = _validation_result(False)
                reporter.finish_validation(result.to_dict())
                return result

            trainer = V2FormalStageTrainer(
                scenario,
                RewardConfig(),
                config,
                engine,
                scenario_sources={
                    'easy': TimedSource(_scenario_payload(0, goal=True), clock)
                },
                validation_runner=validation,
                reporter=reporter,
            )
            original_step = trainer.env.step

            def timed_step(action):
                result = original_step(action)
                clock.value += 3.0
                return result

            trainer.env.step = timed_step

            def timed_plot(*args, **kwargs):
                clock.value += 10.0
                return {'json': 'unused.json', 'png': 'unused.png'}

            try:
                with mock.patch(
                    'brain_uav.trainers.v2_reporting.export_v2_trajectory_views',
                    side_effect=timed_plot,
                ):
                    result = trainer.run()
                    reporter.finish_stage(result.to_dict())
            finally:
                reporter.close()

            episode_rows = [
                json.loads(line)
                for line in (root / 'report' / 'episodes.jsonl')
                .read_text(encoding='utf-8')
                .splitlines()
            ]
            window_rows = [
                json.loads(line)
                for line in (root / 'report' / 'windows.jsonl')
                .read_text(encoding='utf-8')
                .splitlines()
            ]
            stage_end = json.loads(
                (root / 'report' / 'stage_end.json').read_text(encoding='utf-8')
            )

        self.assertEqual(
            [row['episode_elapsed_seconds'] for row in episode_rows],
            [5.0, 5.0],
        )
        self.assertEqual(
            [row['window_elapsed_seconds'] for row in window_rows],
            [5.0, 5.0],
        )
        self.assertEqual(stage_end['stage_elapsed_seconds'], 70.0)

    def test_candidate_validation_failure_continues_and_success_stops(self):
        scenario = ScenarioConfig(max_steps=1)
        engine = _engine(scenario)
        config = V2FormalTrainingConfig(
            stage='easy',
            max_steps=4,
            replay_capacity=16,
            batch_size=2,
            window_episode_count=1,
            max_failures_per_window=0,
            consecutive_qualified_windows=1,
            early_stop_min_steps=1,
        )
        results = iter((_validation_result(False), _validation_result(True)))
        trainer = V2FormalStageTrainer(
            scenario,
            RewardConfig(),
            config,
            engine,
            scenario_sources={'easy': _Source(_scenario_payload(0, goal=True))},
            validation_runner=lambda actor: next(results),
        )
        result = trainer.run()
        self.assertTrue(result.passed_validation)
        self.assertEqual(result.stage_steps, 2)
        self.assertEqual(result.candidate_validation_count, 2)
        self.assertEqual(result.stop_reason, 'fixed_validation_passed')
        self.assertEqual(len(result.windows), 2)

    def test_formal_checkpoint_excludes_replay_and_rejects_failed_or_wrong_predecessor(self):
        scenario = ScenarioConfig(max_steps=1)
        engine = _engine(scenario)
        config = V2FormalTrainingConfig(stage='easy', max_steps=1)
        result = V2FormalTrainingResult.empty('easy', global_steps_start=10)
        result.status = 'passed'
        result.passed_validation = True
        result.stop_reason = 'fixed_validation_passed'
        result.validation_records.append(_validation_result(True).to_dict())
        payload = build_v2_formal_checkpoint(
            engine,
            result,
            config,
            scenario=scenario,
            rewards=RewardConfig(),
            uav_collision_radius=0.0,
            seed_manifest={'base_seed': 7},
            validation_pool_metadata={
                'path': 'easy.json',
                'format_version': 1,
                'curriculum_level': 'easy',
                'master_seed': 20260904,
                'stage_seed': 11,
                'scenario_count': 100,
                'content_digest': 'abc',
            },
            initialization_source={'kind': 'v2_bc_best', 'path': 'bc.pt'},
        )
        self.assertEqual(payload['format'], V2_FORMAL_CHECKPOINT_FORMAT)
        self.assertNotIn('replay', payload)
        self.assertFalse(any('zone_features' in key for key in payload))
        self.assertEqual(payload['scenario_config']['max_steps'], 1)
        self.assertEqual(payload['reward_config']['collision_penalty'], 24_000.0)
        self.assertEqual(payload['seed_manifest'], {'base_seed': 7})
        self.assertEqual(
            payload['bc_schedule']['boundaries'],
            [0, 75_000, 150_000, 250_000],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'easy.pt'
            save_v2_formal_checkpoint(path, payload)
            loaded = load_v2_formal_checkpoint(path, expected_stage='easy', require_passed=True)
        self.assertTrue(loaded['passed_validation'])

        failed = dict(payload)
        failed['status'] = 'failed'
        failed['passed_validation'] = False
        with self.assertRaisesRegex(ValueError, 'passed validation'):
            load_v2_formal_checkpoint(failed, expected_stage='easy', require_passed=True)
        with self.assertRaisesRegex(ValueError, 'predecessor'):
            load_v2_formal_checkpoint(payload, next_stage='hard', require_passed=True)

    def test_medium_handoff_inherits_networks_but_not_optimizer_replay_or_old_bc_reference(self):
        scenario = ScenarioConfig(max_steps=1)
        source = _engine(scenario, seed=9)
        source.replay.add(
            *self._transition_for_replay(scenario),
        )
        source_result = V2FormalTrainingResult.empty('easy', global_steps_start=0)
        source_result.status = 'passed'
        source_result.passed_validation = True
        source_result.stop_reason = 'fixed_validation_passed'
        source_result.validation_records.append(_validation_result(True).to_dict())
        payload = build_v2_formal_checkpoint(
            source,
            source_result,
            V2FormalTrainingConfig(
                stage='easy',
                max_steps=1,
                replay_capacity=16,
                batch_size=2,
                actor_lr=1e-3,
                critic_lr=1e-3,
            ),
            scenario=scenario,
            rewards=RewardConfig(),
            uav_collision_radius=0.0,
            seed_manifest={'base_seed': 7},
            validation_pool_metadata={
                'path': 'easy.json',
                'format_version': 1,
                'curriculum_level': 'easy',
                'master_seed': 1,
                'stage_seed': 2,
                'scenario_count': 100,
                'content_digest': 'x',
            },
            initialization_source={'kind': 'v2_bc_best', 'path': 'bc.pt'},
        )

        components = build_v2_stage_engine(
            ScenarioConfig(max_steps=1),
            V2FormalTrainingConfig(stage='medium', max_steps=1, replay_capacity=16, batch_size=2),
            init_checkpoint=payload,
            rewards=RewardConfig(),
            uav_collision_radius=0.0,
            device='cpu',
        )

        self.assertEqual(len(components.engine.replay), 0)
        self.assertFalse(components.engine.actor_optimizer.state_dict()['state'])
        self.assertFalse(components.engine.critic_optimizer.state_dict()['state'])
        for name, value in source.actor.state_dict().items():
            self.assertTrue(torch.equal(value, components.engine.actor.state_dict()[name]))
            self.assertTrue(torch.equal(value, components.engine.bc_reference_actor.state_dict()[name]))
        actor_ids = {id(value) for value in components.engine.actor.parameters()}
        reference_ids = {id(value) for value in components.engine.bc_reference_actor.parameters()}
        self.assertTrue(actor_ids.isdisjoint(reference_ids))
        self.assertTrue(all(not value.requires_grad for value in components.engine.bc_reference_actor.parameters()))

    def test_easy_rejects_non_best_or_legacy_bc_before_model_loading(self):
        scenario = ScenarioConfig(max_steps=1)
        config = V2FormalTrainingConfig(
            stage='easy',
            max_steps=1,
            replay_capacity=8,
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'final.pt'
            torch.save(
                {
                    'format': V2_BC_CHECKPOINT_FORMAT,
                    'format_version': V2_BC_CHECKPOINT_VERSION,
                    'checkpoint_kind': 'final',
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, 'BC best'):
                build_v2_stage_engine(
                    scenario,
                    config,
                    init_checkpoint=path,
                    rewards=RewardConfig(),
                )
            torch.save({'state_dict': {}}, path)
            with self.assertRaisesRegex(ValueError, 'BC best'):
                build_v2_stage_engine(
                    scenario,
                    config,
                    init_checkpoint=path,
                    rewards=RewardConfig(),
                )

    def test_easy_bc_checkpoint_binds_complete_scenario_and_collision_radius(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, scenario = self._make_v2_bc_checkpoint(root)
            config = V2FormalTrainingConfig(
                stage='easy',
                seed=37,
                max_steps=1,
                replay_capacity=8,
                batch_size=2,
            )
            initialization = load_v2_bc_formal_initialization(
                checkpoint,
                expected_scenario=scenario,
                expected_uav_collision_radius=0.0,
                device='cpu',
            )
            self.assertEqual(initialization.scenario_config, scenario)
            self.assertEqual(initialization.uav_collision_radius, 0.0)
            components = build_v2_stage_engine(
                scenario,
                config,
                init_checkpoint=checkpoint,
                rewards=RewardConfig(),
                uav_collision_radius=0.0,
                device='cpu',
            )
            self.assertEqual(components.engine.actor.hidden_dim, 8)

            mismatches = (
                replace(scenario, target_distance=scenario.target_distance + 1.0),
                replace(scenario, world_xy=scenario.world_xy + 1.0),
                replace(scenario, max_steps=scenario.max_steps + 1),
            )
            for incompatible in mismatches:
                with self.subTest(incompatible=incompatible), self.assertRaisesRegex(
                    ValueError, 'ScenarioConfig'
                ):
                    build_v2_stage_engine(
                        incompatible,
                        config,
                        init_checkpoint=checkpoint,
                        rewards=RewardConfig(),
                        uav_collision_radius=0.0,
                        device='cpu',
                    )
            with self.assertRaisesRegex(ValueError, 'uav_collision_radius'):
                build_v2_stage_engine(
                    scenario,
                    config,
                    init_checkpoint=checkpoint,
                    rewards=RewardConfig(),
                    uav_collision_radius=0.25,
                    device='cpu',
                )

            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            for name, mutation in (
                ('missing_provenance', lambda item: item.pop('dataset_provenance')),
                (
                    'missing_scenario_config',
                    lambda item: item['dataset_provenance'].pop('scenario_config'),
                ),
            ):
                altered = deepcopy(payload)
                mutation(altered)
                path = root / f'{name}.pt'
                torch.save(altered, path)
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, 'provenance|fields'
                ):
                    load_v2_bc_formal_initialization(
                        path,
                        expected_scenario=scenario,
                        expected_uav_collision_radius=0.0,
                    )

    def test_same_seed_formal_build_reproduces_models_and_owned_random_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, scenario = self._make_v2_bc_checkpoint(Path(directory))
            config = V2FormalTrainingConfig(
                stage='easy',
                seed=83,
                max_steps=2,
                replay_capacity=8,
                batch_size=2,
            )

            def build():
                return build_v2_stage_engine(
                    scenario,
                    config,
                    init_checkpoint=checkpoint,
                    rewards=RewardConfig(),
                    uav_collision_radius=0.0,
                    device='cpu',
                )

            first = build()
            second = build()
            for network_name in ('actor', 'critic1', 'critic2'):
                first_state = getattr(first.engine, network_name).state_dict()
                second_state = getattr(second.engine, network_name).state_dict()
                self.assertEqual(tuple(first_state), tuple(second_state))
                for name, value in first_state.items():
                    self.assertTrue(torch.equal(value, second_state[name]))
            self.assertEqual(first.seed_manifest, second.seed_manifest)
            self.assertEqual(
                [first.selector.sample() for _ in range(12)],
                [second.selector.sample() for _ in range(12)],
            )
            np.testing.assert_array_equal(
                first.engine.replay.rng.integers(0, 1000, size=16),
                second.engine.replay.rng.integers(0, 1000, size=16),
            )
            observation = build_v2_observation(
                [0.0, 0.0, 10.0, 0.0, 0.0],
                [5.0, 0.0, 10.0],
                [],
                first.engine.actor.scales,
            )
            first_action = first.engine.select_action(
                observation,
                exploration_noise=0.01,
                exploration_rng=first.exploration_rng,
            )
            second_action = second.engine.select_action(
                observation,
                exploration_noise=0.01,
                exploration_rng=second.exploration_rng,
            )
            np.testing.assert_array_equal(first_action, second_action)


    @staticmethod
    def _transition_for_replay(scenario):
        from brain_uav.envs import V2StaticNoFlyTrajectoryEnv
        env = V2StaticNoFlyTrajectoryEnv(scenario, RewardConfig())
        obs, _ = env.reset(options={'scenario': _scenario_payload(0)})
        return (obs, np.zeros(2, dtype=np.float32), 0.0, obs, False)


if __name__ == '__main__':
    unittest.main()
