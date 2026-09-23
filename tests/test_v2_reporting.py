"""Focused tests for V2 streaming reports and trajectory sampling."""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.image as mpimg

from brain_uav.config import RewardConfig, ScenarioConfig
from brain_uav.geometry import (
    Box,
    Ellipsoid,
    NoFlyZone,
    QuadrangularPyramid,
    Sphere,
    TriangularPyramid,
)
from brain_uav.trainers.v2_reporting import (
    V2BCTrainingReporter,
    V2ExperimentReporter,
    V2TrainingTrajectorySelector,
    V2ValidationTrajectorySelector,
    _format_duration,
    _format_scientific,
    _plot_training_windows,
    export_v2_trajectory_views,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _scenario_payload(zones: list[NoFlyZone]) -> dict:
    return {
        'format': 'v2_static_no_fly_scenario',
        'format_version': 1,
        'state': [-10.0, -10.0, 20.0, 0.0, 0.0],
        'goal': [10.0, 10.0, 30.0],
        'zones': [zone.to_dict() for zone in zones],
        'curriculum_level': 'easy',
        'metadata': {'scenario_seed': 9},
    }


def _episode(number: int, outcome: str, stage_steps: int) -> dict:
    return {
        'episode': number,
        'stage': 'easy',
        'curriculum_level': 'easy',
        'stage_steps': stage_steps,
        'global_steps': stage_steps,
        'outcome': outcome,
        'episode_return': float(number),
        'episode_length': 1,
        'policy_warmup_steps': 0,
        'critic_loss': 2.0,
        'actor_loss': 1.0,
        'bc_loss': 0.5,
        'weighted_bc_contribution': 0.25,
        'terminal_geo_loss': 0.0,
        'bc_lambda': 5.0,
        'exploration_noise': 0.01,
        'policy_noise': 0.01,
        'noise_clip': 0.02,
        'replay_size': number,
        'replay_success_fraction': 0.0,
        'success_replay_size': 0,
        'batch_success_fraction': 0.0,
        'critic_q1_mean': 10.0,
        'critic_q1_std': 2.0,
        'critic_q1_max_abs': 20.0,
        'critic_target_q_mean': 11.0,
        'critic_td_error_mean': 3.0,
        'critic_failure_td_error_mean': 4.0,
        'actor_grad_norm': 0.5,
        'actor_updated': True,
        'actor_update_status': 'updated',
    }


class TestV2TrajectorySelectors(unittest.TestCase):
    def test_progress_crossings_are_deduplicated_and_capped(self) -> None:
        selector = V2TrainingTrajectorySelector(max_steps=10)
        first = selector.select(episode=1, stage_steps=6, outcome='goal')
        self.assertEqual(first.progress_trigger_steps, (1, 2, 3, 4, 5, 6))
        self.assertIn('first_outcome:goal', first.reasons)
        self.assertEqual(selector.progress_sample_count, 1)
        for episode in range(2, 30):
            selector.select(
                episode=episode,
                stage_steps=min(10, episode + 5),
                outcome='goal',
            )
        self.assertLessEqual(selector.progress_sample_count, 20)

    def test_key_budget_reserves_late_first_outcomes_and_75_episode_groups(self) -> None:
        selector = V2TrainingTrajectorySelector(max_steps=1_000_000)
        first = selector.select(episode=1, stage_steps=1, outcome='goal')
        self.assertIn('first_outcome:goal', first.reasons)
        self.assertIn('episode_group:1:goal', first.reasons)
        self.assertFalse(
            selector.select(episode=2, stage_steps=2, outcome='goal').selected
        )
        group_two = selector.select(episode=76, stage_steps=76, outcome='goal')
        self.assertEqual(group_two.reasons, ('episode_group:2:goal',))

        episode = 151
        while selector.key_sample_count < 16:
            selector.select(episode=episode, stage_steps=episode, outcome='goal')
            episode += 75
        self.assertFalse(
            selector.select(episode=episode, stage_steps=episode, outcome='goal').selected
        )
        for offset, outcome in enumerate(
            ('collision', 'ground', 'boundary', 'timeout'), start=1
        ):
            selected = selector.select(
                episode=episode + offset,
                stage_steps=episode + offset,
                outcome=outcome,
            )
            self.assertIn(f'first_outcome:{outcome}', selected.reasons)
        self.assertEqual(selector.key_sample_count, 20)

    def test_validation_selector_limits_failure_types(self) -> None:
        selector = V2ValidationTrajectorySelector()
        self.assertEqual(selector.select('collision'), ('first_failure:collision',))
        self.assertEqual(selector.select('collision'), ())
        self.assertEqual(selector.select('ground'), ('first_failure:ground',))
        self.assertEqual(selector.select('boundary'), ())
        self.assertEqual(selector.select('goal'), ('first_goal',))
        self.assertEqual(selector.select('goal'), ())
        self.assertEqual(selector.sample_count, 3)


class TestV2ReportingPersistence(unittest.TestCase):
    def test_episode_console_fields_and_initialization_preserve_jsonl(self) -> None:
        for model, stage, kind in (
            ('ann', 'easy', 'v2_bc_best'),
            ('snn', 'medium', 'validated_v2_td3_stage'),
        ):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                clock = _Clock()
                reporter = V2ExperimentReporter(
                    Path(directory), stage=stage, model_type=model,
                    scenario=ScenarioConfig(), rewards=RewardConfig(),
                    uav_collision_radius=0.0, max_steps=100, clock=clock,
                )
                output = StringIO()
                with redirect_stdout(output), mock.patch(
                    'brain_uav.trainers.v2_reporting.export_v2_trajectory_views',
                ):
                    reporter.start_stage({'initialization_source': {
                        'kind': kind, 'path': '/existing/model.pt',
                    }})
                    reporter.begin_episode()
                    clock.value = 42.18
                    record = _episode(12, 'goal', 68)
                    reporter.record_episode(
                        record, scenario_payload=_scenario_payload([]),
                        trajectory=[], actions=[], terminal_state=[],
                    )
                    clock.value = 601.0
                    reporter.maybe_report_progress(
                        stage_steps=69, completed_episodes=12,
                        current_episode_steps=1, actor_active=True,
                    )
                text = output.getvalue()
                self.assertIn('initialization=/existing/model.pt', text)
                self.assertIn('loading_existing_model=' + (
                    'BC checkpoint' if stage == 'easy' else 'predecessor passed checkpoint'
                ), text)
                self.assertIn(
                    f'[{"V2 " + model.upper()} {stage}] ep    12 | 68/100 '
                    '| len    1 | goal      | ret      12 |   42.2s', text,
                )
                self.assertIn(
                    f'---- [{"V2 " + model.upper()} {stage}] elapsed 10m '
                    '| 69/100 (69.0%) | eta 4m ----', text,
                )
                saved = json.loads((Path(directory) / 'episodes.jsonl').read_text())
                self.assertEqual(saved['episode_elapsed_seconds'], 42.18)
                for name, value in record.items():
                    self.assertEqual(saved[name], value)
                reporter.close()

    def test_episode_window_and_progress_records_flush_without_waiting(self) -> None:
        clock = _Clock()
        with tempfile.TemporaryDirectory() as directory:
            reporter = V2ExperimentReporter(
                Path(directory),
                stage='easy',
                model_type='ann',
                scenario=ScenarioConfig(),
                rewards=RewardConfig(),
                uav_collision_radius=0.0,
                max_steps=100,
                progress_interval_seconds=60.0,
                clock=clock,
            )
            reporter.start_stage({'requested_device': 'cpu', 'resolved_device': 'cpu'})
            reporter.record_episode(
                _episode(1, 'goal', 5),
                scenario_payload=_scenario_payload([]),
                trajectory=[[-10.0, -10.0, 20.0], [-9.0, -9.0, 20.0]],
                actions=[[0.0, 0.0]],
                terminal_state=[-9.0, -9.0, 20.0, 0.0, 0.0],
            )
            episode_path = Path(directory) / 'episodes.jsonl'
            self.assertEqual(len(episode_path.read_text(encoding='utf-8').splitlines()), 1)
            trajectory_root = Path(directory) / 'trajectories' / 'training'
            self.assertEqual(len(list(trajectory_root.glob('*.png'))), 1)
            trajectory_jsons = list(trajectory_root.glob('*.json'))
            self.assertEqual(len(trajectory_jsons), 1)
            selected = json.loads(trajectory_jsons[0].read_text(encoding='utf-8'))
            self.assertIn('first_outcome:goal', selected['selection_reasons'])
            self.assertIn('progress_thresholds:5', selected['selection_reasons'])
            clock.value = 61.0
            self.assertTrue(
                reporter.maybe_report_progress(
                    stage_steps=6,
                    completed_episodes=1,
                    current_episode_steps=1,
                    actor_active=False,
                )
            )
            row = {
                'window_index': 1,
                'episode_count': 1,
                'goal_count': 1,
                'failure_count': 0,
                'qualified': True,
                'consecutive_qualified_windows': 1,
                'stage_steps': 5,
                'global_steps': 5,
                'candidate': False,
                'episode_start': 1,
                'episode_end': 1,
                'average_return': 1.0,
                'average_length': 1.0,
                'average_actor_loss': 1.0,
                'average_critic_loss': 2.0,
                'bc_lambda': 5.0,
                'average_bc_loss': 0.5,
                'average_weighted_bc_contribution': 0.25,
                'exploration_noise': 0.01,
                'policy_noise': 0.01,
                'noise_clip': 0.02,
            }
            reporter.record_window(row)
            self.assertEqual(len((Path(directory) / 'windows.jsonl').read_text(encoding='utf-8').splitlines()), 1)
            reporter.close()

    def test_later_json_failure_preserves_flushed_history_and_partial_window_is_marked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = V2ExperimentReporter(
                root,
                stage='easy',
                model_type='ann',
                scenario=ScenarioConfig(),
                rewards=RewardConfig(),
                uav_collision_radius=0.0,
                max_steps=100,
            )
            reporter.record_episode(
                _episode(1, 'timeout', 1),
                scenario_payload=_scenario_payload([]),
                trajectory=[[-10.0, -10.0, 20.0], [-9.0, -10.0, 20.0]],
                actions=[[0.0, 0.0]],
                terminal_state=[-9.0, -10.0, 20.0, 0.0, 0.0],
            )
            invalid = _episode(2, 'goal', 2)
            invalid['episode_return'] = float('nan')
            with self.assertRaisesRegex(RuntimeError, 'episodes.jsonl'):
                reporter.record_episode(
                    invalid,
                    scenario_payload=_scenario_payload([]),
                    trajectory=[[-10.0, -10.0, 20.0], [-9.0, -10.0, 20.0]],
                    actions=[[0.0, 0.0]],
                    terminal_state=[-9.0, -10.0, 20.0, 0.0, 0.0],
                )
            self.assertEqual(len((root / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()), 1)
            reporter.finish_stage({
                'status': 'failed',
                'stop_reason': 'test_failure',
                'stage_steps': 1,
                'global_steps_end': 1,
            })
            reporter.close()
            window = json.loads((root / 'windows.jsonl').read_text(encoding='utf-8').splitlines()[0])
            self.assertTrue(window['partial'])
            self.assertFalse(window['candidate'])
            self.assertEqual(window['stage'], 'easy')
            self.assertEqual(window['global_steps'], 1)

    def test_bc_epoch_jsonl_csv_and_curve_use_existing_losses(self) -> None:
        clock = _Clock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reporter = V2BCTrainingReporter(root, clock=clock)
            reporter.record_epoch({
                'epoch': 1,
                'epochs': 2,
                'train_loss': 0.8,
                'validation_loss': 0.9,
                'best_epoch': 1,
                'best_validation_loss': 0.9,
                'refreshed_best': True,
            })
            clock.value = 2.0
            reporter.record_epoch({
                'epoch': 2,
                'epochs': 2,
                'train_loss': 0.7,
                'validation_loss': 1.0,
                'best_epoch': 1,
                'best_validation_loss': 0.9,
                'refreshed_best': False,
            })
            reporter.finish()
            reporter.close()
            self.assertEqual(len((root / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()), 2)
            with (root / 'epochs.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row['epoch'] for row in rows], ['1', '2'])
            image = mpimg.imread(root / 'mse_curve.png')
            self.assertEqual(image.ndim, 3)


class TestV2TrajectoryPlotting(unittest.TestCase):
    def test_five_shapes_and_zero_zone_render_as_redrawable_three_view_png(self) -> None:
        zones = [
            NoFlyZone('sphere', Sphere([-6.0, -4.0, 4.0], 2.0), safety_margin=0.5),
            NoFlyZone('ellipsoid', Ellipsoid([0.0, -4.0, 5.0], 2.0, 1.5, 1.0), safety_margin=0.25),
            NoFlyZone('box', Box([6.0, -4.0, 3.0], 2.0, 4.0, 2.0)),
            NoFlyZone('tri', TriangularPyramid([-3.0, 4.0, 0.0], 3.0, 4.0, 5.0)),
            NoFlyZone('quad', QuadrangularPyramid([4.0, 4.0, 0.0], 3.0, 2.0, 4.0)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, scenario_payload in (
                ('five', _scenario_payload(zones)),
                ('empty', _scenario_payload([])),
            ):
                outputs = export_v2_trajectory_views(
                    root,
                    name,
                    {
                        'scenario_payload': scenario_payload,
                        'scenario_config': asdict(ScenarioConfig(world_xy=20.0, world_z_max=40.0)),
                        'reward_config': asdict(RewardConfig()),
                        'uav_collision_radius': 0.75,
                        'trajectory': [[-10.0, -10.0, 20.0], [0.0, 0.0, 25.0], [10.0, 10.0, 30.0]],
                        'actions': [[0.0, 0.0], [0.0, 0.0]],
                        'terminal_state': [10.0, 10.0, 30.0, 0.0, 0.0],
                        'outcome': 'goal',
                        'episode_return': 1.0,
                        'episode_length': 2,
                        'stage': 'easy',
                        'global_steps': 2,
                        'model_type': 'ann',
                        'source': 'training',
                        'selection_reasons': ['first_outcome:goal'],
                    },
                )
                payload = json.loads(Path(outputs['json']).read_text(encoding='utf-8'))
                self.assertEqual(len(payload['trajectory']), len(payload['actions']) + 1)
                self.assertEqual(
                    payload.get('safety_boundary_visualization_note'),
                    'Conservative visual approximation; not used for collision checking.',
                )
                image = mpimg.imread(outputs['png'])
                self.assertGreater(image.shape[0], 100)
                self.assertGreater(image.shape[1], image.shape[0])


# Key sets captured from pre-change real training output
# (train_result/formal_ann_compile_fix_20260917_004203/v2_td3_easy_metrics_reports).
# They pin the persisted contract: console/plot changes must not touch it.
_EPISODE_KEYS = frozenset({
    'actor_loss', 'actor_update_status', 'actor_updated', 'batch_success_fraction',
    'bc_lambda', 'bc_loss', 'critic_loss', 'curriculum_level', 'episode',
    'episode_elapsed_seconds', 'episode_length', 'episode_return', 'exploration_noise',
    'global_steps', 'noise_clip', 'outcome', 'policy_noise', 'policy_warmup_steps',
    'replay_size', 'replay_success_fraction', 'stage', 'stage_elapsed_seconds',
    'stage_steps', 'success_replay_size', 'terminal_geo_loss', 'weighted_bc_contribution',
    # Diagnostic-only additions (A1/A2/A4 in this pass; H4/H5 in
    # docs/无法早停排查文档.md).
    'critic_q1_mean', 'critic_q1_std', 'critic_q1_max_abs', 'critic_target_q_mean',
    'critic_td_error_mean', 'critic_failure_td_error_mean', 'actor_grad_norm',
})
_WINDOW_KEYS = frozenset({
    'actor_update_status', 'average_actor_loss', 'average_bc_loss', 'average_critic_loss',
    'average_length', 'average_return', 'average_weighted_bc_contribution',
    'batch_success_fraction', 'bc_lambda', 'boundary_count', 'candidate',
    'collision_count', 'consecutive_qualified_windows', 'episode_count', 'episode_end',
    'episode_start', 'exploration_noise', 'failure_count', 'global_steps', 'goal_count',
    'goal_ratio', 'ground_count', 'noise_clip', 'partial', 'policy_noise', 'qualified',
    'replay_size', 'replay_success_fraction', 'stage', 'stage_steps',
    'success_replay_size', 'timeout_count', 'window_elapsed_seconds', 'window_index',
    # A5 in this pass: a parallel "medium curriculum_level only" view of the
    # same window (see P5早停判据离线回放结论_20260917.md).
    'medium_only_goal_count', 'medium_only_episode_count', 'medium_only_goal_ratio',
    'medium_only_failure_count',
})
_WINDOW_CSV_HEADER = [
    'window_index', 'episode_count', 'goal_count', 'failure_count', 'qualified',
    'consecutive_qualified_windows', 'stage_steps', 'candidate', 'episode_start',
    'episode_end', 'average_return', 'average_length', 'average_actor_loss',
    'average_critic_loss', 'bc_lambda', 'average_bc_loss',
    'average_weighted_bc_contribution', 'global_steps', 'exploration_noise',
    'policy_noise', 'noise_clip', 'medium_only_goal_count', 'medium_only_episode_count',
    'medium_only_goal_ratio', 'medium_only_failure_count', 'stage', 'partial',
    'ground_count', 'boundary_count',
    'collision_count', 'timeout_count', 'goal_ratio', 'replay_size',
    'replay_success_fraction', 'success_replay_size', 'batch_success_fraction',
    'actor_update_status', 'window_elapsed_seconds',
]


def _window_row(**overrides) -> dict:
    """A window row in the exact key order the training loop emits.

    windows.csv derives its header from insertion order, so the fixture mirrors
    V2EpisodeWindowTracker.add_episode followed by the window.update(...) call
    in v2_formal_training.py.
    """

    row = {
        'window_index': 1,
        'episode_count': 15,
        'goal_count': 9,
        'failure_count': 6,
        'qualified': False,
        'consecutive_qualified_windows': 0,
        'stage_steps': 518653,
        'candidate': False,
        'episode_start': 946,
        'episode_end': 960,
        'average_return': 14050.855,
        'average_length': 498.9,
        'average_actor_loss': 0.877486,
        'average_critic_loss': 282745.415728,
        'bc_lambda': 5.0,
        'average_bc_loss': 0.5,
        'average_weighted_bc_contribution': 0.25,
        'global_steps': 700713,
        'exploration_noise': 0.01,
        'policy_noise': 0.01,
        'noise_clip': 0.02,
        'medium_only_goal_count': 9,
        'medium_only_episode_count': 15,
        'medium_only_goal_ratio': 0.6,
        'medium_only_failure_count': 6,
    }
    row.update(overrides)
    return row


def _episode_with(number: int, outcome: str, **overrides) -> dict:
    record = _episode(number, outcome, 518653)
    record.update(overrides)
    return record


class _Reporter:
    """Context manager yielding a reporter plus its captured stdout."""

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._directory = tempfile.TemporaryDirectory()
        self.output = StringIO()
        self.root = Path(self._directory.name)
        defaults = {
            'stage': 'medium', 'model_type': 'ann', 'scenario': ScenarioConfig(),
            'rewards': RewardConfig(), 'uav_collision_radius': 0.0, 'max_steps': 750_000,
        }
        defaults.update(kwargs)
        self.reporter = V2ExperimentReporter(self.root, **defaults)

    def __enter__(self) -> '_Reporter':
        self._redirect = redirect_stdout(self.output)
        self._redirect.__enter__()
        self._patch = mock.patch(
            'brain_uav.trainers.v2_reporting.export_v2_trajectory_views')
        self._patch.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        self._patch.__exit__(*exc_info)
        self._redirect.__exit__(*exc_info)
        self.reporter.close()
        self._directory.cleanup()

    def lines(self) -> list[str]:
        return self.output.getvalue().splitlines()


class TestV2ProgressReporting(unittest.TestCase):
    def test_progress_prints_once_per_interval_and_again_after_crossing(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            reporter = context.reporter
            for value in (1.0, 300.0, 599.9):
                clock.value = value
                self.assertFalse(
                    reporter.maybe_report_progress(
                        stage_steps=1000, completed_episodes=1,
                        current_episode_steps=1, actor_active=True,
                    ),
                    msg=f'progress must stay silent at {value}s',
                )
            clock.value = 600.0
            self.assertTrue(
                reporter.maybe_report_progress(
                    stage_steps=1000, completed_episodes=1,
                    current_episode_steps=1, actor_active=True,
                )
            )
            clock.value = 1199.0
            self.assertFalse(
                reporter.maybe_report_progress(
                    stage_steps=2000, completed_episodes=2,
                    current_episode_steps=1, actor_active=True,
                )
            )
            clock.value = 1200.0
            self.assertTrue(
                reporter.maybe_report_progress(
                    stage_steps=2000, completed_episodes=2,
                    current_episode_steps=1, actor_active=True,
                )
            )
            self.assertEqual(len(context.lines()), 2)

    def test_progress_line_reports_elapsed_fraction_percent_and_eta(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            clock.value = 59544.0
            context.reporter.maybe_report_progress(
                stage_steps=517172, completed_episodes=957,
                current_episode_steps=61, actor_active=True,
            )
        self.assertEqual(
            context.lines()[0],
            '---- [V2 ANN medium] elapsed 16.5h | 517172/750000 (69.0%) | eta 7.4h ----',
        )

    def test_progress_line_reports_eta_not_available_before_any_step(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            clock.value = 700.0
            context.reporter.maybe_report_progress(
                stage_steps=0, completed_episodes=0,
                current_episode_steps=5, actor_active=False,
            )
        line = context.lines()[0]
        self.assertIn('eta n/a', line)
        self.assertIn('0/750000 (0.0%)', line)
        self.assertIn('elapsed 11m', line)

    def test_progress_line_drops_episode_and_actor_fields(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            clock.value = 600.0
            context.reporter.maybe_report_progress(
                stage_steps=1000, completed_episodes=7,
                current_episode_steps=61, actor_active=True,
            )
        line = context.lines()[0]
        for absent in ('episodes=', 'current_episode_steps=', 'actor=',
                       'unfinished_episode'):
            self.assertNotIn(absent, line)


class TestV2DurationFormatting(unittest.TestCase):
    def test_duration_selects_seconds_minutes_or_hours(self) -> None:
        for seconds, expected in (
            (0.0, '0s'), (35.4, '35s'), (59.9, '59s'),
            (60.0, '1m'), (2520.0, '42m'), (3599.0, '59m'),
            (3600.0, '1.0h'), (59544.0, '16.5h'),
        ):
            with self.subTest(seconds=seconds):
                self.assertEqual(_format_duration(seconds), expected)

    def test_duration_rejects_negative_and_non_finite(self) -> None:
        for invalid in (-1.0, float('nan'), float('inf')):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _format_duration(invalid)

    def test_scientific_notation_drops_exponent_padding(self) -> None:
        for value, expected in (
            (282745.415728, '2.83e5'),
            (0.0, '0.00e0'),
            (-0.0015, '-1.50e-3'),
            (1e-12, '1.00e-12'),
        ):
            with self.subTest(value=value):
                self.assertEqual(_format_scientific(value), expected)
        for invalid in (float('nan'), float('inf')):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _format_scientific(invalid)


class TestV2EpisodeLineLayout(unittest.TestCase):
    def test_outcome_column_is_aligned_across_outcomes(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            for number, outcome in ((956, 'goal'), (957, 'collision')):
                context.reporter.begin_episode()
                clock.value += 10.0
                context.reporter.record_episode(
                    _episode_with(number, outcome),
                    scenario_payload=_scenario_payload([]),
                    trajectory=[], actions=[], terminal_state=[],
                )
        goal_line, collision_line = context.lines()
        self.assertEqual(goal_line.index('goal'), collision_line.index('collision'))
        goal_pipe = goal_line.index('|', goal_line.index('goal'))
        collision_pipe = collision_line.index('|', collision_line.index('collision'))
        self.assertEqual(goal_pipe, collision_pipe)
        # every separator sits in the same column, so the log reads as a table
        self.assertEqual(
            [index for index, char in enumerate(goal_line) if char == '|'],
            [index for index, char in enumerate(collision_line) if char == '|'],
        )

    def test_episode_line_matches_documented_layout(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            for number, outcome, steps, length, ret, duration in (
                (956, 'goal', 516838, 643, 40581.0, 106.0),
                (957, 'collision', 517111, 273, -17799.99, 28.21),
                (958, 'goal', 517771, 660, 16642.0, 65.9),
            ):
                context.reporter.begin_episode()
                clock.value += duration
                context.reporter.record_episode(
                    _episode_with(number, outcome, stage_steps=steps,
                                  episode_length=length, episode_return=ret),
                    scenario_payload=_scenario_payload([]),
                    trajectory=[], actions=[], terminal_state=[],
                )
        self.assertEqual(context.lines(), [
            '[V2 ANN medium] ep   956 | 516838/750000 | len  643 | goal      '
            '| ret   40581 |  106.0s',
            '[V2 ANN medium] ep   957 | 517111/750000 | len  273 | collision '
            '| ret  -17800 |   28.2s',
            '[V2 ANN medium] ep   958 | 517771/750000 | len  660 | goal      '
            '| ret   16642 |   65.9s',
        ])

    def test_episode_line_drops_actor_field(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            context.reporter.begin_episode()
            clock.value = 5.0
            context.reporter.record_episode(
                _episode_with(1, 'goal'), scenario_payload=_scenario_payload([]),
                trajectory=[], actions=[], terminal_state=[],
            )
        self.assertNotIn('actor=', context.lines()[0])


class TestV2ActorUnfreezeNotice(unittest.TestCase):
    def test_unfreeze_notice_prints_once_on_frozen_to_updated(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            for number, status, steps in (
                (1, 'frozen', 10_000), (2, 'frozen', 20_000),
                (3, 'updated', 25_000), (4, 'updated', 30_000),
                (5, 'not_updated', 31_000), (6, 'updated', 32_000),
            ):
                context.reporter.begin_episode()
                clock.value += 1.0
                context.reporter.record_episode(
                    _episode_with(number, 'goal', actor_update_status=status,
                                  stage_steps=steps),
                    scenario_payload=_scenario_payload([]),
                    trajectory=[], actions=[], terminal_state=[],
                )
        notices = [line for line in context.lines() if 'actor unfrozen' in line]
        self.assertEqual(notices, ['[V2 ANN medium] actor unfrozen at step 25000'])
        lines = context.lines()
        self.assertEqual(lines.index(notices[0]) + 1, lines.index(
            next(line for line in lines if ' ep     3 ' in line)))

    def test_no_notice_when_actor_was_never_frozen(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            for number in (1, 2):
                context.reporter.begin_episode()
                clock.value += 1.0
                context.reporter.record_episode(
                    _episode_with(number, 'goal', actor_update_status='updated'),
                    scenario_payload=_scenario_payload([]),
                    trajectory=[], actions=[], terminal_state=[],
                )
        self.assertNotIn('actor unfrozen', context.output.getvalue())


class TestV2WindowBlock(unittest.TestCase):
    def _record_window(self, context, outcomes: list[str], **row_overrides) -> None:
        clock = context.reporter._clock
        for index, outcome in enumerate(outcomes):
            context.reporter.begin_episode()
            clock.value += 875.8 / len(outcomes)
            context.reporter.record_episode(
                _episode_with(946 + index, outcome, replay_size=500_000),
                scenario_payload=_scenario_payload([]),
                trajectory=[], actions=[], terminal_state=[],
            )
        context.reporter.record_window(_window_row(**row_overrides))

    def test_window_block_matches_documented_three_lines(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            self._record_window(
                context, ['goal'] * 9 + ['collision'] * 5 + ['ground'],
                window_index=64,
            )
        self.assertEqual(context.lines()[-3:], [
            '=== [V2 ANN medium] window 64 | ep 946-960 | step 518653 '
            '| goal 9/15 | streak 0/4',
            '    outcomes  goal 9  collision 5  ground 1  boundary 0  timeout 0',
            '    metrics   return 14051 | length 499 | actor 0.877 | critic 2.83e5 '
            '| replay 500000 | 876s',
        ])

    def test_outcome_line_counts_agree_with_persisted_window_row(self) -> None:
        clock = _Clock()
        outcomes = (['goal'] * 7 + ['collision'] * 4 + ['ground'] * 2
                    + ['boundary'] + ['timeout'])
        with _Reporter(clock=clock) as context:
            self._record_window(context, outcomes, window_index=12)
            saved = json.loads(
                (context.root / 'windows.jsonl').read_text(encoding='utf-8').splitlines()[0])
            outcome_line = context.lines()[-2]
        for outcome in ('goal', 'collision', 'ground', 'boundary', 'timeout'):
            with self.subTest(outcome=outcome):
                self.assertIn(f'{outcome} {saved[f"{outcome}_count"]}', outcome_line)
        self.assertEqual(
            outcome_line,
            '    outcomes  goal 7  collision 4  ground 2  boundary 1  timeout 1',
        )

    def test_streak_denominator_comes_from_required_qualified_windows(self) -> None:
        for required in (2, 4, 7):
            with self.subTest(required=required):
                clock = _Clock()
                with _Reporter(clock=clock,
                               required_qualified_windows=required) as context:
                    self._record_window(
                        context, ['goal'] * 14 + ['collision'],
                        consecutive_qualified_windows=3,
                    )
                self.assertIn(f'streak 3/{required}', context.lines()[-3])

    def test_required_qualified_windows_rejects_invalid_values(self) -> None:
        for invalid in (0, -1, 4.0, '4', True, None):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    V2ExperimentReporter(
                        Path(directory), stage='medium', model_type='ann',
                        scenario=ScenarioConfig(), rewards=RewardConfig(),
                        uav_collision_radius=0.0, max_steps=750_000,
                        required_qualified_windows=invalid,
                    )


class TestV2PersistedFieldStability(unittest.TestCase):
    def test_episode_window_and_csv_fields_are_unchanged(self) -> None:
        clock = _Clock()
        with _Reporter(clock=clock) as context:
            for index, outcome in enumerate(['goal'] * 9 + ['collision'] * 5 + ['ground']):
                context.reporter.begin_episode()
                clock.value += 1.0
                context.reporter.record_episode(
                    _episode_with(946 + index, outcome, replay_size=500_000),
                    scenario_payload=_scenario_payload([]),
                    trajectory=[], actions=[], terminal_state=[],
                )
            context.reporter.record_window(_window_row(window_index=64))
            context.reporter.finish_stage({
                'status': 'passed', 'stop_reason': 'fixed_validation_passed',
                'stage_steps': 518653, 'global_steps_end': 700713,
            })
            root = context.root
            episode_row = json.loads(
                (root / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()[0])
            window_row = json.loads(
                (root / 'windows.jsonl').read_text(encoding='utf-8').splitlines()[0])
            with (root / 'windows.csv').open(newline='', encoding='utf-8') as handle:
                header = next(csv.reader(handle))
        self.assertEqual(frozenset(episode_row), _EPISODE_KEYS)
        self.assertEqual(frozenset(window_row), _WINDOW_KEYS)
        self.assertEqual(header, _WINDOW_CSV_HEADER)


class TestV2TrainingCurvePlot(unittest.TestCase):
    def test_four_panel_curve_renders_from_a_small_window_sample(self) -> None:
        rows = []
        for index in range(4):
            rows.append(_window_row(
                window_index=index + 1,
                episode_start=1 + index * 15,
                episode_end=15 + index * 15,
                goal_count=9 + index,
                failure_count=6 - index,
                consecutive_qualified_windows=index,
                goal_ratio=(9 + index) / 15.0,
                collision_count=6 - index,
                ground_count=0,
                boundary_count=0,
                timeout_count=0,
                average_critic_loss=282745.415728 * (index + 1),
            ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_curves.png'
            _plot_training_windows(path, rows, required_qualified_windows=4)
            self.assertTrue(path.is_file())
            image = mpimg.imread(path)
            self.assertGreater(image.shape[0], image.shape[1])
            self.assertGreater(image.shape[1], 500)

    def test_streak_axis_keeps_streaks_above_the_requirement_visible(self) -> None:
        # The streak keeps climbing past the requirement until stage_steps
        # reaches early_stop_min_steps, so the axis must not clip at 4.
        rows = [
            _window_row(window_index=index + 1, consecutive_qualified_windows=streak)
            for index, streak in enumerate((3, 5, 6, 0))
        ]
        captured: list[tuple[float, float]] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training_curves.png'
            with mock.patch(
                'matplotlib.axes.Axes.set_ylim', autospec=True,
                side_effect=lambda axis, *args, **kwargs: captured.append(args),
            ):
                _plot_training_windows(path, rows, required_qualified_windows=4)
        self.assertIn((0, 6), captured)

    def test_plot_failure_is_reported_not_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing_dir' / 'curves.png'
            with self.assertRaisesRegex(RuntimeError, 'Failed to render V2 training curves'):
                _plot_training_windows(path, [_window_row()],
                                       required_qualified_windows=4)


if __name__ == '__main__':
    unittest.main()
