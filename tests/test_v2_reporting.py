"""Focused tests for V2 streaming reports and trajectory sampling."""

from __future__ import annotations

import csv
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


if __name__ == '__main__':
    unittest.main()
