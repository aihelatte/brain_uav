"""Tests for rendering benchmark episode views from JSON files."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brain_uav.scripts.render_benchmark_episode_views import render_from_input


def _episode_payload(*, episode: int, outcome: str) -> dict:
    return {
        'episode': episode,
        'scenario_id': f'S{episode:03d}',
        'outcome': outcome,
        'scenario': {
            'state': [0.0, 0.0, 5.0, 0.0, 0.0],
            'goal': [10.0, 0.0, 8.0],
            'zones': [{'center_xy': [2.0, 0.0], 'radius': 1.5}],
        },
        'trajectory': [
            [0.0, 0.0, 5.0],
            [2.0, 0.0, 5.5],
            [4.0, 0.0, 6.0],
        ],
        'info': {
            'active_goal_radius': 5.0,
        },
        'config': {
            'scenario': {
                'world_xy': 100.0,
                'world_z_max': 50.0,
                'goal_radius': 5.0,
                'warning_distance': 10.0,
                'ground_warning_height': 4.0,
            }
        },
    }


class TestRenderBenchmarkEpisodeViews(unittest.TestCase):
    def test_single_episode_json_renders_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'ep000001_collision.json'
            json_path.write_text(json.dumps(_episode_payload(episode=1, outcome='collision')), encoding='utf-8')

            rendered = render_from_input(json_path)
            self.assertEqual(len(rendered), 1)
            self.assertTrue(rendered[0].is_file())

    def test_directory_batch_rendering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episodes_dir = Path(tmpdir) / 'episodes'
            episodes_dir.mkdir()
            for idx, outcome in enumerate(('goal', 'collision'), start=1):
                path = episodes_dir / f'ep{idx:06d}_{outcome}.json'
                path.write_text(json.dumps(_episode_payload(episode=idx, outcome=outcome)), encoding='utf-8')

            rendered = render_from_input(episodes_dir)
            self.assertEqual(len(rendered), 2)
            self.assertTrue(all(path.is_file() for path in rendered))

    def test_outcomes_filter_is_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episodes_dir = Path(tmpdir) / 'episodes'
            episodes_dir.mkdir()
            for idx, outcome in enumerate(('goal', 'collision'), start=1):
                path = episodes_dir / f'ep{idx:06d}_{outcome}.json'
                path.write_text(json.dumps(_episode_payload(episode=idx, outcome=outcome)), encoding='utf-8')

            rendered = render_from_input(episodes_dir, outcomes='collision')
            self.assertEqual(len(rendered), 1)
            self.assertIn('collision', rendered[0].stem)

    def test_overwrite_false_skips_existing_png(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'ep000001_collision.json'
            json_path.write_text(json.dumps(_episode_payload(episode=1, outcome='collision')), encoding='utf-8')
            plots_dir = Path(tmpdir) / 'plots'
            plots_dir.mkdir()
            png_path = plots_dir / 'ep000001_collision.png'
            png_path.write_text('sentinel', encoding='utf-8')

            rendered = render_from_input(json_path, output_dir=plots_dir, overwrite=False)
            self.assertEqual(rendered, [])
            self.assertEqual(png_path.read_text(encoding='utf-8'), 'sentinel')

    def test_limit_is_respected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episodes_dir = Path(tmpdir) / 'episodes'
            episodes_dir.mkdir()
            for idx in range(1, 4):
                path = episodes_dir / f'ep{idx:06d}_goal.json'
                path.write_text(json.dumps(_episode_payload(episode=idx, outcome='goal')), encoding='utf-8')

            rendered = render_from_input(episodes_dir, limit=1)
            self.assertEqual(len(rendered), 1)


if __name__ == '__main__':
    unittest.main()
