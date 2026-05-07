"""Render three-view PNGs from benchmark episode JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')


from ..scripts.train_td3 import (  # noqa: E402
    _draw_goal_radius_projection,
    _draw_zone_top_view,
    _draw_zone_vertical_projection,
)
from ..utils.io import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Render benchmark episode three-view PNG files from JSON records.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--outcomes', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _parse_outcomes_filter(text: str | None) -> set[str] | None:
    if text is None:
        return None
    outcomes = {_sanitize_token(token) for token in text.split(',') if token.strip()}
    return outcomes or None


def _sanitize_token(value: str) -> str:
    token = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(value).strip())
    while '__' in token:
        token = token.replace('__', '_')
    return token.strip('_') or 'unknown'


def discover_episode_jsons(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    episodes_dir = input_path / 'episodes'
    search_dir = episodes_dir if episodes_dir.is_dir() else input_path
    candidates = sorted(
        path
        for path in search_dir.glob('*.json')
        if path.name not in {'index.json', 'summary.json', 'efficiency_summary.json'}
    )
    return candidates


def resolve_default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        base = input_path.parent.parent if input_path.parent.name == 'episodes' else input_path.parent
    elif (input_path / 'episodes').is_dir():
        base = input_path
    else:
        base = input_path.parent if input_path.name == 'episodes' else input_path
    return ensure_dir(base / 'plots')


def render_episode_views(payload: dict[str, Any], png_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    scenario = payload['scenario']
    config = payload['config']
    scenario_cfg = config['scenario']
    info = payload['info']
    traj = np.asarray(payload['trajectory'], dtype=float)
    start = np.asarray(scenario['state'][:3], dtype=float)
    goal = np.asarray(scenario['goal'], dtype=float)
    zones = scenario['zones']
    active_goal_radius = float(info.get('active_goal_radius', scenario_cfg['goal_radius']))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 18),
        gridspec_kw={'height_ratios': [6, 1, 1]},
    )
    ax_xy, ax_xz, ax_yz = axes

    ax_xy.plot(traj[:, 0], traj[:, 1], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_xy.scatter(start[0], start[1], color='tab:blue', s=55, marker='o', label='start')
    ax_xy.scatter(goal[0], goal[1], color='tab:green', s=70, marker='*', label='goal')
    _draw_goal_radius_projection(ax_xy, (goal[0], goal[1]), active_goal_radius)
    for idx, zone in enumerate(zones, start=1):
        center_xy = zone['center_xy']
        radius = float(zone['radius'])
        _draw_zone_top_view(ax_xy, center_xy, radius, float(scenario_cfg['warning_distance']))
        ax_xy.text(center_xy[0], center_xy[1], f'Z{idx}', fontsize=8, color='tab:red')
    ax_xy.set_title('Top View (X-Y)')
    ax_xy.set_xlabel('x (km)')
    ax_xy.set_ylabel('y (km)')
    ax_xy.set_xlim(-float(scenario_cfg['world_xy']), float(scenario_cfg['world_xy']))
    ax_xy.set_ylim(-float(scenario_cfg['world_xy']), float(scenario_cfg['world_xy']))
    ax_xy.grid(alpha=0.3)
    ax_xy.legend(loc='upper left')
    ax_xy.set_aspect('equal', adjustable='box')

    ax_xz.plot(traj[:, 0], traj[:, 2], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_xz.scatter(start[0], start[2], color='tab:blue', s=55, marker='o', label='start')
    ax_xz.scatter(goal[0], goal[2], color='tab:green', s=70, marker='*', label='goal')
    _draw_goal_radius_projection(ax_xz, (goal[0], goal[2]), active_goal_radius)
    for idx, zone in enumerate(zones, start=1):
        _draw_zone_vertical_projection(ax_xz, zone['center_xy'][0], float(zone['radius']), f'zone {idx}')
    ax_xz.axhline(
        float(scenario_cfg['ground_warning_height']),
        color='tab:orange',
        linestyle='--',
        alpha=0.7,
        label='ground warning',
    )
    ax_xz.set_title('Side View (X-Z)')
    ax_xz.set_xlabel('x (km)')
    ax_xz.set_ylabel('z (km)')
    ax_xz.set_xlim(-float(scenario_cfg['world_xy']), float(scenario_cfg['world_xy']))
    ax_xz.set_ylim(0.0, float(scenario_cfg['world_z_max']))
    ax_xz.grid(alpha=0.3)
    ax_xz.legend(loc='upper left', ncol=2)

    ax_yz.plot(traj[:, 1], traj[:, 2], color='tab:blue', linewidth=2.0, label='trajectory')
    ax_yz.scatter(start[1], start[2], color='tab:blue', s=55, marker='o', label='start')
    ax_yz.scatter(goal[1], goal[2], color='tab:green', s=70, marker='*', label='goal')
    _draw_goal_radius_projection(ax_yz, (goal[1], goal[2]), active_goal_radius)
    for idx, zone in enumerate(zones, start=1):
        _draw_zone_vertical_projection(ax_yz, zone['center_xy'][1], float(zone['radius']), f'zone {idx}')
    ax_yz.axhline(
        float(scenario_cfg['ground_warning_height']),
        color='tab:orange',
        linestyle='--',
        alpha=0.7,
        label='ground warning',
    )
    ax_yz.set_title('Front View (Y-Z)')
    ax_yz.set_xlabel('y (km)')
    ax_yz.set_ylabel('z (km)')
    ax_yz.set_xlim(-float(scenario_cfg['world_xy']), float(scenario_cfg['world_xy']))
    ax_yz.set_ylim(0.0, float(scenario_cfg['world_z_max']))
    ax_yz.grid(alpha=0.3)
    ax_yz.legend(loc='upper left', ncol=2)

    fig.suptitle(
        f"Episode {payload['episode']} - {payload['outcome']} - "
        f"{payload.get('scenario_id', scenario.get('scenario_id', 'unknown'))}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97], pad=2.0)
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return png_path


def render_from_input(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    limit: int | None = None,
    outcomes: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    json_paths = discover_episode_jsons(input_path)
    outcome_filter = _parse_outcomes_filter(outcomes)
    resolved_output_dir = ensure_dir(output_dir) if output_dir is not None else resolve_default_output_dir(input_path)
    rendered: list[Path] = []

    for json_path in json_paths:
        payload = _load_json(json_path)
        if outcome_filter is not None and _sanitize_token(payload.get('outcome', 'unknown')) not in outcome_filter:
            continue
        png_path = resolved_output_dir / f'{json_path.stem}.png'
        if png_path.exists() and not overwrite:
            continue
        render_episode_views(payload, png_path)
        rendered.append(png_path)
        if limit is not None and len(rendered) >= limit:
            break

    return rendered


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rendered = render_from_input(
        args.input,
        output_dir=args.output_dir,
        limit=args.limit,
        outcomes=args.outcomes,
        overwrite=args.overwrite,
    )
    print(f'Rendered {len(rendered)} PNG files.')


if __name__ == '__main__':
    main()
