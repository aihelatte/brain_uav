"""Render saved TD3 trajectory JSON files into three-view PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .train_td3 import render_episode_json


def _iter_json_files(results_dir: Path, only: str) -> list[Path]:
    subdirs: list[str]
    if only == 'all':
        subdirs = ['step_snapshots', 'goal_examples']
    else:
        subdirs = [only]

    json_files: list[Path] = []
    for subdir in subdirs:
        root = results_dir / subdir
        if root.exists():
            json_files.extend(sorted(root.rglob('*.json')))
    return sorted(json_files)


def main() -> None:
    parser = argparse.ArgumentParser(description='Render saved TD3 episode JSON files into PNGs.')
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--only', choices=['step_snapshots', 'goal_examples', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        raise FileNotFoundError(f'Results directory not found: {results_dir}')

    json_files = _iter_json_files(results_dir, args.only)
    if args.limit is not None:
        json_files = json_files[: max(args.limit, 0)]

    rendered = 0
    skipped = 0
    for json_path in json_files:
        result = render_episode_json(json_path, overwrite=args.overwrite)
        if result['status'] == 'rendered':
            rendered += 1
        else:
            skipped += 1
    print(
        f'[render_saved_trajectories] results_dir={results_dir} '
        f'json_files={len(json_files)} rendered={rendered} skipped={skipped}'
    )


if __name__ == '__main__':
    main()
