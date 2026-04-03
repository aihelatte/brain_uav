"""Run the full experiment pipeline end to end.

这个入口仍然保留，但在课程学习模式下更推荐手动分层执行：
1. generate_dataset
2. train_bc
3. train_td3 --curriculum-level easy/medium/hard
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..curriculum import parse_curriculum_mix
from ..scripts.evaluate import evaluate_policy
from ..scripts.generate_dataset import main as _unused_generate_main
from ..utils.io import log_root_path, model_output_path, now_timestamp, save_json
from ..utils.seeding import set_global_seed
from .generate_dataset import DATASET_VERSION, main as _unused_dataset_main  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a compact curriculum experiment reminder entrypoint.')
    parser.add_argument('--curriculum-level', choices=['easy', 'medium', 'hard'], default='easy')
    parser.add_argument('--curriculum-mix', type=str, default=None)
    parser.add_argument('--model', choices=['snn', 'ann'], default='snn')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--eval-episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    curriculum_mix = parse_curriculum_mix(args.curriculum_mix, fallback_level=args.curriculum_level)
    summary = {
        'message': 'run_full_experiment is kept only as a lightweight evaluation helper in curriculum mode',
        'dataset_version': DATASET_VERSION,
        'curriculum_level': args.curriculum_level,
        'curriculum_mix': curriculum_mix,
        'evaluation': evaluate_policy(args.checkpoint, args.model, args.eval_episodes, args.seed, 'benchmark'),
        'recommended_manual_flow': [
            'python -m brain_uav.scripts.generate_dataset --curriculum-level easy',
            'python -m brain_uav.scripts.train_bc --dataset data/bc_dataset_easy_v5.npz',
            'python -m brain_uav.scripts.train_td3 --curriculum-level easy --init-checkpoint outputs/models/bc/bc_snn_latest.pt',
        ],
    }
    target = args.output or (log_root_path('benchmark') / now_timestamp() / f'curriculum_{args.curriculum_level}_summary.json')
    save_json(target, summary)
    print(summary)


if __name__ == '__main__':
    main()
