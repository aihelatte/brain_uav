"""Plot the main summary chart from a finished experiment run."""

from __future__ import annotations

import argparse
from pathlib import Path

import json
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot experiment summary charts.')
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    output_dir = args.output_dir or args.summary.parent / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ['SNN', 'ANN']
    success = [summary['evaluation']['snn']['success_rate'], summary['evaluation']['ann']['success_rate']]
    infer = [summary['evaluation']['snn']['avg_inference_time_ms'], summary['evaluation']['ann']['avg_inference_time_ms']]
    macs = [summary['profile']['snn_effective_macs'], summary['profile']['ann_macs']]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].bar(labels, success, color=['tab:blue', 'tab:orange'])
    axes[0].set_title('Success Rate')
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, infer, color=['tab:blue', 'tab:orange'])
    axes[1].set_title('Avg Inference Time (ms)')
    axes[2].bar(labels, macs, color=['tab:blue', 'tab:orange'])
    axes[2].set_title('MACs Comparison')
    fig.tight_layout()
    fig.savefig(output_dir / 'summary_overview.png', dpi=200)
    plt.close(fig)
    print({'plot': str(output_dir / 'summary_overview.png')})


if __name__ == '__main__':
    main()
