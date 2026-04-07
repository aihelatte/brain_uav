"""Generate and freeze the paper-grade benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..scenarios import (
    DEFAULT_BENCHMARK_SUITE_PATH,
    generate_benchmark_suite,
    save_benchmark_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate the fixed benchmark suite used for evaluation.')
    parser.add_argument('--seed', type=int, default=20260407)
    parser.add_argument('--count-per-category', type=int, default=100)
    parser.add_argument('--suite-name', type=str, default='fixed_benchmark_suite_v2')
    parser.add_argument('--output', type=Path, default=DEFAULT_BENCHMARK_SUITE_PATH)
    args = parser.parse_args()

    payload = generate_benchmark_suite(
        seed=args.seed,
        count_per_category=args.count_per_category,
        suite_name=args.suite_name,
    )
    saved = save_benchmark_suite(payload, args.output)
    print(f"Saved benchmark suite to {saved}")
    print(
        f"Suite={payload['suite_name']} categories={payload['categories']} "
        f"count_per_category={payload['count_per_category']} total={payload['total_scenarios']}"
    )


if __name__ == '__main__':
    main()
