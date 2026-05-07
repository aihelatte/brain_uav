"""Tests for the upgraded fixed benchmark suite defaults."""

from __future__ import annotations

from collections import Counter
import unittest

from brain_uav.scenarios import (
    BENCHMARK_CATEGORIES,
    DEFAULT_BENCHMARK_SUITE_NAME,
    DEFAULT_BENCHMARK_SUITE_PATH,
    generate_benchmark_suite,
)


class TestBenchmarkSuiteDefaults(unittest.TestCase):
    def test_default_suite_name_and_path_use_v3(self):
        self.assertEqual(DEFAULT_BENCHMARK_SUITE_NAME, 'fixed_benchmark_suite_v3')
        self.assertEqual(
            str(DEFAULT_BENCHMARK_SUITE_PATH).replace('\\', '/'),
            'outputs/benchmarks/fixed_benchmark_suite_v3.json',
        )

    def test_generate_benchmark_suite_defaults_to_800_scenarios(self):
        payload = generate_benchmark_suite()

        self.assertEqual(payload['suite_name'], 'fixed_benchmark_suite_v3')
        self.assertEqual(payload['count_per_category'], 200)
        self.assertEqual(payload['total_scenarios'], 800)
        self.assertEqual(payload['categories'], list(BENCHMARK_CATEGORIES))

        counts = Counter(item['category'] for item in payload['scenarios'])
        self.assertEqual(counts, {category: 200 for category in BENCHMARK_CATEGORIES})


if __name__ == '__main__':
    unittest.main()
