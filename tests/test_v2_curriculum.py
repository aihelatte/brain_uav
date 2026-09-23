from __future__ import annotations

import unittest


class TestV2Curriculum(unittest.TestCase):
    def test_levels_default_mixes_and_stage_sequence_exclude_legacy_level(self) -> None:
        from brain_uav.v2_curriculum import (
            DEFAULT_V2_TD3_CURRICULUM_MIXES,
            V2_TD3_STAGES,
            v2_stage_sequence,
        )

        self.assertEqual(V2_TD3_STAGES, ('easy', 'medium', 'hard'))
        self.assertEqual(
            DEFAULT_V2_TD3_CURRICULUM_MIXES,
            {
                'easy': {'easy': 1.0},
                'medium': {'medium': 0.8, 'easy': 0.2},
                'hard': {'hard': 0.7, 'medium': 0.2, 'easy': 0.1},
            },
        )
        self.assertEqual(v2_stage_sequence('easy'), ('easy',))
        self.assertEqual(v2_stage_sequence('medium'), ('easy', 'medium'))
        self.assertEqual(v2_stage_sequence('hard'), V2_TD3_STAGES)

    def test_mix_validation_is_strict_and_normalized_without_mutating_input(self) -> None:
        from brain_uav.v2_curriculum import normalize_v2_curriculum_mix

        source = {'medium': 8.0, 'easy': 2.0}
        normalized = normalize_v2_curriculum_mix(source, stage='medium')
        self.assertEqual(normalized, {'easy': 0.2, 'medium': 0.8})
        self.assertEqual(source, {'medium': 8.0, 'easy': 2.0})
        invalid = (
            {},
            {'easy_two_zone': 1.0},
            {'easy': -1.0},
            {'easy': float('nan')},
            {'easy': float('inf')},
            {'easy': True},
        )
        for mix in invalid:
            with self.subTest(mix=mix), self.assertRaises((TypeError, ValueError)):
                normalize_v2_curriculum_mix(mix, stage='easy')

    def test_selector_and_component_seeds_are_reproducible_and_separated(self) -> None:
        from brain_uav.v2_curriculum import V2CurriculumSelector, derive_v2_component_seed

        mix = {'hard': 0.7, 'medium': 0.2, 'easy': 0.1}
        first = V2CurriculumSelector(mix, seed=123)
        second = V2CurriculumSelector(mix, seed=123)
        self.assertEqual(
            [first.sample() for _ in range(30)],
            [second.sample() for _ in range(30)],
        )
        seeds = {
            derive_v2_component_seed(7, 'medium', component)
            for component in ('model', 'curriculum', 'easy_generator', 'replay', 'exploration')
        }
        self.assertEqual(len(seeds), 5)
        self.assertEqual(
            derive_v2_component_seed(7, 'medium', 'replay'),
            derive_v2_component_seed(7, 'medium', 'replay'),
        )
        with self.assertRaises(ValueError):
            derive_v2_component_seed(7, 'easy_two_zone', 'replay')

    def test_bc_schedule_has_exact_stage_local_boundaries(self) -> None:
        from brain_uav.v2_curriculum import v2_bc_lambda

        expected = {
            0: 500.0,
            74_999: 500.0,
            75_000: 150.0,
            149_999: 150.0,
            150_000: 30.0,
            249_999: 30.0,
            250_000: 15.0,
            299_999: 15.0,
            300_000: 5.0,
        }
        for step, value in expected.items():
            with self.subTest(step=step):
                self.assertEqual(v2_bc_lambda(step), value)

    def test_noise_schedule_decays_for_first_half_then_stays_final(self) -> None:
        from brain_uav.v2_curriculum import V2NoiseSchedule

        schedule = V2NoiseSchedule()
        self.assertEqual(schedule.values(0, max_steps=100), (0.020, 0.015, 0.030))
        for actual, expected in zip(
            schedule.values(25, max_steps=100),
            (0.0125, 0.0105, 0.021),
        ):
            self.assertAlmostEqual(actual, expected, places=15)
        self.assertEqual(schedule.values(50, max_steps=100), (0.005, 0.006, 0.012))
        self.assertEqual(schedule.values(100, max_steps=100), (0.005, 0.006, 0.012))
        with self.assertRaises(ValueError):
            V2NoiseSchedule(exploration_initial=-1.0)
        with self.assertRaises(ValueError):
            schedule.values(0, max_steps=0)

    def test_default_stage_budgets_and_learning_rates_are_exact(self) -> None:
        from brain_uav.v2_curriculum import v2_stage_defaults

        self.assertEqual(v2_stage_defaults('easy'), (750_000, 1.5e-4, 2.5e-4))
        self.assertEqual(v2_stage_defaults('medium'), (750_000, 1.5e-4, 2.5e-4))
        self.assertEqual(v2_stage_defaults('hard'), (1_000_000, 1.125e-4, 2.125e-4))


if __name__ == '__main__':
    unittest.main()
