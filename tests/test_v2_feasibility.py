"""Deterministic geometric-feasibility tests for V2 random scenarios."""

from __future__ import annotations

import unittest

from brain_uav.config import ScenarioConfig
from brain_uav.envs.v2_feasibility import (
    V2FeasibilityConfig,
    check_v2_geometric_feasibility,
)
from brain_uav.geometry import Box, NoFlyZone, Sphere


def _scenario_config(**overrides) -> ScenarioConfig:
    values = {
        'world_xy': 100.0,
        'world_z_min': 0.1,
        'world_z_max': 100.0,
    }
    values.update(overrides)
    return ScenarioConfig(**values)


class TestV2Feasibility(unittest.TestCase):
    def test_config_validation_and_defaults(self):
        config = V2FeasibilityConfig()
        self.assertEqual(config.xy_step, 50.0)
        self.assertEqual(config.z_step, 25.0)
        self.assertEqual(config.max_expanded_nodes, 100_000)
        self.assertEqual(config.clearance, 40.0)

        invalid = (
            {'xy_step': 0.0},
            {'xy_step': float('nan')},
            {'z_step': -1.0},
            {'max_expanded_nodes': 0},
            {'max_expanded_nodes': 1.5},
            {'clearance': -1.0},
            {'clearance': float('inf')},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                V2FeasibilityConfig(**values)

    def test_unobstructed_and_detour_scenarios_are_reachable(self):
        scenario = _scenario_config()
        config = V2FeasibilityConfig(
            xy_step=20.0,
            z_step=20.0,
            max_expanded_nodes=10_000,
            clearance=0.0,
        )
        start = [-80.0, 0.0, 50.0]
        goal = [80.0, 0.0, 50.0]

        direct = check_v2_geometric_feasibility(start, goal, [], scenario, config)
        blocker = NoFlyZone('blocker', Box([0.0, 0.0, 50.0], 20.0, 40.0, 40.0))
        detour = check_v2_geometric_feasibility(start, goal, [blocker], scenario, config)

        self.assertTrue(direct.reachable)
        self.assertEqual(direct.reason, 'reachable')
        self.assertTrue(detour.reachable)
        self.assertGreater(detour.expanded_nodes, 0)

    def test_sparse_visibility_graph_has_bounded_exact_edge_checks(self):
        class RecordingZone(NoFlyZone):
            def __init__(self):
                super().__init__('blocker', Sphere([0.0, 0.0, 50.0], 10.0))
                self.segment_radii = []

            def violates_segment(self, start, end, uav_radius=0.0):
                self.segment_radii.append(float(uav_radius))
                return super().violates_segment(start, end, uav_radius=uav_radius)

        scenario = _scenario_config()
        zone = RecordingZone()
        result = check_v2_geometric_feasibility(
            [-80.0, 0.0, 50.0],
            [80.0, 0.0, 50.0],
            [zone],
            scenario,
            V2FeasibilityConfig(clearance=40.0),
        )

        self.assertTrue(result.reachable)
        self.assertEqual(result.algorithm, 'sparse_visibility_graph')
        self.assertEqual(result.expanded_nodes, result.examined_nodes)
        self.assertGreater(result.edge_checks, 0)
        self.assertEqual(result.edge_checks, len(zone.segment_radii))
        self.assertTrue(all(radius == 40.0 for radius in zone.segment_radii))
        self.assertLessEqual(result.examined_nodes, 16)
        self.assertLessEqual(result.edge_checks, 135)

    def test_accepted_path_edges_are_exactly_checked_against_broadphase_skips(self):
        class RecordingZone(NoFlyZone):
            def __init__(self):
                super().__init__('far', Sphere([0.0, 90.0, 50.0], 10.0))
                self.segment_radii = []

            def violates_segment(self, start, end, uav_radius=0.0):
                self.segment_radii.append(float(uav_radius))
                return super().violates_segment(start, end, uav_radius=uav_radius)

        zone = RecordingZone()
        result = check_v2_geometric_feasibility(
            [-80.0, 0.0, 50.0],
            [80.0, 0.0, 50.0],
            [zone],
            _scenario_config(),
            V2FeasibilityConfig(clearance=40.0),
        )

        self.assertTrue(result.reachable)
        self.assertEqual(zone.segment_radii, [40.0])
        self.assertEqual(result.edge_checks, 1)

    def test_full_wall_is_unreachable_and_edges_may_not_cross_it(self):
        scenario = _scenario_config()
        config = V2FeasibilityConfig(
            xy_step=25.0,
            z_step=25.0,
            max_expanded_nodes=10_000,
            clearance=0.0,
        )
        wall = NoFlyZone('wall', Box([0.0, 0.0, 50.0], 10.0, 200.0, 100.0))
        result = check_v2_geometric_feasibility(
            [-80.0, 0.0, 50.0],
            [80.0, 0.0, 50.0],
            [wall],
            scenario,
            config,
        )

        self.assertFalse(result.reachable)
        self.assertEqual(result.reason, 'no_path')

    def test_expansion_limit_is_distinct_and_deterministic(self):
        scenario = _scenario_config()
        wall = NoFlyZone('wall', Box([0.0, 0.0, 50.0], 10.0, 200.0, 100.0))
        config = V2FeasibilityConfig(
            xy_step=25.0,
            z_step=25.0,
            max_expanded_nodes=1,
            clearance=0.0,
        )
        args = ([-80.0, 0.0, 50.0], [80.0, 0.0, 50.0], [wall], scenario, config)

        first = check_v2_geometric_feasibility(*args)
        second = check_v2_geometric_feasibility(*args)

        self.assertEqual(first, second)
        self.assertFalse(first.reachable)
        self.assertEqual(first.reason, 'expansion_limit')
        self.assertEqual(first.expanded_nodes, 1)

    def test_invalid_endpoints_and_zone_values_are_rejected(self):
        scenario = _scenario_config()
        config = V2FeasibilityConfig(clearance=0.0)
        with self.assertRaises(ValueError):
            check_v2_geometric_feasibility([0.0, 0.0], [1.0, 2.0, 3.0], [], scenario, config)
        with self.assertRaises(ValueError):
            check_v2_geometric_feasibility(
                [0.0, 0.0, 10.0], [101.0, 0.0, 10.0], [], scenario, config
            )
        with self.assertRaises(TypeError):
            check_v2_geometric_feasibility(
                [0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [object()], scenario, config
            )


if __name__ == '__main__':
    unittest.main()
