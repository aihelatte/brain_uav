"""Predefined benchmark scenarios.

这个文件保存“固定测试场景”，方便你每次都在同样的场景上比较 SNN 和 ANN，
这样结果才公平。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class NamedScenario:
    """A benchmark scenario plus its human-readable name and description."""

    name: str
    description: str
    scenario: dict[str, Any]


def build_benchmark_scenarios() -> list[NamedScenario]:
    """Return the standard benchmark suite used by evaluation scripts."""

    return [
        NamedScenario(
            name='single_detour',
            description='Single hemisphere blocks the direct route.',
            scenario={
                'state': [-620.0, 0.0, 120.0, 0.0, 0.0],
                'goal': [620.0, 0.0, 120.0],
                'zones': [
                    {'center_xy': [0.0, 0.0], 'radius': 170.0},
                ],
            },
        ),
        NamedScenario(
            name='double_channel',
            description='Two zones form a narrow passage.',
            scenario={
                'state': [-620.0, -60.0, 130.0, 0.0, 0.0],
                'goal': [620.0, 60.0, 130.0],
                'zones': [
                    {'center_xy': [-50.0, -150.0], 'radius': 160.0},
                    {'center_xy': [80.0, 150.0], 'radius': 160.0},
                ],
            },
        ),
        NamedScenario(
            name='boundary_margin',
            description='Goal route passes close to the zone boundary.',
            scenario={
                'state': [-640.0, -220.0, 110.0, 0.0, 0.0],
                'goal': [620.0, -100.0, 110.0],
                'zones': [
                    {'center_xy': [40.0, -130.0], 'radius': 145.0},
                    {'center_xy': [280.0, 150.0], 'radius': 120.0},
                ],
            },
        ),
        NamedScenario(
            name='wall_pressure',
            description='Obstacle wall pressures the planner into a safe corridor.',
            scenario={
                'state': [-650.0, 0.0, 140.0, 0.0, 0.0],
                'goal': [650.0, 0.0, 140.0],
                'zones': [
                    {'center_xy': [-120.0, -220.0], 'radius': 140.0},
                    {'center_xy': [-120.0, 0.0], 'radius': 140.0},
                    {'center_xy': [-120.0, 220.0], 'radius': 140.0},
                    {'center_xy': [160.0, -220.0], 'radius': 140.0},
                    {'center_xy': [160.0, 220.0], 'radius': 140.0},
                ],
            },
        ),
    ]
