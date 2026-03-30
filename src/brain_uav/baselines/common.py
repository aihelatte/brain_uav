"""Shared helper functions for baseline planners."""

from __future__ import annotations

import numpy as np


def heading_to_action(current_gamma: float, current_psi: float, direction: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Convert a desired 3D direction vector into incremental control actions.

    环境里的动作不是“绝对角度”，而是：
    - 当前时刻俯仰角改多少
    - 当前时刻偏航角改多少
    所以这里负责把方向向量翻译成动作。
    """

    xy_norm = max(float(np.linalg.norm(direction[:2])), 1e-6)
    desired_gamma = float(np.arctan2(direction[2], xy_norm))
    desired_psi = float(np.arctan2(direction[1], direction[0]))
    d_gamma = np.clip(desired_gamma - current_gamma, -limits[0], limits[0])
    d_psi = np.clip(wrap_angle(desired_psi - current_psi), -limits[1], limits[1])
    return np.array([d_gamma, d_psi], dtype=np.float32)


def wrap_angle(value: float) -> float:
    """Wrap angle into [-pi, pi]."""

    return ((value + np.pi) % (2 * np.pi)) - np.pi
