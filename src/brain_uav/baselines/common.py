from __future__ import annotations

import numpy as np


def heading_to_action(current_gamma: float, current_psi: float, direction: np.ndarray, limits: np.ndarray) -> np.ndarray:
    xy_norm = max(float(np.linalg.norm(direction[:2])), 1e-6)
    desired_gamma = float(np.arctan2(direction[2], xy_norm))
    desired_psi = float(np.arctan2(direction[1], direction[0]))
    d_gamma = np.clip(desired_gamma - current_gamma, -limits[0], limits[0])
    d_psi = np.clip(wrap_angle(desired_psi - current_psi), -limits[1], limits[1])
    return np.array([d_gamma, d_psi], dtype=np.float32)


def wrap_angle(value: float) -> float:
    return ((value + np.pi) % (2 * np.pi)) - np.pi

