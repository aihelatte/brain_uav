from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class Env:
        metadata: dict[str, Any] = {}

    @dataclass
    class Box:
        low: np.ndarray
        high: np.ndarray
        shape: tuple[int, ...]
        dtype: Any = np.float32

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

    class _Spaces:
        Box = Box

    class _Gym:
        Env = Env

    gym = _Gym()
    spaces = _Spaces()

