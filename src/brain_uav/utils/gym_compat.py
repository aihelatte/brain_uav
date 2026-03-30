"""Gymnasium compatibility wrapper.

如果你机器上装了 gymnasium，就直接用正式接口。
如果没装，就退回一个最小兼容实现，这样项目也能先跑起来。
"""

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
