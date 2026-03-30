"""Legacy environment file kept for reference.

当前项目真正使用的是 `static_no_fly_env_runtime.py`。
这个文件保留是为了兼容之前的开发过程，不建议继续从这里导入。
"""

from .static_no_fly_env_runtime import StaticNoFlyTrajectoryEnv, Zone

__all__ = ['StaticNoFlyTrajectoryEnv', 'Zone']
