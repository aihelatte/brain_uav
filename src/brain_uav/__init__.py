"""Top-level package export.

这个文件的作用很简单：让外部代码可以直接从 `brain_uav`
导入最常用的配置对象。
"""

from .config import ExperimentConfig, ScenarioConfig

__all__ = ["ExperimentConfig", "ScenarioConfig"]
