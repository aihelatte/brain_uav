"""Central configuration definitions for the whole project.

读项目时建议先看这个文件，因为环境、模型、训练脚本都会从这里拿参数。
如果你后面要调实验，大多数时候改这里就够了。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ScenarioConfig:
    """Environment-side parameters.

    这一组参数决定“飞行任务长什么样”：
    - 飞机飞多快
    - 世界范围多大
    - 禁飞区有多大
    - 每一步动作允许转多少
    """

    dt: float = 1.0
    speed: float = 25.0
    gamma_max: float = 0.6
    delta_gamma_max: float = 0.12
    delta_psi_max: float = 0.2
    goal_radius: float = 45.0
    world_xy: float = 800.0
    world_z_min: float = 1.0
    world_z_max: float = 400.0
    max_steps: int = 200
    min_no_fly_zones: int = 1
    max_no_fly_zones: int = 3
    no_fly_radius_range: tuple[float, float] = (60.0, 140.0)
    warning_distance: float = 100.0
    boundary_warning_distance: float = 100.0
    ground_warning_height: float = 60.0
    nearest_zone_count: int = 3


@dataclass(slots=True)
class RewardConfig:
    """Reward weights used by reinforcement learning.

    这组参数控制策略会“偏好什么行为”：
    - 更快接近目标
    - 更愿意冲向终点
    - 不要撞禁飞区
    - 不要提前贴近边界
    - 不要用 timeout 的方式苟活
    - 不要动作抖动太大
    """

    progress_weight: float = 2.2
    goal_reward: float = 2500.0
    zone_penalty_weight: float = 450.0
    zone_penalty_cap: float = 1200.0
    boundary_soft_penalty_weight: float = 450.0
    boundary_soft_penalty_cap: float = 450.0
    ground_soft_penalty_weight: float = 180.0
    ground_soft_penalty_cap: float = 180.0
    collision_penalty: float = 3000.0
    step_penalty: float = 3.0
    smoothness_weight: float = 2.0
    boundary_penalty: float = 3000.0
    timeout_penalty: float = 1000.0


@dataclass(slots=True)
class TrainingConfig:
    """Model and optimizer settings.

    这一组参数主要影响训练过程：
    - 学习率
    - batch size
    - TD3 噪声参数
    - SNN 时间窗口 T
    - Actor 冻结保护期

    这里把探索噪声从极小值稍微放大一点，让策略在不严重破坏 BC 底子的前提下，
    有机会跳出“保守耗时直到 timeout”的局部最优。
    """

    seed: int = 7
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.015
    noise_clip: float = 0.03
    policy_delay: int = 2
    exploration_noise: float = 0.02
    replay_size: int = 100_000
    warmup_steps: int = 256
    actor_freeze_steps: int = 5_000
    bc_epochs: int = 10
    snn_time_window: int = 4
    hidden_dim: int = 128
    device: str = 'cpu'


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level config container.

    这个对象会把环境配置、奖励配置、训练配置统一打包，
    方便在脚本里一次性传来传去。
    """

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: Path = Path('outputs')
    data_dir: Path = Path('data')

    def to_dict(self) -> dict:
        """Convert config to a plain dict so it can be saved into checkpoints/JSON."""

        payload = asdict(self)
        payload['output_dir'] = str(self.output_dir)
        payload['data_dir'] = str(self.data_dir)
        return payload
