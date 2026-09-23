# SDDPG Loihi Mapless Navigation 文献整理

## 1. 文献信息

- 论文题目：Reinforcement co-Learning of Deep and Spiking Neural Networks for Energy-Efficient Mapless Navigation with Neuromorphic Hardware
- 作者：Guangzhi Tang, Neelesh Kumar, Konstantinos P. Michmizos
- 研究方向：SNN + DDPG + 无地图机器人导航 + Loihi 神经形态硬件能耗
- 文件位置：`C:/Users/86159/Downloads/Reinforcement co-Learning of Deep and Spiking Neural Networks for Energy-Efficient Mapless Navigation with Neuromorphic Hardware.pdf`

## 2. 论文要解决的问题

这篇文章解决的问题可以概括为：

传统 DRL 方法，比如 DDPG，可以让移动机器人在未知环境中根据局部传感器和目标信息完成无地图导航，但传统深度网络推理能耗较高，不适合资源有限的移动机器人。SNN 具有事件驱动和低功耗优势，但直接训练困难，并且在连续动作值表达方面通常不如 DNN 稳定。因此，论文提出将二者结合，用 SNN actor 负责动作推理，用 DNN critic 提供稳定的价值评估和训练信号。

这一点和本项目的 SNN-TD3 设计逻辑高度一致：SNN actor 用于低能耗策略推理，传统 DNN critic 用于训练过程中的 Q 值估计和梯度更新。

## 3. 方法结构

该文提出的方法是 SDDPG，即 Spiking Deep Deterministic Policy Gradient。

核心结构如下：

| 模块 | 作用 |
|---|---|
| Spiking Actor Network, SAN | 根据机器人状态输出连续动作 |
| Deep Critic Network | 评估状态-动作对的 Q 值 |
| DDPG 框架 | 提供 actor-critic 强化学习训练流程 |
| Loihi 部署 | 将训练后的 SNN actor 部署到神经形态硬件 |

训练时，SAN 生成动作，critic 评估 Q 值，SAN 通过 critic 的梯度信号学习最大化 Q 值。推理时，只需要部署 actor，因此 SNN actor 的低能耗特性具有实际部署意义。

## 4. 状态、动作与任务设置

论文中的机器人是 Turtlebot2 差速轮机器人，使用 RPLIDAR S1 激光雷达。

状态输入包括：

| 输入项 | 维度 |
|---|---:|
| 到目标的距离 | 1 |
| 到目标的方向 | 1 |
| 线速度 | 1 |
| 角速度 | 1 |
| 激光雷达距离观测 | 18 |
| 合计 | 22 |

动作输出为：

| 输出项 | 维度 |
|---|---:|
| 左轮速度 | 1 |
| 右轮速度 | 1 |
| 合计 | 2 |

这和本项目有明显相似性：该文根据目标相对信息、速度状态和局部距离观测输出连续控制动作；本项目根据 UAV 与目标、禁飞区、边界和自身运动状态输出三维连续控制动作。

## 5. 实验设置

| 项目 | 设置 |
|---|---|
| 仿真平台 | Gazebo |
| 真实平台 | Turtlebot2 |
| 中间件 | ROS |
| 传感器 | RPLIDAR S1 |
| 训练环境 | 4 个复杂度递增环境 |
| 训练步数 | 200,000 execution steps |
| episode 上限 | 1000 steps |
| 每步时间 | 0.1 s |
| 仿真测试环境 | 20 m x 20 m |
| 仿真测试起终点 | 200 组随机 start-goal |
| 最小起终距离 | 6 m |
| 真实环境 | 约 215 m2 办公室 |
| 真实任务 | 连续导航到 15 个目标点 |

该文和本项目的相似点：

- 都有仿真环境。
- 都随机生成多组起点和终点。
- 都使用局部观测和目标相对信息进行在线控制。
- 都设置最大 episode 步数为 1000。
- 都使用课程学习使策略逐步适应复杂环境。
- 都关注资源受限移动平台上的低能耗控制。

## 6. 课程学习

该文训练时使用 4 个复杂度递增的环境：

- 简单环境中学习基本导航策略。
- 随着障碍物增加和起终点组合变化，逐步学习复杂策略。
- 作者认为这种 curriculum training 有助于更快收敛和更好泛化。

这可以作为本项目课程学习设计的外部参考。本项目的消融结果显示，去掉课程学习后最终 benchmark 成功率仍接近主模型，但训练过程更不平滑，hard stage 初期表现更差，平均步数和高分位步数也更高。因此，本项目可以将课程学习表述为提升训练平滑性、样本效率和策略稳定性的机制。

## 7. 性能结果

该文报告 SDDPG 在仿真和真实环境中均表现良好。

关键结论：

- SDDPG 比 DDPG 的成功率高约 1% 到 4.2%，具体取决于 forward-propagation timestep size。
- SDDPG 部署到 Loihi 后性能没有明显下降。
- SDDPG 的路径距离和平均速度与 DDPG、DDPG Poisson、move_base 等方法接近。
- 在真实办公室环境中，SDDPG 成功导航到所有目标点。

这个现象对本项目很有参考价值，因为本项目中 SNN-TD3 在 800 个固定 benchmark 中成功率为 99.75%，ANN-TD3 为 99.25%，SNN 也略高于 ANN。这说明 SNN actor + DNN critic 并不必然牺牲性能，已有机器人导航研究中也出现过 SNN 略优于传统网络的结果。

## 8. 能耗结果

该文 Table II 对不同硬件和方法的一次推理能耗进行了比较。

| 方法 | 设备 | Inf/s | 能耗 |
|---|---:|---:|---:|
| DDPG | CPU | 6598 | 8910.84 uJ/inf |
| DDPG | GPU | 3053 | 15252.91 uJ/inf |
| DDPG | Jetson TX2 MAXN | 868 | 2227.41 uJ/inf |
| DDPG | Jetson TX2 MAXQ | 390 | 1171.70 uJ/inf |
| SDDPG T=50 | Loihi | 125 | 131.99 uJ/inf |
| SDDPG T=25 | Loihi | 203 | 67.29 uJ/inf |
| SDDPG T=10 | Loihi | 396 | 28.47 uJ/inf |
| SDDPG T=5 | Loihi | 453 | 15.53 uJ/inf |

作者的核心结论是：SDDPG T=5 在 Loihi 上比 Jetson TX2 MAXQ 上的 DDPG 约节能 75 倍，同时推理速度也更高。

这可以支撑本项目的低能耗动机：类脑方法的事件驱动计算特性确实可能在神经形态硬件上转化为显著能耗优势。

需要注意：

- 该文能耗是 Loihi 硬件实测。
- 本项目目前是 spike-aware theoretical energy estimate。
- 二者证据类型不同，不能直接比较倍率。
- 但二者结论方向一致：SNN 在资源受限平台上具有低能耗潜力。

## 9. 参数量估算

原文没有直接报告 actor 总参数量，但根据文中给出的输入维度和隐藏层规模，可以粗略估算。

已知：

- 输入维度：22
- 隐藏层神经元数量：256, 512 这一行在表中写作 "Neurons per hidden layer for SAN and critic net 256, 512"，结合常见结构和前文讨论，可按两层 256 隐藏层对 SAN actor 做保守估算
- 输出维度：2

若按两层全连接隐藏层，每层 256 个神经元，并包含 bias，则参数量为：

```text
(22 + 1) x 256 = 5,888
(256 + 1) x 256 = 65,792
(256 + 1) x 2 = 514

Total = 72,194
```

即约 7.2 万参数。

本项目 SNN actor 约 1.9 万参数。按这一估算，该文 actor 参数量约为本项目的 3.6 到 3.8 倍。

论文表述时需要谨慎：

- 该参数量不是原文直接报告值。
- 应写作 "roughly estimated" 或 "按网络宽度粗略估算"。
- 可用于说明本项目策略网络较为紧凑，但不能作为严格参数对比结论。

## 10. 与本项目的相同或类似点

| 对比维度 | 该文 | 本项目 |
|---|---|---|
| 策略结构 | SNN actor + DNN critic | SNN actor + TD3 critic |
| 任务类型 | 无地图机器人导航 | 三维 UAV 在线轨迹规划 |
| 仿真环境 | Gazebo | 三维静态禁飞区环境 |
| 起终点 | 随机生成若干组 start-goal | benchmark/curriculum/target switch |
| 输入 | 目标距离、方向、速度、激光雷达 | 目标、速度、禁飞区、边界、几何观测 |
| 输出 | 左右轮速度 | 三维连续动作 |
| 最大步数 | 1000 steps | 1000 steps |
| 推理周期 | 0.1 s | 1 s |
| 课程学习 | 有 | 有 |
| 能耗分析 | Loihi 硬件实测 | spike-aware 理论估计 |
| SNN 性能 | 比 DDPG 高 1% 到 4.2% | 比 ANN-TD3 略高 |

## 11. 与本项目不同的地方

该文更强的地方：

- 完成了真实机器人部署。
- 部署到 Loihi 神经形态硬件。
- 比较了 CPU、GPU、Jetson TX2、Loihi 等不同设备的推理能耗。
- 分析了不同 SNN timestep, T=5/10/25/50, 下的性能-能耗折中。

本项目更突出的地方：

- 任务是三维 UAV 或高速飞行器轨迹规划，而非二维地面机器人导航。
- 有三维静态半球禁飞区约束。
- 有长距离飞行任务。
- 有终端目标捕获需求。
- 有 terminal guidance 机制和对应消融。
- 有 BC 预训练、课程学习、终端引导三类消融实验。
- 有 800 个固定 benchmark 场景。
- 有目标切换实验。
- 对失败类型进行了细分统计，例如 collision、timeout、boundary 等。
- 策略网络规模更紧凑。

## 12. 这篇文献最终可以提炼出的论文作用

### 12.1 支撑 SNN actor + DNN critic 的合理性

该文说明，在连续控制和机器人导航任务中，SNN actor + DNN critic 是一种合理的混合框架。SNN actor 负责低能耗推理，DNN critic 提供稳定的价值评估和训练信号。

可用表述：

> This design is consistent with prior neuromorphic reinforcement learning studies, where the spiking actor is used for energy-efficient action inference while the conventional deep critic provides stable value estimation during training.

中文表述：

> 该设计与已有神经形态强化学习工作中的混合 actor-critic 思路一致，即利用 SNN actor 承担低能耗动作推理，同时保留传统 DNN critic 以提供稳定的价值评估和梯度信号。

### 12.2 支撑 SNN 性能不低于 ANN

该文中 SDDPG 的成功率比 DDPG 高约 1% 到 4.2%。这可以作为本项目中 SNN-TD3 成功率略高于 ANN-TD3 的外部参考。

可用表述：

> Similar observations have been reported in neuromorphic navigation studies, where spiking actor policies achieved comparable or slightly higher navigation success rates than conventional deep actor networks while reducing inference energy.

中文表述：

> 类似现象也出现在已有神经形态导航研究中，即脉冲 actor 策略在降低推理能耗的同时，仍能达到与传统深度 actor 相当甚至略高的导航成功率。

### 12.3 支撑能耗动机和未来部署方向

该文完成了 Loihi 真实部署和硬件能耗测量，这是本项目目前尚未完成的部分。因此，它既能支撑低能耗动机，也可以作为未来工作方向。

可用表述：

> Although this work reports spike-aware theoretical energy estimates rather than hardware measurements, prior Loihi-based navigation studies suggest that the event-driven nature of SNN policies can translate into substantial energy savings on neuromorphic processors. Deploying the proposed SNN-TD3 policy on neuromorphic hardware is therefore an important direction for future work.

中文表述：

> 尽管本文目前报告的是基于 spike-aware 操作数的理论能耗估计，而非神经形态硬件实测结果，但已有 Loihi 导航研究表明，SNN 策略的事件驱动计算特性可以在真实硬件上转化为显著能耗收益。因此，将本文提出的 SNN-TD3 策略进一步部署到神经形态硬件上，是后续工作的重要方向。

### 12.4 支撑模型规模紧凑性

该文 actor 参数量可粗略估算为约 7.2 万，而本项目 SNN actor 约 1.9 万。可以用来说明本项目策略规模较为紧凑。

可用表述：

> Compared with the SDDPG navigation policy, whose actor size can be roughly estimated at about 72k parameters based on the reported hidden-layer width, our SNN actor uses about 19k parameters. This indicates that the proposed policy remains relatively compact while handling a long-range three-dimensional no-fly-zone planning task.

中文表述：

> 根据文中报告的输入维度和隐藏层宽度粗略估算，SDDPG 的 actor 网络规模约为 7.2 万参数；相比之下，本文 SNN actor 约为 1.9 万参数。该对比说明，本文策略网络在规模上较为紧凑，同时仍需处理长距离三维禁飞区轨迹规划任务。

## 13. 推荐放入相关工作或讨论中的综合表述

英文：

> Tang et al. proposed SDDPG, a hybrid reinforcement learning framework consisting of a spiking actor network and a deep critic network for mapless mobile robot navigation. Their results showed that the spiking actor deployed on Loihi achieved comparable or slightly better navigation performance than DDPG while consuming substantially less energy per inference. This provides evidence that SNN-based policies are suitable for energy-constrained robotic navigation. However, their experiments were mainly conducted on ground robot mapless navigation in two-dimensional environments. In contrast, our work focuses on long-range three-dimensional UAV trajectory planning under no-fly-zone constraints, and further evaluates terminal guidance, curriculum learning, expert pretraining, and target-switching robustness.

中文：

> Tang 等人提出了 SDDPG 框架，将脉冲 actor 与深度 critic 结合用于移动机器人无地图导航，并在 Loihi 神经形态芯片上验证了显著的推理能耗优势。该工作表明，SNN 策略不仅可以用于标准连续控制任务，也可以扩展到具有障碍物的机器人导航场景。然而，该方法主要面向二维地面机器人导航，任务空间、动力学复杂度和安全约束均不同于三维 UAV 禁飞区在线轨迹规划。相比之下，本文进一步在长距离三维飞行、静态禁飞区规避、终端目标捕获和目标切换场景下评估 SNN-TD3 策略。

## 14. 最终判断

这篇文章应归为 A 档重点参考文献。

它最适合支撑：

- SNN actor + DNN critic 的结构合理性。
- SNN-RL 可以用于机器人导航，而不只是简单连续控制 benchmark。
- SNN 在导航任务中可以达到不低于传统 DNN actor 的成功率。
- SNN 在 Loihi 等神经形态硬件上具有真实低能耗潜力。
- 课程学习在复杂导航任务中的合理性。

它不能直接支撑：

- 三维 UAV 禁飞区规划已经被已有 SNN 方法解决。
- 长距离飞行和终端精确制导已经被该文覆盖。
- 目标切换鲁棒性已经被该文验证。
- 本项目的理论能耗估计可以和 Loihi 实测能耗直接倍率比较。

最适合的论文定位：

该文是本项目相关工作中用于支撑 SNN-RL 导航可行性和低能耗动机的核心参考，同时也可作为未来硬件部署方向的依据。
