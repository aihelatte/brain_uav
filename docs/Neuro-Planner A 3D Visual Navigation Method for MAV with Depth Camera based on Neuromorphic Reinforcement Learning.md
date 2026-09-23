# Neuro-Planner MAV 3D Visual Navigation 文献整理

## 1. 文献信息

- 论文题目：Neuro-Planner: A 3D Visual Navigation Method for MAV with Depth Camera based on Neuromorphic Reinforcement Learning
- 作者：Junjie Jiang, Delei Kong, Kuanxu Hou, Xinjie Huang, Hao Zhuang, Zheng Fang
- 研究方向：MAV 三维视觉导航 + 深度相机 + SNN actor + DNN critic + HDDPG
- 文件位置：`C:/Users/86159/Downloads/Neuro-Planner A 3D Visual Navigation Method for MAV with Depth Camera based on Neuromorphic Reinforcement Learning.pdf`

## 2. 论文要解决的问题

这篇文章关注的是微型飞行器 MAV 在未知或半结构化环境中的三维视觉导航问题。传统 MAV 视觉导航通常依赖建图、路径规划和轨迹跟踪，计算资源需求较高，并且面对陌生环境时鲁棒性有限。作者希望将深度强化学习的交互学习能力和 SNN 的时序计算、低功耗潜力结合起来，实现基于深度相机的 MAV 三维视觉导航。

这篇文章适合作为本项目的强相关对比文献，因为它同样面向飞行器三维导航，也采用 SNN actor + DNN critic 的混合 actor-critic 框架。但它的应用场景更偏室内低速视觉避障，而本项目更偏长距离三维禁飞区在线轨迹规划和终端制导。

## 3. 方法结构

该文提出的方法称为 Neuro-Planner，训练框架称为 HDDPG, Hybrid Deep Deterministic Policy Gradient。

| 模块 | 作用 |
|---|---|
| Spiking Actor Network, SAN | 使用 TS-LIF 脉冲神经元，根据状态输出 MAV 控制动作 |
| Deep Critic Network, DCN | 使用传统 DNN 评估状态-动作对的 Q 值 |
| HDDPG | 基于 DDPG 的混合 actor-critic 强化学习训练框架 |
| STBP/BPTT/SLAYER | 对比三种 SNN 训练框架 |
| Gazebo + ROS + PX4 | 构建 MAV 软件在环仿真系统 |

训练阶段，SAN 生成动作，DCN 评估 Q 值，并通过 critic 梯度训练 SNN actor。测试阶段，只使用训练好的 SAN 进行动作推理。

## 4. 状态、动作与深度图处理

该文的状态为：

```text
s_k = [r_k, theta_k, phi_k, v_xy,k, v_yaw,k, v_z,k, d_k]^T
```

其中：

| 状态项 | 含义 |
|---|---|
| r, theta, phi | 目标点在 MAV 局部球坐标系下的相对位置 |
| v_xy, v_yaw, v_z | 当前水平线速度、偏航角速度和垂直速度 |
| d_k | 深度图经池化得到的视觉特征 |

动作输出为三维速度控制量：

| 动作 | 含义 |
|---|---|
| v_xy | 水平速度 |
| v_yaw | 偏航角速度 |
| v_z | 垂直速度 |

该文虽然使用 Kinect 深度相机，但没有使用 CNN 对完整深度图做端到端特征提取。作者将深度图划分为若干图像块，并计算每个图像块中有效像素的平均深度，得到紧凑视觉特征。

根据 Table I：

```text
Channels of state Cs = 18
Channels of normalized state C_s~ = 21
Channels of action Ca = 4
```

原始状态中非深度部分为：

```text
目标相对位置 3 维 + 当前速度 3 维 = 6 维
```

因此深度图池化特征维度可反推为：

```text
18 - 6 = 12
```

也就是说，深度图最终大致被压缩成约 1 x 12 的深度特征向量。论文没有明确说明图像块是 3 x 4 还是 4 x 3，但从状态维度可以判断，池化后的深度特征约为 12 个数。

这一点很重要：Neuro-Planner 虽然是视觉导航，但并不是重型视觉网络，而是将视觉输入压缩为低维结构化距离特征。

## 5. 实验设置

| 项目 | 设置 |
|---|---|
| 飞行器 | Iris MAV |
| 传感器 | Kinect depth camera |
| 仿真系统 | ROS + Gazebo + PX4 + CUDA |
| 训练环境 | 4 个复杂度递增环境 |
| 训练环境尺寸 | 12 m x 12 m x 3 m |
| 训练环境障碍物 | 0 / 4 / 6 / 8 个 cuboid pillars |
| 训练 episode | 100 / 200 / 300 / 400 |
| 评估环境 1 | 类似训练环境 #4，但改变部分柱体高度 |
| 评估环境 2 | 20 m x 20 m x 3 m，21 个 cuboid pillars |
| 每个评估环境 | 随机生成 100 组 start-goal |
| 重复次数 | 每种方法重复 3 次取平均 |
| 评价内容 | success rate, average distance, average speed, success/failure trajectories |

该文的训练环境和评估环境明显偏室内低速避障场景。障碍物是长条形立方体柱子，MAV 依靠前向深度相机进行局部感知和避障。

## 6. 速度与任务尺度分析

该文动作映射公式中，水平速度为：

```text
v_xy = 0.225 * (a1 + a2) + 0.05
```

由于：

```text
a1, a2 in [0, 1]
```

所以水平速度范围约为：

```text
v_xy in [0.05, 0.5] m/s
```

训练环境尺寸为 12 m x 12 m x 3 m。如果起点和终点随机生成，平面内最远直线距离约为：

```text
sqrt(12^2 + 12^2) ≈ 17 m
```

若路径长度约为 10-20 m，飞行速度约为 0.2-0.5 m/s，则飞行时间大致为几十秒量级：

```text
10 m / 0.5 m/s = 20 s
17 m / 0.5 m/s = 34 s
17 m / 0.3 m/s ≈ 57 s
```

因此，Neuro-Planner 更适合被理解为室内低速视觉避障型 MAV，例如室内巡检、搬运，或人员难以进入的危险环境导航任务。

## 7. 性能结果

该文在两个陌生评估环境中报告了 DDPG 与 HDDPG-STBP 的成功率对比。

| 环境 | DDPG 成功率 | HDDPG-STBP 相对提升 |
|---|---:|---:|
| Evaluation environment #1 | 82.0% | 最高约 +4.3% |
| Evaluation environment #2 | 77.0% | 最高约 +5.3% |

因此，HDDPG-STBP 在两个环境中大约可以达到：

```text
Environment #1: about 86.3%
Environment #2: about 82.3%
```

这说明 SNN actor 在 MAV 三维视觉导航任务中可以达到不低于传统 DDPG actor 的表现，甚至略高。

不过，由于该文面向未知环境，且深度图被压缩为约 12 维粗粒度特征，模型无法获得障碍物的精确全局几何位置。因此 80% 多的成功率不能简单理解为方法弱，而应理解为在未知视觉感知受限条件下的结果。

## 8. 网络规模估算

Table I 给出：

```text
Channels of normalized state = 21
Channels of action = 4
Hidden layers of SAN = 3
Neurons in each hidden layer of SAN = 512
```

若按 3 层隐藏层、每层 512 个神经元，并包含 bias 粗略估算 SAN actor 参数量：

```text
(21 + 1) x 512 = 11,264
(512 + 1) x 512 = 262,656
(512 + 1) x 512 = 262,656
(512 + 1) x 4 = 2,052

Total = 538,628
```

即约 53.9 万参数。

本项目 SNN actor 约 1.9 万参数。按该估算，Neuro-Planner 的 SNN actor 大约是本项目 SNN actor 的 28 倍。

该对比需要谨慎使用：

- 该文参数量不是原文直接报告值，而是根据网络结构粗略估算。
- Neuro-Planner 使用深度相机池化特征，任务是未知室内视觉避障。
- 本项目使用结构化几何观测，任务是静态禁飞区长距离在线轨迹规划。
- 因此不能简单说谁更优，但可以说明本项目策略网络更紧凑。

## 9. 项目和 Neuro-Planner 的相同点

| 维度 | 相同点 |
|---|---|
| 研究对象 | 都面向 MAV / UAV 类飞行器的三维导航或轨迹规划问题 |
| 方法框架 | 都采用 SNN actor + 传统 DNN critic 的混合 actor-critic 架构 |
| 仿真验证 | 都在三维仿真环境中进行训练和评估 |
| 起终点设置 | 都包含随机生成起点和目标点的实验设置 |
| 课程学习 | 都通过由易到难的环境或任务设置帮助策略逐步适应复杂场景 |
| 低能耗动机 | 都以资源受限飞行平台为背景，强调 SNN 的低能耗潜力 |

## 10. 项目和 Neuro-Planner 的不同点

| 维度 | Neuro-Planner | 本项目 |
|---|---|---|
| 应用场景 | 室内或半结构化未知环境 MAV 视觉导航 | 室外/大尺度/长距离三维禁飞区在线轨迹规划 |
| 任务重点 | 局部避障、点到点导航 | 长距离飞行、禁飞区规避、终端目标捕获 |
| 飞行速度 | 水平速度约 0.05-0.5 m/s，偏低速 | 面向高速飞行器 / UAV 制导语境 |
| 环境尺度 | 训练 12 m x 12 m x 3 m，测试最大 20 m x 20 m x 3 m | 更强调长距离、大范围飞行任务 |
| 障碍物形式 | 室内柱状 cuboid pillars | 三维静态半球禁飞区 |
| 观测方式 | Kinect 深度相机，前向局部观测 | 目标、禁飞区、边界、飞行状态等结构化观测 |
| 网络规模 | SAN actor 粗略估算约 53.9 万参数 | SNN actor 约 1.9 万参数 |
| 成功率 | DDPG 77%-82%，HDDPG-STBP 最高提升 4.3%-5.3% | SNN benchmark 99.75%，ANN benchmark 99.25% |
| 实验完整性 | 比较 STBP/BPTT/SLAYER 和不同 time step | 有主实验、benchmark、BC/课程/终端引导消融、目标切换实验 |

## 11. 关键要点

### 11.1 Neuro-Planner 更适合代表室内低速视觉避障型 MAV

Neuro-Planner 的环境尺寸较小，障碍物是柱状立方体，水平速度约 0.05-0.5 m/s，更像复杂室内环境下的低速局部避障任务，例如室内巡检、搬运或危险环境探索。

### 11.2 本项目更适合定位为长距离高要求制导任务

如果本项目面向导弹或不可回收飞行器的精准制导任务，任务失败代价高，不能依赖反复尝试。因此，接近 100% 的成功率、稳定终端捕获和安全绕障更加关键。

### 11.3 两者对成功率的要求不同

Neuro-Planner 面向未知环境，输入受限，80% 多的成功率已经能够说明其具有导航能力。相比之下，本项目如果面向远程制导或一次性飞行平台，则 99% 以上成功率是更合理的需求。

### 11.4 两者对计算资源的取舍不同

Neuro-Planner 使用约 53.9 万参数的 SNN actor，可能更重视复杂室内视觉避障能力。本项目使用约 1.9 万参数的 SNN actor，更强调资源受限平台上的小模型、低计算代价和高成功率。

### 11.5 Neuro-Planner 的深度图被强压缩

Neuro-Planner 原始状态维度 Cs = 18，其中目标相对位置和速度共 6 维，因此深度图池化特征约为 12 维。也就是说，它不是端到端处理完整图像，而是将深度图划分成若干块后做平均池化，得到约 1 x 12 的粗粒度前向距离特征。

### 11.6 深度图压缩解释了其成功率限制

由于 Neuro-Planner 面向未知环境，模型无法获得障碍物的精确全局几何位置，只能依赖前向深度图的粗粒度池化特征，因此对障碍物结构的感知较模糊，碰撞和失败更容易发生。这可以部分解释其成功率处于 80% 多，而不是接近 100%。

### 11.7 本项目终端捕获相对尺度更严格

Neuro-Planner 的目标捕获半径为 0.54 m。若按 10-15 m 的典型室内飞行距离估算，其终端容差占航程比例为：

```text
0.54 / 10 = 5.4%
0.54 / 15 = 3.6%
```

本项目 benchmark 的目标捕获半径为 5 km，标称飞行距离为 1750 km：

```text
5 / 1750 = 0.286%
```

若按 benchmark 起终距离范围 1575-1925 km 计算：

```text
5 / 1575 ≈ 0.317%
5 / 1925 ≈ 0.260%
```

因此，本项目的相对终端容差约为 0.26%-0.32%，而 Neuro-Planner 约为 3.6%-5.4%。Neuro-Planner 的相对终端容差大约是本项目的十余倍。

这说明本项目不仅飞行距离更长，而且终端捕获在相对尺度上更精细。在策略网络参数量更小的情况下，实现这种长距离高精度终端捕获更加困难。

### 11.8 Terminal guidance 的必要性

由于本项目要求用更少参数实现更精细的终端捕获，因此引入 terminal guidance 是合理的。该机制可以强化模型学习终端精准制导能力。

本项目消融实验也验证了这一点：去除 terminal guidance 后，benchmark 成功率接近 0%，失败主要表现为 timeout，说明该机制对终端精准制导和最终收敛至关重要。

## 12. 可用于论文的表述

### 12.1 英文表述

> Neuro-Planner focuses on indoor MAV visual navigation with a depth camera, where the vehicle performs low-speed obstacle avoidance in relatively compact environments. According to its velocity mapping, the horizontal speed is bounded between approximately 0.05 and 0.5 m/s, and the training environments are 12 m x 12 m x 3 m. In contrast, our work targets long-range three-dimensional UAV trajectory planning under static no-fly-zone constraints, where onboard computational efficiency, compact policy size, and terminal target capture become more critical. Moreover, while Neuro-Planner uses a relatively large spiking actor network, roughly estimated at over 0.5 million parameters, our SNN actor contains about 19k parameters while achieving a 99.75% success rate on the fixed benchmark.

> The terminal tolerance is also much stricter in our task. Neuro-Planner uses a goal threshold of 0.54 m; for a typical 10-15 m indoor flight, this corresponds to approximately 3.6%-5.4% of the travel distance. In contrast, our benchmark uses a 5 km capture radius over a nominal 1750 km flight distance, corresponding to only about 0.286%. Therefore, our terminal tolerance is more than an order of magnitude tighter in relative scale. This stricter terminal requirement, combined with a much smaller policy network, motivates the use of terminal guidance to strengthen precise final-stage target capture. The ablation without terminal guidance further confirms its necessity, as the success rate drops to nearly zero and most failures become timeouts.

### 12.2 中文表述

> Neuro-Planner 主要面向室内 MAV 视觉导航任务，飞行器依靠深度相机在较小尺度环境中进行低速避障。根据其速度映射公式，水平速度约束在 0.05-0.5 m/s，训练环境尺寸为 12 m x 12 m x 3 m，因此更接近复杂室内环境下的低速局部避障任务。相比之下，本文关注的是长距离三维 UAV 禁飞区在线轨迹规划，更强调机载计算资源受限条件下的小模型策略、低计算代价、远程制导和终端目标捕获能力。此外，Neuro-Planner 的脉冲 actor 网络规模可粗略估算为 50 万参数以上，而本文 SNN actor 约为 1.9 万参数，在更紧凑的策略规模下仍在固定 benchmark 中取得 99.75% 的成功率。

> 与 Neuro-Planner 相比，本文任务的终端捕获要求在相对尺度上更加严格。Neuro-Planner 的目标到达阈值为 0.54 m，若按 10-15 m 的典型室内飞行距离估算，其终端容差约占航程的 3.6%-5.4%。相比之下，本文 benchmark 中的目标捕获半径为 5 km，标称飞行距离为 1750 km，仅约占 0.286%。因此，本文的相对终端容差比 Neuro-Planner 至少严格一个数量级。在策略网络参数量更小的情况下，实现这种长距离高精度终端捕获更加困难，这也解释了本文引入 terminal guidance 的必要性。消融实验进一步验证了这一点：去除 terminal guidance 后，benchmark 成功率接近 0%，失败主要表现为 timeout，说明该机制对终端精准制导和最终收敛至关重要。

## 13. 最终判断

这篇文章应归为 A 档重点参考文献。

它最适合用于：

- 说明 SNN actor + DNN critic 已经被用于 MAV 三维导航。
- 说明 SNN actor 在 MAV 三维任务中可以不低于 DDPG，甚至略高。
- 作为室内低速视觉避障 MAV 与本项目长距离禁飞区制导任务的对比对象。
- 支撑本项目在模型规模、成功率、终端捕获精度和任务尺度上的差异化叙事。

它不适合直接用于：

- 证明真实神经形态硬件低能耗部署。
- 证明长距离禁飞区规划已经被已有 SNN-MAV 方法解决。
- 证明目标切换鲁棒性。
- 直接对比本项目的 spike-aware energy 估计。
