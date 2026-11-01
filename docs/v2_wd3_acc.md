# 固定 actor / 固定数据 WD3、ACC 式 critic 诊断（待审）

本文件是运行前预注册，不是实验结果。独立入口：
`python -m brain_uav.scripts.diagnose_v2_wd3_acc`。
未加 `--launch` 时只输出计划，不加载模型、不重放、不拟合、不创建输出目录。
不调用 formal trainer、GPUCompute、MPS 或课程；不修改训练默认值。
SNN 是 actor 类型，其两支 critic 仍为 ANN。

## 已只读核对的输入与缺口（2026-11-01）

当前代码 HEAD：`33f03a4ca4f8f91932315e9589443547d3c41548`。

| 源模型（easy、seed23、500k、原 min） | 本地 checkpoint | SHA256 |
|---|---|---|
| ANN，critic_lr=2e-4 | `E:\wurenji\result2\v2_ann_critic_lr_round2\20261020_221413\ann_clr_2e-4_seed23\v2_ann_td3_easy_failed.pt` | `c48c94a8fa1d7a2ca2a833cfcf9b103e61655cf089c140237d1ded886449e8af` |
| SNN，critic_lr=1e-3 | `E:\wurenji\result2\v2_snn_critic_lr_confirm_round1\20261024_102749\snn_clr_1e-3_seed23\v2_snn_td3_easy_failed.pt` | `c5cb20f60c8181fa03bc1fcd31901cfeae218b1eecd414b186abd15f5710f36d` |

两份已读取实际 checkpoint，仅检查配置、权重结构和 hash，未做模型前向/重放/拟合。
critic 学习率同时校验 formal config 和保存的联合 Adam group，不能通过 CLI 悄悄改掉。
经 CPU 严格加载得到的 actor `cal.state_hash(state_dict())`：
ANN=`7019cead2aa8405d707e86ad86ac0e9114cf536a05f35fa2d3a62e90e353e782`；
SNN=`b07c76e950fef4fe5c9959eb2b45fe204e16fe65d4c0ef5894a201d07942ed96`。

固定 easy 池实际通过 `load_v2_validation_pool(expected_level='easy', expected_count=100)` 读取：
`E:\wurenji\train_result\v2_box085_round1_20261003_003948\validation_pools_600km_seed20260904\v2_validation_easy.json`；
content_digest=`2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b`。
该池只能用于身份排除及固定20场评分，不能进入训练/ACC 校准。

| 现有产物 | 可复用范围 / 缺失 |
|---|---|
| `E:\wurenji\result2\v2_critic_reliability_round1\20261026_171846_464722553` 的 B 20条 | checkpoint 与上述 SNN 完全一致，20个ID唯一，动作、rewards、完整 states、pre_action_hashes、终态和折扣回报齐备，长度关系完整；**全部20场复用评分**，不重新飞行，不进入训练/校准。真实严格重放与冻结actor动作核对仍是后续执行门槛，本轮未执行20场重放。 |
| 同历史的 A 20条 | 是另一份 SNN、critic_lr=2.5e-4；不能当成当前 ANN 或 SNN B 数据。 |
| `E:\wurenji\result2\v2_original_critic_eval_round1\20261031_193709_930923931\ann_min_seed23\other` | checkpoint 身份正确，但 transitions 只有 scenario_id/step/Q1/Q2/G/done；缺动作、observation/next_observation、reward、完整状态/hash，不能据此构建训练或校准数据。 |
| ANN 原训练可视化38场、SNN 原训练可视化39场 | 全部 action 长度及三维位置长度完整；全部缺逐步 rewards、五维 states、pre_action_hashes、终态 terminated/truncated/final_info、冻结 actor/checkpoint 身份。来自变化中的带探索训练策略，不能冒充终态冻结 actor 的无探索校准/评分回报。 |

**数据规模**：每模型共60场，30训练+10校准+20固定评分，按完整场景隔离。
SNN 复用全部20场 B 评分，缺 **40场独立非验证轨迹**；ANN 缺 **60场完整轨迹**：
同样40场独立非验证场景，以及与SNN完全相同固定20场的冻结ANN策略评分轨迹。
固定20场沿用原 helper：7个空场+13个非空场（seed20261026选定），不重新选择评分集。
训练30场取独立 easy 随机流的前30个合格场景，不分层，不按 outcome/Q/G 挑选；
随后仅按几何是否为空，取首个空场和前9个非空场作为校准10场。
这是当前严格输入契约下的数量，不是宣称所有旧训练可视化永远无法恢复。
若以后另行证明旧数据完整性/策略身份，可重新审查复用方案；当前脚本不会凭累计回报
补造逐步证据。新增 collect 默认只打印计划，显式 --launch 才生成独立场景与自然采集；
失败不自动补飞，不用固定验证轨迹补训练缺口。
不按失败结果补采直到满意；失败路径不足时指标明确标 information=insufficient。

## 四臂与预注册参数

四臂同模型共享：source checkpoint 的 online 双 critic、target 双 critic、联合 Adam
moments/step/参数顺序，以及一次 prepare 生成的同一批次计划。学习率用 source 保存值。
一个 Adam（critic1 后 critic2），一次全局梯度裁剪；阈值使用 source 原值。
复用原校准 helper 的独立 eager Adam：fused=False、capturable=False，保留保存的
moments/step/lr/betas/eps/weight_decay及参数顺序；四臂均使用同一种执行后端。

令 `q1,q2` 为当前 target critics 在 next state 上的估计，`m=min(q1,q2)`：

- min：两支标签均为 `r+gamma*(1-done)*m`。
- mean：两支标签均为 `r+gamma*(1-done)*(q1+q2)/2`。
- WD3：两支标签均为 `r+gamma*(1-done)*[0.5*m+0.5*(q1+q2)/2]`。
- ACC：第 k 支标签为 `r+gamma*(1-done)*[beta*qk+(1-beta)*m]`。

WD3 的 min/mean 混合结构与 ACC 的 TD3 critic-specific 插值结构分别来自
[WD3 Eq.8/Algorithm1](https://arxiv.org/abs/2006.12622) 和
[ACC IV-F](https://arxiv.org/html/2111.12673#S4.SS6)。**ACC 不是简单把 WD3 权重调一下**；
两支的标签可以不同。WD3 min 权重固定0.5是本项目选择，不是论文默认值的断言。

ACC 控制规则（**以下标注的数值/归一化均是本项目选择，不冒充论文复现**）：

```text
beta 初值 = 0.5；范围 [0,1]
每100次 critic 更新后，仅在完整专用校准集上：
C = p_empty*mean_empty(G-Qbar) + (1-p_empty)*mean_nonempty(G-Qbar)
S = max(1, p_empty*mean_empty(abs(G)) + (1-p_empty)*mean_nonempty(abs(G)))
u = clip(C/S, -1, 1)
beta_new = clip(beta + 0.05*u, 0, 1)
```

上式每个 mean 先对场景内transition求均值，再对组内场景等权。
校准集1空+9非空，**不按1/10和9/10加权**，更不是原先2空2非空的隐含50%权重。
`p_empty` 从正式入口实际使用的 `V2ScenarioGeneratorConfig().zone_count_probabilities['easy'][0]`
读取；本轮只读核对为0.05，非空合计0.95。源checkpoint没有单独序列化该dataclass；
已核对正式入口以默认config构造generator、两份源训练场景记录均为generator v3，
生成器文件最近修改为2026-10-03的39cc535，早于两份源训练。
此为历史默认配置溯源，不能声称该概率从checkpoint字段直接读出。
完整生成配置、来源说明、实际概率、分层组成及规则进入collect/prepare/fit预注册；
实现不把0.05硬编码到控制公式。输入config变更拒绝。

S 固定于校准 G，不用评分集、训练集、跨臂 Q 或最佳结果归一化；无 EMA、无自适应步长。
正 C 提高 beta（降低 min 的影响），负 C 降低 beta。100次更新后改变的 beta 仅用于
下一次 critic 更新；不反传控制参数，不增加 critic optimizer 步数。
论文 ACC 的一般思想是用回报偏差调目标控制参数；其 TQC 实验含近期回报、EMA 和按环境步
触发的设置。本实验是固定数据的 ACC 式干预，不能冒充论文的在线训练复现或无偏定理验证。

四臂共同冻结 source online actor，用该 actor 给 next action，**目标平滑噪声为0**，
以便与相同无探索固定策略的 G 对照。source target actor 仍严格加载且冻结，但不参与目标
动作计算；不能用 online 权重冒充 source target 权重。
target critics 从 source target 权重开始，按保存 tau/policy_delay 软更新，计数相位接
source last_total_steps。actor/target actor/原 source critics 前后核对 hash 和梯度。
G 绝不进入 TD 标签或 loss；ACC 只读取 calibration 分区 G，其他 G 只作评分。

最多10000次更新（默认10000，可在 prepare 前固定更小预算，fit 不允许改预算）。
batch64，独立固定 seed20261101 均匀有放回训练索引；0/100/500/1000/2000/5000/10000
及预定最终步评估。无 early stop、无 best checkpoint、无按留出选步数/参数。
不进行不同 beta/gain 搜索；缺数据时拒绝，不产生对照通过结论。

## 数据输入契约与重放

`--independent-history DIR` 包含 `manifest.json` 与 `baseline_episodes.jsonl`。
manifest：

```json
{
  "format": "v2_frozen_actor_episodes_v2",
  "checkpoint_sha256": "完整 source hash",
  "actor_sha256": "cal.state_hash(source_actor.state_dict())",
  "exploration_noise": 0.0,
  "scenarios": [{"scenario_id":"独立ID","sequence_index":0,"partition":"train/calibration/score之一","payload":"完整原环境场景字典"}],
  "generation": "实际生成config、p_empty、独立seed20261101和选择规则",
  "backend": "device/torch/CUDA/TF32/threads及eager单观测actor callable",
  "code": "源码HEAD/status及src文件hash",
  "input_hashes": "checkpoint、固定池及复用历史manifest/episodes的hash"
}
```

JSONL 每行沿用 `diagnose_v2_critic` 完整自然 episode 的 record 字段：scenario_id、
sequence_index、empty_scene、actions、rewards、states(N+1个五维状态)、pre_action_hashes、
episode_length、episode_return、discounted_return、outcome、terminated、truncated、
final_steps、final_info。场景/奖励/半径通过 source 严格配置重建。

collect模式：先封存全部60个场景payload、物理指纹、用途和采集计划，再重放或飞行。
SNN 必须提供 --score-history；先严格核对20条B记录并复用，再采40场独立轨迹。
ANN 采同20场固定评分+40场独立轨迹。两模型独立流seed相同，几何选择不依赖actor。
不设置人为飞行budget：正常reset/step到环境自然terminated/truncated（原max_steps timeout有效）。
逐条写完整JSONL，新采记录立即用既有线性重放及冻结actor逐动作精确校验。
`evidence.json`封存manifest和JSONL文件hash；成功summary才标execution_completed=true。
已有输出目录先检查成功标记、hash、身份、60场划分，再严格重放验证；不写入、不重新采集。
不完整目录拒绝自动重试。历史B目录不被写入或重命名。

prepare模式：将collect输出作为 --independent-history，不再重复加 --score-history。
也支持具备同样封存契约的40场独立history+原20场 --score-history；不接受旧12/4/4划分。
先按用途划分并写 manifest，再逐场 reset、按保存动作**线性重放一次**，复用已有
严格 state/reward/hash/终态/回报核对；另外以原单观测 actor callable 精确核对无探索动作。
采集后端、原B后端和prepare重放后端分别记录。跨后端数值不一致也报错，
不通过放宽阈值假装相同策略；原B的CUDA动作未保证能在CPU逐位重现。
以物理 state/goal/zone几何与安全边距指纹防止改 ID 后泄漏同一场景；固定池内任意物理
场景或同名 ID 出现在 train/calibration 均拒绝。

prepare 保存只读数据和10000×64批次计划到 data.pt（可信本项目本地产物，非任意 pickle）；
evidence.json 固定 data/manifest/batch hash。fit 严格检查配置、划分、批次、checkpoint、
固定池以及 prepare 时全部 src Python 文件 hash；各臂独立目录，只读同一 prepared。
源码修改后须复审及重新 prepare，不只根据 HEAD 断言相同实现。

## 产物与解释边界

- manifest：输入 hash、源码 HEAD/status/实际文件hash、配置、划分、初始 actor/online/
  targets/Adam hash；在拟合前落盘。输入/源码变动阻断成功输出。
- fit_curves：双 critic 有符号 Q-G、MAE、RMSE、各自 Q-y、TD误差，train/calibration/score、
  空/非空、terminal/nonterminal、failure_path/failure_terminal及逐场结果；场景等权与
  transition 加权分别记录，空组信息不足。重叠分组不能相加。
- controls：ACC 每次 beta 前后值、C、S、p_empty、归一化信号；完整保留，不下采样。
  summary记录first_boundary_update，未触边界为null；beta到1不代表评分已校准。
- gradient：一次联合 TD batch 的 clip 前全局范数及累计裁剪比例，不能归因给评分子组。
- timing：含初始化复制、最终状态hash、最终 CUDA 同步的总墙钟，以及初始化/评分/控制墙钟。包含检查、排队与同步等待，子项
  不完整覆盖总时间，不是纯 GPU 时间或正式训练吞吐。
- summary：execution_completed 是执行完整性，始终 diagnostic_only、formal_passed=false。
  无通过门槛，无模型权重输出，不生成正式 passed 文件或课程前驱。

固定 actor/数据实验不能代表在线训练能提高 goal 率；G校准改进也不能证明训练问题已解决。
评分20场是7空/13非空的固定富集子集；all汇总仅描述该子集，不是整个easy分布的总体结果。
空/非空分别报告，ACC的源概率加权仅用于控制信号，不把评分子集伪装成总体抽样。
beta 饱和可能说明 min 与各自 Q 的插值范围不足，不能保证消除两支共同偏差。

## 后续命令结构（仅准备，本轮未执行）

每模型先collect补齐并核对数据，再prepare一次；四臂各用独立进程/目录，共用prepared。
以下默认**只打印计划**。
ANN 将 model/checkpoint换为上表 ANN，collect去掉 score-history（新增60条）。
以下collect/prepare/fit都不带 --launch，不执行场景生成、飞行或拟合。

```bash
cd /home/zj/projects/my_project && \
PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B \
  -m brain_uav.scripts.diagnose_v2_wd3_acc --mode collect --model snn \
  --checkpoint outputs/v2_snn_critic_lr_confirm_round1/20261024_102749/snn_clr_1e-3_seed23/v2_snn_td3_easy_failed.pt \
  --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
  --score-history outputs/v2_critic_reliability_round1/20261026_171846_464722553 \
  --device cuda --output-dir /ABSOLUTE/NEW_RUN/other/snn_collected_60

cd /home/zj/projects/my_project && \
PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B \
  -m brain_uav.scripts.diagnose_v2_wd3_acc --mode prepare --model snn \
  --checkpoint outputs/v2_snn_critic_lr_confirm_round1/20261024_102749/snn_clr_1e-3_seed23/v2_snn_td3_easy_failed.pt \
  --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
  --independent-history /ABSOLUTE/NEW_RUN/other/snn_collected_60 \
  --updates 10000 --device cuda --output-dir /ABSOLUTE/NEW_RUN/other/prepared

cd /home/zj/projects/my_project && \
PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B \
  -m brain_uav.scripts.diagnose_v2_wd3_acc --mode fit --model snn --arm acc \
  --checkpoint outputs/v2_snn_critic_lr_confirm_round1/20261024_102749/snn_clr_1e-3_seed23/v2_snn_td3_easy_failed.pt \
  --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
  --prepared-dir /ABSOLUTE/NEW_RUN/other/prepared --updates 10000 --device cuda \
  --output-dir /ABSOLUTE/NEW_RUN/other/acc
```

仅在补齐数据、代码复审及另行授权后添加 `--launch`。
届时外层使用独立时间戳目录、python -u/tee，管道后第一句 `rc=${PIPESTATUS[0]}`，
记录退出码与完整耗时；review 放小 manifest/summary/log，other 放 data.pt、曲线明细。
目前缺失的非验证数据路径是明确占位，不是可直接运行或已核实存在的服务器产物。
