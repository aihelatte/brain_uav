# ANN/SNN 课程训练复现异常问题分析报告

## 1. 分析范围

本报告对比以下训练结果：

- 旧 ANN：`0504_111516_ann_candidates_safety_reward_750k_1000k_ann`
- 旧 SNN：`0504_111525_snn_candidates_safety_reward_750k_1000k_snn`
- 新 ANN：`0827_215958_ann_reproduce_original_v6`
- 新 SNN：`0827_220003_snn_reproduce_original_v6`

目标是解释：新 SNN 为何在 `easy_two_zone` 四候选全部无法早停，以及新 ANN 为何能进入 `hard`、但早停显著变慢。

启动命令中的模型、基础 seed 7、4 candidates、4 workers、CUDA、最大课程 hard 均无明显错误；SNN 也正确使用 torch backend。

## 2. 执行摘要

当前最明确的问题不在 `easy_two_zone` 早停回调，而在课程链更上游的 BC 初始化和 easy 输出 checkpoint。

四次实验的 BC 专家数据逐字节一致，但 BC 训练没有设置 seed。actor 随机初始化和 `DataLoader(shuffle=True)` 的 batch 顺序不受主命令 `--seed 7` 控制。因此，新旧实验从 BC 阶段开始就不是同一条确定性轨迹。

SNN 四个 `easy_two_zone` 候选共享同一个新 easy checkpoint，全部出现相近的低成功率、大量 collision/ground，四个 TD3 seed 均无法恢复，强烈指向共享的上游 checkpoint，而非单个 candidate 偶然失败。

ANN 也同方向退化，只是训练后期形成连续四个合格窗口。它并非正常复现：`easy_two_zone` 早停由约 130k 延迟到约 622k steps，`medium` 由约 337k 延迟到约 646k steps。

| 判断 | 置信度 | 依据 |
|---|---|---|
| BC 未被 seed 控制 | 已确认 | 数据相同，BC loss/权重不同，代码无 BC seed |
| SNN 未早停是性能未达标 | 已确认 | 四候选均无连续四个合格窗口 |
| 新 easy checkpoint 迁移质量较差 | 高概率 | 四候选进入 easy_two_zone 后同时弱于旧实验 |
| first-finisher winner 放大随机性 | 已确认 | 新旧 winner 链发生改变 |
| CUDA/依赖版本进一步影响结果 | 待验证 | 环境未锁定，日志缺版本 |
| CUDA Graph 推理优化破坏训练 | 当前不支持 | 相关改动主要位于 benchmark/inference |

## 3. 数据集与 BC 证据

四个 `bc_dataset_easy_v6.npz` 的 SHA-256 完全相同：

```text
07A34266C1825AA64E637737E9AD55A9626BA7129E08AD0493D9065E17056056
```

这排除了专家轨迹变化、dataset seed 失效、数据版本不同和文件传输错误。

相同数据产生了不同 BC：

| 模型 | 实验 | best loss | best epoch | final loss |
|---|---|---:|---:|---:|
| ANN | 旧 | 2.4196e-7 | 20 | 2.4196e-7 |
| ANN | 新 | 2.9017e-7 | 20 | 2.9017e-7 |
| SNN | 旧 | 3.6161e-6 | 20 | 3.6161e-6 |
| SNN | 新 | 3.8366e-6 | 19 | 3.8861e-6 |

BC MSE 不能单独证明迁移能力，但 loss 轨迹和 state dict 张量哈希均不同：

| 检查点 | 旧哈希前缀 | 新哈希前缀 |
|---|---|---|
| `bc_ann_best.pt` | `b03d332e...` | `e84374d5...` |
| `bc_snn_best.pt` | `ebb28b3d...` | `5543da1b...` |
| `td3_ann_easy.pt` | `e5dd8281...` | `aea52769...` |
| `td3_snn_easy.pt` | `27e4f180...` | `79bdee4a...` |

代码证据：

- `src/brain_uav/scripts/train_bc.py:32` 没有 `--seed`；
- `train_bc.py` 没有调用 `set_global_seed()`；
- `src/brain_uav/trainers/bc.py:33` 使用 `DataLoader(..., shuffle=True)`，无显式 generator；
- pipeline 给 dataset 和 TD3 candidate 传 seed，但 BC 无 seed 可传。

实际 seed 控制范围：

```text
dataset generation：受控
BC actor 初始化：不受控
BC batch shuffle：不受控
TD3 candidate：名义上受控
CUDA 算子确定性：未完全受控
```

这是当前最明确、最靠近课程链源头的可复现性缺陷。

## 4. easy 阶段与迁移

| 模型 | 实验 | winner seed | steps | episodes | 总成功率 | 后 25% |
|---|---|---:|---:|---:|---:|---:|
| ANN | 旧 | 307 | 126,735 | 240 | 78.75% | 98.33% |
| ANN | 新 | 7 | 125,091 | 255 | 85.10% | 100.00% |
| SNN | 旧 | 307 | 126,689 | 255 | 75.29% | 96.67% |
| SNN | 新 | 307 | 128,164 | 270 | 64.07% | 96.67% |

新 SNN easy 能早停，说明 easy 回调正常。但其总成功率比旧 SNN 低 11.22 个百分点，且 ground 更多。“最近窗口通过 easy”不等于“具备相同跨课程迁移能力”。

日志中的课程接力正确：

- easy 从 `bc_*_best.pt` 初始化，`reference_source=bc`；
- easy_two_zone 从 `td3_*_easy.pt` 初始化，`reference_source=previous_stage`；
- ANN medium 从 `td3_ann_easy_two_zone.pt` 初始化；
- `terminal_guidance_disabled=False`；
- `bc_regularization_enabled=True`；
- SNN 使用 CUDA + torch。

没有证据表明 checkpoint 漏载、BC reference 丢失或 terminal guidance 被关闭。

## 5. SNN easy_two_zone 失败

四个候选均训练到 750k：

| seed | episodes | 总成功率 | 前 25% | 后 25% | 最长合格连胜 | 末四窗口 goal |
|---:|---:|---:|---:|---:|---:|---|
| 7 | 1,783 | 59.45% | 30.11% | 77.60% | 2 | 11,12,10,11 |
| 107 | 1,797 | 54.92% | 26.44% | 78.52% | 1 | 14,13,14,10 |
| 207 | 1,754 | 58.15% | 24.60% | 80.18% | 2 | 12,13,13,11 |
| 307 | 1,856 | 52.64% | 22.37% | 78.52% | 3 | 12,14,14,11 |

旧 SNN 对照：

```text
127,474 steps；255 episodes；总成功率 85.49%
前 25% 68.33%；后 25% 98.33%
最长合格连胜 5；末四窗口 15,14,15,15
```

新候选前 25% 只有 22.37%-30.11%，旧实验为 68.33%。四个候选同时下降，说明共享 easy checkpoint 的初始迁移质量明显变差。

失败构成：

| seed | goal | collision | ground | timeout | boundary |
|---:|---:|---:|---:|---:|---:|
| 7 | 1,060 | 456 | 232 | 35 | 0 |
| 107 | 987 | 506 | 257 | 41 | 6 |
| 207 | 1,020 | 457 | 229 | 48 | 0 |
| 307 | 977 | 611 | 225 | 42 | 1 |

主要问题是 collision 和 ground，不是 timeout。因此问题集中在双禁飞区几何绕障和高度安全，而非单纯终端捕获。

部分候选最后 20 个窗口约 78%-79%，说明训练有进步，但平台仍低于每窗口至少 14/15 的早停要求。

## 6. ANN 同方向退化

| 阶段 | 旧 steps | 新 steps | 旧总成功率 | 新总成功率 | 新主要失败 |
|---|---:|---:|---:|---:|---|
| easy_two_zone | 130,159 | 622,192 | 89.05% | 55.58% | collision 670 |
| medium | 336,831 | 645,822 | 62.16% | 51.25% | collision 485、ground 188 |

新 ANN 能进入 hard，只说明后期出现连续合格窗口，不代表复现旧训练效率和稳定性。

easy_two_zone 最后 20 窗口 goal：

```text
4,10,9,9,11,9,9,9,10,10,9,11,11,13,13,13,14,14,15,14
```

medium 最后 20 窗口 goal：

```text
0,0,1,2,6,5,9,10,13,13,14,15,14,12,11,13,15,15,14,14
```

它们确实在末段改善后跨线，不是回调误判，但样本效率显著降低。当前 ANN 目录尚无 hard 完整 metrics，暂不能判断 hard 的失败构成。

## 7. 早停审核

参数：

```text
summary_every_episodes = 15
early_stop_goal_rate = 0.95
early_stop_windows = 4
early_stop_max_failures_per_window = 1
early_stop_min_steps = 125000
```

`train_td3.py:337-389` 的逻辑：

1. 每 15 episode 一个窗口；
2. 失败不超过 1，或 goal rate 达 0.95，窗口合格；
3. 实际最低线为 14/15，即 93.33%；
4. 连续 4 个合格窗口且 steps 不低于 125k 才早停；
5. 任一窗口低于 14/15，连续计数归零。

没有发现实现错误：新 ANN 可触发；新 SNN 最长连胜只有 1-3；`stopped_early`、`stop_reason` 和窗口统计一致；不存在连续四窗口合格但未早停的反例。

早停很严格，但不应先放宽阈值掩盖安全性能退化。

## 8. winner 机制的放大作用

`run_full_pipeline_candidates.py:381-386` 选择第一个完成且 `stopped_early=True` 的候选，并终止其他候选。

```text
旧 ANN：easy 307 -> easy_two_zone 207 -> medium 207 -> hard 207
新 ANN：easy 7   -> easy_two_zone 307 -> medium 307 -> hard 运行中

旧 SNN：easy 307 -> easy_two_zone 207 -> medium 107 -> hard 107
新 SNN：easy 307 -> easy_two_zone 四候选失败
```

该机制依赖训练 seed 和并行完成顺序。共享 GPU 时，完成顺序还可能受调度影响。它选的是“最先早停者”，未必是固定验证集上泛化最好或最适合下一课程的模型。

SNN 新旧 easy winner 虽都是 seed 307，但 BC 起点和最终权重哈希不同，说明 winner seed 不能唯一标识模型。

## 9. 参数与近期代码核对

新旧 easy_two_zone 的 actor freeze、梯度裁剪、噪声衰减、success bias、success batch fraction、actor loss scale、terminal regularization、goal radius curriculum、curriculum mix、早停参数、device 和 backend 一致。

当前 pipeline 新增消融、续跑、失败回退和日志选项，但本次命令未启用 `--skip-bc`、`--hard-only`、`--td3-curriculum-mix`、`--disable-terminal-guidance` 或 `--continue-on-stage-failure`，不应改变默认主实验。

action-only forward、`inference_mode`、SNN reset、CUDA Graph 主要用于 benchmark 推理。目前无证据表明它们进入 TD3 更新。

## 10. 次级风险

### 10.1 optimizer state 可能覆盖阶段 LR

恢复上一 checkpoint 的 optimizer state 时，param group 学习率也会恢复，可能覆盖当前阶段 config LR。该行为新旧均存在，不能独立解释本次差异，但可能放大 checkpoint 敏感性。应记录加载前后的 effective actor/critic LR，并补测试。

### 10.2 seed 与环境元数据不足

selection summary 记录候选 seed，但单个 metrics 可能仍显示 config 默认 seed 7。建议保存：

```text
seed、base_seed、candidate_id、parent_checkpoint_hash、git_commit
torch_version、cuda_version、device_name、deterministic_flags
```

依赖未完整锁定，日志未保存 Python/PyTorch/CUDA/驱动版本。CUDA 非确定性、浮点累计顺序和四进程 GPU 竞争目前只能列为次级假设。

### 10.3 当前未发现的主线错误

没有证据显示以下机制被误改：

- success replay / primary success；
- success clearance 重新过滤；
- SNN 学习率被推理优化改变；
- actor loss 缩放或 BC schedule；
- terminal geometric regularization；
- easy_two_zone curriculum mix；
- checkpoint 接力路径；
- early-stop metrics 保存。

## 11. 最可能因果链

```text
--seed 7
 -> dataset 受控且完全一致
 -> BC 没有 seed
 -> actor 初始化和 DataLoader shuffle 不同
 -> 新旧 BC 权重不同
 -> easy 从不同 BC 起点训练
 -> 第一早停者成为唯一父 checkpoint
 -> easy 能通过，但迁移能力未验证
 -> easy_two_zone 四候选共享较弱父模型
 -> collision/ground 大量出现
 -> SNN 停留在 53%-59%，无法连续达到 14/15
 -> ANN 需约 622k/646k steps 才跨线
```

## 12. 建议验证顺序

### A. 旧、新 easy checkpoint A/B

固定当前代码、环境和 seeds 7/107/207/307，只改变 SNN easy_two_zone 父 checkpoint：

```text
A 组：旧 td3_snn_easy.pt
B 组：新 td3_snn_easy.pt
```

比较前/后 25% 成功率、collision、ground、timeout、最长合格连胜和早停 steps。若 A 恢复而 B 失败，可基本确认上游 checkpoint 是主因。

### B. BC 重复性

补齐 seed 后，同一 dataset、同一 seed 连续训练两次 BC，比较 loss history、best epoch、state dict hash 和固定观测 action。

### C. TD3 单候选重复性

暂时关闭四进程并行。固定父 checkpoint 和 seed 串行运行两次，比较 outcome 序列、窗口统计、阈值 steps 和最终 actor，以区分 BC 与 TD3/CUDA 不确定性。

### D. 实际阶段 LR

记录 checkpoint 加载前后的 configured、loaded、effective actor/critic LR。

### E. 环境信息

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
pip freeze
git rev-parse HEAD
```

## 13. 修复优先级

### P0：BC seed 闭环

- `train_bc.py` 增加 `--seed`；
- 调用 `set_global_seed(args.seed)`；
- pipeline 将基础 seed 传给 BC；
- DataLoader 使用显式 `torch.Generator`；
- checkpoint 和 metrics 保存实际 seed。

### P0：冻结原始 checkpoint

不覆盖旧 BC、easy 和最终模型。记录父 checkpoint 张量哈希，而不只记录文件名和 seed。

### P1：改进 winner 选择

候选达到最低条件后，在固定、独立、无探索噪声的 stage validation 集上统一评估，再依据成功率、安全失败和效率选择 winner。

### P1：完善元数据

保存命令、git commit、实际 seed、候选 ID、父 checkpoint 哈希、依赖版本和确定性配置。

### P2：确认 optimizer LR 语义

明确课程续训是继承上一阶段 optimizer LR，还是加载权重后重设当前阶段 LR，并补测试。

### P2：最后再评估早停

先修复可复现性，再讨论 validation early stop。当前不建议直接减少窗口数或放宽失败数。

## 14. 能说明与不能说明

能说明：

- 专家数据完全一致；
- 新旧 BC/easy 权重不同；
- BC seed 未被主命令控制；
- 新 SNN easy_two_zone 是真实性能不足；
- 新 ANN 同样降低样本效率和稳定性；
- checkpoint 接力与 terminal guidance 没有明显误用；
- first-finisher 会放大上游差异。

不能说明：

- 不能用一次失败断言 SNN 天生比 ANN 更难训练；
- 不能以本次复现失败否定原 benchmark；
- 不能仅凭 BC MSE 判断迁移能力；
- 不能确认 CUDA/PyTorch 版本是主因；
- hard metrics 缺失，不能分析新 ANN hard；
- 不能通过放宽早停证明训练恢复。

## 15. 给审核 AI 的重点问题

1. BC 初始化和 DataLoader shuffle 是否确实未被 seed 控制？
2. first-finisher 是否使 winner 依赖进程调度并传播差异？
3. optimizer state 是否覆盖当前阶段配置 LR？
4. metrics 是否记录默认 seed 而非实际 candidate seed？
5. 当前代码相较 5 月是否存在默认启用并改变训练行为的改动？
6. 如何设计不改变奖励和物理口径的固定 stage validation？
7. 补齐 BC seed 后，CUDA 还需要哪些确定性设置，速度代价如何？

## 16. 最终结论

本次现象不能简化为“随机 seed 不好”，也没有证据支持“早停代码坏了”。更准确的描述是：流水线只对 dataset 和 TD3 候选传递 seed，却遗漏课程链起点 BC；随后 first-finisher candidate selection 将上游随机差异固化为唯一父 checkpoint，并传播到后续阶段。

新 SNN easy checkpoint 能通过 easy 最近窗口，却未复现旧模型的双禁飞区迁移能力。四个 easy_two_zone seed 均出现大量 collision/ground，并在 750k steps 后停留在约 53%-59%，无法满足连续四个 14/15。新 ANN 也同方向退化，只是经过约 622k 和 646k steps 后分别在 easy_two_zone 和 medium 跨线。

首要工作应是旧、新 easy checkpoint 直接 A/B，随后补齐 BC seed、父 checkpoint 哈希和环境元数据。建立可复现闭环后，才适合判断是否需要调整训练超参数、SNN 学习率或早停标准。

