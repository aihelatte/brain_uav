# UAV V2 正式四 seed 全课程基准

实验标识为 `safe-direct-full-course-v1`。ANN 和 SNN 各跑 easy、medium、hard，每阶段均为 seed 7、11、23、41，一共六批、24 个候选，每批最多四个 worker。两个模型都从原 BC best 重训 easy。模型内每次晋级只选一份父权重，下阶段四个 seed 都从这份 actor 起跑。critic、优化器、回放及阶段日程重新初始化。SNN 的 ACC 也在每阶段从 β=.5 和空样本库开始，用该阶段 critic 隔离评分。

| 阶段 | 预算 | ANN actor / critic LR | SNN actor / critic LR | 训练场景混合 | BC 日程 | 失败采样偏置 |
|---|---:|---|---|---|---|---:|
| easy | 500,000 | 1.55e-4 / 2e-4 | 8.25e-4 / 1e-3 | easy 100% | smooth-mid-strong | 2 |
| medium | 750,000 | 1.55e-4 / 2.5e-4 | 9.5e-4 / 2.5e-4 | medium 80%、easy 20% | default | 1 |
| hard | 1,000,000 | 1.125e-4 / 2.125e-4 | 1.125e-4 / 2.125e-4 | hard 70%、medium 20%、easy 10% | default | 1 |

easy 从 `safe-direct-20261119` 解析。medium 的模型专用 actor LR 由 `train_v2_td3.py` 解析。medium 的 critic LR、hard 的两个 LR、后两级预算及场景混合取 `v2_curriculum.py` 默认值。medium、hard 用正式课程 CLI 的默认 BC 日程和 150k 观测快照。BC 权重按阶段步数重置，easy 在 0/75k/150k/200k/250k/300k/335k/370k 的权重为 500/150/75/60/30/20/14/10。medium、hard 在 0/75k/150k/250k/300k 为 500/150/30/15/5。

三个阶段均为 ANN min、SNN ACC η=.01，γ=.99、τ=.005、policy delay 2、warmup 1280 步、actor 冻结 25k 步。探索噪声 .020→.005，目标噪声 .015→.006，clip .030→.012，均在各阶段预算的前一半衰减。正式候选至少训练 125k 步，连续四个 15 完整回合窗口各最多一次失败，并在对应固定 100 场中最多失败五场。保留 `default_v2_cuda_graph_compilation()`、GPUCompute、CPU 去重、批次非阻塞和椭球净空缓存。上界方向性预警 120 米，水平 60 米。直飞几何检查使用 40 米净空及参考动作下一步安全检查，近段至 400 米、系数 3000，远段系数 1500，两段分别求均值。评估由模型独立选动作。

## 拉取

把 `REVIEWED_COMMIT` 填成主聊天审查后已推送的提交。每次新实验用新的代码目录和时间戳结果目录。旧研究目录只读提供原 BC、历史配置证据及原 easy100。

```bash
REVIEWED_COMMIT='填入主聊天审查后已推送的提交'
stamp=$(date +%Y%m%d_%H%M%S_%N)
checkout="/home/zj/diagnostics/my_project_formal_course_$stamp"
root="/home/zj/diagnostics/outputs/v2_course_$stamp"
git clone --single-branch --branch multi-nfz-training https://github.com/aihelatte/brain_uav.git "$checkout" &&
  cd "$checkout" &&
  git checkout --detach "$REVIEWED_COMMIT"
```

## 执行

第一条 Python 命令只打印计划。第二条自动封存场景，再逐批完成 CUDA 门控、正式训练和晋级。两个 hard 胜者齐全后，自动评估 600 场并写出结果表。程序启动时会打印总结果目录。

```bash
runner=(/home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.run_v2_formal_course)
PYTHONPATH=src "${runner[@]}" --root "$root"

run_course() {
  local start rc
  start=$(date +%s)
  printf 'Total result directory: %s\n' "$root"
  if PYTHONPATH=src "${runner[@]}" --root "$root" --run 2>&1 | tee -a "$root.driver.log"; then
    rc=${PIPESTATUS[0]}
  else
    rc=${PIPESTATUS[0]}
  fi
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root.driver.log"
  return "$rc"
}
mkdir -p "$(dirname "$root")" && run_course
```

原晋级池目录默认是旧研究目录下的 `outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904`。可用 `--validation-pool-dir` 指向其副本。easy100 必须匹配历史字节哈希和内容摘要。medium、hard 优先复用同目录中通过校验的原池，缺失时由现有生成器用正式验证种子 20260904 生成。独立池使用 20261121。六份池在第一次 CUDA 门控前全部保存到结果目录，记录原字节哈希、内容摘要和种子，并按初始状态、目标和障碍几何核对重复。

## 查看进度

```bash
PYTHONPATH=src "${runner[@]}" --root "$root" --status
tail -n 60 "$root.driver.log"
find "$root/batches" -name '*.log' -print
```

`missing` 表示还没有该批尝试，`process_failed` 表示已记录门控或训练进程异常，`uncollected` 表示尝试尚未形成可收集结果。`no_formal_pass` 表示四个候选均完成，但无人满足正式门槛；该模型在该阶段停止，另一模型可继续。

## 失败后继续

使用原 checkout、原结果目录和同一批输入，重新执行 `run_course`。已经完成且身份校验通过的训练批次会复用。失败后，新尝试启动前会检查旧尝试的 gate 和 train。清理记录完整时可以直接重试。清理未完成或缺记录时，程序只读核对本机 PID、会话和本批私有 MPS 管道。相关进程已退出才能重试。若进程仍活着或归属无法确认，程序停止并打印批次、PID、管道和日志路径，供人工核对。每次重试使用新 `attempt-*`，旧日志和失败产物保留。每个总结果目录一次只允许一个执行进程。

独立评估开始前，`independent_inputs.json` 封存训练表及两个 hard 选优结果。继续时从冻结结果直接进入独立评估。评估文件和旁边的 `.sha256.json` 摘要齐全且校验通过时直接复用。只有评估文件、缺摘要时，会重新评估对应模型和难度的 100 场。摘要存在但文件、身份或计数不符时，程序报出路径并保留现场。原结果目录仍受原源码清单约束。

```bash
cd "$checkout" && run_course
```

## 结果目录

- `course.json` 保存 24 组实际解析配置和 BC 哈希。 `pools.json` 保存六份池的身份。
- `batches/<stage>-<model>/attempt-*/{gate,train}/` 保存门控、日志、逐场验证、进程与 MPS 记录、训练回合和候选权重。 `selection.json` 保存四个候选、末 50k 分子分母及晋级结果。 `winner.pt` 是选中候选的字节副本，下一阶段记录其哈希路径。
- `training.csv` 包含 24 行计划对应结果。只有正式通过的候选才能晋级，依次比较终态到达数、碰撞数、末 50k 到达率、较小 seed。末 50k 按回合结束步统计，区间为 `终态步数 - 50000 < 回合结束步数 <= 终态步数`。因无人通过而未启动的后续项标为 `blocked_by_no_formal_pass`。
- `independent.csv` 在两个 hard 胜者完成后生成，包含各模型按难度统计的到达、碰撞、冲顶、其他越界、触地、超时。 `independent/<model>-<stage>.json` 保存逐场结果，可在同一难度按 `scenario_id` 对齐对比。对应的 `<model>-<stage>.sha256.json` 绑定结果字节、冻结权重和场景池。
- `course_summary.json` 保存 hard 胜者、模型止步阶段及独立评估场次数。
