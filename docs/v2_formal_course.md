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

## 从 `v2_course_e6bc9be` 续跑

续跑入口使用 `--resume-from`。旧目录固定为 `/home/zj/diagnostics/outputs/v2_course_e6bc9be`，新结果目录示例为 `/home/zj/diagnostics/outputs/v2_course_e6bc9be_resume_<时间戳>`。`144e518` 的原续跑版本由 `scripts/v2_formal_course_resume_source.json` 封存。本 checkout 新训练使用 `scripts/v2_formal_course_snapshot_resume_source.json`。历史 `scripts/v2_formal_course_source.json` 保留用于核对原课程，原 checkout 可沿用它。

启动前按 `e6bc9be` 的 `course.json`、原门控和训练清单、逐 seed 终态及回合记录复核两批 easy，重新核对晋级排名以及 `winner.pt` 和对应候选的 SHA256。`resume_source.json` 保存旧课程、旧池清单、原门控及训练清单、原选优记录和两个胜者的来源。新目录按原字节复制胜者和六份场景池。场景池核对字节哈希、内容摘要、种子与场景契约，沿用晋级和独立场景互斥检查。续跑读旧目录中的封存产物，并使用原 BC 和 `stage_start.json` 作为核验输入。

新目录的两批 `selection.json` 保留 8 个历史 easy 候选，最终 `training.csv` 仍保留这 8 行。`medium-ann` 首先运行新的正式 CUDA 门控，四个 seed 从导入的 ANN 胜者重新起跑。`medium-snn` 使用导入的 SNN 胜者。后续仍按原规则晋级 hard，两个模型均晋级后做独立评估。

把 `REVIEWED_COMMIT` 填成续跑改动经审查后已推送的提交。第一段拉取到新 checkout，第二段输出计划并执行。`--resume-from` 的池来源是旧课程的封存目录。

```bash
REVIEWED_COMMIT='填入审查后已推送的续跑提交'
stamp=$(date +%Y%m%d_%H%M%S_%N)
old=/home/zj/diagnostics/outputs/v2_course_e6bc9be
root="/home/zj/diagnostics/outputs/v2_course_e6bc9be_resume_$stamp"
checkout="/home/zj/diagnostics/my_project_formal_course_resume_$stamp"
git clone --single-branch --branch multi-nfz-training https://github.com/aihelatte/brain_uav.git "$checkout" &&
  cd "$checkout" && git checkout --detach "$REVIEWED_COMMIT"
```

```bash
runner=(/home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.run_v2_formal_course)
resume_course() {
  local start rc
  start=$(date +%s)
  printf 'Total result directory: %s\n' "$root"
  if PYTHONPATH=src "${runner[@]}" --root "$root" --resume-from "$old" --run 2>&1 | tee -a "$root.driver.log"; then
    rc=${PIPESTATUS[0]}
  else
    rc=${PIPESTATUS[0]}
  fi
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root.driver.log"
  return "$rc"
}
cd "$checkout" &&
  PYTHONPATH=src "${runner[@]}" --root "$root" --resume-from "$old" &&
  resume_course
```

```bash
# 查看进度
cd "$checkout" && PYTHONPATH=src "${runner[@]}" --root "$root" --status
tail -n 60 "$root.driver.log"

# 中断后继续。新终端先恢复上次的 checkout、root、old、runner 和函数定义。
cd "$checkout" && resume_course
```

续跑每次复核来源及新目录内的副本。已完成且核验通过的新批次沿用原收集与恢复流程。旧目录里若有失败的 `medium-ann`，它的日志留在旧目录，本次尝试从新目录的 `attempt-000001` 开始。

## 从 `144e518` 的 medium ANN 完成处继续

将 `prior` 设为实际使用 `144e518` 的续跑结果目录，即包含 `batches/medium-ann/selection.json` 的课程根目录。入口从该目录的 `course.json` 找到原 `e6bc9be` 课程，两个旧目录都只读。它核对三批共 12 个候选、旧门控及输入身份、场景池与选优记录。`medium-ann` 如有胜者，会复制到新目录供 hard ANN 使用。原结果若为 `no_formal_pass`，ANN 保持止步于 medium，SNN 继续。首个新批次为 `medium-snn`，使用 SNN easy 胜者，重新执行 CUDA 门控及 seed 7、11、23、41。原 `medium-snn` 如果有遗留进程，须先由既有只读检查确认已退出。

新 `course.json` 和新门控使用 `scripts/v2_formal_course_snapshot_resume_source.json`。`resume_source.json` 同时记录两个旧阶段来源，包括 `medium-ann` 的原 `selection.json` 摘要与晋级状态。周期快照文件名仍用计划步数，内容和周期评估里的 `scheduled_stage_steps` 用于对齐索引，`stage_steps` 记录实际采集或评估步数，评估的 `global_steps` 同样对应实际步数。旧版可读快照仍可使用当前加载器。

审查并推送后，把 `REVIEWED_COMMIT` 和 `prior` 填成实际值。中断后在同一个 checkout 和 `root` 上重跑 `resume_course`。

```bash
REVIEWED_COMMIT='填入审查后已推送的提交'
prior='/home/zj/diagnostics/outputs/填入144e518续跑结果目录名'
stamp=$(date +%Y%m%d_%H%M%S_%N)
root="/home/zj/diagnostics/outputs/v2_course_snapshot_resume_$stamp"
checkout="/home/zj/diagnostics/my_project_formal_course_snapshot_resume_$stamp"
runner=(/home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.run_v2_formal_course)
resume_args=(--root "$root" --resume-from "$prior" --resume-after-medium-ann)
resume_course() {
  local start rc
  start=$(date +%s)
  printf 'Total result directory: %s\n' "$root"
  if PYTHONPATH=src "${runner[@]}" "${resume_args[@]}" --run 2>&1 | tee -a "$root.driver.log"; then
    rc=${PIPESTATUS[0]}
  else
    rc=${PIPESTATUS[0]}
  fi
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root.driver.log"
  return "$rc"
}
test -s "$prior/batches/medium-ann/selection.json" &&
  git clone --single-branch --branch multi-nfz-training https://github.com/aihelatte/brain_uav.git "$checkout" &&
  cd "$checkout" && git checkout --detach "$REVIEWED_COMMIT" &&
  PYTHONPATH=src "${runner[@]}" "${resume_args[@]}" &&
  resume_course
```

查看进度用 `PYTHONPATH=src "${runner[@]}" --root "$root" --status`。

## 结果目录

- `course.json` 保存 24 组实际解析配置和 BC 哈希。 `pools.json` 保存六份池的身份。
- `batches/<stage>-<model>/attempt-*/{gate,train}/` 保存门控、日志、逐场验证、进程与 MPS 记录、训练回合和候选权重。 `selection.json` 保存四个候选、末 50k 分子分母及晋级结果。 `winner.pt` 是选中候选的字节副本，下一阶段记录其哈希路径。
- `training.csv` 包含 24 行计划对应结果。只有正式通过的候选才能晋级，依次比较终态到达数、碰撞数、末 50k 到达率、较小 seed。末 50k 按回合结束步统计，区间为 `终态步数 - 50000 < 回合结束步数 <= 终态步数`。因无人通过而未启动的后续项标为 `blocked_by_no_formal_pass`。
- `independent.csv` 在两个 hard 胜者完成后生成，包含各模型按难度统计的到达、碰撞、冲顶、其他越界、触地、超时。 `independent/<model>-<stage>.json` 保存逐场结果，可在同一难度按 `scenario_id` 对齐对比。对应的 `<model>-<stage>.sha256.json` 绑定结果字节、冻结权重和场景池。
- `course_summary.json` 保存 hard 胜者、模型止步阶段及独立评估场次数。
