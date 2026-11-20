# UAV V2 正式 easy 上界预警与安全直飞

| 批次 | ANN | SNN |
|---|---|---|
| main | seed 7、23，min，actor 1.55e-4，critic 2e-4 | seed 7、23，ACC η=.01、初始 β=.5，actor 8.25e-4，critic 1e-3 |
| seed11 | seed 11，ANN 参数同上 | seed 11，SNN 参数同上 |
| 单独对照 | seed 11，原 ANN 正式配置 | seed 11，原 SNN 正式配置 |

两批新方案均从相应 BC best 开始，使用原 easy100、500k、探索、BC 日程、正式验证与早停。配置锁先逐字段校验原方案，seed11 从同模型 seed7 锁替换 seed 后检查。新方案只覆盖上界预警距离、引导分界和远段权重。seed11 对照沿用旧 250 米引导和原方向性预警。两条实验标识共用一份当前源码清单，每批门控独立绑定实验标识、批次、输入、作业、seed、PID 和报告哈希。在当前源码上精确重放旧批次应使用各自历史提交及清单。

上界方向性扣分以 120 米为起点。水平面、原软惩罚、上限 600 米、触地和越界终止保持原设置。直飞辅助项在策略采样之前使用 40 米的直线路径净空，并用现有动力学预判限幅参考动作的下一条飞行线段。正常和成功回放均保存动作前分段。近段包含 400 米，远段没有外半径。两段所用动作均方误差各自求平均，系数分别是 3000、1500。固定评估和诊断由模型独立选动作。

## 审查并部署后执行

以下为服务器 Bash 命令。先将 **主聊天审查且已推送的提交**填入 REVIEWED_COMMIT。工作区只读借用原研究仓库的前三项输入路径和两份历史 stage_start。旧训练目录继续保留。若服务器上的历史批次已迁移，BASELINE 指向其完整原样副本。汇总核对 manifest、终态评价、回合和进程等文件哈希，并按已封存的权重哈希验收 checkpoint。

```bash
REVIEWED_COMMIT='填入主聊天审查后已推送的提交'
parent=/home/zj/diagnostics/outputs
old=/home/zj/projects/my_project
BASELINE="$parent/v2_formal_easy_20261112_233055_519072036"
stamp=$(date +%Y%m%d_%H%M%S_%N)
checkout="/home/zj/diagnostics/my_project_safe_direct_$stamp"
root="$parent/v2_safe_direct_$stamp"
git clone --single-branch --branch multi-nfz-training https://github.com/aihelatte/brain_uav.git "$checkout" &&
  cd "$checkout" &&
  git checkout --detach "$REVIEWED_COMMIT" &&
  mkdir "$root"
```

在上述 checkout 中继续。第一条 plan 打印全部六组。带 --launch 的门控与训练每次各建一个新目录。保留同批门控目录直至汇总完成。建议按下面顺序执行，任一命令失败即停在该批。

```bash
py=/home/zj/envs/uav_torch/bin/python
run=("$py" -u -B -m brain_uav.scripts.run_v2_formal_easy_baseline)
common=(--source-manifest scripts/v2_formal_easy_safe_direct_source.json
  --ann-bc "$old/outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt"
  --snn-bc "$old/outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
  --validation-pool "$old/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json")
candidate=safe-direct-20261119
control=safe-direct-control-seed11

PYTHONPATH=src "${run[@]}" plan --root "$root/plan" --experiment "$candidate" "${common[@]}"

uav_batch() {
  local experiment="$1" batch="$2" out="$3" start rc
  shift 3
  start=$(date +%s)
  if PYTHONPATH=src "${run[@]}" launch --root "$out" --experiment "$experiment" --batch "$batch" \
       "${common[@]}" --launch "$@" 2>&1 | tee "$out.driver.log"; then
    rc=${PIPESTATUS[0]}
  else
    rc=${PIPESTATUS[0]}
  fi
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$out.driver.log"
  return "$rc"
}

uav_batch "$candidate" main "$root/gate-main" --gate-only &&
uav_batch "$candidate" main "$root/main" --gate-report "$root/gate-main" &&
uav_batch "$candidate" seed11 "$root/gate-seed11" --gate-only &&
uav_batch "$candidate" seed11 "$root/seed11" --gate-report "$root/gate-seed11"
```

本地旧 ANN seed11 的训练记录显示 batch 非阻塞关闭，和正式基准不同。找到的 SNN seed11 评估对应 min/mean 研究权重。用独立实验标识补齐原 ANN 和 SNN seed11。

```bash
uav_batch "$control" seed11 "$root/gate-control" --gate-only &&
uav_batch "$control" seed11 "$root/control" --gate-report "$root/gate-control"
```

查看状态时按对应批次更换目录及实验标识。三个训练批都完整结束后运行收集。

```bash
PYTHONPATH=src "${run[@]}" status --root "$root/main" --batch main --experiment "$candidate" "${common[@]}"
tail -n 20 "$root/main/"*_seed*.log

PYTHONPATH=src "${run[@]}" collect --launch --experiment "$candidate" --batch main \
  --root "$root/main" --second-batch "$root/seed11" --control-root "$root/control" \
  --baseline-root "$BASELINE" "${common[@]}"
```

## 产物与口径

- 汇总在 main/selection_summary.json，ANN、SNN 各选一份权重，保存在 main/selected。每模型保留三个 seed 的记录。选优依次比较正式通过、到达数、平均回报、较少训练步数、较小 seed。验证场景为固定 easy100。
- 同一场若同时违反顶部和水平边界，计为冲出高度上界。触地、碰撞、超时采用实际终止结果。末 50k 取各组终态向前 50k 步，纳入起点及终点均落入窗口的完整训练回合。按 scenario_id 保存逐场对照，并分别汇总原冲顶和原超时到达、保持原失败及其他失败的次数。
- 新方案各 worker 用相同的九个固定诊断场景，空场景、旁侧有障碍且直线路径畅通、直线路径被障碍挡住各三场。诊断文件为 diagnostic_evaluation.json。成功航程是从起点到终止步端点的实际轨迹折线长。三类诊断与 easy100 分开。
- CUDA 门控沿用原容差、更新步号、图复用、退出清理和 128 次物理更新预算。新方案 gate 特意在同一 actor batch 注入两段和空掩码样本，核对两段均值及 eager 对照，同时保留原 SNN ACC beta 与 Graph 重建检查。真实 CUDA/MPS 数值一致性、四进程并发、实际飞行结果均以服务器留档为准。

## 改动文件

- 配置与环境为 `config.py`、`envs/static_no_fly_env_runtime.py`、`envs/v2_static_no_fly_env.py` 和 `observations/v2_builder.py`，均位于 `src/brain_uav/`。
- 正式训练为 `src/brain_uav/trainers/` 下的 `v2_formal_training.py`、`v2_replay_buffer.py`、`v2_td3.py`、`v2_gpu_update.py` 和 `v2_validation.py`。
- 入口和收集为 `src/brain_uav/scripts/` 下的 `train_v2_td3.py`、`run_v2_formal_easy_baseline.py`、`v2_formal_easy_cuda_gate.py` 和新增的 `v2_formal_easy_guidance.py`。
- 新增 `scripts/v2_formal_easy_safe_direct_config.json`、`scripts/v2_formal_easy_safe_direct_source.json`、`tests/test_v2_formal_easy_guidance.py` 和本说明。`tests/test_v2_formal_easy_lr_final.py` 改用原提交校验旧清单，原清单继续按原始源码封存。
