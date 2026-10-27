# GPUCompute pageable blocking/nonblocking A/B（命令草案，待审查）

本轮只接通现有 `non_blocking_batch_transfer`；正式默认仍为 False。
两侧 GPUCompute、CPU 去重、椭球缓存和正式编译不变。
不使用 pinned、打包、预取、额外 stream 或 DeviceReplay。
14 次 Tensor.to 的顺序、dtype、布局及当前 CUDA 流均沿用原实现。

nonblocking 的独立 CPU replay 样本由调用作用域持有至 executor 的原
`output.cpu().tolist()` 完成屏障。上传、噪声准备、绑定/捕获和执行异常时，
仅异常路径同步上传时的原流后才释放源。若流无法排空，engine 持有源并阻止继续更新；
这是明确失败，不回退。正常路径无新增 CUDA 同步，Adam/target 保护未改变。

`local_optimizations.non_blocking_batch_transfer` 记录 requested/effective/reason、
actual_path、实际上传调用数。并发父进程核对请求与实际执行证据。
GPUCompute 正确性先显式生成 blocking reference，再让 nonblocking worker 加载；
reference 除原契约外记录搬运对照来源，禁止拿 CPU 原更新路径或 nonblocking reference 冒充对照。
新增字段只在诊断 reference 中，不进入正式 checkpoint schema/算法配置。
原数值容差、异常注入、恢复更新、加载失效和 Graph 验证不变。

以下完整 Bash 草案先做 ANN/SNN 正确性，再串行 blocking/nonblocking 的 2/8 并发纯测速。
只做 easy；学习率与 smooth-mid 是显式实验输入，不是新默认。两侧保持同一输入及固定池。
correctness 使用原受控步骤序列（含各模式、异常恢复及 checkpoint），不是 500k rollout；
timing 使用 fresh BC + 阶段偏移，不能声称恢复历史训练状态。
命令本轮未执行，服务器应先同步审查后的代码。`--launch` 仅在审查后使用；
去掉它可只看并发计划。任一严格正确性失败都跳过测速，不自动接受 ANN 新差异。

```bash
if cd /home/zj/projects/my_project; then
  root="outputs/v2_compute_transfer_ab_$(date +%Y%m%d_%H%M%S_%N)"
  mkdir -p "$root/review" "$root/other"
  py=/home/zj/envs/uav_torch/bin/python
  ann=outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt
  snn=outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt
  pool=outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json
  digest=2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b
  common=(--device cuda --gpu-td3-compute --validation-pool "$pool"
    --expected-pool-digest "$digest" --seed 7 --validation-seed 20260904
    --batch-size 64 --replay-capacity 500000 --scenario-count 3
    --snn-time-window 4 --comparison-critic-lr 2.5e-4 --comparison-alpha 2.5
    --comparison-bc-schedule smooth-mid --controlled-stage-step-offset 75000
    --warmup-steps 128 --steps-per-level 256 --repeats 1 --correctness-updates 8)
  ok=1
  for model in ann snn; do
    if [ "$model" = ann ]; then bc="$ann"; lr=1.55e-4; else bc="$snn"; lr=8.25e-4; fi
    for mode in blocking nonblocking; do
      flags=(--gpu-compute-write-reference)
      if [ "$mode" = nonblocking ]; then
        flags=(--non-blocking-batch-transfer --gpu-compute-reference "$root/other/${model}_blocking/reference.pt")
      fi
      out="$root/other/${model}_${mode}"
      started=$(date +%s)
      PYTHONPATH=src "$py" -u -m brain_uav.scripts.profile_v2_td3 \
        "${common[@]}" --model "$model" --bc-checkpoint "$bc" \
        --comparison-actor-lr "$lr" --gpu-compute-worker correctness \
        "${flags[@]}" --output-dir "$out" 2>&1 | tee "$root/review/${model}_${mode}.log"
      rc=${PIPESTATUS[0]}
      elapsed=$(( $(date +%s) - started ))
      printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$elapsed" | tee -a "$root/review/${model}_${mode}.log"
      if [ "$rc" -ne 0 ]; then ok=0; break; fi
      "$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); passed=r.get("passed") is True and r.get("completed") is True; json.dump({"strict_passed":passed,"engineering_accepted":False,"detail":sys.argv[1]},open(sys.argv[2],"w"),indent=2); sys.exit(0 if passed else 1)' \
        "$out/correctness.json" "$root/review/${model}_${mode}_gate.json"
      gate_rc=$?
      if [ "$gate_rc" -ne 0 ]; then ok=0; break; fi
    done
    if [ "$ok" -ne 1 ]; then break; fi
  done
  if [ "$ok" = 1 ]; then
    for mode in blocking nonblocking; do
      flags=()
      if [ "$mode" = nonblocking ]; then flags=(--non-blocking-batch-transfer); fi
      started=$(date +%s)
      PYTHONPATH=src "$py" -u -m brain_uav.scripts.v2_concurrent_diagnostic \
        --output-dir "$root/timing_$mode" --ann-bc-checkpoint "$ann" --snn-bc-checkpoint "$snn" \
        --validation-pool "$pool" --expected-pool-digest "$digest" \
        --ann-actor-lr 1.55e-4 --snn-actor-lr 8.25e-4 --critic-lr 2.5e-4 --alpha 2.5 \
        --bc-schedule smooth-mid --seed 7 --validation-seed 20260904 \
        --batch-size 64 --replay-capacity 500000 --scenario-count 3 --snn-time-window 4 \
        --controlled-stage-step-offset 75000 --warmup-steps 128 --steps-per-level 256 \
        --measurement-seconds 30 --repeats 2 --concurrency 2 8 \
        --ready-timeout 1800 --completion-timeout 120 "${flags[@]}" --launch \
        2>&1 | tee "$root/review/timing_$mode.log"
      rc=${PIPESTATUS[0]}
      elapsed=$(( $(date +%s) - started ))
      printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$elapsed" | tee -a "$root/review/timing_$mode.log"
      if [ "$rc" -ne 0 ]; then break; fi
    done
  else
    printf 'Strict correctness failed; timing skipped. Review: %s/review\n' "$root"
  fi
else
  printf 'cd failed; no diagnostic started\n' >&2
fi
```

不传 batch-preparation-updates、batch-preparation-trace-updates 或旧 profiler 选项；
本轮 A/B 的主测速必须无这些插桩。比较 worker 的 TD3 breakdown 中 batch_preparation、
gpu_compute_input_copy、gpu_compute_to_result，同时比较完整 td3_update_wall_seconds、
batch_transfer_window_completion 和 total_wall_seconds/seconds_per_step_including_final_drain。
前者减少但后者没有改善时，不能宣称加速。所有这些是 CPU 完成墙钟/提交等待区间，
不是纯 GPU 耗时；TD3 包含子项，不能重复相加。
共同窗口吞吐仍排除尾步，整步仍包含最终同步与统计回读。
测量期新增编译/捕获、覆盖不足、实际执行证据缺失仍判无效。

review 放小型计划、门控、汇总、错误和日志；other 放 reference、异常状态及详细记录。
不复制 BC/checkpoint 到 review。CPU/mock 测试不证明 pageable CUDA 异步复制正确性或收益；
服务器必须重新通过上述严格正确性检查，才可解释纯测速结果。

## 系数上传屏障旁观 trace（独立运行，命令仅准备）

`--gpu-compute-submission-trace-updates 4` 默认关闭，仅允许真实 nonblocking
GPUCompute 并发 worker。它不调用旧的 blocking transfer 回调，原 14 次 `.to()`
仍由原 `V2ReplayBatch.to(non_blocking=True)` 执行。旧 batch 细测/trace 与
nonblocking 的互斥保护保留，新选项也不能与旧细测或深 profiler 混用。

沿用上轮配置：GPUCompute/CPU 去重/精确椭球缓存与正式编译开启，其他实验关闭；
500k 日程、P1 关/P2/P3 开、batch64、seed7、offset75000、smooth-mid；
ANN actor_lr=1.55e-4、SNN=8.25e-4、critic_lr=2.5e-4、alpha=2.5。
fresh BC 加阶段偏移不是恢复历史训练状态；短循环也不完整复现正式报告及 candidate 负载。

以下函数每次调用使用独立时间戳目录。定义后执行 `uav_submission_trace` 只检查并打印计划；
审查授权后执行 `uav_submission_trace --launch` 才启动短测，不创建 tmux。
2 和 8 条件顺序执行，各一次 30 秒共同窗口，条件内 worker 并发；仅 ann_0/snn_0
在预热后采 4 次更新，其余 worker 持续负载。未采满、跨窗、角色/场景覆盖不足、
新增编译/捕获或真实路径证据缺失均判诊断无效，不延窗或自动重试。

```bash
uav_submission_trace() {
  if cd /home/zj/projects/my_project; then
    py=/home/zj/envs/uav_torch/bin/python
    root="outputs/v2_nonblocking_submission_$(date +%Y%m%d_%H%M%S_%N)"
    ann="outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt"
    snn="outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
    pool="outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json"
    digest="2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b"
    if mkdir -p "$root/review"; then
      started=$(date +%s)
      PYTHONPATH=src "$py" -u -m brain_uav.scripts.v2_concurrent_diagnostic \
        --output-dir "$root/run" --ann-bc-checkpoint "$ann" --snn-bc-checkpoint "$snn" \
        --validation-pool "$pool" --expected-pool-digest "$digest" \
        --ann-actor-lr 1.55e-4 --snn-actor-lr 8.25e-4 --critic-lr 2.5e-4 --alpha 2.5 \
        --bc-schedule smooth-mid --seed 7 --validation-seed 20260904 \
        --batch-size 64 --replay-capacity 500000 --scenario-count 3 --snn-time-window 4 \
        --controlled-stage-step-offset 75000 --warmup-steps 128 --steps-per-level 256 \
        --measurement-seconds 30 --repeats 1 --concurrency 2 8 \
        --ready-timeout 1800 --completion-timeout 120 \
        --non-blocking-batch-transfer --gpu-compute-submission-trace-updates 4 \
        --batch-preparation-workers ann_0 snn_0 "$@" \
        2>&1 | tee "$root/review/driver.log"
      rc=${PIPESTATUS[0]}
      elapsed=$(( $(date +%s) - started ))
      printf 'exit_code=%s\nprocess_wall_seconds=%s\noutput_root=%s\n' \
        "$rc" "$elapsed" "$root" | tee -a "$root/review/driver.log"
      return "$rc"
    else
      printf 'Cannot create output directory; nothing started\n' >&2
      return 1
    fi
  else
    printf 'cd failed; nothing started\n' >&2
    return 1
  fi
}
uav_submission_trace
# 审查后单独执行：uav_submission_trace --launch
```

每个所选 worker 的 `review/gpu_compute_submission.json` 保存小型摘要，
`other/gpu_compute_submission_records.json` 保存有限记录，
`other/gpu_compute_submission_trace.json` 保留原始 runtime、memcpy、kernel、stream、
correlation/flow 信息（以实际 profiler 可见信息为准）。根 review 保留计划、日志与汇总。
不保存张量、checkpoint，不逐步写盘；导出在共同窗口之后。

每样本记录 total_steps、actor_updated、当前/下一 zone shape、起止时刻和各 scope CPU 墙钟。
scope 为 batch_upload、noise_preparation、noise_fixed_input_copy、
coefficients_construct_and_upload、graph_fixed_input_copy、graph_replay_submit、
result_read_and_completion。包含整个 update 的父 scope 不与子项相加。
达到上限立即停止 profiler；异常和 finish 清理幂等，导出延后。
profiler 启停可能同步并扰动并发，因此 instrumented=true、valid_for_speed_comparison=false；
diagnostic_valid 独立判断，有效插桩结果不会阻止继续运行八组条件。

用 correlation/flow 将某次 CPU runtime 上传与 GPU memcpy 对应，再检查对应同步区间及
后续 Graph 提交时间。CPU scope 是提交/等待墙钟，不能称为 GPU 计算时间；
有效性逐采样 update 分别核对 batch 与系数 scope 内 runtime 到 GPU HtoD 的实际关联及
非空输入对应的复制字节；External id 或字段名存在不能证明关联。空张量不要求复制事件，
也不要求必须出现同步事件。缺任一必要关联时标记 unknown、归因不完整、诊断无效。
16B 上传的邻接关系本身不能证明它等待了前面的 batch。缺少关联时标 unknown，
不能补零或推断整卡空闲。单进程 trace 不能证明具体跨进程锁或整卡资源竞争；
本轮不会自动宣布假设成立。CPU/mock 测试不构成 CUDA、速度或根因证据。

可选的服务器只读能力检查（本轮未执行，不安装或修改配置）：

```bash
if command -v nsys >/dev/null 2>&1; then
  nsys --version
  nsys status --environment
  nsys profile --help
else
  printf 'Nsight Systems unavailable; full-process timeline capability unknown\n'
fi
```
