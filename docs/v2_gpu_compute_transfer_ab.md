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
