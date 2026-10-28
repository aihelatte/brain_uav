# GPUCompute 少量 CUDA Event 诊断（默认关闭）

`--gpu-compute-event-updates 32` 为并发诊断增加每个被选 worker 至多32次采样。
默认0时不创建 Event，不新增 Event 报告字段。仅允许现有 A 配置：GPUCompute、CPU去重、
batch nonblocking 开启，系数非阻塞实验关闭，正式编译组合不变。
要求同时显式 `--gpu-compute-lightweight-timing`，保留三项 CPU perf_counter 与完整父级时间。
不兼容 profiler/batch细测/trace；不改变原 profiler 互斥规则。

仅 ann_0/snn_0 采 Event，其他 worker 持续原计算负载（也记录轻量CPU计时）。
一组三个 Event 在既有预热末端同步前首次 record，复用已有同步完成初始化；不插入捕获图。
测量时 E0 在固定输入 D2D 前、E1 在 replay 前、E2 在 replay 后且原结果读取前，均在真实当前流。
原 output.cpu().tolist() 返回后 query 全部 Event，只有已就绪才读 elapsed_time；不新增同步。
未就绪时保存原因并停止后续复用，不填0、不额外等待。异常沿原安全处理传播。

采样优先覆盖各场景/模式，再按共同窗口8个时间段与模式分散，避免固定偶数间隔只采一种模式。
32是上限而非强行凑满目标；有效样本须覆盖全部选定场景、actor和critic_only。
截止后不开始；跨截止和整步尾步排除，不延窗。样本的 scenario_id 指当时真实环境场景，
replay batch 仍按原规则混采，current/next_shape 记录实际采样 batch 的zone张量shape。

worker 的 `other/gpu_compute_event_samples.json` 保存有限明细：步骤/模式/场景/shape、
采样起止、整步窗口判定、Event就绪、CPU三项、GPU两个跨度及无效原因。
GPUCompute metadata 和父进程 review 汇总保存 requested/effective、样本数、限额与独立 diagnostic_valid。
所有带 Event 的结果 instrumented=true、valid_for_speed_comparison=false；其他纯测速严格门槛不变。
新增编译/捕获、窗口覆盖不足或其他真实错误仍阻断后续条件，不能作为有效定位结果。

两个跨度为 GPU **时间轴跨度**，可能包含 CPU 提交间隙、设备调度和其他进程竞争；
E0 执行前的排队不在其中。不是纯 GPU 计算时间，不可与 CPU 墙钟相减求排队/纯CPU时间，
也不能把重叠的 CPU/GPU/TD3/整步相加。单进程 Event 无法识别具体跨进程锁或哪个进程占卡。
这是 fresh BC + offset 的短循环，不是恢复历史训练；不包含正式500k的全部报告/candidate负载。

## 待审查后服务器命令（本地未执行）

默认只输出计划；只有函数显式接收 `--launch` 才启动现有并发 worker。
两组/八组顺序执行，每条件重复2次、共同窗口30秒，不重跑系数A/B。

```bash
uav_compute_events() {
  local mode root py start rc
  mode=${1:-}
  if [ "$#" -gt 1 ] || { [ -n "$mode" ] && [ "$mode" != --launch ]; }; then return 2; fi
  cd /home/zj/projects/my_project || return 1
  root="outputs/v2_compute_events_$(date +%Y%m%d_%H%M%S_%N)"
  py=/home/zj/envs/uav_torch/bin/python
  mkdir -p "$root/review" || return 1
  local launch=()
  if [ "$mode" = --launch ]; then launch=(--launch); fi
  start=$(date +%s)
  PYTHONPATH=src "$py" -u -m brain_uav.scripts.v2_concurrent_diagnostic \
    --ann-bc-checkpoint outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt \
    --snn-bc-checkpoint outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt \
    --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
    --expected-pool-digest 2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b \
    --ann-actor-lr 1.55e-4 --snn-actor-lr 8.25e-4 --critic-lr 2.5e-4 --alpha 2.5 \
    --bc-schedule smooth-mid --seed 7 --validation-seed 20260904 \
    --batch-size 64 --replay-capacity 500000 --scenario-count 3 --snn-time-window 4 \
    --controlled-stage-step-offset 75000 --warmup-steps 128 --steps-per-level 256 \
    --measurement-seconds 30 --repeats 2 --concurrency 2 8 \
    --ready-timeout 1800 --completion-timeout 120 \
    --non-blocking-batch-transfer --gpu-compute-lightweight-timing \
    --gpu-compute-event-updates 32 --batch-preparation-workers ann_0 snn_0 \
    --output-dir "$root/other/run" "${launch[@]}" 2>&1 | tee "$root/review/driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root/review/driver.log"
  # 仅复制现有小型配置/汇总；Event明细留 other，不保存模型/trace。
  if [ -d "$root/other/run/review" ]; then cp -R -- "$root/other/run/review/." "$root/review/" || return 1; fi
  return "$rc"
}
uav_compute_events
# 审查后显式运行：uav_compute_events --launch
```

本地仅 CPU/mock 检查，不构成 CUDA Event 时序、计时准确性或吞吐验证。
