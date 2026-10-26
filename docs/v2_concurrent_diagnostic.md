# ANN + SNN 受控并发短测（待服务器审查）

本入口只复用 `profile_v2_td3` 的 GPU compute timing worker 和真实短循环。
默认输出计划；只有显式 `--launch` 才创建子进程。无 tmux、服务器连接或数据生成。

```bash
cd /home/zj/projects/my_project
PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -m brain_uav.scripts.v2_concurrent_diagnostic \
  --output-dir "outputs/v2_concurrent_$(date +%Y%m%d_%H%M%S_%N)" \
  --ann-bc-checkpoint outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt \
  --snn-bc-checkpoint outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt \
  --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
  --expected-pool-digest 2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b \
  --ann-actor-lr 1.55e-4 --snn-actor-lr 8.25e-4 --critic-lr 2.5e-4 --alpha 2.5 \
  --bc-schedule smooth-mid --seed 7 --validation-seed 20260904 \
  --replay-capacity 500000 --batch-size 64 --snn-time-window 4 \
  --scenario-count 3 --controlled-stage-step-offset 75000 \
  --warmup-steps 128 --steps-per-level 256 --measurement-seconds 30 \
  --repeats 2 --concurrency 2 8 --ready-timeout 1800 --completion-timeout 120
```

以上是**只读计划命令**，本轮未执行。审查后在同一命令末尾增加 `--launch`
才会启动短测。可显式使用 `--concurrency 2 6 8`。模型学习率和日程必填；
示例 ANN 学习率仅为明确的实验输入，不代表已替项目确定最佳值。
支持 `default/smooth-final/smooth-mid/smooth-mid-strong`，不认定 strong 为新基准。

每个条件、每次重复均独立初始化，条件及重复串行，条件内 worker 并发。
GPU compute、CPU 去重、精确椭球缓存固定开启，编译直接使用
`default_v2_cuda_graph_compilation()`；P1 关、P2/P3 开。
其他未采用开关保持关闭，500k 日程和原始零基 BC 时序不变。
固定池由现有 loader 校验 easy 层级、种子和 content_digest，worker 另核对 BC 场景兼容性。

所有 worker 在现有编译、Graph 预热及 CUDA 同步完成后才写 ready。
协调者发布同机 `perf_counter` 的共同未来开始时间及截止时间。
worker 到截止持续负载，不以固定测量步数先退出。
`steps-per-level` 在此是固定场景循环长度：将 256 步分配给所选场景，循环同一顺序；
自然 episode 结束仍在当前场景继续，片段预算结束不会伪造终态。
不同并发条件使用相同每模型配置、初始化、随机种子和场景顺序前缀，
但因固定时长会完成不同数量的步骤。未覆盖全部所选场景时，该轮无效。

等待有 ready/completion 超时；任一进程失败或分发失败，保存错误并仅停止本批拥有的子进程。
真实无效测量返回非零并停止后续条件。批准备插桩本身只取消纯测速资格，
诊断有效时仍继续八组；不会自动重跑、静默回退或改变 batch。
测量期新增 Dynamo 编译、手工图捕获或动作推理 Inductor 图录制均判无效。
Inductor 私有计数接口不可读或 manager 改变同样判无效，不推断“没有捕获”。
动作图 replay 证据沿用现有预热验证；手工更新图另记录测量期实际 replay 次数。
预热验证复用已有小范围 Graph 证据采集，主测量窗口不启用 torch profiler、
固定动作证据或环境细分插桩。需要环境细分时另用原诊断模式，含插桩开销。

批根 `review/plan.json` 记录配置、源码 HEAD/脏工作区、checkpoint 哈希、池 digest、
所选场景类型等；`review/summary.json` 按条件汇总模型延迟、单任务及整批吞吐。
每个条件重复的 `review/` 保存 worker 日志、紧凑汇总及失败原因；
`other/` 保存 worker 原始明细、负载快照和协调控制文件，不复制 checkpoint 或张量。
worker 明细还含 replay 容量/实际占用、场景覆盖、编译/捕获及时间窗口证据。

整批吞吐以共同窗口内完成步数之和除以共同窗口时长计算，末尾跨界步不计入该分子。
每步总耗时包含最后同步及计时统计回读；实际开始、完成时间及尾段超时另记录。
起步偏差超过 0.1 秒或未持续到截止均无效。父子区间不可相加：
TD3 包含采样、batch 准备/搬运和 GPU replay，环境包含推进、奖励、观测及 info。
未归属时间由整步减去互不重叠的父项计算，包含最终等待/计时回读等。
CPU 墙钟含提交与等待，不能称为纯 GPU 计算时间；不为分项增加 CUDA 同步。

fresh BC + 阶段偏移不代表恢复历史训练状态。
短测缺少正式报告、三视图和 candidate 验证负载，不保证 500k 同比例加速。
本工具不授予 ANN 严格数值验收通过结论，也不替代原有正确性验收。
本地 CPU/mock 检查不是服务器 tmux/CUDA 或并发吞吐证据。

## batch 准备及搬运专项诊断（默认关闭，待审查）

只定位原 `GPUCompute update_once → batch.to(device)` 的 CPU 墙钟。
不预设十几毫秒都是 PCIe 传输，不改变采样、14 次默认阻塞搬运、布局或 dtype。
默认纯测速不经过逐张量计时，也不启用此 profiler。

三个模式独立运行，不混用结果：

- 纯吞吐：不传以下任一选项，原严格有效性判据不变。
- 细测：`--batch-preparation-updates 32`，范围 2..64；不启用 torch profiler。
- 短 trace：`--batch-preparation-trace-updates 4`，范围 2..8；与细测互斥，
  使用已有 torch.profiler CPU/CUDA 活动和内存事件，无新依赖。

`--batch-preparation-workers ann_0 snn_0` 默认只采集两名 worker。
被选择的 id 必须存在于所有并发条件，其他 worker 始终维持原真实短循环负载。
双方沿用正式编译及 Graph 预热，ready/CUDA 同步完成后、共同窗口内才开始采集；
采满立即停止采集，仍持续负载至共同截止。样本必须包含 critic-only 和 actor 更新。
trace 达到采集上限、跨截止样本完成或异常时立即停止 profiler；
未采满时由窗口结束后的 finish 清理，停止只执行一次，导出/文件写入仍延后。
profiler 启停本身可能同步并扰动并发负载。
采样记录 `sample_started`、`sample_completed`（同机 perf_counter）及
`within_common_window`。截止后不启动新样本；跨截止样本保留明细，
但不计入 `valid_samples_in_window`，不能用于补满有效样本数或更新类型覆盖。
不足则诊断无效，不延长窗口；原训练尾步、吞吐分子和场景统计规则不变。
无足够样本、场景覆盖不足、worker 异常、就绪/窗口失败、新编译/捕获或计数不可读仍无效。
不扩大多形状预热；此类失败保留证据，不作为纯测速通过。

每个 worker 的 `other/batch_preparation_records.json` 仅保存有限数值明细：
worker/model、更新序号、样本序号、actor_updated、当前/下一 zone shape，
14 个 `tensor_calls` 的名称、dtype、shape、stride、layout、字节数、source_device、pinned、
`cpu_wall_seconds`，三个 `constructors` 的耗时，以及
`batch_preparation_wall_seconds`、`uncovered_wall_seconds` 和总字节数。
构造参数全部求值后才启动构造器计时，子 `.to()` 不重复计入构造器。
总时间使用既有 engine 的 batch_preparation 父项计时（`total_source` 标明来源），
包含元数据、scope、计时器和回调收尾等诊断开销；残差如实保留，不强制为零。
不额外逐张量同步，不读取 `.cpu()`/`.item()`，不保留张量，不逐步刷文件。
仅在窗口结束/异常退出后写明细和汇总。

worker 的 `review/batch_preparation.json` 为小型汇总，含各调用次数/平均 CPU 墙钟、
总准备时间和残差均值、采集数、错误及 `diagnostic_valid`。
原 worker 明细的 `batch_preparation_diagnostic` 和父进程 review 汇总透传这些字段。
trace 存在 worker 的 `other/batch_preparation_trace.json`，有 update、14 次搬运和
三个构造器的命名 scope，可关联可见的 memcpy、分配和 runtime 同步活动。
`trace` 汇总可见 GPU copy/HtoD 次数及可用字节数、runtime 调用次数/墙钟、memory 事件。
未观察到 copy 或缺少字节字段用 null 和原因表示；不会以零冒充“无传输/无等待”。
驱动排队、其他 GPU 进程活动与同步归属仍可能未知，必须结合时间线解释。
CUDA profiler 不可用、导出失败或没有 CUDA 活动时给出具体原因并判诊断无效。
trace 的启停/导出、内存跟踪可能同步并扰动整个并发批次，不能当作正常吞吐。
未开启旧 detailed/compiled profiler，也未取消其与正常测速的互斥限制。

批与 worker 分别报告 `diagnostic_valid` 和 `valid_for_speed_comparison`。
任何含本插桩的批次均为 `instrumented=true`、纯测速资格 false；
诊断完整有效则返回 0 并继续后续并发条件。纯测速仍需原所有严格条件通过。
`batch_steps_per_second` 等原计时保留为插桩时的描述值，不能用于加速验收；
共同窗口外尾步仍从吞吐分子排除。父子计时不能相加。

以下 Bash 函数只准备命令：默认输出计划，不创建 worker。
先确认服务器同步的是审查后的代码；BC、池路径及资源负载未在本地验证。
ANN 学习率及 smooth-mid 为本例显式输入，可按已审查研究配置调整，
不能把 smooth-mid-strong 自行视为基准。

```bash
batch_probe() {
  if cd /home/zj/projects/my_project; then
    mode="$1"
    updates="$2"
    shift 2
    root="outputs/v2_batch_probe_${mode}_$(date +%Y%m%d_%H%M%S_%N)"
    started=$(date +%s)
    PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u \
      -m brain_uav.scripts.v2_concurrent_diagnostic \
      --output-dir "$root" \
      --ann-bc-checkpoint outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt \
      --snn-bc-checkpoint outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt \
      --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
      --expected-pool-digest 2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b \
      --ann-actor-lr 1.55e-4 --snn-actor-lr 8.25e-4 --critic-lr 2.5e-4 --alpha 2.5 \
      --bc-schedule smooth-mid --seed 7 --validation-seed 20260904 \
      --replay-capacity 500000 --batch-size 64 --snn-time-window 4 \
      --scenario-count 3 --controlled-stage-step-offset 75000 \
      --warmup-steps 128 --steps-per-level 256 --measurement-seconds 30 \
      --repeats 2 --concurrency 2 8 --ready-timeout 1800 --completion-timeout 120 \
      --batch-preparation-workers ann_0 snn_0 \
      "--batch-preparation-${mode}" "$updates" "$@" 2>&1 | tee "$root.log"
    rc=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - started ))
    printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$elapsed" | tee -a "$root.log"
  else
    printf 'cd failed; diagnostic not started\n' >&2
  fi
}

# 两份独立计划；不启动任务。
batch_probe updates 32
batch_probe trace-updates 4

# 审查后才可分别执行，独立目录，不与纯吞吐混用：
# batch_probe updates 32 --launch
# batch_probe trace-updates 4 --launch
```

只有以后显式添加 `--launch` 才启动短测；本轮未执行上述服务器命令。
fresh BC＋阶段偏移仍不代表历史状态恢复，短循环缺少正式报告、三视图和 candidate 负载。
