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
无效测量返回非零并停止后续条件。不会自动重跑、静默回退或改变 batch。
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
