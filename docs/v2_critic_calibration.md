# B critic 偏差定位与固定数据短程校准

独立诊断入口：`python -m brain_uav.scripts.diagnose_v2_critic_calibration`。
仅接受指定 SHA256 的 SNN/easy/seed23/500k/critic_lr=1e-3 终态模型。
SNN 仅指 actor；双 critic 为 ANN。严格加载 checkpoint 中的 online/target 权重及联合 Adam。
不调用 formal trainer、GPUCompute、课程、额外策略飞行或固定验证。

输入 `--history-dir` 必须同时含原 `manifest.json` 与 `baseline_episodes.jsonl`。
只取 B 的20场自然轨迹，从固定池原 payload reset 后逐动作重放一次，逐步精确核对
pre_action_hashes、states、rewards、终态及回报；失配直接报错，禁止补飞或放宽容差。
跨平台浮点差异也会阻断，CPU 不能替代原 CUDA 环境的完整重放验收。

固定 seed=20261028，仅按场景 ID 和空/非空属性抽4场留出（各2场），16场训练；
划分在拟合前写 manifest。同一场景不拆分。其余抽样与冻结平滑噪声各用独立 RNG。
G 包含当前奖励，terminated/truncated 后停止，来自旧 online actor 无探索自然轨迹。
y 使用保存的末期 policy_noise=.006、noise_clip=.012，一个 transition 一个冻结噪声标签，
并非噪声期望；不能把 y-G 全部归为实现错误。

`bias_decomposition.jsonl` 保存逐 transition Q1/Q2、reward/done/G、target Q1/Q2、y、
Q-G、Q-y、y-G 及平滑噪声。相邻目标对照为：online critics+online actor 无噪声 →
target critics+online actor → target critics+target actor → 平滑噪声。
各 shift 是后项减前项，含 gamma*(1-done)；min 相对均值的下移为另外一个重叠对照，
不能再次加到上述相邻拆分中。当前/下一观测来自重放，不保存张量或模型。

fixed_td/mc_return 从相同 online critic 权重和 Adam moments/step 出发，唯一标签差异为 y/G。
参数顺序 critic1 后 critic2，一个联合 Adam，一次全局裁剪，保存的学习率和裁剪阈值不变。
诊断 eager Adam 显式使用 fused=False/capturable=False，其余保存的超参数不变。
两组共享预生成的1000×64均匀有放回训练索引；这是固定离线监督实验，不模拟正式 replay。
actor/target 全冻结且不软更新，前后精确状态 hash 核对。0/100/500/1000评估，
最终预定1000，不按留出选择最佳点或延长。

`fit_curves.json` 保存分组指标和每次更新 clip 前全局梯度范数；累计裁剪比例按实际全局范数
超过阈值计算。梯度是联合训练 batch 的量，不能归给留出或单一空/终态分组。
Q-G 有符号均值、MAE/RMSE、Q-y 按 train/holdout、空/非空、terminal/nonterminal 交叉分组。
同时给 transition 加权及场景等权；场景等权 RMSE 为 sqrt(mean(各场MSE))。空组标信息不足。
`summary.json` 的 execution_completed 仅表示执行完成，不是 formal passed。
已有输出目录拒绝覆盖，异常也保存 summary；不输出 checkpoint。
MC 校准成功不等于正式训练已修复；1000次未改善也不证明容量不足。

## 待审查后服务器命令（未执行）

先核实历史诊断已位于下述目录；脚本不会复制、生成或改写旧产物。
以下函数只执行独立离线诊断，无 tmux；本轮开发不执行。

```bash
uav_critic_calibration() {
  cd /home/zj/projects/my_project || return 1
  local root py start rc
  root="outputs/v2_critic_calibration_$(date +%Y%m%d_%H%M%S_%N)"
  py=/home/zj/envs/uav_torch/bin/python
  mkdir -p "$root/review" || return 1
  start=$(date +%s)
  PYTHONPATH=src "$py" -u -m brain_uav.scripts.diagnose_v2_critic_calibration \
    --checkpoint outputs/v2_snn_critic_lr_confirm_round1/20261024_102749/snn_clr_1e-3_seed23/v2_snn_td3_easy_failed.pt \
    --history-dir outputs/v2_critic_reliability_round1/20261026_171846_464722553 \
    --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
    --device cuda --output-dir "$root/other" 2>&1 | tee "$root/review/driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root/review/driver.log"
  # 小型配置/汇总复制到 review；逐 transition/曲线明细留 other。无 checkpoint。
  if [ -f "$root/other/manifest.json" ]; then cp -- "$root/other/manifest.json" "$root/review/manifest.json" || return 1; fi
  if [ -f "$root/other/summary.json" ]; then cp -- "$root/other/summary.json" "$root/review/summary.json" || return 1; fi
  return "$rc"
}
# 审查后显式调用：
# uav_critic_calibration
```

本地测试使用2步环境与微型 CPU 网络，未执行真实20场重放或1000次拟合；
不构成服务器 CUDA、正式训练或因果根因验证。
