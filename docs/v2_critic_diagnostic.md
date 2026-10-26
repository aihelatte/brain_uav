# 终态 critic 评分可靠性诊断（待审查）

仅新增独立脚本 `brain_uav.scripts.diagnose_v2_critic`，不构造 optimizer、replay
或训练执行器。沿用 `diagnose_v2_policy.load_diagnostic_inputs` 严格重建 actor，
并按 checkpoint 声明架构重建 online critic1/2，检查固定 buffer、权重形状和有限性。
失败 checkpoint 仅用于此独立诊断，不改变 passed 标记或 medium 前驱规则。

2026-10-26 本地只读核对：两份给定输入均为 SNN/easy/seed23/500000 步，
actor_lr=8.25e-4、gamma=0.99、投影32、actor hidden192、critic hidden128、
time_window4、动作界限约[0.14,0.2]弧度。架构、奖励、BC smooth-mid-strong、
场景、UAV半径、初始化来源、种子及固定池元数据一致；配置唯一差异为
critic_lr：A=2.5e-4，B=1e-3，保存的 optimizer 参数组 LR 与之相符。
两者更新次数均499937，actor更新237500。训练权重和结果当然不同。
这不提供训练过程中的时间先后或因果证据，也不核实未挂载服务器运行状态。

固定池 loader 读取对象确认100场、7空场、预期 digest 一致。
预选择 sequence_index 为：
`1,8,13,19,32,35,37,40,45,52,53,56,59,69,74,84,90,95,97,98`。
覆盖空场景及 sphere、ellipsoid、box、triangular_pyramid、quadrangular_pyramid。
选择使用 NumPy default_rng(seed=20261026)，不看飞行结果；实际软件版本随 manifest 保存。

每场双方自然飞行后，按场景顺序交替以A/B为anchor，取动作前索引
floor(0.75*(T-1))。每个候选分支重新 reset 原始场景、重放实际 float32 前缀，
精确核对观测/mask、运动状态、steps、prev_action、recent_progress、最佳进度、
线段终止历史、环境RNG及轨迹摘要；不靠拷贝位置恢复。
仅剔除进程局部 geometry identity，以原始场景重建保证几何相同，
P1前次净空上下文的数值部分仍核对。剩余时限不重置。

双方 actor 动作及 anchor 每维归一化±0.25组成共同候选，裁剪并按 float32 精确去重。
Q1是主要口径（当前 actor RL 目标使用online Q1）；同时记录Q2/minQ及分歧。
每侧 G 包含候选动作第一步奖励，之后用自己的 actor 完整续飞至真实终止。
仅在 anchor 自身候选与自然轨迹当步动作完全相同时复用已得后缀。
无探索、eval、inference_mode，SNN普通forward前后重置状态；不收集脉冲数组。
eager运行，精确椭球缓存/CPU去重统一显式开启，不改变正式性能路径。

并列规则预先固定：abs(a-b) <= 1e-3 + 1e-4*max(abs(a),abs(b))。
Q的主要口径先计算每个候选相对该模型自身actor候选的优势ΔQ_i=Q1_i-Q1_own，
再将ΔQ作为a/b判断成对并列及Q1最高候选的近似并列集合；两处使用相同口径，
对Q整体常数平移不敏感。G仍使用原始折扣回报判断并列。
单列Q并列、G并列、双方并列；不将微差记为可靠反向。
原始Q、G、Q-G均保留；regret仍记录Q1原始argmax候选的G损失及近似并列集合的损失范围。
同时记录相对自身actor候选的Q/G增量及预测偏差。场内候选不是独立统计样本。
有效非并列排序对数=一致对数+反向对数，即Q优势和G都不并列的对。
summary按模型及空/非空组明确列出有效对数、一致/反向/Q并列/G并列/双方并列数量、
没有有效排序对的探针数量。零有效对标为information insufficient，不能解释为评分可靠；
即使存在有效对也只提供描述性证据，不自动给出可靠性通过结论。

输出 manifest.json、baseline_episodes.jsonl、probes.jsonl、branches.jsonl、
summary.json、README.txt；目录已存在即拒绝。manifest在首次网络前向前写入。
数值或重放异常会抛错，并在已有输出目录保存 execution_completed=false 及错误。
评分反向属于正常诊断结果，不伪装崩溃，也不输出正式passed结论。
自然飞行动作/奖励/状态及分支续飞保存为普通JSON，不复制模型或输出大量图片。

先看每场 Q1 排序是否与 G 一致，再看 regret、own_Q1_minus_G、双critic分歧和outcome。
任务成功与折扣奖励并不等价。G是当前actor无噪声续飞的参考回报，并非精确TD3训练target；
target-policy smoothing、target滞后、观测未显式表示的历史/剩余时限均影响解释。
actor还受BC、terminal_geo和脉冲正则约束；没有选择候选Q1最大值不证明更新有错。
本检查不能回答历史上谁先退化，也不能把20场富集子集当作100场正式成功率。

以下服务器命令只是草案，需先核对同步后的代码和输入路径，本轮未执行：

```bash
if cd /home/zj/projects/my_project; then
  root="outputs/v2_critic_reliability_$(date +%Y%m%d_%H%M%S_%N)"
  started=$(date +%s)
  PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u \
    -m brain_uav.scripts.diagnose_v2_critic \
    --checkpoint-a outputs/v2_smooth_mid_strong_round1/20261019_142701/snn_seed23/v2_snn_td3_easy_failed.pt \
    --checkpoint-b outputs/v2_snn_critic_lr_confirm_round1/20261024_102749/snn_clr_1e-3_seed23/v2_snn_td3_easy_failed.pt \
    --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
    --output-dir "$root" \
    --device cuda 2>&1 | tee "$root.log"
  rc=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - started ))
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$elapsed" | tee -a "$root.log"
else
  printf 'cd failed; diagnostic was not started\n' >&2
fi
```

A/B服务器路径是由本地产物树推导的待确认映射，尚未连接服务器核实。
日志位于输出目录旁的`$root.log`，预检查失败、尚未创建结果目录时也保留日志和真实Python退出码。
支持 `--device cpu`，仍会完整运行20场及最多240分支，不是快速测试开关。
本地仅测试小型CPU/mock场景与真实输入的只读严格重建；未执行完整诊断或CUDA检查。
