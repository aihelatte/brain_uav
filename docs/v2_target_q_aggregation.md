# min / mean 目标聚合实验（命令待审查，尚未执行）

正式 API `target_q_aggregation='min'|'mean'`，CLI `--target-q-aggregation min|mean`。
默认仍为 min；min 保留 `torch.minimum` 的运算。mean 为 `(Q1+Q2)/2`。
两者均在 `r + (1-done)*gamma*Q` 中使用，done 保持 terminated OR truncated。
单阶段、课程、eager、编译共享目标及 GPUCompute 共用选择；GPU 静态绑定包含模式。
不支持在已绑定 Graph 上直接切换模式；严格重建/失效规则不变。
算法字段进入 formal_config 和 engine.algorithm_config，不改格式版本。
旧产物缺字段仅解释为 min；非法值及 formal/engine 不一致拒绝。
正式/阈值快照均通过原严格诊断加载器，不将快照冒充 passed 或课程前驱。

## 固定数据动态诊断

入口 `python -m brain_uav.scripts.diagnose_v2_target_aggregation`，模式 prepare / fit / evaluate。
本地已只读核实 B SHA256 为
`c5cb20f60c8181fa03bc1fcd31901cfeae218b1eecd414b186abd15f5710f36d`，
历史目录 `v2_critic_reliability_round1/20261026_171846_464722553` 的 B 为20场、12033步。
未执行真实重放。运行时重新核对 checkpoint、历史 manifest/轨迹、固定池 digest。
prepare 在严格线性重放前落盘场景划分；沿用 seed=20261028，16训练、4留出（各2空/非空）。
保存一次共同 `data.pt`、manifest、data_evidence（SHA256/批次 hash），之后只读使用。
这个包含 observation dataclass 的本地可信 pickle 不是模型或正式 checkpoint，勿加载未知来源文件。
fit 重新检查 hash、16/4分组、2000×64确定性训练批次及无留出泄漏；各臂独立输出，拒绝覆盖。

source=min；treatment=min/mean。来源 checkpoint 原样严格加载，干预只作用于独立副本。
online actor 冻结、两臂统一使用它作目标策略、噪声0；target critics 从保存权重开始，
按原 tau 和 policy_delay 软更新，频率相位接在保存的 last_total_steps 后。
每次从当前 targets 重新计算标签；G 不进入损失。
恢复同一联合 Adam（critic1参数后接critic2），同一全局梯度裁剪，eager fused/capturable=False。
不做 actor/BC/脉冲更新，不调用 formal trainer/GPUCompute；不是正式 TD3 的完整复现。
0/100/500/1000/2000评估当前标签拟合误差和 Q-G；不按留出挑最佳，不延长预算。
fit_curves 包含分组及逐场景的有符号误差、MAE、RMSE、TD MAE/RMSE，
transition加权与场景等权，空/非空、terminal/nonterminal；空组标 insufficient。
initial、batch_sha256、actual_updates、target_updates、actor_unchanged 及非有限检查供核对。
两臂 initial 和 batch hash 必须一致，targets 应实际更新；不一致则本轮不可比较。
所有产物 diagnostic_only，不输出模型；诊断改善不能宣称正式训练问题已解决。

以下函数均不会在定义时启动任务，无 tmux、无自动服务器连接。
准备数据不是重新采集策略飞行；两臂可在两个终端中独立启动，勿写同一输出目录。

```bash
uav_target_diag() {
  cd /home/zj/projects/my_project || return 1
  local mode="$1" root="$2" start rc
  shift 2
  mkdir -p "$root/review" || return 1
  start=$(date +%s)
  PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u \
    -m brain_uav.scripts.diagnose_v2_target_aggregation --mode "$mode" \
    --checkpoint outputs/v2_snn_critic_lr_confirm_round1/20261024_102749/snn_clr_1e-3_seed23/v2_snn_td3_easy_failed.pt \
    --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
    --device cuda --output-dir "$root/other" "$@" 2>&1 | tee "$root/review/driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root/review/driver.log"
  for f in manifest.json summary.json data_evidence.json; do
    if [ -f "$root/other/$f" ]; then cp -- "$root/other/$f" "$root/review/$f" || return 1; fi
  done
  return "$rc"
}
# 审查后：先准备一次，成功后把打印出的 batch_root 固定给两个终端。
batch_root="/home/zj/projects/my_project/outputs/v2_target_aggregation_$(date +%Y%m%d_%H%M%S_%N)"
printf 'batch_root=%s\n' "$batch_root"
# uav_target_diag prepare "$batch_root/prepared" --history-dir outputs/v2_critic_reliability_round1/20261026_171846_464722553
# 第1组：uav_target_diag fit "$batch_root/min" --prepared-dir "$batch_root/prepared/other" --aggregation min
# 第2组：uav_target_diag fit "$batch_root/mean" --prepared-dir "$batch_root/prepared/other" --aggregation mean
```

## 四组正式 SNN easy

现有正式 CLI 无奖励覆盖参数；easy 未传 rewards 时使用 RewardConfig 默认（P2/P3关闭），
不能仅给目标聚合参数就保证 P2/P3。
因此下述可执行命令用 Python API 明确传 P1关/P2/P3开，并显式传入正式编译组合。
不新增训练框架，不改学习率默认。每组从同一 SNN BC best 开始。
仅 easy、500k 日程，candidate 早停保留。关闭周期整池验证，阈值快照只保存不验证。
200k/300k/400k/450k 在跨阈值后的 episode 边界保存，结束模型走原终态保存；
早停缺失的快照不补齐。未通过正式门槛返回1并保存 failed，并不等于进程崩溃。

```bash
uav_formal_target() {
  cd /home/zj/projects/my_project || return 1
  local seed="$1" aggregation="$2" root start rc
  root="outputs/v2_formal_target_seed${seed}_${aggregation}_$(date +%Y%m%d_%H%M%S_%N)"
  mkdir -p "$root/review" || return 1
  start=$(date +%s)
  PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u - \
    --seed "$seed" --target-q-aggregation "$aggregation" --root "$root" <<'PY' 2>&1 | tee "$root/review/driver.log"
import argparse
from pathlib import Path
from brain_uav.config import RewardConfig
from brain_uav.models.v2_policy_config import V2SpikeRegularizationConfig
from brain_uav.scripts.train_v2_td3 import run_v2_td3_stage, default_v2_cuda_graph_compilation
from brain_uav.trainers.v2_validation import load_v2_validation_pool
from brain_uav.scripts.diagnose_v2_policy import file_sha256
p=argparse.ArgumentParser()
p.add_argument('--seed',type=int,required=True)
p.add_argument('--target-q-aggregation',choices=('min','mean'),required=True)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
bc=Path('outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt')
expected_bc_sha256='7b2b266fdf32cdb8b4eb220ef26a45008717a0af1538f8d478019f37ddc9adbc'
actual_bc_sha256=file_sha256(bc)
print(f'BC best SHA256 expected={expected_bc_sha256} actual={actual_bc_sha256} match={actual_bc_sha256==expected_bc_sha256}',flush=True)
if actual_bc_sha256 != expected_bc_sha256:
    raise ValueError('BC best SHA256 mismatch; refusing to start training')
pool=load_v2_validation_pool('outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json',expected_level='easy',expected_count=100)
assert pool.content_digest=='2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b'
result=run_v2_td3_stage(stage='easy',model='snn',snn_time_window=4,seed=a.seed,
    init_checkpoint=bc,
    validation_pool=Path('outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json'),
    output=a.root/'other/model.pt',metrics_out=a.root/'other/metrics.json',device='cuda',
    target_q_aggregation=a.target_q_aggregation,max_stage_steps=500000,
    actor_projection_dim=32,actor_hidden_dim=192,actor_lr=8.25e-4,critic_lr=1e-3,
    actor_rl_scale_alpha=2.5,gamma=.99,failure_sample_bias=2.,bc_schedule='smooth-mid-strong',
    rewards=RewardConfig(breakthrough_clearance_gate_enabled=False,
        descent_three_band_enabled=True,boundary_directional_warning_enabled=True),
    spike_regularization=V2SpikeRegularizationConfig(beta_max=1.,target_rate=.17,
        ramp_start_steps=25000,ramp_end_steps=75000),
    gpu_td3_compute=True,cpu_deduplication=True,periodic_snapshot_interval_steps=None,
    weight_snapshot_steps=(200000,300000,400000,450000),
    **default_v2_cuda_graph_compilation(model='snn',resolved_device='cuda'))
print(result,flush=True)
raise SystemExit(0 if result['passed'] else 1)
PY
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root/review/driver.log"
  if [ -f "$root/other/metrics.json" ]; then cp -- "$root/other/metrics.json" "$root/review/metrics.json" || return 1; fi
  for f in stage_start.json stage_end.json; do
    if [ -f "$root/other/metrics_reports/$f" ]; then cp -- "$root/other/metrics_reports/$f" "$root/review/$f" || return 1; fi
  done
  return "$rc"
}
# 第3–6组，审查后分别显式调用；不自动进入medium、不自动并发或重试。
# uav_formal_target 23 min
# uav_formal_target 23 mean
# uav_formal_target 11 min
# uav_formal_target 11 mean
```

同 seed 的 stage_start/metrics 中 `initial_state_sha256` 必须逐字段相等
（actor、两 online critics、三个 targets、两个 Adam）；不相等则拒绝解释聚合收益。
摘要不含算法模式，因此比较不会被预期的 min/mean 配置差异干扰。
运行中，等待同 seed 两组各自的 `other/metrics_reports/stage_start.json` 落盘后，
直接读取它们核对；`review/metrics.json` 此时尚不存在。
下面例子预期 seed23，参数依次为 min 组、mean 组，seed11 时将首个参数改为11：

```bash
PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u - 23 /absolute/min/other/metrics_reports/stage_start.json /absolute/mean/other/metrics_reports/stage_start.json <<'PY'
import json,sys
from pathlib import Path
seed=int(sys.argv[1])
a,b=[json.loads(Path(p).read_text(encoding='utf-8')) for p in sys.argv[2:]]
assert [x['formal_config']['seed'] for x in (a,b)]==[seed,seed], 'unexpected seed pair'
assert [x['formal_config']['target_q_aggregation'] for x in (a,b)]==['min','mean'], 'expected min/mean pair'
fields={'actor','critic1','critic2','actor_target','critic1_target','critic2_target','actor_optimizer','critic_optimizer'}
assert set(a['initial_state_sha256'])==set(b['initial_state_sha256'])==fields
for field in sorted(fields):
    assert a['initial_state_sha256'][field]==b['initial_state_sha256'][field], f'{field}: initial state mismatch; comparison invalid'
print('same-seed initial weights/targets/Adam verified')
PY
```

训练结束后才可改用两组 `review/metrics.json`（或原始 `other/metrics.json`）运行相同核对代码；
不得用终态 metrics 路径作为运行中核对的前提。
本地 BC best 已重新核实为上述 SHA256；正式命令在读取固定池和启动训练前再次校验，
并将 expected/actual/match 写入 driver.log。不匹配直接失败，不更换预期值。

## 训练后只评估（新模型及非终态快照）

同样20个场景，严格加载每个模型，冻结无探索策略，正常 reset/step 到真实结束。
记录 Q1/Q2、含首步奖励的实际剩余 G、逐场景及空/非空/terminal分组误差、goal率。
`transitions.jsonl` 是实际 Q/G 对，不用训练 Q 均值代替误差；不向训练回灌。
评估 manifest/evaluation 记录加载时 checkpoint/pool 哈希和实际读取的两份历史输入身份。
写成功结果前复核这四份文件；变化则 summary.execution_completed=false，
只保留输入身份及已有明细证据，不写成功 evaluation.json。
这里的新模型评估需要策略飞行，和 prepare 仅重放历史动作是两个不同任务。
只在训练后运行，不作为周期停训纠偏或新增早停门槛。

```bash
uav_target_evaluate() {
  cd /home/zj/projects/my_project || return 1
  local checkpoint="$1" root start rc
  root="outputs/v2_target_evaluate_$(date +%Y%m%d_%H%M%S_%N)"
  mkdir -p "$root/review" || return 1
  start=$(date +%s)
  PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u \
    -m brain_uav.scripts.diagnose_v2_target_aggregation --mode evaluate \
    --checkpoint "$checkpoint" \
    --history-dir outputs/v2_critic_reliability_round1/20261026_171846_464722553 \
    --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
    --device cuda --output-dir "$root/other" 2>&1 | tee "$root/review/driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root/review/driver.log"
  if [ -f "$root/other/summary.json" ]; then cp -- "$root/other/summary.json" "$root/review/summary.json" || return 1; fi
  return "$rc"
}
# uav_target_evaluate /absolute/path/to/model_or_snapshot.pt
```

大型共同数据、逐transition证据和曲线留 other；小型 manifest/summary/metrics/日志放 review。
本地仅 CPU/mock 检查，未运行20场重放、2000次真实拟合、策略评估或500k训练。
仍需服务器验证严格重放、CUDA/编译/Graph min/mean 一致性，以及四组实际训练和事后评估。
