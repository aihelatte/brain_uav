# 正式 easy 最后一轮学习率候选（本地复审完成，待服务器门控）

仅使用 `--experiment lr-final-20261113` 选择本轮四组，复用正式 easy 启动器。

| 模型 | seed | 方法 | actor_lr | critic_lr |
|---|---|---|---|---|
| ANN | 7、23 | min | 1.2e-4（原1.55e-4） | 2e-4 |
| SNN | 7、23 | ACC eta=.01、初始beta=.5 | 8.25e-4 | 2.5e-4（原1e-3） |

独立的 `scripts/v2_formal_easy_lr_final_config.json` 记录来源提交、原配置规范化哈希及from/to差异。先完整校验旧配置及historical，再应用两项显式差异；没有改写历史记录。网络、BC best、alpha=2.5、smooth-mid-strong、奖励、噪声、回放、500k预算、正式早停与100场固定池、私有MPS、编译、GPUCompute、CPU去重、精确椭球缓存、batch非阻塞及系数阻塞均继承基准。每组重新从BC best初始化，不恢复旧failed权重。

默认仍为baseline且plan-only，旧CLI数值不变。本轮须显式提供独立源码清单 `--source-manifest scripts/v2_formal_easy_lr_final_source.json`。它包含递归src全部Python文件、原启动shell、旧配置及本轮配置，采用CRLF→LF规范化，共113项。旧112项清单保持原样，在修改后的源码上严格报不匹配；精确复跑旧基准必须使用8d6d9f6的隔离checkout，不能重写旧清单或忽略文件。

experiment、实际resolved配置、输入及源码身份进入plan、manifest和每个worker的ready/门控证据。父进程派发、统一放行、worker、终态collect均使用同一配置选择；门控报告不能跨候选配置或跨源码使用。保留原门控容差、更新预算、beta/图重建检查、安全终止、SIGTERM清理及终态选优。终态100场逐场记录、episodes、windows、ACC controls、训练结果和四份原始权重照常保留。选优仍仅在本轮每模型两个seed之间比较；不自动判定调参成功。

## 独立部署（主聊天审查后执行）

服务器先从推送后的分支创建独立checkout，不更新原研究仓库。克隆后按主聊天交付的提交号执行 `git checkout --detach <提交号>`，将本轮代码固定到已审查版本。以下只准备命令，本轮未执行。

```bash
git clone --branch multi-nfz-training --single-branch https://github.com/aihelatte/brain_uav.git /home/zj/diagnostics/my_project_lr_final_20261113
```

本轮执行文件、独立配置及源码清单均通过Git部署，不再需要SCP覆盖文件。注意：8d6d9f6仅用于原基准精确复跑，不包含本轮学习率接线。

不能只凭checkout的HEAD判断同步；部署后检查实际源码113项哈希。若本地额外修改了其他src文件，必须审查并部署相应源码，不能在服务器重新生成清单绕过差异。

```bash
uav_lr_prepare() {
  cd /home/zj/diagnostics/my_project_lr_final_20261113 || return 1
  old=/home/zj/projects/my_project
  py=/home/zj/envs/uav_torch/bin/python
  parent=/home/zj/diagnostics/outputs
  cmd=("$py" -u -B -m brain_uav.scripts.run_v2_formal_easy_baseline)
  args=(--experiment lr-final-20261113
    --source-manifest scripts/v2_formal_easy_lr_final_source.json
    --ann-bc "$old/outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt"
    --snn-bc "$old/outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
    --validation-pool "$old/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json")
  PYTHONPATH=src "$py" -u -B -c 'import json; from brain_uav.scripts import run_v2_formal_easy_baseline as r; assert json.loads(r.LR_SOURCE_MANIFEST.read_text())["sha256"] == r.identity(r.LR_EXPERIMENT)["sha256"]; print("actual_source_hashes=matched")' || return 1
  PYTHONPATH=src "${cmd[@]}" plan --root "$parent/lr_final_plan_only" "${args[@]}"
}
uav_lr_run() {
  local root="$1" start rc
  shift
  mkdir -p "$parent" || return 1
  start=$(date +%s)
  PYTHONPATH=src "${cmd[@]}" launch --root "$root" "${args[@]}" "$@" --launch 2>&1 | tee "$root.driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\nroot=%s\n' "$rc" "$(( $(date +%s)-start ))" "$root" | tee -a "$root.driver.log"
  return "$rc"
}
uav_lr_prepare
```

## 有限CUDA门控、四组训练

准备成功后，先运行本轮独立门控。与原基准相同，四个MPS worker、每worker八对受控更新、物理更新预算128、放行后300秒截止。失败不训练、不使用旧报告。

```bash
gate="$parent/v2_lr_final_gate_$(date +%Y%m%d_%H%M%S_%N)"
uav_lr_run "$gate" --gate-only --gate-timeout 300
```

确认本次四组门控报告全部通过后，明确绑定该目录启动四组；输入、实际源码、模型/seed/PID、退出码、报告哈希及候选参数仍由入口强校验。

```bash
root="$parent/v2_lr_final_$(date +%Y%m%d_%H%M%S_%N)"
uav_lr_run "$root" --gate-report "$gate"
```

## 状态与收集

其他终端运行相同prepare函数后，将root设置为实际批次绝对路径。不要创建新批次来查询旧状态。

```bash
PYTHONPATH=src "${cmd[@]}" status --root "$root" "${args[@]}"
tail -n 20 "$root"/*_seed*.log
```

父进程正常结束会自动collect；如需手动重新核对并收集：

```bash
PYTHONPATH=src "${cmd[@]}" collect --root "$root" "${args[@]}" --launch
```

本地验证命令（PowerShell设置PYTHONPATH=src;tests）：

```text
python -m unittest test_v2_formal_easy_lr_final test_v2_formal_easy_review test_v2_formal_easy_baseline -q
```

CPU测试用小型内存BC初始化和真实builder/Adam验证学习率，减少的只是夹具replay存储；没有执行真实BC飞行或更新。CUDA门控、MPS接入、实际并发及最终学习效果尚待服务器验收。

本轮相关八个模块共139项：134通过、1项只读BC/池夹具缺失跳过、4项既有错误（1个旧编译mock字段缺失、3个旧场景夹具布尔元数据错误）。未扩大修复。上方三个专项模块共30项通过。
