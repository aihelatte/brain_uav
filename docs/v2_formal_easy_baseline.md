# UAV V2 正式 easy 基准（本地复审完成，待服务器门控）

四个独立 worker 从各自 BC best 初始化。ANN seed7、23 使用 min，SNN seed7、23 使用 ACC（η=.01）。统一使用 easy100 固定池、500k 日程与正式早停。至少125k步后，连续4个互不重叠的15回合窗口各最多1次失败，随后固定池最多5次失败。终态按正式早停或500k预算产生。

完整配置及逐字段来源见 `scripts/v2_formal_easy_baseline_config.json` 的两个 `historical` 记录和四组 `resolved` 记录。历史 ANN/SNN 的场景与奖励配置逐字段相同，分别按原 `stage_start.json` 字节哈希核验。新运行沿用对应模型学习率、完整奖励与场景配置、BC 架构、500k噪声与BC日程。差异是新seed、正式早停与500k上限、取消研究暂停、batch非阻塞搬运开启。保留精确椭球缓存、CPU去重和正式编译。BC 哈希与模型、encoder元数据严格校验。准备后另核对 encoder输出128、投影32、actor宽192、ANN双critic宽128及SNN窗口4、tau2。

ACC只接收自然终止的训练回合及环境实际裁剪动作。原控制器倒算G，并按原完整回合bank、25k门槛及1k最小间隔更新beta。评分副本在编译前建立，每次评分前严格同步权重。正式及周期产物保存ACC合同、beta、控制记录及bank，按正式格式严格读取。旧研究快照仍由研究读取器处理。`load_state` 用于显式恢复诊断状态，正式产物没有可精确续训的环境、完整replay与RNG状态。

四进程共享私有MPS。正式launch在创建Session或CUDA子进程前强制校验`--gate-report`。四份报告分别绑定本批模型、seed、作业、实际PID、退出状态、配置、输入和源码哈希，并核对父进程封存的报告文件哈希。互换文件、缺报告或身份变化均拒绝。正式manifest保存使用的门控证据。

通过独立CUDA Graph能力和安全终止探针后，全部worker完成编译、预热、同步，父进程核对实际PID、派生种子和服务端完整客户端集合再放行。worker和两类能力探针各自创建Unix session，前台Ctrl-C由父进程进入清理。启动器运行期间安装SIGTERM处理器，收到TERM转为带原因的非零受控退出，进入同一清理流程，结束后恢复原处理器；重复TERM不打断正在进行的清理。异常清理复用Session，MPS明确确认安全终止后才发送OS信号。abort/process报告写入失败也继续安全终止及Session.finish，原始异常优先保留。未知响应或外来客户端留下阻断证据。

终态另用同一100场无探索验证，所有候选的正式通过状态来自正式控制器。collect先逐项核对全部四组的作业身份、退出码、BC、配置、场景、奖励、池、权重哈希、实际步数和停止原因；worker_summary的batch_id必须与manifest一致，worker_summary/process/ready三份PID必须为同一正整数，ready作业身份也必须一致。权重限定为对应worker目录中的model.pt/model_failed.pt。全部通过后在临时目录准备ANN/SNN基准并一次发布selected目录。重复collect先重新校验，再返回同一有效结果，既有成功summary及已发布文件在后续身份错误时保留。按正式通过、到达数、平均回报、较少训练步数、较小seed依次选优。两组均未通过时标注`best_available`，缺记录或异常保留未完成与原因。

有限CUDA门控使用同样四个MPS进程及正式配置，每worker八对受控eager/GPUCompute更新，公开更新与捕获中的计算合计上限128/worker。两侧初始化同一batch mask证据，GPUCompute实际记录仍覆盖后严格核对。单场bank观测先按mask去除填充区，控制器标量转换为张量证据。真实调用隔离评分时核对副本独立存储及训练critic、targets、Adam、梯度和RNG前后身份。

门控更新序列保持1、3、4、25001、25002、26002、26003、26004。beta刷新发生在25001、26003，两步分别重用critic-only的bank 1和bank 0；26002首次捕获actor的(1,1)组合，不要求它命中缓存。beta改变后要求executor相同、capture数不增、replay数增加，并在同次target Q上区分新旧beta标签。26003更新完成后加载checkpoint，先核对beta和完整controller，再在26004执行恢复值对应更新并验证Graph重建。正式预热独立记录。沿用`rtol=1e-4, atol=1e-5`，无法区分或严格不匹配即阻断。`--gate-timeout`限定放行后的共同截止时间，默认300秒，卡在同步或Graph执行时使用相同MPS安全清理。正式长训练由训练预算控制。

以下命令分别在服务器已有终端中执行。推送后从远端克隆独立代码目录，原训练仓库仅作只读输入来源。精确部署时使用主聊天给出的提交号执行 `git checkout --detach <提交号>`，不要切换原训练仓库。

```bash
git clone --branch multi-nfz-training --single-branch https://github.com/aihelatte/brain_uav.git /home/zj/diagnostics/my_project_formal_easy_20261111
```

提交包含已采用的椭球缓存默认值、正式接线、门控、配置清单和测试，不再需要从本地覆盖上传源码。

在服务器现有会话中设置参数与记录函数。默认调用只输出完整计划。启动时由入口核验当前实际源码、历史 stage_start、BC 与池的文件哈希，以及池的内容 digest。

```bash
uav_easy_prepare() {
  cd /home/zj/diagnostics/my_project_formal_easy_20261111 || return 1
  old=/home/zj/projects/my_project
  py=/home/zj/envs/uav_torch/bin/python
  parent=/home/zj/diagnostics/outputs
  cmd=("$py" -u -B -m brain_uav.scripts.run_v2_formal_easy_baseline)
  args=(--ann-bc "$old/outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt"
    --snn-bc "$old/outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
    --validation-pool "$old/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json")
  PYTHONPATH=src "${cmd[@]}" plan --root "$parent/plan_only" "${args[@]}"
}
uav_easy_run() {
  local root="$1" start rc
  shift
  mkdir -p "$parent" || return 1
  start=$(date +%s)
  PYTHONPATH=src "${cmd[@]}" launch --root "$root" "${args[@]}" "$@" --launch 2>&1 | tee "$root.driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\nroot=%s\n' "$rc" "$(( $(date +%s)-start ))" "$root" | tee -a "$root.driver.log"
  return "$rc"
}
uav_easy_prepare
```

本地复审完成后，先执行有限门控。确认四份 `cuda_gate.json` 和 `batch_summary.json` 后，使用实际 `gate` 路径启动正式训练。

```bash
gate="$parent/v2_formal_easy_gate_$(date +%Y%m%d_%H%M%S_%N)"
uav_easy_run "$gate" --gate-only --gate-timeout 300
```

```bash
root="$parent/v2_formal_easy_$(date +%Y%m%d_%H%M%S_%N)"
uav_easy_run "$root" --gate-report "$gate"
```

在另一个已有会话中可用同一准备函数及实际 `root` 查询。正常结束由父进程自动收集，手动collect会重新校验已有完整结果。

```bash
PYTHONPATH=src "${cmd[@]}" status --root "$root" "${args[@]}"
tail -n 20 "$root"/*_seed*.log
```

```bash
PYTHONPATH=src "${cmd[@]}" collect --root "$root" "${args[@]}" --launch
```

本轮审查回归的实际命令如下，PowerShell先设置`$env:PYTHONPATH='src;tests'`。

```text
python -m unittest test_v2_formal_easy_review test_v2_formal_easy_baseline test_v2_formal_training test_train_v2_td3 test_v2_target_q_aggregation test_v2_online_wd3_acc test_v2_mps_diagnostic -q
```

最新复审集合实际运行131项：126通过、1跳过、4报错。专项两个模块共22项通过。新增回归先复现26002 actor新bank被误当缓存、SIGTERM跳过清理及错批次/PID仍可collect，再修复。图键测试调用真实GPUCompute.key并跟踪published bank及加载后的重建；SIGTERM测试在独立CPU进程发送信号（Unix用os.kill；Windows用signal.raise_signal，避免Windows的os.kill直接强杀），验证清理及处理器恢复。4项既有错误分别为旧编译mock缺少gpu_td3_update，以及3项场景夹具缺少布尔requested_direct_path_blocker。只读BC/池夹具缺失的1项继续跳过。
推送前另运行环境模块20项：18通过、2项旧场景夹具错误。两项错误均在原HEAD测试中复现，原因是零障碍概率为1而直线路径阻挡概率仍为.5；本轮没有修改场景生成器或扩大修复。
ACC加载先纯校验目标engine算法、网络、BC、Adam和计数，再挂载hook或加载。失败的兼容性检查保持原算法、controller和参数可继续使用。
四组正式运行、MPS接入和有界CUDA检查待服务器验收。
