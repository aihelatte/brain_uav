# seed7 SNN 最终冻结评估（待审，不启动）

独立入口 `brain_uav.scripts.diagnose_v2_seed7_confirmation`，唯一契约是
seed7/easy/SNN的min、fixed_own、ACC(.01)最终requested/actual400000。
不改旧六候选入口、训练算法或共享严格checkpoint加载器。
默认仅输出计划，不读输入、不建目录、不加载模型、不初始化CUDA、不生成场景。

## 身份与预算

候选锁逐份核对原始SHA256、模型、seed、arm、步数，与batch manifest、job summary、
progress、process和400k sidecar交叉核对；实际启动前还严格校验研究快照元数据。
原100场池只用于严格重建checkpoint；40场池只供冻结执行，原始SHA256固定为
`f64082daec0e15cd02e3604b66bf23ecf633a8a6fc12142b1ad67e1dcac8c93e`，
返回对象digest固定为`27a6e7066272e451cf9483c73ef8bfa6358ecefa583020291f5dea47fac86316`。
只复制该封存文件到新输出，不调用任何场景生成器或选择器重选40场。

核对原confirmation成功摘要、全部已封存artifact哈希、20组在线与两份离线
排除覆盖证据、物理排除指纹，再读seed7三组完整reset日志及episode结束历史。
`episodes.jsonl`的结束步数必须与已核对summary中的episodes逐项相等，
严格递增且不超过400000，拒绝两个日志一起截短。最后完成episode早于400000时，
必须还有该episode结束位置开始的未完成尾段reset；只有恰在400000结束才允许省略。
指纹、覆盖不完整、与40场有交集均阻断；40场必须唯一、8空32非空、五类几何。
历史源码身份来自历史manifest；当前源码独立记录，不能冒称完全相同。

最多120条候选自然飞行；同一记录严格线性重放、倒算含当前奖励G，不额外再飞。
复用原Q1/Q2均值/偏差/MAE/RMSE/高估尾部、逐场、分层、terminal/失败末5步统计；
goal、长度、回报和G0保留。空场概率从正式源配置读取，重加权仅是描述性统计。
冻结CUDA eager、无探索，SNN保留原forward复位，前后核对网络与全部输入/源码。

开发集前12场、BC自然轨迹中点规则不变；BC准备显式seed7。
可传 `--snn-anchors /absolute/anchors.json`，仅身份、后端、完整重放及环境/RNG/hash
一致时复用。旧seed23环境RNG可能不一致，失败即停止，不放宽、不自动补飞。
没有可复用输入时最多补采12条seed7 BC自然轨迹，独立计入预算。
三方案共享12个状态与公共动作，BC及每维±.2限幅扰动精确float32去重最多5项，
own最多第6项。三候选全部12状态、共36份动作计划先封存，再执行任何分支，最多216分支。
每分支严格重放前缀、候选首动作后由该快照自己的actor续飞；不重置时间预算。
remaining=max_steps-prefix长度，consumed=prefix长度；公共排名参考BC，含own参考own。
沿用ΔQ并列容差、Q1/Q2各自排名、原始argmax regret；零有效pair信息不足。

最终调用旧入口共用的只读完整性检查，按真实ID/索引核对120条记录、评分、
12共同状态/动作计划、实际去重分支及哈希清单；输入复核和封存全部成功才完成。
失败summary为false并向上报错，无覆盖、重试、自动补飞。已有目录只读复用完整
成功bundle，不加载模型或飞行。review放小型manifest/计划/summary，other放完整记录。
driver日志留在封存目录外，不能运行结束后往其中加入未封存文件。
`diagnostic_only=true`、`formal_passed=false`；不自动给方法结论、不生成课程前驱。

## 独立部署草案（均未执行）

先在服务器创建独立目录；不更新原训练仓库，不碰旧PID/tmux/令牌：

```bash
git clone --no-hardlinks /home/zj/projects/my_project /home/zj/diagnostics/my_project_seed7_eval_20261106 &&
mkdir -p /home/zj/diagnostics/my_project_seed7_eval_20261106/inputs
```

本机PowerShell上传审查后实际源码（HOST替换为已有SSH目标，不依赖tar包）：

```powershell
$dst='HOST:/home/zj/diagnostics/my_project_seed7_eval_20261106/'
scp -r E:/wurenji/my_project/src E:/wurenji/my_project/scripts $dst
scp E:/wurenji/analysis/seed7_400k_20261106/confirmation_candidates_seed7.json ($dst+'inputs/')
```

服务器独立目录核对107项以上的实际清单（新增本入口后108项，CRLF→LF；
输入文件仍原始字节哈希），不单凭HEAD判断同步：

```bash
cd /home/zj/diagnostics/my_project_seed7_eval_20261106 &&
/home/zj/envs/uav_torch/bin/python -c 'import json,hashlib,pathlib; m=json.loads(pathlib.Path("scripts/v2_online_wd3_acc_source.json").read_text()); assert all(hashlib.sha256(pathlib.Path(p).read_bytes().replace(b"\r\n",b"\n")).hexdigest()==h for p,h in m["sha256"].items()); print("source hashes OK",len(m["sha256"]))'
```

准备参数（显式只读绝对输入），默认只输出计划；此段不启动飞行：

```bash
uav_seed7_prepare() {
cd /home/zj/diagnostics/my_project_seed7_eval_20261106 || return 1
old=/home/zj/projects/my_project
out=/home/zj/diagnostics/outputs/v2_seed7_eval_$(date +%Y%m%d_%H%M%S_%N)
cmd=(/home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.diagnose_v2_seed7_confirmation
 --candidate-lock "$PWD/inputs/confirmation_candidates_seed7.json" --checkpoint-path-field server_checkpoint
 --batch /home/zj/diagnostics/outputs/v2_seed7_20261105_124928_019089587
 --sealed-confirmation /home/zj/diagnostics/outputs/v2_confirmation_20261104_204055_826603562
 --validation-pool "$old/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json"
 --snn-bc "$old/outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
 --historical-repo "$old" --source-manifest "$PWD/scripts/v2_online_wd3_acc_source.json" --output-dir "$out")
PYTHONPATH=src "${cmd[@]}"
}
uav_seed7_prepare
```

仅审查和授权后在上述同一shell显式启动；日志在输出目录旁边，不修改封存目录：

```bash
uav_seed7_eval() {
  local start rc
  mkdir -p "$(dirname "$out")" || return 1
  start=$(date +%s)
  PYTHONPATH=src "${cmd[@]}" --launch 2>&1 | tee "$out.driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$out.driver.log"
  return "$rc"
}
# 授权后：uav_seed7_eval
```

本地CPU/mock与极小合成环境检查不能称为120场、216分支或CUDA验收通过。
本轮不执行真实模型评估，不重新训练或生成任何场景。

本轮回归命令（PowerShell先设置 `$env:PYTHONPATH='src;tests'`）：
`python -m unittest test_diagnose_v2_seed7_confirmation test_diagnose_v2_online_confirmation test_diagnose_v2_critic test_diagnose_v2_wd3_acc test_v2_online_wd3_acc.TestOnlineTargets.test_snapshot_explicit_diagnostic_load_and_tamper_rejection test_v2_online_wd3_acc.TestOnlineTargets.test_acc_snapshot_bank_returns_and_chronology -q`
专项8项，旧相关回归45项，共53项；只使用tiny CPU/mock及两步合成环境，
实际候选SHA256及40场raw hash/digest另作只读核对。
局部修复回归同时覆盖缺尾段reset、两个日志同时截短、超400000拒绝、
恰好400000无后续reset通过，以及首条分支之前36份计划均已封存。
