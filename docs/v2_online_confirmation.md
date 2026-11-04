# 在线 WD3/ACC 第三阶段：独立验收（待审查）

入口：`brain_uav.scripts.diagnose_v2_online_confirmation`。原研究入口的
`confirmation` 仍仅输出计划。新入口默认 plan-only，不读取输入、不加载模型、
不初始化 CUDA、不创建目录；显式 `--launch` 才运行。无训练、优化器更新或新 seed。
执行完成只表示资料完整，不代表算法通过；统一算法结论仍须等待既定实验完成。

## 锁定与隔离

读取 `confirmation_candidates.json` 的六个最终400k候选，不搜索历史最佳时刻。
逐个核对原始字节 SHA256、模型、arm、requested/actual=400000，并调用原研究
快照严格校验器。完成性审计中对应六项也必须完整到400k。

历史100场池始终用于严格加载 checkpoint，其返回对象的 `content_digest` 必须
等于原研究 DIGEST。独立40场池仅作为 reset/step 执行场景，独立保存文件哈希、
digest和物理指纹；不改 checkpoint 的历史池身份，不调用20场选择器挑独立场景。

排除读取两批 manifest 的全部16/4组，以及每组完整的 append-only
`other/training_scenes.jsonl`，包含失败ACC、全部续跑阶段和未完成episode的场景。
核对每行物理指纹、从0开始且严格递增的 start_step，并与整份
`other/metrics_reports/episodes.jsonl` 的episode结束步逐项对应（正常终止时可没有最后一次reset）。记录各来源路径、原始哈希、
条数、唯一数及覆盖起止。该覆盖证明依赖研究入口原有的每次reset日志契约；
不能将日志最后一个start_step称为精确结束或失败更新步。
旧离线输入使用指定轮次的 `ann/snn_collect/other/manifest.json` 与
`ann/snn_prepare/other/manifest.json`，交叉核对30 train/10 calibration的payload、
split及scene_key。缺文件、缺组或不匹配立即阻断，不默默跳过。

先保存输入身份、候选及全部排除来源，再以selection seed=20261102的easy派生
随机流顺序取首8个空场、首32个非空唯一物理场景。最多100000次场景提议；
选择后检查五类几何覆盖，不满足就失败，不替换场景。仅按几何/排除集选择，
不依据飞行、Q、G。场景生成只在显式启动后进行。

## 飞行与评分

六候选共享40场，冻结eval actor和双critic，无探索；沿用SNN forward原有复位。
最多240条自然飞行。每条保存完整动作、奖励、N+1状态、pre_action_hashes及终态，
随后线性重放同一记录、严格核对hash/状态/奖励/终态，倒算包含当前奖励的G，
不再飞一条轨迹。保留实际终点和原奖励/折扣/时间限制。

Q1/Q2分别报告Q−G偏差、MAE、RMSE、正误差尾部、逐场配对信息；
复用既有场景等权/transition加权、空/非空、五几何、结局、terminal、失败末5步
分组，空分组保留样本不足。goal率、各结局、长度、回报、G0逐场保存。
重加权概率从source正式默认 `V2ScenarioGeneratorConfig` 读取并记录其来源，
不是8/40；这是分层描述性重加权，不是总体推断或通过门槛。

探针使用原开发20场按sequence顺序的前12场，每模型独立BC自然轨迹
`floor((T-1)/2)`前缀；同模型三候选共享全部前缀和hash。
`--ann-anchors` / `--snn-anchors` 可读取本入口封存的完整 `anchors.json`；严格核对
BC actor身份、历史池、场景顺序、执行后端及逐步重放/动作归属，不放宽容差。
没有可复用记录时最多补采两模型各12条BC轨迹，不采候选开发集自然评估。
目前未发现并宣称可复用的实际BC锚点；启动前应核对现有资料，不能将候选轨迹
冒充BC轨迹。不同模型的锚点可能不同。

公共候选是BC及每维±0.2倍action_limit扰动，clip后精确float32去重，最多5项；
own映射到已有项或第6项。每个候选的12份动作计划先落盘再执行分支。
公共排名以BC为参考；包含own的排名以own为参考，两套索引独立处理。
复用既有ΔQ并列容差和原始argmax regret，零有效非并列对记信息不足。
保存原始Q、G、Q−G、两参考ΔQ及来源映射。不宣称完整六项跨模型/方案相同。
每个分支reset后严格重放前缀，再执行候选首动作与该快照自身actor续飞，
保留已消耗的时间，总分支≤432；G包含候选首步奖励。

## 产物、复用与限制

`review/`：首次manifest、scene_plan、probe_plan、sealed_inputs、最终summary。
`other/`：物理排除指纹、新池、BC完整锚点、240场完整记录及逐步Q/G、有限分支记录。
冻结前后核对actor/双critic状态；结束复核checkpoint、BC、两池、全部历史输入和
CRLF→LF规范化源码哈希。历史源码身份与本轮诊断身份分别记录。

同输出目录仅接受完整成功结果的显式再校验复用：请求计划一致、全部输入/源码
及产物哈希不变、六候选40场/12探针完整。首次执行也使用同一只读完整性检查，
全部输入复核、必需产物核对及最终哈希封存成功后，才设置
`execution_completed=true`；封存读取失败仍保存false与实际错误并向上抛出。

复用不能只相信summary的计数或自行列出的artifacts：从请求、候选锁、审计、两批
20组历史输入和离线排除来源重新推导必需输入，核对BC及源码身份；核对40场池与
scene_plan、sealed_inputs、共同probe_plan和两模型完整anchors.json。按真实场景ID
逐项检查每候选40条完整自然飞行、逐步Q/G和逐场统计；按探针索引、float32去重
动作索引核对12份动作计划、12项探针结果及计划要求的全部分支文件。
分支数由实际去重计划确定，可能少于72，不以summary的自报数量代替。
公共动作与同模型共享状态也逐项对应。完整BC锚点允许由已有合法输入复用，
不额外要求输出目录含有单场BC episode文件。必需文件必须全部列入哈希清单，
实际文件集合及哈希也必须一致；空清单、空占位、重复或错配ID均拒绝。
该复用检查只读取JSON、池和文件哈希，不加载模型、不生成场景、不飞行。

分支元数据中，`remaining_initial_steps=max_steps-len(prefix)`；另用
`consumed_initial_steps=len(prefix)`记录已消耗步数。仅修正记录，不更改环境时间限制。

不会覆盖、自动补飞、重试或修复不完整目录。
失败summary保留 `execution_completed=false` 和实际异常。无formal passed模型输出。
不连接服务器、不操作原研究目录或暂停进程/继续令牌。

## 独立服务器命令草案（本地未执行）

先在独立代码目录部署审查后的源码及源码清单；六候选锁文件与完成性审计作为
只读输入放在该目录的inputs下，不改变内容。锁文件选择 `server_checkpoint` 字段。
确认以下旧离线轮次路径在服务器实际存在，缺失即停止，不生成替代资料。

```bash
uav_confirmation() {
  local code=/home/zj/diagnostics/my_project_confirmation_20261104
  local old=/home/zj/projects/my_project
  local py=/home/zj/envs/uav_torch/bin/python
  local out=/home/zj/diagnostics/outputs/v2_confirmation_$(date +%Y%m%d_%H%M%S_%N)
  local start rc
  cd "$code" || return 1
  local cmd=("$py" -u -B -m brain_uav.scripts.diagnose_v2_online_confirmation
    --candidate-lock "$code/inputs/confirmation_candidates.json"
    --completion-audit "$code/inputs/completion_integrity_audit.json"
    --checkpoint-path-field server_checkpoint
    --historical-repo "$old"
    --original-batch "$old/outputs/v2_online_wd3_acc_20261102_110301_966812111"
    --acc-batch /home/zj/diagnostics/outputs/v2_acc_four_20261103_163517_910401615
    --offline-root "$old/outputs/v2_wd3_acc_round1/20261101_184801_765940445"
    --validation-pool "$old/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json"
    --ann-bc "$old/outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt"
    --snn-bc "$old/outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
    --source-manifest "$code/scripts/v2_online_wd3_acc_source.json"
    --device cuda --output-dir "$out")
  if [ "${1:-}" != --launch ]; then
    PYTHONPATH=src "${cmd[@]}"
    return $?
  fi
  mkdir -p "$(dirname "$out")" || return 1
  start=$(date +%s)
  PYTHONPATH=src "${cmd[@]}" --launch 2>&1 | tee "$out.driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$out.driver.log"
  return "$rc"
}
uav_confirmation
# 审查及授权后才运行：uav_confirmation --launch
```

本地检查只包含CPU/mock与两步合成环境，不代表真实候选、240场、432分支或CUDA
验收通过。没有启动真实生成、采集、训练或服务器任务。方法结论仍待独立验收。

本轮实际本地回归（PowerShell先设置 `$env:PYTHONPATH='src;tests'`）：
`python -m unittest test_diagnose_v2_online_confirmation test_diagnose_v2_critic test_diagnose_v2_wd3_acc -q`
共44项通过；同次运行 `test_v2_online_wd3_acc.TestOnlineTargets` 中的
`test_snapshot_explicit_diagnostic_load_and_tamper_rejection` 和
`test_acc_snapshot_bank_returns_and_chronology`，2项通过，总计46项（原42项加4项专项）。
专项包含19项：默认无副作用、锁文件篡改、场景隔离/覆盖、两池身份、
完整20组排除来源与reset/episode历史对应、动作去重和双参考排名、真实两步合成
环境采集/严格重放、BC锚点复用、候选首步及剩余时间、冻结/非有限、完整产物复用
及失败summary；原排名与离线WD3/ACC测试25项保留通过。
新增/加强回归包括：真实execute最终哈希OSError后summary仍为false；空壳、
空清单、缺必需输入或文件、重封存的重复/错配/占位记录均拒绝；合法完整复用
无模型加载或飞行且不写文件；max_steps=100、prefix=10时剩余元数据为90。
