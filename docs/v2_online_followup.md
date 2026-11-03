# 在线 WD3/ACC 有界后续诊断

本入口独立于研究训练、暂停/继续令牌、旧固定 actor 诊断。默认只打印计划；仅 `--launch`
执行计算。不得向原服务器仓库 git pull，也不得修改原实验或12个暂停进程读取的文件。
把审查后的源码上传到**独立代码目录**，输入均以绝对路径只读指向原实验/BC/固定池；
输出必须在原仓库和原实验之外。工具会拒绝在 `--historical-repo` 原目录运行。

## 事实与假设

历史实验 `v2_online_wd3_acc_20261102_110301_966812111` 的4个ACC进程退出为1，报告illegal
memory access。最后完整episode步数是182688/117971/30795/34904，不是精确失败更新步数。
`output.cpu().tolist()`是异步错误的暴露点，不能证明是根因。
12个非自适应臂停在200k附近。SNN WD3(.15)训练近20场19次成功、冻结开发评估0/20，
两者场景/探索条件不同，不能直接认定算法退化或执行路径有错。

本地只读核对五个SNN requested=200000快照的actual分别为：
min=200449，mean=200101，fixed_own=200266，wd3_045=200648，wd3_015=200185。
ANN ACC001保存75007/150170，ACC010保存75738；SNN ACC没有快照。
这些事实不等于已定位CUDA根因；没有完整replay/RNG/环境，不能精确续训。

`predict_bank`真实调用live critic.forward；critic1可能进入独立编译的critic_guidance encoder，
GPUCompute使用手工Graph/raw计算。交替执行是否造成非法访问仍是**待验证假设**。

## snn-inference

模式时序与正式入口一致：严格BC初始化/build/checkpoint load → 原编译及预热 →
`actor.train()` → 动作或更新；在首次GPUCompute捕获之前完成，不在捕获后切换模式。
双在线critic保持train并检查；targets和BC reference保持eval及冻结，不为隔离评分切换live模式。
`compilation.model_modes`分别记录初始化后、预热后、动作/更新前七个模型的training/frozen状态；
`pre_update_*.json`记录对应更新前模式。冻结评估actor仍独立eval，逐次真实前向记录
training、inference、grad上下文，不把两条路径统一成相同模式。

服务器旧snn-inference通过仅覆盖当时实际执行的模式，不能外推在线actor.train()情形；
保留旧产物。本轮旧acc-bank首个ann/original被“GPU compute requires training online models”
初始化保护拒绝，尚未完成受控更新，不是历史CUDA非法访问复现。
修正后仍需补验SNN动作路径及未完成的ACC诊断，不重跑16组或500k训练。

通过review sidecar的requested/actual/hash查找唯一快照，再严格校验研究格式和加载模型；
不按文件名前缀任选。检查指定固定池加载对象的content_digest，保持原奖励/架构/环境校验。
读取min的200k冻结评估manifest和20条完整episodes.jsonl，核对checkpoint和pool身份。
从预先排序的前2个空场、前9个非空场各取起始/中段/末动作前状态，固定前32个状态。
重放使用原reset/step和严格线性replay helper，核对pre_action_hash、N+1状态、奖励及终态；
不赋值env.state，不根据动作/Q/outcome选状态。所有快照复用同批观测。

训练侧严格构造engine、恢复研究权重/targets/Adam、启用原正式编译组合，真实调用
`select_action(exploration_noise=0,cpu_deduplication=True)`。冻结侧独立严格加载模型并调用
原`actor_action`（eager/eval/inference_mode），无重复eager reference。
每侧独立参数存储，观察真实返回张量，保留raw/clipped动作、逐位比较和rtol=1e-4/atol=1e-5。
不统一两侧training状态或改SNN reset语义以掩盖差异。记录上下文及编译配置。
同次证据收集包含额外只读回读/状态审计，动作墙钟明确带插桩，不作为正常推理性能。

每模型32次双侧比较，加A-A-B-A双侧8次，5模型共360次受控动作调用（上限500）。
编译预热另记耗时及原helper提供的shape/Graph证据；旁观计数host动作调用和Graph探针提交，
并记录原helper每个warmup batch的一次raw actor调用。编译器内部重复追踪次数仍标unknown，
不将host计数冒充设备kernel次数；预热不计入360次。保留已有预热验证同步。
冻结eager侧通过原head返回值观察同次前向两层脉冲；训练compiled callable未暴露脉冲，
明确missing，不做额外前向。`strict_passed`限于动作值及重复状态测试，不代表脉冲等价。

`--closed-loop`是独立显式选项：仅min和WD3(.15)，每个模型一空一非空，两路径各自然飞行，
最多8条episode。记录raw/clipped动作、首个动作/状态差异、outcome、步数、回报和墙钟；
冻结侧保存同次脉冲hash/计数，训练侧缺失，双侧脉冲差异未知。
保持环境自然terminated/truncated，不能截成伪完整轨迹。outcome不同会严格报告失败，
即使单步动作曾在容差内。所有结论仅限本次状态和场景覆盖。

## acc-bank

父进程串行启动ANN original/isolated/beta、SNN original/isolated/beta六个独立CUDA子进程。
不使用tmux、MPS或原暂停进程，不改变CPU线程。主复现默认四进程各6次，共24次；
两个独立beta进程各1次，默认合计26次公共更新。主复现可显式指定3..200次/worker，
但每主进程保守物理上限200，beta进程各100，总上限4×200+2×100=1000。
公共update/replay（包括任何会改变状态的预热）和冷捕获compute调用均提前预留计数；
达到任一上限即失败，不自动增预算。公共更新内的冷compute重复计数属于保守上界。
正式compile预热不得改变更新计数；若改变立即阻断。编译/前向预热、bank评分/状态审计、
update墙钟、捕获/Graph replay统计分开保存。这些诊断墙钟不是纯GPU时间或性能验收。
评分时同时记录Dynamo新增图计数；若首次评分含延迟编译，该编译耗时包含在score墙钟内，
单独耗时标unknown，不伪造可精确分离的时间或把差值解释成GPU等待。

两模型都从同一份合法ANN ACC001 requested75000 bank读取完整episodes，长度：
258/1000/1000/125/417/132/1000/1000。ANN用其自身快照网络/targets/Adam；SNN用严格
加载的min requested200000网络/targets/Adam，把研究目标配置显式干预为ACC001。
ANN数据仅作为SNN的**独立可追溯诊断夹具**，绝非SNN历史训练bank。比较原/候选时，
同模型初始状态摘要、batch/noise hash必须一致。只覆盖eta=.01；不代表ACC010验收。

控制器双方均重新初始化beta=.5；不恢复历史控制过程。依次追加完整历史episode，
通过真实`OnlineResearch.episode -> complete_episode -> predict_bank -> adjust`调用链。
容量/整条淘汰/done/G规则保持原样；绝不切轨迹或改done/G凑长度。
阶段计数明确为合成值：第一次为冻结前奇数步、第二次25000、以后至少增加1000且覆盖
奇/偶更新。完整episode映射到不重叠的合成时间段，**不是重建某次失败现场**。
每次评分/控制后真实GPUCompute更新，再重复评分与更新；覆盖critic_only/frozen_targets/actor。
bank有多种大小和尾batch才算诊断覆盖有效。

**独立beta生效证据**：主复现不启用tensor recorder，Graph、观察顺序保持原样。
另外的beta进程使用同一来源网络和完整首条bank episode、重新初始化的eta=.01控制器，
在明确标为合成的25000步调用真实bank/controller后，以同次GPUCompute recorder记录
`target.q1/q2/td`；不额外前向。记录bank文件/episode、batch/noise hash、控制记录、
调整前后beta、reward/done/gamma、两支实际TD目标及按新旧beta分别复算的目标。
复用原`research_target_pair`公式，分别核对两支critic的新目标，容差仍为rtol=1e-4/atol=1e-5。
两支均须匹配；同次Q下至少一支新旧目标须可区分（取min的一支天然不随beta变化）。
beta未变化或新旧目标不可区分时明确为`information_insufficient`，不计通过，不改G凑结果。
原路径/候选数值比较、主崩溃复现、独立beta检查是三种不同证据；beta进程使用独立诊断Graph，
不能替代主崩溃复现。任一子进程异常、缺证据或检查不通过均阻断整体验收和后续进程；
非法访问后直接退出，不在污染上下文继续、不自动采用候选。各子进程分别保存预算和执行状态。

beta正常完成但返回负结果时，子进程退出码为1。父进程先核对摘要的完成状态、
顶层/结果/检查的strict=false、diagnostic_valid=false、原因与信息充分性、摘要及实际退出码1，
再按StrictMismatch传播`failed`或`information_insufficient`，总报告strict=false。
CUDA异常、超时、摘要缺失/损坏/不完整或退出码不符仍属运行异常，strict=null；
两类都立即阻断后续子进程，总流程execution_completed=false，不把任意rc=1当成数值失败。

受控TD batch64由bank当前obs/action/reward/done确定性组装：非终态next obs取相邻条目，
终态next obs原bank未保存，用当前观测作显式占位（done=1）。success/near-goal/安全标记
也是受控夹具值，不冒充原formal replay；不写入正式replay。固定独立噪声，按原500k日程。
没有G辅助损失，不修改公式、学习率、BC、target/Adam/clip规则或默认性能开关。
GPUCompute保留原online_research_statistics输出，batch blocking、系数非阻塞实验关闭。

**候选**：编译前deepcopy两个critic为独立eager副本；每次评分前原位载入live权重，
只对副本forward，拒绝已编译副本或共享参数存储。绝不临时改live training/compile/注册或地址。
原路径不替换。评分前后审计live网络、targets、Adam、梯度、计数和各RNG；评分不得污染它们。
比较Q、C/A/EMA/beta、后续全部参数/Adam/targets/梯度，沿用原容差、不新增豁免。

非法访问或异常立即退出该进程，保留pre_bank/pre_update及已有记录，不在损坏上下文继续。
父进程遇到异常/超时/覆盖不足/比较失败会阻断后续。超时只清理subprocess.run创建的自有子进程，
不管理历史进程。若original失败，isolated不会自动成为“成功fallback”；需另行审查再决定运行。
未复现仅能写“本预算内未复现”；候选不能据此宣称修复或接入默认ACC。

## 身份与产物

输入原始文件SHA256、历史preflight源码身份、本轮源码逐文件身份分开记录。
本轮源码沿用CRLF→LF规范化；BC/checkpoint/JSON等输入保持原始字节哈希。
`scripts/v2_online_wd3_acc_source.json`应包含新增Python文件（现106项，不固定旧105数量）。
不要在服务器自动重生成清单掩盖不同步；上传本地审查封存版本。
开始及成功前检查实际读取的输入和源码未替换。

输出独立目录：manifest.json、summary.json；review/存共同状态索引、输入身份、进程状态、
日志和紧凑比较；other/存有限逐次动作/控制/参数证据及异常前记录。无正式passed模型或续训文件。
`execution_completed`、`strict_passed`、exit_code区分：执行异常时strict为null；完成但动作或
数值比较失败时strict=false；子worker只完成采集时strict=null，不能解释成比较通过。

## 独立服务器目录命令草案（本轮未执行）

先上传审查源码到下列**独立目录**，不要在原仓库pull、覆盖文件或触碰继续令牌。
独立目录必须是固定到本次提交的可用 Git checkout，不能仅复制 src/scripts：身份核对会调用 Git。
源码清单还包含未提交的椭球缓存环境版本。干净 checkout 的该文件与清单不符；须仅在独立目录
放入已核对的同版 `src/brain_uav/envs/v2_static_no_fly_env.py`（原始 SHA256：
`a7d6d704d9d48c3c134ef2da5b76bda8e8550ebbb376b13ee4a4375dbf1826d8`），保留其未提交状态，
再核对全部106项规范化源码哈希。不要改动原服务器仓库，也不要在服务器重生成清单来绕过校验。
下面函数默认只打印计划，不创建实验目录；显式`--launch`才记录日志并运行。

```bash
uav_online_followup() {
  local code=/home/zj/diagnostics/my_project_online_followup_20261102
  local old=/home/zj/projects/my_project
  local py=/home/zj/envs/uav_torch/bin/python
  local mode="$1"; shift
  cd "$code" || return 1
  local root="/home/zj/diagnostics/outputs/v2_followup_${mode}_$(date +%Y%m%d_%H%M%S_%N)"
  local -a cmd=("$py" -u -B -m brain_uav.scripts.diagnose_v2_online_followup "$mode"
    --history "$old/outputs/v2_online_wd3_acc_20261102_110301_966812111"
    --historical-repo "$old"
    --validation-pool "$old/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json"
    --ann-bc "$old/outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt"
    --snn-bc "$old/outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
    --source-manifest "$code/scripts/v2_online_wd3_acc_source.json" --output-dir "$root")
  local launch=false option
  for option in "$@"; do [ "$option" = --launch ] && launch=true; done
  if [ "$launch" = false ]; then
    PYTHONPATH="$code/src" "${cmd[@]}" "$@"
    return $?
  fi
  mkdir -p "$(dirname "$root")" || return 1
  local started=$SECONDS rc wall
  PYTHONPATH="$code/src" "${cmd[@]}" "$@" 2>&1 | tee "$root.log"
  rc=${PIPESTATUS[0]}
  wall=$((SECONDS-started))
  mkdir -p "$root/review" || return 1
  mv -- "$root.log" "$root/review/driver.log" || return 1
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$wall" | tee -a "$root/review/driver.log"
  return "$rc"
}
uav_online_followup snn-inference
uav_online_followup acc-bank
# 审查后才显式执行；本轮不要运行：
# uav_online_followup snn-inference --launch
# uav_online_followup snn-inference --closed-loop --launch
# uav_online_followup acc-bank --updates-per-worker 6 --worker-timeout 600 --launch
```

## 本地验证边界

```powershell
$env:PYTHONPATH='src;tests'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest test_diagnose_v2_online_followup.TestFollowup test_v2_online_wd3_acc test_v2_target_q_aggregation -q
git diff --check
```

本轮完整回归沿用`docs/v2_online_wd3_acc.md`里的原命令，在`python -m unittest`后追加
`test_diagnose_v2_online_followup.TestFollowup`，原84项加本轮初始化模式回归1项，共85项；诊断类单独19项。
新回归用临时小型合法ANN/SNN BC best（仅序列化，无BC训练）和真实严格prepare/build/load，
先确认actor为eval，再穿过真实make_engine；仅模拟CUDA开关/编译执行及设备信息。
确认预热时仍eval、预热后train，真实CPU ACC更新入口核对在线train/targets和BC冻结eval；
SNN继续调用真实select_action与actor_action进行A-A/A-B-A，确认两侧train/eval及前向上下文。
新增CPU集成用例实际经过bank/controller、eager更新及同次recorder；Graph元数据仅为CPU mock，
不作为CUDA证据。另测真实严格加载的ANN/SNN双向类型拒绝、新/旧beta与信息不足和总报告阻断。
首次导入夹具类时意外收集了旧GPUCompute测试：课程CUDA ABI夹具、BC初始化reference参数夹具、
环境诊断CLI旧断言三个无关用例出现失败（BC用例有ANN/SNN两个子项）。已改用模块导入，
不顺手修复这些旧用例，也不将这85项针对性通过称作全仓通过。

CPU/mock和极小合成更新只验证接线、公式/隔离/门控，**尚无真实CUDA复现、候选修复验收，
也没有SNN动作协议结论**。编译器内部追踪次数和训练侧同次脉冲仍为缺失证据。
本轮不执行完整状态重放、闭环飞行、真实源模型拟合/训练或CUDA。
