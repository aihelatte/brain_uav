# WD3 / ACC 在线研究（待审，未运行）

## 新训练seed复验接线（2026-11-05，待审）

`--seed`默认23，使用正式配置的非负整数校验。显式7贯通计划、预检、派发、
shell worker、实际formal config及model/replay/curriculum/exploration派生种子。
worker在统一放行前核对CLI、manifest、job、formal config和实际seed_manifest；
清单严格包含base/model/replay/curriculum/exploration，以及curriculum_mix中每个level的
`{level}_generator_seed=derive_v2_component_seed(seed,stage,f'{level}_generator')`，
保持完整字典比较，缺失、篡改及额外字段均拒绝。
父进程核对初始化报告的formal_seed。同模型同seed各臂初始摘要须相同；
BC actor来自同一best，跨seed权重可能相同，不要求不同。
status/continue读取原批次manifest并核对seed与所选组；seed7批次查询/继续须显式传7，
不能用新seed混入旧23批次。旧23默认、作业ID及16组模式保留。
算法、学习率、BC/脉冲/ACC、编译和GPUCompute配置均不变。

以下仅为命令草案，不在原有暂停进程的仓库git pull。先把审查通过的代码打包，
在独立目录解包（包须含源码清单，服务器应逐项核对规范化源码哈希）：

```bash
mkdir -p /home/zj/diagnostics/my_project_seed7_20261105 &&
tar -xf /home/zj/diagnostics/reviewed_uav_seed7.tar -C /home/zj/diagnostics/my_project_seed7_20261105
```

三组并行启动，默认仅计划；审查授权后调用函数时加 `--launch`：

```bash
uav_seed7() {
  local code=/home/zj/diagnostics/my_project_seed7_20261105
  local old=/home/zj/projects/my_project
  local root=/home/zj/diagnostics/outputs/v2_seed7_$(date +%Y%m%d_%H%M%S_%N)
  local rc start
  cd "$code" || return 1
  local cmd=(/home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.run_v2_online_wd3_acc
    launch --root "$root" --seed 7 --jobs snn_min snn_fixed_own snn_acc_001
    --bank-scoring isolated --session "seed7_$(date +%s)" --socket seed7_20261105
    --source-manifest "$code/scripts/v2_online_wd3_acc_source.json"
    --snn-bc "$old/outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
    --validation-pool "$old/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json")
  if [ "${1:-}" != --launch ]; then
    PYTHONPATH=src "${cmd[@]}"
    return $?
  fi
  mkdir -p "$(dirname "$root")" || return 1
  start=$(date +%s)
  PYTHONPATH=src "${cmd[@]}" --launch 2>&1 | tee "$root.driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\nroot=%s\n' "$rc" "$(( $(date +%s)-start ))" "$root" | tee -a "$root.driver.log"
  return "$rc"
}
uav_seed7
```

从BC best重新开始，不恢复旧400k权重。500k日程，200k完整episode收尾、保存、
评估后原进程暂停；不会自动继续。确认三组暂停后，在同一独立代码目录执行
（把root替换为启动日志中的实际新批次，不能指向旧批次）：

```bash
cd /home/zj/diagnostics/my_project_seed7_20261105 &&
PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.run_v2_online_wd3_acc continue \
  --root /home/zj/diagnostics/outputs/REPLACE_WITH_ACTUAL_SEED7_BATCH \
  --seed 7 --jobs snn_min snn_fixed_own snn_acc_001 --launch
```

继续仍要求原存活PID及暂停身份，400k硬停止，不进入medium。历史快照严格加载及
旧六候选验收入口保持原样。本地CPU/mock检查不代表新seed CUDA或长期训练通过。
本轮本地命令：`PYTHONPATH=src;tests python -m unittest test_v2_online_wd3_acc test_diagnose_v2_online_followup.TestFollowup -q`
（PowerShell需先设置同名环境变量），45项运行、44通过、1项只读BC/池夹具未挂载跳过。
新增真实CPU prepare/build回归：临时合法BC checkpoint，seed7/23清单通过；
generator seed缺失或篡改、额外字段拒绝。不训练BC、不飞行，不以五字段mock代替构建器证据。
覆盖seed7三组派发与统一放行、worker/config错配拒绝、实际派生种子检查、
seed7快照严格加载/评估传参及原16组、子集、暂停/继续回归；未执行真实tmux或训练。

## 四组ACC有限研究补跑（2026-11-03，接线待审）

仅`ann_acc_001 ann_acc_010 snn_acc_001 snn_acc_010`，从各自BC best重新开始；
旧失败权重不是精确恢复。500k日程、200k完整episode收尾/保存/冻结评估后原进程暂停、
400k硬上限及显式继续令牌规则不变。新批次不自动继续，不重跑其余12组。
`--jobs`现在也限定launch，计划、预检模型集合、派发、初始化摘要和统一放行共用同一集合。
不传时仍为原16组。全部所选初始化完成、同模型摘要一致、独立PID存活后才放行。

`--bank-scoring isolated`仅显式在线研究选择，默认仍original。复用
`trainers.v2_online_research.IsolatedScorer`（独立诊断也导入此实现），编译前创建冻结eager
critic副本，每次评分前严格同步live权重。无参数共享，不切live模式，不全局monkey patch。
ACC公式、eta=.01/.1、完整bank、beta及更新频率、TD损失和更新次数不变；G仅供反馈。
initialized/stage_start/summary记录评分实现、选中组、源码、输入及初始化摘要；原controls轨迹保留。

只读核对已有产物：原SNN路径前6次完成、第7次缓存复用非法访问；候选24次更新、7图、
24次重放、三种模式覆盖。前6次每次1463个状态张量逐值相同，controller通过；
少数Q1评分仍超出原容差，`snn_prefix_comparison.json`的`prefix_strict_passed=false`保留。
独立beta严格验收通过是已有证据。本次接线不代表长期稳定、完全数值等价或ACC效果通过。

使用独立已审源码目录；绝不对有暂停进程的`/home/zj/projects/my_project`执行git pull、
写入、tmux操作或继续令牌操作。外部BC/固定池只读、绝对路径，不复制、不生成、不用软链接。
预检封存实际路径/raw SHA256，worker从manifest读取，评估子进程显式接收相同pool路径
及文件hash并在加载前校验；pool加载器仍校验easy/100场/content_digest及场景契约。
结束时重新核对源码/BC/pool。源码清单CRLF→LF，历史输入仍raw hash，旧历史清单不更新。

以下命令**仅草案，本轮未执行**。默认打印计划，无目录/任务创建；审查后才显式加`--launch`。
新会话/新socket/新批次均与旧实验隔离；shell入口保存各worker tee日志、退出码和耗时。

```bash
uav_acc_four() {
  local code=/home/zj/diagnostics/my_project_acc_isolated_20261103
  local old=/home/zj/projects/my_project
  local py=/home/zj/envs/uav_torch/bin/python
  local root="/home/zj/diagnostics/outputs/v2_acc_four_$(date +%Y%m%d_%H%M%S_%N)"
  local rc start
  cd -- "$code" || return 1
  local cmd=("$py" -u -B -m brain_uav.scripts.run_v2_online_wd3_acc launch
    --root "$root" --source-manifest "$code/scripts/v2_online_wd3_acc_source.json"
    --jobs ann_acc_001 ann_acc_010 snn_acc_001 snn_acc_010 --bank-scoring isolated
    --ann-bc "$old/outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt"
    --snn-bc "$old/outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt"
    --validation-pool "$old/outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json"
    --session "acc_four_$(date +%Y%m%d_%H%M%S)" --socket acc_isolated_20261103)
  if [[ "$1" != --launch ]]; then
    PYTHONPATH=src "${cmd[@]}"
    return $?
  fi
  mkdir -p -- "$(dirname -- "$root")" || return 1
  start=$SECONDS
  PYTHONPATH=src "${cmd[@]}" --launch 2>&1 | tee "$root.driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s dispatch_wall_seconds=%s\n' "$rc" "$((SECONDS-start))" | tee -a "$root.driver.log"
  return "$rc"
}
uav_acc_four
# 审查后才运行：uav_acc_four --launch
```

派发成功不等于训练完成；200k会原地暂停。每worker的review含initialized/progress/summary/
driver.log/process.json，other含controls、metrics_reports、研究快照及冻结评估。
本地mock不证明Linux tmux、CUDA或四组长期稳定性。

本轮本地回归沿用文末命令，额外加入`test_diagnose_v2_online_followup.TestFollowup`：
共88项，87通过、1跳过（原只读BC/池夹具未挂载）。新增三项覆盖四组派发/初始化阻断、
共享评分权重同步和live状态隔离、评估外部pool及原始hash检查；原worker集成用例同时
验证manifest外部路径到评估命令及评分实现透传。Python AST、Bash -n、diff空白和清单均检查。

实施计划：按附件锁定的16组合同，先验证逐支目标与ACC控制，再接真实eager/共享编译/GPUCompute，
然后给原trainer增加默认空的研究hook（不复制训练循环），最后接独立研究快照、评估、
tmux整批初始化门控、存活进程暂停及有界CUDA验收。旧固定actor工具不修改。
不暂存/提交，不改受保护文件；本地仅CPU/mock，真实任务必须显式--launch。

验收顺序：公式/控制器→trainer边界→严格快照→入口与启动预检→有限CUDA（待授权）→16组（待授权）。
评审重点：两支标签、动态beta图输入、200k保持调用栈、400k不造G、评估隔离、全16组放行。

代码职责：`v2_td3.py`负责研究目标/双标签loss及严格engine加载；
`v2_gpu_update.py`复用真实共享计算，`v2_gpu_compute.py`仅扩展研究系数/原末端统计读回；
`v2_formal_training.py`增加默认空hook；`v2_online_research.py`负责完整episode bank、
控制器、存活暂停、严格研究快照和隔离摘要；`diagnose_v2_policy.py`仅新增严格格式分流。
`run_v2_online_wd3_acc.py`负责单份16组矩阵、预检、worker、整批tmux门控和进度；
`diagnose_v2_online_research.py`负责独立冻结评估/共同动作/CUDA验收。
`scripts/online_v2_wd3_acc.sh`负责短命令、tee/退出码/窗口保留，source JSON负责实际文件身份。
必要回归集中在`tests/test_v2_online_wd3_acc.py`，未修改旧测试或受保护文件。

## 锁定的研究合同

这是从BC best开始的在线actor+双critic训练，**不是**旧`diagnose_v2_wd3_acc.py`
固定actor/500k终态事后拟合。旧工具与历史复现入口保持不动。
source HEAD以运行时实际核对为准；本轮实现起点为`e32ff18081bb82d82595ae83eeef1a643934c12d`。

每模型8臂（ANN、SNN共16臂同时初始化、共同放行）：

| arm | target | 参数 |
|---|---|---|
| min | 两支共同minimum | 原公式 |
| mean | 两支共同mean | 原公式 |
| wd3_015 | w*minimum+(1-w)*mean | w=.15 |
| wd3_045 | 同上 | w=.45 |
| wd3_075 | 同上 | w=.75 |
| fixed_own | critic1用q1，critic2用q2 | 固定beta=1 |
| acc_001 | 每支beta*qk+(1-beta)*minimum | beta0=.5, eta=.01 |
| acc_010 | 同上 | beta0=.5, eta=.10 |

q1/q2来自正式target critics；next action来自正式target actor，原平滑噪声和动作裁剪保留。
各自标签均是`reward+gamma*(1-done)*q_target_k`；终态严格reward。
actor仍使用Q1、alpha归一化、原BC/terminal_geo/脉冲项。
MC权重0，G不进入TD标签、loss或额外optimizer更新。
未知模式、MC与研究目标组合、旧GPUUpdate组合及配置冲突拒绝；没有eager回退。

两模型均easy/seed23/CUDA/batch64，warmup1280、policy_with_noise、actor冻结25000。
ANN actor_lr=1.55e-4、critic_lr=2e-4；SNN actor_lr=8.25e-4、critic_lr=1e-3。
alpha2.5/gamma.99/tau.005/policy_delay2；一个联合critic Adam，原联合全局裁剪阈值1，actor裁剪1。
500k noise schedule、smooth-mid-strong；P1关、P2/P3开；failure_sample_bias=2，
其余replay容量500k、成功池比例/近目标偏置及terminal_geo沿正式配置，不调整。
actor projection32/hidden192由BC严格架构继承；SNN time_window4，单边脉冲beta_max1、target_rate.17、25k–75k ramp。
ANN spike beta0。critics两模型都是ANN。

使用`default_v2_cuda_graph_compilation()`的正式组合、GPUCompute、CPU去重及默认精确椭球缓存。
**沿实际正式入口维持batch blocking；系数非阻塞、pinned/打包/新stream/旧设备replay等不开启。**
没有更改正式训练API/CLI的默认目标聚合或默认通过语义。
研究专用标量诊断进入原图输出和原末端读回，不逐步输出网络张量、不新增CUDA同步。
它增加少量统计开销，本研究不作为性能验收。

## ACC数据和控制器

仅ACC臂分配bank。自然训练episode结束后，按真实奖励和gamma逆推包含首步奖励的G；
保存真实observation、环境执行动作、reward、done、收集stage step和G，禁止替换成当前actor动作。
完整episode入队；总长度超过5000时删最旧完整episode，至少留最新一条，超长最新episode完整保留。
训练场景在飞行前记入`other/training_scenes.jsonl`，物理指纹命中固定池立即拒绝；
固定验证/开发集没有途径进入bank。空/非空不分层、不重加权，bank内transition等权。

stage_steps>=25000，距上次至少1000环境步，仅在自然episode边界：

```text
Qbar = (Q1_online(s,a) + Q2_online(s,a))/2  # bank实际动作，no_grad
C = mean(G-Qbar)
A = mean(abs(G-Qbar))
EMA = A (首次)，随后 .95*EMA + .05*A
S = max(1, EMA)
u = clip(C/S,-1,1)
beta_next = clip(beta + eta*u,0,1)
```

新beta仅影响后续TD更新；不反传控制器。control轨迹完整记录C/A/EMA/S/u、beta前后、
实际间隔、bank大小、年龄范围、episode边界、首次触边和触边调整比例。
beta到1不代表已校准。这是探索行为近期回报的近似反馈；不是冻结当前策略的精确无偏校准。
目标结构参考ACC-TD3；eta/间隔/EMA/bank等是本项目锁定参数，不声称是论文统一默认值。

研究目标在共享计算中用[B,2]保存两支标签；普通min/mean继续[B,1]，默认公式不改。
GPUCompute的研究配置由静态检查绑定；动态beta作为现有系数输入的第5个FP32值，
每次真实更新刷新到原系数地址，图内读取`coefficients[4]`。改变beta不改变图key或静态配置，
不重新捕获；加载使原图失效，原梯度bank、有限值/Adam/target保护和捕获恢复保持。

## 预算、暂停与继续

`config.max_steps=500000`始终只是原日程跨度；100k处噪声约`.014/.0114/.0228`。
原trainer增加默认None的研究hook，不复制TD3或在线循环。
研究禁用candidate验证/通过早停，但仍记录原episode/window统计；没有周期100场验证。

75k、150k、200k请求快照在首次跨阈值的完整episode收尾后保存，记录requested/actual。
200k先完成replay、成功/失败池、ACC反馈、episode统计、快照及独立20场评估，然后原进程暂停。
暂停保留调用栈、环境、replay、Adam、梯度、RNG和CUDA对象；只轮询本组继续令牌，
不抽场景、不采样、不更新。解除令牌必须指定存活PID、arm和actual_step。
只有显式选定存活组才继续；不重启、不恢复权重前缀、不自动继续其他组。
400k硬停止；若当前episode没结束，不造done/G或额外episode记录。
300k是episode边界快照；400k最终权重可独立评估，不冒充完整episode或formal passed。
进程丢失不能用这些快照宣称精确续跑；此入口没有resume功能。

暂停/ACC/独立评估/快照/初始编译/实际Graph捕获及进程总墙钟分别保存；训练净墙钟排除这些项。
另保留完整loop墙钟，明确统计/报告开销仍在训练净墙钟内，不能推断500k同比加速。
`release.json`只表示16组初始化一致且共同放行，不表示任何训练完成。

## 快照、评估、共同动作与确认计划

`v2_online_wd3_acc_research_snapshot`是独立严格身份，包含原周期快照、arm、w、beta及controller state。
非自适应臂不能携带controller；未知/冲突/非有限值拒绝。正式loader和课程前驱拒绝该格式。
内层也有research_only身份标记；未清除研究算法字段的内层不能通过普通周期/正式加载校验。
诊断loader显式校验后提供冻结策略视图；原min/mean正式产物继续原严格加载。
不是通用恢复格式，也不产生正式passed模型。

每个请求快照由独立进程在原`select_scenarios`固定20场开发集自然无探索飞行，
记录全部actions/rewards/N+1状态/终止信息/pre_action_hashes及线性重放得到的G、Q1/Q2。
训练端在调用子进程前后核对网络、梯度、Adam、targets、计数、replay RNG/长度、探索/selector RNG、
Python/NumPy/CPU/CUDA RNG和SNN膜状态摘要；评估不接触训练对象。
评估输入、源码和checkpoint hash前后核对；失败summary不会execution_completed=true。
Q1作为actor真实评分单列。双Q分别输出bias/MAE/RMSE、逐场、transition加权/场景等权、
空/非空、goal/各失败、五类几何、terminal/nonterminal、失败终步/最后5步、正误差及分位数尾部。
训练TD loss/Q-y/target分布、clip前梯度范数/裁剪比例、实际actor RL缩放/BC贡献、
快照参数范数/快照间更新范数及真实update计数分开保存；不凭loss大小断言梯度主导。
几何/失败子组会重叠，不能相加；20场已是开发集，不再称盲测。

`probe`是独立、默认plan、不会随训练自动运行：固定20开发集按sequence前12场，
BC自然轨迹中点prefix规则在manifest预注册；用正常reset+动作prefix恢复并精确核对hash。
在所有探针封存后，候选是BC、两维各±.2幅度、当前actor，真实clip后精确去重。
每个候选真实首动作后由待测actor无探索续飞算G；输出双Q相对own动作的ΔQ/ΔG、排序/并列/regret。
复用平移不敏感并列规则；局部动作实验不是完整MAGE或全局梯度证明。

`confirmation`当前仅返回预注册计划：新40场8空/32非空，覆盖五类几何，
排除训练/校准场景journal和原开发池ID/物理指纹；几何预选，seed20261102，
报告宏平均、子组及源easy概率加权结果。此模式即使--launch也拒绝生成：
候选锁定、再次授权后再补执行。此次不生成新池、不跑seed11/7，不自动选胜者。

## 产物和启动门控

固定BC路径/SHA与池digest在脚本常量和计划中只有一份：

- ANN BC best SHA256：`0e103aaad88bb7c60236d5c8719b126c597288af39f6e9c07d4b087809d93b7b`。
- SNN BC best SHA256：`7b2b266fdf32cdb8b4eb220ef26a45008717a0af1538f8d478019f37ddc9adbc`。
- easy固定池digest：`2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b`。

preflight真实使用Path调用严格BC/pool loader，检查源文件hash、BC best、池100场/digest、
默认椭球缓存、CUDA/tmux、GPU信息；不要求脏工作区清空。
16个直接执行worker的tmux窗口均初始化且同模型9项网络/BC/两个Adam摘要相同后才原子放行。
创建或初始化失败写abort，所有未放行worker退出，不启动部分训练。
OOM/资源不足不减少并发、不改batch或线程、不回退；初始化成功仍不能保证长期16并发资源充足。
训练开始后的单组失败留日志及状态，不自动重启，不替换、不自动杀其他组。

`manifest.json`封存输入/source HEAD/status/实际文件hash/完整有效配置。
各组review保存initialized/progress/summary、快照小型hash/参数摘要、driver.log/process.json；
other保存研究快照、完整评估证据、episodes/windows、controller轨迹及紧凑更新统计。
状态：initialized/running/evaluating/paused_at_200k/completed_budget/failed。
other/progress_history.jsonl保留带PID/时间的episode级状态沿革，可核对各进程实际重叠时段；
状态重叠不等于证明GPU内核同时执行或GPU利用率。
终端只返回、窗口创建或Released不能当完成；用status及process/summary实际记录判断。
tmux进程独立于SSH；支持tmux内外调用及私有-L socket，不改已有会话或全局tmux配置。

## 短命令（仅准备，本轮不执行）

服务器仓库`/home/zj/projects/my_project`，Python`/home/zj/envs/uav_torch/bin/python`。
先同步审查代码，核对文件hash；不要只核对HEAD。
仓库内`scripts/v2_online_wd3_acc_source.json`封存本地本轮实际源码hash，启动器默认读取；
与审查代码一并同步，不需外部脚本。源码哈希仅将CRLF统一为LF，兼容Git跨平台换行转换；
其余字节严格比较，BC、固定池及checkpoint仍使用原始文件哈希。
不得在服务器重新生成清单来绕过同步差异。
代码复审修改后需在本地明确重新封存/审查；source模式可另存新清单，拒绝覆盖。

```bash
cd /home/zj/projects/my_project && bash scripts/online_v2_wd3_acc.sh plan

# 默认只输出16组/探针/确认池/有界CUDA计划，不创建作业。
bash scripts/online_v2_wd3_acc.sh launch --root outputs/REVIEWED_NEW_BATCH
bash scripts/online_v2_wd3_acc.sh probe --checkpoint /ABS/RESEARCH.pt
bash scripts/online_v2_wd3_acc.sh confirmation
bash scripts/online_v2_wd3_acc.sh cuda-check --model snn

# 获准后：先分别ANN/SNN有限CUDA门控。失败不得开始16组训练。
# 每模型默认8臂，24个受控更新+1次加载重建/臂/路径，最多400次实际更新/模型。
# 真正eager vs GPUCompute；混合/全空mask、三种模式；beta刷新复用图，raw Kineto Graph launch证据。
# 沿用rtol=1e-4、atol=1e-5；strict_passed与engineering_accepted分开。
# 此入口不自动套用ANN历史接受范围，任何新失败严格阻断，无新增豁免。
uav_online_gate() {
  cd /home/zj/projects/my_project || return 1
  local model="$1" root="outputs/v2_online_gate_${1}_$(date +%Y%m%d_%H%M%S_%N)"
  mkdir -p "$root/review" || return 1
  local start=$SECONDS rc
  PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B \
    -m brain_uav.scripts.run_v2_online_wd3_acc cuda-check --model "$model" \
    --root "$root/other" --launch \
    2>&1 | tee "$root/review/driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s process_wall_seconds=%s\n' "$rc" "$((SECONDS-start))" | tee -a "$root/review/driver.log"
  return "$rc"
}
# uav_online_gate ann && uav_online_gate snn

# 复审且上述严格验收成功后才可手动执行：
# batch="outputs/v2_online_wd3_acc_$(date +%Y%m%d_%H%M%S_%N)"
# bash scripts/online_v2_wd3_acc.sh launch --root "$batch" \
#   --session uav_wd3_acc_seed23 --socket uav_research --launch

# 重连/查看；status的root须为上次实际batch路径。
# tmux -L uav_research list-windows -t uav_wd3_acc_seed23
# tmux -L uav_research attach -t uav_wd3_acc_seed23
# bash scripts/online_v2_wd3_acc.sh status --root "$batch"
# 只解除人工选定的存活暂停组，保持其它组暂停：
# bash scripts/online_v2_wd3_acc.sh continue --root "$batch" --jobs ann_mean snn_acc_001 --launch
```

每组内部`python -u | tee`后第一句立即`rc=${PIPESTATUS[0]}`，保存真实rc及进程耗时。
没有外层set -e/-u，不在用户交互shell末尾exit。窗口Python结束后保留日志路径及结果。
默认不运行任何CUDA门控/采集/评估/训练；本地mock不能证明Linux tmux、CUDA或16并发稳定性。

## 本地针对性验证

仅运行CPU/mock及两步小环境fixture；旧固定actor工具、MC兼容和必要安全保护一起回归。
未执行真实BC训练、源模型20场飞行、10000次拟合、CUDA门控、tmux或16组训练。
以下命令只针对本轮依赖，不代表全仓测试：

```powershell
$env:PYTHONPATH='src;tests'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest `
  test_v2_online_wd3_acc test_v2_target_q_aggregation test_diagnose_v2_wd3_acc `
  test_v2_formal_training.TestV2FormalTraining.test_terminal_non_goal_marks_complete_episode_but_stage_cutoff_stays_unconfirmed `
  test_v2_formal_training.TestV2FormalTraining.test_short_formal_loop_uses_policy_warmup_updates_critic_and_handles_dynamic_counts `
  test_v2_formal_training.TestV2FormalTraining.test_periodic_snapshot_disabled_by_default `
  test_v2_formal_training.TestV2FormalTraining.test_periodic_snapshot_and_validation_sinks_fire_at_step_boundaries_without_touching_early_stop `
  test_v2_gpu_compute.TestGPUCompute.test_real_common_compute_cpu_same_shapes_loss_gradient_and_update `
  test_v2_gpu_compute.TestGPUCompute.test_snn_regularization_uses_real_same_forward_and_resets `
  test_v2_gpu_compute.TestGPUCompute.test_actual_capture_restore_cpu_mock_preserves_values_none_and_masks `
  test_v2_gpu_compute.TestGPUCompute.test_cached_binding_checks_reject_parameter_and_optimizer_replacement `
  test_v2_mc_return.TestMCReturn.test_default_off_sampling_rng_update_and_old_engine_checkpoint `
  test_v2_mc_return.TestMCReturn.test_real_gpu_shared_compute_cpu_matches_eager_three_modes `
  test_v2_mc_return.TestMCReturn.test_formal_snapshot_legacy_strict_loading_and_configuration -q
git diff --check
```

测试会输出模拟第5窗口失败的预期阻断日志；共同动作测试沿用旧score helper，
可能出现list-of-NumPy-arrays构造Tensor的既有性能warning，不扩大修复。
另检查Python AST、新文件空白、启动器/说明Bash语法，以及受保护文件前后SHA256。

本轮局部修复及跨平台源码哈希回归共新增4项，上述同一命令现共66项：SNN actor证据在严格比较前
复用`complete_actor_evidence`，只从已记录spike loss/beta补全weighted spike，不覆盖原证据；
共同探针在选择场景/飞行前校验加载器返回的固定池digest；ACC快照按原gamma和reward
精确复算G（原链路保存Python float，无额外量化），并检查逐episode连续step、跨episode
时间顺序及completed_step等于末步且不晚于快照。旧episode完整淘汰和最新超长episode保留规则不变。
