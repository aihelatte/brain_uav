# 受控 MPS 八并发对照（诊断层，默认仅计划）

入口 `python -m brain_uav.scripts.v2_mps_diagnostic` 复用原并发工具的参数、
`run_batch`、共同窗口、ready/timeout/abort、尾步剔除、场景覆盖、编译/捕获有效性检查。
不改变 worker 的算法、CUDA Graph、checkpoint、正式默认值或 CPU 线程设置。

服务器 RTX4090D / driver550.120 / CUDA12.4、工具在 `/usr/bin` 是用户提供的环境信息；
本地没有访问服务器核实。存在工具/设置环境变量不等于该卡、驱动和 PyTorch 支持实际 MPS Graph。
不安装工具、不设置 MPS active-thread percentage、不更改 compute mode/MPS全局设置、不操作 tmux。

## 执行与证据

- 默认无 `--launch`：只检查输入、输出计划，不启动 MPS/worker/CUDA。
- `--launch --capability-only`：仅独立能力检查。
- `--launch`：能力检查通过后才依次执行 lightweight普通、lightweight MPS、
  throughput普通、throughput MPS；每条件4 ANN+4 SNN、重复2次、共同窗口30秒。
  总8批，条件内并发、条件间串行，不自动重试失败。

能力检查使用新建私有短路径 `/tmp/uav_mps_*` pipe 与本次输出下的独立日志目录，
启动本次 control daemon，以真实 `get_server_list` 查询检查接口并严格解析PID列表；
不查询交互 `help`（550.120实际返回rc=0、Unknown command），不以帮助文本判定能力。
子进程执行少量真实 PyTorch CUDA Graph replay 并同步，然后保留上下文等待父进程核验。
父进程通过 server/client列表核对 **Popen真实PID**，并核对 Graph就绪证据，才释放其同步正常退出。
随后另起一个短客户端，复用同一Graph/同步就绪流程，保持等待父进程；仅核验本私有
服务PID及同步证据后，实际执行 `terminate_client`。只有明确响应 `0` 才确认终止能力，
再发送OS退出信号；不会释放该客户端继续调用CUDA。此探针不替代前一个正常退出检查。
非零、空、未知响应或控制超时均失败，不重试凑通过，不发送OS信号；未同步或未核验时
也不执行终止试验。保留服务/进程并阻断后续对照，供人工核查。
完成能力检查后关闭本次服务，再进行普通条件，避免服务本身污染普通基线。

每批ready之后、共同窗口发布之前，再核对所有8个客户端 PID：
MPS 必须与私有服务 `get_client_list` 精确一致；普通 CUDA 要求没有MPS进程且
`nvidia-smi --query-compute-apps=pid` 恰好是8个worker PID。额外/缺失/无法解释PID均报错。
不支持PID namespace转换；容器PID不一致直接阻断，不能靠环境变量声明effective。
这些是窗口前接入证据，不是整个窗口内对外部负载的全量监控或跨进程锁证据。
开始和条件间检查 GPU/MPS 空闲；若外部任务启动就阻断，不清理它。

配置沿用 Event 诊断 A：GPUCompute/CPU去重/精确椭球缓存开启，batch nonblocking=True，
系数非阻塞=False，正式编译组合；ANN lr1.55e-4/SNN8.25e-4、critic2.5e-4、alpha2.5、
smooth-mid、seed7、batch64、offset75000、原BC best/固定池/奖励、P1关P2/P3开。
普通/MPS来自相同 fresh BC 和原随机配置；不是历史训练恢复，不代表完整500k负载。
本入口拒绝 Event、profiler、细粒度batch插桩、系数实验和外部 CUDA_MPS_* 覆盖。
lightweight阶段仅启用已有CPU三项计时；它们与整步/TD3父项重叠，不相加。
throughput阶段关闭新增轻量计时；原正式短循环的基础分项保持不变。
CPU墙钟含提交、排队与等待，不是纯GPU时间；不能由局部栏变化宣称整步加速。

## 失败清理

NVIDIA legacy Volta+控制接口：
[get_server_list / get_client_list / terminate_client](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)。
对仍存活的本批 MPS worker，先重新核对其在本私有服务中的server/client PID，
调用 `terminate_client <server PID> <client PID>` 成功返回后才允许原 OS terminate/kill。
这先停止该客户端的CUDA上下文，不新增正常测量同步。
控制命令报错/超时、出现外部客户端、存活进程未在列表中时，**不发送OS信号**，
记录 cleanup_errors，保留服务和进程供人工核查，返回非零，禁止继续对照。
不能为“自动清理干净”牺牲MPS上下文安全，也不会kill其他进程。
仅所有自有进程退出且服务无客户端后，才对本私有pipe执行 `quit`。
不删除pipe/log目录，异常证据保留；无需自动清理系统目录。

普通路径沿用原工具清理规则。MPS能力探针和每批worker均在创建时登记，
部分分发、未ready、失败和超时也进入上述安全检查。未知上下文状态会阻断清理。

## 产物

`review/plan.json`：源码HEAD/工作区、池digest/输入hash、真实配置与编译组合。
`review/summary.json`：能力门控、每批接入证据、worker退出码、无效原因、
完整TD3/整步/batch/动作分项、已有CPU Graph提交及结果完成等待、整批共同窗口吞吐。
`other/`：worker日志和明细、Graph及独立终止能力探针日志/就绪/退出码、MPS日志和控制命令响应。
顶层错误与服务清理错误同时打印到stderr，由命令中的tee写入driver.log。
无 checkpoint、模型张量或大型trace。失败也保存summary，不把“分发成功”当完成。
`completed`是诊断执行状态，不是正式训练passed或加速结论。
请分开审阅轻量定位与无额外插桩吞吐重复结果；改善未稳定或等待转移必须如实记录。

## 服务器命令草案（未执行）

以下函数无外层set -e/-u，不在交互shell末尾exit；管道后立即取真实Python退出码。
默认只计划；完整实验显式调用 `uav_mps_compare --launch`。

```bash
uav_mps_compare() {
  cd /home/zj/projects/my_project || return 1
  local root py start rc
  root="outputs/v2_mps_compare_$(date +%Y%m%d_%H%M%S_%N)"
  py=/home/zj/envs/uav_torch/bin/python
  mkdir -p "$root/review" || return 1
  start=$(date +%s)
  PYTHONPATH=src "$py" -u -m brain_uav.scripts.v2_mps_diagnostic \
    --ann-bc-checkpoint outputs/v2_projected_head_round1/ann_20261012_102839/bc/bc_v2_ann_best.pt \
    --snn-bc-checkpoint outputs/v2_projected_head_round1/snn_20261012_102848/bc/bc_v2_snn_best.pt \
    --validation-pool outputs/v2_box085_round1_20261003_003948/validation_pools_600km_seed20260904/v2_validation_easy.json \
    --expected-pool-digest 2e3199a60b8ff23f9cb8f8395d6f44bcb7d73abeddf2ff5a31f1c95c19332f4b \
    --ann-actor-lr 1.55e-4 --snn-actor-lr 8.25e-4 --critic-lr 2.5e-4 --alpha 2.5 \
    --bc-schedule smooth-mid --seed 7 --validation-seed 20260904 \
    --batch-size 64 --replay-capacity 500000 --scenario-count 3 --snn-time-window 4 \
    --controlled-stage-step-offset 75000 --warmup-steps 128 --steps-per-level 256 \
    --measurement-seconds 30 --repeats 2 --concurrency 8 \
    --ready-timeout 1800 --completion-timeout 120 --non-blocking-batch-transfer \
    --output-dir "$root/other/run" "$@" 2>&1 | tee "$root/review/driver.log"
  rc=${PIPESTATUS[0]}
  printf 'exit_code=%s\nprocess_wall_seconds=%s\n' "$rc" "$(( $(date +%s)-start ))" | tee -a "$root/review/driver.log"
  if [ -d "$root/other/run/review" ]; then cp -R -- "$root/other/run/review/." "$root/review/" || return 1; fi
  return "$rc"
}
uav_mps_compare
# 审查后先独立短能力检查：uav_mps_compare --launch --capability-only
# 审查后完整对照（会再次做能力检查）：uav_mps_compare --launch
```

本地只做CPU/mock测试及语法检查；未启动MPS、CUDA Graph或真实worker。
尚不能确认服务器MPS支持、实际接入、清理命令行为或吞吐收益。
