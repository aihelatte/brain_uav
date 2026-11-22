# ANN medium CUDA 短诊断

使用 `e6bc9be` 的原正式 checkout。把审查后的 [诊断脚本](../diagnostics/diagnose_v2_medium_ann_cuda.py) 单独复制到服务器的 `/home/zj/diagnostics/diagnose_v2_medium_ann_cuda.py`。正式 `src/` 和源码清单里的文件保持原样。

入口只读核对 `/home/zj/diagnostics/outputs/v2_course_e6bc9be` 中的第一个 ANN medium 批次、原 CUDA gate、ANN easy 胜者和 medium 固定池。每个 worker 放行前与原批次比较派生种子和初始状态哈希。正式日程仍按 750,000 步解析，每个 worker 在完成第 1,000 步后停止。通过时会写诊断完成状态和实际步数，正式通过字段为空。

## 单 seed

```bash
cd /home/zj/diagnostics/my_project_formal_course_e6bc9be &&
SINGLE="/home/zj/diagnostics/outputs/v2_medium_ann_cuda_seed41_$(date +%Y%m%d_%H%M%S_%N)" &&
env -u CUDA_LAUNCH_BLOCKING PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B /home/zj/diagnostics/diagnose_v2_medium_ann_cuda.py --run --out "$SINGLE" --seeds 41
```

如果出现非法内存访问，保留本目录，停止四进程对照。需要缩小观察范围时，用新 `--out` 重跑出现问题的那一组并加 `--sync-window`。这会在第 675 步进入窗口时和第 675～700 步各标记环节后同步，CUDA Graph capture 内的执行顺序仍由正式代码控制。

## 四 seed

单 seed 在异步模式下完成 1,000 步且 MPS 清理成功后执行。新 shell 中先将 `SINGLE` 指向上面实际打印的目录。

```bash
cd /home/zj/diagnostics/my_project_formal_course_e6bc9be &&
FOUR="/home/zj/diagnostics/outputs/v2_medium_ann_cuda_four_$(date +%Y%m%d_%H%M%S_%N)" &&
env -u CUDA_LAUNCH_BLOCKING PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B /home/zj/diagnostics/diagnose_v2_medium_ann_cuda.py --run --out "$FOUR" --seeds 7 11 23 41 --single-result "${SINGLE:?请设置单 seed 的结果目录}"
```

四个 worker 完成正式初始化且私有 MPS 成员校验通过后，由同一个放行文件一起起跑。每次诊断创建独立 MPS 会话，先运行原 launcher 使用的短能力探针。已有外部 MPS 服务时按现有流程拒绝启动。

## 输出与判读

- `summary.json`、`pids.json` 和 `mps_access.json` 分别保存批次结果、worker PID 和 MPS 成员关系。 `mps/logs/server.log` 保留服务端错误与 PID，`ann_seed*.log` 保留 worker 堆栈。
- `ann_seed*/cuda_trace.jsonl` 逐行刷新，记录第 675～700 步的实际场景等级、障碍数、动作推理及读回、当前及下一观测的回放批次形状、数据搬运、GPUCompute 的实际图缓存键及新建或复用、回放提交和结果读回。
- 异步模式的 `host_return` 表示宿主调用已返回。同步复跑增加 `sync_begin` 和 `sync_done`，便于区分提交和完成。逐行记录会增加 CPU 开销，应结合 MPS 日志和未插入同步的首轮结果判读。
