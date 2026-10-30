# MC 辅助校准的 CUDA 正确性验收

这是新增 MC 路径的短计算验收，不运行环境飞行、正式训练、MPS 或性能对照。
默认仅输出计划；只有 `--launch` 才执行 CUDA 更新。CPU/mock 测试不能代替本验收。

对照使用相同 BC 初始化、critic/target/Adam 状态、固定 batch 和噪声，比较 eager 与
正式配置的 GPUCompute。固定使用 `mean`、MC weight `0.1`、age `100000`；
默认 SNN、batch 64、seed 7、critic_lr=1e-3，500k 日程、smooth-mid，
目标噪声按零基 step-1 取值。G 是诊断构造的受控标签，不是新增采集的飞行回报。

检查全有效、混合、全失效及变化的 G/mask（含零和非零 G）、年龄 `100000/100001`，以及 critic-only、冻结 actor
但更新 critic targets、实际更新 actor 三种模式。用原始 Kineto CPU 事件证明真实
`cudaGraphLaunch`；不能只凭配置、环境变量或 Graph 对象存在判断成功。
数值容差固定为 `rtol=1e-4, atol=1e-5`，失败保留失败报告，不放宽容差。

`summary.json` 的 `passed=true` 只表示这次有限计算验收通过，不证明长期训练一致、
critic 已校准、策略质量提升或完整训练加速。driver 墙钟包含初始化、编译、捕获与
逐项比较，不能用来推算训练速度。

## 服务器启动脚本

审查通过后，在本地准备“实际待同步源码”的哈希清单（包含全部 src Python 文件和
启动脚本，也包含未提交的椭球缓存默认值；不会复制模型）：

```powershell
cd E:\wurenji\my_project
$env:PYTHONPATH='src'
python -B -c "import json; from pathlib import Path; from brain_uav.scripts.diagnose_v2_mc_return_cuda import source_identity; Path(r'E:\wurenji\commands\v2_mc_cuda_source.json').write_text(json.dumps(source_identity(),indent=2),encoding='utf-8')"
```

将经过审查的源码及该清单同步到服务器后，执行以下命令。清单必须来自本地待审版本，
不可在服务器重新生成清单来掩盖同步差异。启动脚本不执行 pull、不改分支、不清理工作区。
只有 --launch 才调用 CUDA；无该参数时不创建输出目录。

```bash
cd /home/zj/projects/my_project && \
bash scripts/validate_v2_mc_return_cuda.sh /home/zj/projects/my_project/v2_mc_cuda_source.json

# 审查批准后才使用显式启动命令：
cd /home/zj/projects/my_project && \
bash scripts/validate_v2_mc_return_cuda.sh --launch /home/zj/projects/my_project/v2_mc_cuda_source.json
```

BC SHA256 和固定池 digest 来自已有本地记录；函数启动时脚本再次核验服务器输入，
不匹配即失败，不自动修改预期值。程序启动时记录源码 HEAD、status、逐文件 SHA256、
BC SHA256、池文件 SHA256、content_digest、实际算法/编译配置；成功前再次核对输入和源码。
源码哈希相等才证明本次文件同步，HEAD 相同本身不够。

脚本使用独立时间戳目录、python -u、tee、管道后立即读取 PIPESTATUS，记录真实退出码、
总耗时并返回非零失败码。小报告与日志位于 review；other 保留诊断结果。
不输出模型、checkpoint、批次张量或大型 trace，不运行吞吐验收。

同 shape/模式下连续变更 G/mask 时，报告 captures_before/after、MC 有效数及基于真实 Q
算出的预期 MC loss；加载双方自身 checkpoint 后检查旧图清空、新执行器及捕获重建。
保留原 eager 异常检查、GPUCompute 梯度 bank/Adam/target 发布及末端屏障；没有新增豁免。
参照是 eager 真实更新，GPUCompute 使用正式编译配置。此处不重复旧性能/MPS/传输矩阵。

这是新建验收入口，尚未取得服务器 CUDA 通过证据。
