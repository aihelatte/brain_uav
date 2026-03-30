# 基于类脑计算的飞行器静态禁飞区在线轨迹规划

这是一个可直接运行的研究工程，用于验证“静态目标点 + 静态半球形禁飞区”场景下的在线轨迹规划方法。项目按“环境建模 -> 基线轨迹 -> 行为克隆 -> TD3 微调 -> 公平对照评估 -> 结果汇总”组织。

## 环境准备

```powershell
pip install -e .
pip install gymnasium spikingjelly -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## 一键跑完整实验

```powershell
$env:PYTHONPATH='E:\wurenji\my_project\src'
python -m brain_uav.scripts.run_full_experiment --dataset-episodes 64 --bc-epochs 8 --td3-timesteps 5000 --eval-episodes 8 --output-dir outputs\full_run
python -m brain_uav.scripts.plot_results --summary outputs\full_run\summary.json
```

## 训练时你会看到什么

在 VS Code 终端里，运行时会按阶段打印进度：

- `[Stage 1/5]`：正在生成 BC 数据集
- `[BC] epoch x/y`：行为克隆预训练的 epoch 进度和 loss
- `[Stage 3/5]`：开始 TD3 训练
- `[TD3] episode=...`：每个 episode 结束时的回报、长度、结局
- `[TD3] progress=...`：每隔一段步数打印一次总体训练进度、buffer 大小、actor/critic loss、最近平均回报
- `[Stage 4/5]`：开始 benchmark 评估
- `[Stage 5/5]`：开始 MACs/参数量统计并写 summary

所以你训练时不用担心“程序卡住了”，只要终端还在刷新这些信息，训练就是正常在跑。

## 运行结束后看哪里

重点看这些文件：

- `outputs/full_run/summary.json`：完整实验汇总
- `outputs/full_run/bc_snn_metrics.json`、`outputs/full_run/bc_ann_metrics.json`
- `outputs/full_run/td3_snn_metrics.json`、`outputs/full_run/td3_ann_metrics.json`
- `outputs/full_run/plots/summary_overview.png`

## 分步运行

```powershell
python -m brain_uav.scripts.generate_dataset --output data/bc_dataset.npz --episodes 64
python -m brain_uav.scripts.train_bc --dataset data/bc_dataset.npz --model snn --epochs 8
python -m brain_uav.scripts.train_bc --dataset data/bc_dataset.npz --model ann --epochs 8
python -m brain_uav.scripts.train_td3 --model snn --timesteps 5000 --bc-checkpoint outputs/bc_snn.pt
python -m brain_uav.scripts.train_td3 --model ann --timesteps 5000 --bc-checkpoint outputs/bc_ann.pt
python -m brain_uav.scripts.evaluate --checkpoint outputs/td3_snn.pt --model snn --episodes 8 --scenario-suite benchmark --output outputs/eval_snn.json
python -m brain_uav.scripts.evaluate --checkpoint outputs/td3_ann.pt --model ann --episodes 8 --scenario-suite benchmark --output outputs/eval_ann.json
python -m brain_uav.scripts.profile_models --snn outputs/td3_snn.pt --ann outputs/td3_ann.pt --output outputs/profile.json
```

## 当前能力

- Gymnasium 风格的三维运动学环境
- 启发式、APF、A* 三类参考轨迹生成
- SpikingJelly SNN Actor 与 ANN 对照模型
- BC 预训练 + TD3 微调
- 基准场景评估、推理耗时统计、有效 MACs 对比
- 汇总 JSON 与结果图片导出
