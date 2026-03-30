# 基于类脑计算的飞行器静态禁飞区在线轨迹规划

这是一个从零搭建的最小可运行研究工程，用于验证“静态目标点 + 静态半球形禁飞区”场景下的在线轨迹规划方法。项目按“环境建模 -> 基线轨迹 -> 行为克隆 -> TD3 微调 -> 公平对照评估”组织，默认提供：

- Gymnasium 风格的三维运动学环境
- 启发式、人工势场、A* 三类参考轨迹生成器
- `SNN Actor + 双 ANN Critic` 的 TD3 训练框架
- `ANN Actor + 双 ANN Critic` 的公平对照模型
- 行为克隆、评估、FLOPs/MACs 与推理耗时统计脚本
- 基础单元测试

## 快速开始

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

如需正式依赖版 Gymnasium 与 SpikingJelly：

```powershell
pip install -e .[full]
```

## 典型流程

```powershell
python -m brain_uav.scripts.generate_dataset --output data/bc_dataset.npz --episodes 32
python -m brain_uav.scripts.train_bc --dataset data/bc_dataset.npz --model snn --epochs 5
python -m brain_uav.scripts.train_td3 --model snn --timesteps 2000
python -m brain_uav.scripts.train_td3 --model ann --timesteps 2000
python -m brain_uav.scripts.evaluate --checkpoint outputs/td3_snn.pt --model snn --episodes 16
python -m brain_uav.scripts.profile_models --snn outputs/td3_snn.pt --ann outputs/td3_ann.pt
```

若未安装 `gymnasium` 或 `spikingjelly`，项目会使用本地兼容回退实现，便于先跑通研究闭环。
