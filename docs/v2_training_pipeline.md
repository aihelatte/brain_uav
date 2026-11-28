# V2 统一训练和离线检查

统一入口以正式课程配置为基础。每阶段默认最多 500,000 步，早停门槛和固定验证池沿用正式课程。ANN 用 min，SNN 用 ACC（η=0.01）。各模型按阶段分别选优，只有正式通过的候选才能晋级。CUDA 每批最多四个 seed，并先走同一批次的门控。CPU 顺序执行，使用 eager。

统一管线的 easy、medium、hard 使用同一套垂直安全奖励。距地面不足 60 m 且下降时，叠加权重为 180 的方向性惩罚，近地软惩罚与方向性分量合计上限为 300。距上界不足 150 m 且爬升时，方向性权重为 200。在 600 m 高度上限下，预警从 450 m 以上开始。水平四壁继续使用 60 m 预警距离和 90 的方向性权重，边界软惩罚合计上限为 300。

easy 和 hard 使用 `smooth-mid-strong`。阶段步数 0、75k、150k、200k、250k、300k、335k、370k 对应 BC 权重 500、150、75、60、30、20、14、10。medium 沿用原 `default` 日程。旧课程入口继续解析原有奖励和 BC 配置。

## 计划和运行

在已审查的 checkout 中执行。保留 `--plan` 可以核对模型、阶段、seed 和批次，打开实际训练时去掉它。输出根目录须为新目录。

```bash
out="/home/zj/diagnostics/outputs/v2_pipeline_$(date +%Y%m%d_%H%M%S_%N)"
PYTHONPATH=src /home/zj/envs/uav_torch/bin/python -u -B -m brain_uav.scripts.run_v2_pipeline \
  --models ann,snn --stages easy,medium,hard --seeds 7,11,23,41 \
  --output-root "$out" --eval-independent --plan
```

可以指定一个模型、一个阶段或一个 seed。跳过直接前置阶段时，通过 `--parent-ann` 或 `--parent-snn` 提供对应模型已正式通过且奖励配置一致的前置阶段权重。使用旧奖励的胜者仍可用于离线检查和原课程。输入树默认为 `/home/zj/projects/my_project`，可用 `--input-root` 指定。固定晋级池、独立评估池、BC 权重和当前源码负责各自的身份校验。

每个目标阶段完成后会更新 `report/README.md`。每个 seed 的原始日志、终态权重和周期快照位于 `runs/<stage>-<model>/seed<X>/`。胜者与晋级记录在 `winners/`，门控和 worker 原始证据保存在 `.batches/`。若所有计划批次均晋级且启用了 `--eval-independent`，末阶段胜者各飞三套独立题库，双模型合计 600 场。

训练和固定验证全程关闭轨迹图。默认 `--trajectory-archive full` 逐局压缩保存所有完整训练回合。换成 `failures` 时，保存全部失败回合，并按回合结束步数，在每个 50k 步区间保存第一个到达回合。坐标、动作和终态保留原单位并四舍五入到四位小数。

## 生成报告和按需重放

```bash
PYTHONPATH=src python -m brain_uav.scripts.build_v2_report --run-dir "$out"

PYTHONPATH=src python -m brain_uav.scripts.inspect_v2_run training \
  --seed-dir "$out/runs/hard-snn/seed7" --windows 12-15 \
  --outcomes boundary,ground --limit 20 --out ./inspect_training

PYTHONPATH=src python -m brain_uav.scripts.inspect_v2_run eval \
  --checkpoint "$out/runs/hard-ann/seed11/model.pt" \
  --pool "$out/pools/validation/v2_validation_hard.json" \
  --scenarios hard_00018,hard_00035 --out ./inspect_eval
```

`build_v2_report` 也可读取历史 `batches/` 目录。检查工具的输出目录放在原实验外。每份诊断同时保存 Markdown 和 JSON，`--limit` 只限制三视图图片数量。训练模式可加 `--no-render`。评估模式默认 CPU，支持终态、失败和周期快照，场景 ID 以实际验证池为准。
