# v8 pilot 校验记录

日期：2026-10-02。新实验目录为 `experiments/v8_multilayer_retention/`；已归档 v1–v7 的 790 个文件保持原哈希。v8 没有修改依赖版本。

## 机制与统计验证

v8 新增 27 项检查，旧版本 73 项；总计 100 项，各版本在独立进程运行。

- 模型种子复现所有权重且不改变调用方随机流；固定图像路由在模型训练后不变；任务头正确选取。
- 倍率 0 的实际 expert 位移为 0，Adam 动量仍更新；倍率 1 与原始 Adam 相同；非法倍率被拒绝。
- 完整 learner/Adam/记忆 checkpoint 复制后，相同未来得到相同状态。覆盖了 PyTorch optimizer 复制时额外 expert 参数分组的保留。
- CIFAR 原始类映射到任务内 0–4；train/monitor/dev-eval 样本互斥；audit ID 空间分离；类别池隔离。
- 记忆总容量不随任务数增加，旧样本不会复活，引用基准跟随保留样本；任意前缀各类别数量差至多 1。
- 控制器专家置换等变，联合对数概率包含两层所有专家；微型实际分支训练能改变参数且费用计数正确。
- 所有已学任务进入结果矩阵；离线评估不改变 learner 或优化器；固定对角遗忘基准不会重置。
- 失败保留在根级汇总；policy replicas 在 root 内先平均；结果网格缺 root/method/seed、重复 replica、下三角缺失/非有限值均被拒绝。

## 真实数据 pilot

官方 CIFAR-100 下载及归档/内部数据校验通过。执行入口：

```bash
python scripts/run_experiment.py v8 --pilot --out runs/v8-pilot-20261002
python scripts/audit_v8_run.py --run runs/v8-pilot-20261002
```

结果原目录 `runs/v8-pilot-20261002/`，完整归档在 `experiments/v8_multilayer_retention/results/pilot/`。40 条选中 dev/audit 结果网格完整；两个模型哈希及 11 个 Python 源文件哈希通过检查。全部方法的相同 root 均共享首任务端点和任务类别顺序。

实际 learner 更新共 4,705 次，候选开发轨迹只计一次；校准、外层训练、模型选择、开发规则和 audit 分账。记录的墙钟约 39.4 秒。原始数据和开发选择轨迹完整保留；后补诊断明确标注 post-run，不用于调参。

## 结果边界

两学习臂都选初始化，所有候选的确定性动作都为全 1。这是可执行的训练流程，不是已学到有效控制。冻结 experts 的遗忘低于全量更新的点估计伴随 25% 更多 learner 更新，不能声称实际计算量匹配。

在线 monitor 逐段达标与官方测试最终保留不同；测试对角记录显示部分任务在自己的学习端点就没有达到 test loss 阈值，不能把全部最终未达标都归为遗忘。只有 3 个 audit roots、1 个 policy seed，且 roots 共享 25 类 pool。报告不作正式优越性/非劣推断。

细节见 [结果解读](../experiments/v8_multilayer_retention/结果解读.md)和[独立审计诊断](../experiments/v8_multilayer_retention/results/pilot/analysis/diagnostics.json)。
