# MoE 学习器状态聚合与后续策略诊断

**正式运行已完成；本轮未验证数据聚合带来的反馈控制优势。** 见 [实验报告](results/main/analysis/实验报告.md)、[结果解读](结果解读.md)、[统计汇总](results/main/analysis/summary.json) 和 [结果图](results/main/analysis/results.png)。

继承 v3 的状态、动作、损失、固定路由和动作评分器，检验闭环状态覆盖能否让反馈信息转化为控制收益。主要比较提前指定为第二轮聚合后的 Feedback vs Progress，而非从测试表现最好的轮次中选择。

## 设计

- 固定 6-expert MoE，旧回放/监测各最多 64 样本，目标新监测 MSE≤0.08；预算与 v3 相同。v3 部分文字写成 0.06，见 [勘误](../../docs/v3-erratum.md)。
- 五档保护组 Adam 位移倍率，每 5 步决策；Progress 已包含 ΔB、初始信息、上次动作、预算等，Feedback 额外读取两个旧知识反馈量。
- BC：初始固定/随机行为路径数据。Aggregate：两轮运行学习器、查询原有固定 continuation 分支教师、聚合后从匹配初始化重新拟合。Matched：每轮每历史查询数相同，但从预设固定/随机行为路径取状态。
- 汇集所有匹配 Progress/Feedback 策略访问的状态，两种输入共用同一数据池；完整状态去重，每条轨迹最多 3 个时间分散快照。不是最优教师，不直接套用 DAgger 理论保证。
- 固定 continuation 仅在校准历史选定，训练标签与拟合规则不随聚合轮次变化。开发集选择方法沿用 v3；最终轮次预先指定。
- 校准/训练/开发/最终测试为 16/64/24/80 个根历史，使用全新的种子；3 个匹配策略初始化。pilot 与正式划分互斥。
- 同时保存每轮分支工作量、行为采样工作量和拟合时间；匹配查询数不意味着匹配离线 FLOPs。
- 终点报告达标率、共同成功配对遗忘、包含失败的代价、初始即达标和紧阈值敏感性分析。

## 后续策略诊断

在冻结策略访问的测试状态上，按预设顺序每历史选一个状态，比较 Qconstant（动作一直固定）、Qfixed（H 步后接校准固定策略）、Qadaptive（H 步后接预先指定的冻结反馈策略）。用未来流 0 的分支标签选择第一动作，统一在独立未来流 1 和同一个 adaptive continuation 下评价，区分续跑方式不同导致的标签变化。

这些读取独立评估数据的 oracle 只用于额外离线诊断，不进入控制器在线选择。Qconstant 本身已到终点，不称为短视 oracle；adaptive 是一条已训练策略，并非全局最优规划。

## 运行

从工程根目录执行：

```bash
python scripts/test_all.py
python scripts/run_experiment.py v4 --pilot --out runs/v4-pilot
python scripts/run_experiment.py v4 --out runs/v4-reproduction
```

结果中的 `plan.json` 在训练前写入，`locked_protocol.json` 在首次生成最终测试历史前写入源码与策略哈希。分析读取 `*_branches.jsonl`、`*_predictions.json`、`test_online.json`，生成 `analysis/实验报告.md` 和 `summary.json`。

`branching.py`、`base_env.py`、`model_core.py` 与机制回归测试从 v3 原样复制；新增逻辑在 `dagger.py`。这仍是合成机制实验，不是 learned-router 验证或 RL 必要性的证明。

数据聚合动机：[Ross、Gordon、Bagnell，2011](https://proceedings.mlr.press/v15/ross11a.html)。
