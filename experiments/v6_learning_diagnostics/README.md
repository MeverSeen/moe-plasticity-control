# 控制器学习诊断：冻结环境，检查训练流程

回应“是否控制器没有学好”的问题，先验证学习能力，再检验目标与筛选方式是否影响实际闭环收益。真实 MoE 的路由问题继续暂缓。

**正式实验已完成。** 小样本拟合与辅助正对照通过基本能力检查；原损失不变，仅改用开发闭环评价筛选，测试遗忘下降约 24.4%，达标率相同。Feedback 相对 Progress 的额外收益仍未验证，也未胜过强时间基线。见 [结果解读](结果解读.md)、[原始报告](results/main/analysis/实验报告.md) 和 [统计汇总](results/main/analysis/summary.json)。

## 四项诊断

1. **小样本记忆**：从新训练集按预定规则取最多 32 个有动作差异且五档均可行的不同历史状态，保留 14→32→32 评分器、Adam 与正则，训练到 3000 轮，报告训练标签上的动作 regret。
2. **辅助正对照**：相同进度输入下，五种均衡反馈值直接确定最优动作。新上下文检查能否学会这个明确可学习的映射。它不是 MoE 结果，也不检验 PPO。
3. **监督损失消融**：原 BCE + log1p/居中 Huber，对比 BCE + 原始遗忘尺度/居中 MSE；两组共用模型、数据和匹配初始化。
4. **筛选消融**：每个训练过程的相同检查点/slack 候选，分别按开发分支 J、开发分支原始遗忘、开发真实闭环原始遗忘筛选。最后两者都优先达标率，工作量打破并列。

正式新根历史 128 train / 48 dev / 96 test，3 个策略初始化，4 个配对未来批次流。环境仍为 v4 的固定 6-expert router、目标 0.08、300 步和原工作量预算；固定后续策略仍为 m=0，不同时修改教师或添加状态聚合。新数据量不同，只有本轮内部匹配消融可用于归因。

每训练过程检查 epoch `{0,50,150,300,600}` 和 slack `{0,.02,.1,1}`。闭环开发筛选开销单列。两项预先指定主要比较是 raw_mse + closed_F 的 Feedback vs Progress，以及该 Feedback vs legacy + branch_J 的 Feedback；其余为探索性消融。区间是未作多重比较校正的名义 95% 根历史配对区间。

预先保存 `plan.json`；权重、基线、源码与选择在任何最终 MoE/辅助测试历史生成前锁定。所有训练曲线保存当前损失及 regret，保留全部开发候选，不只记录历次最优值。小样本拟合、辅助正对照和真实闭环结果应分别解释。

## 复现

```bash
python scripts/run_experiment.py v6 --pilot --out runs/v6-pilot
python scripts/run_experiment.py v6 --out runs/v6-reproduction
python scripts/test_all.py
```

在工程根目录运行，输出目录必须是新的。`model_core.py`、`base_env.py`、`branching.py` 从 v4 原样复制；新增学习逻辑在 `learning.py`，实验编排在 `diagnose.py`。
