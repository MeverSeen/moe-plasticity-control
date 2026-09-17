# MoE Plasticity Control

研究持续学习中的专家更新控制：在固定 MoE 容量、记忆与在线计算预算下，利用路由共享结构和学习反馈分配更新强度，降低达到同一新任务目标时的旧知识退化。

**当前状态：已完成八轮合成实验。v8 的15动作扩展有局部分支价值，但闭环遗忘增加，未验证旧知识反馈或PPO的稳定额外收益。** v8完成64个新测试历史、三个初始化及监督/真实PPO匹配对照；四组15动作控制器在共同成功历史上均比5动作遗忘更多，校正敏感性区间支持该方向。固定 `(0.25,1.5)` 是更强的保留基线，测试达标率84.38%；预设90%概率目标未达到。v7的初步正信号仍保留，不能视为已稳定复现。详见[v8结果解读](experiments/v8_allocation_ppo/结果解读.md)。

## 快速开始

推荐 Python 3.11；原始运行使用 Python 3.9.6、CPU 单线程。无需外部数据集或 GPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/test_all.py
```

共 90 项实验机制与统计检查，各版本在独立进程中测试，避免同名模块串用。已有记录在 `experiments/*/results/`，首次查看结果无需重新训练。

## 实验索引

| 版本 | 研究问题 | 本轮结论 | 报告 |
|---|---|---|---|
| [v1](experiments/v1_expert_control/) | PPO 控制专家更新与保护强度 | 未超过重要性规则基线 | [报告](experiments/v1_expert_control/实验报告.md) |
| [v2](experiments/v2_feedback_control/) | 固定保护组后，旧任务反馈是否改善标量 PPO 控制 | progress/feedback 最终都选中未训练的初始策略，未验证反馈增益 | [报告](experiments/v2_feedback_control/实验报告.md) |
| [v3](experiments/v3_branching/) | 配对分支动作价值与监督评分器能否提取反馈增益 | 一次干预和闭环主要比较的区间均跨零 | [报告](experiments/v3_branching/results/main/analysis/实验报告.md) |
| [v4](experiments/v4_state_aggregation/) | 学习器状态聚合能否把反馈信息转化为闭环收益 | 聚合未优于进度策略或等查询数对照；固定冻结保护组已全部达标 | [报告](experiments/v4_state_aggregation/results/main/analysis/实验报告.md) |
| [v5](experiments/v5_calibration/) | 不限定冲突形态时，固定选择和时序安排是否有可复测收益 | 固定选择参考有局部收益；加入切换计划未稳定胜过时间基线，全可行子集未胜过直接冻结 | [报告](experiments/v5_calibration/results/main/analysis/实验报告.md)、[解读](experiments/v5_calibration/结果解读.md) |
| [v6](experiments/v6_learning_diagnostics/) | 负结果是否来自学习能力、监督目标或模型筛选问题 | 基本学习检查通过；闭环筛选改善结果，但尚未验证实时反馈增益 | [报告](experiments/v6_learning_diagnostics/results/main/analysis/实验报告.md)、[解读](experiments/v6_learning_diagnostics/结果解读.md) |
| [v7](experiments/v7_sequence_feedback/) | 连续多段学习是否让段间或段内旧知识反馈更有价值 | 段内遗忘约 −13.8%，但可行性略低、校正敏感性跨零；段间呈达标/遗忘取舍 | [报告](experiments/v7_sequence_feedback/results/main/analysis/实验报告.md)、[解读](experiments/v7_sequence_feedback/结果解读.md) |
| [v8](experiments/v8_allocation_ppo/) | 双组更新倍率、动态控制与真正PPO是否提供额外收益 | 新动作局部分支有效，但四组闭环遗忘增加；固定组合更强，未验证反馈/RL稳定收益 | [报告](experiments/v8_allocation_ppo/results/main/analysis/实验报告.md)、[解读](experiments/v8_allocation_ppo/结果解读.md) |

八轮实验源码与记录完整保留，版本之间有意保留冻结的模型副本。`docs/import_manifest.json` 校验前三轮导入文件；原有 `locked_protocol.json` 和各版本校验清单也保留。v3 阈值文字错误见 [勘误](docs/v3-erratum.md)，v4 的机制解释见 [结果解读](experiments/v4_state_aggregation/结果解读.md)。真实 MoE 当前路由反馈暂缓，见 [后续待办](docs/future-real-moe-routing.md)。

## 复现入口

从仓库根目录运行；每次使用新的输出目录，已有结果不会被覆盖：

```bash
python scripts/run_experiment.py v8 --out runs/v8-reproduction
python scripts/run_experiment.py v8 --pilot --out runs/v8-pilot
python scripts/run_experiment.py v7 --out runs/v7-reproduction
python scripts/run_experiment.py v7 --pilot --out runs/v7-pilot
python scripts/run_experiment.py v6 --out runs/v6-reproduction
python scripts/run_experiment.py v6 --pilot --out runs/v6-pilot
python scripts/run_experiment.py v5 --out runs/v5-reproduction
python scripts/run_experiment.py v5 --pilot --out runs/v5-pilot
python scripts/run_experiment.py v4 --out runs/v4-reproduction
python scripts/run_experiment.py v4 --pilot --out runs/v4-pilot
python scripts/run_experiment.py v3 --out runs/v3-reproduction
python scripts/run_experiment.py v3 --pilot --out runs/v3-pilot
python scripts/run_experiment.py v2 --out runs/v2-reproduction
python scripts/run_experiment.py v1 --out runs/v1-reproduction
```

`runs/`、虚拟环境及缓存不纳入 Git。原始正式结果已经提交，用于核查已有结论。完整实验会重新训练；单元测试只检验关键机制。

## 评价约定

- 首先报告新任务达标率，再在共同达标历史上作配对遗忘比较；失败不丢弃。
- 当前遗忘指标是独立旧任务评估 MSE 的正增量，不是百分比遗忘率。
- v7/v8 三个 B 分别达标不等于最后仍全部保留；A 记忆固定，B 最终保留另报。
- 达到统一 loss 阈值不等于端点 loss 数字严格相同，具体报告保留敏感性分析。
- v2–v8 动作作用于 Adam 的实际参数位移；路由和初始保护组固定。
- 预算使用明确的工作量代理，并非精确 FLOPs 或相同墙钟时间。
- 已查看的最终测试历史不可用于下一轮调参；新设计应使用新的测试历史。

详细研究范围与后续问题见 [研究说明](docs/research.md)。本仓库尚未指定开源许可证。
