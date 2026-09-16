# MoE Plasticity Control

研究持续学习中的专家更新控制：在固定 MoE 容量、记忆与在线计算预算下，利用路由共享结构和学习反馈分配更新强度，降低达到同一新任务目标时的旧知识退化。

**当前状态：已完成三轮合成实验，尚未验证学习型反馈控制器稳定优于充分调优的进度策略或规则基线。** 本仓库保留负结果、原始数据与冻结协议，不代表真实 learned-router MoE 或语言模型上的验证。

## 快速开始

推荐 Python 3.11；原始运行使用 Python 3.9.6、CPU 单线程。无需外部数据集或 GPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/test_all.py
```

共 23 项实验机制检查，各版本在独立进程中测试，避免同名模块串用。已有记录在 `experiments/*/results/`，首次查看结果无需重新训练。

## 实验索引

| 版本 | 研究问题 | 本轮结论 | 报告 |
|---|---|---|---|
| [v1](experiments/v1_expert_control/) | PPO 控制专家更新与保护强度 | 未超过重要性规则基线 | [报告](experiments/v1_expert_control/实验报告.md) |
| [v2](experiments/v2_feedback_control/) | 固定保护组后，旧任务反馈是否改善标量 PPO 控制 | progress/feedback 最终都选中未训练的初始策略，未验证反馈增益 | [报告](experiments/v2_feedback_control/实验报告.md) |
| [v3](experiments/v3_branching/) | 配对分支动作价值与监督评分器能否提取反馈增益 | 一次干预和闭环主要比较的区间均跨零 | [报告](experiments/v3_branching/results/main/analysis/实验报告.md) |

三轮实验源码与记录完整保留，版本之间有意保留冻结的模型副本。`docs/import_manifest.json` 校验导入文件；原有 `locked_protocol.json` 和各版本校验清单也保留。

## 复现入口

从仓库根目录运行；每次使用新的输出目录，已有结果不会被覆盖：

```bash
python scripts/run_experiment.py v3 --out runs/v3-reproduction
python scripts/run_experiment.py v3 --pilot --out runs/v3-pilot
python scripts/run_experiment.py v2 --out runs/v2-reproduction
python scripts/run_experiment.py v1 --out runs/v1-reproduction
```

`runs/`、虚拟环境及缓存不纳入 Git。原始正式结果已经提交，用于核查已有结论。完整实验会重新训练；单元测试只检验关键机制。

## 评价约定

- 首先报告新任务达标率，再在共同达标历史上作配对遗忘比较；失败不丢弃。
- 当前遗忘指标是独立旧任务评估 MSE 的正增量，不是百分比遗忘率。
- 达到统一 loss 阈值不等于端点 loss 数字严格相同，具体报告保留敏感性分析。
- v2/v3 动作作用于 Adam 的实际参数位移；路由和初始保护组固定。
- 预算使用明确的工作量代理，并非精确 FLOPs 或相同墙钟时间。
- 已查看的最终测试历史不可用于下一轮调参；新设计应使用新的测试历史。

详细研究范围与后续问题见 [研究说明](docs/research.md)。本仓库尚未指定开源许可证。
