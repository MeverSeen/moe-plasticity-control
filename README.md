# MoE Plasticity Control

[English](README.en.md) · [英文技术报告](docs/technical-report.md) · [实验索引](#实验索引) · [复现入口](#复现入口)

研究目标：训练一个根据当前学习状态控制不同 MoE 专家更新的神经网络，在学习新任务时保留已有能力。旧知识退化反馈能否提供额外收益，是其中一个子问题。项目分别检验动作接口、额外反馈和控制器学习，记录失败、离线搜索与评价成本。

**当前进展（2026-10-09）：v11 已完成逐 expert RL 学习诊断。** 延长至 48 轮后，GRPO 确实改变了部署动作；两个协调控制器在开发历史上均选择零专家位移，行为与固定规则完全相同，尚未验证动态协调的额外优势。v1–v9 与 v11 的源码、结果和边界均已归档；旧知识反馈的最有利结果仍是 v7 的初步信号。[v11 结果解读](experiments/v11_rl_diagnostics/结果解读.md)。

| 关键证据 | 结果 | 解释边界 |
|---|---|---|
| [v7 连续合成学习](experiments/v7_sequence_feedback/结果解读.md) | 段内 Feedback 在 56 个共同成功根历史上的原始 A 遗忘约低 13.8% | 达标率非劣效未验证，多重比较敏感性区间跨零；不是全部历史保留 |
| [v8 真实图像 pilot](experiments/v8_multilayer_retention/结果解读.md) | 局部与协调 GRPO 都选中初始化，确定性动作均为全量更新 | 只有 3 个共享类别池的 audit roots；零专家位移的低遗忘伴随 25% 更多更新，前端、头与专家动量仍更新 |
| [v9 学习压力诊断](experiments/v9_learning_pressure/结果解读.md) | 固定每个后续任务 80 步，全量更新训练/开发 CE 为 0.478/2.093；零专家位移为 1.005/1.226 | 180 条开发轨迹来自 6 roots；零专家位移仍训练前端和任务头；不能推出专家无用 |
| [v11 RL 学习诊断](experiments/v11_rl_diagnostics/结果解读.md) | 协调控制器 24/48 轮、两个种子的确定性轨迹均与固定零位移相同；在线达标 4/4，F=0.001300 | 344 条开发轨迹仅四个历史根；局部控制器种子间不一致；没有相对固定规则的稳定协调优势 |

v1–v7 是固定几何路由的合成回归；v8/v9/v11 是固定原始图像路由、任务 ID 可见的多头 CIFAR-100 分类。不同版本的指标、任务和预算不同，不能把其数值合并成统一排名。

**v10 是可选扩展，不是本次交付的前置条件。** 原生 OLMoE 的开发入口、数据准备和 CPU 小模型机制检查已完成，尚未加载 7B 预训练权重做 GPU 实验。继续前需补齐决策目标、独立历史评价、测量分辨率与预算安排；逐 expert 控制器研究已在 v11 CPU 环境继续，GPU 保持暂缓。[v10 说明](experiments/v10_native_moe_feedback/README.md) · [预检与继续条件](docs/v10-preflight.md)。

v11 共运行四次控制器训练和 344 条开发轨迹，用时约 409 秒，记录 55,510 次物理学习器更新；仅使用官方训练数据。后续需检验状态依赖动作是否有超出固定倍率的收益，不能把目前学到的保守动作称为专家协调优势。

## 研究设计与讨论记录

先读英文技术报告可了解完整证据链；下面的方案文档保留研究设计及可选后续，不代表已经执行。2026-09-17 确定多层逐 expert 设计，2026-10-02 完成 v8 pilot 与 v9 诊断。结构化探索、匹配的多层 PPO、充分调优的规则、主动监测和语言模型验证仍未完成。

- [英文技术报告](docs/technical-report.md)：问题、方法、关键正负结果、局限与复现。
- [总体方案与实施主线](docs/mainline-plan.md)：研究主题、三层证据路线、评价/数据协议和 pilot 成本。
- [研究范围与已有证据](docs/research.md)：核心问题、v1–v9 结论与边界。
- [控制器设计](docs/controller-design.md)：目标、输入/动作、共享架构、GRPO 适配、训练数据与实验对照。
- [讨论决策与来源边界](docs/discussion-decisions-2026-09-17.md)：已确定原则、外部意见的采纳/修正、用户报告但未在仓库复现的早期结果。
- [带注释的参考文献](docs/references.md)：20 条文献与平台线索、原始链接、用途与核实状态。
- [真实 MoE 路由待办](docs/future-real-moe-routing.md)：当前路由与历史依赖的后续改进。
- [v8 校验记录](docs/v8-validation.md)：真实数据 pilot、机制测试、完整成本与结果边界。
- [v9 校验记录](docs/v9-validation.md)：独立停止轨迹、模块冻结、泛化诊断与物理成本。

## 快速开始

已有结果可直接阅读，无需重新训练。原始运行使用 Python 3.9.6、CPU 单线程；v1–v7 不需要外部数据集或 GPU。以下检查适用于 v1–v9 与 v11：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/test_all.py
```

共 129 项实验机制与统计检查，各版本在独立进程中测试，避免同名模块串用；归档校验覆盖 885 个文件。v8/v9/v11 重新运行另需官方 CIFAR-100（约 161 MB），不需要新增 Python 依赖或 GPU。可选 v10 使用单独依赖，`python scripts/test_all.py --include-v10` 共运行 157 项检查；CPU 检查不构成真实语言模型实验。

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
| [v8](experiments/v8_multilayer_retention/) | 真实图像、多层逐 expert、固定总记忆下的全历史保留 | pilot 跑通；两个 GRPO 学习臂选中初始化并执行全 1，未验证协调优势 | [报告](experiments/v8_multilayer_retention/results/pilot/analysis/实验报告.md)、[解读](experiments/v8_multilayer_retention/结果解读.md) |
| [v9](experiments/v9_learning_pressure/) | 更严目标与模块冻结能否揭示专家更新的必要性 | 180 条开发轨迹显示全量专家更新存在明显训练—开发差距；未验证其必要性 | [报告](experiments/v9_learning_pressure/results/development/analysis/实验报告.md)、[解读](experiments/v9_learning_pressure/结果解读.md) |
| [v10 可选](experiments/v10_native_moe_feedback/) | 原生语言 MoE 的指定旧域反馈是否改善专家组更新决策 | 开发入口和 CPU 机制检查完成；暂缓真实 GPU 实验，待决策与测量条件补齐 | [说明](experiments/v10_native_moe_feedback/README.md)、[继续条件](docs/v10-preflight.md) |
| [v11](experiments/v11_rl_diagnostics/) | 延长训练后，逐 expert RL 是否改变部署动作并改善保留 | 更新确实发生；协调控制器与固定零位移同轨迹，局部种子不一致；未验证动态协调优势 | [解读](experiments/v11_rl_diagnostics/结果解读.md)、[完整报告](experiments/v11_rl_diagnostics/results/development/analysis/研究诊断报告.md) |

v1–v9 与 v11 的实验源码与记录完整保留，版本之间有意保留冻结的模型副本。`docs/import_manifest.json` 校验前三轮导入文件；原有 `locked_protocol.json` 和各版本校验清单也保留。v3 阈值文字错误见 [勘误](docs/v3-erratum.md)，v4 的机制解释见 [结果解读](experiments/v4_state_aggregation/结果解读.md)。新多层环境的当前路由要求见 [后续待办](docs/future-real-moe-routing.md)，不改动归档实验。

## 复现入口

从仓库根目录运行；每次使用新的输出目录，已有结果不会被覆盖：

```bash
python scripts/run_experiment.py v7 --out runs/v7-reproduction
python scripts/run_experiment.py v8 --pilot --download --out runs/v8-pilot
python scripts/run_experiment.py v9 --out runs/v9-development
python scripts/run_experiment.py v11 --out runs/v11-reproduction
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
- 已归档合成实验的遗忘指标是独立旧任务评估 MSE 的正增量，不是百分比遗忘率；新主线的全部历史评价见总体方案。
- v7 三个 B 分别达标不等于最后仍全部保留；A 记忆固定，B 最终保留另报。
- 达到统一 loss 阈值不等于端点 loss 数字严格相同，具体报告保留敏感性分析。
- v2–v7 动作作用于 Adam 的实际参数位移；路由和初始保护组固定。
- v8 同样控制实际位移；任务 1 后控制两层各 8 个专家，CNN/分类头正常更新。在线监测达标与最终测试达标分开，完整矩阵区分初始泛化差距和后续遗忘。
- v9 仅使用开发数据；early-stop 与固定预算独立运行，首次命中和末点达标分开。额外开发评价不进入动作或停止规则，共同前缀物理开销只计一次。
- 预算使用明确的工作量代理，并非精确 FLOPs 或相同墙钟时间。
- 已查看的最终测试历史不可用于下一轮调参；新设计应使用新的测试历史。

详细研究范围与后续问题见 [研究说明](docs/research.md)。本仓库尚未指定开源许可证。
