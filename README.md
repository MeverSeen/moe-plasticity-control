# MoE Plasticity Control

研究持续学习中的专家更新控制：在固定 MoE 容量、有限旧记忆与可核算的计算成本下，用共享控制器协调逐 expert 的实际更新，降低达到同一新任务目标时的累计旧知识退化。

**启动更新（2026-10-08）：v10 原生语言 MoE 的协议、数据准备和开发入口已实现。** CPU 随机小型 OLMoE 的机制检查通过；真实语料已准备。7B 预训练模型的 GPU 预检等待服务器连接，尚无新研究结论。[v10 说明](experiments/v10_native_moe_feedback/README.md) · [启动校验](docs/v10-preflight.md)。

**当前状态（2026-10-02）：已完成 v8 真实图像多层 MoE pilot，以及 v9 的 180 条学习压力与模块干预开发轨迹；尚未验证学习控制器的优势。** v9 中更严的新任务目标没有使全量专家更新占优。固定 80 步时，全量更新训练/开发 CE 为 0.478/2.093，零专家位移且 stem/任务头可训练时为 1.005/1.226，提示当前配置的泛化与更新稳定性问题。仅开发数据、6 个共享类别池 roots，不能推广为专家无用。[v9 解读](experiments/v9_learning_pressure/结果解读.md) · [学习曲线](experiments/v9_learning_pressure/results/development/analysis/learning_curves.png)。

v8 实现两层逐 expert 更新、固定总记忆、全部历史表现矩阵和局部/协调 GRPO。两个学习臂均选中初始化并执行全量更新；冻结 experts 遗忘较少，但实际更新数多 25%。仅 3 个 audit 根历史，不能作正式优越性结论。[v8 解读](experiments/v8_multilayer_retention/结果解读.md)。

此前 v7 段内 Feedback 在共同达标历史上的 A 遗忘降低约 13.8%，但达标率非劣和多重比较稳健性尚未验证。v1–v7 保持原样。v8 使用固定原始图像路由、任务 ID 可见的多头分类，不代表 learned-router 或语言模型验证。

## 研究设计与讨论记录

2026-09-17 确定多层逐 expert 主线，2026-10-02 在 v8 完成第一版小规模实现。层/全局池化与因子化 GRPO 已进入 pilot；结构化探索、PPO、充分调优的规则、主动监测和语言平台仍待开展。目标 loss、累计遗忘、有限记忆与泛化协议在开发阶段锁定。

- [总体方案与实施主线](docs/mainline-plan.md)：研究主题、三层证据路线、评价/数据协议和 pilot 成本。
- [研究范围与已有证据](docs/research.md)：核心问题、v1–v7 结论与边界。
- [控制器设计](docs/controller-design.md)：目标、输入/动作、共享架构、GRPO 适配、训练数据与实验对照。
- [讨论决策与来源边界](docs/discussion-decisions-2026-09-17.md)：已确定原则、外部意见的采纳/修正、用户报告但未在仓库复现的早期结果。
- [带注释的参考文献](docs/references.md)：20 条文献与平台线索、原始链接、用途与核实状态。
- [真实 MoE 路由待办](docs/future-real-moe-routing.md)：当前路由与历史依赖的后续改进。
- [v8 校验记录](docs/v8-validation.md)：真实数据 pilot、机制测试、完整成本与结果边界。
- [v9 校验记录](docs/v9-validation.md)：独立停止轨迹、模块冻结、泛化诊断与物理成本。

## 快速开始

以下为已归档合成实验的复现入口。推荐 Python 3.11；原始运行使用 Python 3.9.6、CPU 单线程，这部分无需外部数据集或 GPU。新主线的真实数据与硬件需求另见总体方案。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/test_all.py
```

共 112 项实验机制与统计检查，各版本在独立进程中测试，避免同名模块串用。已有记录在 `experiments/*/results/`，首次查看结果无需重新训练。v8/v9 首次运行另需下载官方 CIFAR-100（约 161 MB），不需要新增 Python 依赖或 GPU。

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
| [v10](experiments/v10_native_moe_feedback/) | 原生语言 MoE 的指定旧域反馈是否改善专家组更新决策 | 开发入口和 CPU 机制检查完成；GPU 真实模型实验待开始 | [启动说明](experiments/v10_native_moe_feedback/README.md)、[校验](docs/v10-preflight.md) |

七轮实验源码与记录完整保留，版本之间有意保留冻结的模型副本。`docs/import_manifest.json` 校验前三轮导入文件；原有 `locked_protocol.json` 和各版本校验清单也保留。v3 阈值文字错误见 [勘误](docs/v3-erratum.md)，v4 的机制解释见 [结果解读](experiments/v4_state_aggregation/结果解读.md)。新多层环境的当前路由要求见 [后续待办](docs/future-real-moe-routing.md)，不改动归档实验。

## 复现入口

从仓库根目录运行；每次使用新的输出目录，已有结果不会被覆盖：

```bash
python scripts/run_experiment.py v7 --out runs/v7-reproduction
python scripts/run_experiment.py v8 --pilot --download --out runs/v8-pilot
python scripts/run_experiment.py v9 --out runs/v9-development
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
