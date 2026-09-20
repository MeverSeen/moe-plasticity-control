# MoE Plasticity Control

**本轮有界结果（2026-09-20）：** 单种子PPO完成8条新轨迹、83决策、4批／12梯度步；确认7/8路径完成，四方法共同实跑仅n=1。冻结greedy策略恒等固定1，但同倍率执行出现未解轨迹分歧，不能把较低PPO数值归为控制收益；固定／规则方向也仅描述。第二根PPO因时间预算缺失，不是learner失败。收尾hash字典/字符串比较错误已修复，原终点完整SHA未落盘的限制保留。见[完整结果](experiments/v14_methods_20260920_0814/结果解读.md)。

**最新执行决定（2026-09-20 08:17 UTC）：** 用户已覆盖先前“今日不训练PPO”限定。在同一全层LoRA平台执行固定倍率／同信息规则／PPO三类主表：3训练根均衡轮转最多12条新轨迹、每2条更新；1开发根、2隔离确认根。阶段上限85/30/75分钟，12:40 UTC停止实验、留45分钟归档，原13:25硬截止不变。H=5、共享分层稀疏接口与严格终点奖励固定；仅补alpha=.5/.1的EMA趋势，不再做设计搜索。详见[锁定方法实验](experiments/v14_methods_20260920_0814/README.md)。这条是新执行授权，早先pilot未训练PPO的历史事实保持不变。

研究持续学习中的专家更新控制：在固定 MoE 容量、有限旧记忆与可核算的计算成本下，用共享控制器协调逐 expert 的实际更新，降低达到同一新任务目标时的累计旧知识退化。

**既有合成实验：已完成七轮，v7 连续三段学习出现段内旧知识反馈的初步正信号，但尚未完成同等可行性下的稳定验证。** v6 定位到模型筛选瓶颈；v7 保留参数和 Adam 状态连续学习三个新输入域。段内 Feedback 在共同达标历史上的 A 遗忘降低约 13.8%，但平均达标率略低，多重比较敏感性区间跨零。段间 Feedback 则提高达标率、同时增加遗忘。本仓库保留正负结果、原始数据与冻结协议，不代表真实 learned-router MoE 或语言模型上的验证。

## 当前平台：v14 全层原生 expert LoRA

2026-09-20将下一阶段固定为原生OLMoE的全16层×64 expert低秩适配控制。rank4 LoRA覆盖每个expert三个投影，共37,748,736训练参数，原始基座与router冻结；一条历史持续使用同一套适配器。全层插入、零初始化等价、真实位移及checkpoint恢复已经过A800实测，限额固定/规则比较已完成并归档。单根三路径均达标，规则未改善固定1的CE遗忘；固定.5遗忘更低但多用10次更新。

- [v14结果与限制](experiments/v14_full_layer_lora/结果解读.md)
- [v14平台、实测成本与当前范围](experiments/v14_full_layer_lora/README.md)
- [三类主表与独立通用评价协议](experiments/v14_full_layer_lora/EVALUATION_PROTOCOL.md)
- [v13两层原生权重有限证据](experiments/v13_native_ppo/结果解读.md)：171实际更新、单根三候选；因平台调整提前结束，正式PPO收益未检验。

单根v14短测阶段未训练PPO，不把短测当作全层优于两层的因果对照；随后同平台有界PPO比较已完成，见[方法结果](experiments/v14_methods_20260920_0814/结果解读.md)。预算沿用原分配，13:25 UTC硬截止不变；未执行方法明确标记。

## 已归档：v12 持续动作与联合控制

2026-09-20完成GRPO三组×四种子：A稀疏控制保持10步，B稀疏控制保持50步，C全部8个expert＋trunk联合控制保持50步。1,112条训练分支、244,420次主训练learner更新；本轮全部新增计算共530,410次更新，36个本地模型保留。

长保持减少约75%的训练决策调用，但遗忘改善区间仍跨零；C随机执行的选中模型F比B高.011509，未显示联合控制收益。11/12个greedy模型选中初始化，唯一改变的模型学到另一种固定trunk倍率。强静态参考F=.033779，优于所有组的选中模型均值，但需更多在线更新。全部主要策略10/12根整段达标，两次失败均发生在控制器介入前。开发类别池已用，不能称最终确认；v10/v11及更早版本保持冻结。

- [v12结果、逐种子比较与成本](experiments/v12_persistent_control/结果解读.md)
- [实现与锁定协议](experiments/v12_persistent_control/README.md)
- [v12核验记录](docs/v12-validation.md)

## 已归档：v11 CPU视觉诊断

2026-09-18完成运行，2026-09-20归档。96条奖励配对分支、96条模块范围路径、72条固定校准历史，以及6组×4策略种子的真实训练与复测（1152训练分支）。记录707,190次learner更新，v10代码、协议、模型及结果保留不变。

类别均衡奖励的排序诊断有所改善，尚未产生稳定学习收益；共享trunk的静态倍率.5在12个开发复测根上优于倍率1的平均遗忘，但约多用24%的更新。温启动确实改变随机分布，却仍有22/24个greedy模型选中初始化；未验证超越强静态共享基准的稳定收益。复测仍属已用开发类别池，不能称新类别最终确认。

- [v11结果、逐种子表格与成本](experiments/v11_diagnostics/结果解读.md)
- [实现与锁定学习协议](experiments/v11_diagnostics/README.md)
- [v11核验记录](docs/v11-validation.md)

## 已归档：v10 真实视觉闭环 pilot

2026-09-18 已完成 CIFAR-100 多层MoE的共享逐expert GRPO/PPO训练、粒度/信息/协调消融及独立评价，另检验训练轮数与历史覆盖。总计1,056条真实训练分支；含开发与评价318,250次learner更新。没有验证旧知识反馈增益或超过最佳固定/全局倍率；GRPO相对本轮PPO有探索性正信号，尚非稳定优越性结论。7/8根历史达标，未达到预设90%目标。

- [结果解读与图表](experiments/v10_visual_control/结果解读.md)
- [实现、协议及复现入口](experiments/v10_visual_control/README.md)
- [验证与归档审计](docs/v10-validation.md)
- [后续7B平台的资源缺口](experiments/v10_visual_control/RESOURCE_NOTES.md)

本机视觉pilot已完成；语言原生MoE与全层expert LoRA已在v13/v14接入GPU，证据范围见各自新归档，不追溯改变视觉结果。两组15动作v8已在独立分支 `codex/v8-allocation-ppo` / `9e54f99` 完成，本checkout未复制其归档；原v1–v7保持冻结。

## 研究设计与讨论记录

2026-09-20更新：固定原生OLMoE全层低秩适配控制平台，采用共享编码、层/组结构化接口，三类主表为固定分配、同信息规则、学习控制器（PPO候选）；克隆放诊断表。新任务表现、累计任务遗忘与独立通用能力变化分列。早期视觉/GRPO候选保留历史语境，不代表现在同时推进多个平台。

- [总体方案与实施主线](docs/mainline-plan.md)：研究主题、三层证据路线、评价/数据协议和 pilot 成本。
- [研究范围与已有证据](docs/research.md)：核心问题、v1–v7 结论与边界。
- [控制器设计](docs/controller-design.md)：目标、输入/动作、共享架构、GRPO 适配、训练数据与实验对照。
- [讨论决策与来源边界](docs/discussion-decisions-2026-09-17.md)：已确定原则、外部意见的采纳/修正、用户报告但未在仓库复现的早期结果。
- [带注释的参考文献](docs/references.md)：20 条文献与平台线索、原始链接、用途与核实状态。
- [真实 MoE 路由待办](docs/future-real-moe-routing.md)：当前路由与历史依赖的后续改进。

## 快速开始

以下为已归档合成实验的复现入口。推荐 Python 3.11；原始运行使用 Python 3.9.6、CPU 单线程，这部分无需外部数据集或 GPU。新主线的真实数据与硬件需求另见总体方案。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
python scripts/test_all.py
```

旧v1–v12共95项检查；v13新增27项、v14平台新增16项CPU机制/评价检查，后续同平台方法实验开训前7项接口/恢复/PPO检查，收尾修复新增1项hash比较检查。各版本在独立进程中测试，避免同名模块串用；本轮只重跑相应新增检查，旧结果保持冻结。已有记录在各版本的results或archive目录，首次查看无需重新训练。

## 实验索引

| 版本 | 研究问题 | 本轮结论 | 报告 |
|---|---|---|---|
| v14方法比较 | 同平台固定／规则／PPO | 8条训练轨迹、确认7/8；同动作分歧未解，未建立PPO收益 | [解读](experiments/v14_methods_20260920_0814/结果解读.md) |
| v14平台pilot | 全层原生expert LoRA平台、结构化控制与独立评价 | 接入/真实机制与成本已验证；该历史pilot未训练PPO | [范围](experiments/v14_full_layer_lora/README.md) |
| v13 | 两层原生权重结构化接口 | 单根工程比较；因平台调整提前结束，PPO收益未检验 | [解读](experiments/v13_native_ppo/结果解读.md) |
| v12 | 10/50步保持、全部expert与trunk联合控制 | 决策减少约75%；未验证稳定学习收益或超越强静态参考 | [解读](experiments/v12_persistent_control/结果解读.md) |
| v11 | 奖励排序、模块范围、概率温启动 | 共享静态倍率有效；奖励修正/温启动尚未带来稳定学习收益 | [解读](experiments/v11_diagnostics/结果解读.md) |
| [v1](experiments/v1_expert_control/) | PPO 控制专家更新与保护强度 | 未超过重要性规则基线 | [报告](experiments/v1_expert_control/实验报告.md) |
| [v2](experiments/v2_feedback_control/) | 固定保护组后，旧任务反馈是否改善标量 PPO 控制 | progress/feedback 最终都选中未训练的初始策略，未验证反馈增益 | [报告](experiments/v2_feedback_control/实验报告.md) |
| [v3](experiments/v3_branching/) | 配对分支动作价值与监督评分器能否提取反馈增益 | 一次干预和闭环主要比较的区间均跨零 | [报告](experiments/v3_branching/results/main/analysis/实验报告.md) |
| [v4](experiments/v4_state_aggregation/) | 学习器状态聚合能否把反馈信息转化为闭环收益 | 聚合未优于进度策略或等查询数对照；固定冻结保护组已全部达标 | [报告](experiments/v4_state_aggregation/results/main/analysis/实验报告.md) |
| [v5](experiments/v5_calibration/) | 不限定冲突形态时，固定选择和时序安排是否有可复测收益 | 固定选择参考有局部收益；加入切换计划未稳定胜过时间基线，全可行子集未胜过直接冻结 | [报告](experiments/v5_calibration/results/main/analysis/实验报告.md)、[解读](experiments/v5_calibration/结果解读.md) |
| [v6](experiments/v6_learning_diagnostics/) | 负结果是否来自学习能力、监督目标或模型筛选问题 | 基本学习检查通过；闭环筛选改善结果，但尚未验证实时反馈增益 | [报告](experiments/v6_learning_diagnostics/results/main/analysis/实验报告.md)、[解读](experiments/v6_learning_diagnostics/结果解读.md) |
| [v7](experiments/v7_sequence_feedback/) | 连续多段学习是否让段间或段内旧知识反馈更有价值 | 段内遗忘约 −13.8%，但可行性略低、校正敏感性跨零；段间呈达标/遗忘取舍 | [报告](experiments/v7_sequence_feedback/results/main/analysis/实验报告.md)、[解读](experiments/v7_sequence_feedback/结果解读.md) |

七轮实验源码与记录完整保留，版本之间有意保留冻结的模型副本。`docs/import_manifest.json` 校验前三轮导入文件；原有 `locked_protocol.json` 和各版本校验清单也保留。v3 阈值文字错误见 [勘误](docs/v3-erratum.md)，v4 的机制解释见 [结果解读](experiments/v4_state_aggregation/结果解读.md)。新多层环境的当前路由要求见 [后续待办](docs/future-real-moe-routing.md)，不改动归档实验。

## 复现入口

从仓库根目录运行；每次使用新的输出目录，已有结果不会被覆盖：

```bash
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
- 已归档合成实验的遗忘指标是独立旧任务评估 MSE 的正增量，不是百分比遗忘率；新主线的全部历史评价见总体方案。
- v7 三个 B 分别达标不等于最后仍全部保留；A 记忆固定，B 最终保留另报。
- 达到统一 loss 阈值不等于端点 loss 数字严格相同，具体报告保留敏感性分析。
- v2–v7 动作作用于 Adam 的实际参数位移；路由和初始保护组固定。
- 预算使用明确的工作量代理，并非精确 FLOPs 或相同墙钟时间。
- 已查看的最终测试历史不可用于下一轮调参；新设计应使用新的测试历史。

详细研究范围与后续问题见 [研究说明](docs/research.md)。本仓库尚未指定开源许可证。
