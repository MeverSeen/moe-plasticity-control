# 参考文献与设计对应

整理/核对日期：2026-09-17。保存项目讨论实际用到的研究线索，不是穷尽式综述，也不支持“首次提出”的优先权声明。

核实等级分别为原文相关章节、原始摘要/官方条目、待核实。核实链接和关键方法不等于独立复现结果；只有摘要的资料不推断其完整实现。以下“对本项目的启发”均是我们的设计判断。

## 研究线索索引

| 线索 | 文献 | 用途 |
|---|---|---|
| 架构/资源控制 | [RCL](#rcl)、[RMM](#rmm)、[Replay Scheduling](#replay-scheduling)、[自动样本选择](#automatic-memory)、[L2R 待核实](#l2r) | 对照控制粒度，设计外层环境与元训练数据 |
| 学习更新规则 | [La-MAML](#la-maml)、[Attention 元优化器](#attention-meta-optimizer)、[VeLO](#velo)、[Celo](#celo) | 学习型更新的直接前史、共享架构、任务增强与非 RL 对照 |
| MoE 持续学习 | [CP-MoE](#cp-moe)、[SETA](#seta)、[Agentic MoE 摘要](#agentic-moe) | 专家保护/结构动作、强方法基线与失败边界 |
| 策略学习 | [PPO](#ppo)、[GRPO](#grpo)、[经典控制 GRPO 研究](#grpo-classical)、[DAgger](#dagger) | 训练算法、采样与分布偏移 |
| 路由基础 | [Switch Transformer](#switch) | 区分冻结 router、冻结输入表示和真正不变的路由 |
| 规模、数据与平台 | [DeepSeek-V4.1-Flash](#deepseek-v41)、[Super-NaturalInstructions](#superni)、[OLMoE](#olmoe) | 大规模动机、语言任务划分、原生 MoE 验证候选 |

## 持续学习的外层控制

<a id="rcl"></a>
### R01 — Reinforced Continual Learning

- Ju Xu、Zhanxing Zhu；NeurIPS 2018。
- 来源：[正式论文](https://papers.nips.cc/paper_files/paper/2018/file/cee631121c2ec9232f3a2f028ad5c89b-Paper.pdf)。
- 核实：原文方法。
- 控制器输出各层新增节点/filters 的数量；保留旧参数，训练新增容量；用性能与复杂度评价决策。
- 启发：外层动作经过真实 learner 训练后才能得到回报；结构化低维动作可作用于大模型。
- 边界：扩容且冻结旧网络，不是固定容量下逐 expert 参数可塑性控制。

<a id="rmm"></a>
### R02 — RMM: Reinforced Memory Management for Class-Incremental Learning

- NeurIPS 2021；所读 arXiv 版本发表于 2023，勿把两者混为同一日期。
- 来源：[原文 §4](https://arxiv.org/html/2301.05792v1)。
- 核实：原文状态、分层动作、伪任务与策略梯度训练。
- 一级决定新/旧类记忆分配，二级决定类内配置；构造可获得训练/验证数据的伪持续学习任务，对任务重复运行以估计累计回报，再迁移策略。
- 启发：离线任务族元训练、重复 rollout、结构化联合动作和部署冻结控制器。
- 边界：原文是带回报基线的策略梯度，不是 GRPO；多次任务运行也不能自动视为我们严格同 checkpoint 克隆。额外离线训练有成本。

<a id="replay-scheduling"></a>
### R03 — Learn the Time to Learn: Replay Scheduling in Continual Learning

- Marcus Klasson、Hedvig Kjellström、Cheng Zhang；TMLR 2023，arXiv 初稿 2022。
- 来源：[摘要](https://arxiv.org/abs/2209.08660)、[全文及附录](https://arxiv.org/pdf/2209.08660)。
- 核实：原文方法、泛化实验设置、启发式与环境预计算描述。
- 控制各历史任务的 replay 比例；使用验证表现描述状态。先用搜索研究调度空间，再训练可迁移 RL 策略；改变初始化/任务顺序，并测试未见条件。
- 启发：分别检验动作空间机会和策略学习；比较充分调优的 Global Drop、Local Drop、Accuracy Threshold 等反馈规则。
- 边界：其训练包含预计算并缓存 learner 路径，策略 episode 数不等于新执行的真实训练量；任务平均准确率目标不能原样替代我们的达标约束和遗忘目标。

<a id="automatic-memory"></a>
### R04 — Continual Semantic Segmentation With Automatic Memory Sample Selection

- Lanyun Zhu 等；CVPR 2023，3082–3092 页。
- 来源：[CVF 官方条目](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Continual_Semantic_Segmentation_With_Automatic_Memory_Sample_Selection_CVPR_2023_paper.html)、[原文 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_Continual_Semantic_Segmentation_With_Automatic_Memory_Sample_Selection_CVPR_2023_paper.pdf)。
- 核实：官方摘要与论文摘要；本次直接网页请求曾返回 403，官方检索索引可读取。
- 将记忆样本选择作为决策，使用状态表示与两阶段动作，在 reward set 上评价。
- 启发：样本级控制也可以是结构化动作；区分训练、在线观测与 reward 数据。
- 边界：控制样本而非参数位移；对话中的增强细节、确切动作构造仍需方法章节核对，本文不据此提出定量结论。

<a id="l2r"></a>
### R05 — Learning to Replay（L2R，待进一步核实）

- 对话提供的题名：Learning to Replay: A meta-reinforcement learning approach to experience replay in continual learning。
- 原始线索：[出版方页面](https://www.sciencedirect.com/science/article/pii/S0031320326017097)。
- 状态：本次未能读取出版方页面。对话称其联合学习 replay 时机、强度与内容，但本记录不将这些细节视为已由原文核实。
- 后续需核对：正式作者、在线/卷期年份、DOI、动作、PPO 与内容策略的具体训练，以及可用数据权限。
- 使用边界：保留为待阅读线索，不用于支持当前设计的关键事实，不沿用“最新已证实方法”的说法。

## 学习优化器与更新控制

<a id="la-maml"></a>
### R06 — La-MAML: Look-ahead Meta Learning for Continual Learning

- Gunshi Gupta、Karmesh Yadav、Liam Paull；NeurIPS 2020。
- 来源：[论文条目](https://arxiv.org/abs/2007.13904)、[全文 §5.2 与附录 G](https://arxiv.org/pdf/2007.13904)。
- 核实：方法说明及视觉任务/模型/记忆相关配置。
- 使用元学习与逐参数学习率调节处理持续学习中的干扰，并结合记忆。
- CIFAR-100 为 20×5 类任务、TinyImageNet 为 40×5 类，回放上限分别为 200/400；多遍设置最多 10 epochs/任务，任务增量多头分类。
- 启发：更新幅度学习有直接前史；可借任务协议，以元梯度作非 RL 参照。
- 边界：更换为 MoE 并采用 loss 达标停止后属于协议适配，不是原模型复现；不能据此宣称逐 expert GRPO 有效。

<a id="attention-meta-optimizer"></a>
### R07 — Learning to learn without forgetting using attention

- Anna Vettoruzzo 等；所读 arXiv v2，2024。
- 来源：[原文 §4、§5 与附录](https://arxiv.org/html/2408.03219v2)。
- 核实：元优化器输入、预测权重更新、support/query 与任务设置。
- 用任务材料表示、权重及梯度相关重要性信号产生参数更新，利用注意力建模关系。
- 启发：这是学习更新控制的直接相关方法；可借鉴材料响应编码与内外层数据分离。
- 边界：其 N-way 适应/分类头协议及被控制参数范围，与固定容量原生 MoE、固定旧记忆、等新任务 loss 的目标不同。原文测试时包含额外适应；本项目先测不重新训练的保留能力，另列重新学习实验，不能混比。

<a id="velo"></a>
### R08 — VeLO: Training Versatile Learned Optimizers by Scaling Up

- Luke Metz 等；所读 arXiv v1，2022。
- 来源：[原文 §3.1、附录 B](https://arxiv.org/html/2211.09760v1)。
- 核实：分层网络、全局上下文与元训练方法。
- 参数级小 MLP 与张量级 LSTM 分工，张量模块通过池化上下文协调；其元训练使用演化策略估计平滑元目标的梯度。
- 启发：共享局部处理＋全局协调，规模扩展不必为每个 expert 复制 agent。
- 边界：不是持续学习保护实验，也不是 GRPO；架构可迁移不代表性能结论可迁移。元训练成本必须计入适用性。

<a id="celo"></a>
### R09 — Celo: Training Versatile Learned Optimizers on a Compute Diet

- 所读 arXiv v2，2025。
- 来源：[原文 §3、附录任务增强与 MLCommons 讨论](https://arxiv.org/html/2501.12670v2)。
- 核实：重参数化增强、两阶段训练及大规模迁移边界。
- 通过参数重缩放增加动力学多样性；先学习更新规则，再冻结规则训练步长调度。作者报告从小型图像 MLP 训练的完整 Celo 在 MLCommons 任务上不稳定。
- 启发：增加元训练历史/动力学覆盖，分开学习不同自由度；优先使用尺度合理的可迁移摘要。
- 边界：不能把不同参数化副本当作独立测试历史；小任务有效不保证大 MoE 泛化。外部意见提到的“独立 VeLO 评估”尚缺明确条目，不在此转述为已核实结论。

## MoE 持续学习与保护机制

<a id="cp-moe"></a>
### R10 — CP-MoE: Consistency-Preserving Mixture-of-Experts for Continual Learning

- Yang Liu、Toan Nguyen、Flora D. Salim；所读 arXiv v2，2026，官方条目标注 CoLLAs 2026 接收。
- 来源：[官方摘要](https://arxiv.org/abs/2605.20247)、[全文 §4 与附录 B](https://arxiv.org/html/2605.20247v2)。
- 核实：机制及语言实验相关配置；尚未完成代码复现和同预算实现核对。
- 临时专家承接新任务更新，指导稳定专家的路由与整合，并通过正则保护历史参数。
- 语言设置以冻结 Llama-2-7B 加 LoRA experts；每适配模块 8 experts、rank 4，覆盖 q/k/v/o 与 gate/up/down。SuperNI 连续 8 任务与另外 7 个零样本任务分开评价。
- 启发：语言阶段的模型和任务底座，详细起步配置见 [总体方案](mainline-plan.md)。
- 边界：适配 experts 不等于原生预训练 MoE experts；原方法的临时专家、预热、整合和路由偏置不属于我们的单纯倍率接口。完整复现与机制适配必须分别命名并计费。

<a id="seta"></a>
### R11 — Sparse Subspace-to-Expert Sharing for Task-Agnostic Continual Learning（SETA）

- Fatema Siddika 等；所读 arXiv v1，2026。
- 来源：[原文 §3](https://arxiv.org/html/2606.07500v1)。
- 核实：共享/独有专家、弹性锚定和路由正则的相关章节。
- 根据跨任务重叠形成共享与独有专家；共享保护结合历史承担任务与实时权重位移，另有路由约束。
- 启发：强规则也可以同时使用历史与当前反馈。可借鉴状态依赖的保护强度，而非只比较静态冻结。
- 边界：不要把 SETA 描述为静态规则；其子空间构造、专家拆分和路由演化与本项目不同，完整方法不自动满足相同容量预算。

<a id="agentic-moe"></a>
### R12 — Agentic Mixture-of-Experts for Continual Learning

- Navyatha Gandlaparthi；Iowa State University 硕士答辩活动摘要，2026-07-15。
- 来源：[学校官方页面](https://www.cs.iastate.edu/event/2026/ms-final-oral-exam-navyatha-gandlaparthi)。
- 核实：只有官方活动摘要，不是本记录已读的完整论文或同行评审结果。
- 描述任务切换时逐专家 Share/Clone/Freeze 与 OGP；摘要称两次任务转换下 PPO 未明显胜过简单控制器。
- 启发：相关结构动作已有人探索；模型经验量与控制机会要实测。
- 边界：不能据此断言 PPO 普遍失败，或把作者给出的经验不足解释当作已完成因果证明。

## 学习算法、反馈与序列决策

<a id="ppo"></a>
### R13 — Proximal Policy Optimization Algorithms

- John Schulman 等；2017。
- 来源：[原论文](https://arxiv.org/abs/1707.06347)。
- 核实：原始算法说明。
- 启发：行为策略采样、概率比裁剪与可重复利用的批次更新；作为与 GRPO 同接口、同 learner 交互预算的对照。
- 边界：PPO 本身不保证约束可行性、状态充分或长时信用分配成功；价值网络成本与 learner 训练成本分开。

<a id="grpo"></a>
### R14 — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

- Zhihong Shao 等；所读 arXiv v3，2024，GRPO 原始出处。
- 来源：[原文 §4.1](https://arxiv.org/html/2402.03300v3)。
- 核实：同问题多输出采样、组内优势、概率比裁剪及参考策略正则。
- 启发：把同问题/多回答迁移为同完整学习起点/多闭环轨迹，以组内结果训练小控制器，省去独立 critic。
- 边界：原论文不是我们的控制环境。终局优势不自动解决专家/时间信用分配；没有 critic 不代表 learner 交互更省。分母下限、仅减均值和约束处理是本项目候选改动，需与原始 GRPO 区分。

<a id="grpo-classical"></a>
### R15 — Learning Without Critics? Revisiting GRPO in Classical Reinforcement Learning Environments

- 所读 arXiv v1，2025。
- 来源：[原文 §5–8](https://arxiv.org/html/2511.03527v1)。
- 核实：实验设置与分组限制。
- 在经典控制中比较 GRPO 式方法与 PPO 等基线，讨论长时信用分配、分组和样本效率；实际采用并行环境批次内分组，可能混合不同初始状态。
- 启发：保留匹配成本的 PPO，对照组大小与更新频率；严格记录“同起点”的含义。
- 边界：它没有直接测试我们严格同 learner checkpoint 克隆的分组方案，不能据此断言本项目 GRPO 会成功或失败。

<a id="dagger"></a>
### R16 — A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning（DAgger）

- Stéphane Ross、Geoffrey Gordon、J. Andrew Bagnell；AISTATS 2011。
- 来源：[PMLR 正式条目及论文](https://proceedings.mlr.press/v15/ross11a.html)。
- 核实：原文算法动机与正式摘要。
- 行为会改变随后输入分布；在学习器访问状态上补专家监督并聚合数据，用于减轻离线模仿的误差累积。
- 启发：解释为什么动作预测改善不一定带来闭环改善；v4 的状态聚合是沿这条思路做的诊断。
- 边界：本项目分支教师不是已证明最优专家，标签/观测不足时聚合未必奏效；不能借 DAgger 理论直接保证我们的策略胜出。

## 路由基础

<a id="switch"></a>
### R17 — Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

- William Fedus、Barret Zoph、Noam Shazeer；arXiv 初稿 2021。
- 来源：[路由公式与架构](https://arxiv.org/html/2101.03961v3)。
- 核实：原始路由定义。
- 启发：路由由 router 与其输入表示共同决定。冻结 router 参数不保证上游训练后路由不变；固定 router 且固定输入表示时才可把其样本级分配视为固定。
- 边界：usage/门控权重不等同于重要性、实际损伤或充分状态；更不证明本项目反馈控制有效。

## 规模、数据与平台

<a id="deepseek-v41"></a>
### R18 — DeepSeek-V4.1-Flash 官方模型配置与实现

- 来源：[官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)、[官方推理实现](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)。
- 核实：主干层数、每层 routed/shared experts、top-k 及 Block 的 MoE 构造；未进行训练复现。
- 主干 40 层、每层 384 个路由专家和 1 个共享专家，每 token 选择 6 个路由专家；主干共 15,360 个路由专家、15,400 个专家模块，不含附加推测解码等模块。
- 启发：共享分层控制和结构化探索的规模动机；逐 token 稀疏激活不代表整个训练 batch 只覆盖少量 experts。
- 边界：专家数量不是独立技能数量，也不证明规则失效。不是本项目当前实施平台；如将来选用，需锁定实际配置/代码 revision。

<a id="superni"></a>
### R19 — Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks

- Yizhong Wang 等；2022。
- 来源：[原始论文](https://arxiv.org/abs/2204.07705)。
- 核实：数据集原始说明，1,616 个任务、76 种任务类型。
- 启发：从语言任务族产生元训练历史，按任务身份/来源隔离训练与最终测试；可借 CP-MoE 的指定任务清单作可比序列。
- 边界：换顺序不等于未见任务；不同任务 ID 也可能共享来源和近重复样本。具体数据版本、任务 ID、许可与拆分在实际采用时核对锁定。

<a id="olmoe"></a>
### R20 — OLMoE-1B-7B-0924 官方模型与配置

- Allen Institute for AI，2024。
- 来源：[官方模型卡](https://huggingface.co/allenai/OLMoE-1B-7B-0924)、[配置](https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/config.json)。
- 核实：公开模型卡与配置；未在本项目下载训练。
- 配置为 16 层、每层 64 experts、每 token 激活 8 个，共 1,024 个 expert 位置。
- 启发：区别于稠密基座上的 LoRA experts，可作为原生 MoE 验证平台候选。
- 边界：模型名称中的激活参数规模不等于训练总显存；实际优化器状态、可训练层、复制和观测开销需测量。控制原生 expert 权重与控制新增 adapter 的结论不同。

## 阅读与引用约定

1. 可行性动机、方法借鉴、直接基线、局限证据分别使用，不让一个来源承担它未验证的结论。
2. 本工程提出的 GRPO 环境映射、动作集合、奖励和实验矩阵见 [设计文档](controller-design.md)，分层实验与数据预算见 [总体方案](mainline-plan.md)，不能写成文献已给出的原方案。
3. 新增论文先补原始来源及核实等级；尤其核对在线发表年与卷期年、预印本与会议版本。
4. 后续正式论文写作前补全作者/出版信息及准确 BibTeX，逐项复查方法对齐；本表是注释式研究书目，不是假定已完成全部基线复现。
