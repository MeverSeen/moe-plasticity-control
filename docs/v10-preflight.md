# v10 启动记录：原生语言 MoE

2026-10-09。工程入口、真实语料准备和 CPU 机制检查已经完成；尚未执行 OLMoE 7B 预训练权重的 GPU 实验。暂缓的原因是科学设计与预算门槛尚未闭合，不是单纯等待 SSH 或服务器连接。真实模型的显存、吞吐、学习稳定性、泛化和反馈收益均未验证。

## 已完成验证

- 本地可用 Transformers 4.57.1、Torch 2.8.0；使用真实 Hugging Face `OlmoeForCausalLM` 架构的随机小模型完成工程验证。
- 精确参数冻结、FP32 master AdamW 位移、零倍率 moments、BF16 舍入、下游冻结模块梯度传递、已有 optimizer moments 的重复恢复均覆盖。
- 检查点恢复不会别名原保存的优化器张量，冻结参数的对象、存储与版本变化会被拒绝。冻结 fingerprint 按版本缓存；本工程不使用绕过版本追踪的 `.data` 写入冻结参数。
- 文档/source 分组、重复文档、WikiText 标题变体与重复标题、CPython 包隔离、padding、FP32 token 加权 CE、Python/NumPy/Torch/CUDA RNG 保存恢复、相同块数量但内容不同的数据流恢复拒绝均有检查。
- 28 项 v10 测试及原有 112 项测试通过；原有 839 个归档文件校验通过。
- CPU smoke 完成 6 条基线、20 条配对分支、64 次实际更新；批次配对、动作网格、源码哈希与物理 token/call 计数审计通过。其状态为 `engineering_only`，不得引用随机 token loss 作为研究结果。

## 真实语料

已取得固定版本 WikiText-103 raw 第一训练分片及官方 CPython 3.11.9 源码 archive，构建 1,651 个文档：A 995 篇百科文章、B 656 个代码文件。语料和输入 archive 位于仓库 `.cache/v10-corpus/`，不提交原文。

随后按规范化全文再移除 2 个重复 B 文档，进入划分的 B 文档为 654 个。固定 seed 1701 的文档数量：

| 域 | train | monitor | selection | test |
|---|---:|---:|---:|---:|
| A：百科 | 697 | 101 | 98 | 99 |
| B：CPython | 466 | 46 | 48 | 94 |

数据准备仅读取 test 文本用于去重和划分完整性，不将其 token 化或交给模型评价。source-family split 和精确去重不能保证无近重复、无模型预训练重叠，正式确认前还需审查。

已使用固定 revision 的真实 OLMoE 分词器完成 train/monitor/selection 检查：A 训练池为 2,557,703 个可预测 token，B 为 2,437,532 个；测试文档未 token 化。该步骤只验证数据处理，没有加载模型权重或执行模型评价。

工程记录见 [CPU smoke 审计](../experiments/v10_native_moe_feedback/results/cpu_smoke/integrity.json)、[语料来源](../experiments/v10_native_moe_feedback/results/data_preparation/provenance.json)与[分词检查](../experiments/v10_native_moe_feedback/results/data_preparation/tokenization_check.json)。

## 继续条件与验证顺序

当前 [protocol.json](../experiments/v10_native_moe_feedback/protocol.json)只锁定开发入口和测量语义。以下事项仍是**未完成工作**，并没有被写成代码安全门禁：

- CPU 准备：定义 B 的学习目标与失败处理；用人工数据检查已知有效、无效，以及只改善数值预测却不改善动作排序的反馈。开发校准后，另锁定分支可行性、并列和 ranking-regret 规则。
- CPU 准备：明确整条 root/history 的训练、开发与确认划分，以及校准搜索和单次 GPU 试点的成本上限。两个开发 roots 不能用于可靠确认，共享语料也不提供独立领域证据。
- 最小 GPU 试点：确认 BF16 模型实际位移能区分动作，而非只有 FP32 master 改变；检查 A/B 的专家路由覆盖、B 训练梯度和实际更新。现有日志只有 B 训练路由，A 覆盖诊断仍待补齐。
- 最小 GPU 试点：测量加载与运行峰值显存、host RAM、检查点捕获/恢复、评价和训练成本。它们只能在 GPU 上测量，不能冒充已经完成的 CPU 检查。
- 扩大开发前：确认 B 独立评价上的学习与测量分辨率，补齐长任务中断恢复和失败记录，再按实测成本决定 root、分支与评价预算。当前仅有内存中的分支恢复，尚无完整持久化续跑入口。

当前 CLI 的 `--mode preflight` 是端到端工程预检：三条两步基线，加上有 moments 的共同状态上的五个单步分支，共 **11 次 learner update**；A/B 的 monitor 与 selection 评价合计 **576 次 block 前向**。11 次更新不足以验证学习效果或反馈价值。24GB 显存卡可作为单卡预检的起点，不是已经实测的可运行保证；八张卡也不会被该单进程入口自动聚合。

因此 v10 目前是可选的后续方向，而不是 v1–v9 成果包的前置条件。不会把两个开发 roots、CPU smoke 或将来的短预检当作足以检验反馈增益的样本。整体研究脉络见 [技术报告](technical-report.md)，工程范围见 [v10 README](../experiments/v10_native_moe_feedback/README.md)。
