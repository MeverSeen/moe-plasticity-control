# v10 启动记录：原生语言 MoE

2026-10-08。已实现开发入口，尚未执行 7B 预训练权重的 GPU 实验。当前 Mac 没有 CUDA；未取得 8×4090 服务器连接方式。真实模型显存、吞吐、学习效果和反馈收益均未验证。

## 已完成验证

- 本地可用 Transformers 4.57.1、Torch 2.8.0；使用真实 Hugging Face `OlmoeForCausalLM` 架构的随机小模型完成工程验证。
- 精确参数冻结、FP32 master AdamW 位移、零倍率 moments、BF16 舍入、下游冻结模块梯度传递、已有 optimizer moments 的重复恢复均覆盖。
- 检查点恢复不会别名原保存的优化器张量，冻结参数的对象、存储与版本变化会被拒绝。冻结 fingerprint 按版本缓存；本工程不使用绕过版本追踪的 `.data` 写入冻结参数。
- 文档/source 分组、重复文档、WikiText 标题变体与重复标题、CPython 包隔离、padding、FP32 token 加权 CE、Python/NumPy/Torch/CUDA RNG 保存恢复、相同块数量但内容不同的数据流恢复拒绝均有检查。
- 28 项 v10 测试及原有 112 项测试通过；原有 839 个归档文件校验通过。
- CPU smoke 完成 6 条基线、20 条配对分支、64 次实际更新；批次配对、动作网格、源码哈希与物理 token/call 计数审计通过。其状态为 `engineering_only`，不得引用随机 token loss 作为研究结果。

## 真实语料

已取得固定版本 WikiText-103 raw 第一训练分片及官方 CPython 3.11.9 源码 archive，构建 1,651 个文档：A 995 篇百科文章、B 656 个代码文件。语料和输入 archive 位于仓库 `.cache/v10-corpus/`，不提交原文。

固定 seed 1701 的文档数量：

| 域 | train | monitor | selection | test |
|---|---:|---:|---:|---:|
| A：百科 | 697 | 101 | 98 | 99 |
| B：CPython | 466 | 46 | 48 | 94 |

数据准备仅读取 test 文本用于去重和划分完整性，不将其 token 化或交给模型评价。source-family split 和精确去重不能保证无近重复、无模型预训练重叠，正式确认前还需审查。

已使用固定 revision 的真实 OLMoE 分词器完成 train/monitor/selection 检查：A 训练池为 2,557,703 个可预测 token，B 为 2,437,532 个；测试文档未 token 化。该步骤只验证数据处理，没有加载模型权重或执行模型评价。

工程记录见 [CPU smoke 审计](../experiments/v10_native_moe_feedback/results/cpu_smoke/integrity.json)、[语料来源](../experiments/v10_native_moe_feedback/results/data_preparation/provenance.json)与[分词检查](../experiments/v10_native_moe_feedback/results/data_preparation/tokenization_check.json)。

## 接下来的必要条件

取得 GPU 服务器 SSH 连接方式及工作目录后，运行已准备的 `--mode preflight`：真实 BF16 权重、选中约 100.7M 专家参数、六个基线更新与五个单步分支。测清单卡峰值显存、恢复耗时和学习曲线后再锁定资源配置。

当前 [protocol.json](../experiments/v10_native_moe_feedback/protocol.json)只锁定开发起点。新任务达标目标、分支 regret/可行性目标、独立历史划分、最小有意义效应和闭环确认方案仍待开发校准。不会把两个开发 roots 或 CPU smoke 当作足以检验反馈增益的样本。

复现说明和范围见 [v10 README](../experiments/v10_native_moe_feedback/README.md)。
