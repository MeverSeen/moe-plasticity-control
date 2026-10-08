# v10：原生语言 MoE 的旧域反馈与更新决策

状态：2026-10-08 已完成协议草案、实现和 CPU 机制检查；**尚未运行 OLMoE 7B 预训练权重的 GPU 实验，也没有反馈收益结论**。当前主机没有 CUDA，等待 GPU 服务器连接。v1–v9 的 839 个归档文件保持不变。

本轮问题是：在保护指定旧域 A、适配新域 B 时，在线 A 损失变化是否能改善一个固定专家参数组的更新决策，超过已声明的进度信息基线？A 是预训练模型的百科评价域，本阶段不另做 A 适配；B 是 Python 标准库代码。主张范围是指定旧域保留与标量专家组更新控制，不包含逐专家协调、全部历史保留或通用代码能力。

## 当前完成和待完成

| 部分 | 状态 |
|---|---|
| 精确冻结和选中专家的 FP32 master AdamW 位移控制 | 已实现；真实 HF OLMoE 架构上的离线小模型测试通过 |
| 检查点、优化器 moments、随机状态、数据流恢复 | 已实现；已有 moments 的重复分支及冻结权重完整性测试通过 |
| 文档/source 分组、去重、LM token 加权评价 | 已实现；真实语料已准备，测试角色不进入开发评价 |
| 完整固定倍率、时间调度与配对分支入口 | 已实现；CPU 随机小模型完成 6 条基线和 20 条分支 |
| 7B 真机显存、吞吐、更新稳定性与泛化检查 | 待 GPU 连接，尚未执行 |
| 反馈预测器、独立历史上的动作 regret、闭环确认 | 尚未执行；需要开发校准后的第二次协议锁定 |

CPU smoke 使用随机初始化的小型 `OlmoeForCausalLM` 和生成 token，仅检验工程机制。它不是预训练模型实验，损失数值不能用于研究结论。所有结果带有 `engineering_only` 或 `development_only` 状态。

当前共 140 项测试通过（原有 112 + v10 新增 28）。启动记录保存在 [CPU smoke](results/cpu_smoke/summary.json) 和[数据准备](results/data_preparation/tokenization_check.json)；后者使用了固定版本的真实分词器，未加载模型权重。

## 预检协议

机器可读配置在 [protocol.json](protocol.json)。当前锁定的是开发入口和测量语义；最终语料预算、学习率、B 达标目标、分支决策目标、最小有意义效应和确认性历史划分仍需校准后另行锁定，不假装已完成正式预注册。

- 模型：[OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924)，固定 revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`；模型元信息为 6,919,161,856 个 BF16 参数。
- 参数范围：固定选择第 12–15 层的 expert 0–3，共 16 个专家；其余参数全部冻结。根据官方结构推算约 100.7M 个可训练参数，实际加载时另记录准确清单和数量。选择不依据结果，也不保证这些专家覆盖所有新任务 token。
- 动作：`m∈{0,.25,.5,.75,1}`，作用于 AdamW 的 FP32 master 候选位移，再投回模型 dtype。`m=0` 仍推进优化器 moments；同时记录 master 位移和 BF16 模型实际位移，识别舍入影响。
- 起始资源配置：序列长 256、micro-batch 1、累积 4、学习率 1e-5、weight decay 0、最多 32 次更新；这是短开发测量，不是足以得出统计结论的预算。
- 基线：五个固定倍率，以及从 1 线性降至 .25 的时间调度。后续学习阶段还需要充分开发选择的强进度基线。
- 分支：固定 .5 行为轨迹的 step 0/8/16，两个独立未来批次流，每个未来五动作、每分支四步。两个开发 roots 共 60 条分支，连同基线共 624 次真实 learner 更新。分支来自同一完整可变状态，同 checkpoint/future 的各动作批次哈希必须一致。
- A/B monitor 各 8 个 block，selection 各 16 个 block，按 source 分层固定选取，避免全部 probe 来自一篇长文章；`test` 不 tokenize、不评价。当前不回放 A 样本，replay budget 为 0。
- LM loss 只计算 shifted next-token CE，FP32 累计、按有效 token 加权，排除 padding 和 router auxiliary loss。训练梯度按累积窗口的有效 token 总数归一。

冻结 router 参数不意味着路由固定。每步记录各层实际 top-k 占用和选中专家 token 暴露量。当前记录占用，不声称已经测量逐 token 路由迁移。

## 数据准备

开发数据来自固定版本的 WikiText-103 raw 第一训练分片与 CPython 3.11.9 `Lib/*.py`。WikiText 从 paragraph rows 重建整篇文章，source 是规范化文章标题；CPython 按 package/module 分组，排除 tests、ensurepip、idlelib、tkinter 等目录。保留原始版权文本；语料原文、模型权重和分词器缓存在 `.cache/`，不纳入 Git。

构建时最多读取前 1,000 个 WikiText article occurrences，去重后本次得到 A 995 篇、B 656 个文件。整个 source family 由固定 seed 1701 分配到 train/monitor/selection/test，比例 70/10/10/10；组大小不同导致文档比例并非恰好相等。词级规范化去重仅用于移除重复文档，不证明消除了近重复或模型预训练污染。此任务不能称为学习此前未见的知识。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/v10_native_moe_feedback/requirements.txt
mkdir -p .cache/v10-corpus
curl -L --fail https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/train-00000-of-00002.parquet -o .cache/v10-corpus/wikitext-103-train-0.parquet
curl -L --fail https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz -o .cache/v10-corpus/Python-3.11.9.tgz
python experiments/v10_native_moe_feedback/prepare_corpus.py --wikitext-train .cache/v10-corpus/wikitext-103-train-0.parquet --cpython-tar .cache/v10-corpus/Python-3.11.9.tgz --max-articles 1000 --out .cache/v10-corpus/prepared-new
```

构建结果包含文档 JSONL 和输入 SHA256/provenance。源资料见 [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)、[CPython release](https://www.python.org/downloads/release/python-3119/)及[原始 LICENSE](https://raw.githubusercontent.com/python/cpython/v3.11.9/LICENSE)。本项目不替换上游许可或声称拥有这些语料。

## 运行

每次使用新的输出目录，入口拒绝覆盖结果和直接写入 experiments 归档目录。原有版本不新增依赖，v10 测试显式启用：

```bash
python scripts/test_all.py --include-v10
python scripts/run_experiment.py v10 --mode inspect --out runs/v10-hardware-new
python scripts/run_experiment.py v10 --mode smoke --out runs/v10-smoke-new
python scripts/audit_v10_run.py --run runs/v10-smoke-new
```

在具备硬件 BF16 支持的 CUDA 主机上，先运行预检。以下命令只有显式 `--download` 才允许下载固定版本模型：

```bash
python scripts/run_experiment.py v10 --mode preflight --download --data .cache/v10-corpus/prepared-new/documents.jsonl --device cuda:0 --out runs/v10-gpu-preflight
python scripts/audit_v10_run.py --run runs/v10-gpu-preflight
```

预检执行 0/.5/1 三条各两步基线，并从有 moments 的共同状态执行五个单步分支，共 11 次更新，测量实际恢复成本。默认是单卡执行；八张卡不会自动组成一个模型，也不会自动启动八份任务。确认单卡峰值显存、host RAM 和 checkpoint 传输成本后，才安排跨卡独立运行。

完成资源及学习稳定性检查后，分别用 `--mode baselines` 和 `--mode branches` 执行预锁定开发配置。配置调整应另存 JSON，通过直接入口 `run_native.py --protocol <path>` 使用，并保留每次的 `locked_protocol.json`。不根据结果悄悄修改已有运行。

## 如何解释分支记录

当前保存二维结果：B selection CE 改善、A selection CE 恶化；没有擅自指定 λ 或制造“最优动作”标签。未来的约束目标、并列规则和 ranking/regret 标准需要开发校准后锁定。训练/验证按整条 root/history 分组，相邻 checkpoints 和动作分支不是独立样本；不同 seeds 共享语料，也不是新领域泛化证据。

闭环阶段应预定 B monitor 目标和首次命中规则，并在独立评价上检查表现，报告全部失败、成功率、遗忘和成本。短分支没有信号，仅支持在当前测量范围内暂缓更昂贵的控制器训练，不能证明反馈信息不存在。

所有物理训练、评价、checkpoint 和恢复开销分账，墙钟包含读写等额外时间；时间字段不可再加到总墙钟上。分支共享 checkpoint 的捕获成本只记一次，逐分支恢复/训练/评价则各自计算。
