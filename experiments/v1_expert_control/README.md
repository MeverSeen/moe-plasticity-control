# MoE 专家更新控制器：可复现实验原型

已实现并运行四种主要策略：统一控制、按激活频率保护、按历史重要性保护、PPO 学习型专家控制。另保留未训练控制器作为诊断对照。

**首轮结果没有验证学习型控制器优于规则基线。** 在目标 MSE ≤ 0.04 时，学习型控制器达标 55/80 次，统一控制为 63/80 次，重要性基线为 59/80 次。学习型控制器在共同达标的样本上也没有胜过重要性基线。

- [中文实验报告](实验报告.md)：设计、结果、解释与限制。
- [结果图](results/analysis/results.png) / [矢量图 PDF](results/analysis/results.pdf)。
- [完整汇总](results/analysis/summary.json)：按序列聚类的配对置信区间。
- [测试前锁定的配置](results/data/locked_protocol.json)。
- [原始端点数据](results/data/test_endpoints.csv)。

## 复现

Python 3.9 或更新版本；本次使用 Python 3.9.6、PyTorch 2.8.0、NumPy 2.0.2、CPU 单线程。没有外部数据集下载，也不需要 GPU。

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s . -p test_moe_control.py -v
python3 run_all.py --out results/reproduction
```

`run_all.py` 依次调参、训练、锁定配置、评估、生成统计及图表。请使用空目录；脚本拒绝覆盖既有结果。不同平台可能有数值差异，不保证逐位一致。

也可以分步运行：

```bash
python3 moe_control.py tune --out results/new_run
python3 moe_control.py train --out results/new_run
python3 moe_control.py evaluate --out results/new_run
python3 summarize.py --run results/new_run --out results/new_analysis
```

已提供验证集选出的 `controller.pt`、初始控制器、训练末控制器、训练过程、全部基线候选结果和最终测试记录。配置与模型训练不一致时，评估器会报错；已有锁定协议时拒绝重复覆盖测试。

## 文件

| 文件 | 用途 |
|---|---|
| `moe_control.py` | 合成任务、稀疏 MoE、控制环境、基线、PPO、评估 |
| `test_moe_control.py` | 6 项实验有效性检查 |
| `summarize.py` | 配对聚类 bootstrap、敏感性分析、图表 |
| `run_all.py` | 一次完成全部复现步骤 |
| `results/data/` | 配置、权重、训练及测试原始记录 |
| `results/analysis/` | 已计算的统计和图表 |
| `environment.json` | 本次运行环境 |
| `manifest.sha256` | 交付文件校验值 |

这是固定路由、固定容量的小型合成回归实验。结果不代表真实 MoE 大模型、真实语料上的效果，也不构成整个研究方向不可行的结论。
