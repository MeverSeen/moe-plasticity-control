# 工程初始化检查

检查日期：2026-09-16。环境：macOS ARM64、Python 3.9.6、PyTorch 2.8.0、NumPy 2.0.2、Matplotlib 3.9.4。

| 检查 | 结果 |
|---|---|
| `python3 scripts/verify_artifacts.py` | 153 个导入文件的 SHA-256 与源文件一致 |
| `python3 scripts/test_all.py` | v1 6 项、v2 8 项、v3 9 项，合计 23 项全部通过 |
| `python3 scripts/run_experiment.py v3 --pilot --out runs/init-smoke` | 分支生成、评分器拟合、闭环评估和报告生成均成功 |

本次端到端检查使用已有 pilot 设计，仅验证新工程运行入口，不作为新增研究证据。其输出位于被 Git 忽略的 `runs/`，原有正式实验记录保持不变。本次没有重新运行 v1/v2 的完整训练。
