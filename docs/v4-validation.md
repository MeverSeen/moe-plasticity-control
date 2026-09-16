# v4 验证记录

日期：2026-09-16。Python 3.9.6、PyTorch 2.8.0、NumPy 2.0.2，CPU 单线程。

- 原有 23 项机制检查通过；v4 含 9 项复制的环境回归检查、9 项状态聚合检查、3 项统计检查，合计 44 项通过。
- `python scripts/run_experiment.py v4 --pilot --out runs/v4-pilot` 全流程通过；独立 pilot 划分，约 66.7 秒。
- `python scripts/run_experiment.py v4 --out runs/v4-main` 全流程通过，约 431.4 秒；正式测试仅在源码及 30 个策略哈希锁定后生成。
- 全部锁定源码及权重哈希匹配，根历史划分互斥，训练标签来源有效，逐历史查询数量相同，在线预算及真实目标阈值检查通过。
- 正式结果图已检查；所有原始标签、终点和锁定文件归档到 `experiments/v4_state_aggregation/results/`。
- 前三轮导入的 153 个文件保持原始 SHA-256。v3 的阈值文字错误另记勘误，未覆写原实验记录。

正式实验已查看，后续实验必须重新划分最终测试历史。运行时间与 work 代理都不代表大 MoE 的真实部署成本。
