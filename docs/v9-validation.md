# v9 开发诊断校验记录

日期：2026-10-02。v9 复用冻结的 v8 数据、模型和环境辅助代码，新增独立目录 `experiments/v9_learning_pressure/`；v1–v8 的 821 个归档文件保持原哈希。

## 机制与统计验证

新增 12 项测试，全仓库 9 个版本共 112 项通过，各版本在独立进程中运行。

- early-stop 对初始即达标任务可立即结束；fixed-budget 即使初始达标也执行全部预算，并拒绝提前结束。
- 硬冻结保持对应权重与 Adam moments 不变；零专家位移和硬冻结的语义分开。
- 完整 checkpoint 克隆保持可复现；改变独立开发评价标签不改变训练动作或终点。
- 不同目标独立运行，不能把长轨迹截断来替代短轨迹；共同前缀之后正确重置新任务目标与首次命中状态。
- 非法目标与不兼容的模块条件被拒绝。
- 首次命中、曾命中、末点达标与独立开发达标分别计数；从未命中的任务保留在固定分母中。
- 结果网格缺失、重复、意外条件、共同前缀不一致、矩阵缺失/非有限值、停止语义违例均被拒绝。

## 完整开发运行

```bash
python3 scripts/run_experiment.py v9 --out runs/v9-development-20261002
python3 scripts/audit_v9_run.py --run runs/v9-development-20261002
python3 scripts/test_all.py
python3 scripts/verify_artifacts.py
```

完整归档在 `experiments/v9_learning_pressure/results/development/`，包含协议、源码哈希、环境、共同前缀、180 条完整轨迹、全历史矩阵、成本与图表。6 roots × 30 条件的网格完整；每个 root 的 prefix fingerprint 一致，16 份锁定源码全部匹配。early-stop 与 fixed-budget 的全部后续任务符合独立停止语义。官方 CIFAR-100 仅加载 train 文件，开发评价样本与训练/monitor 互斥，官方 test 和 v8 audit 未参与本轮。

成本从逐行逻辑计数中扣除重复前缀后，与物理账本逐项一致：31,735 次 learner 更新，含共享前缀 115 次；总墙钟约 302.8 秒。单列的 73.3 秒诊断时间已计入 rollout，不重复相加。固定更新数并非 FLOPs 匹配。底层沿用的 `audit_examples` / `audit_seconds` 字段在本轮指离线 development 端点评价。

## 结果与范围

固定 80 步、目标 1.35 下，全量更新训练/独立开发 CE 为 0.478/2.093，零专家位移且 stem/任务头可训练时为 1.005/1.226。严格目标未使全量专家更新表现更好；这提示当前设置的泛化或更新稳定性问题，不能证明专家普遍无用。

仅 6 roots、同一开发类别池、每类 48 个训练样本、一个首任务共同前缀；没有正式新测试、学习控制器或确认性统计主张。完整解释与下一步见 [结果解读](../experiments/v9_learning_pressure/结果解读.md)。
