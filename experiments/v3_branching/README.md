# MoE feedback branching experiment

已完成正式运行。**本轮未验证旧知识反馈优于进度策略。**

完整中文报告：[实验报告](results/main/analysis/实验报告.md)。
结果图：[PNG](results/main/analysis/results.png) / [PDF](results/main/analysis/results.pdf)。

## 实验范围

复用 v2 的 6-expert 非线性合成环境，新增 checkpoint × 5 actions 分支实验和监督式动作评分。
没有训练 PPO/DQN；没有修改 v1/v2 原始代码或结果。`base_env.py` 与 `model_core.py` 是本轮冻结的环境源码副本。
路由为固定几何路由，不是 learned router；这不是之前 4-expert 线性 sanity check 的复现。

- 独立校准/训练/开发/测试历史：16/80/24/64。
- Train/dev/test 共 453 个 checkpoint、4,530 条终点分支。
- 每个分支先执行指定倍率最多 5 步，再采用同一个校准集选定的后续策略，直到目标新任务 MSE≤0.06 或耗尽预算。
- 每个状态/动作使用 2 条独立后续样本流；每条流内各动作配对共享 minibatch。
- 3 个匹配的策略初始化，Progress/Feedback 使用同架构、相同初始信息与训练预算。
- Adam 一二阶矩、步数、预算、最近反馈与预生成 minibatch 索引完整复制。更新无 dropout，随机采样已物化为索引数组。
- 额外评价闭环策略、开发集调优的固定倍率和简单反馈规则。

## 主要文件

| 文件 | 内容 |
|---|---|
| `branching.py` | 数据生成、完整状态分支、监督策略拟合、冻结测试协议、闭环评估 |
| `test_branching.py` | 9 项实验边界与机制检查 |
| `analyze.py` | 历史聚类配对区间、图表、中文报告 |
| `results/main/locked_protocol.json` | 测试前锁定的配置、划分、选择与源码/权重 SHA-256 |
| `results/main/*_branches.jsonl` | 每个 checkpoint 的输入、mask 和所有动作的完整终点结果 |
| `results/main/test_predictions.json` | 策略仅从可用输入作出的动作选择 |
| `results/main/test_online.json` | 闭环轨迹与最终结果 |
| `results/main/analysis/summary.json` | 机器可读统计结果 |
| `results/main/analysis/*endpoints.csv` | 可审查的原始分支及闭环终点表 |

## 复现

运行环境：Python 3.9.6、PyTorch 2.8.0、NumPy 2.0.2、Matplotlib 3.9.4；CPU 单线程。

在此目录执行：

```sh
python3 -m unittest test_branching.py -v
python3 branching.py --out results/reproduction
python3 analyze.py --run results/reproduction --out results/reproduction/analysis
```

脚本拒绝覆盖已有实验目录。`--pilot --epochs 100` 使用单独的少量预检历史。
正式主运行约 149 秒；本机时间不代表大模型成本。

## 结论边界

终点遗忘是独立评估 MSE 的正增量，不是百分比遗忘率。
共同目标阈值不等于数值严格相同，报告同时保留实际端点 loss 和紧匹配敏感性分析。
失败不丢弃；同时报告达标率与全样本诊断代价。共同成功上的遗忘只作配对条件结果。
同样的观测与策略计算额度用于所有方法，这是信息消融，不能作为真实部署算力收益证明。
所谓 oracle 读取离线分支结果，仅用于诊断；第二未来流复测用于检查事后最优是否可重复。
训练/开发世界的独立评估数据是离线元训练标签，最终测试世界的评估数据只在锁定模型后评分。
最终测试结果不应被用于调参；下一轮需要新测试历史。
