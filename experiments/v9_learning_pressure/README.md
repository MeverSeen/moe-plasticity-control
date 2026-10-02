# v9：学习压力与模块干预（开发诊断）

**已于 2026-10-02 完成 180 条完整开发轨迹。** 更严目标没有使全量专家更新占优；固定预算下出现明显训练—开发误差差距。详见[结果解读](结果解读.md)、[完整报告](results/development/analysis/实验报告.md)和[学习曲线](results/development/analysis/learning_curves.png)。本轮没有使用官方 test，也没有训练新控制器。

v9 问的不是“哪种控制器已经赢了”，而是：当每个新任务的学习要求变严时，专家位移是否值得；学习能力又主要留在 stem、专家还是任务头中。本轮不训练 RL、GRPO 或任何新控制器，也没有“应该出现正结果”的先验。

## 已锁定的实验格

完整运行使用 6 个 development roots：`88410` 到 `88415`。每个 root 有一个共同的 task-0 prefix，随后独立运行 30 个 suffix cells，所以共 `6 × 30 = 180` 条结果行。`--pilot` 只使用前两个 roots（`88410`、`88411`），因此产生 60 条结果行。

每个 root 的共同 prefix 都以全模型、`global_1` 训练 task 0，目标 CE 为 1.55，首次达到监测目标时停止，或最多 80 次更新。模型权重、完整 Adam 状态、有限记忆和未来数据抽样 RNG 都在该点完整复制。task 0 的记录永远标记为目标 1.55；tasks 1–4 的记录使用 cell 指定的目标。

每一个 suffix cell 都从同一个 prefix clone 开始，并独立完成 tasks 1–4。它们不是从别的 cell 的曲线截出来的。30 个 cells 是：

| 部分 | 每 root cells | 条件 |
|---|---:|---|
| 目标压力 | 24 | full scope 下 4 个目标（1.55、1.45、1.35、1.25）× 6 条规则（`global_0`、`global_0.5`、`global_1`、`group_rule`、`local_rule`、`coordinated_rule`），均为 early-stop |
| 模块 early-stop | 2 | 目标 1.35：`frozen_stem/global_1` 与 `heads_only/global_0` |
| 固定预算模块臂 | 4 | 目标 1.35：`full/global_0`、`full/global_1`、`frozen_stem/global_1`、`heads_only/global_0`，均强制每个 suffix task 完整 80 次更新 |

早停只查看当前任务的 monitor CE，每 5 次更新检查一次。fixed-budget 的每个 suffix task 无论一开始是否达标都运行 80 步；它的轨迹不能被截断来推导 early-stop 的结论，反之亦然。`first_hit_step` 和 `ever_hit` 记录监测指标的首次命中；这与最终是否仍命中是两件事。

## 数据、记忆和诊断

6 个 root 重用 v8 的 development 25 类池，只从 CIFAR-100 官方训练文件读取样本。它们是初始化和数据流的重复，不是独立抽取的类别总体。官方 test 文件不加载；v8 的 audit 类和 test 结果也不进入本轮。因此任何发现都只是开发诊断，不能当作确认性结论。

每一项都使用同样的固定容量：replay memory 128，monitor memory 128。模型复用 v8 的固定路由两层 MoE 和不可变的 v8 源码；路由由原始图像决定，训练不会改变某个样本被分到的专家。有限记忆保持每个任务各自在学会端点上的固定参照，以测量旧任务损失的后向变化。

每个控制窗口额外测量当前任务的全训练集 CE 与不重叠的 held-out development CE，完成时也记录初始、最小和终点值。它们只用于诊断：不传入 chooser、不用于动作、不用于早停，也不用于选 checkpoint。离线端点评价同样只在 rollout 后进行。

scope 的意义如下：full 允许 stem 和 heads 正常更新，专家的实际 Adam 位移按 action 缩放；frozen-stem 在 task 0 后冻结 stem 的权重和 Adam moments；heads-only 则冻结 stem 与全部专家的权重和 Adam moments，仅任务头可更新。硬冻结与给专家 action multiplier `0` 不相同：后者令本次专家参数位移为零，但 v8 的 ScaledAdam 仍会更新它们的 moments；硬冻结会清除梯度，使 moments 也不变。

## 成本口径

每条 raw result row 保存完整的**逻辑**成本，因此包含该 row 所依赖的共同 prefix，便于按策略查看完整一次学习路径。物理总成本则把每个 root 的共同 prefix 只算一次，再加上每个独立 suffix、clone 和离线端点评价。runner 同时保存 `common_prefix`、`suffixes_and_endpoint_evaluation` 和合计；从单行的 counters 减去该 root 的 prefix counters 可得到真实分支增量。

`diagnostic_train_examples`、`diagnostic_eval_examples` 和 `diagnostic_seconds` 单列，不混进 learner 的训练 passes。诊断时间已经包含在 rollout wall time 内，汇总时不能再重复相加。

## 运行

完整开发格：

```bash
python3 scripts/run_experiment.py v9 --out runs/v9_learning_pressure
python3 scripts/audit_v9_run.py --run runs/v9_learning_pressure
```

两 root pilot：

```bash
python3 scripts/run_experiment.py v9 --out runs/v9_learning_pressure_pilot --pilot
```

首次本地没有 CIFAR-100 官方 archive 时，额外传入 `--download`。结果写入新目录；运行前会锁定 protocol 和源码 hash，运行期间源码变化会让 runner 拒绝完成。
