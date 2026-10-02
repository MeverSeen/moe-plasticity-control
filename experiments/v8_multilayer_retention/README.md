# v8：多层 MoE 的有限记忆保持实验

本目录实现一个小规模、探索性的 CIFAR-100 连续学习实验。目标是比较：在固定的旧样本回放与监测预算下，按层、按专家分配更新强度是否能改善历史任务保持。2026-10-02 已完成 pilot：两个学习臂均选中初始化，未验证协调优势。详见 [结果解读](结果解读.md)与 [诊断图](results/pilot/analysis/diagnostics.png)。该轮不是确认性实验。

## 数据与任务

数据来自 CIFAR-100 官方 Python 归档：`https://www.cs.toronto.edu/~kriz/cifar.html`。归档 MD5 必须为 `eb9058c3a382ffc7106e4002c42a8d85`，读取时还会验证其中 `train`、`test` pickle 的固定 MD5；数据直接从 tar 归档读取，不解压。

100 个 fine classes 以固定种子划分为互不重叠的 50 个 train pool、25 个 dev pool、25 个 audit pool。每个 history 从对应 pool 按该 root 的随机顺序取 25 类，组成 5 个任务、每任务 5 类。每类从官方训练集采样：48 个 learner train、16 个 online monitor、32 个 held-out dev eval。三种样本流彼此互斥。`make_history` 保持调用方给定的类顺序，并把原始 CIFAR 标签按每任务的 `classes` 顺序映射到局部的 0–4；原始类别仍保存在 `history['classes']`。

同一流中各类样本 round-robin 交织，因此任意前缀的各类计数相差至多 1。这一点避免固定内存的前缀截断偏向任务中的第一个类别。官方 test 仅用于 audit：每个被选类别 100 张，共每 history 25 个类别；它不进入训练、目标校准或模型选择。

类的未见性与 history 的独立性需要区分：dev/audit pool 与训练 pool 类隔离，但同一 split 的多个 root 只是同一 25 类 pool 的不同任务顺序和样本抽样。因此 pilot 的 3 个 audit roots 共享候选的 25 个测试类别；对 root 做 bootstrap 只能解释为对这些条件化 root 的不确定性，不是对任务/类别总体的标准推断。

## 模型、训练和内存

模型是两层 residual MoE：8 个 experts、每层 top-2 路由。路由由固定的原始图像 sample routes 决定；CNN stem 和每任务的 5-way head 均可训练。每个任务的局部标签输入对应的 task-ID-visible head。

每个 learner 持有总量固定的 128 replay 样本和 128 monitor 样本。任务完成时，当前任务的 train/monitor 样本以及每个 monitor 样本在该任务完成时的交叉熵参考值才可进入内存。任务变多后，内存按任务均衡配额并作确定性前缀缩减；已删掉的旧样本不会从完整历史重新取回。在线遗忘是对保留 monitor 样本计算、先按 task 均值、再对 task 等权平均的正损失变化。

所有方法在 task 1 使用 `global_1`。task 2–5 才应用控制动作。每次动作为每个 layer/expert 选择 `0`、`.5` 或 `1` 的更新乘数；优化器与完整 Adam 状态持续保留，乘数为 0 时动量本身仍保留。候选分支从 task-1 后的完整模型、Adam、内存和参考值 checkpoint 复制；同一组未来任务的 minibatch 流以 task 和 step 配对，避免动作改变外生训练样本。

## 方法与选择

八个方法臂由 6 条显式规则与 2 个学习控制器构成：

- `global_0`、`global_0.5`、`global_1`
- `group_rule`、`local_rule`、`coordinated_rule`
- `learned_local`、`learned_coordinated`

规则是成本低的 pilot baseline，并未经过充分超参数搜索。学习臂使用 GRPO：local 只使用对应 layer/expert 的局部特征；coordinated 还拼接同层和全局平均表征。动作概率在所有 layer/expert slot 上因子化，但训练时使用其联合对数概率。它不是对控制结构进行搜索，也不使用 PPO。主动监测/主动选择要检查哪些旧任务也尚未实现。

pilot 使用一个 policy seed（881），训练 6 rounds、每 round 4 条成组的真实 learner 分支。奖励由有限 monitor memory 上的最终遗忘、task 2–5 的失败比例和 terminal loss 构成；这只是训练 proxy。候选 checkpoint 包括初始化、中点和最终点。它们均在真实 dev histories 上完成 rollout，并用完整的 held-out dev eval 闭环选择；不是从训练分支的奖励中选择。

在线达标阈值不等同于最终论文式目标。运行前仅在 dev roots 上，以 full-budget `global_1` 校准：收集每个任务见到的最小 monitor CE，取其 90% 分位数加 `.03`，再截断到 `[.8, 1.55]`。官方 test 和 controller 候选均不参与该校准。

最终评估对每一个任务完成时保存的 checkpoint 形成完整下三角 loss/accuracy 矩阵。dev 的评估集来自官方 train 中的 held-out 样本；audit 才读取 official test。成本另行记账，包括 learner updates、train/monitor/audit/route 样本通过数、控制调用、clone/controller/training/selection 时间；它们不能解释为 FLOPs。

## 运行

从仓库根目录运行：

```bash
python scripts/run_experiment.py v8 --pilot --download --out runs/v8-pilot
```

首次运行可用 `--download` 让程序拉取并验证官方归档；若 `.cache/datasets/cifar-100-python.tar.gz` 已存在且通过校验，可省略该参数。运行会在指定的全新输出目录中保存 protocol、校准记录、控制器候选、结果、成本记录和分析。

## 当前限制

pilot 只有 1 个学习 policy seed、3 个 audit roots，统计功效有限。每个 history 只测 25 个 test classes，因此 audit 不是标准的 CIFAR-100 100 类整体成绩。task ID 对 head 显式可见，模型也使用固定 raw-image routing；结论不应直接外推到动态路由或 class-incremental 无 task ID 设置。有限内存遗忘是一个在线代理，离线 test 只在最后评估；PPO、主动监测和完整的结构/超参数搜索尚未实现。
