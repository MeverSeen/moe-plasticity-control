# 离线收益空间诊断

Python 3.9+ 标准库，无 torch/GPU/联网依赖。仅读取已有记录，生成 `headroom.json` 和 `收益空间诊断.md`。不会训练、补跑候选、改协议或读取确认数据作选择。

对应研究解释：[收益空间与决策计划](../../../docs/v13-evidence-and-decision-plan.md)。

## 使用

在仓库根执行（reference 必须替换为开发协议已经选好的完整候选 ID）：

```sh
python3 experiments/v13_native_ppo/diagnostics/analyze_headroom.py \
  --native-dir experiments/v13_native_ppo/results/calibration \
  --reference fixed_reference \
  --output-dir experiments/v13_native_ppo/analysis/calibration_readout_01

python3 -m unittest discover -s experiments/v13_native_ppo/diagnostics -p 'test_*.py'
```

若已有两条**实际执行且明确独立**的未来流，再加 `--select-stream 0 --evaluate-stream 1`。没有这两个参数或没有符合条件的数据时，复测状态是 unknown。本轮执行端目前只计划 stream 0，`independent_future=false`；不要为了让脚本输出 available 而改标记。

输出目录不可包含输入文件，也不要位于会被 `--native-dir` 递归扫描的目录内。拒绝覆盖已有同名分析输出。混合协议或混合 calibration/development 池需分开运行；工具不比较跨协议差值。

原生 schema 尚在执行端落地阶段。适配器不猜测字段，缺失时写入 `adapter_audit`。首次真实结果到达后须确认字段含义与本契约一致；通过合成单测不等于已适配真实结果。

## 原生端点适配

支持执行端约定的 JSON：

- `protocol_revision`：整个比较中的目标、预算、监测集分配规则及优化器语义必须一致。适配器无法仅凭标签验证实际实现，需由归档核验保证。
- `phase` 或 `root.split`：只接受 calibration/development，后者优先；其他用途被列为 excluded，不参与分析。不要将内层训练数据的 train split 混同外层历史的分析用途。
- `root.id, root.tasks`：根历史身份、完整任务序列。
- `checkpoint.base_revision, checkpoint.prefix_sha256`：同一起点的基座＋可变权重/Adam/RNG/记忆/游标指纹；相同根的同一 comparison 若不同则拒绝配对。
- `candidate.name`、候选配置和 `checkpoint.controller_sha256`：同名学习器不同检查点会加上 `@<hash前16位>` 区分。候选 ID 不可复用于不同配置。
- `batch_stream.id, batch_stream.independent_future`：后者必须显式为布尔值。
- `history_completed, history_success, n_fail`：执行端按协议明确给出。completed 包括协议规定的真实提前失败终点；被墙钟中断不是 completed。不得从较短的 tasks 列表猜测未执行任务是否失败。
- `target_spec`：完整历史（包括未执行任务）的目标信息。只有 tasks 列表完整且每项有 target 时，允许从其中提取；部分历史缺完整目标时列为 unusable。
- `monitor.forgetting, monitor.full_memory=true`：完整终点监测代理。轮换小批值不能冒充 full-memory 终点值。
- `evaluation.forgetting`：独立评价的遗忘。字段缺失保留 null，不作为零。
- `cost_actual.updates, cost_actual.wall_seconds`：此次实际计算；`cost_path.updates`：包含复用前缀的完整部署路径更新数。二者不互相替代。actual 总和仅覆盖有记录的轨迹，不含工具不知道的前缀、调试等成本。

只有同根同起点、相同目标/协议/任务集合才能配对。多种比较起点或续跑策略用不同 `comparison_id` 区分，默认仅支持 `whole_history`。首动作分支请用下述规范格式，显式把 continuation 纳入 context 指纹。

## 规范输入契约

也可用 `--input bundle.json`。顶层：

```json
{
  "schema_version": 1,
  "protocol_id": "locked-protocol-hash",
  "metric_id": "explicit-forgetting-definition",
  "records": []
}
```

每条 record 必需字符串字段：`root_id, comparison_id, checkpoint_id, context_id, stream_id, candidate_id, role`。`context_id` 应绑定目标、预算、停止标准、评价/监测样本集合、原始 A 基准、比较类型，以及首动作比较的续跑策略。指纹相同只代表导出者声明匹配，不是工具独立验证了环境状态。

其余字段：

| 字段 | 含义 |
|---|---|
| candidate_spec | JSON 对象，精确描述冻结候选策略/倍率配置；不得混入实际执行出的随机动作或终点结果 |
| candidate_kind | 可选 fixed/rule/clone/ppo/intervention/unspecified |
| completed, success, independent_future | 明确布尔值 |
| n_fail | 按协议计算的非负整数 |
| monitor_forgetting, eval_forgetting | 有限数值或 null |
| path_updates, actual_updates, wall_seconds | 有限非负数值或 null；更新数须整数 |

同一 comparison/stream/candidate 重复会拒绝。独立种子、独立起点和不同候选检查点必须有不同 ID；同根的多检查点仍共用 root_id，以免伪增样本量。

## 分析含义

- **candidate_vs_reference**：匹配起点和未来流的达标、未完成、共同达标遗忘差与路径更新差。正差表示候选更高；遗忘比较只条件于共同成功并有评估记录，缺失与分母另报。
- **within_stream_descriptive**：同一流已执行可行候选的差异、乐观事后最好值、监测与独立评估的排序一致性。它们不是独立验证的改善。
- **common_minimum_forgetting_candidates**：可行且有评估数据的状态中，数值容差内最低遗忘集合的交集。候选菜单不同则 unknown；没有成功候选的状态数量另报。不考虑成本时共享一个最低遗忘候选，不能据“动作标签不同”证明状态信息必要。
- **independent_future**：完整配对菜单上，在选择流挑选候选、在另一实际未来流复测。monitor_selection 使用完整监测代理；privileged_selection 只作特权诊断。二者都保留复测失败。
- 所有汇总先在根内平均，再对根平均。状态数、候选对数和根数分开，不把相关样本当独立样本；不输出显著性、理论最优或自动 GO/NO-GO。

两流若使用同一任务和评估集，复测仅检验未来批次随机性的稳健性，不等于新任务/新评估样本泛化。闭环最佳动作、当前观测充分性、逐 expert 协调和 RL 必要性，均不能由这份摘要单独证明。

## 测试范围

合成单测检验：复测排序反转、复测失败保留、缺失候选/独立流拒绝、同一流选择不算复测、并列候选反例、根级权重、确认集隔离、指纹/协议/候选一致性、非有限值、原生适配的缺失语义和 CLI 输出。合成数值不是本项目实验结果。

## v13 实际归档范围

2026-09-20 主平台转向全层 expert LoRA，正式校准/克隆/PPO/确认均取消。pilot 端点保留 `phase=pilot` 与 `root.split=development`；适配器保留 `source_phase`，报告明确标为工程短测。`reference40` 仅生成目标，明确排除候选对比；指定 `fixed1` 仅是均匀更新参考，不是未执行的强静态搜索结果。独立未来未运行，保持 unknown。
