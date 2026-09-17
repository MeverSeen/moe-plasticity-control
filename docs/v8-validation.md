# v8 执行与审计

基于 ef73c5848fe4b954c64cea2f7d7000ba9483444e 的独立 worktree，分支 `codex/v8-allocation-ppo`。新增 `experiments/v8_allocation_ppo`；v1–v7 的790个归档文件逐字节未变。未发现额外适用 AGENTS.md。

## 实际运行

```bash
python3 scripts/run_experiment.py v8 --pilot --out runs/v8-pilot-final
python3 scripts/run_experiment.py v8 --out runs/v8-main
python3 scripts/run_experiment.py v8 --pilot --out runs/v8-reproduction-check
python3 scripts/test_all.py
python3 scripts/verify_artifacts.py
```

入口自动运行统计分析、`scripts/audit_v8.py` 和 `scripts/summarize_v8_training.py`。前者审计锁定源码及权重、分区历史、全部测试预算、PPO奖励及cost；后者仅做描述性训练/成本汇总，不改变选择。PyTorch 2.8.0、NumPy 2.0.2、Python3.9.6、CPU单线程。

## 机制验证

全部8个独立测试进程共90项检查通过。v8继承12项连续环境检查，新增5项：旧五档动作包含关系；逐组Adam位移倍率及不缩放矩；新动作单维编码；PPO奖励累计/终端失败cost/actor参数确实更新；改变独立评估标签不影响PPO轨迹或奖励。

v8继承预算检查的精确浮点比较改为容忍1e-8，修复累计除法导致的约2e-13舍入差；不改变实际预算。其他v7文件不修改。在线预算按15动作双输出头的上限共同预留；各组预算相同，不以v7/v8的跨版本差异推断因果收益。

## 预检修订

首次预检在保存动作诊断时因NumPy整数不能直接JSON序列化中断；第二次在PPO首批日志输出时因重复batch字段中断。二者均未生成最终测试历史，已保留日志和部分数据。修复仅涉及输出转换和日志字段。正式运行使用随后完整预检通过的相同实验源码；没有按pilot最终测试选择超参数。

主方案先写入版本README与每次运行plan.json。每次运行在创建测试world之前写locked_protocol.json，包含所有部署/对照模型哈希与开发选择的基线；启动与完成时检查源码哈希一致。随机actor作为描述性对照，不加入PPO选优候选；PPO候选包含warm0及训练后检查点。监督epoch0全部作为候选，额外测试对照固定slack=.1。

主统计按根历史聚类，对三初始化先平均；共同成功要求双方全部初始化成功。12项主要比较的Bonferroni敏感性区间另列；非劣效单侧界未作多重校正，不据此作所有比较同时成立的主张。概率目标90%是预定目标，开发点估计达到不能替代总体概率保证。

预检原始运行约62.8秒。独立复跑比较82个核心文件，全部逐字节一致：全部候选/所选模型权重、训练轨迹、开发候选、原始分支、基线选择、测试轨迹与主要统计。墙钟与锁定时间戳不作字节比较，记录见 `experiments/v8_allocation_ppo/results/reproduction_check.json`。预检216条测试rollout、96条PPO训练episode全部通过审计。

补充 `scripts/audit_v8_collection.py` 仅重建原始行为轨迹，逐一核对分支数据中的18维状态并记录采集行为本身的更新与工作量。预检75个状态全部精确一致；重建的16条行为轨迹共1125次learner更新。此审计重跑的额外开销单列，不混入原始实验训练成本。

## 正式运行结果与归档

正式实验约899.9秒，生成534个训练状态、16020个配对分支，完成24个所选监督/PPO模型及初始化/末轮对照。全部4800条测试rollout、2304条PPO episode、1152次PPO优化步通过审计；模型与源码哈希和启动时一致。534个状态的行为轨迹重建全部精确匹配。

正式数据、模型、图表、候选、运行日志归档至 `experiments/v8_allocation_ppo/results/main`；完整预检归档至 `results/pilot`，两次未进入测试的中断预检保存在 `results/preflight_failures`。原始JSON保留，不删失败样本，不重选模型。SHA256清单覆盖版本源码、说明和全部结果。

根目录补充脚本 `plot_v8_comparisons.py` 从锁定主要统计生成区间图，`supplement_v8.py` 描述性比较8组控制器与3个开发所选基线并统计动作使用；这些补充比较只用已冻结策略的原始轨迹，不调参，名义区间未针对24项补充比较校正。两张正式图均已目视检查，标签、坐标和区间完整。

结果偏负面：15动作四组共同成功遗忘均增加，主要校正区间支持该方向；旧知识反馈与PPO稳定收益未验证。最终主要组达标率约82%–85%，低于预定90%目标；不能把开发可行性当作测试保证。详见版本结果解读。

最终v8清单覆盖383个归档文件；根目录验证共1173个文件通过（包括原有790个文件）。正式与预检的归档副本再次通过锁定模型/源码和全部轨迹审计。原始正式测试JSON约77.8 MB，保留全部轨迹，不作有损压缩。
