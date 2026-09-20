# 执行与核验边界

- source_lock.json冻结14个执行时输入文件，实验中不修改这些文件。离线分析、绘图、收集和核验脚本是后续工具，不冒充执行版本。
- 当前执行程序每条历史结束会写临时ppo_current.pt，其文件SHA会进入该条记录；该临时路径随后覆盖。持久保存的ppo_initial.pt和每批ppo_batch_NNN.pt是行为策略的权威复现来源。训练rollout索引i对应初始策略（i<2）或batch floor(i/2)；validate_results.py用这些持久检查点逐条回放实际latent动作及联合log probability，而不将最后一个临时文件冒充全部历史策略。PPO终版与确认选择文件的参数逐位比较，确认不再更新控制器。
- 部署适配器只有A/B因子，没有Adam状态。verify_adapters.py同时核验整个文件SHA及不依赖zip文件名的因子内容SHA；不同文件名造成的字节差异不能当作模型差异。
- EMA只在未缓存的新决策边界推进；缓存重读、A前缀恢复与中途恢复不额外推进趋势。短/长趋势与已有缓存都属于环境状态。
- PPO批次数、实际优化epoch/梯度步、learner更新、完整新历史和控制决策分开统计。未完成轨迹不进入PPO更新。没有决策的完成历史不会被伪造为非零策略梯度。
- 阶段限额使用已有更新边界检查点保护；原独立监督器保留13:25 UTC硬截止。工作流在12:40 UTC之前停止接续更新，完整评价调用的收尾时间也计入真实墙钟；不能把监督器job墙钟称GPU利用率积分。
- 首次默认python3调用不存在，命令未创建远端目录或排队；改用既有venv后成功。一次本地字节编译遇到macOS默认cache路径只读，改用/tmp缓存通过。这些工程事件没有改变训练协议或数据。

## 执行后的明确修复与未解异常

原source_lock仍对应archive/provenance中的执行字节。原workflow把frozen_hash返回的字典直接与字符串比较；当前workflow调用integrity.verify_frozen_hash检查sha256及参数数量。post_run_hash_fix.diff保留改动，test_integrity.py核验匹配、错误SHA、错误参数范围和错误类型。原实际终点hash未持久化，恢复摘要不填造该值。

第一确认根的全1控制轨迹不一致仍未定位。原prefix SHA是缓存标签，不能当每次恢复前的新hash；初始B监测值也是缓存。授权四次root2 checkpoint单步重放均逐位一致且对象未污染，只证明该fixture通过，不替原root1消除异常。debug记录和成本独立归档。
