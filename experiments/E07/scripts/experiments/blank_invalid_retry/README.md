# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [merge_and_analyze_retry.py](<merge_and_analyze_retry.py>) | 合并重试记录、比较逐字输出及恢复率，单独核算重试策略成本。 | [F0262](<../../../../../docs/files/F0262.md>) |
| [parallel_retry.sh](<parallel_retry.sh>) | 调度多个分片/阶段，等待worker、合并记录并推进分析；不应与模型的单条轨迹决策逻辑混淆。 | [F0263](<../../../../../docs/files/F0263.md>) |
| [run_invalid_retry.py](<run_invalid_retry.py>) | 只重跑已标记无效输出，保持原prompt/证据身份并另存重试结果。 | [F0264](<../../../../../docs/files/F0264.md>) |
| [tmux_retry.sh](<tmux_retry.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0265](<../../../../../docs/files/F0265.md>) |
