# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [UPSTREAM_COMMIT.txt](<UPSTREAM_COMMIT.txt>) | 原版说明／依赖／许可文档；辅助理解同目录协议，不是推理入口。 | [F0354](<../../../../../docs/files/F0354.md>) |
| [__init__.py](<__init__.py>) | Python包初始化/命名空间，通常不应作为独立实验入口运行。 | [F0355](<../../../../../docs/files/F0355.md>) |
| [analyze.py](<analyze.py>) | 在所属实验数据上计算/复核analyze.py，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0356](<../../../../../docs/files/F0356.md>) |
| [common.py](<common.py>) | 为所属协议提供共享读写、哈希、评分、token计数/API或状态工具；同名不同hash版本不能直接替换。 | [F0357](<../../../../../docs/files/F0357.md>) |
| [download_planner.sh](<download_planner.sh>) | 下载/安装或初始化该实验依赖、数据和资源；会改变环境或占用磁盘。 | [F0358](<../../../../../docs/files/F0358.md>) |
| [export_review.sh](<export_review.sh>) | 整理/构建或导出结果审阅交付材料，保留所需代码/元数据/选定轨迹；不是模型训练收益本身。 | [F0359](<../../../../../docs/files/F0359.md>) |
| [orchestrate.sh](<orchestrate.sh>) | 调度多个分片/阶段，等待worker、合并记录并推进分析；不应与模型的单条轨迹决策逻辑混淆。 | [F0360](<../../../../../docs/files/F0360.md>) |
| [plan_questions.py](<plan_questions.py>) | 调用规划器分解子问题并估计难度，为Plan-and-Budget提供调度输入。 | [F0361](<../../../../../docs/files/F0361.md>) |
| [prepare_hotpot.py](<prepare_hotpot.py>) | 准备、筛选或冻结Hotpot输入及ID/hash；调用前核对是否读取标签用于离线筛选。 | [F0362](<../../../../../docs/files/F0362.md>) |
| [prompts.py](<prompts.py>) | 本实验的真实模型提示模板；更改本文件会改变实验干预。 | [F0363](<../../../../../docs/files/F0363.md>) |
| [requirements.txt](<requirements.txt>) | 依赖声明或配置快照；请结合所属协议读取具体字段，不是可直接执行的实验。 | [F0364](<../../../../../docs/files/F0364.md>) |
| [run_trajectories.py](<run_trajectories.py>) | 从冻结root按本实验预算协议续写，执行检索循环，保存完整episode与失败/成本。 | [F0365](<../../../../../docs/files/F0365.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0366](<../../../../../docs/files/F0366.md>) |
| [services.sh](<services.sh>) | Bash服务启动/检查入口；不同实验的GPU/端口约束不同。 | [F0367](<../../../../../docs/files/F0367.md>) |
| [tmux_run.sh](<tmux_run.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0368](<../../../../../docs/files/F0368.md>) |
