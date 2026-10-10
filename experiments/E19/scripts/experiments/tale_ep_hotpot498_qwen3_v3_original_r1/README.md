# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [THIRD_PARTY_NOTICE.md](<THIRD_PARTY_NOTICE.md>) | 原版说明／依赖／许可文档；辅助理解同目录协议，不是推理入口。 | [F0371](<../../../../../docs/files/F0371.md>) |
| [__init__.py](<__init__.py>) | Python包初始化/命名空间，通常不应作为独立实验入口运行。 | [F0373](<../../../../../docs/files/F0373.md>) |
| [accounting_integration_test.py](<accounting_integration_test.py>) | 检查accounting_integration_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0374](<../../../../../docs/files/F0374.md>) |
| [analyze.py](<analyze.py>) | 在所属实验数据上计算/复核analyze.py，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0375](<../../../../../docs/files/F0375.md>) |
| [audit.py](<audit.py>) | 对审计执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0376](<../../../../../docs/files/F0376.md>) |
| [common.py](<common.py>) | 为所属协议提供共享读写、哈希、评分、token计数/API或状态工具；同名不同hash版本不能直接替换。 | [F0377](<../../../../../docs/files/F0377.md>) |
| [estimate_budgets.py](<estimate_budgets.py>) | TALE预算估计阶段；需要单独记录调用与token成本。 | [F0378](<../../../../../docs/files/F0378.md>) |
| [export_review.sh](<export_review.sh>) | 整理/构建或导出结果审阅交付材料，保留所需代码/元数据/选定轨迹；不是模型训练收益本身。 | [F0379](<../../../../../docs/files/F0379.md>) |
| [merge_records.py](<merge_records.py>) | 读取并整理merge_records.py记录，合并/筛选时保留ID与条件完整性约束。 | [F0380](<../../../../../docs/files/F0380.md>) |
| [prepare_dataset.py](<prepare_dataset.py>) | 准备、筛选或冻结数据集输入及ID/hash；调用前核对是否读取标签用于离线筛选。 | [F0381](<../../../../../docs/files/F0381.md>) |
| [prompts.py](<prompts.py>) | 本实验的真实模型提示模板；更改本文件会改变实验干预。 | [F0382](<../../../../../docs/files/F0382.md>) |
| [run.sh](<run.sh>) | 该实验族的命令分派入口，转交准备/生成/合并/分析/自测；运行前须查看同版本命令分支与默认值。 | [F0383](<../../../../../docs/files/F0383.md>) |
| [run_trajectories.py](<run_trajectories.py>) | 从冻结root按本实验预算协议续写，执行检索循环，保存完整episode与失败/成本。 | [F0384](<../../../../../docs/files/F0384.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0385](<../../../../../docs/files/F0385.md>) |
| [services.sh](<services.sh>) | Bash服务启动/检查入口；不同实验的GPU/端口约束不同。 | [F0386](<../../../../../docs/files/F0386.md>) |
| [tmux_2gpu.sh](<tmux_2gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0387](<../../../../../docs/files/F0387.md>) |
| [write_retriever_config.py](<write_retriever_config.py>) | 按本地模型、索引、语料路径写Retriever配置；是配置写入工具，不是只读检查。 | [F0388](<../../../../../docs/files/F0388.md>) |
