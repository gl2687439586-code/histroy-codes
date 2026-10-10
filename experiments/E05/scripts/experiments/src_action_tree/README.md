# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [__init__.py](<__init__.py>) | Python包初始化/命名空间，通常不应作为独立实验入口运行。 | [F0160](<../../../../../docs/files/F0160.md>) |
| [analyze_action_tree.py](<analyze_action_tree.py>) | 离线树DP、质量/成本Oracle、动作分布与Pareto分析。 | [F0161](<../../../../../docs/files/F0161.md>) |
| [analyze_qwen3_conditions.py](<analyze_qwen3_conditions.py>) | 比较Blank、Neutral与SRC树，区分自然根/强制根并汇总质量成本。 | [F0162](<../../../../../docs/files/F0162.md>) |
| [audit_qwen3_protocol.py](<audit_qwen3_protocol.py>) | 对审计执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0163](<../../../../../docs/files/F0163.md>) |
| [common.py](<common.py>) | 为所属协议提供共享读写、哈希、评分、token计数/API或状态工具；同名不同hash版本不能直接替换。 | [F0164](<../../../../../docs/files/F0164.md>) |
| [filter_tree_roots.py](<filter_tree_roots.py>) | 读取并整理首轮root记录，合并/筛选时保留ID与条件完整性约束。 | [F0165](<../../../../../docs/files/F0165.md>) |
| [prepare_qwen3_roots.py](<prepare_qwen3_roots.py>) | 导出Qwen自然首搜根，必要时生成强制首搜根，并记录来源。 | [F0179](<../../../../../docs/files/F0179.md>) |
| [probe_services.py](<probe_services.py>) | 发出真实模型生成、Retriever搜索和必要的sandbox请求，验证服务不仅health可达。 | [F0180](<../../../../../docs/files/F0180.md>) |
| [qwen3_common.py](<qwen3_common.py>) | Qwen-only数据/检查点加载、配置和服务模型身份验证、共享事件成本汇总。 | [F0181](<../../../../../docs/files/F0181.md>) |
| [qwen3_prompts.py](<qwen3_prompts.py>) | 定义Qwen3 Blank/Neutral/动作干预真实prompt模板及匹配包装。 | [F0182](<../../../../../docs/files/F0182.md>) |
| [requirements.txt](<requirements.txt>) | 依赖声明或配置快照；请结合所属协议读取具体字段，不是可直接执行的实验。 | [F0183](<../../../../../docs/files/F0183.md>) |
| [run.sh](<run.sh>) | 该实验族的命令分派入口，转交准备/生成/合并/分析/自测；运行前须查看同版本命令分支与默认值。 | [F0184](<../../../../../docs/files/F0184.md>) |
| [run_action_tree.py](<run_action_tree.py>) | 从同一checkpoint递归展开Search/Read/Commit，执行动作约束并保存终止轨迹。 | [F0185](<../../../../../docs/files/F0185.md>) |
| [run_qwen3_natural.py](<run_qwen3_natural.py>) | 生成Qwen3-only Blank/Neutral自然搜索轨迹，并记录条件和成本。 | [F0186](<../../../../../docs/files/F0186.md>) |
| [run_qwen3_three_conditions.sh](<run_qwen3_three_conditions.sh>) | 执行该协议的条件阶段；涉及外部模型/检索或特征计算与否，见风险标记和具体入口。 | [F0187](<../../../../../docs/files/F0187.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0188](<../../../../../docs/files/F0188.md>) |
| [self_test_qwen3_protocol.py](<self_test_qwen3_protocol.py>) | 检查self_test_qwen3_protocol.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0189](<../../../../../docs/files/F0189.md>) |
| [tmux_full.sh](<tmux_full.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0190](<../../../../../docs/files/F0190.md>) |
| [tmux_qwen3_three_conditions.sh](<tmux_qwen3_three_conditions.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0191](<../../../../../docs/files/F0191.md>) |
| [verify_outputs.py](<verify_outputs.py>) | 对verify_outputs.py执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0192](<../../../../../docs/files/F0192.md>) |
