# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [__init__.py](<__init__.py>) | Python包初始化/命名空间，通常不应作为独立实验入口运行。 | [F0199](<../../../../../docs/files/F0199.md>) |
| [analyze_action_tree.py](<analyze_action_tree.py>) | 离线树DP、质量/成本Oracle、动作分布与Pareto分析。 | [F0200](<../../../../../docs/files/F0200.md>) |
| [analyze_openrouter_conditions.py](<analyze_openrouter_conditions.py>) | 在所属实验数据上计算/复核条件，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0201](<../../../../../docs/files/F0201.md>) |
| [analyze_qwen3_conditions.py](<analyze_qwen3_conditions.py>) | 比较Blank、Neutral与SRC树，区分自然根/强制根并汇总质量成本。 | [F0202](<../../../../../docs/files/F0202.md>) |
| [audit_openrouter_protocol.py](<audit_openrouter_protocol.py>) | 对审计执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0203](<../../../../../docs/files/F0203.md>) |
| [audit_qwen3_protocol.py](<audit_qwen3_protocol.py>) | 对审计执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0204](<../../../../../docs/files/F0204.md>) |
| [common.py](<common.py>) | 为所属协议提供共享读写、哈希、评分、token计数/API或状态工具；同名不同hash版本不能直接替换。 | [F0205](<../../../../../docs/files/F0205.md>) |
| [filter_tree_roots.py](<filter_tree_roots.py>) | 读取并整理首轮root记录，合并/筛选时保留ID与条件完整性约束。 | [F0206](<../../../../../docs/files/F0206.md>) |
| [openrouter_common.py](<openrouter_common.py>) | OpenRouter认证、请求、重试、响应/usage规范化与账本公共层。 | [F0222](<../../../../../docs/files/F0222.md>) |
| [openrouter_prompts.py](<openrouter_prompts.py>) | 定义API模型三条件提示模板；与Qwen版本分开保留。 | [F0223](<../../../../../docs/files/F0223.md>) |
| [prepare_openrouter_roots.py](<prepare_openrouter_roots.py>) | 为OpenRouter模型准备自身首Search状态，避免混用其他模型的root。 | [F0224](<../../../../../docs/files/F0224.md>) |
| [prepare_qwen3_roots.py](<prepare_qwen3_roots.py>) | 导出Qwen自然首搜根，必要时生成强制首搜根，并记录来源。 | [F0225](<../../../../../docs/files/F0225.md>) |
| [probe_openrouter.py](<probe_openrouter.py>) | 执行该协议的probe_openrouter.py阶段；涉及外部模型/检索或特征计算与否，见风险标记和具体入口。 | [F0226](<../../../../../docs/files/F0226.md>) |
| [probe_services.py](<probe_services.py>) | 发出真实模型生成、Retriever搜索和必要的sandbox请求，验证服务不仅health可达。 | [F0227](<../../../../../docs/files/F0227.md>) |
| [qwen3_common.py](<qwen3_common.py>) | Qwen-only数据/检查点加载、配置和服务模型身份验证、共享事件成本汇总。 | [F0228](<../../../../../docs/files/F0228.md>) |
| [qwen3_prompts.py](<qwen3_prompts.py>) | 定义Qwen3 Blank/Neutral/动作干预真实prompt模板及匹配包装。 | [F0229](<../../../../../docs/files/F0229.md>) |
| [requirements.txt](<requirements.txt>) | 依赖声明或配置快照；请结合所属协议读取具体字段，不是可直接执行的实验。 | [F0230](<../../../../../docs/files/F0230.md>) |
| [run.sh](<run.sh>) | 该实验族的命令分派入口，转交准备/生成/合并/分析/自测；运行前须查看同版本命令分支与默认值。 | [F0231](<../../../../../docs/files/F0231.md>) |
| [run_action_tree.py](<run_action_tree.py>) | 从同一checkpoint递归展开Search/Read/Commit，执行动作约束并保存终止轨迹。 | [F0232](<../../../../../docs/files/F0232.md>) |
| [run_openrouter_natural.py](<run_openrouter_natural.py>) | 经OpenRouter调用历史API模型，生成Blank/Neutral自然轨迹并保存响应与usage。 | [F0233](<../../../../../docs/files/F0233.md>) |
| [run_openrouter_three_conditions.sh](<run_openrouter_three_conditions.sh>) | 执行该协议的条件阶段；涉及外部模型/检索或特征计算与否，见风险标记和具体入口。 | [F0234](<../../../../../docs/files/F0234.md>) |
| [run_qwen3_natural.py](<run_qwen3_natural.py>) | 生成Qwen3-only Blank/Neutral自然搜索轨迹，并记录条件和成本。 | [F0235](<../../../../../docs/files/F0235.md>) |
| [run_qwen3_three_conditions.sh](<run_qwen3_three_conditions.sh>) | 执行该协议的条件阶段；涉及外部模型/检索或特征计算与否，见风险标记和具体入口。 | [F0236](<../../../../../docs/files/F0236.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0237](<../../../../../docs/files/F0237.md>) |
| [self_test_openrouter.py](<self_test_openrouter.py>) | 检查self_test_openrouter.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0238](<../../../../../docs/files/F0238.md>) |
| [self_test_qwen3_protocol.py](<self_test_qwen3_protocol.py>) | 检查self_test_qwen3_protocol.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0239](<../../../../../docs/files/F0239.md>) |
| [summarize_openrouter_billing.py](<summarize_openrouter_billing.py>) | 整理所属实验的summarize_openrouter_billing.py统计与报告；结果有效性取决于输入记录与评分口径。 | [F0240](<../../../../../docs/files/F0240.md>) |
| [tmux_full.sh](<tmux_full.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0241](<../../../../../docs/files/F0241.md>) |
| [tmux_qwen3_three_conditions.sh](<tmux_qwen3_three_conditions.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0242](<../../../../../docs/files/F0242.md>) |
| [verify_outputs.py](<verify_outputs.py>) | 对verify_outputs.py执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0243](<../../../../../docs/files/F0243.md>) |
