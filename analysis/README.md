# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [analyze_fanoutqa_prior_oracle_mapping.py](<analyze_fanoutqa_prior_oracle_mapping.py>) | 对齐FanOut评分与轨迹，分析需求/Oracle关系并计算同分组留一分配。 | [F0081](<../docs/files/F0081.md>) |
| [analyze_fanoutqa_v31_findings.py](<analyze_fanoutqa_v31_findings.py>) | 在所属实验数据上计算/复核FanOutQA，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0082](<../docs/files/F0082.md>) |
| [analyze_query_prior_signals.py](<analyze_query_prior_signals.py>) | 在所属实验数据上计算/复核query、先验、信号，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0083](<../docs/files/F0083.md>) |
| [analyze_ratio_oracle_prior.py](<analyze_ratio_oracle_prior.py>) | 在所属实验数据上计算/复核Oracle、先验，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0084](<../docs/files/F0084.md>) |
| [audit_cosine_received.py](<audit_cosine_received.py>) | 对审计执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0092](<../docs/files/F0092.md>) |
| [audit_fanoutqa_random100_v31.py](<audit_fanoutqa_random100_v31.py>) | 对审计、FanOutQA执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0093](<../docs/files/F0093.md>) |
| [audit_gpt56_confidence.py](<audit_gpt56_confidence.py>) | 对审计、置信执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0094](<../docs/files/F0094.md>) |
| [audit_query_prior_recovered.py](<audit_query_prior_recovered.py>) | 对审计、query、先验执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0098](<../docs/files/F0098.md>) |
| [compute_fanoutqa_total_token_oracle.py](<compute_fanoutqa_total_token_oracle.py>) | 在所属实验数据上计算/复核FanOutQA、token、Oracle，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0102](<../docs/files/F0102.md>) |
| [evaluate_query_prior_policy.py](<evaluate_query_prior_policy.py>) | 在所属实验数据上计算/复核query、先验、策略，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0105](<../docs/files/F0105.md>) |
| [export_audit_pandb_cosine500.sh](<export_audit_pandb_cosine500.sh>) | 整理/构建或导出审计交付材料，保留所需代码/元数据/选定轨迹；不是模型训练收益本身。 | [F0106](<../docs/files/F0106.md>) |
| [export_audit_pandb_polynomial500.sh](<export_audit_pandb_polynomial500.sh>) | 整理/构建或导出审计交付材料，保留所需代码/元数据/选定轨迹；不是模型训练收益本身。 | [F0107](<../docs/files/F0107.md>) |
| [make_fanoutqa_loose_curve.py](<make_fanoutqa_loose_curve.py>) | 将已有统计结果制作成FanOutQA图表/表格/展示文件；不负责产生新的模型答案。 | [F0110](<../docs/files/F0110.md>) |
| [make_fanoutqa_random100_table.py](<make_fanoutqa_random100_table.py>) | 将已有统计结果制作成FanOutQA图表/表格/展示文件；不负责产生新的模型答案。 | [F0111](<../docs/files/F0111.md>) |
| [make_read_score_loo_figure.py](<make_read_score_loo_figure.py>) | 将已有统计结果制作成阅读、评分图表/表格/展示文件；不负责产生新的模型答案。 | [F0112](<../docs/files/F0112.md>) |
| [make_read_score_loo_reference_style.py](<make_read_score_loo_reference_style.py>) | 将已有统计结果制作成阅读、评分图表/表格/展示文件；不负责产生新的模型答案。 | [F0113](<../docs/files/F0113.md>) |
| [paired_polynomial_cosine_audit.py](<paired_polynomial_cosine_audit.py>) | 按同题配对比较Polynomial/Cosine分配，计算质量成本、McNemar与配对bootstrap。 | [F0131](<../docs/files/F0131.md>) |
| [pandb_hotpot500_dual_qwen_coordinator.sh](<pandb_hotpot500_dual_qwen_coordinator.sh>) | 协调双Qwen的Plan-and-Budget500题运行阶段和日志；模型组合有别于原规划器配置。 | [F0132](<../docs/files/F0132.md>) |
| [start_pandb_hotpot500_dual_qwen.sh](<start_pandb_hotpot500_dual_qwen.sh>) | 控制start_pandb_hotpot500_dual_qwen.sh运行阶段/服务/收尾，存在启动或停止任务等副作用，不能批量盲执行。 | [F0134](<../docs/files/F0134.md>) |
| [start_pandb_hotpot500_dual_qwen_inference.sh](<start_pandb_hotpot500_dual_qwen_inference.sh>) | 控制start_pandb_hotpot500_dual_qwen_inference.sh运行阶段/服务/收尾，存在启动或停止任务等副作用，不能批量盲执行。 | [F0135](<../docs/files/F0135.md>) |
| [start_pandb_hotpot500_planning_only.sh](<start_pandb_hotpot500_planning_only.sh>) | 控制start_pandb_hotpot500_planning_only.sh运行阶段/服务/收尾，存在启动或停止任务等副作用，不能批量盲执行。 | [F0136](<../docs/files/F0136.md>) |
