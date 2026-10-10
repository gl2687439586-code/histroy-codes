# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [analyze_harmonized_metric_transfer.py](<analyze_harmonized_metric_transfer.py>) | 在所属实验数据上计算/复核指标、迁移，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0010](<../../docs/files/F0010.md>) |
| [analyze_mixed_pool_rules.py](<analyze_mixed_pool_rules.py>) | 对Hotpot与FanOut混合池做预算规则迁移/质量成本评估。 | [F0011](<../../docs/files/F0011.md>) |
| [analyze_quantile_transfer.py](<analyze_quantile_transfer.py>) | 在所属实验数据上计算/复核迁移，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0012](<../../docs/files/F0012.md>) |
| [analyze_task_difficulty.py](<analyze_task_difficulty.py>) | 在所属实验数据上计算/复核难度，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0013](<../../docs/files/F0013.md>) |
| [analyze_transfer.py](<analyze_transfer.py>) | 在所属实验数据上计算/复核迁移，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0014](<../../docs/files/F0014.md>) |
| [build_baseline_anchored_table.py](<build_baseline_anchored_table.py>) | 以共同baseline为锚点，整理跨数据集迁移规则的质量/成本对照表。 | [F0015](<../../docs/files/F0015.md>) |
| [plot_rule_validation.py](<plot_rule_validation.py>) | 将已有统计结果制作成规则图表/表格/展示文件；不负责产生新的模型答案。 | [F0017](<../../docs/files/F0017.md>) |
| [plot_transfer_calibrated.py](<plot_transfer_calibrated.py>) | 将已有统计结果制作成迁移、校准图表/表格/展示文件；不负责产生新的模型答案。 | [F0018](<../../docs/files/F0018.md>) |
| [validate_candidate_rules.py](<validate_candidate_rules.py>) | 源域选择超参数，目标域评估硬校准、软预算分布及query结构模型。 | [F0019](<../../docs/files/F0019.md>) |
| [validate_outputs.py](<validate_outputs.py>) | 对validate_outputs.py执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0020](<../../docs/files/F0020.md>) |
| [validate_rule_validation.py](<validate_rule_validation.py>) | 对规则执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0021](<../../docs/files/F0021.md>) |
