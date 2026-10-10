# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [analyze.py](<analyze.py>) | 按暴露预算汇总每题实际总token的范围、分位数与均值，不重新推理。 | [F0071](<../../docs/files/F0071.md>) |
| [audit_generation_caps.py](<audit_generation_caps.py>) | 审计root/续写是否命中生成上限，生成语义复核前的容量过滤与覆盖信息。 | [F0072](<../../docs/files/F0072.md>) |
| [combine_em_semantic.py](<combine_em_semantic.py>) | 合并EM正例与明确语义复核标签；未复核的非EM答案保留未知，不推断为错误。 | [F0073](<../../docs/files/F0073.md>) |
| [direct_semantic_review.py](<direct_semantic_review.py>) | 准备、展示、汇总人工/Codex语义复核批次，管理答案和标签对应关系。 | [F0074](<../../docs/files/F0074.md>) |
| [finalize_full_semantic_review.py](<finalize_full_semantic_review.py>) | 合并语义复核标签并生成最终覆盖/质量/成本汇总。 | [F0075](<../../docs/files/F0075.md>) |
| [full_semantic_review.py](<full_semantic_review.py>) | 建立盲语义复核队列并管理人工标签；本工具不自动预测标签。 | [F0076](<../../docs/files/F0076.md>) |
| [narrow_token_interval_comparison.py](<narrow_token_interval_comparison.py>) | 比较更窄的经验token区间下各预算条件的语义准确率与成本。 | [F0077](<../../docs/files/F0077.md>) |
| [package_full_semantic_review.py](<package_full_semantic_review.py>) | 将完整语义复核结果、来源清单和统计表封装成可审阅包。 | [F0078](<../../docs/files/F0078.md>) |
| [tight_token_intervals.py](<tight_token_intervals.py>) | 计算覆盖指定比例样本的最短实际token区间及分预算统计。 | [F0079](<../../docs/files/F0079.md>) |
| [verify_semantic_export.py](<verify_semantic_export.py>) | 只读核验语义导出包hash、标签覆盖和汇总重算是否一致。 | [F0080](<../../docs/files/F0080.md>) |
