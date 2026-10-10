# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [audit_bamboogle_v5.py](<audit_bamboogle_v5.py>) | 对审计、Bamboogle执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0137](<../../docs/files/F0137.md>) |
| [audit_hotpot_adaptive_k_full500.py](<audit_hotpot_adaptive_k_full500.py>) | 对审计、Hotpot执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0138](<../../docs/files/F0138.md>) |
| [compare_adaptive_k_full500_to_budget.py](<compare_adaptive_k_full500_to_budget.py>) | 在所属实验数据上计算/复核预算，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0142](<../../docs/files/F0142.md>) |
| [readscore1942_conditional_policy.py](<readscore1942_conditional_policy.py>) | 探索策略与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0143](<../../docs/files/F0143.md>) |
| [readscore1942_deeper_audit.py](<readscore1942_deeper_audit.py>) | 探索审计与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0144](<../../docs/files/F0144.md>) |
| [readscore1942_finalize.py](<readscore1942_finalize.py>) | 探索readscore1942_finalize.py与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0145](<../../docs/files/F0145.md>) |
| [readscore1942_old_semantics_overlap.py](<readscore1942_old_semantics_overlap.py>) | 探索readscore1942_old_semantics_overlap.py与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0146](<../../docs/files/F0146.md>) |
| [readscore1942_permutation_audit.py](<readscore1942_permutation_audit.py>) | 探索置换、审计与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0147](<../../docs/files/F0147.md>) |
| [readscore1942_picture_loo.py](<readscore1942_picture_loo.py>) | 探索readscore1942_picture_loo.py与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0148](<../../docs/files/F0148.md>) |
| [readscore1942_prior_analysis.py](<readscore1942_prior_analysis.py>) | 探索先验与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0150](<../../docs/files/F0150.md>) |
| [readscore1942_regime_audit.py](<readscore1942_regime_audit.py>) | 探索审计与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0152](<../../docs/files/F0152.md>) |
| [readscore1942_transfer_probe.py](<readscore1942_transfer_probe.py>) | 探索迁移与预算/质量的关系，通常是离线探针或留出验证；不能当作已部署高收益控制器。 | [F0154](<../../docs/files/F0154.md>) |
