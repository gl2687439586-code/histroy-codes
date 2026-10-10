# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [analyze.py](<analyze.py>) | 五次同500题、共同七档；成功集合、重复成功频率、跨执行迁移及按题隔离先验测试。 本文件负责核心统计计算。 | [F0026](<../../docs/files/F0026.md>) |
| [check_postexecution.py](<check_postexecution.py>) | 先生成一条后再决定是否换预算的离线可行性检查；必须计入已付第一次成本。 | [F0027](<../../docs/files/F0027.md>) |
| [inspect_and_audit.py](<inspect_and_audit.py>) | 解包新三轮并复用raw审计，逐项核对五轮ID、prompt、证据、评分和成本。 | [F0028](<../../docs/files/F0028.md>) |
| [make_report.py](<make_report.py>) | 五次同500题、共同七档；成功集合、重复成功频率、跨执行迁移及按题隔离先验测试。 本文件负责生成阅读报告。 | [F0029](<../../docs/files/F0029.md>) |
| [predict_sets.py](<predict_sets.py>) | 以决策前特征做按题隔离的成功预算/成功频率预测，不将测试题历史标签当输入。 | [F0030](<../../docs/files/F0030.md>) |
| [supplement.py](<supplement.py>) | 五次同500题、共同七档；成功集合、重复成功频率、跨执行迁移及按题隔离先验测试。 本文件负责补充核查。 | [F0031](<../../docs/files/F0031.md>) |
| [test_results.py](<test_results.py>) | 检查test_results.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0032](<../../docs/files/F0032.md>) |
