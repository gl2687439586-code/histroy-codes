# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [analyze.py](<analyze.py>) | 首轮20项免费可用特征、标题分散度、成功集合及单/多候选路由回放。 本文件负责核心统计计算。 | [F0047](<../../docs/files/F0047.md>) |
| [make_report.py](<make_report.py>) | 首轮20项免费可用特征、标题分散度、成功集合及单/多候选路由回放。 本文件负责生成阅读报告。 | [F0048](<../../docs/files/F0048.md>) |
| [report_success_sets.py](<report_success_sets.py>) | 整理所属实验的report_success_sets.py统计与报告；结果有效性取决于输入记录与评分口径。 | [F0049](<../../docs/files/F0049.md>) |
| [success_set_prediction.py](<success_set_prediction.py>) | 从首轮先验预测成功预算集合/候选；区分候选命中上界与真实单答案准确率。 | [F0050](<../../docs/files/F0050.md>) |
| [supplement.py](<supplement.py>) | 首轮20项免费可用特征、标题分散度、成功集合及单/多候选路由回放。 本文件负责补充核查。 | [F0052](<../../docs/files/F0052.md>) |
| [test_analysis.py](<test_analysis.py>) | 检查test_analysis.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0053](<../../docs/files/F0053.md>) |
| [title_dispersion_only.py](<title_dispersion_only.py>) | 仅用首轮Top-5页面标题分散度，做分组统计和预算策略回放。 | [F0054](<../../docs/files/F0054.md>) |
