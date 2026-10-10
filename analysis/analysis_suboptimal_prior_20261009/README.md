# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [evaluate.py](<evaluate.py>) | 次优单预算分配：成功频率软标签、按题隔离、旧2→新3以及4→1执行测试，对比FULL/固定预算。 本文件负责候选方法评估。 | [F0065](<../../docs/files/F0065.md>) |
| [secondary_loo.py](<secondary_loo.py>) | 四轮训练留一执行评估次优预算，同时排除测试题所有训练执行。 | [F0066](<../../docs/files/F0066.md>) |
| [verify_and_report.py](<verify_and_report.py>) | 对verify_and_report.py执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0067](<../../docs/files/F0067.md>) |
