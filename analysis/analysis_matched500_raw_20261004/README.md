# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [audit_raw.py](<audit_raw.py>) | 对审计、原始轨迹执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0036](<../../docs/files/F0036.md>) |
| [inspect_inputs.py](<inspect_inputs.py>) | 对inspect_inputs.py执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0037](<../../docs/files/F0037.md>) |
| [make_report.py](<make_report.py>) | 两轮匹配500题原始prompt/output/证据审计、评分成本重算与协议核对。 本文件负责生成阅读报告。 | [F0038](<../../docs/files/F0038.md>) |
| [supplement.py](<supplement.py>) | 两轮匹配500题原始prompt/output/证据审计、评分成本重算与协议核对。 本文件负责补充核查。 | [F0039](<../../docs/files/F0039.md>) |
| [test_audit.py](<test_audit.py>) | 检查审计相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0043](<../../docs/files/F0043.md>) |
