# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [classify_subem_gap.py](<classify_subem_gap.py>) | 在所属实验数据上计算/复核classify_subem_gap.py，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0085](<../../docs/files/F0085.md>) |
| [extra_diagnostics.py](<extra_diagnostics.py>) | 在所属实验数据上计算/复核extra_diagnostics.py，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0086](<../../docs/files/F0086.md>) |
| [independent_full_audit.py](<independent_full_audit.py>) | 独立复核全量Adaptive-k输入hash、题目/轨迹、EM/F1及成本统计。 | [F0090](<../../docs/files/F0090.md>) |
| [manual_subem_gap_audit.py](<manual_subem_gap_audit.py>) | 对372条SubEM命中但EM不匹配记录附加人工语义类别，区分非规范正确、未整合和语义错误。 | [F0091](<../../docs/files/F0091.md>) |
