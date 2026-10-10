# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [analyze_step_tokens.py](<analyze_step_tokens.py>) | 在所属实验数据上计算/复核token，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0114](<../../docs/files/F0114.md>) |
| [local_action_contrasts.py](<local_action_contrasts.py>) | 在Qwen-only Bamboogle自然根H=3树中比较局部Read/Search/Commit、动作顺序与质量成本bootstrap。 | [F0123](<../../docs/files/F0123.md>) |
| [pareto_frontier_h3_paper.py](<pareto_frontier_h3_paper.py>) | 将已有统计结果制作成pareto_frontier_h3_paper.py图表/表格/展示文件；不负责产生新的模型答案。 | [F0124](<../../docs/files/F0124.md>) |
| [render_pareto_frontier_h3.py](<render_pareto_frontier_h3.py>) | 将已有统计结果制作成render_pareto_frontier_h3.py图表/表格/展示文件；不负责产生新的模型答案。 | [F0129](<../../docs/files/F0129.md>) |
