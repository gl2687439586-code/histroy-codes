# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [analyze_scores.py](<analyze_scores.py>) | 在所属实验数据上计算/复核评分，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0331](<../../../../../../docs/files/F0331.md>) |
| [build_query_file.py](<build_query_file.py>) | 准备、筛选或冻结query输入及ID/hash；调用前核对是否读取标签用于离线筛选。 | [F0332](<../../../../../../docs/files/F0332.md>) |
| [run.sh](<run.sh>) | 该实验族的命令分派入口，转交准备/生成/合并/分析/自测；运行前须查看同版本命令分支与默认值。 | [F0335](<../../../../../../docs/files/F0335.md>) |
| [score_queries_openrouter.py](<score_queries_openrouter.py>) | 执行该协议的评分、query阶段；涉及外部模型/检索或特征计算与否，见风险标记和具体入口。 | [F0336](<../../../../../../docs/files/F0336.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0337](<../../../../../../docs/files/F0337.md>) |
