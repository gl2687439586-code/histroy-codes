# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [THIRD_PARTY_NOTICE.md](<THIRD_PARTY_NOTICE.md>) | 原版说明／依赖／许可文档；辅助理解同目录协议，不是推理入口。 | [F0392](<../../../../../docs/files/F0392.md>) |
| [adaptive_k.py](<adaptive_k.py>) | 依据检索分数间隔选择保留文档数k；不是直接选择LLM总token预算。 | [F0393](<../../../../../docs/files/F0393.md>) |
| [analyze.py](<analyze.py>) | 在所属实验数据上计算/复核analyze.py，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0394](<../../../../../docs/files/F0394.md>) |
| [audit.py](<audit.py>) | 对审计执行完整性、协议、评分、成本或可复现性检查；可能生成派生报告，不应直接视为新实验结果。 | [F0395](<../../../../../docs/files/F0395.md>) |
| [build_roots.py](<build_roots.py>) | 每题恰好一次带分数Top-N检索，根据Adaptive-k规则选择文档并冻结root；不是先让LLM生成首搜query。 | [F0396](<../../../../../docs/files/F0396.md>) |
| [common.py](<common.py>) | 为所属协议提供共享读写、哈希、评分、token计数/API或状态工具；同名不同hash版本不能直接替换。 | [F0397](<../../../../../docs/files/F0397.md>) |
| [export_review.sh](<export_review.sh>) | 整理/构建或导出结果审阅交付材料，保留所需代码/元数据/选定轨迹；不是模型训练收益本身。 | [F0398](<../../../../../docs/files/F0398.md>) |
| [finalize_probe5.sh](<finalize_probe5.sh>) | 控制finalize_probe5.sh运行阶段/服务/收尾，存在启动或停止任务等副作用，不能批量盲执行。 | [F0399](<../../../../../docs/files/F0399.md>) |
| [gate_audit_report.py](<gate_audit_report.py>) | 生成五题Gate的人类可读初步审计摘要。 | [F0400](<../../../../../docs/files/F0400.md>) |
| [merge_results.py](<merge_results.py>) | 读取并整理merge_results.py记录，合并/筛选时保留ID与条件完整性约束。 | [F0401](<../../../../../docs/files/F0401.md>) |
| [plan.py](<plan.py>) | 打印冻结运行计划，不访问模型或检索服务；具体计划字段见所属版本。 | [F0402](<../../../../../docs/files/F0402.md>) |
| [prepare_dataset.py](<prepare_dataset.py>) | 准备、筛选或冻结数据集输入及ID/hash；调用前核对是否读取标签用于离线筛选。 | [F0403](<../../../../../docs/files/F0403.md>) |
| [qa_template.txt](<qa_template.txt>) | Adaptive-k实际QA回答模板文本，必须与生成程序配套保留。 | [F0404](<../../../../../docs/files/F0404.md>) |
| [run.sh](<run.sh>) | 该实验族的命令分派入口，转交准备/生成/合并/分析/自测；运行前须查看同版本命令分支与默认值。 | [F0405](<../../../../../docs/files/F0405.md>) |
| [run_generation.py](<run_generation.py>) | 对选定检索文档生成答案，保留所选k、输出和成本。 | [F0406](<../../../../../docs/files/F0406.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0407](<../../../../../docs/files/F0407.md>) |
| [tmux_2gpu.sh](<tmux_2gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0408](<../../../../../docs/files/F0408.md>) |
| [write_retriever_config.py](<write_retriever_config.py>) | 按本地模型、索引、语料路径写Retriever配置；是配置写入工具，不是只读检查。 | [F0409](<../../../../../docs/files/F0409.md>) |
