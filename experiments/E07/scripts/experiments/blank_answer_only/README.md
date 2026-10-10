# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [__init__.py](<__init__.py>) | Python包初始化/命名空间，通常不应作为独立实验入口运行。 | [F0246](<../../../../../docs/files/F0246.md>) |
| [analyze_answer_only.py](<analyze_answer_only.py>) | 在所属实验数据上计算/复核答案，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0247](<../../../../../docs/files/F0247.md>) |
| [common.py](<common.py>) | 为所属协议提供共享读写、哈希、评分、token计数/API或状态工具；同名不同hash版本不能直接替换。 | [F0248](<../../../../../docs/files/F0248.md>) |
| [estimate_runtime.py](<estimate_runtime.py>) | 用试跑延迟、待完成条数和并发数估算剩余/总墙钟时间。 | [F0249](<../../../../../docs/files/F0249.md>) |
| [evidence_views.py](<evidence_views.py>) | 构造文档槽位及50-token顺序暴露视图；决定预算真正暴露哪些原文。 | [F0250](<../../../../../docs/files/F0250.md>) |
| [merge_shards.py](<merge_shards.py>) | 读取并整理merge_shards.py记录，合并/筛选时保留ID与条件完整性约束。 | [F0251](<../../../../../docs/files/F0251.md>) |
| [offline_lexical_hit.py](<offline_lexical_hit.py>) | 离线词法答案命中统计；不是gold supporting-fact的充分性判断。 | [F0252](<../../../../../docs/files/F0252.md>) |
| [parallel_coordinator.sh](<parallel_coordinator.sh>) | 调度多个分片/阶段，等待worker、合并记录并推进分析；不应与模型的单条轨迹决策逻辑混淆。 | [F0253](<../../../../../docs/files/F0253.md>) |
| [probe_services.py](<probe_services.py>) | 发出真实模型生成、Retriever搜索和必要的sandbox请求，验证服务不仅health可达。 | [F0254](<../../../../../docs/files/F0254.md>) |
| [prompts.py](<prompts.py>) | 本实验的真实模型提示模板；更改本文件会改变实验干预。 | [F0255](<../../../../../docs/files/F0255.md>) |
| [requirements.txt](<requirements.txt>) | 依赖声明或配置快照；请结合所属协议读取具体字段，不是可直接执行的实验。 | [F0256](<../../../../../docs/files/F0256.md>) |
| [run.sh](<run.sh>) | 该实验族的命令分派入口，转交准备/生成/合并/分析/自测；运行前须查看同版本命令分支与默认值。 | [F0257](<../../../../../docs/files/F0257.md>) |
| [run_answer_only.py](<run_answer_only.py>) | 单轮固定证据答案生成：遍历题目×预算，保存原始prompt/output/usage与严格EM/F1。 | [F0258](<../../../../../docs/files/F0258.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0259](<../../../../../docs/files/F0259.md>) |
| [tmux_4gpu.sh](<tmux_4gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0260](<../../../../../docs/files/F0260.md>) |
