# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [actions.py](<actions.py>) | 模型调用、Search/Answer解析、上下文容量约束及终止/兜底逻辑。 | [F0438](<../../../../../docs/files/F0438.md>) |
| [analyze_answer_flips.py](<analyze_answer_flips.py>) | 相邻预算正确性翻转、MSB/SSB、配对统计和Oracle regret。 | [F0439](<../../../../../docs/files/F0439.md>) |
| [analyze_hotpotqa.py](<analyze_hotpotqa.py>) | 按预算汇总Hotpot质量、搜索、成本及事后Oracle。 | [F0440](<../../../../../docs/files/F0440.md>) |
| [analyze_repeats.py](<analyze_repeats.py>) | 新三轮的完整性、质量成本与一致性汇总。 | [F0441](<../../../../../docs/files/F0441.md>) |
| [build_roots.py](<build_roots.py>) | 为每题生成第一条query并调用检索器，冻结首轮checkpoint/证据/usage。 | [F0442](<../../../../../docs/files/F0442.md>) |
| [download_hotpotqa.sh](<download_hotpotqa.sh>) | 下载/安装或初始化该实验依赖、数据和资源；会改变环境或占用磁盘。 | [F0443](<../../../../../docs/files/F0443.md>) |
| [export_repeats.py](<export_repeats.py>) | 打包重复实验的配置、代码、root和轨迹用于审阅。 | [F0444](<../../../../../docs/files/F0444.md>) |
| [full7405.py](<full7405.py>) | 全量7405阶段编排、配置冻结、服务检查、恢复、状态及导出。 | [F0445](<../../../../../docs/files/F0445.md>) |
| [full7405.sh](<full7405.sh>) | 选择正确Python与环境后转交full7405.py；是部署入口，不是生成算法。 | [F0446](<../../../../../docs/files/F0446.md>) |
| [hp_common.py](<hp_common.py>) | Hotpot共享I/O、哈希、token边界、服务客户端、评分与状态存储。 | [F0447](<../../../../../docs/files/F0447.md>) |
| [locate_supporting_evidence.py](<locate_supporting_evidence.py>) | 对齐Hotpot支持标题/支持句到检索和已暴露前缀，记录命中等级。 | [F0448](<../../../../../docs/files/F0448.md>) |
| [merge_roots.py](<merge_roots.py>) | 汇总root分片，校验ID/hash/数量和共享首轮身份。 | [F0449](<../../../../../docs/files/F0449.md>) |
| [merge_trajectories.py](<merge_trajectories.py>) | 合并轨迹分片并验证条件网格、ID、DAG共享节点/保留历史等版本约束。 | [F0450](<../../../../../docs/files/F0450.md>) |
| [orchestrate.sh](<orchestrate.sh>) | 调度多个分片/阶段，等待worker、合并记录并推进分析；不应与模型的单条轨迹决策逻辑混淆。 | [F0451](<../../../../../docs/files/F0451.md>) |
| [prepare_hotpotqa.py](<prepare_hotpotqa.py>) | 将官方Hotpot数据规范化，分离无标签问题与仅离线评分使用的gold/support facts。 | [F0452](<../../../../../docs/files/F0452.md>) |
| [prepare_repeat500.py](<prepare_repeat500.py>) | 从历史数据冻结相同500个题号，防止重复实验随机换题。 | [F0453](<../../../../../docs/files/F0453.md>) |
| [prompts.py](<prompts.py>) | 本实验的真实模型提示模板；更改本文件会改变实验干预。 | [F0454](<../../../../../docs/files/F0454.md>) |
| [repeat500.py](<repeat500.py>) | 同一500题三次独立执行总控；每轮重新生成root并独立保存。 | [F0455](<../../../../../docs/files/F0455.md>) |
| [repeat500.sh](<repeat500.sh>) | 重复500题批次的Bash入口，转交Python总控。 | [F0456](<../../../../../docs/files/F0456.md>) |
| [retrieval.py](<retrieval.py>) | 调用检索服务、缓存与整理返回文档；不生成答案。 | [F0457](<../../../../../docs/files/F0457.md>) |
| [run_trajectories.py](<run_trajectories.py>) | 从冻结root按本实验预算协议续写，执行检索循环，保存完整episode与失败/成本。 | [F0458](<../../../../../docs/files/F0458.md>) |
| [sample_hotpotqa.py](<sample_hotpotqa.py>) | 按seed抽题并冻结选择；不是自动重复旧样本。 | [F0459](<../../../../../docs/files/F0459.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0460](<../../../../../docs/files/F0460.md>) |
| [summarize_full_tokens.py](<summarize_full_tokens.py>) | 按题平均输入/输出/总tokens、证据预算、搜索轮次及Oracle/Pareto汇总，避免只报告预算标签。 | [F0461](<../../../../../docs/files/F0461.md>) |
| [test_full7405.py](<test_full7405.py>) | 检查test_full7405.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0462](<../../../../../docs/files/F0462.md>) |
| [test_repeat500.py](<test_repeat500.py>) | 检查test_repeat500.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0463](<../../../../../docs/files/F0463.md>) |
| [tmux_2gpu.sh](<tmux_2gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0464](<../../../../../docs/files/F0464.md>) |
| [tmux_4gpu.sh](<tmux_4gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0465](<../../../../../docs/files/F0465.md>) |
