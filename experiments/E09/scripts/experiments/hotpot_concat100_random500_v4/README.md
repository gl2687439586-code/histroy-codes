# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [actions.py](<actions.py>) | 模型调用、Search/Answer解析、上下文容量约束及终止/兜底逻辑。 | [F0285](<../../../../../docs/files/F0285.md>) |
| [analyze_answer_flips.py](<analyze_answer_flips.py>) | 相邻预算正确性翻转、MSB/SSB、配对统计和Oracle regret。 | [F0286](<../../../../../docs/files/F0286.md>) |
| [analyze_hotpotqa.py](<analyze_hotpotqa.py>) | 按预算汇总Hotpot质量、搜索、成本及事后Oracle。 | [F0287](<../../../../../docs/files/F0287.md>) |
| [build_roots.py](<build_roots.py>) | 为每题生成第一条query并调用检索器，冻结首轮checkpoint/证据/usage。 | [F0288](<../../../../../docs/files/F0288.md>) |
| [download_hotpotqa.sh](<download_hotpotqa.sh>) | 下载/安装或初始化该实验依赖、数据和资源；会改变环境或占用磁盘。 | [F0289](<../../../../../docs/files/F0289.md>) |
| [hp_common.py](<hp_common.py>) | Hotpot共享I/O、哈希、token边界、服务客户端、评分与状态存储。 | [F0290](<../../../../../docs/files/F0290.md>) |
| [locate_supporting_evidence.py](<locate_supporting_evidence.py>) | 对齐Hotpot支持标题/支持句到检索和已暴露前缀，记录命中等级。 | [F0291](<../../../../../docs/files/F0291.md>) |
| [merge_roots.py](<merge_roots.py>) | 汇总root分片，校验ID/hash/数量和共享首轮身份。 | [F0292](<../../../../../docs/files/F0292.md>) |
| [merge_trajectories.py](<merge_trajectories.py>) | 合并轨迹分片并验证条件网格、ID、DAG共享节点/保留历史等版本约束。 | [F0293](<../../../../../docs/files/F0293.md>) |
| [orchestrate.sh](<orchestrate.sh>) | 调度多个分片/阶段，等待worker、合并记录并推进分析；不应与模型的单条轨迹决策逻辑混淆。 | [F0294](<../../../../../docs/files/F0294.md>) |
| [prepare_hotpotqa.py](<prepare_hotpotqa.py>) | 将官方Hotpot数据规范化，分离无标签问题与仅离线评分使用的gold/support facts。 | [F0295](<../../../../../docs/files/F0295.md>) |
| [prompts.py](<prompts.py>) | 本实验的真实模型提示模板；更改本文件会改变实验干预。 | [F0296](<../../../../../docs/files/F0296.md>) |
| [retrieval.py](<retrieval.py>) | 调用检索服务、缓存与整理返回文档；不生成答案。 | [F0297](<../../../../../docs/files/F0297.md>) |
| [run_trajectories.py](<run_trajectories.py>) | 从冻结root按本实验预算协议续写，执行检索循环，保存完整episode与失败/成本。 | [F0298](<../../../../../docs/files/F0298.md>) |
| [sample_hotpotqa.py](<sample_hotpotqa.py>) | 按seed抽题并冻结选择；不是自动重复旧样本。 | [F0299](<../../../../../docs/files/F0299.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0300](<../../../../../docs/files/F0300.md>) |
| [tmux_2gpu.sh](<tmux_2gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0301](<../../../../../docs/files/F0301.md>) |
| [tmux_4gpu.sh](<tmux_4gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0302](<../../../../../docs/files/F0302.md>) |
