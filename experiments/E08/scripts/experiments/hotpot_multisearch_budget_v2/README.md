# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [actions.py](<actions.py>) | 模型调用、Search/Answer解析、上下文容量约束及终止/兜底逻辑。 | [F0268](<../../../../../docs/files/F0268.md>) |
| [analyze_hotpotqa.py](<analyze_hotpotqa.py>) | 按预算汇总Hotpot质量、搜索、成本及事后Oracle。 | [F0269](<../../../../../docs/files/F0269.md>) |
| [build_roots.py](<build_roots.py>) | 为每题生成第一条query并调用检索器，冻结首轮checkpoint/证据/usage。 | [F0270](<../../../../../docs/files/F0270.md>) |
| [download_hotpotqa.sh](<download_hotpotqa.sh>) | 下载/安装或初始化该实验依赖、数据和资源；会改变环境或占用磁盘。 | [F0271](<../../../../../docs/files/F0271.md>) |
| [hp_common.py](<hp_common.py>) | Hotpot共享I/O、哈希、token边界、服务客户端、评分与状态存储。 | [F0272](<../../../../../docs/files/F0272.md>) |
| [merge_roots.py](<merge_roots.py>) | 汇总root分片，校验ID/hash/数量和共享首轮身份。 | [F0273](<../../../../../docs/files/F0273.md>) |
| [merge_trajectories.py](<merge_trajectories.py>) | 合并轨迹分片并验证条件网格、ID、DAG共享节点/保留历史等版本约束。 | [F0274](<../../../../../docs/files/F0274.md>) |
| [orchestrate.sh](<orchestrate.sh>) | 调度多个分片/阶段，等待worker、合并记录并推进分析；不应与模型的单条轨迹决策逻辑混淆。 | [F0275](<../../../../../docs/files/F0275.md>) |
| [prepare_hotpotqa.py](<prepare_hotpotqa.py>) | 将官方Hotpot数据规范化，分离无标签问题与仅离线评分使用的gold/support facts。 | [F0276](<../../../../../docs/files/F0276.md>) |
| [prompts.py](<prompts.py>) | 本实验的真实模型提示模板；更改本文件会改变实验干预。 | [F0277](<../../../../../docs/files/F0277.md>) |
| [retrieval.py](<retrieval.py>) | 调用检索服务、缓存与整理返回文档；不生成答案。 | [F0278](<../../../../../docs/files/F0278.md>) |
| [run_trajectories.py](<run_trajectories.py>) | 从冻结root按本实验预算协议续写，执行检索循环，保存完整episode与失败/成本。 | [F0279](<../../../../../docs/files/F0279.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0280](<../../../../../docs/files/F0280.md>) |
| [tmux_4gpu.sh](<tmux_4gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0281](<../../../../../docs/files/F0281.md>) |
