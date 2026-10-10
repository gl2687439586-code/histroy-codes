# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [actions.py](<actions.py>) | 模型调用、Search/Answer解析、上下文容量约束及终止/兜底逻辑。 | [F0307](<../../../../../docs/files/F0307.md>) |
| [analyze_answer_flips.py](<analyze_answer_flips.py>) | 相邻预算正确性翻转、MSB/SSB、配对统计和Oracle regret。 | [F0308](<../../../../../docs/files/F0308.md>) |
| [analyze_fanoutqa.py](<analyze_fanoutqa.py>) | 在所属实验数据上计算/复核FanOutQA，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0309](<../../../../../docs/files/F0309.md>) |
| [build_roots.py](<build_roots.py>) | 为每题生成第一条query并调用检索器，冻结首轮checkpoint/证据/usage。 | [F0310](<../../../../../docs/files/F0310.md>) |
| [compare_blank_budget.py](<compare_blank_budget.py>) | 在所属实验数据上计算/复核预算，输出派生统计或诊断；若用于选策略，应检查题目隔离与未来信息边界。 | [F0311](<../../../../../docs/files/F0311.md>) |
| [download_fanoutqa.sh](<download_fanoutqa.sh>) | 下载/安装或初始化该实验依赖、数据和资源；会改变环境或占用磁盘。 | [F0312](<../../../../../docs/files/F0312.md>) |
| [download_wiki2023.sh](<download_wiki2023.sh>) | 下载/安装或初始化该实验依赖、数据和资源；会改变环境或占用磁盘。 | [F0313](<../../../../../docs/files/F0313.md>) |
| [fanoutqa_eval_local.py](<fanoutqa_eval_local.py>) | 仅加载FanOut官方确定性本地字符串指标，避免导入LLM Judge时自动创建API客户端。 | [F0314](<../../../../../docs/files/F0314.md>) |
| [hp_common.py](<hp_common.py>) | Hotpot共享I/O、哈希、token边界、服务客户端、评分与状态存储。 | [F0315](<../../../../../docs/files/F0315.md>) |
| [install_eval_deps.sh](<install_eval_deps.sh>) | 下载/安装或初始化该实验依赖、数据和资源；会改变环境或占用磁盘。 | [F0316](<../../../../../docs/files/F0316.md>) |
| [locate_necessary_evidence.py](<locate_necessary_evidence.py>) | 在FanOut检索与可见passage中定位标注的必要证据页面，区分检索到与实际暴露。 | [F0317](<../../../../../docs/files/F0317.md>) |
| [merge_roots.py](<merge_roots.py>) | 汇总root分片，校验ID/hash/数量和共享首轮身份。 | [F0318](<../../../../../docs/files/F0318.md>) |
| [merge_trajectories.py](<merge_trajectories.py>) | 合并轨迹分片并验证条件网格、ID、DAG共享节点/保留历史等版本约束。 | [F0319](<../../../../../docs/files/F0319.md>) |
| [orchestrate.sh](<orchestrate.sh>) | 调度多个分片/阶段，等待worker、合并记录并推进分析；不应与模型的单条轨迹决策逻辑混淆。 | [F0320](<../../../../../docs/files/F0320.md>) |
| [prepare_fanoutqa.py](<prepare_fanoutqa.py>) | 准备、筛选或冻结FanOutQA输入及ID/hash；调用前核对是否读取标签用于离线筛选。 | [F0321](<../../../../../docs/files/F0321.md>) |
| [prompts.py](<prompts.py>) | 本实验的真实模型提示模板；更改本文件会改变实验干预。 | [F0322](<../../../../../docs/files/F0322.md>) |
| [retrieval.py](<retrieval.py>) | 调用检索服务、缓存与整理返回文档；不生成答案。 | [F0323](<../../../../../docs/files/F0323.md>) |
| [run_trajectories.py](<run_trajectories.py>) | 从冻结root按本实验预算协议续写，执行检索循环，保存完整episode与失败/成本。 | [F0324](<../../../../../docs/files/F0324.md>) |
| [sample_fanoutqa.py](<sample_fanoutqa.py>) | 准备、筛选或冻结FanOutQA输入及ID/hash；调用前核对是否读取标签用于离线筛选。 | [F0325](<../../../../../docs/files/F0325.md>) |
| [self_test.py](<self_test.py>) | 检查self_test.py相关协议、断言或回归；部分测试使用mock，是否会加载模型应查看依赖与风险标记。 | [F0326](<../../../../../docs/files/F0326.md>) |
| [tmux_2gpu.sh](<tmux_2gpu.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0327](<../../../../../docs/files/F0327.md>) |
| [wiki2023_retriever_service.py](<wiki2023_retriever_service.py>) | 将Kiwix Wiki2023页面检索适配为FlashRAG兼容/health和/search；每页选query相关约100-word片段，不读取gold。 | [F0328](<../../../../../docs/files/F0328.md>) |
