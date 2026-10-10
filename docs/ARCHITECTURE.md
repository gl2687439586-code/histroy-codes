# 代码分层与数据流

本仓库保留实验间的科学差异，不把所有程序拼成统一方法。

| 层次 | 职责 | 常见入口 | 后续依赖 |
|---|---|---|---|
| 数据准备 | 固定split、题号、seed；隔离gold | prepare / sample | root与生成阶段 |
| 根状态构建 | 首条query、检索与原始上下文冻结 | build_roots / prepare_roots | 各预算/动作分支 |
| 协议执行 | prompt、证据可见前缀、动作解析、检索 | prompts / actions / retrieval / evidence_views | 逐步轨迹 |
| 实验调度 | 并行分片、重试、断点续跑 | run / orchestrate / full7405 / repeat500 | 输出分片 |
| 合并与审计 | 唯一键、覆盖、协议、成本 | merge / audit / verify | 可比较结果 |
| 离线分析 | Oracle、翻转、稳定性、先验、迁移 | analyze / evaluate / predict | 表格与图 |
| 运维 | 环境、模型/检索服务、资源提醒 | tmux / services / new_server | 受控运行环境 |

在线工作流通常为：样本 → root → 预算/动作条件 → 轨迹 → 合并审计 → 统计。
评分迁移工作流为：问题评分 + 既有预算结果 → 按题隔离学习/留一 → 目标集评估。
五轮工作流为：匹配同一批题 → 各轮独立执行/根 → 对齐 → 成功集合/迁移审计。

## 不能混淆的协议

- E07：Top-5文档槽位、50-token步长、固定证据单次回答。
- E08：Top-10槽位、50-token步长，自然多轮搜索。
- E09/E22/E23：Top-5完整拼接、100-token全局前缀及FULL；各版仍需分别核对实现。
- E05/E06：Search–Read–Commit动作树，不等于自然搜索预算暴露。
- E10：FanOut及Wiki2023检索，指标与Hotpot不是同一ACC。
- Oracle需要事后标签；不能直接作为可部署先验。语义标签管理也不等于自动Judge。

请从实验页查看输入输出和入口，不要从文件名相似推断两套协议相同。
