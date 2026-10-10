# Budget-Control Experiment Code Archive

预算控制实验代码归档：**按实验理解，按文件检索，按版本复核。**

This is a documented research-code archive, not a single installable framework.
It separates online inference, offline analysis and third-party reference code.
No raw trajectories, benchmark answers, paper drafts, model weights or API secrets are included.

## 从哪里开始 / Start here

| 你的目的 | 阅读入口 |
|---|---|
| 看有哪些实验、入口和输入输出 | [实验地图](docs/EXPERIMENTS.md) |
| 查询任一代码文件的具体功能 | [逐文件索引](docs/FILES.md) |
| 理解阶段依赖和代码分层 | [架构与数据流](docs/ARCHITECTURE.md) |
| 准备环境、恢复输入、避免误跑 | [复现边界](docs/REPRODUCIBILITY.md) |
| 看版本缺口与公开副本改动 | [来源和局限](docs/PROVENANCE.md) |
| 让AI开始协作 | [AGENTS.md](AGENTS.md) + [机器可读目录](catalog/files.json) |

## 目录结构

```text
experiments/       按 E编号 隔离的在线实验版本
  E05/...          Qwen3 Blank / Neutral / Search–Read–Commit
  E07/...          Bamboogle 固定证据单次答案生成
  E08/, E09/...    Hotpot 顺序暴露与自然搜索（不同协议）
  E10/, E12/...    FanOut 自然检索、教师需求评分
  E18/–E20/...    Plan-and-Budget、TALE-EP、Adaptive-k
  E22/, E23/...   Hotpot 全量与重复执行
analysis/          离线审计/统计/先验/迁移；保留跨目录引用层次
third_party/       SmartSearch 上游参考子集及原许可
docs/experiments/  每项实验的目的、输入、输出、入口和限制
docs/files/        每个源码/配置/模板的独立功能卡片
catalog/           JSON清单：文件、函数、参数、归属、hash
tools/             不启动模型的仓库完整性检查
```

在线目录保留 `scripts/experiments/...` 结构，便于放回独立SmartSearch仓库副本。
离线 `analysis/` 内保留历史相对层次，**不要把不同目录的同名 `analyze.py` 合并**。

## 安全的第一条命令

```bash
python3 tools/verify_repository.py
```

只校验文件、说明覆盖、链接、hash和Python语法；不安装环境、不联网、不调用模型。
**不要一键执行所有脚本。** 历史入口可能创建服务、下载大文件、覆盖输出或产生API费用。

## 重要边界

- 24项实验/分析的代码映射不等于24项都已有完整原版。19项找到相关工作流，2项部分定位，3项原版未找到。
- 本仓库不宣称每个历史数值都可仅凭代码重现；运行数据、人工标签、固定题号和环境另需恢复。
- 公开副本只做本机用户名/主机地址等机械替换，未重写预算、prompt、动作约束或评分逻辑。变更及双hash逐文件记录。
- 文本中的 `/home/researcher`、`C:/Users/RESEARCH_USER` 等是占位路径，不是可直接运行的服务器配置。
- 不新增全仓库开源许可证；第三方文件遵循其原有许可，作者原创部分的复用授权待仓库所有者决定。

