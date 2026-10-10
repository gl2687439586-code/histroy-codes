# 恢复和复现边界

## 本包提供什么

提供实际历史代码、prompt模板、必要配置/依赖声明、功能说明与来源hash。
不提供模型、FAISS索引、Wikipedia/ZIM、benchmark全文、逐题gold、raw轨迹、人工语义标签、API密钥。
源代码中的字面结果断言/测试期望属于代码内容；本仓库没有另行上传结果报告或论文。

## 恢复在线实验

1. 从实验地图选择一个明确版本，阅读其入口和每份代码的功能卡。
2. 在独立的SmartSearch副本恢复该版本 `scripts/experiments/...`，不要跨版本覆盖公共模块。
3. 准备对应数据、模型/tokenizer、检索服务、root缓存、题号、seed及环境。
4. 调整占位路径、端点、GPU编号、context-length、token池、最大输出和并发。
5. 先只读配置检查，再在获得运行授权后做小规模Gate；不能把历史self-test通过当成当前服务可用。

环境文件是历史依赖提示，不是保证可用的统一锁文件。不要把多个目录requirements合并升级。
第三方框架是源码子集，不是完整安装包。部分README未发布以避免夹带历史结果；对应功能和阶段入口已重写在docs。

## 恢复离线分析

保留 `analysis/` 内部相对层次，提供各实验页列出的已有结果。
匹配500题raw审计需要旧/新两轮input、tokenizer.json和support/data；五轮分析另需repeat_1–repeat_3。
先做raw审计和聚合，再运行依赖matched_arrays等产物的先验分析。
人工语义复核需要原标签ledger，不能凭代码自动重现人的判断。

## 静态检查

```bash
python3 tools/verify_repository.py
```

这只验证目录、代码说明覆盖、hash、Python语法和本仓库生成的导航链接。
没有远程服务、CUDA或完整输入，就不能宣称端到端复现成功。

## 成本与结果口径

严格区分证据token、LLM输入/输出token、逻辑调用与共享前缀的物理调用。
按题计算平均成本，并标注样本集合、固定预算、Oracle/先验及是否发生重试。
不要在EM选择、F1选择、词法ACC评估之间暗中切换；自然根/强制根、完整/部分样本也需分开。
