# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [eval_config.yaml](<eval_config.yaml>) | 原版配置：控制服务或实验参数；需核对资源路径和所属版本。 | [F0469](<../../../../docs/files/F0469.md>) |
| [llm_judge.py](<llm_judge.py>) | 上游LLM Judge评估入口：读取预测文件、请求评判并汇总指标。 | [F0470](<../../../../docs/files/F0470.md>) |
| [run_eval.py](<run_eval.py>) | 上游评测驱动：从配置选择naive/zero-shot/IRCOT/ReCall等推理pipeline并评估。 | [F0471](<../../../../docs/files/F0471.md>) |
| [test.py](<test.py>) | 上游临时测试入口，保留供参考；不是正文实验原始启动记录。 | [F0472](<../../../../docs/files/F0472.md>) |
| [utils.py](<utils.py>) | 上游评测辅助：执行、重试、日志初始化。 | [F0473](<../../../../docs/files/F0473.md>) |
