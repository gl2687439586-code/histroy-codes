# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [__init__.py](<__init__.py>) | Python包初始化与导出定义，供同目录模块导入。 | [F0490](<../../../../../docs/files/F0490.md>) |
| [fid.py](<fid.py>) | Fusion-in-Decoder生成器封装；框架依赖，不代表本实验启用。 | [F0491](<../../../../../docs/files/F0491.md>) |
| [generator.py](<generator.py>) | 本地、vLLM及SGLang远程等生成后端封装。 | [F0492](<../../../../../docs/files/F0492.md>) |
| [multimodal_generator.py](<multimodal_generator.py>) | 多模态生成器封装；本预算QA主线不使用。 | [F0493](<../../../../../docs/files/F0493.md>) |
| [openai_generator.py](<openai_generator.py>) | OpenAI兼容生成器及异步循环封装。 | [F0494](<../../../../../docs/files/F0494.md>) |
| [stop_word_criteria.py](<stop_word_criteria.py>) | 模型生成时的停止词/停止序列判定。 | [F0495](<../../../../../docs/files/F0495.md>) |
| [utils.py](<utils.py>) | 生成长度配置及图像预处理等生成器辅助函数。 | [F0496](<../../../../../docs/files/F0496.md>) |
