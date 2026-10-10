# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [retriever_config.yaml](<retriever_config.yaml>) | 原版配置：控制服务或实验参数；需核对资源路径和所属版本。 | [F0475](<../../../../docs/files/F0475.md>) |
| [retriever_serving.py](<retriever_serving.py>) | 初始化检索器，暴露health、单条search及batch-search HTTP接口。 | [F0476](<../../../../docs/files/F0476.md>) |
| [sandbox.py](<sandbox.py>) | 工具执行沙箱HTTP接口，接收并执行提交的代码；需单独审查服务安全配置。 | [F0477](<../../../../docs/files/F0477.md>) |
