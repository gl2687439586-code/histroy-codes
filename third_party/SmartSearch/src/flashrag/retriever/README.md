# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [__init__.py](<__init__.py>) | Python包初始化与导出定义，供同目录模块导入。 | [F0515](<../../../../../docs/files/F0515.md>) |
| [__main__.py](<__main__.py>) | 检索模块命令入口。 | [F0516](<../../../../../docs/files/F0516.md>) |
| [encoder.py](<encoder.py>) | 检索query/文档向量编码器封装。 | [F0517](<../../../../../docs/files/F0517.md>) |
| [index_builder.py](<index_builder.py>) | 构建稠密/稀疏检索索引，可能加载大量语料。 | [F0518](<../../../../../docs/files/F0518.md>) |
| [reranker.py](<reranker.py>) | Cross/Bi encoder检索结果重排器。 | [F0519](<../../../../../docs/files/F0519.md>) |
| [retriever.py](<retriever.py>) | BM25/DenseRetriever等检索实现，含缓存/重排与批量检索管理。 | [F0520](<../../../../../docs/files/F0520.md>) |
| [utils.py](<utils.py>) | 加载检索模型/语料、池化及向量数据转换。 | [F0521](<../../../../../docs/files/F0521.md>) |
