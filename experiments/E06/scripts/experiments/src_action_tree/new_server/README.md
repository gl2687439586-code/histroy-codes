# 本目录代码说明

本目录属于历史实验的一个版本。不要与其他目录同名模块混用。

| 文件 | 具体作用 | 详情 |
|---|---|---|
| [00_preflight.sh](<00_preflight.sh>) | 新服务器安装前检查空间、内存、工具、目录与硬件边界。 | [F0207](<../../../../../../docs/files/F0207.md>) |
| [01_create_envs.sh](<01_create_envs.sh>) | 建立实验/模型服务环境并安装依赖；可能下载Miniconda或Python包。 | [F0208](<../../../../../../docs/files/F0208.md>) |
| [02_download_assets.sh](<02_download_assets.sh>) | 下载模型、检索语料/索引及所需资源；涉及大量磁盘与网络。 | [F0209](<../../../../../../docs/files/F0209.md>) |
| [03_configure.sh](<03_configure.sh>) | 解析已下载资源路径并写服务/实验配置。 | [F0210](<../../../../../../docs/files/F0210.md>) |
| [04_validate_install.sh](<04_validate_install.sh>) | 安装后的静态数据/模型/Python协议验证；不等于GPU实际生成已成功。 | [F0211](<../../../../../../docs/files/F0211.md>) |
| [common_env.sh](<common_env.sh>) | 集中声明环境路径、缓存、模型和服务参数；应先核对新旧服务器路径。 | [F0213](<../../../../../../docs/files/F0213.md>) |
| [experiment_openrouter_with_monitor.sh](<experiment_openrouter_with_monitor.sh>) | 控制experiment_openrouter_with_monitor.sh运行阶段/服务/收尾，存在启动或停止任务等副作用，不能批量盲执行。 | [F0214](<../../../../../../docs/files/F0214.md>) |
| [experiment_with_monitor.sh](<experiment_with_monitor.sh>) | 控制experiment_with_monitor.sh运行阶段/服务/收尾，存在启动或停止任务等副作用，不能批量盲执行。 | [F0215](<../../../../../../docs/files/F0215.md>) |
| [prepare_bamboogle_input.py](<prepare_bamboogle_input.py>) | 整理Bamboogle问题/标签及原始输入，供新服务器独立运行。 | [F0216](<../../../../../../docs/files/F0216.md>) |
| [resource_monitor.py](<resource_monitor.py>) | 只读采样服务器RAM、swap、磁盘、GPU及完成进度，写资源日志和提醒。 | [F0217](<../../../../../../docs/files/F0217.md>) |
| [session_wrapper.sh](<session_wrapper.sh>) | tmux内运行命令并保留退出码/日志，避免会话关闭后丢失错误原因。 | [F0218](<../../../../../../docs/files/F0218.md>) |
| [tmux_openrouter.sh](<tmux_openrouter.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0219](<../../../../../../docs/files/F0219.md>) |
| [tmux_safe.sh](<tmux_safe.sh>) | 将该目录的服务、worker或协调器放入tmux，提供运行/状态/日志入口；可能启动GPU任务或结束特定会话。 | [F0220](<../../../../../../docs/files/F0220.md>) |
| [write_retriever_config.py](<write_retriever_config.py>) | 按本地模型、索引、语料路径写Retriever配置；是配置写入工具，不是只读检查。 | [F0221](<../../../../../../docs/files/F0221.md>) |
