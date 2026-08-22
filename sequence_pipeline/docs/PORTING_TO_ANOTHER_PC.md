# 迁移到另一台电脑

1. 复制整个项目根目录，而不是只复制 `pose_sequence_pipeline`。
2. 安装 NVIDIA 驱动和 Docker Desktop。
3. 导入或重新构建三个 Docker 镜像。
4. 确认模型权重文件存在。
5. 保持目录层级，或修改 `configs/local.json` 的 `project_root`。
6. 运行 `python run_sequence.py check --models all`。
7. 先用两张图片逐个模型测试。

如果不复制模型仓库，本程序无法单独完成推理，因为它是调度层，不是模型发布包。
