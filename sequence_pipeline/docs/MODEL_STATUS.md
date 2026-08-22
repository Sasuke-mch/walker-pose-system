# 模型接入状态

| 模型 | 接入方式 | 权重路径策略 | 当前注意事项 |
|---|---|---|---|
| YOLO26x detector | 程序自带容器任务 | 配置或 `--weight` | 用于四个 top-down 方法的共享人体框 |
| YOLO26x-pose | 程序自带容器任务 | 配置或 `--weight` | 端到端，不读取共享检测框 |
| PMPose-b | 调用已验证的 BBoxMaskPose benchmark 适配脚本，单轮导出 | 可选本地权重；未配置时模型 API 可能下载 | 复用的是已验证推理接口，不把单轮结果当速度数据 |
| ProbPose-s | 调用当前项目已有批量脚本 | 固定容器文件名 `ProbPose-s.pth` | 迁移时必须复制该辅助脚本 |
| BBoxMaskPose | 调用已验证的 BBoxMaskPose benchmark 适配脚本，单轮导出 | PMPose 权重可覆盖；SAM 由仓库配置控制 | 依赖 `bmp_v2` 和 SAM 检查点 |
| Sapiens2-0.4B | 调用当前项目已有批量脚本 | 固定容器文件名 | 迁移时必须复制辅助脚本和 1.6GB 权重 |
