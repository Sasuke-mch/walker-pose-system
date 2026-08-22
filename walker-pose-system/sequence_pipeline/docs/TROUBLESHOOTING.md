# 故障排查

## Docker daemon 不可用

先打开 Docker Desktop，确认使用 Linux containers。

## Docker image not found

运行 `docker images`，检查镜像标签是否与配置一致。

## libxcb.so.1

说明安装了 GUI 版 OpenCV。Docker 中应使用 `opencv-python-headless`。

## 路径存在但容器读取不到

检查 Docker Desktop 文件共享权限。包含中文的 Windows 路径通常可用，但如果挂载失败，可先用纯英文短路径验证。

## ProbPose/Sapiens2 辅助脚本缺失

从当前已经跑通的模型仓库复制对应 `tools` 脚本，或在 `configs/local.json` 中覆盖命令。

## PMPose 重复下载权重

为 PMPose 配置本地 checkpoint，或把缓存目录持久化挂载。当前程序支持 `--weight pmpose=...`。

## 输出 JSON 有但 common_predictions.json 为空

原始模型 JSON 结构未被通用转换器识别。原始结果仍然有效，需要为该模型增加专用 normalizer。
