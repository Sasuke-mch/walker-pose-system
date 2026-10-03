# Third-party dependencies

本项目通过本地目录和 Docker 镜像调用第三方工程，不在本仓库中重新分发其完整源码和模型权重。

## YOLO26 / Ultralytics

用途：

- YOLO26x-pose 端到端人体姿态估计
- YOLO26x 人体检测

本地目录示例：

```text
D:\path\to\YOLO26-test
```

需要的权重示例：

```text
yolo26x.pt
yolo26x-pose.pt
```

Docker 镜像标签示例：

```text
yolo26-realtime:latest
```

上传公开仓库前，应补充：

- 原始仓库地址
- 使用的 commit 或版本
- 原始许可证
- 本项目修改过的文件

## BBoxMaskPose / PMPose

用途：

- PMPose-b 二维人体关键点估计
- BBoxMaskPose、ProbPose 相关实验

本地目录示例：

```text
D:\path\to\BBoxMaskPose
```

Docker 镜像标签示例：

```text
bboxmaskpose:latest
```

模型缓存不应提交：

```text
model_cache/
*.pth
```

上传公开仓库前，应补充：

- 原始仓库地址
- 使用的 commit 或版本
- 原始许可证
- 本项目修改过的文件

## 数据集

本项目使用过：

- COCO
- OCHuman
- MOT17

数据集文件不包含在本仓库中。请从官方渠道下载，并遵守各自许可证。


## 离线人体拟合资源

- human-body-prior：官方 https://github.com/nghorbani/human_body_prior ，固定源码版本见 realtime_app/requirements-mainline.txt。代码许可证与模型授权按官方说明分别核对。
- SMPL-H / MANO：https://mano.is.tue.mpg.de/ 。人体和手模型通过官方授权取得，本仓库不重新分发。
- VPoser：https://github.com/nghorbani/human_body_prior 。检查下载模型的适用授权；通过 --vposer-dir 指定目录。
- WiLoR：https://github.com/rolpotamian/WiLoR 。拟合读取已有推理记录，并通过 --canonical-mano-right 指定其标准 MANO 右手资源。二维检测器和 WiLoR 推理环境不由离线拟合依赖配置替代。
- COCO17 网格回归矩阵：通过 --regressor 显式提供；本仓库不以历史实验目录作为隐式资源来源。
