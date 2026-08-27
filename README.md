# Walker Pose System

面向助步器场景的人体姿态估计与实时可视化研究代码。

## 当前完成情况

### 已完成

- YOLO26x-pose 端到端二维人体姿态估计
- YOLO26x 检测 + PMPose 两阶段姿态估计
- 视频、图片序列和摄像头输入
- 顺序处理模式
- 模拟实时模式：只保留最新帧，主动丢弃过期帧
- 可视化视频、逐帧 JSONL 和运行统计输出
- YOLO26x-pose、PMPose、ProbPose、BBoxMaskPose、Sapiens2 的速度与精度实验
- OCHuman 和 MOT17 场景测试

### 已完成的双目基线与实验管理

- 双鱼眼相机 ChArUco 标定、相机身份注册和真实采集配对回放
- 基于真实 CSV 配对关系的双目人体关联、鱼眼三角化和重投影检查
- B0 正式基线，以及局部虚拟视角和模型输入去畸变两项消融
- 原始采集、正式实验、筛选实验、工程验证和退役资产的本地归档

### 当前研究边界

- B0 是当前正式基线；局部虚拟视角的收益极小且显著降速，模型输入去畸变为负结果。
- 当前 COCO-17 输出仅含脚踝，不含脚尖或脚跟；完整脚部关键点与步态指标尚未建立。
- 尚未建立严格的真实 3D 真值精度评估；有效 3D 点数不能直接表述为精度提升。
- 步态周期、步频、关节角度、对称性分析和助步器端部署仍待后续验证。

## 仓库结构

```text
walker-pose-system/
├─ realtime_app/          单路实时姿态与双目三角化基线程序
├─ sequence_pipeline/     多模型图片序列实验与评估程序
├─ benchmarks/
│  ├─ speed/              速度汇总
│  ├─ accuracy/           精度汇总
│  └─ qualitative/        少量定性结果
├─ docs/                  项目说明
├─ third_party/           第三方依赖说明
└─ .gitignore
```

## 实时程序数据流

### YOLO26x-pose

```text
图像/视频/摄像头
→ YOLO26x-pose
→ 人体框与 COCO 17 关键点
→ 骨架绘制
→ annotated.mp4 / results.jsonl / summary.json
```

### YOLO26x + PMPose

```text
图像/视频/摄像头
→ YOLO26x 人体检测
→ 矩形人体 mask
→ PMPose-b
→ COCO 17 关键点
→ 骨架绘制
→ annotated.mp4 / results.jsonl / summary.json
```

## 环境要求

主机端：

- Windows 11
- Python 3.11 或更高版本
- Docker Desktop
- NVIDIA GPU 与可用的 Docker GPU 支持

Python 依赖：

```powershell
pip install -r .\realtime_app\requirements.txt
```

第三方模型工程和权重不包含在本仓库中。参见：

```text
third_party/README.md
```

## 配置

复制示例配置：

```powershell
Copy-Item .\realtime_app\config.example.json .\realtime_app\config.json
```

然后修改 `config.json` 中的本地路径。`config.json` 已被 `.gitignore` 排除，不会提交到 GitHub。

## 环境检查

```powershell
cd .\realtime_app
python .\check_environment.py --model all
```

## 运行示例

YOLO26x-pose 图片序列：

```powershell
python .\run.py --model yolo26x_pose --images "D:\path\to\images" --source-fps 14
```

YOLO26x + PMPose 图片序列：

```powershell
python .\run.py --model pmpose --images "D:\path\to\images" --source-fps 14
```

模拟实时处理：

```powershell
python .\run.py --model pmpose --images "D:\path\to\images" --source-fps 14 --simulate-realtime
```

摄像头：

```powershell
python .\run.py --model yolo26x_pose --camera 0
```

双摄像头三角化（使用已验证的鱼眼标定文件）：

```powershell
python .\run_stereo.py --model yolo26x_pose --camera-registry .\camera_registry.json --calibration .\calibration\results\stereo_fisheye.json
```

详细说明：`realtime_app/docs/STEREO_TRIANGULATION.md`。

## 测试

```powershell
cd .\realtime_app
python .\run_tests.py
```

## 输出

每次运行生成独立目录：

```text
annotated.mp4
results.jsonl
runtime.log
summary.json
docker_logs/
```

## 已知限制

- 当前实时程序只接入 YOLO26x-pose 和 YOLO26x + PMPose。
- PMPose 当前使用由检测框生成的矩形 mask，不等同于真实人体分割 mask。
- MOT17 没有人体关键点真值，只用于连续多人场景、稳定性和速度测试。
- 原单路程序只输出二维姿态；双目入口已输出真实采集回放上的三角化基线，但尚未完成严格的真实 3D 精度验证。
- 模型权重、数据集和 Docker 镜像需要用户自行准备。

## 第三方项目与引用

本仓库不重新分发第三方模型权重和完整第三方源码。使用前请阅读各第三方项目的许可证和使用限制。

详见 `third_party/README.md`。
