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

### 进行中

- ProbPose、BBoxMaskPose、Sapiens2 接入实时程序
- 长序列稳定性测试
- 推理线程与显示线程进一步解耦
- 双目输入与左右视图人体关联

### 尚未完成

- 鱼眼双目标定与畸变校正
- 双目关键点匹配和三角测量
- 三维骨骼重建
- 步态周期、步频、关节角度和对称性分析
- 助步器端实际部署

## 仓库结构

```text
walker-pose-system/
├─ realtime_app/          实时姿态估计与可视化程序
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
- 当前结果是二维姿态估计，不代表已经完成双目三维重建和步态分析。
- 模型权重、数据集和 Docker 镜像需要用户自行准备。

## 第三方项目与引用

本仓库不重新分发第三方模型权重和完整第三方源码。使用前请阅读各第三方项目的许可证和使用限制。

详见 `third_party/README.md`。
