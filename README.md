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
- 固定 60 对新采集鱼眼图像上的单侧下肢二维一致性筛选与风险复核
- Sapiens2 原始 308 点中的脚踝、大脚趾、小脚趾和脚跟候选点导出工具

### 当前研究边界

- 当前主线优先级已调整为“先完成端到端技术链，再做针对性优化”：从现有严格三角化结果继续接下肢时序接口、固定坐标、基础运动学、步态候选和统一输出；相机架、近距投影和门限优化暂缓。
- 端到端 T1–T5 软件链已完成：一次双目运行可在同一结果目录内自动写出严格三角化、固定六关节轨迹、基础运动学、受状态保护的坐标归一和非接触运动学转折点候选；运行过程中还会逐对追加仅下游消费的在线下肢状态，并在现有双目预览/标注视频中显示该状态。真实助步器/地面坐标仍待硬件锁定和静态参考采集，当前输出不代表步态参数，也未完成现场实时验收。
- 提供不接触设备的 `preflight_stereo_lower_limb_pipeline.py`：检查本机模型路径、端口配置、标定和相机注册格式，并生成硬件就绪后的建议运行命令；它不替代相机、Docker、GPU 或物理坐标验收。
- B0 是当前正式基线；局部虚拟视角的收益极小且显著降速，模型输入去畸变为负结果。
- 在冻结 M2 单人 C3 比较中，PMPose 是当前髋、膝、踝的工程二维基线：其对项目规定的 Sapiens2 操作性参考一致性高于 ProbPose；这不是外部真实准确率排名。
- Sapiens2-308 可提供脚踝、大脚趾、小脚趾和脚跟候选点。正向图像中必须先确保人体框覆盖脚部；当前正向 YOLO 框在中近距离可能截断双脚。
- Sapiens2 的脚跟是解剖点，不等于鞋底接触点；脚趾和脚跟尚未经过独立人工标注验证，不能直接计算步态事件。
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
├─ docs/                  项目状态与工程说明
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
Copy-Item .\realtime_app\camera_registry.example.json .\realtime_app\camera_registry.json
```

然后修改 `config.json` 中的本地路径，并在 `camera_registry.json` 填入已标定
相机的 Windows PnP `instance_id`。两个文件均已被 `.gitignore` 排除，不会提交到 GitHub。

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

## 当前主线（2026-09-02）

- 采集端以标定角色 `cam0=LEFT`、`cam1=RIGHT` 为准；正向模型输入固定为左图逆时针旋转 90°、右图顺时针旋转 90°，三角化始终回到原始鱼眼像素和标定参数。
- 下肢二维对照以 Sapiens2 经视觉核对后的结果作为工程伪标签参考。连续扩大人体框能减少近距离脚部被框截断的情况，PMPose、ProbPose 与 Sapiens2 的连续可视化和几何回放已固定保存于本地研究档案。
- 近距离单侧二维脚点可见，但左右关联后的完整下肢三维链仍不连续；后续重点是关联、重投影和下肢结构的逐帧诊断，而不是继续扩展检测框。
- DA3 已在原始鱼眼图上验证可提供相对深度层次和遮挡诊断信息。D1 仅将每对同步原始鱼眼图的 DA3 输出用于安全扩展图像 ROI；其相对冻结 C3 的 60 对保存预测几何回放未通过候选替代条件（下肢有效点净减 5，未筛选共同点残差未改善），因此不作为姿态输入替代方案。
- C3/D1 几何重放固定为左逆时针 90°、右顺时针 90°回到原始鱼眼坐标、`max_matches=1`。裁剪外推至原始图像范围外的点只保留在二维审计，不裁剪也不送入关联或三角化。
- DA3 不是米制深度，也不替代鱼眼标定、左右匹配、人物关联或三角化。历史“DA3 选择性去畸变”原型混用了转正图、原始鱼眼深度和标定映射，并复用了代表帧深度，未产生有效的姿态或几何评估；它不属于可用预处理链路。
- 后续主线不再扩展 DA3。冻结单人二维比较中，每张图以最高 detector bbox score 选取唯一人体，额外框仅作 false-positive 审计；Sapiens2-0.4B 是项目规定的二维操作性参考，PMPose-b 与 ProbPose-s 对其一次性比较，Sapiens2 不参与自身评分。该规则不构成外部二维或三维真值。

## 已知限制

- 当前实时程序只接入 YOLO26x-pose 和 YOLO26x + PMPose。
- PMPose 当前使用由检测框生成的矩形 mask，不等同于真实人体分割 mask。
- MOT17 没有人体关键点真值，只用于连续多人场景、稳定性和速度测试。
- 原单路程序只输出二维姿态；双目入口已输出真实采集回放上的三角化基线，但尚未完成严格的真实 3D 精度验证。
- 模型权重、数据集和 Docker 镜像需要用户自行准备。

## 第三方项目与引用

本仓库不重新分发第三方模型权重和完整第三方源码。使用前请阅读各第三方项目的许可证和使用限制。

详见 `third_party/README.md`。
