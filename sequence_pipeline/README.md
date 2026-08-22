# Pose Sequence Pipeline v2

一个面向 Windows + Docker Desktop + NVIDIA GPU 的批量人体姿态估计主控程序。

它接收一个图片序列目录，让用户选择一个或多个姿态估计方法，然后依次调用对应 Docker 环境，保存每个模型的姿态结果、可视化、日志和运行汇总。

## 1. 设计目标

- 用户可以选择 `YOLO26x-pose`、`PMPose`、`ProbPose`、`BBoxMaskPose`、`Sapiens2`。
- 不把五个模型强行安装到同一个 Python 环境。
- 模型源码、权重和 Docker 镜像均由配置文件管理，模型脚本中不写 Windows 绝对路径。
- Top-down 模型共享一次 YOLO26x 人体检测结果。
- 所有模型按顺序运行，避免同时占用显存。
- 支持预检、日志、断点续跑、覆盖输出、命令行权重覆盖和 dry-run。
- 保留模型原始输出，并尽力生成统一格式 `common_predictions.json`。

## 2. 推荐放置位置

把整个文件夹放在项目根目录下：

```text
基于鱼眼双目的助步器人体骨骼三维估计与步态分析系统及方法/
├─ BBoxMaskPose/
├─ YOLO26-test/
├─ external_models/
├─ pose_sequence_pipeline/      ← 本程序
├─ pose_speed_outputs/
├─ sequence_results/
└─ people/
```

默认的 `configs/local.json` 使用：

```json
{"project_root": ".."}
```

因此只要本程序位于项目根目录的下一层，换电脑后通常不需要修改项目绝对路径。

## 3. 主机要求

- Windows 10/11
- Python 3.10 或更高版本
- Docker Desktop，Linux containers 模式
- NVIDIA 驱动和 Docker GPU 支持
- 已构建以下镜像：
  - `yolo26-test-fixed:latest`
  - `bboxmaskpose:latest`
  - `sapiens2:latest`
- 模型源码和权重保持在配置中指定的位置

主机 Python 不需要安装 PyTorch、OpenCV 或 Ultralytics。本程序的主控部分只使用标准库。

## 4. 第一次使用

进入程序目录：

```powershell
cd "D:\my_works\基于鱼眼双目的助步器人体骨骼三维估计与步态分析系统及方法\pose_sequence_pipeline"
```

查看模型：

```powershell
python run_sequence.py list
```

预检全部模型：

```powershell
python run_sequence.py check --models all
```

只检查路径和配置，不检查 Docker：

```powershell
python run_sequence.py check --models all --no-docker
```

## 5. 交互式运行

直接执行：

```powershell
python run_sequence.py
```

程序会询问：

1. 输入图片目录
2. 输出根目录
3. 要运行的模型
4. 是否保存可视化

## 6. 命令行运行

单模型：

```powershell
python run_sequence.py run `
  --input-dir "D:\data\walk01" `
  --output-dir "D:\results" `
  --models yolo26x_pose `
  --save-vis
```

多个模型：

```powershell
python run_sequence.py run `
  --input-dir "D:\data\walk01" `
  --output-dir "D:\results" `
  --models "pmpose,probpose,bboxmaskpose,sapiens2" `
  --save-vis
```

全部模型：

```powershell
python run_sequence.py run `
  --input-dir "D:\data\walk01" `
  --output-dir "D:\results" `
  --models all
```

只生成并打印 Docker 命令，不执行：

```powershell
python run_sequence.py run `
  --input-dir "D:\data\walk01" `
  --output-dir "D:\results" `
  --models all `
  --dry-run
```

## 7. 临时覆盖权重

权重路径优先级：

```text
命令行 --weight
> configs/local.json 中的覆盖
> configs/default.json 中的默认路径
```

示例：

```powershell
python run_sequence.py run `
  --input-dir "D:\data\walk01" `
  --output-dir "D:\results" `
  --models yolo26x_pose `
  --weight "yolo26x_pose=D:\models\custom-pose.pt"
```

可以重复传入：

```powershell
--weight "yolo26_detector=D:\models\yolo26x.pt" `
--weight "pmpose=D:\models\PMPose-b.pth"
```

注意：ProbPose 和当前 Sapiens2 辅助脚本使用固定容器文件名。覆盖它们的权重时，建议保持文件名分别为 `ProbPose-s.pth` 和 `sapiens2_0.4b_pose.safetensors`。

## 8. 输出结构

```text
sequence_results/
└─ run_20260716_183000/
   ├─ manifest.json
   ├─ resolved_config.json
   ├─ detections/
   │  └─ yolo26x_detections.json
   ├─ yolo26x_pose/
   │  ├─ raw_predictions.json
   │  ├─ common_predictions.json
   │  └─ visualizations/
   ├─ pmpose/
   ├─ probpose/
   ├─ bboxmaskpose/
   ├─ sapiens2/
   ├─ logs/
   └─ summary/
      ├─ run_summary.json
      └─ stages.csv
```

## 9. 五个模型的执行方式

### YOLO26x-pose

由本程序自带的 `container_tasks/yolo26x_pose_sequence.py` 执行，权重通过命令行参数传入。

### PMPose

调用 BBoxMaskPose 仓库中已经验证的：

```text
BBoxMaskPose/tools/benchmark_from_yolo26_bboxes.py
```

设置 `warmup=0`、`repeat=1`，并使用 `--save-pred-json` 导出一次完整预测。它在这里被当作已验证的模型适配层，而不是用于正式速度统计。

### BBoxMaskPose

同样调用上面的已验证脚本，使用 `--method bboxmaskpose`。

### ProbPose

调用：

```text
external_models/ProbPose_code/tools/run_probpose_from_yolo26_bboxes.py
```

### Sapiens2

调用：

```text
external_models/sapiens2_repo/sapiens/pose/tools/vis/run_sapiens2_from_yolo26_bboxes.py
```

如果另一台电脑没有这两个辅助脚本，预检会明确报错；应从当前已经跑通的项目中一并复制模型仓库，而不是让程序静默失败。

## 10. 配置文件

- `configs/default.json`：程序默认配置，包含模型结构、相对路径、镜像和命令。
- `configs/local.json`：当前电脑配置。默认项目根目录为 `..`。
- `configs/local.example.json`：本地覆盖示例。

本地修改优先写入 `local.json`，不要直接在模型 Python 脚本里写绝对路径。

## 11. 常用控制参数

- `--save-vis`：保留可视化图片。部分既有模型辅助脚本内部总会先生成可视化；未指定该参数时，主控程序会在模型成功后删除对应可视化目录。
- `--skip-existing`：已有模型输出时跳过该模型。
- `--overwrite`：允许覆盖同名运行目录中的输出。
- `--continue-on-error`：某一模型失败后继续运行后续模型。
- `--run-name`：指定固定运行目录名。
- `--device`：覆盖设备，例如 `0` 或 `cuda:0`。
- `--det-conf`：覆盖 YOLO26x 检测置信度。
- `--imgsz`：覆盖 YOLO 输入尺寸。

## 12. 当前边界

本程序完成的是：

```text
图片序列 → 多模型二维姿态估计输出
```

当前不包括：

- 跨帧人员跟踪
- 左右相机同步
- 双目三角化
- 三维关键点
- 步态周期分割
- 步长、步频和关节角计算

这些应在二维模型输出稳定后作为下一层模块加入，而不应混进当前模型调度层。

详细说明见 `docs/PROGRAM_GUIDE.md`。
