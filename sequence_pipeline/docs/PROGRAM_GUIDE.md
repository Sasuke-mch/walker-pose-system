# 程序完整说明

## 1. 它是什么

`pose_sequence_pipeline` 是主控程序，不是模型仓库。它不包含五个大型模型本身，也不会取代 `external_models`、`BBoxMaskPose` 和 `YOLO26-test`。

它负责：

```text
读取用户选择
→ 检查源码、权重、Docker 镜像
→ 扫描和自然排序图片
→ 创建独立运行目录
→ 必要时运行一次 YOLO26x 人体检测
→ 依次启动各模型容器
→ 保存日志和原始预测
→ 尝试转换为统一 JSON
→ 汇总成功和失败状态
```

## 2. 为什么模型路径不写死

所有主机路径都来自：

```text
configs/default.json
configs/local.json
命令行 --weight
```

默认模型路径是相对于 `project_root` 的相对路径。`local.json` 默认设置 `project_root=..`，因此整个项目复制到另一块磁盘或另一台电脑后，目录层级不变就能继续解析。

Docker 中使用的是容器路径，例如：

```text
Windows: D:\...\external_models\YOLO26x-pose\yolo26x-pose.pt
Docker:  /workspace/model_weights/yolo26x-pose.pt
```

主控程序自动创建挂载关系，模型脚本永远不需要理解 Windows 的盘符。

## 3. 运行阶段

### 3.1 图片清单

程序支持 JPG、JPEG、PNG、WEBP 和 BMP，并进行自然排序：

```text
frame1.jpg
frame2.jpg
frame10.jpg
```

每次运行生成 `manifest.json`，记录稳定的 `frame_index`。

### 3.2 共享人体检测

只要选择 PMPose、ProbPose、BBoxMaskPose 或 Sapiens2，主控程序先运行一次 `yolo26_detector`，生成：

```text
detections/yolo26x_detections.json
```

该 JSON 在容器内记录图片路径 `/workspace/input/...`。后续模型容器都把同一个主机输入目录挂载到 `/workspace/input`，所以路径保持有效。

### 3.3 模型执行

模型不会并行运行。完成一个容器后才启动下一个，防止 12GB 显存同时加载多个大模型。

### 3.4 输出标准化

模型的 `raw_predictions.json` 永远保留。标准化器再尝试生成 `common_predictions.json`。

标准化失败不会删除原始结果。失败原因写入运行汇总，后续可以针对特定模型增加更准确的转换器。

## 4. 配置合并

加载顺序：

```text
configs/default.json
→ configs/local.json 的 overrides
→ 命令行参数
```

建议：

- 通用结构留在 `default.json`。
- 当前电脑的路径、镜像标签和权重放在 `local.json`。
- 临时实验权重使用 `--weight`。

## 5. 调试流程

在新电脑上不要直接运行全部模型。依次执行：

```powershell
python run_sequence.py list
python run_sequence.py check --models yolo26x_pose --no-docker
python run_sequence.py check --models yolo26x_pose
python run_sequence.py run --input-dir "..." --output-dir "..." --models yolo26x_pose --dry-run
python run_sequence.py run --input-dir "..." --output-dir "..." --models yolo26x_pose
```

然后分别增加 PMPose、ProbPose、BBoxMaskPose、Sapiens2。

## 6. 失败处理

每个阶段有独立日志：

```text
logs/yolo26_detector.log
logs/pmpose.log
logs/probpose.log
...
```

默认某个阶段失败后停止。使用：

```text
--continue-on-error
```

可以继续测试后面的模型。

## 7. 断点续跑

使用固定 `--run-name` 和 `--skip-existing`：

```powershell
python run_sequence.py run ... --run-name walk01 --skip-existing
```

已存在预期输出的阶段会跳过。

## 8. 仍需注意的模型差异

- YOLO26x-pose 是端到端模型，不使用共享人体框。
- PMPose、ProbPose、BBoxMaskPose、Sapiens2 使用统一 YOLO26x 框。
- YOLO26x-pose 可能检出不同人数，这是模型差异，不是程序错误。
- Sapiens2 保存 308 点，统一结果不会替代其原始 308 点。
- BBoxMaskPose 还依赖 SAM 配置和检查点；这些由其仓库配置管理。
- PMPose 未设置本地 checkpoint 时可能联网下载权重。

## 9. 下一阶段扩展

二维序列程序稳定后，可新增而不改动当前模型调度：

```text
tracking/
stereo/
gait/
```

分别负责跨帧人员跟踪、双目三角化和步态参数计算。
