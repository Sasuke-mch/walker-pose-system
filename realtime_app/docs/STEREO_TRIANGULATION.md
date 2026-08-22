# 双目采集与三角化链路

## 1. 当前系统分层

```text
HF868 LEFT / RIGHT
        ↓
OpenCV + Windows MSMF
        ↓
pose_app/stereo_camera.py
        ↓
CameraFrame + StereoPair
        ↓
2D Pose（YOLO26x-pose / PMPose）
        ↓
左右 PersonPose / COCO17
        ↓
K1 D1 K2 D2 R T
        ↓
去畸变 + 人物关联 + triangulation
        ↓
正深度 + 重投影误差检查
        ↓
3D COCO17（左相机坐标系）
```

## 2. 摄像头层的唯一正式实现

底层双摄采集只保留在：

```text
pose_app/stereo_camera.py
```

`pose_app/stereo_sources.py` 不再自己打开摄像头或实现另一套配对逻辑；它只把正式相机模块适配到现有 `run_stereo.py` 输入接口，并保留双视频输入。

这样避免出现“两套 StereoCameraSource 逻辑不一致”的维护问题。

### 当前实测硬件配置

```text
LEFT=1
RIGHT=0
1920×1080 @ 30 FPS
backend=MSMF
```

这组参数来自当前 HF868 + Windows 主机实测。换电脑/USB 接口后 camera id 可能变化，应重新运行：

```powershell
python .\tools\camera_probe.py
```

## 3. 时间戳到底是什么

相机线程的核心顺序：

```text
cap.read()
   ↓ 返回图像
perf_counter_ns()
```

因此 `host_return_timestamp_ns` 是主机端 `read()` 返回后的时间。路径中可能包含：

```text
Sensor曝光/读出
→ 相机内部处理
→ USB
→ Windows UVC 驱动
→ Media Foundation
→ OpenCV
→ read() 返回
→ timestamp
```

所以它不能被称为：

```text
曝光时间戳
硬件同步误差
两相机真实曝光差
```

正确表述是：

```text
host-side read-return timestamp
host-side paired-frame time difference
```

## 4. 在线配对

正式相机模块采用：

```text
one-to-one
monotonic
one-frame lookahead
nearest-time online pairing
```

对队头 `L0/R0` 不会只因为差值小于阈值就立即配对，还检查：

```text
|L1-R0|
|L0-R1|
```

是否明显更小。

当前默认：

```text
max_pair_delta_ms = 25
queue_size = 8
```

`max_pair_delta_ms` 是主机侧配对门限，不是硬件同步指标。以后异步鲁棒性实验可以把它作为配置变量，而不是写死在算法中。

## 5. 正式采集工具

```powershell
python .\tools\capture_stereo.py --duration 30 --no-preview
```

输出：

```text
left_capture.avi
right_capture.avi
left_frames.csv
right_frames.csv
stereo_pairs.csv
metadata.json
summary.json
```

### `left_frames.csv / right_frames.csv`

每个实际保存的相机帧记录：

```text
frame_id
host_return_timestamp_ns
read_duration_ms
```

### `stereo_pairs.csv`

记录在线配对关系：

```text
pair_id
left_frame_id
right_frame_id
left_host_return_timestamp_ns
right_host_return_timestamp_ns
signed_host_delta_ms_right_minus_left
abs_host_delta_ms
left_read_duration_ms
right_read_duration_ms
```

### 正式数据完整性要求

`summary.json` 中至少检查：

```text
left_read_failures == 0
right_read_failures == 0
left_recorder_queue_drops == 0
right_recorder_queue_drops == 0
recording_integrity.complete == true
```

`match_drops` 并不等于采集失败，它表示为了得到更合理的左右时间对应而主动舍弃候选帧。

## 6. 双视频模式

`run_stereo.py --left-video ... --right-video ...` 只适合**已经逐帧对齐**的双视频。

`capture_stereo.py` 保存的是两台自由运行相机各自的完整录像，它们的 frame id 不能直接按 N↔N 假定同步。离线研究时必须参考同次采集产生的：

```text
stereo_pairs.csv
```

后续如果需要对完整序列做离线全局重匹配，应单独实现离线匹配，而不要修改实时在线配对器。

## 7. 标定参数

真实三角化必须有：

```text
K1 D1
K2 D2
R T
```

约定：

```text
X_right = R @ X_left + T
```

因此三角化结果位于左相机坐标系，长度单位与 `T` 相同。

支持：

```text
camera_model = pinhole
camera_model = fisheye
```

`calibration.example.json` 仅说明格式，不能用于真实实验。

## 8. 三角化验证顺序

不要直接从“得到 R/T”跳到人体 3D。正确顺序：

```text
固定最终相机位置
→ 采集 A4 ChArUco
→ 左/右单目标定
→ 双目标定
→ 独立图像验证
→ 检查基线与物理测量
→ 已知静态尺寸目标三角化
→ 2D人体左右关键点
→ 人体三角化
→ 正深度/重投影/置信度检查
→ 时序/骨长等物理约束
```

任何移动相机位置、水平角或俯角都会改变外参，应重新标定或至少重新验证。

## 9. 当前限制

- HF868 为自由运行普通 UVC 摄像头，没有可靠的曝光硬件时间戳。
- 当前在线配对是低延迟局部最优，不是整段视频的离线全局最优匹配。
- `run_stereo.py` 当前左右图像仍按顺序分别做 2D 推理，还没有 batch=2 GPU 推理。
- 真实 ChArUco 标定链路尚未加入当前目录。
- 未加入异步运动补偿、Flow Matching、三维骨长约束和步态指标。

这些限制应逐层解决，不应通过伪造同步、伪造标定参数或简单放宽阈值掩盖。
