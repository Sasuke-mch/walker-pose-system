# 双目实时三角化基线

## 1. 当前程序处于什么阶段

原仓库已经包含单路实时二维姿态程序：

```text
摄像头/视频/图片序列
→ YOLO26x-pose 或 YOLO26x + PMPose
→ COCO17 二维关键点
→ 实时预览、视频和 JSONL
```

本次新增的双目基线保持原程序不变，使用独立入口：

```text
run_stereo.py
```

新增链路为：

```text
左、右摄像头持续采集
→ 按主机接收时间进行近邻配对
→ 左图二维姿态估计
→ 右图二维姿态估计
→ 左右人物关联
→ 关键点去畸变
→ OpenCV 线性三角化
→ 正深度和重投影误差检查
→ 双目预览、三维 JSONL 和运行统计
```

这是一条可运行的三角化基线，不代表已经完成实机验证。必须使用真实相机和真实安装状态得到的双目标定参数。

## 2. 新增文件

```text
realtime_app/
├─ run_stereo.py
├─ calibration.example.json
├─ pose_app/
│  ├─ calibration.py
│  ├─ stereo_sources.py
│  ├─ triangulation.py
│  ├─ stereo_visualizer.py
│  └─ stereo_output.py
└─ tests/
   └─ test_triangulation.py
```

原来的 `run.py`、单路输入和模型服务没有被替换。

## 3. 标定文件要求

`calibration.example.json` 只展示结构，其中参数是示例值，禁止直接用于实验。

标定文件至少包含：

```json
{
  "camera_model": "fisheye",
  "length_unit": "meter",
  "left": {
    "image_size": [1920, 1080],
    "K": [[...], [...], [...]],
    "D": [...]
  },
  "right": {
    "image_size": [1920, 1080],
    "K": [[...], [...], [...]],
    "D": [...]
  },
  "stereo": {
    "R": [[...], [...], [...]],
    "T": [...]
  }
}
```

约定：

```text
X_right = R × X_left + T
```

因此输出三维坐标位于左相机坐标系，长度单位与 `T` 相同。若 `T` 用米，三维坐标就是米；若 `T` 用毫米，三维坐标就是毫米。

程序支持：

- `camera_model = pinhole`
- `camera_model = fisheye`

运行分辨率可以与标定分辨率等比例缩放，程序会缩放内参。宽高比变化时程序直接报错，防止错误使用标定参数。

## 4. 双摄像头运行

先复制并修改主配置：

```powershell
cd .\realtime_app
Copy-Item .\config.example.json .\config.json
```

YOLO26x-pose：

```powershell
python .\run_stereo.py `
  --model yolo26x_pose `
  --left-camera 0 `
  --right-camera 1 `
  --calibration "D:\path\to\stereo_calibration.json"
```

YOLO26x + PMPose：

```powershell
python .\run_stereo.py `
  --model pmpose `
  --left-camera 0 `
  --right-camera 1 `
  --calibration "D:\path\to\stereo_calibration.json"
```

先做十组帧对的小测试：

```powershell
python .\run_stereo.py `
  --model yolo26x_pose `
  --left-camera 0 `
  --right-camera 1 `
  --calibration "D:\path\to\stereo_calibration.json" `
  --max-pairs 10
```

无窗口运行：

```powershell
python .\run_stereo.py `
  --model yolo26x_pose `
  --left-camera 0 `
  --right-camera 1 `
  --calibration "D:\path\to\stereo_calibration.json" `
  --headless
```

## 5. 双视频离线复现

```powershell
python .\run_stereo.py `
  --model yolo26x_pose `
  --left-video "D:\data\left.mp4" `
  --right-video "D:\data\right.mp4" `
  --calibration "D:\path\to\stereo_calibration.json"
```

双视频模式目前按各自帧序号同时读取。它适用于已经对齐或从同一次双摄采集中保存的左右视频。

## 6. 关键参数

```text
--keypoint-threshold 0.25
```

左右关键点分数都达到阈值后才参与三角化。

```text
--max-reprojection-error-px 10
```

三角化后分别投影回左右图像。左右平均重投影误差超过阈值时，该三维点保留调试数据，但标记为 `valid=false`。

```text
--max-association-cost 0.05
```

多人情况下，使用去畸变后的极线几何代价关联人物。单人场景不会仅因为不同步导致代价升高就直接丢弃唯一人物对。

```text
--warn-skew-ms 40
```

左右主机接收时间差超过阈值时记录警告，但不丢弃帧对，便于后续研究不同步鲁棒性。

## 7. 输出

每次运行生成：

```text
stereo_annotated.mp4
stereo_results.jsonl
stereo_summary.json
runtime.log
docker_logs/
```

`stereo_results.jsonl` 每行包含：

```text
左右帧号
左右时间戳
主机接收时间差
左右二维姿态结果
人物关联结果
17个三维关键点
每个点的左右深度
每个点的左右重投影误差
无效原因
```

三维关键点有效性不能只看是否存在坐标，必须检查：

```text
valid == true
```

## 8. 时间戳的真实含义

普通 UVC 摄像头通常不能向 OpenCV 提供可靠的曝光时间戳。因此当前时间戳是在：

```text
VideoCapture.read() 返回后
```

使用同一台主机的单调时钟记录的接收时间。

它可以用于：

- 近邻帧配对；
- 统计两路接收时间差；
- 研究不同步对三角化的影响；

它不能证明两只摄像头在该时刻同时曝光。输出中固定记录：

```text
timestamp_type = host_receive_after_capture_read
```

## 9. 当前限制

- 左右图像当前依次进行模型推理，并非一次 GPU 批处理；双目输出速度大约低于单路速度的一半。
- 没有跨帧人物跟踪，`person_id` 仍是每帧模型输出编号。
- 多人关联是第一版极线几何贪心匹配，助步器单人场景优先。
- 没有时序补偿、运动插值或 Flow Matching。
- 没有三维骨长约束和平滑。
- 没有步态参数计算。
- 没有提供虚假的标定参数；必须先完成真实标定。

## 10. 测试

```powershell
cd .\realtime_app
python .\run_tests.py
```

测试包括已知三维点的合成立体投影和反三角化，用于验证三角化公式、外参方向和单位传递。
