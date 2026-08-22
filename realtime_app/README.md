# realtime_app

`realtime_app` 是助步器项目的实时视觉程序。当前代码把三类职责分开：

```text
摄像头/视频输入
    ↓
2D Pose 服务（YOLO26x-pose / YOLO26x + PMPose）
    ↓
双目几何（标定参数、人物关联、三角化、重投影检查）
```

## 当前硬件默认值

针对已经实测的两台 HF868：

```text
LEFT camera id  = 1
RIGHT camera id = 0
1920 × 1080 @ 30 FPS
Windows backend = MSMF
```

不要把 DirectShow (`dshow`) 作为这两台 HF868 的默认后端。此前实测中，DSHOW 在 1080P 下只有约 5 FPS，而 MSMF 可达到约 30 FPS。

## 目录

```text
realtime_app/
├─ run.py                       单路 2D Pose 入口
├─ run_stereo.py                双目 2D Pose + 三角化入口（需真实标定文件）
├─ run_tests.py                 全部单元测试 + 编译检查
├─ check_environment.py         Docker/模型/路径检查
├─ config.example.json          配置模板
├─ calibration.example.json     双目标定 JSON 结构示例，不能直接用于实验
├─ pose_app/
│  ├─ stereo_camera.py          正式双摄采集与在线一对一时间配对（唯一底层实现）
│  ├─ stereo_sources.py         运行层适配器 + 已对齐双视频输入，不重复实现摄像头采集
│  ├─ calibration.py            K/D/R/T 读取与投影/去畸变
│  ├─ triangulation.py          人物关联与 3D 三角化
│  └─ ...
├─ tools/
│  ├─ camera_probe.py           同时显示多个摄像头，确认 camera id
│  └─ capture_stereo.py         正式双摄原始数据/时间戳/配对记录工具
├─ tests/
└─ docs/
   └─ STEREO_TRIANGULATION.md
```

## 第一次使用

从项目根目录 `D:\my_works\walker_pose_system` 进入：

```powershell
cd D:\my_works\walker_pose_system\realtime_app
```

如果本地还没有配置文件：

```powershell
Copy-Item .\config.example.json .\config.json
```

然后根据你电脑上的模型工程和权重位置修改 `config.json`。

安装主机 Python 依赖：

```powershell
pip install -r .\requirements.txt
```

## 先跑测试

```powershell
python .\run_tests.py
```

所有测试必须通过后再继续。

## 确认摄像头编号

```powershell
python .\tools\camera_probe.py
```

当前已确认：

```text
LEFT=1
RIGHT=0
```

如果换 USB 接口、换电脑或 Windows 重新枚举设备，应重新确认编号。

## 正式双摄采集

推荐先做无预览 30 秒完整性测试：

```powershell
python .\tools\capture_stereo.py --duration 30 --no-preview
```

默认输出：

```text
outputs/stereo_capture/<timestamp>/
├─ left_capture.avi
├─ right_capture.avi
├─ left_frames.csv
├─ right_frames.csv
├─ stereo_pairs.csv
├─ metadata.json
└─ summary.json
```

关键含义：

- `host_return_timestamp_ns`：`VideoCapture.read()` 返回后立即记录的主机单调时钟，不是 Sensor 曝光时间。
- `abs_host_delta_ms`：被配成一对的左右帧在上述主机时间戳上的差值，不等于硬件同步误差。
- `match_drops`：配对算法主动丢弃的旧帧，属于正常在线配对行为。
- `overflow_drops`：配对队列堆满导致的丢帧，应尽量为 0。
- `recorder_queue_drops`：录像线程来不及保存导致的原始记录缺失，正式数据中必须为 0。
- `recording_integrity.complete`：保存的视频和逐帧 CSV 是否完整覆盖采集帧。正式实验应为 `true`。

保存的 AVI 是 OpenCV 解码后的 BGR 图像再次编码成 MJPG/AVI，不是 Sensor RAW，也不是原始 UVC 字节流；精确时序以 CSV 为准。

## 单路 2D Pose

```powershell
python .\run.py --model yolo26x_pose --camera 1
```

`run.py` 与双摄底层模块相互独立，保留用于单路模型检查和性能调试。

## 双目 3D

只有真实标定完成后才能运行：

```powershell
python .\run_stereo.py `
  --model yolo26x_pose `
  --left-camera 1 `
  --right-camera 0 `
  --max-pair-delta-ms 25 `
  --calibration "D:\path\to\real_stereo_calibration.json"
```

`calibration.example.json` 只是数据结构示例，禁止直接用于真实三角化。

## 当前代码边界

`pose_app/stereo_camera.py` 只负责：

```text
打开两台摄像头
→ 两个采集线程
→ host read-return timestamp
→ 小队列
→ one-frame-lookahead 在线一对一配对
→ StereoPair
```

它不负责 YOLO/PMPose、标定或三角化。这样以后更换 2D 模型不会改摄像头层，更换配对或相机也不会改 Pose 模型。

下一阶段应完成真实 A4 ChArUco 标定采集、K1/D1/K2/D2、R/T 计算与独立验证，再进行真实人体 3D。
