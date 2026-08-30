# realtime_app

`realtime_app` 是助步器项目的实时视觉程序。当前代码把三类职责分开：

```text
摄像头/视频输入
    ↓
2D Pose 服务（YOLO26x-pose / YOLO26x + PMPose）
    ↓
双目几何（标定参数、人物关联、三角化、重投影检查）
```

## 当前硬件约定

已标定物理相机的逻辑身份为 `cam0 / LEFT` 和 `cam1 / RIGHT`。Windows
OpenCV 索引会随 USB 枚举改变，因此运行时应优先使用
`camera_registry.json` 按 PnP 设备身份解析，不能硬编码索引。

两台 HF868 的 1920 × 1080 采集优先使用 MSMF；DirectShow 仅作回退。

## 目录

```text
realtime_app/
├─ run.py                       单路 2D Pose 入口
├─ run_stereo.py                双目 2D Pose + 三角化入口（需真实标定文件）
├─ run_tests.py                 全部单元测试 + 编译检查
├─ check_environment.py         Docker/模型/路径检查
├─ config.example.json          配置模板
├─ camera_registry.example.json 物理相机身份模板（本机注册表不提交）
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
Copy-Item .\camera_registry.example.json .\camera_registry.json
```

然后根据你电脑上的模型工程和权重位置修改 `config.json`，并将已标定相机的
完整 PnP `instance_id` 写入 `camera_registry.json`。

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

如果换 USB 接口、换电脑或 Windows 重新枚举设备，应重新确认 Camera
Registry 所解析的物理设备；不要把一次探测到的索引写入代码或实验命令。

## 正式双摄采集

推荐先做无预览 30 秒完整性测试：

```powershell
python .\tools\capture_stereo.py `
  --camera-registry .\camera_registry.json `
  --backend auto `
  --warmup-seconds 3 `
  --start-countdown 5 `
  --duration 30 `
  --no-preview
```

采集工具的默认模式同样会读取根目录的 `camera_registry.json`。推荐在命令中
显式写出该路径，便于记录复现实验。它会固定 `cam0 = LEFT`、`cam1 = RIGHT`，
并在本次启动时再解析各自对应的 OpenCV 索引。`--left-camera`/`--right-camera`
仅保留作诊断用途，不能作为正式采集命令或长期相机约定。

相机打开后工具会先显示 `CAMERA PRE-FLIGHT COMPLETE`，完成预热和倒计时；
只有看到 `START RECORDING NOW` 才开始写入正式 AVI、逐帧 CSV 和双目配对记录。
准备阶段的帧不会混入正式采集。

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

使用 Camera Registry 的实时双目运行：

```powershell
python .\run_stereo.py `
  --model yolo26x_pose `
  --camera-registry .\camera_registry.json `
  --max-pair-delta-ms 25 `
  --calibration .\calibration\results\stereo_fisheye.json
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

当前标定和真实 CSV 配对回放已具备。下一阶段是建立下肢/脚部的可解释
质量评估，以及脚尖、脚跟等非 COCO-17 关键点的专项模型与数据。
