# 物理相机注册表与自动左右映射

## 解决的问题

OpenCV 的 `VideoCapture(0)` 和 `VideoCapture(1)` 是运行时枚举索引，不是
相机的永久身份证。两台同型号 `HD USB Camera` 重新插拔后可能交换索引；若
直接把它们用于双目标定，程序可能静默地把左右相机反过来。

此补丁通过 Windows DirectShow/Media Foundation 设备路径与
`camera_registry.json` 中的完整 PnP `instance_id` 匹配，先确定物理相机，
再得到当前 OpenCV 索引。匹配失败、匹配多个设备、或左右解析为同一索引时，
程序会拒绝启动，不会猜测。

## 固定约定

```text
cam0 = LEFT = 使用 cam0_fisheye.json = X_cam0 坐标系
cam1 = RIGHT = 使用 cam1_fisheye.json
X_cam1 = R_cam0_to_cam1 * X_cam0 + T_cam0_to_cam1
```

真实硬件身份是机器相关配置，不能公开或写死在代码中。先复制
`camera_registry.example.json` 为 `camera_registry.json`，再把本机已标定相机的
完整 PnP `instance_id` 填入其中。`friendly_name` 不用于身份判断，因为同型号
相机的名称可能相同。

## 安装一次依赖

```powershell
python -m pip install cv2-enumerate-cameras==1.3.3
```

该依赖为 Windows 相机枚举提供设备路径与对应 OpenCV 索引；项目本身仍使用
OpenCV 采集画面。

## 放置注册表

在 `realtime_app` 根目录创建本机专用的 `camera_registry.json`：

```powershell
Copy-Item .\camera_registry.example.json .\camera_registry.json
```

必须为两个条目设置以下角色：

```json
"cam0": { "role": "left" },
"cam1": { "role": "right" }
```

## 第一次硬件探测（不启动 Docker、不调用模型）

```powershell
python .\run_stereo.py `
  --config .\config.json `
  --calibration .\calibration\results\stereo_fisheye.json `
  --camera-registry .\camera_registry.json `
  --camera-backend auto `
  --camera-probe-only `
  --probe-pairs 60 `
  --log-level DEBUG
```

`auto` 的顺序为 `msmf -> dshow`。只有同一后端同时完成：

1. 以完整 `instance_id` 找到 cam0 与 cam1；
2. 打开两台相机；
3. 接受 1920×1080 的请求分辨率；
4. 收到 60 对近邻时间帧；

程序才返回成功。输出目录中的 `camera_probe_summary.json` 会保存解析到的
设备路径、索引、后端、配对统计和主机侧帧时间差。

## 正式推理

```powershell
python .\run_stereo.py `
  --config .\config.json `
  --calibration .\calibration\results\stereo_fisheye.json `
  --camera-registry .\camera_registry.json `
  --camera-backend auto `
  --model yolo26x_pose `
  --max-pairs 100
```

## 重要区别

`--connect-only` 的原有含义是“连接已运行的姿态推理服务”，并非相机连接测试。
硬件测试应使用新的 `--camera-probe-only`。

## 何时需要重新登记

若把摄像头移到不同物理 USB 端口、换 USB Hub、或换摄像头，PnP `instance_id`
可能改变。此时程序应该报错；确认物理安装与标定关系后，再更新注册表。不要
为了让程序运行而交换 `cam0`/`cam1` 或修改外参方向。
