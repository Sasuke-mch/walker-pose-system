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
├─ run_tool.py                  离线工具目录与统一调用入口
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
├─ walker_tools/                按职责组织的离线工具实现
│  ├─ capture/                 采集与设备预览
│  ├─ calibration/             标定及标定检查
│  ├─ pose2d/                  二维输入、ROI、检测与参考评价
│  ├─ stereo/                  配对、双目观测与几何诊断
│  ├─ lower_limb/              下肢轨迹、运动学及候选评价
│  ├─ scene/                   地面、阶段、助步器与场景诊断
│  ├─ body/                    身体拟合、精修与身体审计
│  ├─ hands/                   手部输入、腕参考与抓握构造
│  ├─ visualization/           查看器、图像和视频导出
│  ├─ commands.json            工具名称与历史文件名对应表
│  └─ catalog.py               不加载模型的入口查询接口
├─ tools/                       历史路径兼容入口及尚未迁移的本地代码
├─ tests/
└─ docs/
   └─ STEREO_TRIANGULATION.md
```

目录按职责分类，不表示方法优先级或验收级别。主线、对照、工程候选、诊断及已撤回方法的身份仍以 `AI_PROGRESS.md` 和各实验协议为准；不能根据目录名升级任何结果。

### 离线工具调用

在仓库根目录查看工具目录或调用原有方法：

```powershell
.\.venv-cuda\Scripts\python.exe -B realtime_app/run_tool.py --list
.\.venv-cuda\Scripts\python.exe -B realtime_app/run_tool.py --list body
.\.venv-cuda\Scripts\python.exe -B realtime_app/run_tool.py body.run_sapiens_cold_smplh --help
.\.venv-cuda\Scripts\python.exe -B realtime_app/run_tool.py body.fit_smplh_wilor_sequence --help
```

`run_tool.py` 只分发命令，后续参数原样传入，不改变工作目录、默认参数或方法选择。也可以使用历史文件名，如 `run_tool.py run_sapiens_cold_smplh.py --help`。原来的 `python realtime_app/tools/<文件名>.py ...` 命令继续有效；原 `tools.<模块>` 和直接模块导入也指向同一个实现，保留公开及既有私有函数。

新增或维护工具时编辑 `walker_tools/<职责>/<原文件名>.py`，不要把算法重新写进 `tools/` 兼容文件。工具依赖使用完整的 `walker_tools.<职责>.<模块>` 导入。新增入口同步登记 `commands.json`；不要在包的 `__init__.py` 中加载模型或执行任务。

`pose_app/` 保留现有公共模块路径，负责复用的运行逻辑、数据结构和算法；`walker_tools/` 负责参数解析、编排与文件输出；`tests/` 验证这些接口；`research_records/` 保存协议与历史实验。模型文件、采集数据、已生成结果和历史实验目录不随代码归类迁移。

核心依赖方向为 `run.py/run_stereo.py → pose_app`、`walker_tools → pose_app`。核心模块不反向导入离线工具。现有扩框规则集中在 `pose_app/person_roi.py`，在线客户端与离线ROI生成器使用同一个 `RULE` 和 `foot_inclusive_box()`；原工具仍导出这些名称，历史调用不变。身体原始观测集中在 `pose_app/body_observations.py`，不再由当前工具运行时加载历史实验脚本。

工具的应用根、仓库根和兼容目录经 `walker_tools._compat` 统一引用 `pose_app.project_paths`；除了直接启动脚本所需的最小包定位，不再重复推算项目路径。特定第三方运行器仍遵守各自的显式路径配置。

### SMPL-H 拟合代码

`walker_tools/body/fit_smplh_wilor_sequence.py` 负责命令行参数，调用 `pose_app/smplh_fitting/pipeline.py` 的 `run_fit(args)`。历史入口和统一调度入口继续使用相同参数；共享 beta、阶段预算、损失权重、失败门和输出字段不变。

| 模块 | 职责 |
|---|---|
| `smplh_fitting/pipeline.py` | 输入审计、模型建立、优化循环、总损失组装及结果保存 |
| `smplh_fitting/initialization.py` | 肩髋刚性坐标基、根旋转和同一旋转模板下的平移初始化；返回 `RigidInitialization` |
| `smplh_fitting/stages.py` | A/B/C/D1/D2/D3 的预算、可训练参数及冻结操作 |
| `smplh_fitting/losses.py` | 二维重投影、身体/手部时序及手部加权残差 |
| `smplh_fitting/contact.py` | 接触输入校验、`SurfaceTargets` 数据结构和手掌/足底表面损失 |
| `smplh_hand_observation.py` | WiLoR 关联、手部观测适配和几何审计；原工具重导出 `read_wilor` |

修改单项损失时，从对应模块进入；修改阶段可训练参数时，从 `stages.py` 进入；查看各损失何时参与总目标，仍读 `pipeline.py`。初始化只计算几何并更新根旋转/平移，审计文件的保存和停止门保留在编排层。接触量通过有字段名的数据结构传递，不使用 `locals()` 注入或无约定的全局字典。导入拟合包本身不加载模型，命令行帮助也不加载求解器。

阶段步数必须是非负整数；学习率、像素尺度和初始化 RMS 上限必须有限且为正，损失权重必须有限且非负。非法参数在建立输出目录和读取模型前拒绝。接触顶点必须是 0–6889 的整数索引，旋转使用项目现有的合法旋转检查；全局手表面目标必须与每个顶点一一对应。优化器更新前检查梯度，更新后检查参数，保存结果前检查模型输出。非有限像素只在运算副本中置零，原始观测和无效标记保留在结果中。

这不是对所有大文件的机械拆分。抓握的平衡目标、约束感知更新和构造抓握保留各自模块，以免混淆方法边界。场景 benchmark、标定工具的大函数尚未在本轮拆分；拟合编排层也仍包含模型装载与产物组装，后续应按输入/输出契约继续提取，而不是为降低行数切成任意小文件。

回归检查从各自子项目目录运行，部分既有测试的子进程依赖该目录：

```powershell
# 当前目录为 realtime_app
..\.venv-cuda\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
# sequence_pipeline 的测试同样从 sequence_pipeline 目录执行。
```

本次迁移只涉及未有本地修改的已跟踪工具。原有未跟踪工具与 `tools/render_raw_visual_handle_interaction_video.py` 的本地修改留在原位，未借整理一并纳入版本。`pose_app` 中既有未跟踪依赖也未被顺带提交；干净环境复现仍需单独核定这些文件的版本归属。

### 接口约定

| 接口层 | 固定入口或结构 | 本次整理的约束 |
|---|---|---|
| 在线运行 | `run.py`、`run_stereo.py` | 不改变启动参数、检测器选择和服务配置 |
| 项目路径 | `pose_app.project_paths` | 项目资源以仓库路径为锚；用户相对输入仍按原工具约定解析 |
| 观测数据 | `pose_app.schema`、`sources`、`stereo_sources` | 不改字段、帧身份、时间语义与相机角色 |
| 人物ROI | `pose_app.person_roi` | 同一扩框公式、阈值及边界裁剪；坐标回映仍由原调用方负责 |
| 正式双目几何 | `pose_app.triangulation` | 不改阈值、人物关联、拒绝原因和单位 |
| 身体软观测 | `pose_app.body_observations` | 原始COCO读取、逆旋转与软权重三角化；保持与正式严格门的区别 |
| 场景与阶段 | `pose_app.realtime_stage_walker` 等原模块 | 不改阶段判定、状态机和地面估计 |
| 身体与手部工具 | `walker_tools.body`、`walker_tools.hands` | 不改共享参数、优化阶段、损失、预算及候选状态 |
| 可视化 | `walker_tools.visualization` | 保留同源输入、数组格式和显示规则 |

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


## 离线主线的独立安装与验证

以下配置面向 Windows、Python 3.12、已有双目二维观测和 WiLoR 输出的 SMPL-H 拟合。实时相机、Docker 检测器另按对应部署说明配置；安装此配置不等于完成这些设备的验证。请在仓库根目录执行：

```powershell
uv venv --python 3.12 --seed .venv-mainline
uv pip install --python .venv-mainline/Scripts/python.exe torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-mainline/Scripts/python.exe -r realtime_app/requirements-dev.txt --no-build-isolation
uv pip check --python .venv-mainline/Scripts/python.exe
.venv-mainline/Scripts/python.exe -B realtime_app/run_validation.py
```

`requirements-mainline.txt` 固定直接依赖；`requirements-mainline-lock.txt` 保存本次 Windows 环境的完整版本清单。严格复装可在安装 CUDA PyTorch 后用该清单替代 requirements-dev.txt。安装需要 Git，以获取固定版本的官方 human-body-prior 源码；该版本的 VPoser 解码代码与本项目原环境一致，避免原环境中包元数据与 PyTorch 版本不一致的问题。旧 chumpy 安装需要预先存在的 setuptools，因此使用 --seed 和 --no-build-isolation。

模型、采集数据及历史实验目录不随代码分发。SMPL-H、MANO、VPoser 应通过各自授权来源取得；COCO 网格回归矩阵也须独立提供。路径全部通过现有拟合参数指定，新增 --canonical-mano-right 允许将 WiLoR 标准右手 MANO 资源放在仓库外，默认路径与原代码一致。

复制 `realtime_app/smplh_fit.example.json`，修改其中观测、标定、模型和输出路径。它是参数模板，不是新的实验协议；实际步数、权重、腕参考和接触设置继续服从对应实验协议。原始二维坐标必须处于原始鱼眼图像坐标系，左右相机和 WiLoR 记录必须来自同一序列；不要用转正图坐标直接进入几何。

```powershell
.venv-mainline/Scripts/python.exe -B realtime_app/preflight_smplh.py --args-json my_fit.json --load-models
$fitArgs = Get-Content my_fit.json -Raw | ConvertFrom-Json
.venv-mainline/Scripts/python.exe -B realtime_app/tools/fit_smplh_wilor_sequence.py @fitArgs
```

预检查缺少资源时返回非零状态；--load-models 还验证标定、回归矩阵、原生 MANO/SMPL-H 接口和 VPoser 解码。正式拟合继续执行原有观测、关联、约束和有限数值检查，预检查不能替代这些检查。输出目录必须为空。人工腕参考只有在显式提供时才启用，不会从旧结果自动读取。

公开测试使用小型合成数据和冻结的历史函数契约，不再要求私有历史脚本。未提供授权模型或历史采集数据的测试会明确跳过；跳过不代表模型前向或实测精度通过。只有实际配置这些资源后才能完成对应验证。
