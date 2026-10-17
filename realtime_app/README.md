# realtime_app

## 当前SMPL-H速度测量（2026-10-13）

本机RTX5070Ti Laptop对当前448帧/560步方法实测：完整solver 42.0031s（10.6659帧/s离线吞吐）；已有参数单帧解码中位14.3135ms，批448解码29.7663ms。数据和详细计时边界见`G20261013_smplh_runtime448_v1/runtime.json`与算法第32节；解码速度不等于从观测求参数的完整拟合速度。

## 当前身体主线：448帧二维鱼眼重投影约束（2026-10-14）

当前正式人体索引为`configs/current_mainline.json/mainline_selection`，指向`research_records/engineering_validation/G20261013_reliability448_view_v1`。该主线使用SMPL-H三维模型生成模型侧COCO-17点，但身体优化的观测约束是左右原始鱼眼二维回投影：`body_3d_weight=0`、`body_observation_mode=reprojection_only`、二维回投影权重`0.004`、像素尺度`25`、踝点二维倍率`2`。因此三角化3D关节点不进入身体拟合残差；三角化数组仍可能用于冷启动/初始化和诊断，不能把“二维约束”误写成完全不读取三维数据。

该结果是448帧全片工程候选，`accepted_for_main_fit=false`仍保留。旧360帧reference25及G20261012踝权重结果保留为历史对照，不混入当前身体参数。场景仍由`mainline -> supported_stop_fixed_lag`解析；独立视觉ORB/cuVSLAM候选未通过覆盖/几何门，不改变场景主线。详细目标、输入、参数和证据边界见[现有算法第35节](../docs/CURRENT_SMPLH_ALGORITHM_20261002.md#35-当前人体主线切换为二维鱼眼重投影约束-2026-10-14)。

速度入口从仓库根执行，每次用新输出目录：

```powershell
& ./.venv-cuda/Scripts/python.exe -X utf8 -B realtime_app/tools/benchmark_smplh_runtime.py --source-args research_records/engineering_validation/G20261013_reliability448_view_v1/fit/fit_args.json --output research_records/engineering_validation/G20261013_smplh_runtime448_repeat_v1 --warmup 10 --samples 50
```

该入口只替换source argv的output-dir，在CUDA完成一次完整拟合并保存独立输出；然后测已拟合参数的float32/no_grad解码。阶段计时通过可选runtime_observer回调在边界同步CUDA；普通拟合不传回调时无额外同步。PMPose/WiLoR二维检测、场景重放和页面渲染不在此速度范围内。

## 当前方法新448帧页面（2026-10-13）

新reliability拟合及页面在`research_records/engineering_validation/G20261013_reliability448_view_v1`：完整离线入口`view/body.html`；运行预览服务后访问`view/body_browser.html`（需同目录`body_data.json.gz`）。此版本沿用最近448的二维身体目标/踝倍率2和native共享手姿，完整冷启动560步，fresh strict/Stage/地面，未使用显示平滑。当前导出入口支持显式reliability选择，并检查保存结果与复现argv的策略；旧reference25导出仍可复现。输入/参数/命令见该目录plan.json、fit/fit_args.json、selection.json；现有正式360选择不自动替换。

## SMPL-H实现说明（2026-10-13）

从输入到输出的完整说明见[现有算法第30节](../docs/CURRENT_SMPLH_ALGORITHM_20261002.md#implementation-guide)：包含模块调用链、坐标/单位、参数维度、置信度及各损失公式、初始化、A/B/C/D阶段变量和预算、接触分支、局部手计算、参数迁移与输出读取。该节依据当前reliability源码；下方reference25复现例属于legacy历史结果。特别注意steps并非总步数硬上限、通用默认未自动启用native/shared手，以及history和最终网格对应不同的更新时刻。

## 2026-10-13 新拟合采用可靠程度权重

body.fit_smplh_wilor_sequence默认--fit-policy reliability；tools/run_smplh_reprojection_experiment.py也默认新策略。身体3D使用双目较低分数及独立几何衰减，2D/3D/MANO损失按原始帧/视图/点的固定总数归约；零分和缺失不获得正权重或被重新归一化成满强度。踝倍率与检测分数分开。根锚使用旋转空间距离及肩髋可靠程度，接触D2/D3继续保留身体数据约束。

新方法尚未经验标定为概率/协方差，固定分母改变了观测相对先验的尺度，旧lambda未重新调优。当前已保存主线网格仍来自legacy；冻结cold profiles与历史命令读取器明确保留旧策略。直接使用旧argv需补--fit-policy legacy；使用新方法则保留body-weighting默认quality（实际解析为新可靠程度公式），不能携带旧reference/uniform/no_*配方。新策略要求身体3D或2D数据项恰有一项为正，避免对同一像素重复计证。

公式、输入拒绝原因和接触证据边界见../docs/CURRENT_SMPLH_ALGORITHM_20261002.md第28节。本轮只做方法和单元检查，未重拟合已保存的360/448序列。下方2026-10-05段落保留旧身体结果语义，场景mainline当前解析为supported_stop_fixed_lag。

## 历史360帧身体工程主线（2026-10-05）

该节只记录历史reference25完整冷启动结果，不再代表当前身体选择。当前选择已切换到上面的448帧二维鱼眼重投影主线；reference25仍作为可复现历史对照保留。通用求解器省略参数的兼容默认不能冒充任一冻结结果。

历史reference25复现必须读取历史360帧自身的参数，不能读取当前`mainline_selection`（那是可靠性448帧选择）。旧`fit_args`省略`--fit-policy`，而通用求解器当前默认已切换为`reliability`，因此历史复现显式补回`legacy`：

```powershell
# 历史360帧 reference25：读取历史自身的fit_args，再显式补回legacy策略。
$historical = 'research_records/engineering_validation/G20261005_old_capture_reference25_v1/runs/reference25/fit_args.json'
$fitArgs = [string[]](Get-Content $historical -Raw | ConvertFrom-Json)
$fitArgs += @('--fit-policy', 'legacy')
$fitArgs[[Array]::IndexOf($fitArgs, '--output-dir') + 1] = 'research_records/engineering_validation/reference25_new_run/fit'
& ./.venv-cuda/Scripts/python.exe -B realtime_app/run_tool.py body.fit_smplh_wilor_sequence @fitArgs
```

当前reliability复现则读取`mainline_selection.body.reproduction_arguments`，不追加`legacy`；两套配置不得混用。

最新主线可视化入口：`build_current_mainline_view.py --output <新目录>`（本地Python从仓库根调用）。它读取当前选择的二维重投影身体，重新计算三角化、重放mainline→supported_stop_fixed_lag，并生成完整自包含body.html；不重拟合身体。随后用`validate_current_mainline_view.py --view <目录>`核对网格/点/严格踝拖尾/相机/助步器与新变换。已验证页面路径保存在current_mainline.json/mainline_selection.visualization。

命令从仓库根运行；本地授权模型/缓存缺失则报告不可用。新显示应读取选择的二维重投影网格和同源supported-stop变换，历史HTML不可冒充新结果。工程主线选择仍不表示独立三维精度或物理接触认证。


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
├─ run_validation.py            公开测试统一入口（两个子项目；CI同源）
├─ run_tests.py                 旧测试入口，当前验证使用run_validation.py
├─ check_environment.py         Docker/模型/路径检查
├─ camera_config.example.json   实时服务配置模板
├─ configs/                     冷启动配置和当前路线索引
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
Copy-Item .\camera_config.example.json .\config.json
Copy-Item .\camera_registry.example.json .\camera_registry.json
```

然后根据你电脑上的模型工程和权重位置修改 `config.json`，并将已标定相机的
完整 PnP `instance_id` 写入 `camera_registry.json`。

仅运行实时采集基础模块的主机依赖：

```powershell
pip install -r .\requirements.txt
```

## 先跑测试

测试前须按下方“离线主线的独立安装与验证”安装完整requirements-dev依赖；只安装requirements.txt不足以运行全部测试。

```powershell
python -B .\run_validation.py
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


## 当前身体路线、固定配置和评价入口

当前推进的是离线整段 SMPL-H 冷启动，Sapiens2 为优先身体候选，PMPose 为完全同流程对照。早期实时三角化的 PMPose 基线继续保留，两种身份不要混用。普通 SMPL 和旧固定抓握精修仍是历史/诊断路线，不是当前冷启动的后续必经步骤。方法级别未改变：两套冷启动均 accepted_for_main_fit=False，人工腕为诊断参考，手部几何未通过，脚底负高度未解决。

`configs/current_mainline.json` 记录候选、对照、输出范围、调用入口和当前标量参数快照；`configs/cold_smplh_joint300.json` 固定当前拟合参数；`configs/cold_smplh_legacy30.json` 保留原入口默认参数。冷启动编排不再从旧 fit_shared_args.json 取参数，仍读取必要的同源观测、WiLoR、腕参考、模型和场景资产。SET_BY_COLD_RUNNER 的三个路径由编排器替换，不能直接把该配置当通用拟合 args JSON 运行。`smplh_fit.example.json` 是自备资源的通用模板，未包含本实验人工腕，不能冒充当前冻结流程。

从仓库根目录复现当前候选，输出选择不存在的新目录：

```powershell
$runDir = 'research_records/engineering_validation/G20261002_sapiens2_body_mainline448_v2'
.venv-mainline/Scripts/python.exe -B realtime_app/run_tool.py body.run_sapiens_cold_smplh --run $runDir --settings realtime_app/configs/cold_smplh_joint300.json --detector sapiens2 --output "$runDir/sapiens_reproduction"
# 同一配置，仅改detector和输出目录，复现PMPose对照。
.venv-mainline/Scripts/python.exe -B realtime_app/run_tool.py body.run_sapiens_cold_smplh --run $runDir --settings realtime_app/configs/cold_smplh_joint300.json --detector pmpose --output "$runDir/pmpose_reproduction"
```

两份命令继续调用同一拟合内核。A30/B30/C300/D1 100/D2 100/D3 0；beta和latent从零，B/C优化同一个共享beta，D阶段冻结身体与体型；每手共享PCA12，腕位置软项1，原生MANO项0.1，身体直接二维项/身体时序项/手二维项为0。未启用表面接触优化，不接回旧脚锚或构造手形。省略 --settings 和 --joint-steps 仍为旧30步配置；--joint-steps 显式覆盖所选配置，覆盖后必须单独记录，不能标作冻结300步复现。

评价已有冻结对照使用：

```powershell
.venv-mainline/Scripts/python.exe -B realtime_app/run_tool.py body.compare_cold_detector_fits --run $runDir --output "$runDir/comparison_reaudit"
```

不指定结果目录时，该入口仍读取 current_mainline.json 所列的两份冻结结果。比较新复现目录时显式指定：

```powershell
.venv-mainline/Scripts/python.exe -B realtime_app/run_tool.py body.compare_cold_detector_fits --run $runDir --sapiens-fit-dir "$runDir/sapiens_reproduction" --pmpose-fit-dir "$runDir/pmpose_reproduction" --output "$runDir/reproduction_comparison"
```

参数指向包含fit/、fit_command.json和body_viewer.html的冷启动运行目录，不能只指向fit/。--run仍提供对应的Sapiens2/PMPose场景；两组身体必须使用这些同源场景输入。评价继续执行原命令差异、阶段一致性与全448帧冷启动检查，不放宽控制。页面链接按实际输出位置生成，记录实际结果目录；应从包含结果和页面的共同父目录启动HTTP服务。跨Windows盘符无法构建同源相对页面链接，应把报告输出选在与输入相同盘符。新运行先核查protocol.json、fit_command.json、fit/fit_summary.json、assessment.json及FAILURE.json（如存在）。不重命名或覆盖冻结结果。

冷启动--settings拒绝重复选项、零联合预算、接触输入和非共享/非原生初始化手姿，防止运行与“无接触冷启动”记录不一致。需要这些旁路设置时使用通用求解器并另立协议。可视化生成是独立输出步骤，页面播放帧率不代表拟合速度。

实现职责：CLI与数据/场景编排在 walker_tools/body；观测转换在 pose_app/body_observations.py；初始化、损失、接触接口、调度和主体分别在 pose_app/smplh_fitting/{initialization,losses,contact,stages,pipeline}.py；人工腕和原生手姿分别在 independent_wrist_reference.py、wilor_mano_prior.py。scene地面分支独立重放，不进入当前身体损失。详见 ../docs/CURRENT_SMPLH_ALGORITHM_20261002.md。

## 主线回归检查

使用已有数据和已安装的CUDA拟合环境，从仓库根目录运行；每次输出选一个不存在的新目录。检查入口与求解入口分开，直接调用现有拟合内核，不改主线预算或默认参数。

```powershell
$runDir = 'research_records/engineering_validation/G20261002_sapiens2_body_mainline448_v2'
$baseline = 'research_records/engineering_validation/G20261003_mainline_regression_v1/baseline'
# 全448帧的短预算回归：五个活动阶段各一步，seed0，与整理前PMPose基准精确比较。
.venv-cuda/Scripts/python.exe -B realtime_app/check_mainline_regression.py short --command "$baseline/fit_args.json" --reference $baseline --output outputs/regression_short_next
# Sapiens2短预算基准由保留的整理前求解器在本阶段计算，来源见manifest，不能称为旧已保存结果。
$baselineSapiens = 'research_records/engineering_validation/G20261003_mainline_regression_v1/baseline_sapiens'
.venv-cuda/Scripts/python.exe -B realtime_app/check_mainline_regression.py short --command "$baselineSapiens/fit_args.json" --reference $baselineSapiens --output outputs/regression_short_sapiens_next
# 当前Sapiens2冻结结果：原始输入、阶段/共享参数、全448帧VPoser与SMPL-H重建。
.venv-cuda/Scripts/python.exe -B realtime_app/check_mainline_regression.py audit --command "$runDir/smplh_cold_start_v2_joint300/fit_command.json" --fit "$runDir/smplh_cold_start_v2_joint300/fit" --observation-reference "$baseline/sapiens_observations.npz" --output outputs/regression_sapiens_next
# PMPose冻结结果使用pmpose_cold_start_v1_joint300和pmpose_observations.npz。
# 仅比较两个已有短预算结果，不拟合、不重建模型。
.venv-cuda/Scripts/python.exe -B realtime_app/check_mainline_regression.py compare --reference $baseline --fit research_records/engineering_validation/G20261003_mainline_regression_v1/short_pmpose/fit --output outputs/regression_compare_next
```

--command支持冻结fit_command.json完整命令列表或fit_args.json选项列表；不支持带SET_BY_COLD_RUNNER占位路径的配置直接求解。--fit/--reference指向直接含result.npz和cold_initialization.npz的目录。--route默认configs/current_mainline.json；audit核查全部冻结标量，short仅将五个活动预算改为1、D3改为0，其余标量仍核查。--chunk-size默认32，必须正数。

short不自动生成或更新golden；--reference必填，缺少其中的observations.npz返回不可用。可用--observation-reference显式指向独立观测基准。audit未给观测基准只能检查当前输入自洽，不表示历史拒绝原因不变。compare不能替代audit。输出report.json、observations.npz；short另有fit/和实际解析参数记录。大数组不打印、不入库。0=所请求检查通过，1=不一致/执行失败，2=资源不可用；资源缺失不会当作通过。

当前本机基准和最终报告位于research_records/engineering_validation/G20261003_mainline_regression_v1；开发期日志和初版报告保留于outputs/mainline_regression_20261003，原模型、数据和冻结结果未覆盖。干净克隆可运行公开单元测试；完整回归需要恢复本机基准、已有数据和授权模型。详见算法说明第18节。通过只证明本次实现保持一致，不证明真实三维精度、触地或接触。


## 全序列受控消融

仅使用已有数据和授权模型。独立运行器读取已保存的冷启动源命令、同一内核及固定预算，输出目录首次必须不存在；已有目录仅在协议完全相同时使用--resume恢复。不要与冻结主线输出混用，算法说明第19节列出完整变量与结论边界。标准冷启动编排拒绝非默认beta策略、先验/身体权重、质量模式和手消融开关；旁路实验通过本入口明确记录。

```powershell
# 新实验；计划固定seed0、448帧及A30/B30/C300/D1 100/D2 100/D3 0。
.venv-cuda/Scripts/python.exe -B realtime_app/run_smplh_ablations.py --output outputs/ablation_next
# 中断恢复；失败证据保留，不改变参数或追加步数。
.venv-cuda/Scripts/python.exe -B realtime_app/run_smplh_ablations.py --output outputs/ablation_next --resume
# 普通评价及全帧参数重建；控制失败或缺失结果返回非零。
.venv-cuda/Scripts/python.exe -B realtime_app/evaluate_smplh_ablations.py --run outputs/ablation_next --replay-parameters
# 本轮既有结果复核：只读拟合，生成审计及汇总，显式引用修复后重试。
$main = 'research_records/engineering_validation/G20261003_smplh_ablation_v1'
$retry = 'research_records/engineering_validation/G20261003_smplh_ablation_retry_v1'
.venv-cuda/Scripts/python.exe -B realtime_app/evaluate_smplh_ablations.py --run $main --retries $retry --replay-parameters
$follow = 'research_records/engineering_validation/G20261003_smplh_ablation_followup_v1'
.venv-cuda/Scripts/python.exe -B realtime_app/evaluate_smplh_ablations.py --run $follow --baseline-run $main --replay-parameters
```

运行追加计划时用--plan realtime_app/configs/smplh_ablation_followup_v1.json指定另一个新目录。输入路径保存在源命令和逐组fit_args/resolved_options；输出含首次FAILURE、执行耗时、阶段检查点、result、指标、逐帧诊断和全帧几何门。读取缓存metrics前仍重新审计输入、掩码、协议、共享参数和冻结状态；--replay-parameters另重建VPoser/PCA/SMPL-H，不能以缓存旧成功跳过检查。缺失本机资源不属于干净克隆自动可运行的公开测试。消融残差、平滑与几何代理不是独立精度，方法级别不因消融改变。


## 检测腕点与已有人工标注比较

直接读取current_mainline中的二维源命令和WiLoR，不运行模型、不拟合人体；保留缺失和几何拒绝。人工标注需manifest逐项指明左右帧、标签及图像源；输出目录必须不存在。

```powershell
.venv-cuda/Scripts/python.exe -B realtime_app/compare_wrist_annotations.py --annotations research_records/annotation_handoffs/H20261001_people1_wrist_joint_10pairs_v5_corrected --output outputs/wrist_manual_next
```

primary WiLoR使用现有PMPose身体腕关联；另报告Sapiens2关联及唯一同侧候选敏感性，不按人工标签选优。二维评价使用原始鱼眼欧氏像素距离，分别报告所有可用点与共同样本。三维仅沿用原人工几何门诊断，未通过的坐标不当真值。源码/完整指标见算法说明第20节及G20261003_wrist_detector_manual_v2。


## 正式场景主线与落地候选（2026-10-04）

`configs/current_mainline.json` 的 `scene_route` 导航到 `configs/stage_ground_profiles.json`。通过 `run_tool.py scene.replay_fitted_body_scene --scene-profile mainline` 调用；必须同时提供同源图像、严格JSONL、标定和地面参考。完整命令、阈值、时序、发布语义及等级见 [阶段与地面实现](../docs/CURRENT_STAGE_GROUND_PIPELINE.md)。落地固定延迟仅为离线候选，尚未接入实时采集入口。其高度闭合不能证明脚锚或接触成立；严格导出掩码和最终发布流修正适用于共用入口。

连续窗口候选：同一场景入口使用 `--scene-profile joint_stage_window`；独立审查使用 `scene.audit_stage_handoff --joint-window`。最终实验为joint_window_gated；速度变化改善但落地闭合和实时耗时未通过，保留离线候选级别。完整残差定义与门见阶段实现文档。

历史360帧场景复现（2026-10-04，360帧PMPose身体候选、strict3031点及baseline场景）：使用历史命令`G20261003_repositioned360_manual_v1/stage_landing_recovery_v1/baseline_profile_review_command.json`，其中已显式声明`--scene-profile baseline`；复现时保持该profile，只把`--output`改到新的不存在目录，绝不覆盖冻结输出。源图像、标定、地面和严格JSONL必须由同次实验提供。

当前配置里的`mainline`只是场景别名，按`current_mainline.json`的`scene_route.default_profile`解析为`supported_stop_fixed_lag`，与历史`baseline`不是同一套参数，因此不能用`mainline`声称复现历史baseline场景。身体范围同样分开：该历史段对应历史360帧PMPose身体；当前`mainline_selection`已切换为448帧可靠性二维回投影身体（见上），历史段落不代表当前身体选择。

身体参数复现使用360帧自己的`pmpose/fit_command.json`；该命令关闭人工腕项。本文前述冷启动回归和消融命令专用于448帧冻结对照，不能自动改用360帧manifest。WiLoR仅提供局部手指旋转先验，不提供米制腕平移；身体共享beta与逐帧姿态求解之后才应用当前场景变换。

实时Stage/地面沿用`run_stereo.py`，选择full SE(3)需要显式`--enable-stage-walker --dynamic-ground-full-se3`，同时提供`--dynamic-ground-reference`和当前标定等既有输入。离线SMPL-H整段拟合没有接入这一实时入口。候选分别显式使用`landing_fixed_lag`或`joint_stage_window`；不能根据页面播放或更平滑就替代主线。严格输入及最终发布流导出修复共享，候选算法等级保持不变。


## 两次采集腕肘诊断

`python -B realtime_app/compare_capture_arm_bias.py --output outputs/arm_review_next`只读已有448/360帧检测、标注对照和冻结身体，并复用旧PMPose有/无腕约束消融。输出目录不能含旧arm_comparison.json。需要本机历史数据；缺失资源报不可用/失败，不运行模型或复用其他采集填补。报告输出arm_comparison.json、arm_records.json和arm_review.html，保留分组、缺失、异常点和三维拒绝状态。统计是人工操作性参考或拟合内部一致性，肘没有独立真值。当前最新主线页面与完整分析位于G20261004_mainline_arm_review_v1，准确构建命令及同源验证见其protocol、viewer_command和viewer_validation。body.html与人体参数bundle由现有visualization.build_grasp_body_canvas_viewer导出，诊断工具不改变身体或场景算法。详细结论见算法说明第21节。
