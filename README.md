# Walker Pose System

## 2026-10-17 路线管理与鞋高基准

身体方法分路线管理，不再固定唯一“主线”。当前高优先级为 `smplh_causal_autograd_contact`：自动微分的因果 SMPL-H q/v/a 跟踪，保留原始双目身体像素、前掌方向和冻结髋观察映射，启用脚部接触；完整448入口为 `.venv-cuda/Scripts/python.exe realtime_app/tools/run_smplh_contact_route.py`。批量重投影和物理跟踪保留各自证据，不在本次展开对照。

裸模型足底参考统一为地面上方25 mm（鞋高）；地面仍Z=0，接触目标25 mm，5 mm容差下界20 mm。路线配置见 `realtime_app/configs/current_mainline.json` 的 `route_management/method_routes`；旧 `mainline_selection` 仅兼容历史批量路线。最新运行状态见 `AI_PROGRESS.md`，详细方法见算法文档5.27。


## 2026-10-13 SMPL-H拟合方法更新

通用拟合与二维实验入口默认使用--fit-policy reliability，修正置信度归约、SO(3)根锚及接触阶段身体观测保持项。已保存360 reference25和448结果仍来自legacy，未重算；冻结profile保留legacy，直接重放历史argv须显式补--fit-policy legacy。公式、输入支持、统计校准及物理证据边界见docs/CURRENT_SMPLH_ALGORITHM_20261002.md第28节，进度见AI_PROGRESS.md；下方历史分数下限0.6不代表新方法。

## 2026-10-12 场景主线更新

用户已将448帧验收的Stage1/2连续过渡逻辑选为场景主线：默认场景入口及`--scene-profile mainline`解析为`supported_stop_fixed_lag`。停止时四脚已落在同一平地是必要采集前提；采用0.5秒固定延迟发布，实时采集入口接入仍待验证。旧baseline显式保留，身体选择不随本次场景提升改变。下方2026-10-05的baseline主线决定为历史记录；当前配置、边界和验证见`AI_PROGRESS.md`与`docs/CURRENT_STAGE_GROUND_PIPELINE.md`。


## 2026-10-05 正式工程主线决定

当前正式工程路线固定为 **PMPose 原始鱼眼观测 → 严格双目几何 → 原 Stage 1/2 识别 → 完整 SE(3) 动态地面基线 → 同源 SMPL-H 冷启动身体及 WiLoR 原生局部手姿**。选择清单位于 `realtime_app/configs/current_mainline.json` 的 `mainline_selection`，场景调用使用 `--scene-profile mainline`，实际解析为 `baseline`。这是一项可复现版本决定，不是物理接触、独立三维精度或整套实时部署认证。

本次场景数据为换机位 360 帧（0..359，原始配对1157..1516）；身体主线改用 `G20261005_reference_weight_fit_v1/runs/reference25/fit/result.npz`。实际权重为 `reference`：回投影平台25px、射线间距平台80mm、置信度参考下限0.6，严重异常使用四次尾衰减。原 `body_quality` 与实际 `body_fit_weights` 分开保存。该结果未启用人工腕软参考，属于正式选择的工程主线，`accepted_for_main_fit=false`仍表示独立物理/接触门未通过。旧360帧quality结果保留为对照；448帧两检测器冻结比较保持原quality参数及等级，配置顶层旧字段仍仅描述448帧。

后续重点：**各关节权重与置信度计算规则**。需要分别研究肘、腕、膝、踝等关节的普通偏差/严重异常界限、可见性与几何敏感性、以及观测和人体先验的平衡；当前共用25px/80mm规则为暂定工程选择。臂均残差75.12→66.38mm、膝踝二阶差分P9573.68→41.92mm/frame²，但左肘及部分腕点变差、固定原质量全身RMS51.10→57.93mm，不能认定所有关节已准确。

`landing_fixed_lag` 与 `joint_stage_window` 保留为显式离线候选：运动指标有部分改善，但落地闭合未通过，联合窗口还未满足耗时预算。只将严格点/掩码导出和最终发布流导出修复纳入共用入口。原基线的硬恢复跳变、缺失测量及静态扶手参考局限继续记录，不用候选或显示平滑掩盖。

详细数据流、参数、命令及停止条件见 `docs/CURRENT_STAGE_GROUND_PIPELINE.md`；身体内核及448帧历史实现见 `docs/CURRENT_SMPLH_ALGORITHM_20261002.md`。后文较早的“当前”判断按其数据集和日期理解；最新选择以本节、`AI_PROGRESS.md` 和选择清单为准。


## 当前路线与入口（2026-10-03）

当前研究推进的是 **448帧离线SMPL-H冷启动身体候选**。Sapiens2为优先身体候选，PMPose为同算法、同预算、同腕参考和WiLoR输入的冻结对照。两套结果仍是工程候选，均未通过完整手部接触门；人工腕参考保持原诊断状态。PMPose实时双目基线继续保留，不能把早期二维基线身份当作当前离线身体选择。普通SMPL、SMPL-X及固定抓握精修是历史/旁路方法，不因整理改级别。

当前路线索引：[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)；详细算法：[docs/CURRENT_SMPLH_ALGORITHM_20261002.md](docs/CURRENT_SMPLH_ALGORITHM_20261002.md)；唯一安装和调用说明：[realtime_app/README.md](realtime_app/README.md#当前身体路线固定配置和评价入口)。本地资料/实验报告中的同名算法说明为备份，旧Word报告为其编写时的历史说明。

| 工作 | 当前入口 | 范围 |
|---|---|---|
| 安装 | realtime_app/requirements-dev.txt；Windows固定环境清单见应用README | 离线拟合和公开测试；不含实时设备/Docker验收 |
| 测试 | realtime_app/run_validation.py | 两个子项目；CI同源；run_tests.py为旧入口 |
| 固定流程拟合 | realtime_app/run_tool.py body.run_sapiens_cold_smplh | 显式指定configs/cold_smplh_joint300.json；不指定仍为旧C30 |
| 自备资源拟合 | realtime_app/run_tool.py body.fit_smplh_wilor_sequence | 通用求解器；参数模板不等于冻结实验配置 |
| 冻结对照评价 | realtime_app/run_tool.py body.compare_cold_detector_fits | 默认冻结结果；可显式指定新冷启动目录，保留失败 |

目录职责不表示验收等级。research_records保存原协议/输出，configs保存可复用参数，pose_app保存算法，walker_tools保存CLI和编排，tools保留兼容入口。历史结果不搬移、不覆盖。下文较早的阶段结论只作背景，当前身份以本节、AI_PROGRESS和实验协议为准。

面向助步器场景的双鱼眼人体姿态估计、双目三维、动态地面和交互可视化研究代码。

## 当前应先阅读的文档

- `AGENTS.md`：持久工作规则和当前冻结约束。
- 版本管理以 `AGENTS.md` 的“Git 版本管理”条款为准：每项已验证的修复、迭代或更新均须独立提交本地 Git，且不得混入已有改动。
- `AI_PROGRESS.md`：最新实验状态、停止条件和主线决策。
- `VISUALIZATION_PIPELINE.md`：人体、地面、相机、助步器可视化的唯一规范。
- `docs/CURRENT_STAGE_GROUND_PIPELINE.md`：Stage 1/2 与动态地面主线。
- `docs/PROJECT_STATUS.md`：当前状态索引，不再复制历史实验细节。
- `CLAUDE.md`：文档权威顺序、项目概览和协作规则。

如果这些文档与旧实验报告或旧日志冲突，以 `AGENTS.md`、`AI_PROGRESS.md` 和当前实验目录协议为准。旧实验记录保留用于追溯，不代表当前规则。

完整测试必须使用应用README中的requirements-dev安装流程；旧基础依赖仅覆盖实时基础模块，不能用于完整测试。

## 系统主链

```text
双鱼眼采集
→ 2D 姿态
→ 左右人物关联
→ 原始鱼眼三角化
→ 下肢 T1–T5
→ Stage 1/2 与地面变换
→ 运动学和接触候选
→ 三维可视化
```

当前系统分为三层：

- `realtime_app/`：单路姿态、双目采集/回放、鱼眼三角化、动态地面和下游输出。
- `sequence_pipeline/`：在独立 Docker 环境中顺序运行多个二维姿态模型并保存统一实验输出。
- `research_records/`：保存正式实验、工程验证、筛选结果、失败证据和实验注册表。

公开查看器位于 `docs/walker-viewer/`，不需要 Python 后端；它只读取公开运动数据，不反向修改算法结果。

## 当前基线与结论边界

- 相机物理身份由 PnP `instance_id` 注册表解析，不能长期硬编码 Windows OpenCV 数字索引；校准角色为 `cam0=LEFT`、`cam1=RIGHT`。
- 模型输入可以按当前协议转正，但二维点进入几何前必须回到原始鱼眼像素；双目计算使用正式鱼眼标定。
- 早期严格双目/下肢工程二维基线为 PMPose；当前离线身体候选与对照身份见本文件开头。Sapiens2 的旧操作性参考角色不构成独立真值。
- 严格三角化使用固定质量门、`max_matches=1`，并保留逐帧、逐点失败原因；无门限或时间补全结果只能作诊断/显示。
- DA3 只用于逐帧图像空间的 ROI/遮挡诊断，不能作为米制深度、标定、人物关联、关键点筛选或三角化依据。
- Stage 1/2、动态地面、助步器结构和接触输出目前都是工程候选；没有外部轨迹、阶段、对象身份或物理接触真值时，不得声称真实精度、触地、支撑、步态或生产通过。
- SMPL/SMPL-X 结果必须按实验阶段门解释。平滑、重投影、有效点数、模型网格或接触代理都不能替代独立三维真值。

### 早期普通SMPL路线与手部旁路

早期普通 SMPL 路线及手部旁路如下；当前离线身体采用 **SMPL-H（6890 个顶点）**，详见开头导航：

- `realtime_app/pose_app/smpl_coco_observation.py` 使用官方 SMPL 网格和固定的 `17 x 6890` `J_regressor_coco.npy`，将 COCO-17 二维观测映射到 SMPL 顶点。
- `third_party/WiLoR/` 和 `third_party/InterWild/` 仅作为可选的人手估计旁路。它们使用 MANO 手部模型（778 个顶点），不能替换 SMPL，也不能把 MANO 参数直接拼接到 SMPL body pose 或 beta。
- 手部结果只有在完成手腕/前臂关联、原始鱼眼像素回映、左右视角质量门和坐标标定后，才允许成为三角化或后续优化的输入；在此之前统一标记为 `model_local_unaccepted`。

本地部署状态（不代表已完成正式实验）：

| 后端 | 代码 | 权重/检测器 | MANO 文件 | 当前状态 |
|---|---|---|---|---|
| WiLoR | `third_party/WiLoR` | `pretrained_models/wilor_final.ckpt`、`detector.pt` | `mano_data/models/MANO_LEFT.pkl`、`MANO_RIGHT.pkl`、`SMPLH_male.pkl` | 共用 CUDA 环境预检通过；拟合使用双 MANO PCA |
| InterWild | `third_party/InterWild` | `demo/snapshot_6.pth` | `common/utils/human_model_files/mano/MANO_RIGHT.pkl`、`MANO_LEFT.pkl` | 权重完整；Windows 环境还缺 `pytorch3d` |

两者的输入都是经过审计的 RGB 手部/人体裁剪图，不应直接把正式实验原始帧静默送入官方 demo。输出均包含手部二维/三维或网格与 MANO 参数，但模型局部三维不能直接当作项目米制坐标。项目侧统一协议由 `realtime_app/pose_app/hand_pose_provider.py` 提供：它只负责资产预检、命令构造和结果字段约定，不在当前阶段自动改变 SMPL 主链。

运行不加载大模型的资产预检：

```powershell
cd .\realtime_app
..\.venv-cuda\Scripts\python.exe tools\hand_model_preflight.py all
```

当前共用环境为仓库根目录 `.venv-cuda`，使用 CUDA 版 PyTorch 和 RTX 5070 Ti。它已经服务 SMPL 主线和 WiLoR；InterWild 权重已就位，但官方代码还依赖 `pytorch3d`，该包在当前 Windows/Python 3.12 环境没有可直接安装的 wheel，因此不能把 InterWild 标记为可运行。官方 WiLoR 依赖中的 `xtcocotools` 在 Windows/Python 3.12 下构建失败，但 WiLoR demo 和本项目预检不导入它，因此暂不安装，失败证据保留在环境安装日志中。

可复现依赖清单见 `realtime_app/requirements-cuda-shared.txt`；CUDA PyTorch 必须先从 `cu128` 索引安装。

环境接入分三层：

1. `smpl_coco_observation.load_smpl_male(..., device="cuda")` 将官方 6890 顶点 SMPL 放到 GPU；`regress_coco17_torch` 把回归矩阵放到顶点所在设备，避免 CPU/GPU 往返。
2. WiLoR 推理使用有序的 `left_ccw90` / `right_cw90` 图像目录；进入模型前分别逆旋转回原始鱼眼方向，官方 `project_full_img` 的二维输出再按原始图像宽高投影。`make_wilor_observation` 要求 `Nx2` 点在 `raw_fisheye` 像素坐标内，不做裁剪、不做静默旋转回映；越界直接拒绝。全序列审计脚本见 `realtime_app/tools/run_wilor_sequence.py`。
3. `hand_pose_provider.py` 只构造官方 demo 命令、检查权重和依赖，并把三维结果标为 `model_local_unaccepted`；它不替换 SMPL，也不绕过原始鱼眼坐标和双目质量门。
4. 正式接入前必须用审计裁剪图完成单图推理，保存原始输出、失败原因、输入裁剪变换和回映后的二维坐标；通过后才能设计手腕/前臂关联适配器。

CUDA 接入的 CPU 对照检查已经完成：同一随机 SMPL 参数下，CPU/CUDA 顶点最大绝对差约 `3.6e-7 m`，COCO-17 回归最大绝对差约 `2.4e-7 m`。这只证明数值前向一致性，不代表姿态精度提升或正式实验结果。

## 快速开始

### 实时程序

```powershell
cd .\realtime_app
pip install -r .\requirements.txt
python -B .\run_validation.py
python .\check_environment.py --model all
```

单路 2D：

```powershell
python .\run.py --model yolo26x_pose --camera 0
```

双目回放或实时运行必须使用物理注册表和真实标定文件：

```powershell
python .\run_stereo.py `
  --model yolo26x_pose `
  --camera-registry .\camera_registry.json `
  --calibration .\calibration\results\stereo_fisheye.json
```

正式双摄采集前应先阅读 `realtime_app/README.md` 和 `realtime_app/docs/STEREO_TRIANGULATION.md`，再运行 `tools/camera_probe.py` 和采集预检。

### 多模型图片序列

```powershell
cd .\sequence_pipeline
python .\run_sequence.py list
python .\run_sequence.py check --models all --no-docker
```

该程序只负责“图片序列 → 多模型二维输出”，不负责跨帧跟踪、左右同步、三角化、三维关键点或步态分析。

## 配置和数据边界

第三方源码、权重、原始采集、视频、JSONL、设备绑定和大型实验资产不随公开仓库提交。配置应从示例文件复制，并在本机填写路径和物理相机身份。

每个正式实验应至少保留：

- `EXPERIMENT.md`
- `VALIDATION_PROTOCOL.md`（如适用）
- `run_metadata.json`
- `command.txt`
- 机器可读结果和失败原因

实验可追溯性使用输入路径、数据范围、代码文件、参数、输出路径和结论边界，不写 Git 哈希。


## 离线拟合复现

已有双目二维观测与 WiLoR 记录的 SMPL-H 主线，安装与运行步骤见 [realtime_app/README.md](realtime_app/README.md#离线主线的独立安装与验证)。依赖版本、参数模板、资源预检查和公开测试入口均已纳入版本管理；模型和采集数据需按说明另行提供。
