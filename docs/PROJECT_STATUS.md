# Project status

## 2026-10-17 独立路线管理

当前高优先级：`smplh_causal_autograd_contact`，自动微分＋前掌方向＋接触，完整448入口 `realtime_app/tools/run_smplh_contact_route.py`。身体方法不再以唯一“主线”决定研究资格；批量重投影、因果跟踪、物理方法分别保存结果和结论。裸模型足底鞋高参考25 mm，真实地面Z=0；配置见 `realtime_app/configs/current_mainline.json/route_management,method_routes`，运行结果见根 `AI_PROGRESS.md`。


## 2026-10-14 Flow采样去除内部重复同步

公开输入校验保留，内部已知合法的Euler时间一次构造，避免每步GPU标量等待；八步F0标量读取17→1，局部采样中位14.77→12.83ms，约减少13.1%。整链交错对照尚不支持稳定提速：几何149.81→146.24ms，F0为40.15→41.69ms；详细不利结果与波动也保留。网络/完整几何/输入绑定不变，相关值/梯度及拒绝检查通过。实现、测量及边界见MODEL_ARCHITECTURE第33节、MODEL_RESULTS第48节和AI_PROGRESS；iMF训练目标仍未接入。

## 2026-10-14 Flow相关新算法适配

六篇少步生成/人体恢复加速的来源与取舍见[LITERATURE_ANALYSIS](../research_records/research/G20261007_flow_mesh_baseline_v1/LITERATURE_ANALYSIS.md)的2026-10-14专题；拟议接口与停止条件见MODEL_ARCHITECTURE第32节，阅读边界见MODEL_RESULTS第47节。优先研究无辅助头iMF训练目标，当前不接入硬冻结、token合并或删几何；本轮仅文献与设计，模型未训练、无新测速。进度以AI_PROGRESS本日对应条目为准。

## 2026-10-14 场景重建：独立视觉世界轨迹路线

已完成面向双鱼眼、相机附着助步器及12GB显存的官方方法核对。建议新增独立visual_rig_pose：ORB-SLAM3 fisheye stereo作基线、cuVSLAM作速度候选、Pi3X小窗口作学习候选；统一相机/助步器/SMPL-H世界变换，将Stage1/2降为可关闭软先验。人体和助步器须同时排除出静态背景；同源视觉/Stage证据不能重复计权，检测或网络分数不自动成为概率/逆协方差。

候选性能范围、RayMap3R/MASt3R-SLAM/BLASt3R取舍、鱼眼/同步/安装适配、置信度逻辑和固定448身体的三路比较计划见[现有场景说明新增节](CURRENT_STAGE_GROUND_PIPELINE.md#2026-10-14-独立视觉世界位姿路线调研与设计决定)。本轮只有调研与设计记录，未安装或运行新模型，未修改主线/已有场景结果；尚不能宣称摆脱Stage依赖或实现世界精度提升。完整进度见AI_PROGRESS。

## 2026-10-13 Flow几何分支GPU提速

隔离next模型保留169维中心差分、全部339解码状态、原鱼眼投影/precision/失败门和K32/Euler8。几何计算只解码COCO回归器全部105个非零支持顶点，最终人体仍6890顶点；全扰动合批及批量候选回传减少重复调用和同步，没有裁几何方向或新增网络。相同GPU执行轮的完整G0/J/G1+A6+网格中位337.149→102.249ms（约3.30倍），J258.654→28.298ms；30次选择ID均一致、无fallback。数值/梯度/坏扰动及资产缓存生命周期的必要检查通过，浮点差异保留说明。

当前仍未训练，不支持完整系统30fps或精度提升；正式SMPL-H/场景选择不改。实现、公式和限制见现有MODEL_ARCHITECTURE第30节，GPU同条件对照与原始计时见MODEL_RESULTS第45节及flow_next_gpu_geometry_runtime_20261013.json，进度见AI_PROGRESS。

## 2026-10-13 当前SMPL-H本机GPU速度

RTX5070Ti Laptop实测当前448帧/560步reliability拟合：一次完整solver为42.0031s，离线吞吐10.6659帧/s；C身体联合优化25.6870s是主要耗时。已有参数单帧解码经10次预热/50次采样，wall中位14.3135ms、P95 22.9080ms；批448解码29.7663ms，再含CPU回传36.6252ms。完整拟合与参数解码须分开理解，均未包含二维检测/场景/渲染；完整计时不含解释器/模块导入和CUDA上下文首次启动。计时记录G20261013_smplh_runtime448_v1/runtime.json，详细范围见算法第32节；未改变模型或主线选择。

## 2026-10-13 Flow借鉴SMPL-H问题处理的定向修正

隔离next模型已分离H2原始二维支持与strict几何门，避免双目拒绝连带删除有效像素；A1及关闭几何的A5删除无用投影/解码依赖；缺失点在非线性投影前隔离，修复NaN/Inf梯度污染。未增加网络或参数，原完整候选和失败语义保留。相关23项检查及六个Python的Ruff通过，没有真实序列重跑、训练或性能/精度声明；旧H2回放数值按旧掩码解释。正式场景/身体选择不改。

共享整片体型、置信度经验公式、固定位置归约、共享手姿与接触没有机械移植。具体理由、接口变化及Word时效见[MODEL_ARCHITECTURE第27节](../research_records/research/G20261007_flow_mesh_baseline_v1/MODEL_ARCHITECTURE.md#27-对照smpl-h主线的定向借鉴2026-10-13)，反例与必要检查见[MODEL_RESULTS第43节](../research_records/research/G20261007_flow_mesh_baseline_v1/MODEL_RESULTS.md#43-2026-10-13对照smpl-h主线的定向迁移)及AI_PROGRESS。

## 2026-10-13 当前reliability新448帧人体页面

完整新冷启动拟合及交互页面位于`research_records/engineering_validation/G20261013_reliability448_view_v1`，入口`view/body.html`，浏览器轻量入口`view/body_browser.html`。沿用最近448二维/踝倍率2/native共享手配置，显式新reliability目标；560步完成，当前strict/Stage/地面重放，无显示平滑。19项来源/坐标及浏览器控件检查通过，13项导出策略/帧映射测试通过。实施记录见算法第31节与AI_PROGRESS；它是当前方法的工程输出，正式360主线和历史448选择不自动提升。

## 2026-10-13 SMPL-H完整实现说明入口

[算法第30节](CURRENT_SMPLH_ALGORITHM_20261002.md#implementation-guide)按实际源码串起输入与坐标、参数形状、置信度/损失、初始化、分阶段更新、接触、计算精简、默认值、输出和失败处理。明确通用reliability实现与已保存legacy网格的区别，并说明手时序归约、steps预算和诊断字段的实际限制。本轮仅补记录，未运行新拟合或改变主线选择；第27—29节仍保存修改原因及已有检查历史。

## 2026-10-13 SMPL-H置信度与目标函数修正

整体复查进一步修正共享手初值（选完整高分假设避免轴角分支相消）、初始化加权门与单模板前向、手关联腕分数范围、逐关节/双髋时间支持及接触内部固定归约。受影响组件/MANO共63项通过；细节见算法第29节，仍无真实拟合结果或精度提升声明。

通用SMPL-H入口及二维实验入口默认采用fit-policy=reliability：身体分数零下限、双目3D取较低视图分数并单独作几何衰减；2D/3D/MANO使用固定原始观测位置数归约，保留低分/缺失导致的证据减弱；踝倍率是任务优先级。手二维缺失box score置零并保留原因。根锚改SO(3) chordal并按初始化肩髋可靠程度衰减；接触D2/D3释放身体时保留所选身体观测、已启用腕项与身体时间项。新策略拒绝同源2D/3D重复作为两组独立数据项。

当前保存的360 reference25、448二维踝倍率结果仍是旧方法，未重算；冻结profile及历史argv读取入口固定legacy，直接重放旧参数须补--fit-policy legacy。场景supported_stop_fixed_lag与Flow研究模块不随此修正改变。共享局部手姿不证明固定抓握，COCO脚踝不证明足底接触，输出明确记录证据缺口；统计概率/方差校准也未完成，不能将新权重称为逆协方差或精度提升。

方法、公式、入口、影响详见[算法第28节](CURRENT_SMPLH_ALGORITHM_20261002.md#28-置信度加权旋转锚与接触保持项的实际修正2026-10-13)。首轮142项相关测试通过；根锚/历史argv补充后受影响33项通过（含2新增）。未执行整片拟合或结果比较。AI_PROGRESS及current_mainline.json同步“源码已修正、所选结果仍为legacy”的区别。

## 2026-10-13 当前主线与Flow研究架构

正式场景入口按2026-10-12决定使用`supported_stop_fixed_lag`；身体仍为360帧PMPose/reference25 SMPL-H及WiLoR。下方较早的baseline说明保留历史范围；选择以`current_mainline.json`及`AI_PROGRESS.md`为准。本次没有将学习模型接入正式主线。

隔离的`learning_flow_mesh_next`已完成固定信息条件一次编码、推理/研究诊断解耦，以及共享状态校验、共享损失归约和评价分组门修复。参数量不变，尚未训练；双生成器、完整手姿/体型和序列共享shape仍无独立效果结论。架构取舍及接口见[MODEL_ARCHITECTURE第25节](../research_records/research/G20261007_flow_mesh_baseline_v1/MODEL_ARCHITECTURE.md#25-2026-10-13架构收敛固定条件一次编码推理与诊断解耦)，具体改动、必要验证与失败见[MODEL_RESULTS第42节](../research_records/research/G20261007_flow_mesh_baseline_v1/MODEL_RESULTS.md#42-2026-10-13架构收敛与纠错)。后续以匹配数据和四法实证决定是否保留G0/J/G1复杂度，不继续堆模型或检查框架。

## 2026-10-13 SMPL-H基线方法结构维护

当前身体选择仍为360帧PMPose/reference25；近期448帧二维拟合/踝倍率2是独立保留的展示结果。场景mainline自2026-10-12解析为supported_stop_fixed_lag，旧360帧baseline场景未重算。下方较早日期的主线段落保留历史范围，不能覆盖本节、AI_PROGRESS与current_mainline.json的当前选择。

共用SMPL-H内核完成按有效目标裁剪计算：只有局部手参数更新且没有有效手二维/接触项时，D阶段直接优化MANO局部旋转/PCA；完整网格仅用于原采样时刻诊断和最终输出。身体/手投影与骨长/接触分支按实际激活执行，常量权重张量移出循环；共享手姿先算一套旋转再与所有原帧/视角目标比较。原A/B/C/D预算、学习率及Adam阶段重置保留，未新增网络、损失或CLI路线。

方法公式、共享手姿/权重/根锚/接触分支的局限、实现及检查见[现有算法说明第27节](CURRENT_SMPLH_ALGORITHM_20261002.md#27-smpl-h基线的方法结构分析与精简2026-10-13)。PCA行范数仅归一化基底，不解释为概率白化。用户要求只优化方法，最终阶段仅单元与静态检查；没有新的完整拟合/精度/性能结论，不替换已存身体、场景或页面结果。Flow Matching由另一聊天处理。

## 2026-10-05 正式工程主线决定

当前正式工程路线固定为 **PMPose 原始鱼眼观测 → 严格双目几何 → 原 Stage 1/2 识别 → 完整 SE(3) 动态地面基线 → 同源 SMPL-H 冷启动身体及 WiLoR 原生局部手姿**。选择清单位于 `realtime_app/configs/current_mainline.json` 的 `mainline_selection`，场景调用使用 `--scene-profile mainline`，实际解析为 `baseline`。这是一项可复现版本决定，不是物理接触、独立三维精度或整套实时部署认证。

本次场景数据为换机位 360 帧（0..359，原始配对1157..1516）；身体主线改用 `G20261005_reference_weight_fit_v1/runs/reference25/fit/result.npz`。实际权重为 `reference`：回投影平台25px、射线间距平台80mm、置信度参考下限0.6，严重异常使用四次尾衰减。原 `body_quality` 与实际 `body_fit_weights` 分开保存。该结果未启用人工腕软参考，属于正式选择的工程主线，`accepted_for_main_fit=false`仍表示独立物理/接触门未通过。旧360帧quality结果保留为对照；448帧两检测器冻结比较保持原quality参数及等级，配置顶层旧字段仍仅描述448帧。

后续重点：**各关节权重与置信度计算规则**。需要分别研究肘、腕、膝、踝等关节的普通偏差/严重异常界限、可见性与几何敏感性、以及观测和人体先验的平衡；当前共用25px/80mm规则为暂定工程选择。臂均残差75.12→66.38mm、膝踝二阶差分P9573.68→41.92mm/frame²，但左肘及部分腕点变差、固定原质量全身RMS51.10→57.93mm，不能认定所有关节已准确。

`landing_fixed_lag` 与 `joint_stage_window` 保留为显式离线候选：运动指标有部分改善，但落地闭合未通过，联合窗口还未满足耗时预算。只将严格点/掩码导出和最终发布流导出修复纳入共用入口。原基线的硬恢复跳变、缺失测量及静态扶手参考局限继续记录，不用候选或显示平滑掩盖。

详细数据流、参数、命令及停止条件见 `docs/CURRENT_STAGE_GROUND_PIPELINE.md`；身体内核及448帧历史实现见 `docs/CURRENT_SMPLH_ALGORITHM_20261002.md`。后文较早的“当前”判断按其数据集和日期理解；最新选择以本节、`AI_PROGRESS.md` 和选择清单为准。


## 当前身体路线与源码职责（2026-10-03）

本节是当前导航，不重新认证历史结果。可执行路线索引为 ../realtime_app/configs/current_mainline.json；算法说明为 CURRENT_SMPLH_ALGORITHM_20261002.md。

| 路线 | 当前角色 | 结论边界 |
|---|---|---|
| Sapiens2 → SMPL-H冷启动，C300 | 当前优先身体候选 | 工程拟合；accepted_for_main_fit=False |
| PMPose → 同一SMPL-H冷启动，C300 | 冻结匹配对照 | 相同求解和预算；不是独立真值 |
| Sapiens2冷启动，C30 | 先行结果/兼容默认 | 不能冒充C300结果 |
| 实时PMPose严格双目、T1–T5 | 保留的几何/运行基线 | 与身体候选身份分开；质量门不变 |
| 固定抓握、corrected、约束感知更新 | 既有实验候选和历史对照 | 不自动接入当前冷启动 |
| 自由关节完整精修v3 | 已撤回推荐的失败证据 | 不重新推荐或清除失败 |
| 5mm腕软界 | 取舍候选 | 不是当前默认惩罚尺度 |
| 其他SMPL/SMPL-X、骨架及时序拟合 | 历史或本地候选 | 不按目录名升级/降级 |

输出根目录为 research_records/engineering_validation/G20261002_sapiens2_body_mainline448_v2：优先候选 smplh_cold_start_v2_joint300，对照 pmpose_cold_start_v1_joint300，场景分别 sapiens2/scene 与 pmpose/scene。两套完整448帧、原拒绝信息和手部失败均保留。没有启用接触拟合；当前地面仅用于场景、后处理和脚高度诊断，不进入身体优化损失。

### 代码边界

- run.py/run_stereo.py：采集/回放运行入口，使用pose_app；sequence_pipeline只负责二维模型顺序运行。
- run_tool.py与walker_tools/catalog.py：已有186个分类工具的查询和分发；只按职责分类，不选择最优方法。
- walker_tools/body/run_sapiens_cold_smplh.py：固定数据集编排、冷启动审计、几何检查及查看器构建；不是通用实时求解器。
- walker_tools/body/fit_smplh_wilor_sequence.py：公共参数解析；pose_app/smplh_fitting/pipeline.py组装输入、模型、目标和输出。
- smplh_fitting/initialization.py、losses.py、contact.py、stages.py：刚性初值、各损失、可选表面目标接口、阶段冻结策略。contact模块存在不代表本次接触项已启用。
- body_observations.py、fisheye_camera.py、smpl_coco_observation.py：拟合观测、鱼眼投影、网格COCO回归；拟合软候选不替代triangulation.py正式严格门。
- independent_wrist_reference.py、wilor_mano_prior.py：人工腕软参考与原生手旋转；WiLoR局部姿态不提供人体米制平移或beta。
- walker_tools/scene与visualization：同源地面重放和显示，不能反向改变身体结果。tools/*.py是旧路径兼容壳，应改分类后的真实实现。

### 结构整理规则与待处理文件

详细清单见 code_inventory.json：对本轮开始时待处理文件记录状态、归属和可定位的词法引用。词法命中不证明实际运行依赖；没有命中也不足以证明可以删除。

未提交的因果骨架、时序prefit、SMPL-X R3/资源门、手脚支路工具及对应测试保留为本地候选。当前冷启动不依赖这些文件；不批量纳入公开测试，也不因本次审查删除。既有标定/地面资产和本地路径保留。未提交的查看器须另审公开数据来源，不等同于拟合器。既有渲染器、实验注册表和报告修改仍待各自任务核验，不随结构整理提交。

已有config.example及四组YOLO阈值配置、requirements-camera-identity的删除状态保留。新camera_config.example.json沿用已发布实时模板的内容与同一应用根路径语义，文档/配置测试使用这个明确入口；不恢复或擅自提交旧文件删除。run_tests.py既有未提交改写保留，当前公开验证统一用run_validation.py。旧CHANGELOG等文件的删除也不在本次代为提交。

当前C300/兼容C30配置已入源码，通用smplh_fit.example.json仍是自备资源示例；它没有本实验人工腕，不能称为冻结复现。详细安装、命令、输入资源和评价工具范围只在 ../realtime_app/README.md 维护。

Last reviewed: 2026-08-30

## What is verified

- The realtime application accepts camera, video, side-by-side video, and recorded stereo-capture replay inputs.
- Two fisheye cameras have a ChArUco stereo calibration at 1920 × 1080. The calibrated coordinate system is `left_camera` and the length unit is mm.
- Live capture records host-side `VideoCapture.read()` return timestamps and produces an explicit one-to-one `stereo_pairs.csv`; formal replay restores that pairing rather than matching equal AVI frame numbers.
- The runtime supports camera-identity resolution, rotation normalization, person association, fisheye triangulation, positive-depth checks, and reprojection-error rejection.
- The former 45-test check is historical. Current public verification uses `python -B realtime_app/run_validation.py`; CI runs the same entry on Python 3.11 and 3.12.
- A targeted visual review cleared all 12 Sapiens2 target-selection ambiguity flags in the fixed 120-image single-view set. Two left-camera images have missing lower-body references and are explicitly excluded.
- The original Sapiens2-308 outputs contain ankle, big-toe, small-toe, and heel landmarks. These have been restored to raw fisheye pixels for all 120 images; 118 images have all eight foot points above the current 0.25 engineering threshold.
- A matched-crop control shows that upright Sapiens2-308 input retains the same 118/120 complete-foot coverage when the input person box contains the feet. The current failure is the upright YOLO box coverage, not a demonstrated pose-head orientation failure.

## Formal replay baseline

The authoritative experiment registry is `research_records/registry/experiment_registry.csv`; raw assets and rendered outputs are deliberately local-only.

| Experiment | Input | Change from B0 | Result |
| --- | --- | --- | --- |
| E20260827-B0 | R20260826-01 approach, 389 true CSV pairs | Baseline | 15.3368 valid 3D points/pair; 3.216 FPS |
| E20260827-L1 | Same 389 pairs | Strict local virtual perspective | 15.3933 valid 3D points/pair; 0.920 FPS; not a useful trade-off |
| E20260827-U1 | Same 389 pairs | Model-input fisheye undistortion | 9.1003 valid 3D points/pair; negative result |

The timestamp deltas recorded for B0 are host read-return deltas, not camera exposure synchronisation errors.

## Evidence limits

- A higher number of valid 3D points is coverage, not proof of greater 3D accuracy.
- No rigorous external 3D ground-truth evaluation has been completed.
- M0 is a five-model engineering screen with consistent 60-pair inputs; it is not a real-world 3D accuracy comparison.
- COCO-17 has ankle points but no toe or heel points. Sapiens2-308 now supplies candidate toe and heel pseudo-labels, but they have not been validated against independent manual labels and do not yet constitute complete foot reconstruction or validated gait events.
- Local virtual perspective and global model-input undistortion should not be retuned without a new pre-registered ablation and an explicit B0 comparison.
- Sapiens2 confidence is an internal model score, not a calibrated probability or pixel-error bound. Its heel landmark is anatomical and must not be treated as the ground-contact point without a separate definition and validation.

## Historical next step (2026-08-30; superseded by current navigation)

Build and evaluate a foot-inclusive upright person-box rule using only current-frame information. It must be compared with the A9 matched-crop control before Sapiens2-308 is used for routine foot pseudo-label export. Then freeze the foot-landmark definitions and manually audit a small stratified set of big-toe, small-toe, and heel points before resuming stereo triangulation or gait-event calculations.

## Repository boundaries

The public repository contains source code, test code, portable configuration templates, compact experiment registry metadata, and small qualitative examples. It deliberately excludes model weights, third-party source trees, raw captures, videos, JSONL outputs, and local device bindings. See [`third_party/README.md`](../third_party/README.md) and [`realtime_app/docs/CAMERA_REGISTRY.md`](../realtime_app/docs/CAMERA_REGISTRY.md).

评价入口已支持--sapiens-fit-dir和--pmpose-fit-dir，不指定时保持原冻结目录；新目录必须与--run的场景同源，原控制检查继续保留。冷启动配置拒绝与记录冲突的接触输入、重复参数、零联合预算及非共享/非原生初始化手姿。


## 2026-10-14 Flow信息编码后续提速

当前next将信息矩阵的旋转/平移块统一批处理；几何、完整候选、PSD和发布网格不变。GPU默认编码中位3.21→0.25ms，同次顺序组完整路径106.50→100.55ms；P95未改善，不升级为稳定实时。39项相关逻辑/梯度测试及Ruff通过，当前仍未训练。最新实现/测量边界见MODEL_ARCHITECTURE第31节、MODEL_RESULTS第46节和AI_PROGRESS；第30/45节保留前轮结果，正式SMPL-H主线不受影响。
