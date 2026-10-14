# 当前两阶段与动态地面主线

## 2026-10-13 身体拟合方法更新的边界

完整的身体求解实现见[SMPL-H算法第30节](CURRENT_SMPLH_ALGORITHM_20261002.md#implementation-guide)，包括原始鱼眼与cam0米制、身体软权候选与严格场景掩码的区别，以及可选接触如何消费同源地面变换。该说明不改变Stage/动态地面选择，不把当前reliability代码误登记为已经重算的主线网格。

新身体求解默认fit-policy=reliability，修正置信度归约、旋转锚与接触保持项，详见CURRENT_SMPLH_ALGORITHM_20261002.md第28节。已保存360 reference25身体及其场景页面未重算；本页旧复现argv需补--fit-policy legacy。场景主线仍为supported_stop_fixed_lag，本次身体方法修改不改变其停止支撑假设、标定或阶段门。新策略没有建立固定抓握或足底接触真值。

## 2026-10-12 连续过渡逻辑进入工程主线

按用户明确选择，场景默认入口和`--scene-profile mainline`现在解析为`supported_stop_fixed_lag`；旧算法通过`--scene-profile baseline`显式保留。当前唯一变量是Stage/地面路由及导出兼容性，原Stage动作标签、严格脚点/旋转门、身体拟合选择和历史结果保留。下方2026-10-05及此前的baseline主线说明是历史决定。

主线使用用户采集前提：助步器每次停止时四脚已落在同一平地，不会悬停。稳定背景停止独立触发支撑恢复，不等待人体运动；0.5秒固定延迟内只修正未发布帧的高度/倾角，保留XY/yaw，原失败/原位姿与修正原因继续记录。超过150mm/8°修正门或长缺测仍拒绝，不能把该先验恢复称独立触地测量。

依据`G20261012_supported_stop448_v1/view_v4`已完成448验证：11个Stage1边界高度步长全零，助步器节点最大帧位移96.65→49.45mm，最大XY步长略增且保留。此次49相关测试通过，实际448重放命令经mainline解析与已验收显式profile全部参数一致，导出器已解除baseline硬绑定并按解析后的profile显示协议。记录在该实验`mainline_promotion.json`；不重复模型/场景运行。实时`run_stereo.py`仍是旧因果路径，0.5秒缓冲尚未接入硬件发布；本次不宣称实时部署完成。

身体360 reference25选择保持；其旧baseline页面/scene_reference仍为历史同源对照，不能当新算法的360验证结果。新场景导出采用主线新profile；其他采集须满足相同停止支撑协议，否则显式baseline。


## 2026-10-05 正式工程主线决定

当前正式工程路线固定为 **PMPose 原始鱼眼观测 → 严格双目几何 → 原 Stage 1/2 识别 → 完整 SE(3) 动态地面基线 → 同源 SMPL-H 冷启动身体及 WiLoR 原生局部手姿**。选择清单位于 `realtime_app/configs/current_mainline.json` 的 `mainline_selection`，场景调用使用 `--scene-profile mainline`，实际解析为 `baseline`。这是一项可复现版本决定，不是物理接触、独立三维精度或整套实时部署认证。

本次场景数据为换机位 360 帧（0..359，原始配对1157..1516）；身体主线改用 `G20261005_reference_weight_fit_v1/runs/reference25/fit/result.npz`。实际权重为 `reference`：回投影平台25px、射线间距平台80mm、置信度参考下限0.6，严重异常使用四次尾衰减。原 `body_quality` 与实际 `body_fit_weights` 分开保存。该结果未启用人工腕软参考，属于正式选择的工程主线，`accepted_for_main_fit=false`仍表示独立物理/接触门未通过。旧360帧quality结果保留为对照；448帧两检测器冻结比较保持原quality参数及等级，配置顶层旧字段仍仅描述448帧。

后续重点：**各关节权重与置信度计算规则**。需要分别研究肘、腕、膝、踝等关节的普通偏差/严重异常界限、可见性与几何敏感性、以及观测和人体先验的平衡；当前共用25px/80mm规则为暂定工程选择。臂均残差75.12→66.38mm、膝踝二阶差分P9573.68→41.92mm/frame²，但左肘及部分腕点变差、固定原质量全身RMS51.10→57.93mm，不能认定所有关节已准确。

`landing_fixed_lag` 与 `joint_stage_window` 保留为显式离线候选：运动指标有部分改善，但落地闭合未通过，联合窗口还未满足耗时预算。只将严格点/掩码导出和最终发布流导出修复纳入共用入口。原基线的硬恢复跳变、缺失测量及静态扶手参考局限继续记录，不用候选或显示平滑掩盖。

详细数据流、参数、命令及停止条件见 `docs/CURRENT_STAGE_GROUND_PIPELINE.md`；身体内核及448帧历史实现见 `docs/CURRENT_SMPLH_ALGORITHM_20261002.md`。后文较早的“当前”判断按其数据集和日期理解；最新选择以本节、`AI_PROGRESS.md` 和选择清单为准。


当前入口、等级和复现参数以本节及 `realtime_app/configs/current_mainline.json` 的 `scene_route` 为准。`scene_rotation_fix_v1` 是冻结工程基线；浏览器中的 `stage_landing_recovery_v1/view_final` 是落地恢复候选的历史展示，不能因展示更新自动升级为主线。下面较早的采集数量保留为历史证据。



最新主线交互页面：`research_records/engineering_validation/G20261005_reference25_mainline_view_v2/body.html`。本轮重新计算strict几何并重放baseline，页面索引与source/browser_validation见current_mainline.json/mainline_selection.visualization。身体参数不变；旧动态场景仅用于事后等值审计。

### 正式版本的处理顺序与接口边界

1. **输入与观测**：物理相机身份仍由instance_id注册表解析；cam0=LEFT、cam1=RIGHT。已有检测输入可转正，但进入几何前逆旋转回原始鱼眼像素。人物关联和逐点几何质量门不改变，最多匹配一对人物；失败帧、拒绝原因及NaN保持可见。
2. **身体分支**：按mainline_selection.body.source_command保存的reference25参数重新构造观测，在相机坐标系求SMPL-H。A30步只优化逐帧transl与VPoser latent，root冻结；B30步只优化全片同一个共享beta；C300步优化共享beta以及逐帧root、transl、latent；D1/D2各100步只优化共享lhand/rhand PCA，身体和beta冻结；D3为0。准确可训练集合来自源码stage plan，不把预算名称当变量集合。WiLoR原生MANO局部旋转以0.1权重提供手指先验，不使用它的global_orient/transl/beta作为人体或腕的绝对位置。360帧人工腕、身体二维重投影、身体时序和表面接触项未启用。冻结结果用于场景对照，不在此阶段重新求解身体。
3. **场景严格点分支**：地面入口实际消费360帧strict JSONL的3031个接受点。身体拟合保存的观测和场景严格点是两个明确的数据产品；源命令可能独立重三角化，不能仅因来自同一检测器就认为接受掩码相同。scene NPZ中的有效XYZ/mask必须与strict文件逐项相同；detector_only重算仅为诊断，不能补充脚锚或填失败点。
4. **Stage识别**：原背景/人体运动证据经原TwoStageRecognizer产生Stage状态和不可用原因。Stage1表达助步器/相机静止、人体可动；Stage2表达人体脚锚静止、助步器/相机可动。它们是工程动作协议，不是逐帧独立真值；保留原状态确认与活动carry，不引入候选交接状态机。
5. **地面位姿**：使用当前原始图像估计被质量门接受的背景旋转；Stage2按严格脚点和因果中值观测估计完整SE(3)地面增量。Stage1仍使用原支撑恢复，缺测保留状态及原因。姿态定义为相机点映射到地面系的T_G<-C_t；场景显示和指标共同消费最终关闭/发布的位姿流。原硬恢复的跳变是已知缺陷，不能凭主线选择消除。
6. **人体、地面、助步器共用变换**：身体相机系网格乘当前T_G<-C_t；地面来自当前位姿；相机刚性连接的助步器使用同一变换及既有结构参考。没有单独抬降人体、脚或助步器的显示补偿。静态助步器安装尚未在本次布置重新认证，只能作为结构/显示参考。
7. **审计与发布**：保留360帧编号、实际配对时间、严格点、阶段、R/t及原始拒绝原因。时间是host-return，不能声称曝光同步。结果生成后核对JSONL/NPZ发布一致性、参数重建和真实模型面。离线整段拟合、实时Stage/地面和页面播放是三个不同接口，不能互相替代耗时验证。

### 复现版本、资源与验证入口

`current_mainline.json/mainline_selection`保存当前选择，`scene_route/default_profile`是场景别名解析的唯一来源；`stage_ground_profiles.json/baseline/options`是场景算法开关来源。未传scene-profile时保留历史调用行为；显式mainline时固定baseline开关，冲突参数直接报错。输出同时保存请求的scene_profile和resolved_scene_profile。别名不会改身体manifest中的448帧冻结参数。

360帧实际身体命令参数：`G20261005_reference_weight_fit_v1/runs/reference25/fit_args.json`；显式冻结25px/80mm/0.6的复现参数：同实验`mainline_fit_args.json`，通过`body.fit_smplh_wilor_sequence`调用。准确场景命令：同目录`stage_landing_recovery_v1/mainline_alias_command.json`。这些路径均位于`research_records/engineering_validation`下。命令列表记录实际输入和输出；复跑必须将output/output-dir换到不存在的新目录，不能覆盖对照。图像、标定、测量地面、WiLoR缓存及授权SMPL-H/VPoser资源均需本地恢复；缺失时报告不可用，不从旧动态场景或其他采集填补。

公开检查使用`realtime_app/run_validation.py`；本次针对路由、严格观测和保留候选运行`test_scene_replay_contract.py`、`test_stage_handoff.py`、`test_landing_pose_recovery.py`、`test_joint_stage_window.py`和`test_mainline_regression.py`。448帧身体回归仍使用该数据集原冻结命令，不自动读取360帧路径。mainline_alias_validation记录360帧别名重放与baseline的逐值对照，通过仅说明路由没有改变计算。

后续改变Stage/落地算法必须另列输入、唯一变量、成功门和停止条件；至少同时检查静止持有、活动carry、原严格脚锚保护、落地高度/倾角闭合、位置/速度/加速度跳变以及测量加计算的总延迟。联合窗口目前只能通过部分运动指标，不能替代基线。若基线也未达到科研目标，应明确两者都未通过，而不是把主线标签当成功判定。

## 2026-10-04 审查后的调用与等级

采用部分替换：严格输入导出和最终发布流导出修正进入共用场景入口；落地恢复不替代默认阶段算法。身体冷启动路线、检测器候选等级和优化参数没有改变。实时 `run_stereo.py` 没有接入固定延迟落地恢复；以下候选入口是离线重放。

场景参数统一保存在 `realtime_app/configs/stage_ground_profiles.json`：

| 配置 | 行为 | 级别 |
|---|---|---|
| `baseline` | 原阶段识别、完整SE(3)、因果踝中值、原Stage 1硬恢复 | 保留工程基线，仍有跳变与落地假设问题 |
| `joint_stage_window` | 未发布窗口内联合SE(3)，原阶段/严格点/有限carry门保留 | 离线候选；累计漂移和实时耗时未通过 |
| `landing_fixed_lag` | 交接状态机、当前踝、最近脚锚、单踝回退、0.5秒未发布缓冲中的支撑恢复 | 离线实验候选，不能称为实时已验证或真实接触 |

`--scene-profile` 固定算法开关，输入路径由实验提供。冲突的显式参数报错；不支持缩写参数。候选必须传入真实 `--frames-metadata`。不传配置仍保留原入口的默认算法，避免历史脚本静默换方法。`baseline_review` 实验是旧 `hard_restore` 的**交接硬恢复对照**，不是 `baseline` 配置，勿混用其结果。

### 输入如何进入计算

1. 读取左右检测帧，要求帧集合相同且从0连续编号；不截断到公共前缀。读取选定图像的 `left_ccw90/right_cw90/pair_XXXX.png`，逆旋转回原始鱼眼像素。标定始终cam0=LEFT、cam1=RIGHT。
2. 读取同次拟合的严格JSONL。每帧最多一对人物，人物编号必须(0,0)；有效关节需正双侧深度、有限XYZ、不重复索引。空人物帧完整保留。接受掩码和导出XYZ直接来自此文件，拒绝点填NaN。独立重算的点仅存为 `detector_only_*` 诊断，不参与有效点导出或地面求解。右膝索引14不被省略。
3. 左右host-return时间取均值作配对时间，分别保存两侧时间和偏差。它们不是曝光同步证明。交接候选禁止用名义FPS代替真实记录。
4. 背景KLT和人体补偿运动进入原阶段识别；旋转跟踪器从当前图像求当前相机←上一相机的旋转。未通过背景质量门的旋转不冒充测量。稀疏背景平移诊断与阶段跟踪分开运行，不能作为可信VO注入。
5. 基线按原规则确认Stage 1/2；交接候选以背景运动/停止证据决定地面动作，两帧确认阶段，但首个运动帧即尝试严格脚约束更新、首个停止帧即保持位姿。未知帧采取有限carry或hold，有动作但不保证能成功测量。Stage 2脚静止来自受控动作协议，没有被独立证明。
6. Stage 1保持相机位姿并更新下一次运动所用脚锚候选。Stage 2取脚世界锚 `a` 和相机坐标踝 `p`，在旋转已知的条件下用 `t=a-Rp` 求平移。双脚结果需通过60mm一致性、120mm步幅、5度旋转等原门；0.45增益更新平移。候选用当前踝和最新锚避免跨走路段的中值滞后。单踝回退需恰有一个严格踝、锚存在、当前累计旋转无缺边，预测偏差≤80mm，并仍过步幅/角度门。它只是降级工程估计。
7. 旋转按当前←上一方向左乘累计，世界旋转使用逆旋转更新。失败帧保留原原因、位姿保持和缺边状态。现有缺边审计只描述局部累计窗，不能认证全片绝对旋转可观性；`world_pose_certified=False` 始终保留。
8. 严格点/人体模型通过 `P_ground=R*P_camera+t` 进入同一地面坐标系。模型顶点由米转毫米后计算；beta、身体姿态和手姿不会因本次场景重放重新优化。助步器只采用静态相机安装模型随同一变换计算；旧机位安装几何不是本次测量真值。
9. 固定延迟候选只在未发布缓冲中修正位姿。关闭writer后重新读取最终JSONL导出NPZ，检查360个ID顺序，禁止将consume返回的临时可变对象作为已发布结果。可视化读取最终同源R/t及真实三角面，不再增加展示平滑或踝点吸附。

### 落地恢复内部如何做

先观察至少两个连续运动样本、运动起止跨度至少0.1秒，再观察至少0.12秒连续停止和至少两个连续的人体/下肢恢复运动样本。人体证据要求同一视图上半身至少3点且补偿运动≥1.2px，踝至少1点且背景残差≥1.2px。单纯背景静止、只有手臂动、短暂停止均不等于落地。本次修正：未知/不明确动作切断这条证据链；停止后再运动重建运动起点；人体恢复票必须连续，不再累计零散票。

显式协议 `flat_ground_supported_after_motion` 提供“落架后回到同一平地支撑姿态”的先验。目标保持XY和平面yaw，恢复初始高度与相对roll/pitch；不是所有姿态归零。高度差>150mm或旋转差>8度拒绝恢复，不裁剪成成功。

固定延迟模式保留0.5秒未发布行，取停止前最多0.3秒的缓冲窗，按真实时间计算 `u` 并以 `3u²-2u³` 混合高度和SO(3)旋转。已经发布的行不修改。确认后重建锚并清空旧平移历史。actual release delay逐行记录，流尾有专门标记；非因果输出明确 `causal=False`。这是确定性协议先验投影，**不是联合优化、接触认证或闭环三维定位**。

因果对照以时间常数0.12秒渐进恢复，限制200mm/s、20deg/s，但因此静止段仍有相机下移；不能把它称为严格助步器静止。固定延迟模式减少此冲突，但会增加可见延迟。

每条修正行保留恢复前R/t、原状态/拒绝原因和原脚锚，另计算恢复后的踝残差；测量质量门的审计范围标为 `before_support_restoration`。通过旧门不代表恢复后仍通过，重新建锚也不等于误差消失。

### 可复现入口

在仓库根目录使用现有环境；以下输入共同对应换机位360帧，不得换入448帧旧腕参考：

```powershell
$src = 'research_records/engineering_validation/G20261003_repositioned360_cold_smplh_v2'
$r = 'research_records/engineering_validation/G20261003_repositioned360_manual_v1'
.\.venv-cuda\Scripts\python.exe -B realtime_app/run_tool.py scene.replay_fitted_body_scene `
  --scene-profile landing_fixed_lag `
  --left-video unused_left --right-video unused_right `
  --input-pair-dir "$src/input" `
  --left-pmpose "$src/left_pmpose_raw.json" --right-pmpose "$src/right_pmpose_raw.json" `
  --calibration "$r/calibration/stereo_fisheye.json" `
  --ground-reference realtime_app/calibration/results/static_ground_20260927_201028/ground_reference.json `
  --walker-model research_records/engineering_validation/G20260918_offline_walker_frame_evidence_v1/coarse_complete_model_v2_camera_rail/coarse_walker_model.json `
  --strict-jsonl "$r/pmpose/first_instance_strict.jsonl" --frames-metadata "$r/frames.json" `
  --output "$r/stage_landing_recovery_v1/your_new_output"
```

基线替换为 `--scene-profile baseline`；不得把这一变化称为只切换落地恢复，因为候选还改变了交接、踝输入与回退。严格单变量对照使用已有 `hard_restore_command.json`，本次准确命令保存在 `baseline_review_command.json`、`lag_review_command.json`。输出目录非空即拒绝覆盖。底层JSON/图像、静态标定与模型需本机提供，源码仓库本身不含全部原始数据/模型资源。已有可视化 `view_final` 固定对应旧 `lag_lowerbody`，不能直接配新重放R/t而仍引用旧结果。

独立审查命令：`python -B realtime_app/run_tool.py scene.audit_stage_handoff --review --root research_records/engineering_validation/G20261003_repositioned360_manual_v1 --experiment research_records/engineering_validation/G20261003_repositioned360_manual_v1/stage_landing_recovery_v1`。它要求本次三份重放目录与冻结对照均存在，不会自动重跑拟合。

复现检查：严格输入mask/XYZ与NPZ逐点对应、最终JSONL与NPZ位姿一致、R正交且det=1、所有帧顺序完整、重放不修改冻结身体参数。关键指标同时报相机跳变、人体变化、恢复前后踝残差、鞋底高度和实际发布延迟。审查结果保存在本实验 `review_validation.json`，以其数值为准。

### 替换决定与尚未解决的冲突

冻结旧候选恢复后踝残差P95约63mm，恢复前约23mm；鞋底最小约-52.3mm。本次360帧修订候选仅在98/150确认两次恢复，后续18次超限拒绝，第333帧高度偏差187.13mm、倾角8.35度；两次恢复的相同12个踝样本残差P95从22.99增至41.07mm。它没有通过闭合和脚锚兼容性要求。旧导出5949个有效点中有2918个不属于严格接受集，修正后3031；硬恢复对照R/t逐值不变。高度归零是施加先验的结果，不是独立精度证据。frame333严格点全部不可用，不能用该帧闭合作为观测验证。缺少物理支撑状态和新安装测量，整套算法暂不能替换基线。当前只采用共用入口的数据来源与发布修正，并给候选明确、可重复的调用位置。下一项科研修正应在同一窗口内同时检查脚锚与支撑姿态兼容性，失败时保留冲突原因；不能通过放宽门或直接更新锚消除报告中的残差。

## 1. 目标与物理约束

系统处理安装在无轮支架式助步器上的双鱼眼相机：

- Stage 1：助步器和相机静止，人体相对相机运动；保持相机地面位姿。
- Stage 2：人的双脚近似静止，助步器和相机运动；用静态背景估计三轴旋转，用左右踝世界锚反推 XYZ。
- Transition：只允许紧邻 Stage 2 的最多 4 帧继续求解落架运动；随后保持位姿，等待 Stage 1 确认。

固定地面变换统一写作：

```text
P_ground = R_ground_from_left_camera * P_left_camera + t_ground_from_left_camera
```

初始 `R,t` 来自已测静态地面参考。当前结果是工程估计，没有外部相机轨迹或人体三维真值。

## 2. Stage 1/2 识别

实现：`realtime_app/pose_app/realtime_stage_walker.py`。

每个相机在 480 像素工作宽度上独立执行：

1. 选择分数最高的主要人体，构造并膨胀人体排除区。
2. 排除多帧积累的相机附着结构候选。
3. 在 3 x 4 网格中均匀提取背景角点，执行 KLT 光流和 RANSAC 部分仿射拟合。
4. 至少 18 个背景内点才可用；可靠视图还要求内点比例至少 0.35。
5. 每侧维护 5 帧运动窗。单帧中位运动至少 1.5 px，或窗口累计位移至少 1.5 px、路径至少 1.8 px且方向一致性至少 0.55，判为相机运动；单帧不超过 0.8 px且窗口累计不超过 1.0 px，判为相机静止。
6. 左右视图按“保运动”规则融合：任一可靠视图确认运动，就不允许另一侧的近零结果把运动隐藏。

人体证据使用选定可靠视图：

- Stage 1 候选：相机静止，背景仿射补偿后的上半身中位运动至少 1.2 px。
- Stage 2 候选：相机运动、左右踝均存在、双踝最大背景预测残差不超过 2.5 px、补偿后踝间距误差不超过 2.0 px。
- 相机运动是 Stage 1 的硬排除条件。

时间状态机：

- Stage 1：最近 5 帧至少 3 票；已确认后允许人体短暂停顿，但背景必须继续静止。
- Stage 2：最近 3 帧至少 2 票；已确认后若背景继续运动，允许最多 3 帧脚部证据短缺失。
- 其余情况输出 `transition`；初始证据不足输出 `warming_up`。

people_1 保存帧回放的最终状态数为 Stage 1/Stage 2/Transition/Warmup=`214/115/114/5`。无人工阶段标签，因此这些数量不是识别准确率。

## 3. Stage 2 完整 SE(3)

实现：

- `realtime_app/pose_app/static_background_rotation.py`
- `realtime_app/pose_app/dynamic_ground_pose.py`
- `realtime_app/pose_app/dynamic_ground_live.py`

### 3.1 旋转

左右相机分别在排除人体、且具有非零流的场景特征上估计相邻帧旋转：

- KLT 前后向误差不超过 1.25 px，流量范围 0.40--70 px。
- 本质矩阵 `recoverPose` 可用时优先采用。
- 小运动或近纯旋转退化时，使用单位视线 Wahba/Kabsch RANSAC 回退；80 次迭代、0.75 deg 内点门。
- 两侧都必须得到旋转；右目旋转变换到左目坐标后，左右差异不得超过 1.5 deg。
- 若某帧旋转不可用，显式保持上一旋转，但仍允许脚约束更新 XYZ，状态记为 `accepted_rotation_held`。

### 3.2 XYZ

Stage 1 中保存最近 5 帧严格左右踝的世界坐标中位数作为锚点 `A_left,A_right`。Stage 2 当前帧仍要求两只严格脚都存在；脚观测使用不含未来帧的最近 5 帧中位数。

给定当前旋转 `R` 和相机坐标脚点 `p_i`，每只脚分别给出相机世界平移：

```text
t_i = A_i - R * p_i
```

两脚推算平移差不超过 60 mm 时取均值。超过 60 mm 时，只有最近两次成功平移形成的运动预测能明确选择一脚才允许恢复：较优脚预测误差不超过 `80 mm x 恢复间隔`，且至少比另一脚优 20 mm；否则拒绝该帧。

基础平移步长门为 120 mm，并按距上次成功更新的帧数扩展，最多 4 倍；旋转步长门保持 5 deg。这样可以在短时证据丢失后恢复累计运动，而不会再次形成固定单帧门造成的拒绝死锁。

Stage 2 已稳定进入后，位姿层允许最多 2 帧低于 0.5 的阶段投票。Stage 2 后最多 4 个 Transition 帧继续求解。确认重新进入 Stage 1 时，仅将高度、俯仰和横滚重锚到刚性支架落稳参考，保留已累计 XY 和偏航。

## 4. 因果滤波与显示

“无视频平滑”不等于“估计器无滤波”：

- 位姿估计保留 5 帧脚点因果中位数。
- 接受的相对旋转完整累计，不能乘0.45；平移使用0.45更新增益。活动段脚点缺失或求解拒绝时保留已接受的旋转边，缺失旋转边另记链完整性。
- 最终显示不做骨架跨帧平均、缺失点保持或 Stage 2 踝吸附。

最终渲染器：`realtime_app/tools/render_raw_visual_handle_interaction_video.py`。

- 当前帧严格双目点优先；缺失关节可使用 force-all 点仅作显示。
- 所有关节、扶手和轨迹使用实线，逐点保留重投影误差与来源。
- 单一 `z=0` 地面和完整两面墙角坐标系。
- 显式绘制层级：地面最低，投影轨迹居中，扶手其次，人体骨架与关节点最前。
- 局部扶手轴只是当前帧双目图像候选；腕到扶手小于 60 mm且腕重投影误差不超过 10 px，只能报告视觉接近候选，不能报告接触。

## 5. 实时入口

主入口为 `realtime_app/run_stereo.py`：

```text
--enable-stage-walker
--dynamic-ground-reference <ground_reference.json>
--dynamic-ground-full-se3
```

保存帧复现顺序：

1. `realtime_app/tools/replay_realtime_stage_walker.py`
2. `realtime_app/tools/replay_realtime_full_se3_branch.py`
3. `realtime_app/tools/reconstruct_stereo_handle_axes.py`
4. `realtime_app/tools/render_raw_visual_handle_interaction_video.py`

最终 people_1 回放：Stage 2 更新 `101/115` 帧，紧邻落架过渡更新 22 帧；第一段 pair 0068--0081 相机位移为 200.6 mm，相机 Z 范围 692.887--746.512 mm。完整 SE(3) 附加支路中位/P95为 11.361/29.041 ms，不含姿态推理、解码、主双目链、显示和编码。

## 6. 当前保留的本地证据

Git 不提交原始数据、JSONL和视频。当前本地只保留以下主线资产：

- 阶段流：`G20260915_stage_classifier_v2_sequential_fixes/final_v2/`
- 静态地面数值参考：`G20260914_static_ground_reference_v1/estimate_stationary_20pairs_v2/`
- 实时 SE(3) 回放：`G20260916_full_se3_static_background_vo_v1/realtime_full_se3_branch_replay_v4_stage2_recovery/`
- Stage 2 恢复审计：`G20260916_full_se3_static_background_vo_v1/fused_stage2_latched_transition_support_v13/`
- 当前帧扶手轴：`G20260916_full_se3_static_background_vo_v1/independent_stereo_handle_axes_v1/`
- 最终视频：`G20260916_full_se3_static_background_vo_v1/realtime_exact_stage2_recovered_visual_v17_foreground_skeleton/visualize_walking_pose.mp4`

## 7. 结论边界

- KLT背景运动和阶段状态没有人工逐帧真值。
- Stage 2的XYZ主要来自脚锚，不是背景视觉独立恢复的平移。
- 脚锚是COCO踝，不是鞋底接触点；不能由此报告真实触地事件。
- 助步器身份排除仍是运动学候选，不是语义真值。
- 保存帧回放时延不能替代物理双摄端到端实时验收。

## 8. 第三、第四优先级验证状态（2026-09-30）

- 第三优先级的外部轨迹指标已实现于 `realtime_app/pose_app/ground_external_validation.py`，包含平移/旋转误差、地面法向漂移、support 脚速、Stage 切换跳变和失败原因计数。
- 当前 360 帧只有内部工程诊断，没有外部刚体轨迹或 IMU，因此 Stage 2 仍是工程估计；内部诊断输出见 `资料/实验报告/stage2_internal_diagnostic_20260930.json`。
- 第四优先级保留 SMPL 固定鞋底 54 点/脚的 heel、ball、toe 分区。接触面积当前只能报告顶点覆盖代理，不能写成物理接触面积；必须用压力鞋垫或人工逐帧标签验证 precision、recall 和事件时序误差。
- C1/C2 逻辑保留，当前权重冻结为 surface=1、nonpenetration=1、tangential=0.001；不因内部指标直接删除。


## 9. 2026-10-04：连续交接候选与当前决策

2026-10-05当前换机位360帧主线展示为`G20261005_reference25_mainline_view_v2/body.html`，使用reference25身体和新重放baseline场景；旧`full_view_rotation_fix_v1/body.html`保留历史展示身份。新实验不替换原`RealtimeDynamicGroundWriter`默认参数；方法级别仍是工程候选。

同源回放新增可选`--frames-metadata <frames.json>`与`--foot-observation-mode current`。前者校验保存图身份、双侧主机时间和偏差，输出配对及左右时间；host_return不是曝光时间。360对实际主机跨度17.4267秒，旧30fps名义时间只有11.9667秒。后者改变当前严格踝观测的因果滤波方式，未改变人体检测与严格三角化门。

`--continuous-stage --frames-metadata <frames.json>`启用`pose_app/continuous_stage_ground.py`。相机静止保持当前R/t；相机运动/交接持续处理当前脚点和旋转，独立建立左右锚，支撑权重按时间渐进激活。缺测最多0.2秒预测，之后显式不可用。首帧单独初始化；缺失旋转链不因重新静止而恢复绝对世界认证。Stage2脚锚是操作性COCO踝参考，不能称鞋底接触。连续策略已逐帧运行，但每帧处理不等于每帧都有可靠测量。

背景VO用`--background-vo-diagnostic`只输出审计；可追加`--background-vo-rectified`比较固定局部校正前端。均直接沿用现有背景前端和标定。20链接/12训练内点及留出重投影门未通过时不能注入地面；人体排除不能认证助步器排除。

最终证据目录：`research_records/engineering_validation/G20261003_repositioned360_manual_v1/stage_continuous_final_v1/`。控制协议`protocol.json`、完整比较`comparison.json`、验证`validation.json`；最终连续运行`continuous_checked/`，其他continuous目录为保留的中间运行。当前点策略仅部分区间改善并增加拒绝；连续候选69帧不可用、123–136区间偏移退化，背景两种前端全片未达匹配门。所有候选`accepted_for_mainline=False`。上游门未通过，身体踝约束和短窗口联合求解尚未执行，不以平滑代替缺失证据。更完整结果见`AI_PROGRESS.md`本阶段记录。


## 2026-10-04：交接诊断候选的当前位置

当前默认仍为既有Stage/地面流程。`stage_transition_diagnosis_v1`保存冻结身体下的完整对照，不以过渡帧数量减少认定方法通过。背景求解按用户要求改为6..20点并保留退化、深度、重投影和运动门；背景仍只有诊断权限，不能注入地面。新增阶段交接、最新静止踝锚和单严格踝回退是显式实验候选，非自动升级主线。

候选每帧执行：背景静止保持当前位姿；运动起始即进入脚锚求解；模糊边最多沿用原4帧活动carry，其余保持并标记不可用。Stage1允许人体暂停，Stage2脚静止来源是受控动作协议。双踝门原样，单踝必须是当前严格有效点且旋转链完整，保留另一脚失败。缺测保持不是无动作，也不是得到正确运动恢复。

最新锚+单踝候选将旧21运动过渡帧中更新数量2提升到20，并改善所列问题段最大骨盆偏移；但frame274仍有约41mm跳变，身体独立拟合与脚锚补偿存在分歧。相机光轴累计负向指标也变差，不能当人体真实后移量解释。详细数字、全部失败及下一步边界见AI_PROGRESS和实验JSON。没有外部轨迹或逐帧阶段真值，保留工程候选级别，不宣称全片问题解决。


### 补充纠正：取消恢复候选不满足跨阶段闭合

同日进一步核查发现：仅取消恢复时，静止frame333高度累计偏差182.7mm、倾角8.46度；最新锚+单踝候选达到302.7mm、14.07度。因此段内骨盆偏移变小不足以支持整体改善，候选仍未通过，不替换默认。详见现有validation.json的height_tilt_closure_audit。正确修正方向是有落地依据的高度/roll/pitch约束及连续交接，保留XY/yaw，不把背景静止直接等同落地，不在显示层分别下移人体或助步器。下一轮尚待实施。


### 协议支持的落地恢复已实现（工程候选）

`landing_pose_recovery.py`由现有StageHandoffGroundWriter消费，现有场景重放入口显式选择`--landing-recovery causal|fixed_lag --landing-protocol flat_ground_supported_after_motion`。原默认不变。要求前序运动、持续停止和人体/下肢恢复运动，按受控平地协议施加高度/倾角先验，保留XY/yaw，并重建脚锚；静止、手臂动或短暂停顿都不自动等同落地。缓冲候选0.5秒，仅修改未发布位姿，JSONL/NPZ同源且causal=False；实际样本延迟和流尾状态记录齐全。它是确定性支撑先验投影，尚非完整联合窗口优化。

stage_landing_recovery_v1的最终lag_lowerbody在98/147/219/300确认恢复，frame333累计高度/倾角闭合，避免原硬恢复97.3mm瞬时相机跳变（降至47.1mm），静止段没有因果恢复的持续相机下移。285–286短暂停止保留未确认状态。修正踝残差P95从23增至63mm，鞋底最小-52.3mm未解决，因此保留工程候选，不升级主线或物理接触证据；完整数字见comparison/validation和AI_PROGRESS。


### 2026-10-04 连续联合窗口：实现与验收

入口仍为 `scene.replay_fitted_body_scene`，新增配置 `--scene-profile joint_stage_window`，对应 `--landing-recovery joint_window`。其他输入沿用本文件完整命令；输出另选非空不存在的目录。窗口求解实现拆入 `realtime_app/pose_app/joint_stage_window.py`，现有landing controller负责缓冲和发布，StageHandoffGroundWriter负责提供未经补齐的原始背景旋转与严格踝点/原脚锚。SciPy已在现有requirements-mainline中固定，不新增模型或重拟合身体。

算法不再将落地修正按固定0.3秒曲线集中到末尾。活动运动帧到达时即尝试联合估计0.5秒未发布窗口中的相机XYZ与三轴旋转；连续停止帧共享末个运动帧的同一位姿。窗口首个状态固定，作为已发布历史和新窗口之间的边界；已发布帧不修改。原ground writer退出活动期后不启动求解，窗口含已退出活动期的非Stage1过渡行时整体拒绝，不能借窗口优化延长原四帧carry。静止可发生在空中，只有原有运动/停止/人体下肢恢复连续证据加显式平地协议满足后才加入支撑先验。

优化采用SO(3)增量 `R_i=Exp(delta_r_i) R_i_base` 和平移增量 `t_i=t_i_base+delta_t_i`。停止共享同一组参数。目标是以下归一化残差的平方和：

- 运动帧严格踝点的世界脚锚偏差，尺度20mm；Stage1人体恢复行不再使用脚静止因子。被双脚间距或平移一致性门拒绝的行不重新用作脚因子。
- 已接受背景旋转边 `R_i≈R_(i-1) R_to_from^T` 的SO(3)残差，尺度0.5度；缺边不填identity，旋转可观性不因此升级。
- 相邻真实时间间隔的速度差除以平均间隔，构成平移/旋转加速度项，尺度分别1000mm/s²、100deg/s²。这是工程正则，并非人体或助步器动力学模型。
- 防止自由变量偏离当前估计过多的弱先验，平移50mm、旋转10度。它不冒充独立观测。
- 仅在落地证据确认时加入末端相机高度残差（2mm尺度）和地面法向残差（0.2度尺度）；不固定XY/yaw，不用末端直接投影覆盖解。

至少3行、2个严格脚因子及1条接受旋转边才求解。`scipy.optimize.least_squares`使用最多20次函数评估、ftol/xtol/gtol=1e-5；此上限不是20次总矩阵计算，也不是墙钟实时限额。预算耗尽、完整目标未下降、原120mm/5度/60mm门失败、脚锚P95比原测量位姿恶化超过2mm，均拒绝整个提案。要求支撑时，末端还必须在5mm高度/0.5度倾角内。外层原150mm/8度恢复幅度门保留。失败不裁剪成成功，不插值严格点。

重复窗口的脚保护始终参照第一次修改前的 `measurement_pose_before_support`，不参照已被上一窗口修改的位姿，避免每次容忍2mm而逐次漂移。被接受的窗口只修改未发布行，另记录`joint_window_attempt/solution`、factor数、目标前后值、残差、求解耗时和修正类型；原measurement/status/reason、严格掩码、world_pose_certified=False保留。当前变量名沿用旧恢复接口，`correction_type=joint_motion_estimate`与`joint_protocol_support`明确区分运动修正和协议落地。

冻结对照 `lag_review`，保留首次 `joint_window_review`、原测量残差保护后的 `joint_window_checked` 与活动门保护后的 `joint_window_gated`。不以中间版本较好看的指标代替最终版本。审查统一命令：

```powershell
.\.venv-cuda\Scripts\python.exe -B realtime_app/run_tool.py scene.audit_stage_handoff --joint-window `
  --root research_records/engineering_validation/G20261003_repositioned360_manual_v1 `
  --experiment research_records/engineering_validation/G20261003_repositioned360_manual_v1/stage_landing_recovery_v1
```

各完整重放命令为实验目录中的 `joint_window_*_command.json`。核查输出 `joint_window_validation.json`；包含全部360帧、严格mask和同源XYZ、发布JSONL/NPZ、合法旋转、阶段序列、静止段位移、人体变化、真实时间速度/加速度及窗口计时。计时不包含检测器推理，不将整次离线重放平均帧耗时当实时部署能力。

最终活动门版本相对lag_review：速度峰值835.93→691.69mm/s，加速度P95 6635.18→2430.77mm/s²；相机最大帧步长47.13→53.00mm（此项反而变差），骨盆步长P95 29.99→28.95mm，静止段最大相机位移均为0。第333帧高度偏差187.13→183.41mm，倾角8.35→8.69度，仍没有闭合。34次窗口提案接受、14次拒绝、64次不可用，其中53次因窗口含 inactive transition；只在150确认一次协议恢复，18次后续恢复仍超限。窗口求解耗时P95约598.63ms、最大686.72ms，名义缓冲0.5s的样本时间延迟P95约0.563s（不含上述计算耗时）。

决定：实现通过工程回归，但科研目标尚未通过，不替代默认主线，也不把中间checked的487mm/s峰值写成最终版本指标。下一阶段先精确划分连续可观测窗口与不可观测边界，避免一个失效过渡帧让整窗长期不可用；保持该边界位姿与拒绝证据。随后检查脚锚来源及绝对支撑姿态之间的兼容性，并优化稀疏/解析Jacobian。不能增加末端恢复力度或放宽150mm/8度门来获得假闭合。现有viewer仍为旧冻结候选，没有冒充新窗口结果。


## 2026-10-12 supported_stop_fixed_lag：448帧明确停止支撑协议候选

此候选依据用户确认的采集前提：助步器每次停止移动时四脚已放在同一平地。动作Stage1仍按背景静止+人体运动分类，其标签不等同相机支撑状态。StageHandoffPolicy(preserve_motion_labels=True)保持原动作分类，独立背景停止给ground_stage=Stage1；运动位姿更新仍需原Stage2/足点门及原vote，不以camera moving放宽。LandingPoseRecovery新协议supported_at_camera_stop不等待human resume，以整段相机运动episode和稳定停止为证据，短暂未确认停止不清空既有motion_samples；未知超过0.3s仍断链，旧flat_ground_supported_after_motion协议不变。

固定延迟15名义帧（30FPS约0.5s），只调整尚未发布缓冲：停止前最多0.3s区间以显式SE3支撑修正完成高度/roll/pitch闭合，保留XY/yaw；不是网页时间插值。记录原位姿、原失败与post-support旧ankle anchor残差，>150mm/8°拒绝。原始静态安装与平地支撑协议是先验，不是自动识别的物理触地。旧448无曝光/host时间，新模式仅名义时间离线验证；此前候选host时间要求不被通用取消。

G20261012_supported_stop448_v1/view_v4完整448验证：11原Stage1边界高度步长全零，所有Stage1四脚参考地面Z数值为零；最大帧高度步长69.52→13.61mm、旋转5.84→1.89°、任一助步器节点步长96.65→49.45mm。XY最大步长30.51→33.24mm，旧anchor残差99.58mm保留，不能说所有运动指标提升。view_v2/v3仍63.8mm支撑误差及回归反例均保留。33测试、22来源检查和浏览器实测通过。默认baseline与正式360选择不变；新候选需显式--scene-profile supported_stop_fixed_lag，完整协议/对照/失败见该研究目录。

## 2026-10-14 独立视觉世界位姿路线调研与设计决定

**状态：文献与现有实现核对后的设计建议，尚未实现新后端、安装模型或运行新场景实验。** 本节不更改 `mainline -> supported_stop_fixed_lag` 的选择，不替换正式360帧或新448帧身体。检索截止2026-10-14；以下排序针对本项目双鱼眼、动态人体、相机安装在助步器及本机12 GB显存条件，不是跨任务的SOTA排名。

### 1. 这次要解决的具体问题

现有场景入口的完整平移仍由 Stage 2 脚锚计算 `t=a-Rp`，背景前端主要提供旋转。这样做把“脚在世界中暂时不动”的动作协议当成了相机平移观测；一旦脚点缺失、两脚不一致或动作标签误判，人体、助步器和相机就会一起产生错误的世界位姿。Stage 1 的保持位姿和 `supported_stop_fixed_lag` 的平地修正可以减少跳变，但它们不能替代静态背景的独立运动估计。

已有诊断不能写成“背景 VO 全部失败”：`research_records/engineering_validation/G20261003_repositioned360_manual_v1/stage_transition_diagnosis_v1/comparison.json` 中 `stage_predict` 为 76/360 帧 accepted，`background_wide` 为 106/360 帧 accepted。不过两条分支均为 `pose_injected=false`。`background_wide` 的中位处理时间约 707.8 ms，最大链接数 37；主要失败原因为少于 6 条链接（241/360）。`causal_stereo_background_pose.py` 仍明确输出 `mask_certified=false`、`metric_pose_usable=false`：仅有人体mask，助步器/背景静态归属未经确认；`continuous_stage_ground.py` 要求可用标记为真才消费视觉平移。因此缺口是可靠、持续、低成本的静态背景米制位姿，不是完全没有 PnP 代码，也不能靠直接翻转可用标记解决。

### 2. 调研后的候选排序

| 候选 | 已核对事实 | 在本项目中的角色 | 主要风险 |
|---|---|---|---|
| **ORB-SLAM3 Stereo/Fisheye** | 官方支持 stereo、fisheye、原始鱼眼图像和 TUM-VI fisheye stereo；鱼眼模式不要求先做图像整流。 | 首个可解释的独立几何基线；直接输出带双目尺度的相机轨迹，适合先验证背景静态点和回环。 | 原生实现是 Linux/C++，Windows 集成成本高；动态人体和相机附着助步器必须显式屏蔽；仍会在低纹理背景中丢跟踪。 |
| **cuVSLAM** | 官方支持 stereo/multicamera、刚性 `rig_from_camera`、动态 mask 和 `world_from_rig`；论文报告 768×480 单 stereo track call 在 RTX 4090 为 0.4 ms、Jetson AGX Orin 为 1.8 ms，明确只计 tracking 调用。 | 速度优先的独立视觉轨迹后端；本机GPU部署候选，WSL2可行性未实测。 | 官方 wheel 面向 Ubuntu 22/24 或 Jetson，无原生 Windows wheel；文档鱼眼模型有 FOV<180°限制；还需核对同步和畸变语义。论文调用时间不是整链路 FPS。 |
| **Pi3X（π³项目）小窗口** | 官方更新支持 rays、intrinsics、poses、连续 confidence 和 approximate metric scale；A800 80 GB/BF16、504×504、512张图的官方记录约39 GB/203.68 s（multimodal=false）。这不支持将448双目对即896张图整批塞入12 GB的方案。 | 独立学习几何候选，也可给几何后端提供候选深度和重定位初始化；先试8–16张图是拟议起点，尚无本机容量/速度结论。 | 条件输入不是硬几何约束；源码将射线化为xy/z并归一化位姿尺度，不能仅凭输入基线宣称米制。需要有效鱼眼射线适配、外部双目尺度约束；较低分辨率或更小窗口另行测量。 |
| **RayMap3R** | 2026 论文提出无训练的双分支动态抑制；RTX A6000 48 GB 上 50/1000 views 均约 9.2/9.4 GB、13.8 FPS，使用隐状态保持常量内存；相机 ATE 从基础 0.114 到完整 0.081，其中 state-aware smoothing 贡献明显。 | 动态人体干扰诊断和候选静态性先验；可作为视觉后端的第二意见，不直接写入主线世界位姿。 | RayMap 公式是针孔射线；尺度对齐和轨迹平滑会改善数值但不等于新观测；12 GB、鱼眼和本项目场景均未实测。 |
| **MASt3R-SLAM / Fast3R / BLASt3R** | MASt3R-SLAM（CVPR2025）约15 FPS/RTX4090，支持跟踪、重定位和回环，使用Sim(3)；Fast3R（CVPR2025）在A100/512×384时8图6.33 GiB、32图13.25 GiB；BLASt3R为2026-09预印本，公开实验主要使用80 GB卡。 | 完整学习SLAM备选、关键帧重建及离线对照；BLASt3R纳入近期方法观察。 | 未选作首个部署候选，不代表已证明不能在12 GB运行。仍需解决动态前景、双鱼眼、尺度与连续延迟；批量吞吐不能当逐帧时延。 |

RayMap3R相机评估包含Sim(3)对齐，不能把ATE当本项目米制漂移保证；其代码 `infer.py` 默认 `model_update_type=cut3r`，完整RayMap门需要显式 `xattn`，不能将默认命令等同论文完整方案。StreamVGGT、CUT3R等流式结构也不能只凭“streaming”名称认定运行显存恒定，要核对KV缓存或整片预编码。

人体世界估计方面，OnlineHMR（CVPR2026）已有“相机系人体 + 增量SLAM + 尺度恢复”的分离结构，完整流程报告3.3 FPS/RTX6000 Ada；JOSH（ICLR2026）联合人体、场景和接触，官方测试环境24 GB。它们支持借鉴坐标分离与人体mask的设计，但不证明可直接用于本机双鱼眼，也不能把这项系统分解本身包装成新颖贡献。当前SMPL-H已提供人体形状和姿态，优先补世界轨迹可减少重复网络。

### 3. 建议的系统结构与传感器适配

拟新增独立的 `visual_rig_pose` 分支，输入原始左/右鱼眼帧、正式鱼眼内参、双目外参和帧时间，输出每帧 `world_from_rig`、相邻增量及后端实际可提供的质量信息：重投影残差、内点空间覆盖、动态/遮挡比例、退化方向、地图/重定位状态和 `pose_reliability`。缺失字段不能伪填已测。该分支不读取 Stage 1/2 标签，不从脚踝估计平移；稀疏静态地图已能支持这项任务，稠密房间表面或3DGS不是第一阶段必需项。

第一后端优先验证 ORB-SLAM3 鱼眼 stereo；如果部署环境允许，第二后端验证 cuVSLAM stereo。两者都必须把人和相机附着助步器从静态背景证据中排除。人体 mask 只能排除人体，不能自动证明助步器是静态世界物体；助步器应使用保守的相机附着 ROI/投影 mask，并把该 mask 标记为工程先验。不得把助步器自身的杆件特征混入世界静态地图。

后端适配成本不同：ORB-SLAM3官方 `System::TrackStereo` 无mask参数，须在特征提取/匹配处接入排除逻辑，不能以涂黑图像就假定边界伪特征已被排除；cuVSLAM已有输入mask接口。前者作为可解释对照，后者作为速度候选，并不预先认定前者部署更省事。

在相机和助步器刚性安装有效的条件下，助步器不需要每帧由大模型重新估计几何和独立轨迹。以左相机C、助步器A、世界W表示坐标，`T_W<-A(t) = T_W<-C(t) T_C<-A`，人体网格 `V_W(t) = T_W<-C(t) V_C(t)`；若后端rig原点不是左光心，先乘固定 `T_rig<-C`。SMPL-H继续拟合相机系人体，三个对象使用同一变换来源。

现有粗助步器模型 `G20260918_offline_walker_frame_evidence_v1/coarse_complete_model_v2_camera_rail/coarse_walker_model.json` 提供相机—助步器参考变换，但换机位安装正确性未认证。`G20261003_repositioned360_manual_v1/camera_coordinate_reference.json` 是coordinate-only、`physical_walker_estimated=false`的身份坐标参考，并不是该粗模型；该目录 `protocol.json` 的 `walker_mount_camera_discrepancy_mm=26.4915` 是历史参考差异，不能解释为当前安装标定误差、协方差或物理精度。安装关系松动或改变时须重新确定该刚体变换。

双目标定能提供尺度来源，不提供绝对世界原点和方向。无地面/重力/已知标志锚时，SLAM地图的整体旋转和平移仍是任意坐标规约；IMU重力可约束倾斜，却不能单独固定全局yaw/位置。采集关系有效的初始ChArUco变换可将视觉地图锚定到初始地面：后续每帧通过视觉轨迹传播，**不能在相机移动后仍原封不动地使用初始相机到地面的变换**。

现有 `realtime_app/calibration/results/cam0_fisheye.json`、`cam1_fisheye.json` 为1920×1080 OpenCV fisheye；中心线HFOV约152.961°/153.429°、VFOV约82.773°/82.360°，这些数值不能证明对角/边角全部射线满足候选模型限制。正式 `stereo_fisheye.json` 约定 `X_cam1=R_cam0_to_cam1*X_cam0+T_cam0_to_cam1`，平移单位mm，适配器须显式转换单位和方向。针孔学习模型可使用由该标定生成的有效虚拟视图，但须保存其回到原始鱼眼射线的映射，不能拿针孔K直接投影原鱼眼像素。cuVSLAM文档明确等距四参数与OpenCV fisheye兼容，但有效视野限制仍需核对；不同后端的投影/反投影定义也不能仅凭同名系数直接互抄。

cuVSLAM API建议相机时间差约1 ms以内；当前旧448对只有名义帧序/时间，没有曝光同步证明。这是数据兼容性缺口：可以做离线候选比较，但不能据此认证动态米制精度。新采集需记录可解释的配对时间，不能以同一帧号代替同步证据。DA3的既有用途限制保持，不将其升级为米制深度或标定输入。

### 4. Stage 1/2 的新角色

拟将Stage 1/2从主要位姿生成机制降为可关闭的软先验。第一实现宜使用后端的相对位姿观测建立小型SE(3)位姿图，后端自己处理背景特征与双目几何。`T_t`表示rig到世界的位姿，`delta_vis_t`表示前一rig系中的当前rig位姿，`Log`将刚体变换残差转为旋转/平移向量：

```text
min  E_visual(T) + lambda_stage E_stage(T) + lambda_support E_support(T)
E_visual = sum_t w_t rho(||D^-1 Log(delta_vis_t^-1 T_(t-1)^-1 T_t)||^2)
```

其中rho为抑制离群的鲁棒损失；D是预先声明的旋转/平移尺度（弧度/米），w是工程可靠度，不是已校准的逆方差。若后端提供有效的信息矩阵，可表达不同方向的约束强弱，但仍需注明其近似假设。后端回环修正后要区分连续里程计坐标与修正地图坐标，避免把地图整体更新误作人体瞬移；重定位断段及修正量显式保留。

置信度逻辑必须同时处理以下问题，不能给各模块一个分数后直接相乘就称概率融合：

- 分开记录静态归属证据、匹配可靠度、双目几何和位姿可观测方向；网络confidence、检测分数、内点比例都不自动等于正确概率或逆协方差。低视差/单一区域纹理可能有很低回投影误差却不能约束全部平移。
- 弱证据不得除以自身权重和而恢复为满强度；缺失、陈旧、遮挡或几何退化应降低绝对观测强度。权重标度在匹配对照中固定，不能按结果临时调高Stage项。
- **同源证据只计一次。** 消费VO相对位姿时，不再把产生它的原特征作为另一套独立观测；如果以后改为统一特征BA，则用原左右像素和已知rig外参，不无条件叠加由同批像素算出的双目深度项。同理，后端已用的回环不在外层重复添加独立回环因子。
- Stage1的背景停止证据与VO可能共享特征，不能假设独立后再累加信息；初版优先让它控制先验是否启用，而不把相同背景结果重复当测量。Stage2踝点高分只说明踝点观测较可靠，**不证明足底接触或世界静止**；脚静止约束要有明确协议/额外证据，缺失时关闭。
- 视觉失效时可输出显式 `predicted` 或 `stage_only` 降级，但不能将其标成视觉测量成功；不用提高先验权重掩盖失效。视觉与Stage冲突时保留残差及拒绝理由。`supported_stop_fixed_lag`的四脚同平地条件仍是用户确认的采集先验，不能记录成自动触地检测。

未来概率/方差校准需要独立留出参考误差；当前无独立世界轨迹真值，只能做工程可靠度分层和留出重投影检查，不能宣布概率校准完成。输出保留 `visual_only`、`stage_only`、`fused_soft` 的估计来源，以及独立的 `tracked/lost/relocalized/predicted` 状态和 `world_pose_certified=false`。

### 5. 实施顺序与停止门

1. 第一个可检查阶段只打通独立视觉后端适配与同源重放导出：原鱼眼、正式标定、完整帧序/失败状态进入，输出 `visual_rig_pose.jsonl`。不先改SMPL-H或叠加新的大人体网络；`static_background_stereo_frontend.py`、`static_background_vo.py`及既有scene replay可复用输入和诊断，不用另建一套场景系统。
2. 优先ORB-SLAM3 fisheye stereo建立基线，cuVSLAM作为GPU速度替代；后者先解决Ubuntu/WSL2依赖、有效视野、同步问题，再计时。Pi3X小窗口或RayMap3R可独立作为替代后端/前端评估，**不要求经典SLAM先通过**；不同候选同样遵守尺度/观测/失败门。
3. 初次场景比较固定 `G20261013_reliability448_view_v1` 的人体拟合及其 `plan.json` 中原始448双目对，0..447全保留；360序列可另作迁移检查，不能混为同一次对照。比较当前Stage路线、视觉路线关闭Stage、同一视觉路线加Stage软约束。第二、三组只改变Stage因子启用与否，第一、二组只比较场景位姿来源；无须重新拟合身体。
4. 在运行前固定mask/几何门和误差/覆盖/预算阈值；以原始图像中静态支持、留出背景重投影、双目rig一致性、连续覆盖和丢失/重定位分段为工程检查。刚体复合恒等式只检查坐标实现，轨迹更平滑只反映平滑性；均不能作为真实精度证据。不能宣称“双目就不会漂移”。本次尚未设定或通过数值升级门，所有候选默认不可注入正式scene writer；未来通过预先声明的工程门后才讨论 `metric_pose_usable=true`，物理认证仍为false。
5. 成功标准首先是Stage关闭后仍有来自静态背景的米制运动观测，且在相同数据上改善事先约定的可用性/留出残差并满足处理预算；不是全片都有非空位姿。没有独立相机/rig轨迹，只能给工程一致性结论。若因纹理、遮挡、同步或安装关系持续失败，停止增大模型，先处理对应观测缺口；保留失败片段，不通过插值、Stage硬填补或显示平滑冒充视觉成功。

### 6. 调研依据

- ORB-SLAM3 官方说明与鱼眼 stereo 示例：https://github.com/UZ-SLAMLab/ORB_SLAM3
- NVIDIA cuVSLAM 论文、性能表和 PyCuVSLAM 文档：https://arxiv.org/html/2506.04359v3 、https://nvidia-isaac.github.io/cuVSLAM/python/api.html
- Pi3/Pi3X 官方代码与容量记录：https://github.com/yyfz/Pi3 、https://raw.githubusercontent.com/yyfz/Pi3/main/benchmark_pi3x.json
- RayMap3R 论文与代码：https://arxiv.org/html/2603.20588v1 、https://github.com/Brack-Wang/raymap3r
- MASt3R-SLAM、Fast3R 和 BLASt3R：https://arxiv.org/html/2412.12392v1 、https://github.com/facebookresearch/fast3r 、https://github.com/naver/blast3r
- OnlineHMR论文与实现：https://arxiv.org/html/2603.17355v1 、https://github.com/Tsukasane/Video-OnlineHMR
- JOSH论文/项目与实现：https://vail.cs.ucla.edu/JOSH/ 、https://github.com/genforce/JOSH

## 2026-10-14 ORB-SLAM3 / cuVSLAM 同源448帧实施与比较

本节接续上面的设计及此前被中断的部署，已实际实现、运行两个独立视觉后端并完成共同诊断。结果未通过升级门，仍不修改mainline或人体拟合。Stage完全未进入两个后端，未用脚锚、旧场景轨迹、插值或Stage回填制造视觉成功。

### 输入、实现与运行边界

固定G20261013_reliability448_view_v1的448对原始图像与fit/result.npz。当前比较在G20261014_visual_rig_compare448_v1，protocol.json保存变量、帧范围和原定门：带原生几何支持的位姿覆盖至少90%，tracking调用P95不超过33.333ms；物理精度仍需独立轨迹。共同input_primary/manifest.json由prepare_visual_rig_comparison.py生成：逆转正的raw fisheye、960×540、按像素中心缩放内参、完整双目R及米制t、人体关键点凸包/肢体padding、参考助步器杆件投影与共同前向半球域。两后端收到相同图像与排除mask；不能因此预设它们内部都完全遵守mask。参考助步器安装和mask仅经视觉核对，未物理认证。

本次manifest仍使用名义30FPS时间，保留左右主机返回差，不能从这些秒数认证曝光同步或人体物理加速度。当前实际主机返回时间跨度16.035239秒，另见身体算法第33/34节。首轮固定栅格留出曾在4帧得零合格双目诊断点，因此保留input但主比较不使用它；primary诊断是同源背景一致性，不是独立留出或世界精度。

ORB部署在Ubuntu-22.04 WSL2，库/runner位于/home/itachi/walker_orb_build/ORB_SLAM3，Pangolin库位于同级pangolin_install/lib；使用项目third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt。源码入口walker_tools/scene/orbslam3_runner.cpp和orbslam3_static_mask.patch。补丁在原生特征提取阶段排除mask及描述子支持，修正右鱼眼读取左畸变的错误；没有改stereo初始化门。依赖补丁对官方clean index检查通过，完整目标已编译。

cuVSLAM使用隔离WSL环境中的17.0.1+cu12，入口walker_tools/scene/cuvslam_backend.py；完整R/t构造以左光心为rig原点，右rig_from_camera为标定逆变换。同步SLAM开启，原生odom pose与map pose分开保存。SDK请求Precision不等于实际primary名单；当前同配置构造检查get_primary_cameras返回[0]，代码记录实际值，不把配置文本当事实。

### 实际结果与结论

| 指标 | ORB-SLAM3最终v4 | cuVSLAM已完成primary |
|---|---:|---:|
| 保留帧数 | 448 | 448 |
| 原生位姿且有非空导出3D支持 | 0（0%） | 373（83.26%） |
| tracking调用中位/P95，ms | 13.77 / 17.48（均未初始化） | 3.80 / 21.99 |
| 完整replay，s | 88.35（另初始化11.55） | 100.57（另初始化0.982） |
| 覆盖门通过 | 否 | 否 |
| 可用于世界速度/加速度或主线升级 | 否 | 否 |

ORB全448为not_initialized，输入缺失0、mask中心违规0、最终地图segment始终0。左右特征数最大49/168，同帧总特征最大217，低于原生StereoInitialization要求N>500；不能据空位姿称通过，也不能用未初始化的调用时间宣称成功定位实时。保留原配置缺overlappingBegin失败、v2在frame51的空描述子崩溃、v3时间溢出中止。已修复空侧rowRange/vconcat处理；runner将timestamp_ns整型token在内存转为等值JSON实数，避免OpenCV32位读入溢出。frame51针对性检查及v4完整448时间逐项匹配通过，无虚假时间倒退/reset。

cuVSLAM所有448帧都有native位姿proposal；包装层将13帧初始化和62帧无导出3D支持列为不可用，所以373不是原生明确报告的“可靠米制成功”。get_last_landmarks只导出last-frame的tracks3d，373帧的中位/P95/最大点数1/3/6不是整张地图或求解器内点数。保守输出分为6段（最长241帧），不等于确认native地图reset。连续367边相邻平移P95约0.384m、最大19.476m；300边平移数值不变，不能由此区分真实静止、保持或预测。SDK未提供可靠视觉更新/预测标志，metric_pose_usable保持false。CPU读图和导出诊断也占时，完整帧中位/P95约180.12/536.75ms，不能将3.80ms调用等同端到端速度。

共同background_diagnostics共30个可投影点，cu残差中位0.767px/P9584.64px；没有任何边有至少4个诊断点，ORB无可用边，所以没有可供排名的充分共同残差或独立精度。不能用较小中位数盖住样本不足和高尾部。

实际mask诊断还发现cu全448导出右目89个观察中76个深入排除区>3px，左目无>3px深层违规；这些只是native导出对应，不能据此确认它们全部被最终求解器作为内点。但35个右目对应且有finite3D的点中，10个右鱼眼回投误差<=2px，其中8个仍深入右mask；原始像素坐标解释与版本源码相符，不能将该现象归为坐标误用。右目排除契约尚未完全实现，具体内部原因未确定。现有原始结果全部保留，没有黑填图、删点或重写轨迹伪造成功。

last landmark的Python文档称world，17.0.1源码与584点回投却显示为当前rig坐标（直接回投中位0.266px；误当world逆pose中位26.32px）。后端没有将该字段当world消费，新增明确语义说明。原生ODOM/map坐标不是初始地面认证坐标；地图重置/优化或包装层断段不能跨段求导。

### 保存结果与下一阶段

最终ORB在orbslam3_primary_v4/visual_rig_pose.jsonl与runner_summary.json，cu在cuvslam_primary.jsonl与metadata；comparison_v1/comparison.json保存共同统计，background_diagnostics.jsonl保存全帧诊断。比较器按同一native变换导出真实冻结人体mesh/COCO、刚体助步器和两个光心到*_world_geometry.npz，失败帧NaN、段标识保留；字段metric_pose_usable=false/world_pose_certified=false，不作为正式世界输出。

当前可复现运行入口是prepare_visual_rig_comparison.py、orbslam3_runner、cuvslam_backend.py和compare_visual_rig_poses.py，实际路径/参数继续从protocol、manifest、settings和metadata读取。8项针对性单测（标定方向、单位、同源网格/对象转换、失败保留、支持不重复计数、深层mask违规）与4个Python入口Ruff E9/F通过；ORB真实编译、边界帧及完整448运行通过。不增加无关全流程验证。

结论是完成两条工程适配与比较，独立视觉世界轨迹尚未达标。下一修复应围绕已定位的静态背景有效特征/双目支持不足及cu右目匹配排除实效，并先用小范围同源输入证明修复，再决定是否全片重跑；不盲增模型、放宽门或把Stage硬填补称视觉成功。Stage辅助融合未实施，因为本轮没有通过的独立视觉观测可供主线融合。人体相机相对运动状态可独立继续研究；世界速度/加速度暂不可用，具体实现意见见身体算法第34节。
