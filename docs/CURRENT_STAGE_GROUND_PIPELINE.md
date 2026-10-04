# 当前两阶段与动态地面主线

## 2026-10-04 正式工程主线决定

当前正式工程路线固定为 **PMPose 原始鱼眼观测 → 严格双目几何 → 原 Stage 1/2 识别 → 完整 SE(3) 动态地面基线 → 同源 SMPL-H 冷启动身体及 WiLoR 原生局部手姿**。选择清单位于 `realtime_app/configs/current_mainline.json` 的 `mainline_selection`，场景调用使用 `--scene-profile mainline`，实际解析为 `baseline`。这是一项可复现版本决定，不是物理接触、独立三维精度或整套实时部署认证。

本次场景数据为换机位 360 帧（0..359，原始配对1157..1516）；身体使用 `G20261003_repositioned360_manual_v1/pmpose/fit/result.npz`。该结果**未启用人工腕软参考**，身体仍为工程候选。人工标注目录名称不能用来推断损失是否启用。448 帧 Sapiens2 身体候选与 PMPose 同流程对照保持原等级和参数；配置顶层的 `frames/body_candidate/frozen_scalar_options` 仍专用于这组冻结对照，不代表360帧运行参数。

`landing_fixed_lag` 与 `joint_stage_window` 保留为显式离线候选：运动指标有部分改善，但落地闭合未通过，联合窗口还未满足耗时预算。只将严格点/掩码导出和最终发布流导出修复纳入共用入口。原基线的硬恢复跳变、缺失测量及静态扶手参考局限继续记录，不用候选或显示平滑掩盖。

详细数据流、参数、命令及停止条件见 `docs/CURRENT_STAGE_GROUND_PIPELINE.md`；身体内核及448帧历史实现见 `docs/CURRENT_SMPLH_ALGORITHM_20261002.md`。后文较早的“当前”判断按其数据集和日期理解；最新选择以本节、`AI_PROGRESS.md` 和选择清单为准。


当前入口、等级和复现参数以本节及 `realtime_app/configs/current_mainline.json` 的 `scene_route` 为准。`scene_rotation_fix_v1` 是冻结工程基线；浏览器中的 `stage_landing_recovery_v1/view_final` 是落地恢复候选的历史展示，不能因展示更新自动升级为主线。下面较早的采集数量保留为历史证据。



### 正式版本的处理顺序与接口边界

1. **输入与观测**：物理相机身份仍由instance_id注册表解析；cam0=LEFT、cam1=RIGHT。已有检测输入可转正，但进入几何前逆旋转回原始鱼眼像素。人物关联和逐点几何质量门不改变，最多匹配一对人物；失败帧、拒绝原因及NaN保持可见。
2. **身体分支**：按该数据集保存的fit_command重新构造观测，在相机坐标系求SMPL-H。A30步只优化逐帧transl与VPoser latent，root冻结；B30步只优化全片同一个共享beta；C300步优化共享beta以及逐帧root、transl、latent；D1/D2各100步只优化共享lhand/rhand PCA，身体和beta冻结；D3为0。准确可训练集合来自源码stage plan，不把预算名称当变量集合。WiLoR原生MANO局部旋转以0.1权重提供手指先验，不使用它的global_orient/transl/beta作为人体或腕的绝对位置。360帧人工腕、身体二维重投影、身体时序和表面接触项未启用。冻结结果用于场景对照，不在此阶段重新求解身体。
3. **场景严格点分支**：地面入口实际消费360帧strict JSONL的3031个接受点。身体拟合保存的观测和场景严格点是两个明确的数据产品；源命令可能独立重三角化，不能仅因来自同一检测器就认为接受掩码相同。scene NPZ中的有效XYZ/mask必须与strict文件逐项相同；detector_only重算仅为诊断，不能补充脚锚或填失败点。
4. **Stage识别**：原背景/人体运动证据经原TwoStageRecognizer产生Stage状态和不可用原因。Stage1表达助步器/相机静止、人体可动；Stage2表达人体脚锚静止、助步器/相机可动。它们是工程动作协议，不是逐帧独立真值；保留原状态确认与活动carry，不引入候选交接状态机。
5. **地面位姿**：使用当前原始图像估计被质量门接受的背景旋转；Stage2按严格脚点和因果中值观测估计完整SE(3)地面增量。Stage1仍使用原支撑恢复，缺测保留状态及原因。姿态定义为相机点映射到地面系的T_G<-C_t；场景显示和指标共同消费最终关闭/发布的位姿流。原硬恢复的跳变是已知缺陷，不能凭主线选择消除。
6. **人体、地面、助步器共用变换**：身体相机系网格乘当前T_G<-C_t；地面来自当前位姿；相机刚性连接的助步器使用同一变换及既有结构参考。没有单独抬降人体、脚或助步器的显示补偿。静态助步器安装尚未在本次布置重新认证，只能作为结构/显示参考。
7. **审计与发布**：保留360帧编号、实际配对时间、严格点、阶段、R/t及原始拒绝原因。时间是host-return，不能声称曝光同步。结果生成后核对JSONL/NPZ发布一致性、参数重建和真实模型面。离线整段拟合、实时Stage/地面和页面播放是三个不同接口，不能互相替代耗时验证。

### 复现版本、资源与验证入口

`current_mainline.json/mainline_selection`保存当前选择，`scene_route/default_profile`是场景别名解析的唯一来源；`stage_ground_profiles.json/baseline/options`是场景算法开关来源。未传scene-profile时保留历史调用行为；显式mainline时固定baseline开关，冲突参数直接报错。输出同时保存请求的scene_profile和resolved_scene_profile。别名不会改身体manifest中的448帧冻结参数。

360帧准确身体命令：`G20261003_repositioned360_manual_v1/pmpose/fit_command.json`。准确场景命令：同目录`stage_landing_recovery_v1/mainline_alias_command.json`。这些路径均位于`research_records/engineering_validation`下。命令列表记录实际输入和输出；复跑必须将output/output-dir换到不存在的新目录，不能覆盖对照。图像、标定、测量地面、WiLoR缓存及授权SMPL-H/VPoser资源均需本地恢复；缺失时报告不可用，不从旧动态场景或其他采集填补。

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

当前换机位360帧默认展示仍为`G20261003_repositioned360_manual_v1/full_view_rotation_fix_v1/body.html`。新实验不替换原`RealtimeDynamicGroundWriter`默认参数；方法级别仍是工程候选。

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
