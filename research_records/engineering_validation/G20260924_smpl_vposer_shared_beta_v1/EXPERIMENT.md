# G20260924_smpl_vposer_shared_beta_v1

## Stage D 学习率规则试验（2026-09-25）

固定开发窗 `60..90`、foot=2.0、hand=30.0。基本收敛停止标准：最后 40 步目标相对下降 ≤0.1%，且参数步长 L2 中位数 ≤0.001。

比较结果：固定 LR 320 步下降 1.624%；指数衰减到 10% 下降 0.255%；余弦到 10% 下降 0.205%；余弦到 1% 下降 0.0431%，参数步长中位数 0.000464，达到当前小窗的 `basic_converged` 标准。

当前推荐规则：Stage D 使用 `CosineAnnealingLR`，`T_max=320`，末步学习率为初始值的 1%；latent/root 从 0.001 衰减到 1e-5，translation 从 0.0005 衰减到 5e-6。该规则用于减少运行时间，不代表数学收敛或全片验证；完整 448 帧尚未运行。

## foot=2、hand=30 的 Stage D 基本收敛诊断（2026-09-25）

固定窗口 `60..90`、A/B/C=120/120/120，Stage D 分别运行 80、320、800 步。暂定工程停止标准为：最后 40 步总目标相对下降不超过 0.1%，且参数步长中位数接近零。

- 80 步：60→80 步下降 0.925%，未达标准；
- 320 步：280→320 步下降 1.624%，未达标准；
- 800 步：720→760 步下降 0.957%，760→800 步下降 0.912%，参数步长中位数约 0.0197，仍未达标准；
- 800 步 3D P95 约 121.04 mm，较 320 步约 131.05 mm 继续变化，说明不能把 80 步当作稳定终点。

本诊断只用于确定小窗优化预算，不是数学收敛证明，也未运行完整 448 帧。机器可读汇总为 `surface_contact_convergence_foot_a2_hand_a30_v1.json`。

## 小窗加权与 Stage D 收敛诊断（2026-09-25）

固定 `60..90` 开发窗和 A/B/C=120/120/120，只测试 D=80 步的三组权重：`both_a1_a30`（foot=1, hand=30）、`both_a2_a30`（foot=2, hand=30）、`both_a1_a45`（foot=1, hand=45）。新增 `--stage-d-trace` 仅记录每步损失，不改变优化更新。

- 三组退出码均为 0，Stage C 参数逐组一致；beta 精确冻结；同进程续跑、surface graph、COCO 梯度隔离和有限性检查通过。
- `both_a2_a30` 脚 residual 更强，但 3D P95 比 `both_a1_a30` 增加约 1.39%；`both_a1_a45` 手 residual 更强，3D P95 增加约 0.21%。
- 三组总损失在第 60 到 80 步仍下降约 0.892%/0.925%/0.934%，最后 10 步没有上升，末步变化仍约 7e-5；80 步未进入平台，不能写成已收敛。
- 机器可读结果：`surface_contact_weight_probe_60_90_v1.json`；逐步记录位于三个 `surface_contact_window60_90_weight_probe_*` 目录的 `stage_d_trace.json`。

本试验只说明更大权重在小窗上能产生额外工程 residual 修正及其观测代价，不冻结更大权重，也不替代完整 448 帧运行或物理接触验证。

## 当前可复现路线（2026-09-25）

本实验当前采用“先全局拟合，再表面接触微调”的顺序。

### 全局拟合阶段

完整运行以 448 帧为拟合范围。每帧独立优化 VPoser latent、global orientation 和 translation；全片共享一组 beta。先执行：

1. Stage A：beta 固定为零，优化逐帧运动参数；
2. Stage B：冻结逐帧运动，只优化共享 beta；
3. Stage C：联合微调逐帧运动和共享 beta，仍不加入手脚接触项。

全局拟合使用当前白名单输入，从零初始化，不读取旧 fitted parameters、旧 temporal prefit、旧 contact target 或旧 HTML。这里的“全局”指全片共享 beta 和统一 448 帧范围，不包含额外 temporal smoothing。

### 表面接触阶段

Stage C 完成后，必须在同一进程中继续 Stage D：

- beta 逐元素冻结；
- 只优化逐帧 latent、global orientation 和 translation；
- 加入 foot surface loss，系数 1.0；
- 加入 hand surface loss，系数 30.0；
- Stage D 观测缩放为 3D=0.70、2D=0.10；
- 接触优化步数为 80；
- foot 只对非零 foot_contact_weight 的有效帧/侧计算，零权重帧保留为 unavailable；
- hand 使用全帧双侧 surface loss，不用 hand_contact_weight 关闭；
- 保留 penetration penalty，用于修正系统性地面以下表面误差。

### 当前固定配置

```text
surface_foot_contact_weight=1.0
surface_hand_contact_weight=30.0
obs_3d_weight=1.0
obs_2d_weight=0.20
stage_d_obs_3d_scale=0.70
stage_d_obs_2d_scale=0.10
contact_steps=80
device=cpu
```

### control 对照

control 与接触路线必须使用相同的全局输入、初始化和 Stage A/B/C；唯一差异是 Stage D 的 surface foot/hand 权重。control 使用 0/0，接触路线使用 1.0/30.0。禁止把 Stage C 结果复制为初始化，禁止把接触项提前加入 A/B/C。

### 当前证据边界

开发窗 60..90 已确认 both_a1_a30 的图构建、梯度、beta 冻结、有限性和数量级修正；该路线可作为工程主线继续使用，但不是最优权重、严格独立窗口冻结或物理接触验证。现有 `full448_formal` 是无接触三阶段全片结果；完整 448 帧 both_a1_a30 运行仍待执行。

本实验验证单帧 VPoser latent 拟合和全片共享 beta 的三阶段流程。每一帧独立优化 32 维 VPoser latent、global orientation 和 translation；不启用帧间时序项。beta 在全片共享。

输入白名单：原始左右 PMPose JSON、双鱼眼标定、官方 male SMPL 6890 模型、17×6890 COCO observation regressor、官方 VPoser V02_05 权重。脚本内部重新执行鱼眼三角化，不读取旧三角化、旧 temporal prefit、旧 beta、旧 contact 或旧 HTML。

阶段 A：beta=0，优化每帧 latent、root orientation、translation。

阶段 B：冻结阶段 A 的每帧运动，只优化一组共享 beta，范围 [-1.5, 1.5]，强 L2 正则。

阶段 C：联合微调，beta 学习率低于 latent/root/translation。

模型侧监督链固定为：male SMPL 6890 vertices → COCO observation regressor → predicted COCO-17。右膝没有 accepted 监督时权重为零。

当前实验不包含 temporal、contact、ground、walker 或可视化结论。beta 只能在二维、三维、逐关节和边界检查后评价。

## 正式运行结果

入口：`fit_vposer_shared_beta.py`；输出：`full448_formal/`。448 帧均使用独立 latent、root 和 translation，beta 全片共享。

- 阶段 A（beta=0）：三维中位/P95 `115.92/204.67 mm`，二维中位/P95 `48.76/153.18 px`。
- 阶段 B（仅共享 beta）：beta 未撞边界；三维中位/P95 `115.41/204.36 mm`，二维中位/P95 `48.41/152.08 px`。
- 阶段 C（低学习率联合微调）：beta=`[0.0944,-0.0890,0.0894,-0.0951,0.0930,-0.0820,-0.0685,0.1048,0.0953,-0.0940]`；三维中位/P95 `105.70/185.58 mm`，二维中位/P95 `41.92/139.09 px`。
- 右膝 accepted 数为 `0`，未进入监督。
- 固定阶段 A 运动做分段 beta 复核后，前/中/后三段 beta 均约 `±0.079`，未撞边界，说明该候选在本数据内具有跨段稳定性；这仍可能反映共享的观测/模型系统误差，不能称真实人体体型。

当前结论：三阶段代码和共享 beta 机制通过工程运行检查；姿态单帧结果仍需逐帧视觉和逐关节门，beta 暂作为候选，不进入接触或最终主线冻结。

参考样式页面已按 `VISUALIZATION_STYLE.md` 重建为 `full448_formal/stage_c_reference_style.html`。该页面只绑定当前 Stage C 的网格、模型 COCO 点、本次三角化点和本次鱼眼重投影指标；地面、助步器和动态相机关系标记为不可用。

## 显示与拟合逻辑审计（2026-09-24）

用户反馈“人体躺在地面且没有助步器”后，对结果和页面分别审计：

1. 当前 `result.npz` 的 `vertices`、`predicted_coco`、`transl` 是左相机坐标系；首帧模型骨盆到肩中心方向为约 `[0.534,-0.095,-0.030] m`。这是相机坐标方向，不是固定地面系的竖直方向。
2. 原页面额外绘制了相机坐标 `y=0` 网格，造成“地面”错觉。本实验并没有从本次原始视频重放得到的 `T_G<-C_t`、地面平面或助步器动态关系，故不能在本页显示地面/助步器，也不能读取旧结果补齐。
3. 该显示问题不能掩盖拟合本身的失败证据：Stage C 三维 P95 为 `185.58 mm`，左踝 P95 为 `244.60 mm`，左膝 P95 为 `165.95 mm`。因此本结果仍未通过单帧姿态视觉门。
4. `build_reference_style_viewer.py` 已改为只显示左相机坐标轴，移除误导性地面网格，并重新生成 `full448_formal/stage_c_reference_style.html`。后续只有在当前运行补齐地面—相机变换后，才能使用固定地面坐标和实体助步器进行物理语义可视化。

## 当前运行 Stage 1/2 与固定地面整合（2026-09-24）

新增模块位于 `pipeline/scene/`，入口为 `replay_current_run.py`。它逐帧读取原始左右视频和原始 PMPose JSON，在本次运行中调用既有 `RealtimeStageWalkerWriter` 完成背景运动/人体运动判别，再调用既有 `RealtimeDynamicGroundWriter` 完成 Stage 2 的全 SE(3) 地面变换。固定地面参考只使用已完成离线标定的 `ground_reference.json`；助步器只使用已记录的静态粗实体拓扑和初始安装位姿，逐帧助步器位姿由当前 `T_G<-C_t` 重新计算。

本次重放输出：`scene_stage_ground_v1/`。

- 448 帧逐帧重放；视频原始帧 484 帧，按当前 PMPose 的 448 帧白名单取前 448 帧并记录在 `scene_sources.json`，没有读取旧 Stage JSONL。
- Stage 1=`254`，Stage 2=`71`，Transition=`118`，Warmup=`5`。
- 当前动态地面更新 `61` 次；状态包括 `accepted=38`、`accepted_rotation_held=23`、`landed_support_reanchored=8`，另有明确 `rejected=38` 和 `held=341`，未静默删除失败帧。
- 固定地面系首帧 SMPL 网格包围盒约为 `x[-0.250,0.431] m, y[-0.312,0.409] m, z[-0.102,1.666] m`；模型 COCO 脚点的地面高度中位数约 `0.033 m`，说明坐标变换已经进入地面语义，但网格仍存在拟合残差造成的局部低于地面现象。

新页面：`full448_formal/stage_c_grounded_stage12_viewer.html`。页面的 SMPL、模型 COCO 点、三角化点、Stage、动态地面变换、固定地面和实体助步器均绑定当前实验输出；无显示平滑，接触仍未作为拟合监督。当前 Stage C 本身三维 P95=`185.58 mm`，所以这个页面是坐标/场景整合诊断，不是基础拟合通过证明。

页面同时导出 `full448_formal/result_grounded.npz`，保存固定地面系下的 `vertices`、`predicted_coco`、`raw_triangulated_points`、`walker_nodes`、`rotation_ground_from_left`、`translation_ground_from_left_mm`、`accepted_mask` 和 Stage 标签。

## XOY 地面坐标可视化修正（2026-09-24）

入口仍为 `pipeline/scene/build_grounded_viewer.py`，新输出为 `full448_formal/stage_c_grounded_xoy_viewer.html`。渲染直接使用固定地面坐标 `[X,Y,Z]`：地面为 `Z=0` 的 XOY 平面，`Z` 向上；SMPL 网格、COCO 点、三角化点和助步器不再经过旧 viewer 的坐标轴交换。页面加入 XOY 网格、XYZ 坐标轴和当前地面变换平移形成的绿色相机轨迹。

当前运行的 SMPL 骨盆从首帧 `[0.083,0.090,0.780] m` 到末帧 `[0.068,0.320,0.779] m`，沿地面 `+Y` 方向约 `0.230 m`；助步器由每帧当前 `T_G<-C_t` 重新转换，质心高度范围约 `0.403--0.514 m`。这些是场景显示审计值，不是对拟合质量或真实物理接触的通过结论。

## Stage 重放输入审计与最终页面（2026-09-24）

前一版场景重放错误地把采集视频的前448帧与 PMPose 448行直接配对。PMPose 的实际图像源是 `V20260908.../people_1/input_448pairs/left_ccw90` 和 `right_cw90` 中的有序 `pair_0000` 到 `pair_0447`；采集视频的连续帧不是该序列。该错位会改变 KLT 背景轨迹、Stage 2旋转和脚锚平移，因此前一版 Stage 计数和助步器移动不可信。

修正后的入口仍是 `pipeline/scene/replay_current_run.py`，增加 `--input-pair-dir` 后使用同源原始双目图像，逆旋转到原始鱼眼像素，再重新三角化质量门、Stage 1/2和 `T_G<-C_t`。三角化监督使用 `q2D=sqrt(cL*cR)>=0.25`、双目正深度和平均重投影误差不超过10 px；没有把最终连续权重 q 当作检测置信度。

输出目录：`scene_stage_ground_v3_pair_replay/`。结果为 Stage 1/Stage 2/Transition/Warmup=`214/115/114/5`，动态地面接受更新=`123`。与历史结果逐帧平移的差异中位数约0.22 mm、P95约1.42 mm，说明当前重放已恢复旧主线的几何逻辑；历史文件仅用于审计比较，没有作为当前输入。

最终页面为 `full448_formal/stage_c_grounded_xoy_viewer_final.html`，助步器使用 `coarse_complete_model_v2_camera_rail/coarse_walker_model.json`：前横杆673.046 mm、两侧横杆442.854 mm、扶手838.389 mm。页面直接绘制固定地面系 `[X,Y,Z]`，地面为 XOY、Z向上，包含SMPL表面、实体助步器、双相机简化标记和绿色相机轨迹。

页面补充了侧面 `0--2.0 m` 高度标尺，并把 OrbitControls 设为无阻尼自由控制，支持旋转、平移和缩放。双目左右光心均通过双目标定外参和当前 `T_G<-C_t` 逐帧计算。可视化规范已写入仓库根目录 `VISUALIZATION_PIPELINE.md`，以后页面修改必须遵守该文件。

同一当前运行场景还生成了视频：`visual_ground_v3_pair_replay/raw_visual_human_partial_handles_ground.mp4`。视频为448帧、30 FPS、960×720，使用当前重放动态地面、当前导出的严格/显示三角化点和 camera-rail 助步器，保持实线和无显示平滑。

浏览器实测发现页面未显示SMPL表面的根因是面索引序列化：`faces` 是13776×3二维列表，旧代码执行 `new Uint32Array(d.faces)` 只产生13776个无效索引。现改为 `new Uint32Array(d.faces.flat())`，得到41328个索引；代表帧140截图中能看见完整着色网格。浏览器自动检查还确认时间轴切到Stage2第141/448帧、自由拖拽使相机位姿变化、Z轴为视角向上方向、无脚本错误。页面有侧面0--2m标尺和双相机光心/细基线。

最终固定地面系SMPL骨盆的Y坐标首尾约为`0.090→-1.152 m`，行进约1.24m。先前基于错误图像配对报告的约0.23m无效。当前SMPL拟合三维P95仍约185.58mm，页面和视频都属于场景可视化诊断，不能作为基础拟合通过或真实接触结论。

页面依据用户给出的墙角示例再次修正：两面竖直墙分别位于`X=0`和`Y=0`，与`Z=0`地面相交，墙角交线为XYZ坐标系原点。SMPL表面使用透明度`0.46`、关闭深度写入，骨架线、COCO点和三角化点使用更高绘制顺序；浏览器代表帧截图已确认表面和骨架同时可见。该规则已同步写入根目录`VISUALIZATION_PIPELINE.md`。

## 显示坐标旋转修正（2026-09-24）

用户复核指出旧页面把运行方向放在错误的显示轴上。此次不改动任何拟合参数或固定地面数据，只统一页面显示变换：保存坐标仍为 `[X_ground,Y_ground,Z_ground]`，页面使用 `X_display=X_ground`、`Y_display=Z_ground`（竖直）、`Z_display=-Y_ground`（人体运行方向为正）。因此地面为 `Y_display=0` 的 XZ 平面，竖直墙为 `X_display=0` 和 `Z_display=0`，三面交界处为唯一原点。SMPL网格、模型COCO点、三角化点、助步器、双目光心和相机轨迹全部通过同一个 `gv()` 映射。旧的旁侧独立高度/Z标尺已移除，实际长度 `AxesHelper(2.0)` 从墙角原点出发。`camera.up`同步改为显示Y轴；浏览器验证无脚本错误、SMPL三角面13776、透明度0.46。

随后发现页面中的旧墙面辅助 `GridHelper` 和旋转错误的 `wallY` 实际形成了 `Y=1.2 m` 的水平假平面，视觉上像地面穿过颈部。该问题只在渲染层，固定地面数据脚点高度仍接近 `Z_ground=0`。已移除两个墙面 GridHelper 和错误水平 wallY，仅保留 `Y=0` 的真实地面、`X=0` 与 `Z=0` 的竖直墙面。正视截图复核确认地面回到脚部高度，页面无脚本错误。

## 地面-only 坐标轴简化（2026-09-24）

按用户要求进一步删除所有沿 Z 方向的墙面/参考平面。页面显示改回固定地面语义：`X_display=X_ground`、`Y_display=-Y_ground`（与人体运行方向平行且正方向朝前）、`Z_display=Z_ground`（竖直）；唯一平面为加深颜色的 `Z=0` XY 地面。地面原点绘制实际长度 X/Y 坐标轴，刻度间隔 `0.5 m`，显示 `0.0--2.0 m` 数值；不再绘制墙角墙面或独立 Z 标尺。SMPL表面、骨架、助步器、相机和轨迹仍使用同一个显示变换。浏览器截图 `full448_formal/stage_c_ground_axis_qa.png` 无脚本错误，地面位于脚部高度。

## 交互与真实帧率修正（2026-09-24）

OrbitControls 改为自然的自由拖拽：左键旋转、右键平移、中键缩放，开启 `dampingFactor=0.08` 的轻微阻尼，启用屏幕空间平移，并将方位角/极角范围设为无限制。播放计时改为 `1000/(30*speed)` ms/帧，使 `1×=30 FPS`、`0.5×=15 FPS`、`2×=60 FPS`，不对几何数据插值。

## 接触损失设计（2026-09-24，尚未接入拟合）

已完成当前运行数据上的脚/手接触逻辑设计，细节见 `CONTACT_LOSS_DESIGN.md`。Stage 2 左右脚均使用当前运行的 accepted 三角化踝点和固定地面平面；Stage 1 左右脚独立按三帧局部速度、脚高和逐点置信度生成连续支撑权重，只对支撑权重计算地面距离损失，避免宽区域吞掉摆动帧。手接触改为当前动态助步器扶手线段的点到圆柱表面距离，不读取旧 interaction/contact 目标。该阶段只完成设计和输入审计，没有修改拟合参数或运行接触优化；下一步必须先生成 contact_labels 审计，再执行无接触→脚→手的门控对照。

## 当前运行接触标签审计（2026-09-25）

已用 `pipeline/contact/build_contact_labels.py` 从当前三角化、Stage、动态地面和静态实体拓扑重新生成标签。输出为 `contact_labels_stage_audit_v2/`。Stage 2 同时满足左右脚有效候选的帧数为101/115；Stage 1 左脚 support/swing/ambiguous/invalid=`8/156/47/3`，右脚=`18/163/30/3`。当前动态扶手距离中位约77--81 mm，按当前质量门手接触候选为0帧，因此手接触不能进入拟合。首轮 `v1` 因标签字符串截断已标记 `invalid_contaminated_run`，未被使用。当前仍未运行 SMPL 接触优化；下一步只做脚-only 小窗口对照。
## 接触损失接入（2026-09-25）

本次只完成实现和静态检查，没有执行接触拟合。拟合器现在可选读取当前运行生成的 `contact_labels_stage_audit_v2/contact_labels.npz` 与 `scene_stage_ground_v3_pair_replay/scene_transforms.npz`。A/B/C 阶段仍为无接触初始化、共享 beta 和联合微调；只有 C 完成后才开启 D 阶段接触微调。

脚接触在地面坐标系中对模型侧 COCO 踝点 15/16 的 z 距离施加 pseudo-Huber，手接触对模型侧 COCO 腕点 9/10 到当前帧动态扶手线段的距离施加 pseudo-Huber。默认权重为零，且本次当前标签审计没有得到可靠手接触候选，因此未运行手接触。

该修改已通过 `py_compile`。下一阶段是 foot-only 小窗口对照，必须与无接触 Stage C 在二维、三维和时序指标上逐项比较；接触代理不能被称为真实触地或真实握力。


## v4 手部权重敏感性对照（2026-09-26，engineering validation）

输入：与 v3 完全相同的 raw PMPose、标定、male SMPL、COCO regressor、VPoser、contact labels、scene transforms、surface sets、静态拓扑；窗口 60..90（31 帧），cpu，base/beta/joint/contact 步数与 v3 一致。唯一变量为 Stage D 表面权重。新增 --force-stage-d-no-contact 开关实现同进程、同优化器、同 80 步、beta 冻结的零接触控制（surface_contact_window60_90_v4_stage_d_no_contact）；不复制 Stage C 结果，不读旧拟合初始化。

四条新路线：hand w0.03、hand w0.10、foot+hand w0.03、foot+hand w0.10（foot 固定 0.02），退出码均为 0。相对零接触控制：2D P95 差值 < 1e-4 px，三维 median 差值 < 4e-4 mm；hand residual median 随权重 0→0.10 仅变化约 1e-4 mm；hand coverage（15/30/50mm）五路完全一致；Stage D 参数变化量五路一致到 1e-6；beta 相对 v3 no-contact 漂移全 0.0；C→D 同进程、beta 精确冻结、COCO 隔离（grad_to_coco=0）全部通过，无 NaN/Inf。

结论边界：当前权重和梯度对拟合影响不足（手部项未产生可观测影响），不得宣布 hand contact 有效；下一步检查损失尺度和优化器，而非放大步数或改阈值。本结果仅为 engineering validation，不是物理握持/握力/承重/三维精度验证。

## v5：3D 主导观测与 Stage D 表面接触权重实验（engineering validation）

动机：旧目标 `l3 + 0.10*l2` 中 2D 像素项（Stage C 典型值约 444.21）数值上压过 3D 项（约 0.003475），不能证明 3D 主导；旧 soft-min 在 K=54/778 时有明显 log(K) 负偏移；手部缺少显式穿透惩罚；Stage D 未降低观测权重。

改动：`smpl_surface_contact.py` 新增 `centered_softmin`（保留旧 `softmin` 接口），脚/手损失改用中心化 soft-min 并加入 `penetration_weight=1.0` 的 relu 穿透惩罚；`fit_vposer_shared_beta.py` 新增 `--obs-3d-weight 1.0`、`--obs-2d-weight 0.25`、`--stage-d-obs-3d-scale 0.70`、`--stage-d-obs-2d-scale 0.10`，`losses()` 中 l3/l2 改为 delta 归一化无量纲形式（metrics 仍以 px/mm 记录），`run()` 显式接收观测系数，Stage A/B/C 用 1.0/0.25，Stage D 用 0.70/0.025 且保持非零观测约束；audit 新增观测系数与脚/手穿透比例。

实验：窗口 60..90，输入/步数与 v4 一致，共 7 条路线（v5_stage_d_no_contact 对照 + foot 0.05/0.10 + hand 0.05/0.10 + both 0.05/0.10），全部退出码 0。相对同一 v5 控制：3D P95 变化 ≤ 0.36%，2D P95 变化 ≤ 0.28 px；脚/手 residual median 增益均 < 1%（未达 5% 门）；穿透比例变化 ≤ 0.02 个百分点；beta 漂移全 0 且精确冻结；gradient audit 全通过；无 NaN/Inf。

结论：无推荐系数。当前接触几何或优化尺度仍未形成可观测收益，停止继续增大权重；不增加步数、不改阈值、不挑选最好看路线。本结论仅为 engineering validation，不是物理接触/握力/承重/真实三维精度验证。
## v5 表面接触几何与坐标审查（几何先于权重）

输入：v5 控制与 both_a010 的 Stage C/D 结果、当前 scene_transforms、surface sets、静态 walker 拓扑；仅只读复算，不重跑拟合，不改动任何 v1–v5 产物。

检查公式：`vg = R_ground_from_left @ v + T_ground_from_left_mm/1000`（逐帧转换到地面系）；脚 z_G 统计与 `fraction(z<0)/z<-10mm/z<-25mm`；扶手端点为 topology `nodes_left_camera_mm` 转换到地面，capsule 长度、palm 到中心线距离、`distance-radius` 残差、负残差比例；一套索引语义与左右侧质心核对；Stage C→D 顶点与 sole z 变化。

关键统计（control Stage C）：
- 脚 z 负比例：左 92.47%、右 95.52%（均 >90%）；脚 z median 左 −27.4 mm、右 −39.7 mm；z<-25mm 比例左 57.83%、右 71.27%。
- 扶手 capsule 长度固定约 0.30 m；端点高度约 0.84–0.88 m（合理）；手 residual median 左 36.2 mm、右 39.6 mm；手负残差比例 左 8.09%、右 6.11%。
- 集合：sole 各 54、palm 各 778，索引范围 [1981,6840] 在 [0,6889] 内，左右集合不相同；左右 sole/palm 质心 x 无异常交换。
- Stage C→D：control 顶点 max-abs 40.1 mm、L2 9.30；both_a010 顶点 max-abs 40.2 mm、L2 9.27（两者接近）。

判定：foot_geometry_status=blocked、hand_geometry_status=pass、coordinate_frame_status=pass、weight_tuning_allowed=false、next_action=geometry_fix。

结论与停止原因：脚部 >90% 候选点 z_G<0 且残差中位仍为负，说明这是当前地面变换与 SMPL 表面之间的工程一致性问题（不是真实触地失败），继续调脚权重没有意义，必须先修几何。手部 median 约 <40 mm 未达 suspect 门，但同样不增加手权重。下一步为 geometry fix（先核对地面 z=0 参考与 SMPL 足底模板偏移），之后再做 Stage C 起点梯度审查。
## v5 sole 映射审计（只读，未重跑拟合）

输入：v5 control Stage C/D 结果、当前 scene、surface sets、官方 male SMPL 模板；输出 `surface_contact_window60_90_v5_sole_mapping_audit.json`。

单位：vertices/coco 为 m（中位 |x| 约 0.28/0.30），raw triangulated 与 scene translation 为 mm（中位约 288/703），与入口 `:136`/`:187` 的 `/1000.0` 一致；`unit_status=pass`，无静默换单位。

变换闭环：同一刚体变换作用于模型与观测时差值恒等，forward/inverse 的模型—观测 3D 误差中位均为 75.37 mm（差 <1e-6），方向检验本身无信息量，记 `ambiguous`；forward 给出 ankle 47/73 mm、pelvis 809 mm 的合理高度，inverse 给出 ankle 约 1026–1176 mm（高于 pelvis），仅作辅助记录，不作为方向证据。

解剖：同侧检查在左相机帧完成（左右均为 True）；ground 系 z 差 median 左 −75.7 mm（零 pose −72.5 mm，漂移 3.2 mm），右 −113.0 mm（零 pose −77.3 mm，漂移 35.7 mm，未达 60 mm suspect 门，记 pass，但不对称性如实记录）；左右 sole 与最低 60 点重叠 15/45，均为同侧正确集合。

集合：heel/ball/toe 各 18，无重复，mirror 平均 |x| 间隙 0.009 m，`mirror_ok=true`；审计中修正了两处脚本 bug（解剖改用 ground 系、侧别改用相机帧），均只影响审计脚本，不涉及拟合。

结论：`single_supported_fix=none`，`fit_rerun_allowed=false`，`next_action=audit_gradient`；仍禁止调大接触权重。
## v5 Stage C 梯度审计（只诊断，不选权重）

sole 映射审计通过后进入 gradient audit：同一 Stage C 参数状态、单次 forward、四项（obs3d/obs2d/foot/hand）分别对 latent/root/transl 求梯度，beta 冻结，Stage D 未执行（审计目录仅含 A/B/C 结果与 gradient_audit.json）。

两次审计（surface 0.10/0.10 与全零对照，输入路径与 Stage A/B/C 完全一致）退出码均为 0；Stage C 观测项 term 值与梯度一致到 1e-6 以内，audit 分支未改变 Stage C。

Term 值：obs3d 0.21256、obs2d 0.28891、foot 0.001868、hand 0.0000954；梯度 L2（all）：obs3d 0.32739、obs2d 0.32769、foot 0.02092、hand 0.000545；surface/observation 梯度比 0.0328 → `surface_gradient_weak_but_active`，graph 为 `pass`（非零、有限）。

结论：表面图有效但弱，不选择权重；下一步 `calibrate_contact_coefficient`（先校准接触系数尺度，而非直接跑接触拟合）。本结论仅为 engineering validation。

## v6 接触系数校准对照（engineering validation）

原因：v5 梯度审计给出 obs3d/obs2d/foot/hand 有效梯度 L2 为 0.327/0.328/0.0209/0.000545；2D 系数 0.25 时 3D/2D 实际梯度几乎相等。降为 0.20 后预计 3D 约为 2D 的 1.25 倍。Stage D 保持 0.70/0.10，即 Stage D 实际 3D=0.70、2D=0.020。系数推导：foot=1/3 对应观测梯度约 3.2%/9.6%，hand=30/60 对应约 2.5%/5%。

实验：窗口 60..90，6 条路线（v6 control + foot 1/3 + hand 30 + both 1/30 + both 3/60），退出码均为 0，command.txt 已存档，未复制旧 Stage C。

结果（相对 v6 control）：foot_a1 通过 3D/2D/穿透/beta/梯度门但 residual 门未过；foot_a3 使脚 |median| 从 32.9 mm 降到 26.1 mm（穿透方向改善约 21%），但 3D P95 恶化 +3.3% 且门公式按 median 下降计为负增益；hand_a30 手 median 44.39→42.74 mm 但左手穿透比例 +1.8pp 超门；both_a3_a60 手增益 +6.6% 达 5% 线但 3D P95 +3.0%、左手穿透 +2.7pp。无路线六门全过。

结论：`recommended_candidate=null`，`stop_reason=contact_geometry_or_pose_response_not_observable`。停止继续放大权重、增加步数或调整阈值。本结论仅为 engineering validation，不是物理接触验证。

## v6 推荐门更正审计（2026-09-25，engineering validation）

v6 汇总对脚部负的有符号中位残差直接计算“下降比例”，把控制组 −32.9434 mm 到 foot_a1 −30.0938 mm 的向零移动误记为负增益。只读复算的绝对中位残差为 32.9434→30.0938 mm，下降 8.6501%；foot_a1 的 3D P95 为 131.579 mm，相对同轮控制 129.490 mm 增加 1.61%，小于原定 2% 门；2D P95 改善 1.58 px，穿透比例未增加，beta 与梯度审计门通过。因此原 `recommended_candidate=null` 是判定公式错误，不应继续引用为当前结论。更正明细保存在 `surface_contact_window60_90_v6_gate_reaudit.json`，原 v6 comparison 保留为历史错误记录，未就地改写。

foot_a1 仅成为开发窗的进一步验证候选：Stage D 后左右 sole 顶点仍有约 93.13%/92.29% 位于当前地面下方，不能称物理触地有效。v6 的 3D/2D 梯度比约 1.25 是从 v5 审计外推，尚未在 v6 Stage C 实测。下一步应先复核脚部接触的逐帧/逐侧改善和当前地面变换质量，再用冻结的 foot=1 方案做独立窗口验证；不得将开发窗通过解释成真实三维精度或承重验证。

## v6 foot=1 独立窗口验证（engineering validation，未冻结）

动机：v6 开发窗 gate 复算（`surface_contact_window60_90_v6_gate_reaudit.json`）给出 foot_a1 穿透幅值改善 8.65%，需三独立窗口验证冻结配置（foot=1/hand=0，obs 1.0/0.20，Stage D 0.70/0.10，80 步）。

执行：129..159 与 278..308 的 control/foot_a1 各退出码 0；373..403 的 control 退出码 0，但 foot_a1 在 Stage D 后审计探针处按设计抛错（`surface loss does not depend on SMPL vertices`）：该窗 31 帧 foot 标签全为 invalid、foot_contact_weight 全零，lfoot 恒为零故无顶点梯度。未改脚本绕过，未事后换窗。

结果：129 窗增益 4.70%、278 窗增益 4.45%，其余门（3D≤2%、2D≤5px、穿透≤1pp、冻结、同图、零 COCO 梯度、有限）均过，仅绝对增益未达 5%；373 窗对子不完整。三窗未全部通过。

结论：`selected_candidate=null`，`stop_reason=foot_a1_not_stable_across_independent_windows`。foot=1 仍只是开发窗候选，不得冻结；脚底约 93% 候选点仍低于地面，不称真实触地。
