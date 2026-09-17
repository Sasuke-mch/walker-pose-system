# 当前两阶段与动态地面主线

本文只描述当前保留、可运行和可复现的主线。历史平面 XY、完整平面 SE(2)、稠密背景 SE(3)、平滑展示和刚性助步器接触原型均不再作为当前入口。

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
- 接受的旋转和平移使用 0.45 更新增益。
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
