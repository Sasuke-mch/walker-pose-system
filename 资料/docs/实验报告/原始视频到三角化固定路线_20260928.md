# Walker Pose System：原始双目视频到三维三角化固定路线

更新时间：2026-09-28

本文冻结当前助步器人体双目处理路线。后续实验从原始双目采集开始，沿本路线逐级输出；不得用旧动态结果、旧三角化结果、临时零点或另一条未记录的旁路替换当前运行结果。

## 1. 路线总览

```text
双目采集目录/左右视频
  -> 帧读取与时间戳配对
  -> 左右相机物理身份解析
  -> 仅为姿态模型旋转图像
  -> PMPose 二维推理
  -> 姿态点逆旋转回原始鱼眼像素
  -> 左右人物身份关联
  -> 双鱼眼几何三角化
  -> 保留所有二维观测、三维点、误差和失败原因
  -> Stage 1/2 与动态地面变换
  -> 助步器拓扑按当前地面变换重构
  -> 人体、地面、助步器和误差可视化
```

主入口是 `realtime_app/run_stereo.py`。离线完整估计入口是 `realtime_app/tools/estimate_complete_stereo_3d.py`，当前推荐直接读取本次运行的 `stereo_results.jsonl`：

```powershell
.venv-cuda\Scripts\python.exe realtime_app/tools/estimate_complete_stereo_3d.py `
  --model pmpose `
  --stereo-jsonl <本次运行>\stereo_results.jsonl `
  --calibration <本次运行>\smpl_calibration\stereo_fisheye.json `
  --output-dir <输出目录>\complete_3d_direct_raw `
  --fps 30
```

## 2. 输入与坐标约定

物理相机身份由 `realtime_app/pose_app/camera_registry.py` 解析 PnP `instance_id` 注册表。`cam0=LEFT`、`cam1=RIGHT` 是角色约定，不能把 Windows OpenCV 数字索引当作永久身份。

采集和回放由 `realtime_app/tools/capture_stereo.py`、`realtime_app/pose_app/stereo_camera.py` 和 `realtime_app/run_stereo.py` 完成。每帧记录原始图像、frame_id、主机返回时间戳、读取耗时和丢帧信息。双目配对必须使用采集目录中的 `stereo_pairs.csv` 及左右帧 CSV；不能把另一段视频按数组下标强行截断后与姿态结果配对。

当前采集图像为原始鱼眼坐标。姿态模型输入可旋转，映射代码在 `realtime_app/pose_app/rotation.py`：`rotate_image_for_model` 只改变模型输入方向，`restore_model_result_to_raw` 用 `model_to_raw_point` 把每个点、框和尺寸恢复到原始鱼眼像素。进入标定、极线关联和三角化前必须完成该逆映射。

当前左右模型旋转为左 `ccw90`、右 `cw90`。原始鱼眼分辨率为 1920x1080；正式鱼眼标定由 `StereoCalibration.load` 读取，三角化长度单位为 mm，坐标系为左相机坐标。

## 3. 二维姿态与原始点保留

PMPose 输出在 `InferenceResult`/`PersonPose` 中保存。每个候选人保留 bbox、bbox score、pose score、COCO-17 点和逐点 score。二维点不能因为分数低而从原始记录中删除。

`realtime_app/tools/export_pmpose_raw_predictions.py` 只用于需要历史 PMPose 接口的离线适配。缺人帧现在写空检测列表；禁止写入全 NaN 人体占位。全 NaN 占位会让后续适配器误以为该视图仍有一个人，进而污染 force-all 或时间补全。

## 4. 人物关联

代码在 `realtime_app/pose_app/triangulation.py`：

1. 对左右候选计算共同 COCO 点的鱼眼去畸变极线代价；共同点不足时记录 bbox 中心退化代价。
2. 候选按代价排序，一对一选择；当前主线单人模式固定 `max_matches=1`。
3. 严格主链使用 `max_association_cost=0.05`。关联失败必须记录 `association_failed` 或 `no_stereo_person`，不能静默复制另一视图。
4. 多人候选不能凭视觉效果随意挑人。完整估计入口对每侧独立处理：恰好一个人可贡献该侧观测；多人标为 `no_unique_left_person`/`no_unique_right_person`；两侧均唯一才有直接双目交会。

## 5. 三角化逻辑

实现文件：`realtime_app/pose_app/triangulation.py`。

对左右原始鱼眼点先调用 `StereoCalibration.undistort_normalized`，再使用左投影矩阵 `[I|0]` 和右投影矩阵 `[R|T]` 调用 `cv2.triangulatePoints`。点通过齐次坐标归一化得到左相机系 `xyz`，再分别投影回左右鱼眼图，记录：

- 左右深度；
- 左重投影误差；
- 右重投影误差；
- 平均重投影误差；
- `negative_or_zero_depth`、`high_reprojection_error`、`non_finite_triangulation` 等质量状态。

严格主链的质量门固定为：左右二维分数均达到 0.25、点有限且在原始鱼眼边界内、左右深度为正、平均重投影误差不超过 10 px。严格点才供 Stage 2 相机姿态和脚锚使用；失败点仍写入 JSONL 的 `reason` 和误差字段。

完整显示/诊断路线不把重投影误差当作删除条件。只要两侧有有限二维点并能得到有限几何结果，就保留 `raw_xyz`，把深度和误差作为标记。越界或非有限点不能进入标定反投影；它们保留二维状态，作为缺失或域外观测记录。

## 6. 单侧、缺人和异常点的处理

单侧观测没有唯一三维深度，不能直接声称“三角化成功”。固定处理如下：

| 观测状态 | 输出 | 是否直接三角化 | 后续含义 |
|---|---|---:|---|
| 左右均有唯一人、点有限 | `stereo_raw` + xyz + 左右误差 | 是 | 直接双目测量 |
| 只有左侧点 | `single_view_temporal_*` | 否 | 当前左相机射线与同关节邻近直接双目时间锚求交 |
| 只有右侧点 | `single_view_temporal_*` | 否 | 当前右相机射线与同关节邻近直接双目时间锚求交 |
| 两侧均缺点但历史有直接双目锚 | `temporal_*` | 否 | 同关节时间插值、外推或保持，来源明确 |
| 两侧均缺且无锚 | `unavailable_no_stereo_anchor` | 否 | 保持不可用，不造点 |
| 某侧多人 | `no_unique_*_person` | 否 | 身份不确定，不选择候选 |
| 点越界 | `out_of_raw_image_bounds` | 否 | 保留二维点和原因，不送入鱼眼几何 |
| 点有限但深度为负/重投影大 | 有限 `raw_xyz` + 质量状态 | 数学上有 | 只作诊断显示，不供严格地面更新 |

旧的完整实现位于 `realtime_app/tools/estimate_complete_stereo_3d.py`。其中 `_triangulate_unfiltered` 只要求有限、边界内的双侧点；`estimate_joint_series` 实现单侧射线投影到直接双目时间锚，双侧缺失使用时间锚。骨长只在估计后统计，不能作为三角化输入或用来伪造观测。

## 7. Stage 1/2、地面和助步器

当前阶段逻辑见 `docs/CURRENT_STAGE_GROUND_PIPELINE.md`、`realtime_app/pose_app/realtime_stage_walker.py` 和 `realtime_app/pose_app/dynamic_ground_live.py`。

- Stage 1：助步器/背景基本静止，人体相对运动；保持相机相对地面姿态。
- transition：证据不足时保留过渡状态，不强行切换。
- Stage 2：背景运动且双踝严格点能作为静止脚锚时，估计相机旋转和平移。
- 地面变换固定为 `P_ground = R_ground_from_left_camera[t] P_left_camera + t_ground_from_left_camera[t]`。
- Stage 2 只能消费严格踝点；完整显示点、单侧时间估计点不能反过来更新地面姿态，否则会把补全误差反馈为相机运动。

助步器结构文件为 `research_records/engineering_validation/G20260918_offline_walker_frame_evidence_v1/coarse_complete_model_v2_camera_rail/coarse_walker_model.json`。它提供固定节点、扶手段、脚节点和相机基线参考；每帧节点必须经过本次 `T_G<-C_t` 变换。助步器“靠近地面”和手腕“靠近扶手”只能记为几何候选，不能写成承重、触地或临床结论。

## 8. 当前 360 帧验证结果

输入运行目录：`realtime_app/outputs/pipeline_20260927_201716_pmpose_mainline`。

- 原始配对：360 对，pair_id 1157–1516；时间戳为采集 CSV 的主机 read-return 时间，平均左右差 9.35 ms，P95 16.57 ms。
- 严格主链：平均每帧 8.27 个 accepted 3D 点；左右边界拒绝计数为 0/168。
- Stage：warming_up 3 帧、Stage 1 212 帧、transition 62 帧、Stage 2 83 帧。
- 动态地面：接受 76 次动态姿态更新；这属于工程重放结果。
- 直接读取原始 `stereo_results.jsonl` 的完整估计输出：`complete_3d_direct_raw/complete_3d_estimates.jsonl`。
- 6120/6120 个关节点有估计：`stereo_raw=5408`、`single_view_temporal_interpolated=677`、`temporal_interpolated=35`。
- 当前三维视频：`complete_3d_direct_raw/walker_ground_stage_video_final/raw_visual_human_partial_handles_ground.mp4`，360 帧、30 FPS；逐点颜色编码重投影误差，画面含地面、相机、助步器、Stage 和手腕/脚踝距离审计。

这些数字只能说明当前软件链的内部输出和可追溯性，不构成三维真值、人体精度、触地、承重或步态准确率。

## 9. 后续固定执行规则

每次新实验必须保存：原始双目输入路径、配对 CSV、左右二维原始点、标定文件、三角化 JSONL、Stage/地面 JSONL、助步器模型路径、命令行和视频输出路径。所有失败帧、拒绝原因、单侧状态、多人身份不确定状态和误差必须保留。

严格三角化、完整诊断估计和可视化必须使用同一运行目录的同源输入。任何新模型或新补全算法只能作为单变量对照，先输出独立目录和状态统计，再决定是否进入主线；不得回写或覆盖本固定路线的严格结果。
