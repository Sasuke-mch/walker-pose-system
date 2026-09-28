# WiLoR + SMPL-H 全序列候选拟合

## 2026-09-28 — VPoser + 双目 WiLoR 修正后完整运行

本次使用 `realtime_app/tools/fit_smplh_wilor_sequence.py`，对 people1 的 448 帧原始 PMPose、左右相机 WiLoR JSONL、当前标定、SMPL-H male 和 VPoser V02_05 进行完整运行。输出目录为 `fit_cuda_vposer_corrected_stereo448_v1/`，未覆盖旧结果。

阶段为 A 身体 VPoser、B 共享 beta、C 身体 VPoser 微调、D1 手部观测、D2 接触接口、D3 受限整体微调；本次未启用接触输入，因为现有 hand contact 权重全为零。左右解剖手均由左右相机共同约束，WiLoR 候选使用腕部距离和歧义门。

输出数组检查通过：vertices `(448,6890,3)`、faces `(13776,3)`、模型 COCO `(448,17,3)`、SMPL-H joints `(448,73,3)`、左右手点 `(448,21,3)`；所有模型数组有限，三角面最大索引 6889。原始三角化点 7534 个有限且 accepted，当前运行没有 finite rejected 点。

拟合效果评估：最终身体 COCO 三维残差 RMS 约 504 mm，手部二维 RMS 约 1092 px（左手）和 699 px（右手）。身体和手部损失虽在阶段内下降，但残差仍很大，且可视化显示人体姿态明显偏离自然人体。该结果只能作为工程链路运行证据，不能作为有效 SMPL-H 姿态或真实手部姿态结果。

同源可视化页面为 `fit_cuda_vposer_corrected_stereo448_v1/smplh_people1_vposer_viewer.html`，使用该次结果的网格、真实三角面、COCO 点、三角化点、SMPL-H 手部点以及当前场景重放的地面、助步器和双相机轨迹。浏览器加载无 JavaScript 错误，页面可逐帧播放和交互旋转。

## 目的

验证 WiLoR 是否能在现有双鱼眼 SMPL 主线上作为手部辅助观测，并把最终人体统一输出为 SMPL-H 的 6890 个顶点。该记录是工程候选验证，不代表手部真实精度或物理抓取成功。

## 输入与坐标约定

- 身体输入：原有 PMPose `raw_predictions.json`，仍由现有严格双目三角化产生 17 个 COCO 点，单位从 mm 转为 m。
- 手部输入：`wilor_left.jsonl` 与 `wilor_right.jsonl`。输入目录分别是有序的 `left_ccw90`、`right_cw90` 图像；推理前左目逆时针目录旋回原始鱼眼方向，右目顺时针目录逆向旋回原始鱼眼方向。
- WiLoR 输出的 MANO 21 点先由模型相机坐标加平移并投影到原始鱼眼像素。它们不是独立二维检测真值，`raw_pixel_bounds_ok=false` 的候选保留在 JSONL 中但不进拟合。
- SMPL-H 使用 `third_party/WiLoR/mano_data/models/SMPLH_male.pkl`。该 pickle 没有 smplx 可选的 PCA 元数据，运行时补入 45 维单位分量和零均值，保持下载模型的 6890 顶点、13776 面、52 个回归关节的 posedirs/J_regressor/weights。
- SMPL-H 的身体顶点仍通过现有固定 `J_regressor_coco.npy` 回归 COCO 17 点；手部二维损失在左目使用 `K0,D0`，右目先用 `R_cam0_to_cam1,T_cam0_to_cam1` 变换到 cam1 再使用 `K1,D1`。

## 实际运行

WiLoR 左、右目各 448 帧，CUDA 推理；随后使用 448 帧从零初始化的 SMPL-H 参数进行 180 步 Adam 优化。每帧优化 global orientation、translation、63 维 body pose、左右各 45 维手部 pose，共享 10 维 beta。

## 审计结果

- 左目 448 帧、914 个候选，21 点和 778 个 MANO 顶点均有限；通过原始鱼眼像素边界的候选 480 个。
- 右目 448 帧、914 个候选，通过像素边界的候选仅 7 个。其余候选不被裁剪、不插值、不混入旧结果。
- 因而进入拟合的 WiLoR 2D 点数为左目 4896、右目 80（只取 16 个可与 SMPL-H 关节直接对应的非 fingertip 点）。
- 输出 `fit_cuda_full/result.npz` 的形状为 `(448,6890,3)`，面为 `(13776,3)`，COCO 回归结果为 `(448,17,3)`。
- 最终候选的身体三维残差约 `0.247 m` RMS，左/右手二维残差约 `76.1/37.6 px` RMS。该数值只说明当前零姿态初始化和损失权重尚未达到可用拟合，不可表述为真实误差。

## 当前结论与停止条件

SMPL-H 接口、6890 顶点统一输出、COCO regressor 接入和 CUDA 全序列执行已经跑通；但 WiLoR 右目像素门通过率极低，且候选拟合残差较大。当前结果只能作为工程候选，不能进入正式抓取实验。下一步必须先定位右目旋转/检测框/鱼眼投影链的失败原因，并在双目 hand candidate 的重投影与关联门通过后再调整拟合权重或手部初始化。

## 二维可视化与失败分析

可视化脚本为 `realtime_app/tools/visualize_wilor_2d.py`，示例输出位于 `visuals_left/` 和 `visuals_right/`。绿色点表示落在 1920x1080 原始鱼眼范围内，红色点表示越界；框颜色和标签中的 `bounds` 使用同一候选级门。

右目失败并不是“没有检测到手”：914 个候选中，每个候选仍有 8--21 个二维点在图内；但只有 7 个候选的 21 个点全部在图内。当前候选级 all-21 门因此把 907 个候选整体拒绝。可视化显示检测框多数覆盖了手腕/手部，但部分 MANO 指尖或掌部投影落在上下边界外，强遮挡和鱼眼边缘使这一现象更明显。

当前最可能的第一原因是候选级边界门过硬，而不是简单的左右旋转错误；第二原因是 WiLoR 的 MANO 相机投影是针孔模型，直接回投到鱼眼图像只能作为近似，不能自动吸收鱼眼畸变；第三原因是右目候选关联尚未完成，不能把不同候选按 `candidate_index` 强行配对。下一步应改成逐关节边界 mask，并要求腕部/掌部最小有效点数，再用双目重投影和同侧候选关联决定是否进入三角化；不能直接放宽为无门限全点使用。

## 2026-09-28 — 审计修复后的实现状态

本次只读审计提出的候选关联分叉、SMPL-H 环指/小指映射无断言、WiLoR 旋转尺寸契约、viewer 右相机光心来源和框级置信度语义已在代码中修正。拟合入口统一调用 `smplh_hand_observation.read_view()`，逐候选关联结果写入 `wilor_association_audit.json`；SMPL-H 运行时执行模板几何断言；WiLoR 记录 `orientation_contract`；viewer 使用标定 `C_right=-R01.T@T01` 并记录与静态 walker 光心的差值；框级 `detector_confidence` 进入四路手部二维残差权重，缺失时使用中性权重 1。

本次修复尚未重跑完整 448 帧拟合。既有 `fit_cuda_vposer_mano_pca12_temporal_v1/` 仍是修复前输出，只用于历史工程对照；短窗对照和手部射线/深度审计通过后，才能启动新的完整运行。新 viewer 代码已用临时输出验证 13776 个三角面对应 41328 个索引，并确认页面证据标签存在。
