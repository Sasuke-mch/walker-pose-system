# WiLoR + SMPL-H 全序列候选拟合

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
