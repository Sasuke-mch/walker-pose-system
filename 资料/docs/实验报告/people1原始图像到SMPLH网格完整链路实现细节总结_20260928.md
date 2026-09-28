# people1 从原始双鱼眼图像到 SMPL-H 网格的完整链路实现细节总结（2026-09-28）

> 备份位置：`资料/docs/实验报告/people1原始图像到SMPLH网格完整链路实现细节总结_20260928.md`
> 对应审计：`资料/docs/实验报告/SMPLH_WILOR_IMPLEMENTATION_AND_AUDIT_20260928.md`
> 实验目录：`research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/`
> 当前主结果：`fit_cuda_vposer_mano_pca12_temporal_v1/`（`fit_summary.json` 标记 `engineering_candidate`）
> 结论边界：全链路产物均为工程候选（WiLoR 模型派生观测的内部一致性结果），不构成真实三维精度、手部精度、握持或承重结论。

---

## 一、链路总览（输入→输出）

```text
原始双鱼眼图像 input_448pairs/pair_0000..0447.png（1920×1080，raw_fisheye）
├─身体支路：PMPose 左右 raw_predictions.json（COCO-17，x,y,conf）
│   → run_clean_full_sequence.raw_triangulate（严格双目三角化，mm，左相机系）
│   → tri(448,17,3) + body_accepted mask + quality（连续权重）
├─手部支路：left_ccw90 / right_cw90 旋转图像目录
│   → run_wilor_sequence.py（WiLoR + detector.pt，init_renderer=False）
│   → 旋转逆变换回原始鱼眼像素（输入/输出尺寸契约校验，记录 orientation_contract；pixel_frame=raw_fisheye）
│   → wilor_left.jsonl / wilor_right.jsonl（每帧多候选：21点+778顶点+局部三维+框级置信度）
│   → 拟合调用统一的 smplh_hand_observation.read_view()：同相机 PMPose 腕部 150px 距离门 + 25px 歧义门，校验 pair_id/重复帧并保存逐候选审计
│   → hand_ll/hand_lr/hand_rl/hand_rr（448,21,2）+ valid（finite）+ bounds（越界软权重0.1）
├─拟合支路 fit_smplh_wilor_sequence.py（CUDA，150步）
│   SMPLH_male.pkl + MANO_LEFT/RIGHT.pkl（真实PCA）+ VPoser V02_05 + J_regressor_coco.npy
│   → 刚体初始化 root/transl（肩髋基 R_global = R_obs @ R_model.T，128.4mm，448/448 通过 300mm 门）
│   → A（身体，冻结root）→ B（仅beta）→ C（释放root + root锚定）
│   → D1/D2（仅12维PCA手部）→ D3（0步，无接触输入时正确跳过）
│   → result.npz + fit_summary.json + initialization_audit.json
└─可视化 build_smplh_people1_reference_viewer.py
    result.npz + audit_20260928/scene（当前场景重放）→ smplh_people1_mano_pca_viewer.html（同源页面）
```

---

## 二、分模块实现细节

### 1. WiLoR 推理（`realtime_app/tools/run_wilor_sequence.py`）

- 模型加载：`WiLoR.load_from_checkpoint(..., init_renderer=False)`；兼容垫片（`inspect.getargspec`、numpy 别名）；detector 为本地 `detector.pt`（`weights_only=False` 仅限加载该检测器）。
- 旋转约定：`left_ccw90→ROTATE_90_CLOCKWISE`，`right_cw90→ROTATE_90_COUNTERCLOCKWISE`；不支持其他取值。现在要求非 raw 输入为旋转后的 `1920×1080` 对应尺寸，逆旋转后必须恢复 raw `1920×1080`，并把 `orientation_contract` 写入每个候选；拟合阶段的腕部距离审计继续保留方向闭环证据。
- 左右手翻转：`multiplier = 2*right-1`，`pred_cam[:,1] *= multiplier`；`joints[:,:,0] *= sign`、`verts[:,:,0] *= sign`。
- 像素回映：弱透视 `cam_crop_to_full` 得 `cam_t`，`xyz = joints + cam_t`，针孔模型 `pixels = xyz[...,:2]/z * focal + (w/2, h/2)`，`focal = EXTRA.FOCAL_LENGTH / IMAGE_SIZE * max(h, w)`。深度 `z<=0` 或非有限直接 `raise`。
- 保存字段：`keypoints_2d_raw_fisheye(21,2)`、`keypoints_3d_model_local(21,3)`、`vertices_3d_model_local(778,3)`、`camera_translation_model`、`detector_confidence`（**框级**，非逐点）、`raw_pixel_bounds_ok`（候选级）、`coordinate_frame_3d=model_local_unaccepted`。
- 现状：左右 valid 点 14343/16947（较历史右目 7/448 显著改善）。

### 2. 身体三角化输入

- 输入：`V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/{left,right}/pmpose/raw_predictions.json` + `stereo_fisheye.json`（cam0=LEFT，cam1=RIGHT）。
- 经 `raw_triangulate` 得 `tri(448,17,3)`（mm，左相机系）+ accepted + quality；拟合内 `/1000` 转 m。
- 当前运行：`body_accepted_points=17192/7616`，初始化 RMS 128.4mm。

### 3. 候选关联（统一 `read_view`，由拟合入口调用）

- 同相机 PMPose 腕部（COCO 9=左腕、10=右腕）选最近 WiLoR 同侧候选；门：最近距离 ≤150px，且第一/第二距离差 ≥25px，否则该帧该手无观测，并在 audit 中保留原因。函数同时检查 `pair_XXXX` 图像名、帧号范围和重复帧。
- `valid` = per-point finite（仅防 NaN）；`bounds_ok` 为诊断元数据，**不剔除**，越界点权重 0.1 保留参与优化。
- `detector_confidence` 作为候选级权重进入四路手部残差；缺失时记录 `missing_neutral_weight` 并使用中性权重1。逐候选结果写入输出目录的 `wilor_association_audit.json`。
- 新增手部跨视角几何审计：拟合入口读取四路同侧观测后，分别对左手和右手的 cam0/cam1 21点执行原始鱼眼边界、去畸变 DLT 三角化、双目正深度、射线夹角和两视图鱼眼回投影误差检查。逐点结果写入 `wilor_hand_geometry_audit.jsonl`，统计写入 `wilor_hand_geometry_summary.json`；这些结果只用于审计，`used_for_fitting=false`，不会把未经验证的手部三维点加入损失。

### 4. SMPL-H 拟合（`realtime_app/tools/fit_smplh_wilor_sequence.py`）

- 模型：`smplx.SMPLH(SMPLH_male.pkl, data_struct=注入双MANO PCA, gender=male, use_pca=False, flat_hand_mean=True, batch_size=448)`。
  `flat_hand_mean=True` 时 smplx 内部手部 mean 置零（`body_models.py` 第614–625行），解码侧加一次 mean → **mean 只加一次**（已核源码）。
- 身体：`beta(1,10)=0` 初始化；`root/transl` 由模板-观测肩髋刚体基初始化（`R_global = R_obs @ R_model.T`→rotvec；translation 用同旋转模板双髋中点对齐；三分支：双髋/单髋/不可用）。
- VPoser：`load_vposer_explicit` 显式目录+checkpoint+latent_dim 校验；`latent(448,32)→decode→body_pose(448,63)`；冻结参数，仅解码身体 21 关节，不碰腕/手指。
- 手部：`lhand/rhand(448,12)=0` 初始化；解码 `θ = mean + (a/s) @ C[:12,:]`，`s_i=||C[i]||`（实测行范数 1.31→0.25 递减，前12行为主成分，截断与归一化合理；约定与 smplx `coeff @ components` 一致）。
- 手部 21 点：`hand21()` = 16 内部关节（HAND_JOINTS：左 `[20,34,35,36,22,23,24,25,26,27,31,32,33,28,29,30]`，右镜像）+ 5 表面指尖（TIP_VERTICES 左 `[2746,2319,2445,2556,2673]`）。经 MANO 模板骨长（10-12 链长于 7-9 链→10-12=环指）+ `vertex_ids['mano']` 指尖顺序（thumb,index,middle,ring,pinky）+ WiLoR `joint_map` 联立验证：**ring/pinky 无错位**（输出 13–16=环指、17–20=小指，与 OpenPose 手部顺序一致）。`hand21` 纯 torch 索引，保持可微。
- 投影：左目直接 fisheye 投影；右目先 `X_cam1 = R01 @ X_cam0 + T01`（T01：mm→m）。四组残差：左手@左目、左手@右目、右手@左目、右手@右目（相机身份×解剖侧正确配对）。
- COCO：`J_regressor_coco.npy` 校验 `17×6890` 行和为1 → `predicted_coco`；body 残差用 quality 连续权重、仅 accepted mask；rejected 点不入监督（当前 finite-rejected=0，属分布声明）。
- 阶段（150步，lr=0.02×阶段缩放）：A[transl,latent]（root冻结）→ B[beta]（body梯度保留）→ C[beta,root,transl,latent]+root锚定0.01 → D1/D2[lhand,rhand]（身体项 detach 冻结）→ D3（无接触输入→0步跳过）。每阶段重建 Adam（仅当阶段变量），`stage_history` 与 summary 一致。
- 损失：`body + 1e-7·hand(≥D1) + 0.02·temporal(≥D1) + 1e-3·PCA先验 + 0.02·VPoser先验 + 0.01·root锚定(仅C) + 1e-3·beta²`。数量级核对：hand 均值约 (300px)²×1e-7≈9e-3，与 body≈7e-3 同量级，手部梯度只流向 PCA 参数——权重设计有效，非"过小无梯度"。
- 新增可选身体模型空间时序项：命令行 `--body-temporal-weight` 默认0以保持旧基线；显式开启时仅在 Stage A/C 对 COCO-17 点减去双髋中点后的轨迹计算连续三帧二阶差分，使用 delta=0.03 m 的 Huber 惩罚，不跨身体无效帧。该项只约束模型运动，不平滑二维/三维观测，也不替代失败帧。
- 时间正则：二阶差分作用于 **PCA 系数**（手形），`E=0.5·(L+R)`，三帧连续有效掩码（任一相机任一点 valid），不跨缺失段，不把整手平移误作抖动。
- 接触：本次无输入（`surface_hand_contact_weight=0`），D2 改名 `D2_hand_refine_no_contact` 且 `contact_active=false`；请求接触但全零权重会 `raise`，逻辑正确。
- 输出（result.npz）：vertices(448,6890,3)、faces(13776,3)、predicted_coco、smplh_joints、hand_points_left/right、左右 PCA(448,12) 与解码 pose(448,45)、vposer_latent、root/transl/beta、tri/mask/quality、WILOR 四组二维与 mask/权重、接触诊断、root_init_source、initialization_body_rms_mm。

### 5. 可视化（`build_smplh_people1_reference_viewer.py`）

- 输入：本次 result.npz + `audit_20260928/scene` 当前重放（scene_transforms.npz 等）；输出 `smplh_people1_mano_pca_viewer.html`（同源）。
- 内容：6890 顶点着色表面（faces 以 gzip-base64 传 JS，Python 侧无 `.flat()`，正确性取决于内嵌 JS 解码，待开页验证 P2-3）、模型 COCO/三角化 COCO/52 关节/双手 21 点、相机、地面、助步器。
- 右相机光心现在使用标定外参 `C_right = -R01ᵀ @ T01`，再经当前地面变换显示；同时计算并记录它与静态 walker 文件右光心的差值，超过5mm时发出警告。页面标签写明 `engineering_candidate`、WiLoR 模型派生手部点和 `contact_active=false`，并在 JS 中检查三角索引数为41328。

---

## 三、当前拟合结果快照（`fit_summary.json`）

```text
status=engineering_candidate, frames=448, steps=150
init_body_rms_mm=128.4（门300，448/448通过），root来源=模板-三角化肩髋刚体基，translation=同旋转模板髋中点
final_body_rms_mm=83.0，hand_l_px=337.1（376.1→），hand_r_px=266.0（283.5→）
hand_temporal=1.7e-05，PCA先验=0.0101，VPoser先验=0.2021，root锚定/骨骼/接触=0
hand_3d_observation_used=false，contact_active=false，D3 skipped
```

解读：身体从 128mm 收敛到 83mm；手部缓慢下降但残差仍大（观测噪声 + 针孔回映 vs 鱼眼投影失配主导，非优化未收敛）；时间项/先验量级正常。

---

## 四、审计发现的问题摘要（严重级别与最小修复集）

| 编号 | 当前状态 | 级别 |
|---|---|---|
| P0-2 | 已统一为 `read_view`，加入 pair_id/重复帧校验并落盘 audit；候选框级置信度进入权重 | 已修复 |
| P0-1 | 已加入模板骨长和指尖最近关节断言 | 已修复 |
| P1-1 | 已加入输入/输出尺寸契约、orientation_contract 和腕部距离审计；语义方向仍以同相机关联距离为证据 | 已修复（结构门） |
| P1-2 | 手部为双目独立二维残差之和，无手部三角化；"双目"表述需收紧（代码诚实标记 `hand_3d_observation_used=false`） | P1 |
| P2-1 | viewer 使用标定 `-R01ᵀT01`，并记录与静态 walker 光心差值 | 已修复 |
| P2-2 | 框级置信度进入四路手部残差权重，缺失值使用中性权重并明确标注 | 已修复 |
| P2-3 | viewer 已验证索引数41328，页面含 engineering_candidate、WiLoR派生和 contact_active=false 标签 | 已修复 |
| P3 | prompt 第 10 项资料（`资料/docs/实验报告/...AUDIT`）在本仓库的实际路径是 `资料/docs/实验报告/SMPLH_WILOR_IMPLEMENTATION_AND_AUDIT_20260928.md`；README 手部表仍写单 MANO；EXPERIMENT.md 主体仍描述旧运行；根目录旧 HTML 需标注 | P3 |

最小修复集已完成：统一候选关联实现（单函数 + 校验 + audit 落盘）→ `validate_smplh` 加 ring/pinky 模板几何断言 → WiLoR 输入/输出尺寸和腕部距离闭环审计 → viewer 右光心公式 + summary 置信度语义修正 → 本文件同步。**不做**：增步数/增手部权重、引入 HMP、平滑替换结果、删高残差帧、旧结果初始化。

---

## 五、下一步推进方向

### 方向一（必须先行）：可审计性修复 + 短窗对照（1–2 天）

1. 已完成最小修复集。验收：重复帧/错位 JSONL 必 `raise`；交换 ring/pinky 必 `raise`；尺寸方向契约错误必 `raise`；正常输入生成逐候选 audit。短窗对照仍需单独运行。
2. 短窗 30 帧单变量实验（手部 valid 密集段，参数与 PCA12 运行一致，A/B/C/D1/D2 各 30 步，新目录），对照同窗切片：身体 RMS 差异 <5mm，手部 valid 点差异可解释，audit 逐帧可查。

### 方向二：手部几何审计（短窗通过后）

1. 手部射线夹角/深度分布审计已接入拟合入口：同一解剖手左右目观测的三角化夹角、正深度、鱼眼回投影误差和逐点拒绝原因均落盘；仍需在具备 OpenCV 运行环境的短窗中执行数值审计。
2. 框级 `detector_confidence` 已接入四路手部残差权重；后续若做单变量实验，应比较“加权/不加权”而不是再次声称尚未接入。
3. 在三角化审计完成前，不加三维手部项、不调大手部权重。身体时序项已实现为显式单变量开关，尚未运行消融。

### 方向三：完整 448 帧重跑（短窗和手部几何审计通过后）

- 修复集为唯一变量，同参数（`--hand-pca-comps 12 --hand-pca-prior-weight 1e-3 --hand-temporal-weight 2e-2 --steps 180`），新输出目录，禁止覆盖旧结果。

### 方向四：接触与场景（独立支线）

- 掌面顶点集 + handle 语义 + 地面变换具备后，先在短窗验证接触项（`contact_active=true` 且残差单调下降），再考虑 D3 有限开放；D3 開放腕部/前臂需单独角度限制并记录身体退化。

### 方向五：文档与证据管理

- EXPERIMENT.md 追加 PCA12 一节并指向最新输出；README 手部表更新为双 MANO；旧 HTML/旧运行标注保留（不删除）；后续实验记录沿用"输入路径 + 代码路径 + 参数 + 输出路径 + 结论边界"追溯，不写 Git hash。

---

## 六、裁决

- 链路基本闭合，产物同源，无旧结果混用。
- 最严重的三个问题：候选关联分叉（P0-2）、ring/pinky 无断言（P0-1）、旋转无闭环校验（P1-1）。
- **当前仍不可直接进入完整重跑**：代码可审计性修复已完成，但必须先完成30帧短窗对照和手部射线/深度审计，再决定是否进行448帧重跑。

