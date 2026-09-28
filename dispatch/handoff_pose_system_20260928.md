# Walker Pose System 交接记录

更新时间：2026-09-28  
项目根目录：`D:\my_works\walker_pose_system`  
实时程序目录：`D:\my_works\walker_pose_system\realtime_app`

## 1. 交接范围和结论边界

本文交接当前双鱼眼助步器人体姿态项目的主链、people1 数据、SMPL-H + WiLoR 手部接入、现有拟合输出、查看器和后续工作。

当前结果只能称为工程验证、内部一致性或操作性参考一致性。没有独立人工二维标注、独立三维真值、力传感器、真实握持标签或临床标签，因此不能把重投影残差、三角化点、模型网格外观、有效点数量、时序平滑度或接触代理解释成真实三维精度、真实握持、真实承重、触地或临床结论。

本文件不替代仓库权威规则。继续工作前必须按以下顺序读取：

1. `AGENTS.md`
2. `AI_PROGRESS.md`
3. `VISUALIZATION_PIPELINE.md`
4. `research_records/registry/EXPERIMENT_RECORDING_POLICY.md`
5. 当前实验目录中的 `EXPERIMENT.md`、`VALIDATION_PROTOCOL.md` 和输出文件

工作区存在大量既有修改、未跟踪代码和实验产物。接手时不能执行 `git reset --hard`、批量删除、覆盖用户文件或把其他人的改动加入本任务提交。

## 2. 当前总体数据链

```text
双鱼眼原始图像
  -> 左右图像模型旋转
  -> PMPose 身体二维 COCO-17
  -> 回到原始鱼眼像素
  -> 左右人物关联
  -> 正式鱼眼标定下的严格双目三角化
  -> COCO-17 三维观测、质量门和拒绝原因
  -> SMPL-H 身体初始化和 VPoser 拟合

左右原始鱼眼图像
  -> WiLoR 人体/手候选
  -> MANO 局部三维和 21 点
  -> 回映到原始鱼眼像素
  -> 按相机内的 PMPose 腕部做同侧候选关联
  -> 左右相机分别对同一解剖手提供二维辅助项
  -> SMPL-H MANO PCA 手部参数拟合

SMPL-H 参数
  -> 6890 个表面顶点、13776 个真实三角面
  -> COCO-17 模型侧点、52 个内部关节、左右手 21 点
  -> 当前 scene replay 的地面变换
  -> 助步器、双相机光心、三角化骨架和交互查看器
```

身体和手部不是两个独立网格拼接。最终 SMPL-H 前向一次生成连续的 6890 顶点；身体 COCO-17 使用项目现有的 `17 x 6890` 表面回归器，手部使用 SMPL-H 内部关节和手指尖表面顶点。

## 3. 固定系统不变量

- 物理相机身份必须从 `camera_registry.json` 的 PnP `instance_id` 解析；不能把 Windows/OpenCV 数字索引当作永久身份。
- 标定角色固定为 `cam0=LEFT`、`cam1=RIGHT`。
- 模型输入可以旋转为正立；进入人物关联、鱼眼投影和三角化前必须回到原始鱼眼像素。
- 双目约定为 `X_cam1 = R_cam0_to_cam1 @ X_cam0 + T_cam0_to_cam1`，平移进入 SMPL-H 前从毫米转换为米。
- 严格三角化保留 `max_matches=1`、逐关节质量门、逐点拒绝原因、原始边界状态和 rejected 点。
- PMPose 的 accepted 监督和连续 quality 权重必须分开记录，不能把连续权重写成严格通过。
- 所有失败帧、无效点和拒绝原因必须保留，不能用插值、平滑或旧结果静默补齐。
- Stage 1/2、动态地面、助步器和 SMPL-H 都不能掩盖上游二维或三角化失败。

## 4. people1 当前输入

### 4.1 身体二维和三角化链

people1 的 PMPose 原始输出：

```text
research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/left/pmpose/raw_predictions.json
research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/right/pmpose/raw_predictions.json
```

当前新相机相关资源：

```text
realtime_app/calibration/results/stereo_fisheye_20260927_195919.json
realtime_app/calibration/results/static_ground_20260927_201028/ground_reference.json
realtime_app/outputs/stereo_capture/20260927_201716_220
```

当前回放为 448 对 people1 图像序列对应的工程运行；不要把视频数组下标直接当作 PMPose 行号，必须使用显式 `pair_XXXX` 或时间戳关系。

### 4.2 WiLoR 手部输入

```text
research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/wilor_left.jsonl
research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/wilor_right.jsonl
```

历史 JSONL 的重要限制：没有 21 个独立 WiLoR 关节置信度，部分记录没有检测框 confidence，只有候选级边界信息。因此缺失 confidence 时使用并记录 `confidence_source=missing_neutral_weight`，不能伪造逐关节 confidence。

### 4.3 模型资产

```text
third_party/WiLoR/mano_data/models/SMPLH_male.pkl
third_party/WiLoR/mano_data/models/MANO_LEFT.pkl
third_party/WiLoR/mano_data/models/MANO_RIGHT.pkl
models/VPoser02_05/V02_05
models/smpl/J_regressor_coco.npy
```

SMPL-H 资产要求：52 个模型关节、6890 个顶点、真实 13776 个三角面；运行时通过 `validate_smplh()` 检查父子链。SMPL-H 的输出 `joints` 在 `smplx` 中可能包含扩展关节，当前输出中可见 `(448,73,3)`；手部映射只使用明确的 52 个内部关节和 5 个表面指尖，不得把额外关节或指尖混为内部关节。

## 5. 关键代码路径

| 路径 | 当前作用 |
|---|---|
| `realtime_app/run_stereo.py` | 双目采集/回放、模型旋转、二维输出和下游主链入口 |
| `realtime_app/pose_app/triangulation.py` | 严格双目三角化、质量门和拒绝状态 |
| `realtime_app/pose_app/fisheye_camera.py` | 标定读取、鱼眼投影和相机坐标变换 |
| `realtime_app/pose_app/smpl_coco_observation.py` | `17 x 6890` COCO 表面回归器及模型侧观测 |
| `realtime_app/pose_app/smplx_fitting.py` | VPoser checkpoint 加载和身体姿态解码辅助 |
| `realtime_app/pose_app/smplh_hand_observation.py` | SMPL-H 手部关节映射、指尖顶点、父子链检查、WiLoR 候选审计 |
| `realtime_app/tools/run_wilor_sequence.py` | WiLoR 序列推理、候选保存、原始鱼眼坐标回映 |
| `realtime_app/tools/fit_smplh_wilor_sequence.py` | 当前 SMPL-H + VPoser + MANO PCA 拟合主入口 |
| `realtime_app/tools/build_smplh_people1_reference_viewer.py` | 从当前 result 和 scene replay 生成同源查看器 |
| `VISUALIZATION_PIPELINE.md` | 可视化唯一规范 |

## 6. WiLoR 手部坐标和点语义

WiLoR 官方 MANO 到 OpenPose 顺序为：

```python
[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6,
 18, 10, 11, 12, 19, 7, 8, 9, 20]
```

项目侧 21 点顺序是：

```text
0       wrist
1..4    thumb_1, thumb_2, thumb_3, thumb_tip
5..8    index_1, index_2, index_3, index_tip
9..12   middle_1, middle_2, middle_3, middle_tip
13..16  ring_1, ring_2, ring_3, ring_tip
17..20  pinky_1, pinky_2, pinky_3, pinky_tip
```

前 16 个是腕部和手指内部关节；后 5 个是表面指尖候选。指尖不是内部骨骼关节。

SMPL-H 内部关节映射：

```python
HAND_JOINTS["left"] = [20,34,35,36, 22,23,24, 25,26,27, 31,32,33, 28,29,30]
HAND_JOINTS["right"] = [21,49,50,51, 37,38,39, 40,41,42, 46,47,48, 43,44,45]
```

指尖表面顶点：

```python
TIP_VERTICES["left"]  = [2746,2319,2445,2556,2673]
TIP_VERTICES["right"] = [6191,5782,5905,6016,6133]
```

`hand21(joints, vertices, side)` 先取 16 个内部点，再按 WiLoR 顺序插入 5 个指尖顶点。运行前必须执行 `validate_smplh(model)`，并检查每根手指的父子链 `[wrist,a,b]`。

## 7. WiLoR 候选关联和拟合约束

候选关联不是跨相机直接比较 `candidate_index`。当前规则是：

1. 在每个相机内部只比较相同解剖 side；
2. 用 WiLoR 第 0 点 wrist 与同侧 PMPose wrist 的像素距离排序；
3. 最近候选超过 150 px 时拒绝；
4. 最近与第二近候选差小于 25 px 时标记歧义并拒绝；
5. 身体腕部不可用时不声称完成可靠关联；
6. 保存候选选择、距离、逐点 finite、逐点边界、失败原因和 confidence 来源。

拟合时的相机/解剖配对必须是：

```text
左模型手：左相机左手 + 右相机左手
右模型手：左相机右手 + 右相机右手
```

不能把左相机的左手和右相机的右手强行比较。

WiLoR 21 点来自 WiLoR 的 MANO 局部三维及其模型投影，因此手部二维项属于模型派生辅助观测。它不能作为独立二维真值，也不能把其三角化结果称为独立真实三维观测。

当前手部损失为 SMPL-H 手部点经两台真实鱼眼相机投影后的像素残差；手部三维观测项当前为关闭状态。身体主项继续使用 SMPL-H 顶点经过固定 COCO 回归器得到的 17 个模型侧点，与三角化 COCO 观测比较。

## 8. 身体初始化、VPoser 和阶段调度

### 8.1 刚体初始化

身体初始化已从“root 全零、translation 用零旋转模型估计”改为：

1. 用 VPoser 均值姿态和 SMPL-H 模板生成模型 COCO 点；
2. 用模板肩髋点构造模型人体基；
3. 用每帧三角化肩髋点构造观测人体基；
4. 计算 `R_global = R_observation @ R_model.T`；
5. 用同一个旋转后的模板髋中点初始化 translation；
6. 记录每帧 `root_init`、`translation_init_source`、初始 body RMS 和有效基帧数。

这一步解决了全局朝向、平移和根旋转不一致的主要问题。默认初始化门为 `max_init_body_rms_mm=300`；当前 448 帧初始化 RMS 约 128.37 mm，448/448 帧 root basis 有效。

### 8.2 VPoser 的作用边界

VPoser 只约束身体局部姿态 latent，不负责：

- 相机坐标系到模型坐标系的整体朝向；
- 模型整体平移；
- 肩髋骨架刚体对齐；
- 手指局部姿态。

因此全局 root 初始化必须独立完成，不能用 VPoser 替代刚体初始化。

### 8.3 当前阶段调度

当前 `fit_smplh_wilor_sequence.py` 的阶段结构：

```text
Stage A: body_vposer
  trainable = translation + VPoser latent
  root 固定为刚体初始化结果

Stage B: shared_beta
  trainable = 全片共享 beta

Stage C: body_vposer_refine
  trainable = beta + root + translation + VPoser latent
  使用 root 初始锚定项

Stage D1: hand_proximal
  trainable = 左右手 MANO PCA 系数
  身体参数冻结

Stage D2: hand_refine_no_contact
  trainable = 左右手 MANO PCA 系数
  没有接触输入时继续只开放手部

Stage D3: hand_contact_refine
  无有效接触标签时步数强制为 0
```

当前没有有效 hand contact 输入，因此不能执行 D3，也不能让 WiLoR 模型派生像素重新牵引 root、translation 或身体 latent。

## 9. MANO PCA 手部参数化和时序项

此前直接优化 45 维手指轴角容易出现不合理姿态和帧间抖动。当前已改为每帧每只手 12 维 MANO PCA 系数：

```text
PCA coefficient (12)
  -> hands_mean + normalized_coeff @ hands_components
  -> 45 维 SMPL-H 局部手指轴角
  -> SMPL-H forward
```

本地 `MANO_LEFT.pkl` 和 `MANO_RIGHT.pkl` 都包含 `hands_components(45x45)` 和 `hands_mean(45)`。当前模型使用 `use_pca=False, flat_hand_mean=True`，由项目代码显式解码 PCA 后再传入 SMPL-H，避免模型内部重复使用 PCA 或重复添加均值。

手部正则包括：

- PCA 系数二次先验，限制偏离 MANO 平均手姿；
- 连续三帧有效时的 PCA 系数二阶差分；
- 不跨缺失段插值；
- 不把时间正则产生的平滑当成真实观测。

当前代码和结果元数据还记录了腕部到指尖的优先权重。接手时必须复核这组权重是否真正乘入手部残差，而不是只写入 `fit_summary.json`。若尚未实际使用，应把它视为待修复实现差异，不能把 `hand_priority="wrist_to_distal_fixed_weights"` 当成已验证事实。

## 10. 当前最新完整输出

### 10.1 当前主结果

```text
research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/fit_cuda_vposer_mano_pca12_temporal_v1/
```

重要文件：

```text
result.npz
fit_summary.json
initialization_audit.json
stage_trace.json（如存在）
smplh_people1_mano_pca_viewer.html
```

实际 `fit_summary.json` 记录：

```text
status = engineering_candidate
frames = 448
total steps = 150
vertices = 6890
hand observation points retained = 21
hand 3D observation used = false
contact_enabled = false
```

阶段步数为 A/B/C/D1/D2/D3 = 30/30/30/30/30/0。D3 被跳过是因为没有有效手部接触输入。

当前指标只能作为工程目标残差：

```text
初始化 body RMS       ≈ 128.37 mm
最终 body COCO RMS    ≈ 83.13 mm（fit history 中 body_m）
左手二维残差          ≈ 337.4 px
右手二维残差          ≈ 266.4 px
左/右手模型点步长 P95 ≈ 0.071 / 0.052 m（已有记录）
```

身体最新结果相对刚体初始化基线已稳定；手部残差仍很大，且 WiLoR 观测本身为模型派生像素，不能解释成真实手部误差。

### 10.2 旧输出的使用边界

以下目录只用于历史对照或错误追溯，不能作为当前结果或查看器输入：

```text
fit_cuda_full/
fit_cuda_full_v2/
fit_cuda_full_v3/
fit_cuda_corrected_full448/
fit_cuda_vposer_rigid_init_fixed_views_stereo448_v1/
```

尤其不能把旧的错误左右手读取结果、旧根初始化或旧 HTML 混入当前页面。

## 11. 当前查看器状态

当前同源查看器：

```text
research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/fit_cuda_vposer_mano_pca12_temporal_v1/smplh_people1_mano_pca_viewer.html
```

生成脚本：

```text
realtime_app/tools/build_smplh_people1_reference_viewer.py
```

页面应使用同一次运行的：

- `vertices[448,6890,3]`
- `faces[13776,3]`
- 模型侧 COCO-17
- 当前运行的 raw triangulation 和 accepted mask
- SMPL-H 内部关节和左右手 21 点
- 当前 `audit_20260928/scene` 的 Stage、地面变换、助步器和双相机数据

页面不能使用旧动态地面、旧助步器位姿、旧顶点或静态手工移动。

按 `VISUALIZATION_PIPELINE.md`，页面必须验证：真实表面和真实三角面、模型 COCO 点、三角化 COCO 点、accepted/rejected 状态、COCO 骨架连线、双手 21 点及连接线、双相机光心/轨迹、地面坐标、实体助步器、Stage 状态、0.5x/1x/2x 播放和 OrbitControls。

当前已完成文件级和数组级检查；浏览器自动化截图证据仍未完成，页面视觉效果不能仅凭 HTML 文件存在而判定通过。

## 12. 已解决的问题

1. **右相机右手读取错误**：旧代码用 `read_wilor(..., "left")` 读取右目文件，已改为按正确解剖 side 读取。
2. **单个越界点导致整只手丢失**：现在保留完整 21 点，逐点记录 finite/bounds；边界状态作为诊断和软权重来源。
3. **错误的 SMPL-H 手部关节索引**：删除猜测式连续编号，改用 `HAND_JOINTS` 和 `TIP_VERTICES`，并运行父子链验证。
4. **把五个指尖当内部关节**：改为 16 个内部关节加 5 个 SMPL-H 表面指尖。
5. **左右相机解剖手交叉比较**：改成同侧跨相机配对。
6. **root 从零开始导致人体朝向错误**：改为模板肩髋基到三角化肩髋基的刚体初始化。
7. **translation 与非零 root 不一致**：translation 改为使用同一旋转后的模板髋中点估计。
8. **Stage A 同时释放 root**：Stage A 冻结 root；Stage C 才释放并保留 root anchor。
9. **无接触时手部像素牵引身体**：无 contact 时 D3 步数为 0，手部阶段只开放手部 PCA。
10. **45 维手指轴角造成姿态不稳定**：改用左右 MANO PCA 12 维系数、PCA 先验和连续帧二阶差分。

## 13. 当前已知问题和风险

### P0：必须优先复核

- `hand_priority_weights_np` 在结果元数据中存在，但需要确认它是否实际进入 `hand_ll_res/hand_rl_res/hand_lr_res/hand_rr_res` 的 loss。若没有，手部“近腕到指尖优先拟合”的实现尚未完成。
- 手部候选关联虽然按腕部距离执行，但当前仍需对左右相机的同人同手稳定性做逐帧统计和可视化，不能只依赖候选读取成功数量。
- WiLoR 采用针孔模型，项目拟合使用真实鱼眼投影；模型投影回鱼眼存在系统模型差异。高像素残差可能来自观测生成链，而不是 SMPL-H 手指姿态本身。
- 必须确认 MANO `hands_components` 的行列方向、归一化尺度和均值只添加一次。当前结果已使用真实 PCA，但需要保留一个单帧数值审计，防止左右资产或转置错误。

### P1：当前结果解释风险

- 手部二维项是 WiLoR 模型派生观测，不是独立二维检测真值；不能把手部二维 RMS 直接命名为手部关节误差。
- 手部三角化尚未作为独立三维监督使用；`hand_3d_observation_used=false` 是当前实际状态。
- 没有 hand contact 标签，D3 contact refine 没有运行；不能宣称手部接触或握持。
- 身体 COCO RMS 仍是内部拟合目标残差，不是外部姿态精度。
- 查看器尚未完成浏览器截图验收；需要确认实际渲染表面、手指线、右膝 rejected 样式、双相机轨迹和时间轴。

### P2：工程维护风险

- 工作区中有大量 SMPL-X、替代身体模型、审计脚本和未跟踪测试文件，不能默认它们属于当前 SMPL-H 主线。
- `AI_PROGRESS.md` 含有多时期旧路线记录；判断当前状态时必须优先读取顶部最新条目和当前 `fit_summary.json`。
- 旧实验记录中的 180 步、旧身体 RMS 和旧手部统计不能覆盖最新输出的实际 150 步结果。

## 14. 接手后的推荐推进顺序

### 第一步：只读接口审计

不重跑全片，先检查：

1. `fit_smplh_wilor_sequence.py` 中手部 priority weights 是否真正作用于每个点；
2. MANO PCA 解码的矩阵方向、scale、均值和左右模型；
3. `hand21()` 的 21 点顺序与 WiLoR 逐项对应；
4. 左右相机同侧候选关联的逐帧歧义率；
5. `result.npz`、`fit_summary.json` 和 viewer payload 的 shape、单位、帧号和来源是否一致；
6. 当前 viewer 是否真的从 `fit_cuda_vposer_mano_pca12_temporal_v1` 和当前 scene replay 生成。

### 第二步：单帧和短窗验证

先选 5–31 帧短窗，只改变一个变量：

- baseline：当前 MANO PCA + 时序正则；
- candidate：确认并实际启用腕部到指尖 priority weights；
- 不同时修改 PCA 维度、学习率、相机投影、候选门和接触权重。

必须保存每阶段参数、loss 分项、逐点 hand mask、候选关联结果、PCA 系数范围、手指轴角范围和停止原因。

### 第三步：手部数据独立诊断

对左右手分别报告：

- 每帧候选数量；
- 选中/歧义/无候选帧数；
- 21 点 finite、边界内、软权重点数；
- 左右相机重投影残差；
- 双目三角化正深度、射线间隙和拒绝原因；
- WiLoR 模型派生来源标签。

没有通过候选关联和坐标审计时，不把手部三维加入统一监督。

### 第四步：完整 448 帧重跑门

只有短窗满足以下条件才重跑：

- body 初始化 gate 通过；
- root 和 translation 没有异常跳变；
- 身体 COCO 残差没有因手部项明显恶化；
- 手部 PCA 系数没有大面积撞边界；
- hand priority weights 确认真正进入 loss；
- viewer 输入和结果数组来自同一运行；
- 所有失败和 rejected 状态仍被保存。

完整运行后才更新 `AI_PROGRESS.md` 和当前实验 `EXPERIMENT.md`，并按 Git 规则只提交本任务改动。

## 15. 不应采取的捷径

- 不要用旧 SMPL、旧 SMPL-H 或旧 viewer 顶点补当前缺帧。
- 不要把 MANO 参数直接拼接到标准 SMPL body pose 或 beta。
- 不要把 WiLoR 局部三维直接当作米制相机坐标。
- 不要把 WiLoR 的模型投影当作独立二维标注。
- 不要因手指越界而删除整只手，也不要把越界点改到图像边界内。
- 不要在没有接触标签时打开 D3 或释放身体 root 来解释手部像素。
- 不要用增加训练步数、调大学习率或显示平滑掩盖坐标系、候选关联和损失实现错误。
- 不要把查看器中“看起来重合”写成姿态精度或真实握持结论。

## 16. 当前交接结论

当前身体 SMPL-H 链已经完成刚体初始化、VPoser 身体先验、共享 beta、COCO-17 观测和 448 帧工程运行。当前手部链已经完成 WiLoR 21 点保留、同侧双相机配对、SMPL-H 16 内部关节加 5 指尖映射、MANO PCA 12 维参数化和有限的时间正则。

当前最可靠的状态描述是：

```text
people1 的 SMPL-H + WiLoR 工程候选拟合链可以运行并输出同源网格、手部点和查看器；
身体初始化和整体朝向问题已有明确修正；
手部仍受 WiLoR 模型派生观测、鱼眼/针孔模型差异、候选关联和 priority loss 实际接入状态限制；
不能进入真实握持、真实三维精度或物理接触结论。
```

