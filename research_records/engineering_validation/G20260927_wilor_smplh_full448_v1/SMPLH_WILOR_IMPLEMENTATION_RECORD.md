# people1：SMPL-H + WiLoR 接入实现记录

## 记录范围

本文件记录 people1 当前把 WiLoR 手部候选接入 SMPL-H 的工程实现、调用路径、已运行结果、已发现错误和修正方向。记录对象是：

```text
D:\my_works\walker_pose_system
```

当前所有结果属于工程候选、内部一致性或操作性参考。没有独立人工二维标注、独立三维真值、力传感器或真实握持标签，因此不能把当前残差、重投影、网格外观或接触接近写成真实精度、真实抓握或承重结论。

## 设计目标

SMPL-H 用一个连续的 6890 顶点人体网格同时表示身体、腕部和手指姿态。原有身体主线继续使用固定的 `17 × 6890` COCO observation regressor：

```text
SMPL-H vertices → J_regressor_coco.npy → COCO-17 model points
```

WiLoR 作为手部候选来源：

```text
原始鱼眼图像
→ 旋转后的 WiLoR 输入图像
→ 人体/手部候选检测
→ WiLoR MANO 21 点和局部三维
→ 回映到原始鱼眼像素
→ SMPL-H 手部二维辅助项
```

WiLoR 的手部局部三维来自模型自身，不能直接作为项目米制三维真值。它的二维点也不是独立人工标注，而是由 MANO 局部三维经过 WiLoR 的针孔相机投影得到。

## 主要文件和作用

| 文件 | 作用 |
|---|---|
| `realtime_app/tools/run_wilor_sequence.py` | 运行 WiLoR，保存逐帧候选、21 点、778 顶点、模型局部三维和原始鱼眼投影 |
| `realtime_app/tools/fit_smplh_wilor_sequence.py` | 读取身体三角化和 WiLoR 2D 点，运行 SMPL-H 前向和优化 |
| `realtime_app/pose_app/fisheye_camera.py` | OpenCV fisheye 投影、双目外参变换和相机坐标转换 |
| `realtime_app/pose_app/smpl_coco_observation.py` | 读取 SMPL/SMPL-H 表面并回归 COCO-17 |
| `third_party/WiLoR/wilor/models/mano_wrapper.py` | WiLoR MANO 关节顺序和额外 fingertip 关节定义 |
| `third_party/WiLoR/mano_data/models/SMPLH_male.pkl` | 本地 SMPL-H male 模型 |
| `realtime_app/tools/build_smplh_people1_viewer.py` | 第一版 SMPL-H 页面生成脚本 |
| `realtime_app/tools/build_smplh_people1_reference_viewer.py` | 参考式页面生成脚本草稿，使用当前 scene replay 数据 |
| `realtime_app/pose_app/smplh_hand_observation.py` | 手部命名映射和父子链校验草稿 |

## WiLoR 调用方法

WiLoR 通过 `run_wilor_sequence.py` 加载：

```python
model = WiLoR.load_from_checkpoint(
    str(checkpoint),
    strict=False,
    cfg=cfg,
    init_renderer=False,
).to(device).eval()
```

`init_renderer=False` 是必要的 Windows 适配。官方默认初始化 `pyrender`，在当前 Windows 环境中会在加载阶段失败，但推理本身不需要渲染器。

输入方向：

- 左目目录 `left_ccw90`：运行前做相应旋转，输出再回到原始鱼眼像素；
- 右目目录 `right_cw90`：运行前做相应旋转，输出再回到原始鱼眼像素；
- 几何和三角化使用原始鱼眼像素，不能继续使用旋转后图像坐标。

WiLoR 每个候选保存：

```text
candidate_index
side
bbox_xyxy
box_center
box_size
keypoints_2d_raw_fisheye
keypoints_3d_model_local
vertices_3d_model_local
camera_translation_model
coordinate_frame_3d
pixel_frame
raw_pixel_bounds_ok
input_image_transform
focal_length_model_projection_px
```

部分新版本 JSONL 还保存 `detector_confidence` 和逐点 valid 字段；旧版 JSONL 没有检测框置信度，使用前必须检查字段，不能默认认为置信度存在。

## 当前拟合入口

原始调用结构是：

```powershell
cd D:\my_works\walker_pose_system
\.venv-cuda\Scripts\python.exe `
  realtime_app\tools\fit_smplh_wilor_sequence.py `
  --left-raw <people1-left-pmpose-raw_predictions.json> `
  --right-raw <people1-right-pmpose-raw_predictions.json> `
  --wilor-left <wilor_left.jsonl> `
  --wilor-right <wilor_right.jsonl> `
  --calibration-dir <stereo-calibration-directory> `
  --regressor <J_regressor_coco.npy> `
  --smplh-model third_party\WiLoR\mano_data\models\SMPLH_male.pkl `
  --output-dir <new-output-directory> `
  --steps 180 `
  --lr 0.02 `
  --device cuda
```

入口先读取两侧 PMPose，执行当前身体三角化，再读取 WiLoR 左右 JSONL。原始版本把一只手的 21 点保留为审计数据，然后用 16 个非 fingertip 点进入优化。

SMPL-H 初始化了：

```text
beta:         1 × 10，共享
global_orient:448 × 3
translation:  448 × 3
body_pose:    448 × 63
left_hand:    448 × 45
right_hand:   448 × 45
```

旧入口执行的是一次性联合优化：

```python
optim = torch.optim.Adam(
    [beta, root, transl, body_pose, lhand, rhand],
    lr=args.lr,
)
```

模型前向得到表面和内部关节。身体点通过 `J_regressor_coco.npy` 得到 COCO-17。左手使用左相机 fisheye 投影，右手先通过双目外参到右相机，再使用右相机 fisheye 投影：

```python
pl = fisheye_project_torch(jl, K0, D0)
pr = fisheye_project_torch(
    (R01 @ jr.transpose(1, 2)).transpose(1, 2) + T01,
    K1,
    D1,
)
```

## 已生成的候选结果

当前旧候选输出：

```text
research_records/engineering_validation/
G20260927_wilor_smplh_full448_v1/fit_cuda_full_v3/
```

数组形状：

```text
vertices:       (448, 6890, 3)
faces:          (13776, 3)
predicted_coco: (448, 17, 3)
```

保存的手部字段包括：

```text
wilor_left_2d
wilor_right_2d
wilor_left_2d_full
wilor_right_2d_full
wilor_left_valid_full
wilor_right_valid_full
wilor_left_bounds_ok_full
wilor_right_bounds_ok_full
left_hand_pose
right_hand_pose
```

旧结果最后记录的工程目标残差约为：

```text
身体项 RMS：约 378.3 mm
左手二维项 RMS：约 189.7 px
右手二维项 RMS：约 191.8 px
```

这些数字是优化目标残差，不是真实误差。旧结果状态只能是 `engineering_candidate`。

## 已发现的拟合逻辑错误

### 手部内部关节索引错误

旧代码中的：

```python
hand_map_l = [19, 21, 22, 23, 24, 25, 26, 27,
              28, 29, 30, 31, 32, 33, 34, 35]
hand_map_r = [20, 36, 37, 38, 39, 40, 41, 42,
              43, 44, 45, 46, 47, 48, 49, 50]
```

把左肘、右腕等错误关节混入左手，把左腕混入右手。按当前 SMPL-H 模型父子链，建议使用：

```python
HAND_MAP_L = [20, 34, 35, 36, 22, 23, 24,
              25, 26, 27, 31, 32, 33, 28, 29, 30]
HAND_MAP_R = [21, 49, 50, 51, 37, 38, 39,
              40, 41, 42, 46, 47, 48, 43, 44, 45]
```

这必须用模型父子链断言和单帧可视化验证后才能用于正式重跑。

### WiLoR 21 点顺序没有稳定命名

旧代码的 `hand_keep` 按数字选择点，注释却使用了另一套顺序。WiLoR 的 MANO 包装器实际把 thumb、index、middle、ring、pinky 排列在 wrist 后面。必须先统一命名，再映射到 SMPL-H。fingertip 只能是表面点，不能冒充内部骨骼关节。

### 越界点进入了损失

旧入口只用 finite mask：

```python
mask_l = torch.tensor(hand_l_mask)
mask_r = torch.tensor(hand_r_mask)
```

`bounds_ok` 只是保存和诊断，没有成为损失权重。因此有限但超出原始图像范围的点仍然以正常权重进入手部损失。正确做法是保留这些点，但给出低权重或只用于审计。

### WiLoR 置信度没有稳定接入

旧版 `wilor_left.jsonl` 和 `wilor_right.jsonl` 没有 `detector_confidence`。对缺失字段使用 `0.0` 会让候选选择退化成列表顺序。检测框置信度不能替代人物关联；应先使用 PMPose 腕部距离关联，再把置信度作为权重。缺失置信度必须在输出中标为缺失来源。

### 还没有真正接入双目手部三角化

旧损失是相机侧二维辅助项：左相机约束一侧，右相机约束另一侧。没有完成同一只手的左右候选关联、手部三角化、深度门、重投影门和三维手部约束。因此当前不能称为双目手部拟合。

### 联合优化会让错误手部项影响身体

旧代码从零姿态同时优化身体、根部、平移、手指和 beta。手部索引或候选错误时，优化会通过改变身体姿态、根部旋转和平移降低手部目标。应采用身体拟合、冻结身体拟合手指、再有限开放腕部的阶段策略。

### 鱼眼和 WiLoR 针孔投影不一致

WiLoR 点由其针孔模型产生，项目 SMPL-H 使用真实鱼眼模型投影。手部像素残差混合了投影模型差异、候选关联误差和姿态误差，不能直接解释为手指误差。

### 身体质量权重和旧输出时间不一致

后续代码草稿已加入身体连续质量权重，但 `fit_cuda_full_v3` 是此前生成的输出，不能假定它使用了新权重。重跑必须使用新输出目录，并在 summary 中记录损失定义版本。

### 平移初始化缺少髋点缺失分支

旧代码使用：

```python
pelvis_obs = np.nanmean(tri_m[:, [11, 12]], axis=1)
```

当双髋都无效时会产生 NaN。需要显式记录双髋、单髋和无髋三种初始化状态。

## 环境错误及解决方法

### Windows 加载 WiLoR 时 pyrender 初始化失败

官方 `load_wilor()` 默认初始化渲染器。当前推理不需要渲染，所以改为：

```python
WiLoR.load_from_checkpoint(..., init_renderer=False)
```

### Python 3.12 缺少 `inspect.getargspec`

旧版 `chumpy` 仍调用 `inspect.getargspec`。加载模型前加入兼容处理：

```python
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
```

### NumPy 删除旧别名

旧版 chumpy 还需要 `np.bool`、`np.int`、`np.float`、`np.object`、`np.str` 等别名。当前环境通过局部兼容层补齐，不能修改全局项目逻辑或把这些别名写入数据结果。

### WiLoR 官方 `xtcocotools` 依赖

当前 Windows/Python 3.12 下构建失败，但 WiLoR demo 和本项目入口没有实际导入该包，因此保留失败证据，不把环境标记为完整官方依赖安装。

### InterWild 依赖缺少 pytorch3d

InterWild 权重存在，但当前 Windows/Python 3.12 没有可用 wheel，因此本阶段继续使用 WiLoR，不把 InterWild 标记为已运行路线。

### SMPL-H pickle 缺少 smplx 可选 PCA 元数据

当前入口运行时补入：

```python
hands_componentsl = np.eye(45)
hands_componentsr = np.eye(45)
hands_meanl = np.zeros(45)
hands_meanr = np.zeros(45)
```

这只满足 `smplx.SMPLH(..., use_pca=False)` 的接口要求。模型本身的 posedirs、权重、关节回归器和面拓扑仍来自下载的 SMPL-H 文件。

## 当前修正状态

已经发现并写入工作区的修正草稿包括：

- `realtime_app/pose_app/smplh_hand_observation.py`：手部命名映射、指尖表面点和父子链检查；
- `realtime_app/tools/fit_smplh_wilor_sequence.py`：候选关联改为优先使用 PMPose 腕部距离，身体项加入连续质量权重；
- `realtime_app/tools/build_smplh_people1_reference_viewer.py`：参考式页面草稿，使用当前 `audit_20260928/scene` 重放数据。

这些修正尚未完成短窗重跑，因此不能把工作区中的代码草稿或旧 `fit_cuda_full_v3` 结果称为已验证修复。Cursor 执行 prompt 位于：

```text
docs/CURSOR_FIX_SMPLH_WILOR_PEOPLE1_PROMPT.md
```

## 后续正确执行顺序

```text
验证 SMPL-H 52 关节和父子链
→ 验证 WiLoR 21 点命名顺序
→ 用 PMPose 腕部做候选关联
→ 建立逐点 finite/bounds/depth/confidence 权重
→ 身体 COCO 拟合
→ 冻结身体后拟合手指
→ 有限开放腕部和前臂
→ 双目手部候选关联和三角化
→ 弱三维手部约束
→ 当前场景重放
→ 参考式可视化
→ 短窗通过后再跑全 448 帧
```
