# people1：SMPL-H + WiLoR 接入、拟合与问题审计记录

## 记录目的和证据边界

本文记录 people1 将 WiLoR 手部信息接入 SMPL-H 的完整工程路线、实际调用方式、数据结构、损失定义、阶段调度、已遇到的错误、已经完成的修正，以及 2026-09-28 对完整 448 帧结果的审计结论。

当前记录的结果属于工程验证、内部一致性和操作性参考一致性。WiLoR 的手部二维点由 MANO 模型局部三维经过模型投影得到，没有独立人工二维标注或独立三维真值；当前也没有力传感器、真实握持标签或人体三维真值。因此，重投影残差、三角化点、模型网格外观、接触代理和播放平滑度不能被表述为真实三维精度、真实握持、真实承重或临床结论。

本文以当前代码、当前输出文件和当前实验协议为准。早期记录中“只优化 16 个点”“Stage 已定义但未执行”“无 CUDA Python”等说法属于历史状态，不能用于描述当前 448 帧运行。

## 1. 总体目标

最终人体由一次 SMPL-H 前向生成连续的 6890 个表面顶点和 13776 个三角面。身体继续使用项目已有的固定 COCO-17 表面回归器，手腕和手指由 SMPL-H 的 52 关节及表面顶点表达。

```text
双鱼眼原始图像
→ PMPose 身体二维点
→ 左右人物关联
→ 原始鱼眼坐标中的双目三角化
→ COCO-17 身体观测

旋转后的 WiLoR 输入图像
→ WiLoR 人体/手候选
→ MANO 局部三维和 21 点
→ 回映到原始鱼眼像素
→ 左右相机候选关联
→ SMPL-H 手部二维辅助项

COCO-17 身体项 + VPoser 身体先验 + 手部项
→ SMPL-H 参数优化
→ 6890 顶点、真实三角面、52 内部关节和左右手 21 点
→ 同源地面/助步器/相机可视化
```

SMPL-H 不是简单把 MANO 网格拼接到 SMPL。当前本地模型保持 SMPL 的 6890 顶点和 13776 面，扩展了骨架、蒙皮权重和姿态修正。它能够表达手指弯曲；原有 SMPL 的问题是没有足够的独立手指旋转自由度。

## 2. 代码和资产

| 路径 | 作用 |
|---|---|
| `realtime_app/tools/run_wilor_sequence.py` | 对 people1 左右相机运行 WiLoR，保存候选、21 点、778 顶点、局部三维、平移和原始鱼眼投影 |
| `realtime_app/tools/fit_smplh_wilor_sequence.py` | 读取 PMPose 三角化和 WiLoR 观测，执行 SMPL-H 前向和分阶段拟合 |
| `realtime_app/pose_app/smplh_hand_observation.py` | SMPL-H 52 关节父子链检查、WiLoR 顺序适配、16 个内部关节加 5 个表面指尖 |
| `realtime_app/pose_app/fisheye_camera.py` | 鱼眼投影、左右相机外参变换、标定加载 |
| `realtime_app/pose_app/smpl_coco_observation.py` | 固定 `17×6890` COCO 表面回归器 |
| `realtime_app/pose_app/smplx_fitting.py` | 显式加载 VPoser V02_05 |
| `third_party/WiLoR/wilor/models/mano_wrapper.py` | WiLoR MANO 输出顺序和 fingertip 定义 |
| `third_party/WiLoR/mano_data/models/SMPLH_male.pkl` | 当前使用的 SMPL-H male 模型 |
| `models/VPoser02_05/V02_05` | VPoser 配置和 checkpoint |
| `realtime_app/tools/build_smplh_people1_reference_viewer.py` | 用当前拟合结果和当前 scene replay 生成参考式交互查看器 |
| `VISUALIZATION_PIPELINE.md` | 网格、坐标、地面、相机、助步器和交互播放规范 |

模型资产检查要求：`J_regressor=(52,6890)`、父节点数组长度为 52、三角面为 `(13776,3)` 且索引不超过 6889。SMPL-H pickle 缺少 smplx 可选的 PCA 元数据时，运行时补入 45 维单位手部分量和零均值，并使用 `use_pca=False`；这不改变下载文件中的 posedirs、蒙皮权重、关节回归器和三角拓扑。

## 3. WiLoR 调用与坐标流程

WiLoR 通过 `run_wilor_sequence.py` 加载。Windows 下必须关闭不参与推理的 pyrender 初始化：

```python
model = WiLoR.load_from_checkpoint(
    str(checkpoint), strict=False, cfg=cfg, init_renderer=False
).to(device).eval()
```

左目推理使用 `left_ccw90`，右目推理使用 `right_cw90`。图像旋转只服务于 WiLoR 输入；模型输出必须回映到原始鱼眼像素，后续人物关联、鱼眼投影和三角化不能使用旋转图像坐标。

每个候选至少保存：`frame_index`、`image`、`side`、`candidate_index`、`bbox_xyxy`、`box_center`、`box_size`、`keypoints_2d_raw_fisheye(21×2)`、`keypoints_3d_model_local(21×3)`、`vertices_3d_model_local(778×3)`、`camera_translation_model`、`pixel_frame`、`raw_pixel_bounds_ok` 和输入变换信息。历史 JSONL 没有逐关节 WiLoR confidence，不能伪造该字段；缺失检测框 confidence 时，记录 `confidence_source=missing_neutral_weight`，并使用候选关联和边界状态构造工程权重。

项目的双目约定为：

```text
X_cam1 = R_cam0_to_cam1 @ X_cam0 + T_cam0_to_cam1
cam0 = LEFT
cam1 = RIGHT
```

SMPL-H 的手部模型点若位于左相机坐标系，左目直接使用 `K0,D0` 投影，右目先使用 `R_cam0_to_cam1,T_cam0_to_cam1` 变换，再使用 `K1,D1` 鱼眼投影。三角化得到的身体点单位为毫米，进入 SMPL-H 损失前除以 1000 转为米。

## 4. WiLoR 21 点的真实语义

WiLoR 的 `mano_to_openpose` 为：

```python
[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6,
 18, 10, 11, 12, 19, 7, 8, 9, 20]
```

它的输出语义是：

```text
0       wrist
1..4    thumb_1, thumb_2, thumb_3, thumb_tip
5..8    index_1, index_2, index_3, index_tip
9..12   middle_1, middle_2, middle_3, middle_tip
13..16  ring_1, ring_2, ring_3, ring_tip
17..20  pinky_1, pinky_2, pinky_3, pinky_tip
```

因此内部关节索引为 `[0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]`，表面指尖索引为 `[4,8,12,16,20]`。指尖不是 MANO/SMPL-H 的内部骨骼关节，不能把它们伪装成关节回归器输出。

SMPL-H 侧的命名映射为：

```python
HAND_JOINTS["left"] = [20,34,35,36, 22,23,24, 25,26,27, 31,32,33, 28,29,30]
HAND_JOINTS["right"] = [21,49,50,51, 37,38,39, 40,41,42, 46,47,48, 43,44,45]
```

这两个列表按 `wrist, thumb, index, middle, ring, pinky` 的内部关节顺序排列。`hand21()` 先取 16 个 SMPL-H 内部点，再插入 5 个手指尖表面顶点，使模型输出与 WiLoR 的 21 点顺序一致。运行时通过 `validate_smplh()` 检查每根手指的父子链 `[wrist,a,b]`。

## 5. 人物和手部候选关联

左右相机的 JSONL 都包含人体解剖意义上的 left/right 候选。不能按 `candidate_index` 跨相机直接配对，也不能把“左相机检测到的手”当成“左手”。每个相机分别用同一相机的 PMPose 腕部点做候选关联：

1. 只比较相同解剖 side 的候选；
2. 使用 WiLoR 第 0 点 wrist 与 PMPose 同侧 wrist 的像素距离排序；
3. 最近候选距离大于 150 px 时拒绝；
4. 最近和第二近候选距离差小于 25 px 时标记歧义并拒绝；
5. 身体腕部不可用时不声称完成可靠关联；
6. 记录候选选择、距离、拒绝原因、逐点 finite、逐点 bounds 和 confidence 来源。

完整 21 点必须保留在输出中。有限但越界的点保留用于审计，进入优化时使用较低软权重；非有限点不产生梯度。单个手指越界不能静默删除整只手。

当前 hand fit 使用左右相机对同一解剖手的二维约束：左手由左目和右目共同约束，右手也由左目和右目共同约束。它仍然是两个相机的二维辅助项；若要称为独立的手部三维观测，必须另外完成跨相机同手候选关联、手部射线三角化、深度/夹角/重投影质量门和逐点三维保存。

## 6. SMPL-H 参数和 VPoser

优化变量为：

```text
beta        1×10，共享人体形状
root        T×3，全局轴角 global_orient
transl      T×3，逐帧平移
latent      T×32，VPoser 潜变量
body_pose   T×63，由 VPoser(latent) 解码
lhand       T×45，左手 15 个关节轴角
rhand       T×45，右手 15 个关节轴角
```

VPoser 只负责身体的 21 个 body joints，输出 `32 → 63`；不能把手指 90 维参数塞入 VPoser，也不能认为 VPoser 会自动解决人体在相机坐标系中的全局朝向。SMPL-H 的手部仍通过左右各 45 维局部轴角控制。

VPoser 的维度、有限值、checkpoint 和 latent 梯度已检查。语义顺序还必须通过零姿态下 SMPL 与 SMPL-H 的身体关节对照、双腕位置和前向结果核查；仅检查 `latentD=32` 不能证明所有关节语义已经完全一致。

## 7. 当前拟合阶段

当前入口 `fit_smplh_wilor_sequence.py` 的阶段调度为：

| 阶段 | 训练变量 | 身体项 | 手部项 | 接触项 |
|---|---|---|---|---|
| A `A_body_vposer` | `root, transl, latent` | 开启 | 关闭梯度 | 关闭 |
| B `B_shared_beta` | `beta` | 对 beta 保留梯度 | 关闭梯度 | 关闭 |
| C `C_body_vposer_refine` | `beta, root, transl, latent` | 开启 | 关闭梯度 | 关闭 |
| D1 `D1_hand_proximal` | `lhand, rhand` | 冻结 | 开启 | 关闭 |
| D2 `D2_hand_surface_contact` | `lhand, rhand` | 冻结 | 保持 | 有效标签时开启 |
| D3 `D3_contact_refine` | `lhand, rhand, root, transl` | 低权重保护 | 保持 | 有效标签时开启 |

手部 21 点使用固定的由近及远权重：腕部最高，近端指骨次之，远端指骨和指尖较低。所有点在每一步都参与，权重改变的是拟合重点，不是分阶段删除关节。这个设计与 VPHOJoint 文本提出的视觉/物理信息分层和由近端到远端聚合思路一致；当前工程没有把该论文中的网络训练方法、公式或独立标签假装已经实现。

身体项是 SMPL-H 表面经固定 `J_regressor_coco.npy` 回归的 COCO-17 与双目三角化点之间的距离，使用三角化质量的连续权重。手部项是 SMPL-H `hand21()` 点投影到左右鱼眼相机后的二维残差。WiLoR 的局部三维没有作为项目米制三维真值使用。

D2 接触项只能使用 SMPL-H 掌面表面顶点、当前地面变换和助步器把手端点胶囊。不能用 COCO wrist 代替掌面顶点，也不能把 hand contact label 的零权重解释成接触成功。接触输入需检查帧数、形状、有限值、非负权重、顶点数和顶点索引；所有 hand contact weight 为零时应拒绝接触模式或明确写入 `contact_active=false`。

## 8. 实际调用命令

people1 当前完整 448 帧运行使用：

```powershell
D:\my_works\walker_pose_system\.venv-cuda\Scripts\python.exe `
  realtime_app\tools\fit_smplh_wilor_sequence.py `
  --left-raw research_records\engineering_validation\V20260908_people0_1_2_pmpose_c3_chain\people_1\c3_predictions\left\pmpose\raw_predictions.json `
  --right-raw research_records\engineering_validation\V20260908_people0_1_2_pmpose_c3_chain\people_1\c3_predictions\right\pmpose\raw_predictions.json `
  --wilor-left research_records\engineering_validation\G20260927_wilor_smplh_full448_v1\wilor_left.jsonl `
  --wilor-right research_records\engineering_validation\G20260927_wilor_smplh_full448_v1\wilor_right.jsonl `
  --calibration-dir realtime_app\calibration\results `
  --regressor models\smpl\J_regressor_coco.npy `
  --smplh-model third_party\WiLoR\mano_data\models\SMPLH_male.pkl `
  --vposer-dir models\VPoser02_05\V02_05 `
  --output-dir research_records\engineering_validation\G20260927_wilor_smplh_full448_v1\fit_cuda_vposer_corrected_stereo448_v1 `
  --steps 180 --lr 0.02 --device cuda
```

该命令每个默认阶段运行 30 步，总计 180 步。正式运行必须使用新输出目录，不能覆盖旧 `fit_cuda_full_v3`，也不能混用旧网格、旧参数或旧 HTML。

## 9. 已遇到的环境和接口错误

### 9.1 Windows 加载 WiLoR 时 pyrender 失败

官方加载器默认初始化 renderer，Windows 下在加载阶段失败，而推理不需要 renderer。解决方案是 `init_renderer=False`，没有改变 checkpoint 和模型前向。

### 9.2 Python 3.12 的 `inspect.getargspec`

旧版 chumpy 访问已经删除的 `inspect.getargspec`。入口在导入相关模型前提供局部兼容别名：

```python
inspect.getargspec = inspect.getfullargspec
```

### 9.3 NumPy 旧别名

旧依赖访问 `np.bool`、`np.int`、`np.float`、`np.object`、`np.str` 等别名。入口只在运行时提供兼容别名，没有修改实验数据和项目算法。

### 9.4 `xtcocotools` 在 Windows/Python 3.12 构建失败

WiLoR 当前 demo 和本项目入口没有实际导入该包，因此保留失败证据并继续运行必要路径，不能把环境宣称为完整官方依赖安装。

### 9.5 InterWild 缺少可用 pytorch3d wheel

InterWild 权重存在，但当前 Windows/Python 3.12 环境没有可直接安装的 pytorch3d wheel。本阶段使用 WiLoR，不能把 InterWild 写成已运行替代路线。

### 9.6 SMPL-H pickle 缺少 PCA 元数据

`SMPLH_male.pkl` 没有 smplx 可选的 `hands_componentsl/r` 和 `hands_meanl/r`。使用 `use_pca=False` 时运行时补入 45 维单位矩阵和零均值，使 smplx 能建立模型；该补丁不替换模型的拓扑和 posedirs。

### 9.7 早期错误的左右手读取

历史代码曾将右目 JSONL 以 `left` side 读取，造成右手统计异常。现已改为每个相机读取两个解剖 side，并按同相机 PMPose 腕部做关联。

### 9.8 早期错误的手部索引和形状

历史代码使用过 `[19,21,...]` 等错误 SMPL-H 索引，也曾将 21 点观测裁成 16 点而模型侧输出 21 点，导致损失形状不一致。现在由命名常量和父子链校验生成完整 `21×3` 模型点，指尖由表面顶点提供。

### 9.9 早期阶段调度没有真正执行

历史版本只定义了 `stage_train` 和 `set_stage()`，主循环仍一次性训练全部参数。当前版本在每个阶段显式设置 `requires_grad`、建立当前阶段 optimizer，并记录 `stage_schedule`。

### 9.10 Stage A/B 形状逻辑错误

历史 Stage A 提前训练 beta，Stage B 又切断 body loss 对 beta 的梯度，导致共享形状无法从身体观测学习。当前 Stage A 固定 beta 为零，Stage B 冻结运动参数并保留身体项对 beta 的梯度，Stage C 再低学习率联合微调。

### 9.11 接触输入静默无效

历史实现可能在 hand contact weight 全零时仍显示“接触阶段”。当前对接触数组执行非零权重检查；当前 people1 接触标签的 hand contact weight 全为零，所以本次 448 帧运行没有接触梯度，D2/D3 的 `contact_active` 为 false，不能称为完成接触拟合。

### 9.12 平移初始化的 NaN 风险

历史代码可能使用 `nanmean` 后再静默归零。当前显式记录双髋中点、单髋和无髋三种来源；无髋时标记 `translation_init_unavailable`，不把缺失观测伪装成可靠初始化。

## 10. 2026-09-28 完整运行结果

输出目录：

```text
research_records/engineering_validation/
G20260927_wilor_smplh_full448_v1/
fit_cuda_vposer_corrected_stereo448_v1/
```

数组检查：

```text
vertices             (448, 6890, 3)
faces                (13776, 3)
predicted_coco       (448, 17, 3)
smplh_joints         (448, 73, 3)
hand_points_left     (448, 21, 3)
hand_points_right    (448, 21, 3)
raw_triangulated     (448, 17, 3)
```

所有模型数组有限，面索引最大值为 6889。三角化点中有限且 accepted 的点为 7534，本次结果没有 finite-but-rejected 点。WiLoR 左右手有限点统计分别为 14343 和 16947，这是两个相机视角累计的观测点数，不是独立真值点数。

本次 summary 的最终阶段损失对应的身体项约 0.504 m 量级，单独按 accepted COCO 点计算的身体 RMS 约 519 mm；手部二维项约为左 1092 px、右 699 px。结果网格虽然完整生成，拟合姿态仍明显偏离自然人体，不能进入真实手部姿态、握持或承重判断。

## 11. 当前最严重的拟合逻辑问题：全局朝向初始化

当前代码曾将每帧 `global_orient` 初始化为零轴角：

```python
root = torch.nn.Parameter(torch.zeros(n, 3, device=device))
```

同时，平移初始化使用零旋转模型的髋部位置：

```python
z = model(global_orient=root, body_pose=zeros, transl=zeros)
zc = regress_coco17_torch(z.vertices, reg)
t0 = observed_hip - zc_hip
```

之后 root 在优化中大幅变化，但 translation 没有按新的根旋转重算。这会让优化器使用 root、translation、VPoser latent 互相补偿。当前可视化中的人体半朝天、抖动和与三角化骨架不重合，首先应按初始化问题排查，不能只增加步数或调手部权重。

正确初始化顺序应为：

1. 用三角化的肩膀和髋部建立观测人体基；
2. 用零姿态 SMPL-H 的 COCO 肩髋点建立模板人体基；
3. 求 `R_global = R_observation @ R_model.T`；
4. 将 `R_global` 转为 axis-angle 初始化 root；
5. 用旋转后的模板髋中点计算 translation；
6. 先固定 root 的大方向，优化身体平移和 VPoser latent；
7. 再低学习率释放 root，并保留身体观测保护项。

人体基必须检查左右方向、竖直方向和行进方向的叉乘顺序及行列约定。不能只凭“肩髋向量看起来相似”确认坐标轴已经对齐。

## 12. 其他仍需验证的问题

### 12.1 VPoser 语义顺序

当前已验证 `latentD=32` 和输出形状 `63`，但维度兼容不等于语义完全兼容。需要用同一零姿态分别送入 SMPL 与 SMPL-H，比较 21 个身体关节、双腕位置和 COCO-17 表面回归结果；若身体关节顺序或轴向有差异，必须增加明确的关节重排/旋转适配。

### 12.2 WiLoR 针孔投影与鱼眼投影失配

WiLoR 21 点由其针孔相机假设产生，而拟合器使用真实鱼眼 `K,D` 投影。当前手部像素残差混合了模型投影差异、鱼眼边缘失配、候选关联误差和手部姿态差异，不能直接解释为手指关节误差。

### 12.3 D3 的身体保护仍需量化

D3 允许手部、root 和 translation 同时变化，虽有低权重 COCO 保护项，但必须记录 D2 到 D3 的身体 COCO RMS、root 旋转变化、translation 变化、手部二维残差和接触残差。没有这些对照，不能声称 D3 只做了安全的局部微调。

### 12.4 手部三维项尚未形成独立观测

当前拟合没有把 WiLoR 的局部三维直接当作米制点，也没有将左右同手候选三角化结果作为统一三维手部项。后续若加入三维项，必须保留深度、射线夹角/间隙、左右重投影误差、候选来源和逐点接受状态，并使用弱 guardrail 或初始化项。

## 13. 正确的后续执行顺序

```text
单帧模板 SMPL-H 前向检查
→ 肩髋人体基和 root 刚体初始化
→ 旋转一致的髋部 translation 初始化
→ A0 固定身体形状和大方向
→ A 身体 VPoser 拟合
→ B 仅共享 beta
→ C 身体联合细化
→ 冻结身体拟合 D1 手部
→ D2 加入有效掌面接触
→ D3 小范围开放腕部/前臂或 root
→ 双目同手候选三角化审计
→ 弱三维手部 guardrail
→ 短窗通过后再跑完整 448 帧
→ 使用同源结果和同源 scene replay 可视化
```

每个阶段应保存起止参数和分项损失。身体拟合通过条件至少包括：模型侧 COCO 点与三角化骨架在同一坐标系、肩髋方向正确、身体 RMS 不出现数量级异常、根旋转不发生逐帧跳变、三角化点和模型点没有被查看器单独删除。手部阶段还要检查由腕部到指尖的误差分布、左右候选稳定性、指尖表面点的来源和鱼眼投影边缘分布。

## 14. 可视化要求和当前页面

查看器必须读取同一次拟合的 `vertices`、`faces`、`predicted_coco`、`raw_triangulated_points`、`body_accepted`、`smplh_joints`、左右手 21 点，并读取当前 scene replay 的地面、Stage、助步器拓扑和左右相机轨迹。

页面应满足：

- 真实 SMPL-H 6890 顶点和 13776 三角面；
- 原始三角化 COCO-17 骨架，accepted 与 rejected 使用不同样式；
- 模型侧 COCO-17 点；
- 52 个内部关节和左右手 21 点；
- `[X,-Y,Z]` 显示变换，Z 为竖直方向；
- XY 地面、实际长度坐标轴和刻度；
- 实体助步器、两台相机光心和轨迹；
- 逐帧访问原始输出，不插值人体、相机或助步器；
- 0.5×、1×、2× 播放，OrbitControls 自由旋转、平移和缩放；
- 页面背景保持浅色，只有地面平面使用明显深色；
- 页面标签明确标注 `engineering_candidate`、WiLoR model-derived 和 contact_active 状态。

当前同源页面为：

```text
research_records/engineering_validation/
G20260927_wilor_smplh_full448_v1/
fit_cuda_vposer_corrected_stereo448_v1/
smplh_people1_vposer_viewer.html
```

它已包含网格、真实面、模型/三角化 COCO 点、SMPL-H 52 内部关节、双手 21 点、手指连接线、场景地面、助步器和双相机轨迹。页面可以作为诊断工具，但当前拟合姿态明显错误，不能用“页面能显示”替代拟合通过。

## 15. 最终结论

当前已经完成 SMPL-H 模型接入、VPoser 身体参数化、完整 WiLoR 21 点保留、左右相机的解剖手关联入口、分阶段优化框架、模型输出落盘和同源可视化。已经修复的历史问题包括错误手部关节映射、21/16 点形状不一致、右目错误 side 读取、候选关联过松、Stage A/B 形状梯度错误、接触输入校验不足和平移 NaN 静默处理。

完整 448 帧运行证明的是工程链路可以执行和输出结构完整。它没有通过人体姿态可用门：身体三维残差约 519 mm，手部二维残差仍很大，且人体整体朝向和逐帧稳定性明显不符合要求。当前最优先的修正是模板到三角化骨架的全局刚体初始化，以及旋转一致的 translation 初始化；VPoser 只能约束局部身体姿态，不能自动修正这两个坐标问题。

在完成单帧初始化验收、VPoser/SMPL-H 身体关节语义验收和短窗身体拟合验收之前，不应继续通过增加手部权重、接触权重、时间平滑或全片步数来掩盖人体整体错位。
