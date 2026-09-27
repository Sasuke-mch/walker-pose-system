# Cursor 执行 Prompt：修正 people1 的 SMPL-H + WiLoR 接入

你正在维护 `D:\my_works\walker_pose_system`。请严格遵守仓库根目录 `AGENTS.md`、`AI_PROGRESS.md`、`VISUALIZATION_PIPELINE.md` 和相关实验协议。不要重置、覆盖或清理用户已有改动；不要把旧结果、旧 HTML 或旧动态地面数据静默混入新运行。

## 目标

修正 people1 的 SMPL-H + WiLoR 工程候选拟合链，并按项目可视化规范重新生成结果。目标是得到可审计的工程候选，不得把结果表述为真实手部三维精度、真实握持或承重结论。

## 必须先检查的输入

重点文件：

- `realtime_app/tools/fit_smplh_wilor_sequence.py`
- `realtime_app/tools/run_wilor_sequence.py`
- `realtime_app/pose_app/fisheye_camera.py`
- `realtime_app/pose_app/smpl_coco_observation.py`
- `third_party/WiLoR/wilor/models/mano_wrapper.py`
- `third_party/WiLoR/mano_data/models/SMPLH_male.pkl`
- `research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/wilor_left_v2.jsonl`
- `research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/wilor_right_v2.jsonl`（若不存在，明确记录使用的实际文件）
- people1 原始 PMPose JSONL 和对应 `input_448pairs`
- 当前正式双鱼眼标定文件

先执行 `git status --short`，记录已有改动。读取至少一个 WiLoR JSONL 行，确认字段、图像方向、21 点顺序、是否有 `detector_confidence`、是否有逐点 valid 字段。不要根据旧注释猜关节顺序。

## 必须修正的逻辑

### 1. 固定 SMPL-H 关节顺序

根据实际 `SMPLH_male.pkl` 的 `kintree_table` 和 `smplx` 前向输出验证 52 个内部关节。当前模型的手部父子链应逐项检查：

- 左腕：20；右腕：21；
- 左手 15 个关节：22–36；
- 右手 15 个关节：37–51。

不要使用当前错误的 `[19, 21, 22, ...]` 和 `[20, 36, 37, ...]` 映射。

WiLoR/MANO 21 点必须先确认实际顺序。按当前 WiLoR `mano_wrapper.py` 的 `mano_to_openpose` 逻辑，预期顺序为：

```text
wrist,
thumb1, thumb2, thumb3,
index1, index2, index3,
middle1, middle2, middle3,
ring1, ring2, ring3,
pinky1, pinky2, pinky3,
tip_thumb, tip_index, tip_middle, tip_ring, tip_pinky
```

SMPL-H 内部关节只能对应前 16 个点；5 个指尖必须使用明确的 SMPL-H 表面顶点索引或单独定义的表面候选，不能把指尖伪装成内部关节。建议的前 16 点映射为：

```python
HAND_MAP_L = [20, 34, 35, 36, 22, 23, 24, 25, 26, 27, 31, 32, 33, 28, 29, 30]
HAND_MAP_R = [21, 49, 50, 51, 37, 38, 39, 40, 41, 42, 46, 47, 48, 43, 44, 45]
```

把这些映射写成有名称的常量，并增加父子链断言。保存 `joint_names`、映射版本和每个点的语义。

### 2. 修正 WiLoR 点顺序

当前 `hand_keep` 不能继续按数字猜测。将 WiLoR 21 点转换成统一的命名顺序，保存：

- `wrist`
- 五个手指的三个内部点
- 五个 fingertip surface 点

输出中同时保留原始 WiLoR 顺序和统一顺序，避免后续无法追溯。

### 3. 候选关联

不能只按 `candidate_index` 或缺失时默认为 0 的检测置信度选择候选。

每个相机、每一帧、每一侧手：

1. 用同一相机 PMPose 的对应腕部二维点作为候选关联参考；
2. 计算 WiLoR wrist 与 PMPose wrist 的像素距离；
3. 设置明确的工程门，例如 150 px 最大距离；
4. 第一候选和第二候选距离差小于 25 px 时标记 `ambiguous`；
5. 关联失败保留候选和失败原因，不进入拟合；
6. 检测框置信度只作为权重，不能替代人物关联；
7. 历史文件没有置信度时，保存 `confidence_source=missing_neutral_weight`，不能声称使用了模型置信度。

左右相机的候选还必须通过同一人物和同一只手的双目关联。不能把“左相机左手”和“右相机右手”直接当作同一只手的双目观测。

### 4. 逐点 mask 和权重

保留完整 21 点，但为每个点保存：

- finite；
- raw image bounds；
- depth valid；
- detector confidence；
- association status；
- projection status；
- final fitting weight；
- reject reason。

损失中不得只使用 `finite` mask。建议：

```text
finite 且边界内：正常权重
finite 但越界：低权重或只用于诊断
非 finite：不进入损失
关联不确定：不进入正式拟合
```

不能静默裁剪、插值或用 0 替代观测值。

### 5. 身体和手部采用阶段拟合

不要从零姿态一次性同时优化所有参数。实现并分别保存：

1. Stage A：固定 beta，使用身体 COCO-17 和严格身体三角化，拟合 root、translation、body pose；
2. Stage B：冻结身体 root、translation、body pose 和 beta，只优化左右手指 pose；
3. Stage C：有限开放腕部、前臂和手指；手部项保持低权重；
4. Stage D：只有通过双目关联和质量门的手部三维候选才能作为弱约束。

每阶段保存参数、损失分项、有效点数和停止原因。手部项不能明显破坏身体 COCO-17 拟合。

### 6. 身体质量权重

身体三角化的连续质量 `quality` 要参与身体损失，但不能改变严格 accepted 监督定义。保存：

- raw triangulation；
- strict accepted mask；
- continuous fitting weight；
- 每关节拒绝原因。

### 7. 初始化和数值安全

髋点缺失时不能让 `nanmean` 产生 NaN，再通过 `nan_to_num` 静默变成零坐标。逐帧处理：

- 双髋均有效：使用髋中点初始化；
- 只有一个髋有效：使用该髋并记录来源；
- 均无效：使用零姿态或明确的上一阶段初始化，并记录 `translation_init_unavailable`。

所有输出必须检查 NaN、Inf、负深度和参数边界。

### 8. 鱼眼投影边界

WiLoR 的二维点来源于其 MANO 局部三维的针孔投影。项目拟合使用 OpenCV fisheye 投影。必须在结果中明确这两种投影模型的差异；不能把手部二维残差直接解释为真实关节误差。

## 可视化要求

使用同一次运行的数据生成页面，必须包含：

- 真实 SMPL-H 6890 顶点和 13776 三角面；
- 模型侧 COCO-17 点；
- 原始三角化 COCO-17 点；
- accepted 点和 finite rejected 点的不同显示；
- COCO 骨架线，包括右膝拒绝时的拒绝样式；
- 52 个 SMPL-H 内部关节，手部单独高亮；
- WiLoR 21 点及其关联/边界状态；
- 双相机光心、轨迹和当前帧辅助线；
- 当前运行重放的 Stage 1/2、地面变换状态；
- 静态拓扑生成的实体助步器；
- XY 地面、Z 轴、实际刻度和 `[X,-Y,Z]` 显示变换；
- 播放速度 0.5×、1×、2×，分别约 66.7、33.3、16.7 ms/帧；
- OrbitControls：左键旋转，右键平移，中键/滚轮缩放，自由环绕。

页面必须使用当前重放的 scene transforms，不得使用旧 HTML、旧动态地面或旧助步器逐帧结果。

## 验证顺序

先做以下检查，再运行短窗：

```powershell
cd D:\my_works\walker_pose_system
\.venv-cuda\Scripts\python.exe -m py_compile `
  realtime_app\tools\fit_smplh_wilor_sequence.py `
  realtime_app\pose_app\smplh_hand_observation.py
```

用 5–31 帧做短窗，检查：

- 左右手映射是否逐项正确；
- 手部 2D 残差是否下降；
- 身体 COCO 残差是否明显恶化；
- 手部参数是否撞边界；
- 候选关联是否稳定；
- 双目重投影和深度是否有效。

只有短窗通过后才运行 448 帧。全片输出使用新目录，禁止覆盖 `fit_cuda_full_v3`。运行完成后更新 `EXPERIMENT.md` 和 `AI_PROGRESS.md`，写明输入路径、代码、参数、输出路径、结果和结论边界，不写 Git 哈希。

如果任何映射、候选关联或坐标系断言失败，保留失败证据并停止，不用猜测值补齐。

最后运行相关测试、检查输出数组形状和有限性，执行 `git diff`、`git status --short`，只提交本次修正涉及的文件。
