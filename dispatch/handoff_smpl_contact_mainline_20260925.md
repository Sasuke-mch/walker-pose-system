# SMPL 手脚表面接触主线交接记录

更新时间：2026-09-25
项目根目录：`D:\my_works\walker_pose_system`
当前实验主目录：`research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1`

## 1. 交接目的

本文件用于把最近一轮 male SMPL 表面手脚接触拟合工作的状态交给下一位执行者。当前主线已经完成接触损失、分阶段拟合、图构建审计、权重校准和开发窗口复核；正在进行 `foot_a1` 的独立时间窗口验证。

所有结论都属于 engineering validation。当前没有人工接触标签、力传感器、外部三维轨迹、真实握力或承重真值，因此不得把拟合结果写成真实触地、真实握持、物理承重或真实三维精度验证。

## 2. 主线输入和数据边界

### 2.1 允许使用的输入

- 原始左右 PMPose 二维预测及其帧 ID；
- 当前双鱼眼标定和相机物理身份注册；
- 官方 male SMPL v1.0.0 6890 顶点模型和真实三角面；
- `J_regressor_coco.npy`，形状为 `17 x 6890`，用于模型侧 COCO-17 观察点；
- 官方 VPoser `V02_05`；
- 当前原始双目图像对重放得到的 Stage 1/2、动态地面变换和助步器静态拓扑；
- 当前生成的 sole/palm 表面候选顶点集合。

### 2.2 严禁读取的内容

不要读取或作为初始化复用：

- 旧 fitted parameters；
- 旧 temporal prefit 或旧 beta；
- 旧 `motion.json`；
- 旧 HTML 或旧逐帧可视化结果；
- 旧 interaction target、旧 hand/foot contact target；
- 旧逐帧助步器位姿；
- 其他实验路线的 Stage C/D 结果。

每个验证窗口必须从同一套允许输入重新完成 A/B/C，并在 C 的同一进程内进入 D。

## 3. 当前 SMPL 拟合流程

### 3.1 参数化

每帧优化：

- 32 维 VPoser latent；
- global orientation；
- translation。

全窗口共享：

- 10 维 beta。

VPoser latent 通过官方 checkpoint 解码为 body pose，再和 root orientation、translation、beta 一起前向 male SMPL。模型输出 `vertices`，再由固定 COCO regressor 计算模型侧 COCO-17。

### 3.2 阶段划分

1. **Stage A**：beta 固定为零，拟合逐帧姿态和根部变量。
2. **Stage B**：冻结逐帧运动，只优化共享 beta。
3. **Stage C**：联合微调逐帧姿态、根部变量和共享 beta；此阶段无接触损失。
4. **Stage D**：从同一进程中的 Stage C 内存继续，逐元素冻结 beta，只优化 latent、root/global orientation 和 translation，加入表面手脚项。

Stage D 的 beta 冻结是硬约束。COCO 右膝 index 14 在当前窗口没有 accepted 观测时不进入监督。

### 3.3 当前观测权重

Stage C：

```text
obs3d = 1.0
obs2d = 0.20
```

Stage D：

```text
obs3d scale = 0.70
obs2d scale = 0.020
```

这组数值来自 v5 梯度审计：原始 3D/2D 有效梯度几乎相等，因此降低 2D 系数，使 3D 在实际梯度上略高于 2D。v6 的 3D/2D 比例尚未单独重新实测，只能把它作为由 v5 审计支持的工程设计。

## 4. 表面接触损失的正式实现

### 4.1 脚部

- 左右各使用 54 个 male SMPL sole 候选顶点；
- 候选集合不使用 COCO ankle；
- 顶点先从相机坐标变换到当前固定地面坐标；
- 每只脚的候选 z 值使用中心化 soft-min；
- 对低于地面的部分加入 `penetration_weight=1.0` 的 ReLU 穿透惩罚；
- 脚部项按当前逐帧/逐侧权重归一化；
- 当前待验证路线为 `foot_a1`，即 surface foot coefficient=1.0。

脚部 residual 保留符号，不能直接用负的 signed median 计算改善比例。正确的窗口级增益是：

```python
control_mag = abs(control_foot_signed_median_mm)
route_mag = abs(route_foot_signed_median_mm)
gain = (control_mag - route_mag) / control_mag
```

同时必须保留 signed median、median absolute residual、逐点 absolute residual 和 `fraction(z<0)`。

### 4.2 手部

- 左右各使用 778 个 palm 候选顶点；
- 当前扶手端点来自本次运行动态地面坐标中的 capsule；
- 扶手半径为当前工程假设 `0.016 m`；
- palm 顶点到 capsule 中心线的距离减去半径得到 signed residual；
- 使用中心化 soft-min 和 pseudo-Huber；
- 所有帧、左右手均计算并均匀平均；
- 不使用 `hand_contact_weight`、`hand_label` 或 `hand_candidate_score` 关闭损失；这些字段只做审计；
- 不把 COCO wrist 作为表面接触点。

全帧手部项是模型假设，不等于所有帧都存在真实握持。

### 4.3 图构建审计

同一次 SMPL forward 同时得到 `result_probe.vertices` 和 `jc_probe`。表面项由 vertices 构造，再检查：

- 对 vertices 的梯度有限且非零；
- 对同一个 `jc_probe` 的梯度为 None 或全零；
- `same_forward_graph=true`；
- `coco_joints_in_surface_contact=false`；
- residual 形状为 foot `[N,54]`、hand `[N,778]`；索引形状另记为 `[54]`/`[778]`。

## 5. 已完成的修正

### 5.1 表面损失尺度修正

原 soft-min 受候选数量影响，脚部 54 点和手部 778 点不能直接比较。已改为中心化 soft-min，并加入显式穿透惩罚。

### 5.2 手部标签门控修正

原流程把 hand label/candidate score 当作是否计算损失的门控。现已删除该门控，所有帧左右手都进入 hand surface loss。

### 5.3 3D/2D 分阶段权重修正

已改成 3D 主导、2D 辅助的 staged fitting。接触只在 Stage D 加入，beta 不被接触项改变。

### 5.4 v6 脚部 gate 公式修正

原 v6 汇总对负的 signed foot median 直接计算下降比例，把 `-32.9434 -> -30.0938 mm` 错判为负收益。只读复算后正确结果是：

```text
absolute median: 32.9434 -> 30.0938 mm
toward-zero gain: 8.6501%
```

因此 `foot_a1` 在开发窗口 `60..90` 通过了预设开发门：

- 3D P95 增加 1.61%，低于 2% 门；
- 2D P95 改善 1.58 px；
- 脚部绝对中位 residual 改善 8.65%；
- 穿透比例没有超出 +1 pp；
- beta 精确冻结；
- surface graph 和梯度审计通过。

这只是开发窗口候选，不是最终冻结结果。原始 v6 comparison JSON 保留为历史记录，错误通过独立 gate reaudit 纠正，没有就地改写历史文件。

## 6. v5/v6 对照结论

### 6.1 v5

v5 在窗口 `60..90` 比较 no-contact、foot-only、hand-only、foot+hand。表面图构建通过，但接触项梯度相对观测项偏弱，路线之间的主指标和 residual 变化不足以选择权重。随后完成几何、sole 映射和 Stage C 梯度审计，未继续盲目放大权重。

### 6.2 v6

v6 比较：

- no-contact control；
- foot coefficient 1；
- foot coefficient 3；
- hand coefficient 30；
- foot 1 + hand 30；
- foot 3 + hand 60。

正确解释：

- `foot_a1` 是唯一通过开发窗口更正后六门的路线；
- `foot_a3` 虽然脚部改善更大，但 3D P95 恶化超门；
- hand-only 和 both 路线有手部 residual 改善，但穿透或 3D 门失败；
- 当前不接受任何手部系数作为正式候选；
- 当前不继续增大接触权重、不增加接触步数、不修改判定阈值。

已完成的独立验证状态：

- `v6_validation_129_159_control`：结果齐全；
- `v6_validation_129_159_foot_a1`：结果齐全；
- `v6_validation_278_308_control`：结果齐全；
- `v6_validation_278_308_foot_a1`：结果齐全；
- `v6_validation_373_403_control`：结果齐全；
- `v6_validation_373_403_foot_a1`：截至本交接尚未完成，不能汇总三窗结论。

不要把前五个目录的存在误写成独立验证已经完成。必须等第六个路线完成并进行统一复算。

## 7. 接下来必须做什么

### 7.1 完成剩余路线

完成：

```text
v6_validation_373_403_foot_a1
```

严格使用与其他五条路线相同的输入、初始化、预算和参数：

```text
surface foot = 1.0
surface hand = 0.0
obs3d = 1.0
obs2d = 0.20
Stage D obs3d scale = 0.70
Stage D obs2d scale = 0.10
contact steps = 80
```

执行时不要修改拟合入口、表面损失、接触标签、scene transforms、surface sets 或任何 v1–v6 历史产物。

### 7.2 统一复算六门

对三个独立窗口分别计算 control 与 foot_a1 的：

1. 3D P95 相对变化，门为不超过 +2%；
2. 2D P95 相对变化，门为不超过 +5 px；
3. 脚部 median absolute residual 向零改善至少 5%；
4. 左右 `fraction(z<0)` 增加均不超过 1 pp；
5. beta 精确冻结；
6. same process、same forward graph、`grad_to_coco=0`、vertices 梯度有限非零；
7. 全部数值有限，无 NaN/Inf。

必须同时保留 signed 和 absolute 统计，不能只输出一个 gain。

### 7.3 冻结或停止

只有三窗口全部通过，才可记录：

```text
engineering_validation_candidate = foot_a1
```

即使全部通过，也必须保持：

```text
physical_touch_validated = false
true_3d_accuracy_validated = false
load_bearing_validated = false
grip_force_validated = false
```

任一窗口失败时，记录：

```text
selected_candidate = null
stop_reason = foot_a1_not_stable_across_independent_windows
```

此时回到几何/坐标或优化尺度审查，不得挑选窗口、修改门槛、继续增大权重或加入手部项。

## 8. 交接时的文件入口

主线说明：

`research_records/reports/G20260924_smpl_vposer_shared_beta_v1_mainline_record.docx`

实验记录：

`research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/EXPERIMENT.md`

最新状态日志：

`AI_PROGRESS.md`

开发窗更正审计：

`research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/surface_contact_window60_90_v6_gate_reaudit.json`

独立窗口执行指令：

`dispatch/task-13-validate-foot-a1-independent-windows.md`

## 9. 交接验收要求

下一位执行者完成后必须回报：

- 六条命令的退出码；
- 三个窗口 control/foot_a1 的完整指标；
- 正确的绝对残差增益；
- 左右穿透变化；
- beta、same process、gradient audit 和 NaN/Inf 检查；
- `selected_candidate` 和 `stop_reason`；
- 修改文件绝对路径；
- 本地 Git 提交信息；
- 明确说明未修改 v1–v6、未执行 push。

不要把“结果目录生成”“可视化更接近扶手”“残差变小”单独写成接触有效。必须经过预设门，并保留工程验证边界。
