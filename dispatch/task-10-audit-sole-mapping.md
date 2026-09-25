# task-10：定位并修正脚部表面接触坐标/候选集合（先审计后决定）

## 背景

项目根目录：`D:\my_works\walker_pose_system`。

v5 几何审查已完成，输出：

`research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_geometry_audit.json`

结论：

```text
foot_geometry_status=blocked
hand_geometry_status=pass
coordinate_frame_status=pass
weight_tuning_allowed=false
next_action=geometry_fix
```

control Stage C 的 sole 顶点统计：

- 左脚 `fraction(z<0)=0.9247`，median `−27.43 mm`；
- 右脚 `fraction(z<0)=0.9552`，median `−39.69 mm`；
- 左右 sole 候选各 54 个，索引范围合法且左右集合不同；
- 扶手 capsule 长度和端点高度正常；
- 手部 median residual 约 36–40 mm，手部几何审查通过。

补充只读检查显示，control Stage C 的模型侧 COCO ankle 地面高度约为：

- left ankle median `47.19 mm`；
- right ankle median `73.13 mm`。

因此不得直接把整个地面平移，也不得未经证据把左右 sole 集合交换。

## 目标

确定脚部问题来自以下哪一类：

1. sole surface candidate 顶点与 male SMPL COCO ankle 的解剖偏移异常；
2. scene transform 的方向/单位在 surface 分支中使用错误；
3. SMPL 顶点、COCO regressor 和 scene transform 的坐标单位不一致；
4. 仅仅是当前 Stage C 姿态导致脚底未落地。

只有证据明确时才做最小修复；证据不明确时必须停在 `blocked_geometry_ambiguity`，不重跑拟合。

## 允许修改的文件

先只允许新增：

- `D:\my_works\walker_pose_system\realtime_app\tools\audit_smpl_sole_mapping_v5.py`
- `D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_sole_mapping_audit.json`
- 在 `EXPERIMENT.md` 追加审计章节；
- 在 `AI_PROGRESS.md` 追加一条日志。

在审计结论为“单一明确修复”前，禁止修改：

- `fit_vposer_shared_beta.py`；
- `smpl_surface_contact.py`；
- `contact_vertex_sets.json`；
- `scene_transforms.npz`；
- v1–v5 历史产物；
- 任何 HTML。

如果审计明确支持单一修复，才允许另外修改拟合入口，并且必须新建 v6 输出目录，不能覆盖 v5。

## 审计输入

```text
v5 control Stage C:
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_stage_d_no_contact\result_stage_c_joint.npz

v5 control Stage D:
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_stage_d_no_contact\result_stage_d_no_contact_control.npz

scene:
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\scene_stage_ground_v3_pair_replay\scene_transforms.npz

surface sets:
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_sets_v1\contact_vertex_sets.json

male SMPL model:
models\smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl

COCO regressor:
models\smpl\J_regressor_coco.npy
```

## 必须完成的审计

### 1. 单位审计

对 `vertices`、`predicted_coco`、`raw_triangulated_points`、scene translation 分别输出：

- shape；
- min/max/median；
- 单位判断依据。

若某数组中位绝对值大于 10，不能直接假设是米；必须根据入口代码和输出字段说明记录是否需要 `/1000`。不得静默换单位。

### 2. scene transform 闭环

对同一 Stage C 文件比较以下变换：

```text
T_forward(X) = R_ground_from_left @ X + T_ground_from_left
T_inverse(X) = R_ground_from_left.T @ (X - T_ground_from_left)
```

对模型侧 COCO 和 raw triangulated points 分别计算：

- ankle 地面高度；
- pelvis 地面高度；
- 3D 模型—观测误差；
- 左右 ankle 的 XY 位置。

只有当某一方向同时改善模型—观测闭环且不破坏当前 3D 误差定义，才可认为方向有证据支持。不能只看脚底 z。

### 3. sole 与 ankle 的解剖关系

用同一批 control Stage C 顶点计算：

- 每侧 sole 54 点的相对模型 COCO ankle 向量；
- z 差的 median/P05/P95；
- sole centroid 与 ankle 的 XY 距离；
- male SMPL 零 pose、当前 Stage C 两种状态分别统计。

明确检查左右候选是否与对应 COCO ankle 同侧；不能只根据 x 正负判断左右。

### 4. 候选集合完整性

读取 `contact_vertex_sets.json` 中 heel/ball/toe/palm_fingers 的来源字段，检查：

- 每个子集数量；
- 是否存在重复顶点；
- heel/ball/toe 是否包含明显非脚部索引；
- 左右集合是否是镜像/对应关系，而不是随机分组；
- 若模型文件含 faces，检查这些顶点邻接的局部连通性。

不得因为“脚底在地下”就重新手写索引。

### 5. 结论门

JSON 必须包含：

```json
{
  "status": "completed_sole_mapping_audit",
  "engineering_validation_only": true,
  "unit_status": "pass|blocked",
  "transform_direction_status": "forward_supported|inverse_supported|ambiguous",
  "left_sole_mapping_status": "pass|suspect|blocked",
  "right_sole_mapping_status": "pass|suspect|blocked",
  "single_supported_fix": "none|unit_fix|transform_fix|surface_set_fix|pose_only",
  "fit_rerun_allowed": false,
  "next_action": "stop|apply_single_fix_then_v6|audit_gradient",
  "physical_touch_validated": false,
  "true_3d_accuracy_validated": false
}
```

判定规则：

- 若单位或变换方向不能唯一确定，`single_supported_fix=none`，`fit_rerun_allowed=false`；
- 若 sole 与对应 ankle 的解剖关系异常且左右都支持同一集合问题，才允许 `surface_set_fix`；
- 若只存在统一单位问题，才允许 `unit_fix`；
- 若 forward/inverse 只改变脚 z 但破坏模型—观测 3D 闭环，不得选 `transform_fix`；
- 若所有映射和坐标均合理，只能标记 `pose_only`，但仍不得把接触权重调大，下一步进入 gradient audit。

## 运行验证

```powershell
& .venv-smplx\Scripts\python.exe -m py_compile realtime_app\tools\audit_smpl_sole_mapping_v5.py
& .venv-smplx\Scripts\python.exe realtime_app\tools\audit_smpl_sole_mapping_v5.py
Get-Content research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_sole_mapping_audit.json -Raw
```

预期：退出码 0，JSON 可解析，所有统计有限；若无法唯一确定修复，必须明确 `fit_rerun_allowed=false`。

## Git

执行前检查 `git status --short`。禁止 `git reset`、`git clean`、`git checkout`、`git push`。

只提交本任务新增脚本和记录文件；实验记录不写 Git hash。建议提交信息：

```text
review: audit SMPL sole mapping before v6 fix
```

## 回报格式

```text
状态：完成 / 受阻 / 需补充信息
改动文件：绝对路径列表
验证结果：编译、审计命令和 JSON 关键统计
审查结论：unit_status、transform_direction_status、左右 sole 状态、single_supported_fix、fit_rerun_allowed、next_action
提交：commit ID 和文件列表
问题：没有则写“无”；明确说明是否重跑拟合以及 v1–v5 是否保持不变
```
