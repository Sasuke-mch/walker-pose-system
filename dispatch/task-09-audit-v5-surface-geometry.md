# task-09：v5 表面接触几何与坐标审查（禁止继续调权重）

## 背景

项目根目录：`D:\my_works\walker_pose_system`。

v5 已完成 7 条 Stage D 路线。代码已经做了：

- 3D/2D delta 归一化；
- Stage A/B/C 使用 3D=1.0、2D=0.25；
- Stage D 使用 3D=0.70、2D=0.025；
- centered soft-min；
- 手脚穿透惩罚；
- beta 冻结和同 forward 图审计。

v5 结果的关键事实：

- 零接触控制组脚部候选点穿透比例约：左 96.36%，右 95.998%；
- 零接触控制组脚部 surface residual median 约 −41.57 mm，P95 仍为负；
- 零接触控制组手部 residual median 约 43.66 mm；手部穿透比例约 4.49%/4.87%；
- foot 权重 0.10 后脚部 median 只改善约 1%；
- hand 权重 0.10 后手部 median 几乎不变；
- 没有路线满足 residual median 至少改善 5%，`recommended_candidate=null`。

因此当前不能继续增加接触权重。必须先判断接触几何是否正确。

## 本任务目标

只做只读几何/坐标审查，不重跑拟合，不修改已有拟合入口和损失实现，不生成新的拟合结果。

## 允许修改或新增

只允许：

1. 新建 `D:\my_works\walker_pose_system\realtime_app\tools\audit_surface_contact_geometry_v5.py`；
2. 新建 `D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_geometry_audit.json`；
3. 在 `D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\EXPERIMENT.md` 追加本次审查章节；
4. 在 `D:\my_works\walker_pose_system\AI_PROGRESS.md` 追加一条日志。

禁止修改：

- `fit_vposer_shared_beta.py`；
- `smpl_surface_contact.py`；
- v1/v2/v3/v4/v5 已有任何结果目录和 JSON；
- contact labels、scene transforms、surface sets、walker topology；
- 任何 HTML 或可视化页面。

## 审查输入

使用以下现有文件：

```text
v5 control:
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_stage_d_no_contact\result_stage_c_joint.npz
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_stage_d_no_contact\result_stage_d_no_contact_control.npz

v5 contact:
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_both_a010\result_stage_c_joint.npz
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_both_a010\result_stage_d_surface_foot_hand.npz

scene:
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\scene_stage_ground_v3_pair_replay\scene_transforms.npz

surface sets:
research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_sets_v1\contact_vertex_sets.json

walker topology:
research_records\engineering_validation\G20260918_offline_walker_frame_evidence_v1\coarse_complete_model_v2_camera_rail\coarse_walker_model.json
```

## 必须检查的内容

### 1. 地面坐标和脚部高度

对 control 的 Stage C 和 Stage D、both_a010 的 Stage C 和 Stage D 分别检查：

- `vertices` 形状 `[31,6890,3]`；
- 使用 scene transforms 的 `rotation_ground_from_left` 和 `translation_ground_from_left_mm/1000` 转换到地面系；
- 对左右 sole 索引分别计算全部候选点的 `z_ground`；
- 输出 min、median、P05、P95、max；
- 输出 `fraction(z<0)`、`fraction(z<-0.010)`、`fraction(z<-0.025)`；
- 输出每帧 soft-min 高度和每帧负点比例。

### 2. 扶手 capsule 几何

从静态 topology 的 `nodes_left_camera_mm` 和当前 scene transform 重建每帧左右扶手端点。检查：

- 每帧端点有限；
- capsule 长度 min/median/max；
- 半径固定假设 `0.016 m`；
- palm 点到 capsule 中心线距离的 min/median/P05/P95/max；
- `distance-radius` 的 min/median/P05/P95/max；
- 负 residual 比例；
- 左右扶手是否在合理高度范围内。

### 3. 顶点索引和左右侧语义

检查：

- sole left/right 均为 54 个顶点；
- palm left/right 均为 778 个顶点；
- 所有索引范围在 `[0,6889]`；
- 左右集合无意外完全相同；
- 逐帧转换后左右脚/手的空间统计不发生异常交换。

### 4. Stage C 到 Stage D 变化

对 control 和 both_a010 分别输出：

- 顶点坐标变化 max-abs、L2；
- sole z 的变化统计；
- hand residual 的变化统计；
- Stage C/D 地面变换不应被重新读取或替换；本审查只使用同一个 scene transforms。

### 5. 结论分类

JSON 必须写明以下字段：

```json
{
  "status": "completed_geometry_audit",
  "engineering_validation_only": true,
  "foot_geometry_status": "pass|suspect|blocked",
  "hand_geometry_status": "pass|suspect|blocked",
  "coordinate_frame_status": "pass|suspect|blocked",
  "weight_tuning_allowed": false,
  "next_action": "geometry_fix|gradient_audit|stop",
  "physical_touch_validated": false,
  "load_bearing_validated": false,
  "true_3d_accuracy_validated": false
}
```

判定规则：

- 若 control Stage C 脚部 `fraction(z<0)` 左右均超过 0.90，`foot_geometry_status=blocked`，`next_action=geometry_fix`，禁止继续调脚权重；
- 若 hand median residual > 0.040 m 且 hand negative fraction < 0.10，`hand_geometry_status=suspect`，优先检查扶手端点/半径/全帧手部假设，禁止继续增大手权重；
- 只有脚部和手部几何均未 blocked，才允许下一步做 Stage C 起点梯度审查；
- 不能把“穿透比例高”解释成真实触地失败，必须说明这是当前地面变换与 SMPL 表面之间的工程一致性问题。

## 运行与验证

在项目根目录执行：

```powershell
& .venv-smplx\Scripts\python.exe -m py_compile realtime_app\tools\audit_surface_contact_geometry_v5.py
& .venv-smplx\Scripts\python.exe realtime_app\tools\audit_surface_contact_geometry_v5.py
Get-Content research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_geometry_audit.json -Raw
```

预期：脚本退出码 0；JSON 可解析；所有统计有限；如果脚部超过 90% 穿透，必须输出 `foot_geometry_status=blocked`、`weight_tuning_allowed=false`、`next_action=geometry_fix`。

## 实验记录

在 EXPERIMENT.md 和 AI_PROGRESS.md 只追加本次几何审查的：

- 输入路径；
- 检查公式；
- 关键统计；
- 判定；
- 为什么停止继续调权重；
- 下一步是 geometry fix 还是 gradient audit。

不要写 Git hash 到实验记录。

## Git

执行前检查 `git status --short`。不得执行 `git reset`、`git clean`、`git checkout`、`git push`。

只提交本任务新增脚本和两份记录（以及必要的 JSON），不要加入模型、原始数据、HTML、v5 旧结果或其他 agent 的改动。提交信息：

```text
review: audit v5 surface contact geometry before weight tuning
```

## 回报格式

```text
状态：完成 / 受阻 / 需补充信息
改动文件：绝对路径列表
验证结果：编译、审查脚本、JSON 关键统计
审查结论：foot_geometry_status、hand_geometry_status、coordinate_frame_status、weight_tuning_allowed、next_action
提交：commit ID 和文件列表
问题：没有则写“无”；明确说明没有重跑拟合、没有修改 v1-v5 历史产物
```
