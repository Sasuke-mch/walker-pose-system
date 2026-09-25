# task-11：Stage C 起点表面接触梯度审计（不改权重、不跑 Stage D）

## 背景

项目根目录：`D:\my_works\walker_pose_system`。

v5 表面接触与 sole 映射审计已完成：

- 3D/2D 已 delta 归一化，Stage A/B/C 使用 3D=1.0、2D=0.25；
- Stage D 使用 3D=0.70、2D=0.025；
- centered soft-min 和手脚穿透项已实现；
- control Stage C 脚部穿透比例高，但单位、forward/inverse 变换和左右 sole 集合审计没有支持单一几何修复；
- sole–ankle 相对高度与 zero-pose 同量级；
- `single_supported_fix=none`，`fit_rerun_allowed=false`，下一步为 `audit_gradient`。

当前任务只回答一个问题：在同一个 Stage C 参数状态下，3D、2D、脚部 surface、手部 surface 对可训练变量的梯度大小分别是多少？

## 严格范围

本任务不得：

- 修改任何损失公式；
- 修改任何权重默认值；
- 运行 Stage D；
- 生成新的拟合结果；
- 修改 v1–v5 历史输出；
- 修改 sole/palm vertex sets、scene transforms、walker topology 或 HTML。

允许修改：

1. `D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py`：只新增 `--gradient-audit-only` 分支和梯度审计函数；正常拟合路径必须保持不变；
2. 新建 `D:\my_works\walker_pose_system\realtime_app\tools\summarize_stage_c_gradient_audit.py`；
3. 新建两份审计输出 JSON；
4. `EXPERIMENT.md` 和 `AI_PROGRESS.md` 只追加记录。

## 新增入口行为

在 `fit_vposer_shared_beta.py` 新增：

```text
--gradient-audit-only
```

行为必须是：

1. 正常从原始 PMPose、标定、male SMPL、regressor、VPoser、当前 scene/contact/surface 输入开始；
2. 正常完成 Stage A、B、C；
3. 保存或构造 Stage C 参数状态；
4. 不进入 Stage D，不调用 `opt.step()` 进行任何 Stage D 更新；
5. 对以下四项分别求梯度：
   - `obs3d_term = obs3d_coeff * l3`
   - `obs2d_term = obs2d_coeff * l2`
   - `foot_term = lfoot`
   - `hand_term = lhand`
6. 对每一项分别对以下变量求梯度：
   - `latent`
   - `root`
   - `transl`
7. beta 必须保持 `requires_grad=False`，不计算 beta 梯度；
8. `allow_unused=True`，None 记录为 0；所有梯度必须有限；
9. 只写审计 JSON 并正常退出。

建议实现独立函数：

```python
def gradient_norms(term, params):
    """Return finite L2 and max-abs gradient per named trainable parameter."""
```

每项必须记录：

```json
{
  "term_value": 0.0,
  "coeff": 0.0,
  "gradient_l2": {
    "latent": 0.0,
    "root": 0.0,
    "transl": 0.0,
    "all": 0.0
  },
  "gradient_max_abs": {
    "latent": 0.0,
    "root": 0.0,
    "transl": 0.0,
    "all": 0.0
  },
  "finite": true
}
```

注意：四个 term 必须来自同一个 Stage C 参数状态；不能为每项重新 forward 或重新优化。每次 `autograd.grad` 要正确使用 `retain_graph=True`，并在最后一项后释放图。

## 两次审计运行

使用同一个窗口 `60..90`、同一输入白名单和同一初始化。每次都必须保存完整 `command.txt`，输出只写审计 JSON，不生成 Stage D 结果目录。

### 审计 1：surface contact graph

输出：

`research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_gradient_audit_surface.json`

命令参数：

```powershell
& .venv-smplx\Scripts\python.exe -u research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py `
  --output-dir research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_gradient_audit_surface_run `
  --left research_records\engineering_validation\V20260908_people0_1_2_pmpose_c3_chain\people_1\c3_predictions\left\pmpose\raw_predictions.json `
  --right research_records\engineering_validation\V20260908_people0_1_2_pmpose_c3_chain\people_1\c3_predictions\right\pmpose\raw_predictions.json `
  --calibration-dir realtime_app\calibration\results `
  --model models\smpl\basicmodel_m_lbs_10_207_0_v1.0.0.pkl `
  --regressor models\smpl\J_regressor_coco.npy `
  --vposer-dir models\VPoser02_05\V02_05 `
  --contact-labels research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\contact_labels_stage_audit_v2\contact_labels.npz `
  --scene-transforms research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\scene_stage_ground_v3_pair_replay\scene_transforms.npz `
  --contact-vertex-sets research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_sets_v1\contact_vertex_sets.json `
  --walker-topology research_records\engineering_validation\G20260918_offline_walker_frame_evidence_v1\coarse_complete_model_v2_camera_rail\coarse_walker_model.json `
  --start 60 --end 90 --foot-contact-weight 0.0 --hand-contact-weight 0.0 `
  --surface-foot-contact-weight 0.10 --surface-hand-contact-weight 0.10 `
  --contact-steps 80 --obs-3d-weight 1.0 --obs-2d-weight 0.25 `
  --stage-d-obs-3d-scale 0.70 --stage-d-obs-2d-scale 0.10 `
  --gradient-audit-only --device cpu
```

### 审计 2：no-surface comparison

输出：

`research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_gradient_audit_nosurface.json`

使用完全相同命令，但：

```text
--surface-foot-contact-weight 0.0
--surface-hand-contact-weight 0.0
```

并仍然传入 `--contact-vertex-sets`、`--walker-topology`、`--contact-labels`、`--scene-transforms`，以保证输入路径和 Stage A/B/C 完全一致；该运行仍然只做 gradient audit，不进入 Stage D。

## 梯度摘要工具

新增 `summarize_stage_c_gradient_audit.py`，读取上面两份 JSON，输出：

- each term value；
- each term gradient L2/max-abs；
- `surface_to_observation_gradient_ratio`；
- no-surface 与 surface audit 的 Stage C 参数/term 一致性；
- all finite。

两次审计的 Stage C 观测项梯度必须一致到 `1e-6` 相对误差以内；如果不一致，说明 audit 分支改变了 Stage C，必须停止。

## 判定规则

只生成诊断，不选择权重：

- 若 `hand_term` 或 `foot_term` 对 latent/root/transl 的所有梯度都为 0，标记 `surface_graph_broken`；
- 若 surface 梯度有限非零，但相对观测梯度小于 `1e-3`，标记 `surface_gradient_negligible`；
- 若比例在 `1e-3` 到 `0.1`，标记 `surface_gradient_weak_but_active`；
- 若比例在 `0.1` 到 `1.0`，标记 `surface_gradient_comparable`；
- 若比例超过 `1.0`，标记 `surface_gradient_dominant`，禁止直接运行接触拟合，先降低接触系数。

JSON 必须写：

```json
{
  "status": "completed_stage_c_gradient_audit",
  "engineering_validation_only": true,
  "stage_d_executed": false,
  "beta_optimized_in_audit": false,
  "surface_graph_status": "pass|surface_graph_broken",
  "surface_gradient_status": "surface_gradient_negligible|surface_gradient_weak_but_active|surface_gradient_comparable|surface_gradient_dominant",
  "weight_selection_allowed": false,
  "next_action": "fix_graph|calibrate_contact_coefficient|stop"
}
```

## 验证

```powershell
& .venv-smplx\Scripts\python.exe -m py_compile realtime_app\pose_app\smpl_surface_contact.py research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py realtime_app\tools\summarize_stage_c_gradient_audit.py
& .venv-smplx\Scripts\python.exe -m unittest realtime_app.tests.test_smpl_surface_contact -v
& .venv-smplx\Scripts\python.exe realtime_app\tools\summarize_stage_c_gradient_audit.py
```

预期：6 项 surface-contact 单测全部 `ok`；两次 audit 退出码 0；无 Stage D 产物；审计中 `stage_d_executed=false`。

## 记录和 Git

在 `EXPERIMENT.md` 和 `AI_PROGRESS.md` 追加：

- 为什么 sole mapping 审计通过后进入 gradient audit；
- 两次 audit 的输入、命令、term 值和梯度；
- 判定 surface graph 是否有效；
- 为什么暂时不选择权重；
- 下一步是修图、校准系数还是停止。

不得写 Git hash 到实验记录。

禁止 `git reset`、`git clean`、`git checkout`、`git push`。只提交本任务代码和审计记录。提交信息：

```text
review: audit Stage C contact gradient scale
```

## 回报格式

```text
状态：完成 / 受阻 / 需补充信息
改动文件：绝对路径列表
验证结果：编译、6项单测、两次 audit、summary 实际输出
审查结论：surface_graph_status、surface_gradient_status、weight_selection_allowed、next_action
提交：commit ID 和文件列表
问题：没有则写“无”；明确说明 Stage D 未执行、v1–v5 未修改
```
