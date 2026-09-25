# task-13：foot=1 候选的独立窗口验证

## 背景

项目根目录：`D:\my_works\walker_pose_system`。

v6 开发窗 `60..90` 的原始 gate 计算有符号残差方向错误，已经由只读复算更正：

- v6 no-contact control 脚部 median = `-32.9434 mm`；
- v6 foot_a1 脚部 median = `-30.0938 mm`；
- 按穿透深度向零的绝对幅值计算，改善 `8.6501%`；
- foot_a1 3D P95 相对 control 增加 `1.61%`；
- 2D P95 改善 `1.58 px`；
- 穿透比例没有超过 `+1 pp`；
- beta 精确冻结，same process、surface graph audit 通过。

更正文件：

`D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v6_gate_reaudit.json`

因此 `foot=1` 只能作为开发窗候选，不能直接称为最终参数。当前脚底仍有约 93% 左右候选点低于地面，不能称真实触地。

## 目标

在三个未参与开发窗选择的独立窗口验证冻结配置：

```text
surface foot contact weight = 1.0
surface hand contact weight = 0.0
obs3d = 1.0
obs2d = 0.20
Stage D obs3d scale = 0.70
Stage D obs2d scale = 0.10
contact steps = 80
```

每个验证窗口都必须有同窗口、同初始化、同预算的 no-contact control。

## 严格范围

允许：

- 新建验证窗口目录和结果；
- 新建 `surface_contact_v6_independent_validation.json`；
- 追加 `EXPERIMENT.md`、`AI_PROGRESS.md`；
- 必要时新建只读汇总脚本。

禁止：

- 修改 `fit_voser_shared_beta.py`；
- 修改 `smpl_surface_contact.py`；
- 修改 v1–v6 既有目录、JSON、HTML；
- 修改 contact labels、scene transforms、surface sets、walker topology；
- 调整 foot=1、obs2d=0.20、Stage D 缩放、步数或判定阈值；
- 复制已有 Stage C 或 v6 结果作为初始化；
- 读取旧 fitted parameters、旧 temporal prefit、旧 HTML 或旧逐帧 contact target。

## 输入白名单

使用与 v6 完全相同的原始输入：

```text
left/right raw PMPose
realtime_app/calibration/results
models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl
models/smpl/J_regressor_coco.npy
models/VPoser02_05/V02_05
G20260924.../contact_labels_stage_audit_v2/contact_labels.npz
G20260924.../scene_stage_ground_v3_pair_replay/scene_transforms.npz
G20260924.../surface_contact_sets_v1/contact_vertex_sets.json
G20260918.../coarse_walker_model.json
```

## 验证窗口

固定三个窗口：

```text
129..159
278..308
373..403
```

每个窗口运行两条路线：

1. `v6_validation_<start>_<end>_control`
   - `--force-stage-d-no-contact`
   - surface foot=0, hand=0
2. `v6_validation_<start>_<end>_foot_a1`
   - surface foot=1, hand=0

所有命令必须显式传：

```text
--obs-3d-weight 1.0
--obs-2d-weight 0.20
--stage-d-obs-3d-scale 0.70
--stage-d-obs-2d-scale 0.10
--contact-steps 80
--device cpu
```

每个输出目录必须保存完整 `command.txt`，开始前确认目录为空；退出码必须为 0。

## 每窗必须记录

- Stage C/D 总 loss 和 l3/l2/lfoot/lhand；
- 2D median/P95、左右 P95；
- 3D median/P95；
- 脚部 signed median、median absolute residual、P05/P95；
- 左右脚 penetration fraction：`z<0`、`z<-25mm`；
- hand residual 只作固定诊断，不作为本轮优化项；
- Stage D vertices max-abs/L2；
- beta drift、beta_frozen、same_process、same_forward_graph；
- surface gradient audit 对 vertices 和 COCO joints；
- Stage 1/2/transition/held/rejected 帧计数（如果当前输出提供）；
- 所有数值有限，无 NaN/Inf。

## 正确的脚部增益定义

不能把负的 signed median 直接相减。对每个窗口、每条路线使用：

```python
control_mag = abs(control_foot_signed_median_mm)
route_mag = abs(route_foot_signed_median_mm)
foot_toward_zero_gain = (control_mag - route_mag) / control_mag
```

同时用逐顶点 `median(abs(residual_mm))` 和 `fraction(z<0)` 交叉检查。若 residual 符号跨越 0，必须保留 signed 和 absolute 两种统计。

## 冻结判定

`foot=1` 只有在三个独立窗口全部满足以下条件，才可冻结为开发候选：

1. 3D P95 相对该窗口 control 恶化不超过 2%；
2. 2D P95 相对 control 恶化不超过 5 px；
3. foot median absolute residual 向零改善至少 5%；
4. 左右脚 `z<0` penetration fraction 均不增加超过 1 pp；
5. beta 精确冻结；
6. same process、same forward graph、grad_to_coco=0、grad_to_vertices 有限非零；
7. 无 NaN/Inf。

如果任一窗口失败：

```text
selected_candidate = null
stop_reason = foot_a1_not_stable_across_independent_windows
```

不得事后挑选窗口、改门、继续增大脚权重、增加步数或加入手部项。

## 结论边界

即使三个独立窗口全部通过，也只能记录：

```text
engineering_validation_candidate = foot_a1
physical_touch_validated = false
true_3d_accuracy_validated = false
load_bearing_validated = false
grip_force_validated = false
```

## 验证命令

```powershell
& .venv-smplx\Scripts\python.exe -m py_compile research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py
```

逐目录检查：

```powershell
Get-ChildItem research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\v6_validation_*\command.txt
Get-ChildItem research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\v6_validation_*\metrics.json
```

## Git

禁止 `git reset`、`git clean`、`git checkout`、`git push`。实验记录不写 Git hash。只提交本任务脚本/汇总和记录，不提交模型、原始数据、大型 HTML 或旧结果。

建议提交：

```text
exp: validate foot contact coefficient on independent windows
```

## 回报格式

```text
状态：完成 / 受阻 / 需补充信息
改动文件：绝对路径列表
验证结果：6条命令退出码、每窗 control/foot 指标、正确绝对残差增益
审查结论：selected_candidate、stop_reason、三窗是否全部通过
提交：commit ID 和文件列表
问题：没有则写“无”；明确说明未修改 v1–v6、未执行 push
```
