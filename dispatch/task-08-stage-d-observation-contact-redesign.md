# task-08：重设计 3D 主导观测与 Stage D 表面接触权重实验

## 背景

项目根目录：`D:\my_works\walker_pose_system`。
当前拟合入口：
`D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py`

当前模型链：原始左右 PMPose → 本次运行内重新三角化 → male SMPL 6890 顶点 → 17×6890 COCO observation regressor → COCO-17 3D/双鱼眼 2D 观测项。当前 Stage A/B/C/D 逻辑为：A 拟合动作和根，B 拟合共享 beta，C 联合拟合，D 冻结 beta 后继续优化 latent/root/transl 并加入表面项。

当前确定问题：

1. `l3` 用米，`l2` 用像素，直接执行 `l3 + 0.10*l2`，当前 Stage C 的典型值约为 `l3=0.003475`、`l2=444.21`，因此 2D 数值项明显压过 3D；现有系数不能证明 3D 主导。
2. 脚部 `softmin(z)` 和手部 `softmin(distance-radius)` 没有按候选顶点数做 log(K) 中心化，K=54 或 778 时会产生明显负偏移。
3. 手部和脚部穿透虽会被 pseudo-Huber 惩罚，脚部有额外 `relu(-z)^2`，但手部没有显式穿透惩罚；需要把穿透统计记录下来并加入对手部穿透的单独惩罚。
4. 当前 Stage D 未降低 2D/3D 观测权重，不能实现“先观测拟合、再加入接触”。

本任务只处理上述确定问题和对应的 v5 小窗口实验，不修改可视化页面。

## 允许修改的文件

只允许修改或新增以下文件：

1. `D:\my_works\walker_pose_system\realtime_app\pose_app\smpl_surface_contact.py`
2. `D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py`
3. `D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\EXPERIMENT.md`
4. `D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_comparison.json`
5. 新建 v5 子目录及其运行产物、`command.txt`、`metrics.json`、`surface_contact_metrics.json`、`surface_contact_forward_audit.json`、`source_audit.json`、`result.npz` 和 Stage D npz。
6. `D:\my_works\walker_pose_system\AI_PROGRESS.md` 只追加本任务一条日志。

禁止修改任何现有 v1/v2/v3/v4 目录内容、现有 HTML、现有旧 comparison JSON、旧 contact labels、旧 scene transforms、旧 surface sets、旧 walker topology。

## 代码修改要求

### A. 中心化 soft-min

在 `smpl_surface_contact.py` 保留现有 `softmin(values, temperature)` 兼容接口，并新增：

```python
def centered_softmin(values: Any, temperature: float) -> Any:
    """Soft minimum corrected for the log(K) offset of K candidates."""
    torch = _torch()
    if values.shape[-1] <= 0:
        raise ValueError("centered_softmin requires a non-empty candidate dimension")
    if not temperature > 0:
        raise ValueError(f"softmin temperature must be positive, got {temperature}")
    k = values.shape[-1]
    return -temperature * (
        torch.logsumexp(-values / temperature, dim=-1) - math.log(float(k))
    )
```

在该文件顶部增加 `import math`。将 `foot_surface_loss` 和 `hand_surface_loss` 的 soft-min 改为 `centered_softmin`。不要改变残差的物理定义：脚部仍使用 `z_G`，手部仍使用 `distance_to_capsule_centerline - radius`。

### B. 穿透惩罚

将脚部损失保持为：

```python
foot_base = robust_scalar(centered_softmin(sole_z, temperature), delta)
foot_penetration = torch.relu(-sole_z).square().mean(dim=-1)
return foot_base + penetration_weight * foot_penetration
```

函数签名改为：

```python
def foot_surface_loss(sole_z: Any, temperature: float = 0.005,
                      delta: float = 0.010,
                      penetration_weight: float = 1.0) -> Any
```

将手部损失改为：

```python
hand_base = robust_scalar(centered_softmin(residuals, temperature), delta)
hand_penetration = torch.relu(-residuals).square().mean(dim=-1)
return hand_base + penetration_weight * hand_penetration
```

函数签名增加 `penetration_weight: float = 1.0`。保持全帧、左右手均参与，不恢复 label gating，不把 COCO wrist 传给 surface loss。

在 `__all__` 中加入 `centered_softmin`。

### C. 2D/3D 无量纲观测损失

在 `fit_vposer_shared_beta.py` 增加命令行参数：

```text
--obs-3d-weight       float default 1.0
--obs-2d-weight       float default 0.25
--stage-d-obs-3d-scale float default 0.70
--stage-d-obs-2d-scale float default 0.10
```

保留旧参数但不再用 `l3 + 0.10*l2` 写死。

在 `losses()` 中改为无量纲形式：

```python
l3 = ((robust_norm(jc - target, 0.10) / (0.10 ** 2)) * w3).sum() / (w3.sum() + 1e-6)
e2 = robust_norm(pl - obs_l, 20.0) + robust_norm(pr - obs_r, 20.0)
l2 = (e2 / (20.0 ** 2) * w * valid2d).sum() / (2 * (w * valid2d).sum() + 1e-6)
```

这两个变量只产生无量纲的 `l3`、`l2`，便于比较。不要修改原始 2D 或 3D 误差的输出单位；metrics 仍以 px 和 mm 记录。

`run()` 改为显式接收两个观测系数：

```python
def run(opt, steps, beta_reg=True, obs3d_coeff=1.0,
        obs2d_coeff=0.25):
```

总目标改为：

```python
loss = (obs3d_coeff * l3
        + obs2d_coeff * l2
        + 0.02 * lp
        + (contact_loss if contact_enabled else 0.0)
        + (0.02 * lb if beta_reg else 0.0))
```

Stage A、B、C 均使用：

```text
obs3d_coeff = args.obs_3d_weight       # 默认 1.0
obs2d_coeff = args.obs_2d_weight       # 默认 0.25
```

这样观测阶段明确 3D 权重大于 2D。Stage D 使用：

```text
obs3d_coeff = args.obs_3d_weight * args.stage_d_obs_3d_scale  # 0.70
obs2d_coeff = args.obs_2d_weight * args.stage_d_obs_2d_scale  # 0.025
```

注意：3D/2D 权重降低后不能变成零；Stage D 仍然保留观测约束。

### D. Stage D 接触权重语义

保留 `--surface-foot-contact-weight` 和 `--surface-hand-contact-weight` 作为 Stage D 的外部接触系数，仍放在已经归一化的 `lfoot`、`lhand` 之后：

```python
contact_loss = (
    args.surface_foot_contact_weight * lfoot
    + args.surface_hand_contact_weight * lhand
)
```

不要把它们乘到单个顶点、soft-min 温度、扶手半径或 COCO 点上。

在 surface audit 中新增：

```json
{
  "observation_loss_units": "dimensionless_delta_normalized",
  "stage_a_to_c_obs_3d_coeff": 1.0,
  "stage_a_to_c_obs_2d_coeff": 0.25,
  "stage_d_obs_3d_coeff": 0.70,
  "stage_d_obs_2d_coeff": 0.025,
  "foot_penetration_fraction": {"left": ..., "right": ...},
  "hand_penetration_fraction": {"left": ..., "right": ...}
}
```

`foot_penetration_fraction` 是 `sole_z < 0` 的候选点比例；`hand_penetration_fraction` 是 `capsule_surface_residual < 0` 的候选点比例。对控制路线也记录这些只读诊断值；控制路线不能伪造 surface gradient proof。

### E. Stage D 标签和 beta 不变

必须保持：

- Stage C → Stage D 同一进程；
- Stage D beta `requires_grad_(False)`；
- Stage D 前后 beta 精确相等；
- COCO 右膝 index 14 不作为 accepted 监督；
- 手部损失覆盖全部窗口帧和左右手；
- `same_forward_graph=true`、surface 对 vertices 梯度有限非零、对 COCO joints 梯度为零；
- 不启用 COCO ankle/wrist surface contact；
- 不读取旧拟合初始化或旧 contact target。

## v5 实验矩阵

窗口固定 `60..90`，同一原始 PMPose、标定、male SMPL、COCO regressor、VPoser、当前 scene transforms、contact labels、surface sets、walker topology。每条路线独立从同样的零初始化开始，不能复制 Stage C 或旧结果。

所有路线都显式传：

```text
--obs-3d-weight 1.0 --obs-2d-weight 0.25
--stage-d-obs-3d-scale 0.70 --stage-d-obs-2d-scale 0.10
--contact-steps 80 --device cpu
```

运行 7 条路线，输出目录必须全新：

1. `surface_contact_window60_90_v5_stage_d_no_contact`
   - `--force-stage-d-no-contact`
   - surface foot=0, hand=0
2. `surface_contact_window60_90_v5_foot_a005`
   - surface foot=0.05, hand=0
3. `surface_contact_window60_90_v5_foot_a010`
   - surface foot=0.10, hand=0
4. `surface_contact_window60_90_v5_hand_a005`
   - surface foot=0, hand=0.05
5. `surface_contact_window60_90_v5_hand_a010`
   - surface foot=0, hand=0.10
6. `surface_contact_window60_90_v5_both_a005`
   - surface foot=0.05, hand=0.05
7. `surface_contact_window60_90_v5_both_a010`
   - surface foot=0.10, hand=0.10

完整命令模板（每条只替换 output-dir 与两个 surface 权重）：

```powershell
& .venv-smplx\Scripts\python.exe -u research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py `
  --output-dir <NEW_V5_OUTPUT_DIR> `
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
  --surface-foot-contact-weight <FOOT_ALPHA> `
  --surface-hand-contact-weight <HAND_ALPHA> `
  --contact-steps 80 --obs-3d-weight 1.0 --obs-2d-weight 0.25 `
  --stage-d-obs-3d-scale 0.70 --stage-d-obs-2d-scale 0.10 --device cpu
```

每条命令保存到对应目录 `command.txt`；退出码必须为 0。

## 结果判定

生成 `surface_contact_window60_90_v5_comparison.json`，至少包含每条路线：

- Stage C / Stage D 的 `l3`、`l2`、总 loss；
- 2D median/P95 和左右 P95；
- 3D median/P95；
- 左右脚穿透比例、左右手穿透比例；
- 左右脚 surface residual median/P95；
- 左右手 surface residual median/P95；
- hand coverage 15/30/50 mm；
- beta 漂移、beta frozen、same process、same forward graph；
- 实际 Stage D 3D/2D 系数；
- 结论边界字段：`engineering_comparison_only=true`，物理接触、握力、承重、真实 3D 精度均为 false。

推荐候选必须同时满足：

1. 相对同一 v5 no-contact 控制，3D P95 不恶化超过 2%；
2. 2D P95 不恶化超过 5 px；
3. 至少一个脚或手的 surface residual median 相对控制下降 5%；
4. 穿透比例不增加超过 1 个百分点；
5. beta 漂移接近 0，beta 精确冻结；
6. surface gradient audit 通过，无 NaN/Inf。

如果所有路线都没有满足条件，必须判定“当前接触几何或优化尺度仍未形成可观测收益”，停止继续增大权重；不要自行增加步数、改阈值或挑选最好看的路线。

## 验证命令

在项目根目录执行：

```powershell
& .venv-smplx\Scripts\python.exe -m py_compile realtime_app\pose_app\smpl_surface_contact.py research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py
& .venv-smplx\Scripts\python.exe -m unittest realtime_app.tests.test_smpl_surface_contact -v
```

预期：编译退出码 0；6 项 surface-contact 单测全部 `ok`。

检查每条路线：

```powershell
Get-ChildItem research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_*\command.txt
Get-ChildItem research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_*\metrics.json
```

## 实验记录

在 `EXPERIMENT.md` 追加一个新章节，写清楚：

- 为什么从原始残差改中心化 soft-min；
- 为什么增加手部穿透惩罚；
- 为什么 3D/2D 改为 delta 归一化；
- 为什么 Stage D 降低观测项而不是删除观测项；
- 输入路径、窗口、步数、7 条路线和输出目录；
- 每条路线实际指标和停止条件；
- 结论仍是 engineering validation。

在 `AI_PROGRESS.md` 追加一条日志，不能重写旧内容。

## Git 要求

工作开始先运行：

```powershell
git status --short
```

不得执行 `git reset`、`git clean`、`git checkout` 或 `git push`。

只暂存本任务修改的代码和记录文件；不要加入原始数据、模型、npz、大型 HTML 或其他 agent 的改动。至少建立两个本地提交：

1. `fix: redesign staged observation and surface contact losses`
2. `exp: compare v5 staged contact coefficients`

提交前分别运行：

```powershell
git diff --cached --check
git show --stat --oneline --name-only HEAD
```

实验记录禁止写 Git hash；Git hash 只在 Cursor 回报中报告。

## 回报格式

```text
状态：完成 / 受阻 / 需补充信息
改动文件：绝对路径列表
验证结果：编译、6项单测、7条实验命令退出码、每条路线关键指标
审查结论：是否有推荐系数；若无，明确停止原因
提交：两个本地 commit ID，以及每个 commit 的文件列表
问题：没有则写“无”；未修改历史 v1-v4 产物则明确写“历史输出未修改”
```
