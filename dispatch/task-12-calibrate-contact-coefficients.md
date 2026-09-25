# task-12：按 Stage C 梯度比例校准 3D/2D 与手脚接触系数

## 背景

项目根目录：`D:\my_works\walker_pose_system`。

Stage C 梯度审计已经完成，摘要：

`research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v5_gradient_audit_summary.json`

有效梯度 L2：

```text
obs3d = 0.327392
obs2d = 0.327691
foot  = 0.020922
hand  = 0.000545
```

当前系数：

```text
obs3d = 1.0
obs2d = 0.25
```

因此当前实际 3D/2D 梯度几乎相等，不能称 3D 略大于 2D。surface graph 有效，但：

```text
foot / observation gradient ≈ 3.2%（系数 1）
hand / observation gradient ≈ 0.08%（系数 1）
```

v5 几何审查已确认：单位、transform 闭环和 sole 映射没有单一可执行修复；禁止重新修改几何。

## 目标

1. 将观测阶段和 Stage D 的实际 3D 梯度调到略高于 2D；
2. 按已测梯度比例比较脚部和手部接触系数；
3. 找到是否有路线能让接触 residual 有至少 5% 可观测下降，同时不明显破坏 3D/2D；
4. 如果没有，停止继续放大权重。

## 允许修改

只允许修改：

- `D:\my_works\walker_pose_system\research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py`；
- 新建 v6 对照 JSON 和 v6 子目录产物；
- `EXPERIMENT.md`、`AI_PROGRESS.md` 追加记录。

不得修改：

- `smpl_surface_contact.py`；
- `contact_vertex_sets.json`；
- scene transforms；
- walker topology；
- v1–v5 历史结果和 JSON；
- HTML 或可视化页面。

## 确定代码修改

只修改默认观测权重：

```text
--obs-3d-weight default = 1.0
--obs-2d-weight default = 0.20
```

Stage D 保持：

```text
stage-d-obs-3d-scale = 0.70
stage-d-obs-2d-scale = 0.10
```

因此：

```text
Stage A/B/C: 3D=1.0, 2D=0.20
Stage D:     3D=0.70, 2D=0.020
```

原因：v5 梯度审计表明把 2D 系数从 0.25 降到 0.20 后，预计 3D 梯度约为 2D 的 1.25 倍，符合“3D 略大于 2D”。

保留 delta 归一化、centered soft-min、穿透惩罚、beta 冻结和 Stage C→D 同进程逻辑。

## v6 实验路线

窗口固定 `60..90`，所有路线从同样的原始输入和零初始化开始。每条命令必须保存 command.txt，不能复制旧 Stage C。

公共参数：

```text
--obs-3d-weight 1.0
--obs-2d-weight 0.20
--stage-d-obs-3d-scale 0.70
--stage-d-obs-2d-scale 0.10
--contact-steps 80
--device cpu
```

运行 6 条路线：

1. `surface_contact_window60_90_v6_stage_d_no_contact`
   - force control，foot=0，hand=0
2. `surface_contact_window60_90_v6_foot_a1`
   - foot=1，hand=0
3. `surface_contact_window60_90_v6_foot_a3`
   - foot=3，hand=0
4. `surface_contact_window60_90_v6_hand_a30`
   - foot=0，hand=30
5. `surface_contact_window60_90_v6_both_a1_a30`
   - foot=1，hand=30
6. `surface_contact_window60_90_v6_both_a3_a60`
   - foot=3，hand=60

系数依据：

- foot=1 的初始梯度约为观测梯度 3.2%；
- foot=3 约为 9.6%；
- hand=30 约为 2.5%；
- hand=60 约为 5%；
- both 1/30 约为 5%–6%；
- both 3/60 约为 14%–15%。

使用与 v5 相同的完整输入路径：原始左右 PMPose、正式标定、male SMPL、COCO regressor、VPoser、contact labels、scene transforms、surface sets、walker topology。

## 结果比较

生成：

`research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\surface_contact_window60_90_v6_comparison.json`

每条路线记录：

- 实际 Stage C/D 3D 与 2D 系数；
- Stage C/D 总 loss 和各项 loss；
- 2D median/P95、左右 P95；
- 3D median/P95；
- 脚和手 residual median/P95；
- 脚和手 penetration fraction；
- hand coverage 15/30/50 mm；
- Stage C→D 参数变化；
- beta 漂移和 beta frozen；
- same process、same forward graph、gradient audit；
- engineering comparison only 及物理结论 false 字段。

推荐候选门：

1. 相对 v6 no-contact control，3D P95 恶化不超过 2%；
2. 2D P95 恶化不超过 5 px；
3. 至少一类接触 residual median 下降至少 5%；
4. 脚/手 penetration fraction 不增加超过 1 个百分点；
5. beta 精确冻结；
6. 无 NaN/Inf，surface 对 vertices 梯度有限非零、对 COCO joints 为零。

如果没有路线通过，必须写：

```text
recommended_candidate=null
stop_reason=contact_geometry_or_pose_response_not_observable
```

禁止继续增加权重、增加步数或调整阈值。

## 验证

```powershell
& .venv-smplx\Scripts\python.exe -m py_compile realtime_app\pose_app\smpl_surface_contact.py research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py
& .venv-smplx\Scripts\python.exe -m unittest realtime_app.tests.test_smpl_surface_contact -v
```

预期：编译退出码 0；6 项单测全部 `ok`；6 条路线退出码均为 0。

## 记录与 Git

在 EXPERIMENT.md 和 AI_PROGRESS.md 追加：

- 为什么 obs2d 从 0.25 改成 0.20；
- Stage C 梯度审计的数值；
- foot=1/3、hand=30/60 的推导；
- 每条 v6 路线结果；
- 是否通过推荐门；
- 如果失败，为什么停止。

禁止在实验记录写 Git hash。不得执行 `git reset`、`git clean`、`git checkout`、`git push`。

代码提交和实验记录提交分开：

```text
fix: make 3D gradient slightly dominant in staged fitting
exp: calibrate surface contact coefficients from gradient audit
```

## 回报格式

```text
状态：完成 / 受阻 / 需补充信息
改动文件：绝对路径列表
验证结果：编译、6项单测、6条实验退出码和关键指标
审查结论：3D/2D实际梯度关系、推荐系数或停止原因
提交：两个 commit ID 和文件列表
问题：没有则写“无”；明确说明 v1–v5 未修改、未执行 push
```
