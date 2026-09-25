# task-07：全帧手部表面接触权重敏感性对照

项目根目录：`D:\my_works\walker_pose_system`

## 目标

验证当前“所有帧都加入手部表面接触损失”的 SMPL 拟合路线对手部损失权重是否真正产生可观测影响。当前 v3 结果中 hand-only、foot-only、foot+hand 的主指标几乎相同，不能据此宣布手部项有效；本任务只做固定窗口、固定初始化的手部权重敏感性对照。

所有结论只能写成 engineering validation。不得写成真实握持、真实握力、承重或真实三维精度验证。

## 当前已冻结内容

- male SMPL 6890 vertices。
- COCO observation regressor 只用于 2D/3D observation loss，不进入表面接触 loss。
- 左右手 palm surface candidate 各 778 个。
- 左右脚 sole surface candidate 各 54 个。
- 扶手几何来自当前运行 `contact_labels_stage_audit_v2/contact_labels.npz` 中的 `handle_ends_ground_m`；只能使用扶手端点几何，不能用 hand_contact_weight、hand_label 或 hand_candidate_score 门控手部损失。
- 手部表面损失覆盖窗口内全部帧和左右手，归一化为 `sum(hand_surface_loss(t,s))/(N*2)`。
- Stage A/B/C 无接触；Stage D 只优化 latent/root/transl。
- beta 在 Stage D 严格冻结。
- 右膝 COCO index 14 排除。
- 不启用 COCO ankle/wrist contact loss。
- 不读取旧 interaction target、旧 motion.json、旧 temporal prefit、旧 HTML、旧逐帧 contact target，也不读取旧拟合作为初始化。

## 只允许修改

优先只运行现有入口，不修改代码。若确有必要修正实验逻辑，只允许修改：

`research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/fit_vposer_shared_beta.py`

不得修改表面函数、顶点集合生成器、单测和已有历史输出目录。

## 固定输入

使用 v3 已验证的完全相同输入：

- 左右原始 PMPose JSON：`research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/left/pmpose/raw_predictions.json` 和 right 对应文件
- 标定：`realtime_app/calibration/results`
- male SMPL：`models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl`
- COCO regressor：`models/smpl/J_regressor_coco.npy`
- VPoser：`models/VPoser02_05/V02_05`
- contact labels：`research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/contact_labels_stage_audit_v2/contact_labels.npz`
- scene transforms：`research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/scene_stage_ground_v3_pair_replay/scene_transforms.npz`
- surface sets：`research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/surface_contact_sets_v1/contact_vertex_sets.json`
- walker topology：`research_records/engineering_validation/G20260918_offline_walker_frame_evidence_v1/coarse_complete_model_v2_camera_rail/coarse_walker_model.json`
- 窗口：`--start 60 --end 90`
- device：`cpu`
- 预算和 v3 完全相同；不能为了放大手部效果增加 contact_steps、base_steps、beta_steps 或 joint_steps。

## 对照路线

已有路线只读，不重跑：

- no-contact：`surface_contact_window60_90_v2_no_contact`
- surface foot-only：`surface_contact_window60_90_v2_foot_only`
- hand-only：`surface_contact_window60_90_v3_hand_only`，hand=0.01
- foot+hand：`surface_contact_window60_90_v3_foot_hand`，foot=0.02、hand=0.01

新建空输出目录，每个目录运行前确认不存在或为空，禁止覆盖：

1. `surface_contact_window60_90_v4_hand_w003`：foot=0.0，hand=0.03
2. `surface_contact_window60_90_v4_hand_w010`：foot=0.0，hand=0.10
3. `surface_contact_window60_90_v4_foot_hand_w003`：foot=0.02，hand=0.03
4. `surface_contact_window60_90_v4_foot_hand_w010`：foot=0.02，hand=0.10

每次命令必须保留在对应目录 `command.txt`，并检查进程退出码为 0。

## 每次运行必须检查

- `hand_contact_mode == all_frames_assumed`
- `hand_labels_used_for_hand_loss == false`
- `hand_contact_weight_gate == false`
- `hand_frames_optimized == 31`
- `hand_sides_optimized == 2`
- hand residual shape 左右均为 `[31,778]`
- foot residual shape 左右均为 `[31,54]`
- `same_forward_graph == true`
- `surface_probe_grad_norm_to_coco == 0` 或全零
- `surface_probe_grad_norm_to_vertices` 有限且非零
- `surface_loss_depends_on_vertices == true`
- `coco_joints_in_surface_contact == false`
- Stage C→D 同进程
- beta 在 Stage D 精确冻结
- 无 NaN/Inf

## 记录指标

为每一路保存并汇总：

1. Stage C 与 Stage D 的总 loss；
2. 左右 2D P95、总 2D P95；
3. 三维 median/P95；
4. 左右脚 surface residual median/P95；
5. 左右手 capsule residual median/P95；
6. hand surface coverage：15/30/50 mm；
7. Stage D 相对 Stage C 的 `vertices`、`predicted_coco`、`body_pose`、`global_orient`、`transl` 最大绝对变化和 L2 变化；
8. beta 与 v3 no-contact 的最大绝对漂移；
9. 手部损失数值、加权后的手部项数值及其梯度范数（如果已有审计字段则直接复用，不要重新定义口径）。

## 判定规则

- 不把 hand residual 下降自动称为真实握持改善。
- 若 hand-only 的 hand residual 随权重增加而下降，但 2D/3D 退化，记录为“手部几何项吸附观测的代价”，不能选为候选。
- 若 hand residual、2D、3D 和姿态都几乎不变，记录为“当前权重和梯度对拟合影响不足”，下一步检查损失尺度和优化器，而不是宣布 hand contact 有效。
- 若 foot+hand 与 hand-only 差异很小，必须报告脚手两项之间的梯度竞争没有被当前指标分辨，不能声称两项互补。
- 只有存在可重复的 residual 变化且没有违反二维/三维/负深度/beta 冻结门，才允许提出下一候选权重。

## 汇总文件

生成：

`research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/surface_contact_window60_90_v4_weight_sweep.json`

同时生成简短 `EXPERIMENT.md` 追加段，说明：输入、唯一变量、四条新路线、失败原因、停止条件和结论边界。不要修改旧段落，不要覆盖历史 JSON。

## 验证命令

```powershell
& .venv-smplx\Scripts\python.exe -m py_compile `
  realtime_app\pose_app\smpl_surface_contact.py `
  realtime_app\tools\build_smpl_contact_vertex_sets.py `
  realtime_app\tests\test_smpl_surface_contact.py `
  research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py

& .venv-smplx\Scripts\python.exe -m unittest realtime_app.tests.test_smpl_surface_contact -v

git diff --check
git status --short
```

## Git

若只运行实验而没有代码修改，不创建虚假代码提交。若修改了入口文件：

```powershell
git add -f research_records\engineering_validation\G20260924_smpl_vposer_shared_beta_v1\fit_vposer_shared_beta.py
git diff --cached --check
git commit -m "exp: sweep all-frame surface hand contact weight"
```

禁止执行 `git reset`、`git clean`、`git checkout`、`git push`。

## 最终回报

按以下格式回报：

- 状态：完成 / 受阻
- 运行目录和每条命令退出码
- 四条新路线的完整指标表
- hand residual 是否随权重变化
- 2D/3D 是否退化
- 四条路线的参数变化量
- 表面图隔离审计结果
- 是否触发停止条件
- 明确写：结果仍是 engineering validation，不是物理接触验证
