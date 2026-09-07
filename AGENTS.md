# 项目启动与交接规则

本文件是本仓库的持久工作约束。开始任何任务前，先检查工作区状态；不得覆盖或重置已有改动。

## 必读上下文

1. 每次任务：`README.md`、`CLAUDE.md`、`AI_PROGRESS.md`。
2. 涉及实验、数据、模型或几何时：`research_records/registry/EXPERIMENT_RECORDING_POLICY.md`，以及当前实验目录的 `EXPERIMENT.md`、`VALIDATION_PROTOCOL.md` 和已有结果文件。
3. 只读取与当前任务有关的记录；不要为获取上下文盲目重跑模型或扫描全部原始数据。

## 当前 DA3 主线

- 当前工程验证：`research_records/engineering_validation/V20260901_DA3_raw_fisheye_safe_roi_C3_control`。
- D1 已完成 60 对逐对原始鱼眼 DA3 前向、D1 安全 ROI 和左右 PMPose-b 推理。下一步仅是 C3/D1 坐标回归测试；通过前不得开始目标人关联或三角化。
- 坐标回归通过后，重放必须使用保存的预测、原始鱼眼标定和固定旋转（左 `ccw90`、右 `cw90`）；每帧最多一个左右人体匹配（`max_matches=1`）。
- DA3 只可作为每帧图像空间的 ROI / 遮挡诊断先验。不得用于标定、人物关联、关键点筛选、三角化或米制深度判断。
- `E20260901-A13_conservative_da3_guided` 是已归档的无效选择性反畸变原型。禁止重跑其姿态推理，禁止把它作为二维、三维或几何证据。

## 工作与结论边界

- 先说明本阶段的验证目标和下一步；一次只推进一个可检查阶段。
- 不得将视觉效果、DA3 有限深度、伪标签一致性、有效 3D 点数或重投影一致性表述为真实二维/三维精度提升。
- 所有失败帧和拒绝原因必须保留；不得挑选三角化更成功的人体候选或静默删除失败样本。
- 不得使用破坏性 Git 或文件操作，除非用户明确要求。

## 记录更新

每完成一个已验证阶段，更新主目录 `AI_PROGRESS.md` 和当前实验目录的 `EXPERIMENT.md`；按需更新 `VALIDATION_PROTOCOL.md`、`command.txt`、`run_metadata.json` 与实验注册表。记录输入、对照、唯一变量、结果、失败原因和结论边界。

仅在主线状态发生实质变化时更新根目录 `README.md` 与 `CLAUDE.md`。实验记录不得写哈希值。
