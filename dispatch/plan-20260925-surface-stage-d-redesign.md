# plan-20260925-surface-stage-d-redesign

目标：把 male SMPL 表面手脚接触拟合改成“先以 3D 主导的观测项拟合，再降低观测项并加入表面接触项”的可审查流程；先修正 signed surface soft-min 的顶点数偏移和穿透惩罚，再比较接触系数。

任务：

1. task-08-stage-d-observation-contact-redesign.md：修改 fit_vposer_shared_beta.py 和 smpl_surface_contact.py，加入无量纲 3D/2D 观测项、3D 主导系数、Stage D 观测衰减、中心化 soft-min 和显式穿透诊断；运行 60..90 新 v5 对照。
2. 后续任务待 task-08 回报后决定：只针对通过停止门的系数做补充实验；若全部失败，停止调权重，改查接触几何/手部全帧假设。

全局边界：
- 不覆盖 v1/v2/v3/v4 历史输出；新输出使用 v5 目录。
- 不读取旧拟合初始化、旧 temporal prefit、旧 HTML、旧逐帧接触目标。
- 继续使用原始 PMPose、标定、male SMPL、COCO regressor、VPoser、当前 scene transforms、surface vertex sets 和静态 walker topology。
- 所有结论仍为 engineering validation；不称真实触地、握力、承重或真实 3D 精度。
- 每个可独立验证的代码/记录更新单独本地提交，不 push。
