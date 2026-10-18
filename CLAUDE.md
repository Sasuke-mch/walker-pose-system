# Walker Pose System - Claude 工作指南

## 2026-10-17 用户路线决定

不固定唯一身体主线，方法按独立路线管理。当前高优先级 `smplh_causal_autograd_contact` 使用自动微分＋前掌方向＋接触，入口 `realtime_app/tools/run_smplh_contact_route.py`，单次完整448；不自动增加FD/无接触/局部旋转对照。裸模型足底以真实地面上方25 mm为鞋高参考，5 mm容差下界20 mm，物理地面保持Z=0。路线与优先级读取 `current_mainline.json/route_management,method_routes`；旧 `mainline_selection` 只是批量路线兼容字段。历史决定按日期解释，最新结果看 `AI_PROGRESS.md`。


## 2026-10-13 SMPL-H拟合方法更新

通用拟合与二维实验入口默认使用--fit-policy reliability，修正置信度归约、SO(3)根锚及接触阶段身体观测保持项。已保存360 reference25和448结果仍来自legacy，未重算；冻结profile保留legacy，直接重放历史argv须显式补--fit-policy legacy。公式、输入支持、统计校准及物理证据边界见docs/CURRENT_SMPLH_ALGORITHM_20261002.md第28节，进度见AI_PROGRESS.md；下方历史分数下限0.6不代表新方法。

## 2026-10-12 场景主线更新

用户已将448帧验收的Stage1/2连续过渡逻辑选为场景主线：默认场景入口及`--scene-profile mainline`解析为`supported_stop_fixed_lag`。停止时四脚已落在同一平地是必要采集前提；采用0.5秒固定延迟发布，实时采集入口接入仍待验证。旧baseline显式保留，身体选择不随本次场景提升改变。下方2026-10-05的baseline主线决定为历史记录；当前配置、边界和验证见`AI_PROGRESS.md`与`docs/CURRENT_STAGE_GROUND_PIPELINE.md`。


## 2026-10-05 正式工程主线决定

当前正式工程路线固定为 **PMPose 原始鱼眼观测 → 严格双目几何 → 原 Stage 1/2 识别 → 完整 SE(3) 动态地面基线 → 同源 SMPL-H 冷启动身体及 WiLoR 原生局部手姿**。选择清单位于 `realtime_app/configs/current_mainline.json` 的 `mainline_selection`，场景调用使用 `--scene-profile mainline`，实际解析为 `baseline`。这是一项可复现版本决定，不是物理接触、独立三维精度或整套实时部署认证。

本次场景数据为换机位 360 帧（0..359，原始配对1157..1516）；身体主线改用 `G20261005_reference_weight_fit_v1/runs/reference25/fit/result.npz`。实际权重为 `reference`：回投影平台25px、射线间距平台80mm、置信度参考下限0.6，严重异常使用四次尾衰减。原 `body_quality` 与实际 `body_fit_weights` 分开保存。该结果未启用人工腕软参考，属于正式选择的工程主线，`accepted_for_main_fit=false`仍表示独立物理/接触门未通过。旧360帧quality结果保留为对照；448帧两检测器冻结比较保持原quality参数及等级，配置顶层旧字段仍仅描述448帧。

后续重点：**各关节权重与置信度计算规则**。需要分别研究肘、腕、膝、踝等关节的普通偏差/严重异常界限、可见性与几何敏感性、以及观测和人体先验的平衡；当前共用25px/80mm规则为暂定工程选择。臂均残差75.12→66.38mm、膝踝二阶差分P9573.68→41.92mm/frame²，但左肘及部分腕点变差、固定原质量全身RMS51.10→57.93mm，不能认定所有关节已准确。

`landing_fixed_lag` 与 `joint_stage_window` 保留为显式离线候选：运动指标有部分改善，但落地闭合未通过，联合窗口还未满足耗时预算。只将严格点/掩码导出和最终发布流导出修复纳入共用入口。原基线的硬恢复跳变、缺失测量及静态扶手参考局限继续记录，不用候选或显示平滑掩盖。

详细数据流、参数、命令及停止条件见 `docs/CURRENT_STAGE_GROUND_PIPELINE.md`；身体内核及448帧历史实现见 `docs/CURRENT_SMPLH_ALGORITHM_20261002.md`。后文较早的“当前”判断按其数据集和日期理解；最新选择以本节、`AI_PROGRESS.md` 和选择清单为准。


## 当前身体研究路线与导航（2026-10-03）

当前离线身体路线为SMPL-H冷启动：Sapiens2优先候选，PMPose同流程冻结对照。两者accepted_for_main_fit=False；人工腕是诊断参考、手部几何未通过、脚底负高度未解决。实时PMPose三角化基线与这一离线身体候选是不同层面的身份。旧SMPL/SMPL-X与固定抓握方法保留历史级别，不接回旧beta、手形或脚锚。

开始身体任务应读 docs/CURRENT_SMPLH_ALGORITHM_20261002.md 和 realtime_app/configs/current_mainline.json，再读实际protocol/fit_command/assessment；实验报告备份辅助核对，不覆盖更新的源码和实际协议。结构和待处理文件见 docs/PROJECT_STATUS.md、docs/code_inventory.json。

公开测试与CI使用 realtime_app/run_validation.py；run_tests.py为旧入口。离线调用统一用 realtime_app/run_tool.py。当前候选用 body.run_sapiens_cold_smplh --settings realtime_app/configs/cold_smplh_joint300.json，省略配置保留C30，不能误称C300复现。安装、资源检查、完整命令与评价入口统一见 realtime_app/README.md，避免在多个指南中维护另一组默认值。

## 文档权威顺序

为避免同一规则在多个文件中重复维护，按以下顺序判断：

1. `AGENTS.md`：仓库持久工作规则、实验推进门和当前冻结约束。
2. `AI_PROGRESS.md`：最新工作日志、实验结果、停止条件和当前阶段状态。
3. `VISUALIZATION_PIPELINE.md`：涉及人体、地面、相机、助步器可视化时的唯一显示规范。
4. `research_records/registry/EXPERIMENT_RECORDING_POLICY.md`：实验记录、归档和可追溯性规则。
5. 当前实验目录中的 `EXPERIMENT.md`、`VALIDATION_PROTOCOL.md` 和结果文件：该实验的局部协议和证据。
6. 本文件与根目录 `README.md`：只提供导航和稳定的项目概览，不复制上述规则。

若文档之间存在冲突，以较新的权威文档和当前实验协议为准；旧日志和旧实验记录只作为历史证据，不作为当前规则。

## 项目概览

本项目研究助步器场景下的双鱼眼人体姿态估计、双目三维、动态地面和交互可视化。主链为：

```text
双鱼眼采集
→ 2D 姿态
→ 人物关联
→ 鱼眼三角化
→ 下肢 T1–T5
→ Stage 1/2 与地面变换
→ 运动学/接触候选
→ 三维可视化
```

`realtime_app` 是实时和双目主程序；`sequence_pipeline` 是多模型图片序列调度程序；`research_records` 保存正式实验、工程验证和失败证据；`docs/walker-viewer` 是公开的静态交互查看器。

## 当前基线与边界

- 双目身份必须使用 `camera_registry.json` 的物理 PnP 身份；不得把 Windows OpenCV 数字索引当作永久相机身份。
- 物理角色固定为 `cam0=LEFT`、`cam1=RIGHT`；模型输入可按当前协议转正，但几何计算必须回到原始鱼眼像素和正式标定。
- 实时双目工程基线保留 PMPose；当前离线身体优先候选使用 Sapiens2，其结果仍不是独立真值。
- 严格三角化必须保留失败帧、逐点拒绝原因、坐标边界和质量来源；`max_matches=1` 是当前主线约束。
- DA3 只允许作为逐帧图像空间的 ROI/遮挡诊断先验，不得用于标定、人物关联、关键点筛选、三角化或米制深度。
- Stage 1/2、动态地面、助步器和接触输出都是工程候选，缺少外部轨迹、阶段、对象身份或物理接触真值时不得升级为真实精度或物理结论。
- SMPL/SMPL-X 结果必须遵守当前实验的阶段门；模型内部一致性、重投影、平滑、接触代理和有效点数量都不能替代外部真值。
- 可视化必须遵循 `VISUALIZATION_PIPELINE.md` 的同源输入、当前运行重放、固定地面坐标、真实三角面和浏览器验收要求。

## 工作规则

- 所有 agent 完成可独立验证的修复、迭代或更新后，必须按 `AGENTS.md` 的“Git 版本管理”条款创建仅含本次任务的本地提交；不得混入已有未提交改动，未获明确授权不得推送远端。
- 每次开始工作先检查工作区状态，并保留已有改动；不得覆盖、重置或删除用户已有成果。
- 先说明验证目标、输入、对照、唯一变量和停止条件；复杂任务一次只推进一个可检查阶段。
- 实验、数据、模型或几何任务必须先读取实验记录规则和相关实验协议；禁止为获取上下文盲目重跑模型或扫描全部原始数据。
- 未有独立人工/物理/三维真值时，只能表述为工程验证、内部一致性或操作性参考一致性。
- 不得通过补点、插值、平滑、旧结果混用或挑选成功帧掩盖失败；所有拒绝原因必须可追溯。
- 完成已验证阶段后更新 `AI_PROGRESS.md` 和对应实验记录；实验记录使用路径、数据范围、代码、参数和输出追溯，不写 Git 哈希。
- 用户使用中文时用中文回复；优先小修改，避免无明确授权的大规模重构。

## 代码与实验入口

- 双目实时/回放：`realtime_app/run_stereo.py`
- 单路 2D：`realtime_app/run.py`
- 测试：`realtime_app/run_validation.py`
- 多模型序列调度：`sequence_pipeline/run_sequence.py`
- 当前阶段与动态地面说明：`docs/CURRENT_STAGE_GROUND_PIPELINE.md`
- 可视化规范：`VISUALIZATION_PIPELINE.md`

详细当前进展只维护在 `AI_PROGRESS.md`，不要在本文件追加逐次实验日志。
