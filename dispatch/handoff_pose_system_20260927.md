# Walker Pose System 总体交接记录

更新时间：2026-09-27  
项目根目录：`D:\my_works\walker_pose_system`  
实时程序目录：`D:\my_works\walker_pose_system\realtime_app`

## 1. 交接目标与证据边界

本文件交接的是“相机重新布置后，从双鱼眼图像到二维检测、三角化、地面/助步器状态和 SMPL 主线”的当前工程状态，同时记录 WiLoR/InterWild 手部模型的候选位置。

当前所有结果仍属于工程验证、内部一致性或可操作性参考一致性。没有人工二维标注、独立三维真值、力传感器或临床标签，因此不得把有效点数、重投影误差、模型置信度、可视化接触或平滑结果写成真实三维精度、真实触地、真实承重或真实握持结论。

## 2. 当前主线

### 2.1 固定不变量

- 物理相机身份由 `camera_registry.json` 的 PnP `instance_id` 解析，不能把 OpenCV 数字索引当作永久身份。
- 标定角色固定为 `cam0=LEFT`、`cam1=RIGHT`。
- 模型输入可旋转为正立，但所有模型输出在三角化前必须逆变换回原始鱼眼像素。
- 当前人体主线仍是 COCO-17 二维观测、严格单人关联、严格质量门和双目三角化。
- 三角化保留逐关节有效/拒绝状态、拒绝原因、原始边界状态，`max_matches=1`。
- SMPL 主线使用 male SMPL 6890 顶点模型和固定 `J_regressor_coco.npy`，模型侧仍通过 COCO-17 回归器对齐二维/三维观测；SMPL 表面顶点用于手脚表面项。
- 后续 SMPL/SMPL-X、地面、助步器和接触模块不能掩盖上游二维检测或三角化失败。

### 2.2 当前新数据与标定

- 双目标定：`realtime_app/calibration/results/stereo_fisheye_20260927_195919.json`
- 静态地面参考：`realtime_app/calibration/results/static_ground_20260927_201028/ground_reference.json`
- 地面标定板与地面距离：24 mm，记录为测量参考，不可在后续代码中重复加一次偏移。
- 步行器拓扑结构沿用原有拓扑；相机位置变化只应通过当前标定和当前运行的场景变换重新计算，不得复用旧相机基线或旧逐帧位姿。
- 新采集双目回放：`realtime_app/outputs/stereo_capture/20260927_201716_220`
- 该回放共有 360 对真实配对帧，原始采集约 30 FPS；离线模型推理速度较低，不等同于实时输出帧率。

## 3. 已完成的代码与验证

### 3.1 PMPose 主线

PMPose 的调用链为：

```text
旋转后的左右图像
  -> YOLO 人体检测服务
  -> 自适应扩框
  -> PMPose-b
  -> 关键点逆旋转
  -> 原始鱼眼像素
  -> 既有单人关联与严格三角化
```

关键文件：

- `realtime_app/run_stereo.py`
- `realtime_app/pose_app/http_client.py`
- `realtime_app/server/yolo_detector_server.py`
- `realtime_app/server/pmpose_server.py`
- `realtime_app/pose_app/rotation.py`
- `realtime_app/pose_app/triangulation.py`

当前推荐模型输入方向：

```text
LEFT  --left-model-rotation ccw90
RIGHT --right-model-rotation cw90
```

### 3.2 自适应扩框

项目原有的正式实现是 `realtime_app/tools/build_continuous_foot_inclusive_roi.py` 中的 `foot_inclusive_box()`。实时 PMPose 链路现在直接调用这一实现，不再保留另一套 `adaptive_bbox.py` 或重复公式。

- 正式实现：`realtime_app/tools/build_continuous_foot_inclusive_roi.py`
- 实时调用：`realtime_app/pose_app/http_client.py`
- 默认开启：`--pmpose-adaptive-box on`
- 关闭方式：`--pmpose-adaptive-box off`
- 默认阈值：人体框宽度 / 模型输入图像宽度 `0.37`
- 默认增长幂：`0.75`
- 最大相对扩展：左右 `0.20`、顶部 `0.08`、底部 `1.30`

规则含义：远处或较小人体框保持原框；近处人体框连续增大，底部扩展更强，目标是减少脚部和被遮挡下肢被裁剪。扩框只在旋转后的模型输入坐标中发生，原始框作为请求附加信息保留，扩框数量和参数写入 `stage_times_ms` 及 `stereo_summary.json`。

按原有逻辑完成了一次 360 对回放：输出为 `realtime_app/outputs/pipeline_20260927_201716_pmpose_original_roi`；720 次左右模型调用中 887 个检测框扩展，平均有效三维点 8.27/17，平均配对处理速度 2.308 对/秒，右侧越界拒绝点 168 个。此次运行的作用是确认主线接入和输出完整性，不把这些统计解释为真实精度。

### 3.3 旧回放基线

之前未开启自适应扩框的 PMPose 回放输出：

`realtime_app/outputs/pipeline_20260927_201716_pmpose`

已知内部统计：

- 360 对处理完成；
- 左侧平均有效二维点约 16.37/17，右侧约 15.45/17（阈值 0.25）；
- 平均三维有效关键点约 7.33；
- 离线推理平均配对处理速度约 1.408 对/秒；
- 这些数字是当前工程链路统计，不是人体姿态精度。

旧可视化：

- `realtime_app/outputs/pipeline_20260927_201716_pmpose/visualized_2d_frames`
- `realtime_app/outputs/pipeline_20260927_201716_pmpose/visualized_2d_model_input`
- `realtime_app/outputs/pipeline_20260927_201716_pmpose/stereo_annotated.mp4`

下一次应使用同一输入、同一标定、同一阈值，只改变 `--pmpose-adaptive-box`，形成严格对照。

## 4. 地面、助步器与人体主线状态

### 4.1 地面与助步器

- 静态地面参考已经由新位置重新建立；动态地面使用当前运行的动态变换输出。
- 助步器拓扑保留现有结构，不能用可视化临时连线替代真实拓扑。
- 之前动态/阶段链路中存在覆盖率不足、动态地面更新为 0、助步器重构部分失败等记录；这些失败必须保留，不得用后续 SMPL 拟合或平滑结果覆盖。

### 4.2 SMPL 主线

- 当前正式人体表达主线仍为 SMPL，不是把 MANO 直接塞进标准 SMPL。
- 标准 SMPL 没有独立手指自由度；手部弯曲只能通过近似身体姿态表达，不能满足精细抓取。
- WiLoR 输出的是 MANO/SMPL-H 风格手部参数和手部关键点；要保留整体 SMPL 主线，当前合理工程方式是：继续用 SMPL 的身体和 COCO 回归器作为主体，手部模型只作为手部观测/候选约束，不能直接声称已经得到统一 SMPL 手指参数。
- 若要最终得到包含手指的统一人体模型，应转为 SMPL-H 或 SMPL-X 参数化，并重新定义回归器、手部关节映射和拟合接口；这不是当前 PMPose 扩框改动的一部分。

### 4.3 时序与接触实验

已有 full17 SMPL/表面接触和统一时序结果，均属于工程验证。当前结论边界：统一时序可以降低部分运动变化，但可能增加脚部地下比例或表面接触误差；不能直接把视觉结果当作物理接触。

## 5. WiLoR 与 InterWild 状态

### WiLoR

- 已有本地模型下载、序列运行、二维可视化和部分 SMPL-H 拟合尝试。
- 失败/不足点已经记录：强遮挡时手框和关键点会缺失或不稳定，二维可视化中部分关节不可见并不代表模型输出一定没有对应槽位。
- WiLoR 的输出应优先作为手部观测，包括手部关键点、MANO/SMPL-H 姿态、形状和相机相关参数；不要再把其 MANO 结果误读为标准 SMPL 参数。
- 当前没有完成“WiLoR 手部参数稳定融合到最终统一身体模型”的正式实验。

### InterWild

- 已完成候选环境/模型路径的规划，但尚未作为主线稳定接入并通过遮挡场景对照。
- InterWild 适合作为强遮挡手部候选和补充诊断，不应在没有同一输入、同一遮挡片段和同一评价协议前替换 WiLoR 或 PMPose 主线。

## 6. 当前正在做什么

当前不是扩框算法对比实验，而是在固定使用既有扩框逻辑的前提下，恢复新相机位置后的完整主线。扩框只是二维 PMPose 输入预处理，不是独立研究变量。

当前主线顺序必须保持：

1. 读取新双目标定和新静态地面参考；
2. 用真实配对回放执行左右图像旋转；
3. PMPose 前调用原有 `foot_inclusive_box()`；
4. 保存左右二维关键点、框、置信度和失败状态；
5. 用原始鱼眼像素执行单人关联和严格三角化；
6. 在三角化结果上运行下肢 T1–T5、Stage 1/2、动态地面和助步器候选重构；
7. 对通过上游门槛的结果导出 SMPL 主线输入；
8. 按既有 Stage A/B/C/D 流程拟合 SMPL，最后才生成同源可视化。

每一步必须先检查输入输出契约，再进入下一步。某一步失败时保留失败帧和拒绝原因，不能用后续 SMPL、插值、平滑或旧结果补齐。

## 7. 下一步主线任务

### 7.1 完成含下游模块的 PMPose 回放

在已经验证的 360 对回放基础上，重新运行同一输入，但开启当前主线的下游模块：下肢状态、Stage walker、静态地面参考和完整动态 SE(3) 选项。唯一意图是生成当前相机位置下的一套同源中间结果，不做扩框开关对照。

必须核查：

- 左右旋转方向是否使人体正立；
- PMPose 二维结果是否仍回到原始鱼眼坐标；
- 三角化是否使用新标定而非旧外参；
- 右侧越界拒绝点是否完整保留；
- Stage 1/2 是否按当前协议转换；
- 动态地面是否真的更新，不能只看文件存在；
- 助步器节点/边是否来自现有拓扑；
- 所有输出是否来自同一批 360 对输入。

### 7.2 生成 SMPL 输入并执行主线拟合

使用本次 PMPose 回放的 `stereo_results.jsonl`，通过现有 PMPose 导出适配器生成 SMPL 输入。SMPL 拟合仍使用固定 COCO-17 regressor、male SMPL 6890 顶点和当前地面/助步器场景变换。

拟合顺序保持：

- Stage A：beta 固定，逐帧姿态和根变量；
- Stage B：冻结运动，只更新共享 beta；
- Stage C：联合拟合；
- Stage D：同一进程继续，逐元素冻结 beta，再加入表面手脚项或当前已冻结的时序项。

在 SMPL 之前必须先检查：输入帧数、17 个 COCO 索引、finite/NaN、accepted/rejected mask、相机坐标系和地面变换来源。若二维或三角化输入不完整，停止在 SMPL 入口，不用拟合结果掩盖上游问题。

### 7.3 手部模型暂不改变身体主线

WiLoR/InterWild 仍作为后续手部观测候选，不阻塞当前 SMPL 身体主线。下一步先把 PMPose+三角化+SMPL 主线在新相机数据上跑通，再在同一帧范围上接入 WiLoR 手部观测；只有手部观测契约稳定后，才讨论 SMPL-H/SMPL-X 统一参数化。

## 8. 推荐下一次运行命令

下一次直接运行包含当前主线下游模块的 PMPose 回放，不切换扩框开关：

```powershell
cd D:\my_works\walker_pose_system\realtime_app

..\.venv-cuda\Scripts\python.exe run_stereo.py `
  --config .\config.json `
  --model pmpose `
  --left-model-rotation ccw90 `
  --right-model-rotation cw90 `
  --stereo-capture-dir .\outputs\stereo_capture\20260927_201716_220 `
  --calibration .\calibration\results\stereo_fisheye_20260927_195919.json `
  --max-pair-delta-ms 25 `
  --warn-skew-ms 25 `
  --keypoint-threshold 0.25 `
  --max-association-cost 0.05 `
  --max-reprojection-error-px 10 `
  --stereo-subject-mode single `
  --pmpose-adaptive-box on `
  --enable-lower-limb-pipeline `
  --enable-stage-walker `
  --walker-reconstruction-interval 10 `
  --dynamic-ground-reference .\calibration\results\static_ground_20260927_201028\ground_reference.json `
  --dynamic-ground-full-se3 `
  --output-dir .\outputs\pipeline_20260927_201716_pmpose_mainline `
  --output-fps 30 `
  --headless
```

## 8. 工作区与交接纪律

当前工作区存在大量用户已有修改、实验产物和未跟踪文件，不能执行 `git reset --hard`、批量删除或覆盖。自适应扩框应继续维护原有 `build_continuous_foot_inclusive_roi.py` 单一实现；交接者只应修改与当前阶段直接相关的文件。

实验记录必须写输入路径、帧范围、参数、输出路径和结论边界，不在实验记录中写 Git 哈希。所有失败帧和拒绝原因必须保留。当前先完成主线闭环，不把扩框统计写成“准确性提升”。
