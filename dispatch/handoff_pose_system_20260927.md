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

按原有逻辑完成了一次 360 对回放：输出为 `realtime_app/outputs/pipeline_20260927_201716_pmpose_original_roi`；720 次左右模型调用中 887 个检测框扩展，平均有效三维点 8.27/17，平均配对处理速度 2.308 对/秒，右侧越界拒绝点 168 个。此次运行确认了接入路径，但没有证明精度提升；仍需关闭扩框做严格 A/B 对照。

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

## 6. 下一阶段实验方向

必须按以下顺序推进，每次只改变一个主要变量。

### 阶段 A：验证自适应扩框是否真的改善二维观测

固定：新标定、360 对回放、旋转方向、YOLO 权重、PMPose 权重、阈值、关联和三角化质量门。

只比较：

1. `--pmpose-adaptive-box off`
2. `--pmpose-adaptive-box on`（本次已完成）

必须输出并对齐：

- 每帧左右原框与扩框；
- 17 个关键点置信度和缺失情况；
- 每关节双目有效/拒绝数量；
- 三角化重投影误差、射线间隙和拒绝原因；
- 下肢关键点覆盖；
- 失败帧完整保留。

停止条件：扩框导致有效点减少、误检增加、关联错误或几何质量下降时，不能默认保留扩框；应退回原框或改成分段规则。

### 阶段 B：只在 A 通过后测试局部虚拟透视

当前已有 `--model-input-local-perspective {off,auto,always}`。它是第二次局部模型输入推理，不等同于扩展 YOLO 框。

建议先比较：

```text
adaptive on + local perspective off
adaptive on + local perspective auto
```

不能同时改变扩框阈值、局部 margin、三角化门和 SMPL 拟合权重。

### 阶段 C：手部模型候选对照

只使用 A/B 确认后的二维人体主线作为固定身体输入，分别运行：

1. PMPose 身体主线；
2. WiLoR 手部观测；
3. InterWild 手部观测。

比较内容应是遮挡片段的手部观测可用率、左右一致性、关键点置信度、重投影/三角化内部一致性和失败模式，不得直接比较“谁更像真实手势”。

### 阶段 D：统一人体模型方案选择

只有当 WiLoR/InterWild 手部观测在固定片段上稳定后，才选择：

- 继续 SMPL 身体 + 手部观测约束；或
- 迁移到 SMPL-H/SMPL-X，重新建立身体、手部和 COCO 回归器的统一参数化。

迁移模型必须重新验证模型顶点、关节回归器、坐标系、相机投影和手脚表面项，不能只替换 checkpoint。

## 7. 推荐下一次运行命令

先做自适应扩框 A/B 对照，暂不进入正式 SMPL 拟合：

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
  --output-dir .\outputs\pipeline_20260927_201716_pmpose_adaptive `
  --output-fps 30 `
  --headless
```

然后只把 `--pmpose-adaptive-box on` 改为 `off`，输出到另一个目录。两次都不要启用 SMPL、动态地面或助步器后处理，先完成二维/三角化数据审计。

## 8. 工作区与交接纪律

当前工作区存在大量用户已有修改、实验产物和未跟踪文件，不能执行 `git reset --hard`、批量删除或覆盖。自适应扩框应继续维护原有 `build_continuous_foot_inclusive_roi.py` 单一实现；交接者只应修改与当前阶段直接相关的文件。

实验记录必须写输入路径、帧范围、参数、输出路径和结论边界，不在实验记录中写 Git 哈希。所有失败帧和拒绝原因必须保留。完成 A/B 后，再更新 `AI_PROGRESS.md` 和对应实验记录，不能先写“准确性提升”。
