# SMPL-H二维运动跟踪交接

更新日期：2026-10-15。仓库：`D:\my_works\walker_pose_system`，优先PowerShell 7。本文用于新窗口接手当前工作；有效算法与公式见[算法说明](CURRENT_SMPLH_ALGORITHM_20261002.md)。本文与算法说明均不覆盖`AGENTS.md`、`AI_PROGRESS.md`及实验协议。

本次只整理文档并写交接，没有启动新的拟合、训练、短窗实现或数据采集。下述“已完成”来自保存结果及此前针对性检查；“下一步”尚未实施。不要因为交接列出了命令就默认重跑全部实验。

## 1. 用户目标与不能改变的约定

用户要求先用第一帧初始化人体，初速暂设0，用当前及历史双目二维关节点驱动人体运动，得到速度、加速度曲线，并逐步减少场景对Stage规则的依赖。后来进一步明确：

- 拟合对象与最终结果都必须是完整SMPL-H；通过regressor构造骨架，只是为了约束这个模型，不交付一个脱离模型的自由骨架。
- 跟踪损失直接使用左右原始鱼眼二维像素，不添加逐帧三维关节点观测；模型内部三维前向不可省略。
- 优先肩、肘、腕、髋、膝、踝；不优化手指和足部细节，冻结这些局部姿态。
- 首帧人工标签的hand=腕、foot=踝；不将踝解释成heel。
- 世界实验中，移动助步器阶段全段强制固定两个脚跟，作为明确假设对照；不能缩成“人工确认静止”的子集，不能悄悄松成软约束。
- 改进必须针对已定位问题，先分析现有方法和相关研究，不不断增加网络、平滑器、物理模块或验证设施。
- 原失败/拒绝/不可用帧全部保留。没有独立三维/物理真值时，低像素误差、平滑、骨长不变和接触代理都不是精度证明。

当前“二维-only”还需分清初始化历史：离线主线仍用三角化初始化；因果模型读取的是frame0-only旧模型参数，其历史初始化曾用三角化，但本轮读取时不把保存的三维点当观测。人工细化与之后跟踪都只用二维。不要声称完全没有任何历史三维初始化。

## 2. 接手时先辨明三种身份

| 路线 | 当前身份 | 权威位置 |
|---|---|---|
| 448二维reliability离线SMPL-H | 用户2026-10-14选定的身体工程主线 | `realtime_app/configs/current_mainline.json/mainline_selection` |
| 完整SMPL-H因果运动跟踪 | 已实现、完整448；latent1.2为下一轮候选，未晋升主线 | `research_records/engineering_validation/G20261014_smplh_motion448_v1/protocol.json` |
| 物理PPO | 已训练和回放，当前候选失败并停止 | `research_records/engineering_validation/G20261014_rl_physics448_v1/protocol.json` |

当前身体主线是`G20261013_reliability448_view_v1/fit/result.npz`，不是360 reference25。主线参数为reliability/quality、3D权重0、二维权重0.004、像素尺度25、踝倍率2、身体时序0；共享体型，native MANO0.1/共享PCA12，手二维和接触关闭，A30/B30/C300/D100/D100/D3zero。`accepted_for_main_fit=false`仍成立，工程选择不等于物理认证。

场景主线为`mainline -> supported_stop_fixed_lag`，名义发布延迟0.5s。其停止即同平地四脚支撑的采集假设仍在；实时接入未认证。独立ORB/cuVSLAM未通过，未替换场景；Stage脚锚不能自证独立世界轨迹。

主线离线结果使用全序列联合优化/共享体型，包含未来信息，不能作为因果跟踪的同信息真值。因果实验首帧禁止直接读取该全片拟合的共享beta或未来姿态。

## 3. 最小阅读顺序和文件职责

1. `AGENTS.md`，然后`AI_PROGRESS.md`顶部2026-10-15与相关2026-10-14条目。
2. `docs/CURRENT_SMPLH_ALGORITHM_20261002.md`第1节核对身份，第4–6节核对跟踪与计划。其第2–3节供修改离线拟合器使用。
3. 因果实验根`G20261014_smplh_motion448_v1/protocol.json`，再读`run_latent_cap12/run_parameters.json`和`summary.json`；0.4对照见`run_final_v3`。
4. 修改时读取下表对应源码；不为找上下文重跑检测、拟合、训练或遍历全部原始图。

| 文件 | 负责什么 |
|---|---|
| `realtime_app/pose_app/smplh_motion_tracker.py` | 实际模型映射、SO(3)状态、预测、二维EKF反馈、协方差、失效恢复、模型点导数 |
| `realtime_app/pose_app/smplh_first_frame_labels.py` | 人工标签、源图核对、逆转正、首帧二维细化 |
| `realtime_app/tools/run_smplh_motion_baseline.py` | 独立回放入口、配置/全模型NPZ/JSONL/统计/曲线 |
| `realtime_app/tools/run_stereo_pva_baseline.py` | 被复用的二维/时间读取与画图；其自由点滤波器不是当前人体拟合对象 |
| `realtime_app/tests/test_smplh_motion_tracker.py` | 11项针对性状态/反馈/导数测试 |
| `realtime_app/tests/test_smplh_first_frame_labels.py` | 2项标签/图像身份测试 |
| `realtime_app/pose_app/smplh_fitting/{pipeline,losses,initialization,stages}.py` | 现有离线整段拟合器，与独立跟踪器分开 |

当前完整SMPL-H跟踪直接调用tools入口，尚无统一`run_tool.py`别名，不要编造`body.run_smplh_motion_baseline`命令。PPO的统一入口是另一条路线。

## 4. 数据、标注、资产与时间

以下路径相对仓库根；所有输入必须保持同源、有序，不能截取视频前448帧替代。

| 输入 | 路径/契约 |
|---|---|
| 左二维 | `research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/left/pmpose/raw_predictions.json` |
| 右二维 | 同根`c3_predictions/right/pmpose/raw_predictions.json` |
| 有序图像 | 同一people_1根`input_448pairs`，pair0000..0447 |
| 采集时间 | `realtime_app/outputs/people_1/20260908_203954_366/stereo_pairs.csv` |
| 标定 | `realtime_app/calibration/results`，正式`stereo_fisheye.json` |
| 用户首帧标签 | `research_records/annotation_handoffs/H20261014_people1_pair0000/labels/{left,right}_pair_0000.json` |
| 原始frame0-only种子 | `research_records/engineering_validation/G20261014_stereo_pva_baseline448_v2/first_frame_fit/result.npz` |
| 当前已细化首帧 | `research_records/engineering_validation/G20261014_smplh_motion448_v1/run_manual_v1/first_frame_fit/result.npz` |
| SMPL-H与MANO | `third_party/WiLoR/mano_data/models/SMPLH_male.pkl`及`MANO_LEFT.pkl`、`MANO_RIGHT.pkl` |
| 观测regressor | `models/smpl/J_regressor_coco.npy`，17×6890 |
| VPoser | `models/VPoser02_05/V02_05` |
| Python环境 | `.venv-cuda/Scripts/python.exe`；当前回放使用CUDA |

cam0=LEFT、cam1=RIGHT，标定使用完整R/t。原图1920×1080，模型输入转正为1080×1920。标注/自动点进入几何前：左raw=(1919-v,u)，右raw=(v,1079-u)。标定T原为mm，模型内m；残差px，旋转rad，关节导数导出mm/s和mm/s²。不能把mm数组直接与模型米制相减。

时间为左右host-read-return时间的中点、减首帧；全片跨度16.035239s，dt为0.0069816..0.0695054s，中位0.0397631s，不使用名义30fps积分。左右返回时差中位3.5254ms/P955.97451ms。返回时间不是曝光时间，不能据此证明双目同步或生理运动时基正确。

人工左右各8点，共16点：肩5/6、腕9/10、膝13/14、踝15/16；`leftt_hand`显式映射左腕。源图像素身份、尺寸、重复点、越界和中文路径已有检查。首帧肘/髋仍取自动点，frame1起全部使用自动点。

首帧二维细化已做Adam500步：latent lr0.02、根/平移0.005、beta0.01；人工/自动sigma4/15px、3sigma Huber、latent prior sigma1.5、beta prior sigma2，手足颈头冻结。细化后冻结beta；人工16点残差median10.38450/max61.64695px。人工点参与拟合，不能再作为当前SMPL-H的held-out参考。重新回放复用此结果并显式`--first-frame-refine-steps 0`，不重复500步。

## 5. 当前跟踪器实际逻辑

- 名义姿态q=(根平移3、根SO(3)局部3、VPoser32)，局部38维；v/a各38，误差协方差114×114。
- VPoser、SMPL-H及regressor冻结；beta与双手各45维冻结；排除root的身体索引9/10（足）、11/14（颈头）固定。只控制其余身体链，不独立驱动12个自由xyz点。
- 每步按真实dt预测q/v/a，根采用右乘Exp；白jerk过程噪声使加速度可变化。平移/root/latent jerk谱密度分别0.25/1/4，不能称物理驱动力。
- 初始位姿标准差(t/root/z)=0.1m/0.35rad/0.5；速度0.5m/s/1rad/s/1.5每秒；加速度1m/s²/2rad/s²/3每秒²。R/Q和P是工程设定，未校准。
- 同一网格回归COCO17，使用5..16的12点。每视图score0.2..1且有限/界内；双目同名点做raw像素Sampson35px极线门，不三角化。允许单目支持，不将其称独立深度。
- 至少4个不同关节点可观测、12个模型点双目均正深度时更新。±0.001中心差分产生77个批模型评估，H只依赖姿态，v/a经交叉协方差同时被像素纠正。
- 基础sigma8px、Huber25px、latent prior sigma1.5；只做一次EKF反馈，不重复用同帧像素迭代收缩P。
- t/root/latent修正范数上限0.08m/0.12rad/默认0.4；1.2候选只覆盖latent。取共同s，检查s/s2/s4真实非线性总目标不增、正深度和|latent分量|≤6。
- 均值q/v/a与Joseph协方差必须共用sK；接受后做SO(3)误差重置并刷新点Jacobian。更新失败仅预测并保留原因。
- 整体失观0.1s后失效；恢复从最近接受的模型位姿用当前二维校正，v/a清零、P恢复、新segment。不读取三维重启点或未来拟合结果。

关节v/a由同一模型在q附近按eps0.01s计算瞬时导数，包含运动学曲率；不是对最终逐帧位置做差分，也不是简单J乘参数加速度。参数速度不是直接的关节角速度/肌肉驱动。

导数发布还需本点近期0.1s像素与好拟合、segment热身0.15s、当前位置标准差≤0.1m、当前所有合法raw视图回投≤25px。被极线或更新拒绝的原始像素仍参与质量判断。未通过时发表值NaN，raw导数及原因保留，不补洞或裁剪尖峰；数值可解码和导数可用是不同标志。

当前坐标是每时刻左相机系，v/a包含相机运动；`world_state_available=false`。没有Stage、地面、脚跟锚或动力学。世界导数需要Rdot/Rddot和相机平移导数，不能只乘旋转矩阵。

## 6. 已有结果：严格区分单变量和多因素版本

实验根记为`research_records/engineering_validation/G20261014_smplh_motion448_v1`。

| 项目 | 原始完整基线run_final_v3 | 当前候选run_latent_cap12 |
|---|---:|---:|
| latent纠正上限 | 0.4 | 1.2 |
| 数值可解码 | 448/448 | 448/448 |
| 全部raw像素槽 | 10752 | 10752 |
| raw误差median/P95 | 36.23725/135.28643px | 18.04503/61.76466px |
| raw误差max | 486.76369px | 143.50354px |
| raw≤25px | 34.18899% | 66.88058% |
| 满足导数条件 | 1065/5376，19.81027% | 2846/5376，52.93899% |
| step median/P95 | 71.97225/99.61367ms | 73.39020/98.36702ms |
| 重启 | frame76 | frame406 |

10752=448×12×2视图，5376=448×12关节点时刻；不能把实际used视图当全部分母。已核这两版configuration唯一差异是latent上限；其他输入、形状、手姿、faces、时间、标定、R/Q和质量门均相同。

问题定位：原版443次正常更新全部由latent主导共同限幅，423次达到0.4上限，占95.49%；根/平移未达上限却一同被s抑制。只放宽这一数值瓶颈后，七个64帧连续区间的像素中位数均改善。0.4和1.2不是具有直接物理单位的运动上限；1.2也不是全局最优值，决定不继续扫更多上限。类/CLI默认仍为0.4，下一轮需显式传1.2。

候选状态initialized1/updated444/predicted2/reinitialized1；导数available2846、warmup120、poor/stale2358、无近期本点52。共同可用892槽的a模长P95为3.96564→4.02471m/s²，基本未改善。两版各自发表集合变化很大，不能用较小峰值或曲线平滑说明加速度更准。

1.2逐关节剩余问题：

| 点 | raw median/P95，px | 导数帧/448 |
|---|---|---|
| 左/右肩 | 15.08/35.14；16.38/31.65 | 278；286 |
| 左/右肘 | 16.57/43.87；11.96/29.32 | 283；362 |
| 左/右腕 | 13.05/35.77；10.56/30.65 | 325；359 |
| 左/右髋 | 36.22/72.71；41.48/80.98 | 28；22 |
| 左/右膝 | 25.04/69.14；31.13/92.25 | 169；96 |
| 左/右踝 | 14.21/40.78；16.17/43.83 | 332；306 |

### 6.1 必须保留的问题与失败教训

- 髋点仍差且没有人工髋/肘参考；单帧beta和表面回归语义可能有偏差，尚未区分原因。不能删髋或增点偏置来粉饰总体误差。
- 12点双目48像素残差经12×3点位，瞬时姿态H的秩最多36，小于38个姿态变量；先验补弱方向不等于像素识别该方向，完整表面尤其未被骨架充分约束。
- manual/auto首帧16点差mean28.98/max68.64px，frame1切回自动可能产生暂态；warmup不消除偏差。
- 首帧右肩raw Sampson约30.17px、细化左目残差61.65px，不能全归滤波，不能用人体拟合代替独立相机标定。
- 同模型当帧静态二维诊断：frame224 median36.37→10.83px，frame412 151.01→36.66px。仅作诊断、不插入轨迹；见`run_final_v2/static_fit_diagnostic.json`。
- 早期`run_final`因mean限幅但P完整收缩导致漂移，P95677.54px；已放弃该迭代实现。`run`仅36帧可用，其他中间目录及失败日志保留。
- `run_manual_v1`整段轨迹不是当前候选，但其first_frame_fit被复用。其他早期版本多因素改变，不能当单变量消融。
- 导数是后验模型的瞬时估计；观测纠正引起的帧间位置跳变和模型瞬时速度不同，目前没有独立3D或导数真值验证。

## 7. 输出与已有验证能证明什么

每版目录包含：

- `result.npz`：448×6890×3米制vertices、13776×3真实faces，根/63维body_pose/冻结beta及双手45维、VPoser32、COCO17/原生52点、广义v/a、时间/segment、pose与derivative标志。
- `joint_velocity_mm_s/joint_acceleration_mm_s2`为448×12×3发布值，失败NaN；`raw_joint_*`保存未通过质量门的原始模型导数。
- `trajectory.jsonl`全部448有序记录：q/v/a、所有raw误差、used/rejected/导数原因、协方差对角、s、时差和世界不可用标志。
- `run_parameters.json`、`summary.json`；`velocity_components.png`、`acceleration_components.png`和`full_mesh_stereo_overlay.jpg`。候选另有`step_bound_comparison.png`。图不平滑、不插值、不裁掉尖峰。

此前13项针对性测试、五文件编译和Ruff E9/F通过；1.2入口已实际完整回放并核对单变量、冻结字段、帧数、导数NaN门。真实模型参数再生mesh最大差约4.17e-7m，发布轴角再生约3.58e-7m，全448网格regressor映射约1.19e-7m。证明的是接口/数值内部一致性，不是真实姿态或加速度准确。

本次文档整理只核对当前身份、记录字段/数值、引用与链接，不重跑这些已通过的模型检查。大型NPZ/图像、标注和权重未随源码提交；克隆仓库不等于已有全部运行资源。

## 8. 命令与恢复规则

仅在准备复现或修改相关算法后需要时使用；本交接不要求重新运行已完成的448结果。

### 8.1 复现1.2候选

在仓库根执行，输出目录必须不存在或为空；不要覆盖现有`run_latent_cap12`。

```powershell
Set-Location D:\my_works\walker_pose_system
& ./.venv-cuda/Scripts/python.exe realtime_app/tools/run_smplh_motion_baseline.py `
  --left-predictions research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/left/pmpose/raw_predictions.json `
  --right-predictions research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/right/pmpose/raw_predictions.json `
  --input-pairs research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/input_448pairs `
  --initial-fit research_records/engineering_validation/G20261014_smplh_motion448_v1/run_manual_v1/first_frame_fit/result.npz `
  --first-frame-labels research_records/annotation_handoffs/H20261014_people1_pair0000/labels `
  --first-frame-refine-steps 0 `
  --max-latent-correction-norm 1.2 `
  --output research_records/engineering_validation/G20261014_smplh_motion448_v1/new_candidate12
```

若需0.4对照，显式改上限和新输出目录。若需重做首帧细化，改用第4节未细化的frame0-only种子并将refine设500；这会改变实验初值，不能混称原单变量对照。禁止整段模型文件作初值，读取器会检查pair_id=[0]。

入口拒绝非空输出，没有恢复Adam/滤波器的断点继续机制。中断目录和日志保留；诊断具体失败后，在新目录完整回放。不能只从中断帧续接而丢掉历史协方差，也不复制成功帧伪装完整输出。图像/标定/帧集合/资产错误先停，不能放门或混旧结果。

### 8.2 修改相关源码后最小测试

在仓库根进入应用目录，使用同一Python运行两个现有测试文件；这两个命令合计对应13项，不运行全仓测试。

```powershell
Push-Location realtime_app
try {
  & ../.venv-cuda/Scripts/python.exe -m unittest discover -s tests -p test_smplh_motion_tracker.py
  & ../.venv-cuda/Scripts/python.exe -m unittest discover -s tests -p test_smplh_first_frame_labels.py
} finally {
  Pop-Location
}
```

测试检查特定状态/映射契约，实际资产映射只在改变模型/解码/regressor时才需再次核对。纯文档整理不需要重跑模型或全仓测试。

## 9. 下一步：按问题推进，不堆模块

### 第一阶段：观测与已知运动，优先做这一阶段

冻结1.2候选、同一模型/体型/二维门/时间，先核对髋和右膝的COCO索引、regressor语义、遮挡和标定。独立少量多帧人工参考不得同时进入拟合；现有16首帧点不能再复用作独立验证。曝光问题需要真实采集证据，不能从host日志推断已解决。

随后用同SMPL-H和鱼眼投影生成小规模已知运动：静止、匀速、匀加速、正弦肢段、停止转向、人体静止/相机移动；真实非均匀dt、2/8px已知噪声及短遮挡。不能只给常加速度模型出二次轨迹；已知3D点/导数只评价，不进入二维损失。完整clip分开发/held-out，记录位置、速度、加速度误差、滞后、覆盖和共同可用集合。

此阶段尚未实现。它回答估计器能否在已知噪声/运动下恢复导数，不回答真实人体三维准确率。遇到观测/语义契约冲突先处理冲突，不继续扩展时序模型。

### 第二阶段：证实时序不足才替换为因果短窗

拟议窗口[t_k-0.3,t_k]、当前(q,v,a)共114个局部变量，生成历史模型并直接投影各历史双目像素；固定形状和观测定义，只发布当前帧，不重写过去。替换EKF反馈，不在其后再叠平滑器/第二个同源融合器。

平移/latent采用局部二次轨迹，根R_i=R_k Exp(omega*tau+0.5*alpha*tau²)。omega/alpha仅是当前时刻局部体角速率；历史体角速度为J_r(xi)(omega+alpha*tau)，不假设全窗恒定物理角加速度。历史点导数从完整模型计算。

上一结果只作warm start。每窗只用当前窗口像素和固定弱姿态/速率先验独立求解；不再叠加已经含同窗像素的EKF后验当独立先验。相邻输出相关。当前不建边缘化或图优化设施。

不能继承旧EKF的P/不确定性门。最小拟议工程近似是数据曲率Jdata^T W Jdata加固定先验曲率Lambda后的逆，再经模型点Jacobian传播；病态/非有限不可用。必须区分数据弱方向与先验贡献，在已知噪声试验检验后确定阈值，不因正则逆Hessian小就声称可辨识或概率已校准。

预设工程门：完整held-out已知运动加速度RMSE比固定候选至少下降20%，位置误差相对恶化不超过5%，滞后不明显变差；全样本覆盖和共同可用集合同时报告。真实448全raw median/P95与髋膝分项保留，不靠丢点/未来/改历史输出改善。这些门尚未通过，不是论文保证；最多两次有依据修订，未通过则停并分析，不自动换神经网络。

### 第三阶段：可靠世界轨迹与全段双heel假设

独立相机相对运动可靠后，在固定局部坐标内估计人体并用各历史相机变换投影，再统一人体/相机/刚体助步器到世界。无需先重建完整房间。现有ORB/cuVSLAM跳变不满足此条件。

再做“无heel锚 / 移动助步器全阶段两个真实模型heel世界硬固定”的单变量对照。现有Stage2计115帧，transition另保留；不能改区间让结果好看。不能用踝替heel，不能悄悄软化锚或隐藏投影冲突/不可行。该约束尚未接入当前因果SMPL-H。

### 学习方法和速度优化的进入条件

只有经典同定义估计器在独立参考上仍有明确未解决误差，才考虑一种学习运动先验的额外收益。当前不新增HuMoR、RL或另一个图像网络；已有PPO失败不等于整个RL范式被否定，也不构成继续盲目加训练的理由。

相关研究已读：[SMPLify-X/VPoser](https://arxiv.org/html/1904.05866)、[HuMoR/VPoser-t](https://arxiv.org/html/2105.04668)及[官方代码](https://github.com/davrempe/humor)、[SLAHMR作者项目](https://vye16.github.io/slahmr/)。借鉴模型参数拟合、姿态/运动先验区分、过度平滑风险与相机/人体解耦，不声称现方案有新的研究优先权。具体适配限制见算法第6节。

拟合与导数通过后，再在同一模型内考虑Jacobian成本优化。77姿态批评估已是主要成本；自动微分/解析替换未实施，不先承诺短窗实时。

## 10. 工作区与交付注意

本轮开始即有大量其他任务的修改、删除和未跟踪源码；这些不属于本次交接，不得清理、重置或顺带提交。每次先`git status --short`和实际diff，只提交自己的文件/代码块。`AI_PROGRESS.md`已有两处旧空行增量，本轮日志需按新增块隔离暂存。

不得把原始数据、模型权重、大NPZ/图像顺带提交。按AGENTS要求为已验证的独立修改创建本地提交，不推送；实验/交接记录不写Git哈希，提交ID只在用户交付消息中报告。

算法说明已重排：旧27–30节→现2–3，旧31–32/35→现1，旧39→现4，旧40→现5–6；旧历史结论在现7。四个仓库依赖锚点已保留。旧日志节号按原时间理解，不为整理本页重写所有历史记录。

接手后的第一项实质任务是第9节第一阶段，先用结果说明问题再决定实现范围。不要把本次文档交接当成授权立即启动全部未来路线，也不要在交接后自动重跑448、训练或新增模块。
