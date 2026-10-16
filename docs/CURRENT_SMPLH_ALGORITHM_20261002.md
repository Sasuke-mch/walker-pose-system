# 当前 SMPL-H 算法：有效实现、结果与改进边界

更新日期：2026-10-15。仓库根目录：`D:\my_works\walker_pose_system`。本文整理当前有效算法，历史运行日志以`AI_PROGRESS.md`及各实验协议为准；不再按日期追加重复实现。权威顺序仍遵守`AGENTS.md`。

本次整理未改求解器、未重跑模型、未切换主线。旧版错误公式、被替代的计划、旧展示端口和重复结果表从本文移除；原实验的失败帧、拒绝原因、数值结果和日志均保留原位置。

接手顺序：先读第1节区分主线与候选，再读第4–6节理解当前跟踪；修改离线整段拟合器时读第2–3节。详细路径、运行命令和下一阶段任务见[交接说明](HANDOFF_SMPLH_TRACKING_20261015.md)。

- [1. 当前选择与范围](#current-selection)
- [2. 公共输入、坐标、模型与模块](#model-contracts)
- [3. 离线整段拟合器](#implementation-guide)
- [4. 完整SMPL-H因果二维运动基线](#causal-smplh)
- [5. 全448结果、单变量对照与剩余问题](#tracking-results)
- [6. 研究依据与未实施改进](#next-plan)
- [7. 已停止路线与必须记住的修复](#historical-boundaries)
- [8. 验证范围与使用入口](#validation-and-use)

<a id="current-selection"></a>
<a id="35-当前人体主线切换为二维鱼眼重投影约束-2026-10-14"></a>

## 1. 当前选择与范围

正式选择读取`realtime_app/configs/current_mainline.json/mainline_selection`。该JSON顶层的早期Sapiens2、360/448旧配置等字段只保留历史语义，不能覆盖`mainline_selection`。

| 对象 | 当前身份与路径 | 边界 |
|---|---|---|
| 身体主线 | `research_records/engineering_validation/G20261013_reliability448_view_v1/fit/result.npz`，PMPose、448帧、离线整段SMPL-H | 2026-10-14已选择的二维工程路线；`accepted_for_main_fit=false` |
| 场景主线 | `mainline -> supported_stop_fixed_lag` | 同平地、停止即四脚支撑的采集假设；名义发布延迟0.5s；未完成现场端到端认证 |
| 当前固定体型跟踪候选 | `G20261014_smplh_motion448_v1/run_calibrated_beta12` | 用户指定全帧校准beta；姿态二维因果反馈，脚部/像素拟合仍不足，未替换主线 |
| 原1.2跟踪候选 | 同根`run_latent_cap12` | 首帧联合细化beta；保留完整原结果及0.4/1.2单变量证据 |
| 跟踪原始对照 | 同根`run_final_v3` | latent修正上限0.4，仍为类/CLI默认；下一轮显式使用1.2候选 |
| 历史身体对照 | `G20261005_reference_weight_fit_v1/runs/reference25`，360帧 | 已被448二维选择替代；保留结果，不混入当前参数 |
| 物理PPO / 独立视觉位姿 | `G20261014_rl_physics448_v1` / `G20261014_visual_rig_compare448_v1` | 已真实运行但未通过，未晋升；见第7节 |

上表缩写实验根均位于`research_records/engineering_validation/`。离线整段方法使用全片共享体型和全片联合优化，不能作为与因果方法信息量相同的在线对照或独立真值。

### 1.1 当前离线主线实际配置

```text
fit_policy=reliability; body_weighting=quality
body_3d_weight=0; body_observation_mode=reprojection_only
body_reprojection_weight=0.004; body_reprojection_scale_px=25
body_ankle_reprojection_multiplier=2; body_temporal_weight=0
VPoser=0.01; beta=0.001; root_anchor=0.01
native MANO=0.1; native初始化; shared PCA12; hand_2d_weight=0
无人工三维腕约束；无接触/脚跟约束
A30 / B30 / C300 / D1 100 / D2 100 / D3 0；seed0
```

身体优化只比较模型回归点的左右原始鱼眼二维投影，不使用三角化关节点的三维残差或梯度。SMPL-H自身仍是三维模型；三角化仍用于冷启动、初始化门和诊断。这与第4节跟踪器“更新时不三角化、不读取三维观测”要分别理解。

踝倍率2只改变二维Huber项中COCO15/16的任务优先级；reliability归约的分母固定为原始槽数，不是置信度和。它不表示踝接触、脚跟固定或足底受力。

### 1.2 当前产物与实测口径

当前离线根目录内：`plan.json`记录本次生成时协议；`fit/fit_args.json`是实际argv，`fit/resolved_options.json`是解析参数，`fit/fit_summary.json`记录阶段与结果；`view/body.html`是完整离线页，`view/body_browser.html`与`body_data.json.gz`配套，`view/scene/`保存同源场景。生成时plan中的“尚未选择主线”是历史字段，后来选择身份以2026-10-14的`mainline_selection`为准。

560步完成，448帧模型有限，初始化条件加权RMS127.15mm（原门300mm）。既有19项同源/坐标检查、入口测试及浏览器控件检查通过；这些证据只表示工程输出与显示一致性，不证明三维或接触精度。显示规范见`VISUALIZATION_PIPELINE.md`。

本机RTX5070Ti Laptop、PyTorch2.11/cu128、Python3.12.12下，独立一次完整448拟合耗时42.0031s（准备3.5750s、优化36.7760s、最终解码0.0282s、回传/保存1.6239s）。这是离线吞吐10.6659帧/s，不是单帧因果延迟。已有参数解码经10次预热/50次采样：单帧wall中位14.3135ms，批448中位29.7663ms；均不含检测或拟合。完整拟合只测一次，没有完整拟合P95。原始口径见`G20261013_smplh_runtime448_v1/runtime.json`，不因本次整理重新测速。

<a id="model-contracts"></a>

## 2. 公共输入、坐标、模型与模块

### 2.1 入口和模块职责

| 文件或函数 | 实际职责 |
|---|---|
| `run_tool.py body.fit_smplh_wilor_sequence` | 统一命令分发，不改参数与工作目录 |
| `walker_tools/body/fit_smplh_wilor_sequence.py::build_parser/main` | 参数默认值、解析，调用`run_fit` |
| `tools/fit_smplh_wilor_sequence.py` | 历史入口兼容转发，不是第二套算法 |
| `pose_app/smplh_fitting/pipeline.py::run_fit` | 读取、初始化、组装损失、优化循环和输出 |
| `pose_app/smplh_fitting/stages.py` | 参数合法性、每阶段预算、变量冻结、局部手计算路径 |
| `pose_app/smplh_fitting/losses.py` | 有效像素、身体权重、旋转锚、观测归约和时序项 |
| `pose_app/smplh_fitting/initialization.py` | 单模板、肩髋方向、同模板平移和初始化门 |
| `pose_app/body_observations.py` | 身体JSON读取、逆转正、软权候选三角化 |
| `pose_app/fisheye_camera.py`、`pose_app/smpl_coco_observation.py` | 标定投影与posed vertices到COCO17回归 |
| `pose_app/smplh_hand_observation.py` | 双相机×双解剖手关联、21点顺序、像素辅助观测 |
| `pose_app/wilor_mano_prior.py` | 资产/参数约定、局部旋转转换、PCA编码、完整初值选择和旋转损失 |
| `pose_app/smplh_fitting/contact.py`、`pose_app/smpl_surface_contact.py` | 可选接触输入检查、坐标变换与表面损失 |

修改数值公式应定位到对应组件；阶段何时启用该项则要同时看`pipeline.py`。只阅读CLI帮助或单个损失函数，无法确定最后参与反向传播的完整目标。

### 2.2 输入数据、帧顺序、坐标和单位

记序列长度为`N`，身体关节数`J=17`，相机数`V=2`。

| 输入 | 内容与契约 |
|---|---|
| `--left-raw / --right-raw` | JSON的`images`记录；每项使用`keypoints[0]`，接受17×3或23×3，23通道只取前17个COCO点；三列为u、v、score |
| 帧编号 | 优先`image_id`，其次`frame_index`；左右帧集合必须相同，并且排序后为0..N−1；不会取交集静默删帧 |
| `--wilor-left / --wilor-right` | 分别表示左/右**相机**的JSONL；每个文件都可能含解剖学左手与右手，不能按文件名只读一只手 |
| `--calibration-dir` | 双鱼眼K、D、cam0到cam1的R与毫米制T及图像尺寸；拟合不优化标定 |
| `--regressor` | 从6890个posed vertices回归COCO17的固定矩阵 |
| `--smplh-model` | 真实SMPL-H模型资产，检查6890顶点及手关节/均值契约 |
| `--vposer-dir` | CLI解析层允许省略，但`run_fit`运行时必需；不会默默切换为自由63维身体轴角 |
| `--mano-left / --mano-right` | 对应手的均值与PCA基；native路径还核对canonical right MANO资产 |

人体检测是在转正后的图像上产生的。当前读取器沿用1920×1080原图约定：左侧`(x,y)=(1919−v,u)`，右侧`(x,y)=(v,1079−u)`；先逆转正，再做关联与鱼眼几何。这里的1920/1080是实现中的固定约定，不是自动适配任意分辨率；更换分辨率/旋转方式必须修改并核对读取约定，不能仅更换标定文件。

左右鱼眼点先去畸变到归一化成像平面，以`P0=[I|0]`、`P1=[R01|T01]`三角化。`X_cam1=R01 X_cam0+T01`。原始三角化、射线距离及标定平移使用**mm**；进入SMPL-H拟合后统一换为**m**，像素残差为**px**，轴角为**rad**。地面变换的平移也必须从mm换成m后与网格运算。

身体输入数组是`N×17×3`，三角化目标同形。三角化还返回左右回投误差、射线最近距离、深度、accepted和逐点原因。此读取器是身体软权候选路径，**不是严格场景三角化门**；`body_accepted`、`body_fit_support`与Stage/地面使用的严格掩码是不同变量，不能互相替换。二维模式仍使用三角化进行初始化和诊断，并非完全取消双目几何依赖。

### 2.3 参数、张量维度与解码

令`K`为每只手的PCA维度（默认12），`S=1`表示全片共享手姿，否则`S=N`。

| 代码变量 | 优化形状 | 含义 |
|---|---|---|
| `beta` | `1×10` | 全序列同一个人的体型，调用模型时展开到N帧 |
| `root` | `N×3` | 根全局旋转的轴角表示 |
| `transl` | `N×3` | cam0坐标中的全局平移，m |
| `latent` | `N×32` | 当前VPoser资产的身体潜变量，解码为`N×63`身体轴角 |
| `lhand / rhand` | 各`S×K` | 两只解剖手的局部PCA系数 |

32维对应当前VPoser资产；实现会检查解码的batch和身体维度。按此维度计算，总未知量为`38N+10+2SK`。共享12维双手时为`38N+34`，逐帧12维双手时为`62N+10`。这只是参数数量，不是每阶段都一起更新。

每只手独立使用均值`μ∈R^45`、PCA行基`C∈R^(K×45)`及行范数`s∈R^K`：

\[
\theta^{hand}=\mu+(c\oslash s)C.
\]

`s`用于基向量尺度归一，不是由当前视频估计的误差标准差。`c=0`得到MANO平均手姿，而不是45个零轴角。共享模式先解码一组局部手姿，再展开到N帧；它只固定手指相对父关节的旋转，不固定手腕位置、手腕朝向或手相对扶手的刚体变换。

VPoser将latent解码为21个身体关节的轴角；SMPL-H结合共享beta、身体与手姿，经固定形状/姿态修正、运动学与线性混合蒙皮得到`N×6890×3`网格。身体观测点使用`COCO17=regressor @ posed_vertices`，不是直接抽取内部骨架的17个关节。手21点来自内部腕/手指关节和5个指尖表面顶点，并重新排列到WiLoR/OpenPose顺序。输出内部`smplh_joints`可能含库附加点，不能仅凭SMPL-H运动学52关节就假定输出恰好52行。

<a id="implementation-guide"></a>
<a id="27-smpl-h基线的方法结构分析与精简2026-10-13"></a>
<a id="28-置信度加权旋转锚与接触保持项的实际修正2026-10-13"></a>

## 3. 离线整段拟合器

通用入口使用固定标定、COCO回归器、SMPL-H、VPoser与MANO资产，分阶段Adam优化共享体型、逐帧根/身体参数及双手PCA。网络权重不训练。当前主线选择第1节的二维目标；以下同时说明仍受维护的三维/手部/接触可选分支，不表示这些分支已在当前结果启用。通用CLI默认与所选实验参数不同，必须读取保存的argv。

```text
左右身体JSON → 原始鱼眼像素/有效性/分数
             → 固定标定几何 → 肩髋初始化与诊断
共享beta、逐帧root/transl/VPoser、双手PCA
             → 固定解码器/网格/回归关键点
             → 已选择的一种身体数据项 + 手项 + 参数先验
             → A/B/C/D阶段 → 最终完整模型和来源状态
```

### 3.1 身体可靠程度：从原始score到最终权重

新策略把三个概念分开：**是否可用**由mask决定，**该观测有多可信**由score与几何衰减决定，**任务强调程度**由外部λ或踝倍率决定。mask不是分数，λ也不是概率。

每视图二维支持要求坐标与分数有限、在原始图像范围内且`0<score≤1`。非法或缺失观测保留原始数据和原因；不把负数、超过1的数或NaN裁成有效高分。

三维支持`M_tj`还要求双目几何accepted、三角化有限、左右二维均有效，以及回投误差/射线间距有限非负。对支持点定义：

\[
c_{tj}=\min(c^L_{tj},c^R_{tj}),\qquad
q_{tj}=\max\left(\frac{e^L_{tj}}{25\,px},\frac{e^R_{tj}}{25\,px},\frac{g_{tj}}{80\,mm}\right),
\]
\[
a_{tj}=\frac{1}{1+\max(q_{tj}-1,0)^4},\qquad
w_{tj}=M_{tj}\,c_{tj}\,a_{tj}.
\]

25px/80mm分别由`body_reference_reprojection_px`和`body_reference_ray_gap_mm`配置。取较低视图分数，是因为三角化需要两侧共同支持；几何仅作进一步衰减。平台内不额外降权，超出后连续衰减，无正的置信度或尾权下限。实现采用稳定的倒数形式，避免极大severity直接四次方溢出。

这些权重在优化前从输入观测算好，进入损失时detach；不会根据当前模型拟合得好不好反过来提升观测分数。原`body_quality`仍保存历史几何质量作诊断，新策略实际用`body_fit_weights`。同一个几何问题也不再被多个指数因子连续相乘重复抑制。

**这是一套保留可靠程度的加权逻辑，还不是概率/逆协方差模型。** 检测score没有独立误差标签的经验校准；min规则、25px/80mm及四次尾属于明确的工程选择。小视差导致的三维方向性不确定度并未用完整协方差表示。不能将“分数0.8”解释为“80%概率误差小于某毫米阈值”。

### 3.2 身体数据项、归约与梯度含义

记模型COCO点为`X_tj(Θ)`，三角化目标为`Y_tj`。三维数据项为：

\[
L_{3D}=\frac{1}{NJ}\sum_{t,j}w_{tj}\|X_{tj}(\Theta)-Y_{tj}\|_2^2.
\]

单位为m²，分母固定为原始`N×17`位置数，不是有效点数或权重之和。无效槽不参与算术；实现先选择/隔离支持点，避免`NaN×0`污染梯度。零占位只用于内部安全计算，原始缺失观测和拒绝原因仍保存。

二维数据项直接比较原始鱼眼像素。设`π_v`为固定相机投影、`u_vtj`为观测、`σ_px`为像素尺度：

\[
H(x)=\begin{cases}\tfrac12x^2,&x\le1,\\x-\tfrac12,&x>1,\end{cases}\qquad
L_{2D}=\frac{1}{2NJ}\sum_{v,t,j}m_{vtj}c_{vtj}p_j
H\!\left(\frac{\|\pi_v(X_{tj})-u_{vtj}\|_2}{\sigma_{px}}\right).
\]

`p_j=1`，仅在指定踝倍率时COCO15/16改为该倍率；该参数必须配合`reprojection_only`，明确属于任务优先级。默认`σ_px=100`；历史448二维配方的25px不是通用默认。两个相机共同除以`2N17`，不会各自归一化后再等权平均。模型在有支持的相机点后方时直接失败，不以安全占位掩盖这些点。

相同残差与雅可比下，权重1和0.1对数据梯度的贡献比为10:1。若整段分数都乘0.1，数据梯度也乘0.1，先验强度不变；这正是低质量整段数据应减少对先验覆盖的逻辑。原来的`Σwr/Σw`会抵消这种整体下降。另一方面，固定分母使缺失更多时数据总强度下降，这是设计结果，不是将缺失点恢复成低误差。

新策略要求`body_3d_weight`与`body_reprojection_weight`**恰有一项为正**。同一双目像素及其三角化结果不是两份独立证据，因此不同时作为双正身体目标。`body_observation_mode=legacy`只是旧命名的普通数据模式，不等于`fit_policy=legacy`；前者与新的可靠程度策略可以同时存在。

`body_m`仍是`√(Σw‖X−Y‖²/Σw)`的条件加权诊断；`body_objective_m2`才是上述固定分母三维项。`body_2d_px`是按检测分数加权的原始像素距离均值，不含Huber和踝优先级。诊断数值不能直接相加还原总loss，更不能解释成独立三维精度。

### 3.3 冷启动和初始化拒绝门

1. beta、latent和双手PCA系数从零开始，不加载历史拟合参数或Adam状态。先用一帧零beta/零root/零transl、零latent解码出的身体姿态生成模板；初始化模板明确使用零45维手轴角，与后续PCA零系数对应的平均手姿不同。
2. 用模板及每帧三角化的肩5/6、髋11/12构造身体正交基：左向量来自左右肩差，上方向来自肩中点减髋中点，正交化后取叉积补第三轴。根旋转为`R_init=B_observed B_template^T`，再转轴角。
3. 将同一模板身体姿态展开到N帧，按各帧初始根旋转重新调用SMPL-H。平移为观测髋中点减去**这个旋转后模板**的髋中点，避免方向和平移引用不同模板状态。
4. 所有帧必须满足现有肩髋方向与支持门。函数虽保留单髋平移回退的来源记录，但不能据此声称缺失肩髋的序列已获准绕过总体初始化门。
5. 新策略以`√(Σw‖X_init−Y‖²/Σw)×1000`检查初始化几何尺度，默认阈值300mm；同时记录原无权RMS。非有限、无有效支持或超过门值则停止，不自动扩大阈值、插值或换旧结果。

初始化门检验的是有支持点上的几何误差，所以使用条件加权均值；如果也除原始位置数，整段极低分反而可能把错误初始化压到门下。它的目的与混合先验的拟合目标不同，不能机械统一分母。冷启动仍依赖观测、标定、预训练资产和初始身体方向，不等于不使用先验。

### 3.4 手部观测、旋转约定与初值

WiLoR的原生MANO路径使用15个局部手指旋转，忽略其全局root、translation和betas，不将WiLoR相机尺度或手部形状当作人体真值。读取器核对参数约定、父相对旋转和关节顺序；左手转换为`M R M`的双侧基变换，不能只做`M R`。候选需与对应人体腕关联，新策略要求关联腕有限、在边界内且分数0.25..1；异常身体腕不能绕到手分支继续使用。

同一相机可保留左右两只解剖手，两相机给每只手至多两份局部旋转假设。native权重`b_tvs`来自有效`detector_confidence`，缺失/非法分数不给正权重。没有有效native覆盖的手会被拒绝，具体候选选择及拒绝原因写入既有参数检查文件。

将旋转转为45维轴角后，以`(θ−μ) pinv(C) ⊙ s`编码PCA。共享模式选择所有有效帧/视图中box score最高的**完整假设**，分数相同取最先出现者；逐帧模式选择该帧最佳视图，缺失帧初值为零。这样避免对+179°/−179°主值轴角直接平均后相消；不意味着最高分候选就是真实手姿，也不消除非凸问题。

native初值直到进入D1才复制到手参数，A/B/C不受其改变；即使D1预算为0，这个阶段入口赋值仍发生，后续D2可使用该初值。所有有效假设继续参与手损失，不只拟合被选为初值的一帧。

对一只手，局部旋转项为：

\[
L_{MANO}^{s}=\frac{1}{2N}\sum_{t,v}b_{tvs}\frac{1}{15}
\sum_{k=1}^{15}\frac{\|R_{tsk}(\Theta)-\widehat R_{tvsk}\|_F^2}{4},
\quad L_{MANO}=\tfrac12(L_{MANO}^{left}+L_{MANO}^{right}).
\]

单关节`‖R−R̂‖²/4=1−cos(角差)`，是有界旋转弦距离，不是角度平方的精确测地距离。共享手仅计算一次预测旋转，然后广播给所有N×2观测；不会复制独立手参数。`b`是候选框分数，不是逐手指测量精度。

可选WiLoR二维项使用其模型投影21点，是辅助信息。每相机、每手权重为box score×边界因子×关节优先级：边界内1、外部保留点0.1，腕1，五指各沿近端到指尖为0.95/0.82/0.68/0.48；缺失或非法box score为0。这条辅助路径的边界处理不同于身体二维硬支持，不能统一描述为所有越界手点已删除。每个相机×手项除以`N×21`，**四项相加**，没有再除4；单位px²，不使用身体二维的Huber尺度。关闭`hand_2d_weight`后不计算其投影损失，不能把WiLoR投影称独立人工2D真值。

### 3.5 先验、时序项与总目标

参数正则为：

\[
L_\beta=\operatorname{mean}(\beta^2),\quad
L_z=\operatorname{mean}(z^2),\quad
L_{PCA}=\operatorname{mean}(c_L^2)+\operatorname{mean}(c_R^2).
\]

根锚使用同源初始化作为软正则。令`a_t`为该帧两肩两髋四个`body_fit_weights`的最小值：

\[
L_{root}=\frac1N\sum_t a_t\frac{\|R(r_t)-R(r_t^{init})\|_F^2}{6}.
\]

这是旋转空间弦距离；局部小角度下约为角差平方/3。sinc形式Rodrigues避免直接`acos`端点问题，但180°仍可能出现驻点。该项是同源方向保持，不是新增独立观测，不能证明根朝向已准确。

可选身体时间项对模型COCO相对髋中点的位置作二阶差分，以0.03m为Huber阈值。新策略要求对应关节连续三帧有`body_fit_support`，并要求三帧左右髋均支持；除以原始`(N−2)×17`槽数，不跨缺口。即使是二维数据目标，这一时间支持仍依赖三角化支持。该项默认关闭。

逐帧手模式的时间项对PCA系数作二阶差分，先在K维取平方均值，再对**有效三帧组**求均值，最后左右手平均；支持来自有效手像素或native假设。它目前只使用是否有效，不按连续box分数缩放，归约也不是固定全序列分母。共享手模式该项为零。身体/手部二阶差分均未除采样时间平方，不能直接当成m/s²或角加速度；换帧率时不能无条件复用同一λ。

骨长项比较当前内部骨架的10组长度与零beta模板长度，默认关闭，因为启用会把共享体型拉向模板。独立人工腕参考也是显式可选项，参考路径与正权重必须一起提供；当前448主线未启用。没有这些输入就没有相应信息来源。

每阶段总目标可写为：

\[
L=\lambda_{3D}L_{3D}+\lambda_{2D}L_{2D}
+\lambda_{MANO}L_{MANO}+\lambda_{hand2D}L_{hand2D}
+\lambda_\beta L_\beta+\lambda_z L_z+\lambda_{PCA}L_{PCA}
+\lambda_{root}L_{root}+\lambda_{timeB}L_{timeB}
+\lambda_{timeH}L_{timeH}+\lambda_{bone}L_{bone}
+\lambda_{wrist}L_{wrist}+L_{contact}.
\]

这个式子列出可用项，不代表每阶段全部激活；实际开关见下一节。各项量纲不同，λ同时承担数值尺度与偏好，不能只按λ大小比较约束强弱。改为固定分母后，旧λ尚未重新调优，不应标成统计最优配置。

### 3.6 分阶段优化：更新变量与实际开关

| 阶段 | 更新变量 | 默认lr倍率（基础0.02） | 观测与特殊项 |
|---|---|---|---|
| A `A_body_vposer` | transl、latent | 1（0.02） | 身体数据；可选身体时序、骨长、腕参考 |
| B `B_shared_beta` | beta | 0.1（0.002） | 身体数据；可选腕参考；不启用身体时序/骨长 |
| C `C_body_vposer_refine` | beta、root、transl、latent | 0.25（0.005） | 身体数据、根锚；可选身体时序、骨长、腕参考 |
| D1 `D1_hand_proximal` | lhand、rhand | 0.5（0.01） | native与/或手二维、手时序；身体冻结 |
| D2 无接触 | lhand、rhand | 0.25（0.005） | 同D1，继续手部细化 |
| D2 有接触 | lhand、rhand、root、transl、latent | 0.25（0.005） | 手项、表面接触、根锚，并保留身体数据及已启用腕/身体时序 |
| D3 有接触 | 同接触D2 | 0.1（0.002） | 继续接触细化，保留相同身体约束 |

`beta_policy=freeze_after_b`仅从C的更新列表移除beta；D阶段始终冻结beta。无接触时D3强制0步，不能让WiLoR模型投影单独推动全身。参数正则在总式中一直存在，但冻结变量对应的常数不会产生参数更新；“计算过某项”不等于“它在该阶段有效优化身体”。

每阶段新建Adam，动量不跨阶段继承。每步依次清梯度、解码、构造启用项、检查总loss有限、反传、检查活动梯度、更新、检查活动参数。没有学习率调度器、自动早停、梯度裁剪或失败后回滚重试；不能在实现说明中暗示这些机制存在。

六个预算分别来自`base/beta/joint/hand/contact/contact-refine-steps`。省略的预算各取`max(1, steps//6)`，显式0可跳过优化循环。**`steps`不是总迭代硬上限**：默认180且无接触时实际为30+30+30+30+30+0=150；要完全跳过各阶段循环，需显式设置各阶段预算为0，而不能仅设`steps=0`。当前448及历史360所用阶段预算均为30/30/300/100/100/0，与通用默认不同；其目标和策略不同，不能混用。

### 3.7 可选接触分支的具体计算和边界

启用必须同时给出`contact_labels`、`scene_transforms`、`contact_vertex_sets`、`walker_topology`，且手/足表面总λ至少一项为正。读取器检查有限值、数组形状、旋转与顶点索引等；reliability下手/足标签在[0,1]，标签全零失败。几何形状检查不等于对采集同源性或物理标签真实性的独立认证，调用者仍须遵守当前实验协议。

cam0网格变到每帧地面：`V_G=R_GC V_C+t_GC`。手掌顶点计算到扶手线段胶囊表面的有符号残差`d=distance_to_segment−0.016m`；足底顶点使用`d=z_G`。它们都使用真实模型表面，不能用COCO腕/踝替代。

对K个候选表面点，中心化softmin和pseudo-Huber为：

\[
\operatorname{csoftmin}_\tau(d)=-\tau\log\left(\frac1K\sum_i e^{-d_i/\tau}\right),\quad
\rho_\delta(x)=\delta^2\left(\sqrt{1+(x/\delta)^2}-1\right),
\]
\[
\ell(d)=\rho_\delta(\operatorname{csoftmin}_\tau(d))
+\frac1K\sum_i\max(-d_i,0)^2.
\]

当前`τ=0.005m`，手`δ=0.015m`、足`δ=0.010m`，穿透项系数1。softmin减去了点数造成的偏移，是平滑的表面距离代理；不是要求所有候选顶点同时贴在平面，也不是精确的最小距离。

手、足分别按`Σ_(t,s) label_ts × ℓ_ts/(2N)`归约，再乘各自λ。显式全片手表面先验在固定相机—助步器变换下比较对应掌顶点，距离除0.03m后作Huber，顶点均值乘该帧手标签，再按帧和左右手平均；零标签帧不再接受满强度先验。

接触D2/D3释放身体变量，所以新策略同时保留原来选择的身体数据项，避免全身只为满足接触而漂移。接触标签、扶手尺寸和地面仍是外部输入/假设；共享手PCA不证明固定抓握，只有踝观测也不证明足底接触。未提供独立物理信息时，输出的`physical_evidence`明确保留未建立状态，不从拟合残差自动生成“真实接触”。

### 3.8 减少冗余计算的实际路径

当阶段≥D1、训练变量恰为双手PCA、`hand_2d_weight=0`且该阶段无接触目标时，native局部旋转、PCA先验和手时序已经能完整计算活动梯度。此时采用`local_hand`路径：普通优化步不调用全身SMPL-H，日志步才在`no_grad`下重算完整模型诊断。若手二维或接触依赖空间位置，则仍走完整网格路径，不近似替代蒙皮。

日志按全局步`step%10==0`或最后一个优化步记录。共享手的15组预测旋转只转换一次，观测仍保留全部帧/视图；固定手映射、置信度和关节优先级移到循环外。初始化的首个零状态模板只前向1帧；带各帧root的第二次模板前向仍为N帧。最终输出也始终用最终参数重新解码完整网格。

手指蒙皮可能改变表面回归的COCO腕，所以“身体参数冻结”不能推出所有COCO点完全不变；日志与最终输出重新计算这些点。`stage_schedule.model_evaluations`只计阶段循环中的模型调用，不包含初始化、最终解码或外部observer，不能直接当作整次运行总调用数/性能测试。

### 3.9 默认参数、历史复现和新方法的调用边界

| 参数 | 通用CLI当前默认 | 使用时需区分 |
|---|---|---|
| fit_policy | reliability | 历史未带策略的内部Namespace可能按legacy兼容；直接调用者应显式设置 |
| body_weighting | quality | 新策略在此名称下使用第3.1节公式；reference/uniform/no_*仅允许legacy |
| body_3d_weight / body_reprojection_weight | 1 / 0 | 新策略恰有一项为正 |
| body_observation_mode | legacy | 是普通模式的旧名称，不决定fit_policy |
| body_reprojection_scale_px / ankle倍率 | 100 / 1 | 踝倍率≠1须用reprojection_only |
| beta_policy / beta_prior_weight | joint / 0.001 | 始终共享beta；C是否继续优化可显式选择 |
| vposer_prior_weight / root_anchor_weight | 0.02 / 0.01 | 不是已重新调优的最优值 |
| hand_pca_comps / hand_pca_prior_weight | 12 / 0.001 | profile标签pca12/pca24/full45需与维度一致 |
| mano_pose_weight / mano_pose_init | 0 / 关闭 | 通用默认并没有自动启用native手姿 |
| shared_hand_pose | 关闭 | 开启通常还必须native初始化且native权重>0；先验消融需显式许可开关 |
| hand_2d_weight / hand_temporal_weight | 1e−7 / 0.02 | 共享模式手时序恒为0；native-only需显式把手二维设0 |
| body_temporal / bone / wrist / contact权重 | 全部0 | 没有默认接触、人工腕或身体平滑 |
| lr / device / max_init_body_rms_mm | 0.02 / cuda / 300 | CUDA不可用就失败，不自动转CPU |

`body_reference_confidence_floor`解析默认仍为0.6，只供legacy reference使用。新策略实际floor恒为0，并拒绝修改这个历史专用参数；不能因帮助里看见0.6就断言新策略仍保留0.6下限，也不要用`--body-reference-confidence-floor 0`试图开启新策略。

从仓库根查看实际入口参数（只显示帮助，不运行拟合）：

```powershell
& ./.venv-cuda/Scripts/python.exe -B realtime_app/run_tool.py body.fit_smplh_wilor_sequence --help
```

实际运行须补齐第2.2节的输入和独立空输出目录。重放当前448时由`configs/current_mainline.json/mainline_selection.body.reproduction_arguments`定位参数数组；历史360 reference25使用其原实验`runs/reference25/fit_args.json`。不能把新CLI默认偷偷套到旧结果上：

| 目的 | 应明确设置的策略/变化 |
|---|---|
| 重放已保存reference25 | 原完整参数数组+`--fit-policy legacy`，仅换新输出；见`realtime_app/README.md`现有示例 |
| 在同样输入/预算上使用新3D方法 | 明确reliability、body-weighting=quality、body-3d-weight>0、body-reprojection-weight=0；移除旧reference消融选项；原有native/shared手设置需明确保留 |
| 使用新二维身体目标 | reliability、body-observation-mode=reprojection_only、body-3d-weight=0、body-reprojection-weight>0；明确像素尺度；不允许人工3D腕或接触目标 |
| 仅使用native手部信息 | mano-pose-init、mano-pose-weight>0、hand-2d-weight=0；是否shared是独立建模选择 |

更换策略并不自动获得同等λ尺度，也不自动成为新主线。通用入口没有自动为实验固定所有随机种子；实验包装器与所保存参数/协议的设置需另外保留，不宣称跨设备逐位复现。本节说明实现与调用契约；当前已完成的448结果见第1节，不从源码默认推断实验结果。

### 3.10 输出、失败处理与阅读结果的方法

| 文件/字段 | 记录内容 |
|---|---|
| `cold_initialization.npz` | 优化前beta、root、transl、VPoser latent、pair_id；不是完整优化器检查点 |
| `initialization_audit.json` | 初始化来源、门值、加权及原无权RMS等；不能只读通过/失败而忽略支持条件 |
| `wilor_association_audit.json` | 手候选关联与拒绝信息 |
| `wilor_mano_parameter_audit.json` | 启用native时的资产、参数约定、覆盖与编码情况 |
| `wilor_hand_geometry_audit.jsonl`及summary | 启用相关诊断时的跨视图手几何信息；不作为手3D拟合目标；禁用时看summary状态 |
| `wrist_reference_audit.json` | 仅在启用外部腕参考时记录来源与使用范围 |
| `result.npz` | 最终参数、真实网格faces/vertices、模型COCO/内部关节/手21点、原始观测、权重、有效状态和可选接触诊断 |
| `fit_summary.json` | 策略、输入资产路径、参数语义、阶段预算/变量/lr、历史loss、初始化门、证据边界 |

`result.npz`中的`vertices/predicted_coco/transl`为cam0米制；`raw_triangulated_points`保留毫米，不能直接与前者相减。输出`left/right_hand_pca`展开为N行，即使真实训练参数只有1×K；需要结合`shared_hand_pose`与`hand_pose_parameter_frames`判断自由度。`body_left/right_confidence`、二维valid/reasons、`body_accepted`、`body_fit_support`、`body_fit_weights`和新增权重分解同时保留，不能只读最终网格丢弃失败状态。

history的loss/诊断来自该步**更新前**前向，虽然日志在`optim.step()`后写出；`result.npz`是最后更新后重新解码。因此最后一行history不等于最终网格误差的重新计算。`weighted_terms`只记录若干选定项，未覆盖手二维、全部时序、腕和接触等；不能把其求和当作完整总目标。`reprojection_only`的局部梯度采样实际覆盖阶段索引0..2（A/B/C）中存在的local_step 0/30，元数据中的旧`A_C`文字不精确；这也不是完整目标的梯度分解。

拟合拒绝非空输出目录；预算/权重/尺度先校验，但建目录后仍可能在资产、帧、初始化或优化阶段失败，所以失败目录可能已有输入检查记录，不能把“目录存在”当作完成。总loss、活动梯度、更新后参数和最终张量有有限值检查；失败立即抛出，不自动换旧网格、补帧或隐藏拒绝点。日志不包含可恢复Adam状态，也没有自动断点续跑。

当前有效机制包括固定分母归约、身体3D/2D互斥、SO(3)根锚、接触释放身体时保留身体数据、完整共享手初值、逐关节时间支持及local_hand计算。当前未完成的是检测分数/协方差校准、完整三角化各向异性误差传播、新权重的独立调优及真实三维/接触验证。当前448已经实际运行reliability，不能再写成“新策略尚未完整拟合”。

<a id="causal-smplh"></a>

## 4. 完整SMPL-H因果二维运动基线

拟合对象和最终输出均为完整SMPL-H；回归骨架只作为二维观测接口。跟踪器不运行三角化、不读取逐帧三维观测、不添加三维位置目标，也不使用Stage、地面、接触或物理仿真。每次模型前向仍生成三维网格。当前结果的身份与0.4/1.2区别见第1、5节。

### 4.1 输入、来源与首帧人工标注

- 左/右二维输入：`research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/{left,right}/pmpose/raw_predictions.json`。
- 图像与时间：同一people_1下的`input_448pairs`，pair0..447。读取采集记录的左右host-return时间中点，减首帧形成秒；完整448帧对应16.035239s，dt范围0.0069816..0.0695054s。返回时差median3.5254/P955.97451/max9.2775ms不能证明曝光同步。
- 标定：`realtime_app/calibration/results`；cam0=LEFT、cam1=RIGHT，原始图1920×1080。右目点用完整`R_cam0_to_cam1`和`T_cam0_to_cam1_mm/1000`变换，内部统一米，投影残差为像素。
- 模型资产：`third_party/WiLoR/mano_data/models/SMPLH_male.pkl`及同目录MANO_LEFT/RIGHT；`models/smpl/J_regressor_coco.npy`；`models/VPoser02_05/V02_05`。MANO只补旧SMPL-H资产缺失的PCA/mean字段；最终用SMPLHLayer真实矩阵旋转与面，VPoser和模型权重冻结。
- 原0.4/1.2对照起点为之前当前方法的**frame0-only**模型参数。其历史冷启动用过三角化；读取字段仅transl/global_orient/body_pose/betas/VPoser/手姿，未读保存的三角化点或predicted_coco作为测量。这两个旧对照不使用整段beta或未来帧初值；下述用户指定校准体型版本例外只读取全片共享beta。
- 人工标注：`research_records/annotation_handoffs/H20261014_people1_pair0000/labels/{left,right}_pair_0000.json`，左右各8点，共16点。用户明确确认hand=腕关节、foot=踝关节；左图`leftt_hand`作为显式别名映射到左腕，不作模糊猜名。

标注映射：左右肩COCO5/6，左右腕9/10，左右膝13/14，左右踝15/16。肘7/8、髋11/12仍用首帧自动点；首帧以后全部回到原始自动检测，不传播人工真值。

先验证标注JSON声明1080×1920，`imagePath`所指图与选定pair0000转正图像素相同，避免标错来源。左图原先逆时针转90度，标注(u,y)恢复raw=(1919-y,u)；右图原先顺时针转90度，恢复raw=(y,1079-u)。越界、非有限、重复解剖点和不支持的标签直接报错。OpenCV读图用imdecode+fromfile，解决Windows中文路径读取失败。人工score=1只表示纳入，不表示校准概率。

首帧另做一次纯二维SMPL-H细化：Adam500步，latent学习率0.02、根轴角/平移0.005、beta0.01；人工/自动点sigma分别4/15px，二维点向量范数采用3sigma Huber，VPoser先验sigma1.5、beta先验sigma2。优化根、姿态、10维beta，手/足/颈/头局部旋转固定；细化结束后beta冻结，初速/初加速度均设0。没有三维数据项。初值结果位于`run_manual_v1/first_frame_fit/result.npz`，原始500步损失与16点残差见同目录summary。

0.4与1.2对照均复用这一个已细化的首帧，`--first-frame-refine-steps 0`避免重复优化。frame0只初始化，不再次当作EKF新测量。人工16点首帧残差median10.38450/max61.64695px；最大的左目右肩残差没有被删除或改动标注。

**2026-10-15用户指定校准体型。** `run_calibrated_beta12`显式传`--shape-fit G20261013_reliability448_view_v1/fit/result.npz --first-frame-shape-policy fixed`，只读取共享1×10beta，禁止读取未来姿态/网格。以原始frame0-only种子和相同人工二维重新500步优化q，beta不参与Adam，再运行原1.2跟踪器。beta为[1.1298097,-0.3122407,0.6401215,-0.0228373,0.2672804,-0.2343161,-0.1101328,-0.0018150,-0.2465878,0.2786286]；输出与主线beta逐值相同，期间冻结。它明确使用全片预校准体型，不能混称首帧-only体型或与旧对照同信息。旧`refine`默认及结果保留；首帧16点仍是拟合输入而非held-out。

### 4.2 完整模型、观测接口与状态定义

实现文件：

- `realtime_app/pose_app/smplh_motion_tracker.py`：MotionPose、真实SMPLHObservationMap、状态预测/二维反馈/导数与失效处理。
- `realtime_app/pose_app/smplh_first_frame_labels.py`：显式标注语义、逆旋转、首帧二维细化。
- `realtime_app/tools/run_smplh_motion_baseline.py`：有序回放、全模型NPZ、逐帧JSONL、统计与曲线；复用已有PVA工具的原始二维读取、时间读取和画图函数。
- 两个对应测试文件：`realtime_app/tests/test_smplh_motion_tracker.py`和`test_smplh_first_frame_labels.py`。

名义位姿是q=(t,R,z)：t为3维米制根平移，R为SO(3)根旋转，z为32维VPoser。局部误差维度38，速度和加速度各38，联合协方差114×114；不是12个自由点各自运动。R存3×3矩阵，局部误差/更新用右乘旋转向量，避免根轴角跨π造成假跳变。

VPoser解码21个身体局部旋转后，冻结排除root的索引9/10（双足）、11/14（颈/头），其余身体旋转由同一latent控制；双手各45维姿态和10维beta冻结。身体全局姿态、手部完整网格仍随根和身体链运动，腕点可观测；没有从hand标签去优化手指。

模型侧点严格为`J_COCO @ vertices`，得到COCO17；跟踪使用5..16的肩、肘、腕、髋、膝、踝12点。每次评估均由同一组完整模型参数生成网格和回归点。原生52关节用于内部骨长检查；COCO是表面回归点，其距离受pose blend影响，不要求像原生运动学骨段一样绝对刚性，不补一个自由骨架或逐帧骨长项。

### 4.3 预测、过程噪声及二维反馈的实际顺序

帧k开始先用真实dt和上一帧后验预测：

\[
t^-_{k+1}=t_k+v_tdt+\tfrac12a_tdt^2,\quad
R^-_{k+1}=R_k\operatorname{Exp}(\omega dt+\tfrac12\alpha dt^2),\quad
z^-_{k+1}=z_k+v_zdt+\tfrac12a_zdt^2,
\qquad v^-=v+adt.
\]

加速度不是全片硬常量：每个预测步加入白jerk过程噪声，单坐标[p,v,a]的Q块是

\[
q_j\begin{bmatrix}
dt^5/20&dt^4/8&dt^3/6\\
dt^4/8&dt^3/3&dt^2/2\\
dt^3/6&dt^2/2&dt
\end{bmatrix}.
\]

平移、旋转、latent的谱密度分别0.25m²/s⁵、1rad²/s⁵、4latent²/s⁵。F包含根旋转误差的Exp(-drive)和SO(3)右Jacobian传播；其余为常加速度块。初始标准差按(t,root,z)分组：位姿(0.10m,0.35rad,0.5)，速度(0.5m/s,1rad/s,1.5/s)，加速度(1m/s²,2rad/s²,3/s²)。这些都是工程设定，不是数据校准的分布。

随后执行：

1. 检查当帧各视图有限值、原始边界及score∈[0.2,1]。分数只作门，不直接换算逆方差。两眼均有效的同名点再作二维极线检查：鱼眼反畸变到归一化射线，以E=[T]×R计算Sampson一阶raw像素距离，超过35px拒绝这对点；不产生三维点。单目有效时可参与更新，但不被解释为独立米制深度。
2. 至少4个不同关节有有效像素才更新；模型12点在两相机中都需正深度。投影模型用原始鱼眼函数，不把转正坐标直接送入标定。
3. 在当前预测模型上，以各参数±0.001作中心差分，77个模型批量评估产生像素和模型点Jacobian。观测H只依赖位姿块，v/a通过协方差交叉块被同一二维残差纠正，不通过差分二维再造一个三维加速度测量。
4. 每个二维点残差向量采用25px Huber，基础sigma8px；大残差提高该点的等效噪声。把latent零均值sigma1.5作为姿态正则一起处理；它不是三维观测，也不能用于准确性证明。
5. 计算一次EKF增益与联合状态修正。平移/root/latent每帧修正范数上限分别0.08m/0.12rad/默认0.40（1.2候选显式覆盖latent上限）；取共同缩放s，并检查s、s/2、s/4的真实非线性投影。候选须保持正深度、|latent分量|≤6，且“Huber像素+状态位移先验+latent先验”总目标不增。此检查保证总目标，**不保证每个像素误差下降或达到25px**。
6. 接受后，q、v、a都用同一s修正；协方差用同一个缩放增益K'=sK做Joseph更新：

\[
P^+=(I-K'H)P^-(I-K'H)^T+K'R_{\rm eff}K'^T.
\]

再执行SO(3)误差坐标重置、对称化，并在已接受模型重算点Jacobian。禁止“均值只走小步，协方差却按完整增益收缩”。没有内循环把同一批像素重复当新证据。拒绝更新则只保留预测及原因。

滤波状态递归保留历史信息，当前更新没有读取未来帧；此版未实现历史像素短窗；未来0.3s候选见第6节。加速度是被估计的运动状态，不是已识别的关节力矩、驱动力或接触力。

### 4.4 失观、恢复和导数的保守可用性

整体连续0.10s没有成功更新就失效；短失观可预测。恢复时从最近接受的模型位姿开始，只用当前二维点重新校正，v/a清零、协方差恢复初值，segment加1；不从三角化或旧逐帧拟合补洞。数值可解码、正深度只表示`pose_available`，不表示模型贴合图像。

每个关节点还需要自己的像素支持：缺失0.10s以后虽可被运动学链推断位置，其导数不再发表。新segment的前0.15s留空；位置协方差经当前点Jacobian传播，最大主轴标准差需≤0.10m；当前所有有限/界内/达score门的原始像素最坏回投需≤25px，最近一次被实际使用且拟合良好的本点观测也不得早于0.10s。当前被拒绝的像素仍参与拟合质量诊断，不能用空used mask把坏投影变成“良好”。

失观预测会刷新当前模型的点Jacobian，避免把上一姿态的Jacobian冒充当前位置不确定性。输出保留`updated/predicted/inferred/initialized/reinitialized/expired`，以及`derivative_warmup/poor_or_stale_pixel_fit/no_recent_own_pixels/position_uncertainty/model_unavailable/available`导数原因。字符串采用object数组，修正了早版把reinitialized截成reinitial的实际问题。

速度/加速度从同一模型的瞬时参数轨迹求得。eps=0.01s，在q上分别施加±v eps+0.5a eps²、由完整SMPL-H回归点中心差分得到v与a；包含旋转及非线性模型映射的曲率项，不把a简单写成J a_q。测试包含恒角速度的向心加速度。它们是当前后验运动模型的瞬时导数，不是把所有观测修正跳变后的网格序列再做位置差分。

不满足导数条件时，公开曲线与`joint_velocity_mm_s/joint_acceleration_mm_s2`为NaN；原始模型导数另存`raw_joint_*`及JSONL的raw字段，用来查尖峰，不静默删除。不生成插值、后处理平滑或伪连续段。模型失效时公开网格、模型参数和点为NaN/JSON null；JSONL内部名义参数可保留调试，但必须看pose_available标志。

### 4.5 最终输出格式与实际结果

0.4与1.2均保存同一输出契约，`result.npz`主要字段：

| 字段 | 形状/单位与含义 |
|---|---|
| vertices / faces | 448×6890×3米 / 13776×3真实拓扑 |
| transl / global_orient / root_rotation | 448×3米 / 448×3轴角弧度 / 448×3×3 |
| body_pose / body_rotations | 448×63轴角弧度 / 448×21×3×3，含冻结关节的实际旋转 |
| betas / left_hand_pose / right_hand_pose | 1×10 / 各1×45弧度，整段冻结 |
| vposer_latent | 448×32，供跟踪状态复现；也可直接用body_pose重建网格 |
| predicted_coco / smplh_joints | 448×17×3米 / 448×52×3米 |
| generalized_velocity / generalized_acceleration | 各448×38；平移、根角速率、latent分组单位不同 |
| joint_velocity_mm_s / joint_acceleration_mm_s2 | 各448×12×3，导数条件失败处NaN |
| raw_joint_velocity_mm_s / raw_joint_acceleration_mm_s2 | 同形状，保存尚未通过导数条件的模型估计 |
| pair_id / timestamp_s / pose_available / derivative_valid / segment_id | 完整帧、真实dt来源、数值状态、12点导数门、重启段 |

`trajectory.jsonl`保留448条有序记录：当前全部点、所有raw回投误差、used/views拒绝原因、导数原因及原始/发表导数、q/v/a、协方差对角、缩放增益、返回时差、世界不可用标志。summary分别统计全部原始有效观测与实际使用观测；极线拒绝点仍进入全部raw误差统计，不靠筛选改善结论。

两版都有12面板XYZ速度/加速度曲线，黑色不同线型、灰预测带，断段/不可用留空，不裁剪尖峰。`full_mesh_stereo_overlay.jpg`使用本次vertices/faces及同源双目图，展示frame0/224/412/447完整模型；第0帧采用人工点。它们没有混入Stage/地面或旧网格。

<a id="tracking-results"></a>

## 5. 全448结果、单变量对照与剩余问题

结果根：`research_records/engineering_validation/G20261014_smplh_motion448_v1/`。原始参考为`run_final_v3`，当前下一轮候选为`run_latent_cap12`；参数与逐帧状态分别见各自`run_parameters.json`、`trajectory.jsonl`，统计见`summary.json`。

### 5.1 从完整448帧定位限制，不能只看两个静态样本

先读取`run_final_v3/result.npz`和448条JSONL，按实际dt用上一后验q/v/a复算当前预测，再计算“当前后验相对预测”的平移、SO(3)根旋转、latent修正范数，分别除以0.08m/0.12rad/0.4。不重跑估计器。

443个正常updated帧中，443次最大归一化修正都来自latent，423次恰好达到1，即latent修正0.4，比例95.4853%。根旋转最大归一化修正0.370878（约0.04451rad）、平移0.227520（约0.01820m）；这两项远未达到各自上限，却被共同s缩小。所有accepted更新含一次重启的s中位数0.23312。可见这里的“保守”严重抑制观测纠正，不是物理运动有证据地需要如此小步。

0.4是VPoser坐标的数值限幅，没有直接的米、角度或关节驱动含义；每帧固定限幅也不随真实dt变化。423次饱和不能简单归咎检测器或人体模型表示能力。下述静态二维诊断已提示时序限制，本全片统计进一步给出可检验原因。

### 5.2 只改0.4→1.2，其他输入和配置固定的真实对照

只把latent修正上限改成1.2，调用同一个回放工具，输出`run_latent_cap12`，完整日志为同根`run_latent_cap12.log`。两次run_parameters逐字段比较，configuration唯一差异就是此项；左右原始二维、16点首帧标签、已细化首帧、beta/手姿/faces、标定、时间全部一致。状态与Joseph仍共用sK；没有放宽25px导数门或35px二维极线门。1.2是一次有理由的数值探针，不是经过大规模搜索证明的最优值。

复现时沿用交接说明中的输入和`--first-frame-refine-steps 0`，额外指定`--max-latent-correction-norm 1.2`，并把`--output`改成新的空目录；已完成的run_latent_cap12不覆盖重跑。其真实输入/配置见同目录run_parameters。

| 指标 | 原限幅0.4，run_final_v3 | 对照限幅1.2，run_latent_cap12 |
|---|---:|---:|
| 数值可解码帧 | 448 | 448 |
| 全部10752二维槽median | 36.23725px | 18.04503px |
| 全部二维槽P95 / max | 135.28643 / 486.76369px | 61.76466 / 143.50354px |
| 全部raw≤25px比例 | 34.18899% | 66.88058% |
| 满足导数条件的关节槽 | 1065/5376，19.81027% | 2846/5376，52.93899% |
| s中位数 | 0.23312 | 1.0 |
| 单步耗时median/P95 | 71.97225/99.61367ms | 73.39020/98.36702ms |
| 重启帧 | 76 | 406 |
| 原始a模长P95/max，全部5376 | 5.15842/43.51731m/s² | 4.70508/24.44220m/s² |
| 发表a模长P95/max，各自有效集合 | 3.83907/32.26396m/s² | 5.47644/16.46524m/s² |

整体median下降约50.20%，P95下降约54.34%。这证明该限幅设置及其共同融合强度显著限制了本片的投影贴合，**不证明三维或导数准确性**。对照数值状态为initialized1、updated444、predicted2、reinitialized1；导数warmup120、poor/stale fit2358、无近期本点52、available2846，所有状态保留。

不能把两个“发表a P95”直接当同一批点比较：1.2版额外纳入很多原本不贴合的区间。两版共同通过导数条件的892槽，a模长median/P95/max为：

- 0.4：1.02614 / 3.96564 / 32.26396m/s²。
- 1.2：1.05962 / 4.02471 / 13.75064m/s²。

共同集合的P95基本没有改善。更小的max与更好的二维误差都不能取代已知运动真值。

完整时间轴按连续64帧分组，七组都改善：

| 帧范围 | median0.4→1.2，px | 导数比例0.4→1.2 |
|---|---|---|
| 0..63 | 21.56→18.91 | 41.15%→49.74% |
| 64..127 | 26.97→19.38 | 33.07%→49.61% |
| 128..191 | 48.75→19.71 | 9.64%→47.92% |
| 192..255 | 38.86→15.94 | 13.93%→59.51% |
| 256..319 | 27.15→16.32 | 24.48%→58.46% |
| 320..383 | 36.13→16.79 | 13.93%→59.51% |
| 384..447 | 73.70→18.86 | 2.47%→45.83% |

图`run_latent_cap12/step_bound_comparison.png`直接画全部帧median和12点导数输出比例，白底黑色实/虚线，无平滑或插值。新版本两张XYZ运动曲线及同源完整mesh双目叠图均已生成并实看；旧0.4结果不覆盖。

### 5.3 剩余问题分布：髋、膝与整体表面

1.2版逐关节结果，误差统计包含每个关节全部896个左右视图槽：

| 点 | raw median/P95，px | 导数输出帧/448 |
|---|---:|---:|
| 左肩 / 右肩 | 15.08/35.14；16.38/31.65 | 278；286 |
| 左肘 / 右肘 | 16.57/43.87；11.96/29.32 | 283；362 |
| 左腕 / 右腕 | 13.05/35.77；10.56/30.65 | 325；359 |
| 左髋 / 右髋 | 36.22/72.71；41.48/80.98 | 28；22 |
| 左膝 / 右膝 | 25.04/69.14；31.13/92.25 | 169；96 |
| 左踝 / 右踝 | 14.21/40.78；16.17/43.83 | 332；306 |

改善主要来自手臂和踝，髋并未解决。骨长内部刚性、全网格输出、手臂贴合不能推导出髋/膝或完整人体运动已恢复。当前人工16点没有髋/肘，髋点定义、遮挡与回归器语义缺少独立核对；体型从单帧骨架估计也缺少独立尺寸/轮廓证据。尚不能区分其中哪一项主导，不自动添加点偏置或删髋点来提升总体数值。

还有可辨识性上限：两眼12点虽有48个像素残差，但投影经过12×3维模型点，位姿观测Jacobian的秩最多36，当前位姿局部变量38维；不能把48当作48个独立姿态约束。轴向旋转与表面厚度还可能更弱。历史运动与先验能提供额外限制，不能保证每种姿态都可唯一辨识；首帧同时优化beta使问题更欠定。因此“完整SMPL-H能生成”与“完整姿态/表面已由图像识别”必须区分。这是本模型接口的维数分析，不是独立人体精度测量。

还存在首帧来源切换：人工16点与该帧原自动点的距离mean28.98083/median28.46762/max68.63545px，下一帧又使用自动点。这样可能产生初始暂态，不能把所有变化当真实人体运动；0.15s导数warmup只避免立即发表，不能消除模型/来源偏差。人工首帧已经参与拟合，故其残差**不是held-out验证**。后续需要未参与拟合的少量多帧标签/遮挡状态，不能继续用这16点自证准确。

host-return中点依旧只是当前工程时基。真实采集回读抖动与双目曝光不同步，会被速度/加速度放大；合成中用此dt证明实现正确，也不能证明真实物理采样时间正确。

### 5.4 已有诊断与尚不能归因的误差

在0.4轨迹同一冻结模型上，仅移除时序先验/步长上限作当前帧静态二维Adam200步，frame224的median/P95为36.37/88.66→10.83/36.70px，最差frame412为151.01/395.04→36.66/122.65px。诊断未插入轨迹；记录为`run_final_v2/static_fit_diagnostic.json`，v2/v3模型状态相同。它提示求解限制，也表明残余冲突不能全部归因时序。

首帧人工同名点raw Sampson距离中，右肩约30.17px；细化后左目右肩残差61.65px，右目约9.98px。需要分别核对点定义、遮挡、标定与同步；不能以人体拟合“校准”相机，也不能修改人工点去消除冲突。manual sigma4px只是工程假设。

0.4轨迹frame292右踝虽两眼误差23.55/6.90px，模型加速度仍达32.26m/s²。较好的像素贴合和通过质量门都不能证明导数。单帧beta冻结避免体型逐帧漂移，但无法消除初始尺寸/表面偏差。

Jacobian目前需要77个姿态批评估及当前模型重线性化，是已知成本；单步约73ms中位数不包含检测、初始化、输出。自动微分/解析Jacobian替换尚未实施，应在拟合和导数定义通过后单独验证，不能先以GPU调用或批量吞吐宣称实时。

### 5.5 2026-10-15观测核对：髋与右膝不是同一种残差

本轮只读取1.2候选保存的模型点、轨迹、原二维与标定，不重新拟合或改变任何门。统计frame1..447的447帧；frame0使用人工替换点且参与初始化，不混入自动点诊断。全部原448帧及失败状态仍保留。保存COCO点经现有鱼眼投影得到的误差与这447帧日志逐项相同。

COCO11/12/14确为左髋/右髋/右膝，当前使用posed vertices经17×6890回归器得到观测点，没有用native关节替代。本地回归器与Pose2Mesh作者发布数组逐值相同；[作者COCO实现](https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/master/data/COCO/dataset.py)先回归17点，再另加双髋均值Pelvis。11/12不是SMPL-H原生1/2关节的同一算子，因此也不能直接称为native髋中心。来源相符只确认映射契约，不保证其与本片遮挡下的检测点无偏差。

| 点 | raw Sampson median/P95，px | 超35px帧/447 | 左目/右目模型回投median，px |
|---|---:|---:|---:|
| 左髋 | 9.18/22.64 | 2 | 36.52/35.49 |
| 右髋 | 8.89/24.13 | 2 | 37.12/43.83 |
| 右膝 | 21.82/46.30 | 80 | 21.82/41.33 |

Sampson使用同一标定的E=[T]×R及逆鱼眼局部像素Jacobian，扰动±0.25px；只检验双目同名点与当前标定的极线相容性，不三角化，也不能分辨检测偏差、曝光不同步或标定偏差。右膝占全部111次同名点极线拒绝中的80次；其左右最低score仍为0.687/0.717，高score没有排除冲突。原JSON另有visibility/presence，读取器只用keypoint score；右膝visibility两目median0.954/0.949也只是同一检测器输出，不能替独立遮挡标签，不据此增加新的门。右目右膝的模型点相对检测点在全部447帧都位于转正图下方，纵向偏差median38.39px。Sampson≤5px的13帧两目回投median7.55/9.08px，但这是小型条件子集，不是独立验证，不能推导因果归因。

髋点即使Sampson≤5px仍不贴合：左髋111帧两目median35.50/29.37px，右髋136帧41.21/49.24px。模型双髋投影间距/检测双髋间距的逐帧比值median为左0.7430、右0.7420；膝为0.9692/0.9877、踝1.0245/1.0088。两眼一致的髋间距偏小是本候选的投影现象，不能据此断言真实骨盆窄、回归器错误或必须重估beta。

固定抽取七个64帧区间中点32/96/160/224/288/352/416，保存同源转正图上的自动点圆圈、模型投影叉号及COCO编号于`run_latent_cap12/observation_diagnostic_7frames.png`并实看；衣物覆盖髋膝，不能把目测或本图变成独立解剖标签。数值与方法记录在实验根`protocol.json/observation_diagnosis_20261015`。

阶段决定：完成这一轮观测诊断，不改模型、删点、补偏置、松门或晋升1.2；短窗仍未获时序真值证据。真实归因需要未参与拟合的多帧髋膝点及可见/遮挡状态，遮挡点只能标不可确认，不能要求人工猜测。已知运动仍可另行检验估计器，但同映射合成主动消除了真实点定义/体型错配，不能借其成功宣布本片髋问题解决。

### 5.6 固定主线beta与停止恢复修复后的结果

用户看到的体型变化确来自首帧二维联合优化：旧beta为[3.1383,-1.7428,2.3555,0.8654,0.0179,0.5304,1.6632,-2.8012,-3.4949,2.5916]，不是显示时读错模型。按用户指定固定第4.1节全片主线beta后，当前candidate保存`run_calibrated_beta12`；全448可解码，updated441/predicted4/reinitialized2（111/263）/initialized1。全10752 raw median19.795/P95110.569/max216.988px，发布导数2229/5376；均不能写成比原1.2更优。初始化策略/体型信息改变，旧0.4/1.2单变量结论仍只针对原共享初值。

可视化使用本轮严格点与新重放Stage场景。停止恢复在真实非均匀时间下漏掉368..376事件：只从最后运动时刻计算歧义间隔，忽略373的新停止证据。修复后按最后运动或停止证据保持0.3s间隔界限，歧义不刷新证据/不累计停止时长；确认、最大修正、strict和已发布输出规则不变。事件增加379，380/391的助步器四脚从最高24.77mm恢复为0。Stage1只余217确认前单帧最高18.19mm，未把短暂动作标签直接当支撑事实。

同一修复场景的左右脚底候选height有符号median从56.524降至13.939mm，但新absolute median33.778/P9587.774mm，301/896脚时刻绝对偏离>50mm；frame188右脚+131.287mm、335左脚−117.604mm。脚部固定局部姿态与旧主线相同，仍随小腿朝向变化；335左踝16.72/8.44px回投不能保证脚底高度。该高度来自Stage显示参考，不是独立接触真值；没有将脚或人体向地面吸附，也没有通过旋转v/a假造世界导数。

旧离线主线的reliability、踝二维倍率2、共享体型与弱先验可供后续求解参考，但其使用全片信息且surface-foot-contact=0，不能直接把它当当前因果精度真值。残余下肢姿态/点深度/足底朝向未解决，需遵循已知运动与world阶段门；本轮不盲目扫权重、放门或增加地面/heel观测。当前页面为`run_calibrated_beta12/view/body.html`及浏览器版，beta与左右脚底/助步器四脚height明确显示，原raw/NaN、拒绝与segment均保留。

<a id="next-plan"></a>

### 5.7 用户指定10mm整足约束的Stage条件世界估计

本轮按用户新授权实现工程假设对照，未将独立视觉轨迹/已知运动研究门改写成已通过。入口仍为run_smplh_motion_baseline.py，显式`--world-scene`和`--ground-constraint-policy contact`启用；未提供时保留原相机系基线。新代码为pose_app/smplh_world_motion_tracker.py与smplh_ground_constraints.py。体型读取原主线共享beta后冻结，复用已按该beta细化的frame0-only初值；复用时逐值核对beta，避免再次500步细化。当前原始17点、12拟合点、鱼眼/极线门及真实时基保持，没有三维人体观测。

**坐标与状态。** 内部直接在固定地面系估计q/v/a，每帧使用当前R_GC、t_GC将模型点逆变换为p_C=R_GC^T(p_G−t_GC)，再原双目投影。由于SMPL-H绕shaped rest pelvis J0旋转，根参数必须按t_G=R_GC(t_C+J0)+t_GC−J0、R_root,G=R_GC R_root,C转换；直接R t_C+t会使网格错位。保存result的vertices/transl/root仍是当前相机系供旧接口再生，同时另存world_vertices/world_transl/world_root_rotation；generalized及关节v/a明确为Stage条件地面系。不是旋转旧相机速度得到世界导数。

**同帧约束MAP。** 先由原q/v/a与真实dt预测，使用该预测联合协方差的Pqq作为姿态先验。一次同帧目标为原双目Huber像素项、VPoser弱先验和dq^T Pqq^−1 dq/2；SLSQP迭代只求同一目标，不逐次重复收缩P。得到dq后，联合均值通过P_xq Pqq^−1 dq同步纠正v/a。收敛时才在最终点构造一次局部Laplace/Joseph近似，并做SO(3)误差坐标transport；active几何约束的1mm近似尺度及所有R/Q均为工程假设，不是校准置信概率。MAP不沿用旧EKF的latent1.2共同gain限幅，仍有|latent_i|≤6边界；因此world-none EKF与contact MAP不是纯足底项单变量消融。

**几何范围。** 每侧54个真实候选sole点，取各侧min z定义近地面；heel顶点3387/6787仅做相对运动锚，不把heel的z置0，因为它们不在sole集合。所有帧候选sole不得低于−5mm；全部115个operational Stage2帧双sole min∈[−5,5]mm，首次可行姿态建立两侧heel及全部sole点的三维episode锚，之后每点距其对应锚≤10mm。短transition保留锚供Stage2恢复，不重新设锚；transition本身仍按非Stage2近地面条件，不伪称移动/静止已知。明确Stage1后解锚。非Stage2另采用至少一侧sole接近地面的本段行走假设，允许另一脚摆动；这是新增假设，不是原主线contact classifier的真值。solver使用向内0.2mm余量，实际输出按物理5/10mm界检查，只有1微米数值容差，不将10.2mm算通过。

**数值求解与失败。** 逐点z下界及逐点平方距离球避免min/max的活动顶点切换造成有限差分不稳定；每帧固定预测模型的候选支撑点来实现上界。新双脚约束尚不可行时，只用几何/预测先验寻找warm start，不消耗像素或收缩P。之后保留同一帧目标下真正改善且满足全部物理界的可行iterate；若最终迭代不可行/达预算，仅可采用比该帧可行初值更低目标的incumbent，标记feasible_unconverged。optimizer最后cost、slack、foot geometry和失败原因全部另存，不能改称收敛；该状态仅更新联合均值并transport协方差，不收缩像素/active/导数先验信息，不发表v/a。没有可行改善仍constraint_conflict，预测/失效及原始导数保留，发表导数NaN，不吸附显示或补洞。

收敛Stage2另以真实模型q(t±eps)计算heel v/a，包含曲率，进行保持q固定的零heel导数条件近似；只是工程先验，既未证明整足瞬时速度精确0，也未证明真实加速度。角度/latent曲率对v的完整Jacobian未纳入该单次近似，相关结构保持是局部近似；不得把较小曲线峰值当准确。

**输入与证据边界。** 本轮world_contact_20261015/scene由原有序448图像/二维/strict和host-return时基新重放，未读取旧动态场景。该相机变换已用脚踝锚、停止支撑假设和名义0.5s固定延迟，因此身体接触符合它只证明条件工程一致性，不能自证独立世界轨迹。known-camera-motion零世界速度、J0所有网格变换、非均匀q/v/a、全sole球、短失观/NaN、同帧可行incumbent及失败协方差路径已有针对性测试；这不是第6节完整合成clip/噪声/held-out导数验证。实际所有完整/中断试跑、参数、冲突数量和停止决定见实验protocol的world_contact_20261015，最终数字以完成summary为准，不晋升正式主线。

最终完整v7保留448帧，441可解码，满足全部几何434帧；移动Stage2满足101/115，14帧76..79、139..140、275、436..442仍失败。满足的移动帧sole min最大绝对高度4.959mm、每侧全部sole点最大位移9.975mm；所有移动输出含失败最大40.487mm。constrained390/feasible_unconverged43/conflict15；raw可计算10584/10752槽median41.397/P95156.791px，发表导数635/5376，step median204.18/P951383.66ms。world-none EKF参考448帧raw23.211/84.275px。约束改善了几何接近性，却恶化像素拟合并仍有失败，因此整段10mm目标和实时要求均未达成；停止本轮求解扩展，不放宽条件或换更多模型。结果与页面留world_contact_20261015/run_contact_v7，所有早期失败目录保留。

### 5.8 人体异常诊断与已复现的实现错误（2026-10-15）

用户询问三角化是否参与人体优化，并要求完整黑色骨架。当前`run_calibrated_beta12`的人体残差只有原双目二维、姿态及递归状态先验，三角化/Stage仅作显示；world/contact同样没有人体三维观测残差，但其相机变换来自Stage三角化脚踝锚，存在间接依赖。历史frame0初始化也曾使用三角化种子，不能概括为全流程从未用三角化。三组当前beta与原主线共享beta逐值相同，未发现读错体型或180°根旋转证据。

**显示错误已修。** motion exporter误将scene的`raw_triangulated_points_mm`当原始点，实际该字段仅存严格accepted点。448帧7534个有限原始点中2248个被拒绝点因此消失，原16边黑骨架只能显示3685条边实例。改用同源`detector_only_triangulated_points_mm`作诊断显示，同时补齐项目18边拓扑的耳肩连接；两页均保存7915/8064条有限边实例。误差/拒绝只作显示统计，不隐藏有限端点，真正缺失坐标不插值；严格5286点、Stage及accepted踝拖尾原值保持，不将显示原始点升级为人体监督。已更新calibrated和contact_v7页面，数据/脚本检查通过；本次浏览器安全策略拒绝页面读取，浏览器复核未完成，metadata明确记为blocked，不能沿用旧截图证明新页通过。

**姿态先验与原主线不等价。** 原reliability二维项为`.004/(34×625) Σscore×task×Huber25px`，姿态项为`.01/32 Σz²`；当前为`ΣHuber25px/64`及`.5/2.25 Σz²`，另有完整递归Pqq先验。忽略score与踝倍率时，前两项的相对先验强度约弱117倍，不能仅凭同为VPoser就声称沿用了原主线的先验力度。latent范数中位数原主线2.05，相机7.44、world8.78、contact7.36；未据此推断训练分布外，但保存的局部旋转确有异常：world左/右膝局部轴角x小于−5°分别356/320帧，contact左膝局部轴角y中位−20.99°，335帧−28.07°、原主线同帧−1.97°。这些是模型局部旋转指标，不是临床关节角。尚未用单变量先验强度实验确定各因素贡献，不把117直接当新权重。

**当前帧充分拟合也未解决。** 固定相机188/335两帧的beta、手足、门、像素和Huber/先验，只去掉递归先验做200步当前二维诊断：188 raw中位27.080→14.486px、P95158.772→149.492px；335中位26.217→21.837px但P9584.450→100.540px，膝部扭转仍存在。髋尾部残差仍超过100px。因此时序反馈有影响，观测/模型接口仍有未解决错配；不能用更充分优化或新三维损失保证合理人体。记录在`current_frame_pose_diagnostic_20261015.json`，不插入旧轨迹。

**预测越界漏洞已修。** 原`max_abs_latent=6`只约束update proposal，预测`z+v dt+.5a dt²`绕过它，相机109/110/261/262、world400/401/402仍发表越界predicted姿态。现每次预测后检查边界；越界保留`prediction_rejection`原预测值，不裁剪。恢复最后接受姿态、v/a清零、P重置，再使用当前二维重启；不足则expired且模型/导数NaN，成功才增加segment。runner JSONL保存拒绝原因。缺测与当帧可恢复两项回归在旧代码失败、修后通过。

**约束优化参数尺度已修。** v7中373/448帧SLSQP只迭代1次即success，这些帧raw中位43.51px。精确重放0..48并捕获完整真实Pqq（条件数约109990）后，同起点/像素/目标/约束/边界/30次预算，仅作`dq=diag(sqrt(diag(Pqq))) y`坐标缩放：frame48目标480.204→338.821，raw中位65.734→34.774px、P95181.843→118.127px，物理最小余量仍+3.440mm。已在bootstrap/MAP正确映射梯度、约束Jacobian、起点和边界，返回值/可行incumbent保持物理单位，完整precision未对角替代。修后实际路径精确复现该对照；30次仍未收敛，明确feasible_unconverged、P不收缩且发表v/a全NaN。该单帧证明旧数值求解不足，不证明修后全片或人体合理。完整记录含真实Pqq在`world_contact_20261015/map_parameter_scaling_diagnostic_frame48.json`。

未解决的结构风险还包括：非Stage2求解器固定预测支撑脚/顶点，比公开“任意一脚近地面”条件更强；几乎不动的足局部旋转与整足54点10mm约束可能迫使髋膝代偿。旧主线VPoser足旋转也近0，单纯取消足覆盖未必解决。上述两项尚无单变量贡献验证，本轮未改策略或堆新损失。源码修复未重算整段448，现有两页只修显示、人体结果及所有失败保持，不晋升主线；下一阶段应先核对/统一姿态先验尺度与关节可行域、髋观测语义，再作完整比较。

### 5.9 二维优先、整体VPoser与联合运动估计对照（2026-10-15）

**本轮用户约束。** 优先使完整模型COCO关节点符合原双目二维，三角化仅诊断；保留整体VPoser，不增加膝/肘等局部姿态先验。原主线共享beta、首帧姿态、冻结手足颈头、12关节与原始鱼眼门、真实非均匀时基保持。本节覆盖5.8之后的实施与选择，不能把此前“可考虑局部弯曲先验”的建议当成当前授权。

**研究与采用范围。** [Särkkä/Svensson 2023第7.4–7.5节Algorithm7.9](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf)、[流形IEKF指南](https://arxiv.org/abs/2307.09237)支持同一预测均值/P上反复改善观测线性化，最后只计算一次后验；[LM/线搜索迭代平滑研究](https://users.aalto.fi/~ssarkka/pub/lm-eks-camera.pdf)只借鉴数值求解思想，不在现有滤波后叠平滑器。[WNOJ RA-L2019](https://arxiv.org/abs/1809.06518)联合估计pose/v/a，现有_predict的dt^5/20等Q已经具有该形式，因此保留，不冒称新加运动模型。[VPoser官方模型](https://github.com/nghorbani/human_body_prior/blob/main/human_body_prior/models/vposer_model.py)为整体21关节姿态先验，不提供时间动力学；[HuMoR原文](https://openaccess.thecvf.com/content/ICCV2021/papers/Rempe_HuMoR_3D_Human_Motion_Model_for_Robust_Pose_Estimation_ICCV_2021_paper.pdf)及官方优化器区分姿态/运动先验，其固定30fps、静态针孔相机地面优化不能直接套入本移动双鱼眼。本轮未引入HuMoR或新学习模型。

**可选同帧迭代MAP。** 新增pose_app/smplh_iterated_motion_tracker.py；runner增加--update-method iterated-map、--latent-prior-sigma、--map-iterations。默认仍原EKF。每帧固定完整预测P，使用Cholesky(Pqq)白化delta，最小化原Huber25px/sigma8像素、整体VPoser及delta预测先验；每步重新生成完整SMPL-H/原鱼眼投影和相对固定预测chart的Jacobian，回溯真实目标。15次默认预算，不沿用原latent1.2共同修正限幅；latent绝对边界6保持。只优化38维q，v/a通过P_xq Pqq^-1 delta一次条件纠偏；收敛时最终一次IRLS-GN/Joseph近似与根SO(3)切空间transport。未收敛但可行改善仅保留姿态和raw导数，不收缩信息、不刷新good-fit、不发表导数，JSONL明确iterated_update；异常Jacobian诊断使用null而非Infinity，防止严格JSON写出中断。没有人体3D项、局部先验或逐次重复计入像素。

**支撑选择实错已修。** 旧solver把预测支撑点固定，两个物理可行反例被错误拒绝，余量−20.2/−15.2mm。简单改双脚union min仍在平底切换处1步假停，失败方案未采用。最终每次模型求值使用当前sole最低点；非Stage2分别求左/右支撑分支，共享同一prior/P、选目标更小可行解，最后只形成一次后验，额外1次MAP及可能bootstrap次数全部记录。Stage2仍1次MAP，10mm整sole/heel球和episode锚保持。14项world测试包括换脚、换同脚接地点及解析倾斜模型：右/左分支目标76.351/81.552，mean像素0.5213→0.4395px，左右sole7.965/4.805mm；另补当前分支检查，不能由另一脚触地替代所选支撑分支验证；新增回归先复现错误接受、修后通过。仅机制检查，未声称真实448接触改进。

**完整真实448三组比较。** 所有组共享已固定beta细化的frame0-only初值（不重复500步，frame0像素不重用）、同10752槽、同dt、像素门和q/v/a过程。原EKF采用共同修正限幅、MAP不再采用，所以这是反馈求解器整体对照，不能把改善全归于迭代次数；两个MAP的VPoser强弱比较才只改变整体先验权重。修后EKF sigma1.5：median20.835/P95107.055/max203.844px、≤25px58.43%、导数2143/5376；15步MAP sigma1.5：19.148/80.184/399.170px、62.99%、导数0；MAP sigma0.75（整体先验权重4倍）：18.061/95.198/227.893px、67.19%、导数0。三组均448可解码，原权重MAP较EKF中位/P95改善8.10%/25.10%，但不能隐藏max恶化。两个MAP均447/447更新未收敛，原权重439预算/8线搜索失败，强权重445预算/2线搜索失败；不是“updated所以已收敛”。step median/P95分别69.07/95.30、754.75/1404.75、636.01/4162.60ms，不实时。beta和时间逐值相同，全部失败和raw保留。结果/命令/参数/日志在motion_observation_20261015，比较由analyze_smplh_motion_comparison.py生成comparison.json及黑白pixel_comparison.png。

**整体先验权重结论。** 四倍权重将latent范数中位7.088降至4.565，膝局部rotvec x<−5°左/右316/252帧降至144/0，显示整体VPoser确能改变此类模型姿态异常；该轴角不等于临床角或真实合理性认证。但双髋合并视图中位原权重MAP47.49/55.59px，四倍变57.93/61.05px，P95也由141.50/137.25变163.04/169.88px。它改善总体median并牺牲髋/尾部，不能宣布全面优于原权重，也不自动设成默认。未盲扫117倍或继续更多权重。

**已知运动与导数。** 新增evaluate_smplh_motion_known_motion.py，实际SMPL-H固定局部姿态的6×64帧连续刚体/相机运动，2开发clip、4held-out，真实非均匀dt、0/2/8px噪声及一段缺测；位置/导数由独立解析式计算，初速/加速度真值不交给tracker。初始形状/姿态已知是披露的工程条件，尚未验证关节随时间变化的latent运动。解析式与独立五点数值导数3测试通过，同SMPL-H刚体动作最大差约1e−7m。热身0.3s后的held-out完整2592槽中，EKF/原权重MAP发布2500/1224，位置可用RMSE5.225/7.960mm、全raw a RMSE0.7465/0.7679m/s²；共同发布1224槽a RMSE0.2621→0.2022m/s²仅为条件子集改善，不能覆盖全片退化。四倍先验发布48/2592、位置RMSE8.961mm。两段8px的MAP发布0，不能以筛后较小曲线宣称导数更准。全部三算法/6clip数值和失败保存在known_motion_v1，结果补全raw、完整分母及两两共同集合。

**有边界的数值复核。** 同真实frame1固定prior/像素/目标/15预算，仅内存试一次自适应LM：末梯度6.66295→0.84659但目标180.98719→180.99246略差，仍未收敛，未采用/未重跑新全片。原算法15→60次目标仅降到180.97278，梯度仍2.89299。15次终点有限差分0.0005/0.001/0.002梯度∞均约6.65，autograd6.65341与原6.66295一致，不能全归float32差分噪声；曲率谱约1.114..17241，白化梯度也不等于物理误差。保留lm_damping_frame1_diagnostic.json，不通过放宽阈值伪造收敛。

**三角化诊断定位结构差异。** 新triangulation_diagnostic.json对同一自动二维全448的10752槽，自由三角化回投median6.106/P9521.230px/≤25px96.78%，旧calibrated模型19.842/110.569/60.72%；与旧19.795差异为frame0自动/人工输入口径，自动447复现日志误差差4.37e−13px。自动447右目左髋/左目右髋模型median122.31/114.74px且均0/447≤25，髋各仅2帧极线拒绝；自由点双髋距中位197.85mm、模型134.29，大腿自由点264.82/262.60mm、模型357.64/366.17。它提示COCO检测点定义、观测回归算子或共享形状/结构的操作性差异，不能当真实尺寸或beta错误证明。右膝80/447极线拒绝，双目冲突和模型欠拟合同时存在。所有7534有限/82原始越界缺失、strict5286及拒绝保留，未挑strict子集、未把三角化加入损失。

**阶段决定。** 38项针对性测试（base13/world14/iterated8/解析评价3）、Ruff E9/F/编译通过；三个真实448和18个64帧clip运行完整保留。保留可选迭代求解与已验证支撑bug修复，但不晋升任一MAP/四倍先验为默认，不宣称人体已合理或导数已准确，不新增浏览器页/未完成新浏览器复核。下一项有效工作应先解决髋等观测-模型接口差异及非线性求解收敛，再判断是否需要因果短窗；继续堆权重、扩大模型或接独立世界融合没有当前证据支持。

### 5.10 首帧Sapiens2重初始化与下肢复核（2026-10-15）

**范围及跑前结论。** 用户明确仅首帧Sapiens2、后447帧PMPose，原主线beta冻结。输入来自G20261002_sapiens2_body_mainline448_v2的左右COCO17 adapter及native308，同input_448pairs/pair_0000身份、原图逆旋转及映射核对通过；保留原置信度、无人工覆盖。原16人工点没有髋，旧髋本来为PMPose，不能直接把系统髋残差归人工标错。Sapiens髋被衣物遮挡、两膝自由回投约15.70/11.22px，也非独立真值。

旧首帧躯干倾角19.93°、spine1局部x22.41°，原frame0-only种子约10.60°；种子beta稍异，不能单一因果归责。VPoser关节顺序/鱼眼映射和显示R/t未发现错误；body9/10为足末端，真正踝body6/7自由。浮空与负高度存在于保存网格，不是取景错误；只拟合COCO踝缺少脚掌朝向直接观测。额外Sapiens趾/跟低回投仍不能确认鞋边/裤脚与皮肤sole对应，本轮不直接加入目标或局部先验。

**实施与首帧选择。** smplh_first_frame_labels.apply_first_frame_observations接受raw双目17点、只替换frame0并检查同图/边界/score；runner与人工互斥，viewer复现实际替换值。从原单帧种子出发，只读取指定全片模型的共享beta。保留Adam500，增加默认关闭的L-BFGS、整体先验sigma及双精度/尺度归一选项；latent/root/translation尺度0.5/0.35/0.1仅改变求解坐标，无部位先验/3D残差。正常或异常均恢复float32，保留最终梯度，不把预算结束称收敛。

仅比较既有1.5/0.75两个整体先验：1.5候选左膝局部x−7.90°，保留该模型和164帧中止试跑；选0.75后左/右约−0.41/6.59°，非临床角。选中首帧同Sapiens24槽旧→新median29.87→17.22/P9595.87→56.16/max104.05→70.38px；躯干倾角19.93→11.05°、脊柱夹角29.50→20.94°、sole最低点+20.85/+35.45→+10.87/−8.96mm。原来两个低误差髋也退化，完整四视图保留。beta逐值一致；观测与首帧整体先验共同改变，是初始化策略对照而非纯检测器消融。

**完整重跑与停止。** first_frame_sapiens_20261015/replay_selected_init下三组各完整448、均可解码。与5.9相同方法比较后447帧/10728槽：

| 方法 | median旧→新 px | P95旧→新 px |
|---|---:|---:|
| EKF | 20.841→23.711 | 107.020→84.643 |
| MAP sigma1.5 | 19.151→19.621 | 80.015→76.588 |
| MAP sigma0.75 | 18.064→18.136 | 95.022→93.746 |

两MAP各447更新全feasible_unconverged、发布导数0；EKF发布2143→1957/5376。同一新Stage参考下EKF脚底绝对height median34.84→40.75mm，>50mm浮空111→0却<−50mm穿地155→330/896，膝局部x<−5°左/右159/101→441/331。四倍MAP脚底绝对median30.72→30.48/P9574.31→74.73mm，188帧仍约+72.4/+68.9mm。初始化未解决持续下肢问题，不替换默认/主线，不放宽门，不继续扫首帧权重。应先核实髋点语义/模型对应，以及持续足部可观测性和有依据的接触条件。

**记录与显示。** 本次protocol.json记录全部首帧分支、实际命令、选择及停止；selected_initialization_diagnostic.json、distal_foot_diagnostic.json和replay_selected_init下comparison.json/initialization_replay_comparison.json分别保留首帧/足部、完整448/同后447比较。场景同图PMPose本轮新重放，仅显示参考；首帧Sapiens人体与黑色PMPose三角化来源明确分开，不读旧动态、不回填strict、不平滑/吸附。入口replay_selected_init/visualization/index.html实测，33测试、Ruff、编码数组/实际观测与生成脚本语法通过，失败保留。物理正确性、独立世界和真实三维精度未建立。

### 5.11 静止助步器阶段的斜上运动诊断与接触修补尝试（2026-10-15）

本轮保存于`G20261014_smplh_motion448_v1/slope_motion_20261015`。检查全部11个连续Stage1段，不用摆动脚拟合地面，也不从视觉爬坡直接断定鱼眼出错。当前模型全部双目正深度、观测最大离轴约80.2°；NumPy/Torch64原始鱼眼投影与OpenCV最大差约6.83e−13px、Torch32约0.000274px。左右转正回映、原始1920×1080分辨率和同图身份通过，原观测去畸变再回投最大差约4.55e−13px。这只排除这些实现错误，不证明真实标定没有偏差。

另一层鱼眼风险仍保留：实际PMPose和Sapiens入口只读取转正原鱼眼PNG，未做标定透视；本地训练配置未见显式鱼眼增强，不能据此断言预训练中从无鱼眼。现成LocalPerspectiveModelInput默认参数6视图预检查出现脚踝出界或behind，故不直接换输入/裁脚后宣称改善。网络域偏差与全身覆盖透视应作为独立前端试验，记录network_input_domain_diagnostic，当前448输入保持。

**定位到局部姿态与深度的耦合。** 所有长Stage1段发布R/t恒定。35..61中，sigma0.75模型双踝均高上升67.81mm，精确分解为骨盆平移8.28、根旋转11.09、局部姿态/pose blend48.44mm；双sole固定54点质心升77.57/95.92mm，局部项80.58/87.35mm。同段自由三角化双踝只升10.23mm、旧离线模型下降3.79mm，均仅诊断而非真值。MAP1.5双踝升91.55mm而EKF下降15.38mm。故该段不能以逐帧地面漂移或统一全身刚体上漂解释；391..422自由踝自身又升43.90mm，229..263整体下降，也不能把全片异常压成一个固定坡度。原始髋语义/结构不一致、二维深度-姿态折衷和未收敛求解仍是未分离的机制风险，不凭分解结果声称已唯一确定根因。

**两项看似合理的运动修补被实际否决。** 0..61原模型/时间/二维精确复现；保留pose/P、仅禁止未收敛更新反馈v/a，右踝/sole阶段升高50.29/68.83→51.03/73.17mm，未改善。首帧Sapiens→PMPose的24槽增量median29.00px，分别自身相邻帧仅2.71/3.43px，的确有检测器域突变；但仅frame1重置v/a及其交叉协方差、保留Pqq和姿态，右踝/sole升高反增至82.60/119.44mm。两项均未进入生产，不把“可能扰动”写成已证明的爬坡主因。

**同帧接触可行性。** 固定同一诊断P、初值、像素、beta及整体VPoser sigma0.75，使用同SLSQP有/无地面约束对照。61帧sole81.24/99.69→52.33/4.82mm，原始二维median/P9519.04/154.49→19.17/156.63px；188帧72.33/69.30→63.21/4.81mm，17.36/153.69→16.91/154.11px。单脚支撑条件可消除双脚同时悬空，却允许另一只脚仍高52–63mm；不能称两脚已合理。诊断P是明确的新初始对角P，不是实际递归P。上述两次支撑解需58/63迭代，按既有ftol0.01终止，不证明全局最优。

另以35/48/61/188冻结参数只释放足末端body9/10至整体VPoser解码，sole最低点最大改变0.00361mm、投影最大0.00223px，排除这些帧由足末端冻结主导浮空；没有修改冻结策略。保持同网格只将髋回投点换成native1/2，后447髋median/P95反由59.60/165.48变74.79/184.18px；不能靠未经校准的关节替换修复语义。两个反事实均未采用。

因此runner只增加显式`--ground-map-iterations`和`--ground-bootstrap-iterations`，默认30/80保持；80次是本轮有依据的候选预算，未放宽5mm/10mm、观测门或收敛定义。完整`run_contact80`使用新鲜同图Stage场景、相同Sapiens首帧/后447PMPose和冻结beta、整体VPoser0.75，按既有世界接触路径重新估计模型。其与相机MAP的完整比较涉及坐标系及求解器变化，只能作为路线比较；同求解器接触隔离以同帧诊断为限。完整覆盖/全部Stage2条件/原始二维median及P95退化不超过5%作为预先工程停止门，结果在该目录protocol及repair_comparison中保留，不自动提升默认。

静态地面来自9月14日板测、当前人体图像为9月8日；用户本轮明确确认两次采集之间相机相对助步器的安装位置/倾角始终不变，因此沿用原参考，不把日期差异当作病因。该确认仍不是独立动态地面或接触真值。助步器四脚零高度由同源地面/模型构造，不能反过来作为地面正确的独立证据。strict从原PMPose重新计算448，全部记录与场景输入一致，5286通过点、失败及拒绝理由完整保留。

**实际可行解误拒与修补。** 修前run_contact80完整448有9次优化拒绝（全部门前24/24有效），447帧可解码、440满足实际几何、Stage2满足107/115。多帧预算末公共物理slack已正，却因没有可行初值目标而被判constraint_conflict，退回的预测反而越界。现在仅status9预算结束且存在真正可行、有限目标的当前帧解时，缺少可行初值也可保留最低目标incumbent为feasible_unconverged；有可行基准仍必须改善。非9失败、坏几何/分支、非有限解不接收；不收缩预测信息、不发布未收敛v/a。新增feasible_start_available及budget_acceptance_reference，显示实际求解器/预算/停止，不再把内部约束MAP误标EKF。

真实75帧：修前退回预测的整sole位移10.82/11.83mm，修后保留当前可行解9.98/9.91mm、仍无导数发布；0..74保存q/v/a和P对角逐值相同。修后同80预算run_contact80_feasible完整448，448可解码/满足设定几何、Stage2全部115满足，整sole最大位移9.984005mm；438按既有规则收敛、10可行未收敛，后者不发布导数，最终1798/5376样本通过导数门。回归先在旧代码两个子例失败，修后world/ground/base42项及4子测试通过；已知可行基准不改善仍拒绝、早先可行而末解坏可保留incumbent、其他失败仍拒绝均覆盖。

**保留修补但不提升路线。** 修后10752槽raw median23.364/P95112.973/max242.923px；同修前10728有限槽median24.567→23.366、P95109.679→112.977，最大473.788→242.923px，分位存在取舍。相机MAP075原18.132/93.577更好，修后髋median81.57px且35..61双踝均高仍升42.45mm（原67.81）、骨盆升27.01mm（原8.28），不能称爬坡全部消除。单脚支撑还允许另一脚摆动，全片sole最高72.13mm不等于双脚全部贴地；没有独立支撑真值。接触几何门通过，但原始二维相对相机路线退化超过预定5%门，故不提升主线、不继续盲扫预算/先验。完整结果以repair_comparison及protocol为准；同帧页面保留原相机/接触修前/接触修后三组和失败147，浏览器核对实际source、方法标签、预算、未收敛禁发与帧切换。

### 5.12 再审查与当前实际实现逻辑（2026-10-16）

本节汇总当前研究候选，不改变正式离线身体主线。5.7最初的active几何1mm协方差近似、固定预测支撑顶点及预算接受说明均为历史实现；支撑分支和首次可行解按5.8/5.11，协方差按本节修正。

1. **输入与初始化**：448对同源有序图，cam0左/cam1右；网络使用左逆时针90°/右顺时针90°的原鱼眼图，几何前逆旋转回1920×1080原像素。首帧只用用户指定Sapiens2观测细化，1..447帧仍PMPose；全片主线共享10维beta作为已知体型冻结，未借用其未来姿态。整体VPoser sigma0.75，手及缺少目标像素的足末端/颈/头局部旋转冻结，真正踝仍自由。
2. **模型与状态**：SMPL-H真实6890顶点/13776面，COCO17由固定回归器作用于网格生成；优化双目肩至踝12点。q=(t,R,z)的局部维度38，v/a各38，P为114×114；没有12个自由三维关节变量。真实host-return非均匀dt驱动q预测和v更新，SO(3)用右乘Exp及对应Jacobian，Q为白jerk模型。host-return不是独立曝光时间。
3. **同帧二维纠偏**：原始鱼眼K/D与完整双目R/T投影；finite/图内/score≥0.2及35px原像素极线门先行。目标为双目Huber25px、sigma8px像素项，加预测Pqq完整精度的状态偏离项及整体VPoser项。三角化人体点不进入此目标；strict三角化踝通过Stage间接影响世界相机变换，不能称整条世界路径完全不依赖三维脚点。
4. **世界接触求解**：场景静止段保持R/t，移动段背景旋转与严格双踝锚估平移，停止按既有0.5s固定延迟恢复高度/roll/pitch。每帧世界状态经当前R/t逆变换后与原双目像素比较；根平移变换包含shaped pelvis J0。SLSQP同帧只用一个预测先验；变量按P对角尺度归一，保留完整精度目标。当前80次为显式实验参数，默认仍30次；几何bootstrap默认80，不重复使用像素。
5. **接触条件**：所有sole点不低于−5mm；非Stage2按左右支撑两分支分别求解，至少一侧sole min≤5mm，另一侧允许摆动。Stage2双sole min在±5mm内，首个可行状态建立两侧54点sole及heel三维锚，整个片段各对应点位移≤10mm；transition保留锚，明确Stage1才清除。硬约束作用于完整模型参数，未将显示顶点吸附或压平。
6. **均值与信息更新**：同帧解通过P_xq Pqq^-1 dq同步纠正v/a；只在优化收敛时做一次像素和整体VPoser的局部Joseph后验，之后SO(3)误差坐标重置。预算末首次找到可行解可保留为feasible_unconverged，但不收缩信息、不发布导数。Stage2收敛解另以零heel速度/加速度的明确工程假设条件化导数，未把其当实测。
7. **输出与显示**：关节v/a由模型局部时间展开取得，含旋转曲率；逐点检查最近观测、回投、暖机、位置方差、收敛与几何，失败保持NaN。当前scene/mesh/COCO/原始有限三角化黑线/助步器使用同帧R/t；高误差不删除黑线，真正缺失不补。world_state_available只表示所声明坐标状态可用，world_pose_certified始终false。

**新增已复现错误：硬不等式被误作多份独立观测。** 原代码将距边界0.5mm内的每条几何法向，以sigma1mm追加到H/R。相同可行域只复制约束、不增加像素，初始化Pz标准差100mm仍被压至0.1194mm；复制2/10次进一步压至0.08444/0.03776mm。这证明后验随网格采样密度产生虚假信息，可能阻碍后续纠偏，但尚不能据此唯一归因真实爬坡。现在硬约束求解保持，只从高斯信息更新移除这些人工几何行；初始化无新信息时P保持，普通更新保留像素/整体VPoser信息。该P是没有硬截断后验校准的局部近似，显式contact_covariance_certified=false。

真实0..61同变量对照记录在contact_covariance_prefix.json，唯一改变上述协方差政策；同源像素、beta、VPoser、Stage及5/10mm/80次均保持。完整448重跑先要求62帧全部模型/接触可用，原始二维median/P95各不退化超过5%，35..61均踝升高不超过旧42.4487mm；失败则保留证据，不追加权重/预算搜索。后续实际结果及决定写入本实验protocol，不将数学修补直接写成真实运动改善。

实际62帧全部可解码并满足设定几何、4/4 Stage2满足；raw median21.154→18.852、P95132.195→124.076px，但35..61均踝升高42.449→45.017mm。像素改善没有带来爬坡改善，未过完整重跑准入门，故没有运行新448。保留协方差数学修复及62帧证据；最新完整448与当前网页仍为5.11的run_contact80_feasible，不能将旧页面冒称已展示本次新P结果。

**可选鱼眼透视前端的具体修补。** local_perspective.py原先将[x/z,y/z,1]直接求和，使边缘射线因sec(theta)长度过权；现改平均单位方向。焦距原用max−min跨度但主点固定中心，非对称支持仍会裁点；现按maxabs半径与输出半宽/半高计算，并对所需HFOV超过原130°上限显式拒绝，不静默截断。虚拟栅格映回原相机若z≤1e−9，旧OpenCV投影会折回前半球；现remap为−1黑底，单点virtual_to_raw同域显式拒绝。正常正深度往返和bbox中心策略保留。5个新增回归均先在旧码失败，修后本文件10测试、相关调用/旋转5测试及15子测试、Ruff通过；独立复审后补齐普通在线协方差回归，world/ground/base共44测试及4子测试通过。

六个真实0/35/61左右视图在修后默认仍4个FOV超限、2个方向拒绝，故未运行网络或改变当前448观测。诊断性角度范围居中可容纳全部已选17点，但需要HFOV130.27–149.25°，不是已采用算法；图像有效原像素面积约61.38%–89.01%，不能把点位覆盖当整图覆盖。旧记录score0.25口径也发现左61第6点低于阈值，新local_perspective_review.json保留17点原对照，另列16点阈值子集，不静默删点。该前端原本不在本轮448路径中，修复不能被写成本轮人体改善的原因；后续需独立检验完整覆盖、映射和同模型推断，再决定是否重跑。

### 5.13 静止段的逐步因果排查：膝观测恢复、弱深度与求解停滞（2026-10-16）

本轮固定5.10的相机系MAP sigma0.75候选，追踪0..61，而不是将5.11接触候选当作相同求解器的消融。输入仍首帧Sapiens2、其余PMPose、原全片beta、原host-return时间；35..61使用同一恒定Stage地面R/t作诊断，不把它当独立物理真值。没有新增人体、地面或局部关节约束，没有修改生产跟踪器，也未运行新448。

**抬升在哪一步发生。** `trace_motion_chain.py`精确复现已保存COCO点，最大差为0。35..61双踝平均上升67.813mm，其中每帧预测累积36.271mm，二维更新累积31.542mm。37/59帧二维更新分别直接抬高32.151/22.125mm；不是单纯相机或地面转动造成。`recover_event_snapshots.py`从实际35帧预测P恢复事件前状态，36..59每帧pose与原结果逐值一致；同帧对照使用这些实际P，未用重置的对角协方差替代。

**右膝恢复具有局部因果证据，但不能直接删点。** 右膝Sampson原像素距离36→37为36.713→30.311px，58→59为35.868→23.223px；因此跨35px硬门后，两目立即重新进入，基础sigma仍8px。37/59的score均较高，恢复来自几何门变化。固定相同prior/P/beta/其余观测与15步预算，仅在诊断中移除右膝两目项：37帧更新从+32.151变为−15.644mm，59帧从+22.125变为+0.482mm。去左膝也会改变结果，说明完整VPoser潜变量下存在耦合；移除观测后的目标不同，不能将这些低高度诊断直接提升为更准轨迹或生产筛点策略。所有24个原始残差与原目标均保留，见`leave_pair_fixed_prior_diagnostic.json`。

**为何原目标接受抬升。** 37帧膝数据代价下降32.484，髋反而上升17.333、整体VPoser上升2.422，但总目标208.924→186.656；59帧膝/腕分别下降37.292/28.193，髋/踝分别恶化7.905/1.724，总目标292.259→226.658。因此大髋残差不是这两次抬升的直接获益项。初步髋梯度小扰动使脚下降的反向证据与此相符，不能按残差大小排序病因。

当前相机MAP此前每帧未收敛，接受改善姿态并条件回写q/v/a，却不加入像素后验信息。实际37/59的Pqq平移标准差为1.168/2.668m、latent标准差3.582/8.263；这是滤波器内部预测先验的量级，不是校准真实误差。两次实际更新的时间项仅0.04736/0.03001，已经不足以强力阻止此类变化。P条件数约9.4..9.6，问题是信息没有积累，不能称矩阵病态，更不能人为压P制造可信度；此前禁止未收敛v/a反馈亦未改善，单改该回写不是已证明的修复。

**双踝像素对相应深度方向较弱。** 独立XYZ只作观测可辨识性诊断，绝不喂回身体拟合。61帧模型两踝高度比同观测自由点解高45.6/46.6mm，差向量与最弱像素方向绝对余弦0.997/0.999；sigma8局部线性化的高度尺度约20.6/21.1mm，未校准为真实不确定度。59帧将踝高固定到模型35帧高度、只调XY，左右踝双目残差可达[3.0,4.2]/[8.8,7.3]px；原模型约[14.6,13.8]/[13.3,13.1]px。说明像素容许更低解，身体运动学/全身目标的折中选择了更高解；不证明低解解剖正确。另踝点不等于heel或sole，PMPose原23通道的额外6点是MPII/AIC躯干/头点，并非被截掉的脚趾。

**数据确有重复，但单变量对照不支持以其修复爬坡。** 35..61共54张PNG逐像素比较，两路各26个相邻对中8对相同，发生于36/41/44/47/50/53/56/61；12项原AVI位置对照与PNG相同，未见该窗口一目复用而另一目新图。35→36仍有9.563ms主机时间差，60→61为7.910ms。frame_id仅为成功read计数，CSV没有传感器曝光时间/counter，不能将其称物理时间或曝光同步证据。全448的关键点相等计数只作筛查，不扩大为全448图像已比对。

`diagnose_repeated_stereo_frames.py`仅在两路解码图都等于前一图时跳过更新，保留全部62行、原dt、预测和原始误差；0..61共17对重复。35..61上升67.813→95.181mm，未通过预设20%改善门；但61帧绝对踝高161.428→161.952mm仅差0.525mm，差值主要来自35帧起点93.614→66.771mm，不得说末帧额外浮高27.37mm。raw median16.181→16.693、P95128.041→124.721px。当前全未收敛MAP本来就没有重复高斯后验收缩，重复发生的是均值求解/回写；该对照不支持直接跳帧作为修复，也不能证明重复数据或host时基无害。

PMPose当前本地配置`freeze_oks=False`，实现将predicted OKS导出为score，visibility/presence另存；59帧右目两髋score约0.787/0.782、visibility约0.213/0.309。因此score不是可见概率，也未标定为8px噪声；右膝visibility并不低，不能以低可见性解释其恢复。本地源码/config未独立认证历史Docker实现，来源与事件原始字段保存于`data_timing_diagnostic.json`，不据此回改旧检测器输出。

**求解未收敛还有真实数值问题。** 37/59帧15步末的FD与同目标autograd梯度L2差仅0.165%/0.387%，大梯度不是差分伪造。精确重现80预算后：37帧末点eps1e−3跨LeakyReLU折点，以统一阻尼1e−4重构的GN单位方向，FD斜率−4.954、真实AD/+double方向导数却为+2.449；59帧存在窄下降域与float32目标约1e−4量级扰动。这里不是终止前四个阻尼方向的完整重建。直接将eps缩至1e−5使15步梯度误差反增至11.7%/34.6%，不可当作修补。

最后仅两事件各做一次double+exactAD的L-BFGS-B数值参考，完整Pqq/Huber/VPoser/原latent±6域和同prior起点保持，maxiter80/maxls20，未向时序反馈。两次均预算结束、不称收敛；37/59仍分别抬高30.499/20.978mm，目标187.118/229.601还高于原MAP15的186.656/226.658。这既不支持替换优化器，也表明修正差分数值问题不足以单独解释或消除主要抬升。协议、失败及方向限制见`solver_numerical_probe.json`。

阶段决定：保留诊断和失败，不晋升跳重复帧、删膝点、缩eps或双精度L-BFGS候选。优先处理原始膝观测/模型点定义及几何门恢复的观测契约，再以同目标、可靠导数检验求解；持续保留全身原始残差与物理失败。真实曝光时基及独立脚部接触证据仍缺失，当前结果不构成独立三维或动力学正确性。

### 5.14 去除SMPL-H的双目骨架跟踪对照与下一步（2026-10-16）

按用户要求，用现有`StereoPVATracker`重新跟踪完整448帧的12个肩肘腕髋膝踝点。首个合格双目点仅用于初始化，各点维护9维[p,v,a]，此后用左右原始鱼眼4维像素EKF反馈；几何三角化只负责质量门与断段重启，不作为三维观测融合。跟踪过程不加载任何SMPL-H初姿、beta、网格或预训练人体模型。保留原Sapiens2首帧、后447帧PMPose、标定、host-return时间及所有拒绝/失效状态；SMPL-H结果只在跟踪结束后读取作比较。

新增入口`realtime_app/tools/diagnose_free_skeleton_motion.py`，输出`G20261014_smplh_motion448_v1/skeleton_tracking_20261016/`。预先固定两次完整448：原PVA默认sigma4，以及仅将像素sigma改为8的控制，其他PVA参数不动，不扫参数。原始空间运动/状态/拒绝/分段/协方差对角在各NPZ，协议与结论在protocol/summary.json，四面板图为comparison.png。既有同源slope scene只作数值后处理地面参考，不输入跟踪，也未制作新的场景网页；重新从该scene识别11个连续Stage1段，段内R/t均恒定。

主窗口35..61的结果：

| 方法 | 左踝高度变化mm | 右踝高度变化mm | 双踝平均变化mm | 首尾各5帧中位数的均踝变化mm |
|---|---:|---:|---:|---:|
| 原SMPL-H相机MAP0.75 | +54.77 | +80.85 | +67.81 | +35.19 |
| 无SMPL-H PVA sigma4 | −13.55 | +28.12 | +7.29 | +6.19 |
| 同PVA仅sigma8 | −11.74 | +25.40 | +6.83 | +6.25 |

两PVA主窗口双踝54/54槽均为updated、无重启或缺失；没有用平滑/缺失屏蔽压低上升。37/59相邻帧均踝变化，原SMPL-H约32.30/22.85mm，PVA4为0.64/0.33mm、PVA8为1.17/2.16mm。全448原PVA状态为初始化12、像素更新5024、预测310、重启13、失效17；sigma8为12/5068/275/8/13。因此这是持续像素反馈跟踪，不能说成逐帧直接三角化替代。

髋部在同一1788/1792个原始双目槽位比较：原SMPL-H回投median/P95为59.60/165.48px，PVA4为6.40/16.53，PVA8为6.52/16.78。4个缺失槽保留在完整分母中。这说明当前人体路线无法像自由点那样解释这些髋观测；不证明自由点三维位置或原始检测点定义正确，也不能仅凭该差异判beta错误。

边界：PVA右踝在主窗口仍升25..28mm；86..123升约66..69mm、391..422升93..96mm，左踝相应未同升。单侧变化可能含摆动、深度偏差和跟踪误差，不能都称悬空。自由点没有固定骨长：原PVA右小腿全片216.6..1261.1mm（预测异常保留），sigma8仍161.0..1035.4mm；所以它不是物理正确骨架或替换主线的候选。不同状态、初始化、Q与几何门仍有混杂；sigma8只排除了差异完全来自4px这个单一设定，不能称严格“只去掉SMPL-H”消融。

**阶段判断：优先改进SMPL-H耦合路线。** 原35..61双踝同步抬升与两个跳变在无SMPL-H跟踪中未复现，髋残差也显著减小。因此不支持把该段主要问题归为所有跟踪方法共有的p/v/a积分；优先定位完整人体的观测点对应、共享latent状态、先验及求解逻辑。后续单侧变化和骨长漂移仍要求保留数据/时基问题，不能把判断扩大为SMPL-H模型资产本身错误或全片问题已解决。

下一步按顺序只推进一条改进路线：先基于已有髋宽/肢段及双目残差证据，解决模型侧COCO髋点与检测点能否由冻结体型一致解释，不重复鱼眼投影检查或直接加髋权重。随后保留SMPL-H模型与体型，优先试根位姿加显式局部关节旋转的跟踪状态，避免把全身VPoser32的速度/加速度当唯一运动状态；膝踝仍服从运动学链，不强求独立。姿态先验与运动噪声的对应变化必须显式披露，不称单一参数消融。先仅35..61、无新增接触条件，要求双踝升高与髋回投一起改善且不靠骨长漂移/删点；成立才跑完整448，否则停止该候选。自由PVA仅保留诊断参照，不把同源三角化再次作为独立信息叠加到像素损失。

本轮仅上述两次PVA运行，后续数值摘要/字符串修正使用`--analyze-only`读取保存数组，没有重跑跟踪。既有PVA8项测试、入口编译及Ruff E9/F通过；首次测试由仓库根启动的导入失败更正为realtime_app目录后通过，不称首次成功。图像已实看，独立只读复审无阻断。没有训练、新SMPL-H拟合或正式主线切换。

本节结果的后续交互可视化已于2026-10-16交付：`skeleton_tracking_20261016/view/index.html`，三组448同帧同视角切换、原SMPL-H真实网格/骨架叠加、跟踪状态、脚高/髋回投曲线和双目原图；支持35..61循环及关键帧。场景在此次可视化任务从原始同图新重放，旧动态不输入，保存人体/跟踪不重跑。55帧sigma4右膝失效保留空值与断边，37/59/61关键帧及循环/原图实际浏览器检查通过；预览与来源在view/metadata.json。后续新页面不将纯数值阶段使用的旧参考直接当新动态输入，也不改变5.14的结论。

### 5.15 髋为何后偏、脚升放大机制及最小改正（2026-10-16）

本轮仅继续5.14的定位，输出`G20261014_smplh_motion448_v1/hip_pose_repair_20261016/`。输入仍为同一448对图、Sapiens首帧/后447帧PMPose、原标定与时间、冻结beta和初姿、原像素门、VPoser0.75与15次迭代；未加入新网络、接触损失、骨长自由度或平滑。入口为`diagnose_smplh_pose_capacity.py`和`probe_smplh_covariance_feedback.py`，各protocol保存实际命令和参数，`comparison.json`使用同一原始槽位汇总。

**髋后偏不只是首帧，也不能直接认定真实骨架比例错。** 相对独立PVA的COCO髋中点，原模型在操作性地面Y的后偏从frame0约48.9mm增到35..61中位183.6mm。模型髋宽约132–137mm，自由跟踪约190–198mm；两者髋连线方向余弦约0.99，但髋膝距离分别约360mm与215–271mm。同图Sapiens的髋宽约173–177mm、髋膝约269–311mm，和PMPose也有系统差异；这些都是观测定义和几何推断下的尺寸，不能当作解剖测量。

0/35/61每帧每配置仅一次最多60次的无强正则能力探针中，释放局部旋转，甚至允许单帧体型在±3范围变动，可以降低髋残差，但伴随许多上百度关节扭转和多个体型参数触界，且各帧体型不同。该结果不支持直接采用自由旋转解或解冻beta来修补。冻结姿态仅将整个人前移也无法解决：35帧前移50mm，髋Huber代价101.33→101.37几乎不变，上肢代价30.13→226.75。现有二维目标允许牺牲髋来维持其余观测，不能靠根平移或加髋权重消除结构性冲突。

**脚升有跟踪信息反馈的具体放大器。** 原447次更新均未收敛，因此整段没有一次P的观测信息更新；此前事件37/59的平移标准差已达米级，时间先验逐渐失去约束。新增默认关闭的`condition_accepted_iterate`：只对已接受且Jacobian有效的最终迭代作一次IRLS/GN近似后验，原q/v/a条件均值公式不变；未收敛状态及导数禁发不因P更新解除，拒绝/缺测不写信息。这是局部近似，不是数值收敛或统计校准证明。

**髋只改现有观察算子，不改解剖运动学。** 借鉴[KBody原文3.1.2](https://openaccess.thecvf.com/content/CVPR2023W/RHOBIN/papers/Zioulis_KBody_Towards_General_Robust_and_Aligned_Monocular_Whole-Body_Estimation_CVPRW_2023_paper.pdf)关于检测器虚拟关节点的处理，以左右共享的两个系数替换现有COCO回归器的两行：

\[
 h'_L=h_L+a(h_L-h_R)+b(k_L-h_L),\qquad
 h'_R=h_R+a(h_R-h_L)+b(k_R-h_R).
\]

这里只有检测髋点的宽度和沿大腿位置发生定义适配，系数b不是SMPL形状beta。1..10帧先用原映射跟踪，再对这些固定姿态和原始双目像素标定a∈[0,0.5]、b∈[0,0.4]，得到a=0.3621331364、b=0.1094125322，未触界；11帧后冻结。beta、原生关节中心、父子链和网格生成不变，原生骨长全448极差小于0.0003mm。它是当前检测器/当前人的局部语义校准，不是人体比例重新估计，也未在其他人或采集上验证。

| 35..61配置 | 左踝变化mm | 右踝变化mm | 均踝变化mm | 同108髋槽median px |
|---|---:|---:|---:|---:|
| 原相机MAP | +54.77 | +80.85 | +67.81 | 81.02 |
| 仅接受迭代的信息反馈 | +10.50 | +17.17 | +13.84 | 79.40 |
| 再改两行髋映射 | −10.11 | −0.96 | −5.54 | 29.95 |

仅信息反馈没有达到髋残差降低20%的联合门，止于62帧。联合候选通过后只跑一次448，使用上轮冻结系数，不再标定；前62点与短程结果逐值相同，beta与原结果逐值一致。全部448可用，无重启；同10752槽median18.132→18.711px（+3.2%，在预定5%门内），P9593.577→58.446；同1792髋槽median59.571→24.692、P95165.438→64.300。11..447未参与校准的10488槽median18.178→18.707、P9592.742→58.126。这里1792采用所有合法raw髋槽，5.14的1788是与PVA同时有有限残差的交集。

原生骨盆35..61相对原结果前移中位129.2mm，61帧前移147.7mm；因此改善不完全由挪动观察点造成。观察髋相对自由COCO髋的后偏中位183.6→43.5mm，但自由COCO髋不是原生骨盆真值。原35..61和156..194两个长静止段的双踝持续升高判据均不再触发；后者均踝+34.73→−19.33mm，其他所有静止段仍逐项保留。不能把静止助步器推断为整段人体脚必须完全不动。

**尚未解决的代价必须保留。** 非髋8960槽median15.572→17.499、P9549.104→56.364px，均有恶化；447次更新109收敛、338仍未收敛，实际仅681/5376关节导数通过发布门。对相同操作性地面，54点模型足底低于−5mm的足帧槽406→315/896、最小−89.35→−72.68mm，但35..61最低0.78→−11.05mm，局部出现新的下穿。该地面包含旧脚锚假设，不是独立触地真值；即便按这个工程参考，也不能宣布物理一致性已经修好。

阶段决定：保留这两个最小改动及完整候选，通过了预定工程门，正式路线和原网页不切换。当前更支持“检测髋定义不匹配造成拟合折中，加上不写信息的弱时间先验放大脚升”，不支持只怪首帧、模型资产或通用积分。下一阶段固定髋观察定义，优先把运动状态改为根位姿和显式局部SO(3)旋转，VPoser只承担姿态先验，先在35..61检查非髋残差、收敛和足底高度能否一起改善；不先加接触/新网络或扫描权重。10项MAP测试、四文件编译和Ruff检查通过；TRY004类型风格规则忽略以保持原ValueError接口，未重跑无关全量测试。

#### 5.15.1 修改落点与保持项

此次没有把自由PVA的三维坐标接入SMPL-H，也没有改变人体状态的定义。状态仍为38维q=[平移3、根SO(3)切向3、VPoser32]及其38维v、a，完整协方差为114×114；体型是此前全序列预校准的同一组beta，手姿和四个缺少直接观察的身体旋转仍冻结。

| 实现位置 | 原行为 | 本轮行为 | 启用范围 |
|---|---|---|---|
| `realtime_app/pose_app/smplh_iterated_motion_tracker.py`，`IteratedMAPConfig` / `_update` | 只要数值未收敛，就完全保留预测协方差的信息量 | 新增`condition_accepted_iterate=False`；显式启用后，已接受的最终迭代也可写一次近似后验 | 原默认不变；本轮probe显式设True |
| `realtime_app/tools/probe_smplh_covariance_feedback.py`，第10帧后的标定/冻结分支 | 17行COCO观察回归器始终不变 | 只替换第11/12行；第13–16行膝踝观察定义和SMPL-H内部关节均不改 | 仅`--hip-adaptation`候选，第11帧起使用 |
| 同一MAP的`_output` | 未收敛时不发布v/a | 保持不变，近似P更新不解除数值状态和导数门 | 所有候选 |

原始输入、frame0-only初姿及资产路径完整保存在源运行`first_frame_sapiens_20261015/replay_selected_init/map_sigma075/run_parameters.json`；本轮每个protocol再次保存有效配置和命令。像素sigma8px、二维向量Huber平台25px、极线门35px、最低分0.2、最少4关节、VPoser sigma0.75、latent域±6、单帧15次迭代、梯度容限0.05及步长/目标容限0.001均未调参。没有新增髋权重、局部膝先验或关节编号排除。

#### 5.15.2 为什么补信息，以及近似的具体含义

原实现接受部分未收敛但下降的姿态，同时允许q修正通过预测交叉协方差反馈到v/a，却因`converged=False`不缩减P；连续447帧如此，Pqq持续增大，使后续MAP中的时间代价变弱。这是原先保守处理未收敛结果的实际代价，不能把它说成PVA积分公式本身写错。

本轮不放宽姿态接受门：最终Jacobian必须有效，而且必须数值收敛，或相对预测初值有超过0.001的目标下降。没有足够观察、Jacobian无效或更新被拒绝时仍不写信息。新增条件仅为：

```text
condition_posterior = accepted AND (converged OR condition_accepted_iterate)
```

在同一预测状态/P上完成所有迭代，最终只计算一次后验。H由最终像素Jacobian和原32维VPoser先验行组成，并补零到完整q/v/a状态；Σ仍使用原Huber权重对应的像素方差及原VPoser方差。因此这次补回的是现有像素与姿态先验的信息，不是加入新的独立测量。

\[
K=PH^T(HPH^T+\Sigma)^{-1},\qquad
P^+=(I-KH)P(I-KH)^T+K\Sigma K^T.
\]

随后仍作原SO(3)误差坐标重置和对称化。q修正反馈到v/a的条件均值公式保持原样。`optimizer_success`仍只由数值收敛决定；近似信息更新标记为`one_accepted_iterate_linearized_posterior_not_converged`，并记录`posterior_updates=1`。它不是在每次内部迭代重复使用像素，也不证明P在非线性、有偏观察和未收敛状态下具有统计校准。实际109次收敛、338次未收敛及681个可发布关节导数的计数均原样保留。

两项新增测试针对这一契约：同一第一次迭代下两种策略的q/v/a均值相同，P可以不同但未收敛导数仍为空；无合格观察时，即使开启选项也不能产生信息更新。已有8项MAP测试继续覆盖一次后验、整状态条件均值、拒绝与恢复等逻辑，共10项通过。

#### 5.15.3 髋映射怎样标定，为什么不能等同真实骨骼修正

标定用的是**仅信息反馈组第1–10帧已经求得并固定的姿态**，不是每次同时修改姿态和髋点来压低同一帧误差。在原二维门允许的左右髋槽位上，求解：

\[
\min_{0\leq a\leq0.5,\ 0\leq b\leq0.4}
\frac12\sum_{t=1}^{10}\sum_{v,j\in\{L,R\}}
\left\|\frac{\pi_v(h'_j(q_t;a,b))-u_{tvj}}{8}\right\|^2.
\]

此标定是普通L2，不是跟踪阶段的Huber目标；使用一次SciPy TRF least-squares，从(0.15,0.15)开始，最多30次函数评估，实际6次，最终optimality约0.001733。得到(a,b)=(0.3621331364,0.1094125322)，两者都没有触界；第10帧输出仍属于原观察定义，完成该帧后更新回归器，从第11帧起生效。完整448运行读取短程已保存的两个系数，没有重新标定或利用后续帧调整它们。

以左髋为例，新回归器行就是`J'_11=(1+a-b)J_11-aJ_12+bJ_13`，右侧同理。两行权重和仍为1，所以整体平移传递关系保持，但对侧髋的系数为负，**这是仿射适配，不是有身体内部位置保证的凸组合**。它只借鉴KBody“适配检测器观察点定义”的思想，没有复现其完整系统，也没有取得其约束或精度保证。两个系数可能同时吸收检测偏差、初期姿态误差或残留几何误差，不能把它们解释为人的真实髋宽或股骨比例。

本轮原生骨盆相对原结果前移约129mm，说明人体状态确实改变；但自由PVA的COCO髋中点仍不是原生骨盆真值，移动方向更符合当前观察也不是独立三维正确性的证明。形状beta本来来自整段预校准，因此11..447只对**新增髋映射标定**留出，不能称独立体型验证或严格无未来信息的全系统评测。

#### 5.15.4 实验顺序与保留结果

所有路径相对于`research_records/engineering_validation/G20261014_smplh_motion448_v1/hip_pose_repair_20261016/`：

| 目录/文件 | 实际运行范围 | 唯一问题与处理决定 |
|---|---|---|
| `capacity/` | 0/35/61各三种有界单帧求解，共9次 | 比较latent、显式旋转、旋转加体型的观察解释能力；无强姿态正则的低残差解有严重扭转/触界，不采用，也不证明冻结体型全局无解 |
| `detector_geometry.json` | 只读同图PMPose/Sapiens的0/1/35/61观察 | 分辨检测器定义差异；诊断三角化不进入跟踪，不作三维真值 |
| `translation_cost_tradeoff.json` | 固定35/61姿态，整体平移的像素代价计算 | 说明仅把整个人前移会破坏其他观察，不是新的时序拟合 |
| `covariance_feedback/` | 一次0..61 | 只改信息反馈；脚升减轻但髋未过门，停止，不补跑448 |
| `hip_operator_feedback/` | 一次0..61 | 在上一策略上只改两行髋映射，达到联合工程门 |
| `hip_operator_feedback_full448/` | 一次0..447 | 冻结短程系数验证推广；保持全部帧和失败状态 |
| `comparison.json`、`sole_reference_diagnostic.npz` | 保存结果后处理 | 同槽统计、全部Stage1段、原生骨盆/骨长、54点足底参考高度，不再次优化 |

“通过工程门”仅指预先固定的可用性、无新重启、髋median降低20%、全raw median最多增5%及P95不恶化。这些指标没有要求非髋单独改善，也没有证明正确触地。实际非髋median/P95恶化、315/896足帧槽低于参考地面−5mm、35..61新增最低−11.05mm，必须与通过门一起呈现，不能只引用`full_gate_passed=true`。

#### 5.15.5 复现入口、保存范围和下一步

从仓库根目录使用现有CUDA环境；下面三条跟踪命令按顺序对应三个独立输出目录。它们会拒绝覆盖已有目录，已有结果应直接读取，不能为了重现记录而再次运行：

```powershell
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_covariance_feedback.py
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_covariance_feedback.py --hip-adaptation
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_covariance_feedback.py --hip-adaptation --frames 448
```

上游原始观察、首帧来源、模型资产和初姿由源`run_parameters.json`解析；输出`protocol.json`记录每次有效配置与完整参数。`trajectory.jsonl`保留每帧优化目标/迭代史/终止原因、使用槽位和拒绝原因，不能把`updated`当作收敛。

联合短程和448的`result.npz`保存平移、根/身体旋转、latent、原生关节、观察点、beta及最终观察回归器，可由它们重新生成同一人体网格。仅信息反馈的较早62帧文件只保存点、误差、状态和协方差对角，没有保存完整人体参数，因此本次网页不伪造该组网格。原型也未保存完整114×114协方差及全部v/a数值，不能从终帧直接继续滤波，或用位置差分填补缺少的导数；如下一算法阶段需要精确续跑，应先在那个明确用途下补足状态保存。

本轮已经完成原因定位和改正尝试，不继续扫体型、先验或迭代预算。下一阶段尚未实施：固定髋观察定义，用根位姿及显式局部SO(3)旋转维护运动，VPoser仅作姿态先验；先在35..61要求非髋残差、未收敛和足底下穿一起改善。若达不到，不增加接触或平滑把问题盖住。人体和地面缺少独立真值的边界保持。

### 5.16 上述修改的448帧可视化（2026-10-16）

入口`hip_pose_repair_20261016/view/index.html`，生成代码`realtime_app/tools/build_smplh_repair_viewer.py`。初版使用`smplh_repair_comparison.html`与共享`free_skeleton_comparison.js`，提供修正前后同帧同视角切换、原生骨盆/观察髋区分、足底标记、四幅曲线及同帧双目回投。相关数值和原截图保留，以下简约版更新同一入口。

用户随后要求按根目录规范简化视图。当前模板仍为`smplh_repair_comparison.html`，脚本改用专用`smplh_minimal_view.js`；只渲染已保存的联合修正候选及其自身真实网格，不解码或叠加旧结果。移除结果对照、曲线、双目图、关键帧按钮和常驻诊断侧栏，主画布铺满剩余窗口；顶部保留播放、倍率、帧号、时间轴和四个视角。显示开关与当前帧信息分别折叠，底栏直接显示Stage、地面变换和未收敛状态。黑色原始18边、有限拒绝点橙叉、绿色模型观察点、灰绿色XY地面、Z轴、0.5m刻度、实体助步器、双相机和轨迹仍保留；足底标记可选。统一gv映射[X,−Y,Z]，坐标轴标注按显示坐标正方向绘制。

本次只替换当前view的HTML/JS，复用该次已导出的`data.json.gz`及同源场景，不重算人体、网格或Stage/地面，也不把旧场景重标为新运行。压缩包中的历史对照数据仍保留，但简约页只解码修正候选；接触与时序方案仍未实施。后续用原导出入口生成页面时同样采用简约模板。

初版可视化导出时，从原有序448对图、原始PMPose重新计算strict及重放mainline Stage/地面。人体跟踪和拟合未重跑；修正后网格仅由上次保存的旋转、平移、beta和手姿重建，使用真实6890顶点/13776三角面。再生COCO/原生关节与保存结果最大差分别1.79e−7/2.61e−7m；新地面R与此前数值参考最大差0，t差5.68e−14mm，旧动态只在完成显示数据后用于该数值比较，没有进入当前场景计算。

所有人体、原始有限三角化点、严格踝拖尾、助步器实体和双相机使用同一帧变换及[X,−Y,Z]显示；有限拒绝点保留橙色叉号和完整黑色骨架边。Stage/地面held或rejected状态、人体未收敛、严格门拒绝与人体二维门分别保留。没有平滑、插值、人工吸附或接触目标替换。该网页用于检查此次工程修改，不是新的独立物理验证，也不改变正式主线。场景来源、逐关节计数及浏览器检查记录见`view/metadata.json`。

### 5.17 下一阶段规划：支撑条件下的软接触与短窗时序估计（2026-10-16，尚未实施）

用户在查看5.16修正对照后，要求解释机制，并规划脚地接触损失和适当时序平滑。本节是计划，不是新的实验结果。按新要求，将下一阶段顺序调整为：冻结当前髋映射/形状/观察，先在现有估计器验证软接触，再在同一目标内验证短窗时序；显式局部旋转状态保留为求解/耦合问题仍阻断时的后续单独改动，不和接触、平滑同时更换。正式路线仍不变。

**为何两处修改有效。** 髋映射改善模型与检测器观察点的语义对应，不改变原生运动学骨长；它让二维目标不必继续用骨盆后偏、屈膝和其他姿态折中来解释不相容的观察。信息反馈则让已接受的像素/姿态先验真正减小预测不确定性，防止连续不写信息使时间先验变得极弱。两者分别针对系统性拟合冲突和时序放大：35..61的同一逐项对照为均踝+67.81→+13.84→−5.54mm，髋median81.02→79.40→29.95px。它们减轻了问题，没有推导或测得真实接触，仍有338次未收敛、非髋误差增加及足底下穿。

**固定基线与坐标。** 以`hip_operator_feedback_full448`作为A组，使用其冻结a/b及原beta、手姿、二维门、时间、观察强度；保留第11帧开始使用新映射的相同时序。接触和跨帧物理运动在同一个固定地面坐标中表达，二维项仍投回各帧左右鱼眼相机。地面R/t在对照间固定，禁止通过优化地面/人体尺度消除脚地残差。首先用R/t恒定的35..61判断原因；推广448时明确现有Stage地面含踝锚假设，不能把接触项再当独立地面测量，也不能把相机系v/a仅乘R当世界导数。鞋底与SMPL裸足表面定义的固定差异须在设定目标前说明，首轮沿用当前模型表面对操作性地面的工程约定，不逐帧自由调偏置。

**第一阶段B：只加入脚地软损失。** 复用现有54个足底候选点及heel/ball/toe分组，不加载新接触网络，也不直接切回旧整足硬约束分支。防穿地项对两足所有候选点持续有效；贴地项只用于有支撑证据的脚跟或前掌区域；同一支撑区域持续接触时才约束其地面切向移动，足部滚动/支撑切换可释放旧区域，不锁死整只脚的54个点。

设h为候选表面点相对参考地面的有符号高度，S为候选点集，f为脚，k为脚跟/前掌候选支撑区域。可采用下面的最小目标形式：

\[
 E_{\rm foot}=\frac{1}{|S|}\sum_{t,s\in S}
 \left(\frac{\max(0,-h_{ts}-\epsilon)}{\sigma_p}\right)^2
 +\sum_{t,f,k}c_{tfk}\,\rho\!\left(\frac{h^{\rm support}_{tfk}-h^*}{\sigma_h}\right)
 +\sum_{t,f,k}c_{tfk}c_{t-1,f,k}\,
 \rho\!\left(\frac{\|\Pi_{XY}(s_{tfk}-s_{t-1,f,k})\|}{\Delta t\,\sigma_v}\right).
\]

h_support取支撑区域的贴近地面表面高度，s取同一固定局部支撑标记；不能把不断切换的全足最低顶点当同一个材料点计算防滑，也不能把整足表面所有点都拉到z=0。epsilon保留既有5mm工程容差；sigma_h=10mm、sigma_v=20mm/s可作为首次归一化尺度，sigma_p=5mm，均为待执行的工程起点，不是实测噪声或人体标准；首轮固定，不扫权重。优化若用softmin近似支撑高度，评价仍使用真实顶点最小高度，softmin为零不构成无穿地证明。

c是固定的支撑标签/权重，不在同次优化中自由估计，避免令c全部归零来逃避贴地项。先从现有同帧双目图像中辨认35..61的支撑/摆动/不确定时段并保存标签与理由，不依赖当前偏高的模型脚高作为唯一触发条件；不确定区域只保留防穿地项。Stage1仅表示助步器静止，不代表双脚支撑；Stage2双脚固定只有在当前采集协议与观察一致时才作为显式假设使用，过渡帧不强加双足锁定。标签切换处可短暂减弱接触权重，不能对原始观察或输出轨迹做插值。

接触项直接参与同一次人体参数求解，接受后同步生成网格、骨架及状态，不在显示端移动顶点。持续重投影冲突必须保留；软约束不保证精确满足5mm几何门。只允许复用一次最终线性化更新，不能把工程接触假设当作1mm真值反复压缩协方差。

**第二阶段C：在B基础上，用短窗联合估计实现时序平滑。** 使用按真实时间选取的过去0.3s窗口，窗口长度按时间而非固定帧数；只使用截至当前帧的观察，发布当前帧，不回写已发布历史。该窗口不额外使用未来人体图像，现有场景发布延迟另计。它替换单帧反馈求解，不叠加到现有后验输出后再滤一次。

窗口内仍优化同一人体q/v/a，原始像素、VPoser和接触目标保持，跨帧使用现有白噪声jerk的状态转移及Q(delta_t)形成一个时序项：

\[
 E_W=E_{\rm boundary}+\sum_{i\in W}(E_{{\rm pixel},i}+E_{{\rm pose},i}+E_{{\rm foot},i})
 +\frac12\sum_{i\in W}\|x_i\boxminus F_{\Delta t_i}(x_{i-1})\|_{Q(\Delta t_i)^{-1}}^2.
\]

根旋转的差通过SO(3)局部对数/重置计算，不平均轴角。当前先保留VPoser状态以控制变量；若之后改显式局部关节状态，另起单独对照。已有jerk先验已经表达时间连续性，不再重复叠加“速度趋零”“加速度趋零”和多级低通；窗口的作用是让相邻观察共同约束当前估计，允许真实摆动和较大但有观察支持的加速度。边界先验只包含窗口外的信息；不能用已吃过窗口像素的当前后验作先验又重放这些像素，重叠窗口移动时须正确边缘化。缺测/拒绝/失败不借窗口静默补成已观察或已收敛。

这一建议只借鉴[HuMoR原文4.2](https://arxiv.org/html/2105.04668)对接触高度/接触速度的分离；该文也显示单纯更平滑可得到更小加速度却更差运动误差。这里不引入HuMoR模型，不照搬其静态相机、30Hz或整段优化假设，也不把它的权重用作本项目数值依据。

**最少对照和停止条件。** A为已保存的联合修正结果；B=A加软接触；C=B改短窗求解，按B、C顺序推进，每个阶段只改一项，不做参数网格。先从原始初值跑0..61并以35..61为主窗口，不能从缺少完整P/v/a的旧点文件假装精确续跑。每组必须保留全时轴、原始raw槽、接触标签、各项损失、求解状态以及完整q/v/a/P供明确续跑使用。

拟执行的工程门：相比A，B先要求参考穿地幅度/频次明显下降（首轮目标各至少50%）、已标定支撑区域接地和切向滑移改善；髋与非髋的同槽median/P95分别不得恶化超过5%，无新增失效/重启。C再要求减少无观察支持的高频抖动，且摆脚峰值至少保留B的90%、人工可见的起落脚时刻变化不超过1帧，脚地与像素指标继续满足B门；这些幅值/时刻条件只作相对基线的防过平滑检查，不是动作真值。不能以更小加速度或损失自己的下降单独判成功。通过前缀门后各自才允许一次完整448推广，必须保留267帧等旧失败区域，不挑成功段。若接触改善只能以像素/非髋退化或大面积未收敛换取，停止加平滑，回到求解/显式旋转状态单因素修正，不继续提高接触权重。

本节只记录规划，尚未新增接触标签、损失实现、时序求解器或运行结果；无需重跑模型、旧测试或刷新当前可视化。

### 5.18 朝向问题的只读诊断与改进建议（2026-10-16，尚未修改算法）

用户指出脚与髋朝向不符、转动幅度偏大，要求通过合理建模避免错误，不采用直接压低旋转幅度。此次只读取原map_sigma075与hip_operator_feedback_full448的result.npz和当前view/scene/scene_transforms.npz，没有重跑跟踪或改变可视化。全部448帧统一使用相同R_ground_from_left；骨盆方向代理为原生J2−J1，左右脚方向代理为J10−J7、J11−J8，取地面XY投影的atan2并连续展开，全片极差作为角度范围，相邻帧圆周角差绝对值取最大值。脚代理是模型踝→足部关节方向，不是人工鞋尖方向或真实转角误差。

| 方向代理 | 原范围deg | 联合修正范围deg | 原最大单帧变化deg | 联合修正最大单帧变化deg |
|---|---:|---:|---:|---:|
| 原生骨盆横轴 | 14.87 | 42.16 | 8.62 | 4.70 |
| 左脚 | 42.06 | 86.90 | 34.93 | 8.21 |
| 右脚 | 48.05 | 69.24 | 21.63 | 10.10 |

先减去骨盆横轴方向再展开，左/右脚相对骨盆的范围为原42.39/49.16deg、修正后50.97/37.49deg。故世界脚朝向范围增加不能全部归结为踝部乱转；骨盆整体转向也是贡献。髋/踝局部SO(3)最大逐帧变化均减小，足部末端body索引9/10相对首帧旋转变化为数值零。这一检查支持“跳变减少而整体方向范围扩大”，不证明扩大部分全为错误，也不能把全片极差当真实姿态误差或单向漂移量。

当前跟踪只使用肩至踝的12个二维身体点，没有脚跟/大脚趾/小脚趾。冻结足部末端局部旋转不能固定其世界朝向：骨盆、髋、膝、踝的父链旋转仍会改变脚。COCO表面回归点可能对部分扭转有弱响应，但没有直接足部方向证据；足端位置拟合与整足朝向正确不能等同。HybrIK原文3.2以swing/twist分解说明3D关节位置不足以唯一确定局部旋转，另用图像信息估计twist：https://arxiv.org/html/2011.14672 。这里借鉴其可辨识性分析，不直接安装其网络或把同源三角化再叠为独立测量。COCO-WholeBody已有双足共6点定义，可作为未来同一二维观察接口的参考：https://github.com/jin-s13/COCO-WholeBody/blob/master/data_format.md 。

另一个待核查的具体耦合来自当前髋映射h'=h+a(h−h_other)+b(k−h)：b=0.10941使观察髋随膝的位置改变，且a/b标定依赖初期已固定但未必正确的姿态。可能发生宽度、骨盆转向、腿姿之间的补偿；当前没有证据唯一认定这是新朝向问题的原因。应先分开原生骨盆转向、髋轴向扭转、踝/脚朝向，做少量像素敏感性比较；若确认映射通过膝运动解释髋误差，再单因素比较骨盆局部坐标内的固定观察点偏移，校准范围和其余参数保持一致，不为角度更小就接受新映射。

建议顺序：先完成上述最小定位，检查既有修正是否以不受观察支持的朝向代价换取髋残差；再在原双目图中用少量可见足跟/前掌点验证方向信息价值，部分标签留作未参与拟合的二维参考。鞋面与裸足模型标记的语义差异、遮挡与拒绝状态必须保留；有效时才将足部点接入同一个鱼眼重投影目标。若仍存在全身latent把局部误差转移到无关旋转，再单独改根位姿加局部SO(3)状态，VPoser保留姿态先验；不将换参数化本身说成获得新观察，弱可辨识方向仍应保留不确定性。每一步先0..61/35..61，单因素推进，不同时换髋映射、足部输入、运动状态。

5.17的接触和短窗计划继续保留，但不能把它们当作朝向消歧依据：只有贴地高度不能确定平面内脚尖方向，短窗也不能恢复整段都未被观察到的自由度。下一步先定位朝向问题，评价使用未拟合足部像素/可见方向、既有髋与非髋残差和脚高，不能以旋转幅度更小或曲线更平单独判成功。本轮仅形成诊断和建议，没有实施新映射、足部标注/检测器、旋转状态或接触/平滑。

### 5.19 朝向路线第一阶段：敏感性定位与骨盆局部固定髋候选淘汰（2026-10-16）

用户授权按5.18继续。本阶段只推进髋观察定义，未直接限制任何转角，也未同时加入脚部点、接触、时序窗或显式关节状态。入口`realtime_app/tools/probe_smplh_orientation.py`；输出为`G20261014_smplh_motion448_v1/hip_pose_repair_20261016/orientation_20261016/`（以下简称O）。原`hip_operator_feedback`为A，试验`O/pelvis_local_prefix`为B。当前448、简约网页和正式选择均保持。

**输入与最小定位。** 沿用A的source_run/run_parameters、同源Sapiens首帧/PMPose后续raw像素、原标定/时基、冻结beta/初姿/手姿和所有门。先从A保存姿态读取35/47/61，分别扰动整体绕地面法向的yaw、髋局部x轴转动、髋绕rest髋膝骨轴twist，以及踝绕rest踝足骨轴twist；每种只取−10/−5/0/+5/+10deg。髋x轴转动不是严格解剖学swing分解；静态探针不是时间状态或新的运动解。

同一探针比较原动态髋与“在该中心姿态重合、随骨盆刚性运动”的髋，只比较观察导数，不把中心匹配当作可部署标定。按当帧原二维门和Huber25/sigma8分开记录髋、膝、踝、上半身像素代价。47帧左髋局部x转+10deg使髋代价下降1.05、非髋却上升51.62；35帧root yaw+10deg髋下降1.09、非髋上升203.91。这些是无量纲像素目标变化，不包括latent/时间先验；不能据此认定整套优化会选择这些孤立扰动。它们确认髋/腿耦合可改变髋损失，同时反驳“只要压住髋两点就能解释全部骨盆转向”的强假设。

11..61的原适配髋转到骨盆局部坐标后，前后坐标范围左40.88/右34.26mm。解析逆解原两髋仿射映射，分出原始髋表面、扩宽、膝方向三项：原表面前后范围10.08/6.02mm、扩宽2.74/2.74mm、膝项28.62/27.80mm（各项极值不同时发生，不能直接相加）。故膝项确实造成额外局部观察移动，但这还不是错误方向的唯一因果证明。详见`O/diagnosis.json`和`hip_motion_decomposition.json`，未重拟合这些历史帧。

**B的唯一算法改动：固定骨盆局部髋观察点。** 对1..10的A保存姿态，用原始COCO观察点h、膝k、原生骨盆p及root R计算：

\[
u_s=\operatorname{mean}_{1:10}R_t^T(h_{ts}-p_t),\quad
w_s=\operatorname{mean}_{1:10}R_t^T(h_{ts}-h_{t\bar s}),\quad
d_s=\operatorname{mean}_{1:10}R_t^T(k_{ts}-h_{ts}),
\qquad o_s=u_s+a w_s+b d_s,\quad h'_{ts}=p_t+R_t o_s.
\]

仍只拟合两个左右共享系数，使用相同40个raw髋槽、sigma8平方残差及a∈[0,.5]/b∈[0,.4]，最多30次函数评价；1..10外像素不参与标定。6次评价得到a=0.3608484234、b=0.1099658298，未触界，cost929.96；原动态映射标定cost931.16。偏移左[.120278,−.129698,.010281]、右[−.117813,−.128415,.010153]m，均为SMPL骨盆局部坐标，不能把局部y/z当作地面高/前后。

第11帧开始采用该固定偏移。实现为本实验的`PelvisHipMap`子类覆盖两个COCO输出点，其他点仍由原regressor获得，原生SMPL-H运动学/mesh不改；该子类是“所有观察点均J_obs V”常规接口的明确隔离实验例外，未修改生产基类。它同时去掉原髋表面局部变形和新增膝依赖，因此失败不能单独归因于删除b项。不是可断言正确的解剖模型，两个标定系数也不是形状beta。

**运行前门与完整结果。** `O/protocol.json`在重放前保存：A/B同一0..61、35..61主窗口和11..61未参与新标定窗口；髋/非髋同槽median与P95各最多恶化5%；全部62姿态可用、无新重启、未收敛不增加；均踝35→61增高须低于30mm、最低sole最多恶化5mm、低于−5mm足帧不增加。转角幅度不设最小化目标，小转角不能单独通过；数值门通过也仍需未拟合的图像方向证据才能判断朝向。

| 指标 | A原动态髋映射 | B固定骨盆局部髋 |
|---|---:|---:|
| 35..61髋median/P95，108槽，px | 29.95 / 68.95 | 41.26 / 84.39 |
| 35..61非髋median/P95，540槽，px | 20.11 / 83.51 | 18.95 / 69.59 |
| 11..61髋median/P95，204槽，px | 36.09 / 67.80 | 43.34 / 74.02 |
| 11..61非髋median/P95，1020槽，px | 17.28 / 66.26 | 16.46 / 58.56 |
| 双踝35→61均高变化，mm | −5.54 | +36.59 |
| 35..61最低候选sole，mm | −11.05 | −9.11 |
| 35..61低于−5mm足帧数 / 54 | 2 | 3 |
| 骨盆横轴方向范围，deg | 20.28 | 21.10 |
| 左/右脚方向范围，deg | 24.54 / 27.61 | 32.82 / 29.59 |
| 61次更新收敛 / 未收敛 | 14 / 47 | 13 / 48 |

B一次62帧36.59s，全部62可用/单segment；切换前0..10模型COCO与A逐值相同。评价raw槽在极线门之前按有限/score≥.2/图像边界统一统计，拒绝点不从评价分母删除。两个54点sole集合及同一保存R/t仅用于操作性地面评价，未喂跟踪器或改变场景；这里只是历史结果/候选数值对照，没有生成新三维场景或读取旧场景当新网页。方向定义同5.18，范围不是真实方向误差。

**决定：B失败，淘汰，停止扩展。** 非髋改善不足以抵消髋恶化、重新双踝升高及更多下穿足帧；不跑448、不调权重补救、不把B接入当前页面。该结果只否定此次标定/刚性观察算子的替换，不证明动态映射正确，也不排除其他局部定义。保留完整失败轨迹及q/v/a/P、观察使用/拒绝原因、未收敛状态。未标定独立方向准确性，不宣布朝向问题已修复。

**足部图像可用性与下一步。** `O/foot_visibility.jpg`是原图0/35/47/61双目upright下部裁图，来源/裁切/缩放见`foot_image_sources.json`。已实际查看：双脚前掌、鞋面与部分脚趾可见；脚跟受裤脚/鞋部遮挡，不能可靠定位。此为助手目视可用性检查，未生成点坐标、人体接触标签或人工真值。下一阶段保留A，只标注清楚可见的big/small toe或前掌参考，逐点不确定/遮挡状态保留，先冻结拟合帧与留出帧再接鱼眼像素项；不能把拖鞋边缘、鞋面图案或推测脚跟冒充裸足模型表面点，也不能凭这几张图声称真实触地。没有脚跟不等于所有前掌方向信息都无效；需要验证其可辨识性和语义误差。

本阶段10项现有iterated MAP测试通过，入口编译和Ruff通过。大型NPZ、完整trajectory及裁图留本地；仅源码、现有文档和必要小protocol/summary/诊断数据提交。不创建额外验收报告。复现/后处理命令如下，已有目录和试验拒绝覆盖，不为核对而重跑：

```powershell
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_orientation.py
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_orientation.py --replay
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_orientation.py --inspect-saved
```

### 5.20 补充前掌方向观测：点约束448未过门，方向约束62亦停止（2026-10-16）

按用户继续分析/修正/测试的要求，保留A的动态髋映射，不重试5.19已失败的骨盆刚性算子。发现`G20261002_sapiens2_body_mainline448_v2/{left,right}_sapiens_raw.jsonl`已保存同源448图的全部308点，原COCO-17适配丢掉了足部点。本次复用这些结果，不新推理/训练、不引入足部网络或光流链。初期考虑手动种子加LK的方案在执行前被替代，未实际运行；协议保留此变更与原始不可靠图像读点，不把未实施方案写成结果。

**单一改动与输入契约。** 新隔离入口`realtime_app/tools/probe_smplh_foot_observations.py`，输出`hip_pose_repair_20261016/foot_direction_20261016/`（F）。Sapiens308定义读取本地官方`third_party/sapiens2/repo/sapiens/pose/configs/_base_/keypoints308.py`最终`dataset_info`，不是文件前半的COCO定义：15/16/18/19分别为左大/小脚趾、右大/小脚趾；17/20为heel，因遮挡语义不足不纳入。SMPL-H官方表面顶点3216/3226/6617/6624从已安装smplx的`vertex_ids.py`解析定义，未把native踝关节当toe。

308点坐标原为left_ccw90/right_cw90的1080×1920模型输入像素，经已有`model_to_raw_point`逆转正为1920×1080原鱼眼像素；左右分别为(u_raw,v_raw)=(1919−v_model,u_model)/(v_model,1079−u_model)。原帧名/pair_id和同一冻结prompt首实例顺序核对，不改人物关联。置信度仍只作≥.2有限/边界门，不能解释为可见概率；复用35px极线门，保留拒绝原因。脚跟没有被推测或补齐，鞋面/拖鞋边缘不能当裸足顶点的人工真值。

`FootPointMap`只在完整SMPL-H前向后附加四个真实表面顶点，原17点/native骨架/网格不变。`FootPixelTracker`把这四点与原12个身体点一起加入同一次current-frame MAP，以相同sigma8px、向量Huber25px与原15步预算求解。不是对已求好的脚朝向做后处理，也不直接压低转角；各关节通过同一38维t/root/VPoser状态影响最终人体。原q/v/a/P预测、VPoser0.75、原beta/手姿/足部末端冻结及第11帧应用原a/b完全保留，a/b没有用新增脚部数据重新标定。每帧仅最终线性化更新一次P，非收敛导数仍禁发，发布接口保持12身体点；本实验允许二维求解额外使用四toe。原至少4个身体关节的支持门不得被toe绕过，已有回归覆盖；实际本次已处理帧支持远高于4。

**评价与预先门。** 首轮1..61均可用足部检测，0帧不重复初始化观察；35/47/61整帧足部点留出，身体点仍按原规则使用，不误称全帧所有观察留出。图像读点在模型优化前冻结：初次网格目视读点中的右视图/小趾位置不可靠，在查看源检测叠图后标为不可用，原坐标仍保留在`F/heldout_reference.json`；实际可用仅左视图两只大脚趾×3帧=6槽，声明8px读点不确定度。它们是助手图像读取的操作性参考，未经独立人工复核，也不是盲标或真实三维真值。Sapiens对这6槽median/P95为6.52/9.16px。

在拟合前保存`F/protocol.json`：髋与非髋35..61及11..61同raw槽median/P95各最多恶化5%，无新失效/重启，sole最低不恶化超过5mm且低于−5mm足帧不增；六槽图像读点median改善≥25%，未参与拟合toe的大小趾连线图像方向median改善≥30%。后者是前掌横向方向一致性，不是真实脚尖yaw误差或独立准确性。没有把转角范围更小写成成功门。

**一次62帧结果。** 45.85s，62/62可用、单segment；留出24个toe像素槽median/P95由28.89/57.54降至12.92/23.00px，12个foot-view横向方向差median/max由11.15/30.36降至3.01/10.97deg。六槽图像读点median/P95由41.86/57.75降至15.47/21.99px。35..61髋median/P95 29.95/68.95→31.42/71.33px（median+4.92%，接近5%门）；非髋20.11/83.51→18.28/77.36px。11..61髋36.09/67.80→32.38/66.88、非髋17.28/66.26→17.77/69.08px，全部守门。

35..61均踝变化−5.54→−4.59mm，最低sole−11.05→+1.00mm，低于−5mm足帧2→0；骨盆/左足/右足方向范围20.28/24.54/27.61→12.66/11.90/11.34deg，仅为伴随变化，不是真实角误差。61次更新收敛14→12、未收敛47→49，仍存在数值问题；本阶段未把“未收敛不增”设为门，也不隐藏该代价。464个toe观测槽实际用于求解，留出三帧使用0槽。原图同帧红A/绿B/黄留出检测叠图`F/heldout_projection.jpg`已实看，47帧原鞋外toe回到前掌附近，但鞋/模型定义和残差仍存在。

**完整448协议。** 前缀通过后才执行一次，全部参数不调。新增267/335/412为推广阶段未参与拟合的toe帧，仍留出35/47/61；只有新增三帧可称本轮未用于前缀筛选的时间留出，仍同片同检测器、beta来自历史全片。运行前在`F/toe_pixels_full448/protocol.json`固定：全448髋/非髋median/P95各≤A的1.05倍、448姿态完整且无新重启、全片最低sole不额外下降超过5mm/−5mm足帧不增；新增留出toe median改善25%、前掌横向方向median改善30%。接触/短窗/显式局部旋转均未同时改变，操作性地面R/t仅为已保存候选的数值比较，不作为跟踪目标或新场景显示输入。

**完整448实际结果与停止。** `F/toe_pixels_full448`一次运行346.39s，448全部可用/无重启，前62模型COCO与前缀逐值相同、beta不变。新增267/335/412的24个toe像素槽median/P95 60.59/92.37→22.33/36.38px，12个foot-view方向差median/max 19.42/39.63→6.75/19.99deg；全部六留出帧48像素槽42.29/88.93→15.02/34.46px，24方向槽median14.66→4.89deg。全片髋1792槽median/P95 24.69/64.30→25.28/65.83，非髋8960槽17.50/56.36→19.13/52.41px；非髋median+9.32%，唯一失败的预先门。sole最低−72.68→−36.87mm，低于−5mm足帧315→259/896，仍不能说接触正确或无下穿。447更新收敛109→90、未收敛338→357。3528个toe观测槽用于求解，六留出帧使用0槽，全部原身体评价像素仍在，不删异常。

全片失败不能被方向改善覆盖，**点约束B不采用、不切网页/正式选择、不继续扫权重**。`F/body_tradeoff.json`逐关节分解显示，肩/肘/膝median变化较小，主要代价在双踝median10.82/10.86→16.98/16.42px；0..61之后各时段非髋median均恶化。源图留出回投`heldout_projection_full448.jpg`已逐排实看：267/412显著靠近前掌，但335左足仍有明显位置残差。此证据支持检查脚趾绝对位置/裸足鞋面语义与踝目标的折中，不证明冲突唯一来自哪个模块。

**一次有根据的修订C：只约束前掌图像方向。** 相对B只更换足部残差，不调整人体/原身体像素/髋映射/状态/先验/15步预算/sigma8。对每foot/view令d为投影small−big toe二维向量，d_obs为同源检测向量，目标残差为：

\[
r_{\mathrm{dir}}=\|d_{\mathrm{obs}}\|\left(\frac{d}{\|d\|}-\frac{d_{\mathrm{obs}}}{\|d_{\mathrm{obs}}\|}\right).
\]

沿用向量Huber25/sigma8，每foot/view只计一次二维方向残差，不把同一趾对算两个独立方向传感器。两toe都通过原像素/极线门且图像间距≥16px才可用，短于两倍8px尺度的方向拒绝；不约束前掌中心或投影宽度，不拉向零转角。观测间距作为单位方向噪声传播的一阶尺度，仍未声称已完成统计标定。代码把这项等价嵌入同一16槽线性化求解；unused半槽保持未使用。不会把编码的方向像素当作实际脚点发布，输出仍为真实mesh表面toe。

运行前`F/toe_direction_prefix/protocol.json`记录原因、公式口径和原门；35/47/61继续留出足部，已看过的267/335/412不得再称修订C的新留出。C一次62帧37.74s：原三留出帧toe median/P95 28.89/57.54→21.71/40.74px，方向median/max11.15/30.36→2.73/6.35deg，六槽助手读点41.86/57.75→20.86/31.03px。35..61髋median/P95 29.95/68.95→32.28/74.14px（+7.79/+7.53%，两项失败），非髋20.11/83.51→17.54/70.43px。11..61髋35.42/67.90、非髋16.32/65.30px，各守门。最低sole+0.71mm、低于−5mm足帧0/54、均踝变化−6.58mm；骨盆/左脚/右脚范围7.30/12.63/8.92deg；收敛14→21、未收敛47→40。232个方向槽实际使用，留出帧0槽。`heldout_projection_direction.jpg`已实看，位置残差保留，方向好看不能覆盖髋门失败。

**最终决定。** B的完整448和C的62前缀都未过其完整门；保留两者为方向信息有效但身体折中仍未解决的失败候选，不跑C448、不继续调权重/换评价门，也不加接触或平滑。不能凭本轮就把耦合唯一归为latent：脚趾/踝语义、固定体型、局部可表达性与未收敛仍是竞争解释。下一阶段应先在35/47/61固定root/上半身，做真实SMPL-H局部髋/膝/踝SO(3)静态可表达性诊断：同一原身体+足部方向目标下，检查局部旋转是否能保持髋/踝位置并改善方向；有明确改善后再单独改运动状态，VPoser仅作姿态先验。静态探针不是跟踪结果，不能拿关节角更小当通过。此下一阶段尚未实现。

15项现有MAP及新增足部接口回归通过，覆盖无toe时与原状态/协方差逐值相同、有toe单次P更新、原最小身体支持门、方向残差不拟合中心、缺一toe只拒绝本趾对；入口编译及Ruff通过。支持门检查晚于第一次运行代码补充，实际三组所有更新原身体支持最少10/9/10，新增分支均未触发，无需重跑。全部大NPZ/轨迹/原图留本地，协议/必要小统计与源码入Git；未建新三维网页，当前448及简约页保留。

复现命令（已有目录拒绝覆盖，直接读取保存结果，不重复跑已通过前缀）：

```powershell
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_foot_observations.py
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_foot_observations.py --frames 448
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_foot_observations.py --inspect-saved
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_foot_observations.py --frames 448 --inspect-saved
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_foot_observations.py --observation-mode direction
.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_foot_observations.py --observation-mode direction --inspect-saved
```

### 5.21 两个足部候选的独立简约视图（2026-10-16）

用户要求分别可视化5.20的两项修正。入口分别为`foot_direction_20261016/toe_pixels_full448/view/index.html`（真实完成448帧）与`toe_direction_prefix/view/index.html`（真实完成62帧）。新增`realtime_app/tools/build_foot_observation_viewers.py`，共享既有`smplh_repair_comparison.html`和`smplh_minimal_view.js`；共享脚本仅增加由payload决定帧范围/标题/失败原因，仍兼容原简约页。当前两页均停47帧侧视，首帧原始观察髋中点作为固定取景中心，使两页取景一致；保留播放、倍率、帧/视角和折叠显示/帧信息，不增加曲线或常驻诊断面板。

场景由本任务重新从同源有序448双目图与PMPose生成strict，重放当前mainline Stage/地面，保存于`foot_direction_20261016/visualization_20261016/`；读取历史命令仅复用静态配置，不读旧动态地面/助步器姿态。两页共享同一任务的新场景，方向页只读取对应前62帧。人体跟踪/拟合未重跑，网格只由各自保存的旋转/平移/beta和原冻结手姿重建，不读旧网页顶点。COCO与原生关节同保存值的最大差分别保存于两页metadata，均≤2e−6m；真实6890顶点/13776三角面/41328索引、统一同帧R/t与[X,−Y,Z]、XY地面/Z轴/0.5m刻度、实体助步器/双相机/轨迹保持。黑色17点18边仅缺失端点时断开，有限strict拒绝点仍保留橙叉；不按编号静默排除右膝。

页面标题明确448/62帧，状态栏保留数值未收敛与“候选未通过”，帧信息保留原脚底/髋回投/strict/地面原因和各自失败门。方向页不填充62帧以外的数据，不平滑/插值/显示吸附。场景黑点首帧为PMPose，人体首帧Sapiens2，两者同图但不能误称同检测点；metadata说明。初次序列导出因默认GBK读UTF-8场景JSON失败，改显式UTF-8后只导出两页，已完成的场景未重跑。

实际浏览器检查47同帧侧视、位置267和方向61、播放47→52/53、帧上限447/61和console无error；两页分别保存`preview_frame47.png`。入口编译/Ruff与共享JS语法通过，网页只供观察本次未通过门的工程候选，不改变正式身体选择或物理结论。调用`.venv-cuda/Scripts/python.exe realtime_app/tools/build_foot_observation_viewers.py`先新场景再导出，`--export-only`只复用本任务已完成场景；拒绝覆盖已填充的view目录，已有页直接读取。大型GZ/场景/截图留本地，仅源码/既有文档/小metadata与场景命令提交。

### 5.22 方向候选上的软接触试验：已实现，62帧未过门（2026-10-16）

用户暂时接受5.20的方向候选作为后续研究基线，授权尝试接触。此次对照是`toe_direction_prefix`对`soft_contact_prefix`，均位于`hip_pose_repair_20261016/foot_direction_20261016/`；不改变正式选择，也不把此前方向候选未通过原A髋门改写成已通过。入口`realtime_app/tools/probe_smplh_soft_contact.py`从同一初值重放0..61，35帧才启用接触，0..34的平移/根旋转/latent/完整P与保存方向基线逐值相同；beta、手姿、髋映射、raw观察、足趾留出35/47/61、原门、真实时间和15步预算保持。

**标签与几何。** 在查看优化结果前，读取原双目35/39/43/47/51/55/59/61足部图，保存`foot_direction_20261016/contact_source_review.jpg`。右前掌35..47在左脚迈步时近似静止，左前掌49..61在右脚迈步时近似静止；48帧与其余不确定支撑均不加贴地，未标注脚跟。冻结26个支撑足帧，标签不由模型高度推导。这是拖鞋前端可见界面支持的工程支撑假设，不是裸足接触真值，也没有训练在线接触判别器。

固定与方向基线相同的`hip_pose_repair_20261016/view/scene/scene_transforms.npz`，确认35..61的R/t恒定，行向量地面点为`v_C @ R.T + t`。没有重估地面、形状或人体尺度；现有Stage地面含踝锚，不提供独立物理测量。沿用左右各54个sole候选、各18个ball候选；防穿地残差为`max(0,-h-0.005)/(0.005*sqrt(108))`，108点归一化避免点数成为额外权重。贴地只取已标记ball区域的**真实最低顶点高度**h，目标0、尺度10mm，使用pseudo-Huber能量`sqrt(1+(h/0.010)^2)-1`，没有把全部点压平。未使用有正偏的centered softmin，未增加防滑/短窗/平滑。裸足表面对操作性地面z=0仍是本轮约定，鞋底差异未解决。

**同次求解。** `SMPLHIteratedMotionTracker._engineering_residuals(data)`默认空；实验子类从同次SMPL-H前向保留108表面点，工程残差进入相同MAP目标、批量中心有限差分、GN方向及最终梯度认证。没有第二次位姿修补或显示吸附。身体支持不足仍拒绝，接触不能单独维持跟踪；非有限正则/雅可比保留拒绝状态。每帧最多一次原像素/姿态信息反馈，接触曲率不作为独立测量写入P。最终姿态改变会间接改变像素Jacobian/P，因此这是“工程正则化点估计+原项局部信息近似”，不是完整统计接触后验，也不能说接触完全不影响协方差。原预测P的交叉协方差将同一q修正传递给v/a，未收敛导数继续禁用。

**预先门与结果。** 原方向基线35..61已无低于−5mm足帧，故不再要求零穿地频次下降50%；改要求26支撑足帧绝对高度median/P95各下降至少50%，无新下穿/缺失/重启，主窗口及11..61的髋/非髋raw median/P95各不恶化超过5%，未收敛不增，留出方向median≤max(3deg,基线×1.1)、max≤基线+2deg。协议在求解前保存，固定尺度只跑一次，不扫权重。

| 工程量 | 方向基线 | 加软接触 |
|---|---:|---:|
| 支撑ball绝对高度median/P95，mm，26足帧 | 27.73 / 56.57 | 18.78 / 52.56 |
| 主窗口髋raw median/P95，px，108槽 | 32.28 / 74.14 | 31.80 / 72.80 |
| 主窗口非髋raw median/P95，px，540槽 | 17.54 / 70.43 | 17.58 / 71.79 |
| 主窗口最低sole，mm | +0.710 | +0.545 |
| 主窗口低于−5mm足帧 / 54 | 0 | 0 |
| 35/47/61留出方向median/max，deg，12项 | 2.73 / 6.35 | 2.98 / 10.02 |
| 主窗口未收敛 / 27 | 21 | 23 |

62姿态均可用、无重启，完整q/v/a/P均有限，运行36.05s。11..61髋35.42/67.90→35.15/67.61px，非髋16.32/65.30→16.41/64.72px，所有身体像素门通过；留出脚趾使用0槽。支撑median改善32.25%、P95仅7.09%，均未达50%门；方向max及未收敛门也失败，整体`passed=false`。

分开看，左支撑49..61高度median41.18→32.82mm，右支撑35..47为13.17→11.37mm；剩余较大间隙主要在左支撑段。最差留出方向移到61帧右目左足（10.02deg），正是正在拉近地面的左足；61帧该足左目也5.85→8.87deg。这支持“贴地与当前方向/姿态表达发生折中”，但不足以唯一认定latent耦合、图像误差、地面或裸足/拖鞋语义中的哪一项是主因。某些帧工程项自身上升而联合目标下降，亦保留在轨迹中，不能把接入损失当每项必然改善。

**决定：本次软接触候选不采用，不跑448、不提高权重、不叠加平滑；暂保方向基线。** 保留实现供后续明确对照，但不自动接入原页面/正式身体。下一步先在保留的61帧检查支撑标记语义和局部足姿能否同时满足像素方向/高度，必要时再单因素检验显式局部旋转；换参数化、鞋底偏置或更强权重都尚未执行。

`soft_contact_prefix/protocol.json`保存输入/固定地面/标签/尺度/门，`comparison.json`保存全部门及按足/视角诊断，`trajectory.jsonl`保存全帧使用/拒绝、初末工程能量及最终防穿地/贴地分量，`result.npz`保存完整状态/网格再生参数及支撑高度；`contact_comparison.npz`保存同槽地面量。大型数组/原图留本地。21项相关测试通过，覆盖真实min、地面R/t方向、单边容差/归一化、支撑切换、正则目标与像素信息协方差、身体支持门和非有限拒绝；新增入口/测试Ruff通过，核心旧整数类型异常TRY004提示未改。命令：`.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_soft_contact.py`；输出目录已存在时拒绝覆盖。此轮未刷新三维页，未增加新的时序算法，接触高度是同参考工程量。

### 5.23 接触失败后的局部可表达性诊断：不直接切换裸SO(3)（2026-10-16）

用户要求继续尝试。精确再生保存方向/软接触网格：左支撑49..61的sole最低高度median27.229→26.973mm，ball高于sole最低点的差median13.207→2.266mm；右支撑35..47 sole最低12.231→10.354mm。因此上轮左ball主要靠前掌变平，整足下降不足。54sole来自模板蒙皮/低高度候选，ball18只是纵向分组，未校准为真实解剖跖骨头；但它包含模板真实最低点，扩大到原完整足候选最低仅约多降1mm，不能抹去当前2–5cm间隙。未修改支撑区、鞋底偏置或权重。

隔离入口`realtime_app/tools/probe_smplh_contact_expressibility.py`，固定方向候选35/40/47/55/61的root、平移、beta、手、髋回归器、地面及其他15个body旋转，仅双髋0/1、双膝3/4、双踝6/7可动，feet9/10固定。参数图分别为VPoser32解码后覆盖所有非下肢旋转、及6个局部SO(3)右乘18维；每种各做data-only与data+原110接触残差，共20次静态求解。两边同时去掉时序P与VPoser先验，不拿latent单位与弧度设相同先验。这是宽松几何诊断，不替换原跟踪MAP。35/47/61脚方向仍留出，40/55使用原门通过的方向；径向25px Huber/8px与接触尺度不变。TRF/x_scale=jac、中心差分1e−3、最多60评价、gtol0.05、ftol/xtol1e−4，无参数网格或轨迹拼接。

| 帧 | 原支撑ball mm | latent+接触 mm | SO(3)+接触 mm | SO(3)最大踝改变量deg |
|---|---:|---:|---:|---:|
| 35 | 2.34 | 0.19 | 0.34 | 91.85 |
| 40 | 11.81 | 9.58 | 0.21 | 109.03 |
| 47 | 30.33 | 10.62 | 1.65 | 91.86 |
| 55 | 41.18 | 34.96 | 0.81 | 176.28 |
| 61 | 31.37 | 3.02 | 0.68 | 98.24 |

SO(3)五帧支撑sole分别−12.68/−10.32/−19.50/−11.16/−12.61mm；35/47/61方向max46.87/92.60/78.63deg，55虽拟合方向仍有176.28deg踝变化。低ball/总代价不能证明合理接触。latent在61把ball31.37→3.02mm并保像素/方向门，但另一足下穿；40/55支撑未减半、47留出方向恶化。所有静态接触候选未过全门，全部角变量和分项成本保留。

五帧latent→下肢局部18维旋转Jac均rank18（相对阈值1e−4），条件57.27/85.55/89.94/87.03/48.31，不支持“缺自由度所以直接换SO(3)”。满秩不证明全局可表达或先验适合；一次局部优化失败不证明不可行。18次因ftol/xtol退出，47/61的SO(3) data-only另两次耗尽60次评价；20次均未满足梯度门，接触10组optimality0.23..3.03仍超过0.05。`success=True`不等于梯度收敛。有限差分浮点噪声、真实min活跃点切换、条件数及目标冲突须区分，下一阶段仅核查导数数值。

完整目录`foot_direction_20261016/contact_expressibility_complete/`保留协议、全部20组summary与static_poses.npz，仅是静态姿态，不混回序列；7.30s，真实前向COCO与保存姿态差均≤2e−6m。首轮`contact_expressibility/`在35帧后因NumPy bool写JSON中断，协议与`contact_expressibility_run.log`保留；修复标量转换后新目录完整执行，未覆盖失败。2项新测试检查指定关节右乘/冻结及径向Huber，Ruff通过。命令`.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_contact_expressibility.py`。未换时序状态、未改变默认或页面，独立物理真值边界沿用5.22。

### 5.24 自动微分的配对62帧：导数实现吻合，但两组均未通过（2026-10-16）

5.23发现多次目标变化停止却梯度不合格，因此新增`realtime_app/tools/smplh_contact_autograd.py`，只将同一次MAP的目标Jacobian由固定步长中心差分换为Torch自动导数。原类仅抽出可覆盖的`_objective_jacobians`，默认有限差分数值/路径不变。新类完整经过原VPoser→SMPL-H→相同COCO及toe点→raw鱼眼→方向编码/同接触残差；root在原预测切空间用右乘指数，零角导数有限。模型/权重仍float32，重撤回和投影float64匹配原NumPy；NumPy代价、门、15迭代预算、原参数/qva过程及单次原像素/姿态信息反馈不变。最终信息更新使用该自动pixelJ；发布点导数仍用旧有限差分，不谎称全部路径已改。

实际资产55帧、非零delta（根旋转0.01/−0.015/0.005rad）、方向与接触均开：Torch/NumPy像素forward最大差2.27e−13px、contact差0；随机方向的中心差分h=1e−3与自动导数相对L2差pixel1.09e−4、contact2.83e−5，所有J有限。body12/no-contact、direction/no-contact、direction/contact三个分支的forward和维度均核对；h=3e−4误差反升，提示float32量化存在，但不能把所有未收敛归因于它。暖态J约无接触0.068s、有接触0.233s；这是局部调用耗时，不是实时端到端性能。

运行`probe_smplh_soft_contact.py --jacobian autograd --without-contact`得到`toe_direction_autograd_prefix`，随后同代码`--jacobian autograd`得到`soft_contact_autograd_prefix`。两组均从原0帧初值跑62；接触组仅35帧开启原软接触，0..34 q/P与匹配自动导数无接触对照逐值相同。与原有限差分对照的原始12关节、脚方向、标签、beta、手、地面和权重相同，未加平滑/偏置。必须用无接触数值对照分离导数改动与接触，不能只展示新接触组较旧结果gap更小。

| 35..61工程量 | 原方向/有限差分 | 自动导数/无接触 | 自动导数/加软接触 |
|---|---:|---:|---:|
| 髋median/P95 px | 32.28 / 74.14 | 43.91 / 75.06 | 34.95 / 75.04 |
| 非髋median/P95 px | 17.54 / 70.43 | 19.42 / 72.49 | 18.93 / 75.99 |
| 支撑ball绝对高度median/P95 mm | 27.73 / 56.57 | 7.81 / 18.22 | 9.68 / 31.41 |
| 最低sole mm | +0.71 | −47.65 | −13.92 |
| 低于−5mm足帧 / 54 | 0 | 28 | 13 |
| 留出方向median/max deg | 2.73 / 6.35 | 3.75 / 10.74 | 3.40 / 11.55 |
| 未收敛 / 27 | 21 | 26 | 24 |
| 完整62运行s | 37.74 | 98.98 | 174.74 |

无接触自动导数相对原方向基线即失败；第一帧最大COCO差已9.50mm，13帧首次超过10mm、22帧超过30mm，61帧81.69mm。每组的prior随自身反馈不同，后续总代价不能直接逐帧当同一目标横比。自动接触相对匹配自动无接触减少下穿，但支撑高度median/P95反而升高，11..61非髋P95 60.40→67.49px、绝对sole门等失败；两组整体passed=false。62均可用、无重启，完整状态有限，35/47/61脚趾使用0槽。全61更新的未收敛原方向40、自动无接触56、自动接触54，自动两组45/44次耗尽预算、11/10次线搜索失败，不将有限值或梯度代码正确说成接触成功。

**决定：自动导数实现作为隔离研究工具保留，不替换当前方向版或正式主线；不跑448、不加迭代预算/权重或平滑。** 接触最小高度仍有活跃点切换，当前优化/反馈路径对数值方法敏感，尚未唯一归因。下一步固定同一保存前帧的q/v/a/P先预测，逐帧比较FD/自动求解；如尝试几何不等式接触，也须在同一prior/raw目标下明示条件并保留不可行结果，不能把新方法与已漂移的prior混比。

25项相关测试通过，默认旧MAP、正则信息更新、raw门、接触几何、静态旋转及新零角SO(3)/鱼眼导数均检查；新增/修改工具及测试Ruff通过，核心仅原TRY004类型异常提示。两个目录的protocol/comparison小记录纳入Git；完整q/v/a/P、轨迹、toe/sole与运行日志留本地。未刷新网页、未新增加时序算法，固定踝锚参考地面/拖鞋与裸足定义的工程边界仍在。

### 5.25 同一预测先验下的接触可行性：40通过，55/61仍失败（2026-10-16）

为分离5.24的跨帧反馈累积，入口`realtime_app/tools/probe_smplh_contact_feasibility.py`只检查40/55/61，各自从原方向结果的**前一帧**完整q/v/a/P恢复并调用原`_predict(actual dt)`，同帧三组预测q/P逐值相同；不使用当前帧后验作先验，不跨帧传递新候选。原身体raw槽/质量门、40/55脚方向、61脚方向留出、beta/手/髋映射/地面/支撑标签、VPoser先验和P白化保持。原FD与自动微分各用同15步软接触MAP；第三组把软接触改成明确几何不等式，最小化原pixel+latent+预测P目标，SLSQP最多60步、ftol1e−6。不同预算/目标仅作定位，不当算法速度或公平性能排名。

几何条件为108sole每点`h>=−5mm`，已标记支撑ball真实最低点`min(h)<=10mm`，并保留原`|latent|<=6`及双目拟合点正深度域。固定原裸足表面对操作性地面约定，不调鞋底偏置或地面；只要求至少一个区域点接近，未声称整片前掌接触、防滑或真实触地。真实min的上界是非凸并集，某次局部失败不证明全局不可行。几何验收预先固定1e−6m数值容差，最终精确网格实际高度公开。边界约束最优时无约束梯度不必为0，因此记录solver状态、实际margin与估计active-set KKT，不套用原无约束梯度门。

| 帧/量 | 保存方向基线 | 同prior FD软接触 | 同prior自动软接触 | 原MAP+几何条件 |
|---|---:|---:|---:|---:|
| 40支撑ball mm | 11.81 | 11.26 | 11.28 | 10.0000 |
| 55支撑ball mm | 41.18 | 39.91 | 39.73 | 10.0001 |
| 61支撑ball mm | 31.37 | 30.88 | 30.90 | 10.0023 |
| 55非髋P95 px | 66.23 | 67.30 | 67.36 | 88.97 |
| 61非髋P95 px | 104.37 | 104.71 | 104.70 | 113.78 |
| 55方向median/max deg | 2.83 / 5.14 | 2.80 / 5.17 | 2.84 / 5.18 | 4.03 / 7.49 |
| 61留出方向median/max deg | 5.54 / 6.35 | 5.56 / 6.40 | 5.56 / 6.40 | 4.79 / 6.02 |

同prior时FD/自动软接触解非常接近，与5.24整段差异远小，支持跨帧状态/信息反馈会累积数值路径差异；不能因此断言协方差机制错误或导数选择是唯一病因。40的几何条件组25步成功、所有像素/方向/可行门通过；55的38步成功、实际支撑ball违反约0.000068mm可在预定数值容差内，双足sole9.97/13.62mm，但非髋P95增加34.33%、方向门均失败。61耗尽60步，支撑ball仅越10mm上界0.002275mm，大于预定0.001mm容差；这不是物理不可行证据，且该帧非髋P95+9.02%仍独立失败。其方向保持，不能把失败统一叫“脚方向都坏了”。

55几何组的pixel能量159.67→152.87反降，但预测先验28.04→63.87、原MAP总目标207.48→239.33；61分别pixel223.75→231.38、预测7.24→42.33、原MAP249.88→293.52。55较保存C的根平移改36.04mm、根转0.965deg、最大body局部改6.78deg；61为13.41mm/0.954deg/3.50deg，40为1.10mm/0.281deg/0.95deg。三帧拟合点双目最小正深度0.175..0.184m，无镜像解或latent越域；这次失败不能照搬5.23裸SO(3)大角扭曲解释。55恶化主要在膝：左目左膝78.29→102.04px，右目左/右膝64.11→86.67、96.55→114.23px；61左目左/右膝103.35→112.93、69.78→79.48px，右目65.86→77.94、123.79→129.86px。全部raw槽保留，髋改善不能抵消这些尾部退化。总代价可通过牺牲局部raw残差/方向重新分配，不能只看pixel总和或地面gap。

55保存几何候选上的白化y随机方向导数检查另有数值限制：步长0.003/0.001，AD目标方向导数0.55909，中心差分0.51238/0.62822，相对差8.35%/12.37%；高度方向相对L2差11.89%/13.90%。这与5.24在原delta坐标固定点的1e−4级吻合不同；白化后物理扰动小、float32量化/局部非光滑均需区分，尚不能唯一归因。未新增优化或放宽门，不能把SLSQP success或估计KKT当严格最优证书；55图像失败和61预算/图像失败不依赖这一数值解释。

**决定：三帧全门未过，不运行62几何候选，不扫权重/预算，不把40成功帧拼入原序列。** 下一步的有效问题是具体raw观察/固定人体与地面语义、运动先验为何冲突，以及是否有独立证据支持修改；不是再堆一个接触/平滑模块。当前原方向研究基线、正式路线、页面均不变。SLSQP未发布新P或速度/加速度，不能把静态姿态当新的时序结果。

`foot_direction_20261016/contact_matched_prior_feasibility/`保存运行前protocol、全部9组结果及分项cost/几何margin/原梯度/KKT/域/完整角改变量、同prior/candidate参数NPZ，完整53.04s。恢复/预测同源与三组q/P相等已在真实运行核对，最终域和角改变量从保存候选追加，无重优化。工具Ruff/编译通过，沿用已通过25项相关数学/MAP测试，不为纯诊断重复运行整套模型。命令`.venv-cuda/Scripts/python.exe realtime_app/tools/probe_smplh_contact_feasibility.py`，已有目录拒绝覆盖。失败、输出/输入路径与物理边界保留，不推送。

## 6. 研究依据与未实施改进

本节研究设计仍未整体完成；已实现第4节估计器、第5节对照、5.7的Stage条件约束及5.9的迭代MAP与已知刚体运动检查。完整关节运动held-out、短窗和独立世界融合仍未实施；5.9的固定局部姿态刚体/相机clip不能替代完整研究门。保留同一SMPL-H/回归器/观测契约，先处理已定位问题，不以新网络或物理模块替代诊断。下方原步骤以5.9最新实施/停止决定为准。

### 6.1 核验的已有研究及实际借鉴范围

本阶段只核验与当前最小改进直接相关的三条主线，未做“最新全球SOTA”排名：

1. [SMPLify-X / VPoser，CVPR2019原文](https://arxiv.org/html/1904.05866)：二维特征约束参数化全人体，VPoser提供姿态先验。它支撑“优化模型参数、回归骨架作观测接口”的做法；没有证明在VPoser坐标中作常加速度预测就得到真实人体加速度。现场VPoser.decode输入latent，输出21局部旋转，未读取时间/速度；静态姿态先验与运动先验须区分。
2. [HuMoR，ICCV2021原文](https://arxiv.org/html/2105.04668)及[官方代码](https://github.com/davrempe/humor)：作者VPoser-t基线已有姿态先验加时序约束，全方法则学习状态转移。原文还展示“加速度更小但运动误差更大”的过度平滑情形。因此本项目不能把曲线小当目标成功。官方RGB流程假设静态相机、30fps、整段/重叠批优化，不能直接照搬到移动双鱼眼和非均匀dt。其模型三维关节平滑也不是本项目已实现的二维-only目标。
3. [SLAHMR，CVPR2023作者项目](https://vye16.github.io/slahmr/)：将相机与人体运动解耦，说明世界运动需要相机运动证据。借鉴坐标组织与联合投影原则，不在当前基线再安装其DROID-SLAM/PHALP等组件。

由研究机制与本地证据作出的**本项目判断**是：先改善观测契约、时序目标与求解，而不是立即增加新的学习/物理系统。这不是上述论文已验证本项目的结果。对助步器辅助运动的先验适配、当前fisheye标定、曝光时间与人体尺寸仍需本地证据。

### 6.2 对用户“二维构造加速度再积分”思路的具体判断

可以让加速度作为待估变量参与积分预测，但二维差分不能被直接视为三维加速度。即使单点p在固定相机系中，

\[
\dot u=J_\pi(p)\dot p,\qquad
\ddot u=J_\pi(p)\ddot p+\dot J_\pi(p,\dot p)\dot p.
\]

深度和投影曲率都参与；若相机在动，还有坐标运动项。均匀dt时，独立像素噪声sigma的二阶差分标准差为sqrt(6)sigma/dt²；示意取8px和0.04s，约12247px/s²，并非实际测得噪声，也不能直接换算人体米制加速度。所以现基线直接拟合原始像素，再通过同一状态预测与反馈估计运动，避免把差分噪声先放大。

多关节体的端点加速度也不能独立自由设置。若p_j=f_j(q)，则

\[
\dot p_j=J_j(q)\dot q,\qquad
\ddot p_j=J_j(q)\ddot q+\dot J_j(q,\dot q)\dot q.
\]

当前q是根和VPoser参数，v/a是其局部坐标变化率；关节物理曲线由完整模型映射生成，包含曲率。它是运动学状态估计，不是已识别的关节力矩/质量/受力模型。若将来学习驱动，必须明确学的是状态转移、控制量还是力矩，不能混名。

### 6.3 下一步按证据门推进，不增加平行模块

**步骤1，保留当前更好的1.2像素拟合候选，核对观测与时基。** 0.4留作原始参考；类/CLI默认0.4未自动变更，推荐下一轮显式传1.2。不继续扫描更多限幅值。针对髋/右膝核对COCO索引/回归器来源、raw逆旋转、独立标定与遮挡语义；少量新人工点只作未参与拟合的参考，不允许先拟合再评价同一标签。研究采集时基需真实曝光证据，不能凭返回日志宣称同步。

**步骤2，先用已知运动检验当前估计器。** 沿用同一SMPL-H/回归器/鱼眼投影生成小规模已知运动；包括静止、匀速/匀加速、正弦肢段、停止转向，以及人体静止/相机移动。沿用非均匀时间和2/8px已知噪声、短遮挡；不能只生成局部二次轨迹给同类模型出容易题。已知三维点和导数仅用于评价，不进入二维拟合损失。开发与评价按完整clip分开，报告位置、速度、加速度误差、滞后、不可用率，保留全时轴与共同可用集合。合成只检查模型内估计能力，不替代真实人体验证。

实现时直接复用模型映射和tracker.step，显式提供raw像素/合成采样时刻，不把raw合成点塞入会再次逆转的PMPose读取器或伪造采集元数据。先无噪声/无遮挡，再单独加噪声或遮挡。固定模型资产、形状与预设种子时要披露“已知模型”条件；生成器真实q0/v0/a0不能混入以首帧二维初始化为主评价的运行。真导数从生成器连续运动独立计算，不能复制tracker的局部二次导数算式；相机运动先变换固定世界模型点，评价相机系导数，不要求其为零。这些实现仍未开展。

**步骤3，若已知运动证实当前时序模型/反馈仍不够，再替换为0.3s因果短窗拟合。** 它替换现有EKF反馈，不在EKF后另叠平滑器或融合第二个同源估计。继续固定beta/手足颈头、同一二维点/门和model regressor；每帧只使用[t_k-0.3,t_k]，只发布当前帧，不重写已经发布的历史帧。

最小候选使用(q_k,v_k,a_k)共114个局部变量，在窗口历史时刻tau=t_i-t_k≤0生成：

\[
t_i=t_k^{\rm root}+v_t\tau+\tfrac12a_t\tau^2,\quad
R_i=R_k\operatorname{Exp}(\omega_k\tau+\tfrac12\alpha_k\tau^2),\quad
z_i=z_k+v_z\tau+\tfrac12a_z\tau^2.
\]

这里t_k^{root}是根平移，不是时间戳。omega_k、alpha_k表示当前时刻的局部体角速度/角加速度；令xi(tau)=omega_k tau+0.5 alpha_k tau²，历史时刻实际体角速度为J_r(xi)(omega_k+alpha_k tau)，一般不等于omega_k+alpha_k tau。历史导数必须从完整旋转轨迹计算；这个局部表达式不保证全窗恒定物理角加速度。每个历史模型重新经左右raw-fisheye投影，目标形式为

\[
E=\sum_{i,c,j}\rho(\|\pi_c(f_j(q_i))-y_{icj}\|/\sigma_{icj})
 +E_{\rm pose}(z_i)+E_{\rm rate}(v_k,a_k).
\]

E_rate是明确记录单位/尺度的弱数值先验，不声称物理真实性。只用二维数据项，不增加三维观测；局部二次/0.3s是待证假设，不能假设全片恒加速度。优化求解使用迭代数/下降目标/步长控制作为数值条件，不能再以“每帧必须改latent≤0.4”作为物理限制。

上一结果仅作warm start，不能把已经含窗口内像素的EKF后验当独立先验，再把同一窗口像素重新计入一次。最小版每次依据当前窗口和固定弱先验独立求解；如果以后要递推概率边界，必须只包含窗口以前的信息，届时另做单独设计，当前不引入边缘化/图优化设施。

短窗只沿用当前像素质量、时效、热身条件，**不能沿用旧EKF的P或默认已保留其位置协方差门**。待实施的最小工程近似可用当前窗口线性化数据项I_data=J_data^T W J_data及固定先验曲率Lambda，定义P_win=(I_data+Lambda)^(-1)，再以当前模型点Jacobian G_p映射为G_p P_win G_p^T；病态/非有限时不发布导数。该近似条件于冻结模型、工程像素噪声和先验，不是校准后验概率；正则填补弱方向不代表像素识别了该状态。应分别记录数据项的弱方向及先验贡献，位置标准差阈值须在步骤2的已知噪声/运动中检验，再确定是否沿用0.1m。不能仅因正则逆Hessian数值较小就标记导数可靠，也不能因初始化来自上一帧而积累其旧协方差信息。相邻重叠窗口输出相关，不按独立样本解释。

**步骤4，先通过导数门，再接世界。** 独立相机轨迹可靠时，将同一模型在固定局部坐标内估计，用各历史相机变换投影；不直接二阶差分噪声相机pose。仍不要求完整房间表面重建。人体/相机/助步器共用逐帧变换，Stage只辅助；随后按用户指定比较“移动助步器全段两个真实模型脚跟世界硬固定”和无锚版本，不替换为踝、不裁剪冲突区间，保留投影残差/不可行。

**学习方法的进入条件。** 同定义经典估计器在已知运动和独立标签上仍有明确未解决误差，且观测、坐标、时基、体型已可检验时，才评估一种学习运动先验是否提供额外收益。当前不实施HuMoR、RL或另一个图像网络；研究借鉴的是机制和限制。

### 6.4 下一阶段评价目标和停止条件

这些是未来门，不是已达到的效果：

- 固定本次1.2候选为下一轮基线，固定输入、体型、标定、二维门及输出条件；一次只改变时序估计方式，不同时换检测器/回归器/体型/相机后端。
- 已知运动的held-out加速度RMSE目标至少下降20%，位置误差与滞后不得明显恶化（位置误差相对增加不超过5%作为首轮工程门）；同时报告全样本覆盖和共同可用集合，不靠更多拒绝点换较低误差。这些数值是本项目预设目标，不是文献性能保证。
- 真实448全部raw误差median/P95不得靠舍弃髋/膝、混未来帧或重写旧输出改善；逐关节统计与未参与拟合的人工像素一致性必须保留。较低真实2D误差仍不叫真实3D/导数准确。
- 如果数据/标签/标定契约冲突，先停止时序扩展并处理该冲突；若短窗经过最多两次有依据的修订仍未通过已知运动门，停止继续堆模型，不自动进入神经网络路线。
- 运行时间单独记录；拟合与导数方法通过后再在同一模型内优化Jacobian计算成本，不在本阶段承诺短窗实时。

### 6.5 相机与世界运动的边界

当前p_C(t)是每时刻左相机中的坐标信号，包含相机运动。若p_W=R(t)p_C+t(t)，则

\[
\dot p_W=R\dot p_C+\dot Rp_C+\dot t,\qquad
\ddot p_W=R\ddot p_C+2\dot R\dot p_C+\ddot Rp_C+\ddot t.
\]

速度/加速度不能只乘R。当前独立ORB/cuVSLAM结果未通过，不能将其跳变传入人体加速度；Stage场景部分使用脚锚，不能用同一脚锚自证独立世界运动。优先在固定局部坐标内利用窗口相机相对运动做投影；完整房间表面重建不是必要前提。

<a id="historical-boundaries"></a>

## 7. 已停止路线与必须记住的修复

### 7.1 历史结果只保留定位与决定

| 路线/实验根 | 保留的结果与边界 | 当前决定 |
|---|---|---|
| `G20261005_reference_weight_fit_v1/runs/reference25` | 360帧legacy三维身体路线 | 历史身体对照，非当前选择；重放须原argv与legacy |
| `G20261002_sapiens2_body_mainline448_v2` | Sapiens2/PMPose冷启动及腕参考/手先验消融 | 历史工程对照；不把人工腕或旧权重带回当前主线 |
| `G20261012_ankle_weight448_v1` | 448二维踝权重历史来源 | 不混入当前reliability模型参数 |
| `G20261014_stereo_pva_baseline448_v2/tracking_final` | 独立12点因果PVA，低像素误差但骨段距离漂移，长失观外推明显 | 保留自由点对照；完整SMPL-H已实现，不再以“接骨架”为未来任务 |
| `G20261014_rl_physics448_v1` | 三种子合计301056步PPO；真实448无锚数值可用69/448/63帧，回投median223.73/432.38/207.80px；最佳合成位置改善仅约1.3% | 当前候选失败并停止，不继续盲目加训练或直接接回主线 |
| `G20261014_visual_rig_compare448_v1` | ORB全448未初始化；cu373/448有导出几何支持但位姿跳变和右目mask问题未解 | 无合格独立世界轨迹，未融合Stage/未注入writer |

表中实验根同样位于`research_records/engineering_validation/`。详细失败状态与参数从原`protocol.json`、summary、日志及`AI_PROGRESS.md`读取；本次只删除文档中的冗余展开，不删除实验数据。自由点、离线模型、完整SMPL-H、PPO的约束/初值/可用集合不同，不能把其像素数值当严格单变量优劣排名。

PPO使用未识别质量/胶囊/腕支撑假设，不能称真实人体动力学。训练/合成held-out使用旧rigid heel；最终真实回放改用SMPL-H表面3387/6787硬锚，固定权重未重训，两者不可混称同约束训练。PPO首帧人工点只作诊断，而第4节人工点参与了首帧细化；同一标注在不同实验的使用身份不同。这里的heel实现没有接入当前SMPL-H因果基线。

### 7.2 废弃逻辑不再作为实现说明

| 已废弃的操作 | 当前有效处理与原因 |
|---|---|
| 用观测权重和自归一化，整体低分仍恢复满强度 | reliability使用固定原始槽数，数据强度可随证据减弱 |
| 根轴角直接作欧氏锚；平均完整PCA主值轴角初值 | SO(3)旋转弦距离；选完整有效手假设，保留所有观测 |
| mean限幅、P仍按完整增益收缩 | q/v/a和Joseph共用sK；旧`run_final`漂移P95677.54px明确失败 |
| 只检查被使用的像素或沿用旧点Jacobian | 当前所有合法raw像素参与导数质量诊断；刷新当前点Jacobian |
| 把数值可生成/低残差/小加速度等同恢复准确 | 分开记录可用状态、全raw误差、导数门与独立验证缺口 |
| 将像素二阶差分直接当米制加速度再开环积分 | 原始二维位置闭环纠正同一模型状态；投影/运动学曲率不可省略 |
| 将踝点、rigid heel site或共享手姿当实际接触事实 | 明确点定义；世界heel仅作用户指定假设对照；保留冲突和不可行 |

跟踪早期`run/run_v2/run_v3/run_manual_v1/run_final/run_final_v2`仍保留在原实验根。`run`只有36/448可用；`run_final`是协方差不一致失败；`run_manual_v1`的首帧细化被复用，其整段轨迹不与当前结果拼接。中间版本多因素改变，不把修复前后写成单变量精度证据。

<a id="validation-and-use"></a>

## 8. 验证范围与使用入口

已有13项跟踪/标注测试覆盖真实dt、已知加速度、旋转曲率、失观/恢复、本点时效、限幅与协方差一致性、极线坏点、raw拒绝点质量、根跨π、标注别名/逆旋转/错图/中文路径等。五个实现/测试文件的编译、Ruff E9/F已通过；1.2候选实际完成448并核对配置唯一差异及导出契约。本次文档整理不重复这些模型运行。

真实资产内部映射已验证：三帧由保存latent/根再生mesh最大差4.17e-7m，发布轴角再生最大差3.58e-7m，全448网格回归COCO最大差1.19e-7m。它们证明模型参数/网格接口一致，不验证真实人体三维位置或导数。

复现参数、当前候选完整命令、失败后如何处理、下一步任务见[交接说明](HANDOFF_SMPLH_TRACKING_20261015.md)。不覆盖已有输出；不从整段beta或终帧初始化因果实验；不把未实施短窗写成已有代码。大型NPZ、原图、标注与权重留本地，不能假定仅克隆Git即可重跑。

### 8.1 旧章节引用兼容

本文已从40个追加章节整理为8个主题章节。旧第27–30节对应第2–3节，旧第31–32/35节对应第1节，旧第39节对应第4节，旧第40节对应第5–6节；旧第1–26/33–38节仅保留第7节所需历史结论。旧日志中的数字节号按其编写时理解，不重写历史日志。

现有仓库链接依赖的`implementation-guide`及旧第27/28/35节四个锚点保留显式兼容。历史细表、失败全过程从原实验记录与Git历史定位；它们不再作为本页的现行操作说明。
