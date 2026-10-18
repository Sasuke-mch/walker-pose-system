# 自动微分接触路线：现有逻辑与448帧记录

记录日期：2026-10-18。运行日期：2026-10-17。本次依据现存代码、protocol.json、summary.json、trajectory.jsonl及结果数组补充说明，没有重新优化、调参或重跑448帧。

**当前结论：这条路线已完成完整448帧回放，保持高研究优先级，但鞋高、贴地和求解收敛检查未通过。** 下面记录实际执行逻辑，不能把设计目标当作已经实现的效果。

## 1. 路线身份与记录位置

路线标识为 smplh_causal_autograd_contact。它使用SMPL-H完整前向模型，在逐帧运动预测后，以原始双目身体像素、前掌横向方向、姿态先验及脚地软接触共同修正身体状态；MAP目标的雅可比使用自动微分。

路线管理以 realtime_app/configs/current_mainline.json 的 route_management、method_routes 为准。不同方法独立保留输入、配置、结果和证据，不固定永久唯一的身体“主线”。旧 mainline_selection 和场景 mainline 都是兼容选择器，不能据此认定某方法全面优于其他方法。本文不描述其他强化学习或批量拟合路线的运行逻辑。

本文路径以仓库根目录 D:/my_works/walker_pose_system 为基准。实验目录为：

    research_records/engineering_validation/G20261014_smplh_motion448_v1/
      hip_pose_repair_20261016/foot_direction_20261016/
      autograd_contact_shoe25_full448_20261017/

后文称其为“本实验目录”。本文件是该运行的详细说明；算法总文档为 docs/CURRENT_SMPLH_ALGORITHM_20261002.md 第5.27节；按用户要求，同内容副本保存于 docs/实验报告/自动微分接触路线说明_20261018.md。实验状态仍以本实验目录的原始protocol、summary和逐帧记录为依据，副本不改变其证据等级。

## 2. 实际输入与冻结资产

### 2.1 图像、时间与观测

| 输入 | 实际来源与使用方式 |
|---|---|
| 原始448对图像 | research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/input_448pairs；帧号0～447 |
| 身体二维观测 | 同一 people_1 目录下 c3_predictions/left/pmpose/raw_predictions.json、c3_predictions/right/pmpose/raw_predictions.json；仅身体COCO5～16进入跟踪 |
| 首帧身体观测 | G20261014_smplh_motion448_v1/first_frame_sapiens_20261015/first_frame_observations.json；替换首帧PMPose身体观测供初始化及评价，首帧不再重复做像素更新 |
| 前掌二维观测 | G20261002_sapiens2_body_mainline448_v2/left_sapiens_raw.jsonl、right_sapiens_raw.jsonl；308点中的15、16、18、19 |
| 双目标定 | realtime_app/calibration/results；cam0为LEFT，cam1为RIGHT，进入几何前均回到原始鱼眼像素 |
| 真实时基 | 图像选择清单与原采集时间；时间0～16.035239s，帧间隔0.0069816～0.0695054s，不按固定30Hz计算运动 |
| 同源场景 | 本实验目录 scene_input/strict.jsonl、scene_input/scene/scene_transforms.npz；由本次同图输入重新计算strict并重放Stage/地面 |

表中以G开头的目录均位于 research_records/engineering_validation。身体跟踪没有把三角化点作为独立三维测量；strict参与场景重放和显示，地面变换通过接触残差影响当前身体解。

Sapiens2输入转正方向为左图ccw90、右图cw90；观测先映射回1920×1080原鱼眼坐标。读取前掌时检查pair_id和pair_XXXX.png身份，并沿归档的同一实例读取，不按拟合残差重新选人。缺失点不补造。

### 2.2 初始化、体型与人体资产

源运行配置为 G20261014_smplh_motion448_v1/first_frame_sapiens_20261015/replay_selected_init/map_sigma075/run_parameters.json。实际初态读取同根 first_frame_sapiens_20261015/initialization_double_sigma075/result.npz 的单独首帧，不从上一条路线的终帧接续。

体型beta来自 G20261013_reliability448_view_v1/fit/result.npz 的全序列共享预校准值，源配置明确记作 precalibrated_shared_shape_from_full_sequence。本路线中beta固定，不随当前448帧重新估计。因此“逐帧身体递归”不能扩展解释为所有信息均只来自当前及过去帧。

资产包括 third_party/WiLoR/mano_data/models/SMPLH_male.pkl、MANO_LEFT.pkl、MANO_RIGHT.pkl，models/smpl/J_regressor_coco.npy，以及 models/VPoser02_05/V02_05。SMPL-H真实网格为6890个顶点、13776个三角面。MANO资产用于该模型的手部构造，运行中没有逐帧重新估计手姿。

两手各15个局部旋转固定为首帧值。VPoser解码21个身体局部旋转后，零起始索引9、10、11、14覆盖回首帧值，分别对应双脚、颈、头的固定分支。脚部局部旋转固定不等于整条腿不动，上游髋、膝、踝及根运动仍影响脚的位置和方向。

## 3. 身体状态、观察点与坐标

### 3.1 优化的状态

当前姿态为 q=(p,R,z)：根平移p有3维，根旋转R属于SO(3)，VPoser潜变量z有32维。姿态误差在38维切空间中表示，同时维护38维速度v、38维加速度a和114×114联合误差协方差P。

局部修正δ通过右乘回缩作用于姿态：

\[
p'=p+\delta_p,\qquad R'=R\operatorname{Exp}(\delta_\theta),\qquad z'=z+\delta_z.
\]

速度、加速度里的平移分量分别具有m/s、m/s²单位，旋转与潜变量分量有各自单位，不能把38维数组直接取范数就称为某个人体关节的米制速度。

全部身体状态仍储存在当前左相机坐标下。接触时用本次场景的地面变换计算高度；显示时再统一变换到固定地面系，不在求解后抬高人体或移动地面。

### 3.2 模型观察点与原生关节的区别

身体像素对应的是模型网格经17点COCO回归器得到的观察点，不是SMPL-H原生52关节的直接同名替换。本路线监督肩、肘、腕、髋、膝、踝共12点；鼻、眼、耳不进入该身体像素项。

从第11帧起，左右髋回归行改为冻结的虚拟观察映射：

\[
J'_L=J_L+\alpha(J_L-J_R)+b(J_{K_L}-J_L),\quad
J'_R=J_R+\alpha(J_R-J_L)+b(J_{K_R}-J_R).
\]

其中 α=0.36213313641862127，b=0.10941253220794421；第一项使观察髋向外扩，第二项使其向同侧膝方向移动。参数此前仅用1～10帧拟合并冻结，0～10帧仍用原回归器。本次448没有再次标定它们。

这一操作改变二维观察模型，不修改beta、原生关节中心、骨长或网格拓扑。髋曲线随之表示映射观察点的运动，不能据此直接判断解剖髋中心或原生骨盆的真实运动。图像坐标的髋标定留出也不是对全片预校准体型的独立验证。

## 4. 已有时序算法：真实dt的q/v/a预测

本路线已经包含递归时序预测，不是逐帧独立拟合，也没有额外套一层轨迹平滑。每帧使用实际dt：

\[
d=v\,dt+\tfrac12 a\,dt^2,\quad
q^-=q\oplus d,\quad v^-=v+a\,dt,\quad a^-=a.
\]

协方差按 P⁻=FPFᵀ+Q 传播，根旋转部分使用SO(3)右雅可比及旋转传输。白噪声jerk驱动下，每个欧氏分量的过程噪声块为：

\[
Q_j=\eta_j
\begin{bmatrix}
dt^5/20&dt^4/8&dt^3/6\\
dt^4/8&dt^3/3&dt^2/2\\
dt^3/6&dt^2/2&dt
\end{bmatrix}.
\]

平移、根旋转、潜变量的jerk密度分别为0.25、1、4；并非对所有关节直接施加同一个“加速度要小”的惩罚。预测和当前观测共同更新同一身体模型。

初始v、a为0。初始独立标准差按“平移/根旋转/潜变量”分组：姿态0.10/0.35/0.5，速度0.5/1/1.5，加速度1/2/3；初始P为其平方构成的对角矩阵，后续允许交叉相关。

## 5. 当前帧观测如何进入目标

### 5.1 身体像素门

每个视图检查有限值、原图边界、score在[0.2,1]内。两个视图都有合格点时，再检查鱼眼去畸变射线的像素化极线距离，超过35px则成对拒绝；只剩一个合格视图时可保留单目约束。至少4个身体关节在任一视图有效才进行当前帧MAP。

score在此主要用于门控，不是把score直接乘成连续的像素权重。参与优化的身体点采用像素尺度σ=8px、二维残差范数Huber阈值25px；大残差会降权，但不等于从最终全raw评价中删除。

模型投影要求左右相机中所有待投影点有限且具有大于10⁻⁴m的正深度。被拒绝观测仍保留在逐帧记录中，不以模型点或历史结果替换。

### 5.2 前掌横向方向

四个模型表面点使用顶点3216、3226、6617、6624，分别组成左右脚的两端点；与Sapiens2前掌点配对。这里观测的是大趾到小趾一侧的前掌横向图像方向，不是脚跟到脚尖的完整纵向朝向，也不直接提供脚底高度。

每个脚/视图要求两个原始端点都过门，且观测端点距离至少16px。令观测横向向量为d_obs、长度为ℓ，模型回投向量为d_pred，方向残差为：

\[
e_{\rm dir}=\ell\,\frac{d_{\rm pred}}{\|d_{\rm pred}\|}-d_{\rm obs}.
\]

它保持有向方向，去掉模型端点绝对位置和预测长度的直接约束；ℓ保留了该项的像素尺度，因此不是每个方向一律同权的纯角度损失。方向项沿用8px尺度和25px Huber；每脚每视图只计一个二维方向向量，不把两端点当两份独立方向证据。

35、47、61、267、335、412帧的前掌方向全部留出；首帧也不加入前掌更新。六帧留出仅针对新增方向观测，身体像素仍用于该帧跟踪，不能将其称作完全独立测试集。

## 6. 完整目标与25mm鞋高

### 6.1 五类目标，六个可分解能量项

在当前帧固定预测先验后，目标可写为：

\[
E=E_{\rm body}+E_{\rm dir}+E_{\rm pred}
 +E_{\rm latent}+E_{\rm sole}+E_{\rm support}.
\]

前两项是身体像素与前掌方向，第三项保持与运动预测的一致性，第四项限制VPoser偏离合理姿态区域；最后两项合称脚地接触。它们不是六份独立真实测量。

对任一身体或方向二维残差e，像素代价为 ρ₂₅(‖e‖)/8²，其中：

\[
\rho_\kappa(r)=
\begin{cases}\tfrac12r^2,&r\le\kappa,\\
\kappa(r-\tfrac12\kappa),&r>\kappa.
\end{cases}
\]

设δ相对该帧固定预测姿态，Pqq=LLᵀ，δ=Ly，则：

\[
E_{\rm pred}=\tfrac12 y^\top y
 =\tfrac12\delta^\top P_{qq}^{-1}\delta,\qquad
E_{\rm latent}=\frac{\|z\|^2}{2(0.75)^2}.
\]

运动预测项惩罚“本次姿态修正相对预测有多意外”，不是简单要求根或每个关节静止。潜变量先验惩罚VPoser坐标幅度，不直接惩罚各骨骼角度、关节速度或身体像素。

### 6.2 防下穿：足底最低允许高度20mm

公共鞋高为 realtime_app/pose_app/foot_ground_reference.py 的 SHOE_HEIGHT_M=0.025m。物理地面始终是Z=0；裸人体足底的贴地参考为Z=25mm，原有5mm容差对应允许下界20mm。

接触集合每脚含heel、ball、toe各18个顶点，去重后每脚54个，共108个。对顶点i按本次地面变换计算：

\[
h_i=[R_{G\leftarrow C,t}V_i+t_{G\leftarrow C,t}]_z,\qquad
r_i=\frac{\max(0,0.020-h_i)}{0.005\sqrt{108}},
\quad E_{\rm sole}=\tfrac12\sum_{i=1}^{108}r_i^2.
\]

它只惩罚低于20mm的足底表面点；高于下界时该项为0，不会把摆动脚拉向地面。除以√108使能量采用固定顶点集合归一化，而不是增加顶点数就自动加大接触权重。

这里的heel是足底表面候选区域，不是要求某个踝点或历史脚跟标志点恰好位于25mm。25mm是用户指定鞋高，不是通过图像重新估计的标定量。相机、助步器、Stage、物理地面不随它抬高，历史已保存结果也不按新基准回写。

### 6.3 已标注前掌贴地：目标25mm

对每只脚的18个ball顶点取真实最小高度b_f，标签c_f∈{0,1}，令 u_f=(b_f−0.025)/0.010，则：

\[
E_{\rm support}=\sum_f c_f\left(\sqrt{1+u_f^2}-1\right).
\]

实现中使用等价的稳定残差 r_f=c_fu_f√[2/(√(1+u_f²)+1)]，与防下穿残差一起进入最小二乘框架。这是相对25mm的稳健高度惩罚，10mm是尺度，不是硬约束的可行容差。

标签在求解前冻结：右前掌35～47帧、左前掌49～61帧，共26个足帧。48及其他未标注区间不加入贴地项，不根据拟合脚高自动扩展支撑，也不把摆脚强行吸附。标签来源为离线查看原图的操作性假设，不是独立足底传感器或三维触地真值。

从帧1到447均启用防下穿；首帧使用保存初态，仅按新参考评价，不平移修正。本路线没有新增防滑、摩擦、地面反作用力、动力学平衡或在线接触分类。

### 6.4 软目标的实际含义

防下穿和贴地都是软损失，像素、姿态与运动预测可与它们形成折中。求得更小总目标、达到梯度停止条件，都不保证脚高满足20mm或前掌恰好25mm。ReLU和真实最小顶点还使目标具有分段非光滑性，自动微分给出当前分支的导数，不是全局最优或硬接触可行性的证明。

## 7. 自动微分、求解与信息反馈

### 7.1 每帧只固定一次预测先验

每个当前帧先取预测q⁻和P⁻，对Pqq完整Cholesky白化，在同一个δ切空间里做最多15次IRLS/Gauss–Newton迭代。每轮重算当前模型、投影、稳健权重及接触残差，先验始终固定，不把同一张图当成多次新观测不断更新协方差。

MAP的像素、方向、接触雅可比在 AutogradContactTracker 中用 torch.autograd.functional.jacobian 计算。根回缩和鱼眼投影使用float64，VPoser及SMPL-H前向仍为float32；没有把已训练模型整体转为双精度。目标值及线搜索沿用原NumPy路径。潜变量先验与白化预测项的导数直接按其解析形式组合。

原 SMPLHMotionTracker._linearize 仍用参数步长10⁻³做点雅可比和不确定度计算；最终关节导数也仍用数值模型报告公式。因此“自动微分路线”精确指当前MAP目标的雅可比，不能写作全部几何、协方差与输出导数都已自动微分化。

### 7.2 预算与状态判定

| 参数 | 当前实际值/作用 |
|---|---|
| 最大线性化迭代数 | 15 |
| 白化梯度无穷范数停止阈值 | 0.05 |
| 步长、目标变化阈值 | 均为0.001；小步和小目标变化后仍须通过梯度检查 |
| 每次线搜索最大回退次数 | 12；步长依次减半，Armijo系数10⁻⁴ |
| 初始阻尼及失败尝试 | 10⁻⁴；每轮最多4组阻尼尝试，失败时增大10倍 |
| VPoser有效域 | 每个潜变量绝对值不超过6 |
| condition_accepted_iterate | true，见下节 |

converged表示达到本实现的梯度停止门。若最终目标与雅可比有效，且目标下降超过0.001，但梯度未达门，则保留为feasible_unconverged。名称里的“feasible”只表示这一候选被当前实现接纳，**不表示接触几何已可行**。没有达到这些条件则拒绝，并保留原因。

配置中继承的 max_translation_correction_m=0.08、max_root_correction_rad=0.12、max_latent_correction_norm=1.2 属于旧单步更新器的配置字段；当前15步MAP的_update没有按这三个数执行硬截断。不能把它们写成当前MAP实际生效的单帧位移/转角/潜变量修正上限。

### 7.3 q、v、a、P如何一起修正

接受δ后，q取当前解，v和a通过预测联合协方差中的交叉项同步修正：

\[
\Delta v=P_{vq}P_{qq}^{-1}\delta,\qquad
\Delta a=P_{aq}P_{qq}^{-1}\delta.
\]

不是先独立优化关节点位置再用相邻帧差分估计v/a。协方差在最后接受点最多更新一次，使用身体/方向像素和潜变量先验的局部IRLS信息，采用Joseph形式，并做SO(3)切空间重置与对称化。

**当前配置允许未收敛接受点也做这一次近似信息更新。** 原运行逐帧记录为70个converged和377个feasible_unconverged，447帧各有一次posterior_updates=1；首帧未运行MAP。不能写成“未收敛时协方差必然不更新”。

接触曲率没有加入这次协方差信息矩阵，尽管接触已经改变均值，并可能间接改变最终像素雅可比。因而它是工程正则化的点估计加局部像素/姿态信息近似，不是完整统计一致的接触后验；未收敛接受点的P更不能被当作校准过的不确定度。

### 7.4 失观与重启

最长允许缺少已接受身体观测的预测时间为0.10s，这里的观测支撑不是脚部触地支撑。若超时或预测潜变量越域，状态失效；恢复分支使用最近已接受姿态重置v/a和初始协方差，再尝试当前有效像素，成功才增加segment_id并重新暖机。没有足够观测则保持expired，不补帧或插值。本次448实际只有一个segment，没有触发重启。

## 8. 关节速度、加速度及其有效性

### 8.1 从模型状态换算，单位不能混用

对同一当前状态q、v、a，取ε=0.01s，经相同回缩和模型观察函数g得到：

\[
p_0=g(q),\quad
p_\pm=g\!\left(q\oplus(\pm\epsilon v+\tfrac12\epsilon^2a)\right),
\]
\[
\hat v_J=\frac{p_+-p_-}{2\epsilon},\qquad
\hat a_J=\frac{p_++p_--2p_0}{\epsilon^2}.
\]

该式沿当前状态的局部运动求导，包含模型运动学曲率的数值作用；不是直接对连续图像位置做一阶、二阶差分。原跟踪接口发布单位为mm/s、mm/s²；绘图派生数组使用m/s、m/s²。

它们是当前左相机坐标下的模型状态导数。地面速度的导数还包含相机运动和地面变换的时间导数，不能只乘一个R就称为地面系真实速度或加速度。

### 8.2 有效导数发布门

每个关节必须同时满足：模型可用、该点自身观测未超过0.10s、当前及最近良好像素拟合时效合格、相关原始观测最大回投误差不超过25px、位置最大主方向标准差不超过0.10m、段起点后已暖机0.15s。MAP未收敛时还会强制将全部12点的derivative_valid清零，将公开速度/加速度置NaN，并保留iterated_map_unconverged原因。

因此保存了有限原始估计，不等于通过了发布门。绘图为了展示完整当前结果，显示全部有限原始估计并直连相邻帧；小圆点标出通过原门的样本，不把大段无效区间伪装成有效导数。

| 关节组 | COCO索引（左/右） | 通过导数门帧数（左/右，各除以448） |
|---|---|---|
| 肩 | 5 / 6 | 11 / 9 |
| 肘 | 7 / 8 | 39 / 43 |
| 腕 | 9 / 10 | 55 / 57 |
| 髋观察点 | 11 / 12 | 15 / 8 |
| 膝 | 13 / 14 | 23 / 10 |
| 踝 | 15 / 16 | 61 / 65 |

本次总有效槽数396/5376，每槽是一个关节在一帧的三维导数状态，不再乘3或乘视图数。曲线中的原始加速度最大49.30534m/s²、速度最大1.09809m/s，均包含未收敛帧，不能将这些峰值认定为实测人体运动。

## 9. 保存结果、三维页面和逐关节图片

| 文件 | 内容与解释 |
|---|---|
| protocol.json | 路线、0～447范围、鞋高/标签、实际配置、源运行、当前场景、因果边界 |
| result.npz：predicted_coco、smplh_joints | 分别448×17×3模型观察点、448×52×3原生关节；二者含义不同 |
| result.npz：transl、root_rotation、vposer_latent、body_rotations | 448×3、448×3×3、448×32、448×21×3×3；配合冻结资产可再生真实网格 |
| result.npz：velocity、acceleration、covariance | 448×38、448×38、448×114×114，前两者是广义参数导数 |
| result.npz：pose_available、segment_id、derivative_valid | 完整帧可用/重启段信息及448×12导数发布掩码 |
| result.npz：toe_m、sole_min_m、support_ball_height_m、support_ball | 448×4×3前掌表面点，以及448×2足底最低高、ball最低高、冻结支撑标签 |
| result.npz：betas、regressor、timestamp_s、foot_ground_reference_m | 固定体型、最终回归器、真实时间、0.025m参考；第0～10帧须用原回归器重建 |
| trajectory.jsonl | 每帧求解状态、迭代历史、posterior_updates、身体/方向使用与拒绝原因、足高、接触成本和导数有效点数 |
| summary.json | 全帧/全raw槽评价及各项检查，不只统计已使用或较好样本 |
| scene_input/ | 同图重算strict、Stage/地面重放、相机/助步器场景文件及命令记录 |
| view/index.html | 本路线独立448帧真实网格页面；同帧原始双目回投、求解失败状态、绝对足高及相对25mm净高 |
| motion_curves/joint_derivatives.npz | 由同一保存q/v/a和冻结资产按第8节公式再生的12点米制导数、时间和原有效性掩码 |
| motion_curves/lower_limb_velocity_acceleration.png、.svg | 下肢左右对照：髋、膝、踝分行，速度与加速度分列 |
| motion_curves/关节曲线/ | 12张2400×1536 PNG，左右肩、肘、腕、髋、膝、踝各一张，命名左肩.png、右肩.png等；上速度幅值、下加速度幅值 |

逐关节图片覆盖的是COCO5～16身体观察点，不是SMPL-H52个原生关节和手指的全量图。COCO0～4头部点不输出。曲线为白底连续实线，不平滑、不删峰值、不跨失效帧或重启段连接，髋图注明映射点含义。

三维页面使用本运行保存姿态和同源场景，模型、观察点、严格三角化、相机及助步器统一经本次R/t和显示映射[X,−Y,Z]，物理地面Z=0，真实模型三角面保持。页面按20mm下界标记违约，不通过平移网格制造接触。原记录已检查55、267、447帧、35→36播放、视角、同帧原图和末端边界，浏览器错误0；这属于显示与数据组织验证，不是物理认证。

## 10. 完整448帧结果与失败事实

以下直接来自本实验summary及保存数组。原始身体槽含所有12点和双视图，不按优化是否使用、score或求解是否收敛筛除；点数有限和回投误差小均不是独立三维精度证明。

| 指标 | 结果 | 分母/范围 |
|---|---|---|
| 姿态可用 | 448/448 | 原始0～447帧 |
| 重启 | 0，segment总数1 | 完整序列 |
| 收敛/未收敛接受 | 70 / 377 | 447个更新帧，另有1个首帧未运行MAP |
| 身体raw回投 median / P95 / max | 19.06592 / 56.80706 / 147.06147px | 10752=448×2视图×12点 |
| 髋raw回投 median / P95 | 29.54650 / 68.41069px | 1792槽 |
| 非髋raw回投 median / P95 | 17.16505 / 50.97274px | 8960槽 |
| 35～61帧身体raw median / P95 | 19.22460 / 73.73795px | 648槽 |
| 足底最低高度median | 11.22190mm | 896=448×双脚足帧 |
| 低于鞋高容差下界20mm | 597/896足帧 | 左右脚分别计数，不是597个视频帧 |
| 低于物理参考地面0mm | 221/896足帧 | 原始值保留 |
| 全片最低足底高度 | −15.80933mm | 没有事后抬脚 |
| 支撑ball相对25mm绝对误差 median / P95 / max | 6.89131 / 23.13701 / 25.00744mm | 26个已标注支撑足帧 |
| 支撑ball在25±10mm内 | 17/26 | 未标注区间不冒称通过贴地门 |
| 留出横向方向误差 median / P95 / max | 5.99466 / 18.60202 / 24.31741° | 24=6帧×2视图×2脚 |
| 有效导数槽 | 396/5376 | 448×12关节点 |
| 主体回放与统计计时 | 1745.85794s，约29.10min | 不含前置场景重算及后续可视化 |

五项工程检查中，全448姿态可用、无重启通过；足底下界、全部标签贴地误差、全部更新收敛均失败，all_engineering_checks_passed=false。

为区分“未收敛”与“软目标允许违约”，本次只读复核得到：70个收敛帧中仍有50帧至少一只脚低于20mm；第320帧已收敛，但左右足底分别9.23109/8.91517mm。旧简写“320帧收敛仍低于20mm”应读作“第320帧”，不是320个收敛帧。不能仅靠宣称增加求解迭代就保证接触条件成立。

已保存真实初态的NumPy/Torch目标值最大差2.2737367544323206×10⁻¹³，用于发现双实现鞋高偏置不一致；它不证明各处梯度或接触正确。绘图重建时，448帧同源模型COCO点与保存值最大差0m，证明复用了同一模型/映射，不构成真实速度验证。

## 11. 实现索引与复现方式

| 实现文件 | 职责 |
|---|---|
| realtime_app/tools/run_smplh_contact_route.py | 当前单路线448入口、场景重放、帧循环、保存与全槽统计 |
| realtime_app/pose_app/smplh_motion_tracker.py | SMPL-H观察模型、q/v/a预测、像素门、失观恢复、原导数输出 |
| realtime_app/pose_app/smplh_iterated_motion_tracker.py | 固定预测先验的15步MAP、接受/收敛判断、v/a条件修正与一次P更新 |
| realtime_app/tools/probe_smplh_foot_observations.py | 同源身体/前掌读取、方向残差、留出帧与髋映射源配置 |
| realtime_app/tools/probe_smplh_soft_contact.py | 108足底点、两脚18点ball集合、标签、防下穿和贴地残差 |
| realtime_app/tools/smplh_contact_autograd.py | 与原目标路径匹配的Torch前向与MAP雅可比 |
| realtime_app/pose_app/foot_ground_reference.py | 共用25mm裸模型鞋高参考 |
| realtime_app/walker_tools/visualization/_scene_export.py | 同图strict及新Stage/地面重放，仅读历史静态重放配置 |
| realtime_app/tools/build_foot_observation_viewers.py、build_contact_experiment_viewers.py | 保存姿态再生真实网格、同源显示和原图回投 |
| realtime_app/tools/plot_smplh_contact_derivatives.py | 模型关节导数重建、总图与逐关节曲线 |

复现时先读取本实验protocol.json和源run_parameters.json；大型原图、NPZ、权重及标定均在本地，不能假设仅克隆代码即可运行。完整回放必须指定尚不存在、且位于foot_direction_20261016下的新目录；不能覆盖本次结果。例如：

~~~powershell
Set-Location D:\my_works\walker_pose_system
.\.venv-cuda\Scripts\python.exe realtime_app/tools/run_smplh_contact_route.py --output research_records/engineering_validation/G20261014_smplh_motion448_v1/hip_pose_repair_20261016/foot_direction_20261016/autograd_contact_shoe25_full448_new
~~~

上述仅为重跑入口说明，本次整理没有执行。默认输出目录已经存在，直接再运行默认命令会被拒绝；不要通过删除旧目录消除该拒绝。可视化入口同样不覆盖已有非空view。

仅重新排版现存曲线时，不用重跑模型或身体跟踪：

~~~powershell
.\.venv-cuda\Scripts\python.exe realtime_app/tools/plot_smplh_contact_derivatives.py --render-only
.\.venv-cuda\Scripts\python.exe realtime_app/tools/plot_smplh_contact_derivatives.py --render-only --per-joint
~~~

去掉render-only会从保存q/v/a和冻结模型重建报告导数，不会重新执行身体优化；默认读取本实验，换实验须显式传入run。网页在仓库根HTTP服务下的地址为：

    http://127.0.0.1:8765/research_records/engineering_validation/G20261014_smplh_motion448_v1/hip_pose_repair_20261016/foot_direction_20261016/autograd_contact_shoe25_full448_20261017/view/index.html

## 12. 当前边界与后续工作的停止条件

当前方法具有真实dt的时序状态预测和原始像素闭环，接触已进入均值优化，自动微分已用于MAP雅可比；这些是已经实现的逻辑。完整回放完成、软件接口一致、能显示连续曲线，与真实接触或三维正确是不同结论。

尚未解决的是：鞋高下界与标签贴地违约、较多未收敛更新、有效导数稀疏，以及固定体型/观察点定义/参考地面和图像之间的冲突。参考地面含Stage及踝锚依赖，场景有固定延迟、标签离线、beta全片预校准，因此不称即时端到端在线接触系统，也没有独立三维、力学或临床验证。

本次已经按用户要求完成单路线完整448，不自动增加有限差分/无接触对照、权重搜索、局部旋转路线或显示平滑。若后续修改算法，必须同时看全raw身体/方向、鞋高/贴地、收敛和有效导数；保留失败，不能用改善某一个高度统计或更光滑的图替代其他失败项。本次文档工作以实际逻辑、参数、结果和文件位置说明清楚且副本保存完成为停止条件。
