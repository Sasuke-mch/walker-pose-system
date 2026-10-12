# Flow Matching 隔离模型架构：当前实现与维护入口

本文件依据当前 `realtime_app/learning_flow_mesh_next/` 源码维护。对应的逐项实验条件、数值、失败和解释见 [MODEL_RESULTS.md](MODEL_RESULTS.md)。二者不替代 AGENTS.md、AI_PROGRESS.md 或局部实验协议。更新时间：2026-10-12。后续改变模型、输入、状态、信息编码、输出或评价接口时，应同步修订对应段落；保留旧结果的版本与条件，不能改写成新模型的效果。

## 1. 模型身份、范围与证据级别

当前研究实现为 JointFlow：在一个联合状态中表达 SMPL-H 局部姿态旋转 R、人物整体位置 T 和体型系数 β，用条件线性 Flow Matching 建模。当前尚未训练，未加载神经网络 checkpoint。测试中的随机模型只能验证接口、依赖和数学行为。正式主线仍为 PMPose 原鱼眼→严格双目→Stage/动态地面→reference25→WiLoR；本模块没有替换其调用。

必须区分四种身份：

- `learning_flow_mesh/` 的历史 144019 参数模型：曾完成合成训练；其结果是人工任务参考一致性。
- 同目录后续 H8 167955 参数隔离工程候选：具有原生手条件、共享体型与 beta 条件接口；不得将旧 baseline 的训练数值归到 H8。
- `learning_flow_mesh_next/`：本文件主要描述的 711443 参数全 SMPLH52 候选，与前两者是不同网络；尚无训练效果。
- 同目录 `so3_flow.SO3JointFlow` 是显式构造的 F0 表示替换候选；D01存储简化后默认52节点695952参数，显式retain_inactive_information=True保留原711056分配。当前默认工厂和双阶段管线仍构造上述欧氏模型。它同样未训练，不能继承任何历史训练数值。

当前 next 实现没有 RGB 特征提取器、预训练视觉模型、独立手部条件 memory、骨架 parent graph、动态地面或接触损失。观测是 COCO17；全模型输出包括52个局部旋转节点，不代表这些手部旋转已从当前观测中被独立约束或准确恢复。

2026-10-12最新状态：研究单模型候选为第13节的SO3JointFlow，默认ambient JointFlow与既有G0/J/G1仍保留。完整端点物理代价VJP已独立实现，方向对照通过，但全K32成本门失败，未接入生成器并停止该反馈晋升；接口与结果见第14节/MODEL_RESULTS第29节。第2节描述默认双阶段数据流，不能当作单模型SO3候选已启用第二生成器的证据。

## 2. 总数据流

```text
单物理帧 Observation + 上一帧 TemporalState
  → StateEncoder：34个视图/关节槽、准入池化、GRU128、物理时间
  → context[1,128]
  → G0：独立帧噪声 → 8次Euler → K32完整候选
  → 冻结G0 medoid anchor → 局部扰动解码/鱼眼投影 → J → Λ
  → G1：自身观测状态/context + 冻结Λ + 独立G1噪声
  → 8次Euler → K32完整候选 → 各候选自己的R/T/β解码
  → A0…A7对照；正式动作从候选池选择一个完整人体
  → 当前缺观测：不发布；失败/歧义/所有候选保留
```

G0和G1是两个不同模型实例，不是一次网络调用的别名；同拓扑不自动证明初始化、训练数据、优化预算或 wall-time 已匹配。当前 prepare 宣告同 core 容量，真实训练对照未执行。TwoStageInference把G0设为eval且关闭参数梯度；G1保留独立状态。不把选中的人体反馈给观测GRU。

## 3. 输入与坐标契约

`schema.Observation`只接收显式字段，无训练标签、已知根位置、strict XYZ真值或任意额外字典。

| 字段 | 单帧维度 | 含义/约束 |
|---|---|---|
| xy | 2×17×2 | LEFT/RIGHT原鱼眼像素；不得仍处在模型转正域 |
| confidence | 2×17 | 原检测置信度，非噪声协方差 |
| present / quality / geometry | 各2×17 bool | 来源可用、原质量、射线几何标志，三者分开 |
| views | 2 bool | 视图可用，admitted=present∧quality∧views |
| origins / directions | 各2×17×3 | 相机0共同坐标、米；准入方向为单位向量 |
| rejection_reasons | 2×17 | 原失败理由，不静默删除 |
| camera_ids | 2个PnP ID | LEFT/RIGHT顺序，不用Windows数字索引当身份 |
| identity / sequence / frame | 显式 | 人物/序列/原帧身份；frame非负 |
| timestamp / timestamp_kind | 秒/来源 | 物理时间；view timestamps可分别保留 |
| operator / units / pixel_domain | 固定声明 | COCO17_SURFACE_REGRESSOR_6890 / metres / raw_fisheye |

present槽必须有限；真正缺失槽可NaN，不能把present但质量拒绝的NaN悄悄归零。几何只在admitted∧geometry的槽输入。当前真实重放的时间是host read返回时间，不是曝光时间或硬件同步误差证明。

## 4. 观测与因果状态编码

源码 `state_encoder.py`：每槽15维，顺序为xy2/1000、confidence1、origin3、direction3、present1、quality1、有效geometry1、log1p(incoming_age)1、prior_seen1、dt1。D08修复后，incoming_age是上一状态age加当前dt、但尚未因本观测清零的测量间隔；prior_seen是在当前观测之前是否曾见过该槽。xy/score仅在admitted时进入；射线仅在有效geometry时进入。除以1000是固定输入尺度，非训练得到的图像归一化。

34槽经Linear15→128后加17×128关节身份与2×128视图角色参数，再经GELU、Linear128→128。身份必须在非线性之前与内容结合；旧版MLP后相加会在均值池化中丢失槽位与观测的对应。2026-10-12修复仅调整既有算子顺序，参数仍不变。只有准入槽参与均值池化，池化观测经GRUCell128→128更新hidden。没有准入槽时完全不更新hidden；年龄按dt增加，不发布人体。初始hidden/age为0，seen为False，未知age由seen明确区分。

另一个时间分支输入 `[dt, mean(age), admitted_fraction, seen_fraction]` 的log1p值，经Linear4→128、GELU、Linear128→128。context=hidden+time_context；全缺失时context可因时间变化，但观测hidden仍冻结，不能混淆这两者。

D08修复区分两个时间位置：local MLP读取重置前的incoming_age/prior_seen；返回TemporalState与上述global timing仍使用本次观测后的age/seen（准入age0、seenTrue，缺失age继续增加）。旧实现在清零/置True后才构造local特征，所有实际池化槽位的age/seen恒为0/True。新顺序不增加参数，不改变age/seen或GRU的更新方程，但hidden/context数值因收到新信息而合法改变。旧版本已保存的hidden/模型输出不作为新版本结果使用；已有历史成本不改写成新输入语义性能。

TemporalState保存identity、sequence、camera_ids、hidden128、age2×17、seen2×17、last_timestamp和last_frame；身份/序列/rig改变必须显式reset。时间和frame严格递增。序列化/分块carry保存真实状态，detach用于截断梯度而非清空状态。模块不处理未来帧，不用最终选中候选更新这段memory。

## 5. 联合状态、token与速度网络

全SMPLH52输出维度 D=52×6+3+10=325，决策局部切空间维度 M=52×3+3+10=169。BODY22部分模型D=145、M=79，不能作为完整手部/SMPL-H网格结果发布。

6D旋转为局部旋转矩阵前两列打包；线性流在欧氏6D参数空间进行，不是SO(3)上的Riemannian flow。最终经guarded 6D→matrix转换检查；退化/非有限候选保持拒绝状态，不能凭旋转转换给出真实姿态准确性。

一个样本包含52个pose token、1个position token、1个shape token，共54个token。pose Linear6→128；T Linear3→128；β Linear10→128。加可学习54×128 token IDs、同一context和流时间embedding。流时间s的特征为[s,sin(2πs),cos(2πs)]，Linear3→128→GELU→Linear128→128；一个联合样本所有状态块共用s。

4个TransformerEncoderLayer：width128、heads4、每头32维、FFN256、GELU、dropout0、batch_first=True、norm_first=True。attention在同一帧人体状态token之间进行；跨时间来自前述GRU，不是双向时间attention。pose输出Linear128→6共享于52token；position输出128→3；shape输出128→10；拼回325维速度。

## 6. 参数与容量账本

下表为本轮构造随机模块并计数的结果；没有前向采样、优化或神经权重加载。两个信息编码模式参数数相同；固定probes是非训练buffer。

| 模块 | 学习参数数 |
|---|---:|
| StateEncoder | 137216 |
| pose_in | 896 |
| position_in | 512 |
| shape_in | 1408 |
| token_ids | 6912 |
| flow_time | 17024 |
| info_pose | 1152 |
| info_position | 1152 |
| info_shape | 12800 |
| 4 velocity Transformer blocks | 529920 |
| pose_out | 774 |
| position_out | 387 |
| shape_out | 1290 |
| 全52节点总数 | 711443 |

BODY22部分模型707603。两个G0/G1全模型若同时保存，共1422886学习参数，不能只报单模型容量。每帧16次velocity batch调用，G0/G1各K32×8，共512候选状态求值；解码/J成本另计，不因为batch调用16就称16个候选求值。

## 7. G0局部观测信息与G1条件

`geometry_information.py`选择数值合法G0候选的A4 medoid作为anchor，冻结其R/T/β。局部坐标是右乘局部旋转Exp扰动、位置和beta扰动，使用明确TangentScale；旋转每块要求等方尺度。真实surface decoder输出6890顶点，再由COCO17回归器投影至原鱼眼像素。

J为原admitted像素相对于标准化局部169维坐标的中心差分，完整34点时68×169。默认精度审计float64、step1e-4，时序性能旧检查float32 step1e-3；一个anchor加正负169扰动为339个解码状态，按chunk批处理。Λ=JᵀWJ，W是每view/joint的2×2precision块对角；不包含跨视图/时间误差相关性。当前sigma10px是proxy，calibrated=False，不由confidence直接推成协方差。

默认 `diagonal_blocks`：每pose和T读取对应3×3对角块展开9维；β读取10×10展开100维；signed_log(x)=sign(x)log(1+abs(x))，经无bias Linear9/100→128加到token。信息detach，G1不反传至anchor/J；同一帧Λ被全部候选共享。默认表示不能传递跨块耦合，不能将其称为完整联合观测信息。

可选 `cross_block_sketch` 保持9/100输入宽度，以固定Rademacher probes投影非对角行，再signed-log叠加。3列pose/T probe和10列β probe种子固定，不消费模型初始化RNG，float64全模型17576bytes buffer。该模式已因固定anchor的重建可见性筛选暂停晋升，仅保留数值对照，不是默认。没有无损或JL保证，不能称已改善预测。

Sβ=Λββ−Λβu pinv(Λuu)Λuβ只作为局部诊断；Gram会平方J条件数，截断阈值可产生虚假残余shape信息，需直接whitened-J SVD和阈值审计。局部rank不证明共享beta可辨识或真实准确。

冻结信息保存当前观测、mask/ray/时钟、实际K/D/R/T/图像尺寸/PnP角色及噪声precision/source/calibrated快照。同frame但实际输入改变必须回退。新缓存记录当前观测快照；旧缓存明示legacy_unverified。不认证全历史、decoder资产内容或源数据独立性。

## 8. 训练目标与未执行配置

标准化目标x来自完整标签R/T/β；Normalization仅允许train真实声明统计，工程fixture单位常数不是训练配置。ε~N(0,I)，s~Uniform[0,1]，z_s=(1−s)ε+sx，目标速度x−ε。L=(mean_pose MSE + mean_T MSE + mean_beta MSE)/3，各块按自身维数平均，避免312维pose自动压过3维T与10维β。

2026-10-13 D10.1修复Normalization统计合同：fit仍用原mean/std和1e-4 floor，但结果detach；序列化保留dtype/split/source_kind。训练来源声明只能是train+fitted_reference/independent_reference，工程常量明确engineering_fixture+analytic_fixture，旧四字段记录保持legacy_unverified、原float32默认，不能由source字串自动认证训练来源。require_training_statistics检查声明，不认证独立数据或划分。float32/64回读值逐值保持；mean/scale必须冻结且同dtype，推理入口和回归loss明确validate，SO3 codec绑定来源字段，候选provenance保存这些字段。数据统计真实性仍待真实loader与注册数据证据。

同capacity regression使用同速度网络、输入状态0和s=1，经block平均MSE。2026-10-12已补齐可选information接口：regression、regression_loss与train.objective透传同一冻结Λ，复用原信息投影和全部参数；默认不传Λ与旧输出逐值一致。损失接口已支持信息条件回归；2026-10-12进一步补齐candidate_store.infer的Λ透传，仍限K1且保留拒绝/状态/资源记录。完整训练评估runner未实现，尚未训练；如比较G1，仍须匹配G0生成anchor/J的完整成本和条件来源，不能只匹配单网络参数。CFM没有默认endpoint MSE、best-of-K loss或独立网格损失。

预置AdamW、lr3e-4、weight_decay0.01、clip1、warmup1000、cosine、effective_batch16、unroll32、smoke2000/main50000、validation每1000、seeds7/17/27。`train.py`只有configuration/prepare/objective，无真实数据loader/optimizer循环/训练checkpoint生命周期；以上均未执行。用户未授权训练、motion labels未取得，不能凭配置声称可学习性或完整复现。

## 9. 候选采样、解码与发布

默认K32、Euler8、无TTA；z←z+v(z,s,context,Λ)/8，起点标准高斯，终点反归一化。帧噪声由identity/sequence/frame/stream/candidate index产生稳定seed；K1/8/32/64共享前缀、帧之间独立，G0/G1 stream分开。无拒绝后补采样、无旧结果混合。

CandidateStore保存raw、全部R/T/β、numerical/actions、原reasons、ids/seeds、topology、provenance/resources，缓存独占创建不覆盖。当前actions=numerical∧current_observed；预定的额外观测一致性校准动作门尚未实现/校准。投影不可用可导致几何成本inf，但不能冒充已完成独立动作quality gate。

SurfaceDecoder使用已有male SMPL-H/MANO统计资产；每候选自己的β/T解码，不平均β、不拼不同候选关节。physically静态体型的当前估计允许随证据变化，但未实现独立稳态beta认证，也不把EMA beta缝接到另一pose。

## 10. 全候选决策与时间规则

| 方法 | 当前实现 | 可输出完整人体 |
|---|---|---|
| A0 | 原第一个候选；不是第一个可用候选替代 | 原候选可作为action时是 |
| A1 | absolute COCO17坐标均值 | 否，诊断 |
| A2 | 最低原鱼眼观测成本 | 是 |
| A3 | 逐关节JPMA/top-k RPEA工程改写 | 否，诊断 |
| A4 | 完整R/T/β距离的medoid，均值距离 | 是 |
| A5 | 等方二次风险+相同几何成本，均值平方距离 | 是 |
| A6 | B二次风险+几何成本 | 是 |
| A7 | A6近等价动作内的因果时间选择 | 是 |

B=λ0(Λ+λ0I)⁻¹，是决策metric而非后验协方差。δ_kl使用两候选最短SO3对数，运输到anchor切空间，并含标准化T/β差；期望l涵盖所有数值合法原候选，不能只对动作子集重归一化后隐去其余状态。Rk=mean_l δᵀBδ+λg Eobs(k)。Eobs为准入点原鱼眼白化Huber成本，threshold3；没有GT-root/DA3/calibrated posterior weights。

2026-10-12修正该Huber成本的零残差反向边界：平方残差在二次区间时，未启用的线性分支使用安全平方根输入，避免sqrt(0)的无限导数污染where反向。前向Huber定义、原始投影/准入和决策规则不变；精确匹配点的代价与一阶导数均为零。此修复只保证相关数值边界，不代表物理反馈架构已经接入。

A7只在Rk≤minR+validation delta的近等价集合中最小化相对上次发布完整人体的距离/dt²；gap>0.5s关时间偏好。delta0对照保持A6，不能凭这一无变化检查证明时间提升。近πchart、失效信息或实际绑定失配明确回A4，保留歧义/理由，不静默平滑或回写历史。低risk gap非真实物理歧义概率，当前ambiguity未校准。

## 11. 离线评价层与尚缺资源

`evaluation_contract.py`按train原参考分别标准化own-T COCO17和同basis beta10，ES采用固定K V-statistic；raw/legal/action分开，coverage90与width并报。未知参考块unavailable，非法raw池不拿legal分数冒充。`acceptance.py`根据每block三seed配对记录推导收益与CI，不接受直接填true的认证。

`evaluation_population.py`核登记人物/原动作/序列划分、原frame、block分区、source/rig/stratum与train统计origin。`evaluation_blocks.py`从每原帧两方法报告生成完整block均值；任意缺失失败阻断整体配对evidence，原报告仍保存。登记元数据不认证报告对应真实tensor/files、许可、独立真值、power、实际wall-time或外部test锁。

真实motion manifest records=0；独立原鱼眼像素reference records=0；独立3D与shape scan缺失。AMASS同族许可/sex/beta/hand convention待取得核验；不允许SMPL-X beta直接拷入SMPL-H。360重放观测不是训练真值或heldout独立测试。剩余方法收益、三训练seed真实结果、真实校准与实时要求都未完成。

## 12. 文件对应与后续维护

2026-10-12移植审查：作者guidance的端点梯度与经过速度网络回到flow state的梯度并不相同。实际鱼眼project反例中完整GN步虽令线性残差为0，真实残差却增加；合法投影也不能替代下降检验。两个合法旋转的欧氏6D均值可能退化。当前没有接入端点修正/HQS或替换G0/G1，详见MODEL_RESULTS第12节；不可把论文的线性Gaussian理论直接称本模块非线性后验保证。下一阶段先审查冻结局部J计算的自动微分替代及完整成本，再决定是否改变生成器。

直接架构：schema.py、state_encoder.py、joint_flow.py、information_encoding.py。候选与几何：candidate_store.py、body.py、geometry_observation.py、geometry_information.py、tangent.py。双生成与决策：pipeline.py、aggregate.py、information_risk.py。标签/评价：data_firewall.py、train.py、evaluate.py、evaluation_contract.py、evaluation_population.py、evaluation_blocks.py、acceptance.py。

结果描述必须记录：输入范围和来源、model身份/参数/是否训练、seed与候选预算、唯一变量与对照、所有适用指标单位、失败/不可用、统计单位、测时边界、是否重跑，以及结论所能与不能支持的内容。变更后只重跑相关检查，不能用重复绿测试替代真实效果。下一轮在新数据、授权或复现缺陷出现前，不主动扩容或新开无法验收的架构变体。


2026-10-12局部J审计补充：实际SMPL-H两个人工锚点的向量化反向微分与FD导数误差约1e-9，但额外显存约3.4倍、跨运行耗时明显不稳定，且raw AD缺338邻域状态拒绝检查。默认仍为FD；未证实guard等价及完整成本收益，不晋升替换。人工rig实际D=[.03,-.01,.001,0]，v2/v3元数据误写D0的更正及全部测时见MODEL_RESULTS第13节。


2026-10-12邻域门审查：实际local_information的解析边界反例确认“锚点投影合法+AD有限”不等于原FD邻域合法；MODEL_RESULTS第14节保存原拒绝。关闭anchor-only直接替换捷径，暂停局部AD性能替换，不新增边界启发式；后续转回单生成器条件机制的简约性审查。


2026-10-12输入公平性补充：G0的Observation条件不含外部NoiseModel.precision，G1经Λ间接接收；已有码origins/directions不能忽略，也不能等同完整K/D和噪声条件。变化noise的单/双阶段对照必须登记相同可用信息；相同Λ控制需计入anchor/J成本。实际sigma1/2审计见MODEL_RESULTS第16节，未新增条件模块。


2026-10-12 Flower审查不接入新模块：其无条件先验、逐步线性求解与图像样本平均不能直接替代当前观测条件JointFlow及完整人体决策。官方源码范围、默认调用成本和CG停止表达式反例见MODEL_RESULTS第17节。后续文献检索转向直接人体/第一视角条件flow与几何表示。


2026-10-12质量路径候选未接入：StableHand训练/推理起点及发布版时间日程需区分。暂保留同观测起点的条件桥作为简约候选，但COCO17不是完整325D状态，未知beta/手姿不得伪装已知或额外堆proposal网络。解析证据和资源门见MODEL_RESULTS第18节。


2026-10-12起点资源审查修正：部分观测可仅作用于同空间分量，其余q0保留噪声，不要求全325D测量。但硬q1会损失条件源随机性，且WiLoR带来额外输入条件；质量桥尚不接入，继续使用高斯源。详见MODEL_RESULTS第19节。


2026-10-12消融前提更新：用户确认暂无额外数据，训练与噪声参考records仍0；四法Flow/回归×无Λ/有Λ及名义容量/完整成本约束见flow_research_plan_v1.json追加消融和MODEL_RESULTS第20节。仅修复信息回归infer透传，不据接口通过提升模型效果。


2026-10-12骨架流形移植门：同beta两合法姿态，51内部FK边长度变化约数值零，8条COCO表面边最大变化6.4614mm。因此不额外施加COCO骨长恒定，不用骨架一致性宣称SO3概率流。ManiPose作者代码/范围与实际解码反例见MODEL_RESULTS第21节。


2026-10-12SO3概率路径门：当前shortest_log用于局部风险，近π整张量拒绝，单位旋转端点反向Jacobian非有限；冻源/目标的时间JVP仍能有限。不能直接作为全局可微log移植，保留原决策回退；RFM仅为表示替换候选，源测度/速度metric/积分与成本须独立核验。详见MODEL_RESULTS第22节。

2026-10-12专用SO3实现门：FrameFlow legacy 的近π log分支在所检验的固定端点上未满足端点一致性，停止直接移植；其冻端点时间JVP检查实际通过，不能据端点误差泛称时间导数失败。训练的目标中心IGSO3与推理Haar源、线性/指数日程须分别登记。独立SciPy冻端点log加matrix_exp只作为无需拒绝/重采样的训练路径控制通过；没有新增神经网络或改变当前325D模型。下一步只考虑在同条件单模型对照中替换旋转表示与积分，保留相同观测/预算和强回归；不复制蛋白质全局中心化、IPA或其它模块。详见MODEL_RESULTS第23节。

## 13. SO3JointFlow：单模型旋转表示替换候选

2026-10-12按已有长期协议先登记该因素，再实现 `so3_flow.py`。构造时必须传入现有Normalization和ModelConfig；StateEncoder、54token、128width/4heads/4blocks、T/beta输入输出及观测信息保持现有设计。仅pose_out从128→6改为128→3。角速度只表达方向变化，旋转长度和正交关系由群更新维持。旧单模型F0和SO3候选是两个表示对照，未将当前G0/G1/J删除。

数学状态为SO3^N×R3×R10，内部网络仍读取旋转矩阵前两列的6维嵌入，T/beta为原标准化量。速度为3N+13维：52节点169，22节点79。每步右更新 R_next=R·Exp(dt·skew(omega))，T/beta保留Euler。实际携带完整3×3矩阵；默认旋转指数核经第26节检查后使用Rodrigues，显式rotation_kernel="matrix_exp"保留原核对照。速度维度减少本身不代表内存或运行成本降低。

训练和采样均从同一Haar SO3源开始。现有每候选seed及6N+13 Gaussian噪声接口复用，每节点只取前4个独立Gaussian并归一化为wxyz四元数；另外2个分量不进入动力学。四元数到矩阵不取近π拒绝门、不重抽；外部零/非有限源显式报错并给出索引。训练路径使用冻结端点SciPy principal log及R0·Exp(t·omega)，训练路径/sample/regression复用Rodrigues闭式核，线性日程与推理一致。CPU目标准备需要传输与求log，不能作为端点可微guidance接口或训练加速结论。

CFM三块等权：固定1rad单位下每角速度坐标MSE（SO3 canonical metric的常数缩放）、标准化T3和beta10 MSE。确定性对照使用同一SO3网络，一次前向预测相对I的角向量并Exp成合法旋转，再调用现有regression_loss；其标准化6D损失与角速度CFM损失不用于横比效果。两种目标参数完全相同。第24–29节检查时的原分配版本52候选711056参数、22候选707216，均比对应ambient模型少387，其中15104个信息投影参数分配但不激活。D01进一步移除这些不可达分配，默认52/22为695952/692112，旧模式显式保留；活跃计算与梯度逐值相同，任何Lambda输入仍拒绝。

公开输出按绑定Normalization导出完整6D/T/beta，沿用CandidateStore的完整候选、原掩码、缺失观测禁止发布、来源和seed。objective与infer增加显式codec检查，防止另一套均值/尺度把合法列错误解释；资源标记LieEuler和表示身份。现有prepare/config、训练开关、作者代码及正式主线未修改。相关20项测试与改动文件Ruff通过；无训练、权重加载、真实网格效果或速度证据，详见MODEL_RESULTS第24节。

2026-10-12固定预算成本已测，见MODEL_RESULTS第25节：同K32/8调用、真实SMPL-H解码及A4决策，SO3matrix_exp完整p50为44.87ms，对照35.20ms，1.2747倍未过预注册1.25门。少387参数和169维速度未给当前实现提速；候选继续隔离，不晋升。后续最多一次有数学依据的指数核替换，不改变模型、源或日程。结果属于固定人工观测/随机网络成本，非正式系统速度或精度。

2026-10-12第二轮核替换已完成，见MODEL_RESULTS第26节。Rodrigues用K²=wwᵀ−||w||²I及half-angle sinc省去通用矩阵指数；θ²<1e-4用四阶A/B Taylor、regular分支先clamp再sqrt，避免零角 inactive 分支的除零/导数非有限。网络参数、噪声源、日程和步数不变。17项相关测试通过；同次全链p50闭式38.33ms、旧核43.63ms、ambient33.23ms，1.1536倍通过成本门，仍慢于ambient，显存增量未下降。保留显式构造的研究候选；两轮已耗尽，结束核优化，未将其晋升默认、宣称精度或改变G0/G1/J。

2026-10-12统一机制阅读见MODEL_RESULTS第27节：FMPE/条件transport支持当前F0的离线joint配对、在线无G0/J采样，但该通用机制已有先例且数据/噪声充分性未满足。SBI-SIM另加控制网与逐步物理反馈，当前不叠加；当时提出单次晚阶段代价/角向VJP替换anchor信息输入的假设，现已完成独立算子而成本晋升失败，见第14节。A6不等同后验校准/密度/MH，当前无log_prob或真实prior/likelihood；不根据论文的density/覆盖论证升级证据。训练和默认架构继续保持。

## 14. 独立端点代价余切量：已实现，未接入神经网络

`geometry_information.endpoint_cost_vjp` 接收当前完整CandidateStore、严格绑定的原鱼眼Observation、Camera、NoiseModel、TangentScale及独立逐行surface callable。只允许SMPLH52，K个原槽位输出gradient[K,169]、available[K]、costs[K]、原ID/原因、绑定快照和实际解码/reverse计数；不改actions、不选anchor、不建立J或Λ。缺失/过期绑定与无当前观测在解码前不可用；数值非法计算行先排除，全部原输出槽位保留。准入投影失败、非有限代价/梯度及批异常明确记录，unavailable梯度NaN，合法零梯度available。

对每个冻结端点(Rhat,That,betahat)，只对零标准化delta169求导：R(delta)=Rhat Exp(S_pose delta_pose)，T(delta)=That+S_T delta_T，beta(delta)=betahat+S_beta delta_beta。复用Rodrigues右乘核，不用可微log；端点、观测、相机、precision和尺度全部detach。E是当前准入点mean Huber原鱼眼代价，对候选E求和一次reverse，使每行返回各自gradient，不被K缩小。二次区g=JᵀWr/N_admitted；全局真实LBS逐样本独立是批求和成立前提。

该输出是标准化坐标下的代价梯度/余切量。将它变为更新需要明确metric/步长/日程；本阶段没有物理修正、control网络或velocity条件接口。它不能保留Λ的rank/Schur/全耦合，也没有验证FD邻域，不能作为旧information的语义等价替换。

晚阶段s=7/8沿base velocity预测到s=1，正好等于原8步LieEuler最后一步端点，不额外调用速度网络。成本可行性因此在原端点上测旁路；VJP解码仍不能冒充任何未来修正后人体的最终解码。一次真实CPUfloat64对照relative_error=3.9459897e-11、pose/T/beta均有限；同次CUDAfloat32 K32/8完整p50无VJP45.61805ms、有VJP81.30155ms，ratio1.7822233>1.25，停止全K32晚反馈晋升。后者实际网络8batch/256状态、人体2batch/64状态、reverse1；前者人体1batch/32状态、reverse0。详细记录和边界见MODEL_RESULTS第29节与flow_next_endpoint_vjp_cost_v1.json。训练、神经checkpoint和正式主线继续关闭/不变。

## 15. 当前研究方案架构图与阅读顺序

按用户要求生成第24–29节原SO3分配版本对应的中文论文风格图：[可编辑SVG](figures/jointflow_so3_current_20261012.svg)、[矢量PDF](figures/jointflow_so3_current_20261012.pdf)。白底黑线，中文字与数学符号已实际查看PNG并修正；不使用历史H8图替代next，也不添加当前不存在的RGB、手部memory、运动树图、contact或控制网络。该图保留D01/D08之前711056参数及旧时间输入语义版本；主模块连接不变，当前默认容量以第1/13/16节、当前age/seen语义以第4节为准，不覆盖历史图或成本。

![当前SO3JointFlow架构](figures/jointflow_so3_current_20261012.png)

图a为原鱼眼34槽位特征的准入池化/观测GRU/时间条件；图b为Haar源、54tokens联合速度场与8步右乘LieEuler；图c保留每条候选完整R/T/beta、真实SMPL-H解码及A4整候选选择。图d明确为独立端点VJP算子，未回连生成器且成本门失败；图e表示既有ambient G0/J/G1对照仍保留，不代表单模型SO3内部包含两个生成器。图中参数与模块来自当前源码，不代表已训练、已验证人体精度或正式主线切换。

当前研究计算链为Observation/历史观测状态→context128→条件SO3JointFlow→K32完整候选→真实解码与整候选决策。旋转读取6D嵌入、实际携带3×3矩阵，输出3D角速度；T/beta沿标准化接口，公开状态325维、局部方向169维。物理人体beta应跨时间不变，但next未实现H8的prefix冻结；当前只是联合生成每帧的beta估计。仅COCO17观测也不直接提供独立手部/shape标签。下一项研究应先证明这种条件分布与整候选决策的任务价值，而不因图形完整或SO3合法就认定创新/有效。

## 16. 长期不足台账与逐项改进协议（2026-10-12）

用户新授权：分析现有模型不足、逐项改进、详细记载，作为当前聊天的持续目标。使用continuous_thread_goal，不恢复定时任务。训练/神经checkpoint、正式主线及独立效能门继续沿用既有授权；“逐项改进”不授权以新增模块代替证据，也不将每一项待验证设计都当作已证实缺陷。

证据等级：源码确认=当前确实存在的接口/实现边界；实测失败=已按协议发生的失败；设计假设=存在合理反例或风险但未证明降低任务效果；资源限制=需要独立标签/训练/部署条件才能解决。排序按明确风险、可判别性和现有资源，先实现可证实的小改进，再决定哪些研究假设值得投入。每项更新保留输入/代码/唯一变量/对照/成功与停止标准/原失败/实际结论，结果统一追加MODEL_RESULTS、协议和AI_PROGRESS；不新建审计/验收文档。

|ID/顺序|不足及当前证据|具体改进或判别动作|成功标准与停止条件|当前状态|
|---|---|---|---|---|
|D01/P0|SO3 F0拒绝所有Λ，却继承15104个info投影权重和固定probes；so3_flow.py/joint_flow.py，源码确认。|移除SO3默认常驻的不可达层/buffers，保留显式legacy分配对照；不动ambient G0/G1。保留构造RNG顺序，对比全部活跃参数、完整候选/状态、CFM/回归值及梯度。|52参数695952、22参数692112；全部活跃计算逐值相同、拒绝语义不变。若需广泛重构或隐式checkpoint迁移则停；只证明存储简化，不测时追分。|已完成存储简化；25相关测试/Ruff通过，MODEL_RESULTS第30节。|
|D02/P1|34槽经均值池化/GRU到context128，人体tokens接共同条件；实际局部关系受损未证。第32节窄102D像素/score输入三seed局部满列秩，初始化敏感性不均。|区分输入自由度、全局非单射与训练后任务充分性；保留原结构，不从34×15原特征槽数推导必须扩容。身份路由仅为数学控制，未移植。|当前结论排除“均值池化/128维本身已经证明该输入范围丢信息”的理由；不认证全局/实际ray/history信息充分，真实对照仍需数据。|维数直接扩容停止；原结构保留，效能问题待独立训练/数据。|
|D03/P1|next逐帧联合估计beta，没有序列静态shape状态；物理静态体型与估计随证据变化不同，源码确认，实际劣化未证。|审查联合序列状态/条件读出是否能表达共享shape；与逐帧及H8前缀冻结作因果、完整候选的公平对照，不能把旧beta拼入新pose。|不泄漏未来、不回写、不通过EMA隐藏失败；必须独立shape/姿态效能支持收益。无标签先做合同推导，不能以beta平滑认定改进。|待机制分析与数据。|
|D04/P1|COCO17输入、52旋转输出，但无独立手部/shape监督和验证；源码及空manifest确认。|逐项声明可观察输入与标签范围；BODY22只能作部分输出对照。额外WiLoR/RGB是信息增加，必须另登记匹配输入，不自动叠加。|取得同family/basis/手均值的真实完整标签并独立评价；不以合法手旋转或表面投影证明真实手姿。数据未获则效果项未完成。|资源限制，暂无新数据。|
|D05/P1|F0不接收NoiseModel.precision，G1通过Λ间接接收；已有第16节noise控制，输入公平性边界确认。|固定noise时保留；变化noise时先明确共同可用信息与接口，再做噪声因素/架构因素分离对照。confidence不得当校准协方差。|信息来源、容量和完整获取成本均匹配；无独立噪声校准数据不晋升概率/置信度结论。|待输入协议与校准资源。|
|D06/P2|默认Λ对角块丢交叉耦合，固定sketch非单射且原晋升已停；实测反例/第7节确认。|优先问任务是否需要Λ及其耦合，而非不断扩展编码；继承Flow/回归×有/无Λ对照和全部anchor/J成本。|按原held-out门证明收益，不能用sketch重建可见性代替任务效能；不恢复已停跨块扩展。|保留对照，未证明信息必要性。|
|D07/P2|A4 medoid表达候选群体代表性，不保证最接近真值；算法语义确认，真实误选率未知。|同缓存对比最强可部署决策、保留全槽位与风险覆盖；检查多模态及候选偏置，禁止oracle根位置/逐关节拼接。|原选择误差至少5%配对改善及保留率门；无独立真值不选赢家，不新加learned ranker。|待独立效能。|
|D08/P1|原local特征在准入age清零/seen置True后构造，实际池化槽的这两维恒定；6组context精确碰撞及真实时间顺序检查确认。更广的连续时间/运动建模仍未证。|local MLP改用incoming_age/prior_seen；输出状态与global time沿原重置规则。无新参数，不添加hidden衰减或人体结果反馈。|重现间隔/首次见到差异，并保留缺失/未来/分块合同；context响应不等于真实姿态收益。|本地特征缺陷已修复，MODEL_RESULTS第31节；实际缺失/采样节奏效能仍待独立验证。|
|D09/P0|SO3生成仍慢于ambient；全K32 VJP比1.7822超过1.25；第26/29节实测。|保留SO3表示候选，停止现全K32反馈晋升；若有全新有据替代问题先预注册。原核两轮额度不重置。|必须在原K/NFE和全成本口径内达门；不能减K/步数或反复测时制造通过。方向通过不算精度或速度收益。|原路径停止，独立VJP保留。|
|D10/P1|train.py只有prepare/objective，无真实loader/optimizer/checkpoint生命周期；原Normalization.fit保留grad并回读丢dtype/来源，实测确认。|D10.1已固定统计计算图及dtype/来源序列化；接续补因果序列context→匹配目标loss。实际loader/optimizer/checkpoint仍需逐步完成且训练默认禁止。|真数据来源、train-only统计、同条件回归、验证选点与test锁均落实；声明不认证真实性，缺资源不伪造已训练成果。|D10.1完成，41原相关项+2新增入口项/Ruff通过；完整runner仍未实现/数据门关闭。|
|D11/P0|训练、独立noise/3D/shape记录为空，未证实生成/shape/选择/校准改善；manifest确认。|取得独立资源后按原energy score、选择误差、coverage-width、无效率和三seed参与者配对协议评价。现有拟合结果不自证准确。|继承原5%改善、CI/coverage/失败门；数据缺口不能靠工程测试补齐，长期目标保持未完成。|资源限制。|
|D12/P2|当前无后验密度/可信likelihood/校准；全链采集+检测实时与跨rig外推也未认证；源码/既有成本范围确认。|校准/概率与部署分别在相应数据/硬件/完整输入边界评价；先明确采样分布、失败率和时间口径。|不把等权候选、B矩阵、合法旋转或纯网络耗时叫后验/30fps；无证据保留明确未认证状态，不叠加density/MH模块。|待独立概率/部署验证。|

D01的竞争解释为：不可达分配可无损删除；存在隐藏依赖；输出差异其实来自构造RNG偏移。唯一变量为SO3 inactive信息层/buffer分配。保留原super构造和姿态头初始化顺序，显式retain_inactive_information=True作为旧分配模式；默认删除分配，不改变函数、观测、Normalization、Haar源、8步或完整候选接口。严格state_dict不自动忽略旧层，旧已保存结果继续按原容量解释。当前所有下一项均为待判别/资源状态，不能提前写成改进完成。

D01实际通过：默认SO3删除info_pose/info_position/info_shape，并将非持久pose/shape probes置None。原ambient构造不变，profile共同参数复制只取两个模型实际共有的keys。构造仍先创建旧层再移除，保持活跃参数初始化逐值相同，故不宣称减少初始化峰值；原分配模式用于历史比较，不是另一个生成器。float32三组权重常驻/serialized张量减少60416bytes，固定probes移除13M个元素（原float64时52节点17576bytes、22节点8216bytes；随to(dtype)改变原buffer字节）。活跃容量不变，无新性能测量；旧strict state_dict在新默认模式显式报unexpected keys，未加入隐式迁移或加载checkpoint。

D08先在现有协议登记时间特征碰撞控制，CPUfloat64 seeds7/17/27、同incoming hidden/观测/时钟，仅一个准入槽的age或prior_seen不同。旧版六组context差严格0，实际0→缺失0.2→恢复0.9秒序列的local输入也错把第一次seen设True。修复后六组context均可区分（测试阈值1e-8），重置后的age/seen/身份/时钟相同，hidden按新观测特征变化；首次age0/prior_seenFalse、恢复gap0.9/prior_seenTrue正确。27相关项先通过；6项因错误要求hidden相同而失败，修正测试约束后仅该6项通过，两文件Ruff通过。共有33相关检查分次覆盖，不重复此前27项。无训练、checkpoint、测时或真实缺失姿态收益结论。

D02限定范围的结果见MODEL_RESULTS第32节/flow_next_encoder_dimension_v1.json。当前帧34个像素对和score共有102个自由连续坐标；掩码/身份是离散条件，射线/时钟在该检查固定，不能把34×15全当独立实数。身份路由到34×3=102坐标并给score加1，mean后可由非零score块恢复准入集合及N，再乘N恢复值；四个例包括零score与全缺失。该控制未加入实际MLP。实际修复后context的128×102 Jacobian在三随机seed和四个阈值均秩102，但smin/smax约6.3–6.6e-7；这些是局部/初始化信息，不证明全局可逆、实际足够或trained误差。保留现有mean/GRU，停止仅凭维数增加cross-attention/RGB/宽度；下一步进入D10真实训练准备接口的资源与因果合同检查，优化与权重加载仍关闭。
