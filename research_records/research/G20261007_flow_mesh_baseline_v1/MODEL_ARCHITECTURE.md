# Flow Matching 隔离模型架构：当前实现与维护入口

本文件依据当前 `realtime_app/learning_flow_mesh_next/` 源码维护。对应的逐项实验条件、数值、失败和解释见 [MODEL_RESULTS.md](MODEL_RESULTS.md)。二者不替代 AGENTS.md、AI_PROGRESS.md 或局部实验协议。更新时间：2026-10-12。后续改变模型、输入、状态、信息编码、输出或评价接口时，应同步修订对应段落；保留旧结果的版本与条件，不能改写成新模型的效果。

## 1. 模型身份、范围与证据级别

当前研究实现为 JointFlow：在一个联合状态中表达 SMPL-H 局部姿态旋转 R、人物整体位置 T 和体型系数 β，用条件线性 Flow Matching 建模。当前尚未训练，未加载神经网络 checkpoint。测试中的随机模型只能验证接口、依赖和数学行为。正式主线仍为 PMPose 原鱼眼→严格双目→Stage/动态地面→reference25→WiLoR；本模块没有替换其调用。

必须区分三种身份：

- `learning_flow_mesh/` 的历史 144019 参数模型：曾完成合成训练；其结果是人工任务参考一致性。
- 同目录后续 H8 167955 参数隔离工程候选：具有原生手条件、共享体型与 beta 条件接口；不得将旧 baseline 的训练数值归到 H8。
- `learning_flow_mesh_next/`：本文件主要描述的 711443 参数全 SMPLH52 候选，与前两者是不同网络；尚无训练效果。

当前 next 实现没有 RGB 特征提取器、预训练视觉模型、独立手部条件 memory、骨架 parent graph、动态地面或接触损失。观测是 COCO17；全模型输出包括52个局部旋转节点，不代表这些手部旋转已从当前观测中被独立约束或准确恢复。

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

源码 `state_encoder.py`：每槽15维，顺序为xy2/1000、confidence1、origin3、direction3、present1、quality1、有效geometry1、log1p(age)1、seen1、dt1。xy/score仅在admitted时进入；射线仅在有效geometry时进入。除以1000是固定输入尺度，非训练得到的图像归一化。

34槽经Linear15→128后加17×128关节身份与2×128视图角色参数，再经GELU、Linear128→128。身份必须在非线性之前与内容结合；旧版MLP后相加会在均值池化中丢失槽位与观测的对应。2026-10-12修复仅调整既有算子顺序，参数仍不变。只有准入槽参与均值池化，池化观测经GRUCell128→128更新hidden。没有准入槽时完全不更新hidden；年龄按dt增加，不发布人体。初始hidden/age为0，seen为False，未知age由seen明确区分。

另一个时间分支输入 `[dt, mean(age), admitted_fraction, seen_fraction]` 的log1p值，经Linear4→128、GELU、Linear128→128。context=hidden+time_context；全缺失时context可因时间变化，但观测hidden仍冻结，不能混淆这两者。

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
