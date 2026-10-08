# 人体模型Flow Matching baseline：原始论文分析

日期：2026-10-07。这是分析记录，不是复现结果或已训练模型。每篇链接指向原始论文；本地公开HTML与带行号文本位于sources/。阅读完成指方法与主要实验/消融正文已检查，不代表逐项重证全部附录证明或已核验所有SOTA排名。方法图说明来自正文图注，尚未逐图进行视觉审计。没有核对的超参数/硬件或数值明确留作缺口，不用猜测补全。不同论文的指标和输入不可直接排成统一准确率榜。

## P01 — Flow Matching for Generative Modeling

[原文](https://arxiv.org/abs/2210.02747)。正文依据：sources/P01.txt L0094–L0230；实验L0255–L0386，尤其Table 1、Figures 5–7。

- 流程：先指定噪声到数据的条件概率路径，再用网络回归该路径的向量场；训练不用逐步求解ODE，推理才积分。高斯路径支持diffusion与条件OT插值。
- 模块/目标：图像实验使用U-Net；CFM与边缘FM梯度等价。线性路径的目标可写为数据减噪声。条件OT不等于对整个训练数据求出了全局最优输运配对。
- 实验检查：同架构对比不同训练目标，并报告FID、BPD、NFE。部分跨论文ImageNet对比的模型规模及训练吞吐不同，不能概括为无条件更快。
- 采用：baseline采用可复现CFM与固定步长ODE，显式记录NFE；不据图像实验保证人体实时性。
- 缺口：本文没有双目人体/SMPL-H实验，需另行验证旋转表示和条件编码。

## P02 — Flow Matching on General Geometries

[原文](https://arxiv.org/abs/2302.03660)。正文依据：P02 L0128–L0264；实验L0358–L0426。

- 流程：在流形切空间回归向量场，通过满足条件的premetric构造概率路径；具有闭式测地线的空间可免训练期ODE。
- 目标：使用黎曼度量比较切向量；一般几何使用谱距离，可能仍需模拟条件路径，不能把所有RFM都说成完全无模拟。
- 实验检查：球面、环面、高维及三角网格上的密度建模；Table 1比较算法性质，Figure 5研究高维扩展。网格上的点分布生成不是整个人体网格生成。
- 采用：明确旋转的几何边界；起步可用欧氏6D表示，但需合法旋转解码和退化检查，RFM作为单独后续对照。
- 缺口：本文不能直接提供条件人体恢复的架构、标签或效果依据；SO(3)数值边界需本地测试。

## P03 — FMPose3D

[原文](https://arxiv.org/abs/2602.05755v1)。正文依据：P03 L0102–L0201；实验L0544–L0746，Tables 1/4/5。

- 流程：二维关节条件、当前三维噪声状态、生成时间各自嵌入，送入并联局部GCN与全局注意力，输出每关节三维速度；Euler生成多个骨架。
- 目标：线性条件FM速度MSE；RPEA以回投影损失的指数作伪似然，筛选并聚合候选。
- 实验检查：Human3.6M按受试者划分；本文骨架baseline MPJPE49.3mm，40候选/翻转/聚合47.3mm，不是相同推理预算。架构消融比较并联与串联；速度表是RTX4090骨架网络，不能套到完整双目网格系统。
- 采用：速度场的局部/全局分支和条件缓存；不把GCN+FM称新贡献。不直接逐关节拼接或平均人体旋转候选。
- 缺口：输出是骨架，没有形状、手指或人体网格监督。

## P04 — ProHMR

[原文](https://arxiv.org/abs/2108.11944)。正文依据：P04 L0082–L0156；实验L0157–L0331，Tables 1–5。

- 流程：图像编码条件normalizing flow生成SMPL姿态分布，另一个MLP估计形状和相机；SMPL解码网格。支持测试期多视图一致性优化。
- 目标：NLL、随机样本二维/先验、指定mode监督、6D正交项。零噪声为mode依赖其特定flow Jacobian，不可推广到FM。
- 实验检查：训练含真实三维及SPIN伪标签；最小候选误差使用真值挑选，必须与可部署点预测分列。人体拟合表控制相似训练数据及初始化。
- 采用：姿态分布与形状/位置分支分开；标记伪标签来源；FM零噪声预测不称mode。
- 缺口：6D不是唯一旋转表示，正文报告不加约束时存在监督不一致/模式坍缩；SMPL身体标签不等于SMPL-H手指标签。

## P05 — HuManiFlow

[原文](https://arxiv.org/abs/2305.06968)。正文依据：P05 L0070–L0117、L0209–L0300；消融L0374–L0379，Tables 1/2/5。

- 流程：边缘/二维关键点代理输入，形状分布条件化身体姿态；按运动学祖先生成各关节SO(3)旋转分布，SMPL解码网格。
- 目标：姿态和形状NLL、根旋转、可见二维样本误差；文中不用三维点估计MSE，并通过消融讨论其压缩多样性的作用。
- 实验检查：合成代理输入来自既有姿态标签并施加噪声/截断；评价区分点估计、min-of-100、可见二维一致性和关节分散。min-of-100不是实际选择结果，分散也不是校准置信度。
- 采用：运动学图有依据，报告精度/证据一致性/多样性三个维度；辅助端点损失需单列消融，不能默认越多越好。
- 缺口：祖先自回归采样的计算性质不同于并行FM；不能直接移植似然或零噪声mode解释。

## P06 — ScoreHMR

[原文](https://arxiv.org/abs/2403.09623)。正文依据：P06 L0086–L0147；实验L0250–L0305、L0369–L0373，Tables 1–4。

- 流程：冻结回归图像特征条件化6D SMPL姿态diffusion；已有估计DDIM inversion后以二维、多视图或时间观测guidance精修。
- 目标：标准噪声预测MSE，训练姿态含伪标签；测试期guidance梯度经过参数到观测的可微映射。形状默认与姿态生成分开。
- 实验检查：Table 1变更特征和伪标签；正文区分ScoreHMR-a/b，最优特征/标签设置不能混成单因素架构增益。PA-MPJPE不能检验全局位置。正文网格顶点数有6980表述，baseline不得照抄为模型资产拓扑。
- 采用：预训练特征可先冻结；真实可微人体层必要；guidance后续添加，先做直接条件baseline。
- 缺口：迭代精修预算、相机假设和原始SMPL回归初始化均不同于双鱼眼实时直接生成。

## P07 — HMR 2.0 / Humans in 4D

[原文](https://arxiv.org/abs/2305.20091)。正文依据：P07 L0124–L0174、L0518–L0541，Table 6。

- 流程：ViT-H/16图像patch → 单查询Transformer cross-attention → SMPL姿态、形状和透视相机平移；真实人体层产生6890顶点。跟踪另由PHALP′与运动模型实现。
- 监督：有标签字段分别用参数/3D/2D损失，另有人体先验判别；扩充数据采用ProHMR拟合伪标签，不能当成独立真值。
- 实验检查：B1/B2控制回归头比较ResNet与ViT，D1/D2/B2改变训练数据。模型、数据、伪标签各有贡献；不能把全部增益归到Transformer。正文报告大量配置/checkpoint在3DPW及LSP-Extended比较，自己的研究须保留独立验证集。
- 采用：图像信息不能只剩二维骨架；图像特征与姿态查询cross-attention合适。
- 缺口：巨大预训练骨干、单帧回归、固定相机假设不等于双鱼眼实时FM。HMR2单帧网络和后续跟踪器需分开讨论。

## P08 — SPIN

[原文](https://arxiv.org/abs/1909.12828)。正文依据：P08 L0074–L0138，Table 1。

- 流程：ResNet50迭代参数回归 → 初始化SMPLify二维拟合 → 优化参数/网格监督回归器；每图保存最佳拟合，再随训练更新。
- 监督：拟合中的二维重投影、姿态/形状先验；回归中拟合参数/顶点与二维损失。拟合差时保留二维监督；异常beta处理有明确规则。
- 实验检查：静态伪标签对照与在线拟合循环有区别；测试报告直接回归输出，不额外测试拟合。3DPW用于测试；混合2D/3D训练标签需逐样本区分。
- 采用：数据记录必须带label_source；拟合结果可用于弱监督预训练，但不能用同一拟合器结果证明真实精度。
- 缺口：重投影低只证明图像相容，不能消除错误左右身份、深度歧义或先验偏差；无需在第一版引入昂贵拟合循环。

## P09 — VIBE

[原文](https://arxiv.org/abs/1912.05656)。正文依据：P09 L0135–L0196、L0315–L0331、L0379–L0412，Tables 1–3。

- 流程：冻结图像特征 → 双向GRU → SMPL迭代回归；整个clip平均beta。AMASS运动GRU/attention判别器约束序列；训练clip长度16。
- 监督：2D/3D/SMPL损失，加运动对抗损失；AMASS运动与视频标签来源不同。MPoser作为另一种先验对照。
- 实验检查：含/不含3DPW训练要分开；单靠时间模块可能更平滑而逐帧误差更差。运动判别器帮助泛化，强先验也会压掉罕见运动。
- 采用：形状跨帧共享；分别报告位置、加速度及运动多样性，禁止只凭平滑度选模型。
- 缺口：双向GRU和整clip形状平均使用未来帧；不可原样称实时因果模型。第一版不加入GAN判别器。

## P10 — WHAM

[原文](https://arxiv.org/abs/2312.07531)。正文依据：P10 L0113–L0167、L0527–L0547，消融表。

- 流程：单向RNN编码2D运动并解码3D/SMPL；冻结图像特征整合；AMASS合成相机预训练后真实视频微调。另由相机角速度、世界根姿态/速度、脚接触和轨迹精修恢复世界运动。
- 监督：合成运动有参数真值，视频使用可用字段；训练顺序、相机运动和运动先验同时重要。
- 实验检查：去掉预训练、图像或神经初始化均退化；去掉相机角速度主要损害轨迹。轨迹精修改善脚滑/全局指标，却可增大jitter，说明抖动和全局精度不是同一目标。
- 采用：因果历史加图像；身体局部姿态和全局位置分支分开，坐标系显式记录。
- 缺口：接触概率和轨迹修正依赖场景假设；助步器使用者动作与AMASS有域差，不能直接证明触地或借此掩盖Stage/地面错误。

## P11 — GVHMR

[原文](https://arxiv.org/abs/2409.06662)。正文依据：P11 L0189–L0333、L0803–L0820，Table 3。

- 流程：2D/bbox/HMR2图像/相机相对旋转分别嵌入；12层512维RoPE Transformer预测SMPL-X身体参数、相机/GV根姿态、速度、静止点。重力与视向构成GV坐标，后处理与IK修正全局运动。
- 监督：参数MSE、静止BCE，另有2D/3D/顶点和相机/世界平移损失；AMASS、BEDLAM、H36M、3DPW混合，120帧训练。
- 实验检查：相对世界姿态逐帧积累比GV直接姿态差；后处理与RoPE各影响全局指标。滑窗对照并不等价于因果处理。
- 采用：显式坐标契约和相机运动条件；Transformer可以有全身信息，关键在时间mask与计算预算。
- 缺口：正文mask为双向局部窗口，使用未来帧；世界速度积分仍会累积平移误差。第一版不复制重力/接触后处理；重力与视向退化需要自己的检测。

## P12 — MotionBERT

[原文](https://arxiv.org/abs/2210.06551)。正文依据：P12 L0089–L0148、L1139–L1152、L1210–L1224。

- 流程：2D坐标与置信度序列 → 双流空间/时间attention和自适应融合；遮挡/噪声恢复3D运动预训练。mesh任务头预测SMPL姿态与全序列共享beta，也可与RGB初始姿态残差结合。
- 监督：AMASS投影3D、真实视频重投影、位置/速度损失；下游mesh使用参数和网格/关节监督。
- 实验检查：随机初始化、冻结骨干、不同mask、加入真实2D分别消融；mesh评估含关节与MPVE，作者指出二维点缺少形状信息。
- 采用：噪声/缺失增强，以及低成本运动编码；置信度作为输入参考，与观测可用mask分开。
- 缺口：时间attention默认不是因果；全序列beta pooling泄露未来。只有骨架输入不能充分观察体型，不直接照搬为图像人体恢复完整方案。

## P13 — VideoPose3D

[原文](https://arxiv.org/abs/1811.11742)。正文依据：P13 L0122–L0173、L0673–L0692。

- 流程：膨胀TCN把2D序列提升为3D关节；243帧感受野，明确提供预测窗口最右端的因果变体。半监督中另估计全局轨迹并重投影二维。
- 监督：3D关节、无标签二维回投与骨长约束；预测骨架，不生成网格或体型。
- 实验检查：固定二维输入下长上下文改善关节误差；因果版本性能弱于双向，延迟/精度需同预算比较。
- 采用：单向历史可先用小GRU/TCN而非很大Transformer；推理须验证未来帧扰动不影响过去。
- 缺口：H36M镜头畸变可忽略的论断不适用于鱼眼；无标签重投影不能证明三维准确。不能把其骨架目标当作本任务完成。

## P14 — MotionAGFormer

[原文](https://arxiv.org/abs/2310.16288)。正文依据：P14 L0080–L0140、L0398–L0404。

- 流程：含置信度的二维序列；空间/时间Transformer流与GCNFormer流并行、自适应融合。空间人体图，时间图按特征相似KNN，目标为3D骨架。
- 监督：位置和速度损失；多种深度/宽度模型比较计算量。
- 实验检查：GCN、TCN/TGCN、并行/串行、位置编码分别消融；局部图与全局attention互补，不表示任意加图就必然提升。
- 采用：固定运动树的轻量图分支可作为单因素开关；和attention-only同数据/预算对照。
- 缺口：时间KNN遍历全clip会接触未来；不能直接宣称实时。图节点在本baseline对应旋转参数而非三维点，需要真实SMPL-H父子表。

## P15 — MVGFormer

[原文](https://arxiv.org/abs/2311.10983)。正文依据：P15 L0094–L0140、L0582–L0589。

- 流程：3D几何查询投影到各相机，以deformable attention提取图像信息，预测2D偏移/置信度，再解析可微三角化更新3D；外观查询另融合图像。多人体地面锚点与查询分类控制重复候选。
- 监督：中间层3D/2D L1和分类；已知相机、空间范围和训练锚点匹配。
- 实验检查：查询数、解码层数、匹配半径、融合方式各消融；1024查询/4层的设置服务多人任务，不能无条件迁移到单人小系统。
- 采用：把相机几何作为条件，而不是让网络背相机编号；后续可做图像-几何交替融合。
- 缺口：3×4透视投影与鱼眼映射不同；单人baseline只需匹配后的目标，不需要1024查询。质量学习不等于允许改掉工程严格几何门。

## P16 — Learnable Triangulation

[原文](https://arxiv.org/abs/1905.05754)。正文依据：P16 L0068–L0149、L0545–L0578。

- 流程：每相机heatmap soft-argmax；学习各camera/joint权重后加权DLT+可微SVD。第二方案将图像特征反投影到64³体素，再V2V和三维soft-argmax。
- 监督：代数分支鲁棒3D损失，体积分支位置及heatmap损失；已知相机，同步单人，体积中心先从骨盆三角化求得。
- 实验检查：H36M绝对误差报告剔除了特定错误真值序列；单相机消融的中心来自4相机，是偏乐观条件。Panoptic部分“真值”由更多相机三角化，不等于外部物理测量。
- 采用：观测存在与质量分开、显式标定几何；保留被拒绝信息。把置信度自动学成趋零需另有检验。
- 缺口：体积开销高且依赖可见骨盆；透视/去畸变假设须适配鱼眼。第一版输入射线，保留三角化参考而不重写主线。

## P17 — METRO

[原文](https://arxiv.org/abs/2012.09760)。正文依据：P17 L0110–L0142、L0237–L0251，Tables 2–3。

- 流程：CNN图像特征拼模板关节/顶点查询，Transformer逐渐降维直接预测3D坐标；431粗顶点经MLP上采样6890，不输出SMPL旋转参数。
- 监督：直接顶点、关节、网格回归关节L1和2D损失；可用标签开关控制混合训练。随机masked queries促使跨部位信息恢复。
- 实验检查：同设置MVM消融H36M MPJPE由61.0降到54.0，PA由39.1到36.7；超过约30%遮挡收益变小。
- 采用：有空间身份的查询和观测缺失增强；网格误差需独立于骨架误差报告。
- 缺口：自由顶点回归不自动维持旋转合法性/人体拓扑几何；大顶点状态比参数生成更重。第一版保留真实SMPL-H解码，不直接流匹配6890×3坐标。

## P18 — Mesh Graphormer

[原文](https://arxiv.org/abs/2104.00272)。正文依据：P18 L0076–L0111、L0211–L0230、L0275–L0329。

- 流程：图像grid tokens+模板查询；MHSA后图残差块与MLP；431粗网格上采样完整网格，直接坐标回归。
- 监督：顶点/关节/2D L1、粗网格中间监督和MVM；H36M人体网格训练采用伪标签，3DPW设置包含其训练数据。
- 实验检查：图层放前两个encoder无益、后层有效；只有graph或只有grid提升小，两者一起更大。这反驳“堆GNN即可提升”的假设。
- 采用：保留局部图/全身attention可开关；图像信息要与节点发生交互。
- 缺口：其图是网格邻接，而本baseline图是旋转运动树；不能声称复现Graphormer。粗网格、预训练CNN和伪标签规模同时影响结果。

## P19 — EgoHMR（Zhang et al., ICCV 2023）

[原文](https://arxiv.org/abs/2304.06024)。正文依据：P19 L0130–L0264、L0442–L0465。此文不同于ECHP同名EgoHMR。

- 流程：第一视角图像中的交互对方+真实场景点云；图像/局部场景/可见性条件的GCN diffusion生成SMPL旋转，形状和平移确定性预测；COAP碰撞梯度guidance。
- 监督：clean姿态、3D/2D、beta、6D正交及碰撞损失；测试可见/不可见关节分开，后者使用min-of-n和多样性。
- 实验检查：EgoBody官方划分；20候选best-of评估依赖真值选样，不是部署选样。碰撞/2cm接触为几何代理，不能证明真实接触。
- 采用：图节点可以是人体旋转；未知部位保留生成不确定性；形状/位置单独分支。
- 缺口：头戴者观察别人，不是助步器固定双鱼眼或观察自身；真实3D场景点云条件不能由DA3米制深度替代。baseline不加入碰撞guidance。

## P20 — Fish2Mesh Transformer

[原文](https://arxiv.org/abs/2503.06089)。正文依据：P20 L0086–L0150、L0230–L0251、L0295–L0301；按取得的arXiv正文版本分析。

- 流程：头戴180°鱼眼自身图像；球面/等距柱状几何位置表征加Swin，SMPL姿态/形状、相机和朝向多头；参数加2D/3D损失。
- 监督：扩展ECHP使用第三人称4DHumans自监督标签，不是独立人体网格真值。输入本身没有第三人称图像，标签来源仍须声明。
- 实验检查：位置编码和新数据分别消融；无新增数据ECHP PA-MPJPE由57.67到71.44。PA对齐去掉全局尺度/姿态误差，不能据此证明米制双目定位或身份正确。
- 采用：鱼眼应显式带成像几何条件；先用本地标定射线，不照搬未经核对的球面半径/离散位置表。
- 缺口：头戴摄像机视野/标签域与固定双鱼眼不同；正文“恢复三维空间”不能解读为单张鱼眼直接得到真实深度。形状/相机损失权重、标签流程需读补充/代码才能完整复现。

## 横向比较与决策

|论文|网络真正生成/回归对象|时间或几何条件|对本baseline的直接作用|
|---|---|---|---|
|P01 FM|一般连续状态|概率路径、ODE|CFM基础目标|
|P02 RFM|流形状态|测地路径/度量|后续SO(3)对照|
|P03 FMPose3D|3D关节点|2D、图+attention|速度场设计；不冒充mesh|
|P04 ProHMR|SMPL姿态分布|图像、normalizing flow|形状/相机分开|
|P05 HuManiFlow|SO(3)姿态分布|祖先条件、proxy|参数旋转与不确定性|
|P06 ScoreHMR|SMPL6D姿态|图像、观测guidance|可微解码/后续guidance|
|P07 HMR2|SMPL参数|图像cross-attention|图像输入支路|
|P08 SPIN|SMPL参数|拟合伪标签|标签来源契约|
|P09 VIBE|SMPL参数|双向GRU|共享shape、未来泄露检查|
|P10 WHAM|SMPL+世界运动|单向RNN、相机运动|因果历史与坐标分离|
|P11 GVHMR|SMPL-X身体+轨迹|双向RoPE、重力|世界坐标后续适配|
|P12 MotionBERT|关节/SMPL任务头|双向空间时间|缺失增强、形状信息缺口|
|P13 VideoPose3D|3D关节|TCN含因果变体|最小时间对照|
|P14 MotionAGFormer|3D关节|图+空间时间attention|图开关，不先当创新|
|P15 MVGFormer|3D关节|标定、多视图特征|显式几何条件|
|P16 Learnable Triangulation|3D关节|学习DLT权重/体素|置信度与观测mask分开|
|P17 METRO|自由网格顶点|图像、模板queries|顶点目标和参数目标区别|
|P18 Mesh Graphormer|自由网格顶点|图像grid、网格图|图结构须匹配对象|
|P19 EgoHMR|SMPL姿态分布|可见性、场景点云|旋转图生成、候选预算|
|P20 Fish2Mesh|SMPL参数|头戴鱼眼位置编码|射线几何；伪标签边界|

第一版选择Euclidean 6D条件FM：52个SMPL-H局部旋转（含根）的312维状态，线性Gaussian→data路径，固定步数Euler；终点Gram–Schmidt投影到SO(3)，中间状态不宣称物理人体。它是可检查的起点，不是SO(3) RFM复现；近共线/零向量需显式无效状态。beta按序列首次可用前缀估计或外部给定后冻结；平移独立逐帧回归到统一米制参考坐标。真实SMPL-H层产生6890顶点和三角面。

条件为多视图二维坐标/置信度、存在mask、标定后的原点/单位射线，以及可选RGB或图像特征。图像特征可用小CNN验证接口，也可接冻结预训练特征；未下载预训练权重时不得称预训练图像baseline。低置信度作为输入，不乘成趋零的硬门；不存在观测才mask。GRU编码过去与当前；旋转queries在帧内cross-attend观测，再经空间attention和可切换父子图。不对整clip做时间双向attention、不从未来平均shape。

先执行同条件确定性回归与FM训练/解码工程检查，再进行相同数据/优化预算的graph开关和单帧/历史对照。候选1为部署基础，候选更多时记录NFE×候选数，best-of必须标记oracle；禁止将旋转或自由网格样本算术平均后当合法人体。真实数据阶段要求跨被试/采集划分、标签来源、手部标签mask、左右身份审计；当前360/448连续短视频不满足独立准确性训练/测试设计。

### 作者代码核对

- [FMPose3D作者实现](https://github.com/AdaptiveMotorControlLab/FMPose3D/blob/main/fmpose3d/models/fmpose3d/model_GAMLP.py)：实际检查TimeEmbedding、Block、Model.forward。状态/二维/时间拼接，GCN与attention残差并行，按B×F展平执行帧内网络。论文示意不能推断其有因果历史；本baseline另设历史编码。这里只参考设计，自写实现，不复制代码。
- [ScoreHMR训练入口](https://github.com/statho/ScoreHMR/blob/master/train.py)：use_betas默认false，img_feats默认pare；人体形状不必进入生成状态，图像特征来源明确可配。其训练流程未本地执行，不能声称复现论文。
- 其余18篇作者代码尚未逐库运行或全面审阅；正文方法/主要实验分析已完成，精确复现仍需补充材料与实现核对。未进行20篇全部附录证明、全部图的视觉核验或最新SOTA系统审计。

## 来源缺口

PoseD-Flow项目页：https://circle-group.github.io/research/PoseD-Flow/。页面显示旋转流形FM和人体恢复应用，但2026-10-07实查论文/代码按钮为空链接，未获得原始正文，不能计入20篇已读论文，也不据项目页完整推断其实现。暂用Mesh Graphormer补足正式20篇。后续如取得正文再追加，不虚报阅读。

## 2026-10-08 十篇作者实现审查：第一阶段

本阶段重新核对下面十个作者仓库，不用已有二十篇阅读数冒充十个结构复现。十份源码/配置/许可快照在`third_party/architecture_reproduction/`；逐文件来源、字节数和取得时间见`architecture_sources_v1.json`和各快照`SOURCE_SNAPSHOT.json`。不提交作者代码、资产或完整原文。外部仓库可能随时间变化，复现使用本地快照，不假定之后远端同名分支仍相同。

|论文及官方代码|实际检查的源码位置|对本系统的具体启示|当前执行范围|
|---|---|---|---|
|[ProHMR](https://github.com/nkolot/ProHMR)|heads/fc_head.py 7–51；heads/smpl_flow.py 12–90|图像特征MLP输出beta/camera，沿候选轴重复；概率姿态与确定性shape分开。camera不是双鱼眼米制translation。作者使用修改版nflows的ConditionalGlow，标准nflows不够|原ResNet50/flow/head及密度往返通过；neutral网格及原评价待完成|
|[HuManiFlow](https://github.com/akashsengupta1997/HuManiFlow)|models/humaniflow_model.py 33–340|shape均值/标准差定义分布；shape、root、camera及祖先旋转共同条件化关节SO(3)分布，默认采样shape会随候选变化，use_shape_mode_for_samples另有语义。不是FM，也不是我们的并行图平均|原proxy/SO(3)层级采样/密度通过；原网格/输入代理/评价待完成|
|[HMR2](https://github.com/shubham-goel/4D-Humans)|heads/smpl_head.py 17–111；utils/geometry.py 47–62|单查询cross-attention看完整图像patch，输出pose/beta/camera残差；不等于52个独立姿态查询或多帧shape。6D先按两组三维向量排列，必须核对转换函数而非只看名字|原ViT-H/Transformer参数头通过；neutral网格/检测裁剪/评价待完成|
|[VIBE](https://github.com/mkocabas/VIBE)|lib/models/vibe.py 27–117；spin.py 209–293；configs/config.yaml|正文P09 L137/L143–144描述双向GRU和整段平均shape；当前发布配置BIDIRECTIONAL=false，VIBE.forward把N,T特征展平，Regressor逐帧输出shape，这条代码路径未做平均。必须区分论文版本与发布实现|原TemporalEncoder完成；完整SMPL回归路径未执行|
|[WHAM](https://github.com/yohanshin/WHAM)|lib/models/wham.py 15–201；layers/modules.py 98–217|MotionEncoder/Decoder/TrajectoryDecoder均按过去递归；beta实际为B,T,10，不天然全片共享。preprocess的mask=True表示缺失，与本项目presence mask相反。世界轨迹/脚修正需真实人体层及外部相机语义|原网络构造、三个原组件和mask完成；真实人体层及全局脚/精修未执行|
|[MotionBERT](https://github.com/Walter0807/MotionBERT)|DSTformer.py 269–361及forward_temporal 188–200；model_mesh.py 9–80|姿态主干空间/时间两顺序分支融合；时间attention看整clip。mesh head先平均时间特征再预测一套beta，随后广播；与VIBE论文先预测beta再平均不同。shape池化使用未来，实时适配必须单列|原H36M姿态网络243帧、512宽、5层完成；mesh分支未执行|
|[MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)|MotionAGFormer.py 68–160/196–284；attention.py 50–62；graph.py 10–128|attention与图分支各含空间和时间操作，再自适应融合；时间attention及相似图没有因果屏蔽。原图为H36M17且图层有节点BatchNorm，不能将我们的52节点相邻平均称原结构复现|原xsmall网络27帧、64宽、12层完成|
|[FMPose3D](https://github.com/AdaptiveMotorControlLab/FMPose3D)|model_GAMLP.py 111–152/216–247；scripts/FMPose3D_train.sh|2D、3D状态与生成时间拼接，B,F展平成B×F后执行图/attention，网络是帧内速度场，不含我们的GRU，也不输出mesh/beta。完整论文另有采样、RPEA及评价预算|原速度网络、Euler、flip-TTA及RPEA依赖通过；发布聚合读取GT根，原数据/检测评价待完成|
|[Learnable Triangulation](https://github.com/karfly/learnable-triangulation-pytorch)|mvn/models/triangulation.py 18–129/131–200/203–355；utils/multiview.py；experiments/human36m/eval/|原ResNet152、heatmap积分、跨相机归一化confidence和加权DLT；体素分支依赖预测人体中心，实际从batch相机重建P。RANSAC半距离门、强制保留抽中的视图。透视P不能直接处理原始鱼眼像素，也没有项目严格门|原代数、64³体素/V2V完整前向通过；原RANSAC经显式Python集合采样兼容后通过；人工针孔输入、随机权重，原数据/检测/预测中心流程及性能待完成|
|[Riemannian FM](https://github.com/facebookresearch/riemannian-fm)|manifm/model/arch.py 75–106；model_pl.py 556–615|网络向量场投影到当前位置切空间；测地路径用jvp求目标速度，再用流形度量比较。不能把欧氏6D直线后投影终点称SO(3) RFM；此仓库也不是现成条件人体恢复网络|原球面路径/网络/损失及1000步采样通过；原数据/似然及人体SO(3)未完成|

### 对架构的阶段性判断

1. 共享体型是我们需要显式规定的任务契约，不能假设视频方法都天然共享。P09正文、发布实现、P10发布实现和P12 mesh head实际做法不同。当前首帧beta分支应保留为冻结对照；拟议候选采用因果多帧校准、完成后冻结，并在校准期明确shape unavailable，不追溯改写已发布帧。池化是否提升体型精度尚未验证，不训练阶段只验证信息来源和共享/因果语义。
2. 原MotionBERT和MotionAGFormer看未来是其离线算法正常行为，不是作者代码bug；如果任务要求零延迟在线，应另写因果适配并报告改法及延迟，不能只把整clip放入网络再称实时。
3. 局部关节旋转、全局根旋转、相机坐标平移和世界轨迹必须分开；弱透视camera不能改名为translation_m。Stage/地面仍由原系统负责，网络组件不能掩盖该层失败。
4. 缺失语义是必须补齐的接口，而非可选小技巧。原三种骨架网络不收项目presence mask，一个NaN输入可污染整clip/整帧；项目适配须显式保留拒绝/无效输出。WHAM缺失mask极性相反，直接传入会把正确点遮掉。
5. 当前图平均与GRU模块均未证明有效。原作者空间、时间和图结构复杂程度不同；先保留同容量确定性对照，再单因素搭建候选。旋转RFM需独立概率路径和切向速度定义，不能仅替换输出转换函数。

这只是源码与无训练结构证据，未形成新精度结果，也未完成十篇的全部结构/适配复现。正式状态以architecture_research.json为准。

## 进度

20篇公开正文的方法与主要实验分析已完成，含监督/对照/局限及设计依据。正文逐行存档于sources/；图目前依据正文图注解释，未声称逐图视觉检查。随后已实现并训练隔离baseline，结果见IMPLEMENTATION.md；只有合成工程证据，没有真实精度结果。


## 概率分支与架构取舍的实证补充

ProHMR概率性集中于姿态6D流；FCHead由图像条件回归β/相机，再复制到姿态候选。HuManiFlow预测β的Normal分布，并以采样β、全局旋转、相机和祖先姿态作为下游关节SO(3)流条件。use_shape_mode_for_samples只约束同图候选，不能代表跨视频共享体型。HMR2单查询Transformer同时读出姿态/β/相机，单图参数头没有全片共享协议。不能直接移植后宣称共享体型已解决。

HMR2公开ViT-H/16和头共6.7046亿参数；结构规模已确认，构造/执行总耗时不能当部署延迟。实时目标下应先测预算，不能直接照搬。HuManiFlow要求18通道边缘/热图代理；借鉴SO(3)层级设计还需处理代理构建及23关节SMPL到52旋转SMPL-H语义，不能直接补零认作相同模型。

Riemannian FM作者支持Sphere/FlatTorus/SPD/Mesh等，不直接提供人体SO(3)条件流。原球面路径和切向网络成功不证明改成SO(3)更准确。下一项隔离H1因果多帧体型校准，保持姿态生成器；随后做H2显式相机/关节/旋转适配。原完整网格/输入/评估缺口另列，长期目标继续。

## 原组件组合之前的语义核对（H2）

同为17点不表示相同监督：MotionBERT dataset_wild.py的halpe2h36m采用右腿在前的H36M序列，并依赖Halpe的颈/髋/头顶；项目为COCO表面点。WHAM CustomDataset初始化以coco根中心处理，detector.py指定COCO ViTPose配置；不能将其工具文件中H36M列表当成实际推理输入。

同为6D也不表示相同旋转：ProHMR/HMR2 geometry.py顺序列，HuManiFlow rigid_transform_utils.py默认交错列，MotionBERT utils_mesh.py交错列；WHAM transforms.py的rotation_6d_to_matrix为顺序行，而另一个rot6d_to_rotmat为交错列。六个原函数已用20个非对称旋转验证，显式适配一致；错误行/列接线仍通过正交和det=+1，必须测试矩阵含义。来源为本地作者原函数，与论文概念性“6D”描述分开。

当前架构更合理的推进顺序是先保证输入/输出约定，再讨论空间图和Flow路径。观察tokens的COCO语义与52个旋转queries分开，连接由条件attention承担，模型侧观测由真实表面回归器产生。它是明确的候选设计，尚未验证学习是否能正确对应；不能从模型骨架里挑相近节点或把弱透视camera直接变成米制平移。H1只修共享形状读出时机，H2只修候选接口；两者均不证明精度改善，也没有替代主线。

## 采样和部署条件审查（S1）

重新阅读P03正文L0126–L0138的Euler推理与L0139–L0196的RPEA，运行作者采样与聚合代码。论文的二维回投影伪似然不能自动解释为无需其他三维输入：当前发布aggregation_methods.py实际读取gt_3D的根，H36M加载器保留绝对相机根，评估脚本把它直接传入。固定候选/相机/二维点，仅改变根深度就改变原聚合输出；非根真值变化无影响。这里核实的是这条发布路径的部署条件，没有据此判断所有论文实验无效。

原RPEA还用误差截断和NaN权重均匀替换：全选中候选误差超过20×temperature时退化均值；输入NaN根仍可得到有限结果。项目候选不能借这两种行为掩盖不可用几何。若以后研究回投影候选评分，根位姿须来自明确、可用的观测/估计，投影须沿现有鱼眼标定，缺失须保留；这些改变都属于项目新适配，应另设对照，不能称原RPEA已可直接部署。

原单帧3步FMPose3D使用flip-TTA需6次速度网络前向；多候选进一步乘预算，论文的“少步”不能直接当系统实时证据。原HumanPostProcessor的根置零与可选示例world旋转/最低z平移也不能替代项目Stage/地面标定。本次只有根置零且关闭world旋转的单元前向，未迁移显示归地。

原RiemannianFM配置球面sample为1000步投影Euler；完整轨迹比last-only多一次终点速度查询。网络/测地损失/采样结构已经核实，但这些不等于人体SO(3)生成、更不证明比现有欧氏6D更准。后续旋转路径改进必须单列结构、梯度、推理预算和相同条件对照。

六个原SMPL网格方法的neutral模型、专用回归器和原初始化文件目前有明确缺项，详见architecture_full_path_resources_v1.json；不以其他性别/模型族或合成回归器补齐并升级复现级别。原方法完整结构、项目适配和已训练性能继续分开。

## 多视图空间结构的适用条件（M1）

P16原体素路径已按发布评估配置执行ResNet152→32通道特征→64³反投影聚合→V2V→17点积分。体素不是自动发现人体的空间模块：use_gt_pelvis=false时仍需batch.pred_keypoints_3d，MPII中心为index6；评估YAML指定预先计算的代数预测文件。2500mm立方体围绕这个外部中心建立。本次人工中心平移100mm，原输出随之变化；这只证明依赖，不是人体精度。若用于本项目，中心必须来自明确、可用的估计，缺失时应不可用，不能用人工真值或旧结果偷偷补齐。

原VolumetricTriangulationNet.forward将传入proj_matricies覆盖为batch.cameras深复制并缩放到heatmap后的P。保持batch相机，仅传NaN矩阵不影响输出；改变batch相机主点才影响结果。Camera.projection为K[R|t]，不包含鱼眼畸变。项目接入前应统一相机权威来源；原结构复现与鱼眼改造是两项不同任务，不能把适配鱼眼的射线直接塞进P矩阵位置。

原RANSAC的calc_reprojection_error_matrix返回0.5×像素欧氏距离，所以默认epsilon15对应严格小于约30px欧氏距离；这不是项目当前的重投影质量门。原随机抽样两视图直接进入inlier_set，四视图人工外点能排除，但两视图没有额外的一致性投票。人工异常双目仍保留两者并产生大偏移，不能把inlier_count=2当可靠三角化。forward还返回全零confidence占位、丢弃逐关节inlier列表，没有项目逐点拒绝状态。

架构判断：双鱼眼主线继续使用原严格几何与明确质量状态，优先将可用二维/射线及观测语义作为条件，再生成52旋转；不要为了“空间模块”照搬昂贵体素，或以RANSAC替代缺少冗余的双目质量门。三条P16分支运行证据完善了结构比较，但随机前向不证明体素/学习置信度优于当前算法。之后H1/H2组合只研究输入契约、共享体型和因果性，精度提升需另行训练和受控验证。

## 真实输入组合后的架构判断（H3）

H3已经验证COCO17条件与52旋转queries可通过原cross-attention接通，而不伪造H36M头部或把17点补成22点；输出采用真实6890顶点网格再回归COCO17表面观测。该连接是随机网络的可计算接口，不证明attention已经学会正确对应，不证明手部姿态可从17点识别。真实任务需明确WiLoR等手部信息的来源和旋转约定；当前候选未接入这些信息。

共享shape的K1/K8对照只改读出，姿态状态及平移完全相同；K8完成后β常数、缺失帧不能作为新体型证据，也不能因内部网络返回有限值而发布人体。原GRU仍在缺失帧消费null条件，且历史特征平均可能重复计入早期信息。这些是下一阶段需要分开研究的因果状态和信息来源问题，不以共享常数或平滑输出说明体型正确。

用真实360帧检验逐帧/分块一致性和未来隔离，补足了此前人工接口测试；它没有把原主线拟合值当训练标签，也没有重跑检测或调整严格门。当前约束仍是“不训练”，因此结构改进结论限于语义、状态管理、因果性和可追溯性；是否提高三维准确性不作判断。后续优先审查手部条件、缺失时的历史状态更新、共享形状的独立信息来源，再考虑新空间或流形模块，并继续补齐原方法资源和完整输入路径。

## H4手部来源审查后的结构取舍

辅助核对[WiLoR官方实现](https://github.com/rolpotamias/WiLoR)及现有run_wilor_sequence/wilor_mano_prior，不算新增选定论文。原网络的canonical-right父相对15旋转，左crop镜像、MANO mean状态和21输出点重排分别处理；只有经过原镜像与资产审计的局部旋转进入类型明确的条件。global_orient、MANO β、弱透视相机和投影手点有各自坐标语义，不应混入已标定cam0米制人体位置。关联仍用原始检测框与身体腕点及原候选歧义规则。

360帧显示两视图覆盖很不对称，候选每只手只有33帧同时具备两视图；旋转分歧左P95 25.58°、右37.94°（33×15个关节角差）。因此更合理的下一步是保留相机/手/节点身份、缺失掩码和独立假设，在人体条件查询中融合，不直接选最大框分数或平均后硬覆盖人体手旋转。框分数不是关节不确定性，交叉视图分歧不是精度；随机条件分支的作用还需训练/受控对照才能评价。

原共享PCA手姿初值以及后续共享手拟合使用整段视频，属于冻结工程主线的离线选择；逐帧条件契约无未来依赖。不能把离线共享手结果作为在线参考输入，否则会引入不可用未来信息。当前H4只补齐输入契约，H3未接入学习手分支；将共享体型、因果状态、空间融合和手部局部条件分别搭建及检查，避免一次改变多个来源和输出规则。十个原方法的默认资源和完整数据路径仍需继续，不把候选人体解码充当原论文复现。

## H5：把手部来源放入可检查的条件路径

P07正文L134–138及原smpl_head.py说明query通过cross-attention读取image memory。本项目借用条件与查询分开的形式，把H4两个视图的native15旋转、box score及camera/node身份交给30个同手queries；不是照搬P07单query或其图像语义。选择局部残差而非硬替换，是为了保持输入假设与完整人体生成结果可区分；保留两相机及缺失状态，是为了保留H4发现的覆盖差异和冲突，尚不代表融合学到了正确权重。

H5源于不同的信息用途：手指局部R可以帮助姿态生成，但没有理由把MANO局部shape、弱相机或global orient用于确定完整人体β/米制位置。因此本次独立分支只改30手节点condition，body global feature/GRU、共享shape与T保持原来源。现有全身空间层仍可传播手信息，随机测试观察到非手旋转变化，后续必须评价这种传播是否恶化身体；“局部输入”并不保证“只改变手部最终输出”。只有受控训练及测试才能评价精度，当前不训练只证明这条路径可计算/可微、因果、缺失处理明确。

关闭手条件后完整360帧与旧H3参数完全相同，开启后共享β/T/hidden不变，未来影响0；新增23616参数、单分支CPU检查约0.65–0.75ms不含整个系统。不能以这些检查证明新增attention必要，不能把48.83mm随机网格变化称改善，也不能把组件时延当整套实时。下一步继续原方法公共资源/完整输入缺口、null帧状态和shape独立信息来源，逐项保留控制；十篇完整方法与最终架构优化仍未完成。

## R1：原资产完整性不能代替方法复现

七个公共辅助文件补齐后六条原neutral人体路径仍缺14项，不能将资源接口通过称完整推理。HuManiFlow三个原regressor保留9/17/19行语义，不强制非负或归一化；人工顶点线性/梯度检查只说明原线性接口可用，不说明人体关节准确。

ProHMR FCHead只取mean的shape/cam，HMR2头还读取pose。两个实际获取的官方mean与旧MMPose fixture原字节相同，撤销仅这两个比较对象的“等价未确认”，不能顺势替代其他论文初始化。根6D残量约1.51e-7，小于项目1e-6退化门却大于作者normalize eps；合法原初始化不等于可靠姿态观测。应区分参数初始化与观测接口，而不是为文件通过放宽几何/旋转门。

下一因素为缺失历史：不发布null帧仍可改变hidden并污染后续shape认可特征。冻结hidden也有忽略elapsed time的代价，不能预认定更好。保持shape pooling和生成器不变，先验证状态与信息来源，再讨论学习收益。原完整路径与隔离候选继续分别记录。

## H6：因果与经过时间是两个问题

原H5无观测帧不发布，但原GRU依然消费null条件，360帧中29缺失帧hidden最大变化0.547084。冻结候选确实切断这类状态变化，却在人工1/8/64缺失恢复及帧号间隔10倍探针中完全不响应时间。于是“冻结消除缺失污染”最多是状态接口性质，不能推出运动预测更正确；长缺失期间不变化可能忽略真实运动。原null推进也没有验证经过时间的正确含义。

本次没有同时加入time embedding、修改shape池化或更换生成器。所有发布数量/失败状态保留、关闭策略与旧H5精确一致、未来隔离和stream检查通过；随机网格最大改变266.337mm只表明历史变量确实影响下游，不是误差减少。当前没有准确性优胜者。后续原完整输入/mesh路径继续，shape信息重复计入应另立因素，显式elapsed-time动态也必须新协议与独立对照。

## R2：配置声明和实际入口不能混同

HuManiFlow的CLI帮助列出male/female，执行却对非neutral主动NotImplementedError。这说明“官方参数列有选项”不是路径跑通证据。现有male标准SMPL可由原library wrapper处理，原随机概率头点估计/50采样→body库网格实际运行通过，独立保存重建一致，但只算显式替代male库配置，不算原CLI/defaultneutral。

原90joint输出拼接45+9+19+17；同样6890顶点也不能因此忽略regressor语义、gender或参数拓扑。现有SMPLX_NEUTRAL不是默认SMPL_NEUTRAL，SMPL-H也不是23body原模型替代。随机网格有限及SO3合法只支持接线与解码，缺完整图像proxy/权重/评价的情况下没有准确性、采样不确定性或男女泛化结论。默认14项资源缺口继续保留。

## R3：proxy的有限性和可见性假设

P05正文L207–209、附录L886说明Canny/HRNet-W48 proxy；原图像预测路径L74–115实际在confidence>0.75后强制COCO indices0..6可见。零或负HRNet热图仍返回0坐标，随后7张Gaussian非零，这个强制先验不能被项目当作“已观察到头肩”。NaN Gaussian乘false仍NaN，原mask不是NaN隔离接口。原算法对正常检测的假设与本项目失败帧契约必须分开，不能沿用其finite输入即可认为有效。

原图像路径use_shape_mode_for_samples=True，50候选共享shape mode；R2的原库默认False是另一个采样配置，不能混记。它们都不是跨视频因果共享β。固定真实保存RGB通过原HRNet/crop/Canny/sourceAST到原网络并解码male网格，只证明该原输入子路径可运行；随机HRNet/概率参数无法说明检测准确、模态可辨识或不确定性校准。完整predict函数先依赖renderer，原neutral/CLI/检测/评价未覆盖，不升级十篇完整复现。

### 2026-10-08 R4：HMR2原前向方法与明确male人体配置

执行作者HMR2.forward_step完整AST方法（源89–140行），原ViT-H/16+六层Transformer共670459037参数、seed43人工RGB256裁剪192，使用R1作者均值/J19和用户已有male SMPLLayer。未执行Lightning初始化、检测/裁剪估计、neutral默认、训练/权重/评价；44关节为25OpenPose+19回归，不能当COCO17或SMPL-H52。6890顶点/13776面，独立同设备SMPLLayer顶点精确、J19误差2.98e-8m、独立投影精确；保存NPZ在CPU另行重建误差2.98e-7m。重复精确，参数版本计数不变。

裁剪内单个NaN污染全部20670顶点坐标、132关节坐标和88投影坐标；原投影对零/负尺度派生平移仍返回有限值（深度1e13及-39.0625），有限输出不证明物理合法或有效观察。初次v1失败CPU均值buffer/CUDA设备冲突已保留，检查器显式head.cuda后v2通过，作者文件未改。Ruff首次格式/S102问题修正，141实际回归测试/15warning、47AST，另实际原方法与独立保存数据检查；原timm/cross/SMPL提示保留。原11路径不变、719作者文件仅存在/字节数核对，14默认资源缺口仍在，十篇长期任务保持active。

### 2026-10-08 R5：ProHMR原推理初始化依赖与64候选人体子路径

原initialize（66–84）及forward_step（86–159）AST方法实际执行，ResNet50/ConditionalGlow共51513389参数，seed47人工RGB224、seed59采样，原63随机+首个零latent候选，R1原mean/J19及用户male SMPLLayer。初始False下仅图像冷启动原报KeyError(smpl_params)，人工标注用于依赖探针后原方法设顶层True；eval模式4个ActNorm仍False且所有network/body state_dict逐值不变，未做数据初始化或训练。改变人工标注并重置真实顶层False后重新执行初始化，同seed全部网格仍一致；不能称标注驱动初始化完成。

原6890顶点/44关节/13776面，64beta共享、零latent首位与独立头一致；同资产独立解码顶点精确，J19误差1.19e-7m、投影精确、旋转合法。原RGB单NaN污染全部1322880网格坐标及关节/投影，但64log_prob仍有限；原加性耦合零logdet、未初始化ActNorm与初始LU令此log_prob反映latent密度而非网格有效性/可靠性。v1漏已有_deps导致UMNN导入失败保留，检查器补原既有隔离依赖路径后v2通过，未装新依赖/改作者。保存NPZ另进程20线程重建逐值断言失败2.38e-7m保留于rebuild_v1，统一原2线程独立重建精确；两者均满足原预注册2e-5门，未放宽。

141实际回归测试/15warning、Ruff/48AST，原两方法及保存数据独立检查另行执行，不称回归测试覆盖它；原autocast/triangular_solve/cross/SMPL提示保留，11冻结路径不变、719作者文件仅存在/字节数核对。原neutral/Lightning初始化/renderer/完整评价未执行，14默认缺口及长期active不变。同步P07逐篇coverage补R4，P04补R5，保留所有旧check与失败。

### 2026-10-08 R6：RiemannianFM原球面密度与异常返回审查

作者compute_exact_loglikelihood全AST、div_fn/output_and_div与Sphere×Euclidean原积分器执行，原width512/eightlayer/Unbatch/tangent网络1816966参数、seed53、2个声明球面点、exactdivergence、1000投影Euler步，未训练或加载权重。randomlogdensity为-2.5590150/-2.5883312、NFE1000、重复精确、网络state_dict不变。解析零场/旋转场对-log(4pi)误差4.44e-16；原投影常量轴梯度流散度-2az精确，解析对数密度积分误差8.6131e-5<预注册3e-3，积分符号通过；normalize_loglik按ambient3除，不是intrinsic2。名字exact只描述散度，不代表Euler无离散误差或准确人体密度；docstring负loglik与实际logdensity返回不符。

故意vecfield异常原broadexcept打印traceback后返回[0,0]，即有限logdensity0/指数密度1，并从请求的(logdensity,projerror)tuple变成tensor，没有nfe/integ_error日志；这是失败，不能当正常概率结果。NaN场不抛异常，原1000步传播2个NaN密度，integ_error非有限；不清洗。保留完整traceback与stderr。v1检查器将float32点转double而非直接生成声明float64，严格散度检查失败保留；v2正确float64通过。v2含PythonJSON NaN日志保留，v3重新完整执行，将非有限日志明确字符串记录并严格JSON校验，数值结果与v2精确一致。未改作者或放宽门。

141实际回归测试/15warning、Ruff/49AST，原密度/解析/异常路径另实际执行，未称回归测试覆盖它；719作者文件仅存在/字节数核对、11冻结路径不变。原Lightning/EMA构造、volcano真实数据/指标、其他流形与人体SO3未执行；默认14人体资源缺口仍在，十篇长期目标active。现有逐篇coverage及六份记录同步，原invalid-escape语法warning保留。

### 2026-10-08 R7：WHAM辅助资源受限GET失败与预算持久化

核对本地和实时作者fetch_demo_data.sh辅助body_models源，使用原R1选择成员/大小/20s安全获取器，唯一一次本阶段GET，连同此前交接报告HEAD超时计来源两次预算。只选择已核对H36M17×6890、feet4×6890和mean；WHAM关节回归总行数不由输入17擅自推断，未以其他回归器代替。调用前以exclusive创建attempt记录，进程崩溃或记录不完整也阻止重复claim，不自动重试。

实际约21.047s，重定向drive.usercontent.google.com后20s ReadTimeout；归档inventory为空、LimitedReader body字节0、installed空、3个选择成员仍missing，所有未访问成员不证明服务器文件不存在。0字节仅归档body计数，不代表HTTP/TLS/重定向无流量。默认缺口14→14，新full_path_resources_v4为现状复查，未改任何授权人体资产。来源网络预算现耗尽，不能下轮重试或使用镜像/账户/TLS绕过。WHAM完整body/globalfeet/refiner仍未执行，其余独立路径可继续；长期目标不完成。

新增2个持久claim测试（已claim后重试及非JSON已有记录拒绝、不覆盖），143实际全套测试/15warning、Ruff/51AST，719作者文件存在/字节数和11冻结路径不变；首次测试import格式Ruff修正。本阶段验收的是传输失败证据与预算执行，没有宣称资源获取成功或WHAM已复现。六份现有记录同步，小型claim/result/audit保留，本地提交，不推送。

### 2026-10-08 R8：MotionBERT原wild输入/尾片段/TTA/尺度恢复

执行原infer_wild CLI默认global_lite配置（width256/depth5，共16001549参数，与已测fullsize配置分开），原load_backbone、halpe2h36m/read_input/WildDetDataset和全no_grad loop AST66–88及pixel恢复93–96。seed61人工248条HALPE26像素观测，明确focus7，243主片段+5尾片段不padding；原flip=true/rootrel=false/no_conf=false/gt_2d=false。独立26→17行/均值及像素归一化精确，独立左右置换、两分支平均、根和pixel恢复全部精确；重复及state_dict精确，保存数组独立验证通过。未运行权重加载、视频/检测、渲染、顶层CLI、SMPLmesh或真实数据精度。

原full-sequence crop对全序列求坐标范围，只改变最后frame247的future点，prefix0..119输入最大变化1.581047；pixel固定图像尺度prefix0变化。这是前处理的未来依赖，不能只用网络因果mask判断整条路径因果。read_input不读取image_id，只沿JSON顺序处理；未focus时人物7/9成为连续记录，focus99空选择报ValueError，保留而不造空帧。全confidence0的crop返回全零，随机网络后仍全部有限且pixel输出254/255非零；负confidence因!=0仍参与crop。单NaN坐标使crop所有8432xy值非有限，5帧原network/pixel输出254/255非有限，唯一首根z被赋0，未清洗。

默认global分支仅每个clip首根z=0（frame0、243），其他246根z非零，不是逐帧根相对或世界轨迹。pixel模式全xyz乘min尺寸/2，xy加图像中心，是图像相对尺度而非米制深度。新增3个原reader/crop语义测试，146实际全套测试/15warning、Ruff/53AST；原完整子路径另实际执行。v1通过，提取共用原reader测试和输入文件随版本命名后v2重新执行，所有保存数组含NaN位置精确一致；两组人工input/NPZ本地留存、不提交大产物。719作者文件仅存在/字节数和11冻结路径不变、14默认资源缺口仍在、长期active。现有六份记录及P12coverage同步，保留首次import-format lint问题。

### 2026-10-08 R9：MotionAGFormer默认base完整网络及原评价边界

此前仅xsmall，现按train.py默认base原load_model完整实例化16layers/128feat/512rep/243frames，11723811参数，seed67人工H36M17条件。全evaluate函数93–221实际AST执行，原DataReaderH36M经__new__跳过文件读取，明确填人工dt_dataset及单243clip缓存，实际原denormalize/get_hw/read_hw/get_split_id方法运行；真实dataset文件/read2d/read3d/训练权重/检测/顶层CLI/人体mesh没有执行。原TTA/逐帧根0与独立左右置换融合一致，width1000对应xy+中心、xyz×500恢复精确。与独立范数、SciPy旋转最小二乘、二阶差分聚合误差最大5.67e-8，保存数据独立核对；所有数值只为人工参考运行诊断，不作H36M或鱼眼精度。

预注册243全覆盖范围触发原IndexError(index241,size241)，完整保留；245总帧/单243覆盖范围执行，但原action加速度均值含末两项初始化0（未修正分母）。完全一致人工参考因e1_all>0排除有效0误差帧，mean(empty)及nan断言失败保留。原acc_error不除dt²/FPS²，尽管原打印mm/s²，实际二阶差分单位每帧²；新增解析t²/隔帧采样验证2→8。原dataset2.5d_factor1→2时人工MPJPE改变，而Procrustes约72.203不变；P-MPJPE可掩盖整体尺度，不证明米制相机恢复。

未来帧121..242扰动使prefix最大3.3081e-6变化，为原随机默认网络非因果证据；原confidence0仍输出12393有限非零坐标、单NaN导致raw网络全部12393非有限。repeat/state_dict精确。v1数值结果及NPZ已保存，但检查器最后GBK控制台打印²失败退出1；v2仅stdout ASCIIescapedJSON修复，重新完整执行退出0，全部结果及数组含NaN位置与v1一致，不改作者。新增3个原指标控制，149实际全套测试/18warning通过；Ruff/55AST、719作者文件存在/字节数和11冻结路径不变。原timm/invalidescape及perfect-reference warnings保留，默认14资源缺口和长期active不变。六份现有记录/P14coverage同步，只提交小型结果及代码，本地NPZ留存。

v3以process-local denormalize只读observer调用未变原方法并复制中间结果，逐值核对完整evaluate的TTA/根/affine数组精确，避免只用聚合指标推断逐点一致。v3重新实际运行退出0、旧字段与全部保存数组含NaN位置与v2精确一致；当前结果引用v3，v1/v2保留。

### 2026-10-08 R10 十篇覆盖审查与体型来源纠正

本轮只核对已有证据与原代码，不运行网络、不训练、不载入权重、不联网。architecture_r10_results_v1.json逐篇列出实际执行范围、关节/旋转/坐标、缺失、时序/体型与未执行项，并保存对应结果文件的实际字段。VIBE仍仅TemporalEncoder，WHAM仍构造及motion/trajectory组件；其他原网络、AST输入/采样/评价子路径和male替代资产均不升级为默认完整方法。六条默认人体路径14项资源缺口现场核对未变；WHAM同源请求预算已耗尽。

纠正一处来源判断：P09原论文3.1第147–150行说明逐帧参数与T个回归器，原VIBE.forward把B*T特征逐帧送Regressor，后者逐帧更新beta，没有全片平均。P12原SMPLRegressor才有时间平均与beta展开；本轮只是AST静态核对，该mesh head尚未执行。旧P09_v1与H1预注册的“VIBE clip-shared/averaged beta”说法保留为历史错误，由本条和当前architecture_research.json明确纠正。H1前缀K8冻结属于项目因果改造，不能称原VIBE或原MotionBERT复现。

总stage/completed列表原停在R3，现按实际R4–R10证据同步。十篇638个论文源码文件仅核对存在/字节数，不称内容一致；11冻结路径与当前HEAD内容核对一致。R9的149测试/18warning是已有回归证据，本轮记录修改未重跑或增加测试。检查十篇ID/所有引用JSON/字段、14资源状态、4原类方法AST、六份记录同步、暂存diff；全部通过，所有既有失败保留。

下一检查阶段H7仅改变共享shape特征来源：局部单帧特征对比GRU历史摘要，保持K8、head、pose/history状态、noise、输入和发布/几何门不变。需预注册、信息/梯度归因和完整360逐帧/分块控制；H6不编码缺失时长的限制保留。原完整输入/mesh/stream/退化多视图缺口及最终综合继续跟踪，长期目标未完成，不以记录审查代替原路径执行。

### 2026-10-08 H7 共享体型信息来源单因素检查

新增隔离shape_features.py，继承未变H6，读取原GRU输入pooled（原normalized52queries的逐帧均值）供原PrefixSharedShape；对照history路线逐值复现保存H6。仅改变shape所取特征，K8、shape_head、pose/T/history、nativehand、noise97、8Euler及原发布/严格门不变；167315参数无新增/更新。携带状态绑定feature source与missing policy，切换必须显式reset。仍为只读eager pre-hook及原H6 hook，不支持并发/export部署或时延结论。

完整360输入/17sourcefields/763hand源槽/706admitted/3031strict全部保存并独立核对，local与history除beta外全部输出、隐藏状态、noise精确一致。发布仍324available/7pending/29missing，不可发布mesh保留NaN；frame0..7完成K8后同一beta，独立原局部特征均值/head核对精确。逐帧/chunk11/64差值最大9.5367e-7，future120后扰动对prefix影响0。早期frame0扰动不影响后续local特征，history后续差值0.11979。

明确人工两帧observed但不admitted、后八帧admitted fixture：local早两帧shape梯度0，history梯度L1为0.05074/0.07223；只改变早两帧特征，localbeta变化0、historybeta变化0.021306。它检验历史摘要把未获shape准入的观测带入后续shape的路径，不证明真实观测统计独立或β更准确。完整真实条件下beta最大变化0.94872、mesh最大变化41.991mm均为随机模型结构差值，不是误差改善；pose/T未变。真实SMPLH6890/13776与独立保存参数进程重建误差2.3842e-7m通过。

6新增测试，155实际全套测试/18warning、Ruff/58AST通过；638论文源码仅存在/字节数与11冻结路径核对，不称源码内容比较，默认14资源缺口仍在。首次targeted2failed/3passed因未先排除可变beta而调用带断言parity、缺失视图fixture未关闭nativehands；测试修正后5通过。新增mixedrow后首全套154pass/1fail：不同批量行抽样noise不同，不能比较pose；修正为shape/hidden行独立性和同batch完整对照，失败XML保留，v2全通过。初次2lint项修正；requiresgrad转诊断scalar与原资产/依赖warning保留，无optimizer/权重加载。

六份现有记录同步，小型result/validation/rebuild/tests与隔离代码本地提交，NPZ大产物留本地。H7完成R10所列feature-source检查，H6缺失时长盲区和部署限制不改级别；不选择“更准确”候选。下一阶段继续VIBE原regressor/body资源及未执行原路径，WHAM同源预算保持耗尽；长期目标与最终综合未完成。

### 2026-10-08 R11 VIBE原辅助包Range索引获取失败

现场核对作者prepare_data.sh原GoogleDrive ZIP入口与VIBE SMPL extra/mean依赖，先预注册并持久化exclusive attempt再发唯一GET，Range bytes=-65536/identity/stream/timeout20；只允许206合法尾部范围，拒绝200整包回退。ZIP含训练权重/人体资产，不能沿原脚本整包下载。两个预定辅助成员J_regressor_extra.npy、smpl_mean_params.npz仅作待选择目标，未以HuManiFlow或MMPose文件冒充作者资源。

实际约1.438s，重定向drive.usercontent后HTTP200、Content-Range无，检查器在正文读取前拒绝；inventory/installed为空、应用层archive body reader计数0，两个成员not_acquired，默认14缺口不变，v5库存保存。0不代表HTTP/TLS/重定向无流量；未读取200内容，不能推断是ZIP、确认页或服务器全面不支持Range。该来源1次已消耗、尚余1次，现有claim禁止本阶段重试；第二次须先解决选择性获取方案并另行预注册，不绕账户/TLS/镜像，不整包下载、不获取neutral/权重。

新增3项Range边界控制：200/重定向/HTTP错误在读取前拒绝、64KiB与archiveend约束、encoded transport拒绝；158实际全套测试/18warning、Ruff/60AST通过，638论文源码存在/字节数与11冻结路径核对，源码内容比较未做。六份现有记录同步，实际失败/claim/库存与检查器本地提交；验收的是失败与回退边界，不是资源成功或VIBE完整复现。H7结论保留，原完整路径与最终综合仍未完成。

### 2026-10-08 R12 VIBE原缺失特征与分窗状态契约

R11未读200正文，无法验证确认页/下载Range地址；现场查看gdown原实现需要从确认内容解析下载入口，没有证据支持猜测confirm参数耗用VIBE最后一次预算。本轮VIBE资源GET为0（只核对gdown源代码），不安装gdown/账户cookies，预算仍余1。改为完成R10明确未执行的P09missing/stream组件契约，原完整regressor/body缺口保留，未以组件替代完整方法。

原TemporalEncoder完整class AST27–63，released2layers/hidden1024/linear/residual/oneway，17840128参数、seed113，默认16window另人工48帧feature诊断。native forward签名(x)，没有hidden/mask/frameID输入且丢弃GRU返回hidden。三个16window直接重置后与whole后段差0.0356448，首段精确；显式GRU pre-hook传入carriedhidden/output-hook读取hidden的诊断适配，仍执行未变原forward，与whole逐值精确。该适配不是原生stream API，不称并发/export或物理时间正确。future24后扰动对prefix24影响0、重复/state_dict不变。

原全零feature输出98304个有限非零值，不表示当前观测可用。frame17单NaN对prefix0..16精确无影响，但frame17..47每帧全部2048非有限；不清洗、不造mask、不以finite输出作发布规则。独立进程从保存feature重新随机seed原实例化，whole逐值一致，savedcarry/reset/NaN边界逐值验证；NPZ本地保留。

新增2个原carry/reset及坏hidden失败清理控制，160实际全套测试/18warning、Ruff/62AST通过，638论文源码存在/字节数与11冻结路径核对、14默认资源缺口不变；初次unusedBASE lint修正，没有失败数值或测试结果。run_components的未来metadata生成文字同步纠正P09“paper clip-averaged beta”错误，旧P09_v1/H1预注册保留历史，不改作者文件/实际算法。六份现有记录同步、小型结果/验证/成功XML与检查器本地提交，无训练/权重/mainline变化。

下一阶段整理原已运行结构与H1–H7受控契约的架构综合，明确每个选择的证据和反例；原完整人体资源/输入与未执行多视图/时序缺口继续跟踪。不能因carry或缺失语义已清楚而宣称十篇完整方法或研究目标完成。

### 2026-10-08 R13 架构综合与β/平移依赖审查

architecture_synthesis_v1.json形成十篇当前实际覆盖、八个模块决定及证据字段绑定：原鱼眼射线/COCO17与52旋转query接口、H7局部前缀shape、H6显式历史与缺失策略、空间/手条件、6D与流形边界、参数耦合、FM/回归对照、概率/评价边界。没有用投票或随机前向“选最佳模型”；每项记录反例/未执行范围，default全方法不标完成。P09原逐帧β、P12静态时间pool与项目K8冻结区分，S2原RFM不能冒充人体SO3流，原SMPL24不等同SMPLH52。

本轮新增依赖干预仅frame0..15/原pair1157..1172（16帧、9ready），非新增360验证。model19/hand29/noise97/K8/8Euler保持H7；与原savedH7prefix差最大4.7684e-7在既有2e-5门内。只加β0.25时state/rotation/T逐值不变，只加T向量[0.1,-0.2,0.3]m时state/rotation/β逐值不变。autograd返回velocity endpoint对β的edge为None（无计算依赖，不是饱和0）；同一真实SMPLH网格对β梯度L1=0.0122963，β干预网格最大变化10.8493mm，T平移响应误差2.9802e-7m。说明当前β/T是独立读出并只参与mesh，不直接条件化姿态流；共享encoder相关性与β反馈是不同结构。前7mesh NaN保留、state_dict/冻结核心未变，无训练/权重加载。

历史verification_summary中同容量144019参数、同1000步合成FM110.4661mm/8NFE，回归50.0678mm/1NFE，关闭graph99.7432mm、关闭history98.2499mm；只读取既有metadata，不载入checkpoint/重训练。低维人工运动/代理图像任务不支持FM优于回归，也不能推断真实鱼眼、图结构/历史或概率校准收益；no-graph参数135699存在容量差，不能升级成graph有害的结论。FM与匹配回归继续双路控制，不指定准确性胜者。

下一H8只检验显式sharedβ进入现有velocity condition的单因素：保持H7体型读出/冻结、pose网络、history/T/nativehand、solver/noise、严格与发布门，提供关闭分支精确H7控制及主动β梯度/响应。未readyβ不得以NaN/默认0被当有效shape；记录新增conditioning容量，360/流式/未来/真实解码/梯度须独立验证。该结构受P05 shape context与本轮缺边启发，不是原SO3祖先流移植，也不预设条件化提高精度；H8尚未预注册或执行。

本轮真实干预/梯度/保存数组独立核对通过，八决定证据字段及十篇引用逐项验证；Ruff/63AST、638论文源码存在/字节数与11冻结路径通过，14默认缺口/WHAM耗尽/VIBE余1不变。R12的160测试/18warning为既有回归证据，本轮新检查器以实际运行及独立数组检查验证，没有镜像实现的新单测或重跑旧全套。首次importformat lint修正，SMPLH10系数warning保留。六份现有记录同步与小型证据/检查器本地提交；原完整输入/mesh等缺口、SO3/退化多视图等未执行范围继续跟踪，长期目标未完成。

### 2026-10-08 H8 显式体型条件进入姿态生成：单因素验收

按 architecture_h8_protocol.json 预注册，仅增加无偏置 Linear(10,64) 的共享体型条件残差，广播至52个query后进入原8步Euler；新增640随机参数，总167955。H7的K8体型估计/冻结、原history、nativehand、translation、noise及严格发布门不变。未ready体型仅条件残差为0，公开beta和mesh仍保留NaN；开关随state绑定，改变策略须显式reset。不是原HuManiFlow层级SO3完整移植。

architecture_h8_results_v1.json：完整360帧/pair1157..1516；关闭分支与同seed H7及saved H7逐值一致。开启后beta/T/mask/hidden/noise不变，前7未ready帧state一致；324available、7calibration_pending、29no_current_observation，3031严格点、763原手部slot/706admitted及17项body原字段完整保留。chunk1/11/64最大误差9.536743e-7，修改120以后body/hand对此前输出影响0。固定条件仅beta加0.25，前16帧endpoint变化0.1825393，beta梯度L1=0.0806542、projection梯度L1=2.1161964，前7未知beta梯度0；实际SMPLH测试固定decoder beta为0，另验证beta经pose进入mesh的梯度。随机mesh差134.586mm仅为路径响应，不是精度或改善。

独立进程读取saved NPZ核对原body/hand、未改字段和不可用NaN；重新由第7/359帧旋转、beta、T构建真实SMPLH，vertices最大误差2.384186e-7m，faces一致，见 architecture_h8_rebuild_v1.json。大型NPZ只留本地。实际全套168passed/21warnings，新8项测试；Ruff/66AST通过，638论文源文件存在/字节数及11冻结路径核对，源内容未作全面比对。首次7项targeted失败来自测试fixture引用不存在schema属性，改为显式COCO17后通过；首次import order lint已修正；失败描述与运行warning保留在 architecture_h8_validation.json。

六项现有记录同步。本阶段不训练、不加载checkpoint、不改变冻结主线或证据等级；不存在精度胜者结论。14项默认资源缺口仍在，WHAM同源预算耗尽，VIBE剩1次需先有实际选择性访问证据。下一阶段继续原作者完整路径与未执行退化多视图/时序/流形契约；长期目标未完成。
