# Flow Matching 人体模型 baseline 长期研究

## 2026-10-13 历史产物清理

用户授权清理不再使用的结果和测试产物。本目录已移出旧版架构结果及数组、H1 v3/v4结果与独立protocol/validation、全部通过的重复测试XML、v1/v2/v3架构图；文件进入Windows回收站。H1协议和结论保留在本文与IMPLEMENTATION.md，v1/v2失败原记录保留。H2 v1/v2已清理，当前replay.py直接读取的H2 v3 JSON/NPZ、H2协议/validation以及H3/H4/H5/H6/H7仍被代码读取的数组保留。其余试验最新数值结果、失败与网络尝试记录、当前Flow next协议和成本记录、论文、源码及维护中的测试不变。

下文旧文件名和命令仍是历史记录，不保证每份旧中间产物现在都存在；恢复已清理的旧运行产物应从回收站取回，或按原入口显式生成新输出。不得因清理重写失败为通过，也不以本次文件整理增加任何精度或完整复现结论。当前维护重点仍为learning_flow_mesh_next与正式SMPL-H主线。

开始日期：2026-10-07。用户授权：阅读并分析20篇相关研究，先分析再实现通用baseline，记录分析和实现过程，并持续推进。当前正式工程主线仍按 current_mainline.json，不由本实验替换。

## 目标与阶段门

1. 阅读20篇原始论文正文，覆盖生成式人体恢复、Flow Matching及旋转几何、视频、双目/多视图、鱼眼/遮挡、骨架图与网格模型。逐篇记录正文位置、输入输出、模块、损失、实验控制、计算量和复现缺口；摘要或项目网页不计作正文完成。
2. 形成横向比较与可追溯设计决定，明确通用baseline的输入契约、人体参数表示、形状共享、旋转、因果性、标签来源和最小对照。阶段1未完成前不以已定架构开始实现。
3. 隔离实现学习模块、配置、训练/推理入口和可微真实SMPL-H解码。不把假网格或仅骨架输出当成人体模型恢复完成。
4. 运行数值、梯度、掩码、因果性、checkpoint往返、训练及推理检查；以小规模可复现工程实验验证可学习性，真实数据准确性另有协议。失败和限制保留。

## 验证协议

- 输入：原始论文正文及作者实现；实现阶段仅使用显式许可的本地人体资产与明示标签类别的数据。
- 对照：同条件编码的确定性回归；FM生成器中attention-only / graph+attention；单帧与因果历史；同数据划分、标签来源、优化预算和推理候选预算。
- 唯一变量：每个对照一次只更改一个因素；未执行的对照明确标记未执行。
- 成功标准：20篇正文分析有来源和正文锚点；实现能训练、保存、加载、采样并解码真实网格；生成旋转合法，未知/缺失状态保留，未来帧不影响过去；人体参数和SMPL-H网格损失可反传。
- 停止条件：许可资产/标签缺失、数值不稳定、来源或划分不明时停对应环节，保留失败并继续不依赖该问题的工作；不得以当前拟合结果同意自身证明真实精度。
- 长期跟进：本聊天目标与每小时heartbeat，里程碑通知；不同时运行多个写入本实验的进程。

## 记录与交付

- papers.json：20篇候选及正文获取/阅读状态，区分检索和阅读。
- LITERATURE_ANALYSIS.md：逐篇卡片与横向比较。
- IMPLEMENTATION.md：设计依据、输入输出、代码、配置、命令、验证、失败及剩余工作。
- progress.json：阶段、已完成项、下一项、自动跟进状态。
- sources/：公开原文和抽取的阅读材料，仅本地保存，不提交完整论文或模型。
- outputs/：实际验证产物和checkpoint，仅本地保存；小型结果可单独提交。
- 本地提交只包含本次代码与小型记录，已有工作区改动不纳入，不推送；实验记录不写Git哈希。

## 当前状态

20篇方法/主要实验分析、隔离网络及4组1000步合成训练/推理完成。旋转、因果性、缺失、真实解码梯度、全训练重放通过；具体证据见IMPLEMENTATION.md和verification_summary.json。合成回归优于FM，图/历史无已证增益；无真实RGB/鱼眼或独立精度结果。交付前只提交本任务文件，完成后停用heartbeat。

## 2026-10-08 新授权：十篇结构复现与架构改进（不训练）

用户要求深入研究十篇相似方法，先搭建、复现和逐步调整架构，设为长期任务。此前二十篇正文分析及合成训练为独立历史阶段；不能把它们算作十个作者实现复现。本阶段不训练，不改变reference25主线及方法证据级别。

执行协议和逐篇状态见`architecture_research.json`。选择ProHMR、HuManiFlow、HMR2、VIBE、WHAM、MotionBERT、MotionAGFormer、FMPose3D、Learnable Triangulation和Riemannian FM，覆盖体型/姿态分支、空间/时序、标定视图融合和流形路径。作者源码留`third_party/architecture_reproduction/`，只提交自写运行器及小型来源/结果记录。区分原组件、原完整网络、项目适配和自写候选，不把随机前向称为精度复现。许可/资源阻碍保留并继续其他独立部分。

先锁定协议，再执行结构检查；逐项候选改动必须写明输入、对照、唯一变量、预期作用和失败条件。先核对共享体型信息来源及因果性，再决定空间模块和旋转流的修改，不预设图/历史/FM有效。每小时本聊天自动跟进仅在里程碑、实质失败或需输入时通知。


### H1 因果前缀共享体型校准预注册

见`architecture_h1_protocol.json`，编码器、shape_head、姿态生成器及参数保持原样，仅替换shape读出：首帧→前8个调用方明确认可帧的条件特征均值。校准完成前β为不可用，NaN与ready=False同时保存；完成后同一序列冻结一个β，不回写已发布帧。严格递增帧号和序列状态隔离，缺失帧不计入8帧。8为结构检查阈值，不证明真实体型可辨识。检查batch/stream、未来隔离、缺失、冻结、真实SMPL-H解码/梯度；不训练，不接入主线。协议先提交后实现。

### H2 关节、旋转、掩码和鱼眼坐标接口预注册

见`architecture_h2_protocol.json`。只搭建隔离适配层，核对作者旋转函数及已有360帧保存输入。显式区分COCO表面观测与SMPL-H运动学旋转节点；区分列打包、列交错、行打包三种6D旋转；显式声明掩码极性和像素域。相机射线统一到cam0、米，严格接受状态与拒绝原因逐点照录。二维文件名中的raw不作为坐标域证据；与现有转正逆变换作精确对照。无训练、无检测重跑、无拟合、无主线更换。成功仅代表接口和内部几何一致。协议先提交后实现。

### S1 原采样路径及输入依赖预注册

见`architecture_sampling_protocol.json`。运行P03原Euler和flip-TTA，固定候选/相机/二维点，单独扰动传入RPEA的三维根位置与非根值，核实实际依赖；运行P02原1000步球面投影Euler的完整轨迹和末帧路径，检查单位范数、NFE和端点一致。无训练、无学习权重或主线调用替换。各原网格路径缺失的neutral SMPL与原回归器列清单，不能用male/COCO代用品冒充补齐。协议先提交后执行。

### M1 原多视图体素与RANSAC预注册

见`architecture_multiview_protocol.json`。保留P16原384×384/ResNet152、64³/V2V/2500mm与RANSAC10次15px的配置，明确人工针孔相机、预测骨盆输入；检查实际相机/中心依赖、深复制、体素归一化与几何控制，不使用项目真值或更改质量门。原Python集合采样如失败，先记录再用局部明确兼容接口，不改作者文件或采样算法。无训练、权重或主线替换，完整原数据评价另列。

### H3 真实COCO17条件与共享体型组合预注册

见`architecture_h3_protocol.json`。复用H2完整360帧数组与原严格状态，明确17个表面观测和52个局部旋转；fresh随机模型显式J17，不加载原J22合成checkpoint。H1前8认可帧共享shape与首认可帧对照，其他生成/平移/噪声和输出规则一致。形状未完成及当前观测缺失的帧保留NaN/不可用，不回填；严格XYZ仅诊断，不输入网络。验证逐帧/分块/整段、未来隔离、真实网格与COCO表面回归、独立小梯度检查；不训练、不改主线，失败和原方法资源缺口保留。协议先提交后实现。

### H4 原生手部条件与因果输入审查预注册

见`architecture_h4_protocol.json`。先按原read_parameter_view复现主线全部360帧的选择、旋转和权重，再显式与H2人物可用状态相交用于H3候选；原缺失identity占位不得变为有效手姿。核清左右镜像、15节点顺序、SO(3)/6D和box置信度，双视图冲突逐项保留。运行原整段shared-PCA初始化赋值的未来依赖探针，区分离线拟合与逐帧条件；全局朝向/形状/弱相机/投影点不输入。此阶段不加学习分支、不训练、不改主线/原门，先通过输入契约再搭建手部网络。

H4执行记录：v1因检查器body dtype未按原pipeline转float32而与原审计有微差，失败保留；只修转换位置后v2/v3全部通过。763个原接受手/视图槽独立核对旋转/框score一致，706个满足H2候选视图可用，57个仅保存源信息；3031严格点、29关联失败帧不改。每手33帧双视图旋转分歧仅作模型一致性；原离线shared-PCA初值有整段未来依赖，新逐帧tokens无未来依赖。完整细节见IMPLEMENTATION.md，结果/独立检查见architecture_h4_results_v3.json和architecture_h4_validation.json。新增15项测试，全模块90及Ruff/AST通过。仍没有学习手分支，未训练/替换主线，原完整方法资源缺口继续推进。

### H5 原生手部条件分支预注册

见`architecture_h5_protocol.json`。参考P07参数query读取独立条件memory的接口形式，项目候选使用30个手节点query及带相机/解剖节点身份的两视图native15旋转memory，同手cross-attention残差只加到对应手condition，随后复用原52节点生成器。明确不是原HMR2复现或训练结果。唯一变量为手部条件残差，body encoder/GRU/globalfeatures、β读出/冻结、平移、8步Euler和发布规则不变。没有手输入时跳过attention、贡献精确零并与H3对照；全部360帧、逐帧/分块、未来隔离、真实解码/梯度及限定CPU成本检查。原candidate只增加编码委托接口复用工作流，不复制状态逻辑；失败保留，不训练/替换主线。

H5执行：v1运行器误取COCO回归矩阵一行而失败，已修运行器并保留；v2/v3通过。源17字段及H4来源照录，无手控制与旧H3参数逐位相同；β/T/hidden/发布规则不变，完整360与1/11/64分块、未来隔离、真实人体重建通过。新增23616参数，12项测试，全模块102/Ruff/AST通过。分支CPU中位0.650/0.754ms仅有限组件计时，不含检测/完整解码/Stage。随机姿态/网格变化不代表改进，详见IMPLEMENTATION.md及architecture_h5_validation.json；长期完整方法/资源和其他独立架构因素继续。

### R1 原论文公共辅助资源预注册

见`architecture_resources_protocol.json`。按官方仓库/资源脚本获取明确的原regressor/mean，检查形状、数值、单位响应和梯度，再重新核对六个默认人体路径的资源清单。只获取小型辅助文件，限制请求时间/字节，压缩包只读取allowlist常规成员，不展开模型/权重/数据集，不覆盖既有文件、不用同形同名或male/SMPL-H替代默认neutral。失败保留，原源码/core/主线不改。文件存在不代表完整方法或精度通过；许可模型和完整输入/评价仍单列。

### R1 实际验收

七项原公共辅助文件schema及人工线性/梯度本地核验通过，本次网络0，保留v1/v2获取和历史WHAM传输失败。唯一变量仍是辅助资源，未改正式主线、作者原源码或候选。获取器唯一目标完成/EOF缺项/重复成员/读后deadline和已记录两次网络尝试保护已检查。新architecture_resources_results_v3.json与full-path资源v3为当前证据，v2保留。六个默认mesh路径仍缺14项，未执行完整neutral网格或原评价。124测试/14warning、Ruff/40文件AST通过，详细边界与不变路径在architecture_resources_validation.json。长期目标未完成。

### H6 缺失帧历史更新预注册

见architecture_h6_protocol.json；先锁协议再执行。唯一变量是无当前观测帧的GRU transition：原null照常更新对照，候选冻结hidden。保持H5所有参数/shape池化/native条件/noise97/K8/8Euler/发布规则。冻结不编码经过的时间，以真实360缺失和人工1/8/64长缺失检验恢复及时间忽略，不能先认定更好。使用临时eager原GRU输出hook复用原参数，额外重复计算及并发/编译限制明确保留。无训练/权重加载/主线变更/准确性结论。

### H6 实际验收

architecture_h6_results_v1.json/NPZ与validation通过。原H5随机参数/手来源/shape池化/noise/K8/8Euler不变；唯一变量为无原当前观测时冻结GRU transition。真实360完整源字段/strict3031/native763→706/失败29保存；324/7/29输出不变，missing参数网格NaN。原missing hidden变化0.547084，freeze为0，未来影响0，分块<=9.54e-7、真实网格重建2.38e-7m。人工missing1/8/64恢复及稀疏帧号证明freeze忽略经过时间，不能以结构通过选为更准主线。135全测试/14warning、Ruff/43AST通过，无训练/更新/新参数或主线变更。完整原方法及14资源缺口仍待推进。

### R2 用户现有资产及HuManiFlow男模型库子路径预注册

用户提供三个本地资产目录并授权有用则用；实际无neutral SMPL，SMPLX_NEUTRAL不是替代品。HuManiFlow CLI列出male/female但非neutral实际NotImplementedError，不能当CLI支持证据。见architecture_r2_protocol.json，独立配置以现有male SMPL调用原模型库点估计/50候选→原SMPL wrapper，核对真实网格/90关节及原regressor；不加载训练权重、不改作者文件、不补齐默认neutral，不宣称完整图像/CLI推理。

### R2 实际验收

原HuManiFlow男模型库子路径v1通过：原点估计/50采样头接原SMPL wrapper输出6890/13776/90，另进程50保存参数独立重建及原regressor块核对。随机seed31/人工proxy，参数不改、网络0、无训练/训练权重；原CLI对非neutral拒绝和默认neutral未覆盖明确保留，PyTorch3D缺失提示保留且未渲染。135现有回归测试重新通过/14warning，原子路径由另验覆盖，Ruff/44AST。默认14缺口未变；architecture_r2_validation.json保持whole_goal_complete=false。

### R3 HuManiFlow原RGB/proxy输入路径预注册

见architecture_r3_protocol.json。固定保存frame0/原pair1157/LEFT转正全视图，作者no-detector分支只作结构输入不宣称已居中人物crop。原HRNet-W48/384×288、Canny、18×256proxy及源AST输入→pred语句实际运行，使用原图像路径50候选/sharedshape=True；R2的默认False采样另记库配置，非原图像路径设置。另审查zero/negative热图、强制头7可见、NaN mask及非对称坐标。随机无权重、male库资产独立标记，renderer/原CLI/neutral/评价未覆盖，作者文件/主线不改。

### R3 实际验收

固定frame0 RGB→原HRNet-W48→原crop/Canny/高斯/visibility连续AST L74–115→原50sharedshape参数→male原SMPL wrapper通过。独立保存RGB/argmax/conf/Canny精确、Gaussian误差5.96e-8、仿射0、50保存mesh/前45点/面重建精确；141全测试/15warning、Ruff/46AST。zero/negative热图强制7Gaussian、maskedNaN仍NaN及bbox原位改变反例保留，作者不修改。随机无训练/权重/网络，仍非已认可人物crop、未做detector/defaultneutral/renderer完整函数/CLI/评价，默认14缺口保留。R2默认False库采样与R3原图像True共享shape显式分开。结果v1/validation保持whole_goal_complete=false。

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

### 2026-10-08 R14 作者DLT缺失置信度与退化几何审计

按 architecture_r14_protocol.json，直接加载未改动P16 mvn/utils/multiview.py 原NumPy/Torch点DLT及batch函数；仅人工无畸变针孔、毫米float64，K焦距384/主点192、相机x=±300、点[80,-40,3000]。不分配作者网络、不加载权重、不访问网络。10组NumPy/Torch病例及1组batch dtype病例完整保存，NumPy没有confidence参数，其对应结果明确是无权重控制，不用于声称拒绝缺失置信度。

健康Torch误差1.86446e-11mm，独立双射线视差深度公式误差4.54747e-13mm；省略/全1权重逐值相同，统一乘7误差9.09495e-13mm。全0置信度Torch返回有限[0,0,0]、无warning，实际投影却非有限；仅1有效视角或重合相机也可返回有限但不唯一的点，两SVD后端重合相机数值显著不同，不把具体坐标当稳定契约。负深度点被正常返回，无cheirality拒绝。0/1视角原函数IndexError，NaN点/相机/置信度触发原SVD异常，均保留而不填补。batch输入float64却输出float32。这里证明有限坐标不足以判定有效几何，不是物理精度，也不替换项目严格质量门或鱼眼模型。

architecture_r14_results_v1.json/v2.json数值完全一致；首次6项B023 lambda变量绑定lint已以显式default capture修复，未改作者函数。独立双射线对照、零置信度反例与缺失/NaN异常3项targeted测试通过，全模块Ruff/68AST、638论文源存在/字节数与11冻结路径核对通过；没有全面源内容比较。H8的168测试/21warnings保留为上阶段证据，本阶段没有重跑完整套。见 architecture_r14_validation.json/architecture_r14_tests.xml。六项记录同步，14默认资源缺口及原完整路径待执行范围保留，长期目标未完成。

### 2026-10-08 R15 MotionBERT官方网格辅助资源目录访问

按 architecture_r15_protocol.json 核对P12 docs/mesh.md第5行官方OneDrive目录，该文档只将SPIN链接作为regressor说明，不能据此把SPIN同名文件或HuManiFlow旧文件直接换入P12。网页工具open未能读取链接，无正文返回；为保守预算计1次同源访问。预注册后仅1次direct GET，持久exclusive claim防止重启重复请求。无本研究真实进程运行，无训练/模型/权重分配或主线修改。

官方短链经301/302到onedrive.live.com，返回HTTP200 text/html，声明Content-Length404347字节，超过预注册262144上限，读取正文前拒绝；body_bytes_read=0、清单空、安装空、辅助资源0。保留完整源URL/redirect/header/原异常在 architecture_r15_results_v1.json 与 architecture_r15_attempt.json。这不是服务器不可达、文件不存在、资源下载成功或完整作者路径的证明。未尝试账号、猜Graph/确认URL、镜像、整包数据或权重。该链接当前保守2次访问预算耗尽，不能自动提高限额或重试。

8项针对性测试验证HTTP/类型/编码/超限头拒绝及existing claim在网络session前拒绝；全模块Ruff通过，首次test importformat已修复；70AST、638论文源存在/字节数、11冻结路径核对通过。没有全面源内容比较或本阶段全套重跑；H8的168tests/21warnings与R14的3tests仅是先前证据。六项现有记录同步，小型请求/验证证据提交；14项默认缺口不变，neutral及P12三个辅助文件尚缺。只停止该资源入口，继续其他独立原推理/时序/流形路径。长期目标未完成，见 architecture_r15_validation.json。

### 2026-10-08 R16 HMR2原保存图裁剪至完整随机网格子路径

按 architecture_r16_protocol.json 固定frame0/cam0 LEFT正立CCW90图pair_0000.png（原pair1157），1920×1080；输入框显式整图[0,0,1080,1920]，不是检测器输出或已认证人物框。直接加载作者ViTDetDataset及utils；采用原ViT配置256输入、原loader第75..78行BBOX_SHAPE=[192,256]，未读取checkpoint。原crop实际Gaussian抗混叠downsampling_factor=7.50000048，size1920.000122/center[540,960]；BGR转RGB及ImageNet归一化后接完整670459037参数ViTH/6层头/原forward_step AST/合法现有maleSMPL44joint，原seed43不变。只为R4 harness增加可选normalized_image参数，不改作者或任何冻结核心。

同图重复逐值一致、参数version不变；原GPU独立mesh误差0/joint5.96e-8m/projection0，另独立CPU进程从saved旋转/beta重建mesh最大误差2.384186e-7m，faces一致，保存输入像素与源图完全相同。小型非抗混叠非对称fixture的独立affine/BGRtoRGB/归一化与作者逐值相同；不将此控制扩展为完整图Gaussian独立等价证明。原NaN输入产生20670个非有限vertex等，照原行为保留。

原emptybox length0；图外框仍返回有限normalized black并保留personid；zero-size框产生scalar divide warning却返回有限patch；NaN框原AssertionError，不修补。报告channel std使用float32累计会有小数值残差，测试按逐像素black广播控制，不把std残差当图像内容变化。作者原框处理不证明人体观测有效，更不提供原鱼眼标定/严格几何或精度。

首次完整运行在新增报告汇总处变量error被except绑定清除，UnboundLocalError；v1 JSON/input/mesh NPZ保留，更名affine_error后v2完整重跑通过。初次targeted2failed/1passed分别是bbox float32舍入误差与恒定patch均值累计误差，修正测试控制，不改作者输出；失败XML原样本地保留，failure描述提交。实际全套182passed/21warnings，新3项；Ruff/72AST、638论文源存在/字节数及11冻结路径核对通过。没有全面源内容比较。六项记录同步，14默认资源缺口不变；neutral、作者initializer/detector/完整demo与真实性能仍未执行，不标完整复现。所有大型NPZ留本地，小型结果/重建/验证/成功XML及本阶段代码本地提交。长期目标未完成。

### 2026-10-08 用户停止扩展后的最终架构综合分析

用户明确要求“现在停止继续深入了。你基于现有的结果，进行详细分析。然后达成我的目标”。据此停止资源获取、作者路径扩展与新候选实验，原长期扩展goal暂停、automation已删除。原十篇默认完整复现仍未达成；本次完成的是基于现有证据的架构分析与参考实现收敛，不将用户停止研究写成全部原方法完成。architecture_conclusion_v1.json整理最新十篇覆盖与设计决定；R13旧综合保留为历史。

十篇的作用是明确结构适用条件，而非模块投票。ProHMR提供条件分布与共享shape样本设计，但image-only冷启/ActNorm及NaN有限logprob有边界；HuManiFlow明确shape参与pose条件，可启发H8依赖边，不能直接换SMPL24为SMPLH52；HMR2提供query-memory/原RGBcrop与参数头路径，R16固定图原crop+完整网络/male网格已运行，detector/defaultneutral仍未完成。VIBE原TemporalEncoder按窗口重置且完整regressor为逐帧beta，不支持此前误称的clip平均；MotionBERT与MotionAGFormer可启发空间时序关系，但整序列未来依赖不满足当前在线因果合同。WHAM的世界运动、脚与refiner不因运行motion组件就具备。LearnableTriangulation是无畸变针孔/显式camera/center方法，不能替换原鱼眼质量门；零confidence返回有限点更表明finite不是valid。FMPose3D的GT-root参与RPEA不允许直接用于未知根位置的实际推理。RiemannianFM现有S2不是人体SO3，6D旋转合法也不等于SO3概率流。

收敛为现有H8研究参考：同源原鱼眼条件/显式COCO17观测与52rotationquery分离 → 现有空间编码及H5手部局部memory → H6单向GRU显式carry/无当前观测冻结hidden → H7原preGRU局部特征在8个admitted帧池化共享10D beta并冻结 → H8 readybeta经无偏置10to64残差进入原condition → 原8Euler/guarded6D局部旋转 → 真实maleSMPLH解码及严格availability发布。167955参数，其中native手条件新增23616、H8新增640；关闭H8精确回到H7，已存在可审计单因素对照。base config默认22观测不是本合同：H8 runner显式observation_joints=17/COCO17_SURFACE，不能直接套默认旧infer入口宣称H8完成。

关键选择依据是契约而非精度：H7隔离非admitted早期信息进入shape池化，但局部帧不代表统计独立、K8不代表可辨识；H6避开null编码隐藏态漂移，但长缺失压缩了物理时间，不能称已处理时间；H8证明beta→pose→mesh路径有响应/梯度，并未证明共享shape准确或耦合有收益；native手部条件只作用声明的局部query，不能解释成手/接触改善。保留图与空间attention为现有参考结构和可隔离因素，没有证据宣布graph/history优于简化对照。已保存360帧324available/7pending/29no-current，3031严格点、所有拒绝与NaN保留；availability不是精度通过率。

历史同1000合成步/144019参数：FM110.4661mm、8NFE；确定性回归50.0678mm、1NFE。仅该人工任务内部一致性支持回归为更强对照，未证明真实双鱼眼性能，不能推出FM优于回归。no-graph99.7432mm且参数135699，不能归因graph有害；no-history98.2499mm也未建立真实时间规律。研究参考保留FM以表达生成任务，确定性回归必须保留为对照；若只追求单一人体点估计，当前证据不支持付出FM额外采样成本。多样样本、finite密度、smooth或双视角一致不能替代独立准确度/校准证据。

最终交付是一套已实现、接口与依赖清楚、可复核的研究候选及十篇边界分析，不是已训练可用的真实人体恢复系统。14资源缺口、完整作者路径、SO3替代、elapsed time/部署、独立三维与接触/体型精度保持未执行，不继续追补也不消除历史失败。最新实际回归证据R16全套182passed/21warnings、Ruff/72AST；原GPUmesh重建0、独立CPU2.384186e-7m仅内部一致性。本次只综合已有结果，无新实验/训练/权重/主线改动。六项记录同步，本地提交，不推送。


### 2026-10-11 同规模长期改进：白化SVD体型混淆审计

用户授权继续长期改进，要求参考已有研究、不堆模型规模，接受暂时无提升。flow_next_longterm_protocol_v1.json锁定full52的width128/heads4/blocks4/711443参数，一次只改变一个机制，同规模强对照，沿用既有独立held-out预注册门与每机制两轮修订限制。新聊天目标已启用；每6小时heartbeat仅在实质完成、失败或需要输入时通知。训练及神经权重加载仍关闭，冻结正式主线不变。

首轮依据LAPACK秩亏最小二乘SVD参考，新增白化J的直接SVD消去姿态/平移干扰审计与五档阈值敏感性，保留旧Gram Schur及单阈值结果；生成器、Lambda、A6/A7决策未修改。解析反例中弱干扰方向与体型完全混淆，Gram rtol1e-6误留体型信息1，J rtol1e-6残差0；对应Gram阈值应为1e-12。正交观测坐标变换、干扰列置换、独立体型方向及零信号检查通过；32项累计测试与Ruff通过，见flow_next_shape_audit_v1.json。不支持真实体型可辨识、精度或速度提升；新增诊断有额外计算，旧耗时不代表新版本。下一轮复用既有固定候选，审计有限差分步长与proxy噪声精度敏感性，不生成新候选、不训练。
