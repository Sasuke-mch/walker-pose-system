# 条件 Flow Matching SMPL-H baseline

这是隔离的研究模块，不参与正式实时主线。先读20篇原始正文后确定设计，依据和失败记录在 [研究目录](../../research_records/research/G20261007_flow_mesh_baseline_v1/)。需要本地有许可的SMPL-H与匹配MANO资产；不会自动下载数据或模型。

当前推进十篇作者结构研究，按用户要求不训练。下方训练命令属于历史合成baseline，不是当前任务。原model.py与config.json仍为冻结对照；H1前缀共享体型候选在shared_shape.py，H2接口适配在interfaces.py，两者尚未替换主线或原模型默认行为。阶段状态和范围见研究目录architecture_research.json。

## 网络与坐标

观测二维点/置信度、存在mask、相机原点/单位射线、可选RGB或图像特征 → 52个旋转查询cross-attention → 单向GRU历史 → 条件速度场（帧内全身attention、可切换运动树GCN）→ 6D终点投影 → 真实SMPL-H网格。人体体型beta由初始帧估计后冻结，平移逐帧回归。历史/图开关在config.json。另有同条件编码的确定性回归入口。

旋转状态是 `(R6D - identity6D)/0.25`，6D存**旋转矩阵前两列依次拼接**，不与前两行惯例混用。52节点依次为根、21身体、15左手、15右手；图父子表读取真实模型。人体层直接接收绝对局部旋转矩阵，不加MANO均值；MANO字段仅用于官方SMPLHLayer资产构造，未把单位PCA当手先验。第一版模型资产为male，10维shape，6890顶点/13776真实三角面。

采样为固定步数Euler，默认8 NFE、1候选。不同噪声产生候选，同一序列各帧共享噪声以避免人为随机抖动；这不是时间准确性证明。训练FM采用随机时间线性CFM路径；未知姿态标签节点保留噪声并不参加目标损失，不假造手姿标签。可选mesh辅助损失作用于单步终点估计，不是精确积分终点，更不是接触约束。近共线6D/非有限旋转保留invalid和NaN，不静默替换。

第一版不是SO(3) RFM，不是论文完整复现。小CNN随机初始化；若传冻结特征，其权重/来源要写入数据metadata。当前没有人体真实RGB数据训练、预训练视觉骨干或项目鱼眼接入。

## NPZ输入契约

维度 `N,T,V,J` 为样本序列、帧、视图、观测关节。默认J=22使用SMPL-H前22运动学关节；COCO17须改config并在数据适配器声明拓扑，不能偷偷当同一组点。

|字段|形状|说明|
|---|---|---|
|xy|N,T,V,J,2|坐标归一化方式写metadata；几何射线须来自原始成像坐标|
|confidence|N,T,V,J|0..1参考值；不硬乘到趋零损失|
|observation_mask|N,T,V,J|bool，真正缺失；低置信度不自动等于缺失|
|view_mask|N,T,V|bool，视图可用状态|
|ray_origins / ray_directions|N,T,V,J,3|米制公共参考系原点及单位方向|
|geometry_mask|N,T,V,J|bool，射线是否存在；可与二维存在不同|
|image_mask|N,T,V|bool；图像/特征是否存在|
|images 或 image_features|N,T,V,3,H,W 或 N,T,V,F|可选其一，float RGB [0,1]；F由config指定|
|rotations / pose_mask|N,T,52,3,3 / N,T,52|训练绝对局部旋转标签及bool；缺失不能作identity真值|
|betas / shape_mask|N,10 / N|共享体型及bool标签mask|
|translation / translation_mask|N,T,3 / N,T|米制平移及bool标签mask|
|metadata|scalar JSON string|来源、身份、坐标、单位、拓扑、split等|

metadata必含label_source/reference_frame/units/rotation_convention/subject_ids/sequence_ids/split/observation_topology/geometry_source/image_source。units=metres，rotation_convention=absolute_local_matrix_columns；每样本有被试/序列ID。训练、验证和测试跨被试/序列不重叠。推理允许不提供全部目标字段；评估若提供标签则必须包含完整目标字段与mask。输入中的不存在项可NaN，存在项非有限会报错，禁止隐式清理成可用观测。

鱼眼适配由上游标定把**原始鱼眼像素**转成参考系单位射线，本模块没有把透视P矩阵直接用于鱼眼。真实相机ID、同步、左右身份、Stage及动态地面仍由正式工程定义；本研究不更改严格三角化质量门。

## 可复现命令

从仓库根运行，Windows PowerShell 7。使用已有CUDA环境；CPU可复制配置改device，无需下载资产。输出目录必须不存在，防止覆盖失败证据。

```powershell
$env:PYTHONIOENCODING='utf-8'
$py = '.\.venv-cuda\Scripts\python.exe'
$rec = 'research_records/research/G20261007_flow_mesh_baseline_v1/outputs'
$cfg = 'research_records/research/G20261007_flow_mesh_baseline_v1/config_learning1000.json'
& $py -B -m realtime_app.learning_flow_mesh.synthetic --config $cfg --output "$rec/my_data_replay"
& $py -B -m realtime_app.learning_flow_mesh.train --config $cfg --data "$rec/my_data_replay" --output "$rec/my_flow_replay" --mode flow
& $py -B -m realtime_app.learning_flow_mesh.train --config $cfg --data "$rec/my_data_replay" --output "$rec/my_regression_replay" --mode regression
& $py -B -m realtime_app.learning_flow_mesh.prepare_verification --config $cfg --output "$rec/my_verification_inputs" --seed 1234
& $py -B -m realtime_app.learning_flow_mesh.infer --checkpoint "$rec/my_flow_replay/checkpoint.pt" --input "$rec/my_verification_inputs/fresh_test.npz" --output "$rec/my_flow_inference" --seed 99 --steps 8
& $py -B -m pytest realtime_app/learning_flow_mesh/tests -q
uvx ruff check realtime_app/learning_flow_mesh
```

合成数据为真实人体层解码的人工低维相关旋转族，2个透视相机、3通道关节heatmap代理图像；不是自然人体RGB或鱼眼。64训练、16验证、16测试序列，每序列4帧，ID分开。验证CFM使用固定噪声/时间，推理独立seed，不用真值选候选。结果含标签一致性的顶点误差和打乱观测对照；不得据其声称真实人体恢复精度或泛化。

流式API：`encoder(batch, hidden=previous_hidden, fixed_betas=initial_betas)`。首次帧无观测且无图像时shape_available=false；后续可在观测到来时初始化新的前缀，不追溯填补先前失败帧。单纯滑窗重置hidden/beta不等于持续在线状态，需要调用者携带它们。transport返回rotation_valid/frame_observed，推理产物逐帧保留frame_valid及NaN失败。完全缺失帧可由历史先验生成，不能称观测支持的真实姿态。

## 验证范围

测试检查旋转正交/行列式、零/近共线拒绝、未来扰动不泄露、前缀与流式等价、缺失NaN不污染、全缺失状态、零置信度仍使用坐标、真实网格输入梯度、网络→ODE→网格梯度、checkpoint/seed一致性和划分拒绝。训练损失下降仅是基础学习证据。真实研究下一阶段需要有许可的公开/合成规模数据及真实场景独立测试；当前360/448连续工程拟合不能充当独立三维真值。

最新验收使用上述1000步配置；同容量回归、关闭graph、关闭history均执行，详见研究目录IMPLEMENTATION.md和verification_summary.json。21项测试通过。FM fresh合成顶点一致性110.47mm，回归50.07mm；图/历史没有在这组人工任务证实收益。FM全训练重跑、fresh推理与输入重建均逐位相同（限本环境）；全缺失64帧保留无效。默认模块config.json的200步仅为快速检查，不等同最新1000步结果。

## 不训练的结构与接口检查

原组件运行器在reproduction目录，不能把组件前向当完整论文复现。H1与H2累计后全模块54项测试通过；没有重新训练。H2逐项声明COCO表面观测、MotionBERT H36M顺序、SMPL-H旋转节点、6D排列、掩码极性、原始/转正像素域及相机单位。严格质量门来自冻结文件，射线能计算不代表严格接受。H2读取360帧，保留29个关联失败帧；这些帧不给隔离候选指定人物。现有身体软观测读取器另有契约，未改变。

```powershell
& .venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.check_interfaces --output research_records/research/G20261007_flow_mesh_baseline_v1/architecture_h2_results_new.json
& .venv-cuda/Scripts/python.exe -B -m pytest realtime_app/learning_flow_mesh/tests -q
```

结果JSON与同名NPZ不覆盖已有版本。NPZ含完整帧/物理pair编号、人物索引、原始像素、归一化像素、置信度、cam0米制光心与单位射线、观察/几何/严格掩码、拒绝原因、quality_flags及有限拒绝三维诊断值。未执行真实图像网络前向、拟合或精度评价。

S1已增加FMPose3D原Euler/flip-TTA/RPEA输入依赖、RiemannianFM原1000步球面采样检查，全模块累计60项测试通过。`reproduction.run_sampling`分别使用`--paper P03 --device cuda`或`--paper P02 --device cpu`，并显式给新`--output`；`reproduction.audit_mesh_resources`输出各原网格路径的资源缺口。RPEA发布评估路径读取真值根，不能直接作为无真值在线组件；有限fallback不代表几何可用。原默认neutral人体及专用回归器仍缺失，未用现有male/SMPL-H或COCO回归器冒充。

M1补齐P16原64³体素/V2V和原RANSAC完整前向，累计66项测试通过。`reproduction.run_multiview --kind volume --device cuda --output <new.json>`；RANSAC先不带兼容开关保留Python集合采样失败，再用`--kind ransac --python-set-compatibility`显式运行兼容版本。使用人工四相机针孔输入、随机权重，不是原数据性能复现。体素要求外部预测中心，并从batch相机重建投影；RANSAC原阈值为半欧氏距离，双目强制保留两视图，输出confidence全零占位，不能用来替换项目严格门。详见`architecture_multiview_validation.json`和研究记录M1节。

H3在`candidate.py`组合H1/H2和原生成器，明确使用COCO17随机模型，不加载J22合成checkpoint。`reproduction.check_joint_candidate --device cpu --output <new.json>`完整运行既有360帧保存契约→52旋转→真实SMPL-H→COCO17表面回归。前7帧体型待校准及29个无观测帧保留不可用；内部推测另存diagnostic，原严格掩码/拒绝XYZ不作网络目标。K1/K8只改变共享shape读出；姿态与平移相同。新增9个契约/数值故障/梯度测试，全模块75通过。详见`architecture_h3_validation.json`；仅真实输入接线和随机结构，不是准确性或实时结果。

H4新增`hand_conditions.py`和`reproduction.check_hand_conditions --output <new.json>`，按原WiLoR读取器复现360帧全部原生手姿选择及拒绝记录。左右镜像仍由原`convert_record`完成；输入为15个父相对旋转，不能以21个OpenPose投影点代替。保留两个相机的独立假设，6D列拼接加检测框置信度形成`B,T,V,2,15,7`条件。原零权重identity及H2人物不可用视图均保持missing/NaN；source-present与候选admitted分别保存。原主线763个接受手/视图槽中，候选准入706个、另外57个有源手姿但H2视图不可用。此阶段只增加类型明确的手部输入，没有把它接入生成器，没有改H3/正式主线。

`architecture_h4_validation.json`记录763项原始记录的独立镜像/权重核对、17个H2字段原样保存、未来隔离及6D往返。新增15项测试，全模块90项通过，Ruff/AST通过。原共享PCA手姿初始化用整段视频，这是离线设计；新逐帧条件没有这种未来依赖。初次audit因检查器误用float64身体像素失败，保留v1，按原pipeline先精确核对raw像素再cast float32，v2/v3通过。双视图旋转分歧是模型一致性诊断，框置信度不是关节不确定性；不据此声明准确手形、抓握或实时部署。

H5在`hand_candidate.py`接入原生手部条件：两视图独立memory含6D/框score、camera ID及30个解剖节点ID，同手15queries读取同手memory，只把残差加到对应30手节点condition，再由原52节点生成器传播。全缺失的手跳过attention并贡献精确零；输入不直接覆盖输出旋转，不进入共享β/平移/GRU。`candidate.py`只增加编码委托接口，其他状态及发布逻辑复用；关闭手条件的完整360帧结果与旧H3保存结果完全相同。

命令：`-m realtime_app.learning_flow_mesh.reproduction.check_hand_candidate --output <new.json>`，CPU2threads，必须已有H2/H4保存契约和本地人体资产。新增分支23616参数、候选总167315；新增12测试，全模块102项/Ruff/AST通过。完整360/分块/未来/真实解码与独立重建检查见`architecture_h5_validation.json`。随机条件影响不能当精度改进；两次分支CPU中位数0.650/0.754ms不包含检测/人体解码/Stage，也不代表整套实时性。首轮误取COCO regressor一行的运行器失败保留；没有训练、模型权重更新或主线更改。原方法默认资源缺口、null帧GRU、shape池化和eager部署限制仍在。

### R1 原辅助资源本地验收

七个已有官方regressor/mean通过schema、人工6890顶点线性/平移/梯度检查；本次不联网。命令`python -B -m realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources --local-only --output <new.json>`；网络入口拒绝记录中已有两次的第三次尝试。归档以唯一目标集合确定完成，EOF缺项和重复成员保留。

实际全模块124测试、14warning及Ruff/40文件AST通过。两个官方mean与MMPose fixture字节一致仅限明确对象；mean根退化与观测codec门不同，不改门。当前architecture_full_path_resources_v3.json仍缺14项，六条默认neutral路径未运行；validation保持whole_goal_complete=false。无训练/权重加载、无主线变化，无精度结论。

### H6 null-frame历史对照

独立`missing_history.py`只改变无当前观测时的原GRU hidden更新；不改H5/core文件或shape池化，关闭策略与保存H5完全一致。真实360保留324可用/7校准/29无观测与NaN；freeze在缺失hidden变化0，原控制最大0.547084。

135测试/14warning及Ruff/43AST通过。缺失1/8/64后恢复相同、帧号10倍无响应说明freeze忽略经过时间；没有准确性优胜结论。临时eager hook先运行原GRU再复算，仅验证harness，无并发/编译或实时性能结论。运行`python -B -m realtime_app.learning_flow_mesh.reproduction.check_missing_history --output <new.json>`；结果/validation在已有研究目录，长期目标未完成。

### R2 现有male原库网格路径

`check_existing_smpl_variant`使用用户授权现有male标准SMPL，运行HuManiFlow原随机网络点估计/50候选→原SMPL包装器。输出6890顶点/13776面/90原节点，保存参数独立smplx重建精确，三个原regressor块独立误差<=1.19e-7m。

仅替代male库子路径：作者CLI非neutral实际NotImplementedError，默认neutral/原图像proxy/渲染/评价未做。135回归测试及实际原子路径独立验证通过、Ruff/44AST；默认14资源缺口仍在，无训练/权重加载/主线变更/准确性结论。命令`python -B -m realtime_app.learning_flow_mesh.reproduction.check_existing_smpl_variant --output <new.json>`，JSON及同名NPZ保留不覆盖。

### R3 原HuManiFlow图像proxy路径

固定保存frame0经原HRNet-W48、crop/Canny/Gaussian及predict源AST十二句到原50候选/共享shape，再解码显式male库网格。141测试/15warning及Ruff/46AST通过；独立保存参数50mesh重建精确。R2库默认允许候选shape变化，R3按原图像路径use_shape_mode_for_samples=True，两者不混记。

原zero/negative HRNet热图仍强制7通道可见并产生Gaussian，maskedNaN仍NaN；不是有效观测门。不改作者/主线、无训练/权重，defaultneutral/renderer/完整函数/CLI/检测评价仍缺。命令`python -B -m realtime_app.learning_flow_mesh.reproduction.check_original_proxy --output <new.json>`，结果只证明原子路径接线。

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
