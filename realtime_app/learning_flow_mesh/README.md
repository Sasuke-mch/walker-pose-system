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
