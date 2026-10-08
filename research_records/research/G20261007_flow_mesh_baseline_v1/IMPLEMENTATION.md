# 通用条件FM人体模型baseline实现记录

2026-10-07。P01–P20方法/主要实验分析与设计门先完成，然后实现；文献依据见LITERATURE_ANALYSIS.md及papers.json。隔离代码realtime_app/learning_flow_mesh/，没有接入正式reference25或Stage/地面链。

## 设计依据与选择

|决定|原文依据|第一版边界|
|---|---|---|
|线性CFM，预测速度，Euler积分|P01、P03|Euclidean6D；SO(3)RFM P02留作对照|
|SMPL-H旋转参数生成，真实人体层|P04/P05/P06/P19；与P17/P18自由顶点对比|52旋转，male资产，shape/translation独立|
|全身attention+运动树图可切换|P03/P14/P18/P19|节点为旋转，图来自资产；不是创新结论|
|图像及二维/射线条件cross-attention|P07/P15/P16/P20|小CNN无预训练；鱼眼射线需上游标定适配|
|单向GRU历史|P10/P13，反查P09/P11/P12/P14未来帧|prefix/stream测试，shape只初始帧冻结|
|缺失mask与confidence分离|P12/P16及当前用户置信度需求|0置信度仍是输入，不丢弃存在观测|
|同条件确定性回归|P04/P06/P19评估候选预算问题|同训练步数，实际CFM与回归目标不同，不能把损失值横比|
|标签来源/被试划分显式|P08/P16/P18/P20|人工SMPL-H合成工程检查；不认定真实精度|

旋转前两列表示、归一化、父子表、形状共享、输入字段和完整命令见模块README。新增body.py/rotations.py/model.py/data.py/synthetic.py/train.py/infer.py/config.json及独立tests；已有主线代码未修改。

## 实际资产与环境

- Python：仓库.venv-cuda/Scripts/python.exe；PyTorch 2.11.0+cu128，CUDA可用；已有smplx/numpy/pytest。
- PowerShell：7.6.6。Ruff用uvx隔离缓存运行，不修改训练环境依赖。
- SMPL-H：third_party/WiLoR/mano_data/models/SMPLH_male.pkl，匹配MANO_LEFT.pkl/MANO_RIGHT.pkl；实际顶点6890、三角面13776、父子节点52、beta10。未下载受许可数据/模型。
- 首次直接加载因旧chumpy inspect.getargspec失败，局部legacy API兼容后读取真实资产并补匹配MANO字段。没有修改资产或伪造手PCA。官方SMPLHLayer接矩阵，不对手姿重复加均值。

## 检查与试验记录

- 第一次pytest：16通过、1失败。近pi参考由官方batch_rodrigues生成带epsilon数值偏差（2.415e-8）；更换为double torch.matrix_exp精确SO(3)参考，原1e-8容差未放宽。
- 修复后17项测试通过，保留4条旧资产scipy/chumpy弃用warning；Ruff初次3个import排序错误，自动整理本模块后通过。失败详见validation_attempts.json。
- 合成数据outputs/synthetic_v1/：64/16/16独立ID样本、每样本4帧、双透视、22身体观测、52旋转真值、不同视图/点缺失、0.25..1参考置信度、带噪声射线、关节heatmap代理RGB。这个易学习的人造相关旋转族不是自然姿态数据分布，也没有真实鱼眼/遮挡数据泛化证明。
- 训练/测试结果在实际执行后追加；不得预填通过。默认200步、batch8、lr0.001、seed7；CFM/回归同条件、同优化预算，mesh辅助0.02、shape0.1、translation1，采样8步1候选。mesh辅助作用单步终点估计，可能偏向单解，不声称分布校准。

## 尚未执行/待扩展

未进行真实RGB/鱼眼训练、未复现20个作者仓库、未评估公开人体恢复排行榜；SO(3)路径、完整视觉heatmap骨干、正式数据消融、概率校准、重投影guidance、全局世界运动/接触均不是当前验收必要项。手部二维缺失时生成依赖标签中学到的先验，不宣称准确手部。需后续先明确标签/数据许可和独立场景拆分，再做这些研究；本工程baseline不得替代原有质量/Stage审计。

## 已执行的训练与对照

200步v1结果全部保留，CFM验证1.571→0.251、原test顶点一致性285.2mm、打乱观测291.1mm，条件利用很弱。代码审查又发现回归分支绕过空间decoder，不能当同容量对照；修正为同一state/time projection、空间attention/graph和输出头，回归输入零状态与固定时间。v1重构源在outputs/historical_v1_source/，旧checkpoint仅属于该历史实现，不与新版本混用。

冻结新架构后，采用config_learning1000.json：width64、heads4、2层、seed7、batch8、1000步、lr0.001，其余默认不变。config_no_graph1000.json只变use_graph，config_no_history1000.json只变use_history。四组独立从头训练，同划分、优化步数和候选1；FM使用8 NFE，回归一次前向，计算预算不同，不能掩盖。关闭history时GRU参数保留但不参与前向；关闭graph时去掉对应图层参数。

原seed9 test曾用于工程诊断和预算调整，因此不将它包装成冻结科研测试。四组架构/预算完成后，另生成seed1234的16×4帧fresh_test1234.npz；测试后不再调架构/预算。其ID、生成流程和meta明确，但仍是同一低维人工分布的合成检查。prepare_verification.py可重建fresh输入及无标签全缺失case；数据数组、NaN状态和metadata重建完全相同。

|模型|参数数|各自验证loss初始→最终|fresh顶点一致性mm|打乱条件mm|NFE/候选|
|---|---|---|---|---|---|
|FM图+历史|144019|1.571231→0.053356|110.47|443.44|8/1|
|同容量确定性回归|144019|0.543262→0.003263|50.07|468.93|1/1|
|FM关闭图|135699|1.429791→0.045283|99.74|453.74|8/1|
|FM关闭历史|144019（GRU未激活）|1.440046→0.079666|98.25|459.53|8/1|

loss目标不同，不能跨FM/回归比较数值。fresh的4组均保留完整64帧，旋转合法、顶点有限，最大正交/行列式误差小于5e-7，网格6890顶点/13776面。打乱条件明显退化证明在此人造任务利用了观测；不能证明真实鱼眼图像可泛化。**FM未超过确定性回归；图和历史在本合成对照未带来收益。** 不能据此否定真实遮挡多解场景中的生成方法，也不能先认定GNN或FM就是研究贡献。该简单单解任务不验证多模态覆盖或概率校准。

### 可复现与失败保留

- flow_v2_replay按同配置从头再训练1000步，所有state tensors逐位相同；重新加载并推理fresh，state/旋转/beta/translation/真实顶点/关节/面逐位相同。保证限于本机同环境，未宣称跨硬件位级复现。
- all_missing_input.npz没有姿态/体型/平移标签，所有观测/图像mask=false、数值NaN；真实infer CLI保留64/64帧，frame_valid全false、顶点/关节全NaN、metrics为空，没有用旧结果或identity补齐。
- 最终21项行为测试通过，含RGB前缀梯度因果检查、缺失初始shape、zero-confidence观测仍影响条件、真实左右手/体型/平移梯度，以及network→ODE→真实网格反传。pytest_final.xml和日志在outputs/。4条旧chumpy/scipy弃用warning保留。
- 模块、来源抓取及结果审计脚本Ruff通过；Python语法解析通过。来源抓取首次扩展linter发现3条literal/异常类型诊断，已修；重复读取缓存不得重置阅读状态，papers.json行数与当前抽取文本同步。
- verification_summary.json由audit_results.py重新读取各训练日志、checkpoint、推理NPZ、pytest XML、配置和文献记录生成；它是工程一致性核对，不代替逐篇阅读或独立准确性验证。

### 已执行命令

从仓库根使用.venv-cuda/Scripts/python.exe -B：synthetic --config realtime_app/learning_flow_mesh/config.json生成outputs/synthetic_v1；train --config config_learning1000.json --data outputs/synthetic_v1 --output outputs/flow_v2 --mode flow以及同预算regression_v2；两个开关配置对应flow_no_graph/flow_no_history；原配置重跑到flow_v2_replay。上述train均是模块realtime_app.learning_flow_mesh.train。infer用对应checkpoint、--input outputs/fresh_test1234.npz、--seed 99 --steps 8，输出fresh_flow/fresh_regression/fresh_no_graph/fresh_no_history/fresh_replay；全缺失输入输出all_missing_infer。所有相对路径均以本实验目录为前缀，完整可复制重放命令见模块README；输出目录必须改新名称以保护证据。

## 对实际目标的验收

|要求|直接证据|结论|
|---|---|---|
|先分析20篇再实现|逐篇正文/锚点/cards，文献阶段记录先于代码|方法和主要实验阅读完成，附录/视觉全图/全部作者代码不在已读范围|
|人体模型而非仅骨架|真实BodyDecoder、faces、4组prediction.npz|6890顶点/13776面实测，非替代网格|
|可训练/保存/加载/推理|4组1000步logs/result/checkpoint，infer NPZ，完整重跑|合成工程学习完成，FM真实数据优势未成立|
|旋转合法性|精确SO(3) roundtrip和退化测试，fresh输出逐矩阵核对|合法输出；退化保持无效|
|因果性与共享体型|未来扰动/prefix/stream/RGB梯度/初始缺失shape测试|过去输出不接触未来；beta前缀冻结|
|缺失与低置信度|NaN masked-input测试、缺失手标签、zero-confidence、全缺失真实CLI|缺失/低置信度分开，失败保留|
|可微真实解码|真实左右手/体型/平移梯度及ODE→mesh测试|梯度有限且非零|
|主线不替换/本地提交|代码仅新增隔离目录，AI_PROGRESS和本实验更新；提交前暂存检查|未改Stage、地面、current_mainline或现有拟合|
|记录和结论边界|本文件、协议、papers/progress、失败及verification_summary|实验只证合成工程与标签一致性；无独立精度声明|

本轮目标是建立可研究的通用baseline，已完成相应实现与基础验证；真实人体数据训练和新的科研结论须另立数据/评估协议，不把它们偷换成当前通过项。

实现与小型记录已单独本地提交，既有89条状态保留；完整原文/资产/数据/checkpoint/网格NPZ留在本地，未提交或推送。停用hourly请求返回自动任务不存在，标准本地automations目录无对应条目；未重新创建任务，progress记为ABSENT_ON_COMPLETION，而不是伪称已暂停成功。

## 2026-10-08 架构图

按用户要求绘制当前已实现Flow Matching人体模型的论文式架构图，输出在本目录`figures/architecture_en.{png,svg,pdf}`与`architecture_zh.{png,svg,pdf}`。复现脚本为`realtime_app/learning_flow_mesh/draw_architecture.py`，使用已有1000步配置和`outputs/fresh_flow/prediction.npz`第0序列第0帧的真实SMPL-H顶点/面；没有重新训练、推理或改正式工程主线。

实际查看了已下载论文正文中对应的原图：ProHMR Figure3（条件编码、姿态生成和形状分支）；FMPose3D Figure1（高斯初值、线性CFM路径与速度监督）；MotionAGFormer Figure2（总览加模块放大、全局注意力和局部图）。原图参考副本仅保存在`figures/references/`，来源和借鉴范围记录在`figure_sources.json`。最终图为原创重绘：本实现是欧氏6D CFM、单向GRU、同帧注意力加可切换运动树图，不是上述任一论文完整复现；不复制ProHMR可逆似然分支、FMPose3D多候选聚合或MotionAGFormer时间GCN。

图分(a)条件编码—速度场—8步Euler—6D转SO(3)—固定可微SMPL-H，形状/逐帧平移为独立回归分支；(b)两层空间模块中的全局注意力与局部图并行残差结构；(c)带标签掩码的CFM和形状/平移/单步终点网格辅助损失。52节点为根1、身体21、双手各15，22输入观测关节不能冒充COCO17。首帧共享beta与原工程全片优化beta是不同策略。Stage/地面/接触未画入学习网络。

Matplotlib精确生成可编辑SVG及带矢量框线文字的PDF，人体缩略图由真实三角面渲染并嵌入，以避免13,776面导致编辑文件过大。PNG为4320×2520。两种语言均实际查看，文字边界自动检查、配置契约检查和源网格尺寸检查通过；Ruff初轮发现import排列及pairwise风格问题，修复后通过。未运行模型测试或产生新精度指标。

复现：从仓库根执行`.venv-cuda/Scripts/python.exe -B realtime_app/learning_flow_mesh/draw_architecture.py --output research_records/research/G20261007_flow_mesh_baseline_v1/figures`。配置契约变化时脚本拒绝继续生成陈旧标注，需核对新代码并同步图。

## 2026-10-08 十篇原结构复现：源码及首批运行

协议锁定后执行。新增隔离运行器`realtime_app/learning_flow_mesh/reproduction/`：fetch保存十份原代码/配置/许可及来源清单；run_original运行P03/P12/P14原神经网络；run_components运行P09/P10原视频组件与P16原代数网络。源码仅在third_party本地保存，没有复制进维护代码，也没有安装旧torch/torchvision到现有环境。

全部为CPU、随机初始化、eval/inference_mode、seed19，不训练、不读取现有被试图像、不比较随机输出精度。Python3.12.12、torch2.11.0+cu128。作者旧timm导入warning保留。run_original的整个过程时间含构造/诊断，不是推理延迟；没有整体实时认证。

|实际运行|构造/输入|已验证行为|未覆盖|
|---|---|---|---|
|P12 DSTformer|原H36M配置512宽/5层/8头；1×243×17×3；42,466,317参数|输出1×243×17×3有限、重复逐位一致；改变未来半段，过去最大变化0.110636；一个缺失NaN造成12,393个非有限输出|SMPL mesh head、真实精度、项目缺失/因果适配|
|P14 MotionAGFormer xsmall|原配置64宽/12层/8头/27帧；2,241,531参数|输出有限/重复一致；过去变化4.10e−6，虽小仍非因果；NaN污染全部1,377输出|训练效果、因果/52节点适配|
|P03 FMPose3D Model|原512宽/5层/17节点；16,045,837参数；原单帧条件2D/状态3D/时间1|输出速度有限/重复一致；附加5帧检验过去变化0；NaN只污染该帧51输出|作者完整ODE、RPEA、多候选选择和mesh/beta|
|P09 TemporalEncoder|发布config.yaml：2层/1024隐层/单向/线性+残差；1×16×2048|原配置未来扰动过去变化0；明确改双向的诊断0.213605，不能混作发布配置|完整回归/人体层；原组件按AST加载，未改类代码，绕过资产依赖顶层导入|
|P10 WHAM组件|原model_base配置；Network(smpl=None)只构造，原MotionEncoder/Decoder/TrajectoryDecoder前向；1×8×37|过去pose/keypoint/beta变化均0；beta为1×8×10，同序列最大变化0.000723；缺失点NaN通过原preprocess清除|没有假SMPL替代真实人体层；全Network.forward、脚世界坐标/轨迹精修仍待资源|
|P16代数网络|原ResNet152、384×384、置信度头；79,521,890参数；1×2×3×384×384|完整原分支输出1×17×3及双目heatmap/置信度有限；原DLT已知3D点最大误差9.54e−7m|只关闭预训练初始化，没有改变结构/分辨率；体素/RANSAC/鱼眼适配/严格拒绝门未执行|

P03通过namespace package路径绕过高层detector/动物/自动权重下载facade，原Model/Graph未修改。P09只编译实际原TemporalEncoder AST；不能把这一组件检查称完整VIBE复现。P10保持原配置字段，但实际Network根据17关节计算37维输入，配置in_dim=49不决定该构造器的输入；layers/modules读取rnn_type默认LSTM，配置layer字段未自动等于接口名。这里记录实际构造而不是猜测字段含义。

复现命令（仓库根；输出必须不存在，改用新版本文件保留历史）：

```powershell
./.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.fetch --output research_records/research/G20261007_flow_mesh_baseline_v1/architecture_sources_v2.json
./.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.run_original --paper P12 --output research_records/research/G20261007_flow_mesh_baseline_v1/architecture_checks/P12_v2.json
./.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.run_components --paper P16 --output research_records/research/G20261007_flow_mesh_baseline_v1/architecture_checks/P16_v2.json
./.venv-cuda/Scripts/python.exe -B -m pytest realtime_app/learning_flow_mesh/tests/test_reproduction_fetch.py -q
uvx --offline ruff check realtime_app/learning_flow_mesh/reproduction realtime_app/learning_flow_mesh/tests/test_reproduction_fetch.py
```

原source acquire前验证全部ZIP成员再写入，拒绝目录逃逸、Windows驱动路径和反斜杠；未完成/不一致缓存不覆盖且记录失败。7项安全/缓存保留检查通过。初次linter的2项import已修正；执行已审查原类的AST和异常报告边界有局部注明，仍返回失败退出码，不吞错假报通过。当前cuda环境未安装ruff，使用已缓存uvx离线版本；没有为此更换环境依赖。

下一阶段：P04/P05/P07/P02原概率/参数/流形网络；随后补齐真实人体层及正式接口。核心SMPL-H学习baseline及reference25主线没改。H1因果多帧共享体型与H2语义适配仅有协议，尚未实现；十篇复现和架构优化长期目标保持进行中。


## 2026-10-08 概率姿态和参数回归原结构检查

运行器：`realtime_app/learning_flow_mesh/reproduction/run_probability.py`。每篇一个独立进程，随机种子19，float32，不创建优化器，不反传更新，不运行训练。依赖只安装到`third_party/architecture_reproduction/_deps`，版本、来源、作者nflows快照和均值文件见`architecture_probability_resources_v1.json`。公开MMPose均值文件与本地副本相同，但未证实与两篇原发布包一致，不称完整原包重放。

| 论文 | 实际覆盖 | 检查结果 | 未覆盖 |
|---|---|---|---|
| ProHMR | 原ResNet50、144维ConditionalGlow、β/弱透视头、63随机＋1零潜变量 | 51,513,389参数；逆变换最大误差2.38e-7；密度往返差0；64候选β一致 | 原GT ActNorm初始化、neutral SMPL、完整损失/评估 |
| HuManiFlow | 原18通道256×256代理、ResNet18、23个SO(3)层级流、形状分布、采样/密度 | 12,490,497参数；旋转正交/行列式通过；4候选默认β不同，shape-mode开关使β一致 | 检测/边缘代理生成、neutral SMPL、完整评估 |
| HMR2 | 原ViT-H/16＋6层Transformer头；RGB256×256按原代码裁成256×192 | 670,459,037参数；24旋转＋10β＋3弱透视参数；重复/旋转通过 | 检测裁剪估计、neutral SMPL、原checkpoint/评估 |
| Riemannian FM | 原球面geodesic/JVP、512宽8层time-MLP、切向投影、原rfm_loss_fn | 1,816,966参数；端点误差1.79e-7；目标切向误差2.38e-7；网络切向误差3.73e-9 | 原数据密度、ODE、Mesh/SPD等分支、人体SO(3) |

命令：`python -B -m realtime_app.learning_flow_mesh.reproduction.run_probability --paper P04 --device cuda --output <new.json>`；P05同样，P07加`--hmr2-backbone`，P02使用CPU。实际解释器`.venv-cuda/Scripts/python.exe`；输出为新文件，禁止覆盖。最终代码重放检查分别P04_v4、P05_v3、P07_v3、P02_v2；前序版本及失败保留。

失败保留：P04_v1骨干默认pretrained=True，加载ImageNet后因UMNN缺失失败，该JSON“random_initialization”字段不准确，以资源记录勘误为准，排除通过证据。最终显式pretrained=False，并在导入作者模型前拦截权重下载。P04_v2暴露作者固定多候选noise路径context行数不匹配；未改原源码，v3使用原发布63随机＋1零路径通过。P05_v1缺opt_einsum，隔离补依赖后v2通过。P07_v1原ViT.train不返回self，不能链式赋值eval；改运行器原地调用后v2通过。首次guard调用lazy model_zoo未显式import，在模型执行前退出，修正后捕获边界覆盖此处。资源记录生成一次未指定UTF-8而失败，源/结果未损坏，显式编码后重建。弃用警告和HuManiFlow非执行分支pytorch3d提示保留。

四篇都未验证真实三维精度。十篇选定组件通过不等于十个完整方法、项目接入或架构优化完成。


## H1：因果多帧体型校准候选（隔离、不训练）

代码`shared_shape.py`，验证入口`reproduction/check_shared_shape.py`。使用同一个既有ConditionEncoder和shape_head，无新增学习层或重新初始化已有参数。原编码器未暴露global_condition；检查适配器以临时只读hook取得GRU输出（关闭history时取原norm输出的52节点均值），finally移除hook。编码器源码、空间模块、姿态生成器、8次Euler预算、平移、所有主线配置未改。该hook只用于eager检查，尚无并发/编译/导出接口保证，不作为部署完成。

流程：输入B,T,64条件特征、调用方明确认可的B,T bool、B,T int64帧号及每行独立序列ID。状态保存特征和、有效数、β、ready和上一帧号。缺失值用where先隔离，不能以0乘NaN代替。前8个认可帧等权平均，用原Linear(64,10)读出一次β；此前每帧β=NaN、ready=False。第8个有效帧开始发布，随后不再更新特征和/β；缺失帧不计数。跨chunk继续同状态，换序列显式reset；重复/倒序帧、串序列、认可的非有限特征均报错，不猜测修复。不可用前缀仍保留，不生成或回填网格。

原模型联调使用真实SMPL-H/MANO资产生成2×13帧人工合成代理，网络随机种子19、生成种子31；没有调用训练或载入学习checkpoint。序列0第1帧缺失，序列1前3帧缺失，首次可用帧分别8、10。原条件/平移逐位不变，参数快照完全不变。模型逐帧和批处理β最大差5.96e-8；未来11帧后扰动造成已发布β变化0；冻结后β完全相同。只解码ready片段，真实网格分别1×5×6890×3、1×3×6890×3，有限。独立梯度检查确认有效校准特征与β头收到有限非零梯度，未来及缺失特征梯度为0。1帧控制在相同批形状下精确等于原公式；B=1/B=2 GEMM误差2.98e-8单独记录。对照改变β的L2=0.3578只是读出变化，不是改进精度。

验证：全模块38项测试通过（H1新增8项），Ruff/AST通过，原模型/训练/配置及主线文本不变。v1失败由广播数组写入别名导致，将输入clone后才注入缺失修复；v2对不同批大小过度要求逐位相同导致失败，保留误差并增加同批精确公式控制。v3/v4成功，其中v4关闭整个结构运行器的梯度记录；梯度检查只在单元测试中反传，不做参数更新。详见architecture_h1_results_v4.json及architecture_h1_validation.json；NPZ本地保存完整不可用前缀和掩码，不提交数组。

命令：`.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.check_shared_shape --output <new.json>`。输出同名NPZ，均不覆盖。既有条件编码器、历史GRU在缺失帧仍运行，H1只控制shape读出不改这一路；池化历史特征可能重复计入早期信息，尚未证明不偏或可辨识。8帧是工程阈值，需后续质量/时延研究；共享常数本身不证明体型正确。候选保持隔离，后续H2语义适配、原完整方法路径继续。

## 2026-10-08 H2：把作者组件接口接清楚

协议architecture_h2_protocol.json先提交后实现。新增interfaces.py和reproduction/check_interfaces.py，没有更改原model.py/train.py/config.json或current_mainline.json。

关节：COCO17_SURFACE为表面观测，现有17×6890回归器通过原验证；它不等于SMPL-H内置腕/肘节点。SMPLH52_ROTATIONS是根、21身体、30手的局部旋转。MotionBERT H36M17由其dataset_wild.py的halpe2h36m定义，右腿在左腿前，含pelvis/torso/neck/head_top；COCO没有全部这些输入，适配层拒绝因同为17而直接复用、补零或合成头部。WHAM实际检测输入来自COCO ViTPose路径，不拿kp_utils中另一个H36M列表替代它。

旋转：ProHMR/HMR2与当前baseline使用前两列顺序拼接；HuManiFlow默认、MotionBERT使用前两列交错；WHAM人体/root路径rotation_6d_to_matrix使用前两行顺序拼接，其同文件legacy rot6d_to_rotmat又是交错列。显式SixDLayout通过矩阵转换，不凭6维判断。20个非对称float64合法旋转对六个原函数作原样AST运行，结果与适配完全一致。故意将WHAM行布局交给列解码，仍是合法旋转，但最大矩阵差1.74154；正交/行列式检查抓不到这种接线错误。退化、共线、NaN的6D保持invalid/NaN，不替换identity；转换梯度单独检查。

掩码：项目True为存在，WHAM预处理True为缺失，MaskMeaning必须明示；不接受数值mask。相机：PixelDomain明确原始/转正，不能根据raw文件名猜测。复用现有model_to_raw_point和StereoCalibration，先回原鱼眼，再反畸变生成单位射线。R/T遵循Xright=R·Xleft+T；右光心cam0坐标为−RᵀT，右射线为Rᵀdright；mm转m只做一次。弱透视尺度/偏移不能改名成米制XYZ。像素坐标有效、射线反投影一致、冻结严格质量门为三个独立状态；逆变换不一致保留原因，不删除仍有效的二维观测。反投影往返1e-4px为隔离适配器内部数值检查，未替换或放宽原严格门。

真实保存输入：G20261003_repositioned360_cold_smplh_v2左右pmpose_raw.json及该目录calibration/stereo_fisheye.json，严格参考为G20261005_reference25_mainline_view_v2/first_instance_strict.jsonl。360帧0..359、物理pair1157..1516，共6120个逐点槽。331帧关联成功，左右人物索引均0；29帧association_failed，保留493个不可用槽，不给候选选第一人。左右分别162/203帧含多个检测候选，故索引核对必要。与原身体raw_side读取器在全部360帧的逆变换差为0；29帧的原身体软观测行为未修改，严格场景关联与身体软质量是不同协议。

3031个严格接受点与每个原拒绝reason照录；2091个高重投影误差点有限XYZ仍保留诊断，未变成接受点。其余原拒绝为低2D分数369、越界136；关联失败493为无逐点源条目的框架不可用状态，同时保存原pair_outcome。双视图像素/射线有效11118；单位射线范数误差3.33e-16；沿光线点重投影最大误差左6.85e-13px、右2.74e-12px，仅为内部一致性。cal单位mm、基线0.27591249m；世界/动态地面未接入、不借这套检查证明人体准确性。

命令：`.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.check_interfaces --output <new.json>`，结果带同名本地NPZ，保留所有槽、人物与拒绝状态且拒绝覆盖。v1通过；v2增加原置信度照录；v3补物理pair范围和置信度范围检查。单元新增16例，全模块54通过；初次故障注入尝试修改frozen校准实例失败，改为测试内临时类方法注入，不改校准类。Ruff初次导入排序及检查器边界标记修正后通过。详见architecture_h2_validation.json。完整原方法网格/输入/评估仍未全部完成；下一阶段先补可运行的原路径与资源清单，再将H1/H2接入一个声明COCO17、52旋转的隔离候选，不能把随机输出当改进精度。

## 2026-10-08 S1：原采样、聚合输入及网格资源

协议architecture_sampling_protocol.json先提交后执行。新增reproduction/run_sampling.py与audit_mesh_resources.py，原作者文件、原baseline和工程主线均未修改，没有优化器更新或学习权重加载。结果P03_sampling_v4.json、P02_sampling_v1.json及architecture_full_path_resources_v1.json。

P03使用原512宽/5层/17点GAMLP，参数16045837，单帧H36M归一化二维人工输入、固定三维噪声。实际执行原common/utils.py的Euler与原FMPose3DInference._run_euler_sample方法；只跳过加载器/检测器外壳，不复制改写模型。1/3/5指积分步数，分别1/3/5次速度前向，独立Euler算术对照差0；同噪声完全重复。发布默认3步，flip-TTA原两路独立噪声、左右重排/反翻转/均值，实际6次前向，同seed完全重复。并非整段视频因果推理或二维检测器复现，原helper输出为后处理前的原始17点。单元另运行原HumanPostProcessor(camera_rotation=None)，确认原地将根置0、其他已根相对的节点不变、world为复制；默认示例四元数/最低z平移没有接入项目。单个NaN腕输入导致原采样输出51个非有限值，完整保留，没有补帧。

P03 RPEA的实际输入比注释更多。aggregation_methods.py中gt_3D不仅用于B/F/J形状，还取gt_3D[:,pad,0]，把候选的非根点加上这个根，再用Human3.6M透视/径向切向畸变投影函数计算二维误差；原投影还clamp x/z,y/z到[-1,1]，不是鱼眼模型。load_data_hm36.py 86–90将非根点减根、保留根的绝对相机坐标；scripts/FMPose3D_main.py 135将该gt_3D直接传入聚合。因此这条评估路径使用了真值根位置，不能把代码里的shape-only注释当在线输入契约。

依赖对照：三个人工根相对17点候选、同一归一化针孔相机、同一二维目标，只有传入根深度1m→3m变化。硬选择输出最大变化0.4m，原topk=3/exp_temp=0.002聚合变化0.299993m；改动非根gt_3D到100，输出变化0。以上为人工依赖检查，绝不是模型在人身上的误差测量，也不据此否定论文整体。原RPEA逐关节权重可混合不同候选，不能默认保留人体骨长或合法旋转。

边界探针：代码将选中误差截断到20×temperature，若全部超过这个数，指数权重相同；远误差人工例退化为均值，最大差5.55e-17。根设NaN时，代码用uniform_weights代替NaN权重，仍返回有限均值，差1.39e-17；有限输出不代表有效几何。可选反传探针因原dist[:,:,0]原地修改norm结果失败，错误信息保留；这是原评估聚合的反传限制，不阻止其no_grad前向，不修作者文件、不偷换可微公式。P03_sampling_v1因把这项可选能力当必过条件整体失败；v2起独立记录前向结果与反传失败。v3增加NaN根探针，v4去掉沿用旧检查器的“五帧”注释，明确本次为单帧/多积分步。

P02使用原Sphere、512宽/8层切向网络1816966参数以及原sample/sample_all方法。按train.yaml的eval_projx=True/local_coords=False，保留1000次投影Euler，未缩预算。完整轨迹1001×4×3，last-only前向1000次，完整轨迹额外查询终点速度所以1001次。两条路径末点差0、同初始状态重复一致；全过程单位范数最大误差1.19e-7。已知赤道切向场旋转1弧度的float64解析对照误差2.81e-7。原网络state_dict完全不变。仅验证球面采样，不是volcano真实数据、似然评价或人体SO(3)概率模型；构造/检查总耗时不当实时延迟。

网格资源审计读取六个原SMPL路径及当前smplx默认gender，确认默认neutral。ProHMR/HMR2还需SMPL_to_J19和原mean；HuManiFlow需extra/cocoplus/h36m；VIBE需extra和mean；WHAM需wham/h36m/feet；MotionBERT需extra/h36m_correct和mean。默认路径共21项先决文件当前均缺失；models/third_party定向名称检索只找到项目COCO回归器，没有neutral SMPL及这些原回归器。现有male/female SMPL、SMPL-H、neutral SMPL-X及公共mean fixture列为不同资产/未确认等价来源，没有自动替换。WHAM作者下载脚本要求SMPLify账户许可，不读取凭据或尝试认证下载。审计完成不等于六个原网格方法完成。

命令：`.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.run_sampling --paper P03 --device cuda --output <new.json>`；P02实际验证使用`--device cpu`。资源命令为`-m realtime_app.learning_flow_mesh.reproduction.audit_mesh_resources --output <new.json>`；输出均不覆盖。新增6个原源码行为/解析控制测试，全模块60通过，Ruff/AST通过。初次测试将0.2×3与3/5要求逐位相同，差1.11e-16，改为明确1e-15算术容差；没有改采样器或几何门。下一阶段取得可公开获取的原辅助资源，并预注册H1/H2真实COCO17输入的隔离候选联调；不可取得的许可资产仍列缺口。不要直接照搬RPEA真值根、有限fallback、示例世界旋转或逐帧beta行为。

## 2026-10-08 M1：原多视图体素与RANSAC

architecture_multiview_protocol.json在实现前锁定。新增reproduction/run_multiview.py、tests/test_original_multiview.py；直接导入作者模型，关闭backbone权重初始化及自动权重下载，其余原网格/宽度/分辨率保留。人工四相机Kf384、图像384×384、光心为X/Y正负300mm、单位mm、无畸变；RGB为显式人工噪声。没有调用检测器、原数据集、预计算骨盆预测文件或训练权重，没有改作者文件、核心baseline或工程主线。

体素最终P16_volumetric_v2.json：原80,588,050参数，eval human36m_vol_softmax.yaml，softmax跨视图聚合、volume_size64、cuboid_side2500mm、原V2V。输入1×4×3×384×384；特征1×4×32×96×96；17个64³概率及1×17×3输出。人工pred_keypoints_3d提供MPII index6中心[80,−40,3000]mm，体素边界中心±1250mm。概率和误差5.96e−8，同输入重复差0，state_dict不变。无预测中心报KeyError，未伪造中心。移动输入中心X100mm，输出最大变化100.521mm；这只是依赖探针，不是误差指标。

原forward实际覆盖proj_matricies：保持batch.cameras，传NaN矩阵输出差0；保持传入矩阵，把batch相机cx增加20px，输出变化0.762mm。相机从batch深复制后缩放，原K未被改变。v1已通过；v2增加实际相机扰动对照。峰值allocated1,218,136,064bytes含检查用state_dict克隆、输出和复测，构造及复测总耗时也不是部署显存或实时延迟。原torch.meshgrid未来警告保留，不修改外部源码。

RANSAC P16_ransac_v1.json记录原random.sample(set)在Python3.12的TypeError。v2使用明确开关--python-set-compatibility：仅该模块random命名空间改为将整数集合排序为list的抽样接口，finally恢复；不改全局random.sample、作者文件、10次采样或几何公式。固定seed与排序序列抽样一致，但不声称与作者旧Python的集合迭代顺序逐位相同。完整原68,635,473参数ResNet152/heatmap argmax/CPU RANSAC/SciPy Huber直接优化通过，四视图输入，1×17×3及1×4×17×2输出有限，参数未更新。confidence是原全零占位，不作为缺失/接受判断；完整forward不导出inliers。

解析已知点[120,−40,3000]mm：四视图原求解误差2.37e−11mm；向第3视图添加[130,−100]px外点，保留0/1/2，误差4.37e−11mm；仅保留0/3两视图时两者被强制接纳，人工输出偏移3453.57mm。数值不是实际人体误差，只展示缺少冗余的行为。原calc_reprojection_error_matrix为0.5×欧氏像素距离，20px偏移返回10；默认epsilon15意味着欧氏距离约30px，不是15px。新增29.99/30.01px边界对照，不改默认阈值或项目门。原针孔投影也不包含鱼眼畸变。

执行命令：`.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.run_multiview --kind volume --device cuda --output <new.json>`；RANSAC先用--kind ransac不加兼容开关保存失败，再以新输出路径加--python-set-compatibility。同名NPZ本地保存，输出拒绝覆盖。新增6个原几何/兼容/体素积分测试，全模块66通过。首次测试AST环境遗漏原nn导入导致1失败65通过，补齐测试符号后通过；初次Ruff未用变量/导入格式诊断均已修正。失败结果/XML保留，见architecture_multiview_validation.json。

M1只补齐P16原三种求解分支的结构证据；输入构建、真实原数据、原检查点/指标、鱼眼适配、严格质量门与人体网格均不在这项证据内。后续隔离候选必须声明中心/相机来源和有效状态，不能把体素输出、confidence占位或双目inlier_count当成真实性。长期目标继续，六个原默认mesh资源缺项及候选组合尚未完成。

## 2026-10-08 H3：真实COCO17输入到完整人体接口

architecture_h3_protocol.json先锁定后实现。新增candidate.py、reproduction/check_joint_candidate.py、tests/test_joint_candidate.py；原model/train/infer/config、工程主线、作者源码及资产未修改，不加载学习权重、不运行优化器。使用H2 results_v3.json/npz保存的360帧0..359、物理pair1157..1516。完整原始像素、人物索引、掩码、拒绝原因、quality_flags、有限拒绝XYZ照录在输出source__字段；未重新关联、检测、三角化或改变质量门。

saved_contract_conditions要求显式COCO17_SURFACE/cam0/meter。检查T×2×17形状、布尔掩码、选中人物、严格门双视图一致以及帧号；原float64条件只在这里转换为模型float32，IDs从int32转int64。像素、confidence、严格接受射线/光心、view/image mask作为条件；没有原始三维XYZ、β、姿态标签或世界地面输入。改写诊断XYZ不会改变任何网络输入。0置信度若观测存在仍保留；没有加新的置信度门。image_mask明确全false，不把随机RGB或虚构特征补成真实输入。

模型fresh seed19，ModelConfig仅显式J22→COCO17，width64/heads4/layers2、图/历史、rotation_scale0.25/root_offset(0,0,3)保持原值，143699参数。旧J22人工模型checkpoint不兼容这套观测语义，本轮未加载。条件仍由原52queries cross-attention及GRU产生，H1临时只读hook取global feature。K8前8个认可帧取均值后调用同一shape_head；对照K1复用完全相同模型/生成器/输出规则。实际frame0认可，K1与原首帧β差0。认可条件是当前存在任意原严格接受COCO点，只是操作性校准规则，不保证足够体型信息。隐藏状态仍由原GRU逐帧处理，包括null帧；没有悄悄冻结或修正该状态。

CandidateState携带原GRU hidden、H1 ShapeState和每序列seed97的B×1×52×6噪声；每块扩展同一噪声到各帧，不逐帧重抽。H1按sequence_id及严格递增frame_id隔离状态；跨序列/重复帧/换噪声seed拒绝。原Euler为8步、候选1个，姿态编码/解码没有改成RFM。K1/K8的raw_state和raw_translation完全相同；βL2差0.44246仅说明读出变化，不是体型误差或收益。

输出明确分三层：原严格观测状态、内部network diagnostic、候选可用状态。shape未校准时beta为NaN/ready=False，人体参数和网格不可用；校准完成后β全序列冻结。当前帧无观测则status=no_current_observation，published rotations/translation/mesh为NaN，即使GRU/网络内部推测有限，也只存raw_*_diagnostic。旋转退化和参数非有限另留状态。没有前缀回填、插值或旧网格替换。这一可用状态只表示数值与当前输入契约，不等于原严格门全部通过、物理可靠或真实姿态准确；没有改变冻结baseline的infer.py行为。

真实BodyDecoder按52矩阵拆根/21身体/左右15手，使用原本地SMPLH_male/MANO资产及flat_hand_mean=True，无伪造人体层。按可用帧选择解码，每一序列复制同一已冻结β到该序列的各帧。保留6890顶点/13776面/52运动学节点，另由models/smpl/J_regressor_coco.npy生成17个表面点。运动学关节与表面点分别保存，不拿腕或肘的名字当恒等映射。当前随机模型没有WiLoR手部条件，30手旋转只是结构输出，不能据此评价抓握。

完整结果v1/v2：原3031严格接受点、29关联失败帧完整保留；K8第7帧首次ready，324可用、7待校准、29无当前观测、0数值故障。逐帧/11/64分块与整段的掩码/status一致；β最大差5.96e−8、rotation4.77e−7、translation2.38e−7m、flow state1.43e−6。11帧分块网格差4.77e−7m、COCO表面点7.15e−7m，均小于预注册2e−5容差。修改120帧后的二维/置信度，此前参数差0，冻结β不变。保存参数首/末可用帧重建网格差2.38e−7m，表面回归重复差0。上述差为算术一致性，不是三维测量误差；总检查耗时含模型构造/多种复测与解码，不能当实时部署延迟。

命令：`.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.check_joint_candidate --device cpu --output <new.json>`，必须先有H2声明的保存输入与本地许可资产。新输出同名NPZ保留完整人体和诊断，仅本地保存，不覆盖旧版本。新增9项测试含数值故障注入及真实SMPL-H/表面点反传；输入、shape head、translation head、velocity head及观测identity梯度有限非零，参数未更新，没有训练。全模块75测试、Ruff/AST通过；初次Ruff仅导入格式诊断已修正，原chumpy/scipy警告保留。独立NPZ/源输入/参数重建检查见architecture_h3_validation.json。

H3完成的是接口组合；尚未证明shape可辨识、池化不偏、手姿正确、世界/地面恢复或实时性。下一阶段单独研究缺失帧历史状态和手部条件；六个原默认neutral SMPL路径/输入/评估仍为十篇完整方法缺口，不用此处SMPL-H候选解码冒充原方法补齐。

## 2026-10-08 H4：原生手部条件、来源与离线/在线边界

architecture_h4_protocol.json先锁定，新增hand_conditions.py、reproduction/check_hand_conditions.py及15项测试；本阶段不加学习分支、不训练、不重跑检测/拟合、不改变H3或原正式身体/手流程。来源为reference25/resolved_options.json指向的左右WiLoR JSONL、fit/wilor_mano_parameter_audit.json和H2 results_v3数组；全360帧0..359、物理pair1157..1516保留。官方WiLoR作为原生手信息的辅助来源，不增加十篇选定论文的完成计数。

先复用原raw_side身体读取器，所有H2选中人物帧的float64 raw像素与H2逐位相同，再按原smplh_fitting/pipeline.py转float32后调用read_parameter_view。按原阈值检查原始鱼眼检测框与同侧身体腕点距离、候选歧义、元数据、15个局部SO(3)及检测框置信度；两相机的全部审计列表与原保存主线精确相同。不得以投影21点、弱透视相机或MANO β提供世界手位。原主线手来源接受与候选H2人物可用是独立层，先保存原状态，再相交，不改变原质量规则。

原convert_record核对canonical-right约定、index/middle/pinky/ring/thumb顺序、parent_relative_rotmat、没有加mean和左crop镜像标记。左手转换M R M，M=diag(-1,1,1)，右手原样；不是M R，也不对已经转好的矩阵再次镜像。原15节点映射到SMPL-H左22..36/右37..51，父节点分别接腕20/21及各三节链。官方MANO wrapper的21点OpenPose重排只用于输出关节点，不能套到15个旋转上。审计资产与原主线相同；左右MANO rest镜像差0，但MANO/SMPL-H rest骨向差最大左11.57°/右7.07°，原45°只排明显轴/顺序错，不证明模型模板或精确局部基完全相同。

接口NativeHandTokens要求B,T,V,2,15,3,3旋转和B,T,V,2框分数、显式解剖左右顺序、约定、bool视图可用和严格递增int64帧号；源/目标帧号必须相同。验证有源选择的SO(3)，即使候选视图随后被屏蔽。source_present为原weight>0，admitted为source_present与H2视图可用相交，present广播15节点。采用既有列拼接6D和同一框score形成B,T,V,2,15,7；框score只表示检测框，不伪装每关节置信度。缺失的token/local_rotation保留NaN，源的零权重identity不准入。两相机单独保存，不选更好者、不平均轴角/PCA、不补帧、不转移global_orient/β/weak-camera/投影2D。实际手部神经网络分支另立协议；当前H3生成器没有接收这些tokens。

真实数据按“相机/解剖手”统计：左相机原接受左40/右356，候选准入左34/右328；右相机原接受左334/右33，候选准入左311/右33。原763个手/视图槽中706准入，57因H2视图人物不可用而保留源信息但不准入；共10590个15关节token。29个H2关联失败帧不赋予候选人物，3031个原严格接受点及全部拒绝状态不变。每只手各33帧有双视图同时准入；以这33帧×15关节的旋转角差作统计，左median11.43°/P95 25.58°、右13.11°/37.94°。这是交叉视图模型一致性，覆盖少且选中了特定可用帧，既不是三维真值误差，也不能推断单视图其余帧的准确性。

逐帧token未来隔离：只改第120帧及以后的原生旋转，先前token差0。另从原pipeline AST提取并执行唯一shared_hand_pose分支initial赋值，不改写原公式。该PCA初始化按整段所有接受帧/视图加权，未来旋转扰动改变初值，具体系数变化见结果JSON；这是原有意离线处理，不是主线bug。后续共享手姿拟合也使用整段信息，不能以这份冻结结果充当在线输入。零权重保持原不可用，原PCA12/mean/scale与模型资产一致。

结果v1失败保留：检查器先前把float64 body直接传原读取器，与原pipeline float32导致box_wrist_distance_px微差；只修检查器dtype，不放宽“完全相同”检查。v2/v3全部通过，27个非metadata数组在新进程重复逐位一致；独立读取原JSONL和审计选中record，对763个接受槽验证原native矩阵、M R M和框权重完全相同；6D重建相对float64源最大差2.98e-7。所有17个H2数组原样保存。15项测试涵盖缺失identity、视图分离、镜像及排除字段、错误语义/帧/掩码/旋转/tree、原读取器未来隔离、离线初始化依赖、旋转及score反传；无优化器。首轮两个fixture分别错用了confidence键和含side的PCA维度，已按原接口改正；Ruff五项dict风格问题修正。全模块90通过，10个既有chumpy/scipy warning保留；Ruff及35个Python文件AST通过，719个作者文件存在及字节数一致。详见architecture_h4_validation.json。

复现命令：`.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.check_hand_conditions --output <new.json>`；必须提供未使用的新名字，输出同名NPZ。原始WiLoR、论文/作者文件、许可资产、大型数组留本地，不提交。下一阶段在独立候选中接入带视图/手/关节身份的原生手部条件，检查全缺失分支严格为零、时间因果性、梯度和算力；不以随机前向宣称手部改善，仍需完成原默认方法的许可/公共资源路径。

## 2026-10-08 H5：手部条件与完整人体生成器接线

architecture_h5_protocol.json先锁定。新增hand_candidate.py、reproduction/check_hand_candidate.py及12项测试，candidate.py只增加受保护的_encode委托并把原调用移入该方法；原H1形状/noise/状态/发布工作流不复制。core model/train/infer/config、H1/H2/H4工厂、原手读取器、作者源和reference25主线保持原样。参考依据是P07正文L134–138与官方smpl_head.py17–91中参数query读取条件memory的接口；P07原来是一个query读取ViT图像tokens，本项目为30个手节点query读取native手信息，两者明确不是相同架构，也不把候选算新论文复现。

输入是H2全360帧原始鱼眼归一化像素/严格射线条件和H4保存的原生手姿来源；saved_native_hands重新调用H4工厂，由source_native_rotations/source_box_confidence和明确cam0/cam1人物状态重建tokens。帧号必须与body step相同，解剖节点必须左22..36/右37..51，掩码为bool、矩阵编码/分数与模型float32/device一致；不读取离线拟合出来的共享手姿。原763源手/视图槽与706准入、10590关节tokens及所有17个H2字段保存，源缺失与明确controlled-ablation分别处理。关闭条件时保留源source_present，但admitted/present全false且token/localR全NaN，说明人为消融，不改写原来源。

NativeHandConditioner的参数包括MLP(7→64→64)、2×64视图identity、2×15×64解剖节点identity、memory/query LayerNorm以及4头64维cross-attention，总23616；原模型143699，总167315。每个(batch,time,hand)为一行，15个body手节点condition作query，2camera×15个native节点作memory；相机与解剖identity加到memory，保证两视图不是不明身份的混合。左右手分别处理，不能在新增attention中互相读取，但一只手可以读取同手其他关节与两个视图的信息。框score是7维输入的一部分，不是关节概率、质量门或硬权重。

缺失memory在投影前由where隔离NaN，并在attention中mask；整个hand-row都缺失时完全跳过attention，避免全masked softmax及输出bias制造信息。生成B,T,52,64的delta，其中前22个节点恒为精确零；只有对应30手节点有残差，加到原condition后再运行原8次Euler。没有把WiLoR矩阵写入人体输出，没有平均视图/PCA，没有新增手部时序状态，也没有回填缺失。新分支不进入body encoder/GRU/globalfeatures；因此同源H1共享β、平移、hidden、noise和认可/冻结时机不受手条件影响。原52节点空间attention/图层可把手condition的影响传到身体旋转，这是后续受控研究对象，不是手影响身体一定有益的证明。

全流程结果v2/v3：共享β仍第7帧ready并冻结，324可用/7待校准/29无当前观测；后者参数/网格保持NaN。即使人工契约探针给了手条件但没有当前body观测，也不把帧发布为身体可用或新shape证据。控制关闭手条件后，完整360帧全部H3参数/掩码与本次独立H3及上阶段保存结果逐位相同。开启后β/T/shape认可/status与H3完全相同，hidden/noise相同；非手condition delta为0，无手帧flow state也与控制完全相同。随机输出与native源最大矩阵差1.4045，证明并非直接替换；flow state最大改变0.5251、网格最大变化48.83mm及非手旋转矩阵最大变化0.11256只表示条件影响，绝不是精度/误差或收益。

逐帧/11/64分块状态及掩码一致；flow最大算术差1.19e-6、矩阵4.77e-7、T2.38e-7m、β5.96e-8、网格与COCO表面点7.15e-7m，均满足预注册2e-5。未来120帧以后同时扰动身体像素/score和原生手旋转，前120帧全部参数差0，β冻结不变。v2/v3新进程60个非metadata数组完全重复；独立读取保存R/β/T，重建首/末可用人体及COCO表面点最大差2.38e-7m。旧H3参数对照、source H2/H4字段、NaN/status/β和delta逐项独立核验在architecture_h5_validation.json。

成本仅为本机CPU2线程、eager、真实第8帧条件，warmup5后50次：v2/v3 branch有手中位0.650/0.754ms，无手0.119/0.146ms；同一8NFE候选H3 7.618/8.291ms，H5 8.440/8.537ms。计时在各模式内顺序进行且样本少，有调度/缓存波动，不能把中位数差当稳定增量、加速率或硬实时证据；不含检测器、raw/strict准备、手来源读取、完整人体解码、Stage和显示。没有作吞吐外推或整套FPS声明。

12项测试检查精确零控制、shape/T/hidden/noise不变、同手memory/跨手隔离、被mask输入无效、错误帧/节点/掩码/非有限/dtype/人物视图拒绝、body缺失发布、逐帧/分块/未来及真实人体梯度。真实网格/COCO loss反传到当前有效native旋转、框score、projection/节点/视图identity/attention，梯度有限非零；当前缺失相机及其他过去/未来手条件梯度严格0，无优化器，参数不更新。全模块102通过、13个既有资产弃用warning保留，Ruff及38文件AST通过。候选文本逆去_encode委托后与原源码完全相同；原core/主线等8路径文本与已提交版本相同，719作者文件存在及字节数一致。

首次完整v1失败由运行器错误取load_coco17_regressor(...)[0]造成6890向量，修运行器为完整17×6890矩阵；不改回归器/人体层/原阈值，v1 JSON保留。静态检查的TypeError/导入格式/nested-if修正，初轮12行为测试通过后增加“当前masked camera梯度0”覆盖再跑102全套。命令：`.venv-cuda/Scripts/python.exe -B -m realtime_app.learning_flow_mesh.reproduction.check_hand_candidate --output <new.json>`，输出同名NPZ不覆盖。候选为未训练的接线，尚无手形、β可辨识、世界位置/接触精度结论；null-frame GRU、history feature池化及eager hook部署限制不变。后续继续原方法公共辅助资源/完整输入路径，以及独立缺失历史状态对照，不扩大本阶段变量。

## R1 公共辅助资源验收（2026-10-08）

只改变原公共小型辅助资源的可用性和本地验收，不改变模型或正式工程路线。本次接手不再下载：使用`fetch_auxiliary_resources --local-only --output <new.json>`验证v2记录的七个文件，回归器来自HuManiFlow官方raw及ProHMR/HMR2官方包选择性读取；HMR2默认位置为用户.cache/4DHumans/data，早期隔离镜像不算默认路径。来源/字节/实际schema与人工6890顶点原vertices2joints、独立einsum、平移行和响应和有限非零梯度保存在architecture_resources_results_v3.json。最大线性差约2.78e-17、平移差2.22e-16；没有真实关节点精度结论。

获取器不再以循环结束或成员次数表示成功：记录wanted/acquired/missing集合，只有全部唯一目标取得才all_selected_acquired；正常EOF缺项和重复目标分别失败/停止。保留先取得的文件，不展开大型未知成员、不extractall、不读symlink、不覆盖不同字节；读前后检查deadline，原raw8MiB/archive16MiB/request20s限制继续。已有v1/v2记录两次尝试，网络入口读取记录并拒绝第三次，本阶段网络0。历史WHAM HEAD重定向12s ReadTimeout来自接手报告，单列historical_preflight_records；不是本次实际请求或文件/许可缺失证据。

mean原pose144/shape10/cam3，HMR2 heads/smpl_head.py 49–55初始化读取全部三项，ProHMR heads/fc_head.py只读取shape/cam。packed columns根正交残量1.5099581673894136e-7；作者F.normalize默认eps1e-12仍输出合法旋转，项目eps1e-6拒绝节点0，其余节点原decoder与adapter差0。初始化不是有效姿态观测，不修改资产、不放宽门。两个官方mean与旧MMPose fixture的keys/数组/dtype/原字节全相同；不得外推VIBE/MotionBERT其他同名文件。

原audit_mesh_resources入口执行v2，并在明确fixture比较后新建v3（不覆盖v2）；当前默认缺14项，P04/P07/P05各1、P09为3、P10/P12各4，六条mesh都未就绪。raw/原包/早期失败v1/v2全部保留。完整模块实际124测试、R1新增22、14warning、Ruff、40文件AST通过；719作者文件按清单核对存在/字节数，非内容逐字比对。core/主线、H3/H5、原WiLoR读取器10路径未改，无训练/权重加载/optimizer或完整原mesh。验收architecture_resources_validation.json的whole_goal_complete=false；本地提交不含辅助资产、作者源码、模型、原论文或大产物。

## H6 缺失历史单因素候选（2026-10-08）

`MissingHistoryMeshCandidate`继承H5，只在原history输出上临时拦截；`update_all`直接走原H5，`freeze`复用原GRU.forward逐帧，missing行hidden原样保留，初始missing为零hidden，当前全局输出取最后一层hidden。原ConditionEncoder仍计算局部null条件，shape的认可只由原admitted决定，missing不变成体型证据。当前有图像或2D观测仍更新历史，严格点全拒绝不等于没当前观测。K8池化规则、手分支、随机noise、生成器及所有head未改。共享β/T/姿态可因输入历史变化而改变，这属于唯一变量的下游效应，不宣称它们与H5必须相同。

该harness使用scoped eager hook，原整段GRU实际先执行再复算逐帧，direct forward绕过hook递归；finally清理自身hook，H1特征读取hook同步读取替换后的真实历史输出。没有额外持久状态/参数、训练/权重加载/optimizer。共享模块并发/compile/export不适用，额外计算不能当加速；部署接口应另设协议。

输入为H2/H4全部360保存槽及H5保存控制，model19/hand29/noise97、8Euler。当前v1关闭策略与旧H5所有控制参数逐位相同；17身体及11手源字段、3031严格点、763源/706准入手槽全部独立NPZ核对。仍324/7/29发布状态，missing参数/真实网格NaN；β第7帧ready后常数，policy间β差5.96e-8来自逐帧复算舍入，不能称体型改善。29无观测时原hidden最大变化0.547084、freeze精确0，policy间hidden差0.822174、随机网格差266.337mm仅结构影响。

逐帧/11/64分块原固定容差2e-5通过，最大raw state差9.54e-7，未来120后身体/手token扰动影响0；真实SMPL-H6890/13776/52及COCO表面读取，独立保存参数重建差2.38e-7m。长missing1/8/64后恢复与无gap一步完全相同，frame IDs乘10（身体/native同步）除编号外全部输出差0。freeze只知道观测次数，完全不编码时间；不能称运动建模更正确，原null update也没有物理经过时间验证。

新增11行为测试：all-observed原GRU一致、混合行隔离、缺失前缀/恢复、hidden不原位修改、状态帧号继续前进、shape认可不等于观测存在、masked/future输入梯度0而恢复后当前/过去观测及原GRU梯度有限非零、异常移除hook，无参数更新。全模块实际135通过/14warning，Ruff及43AST通过，10原路径相对提交未改，719作者文件存在/字节数核对非逐字比对。初轮9测试通过，Ruff2导入格式问题修正后扩大11测试；实际JSON/NPZ及validation保留。命令`python -B -m realtime_app.learning_flow_mesh.reproduction.check_missing_history --output <new.json>`，不覆盖输出，NPZ大文件不提交。证据仅因果/状态/真实人体接口，不构成accuracy或主线推广。

## R2 现有male SMPL的HuManiFlow原库网格子路径

用户明确授权使用三个给定资产目录。实际是SMPL male/female、SMPL-H male/female、MANO和SMPL-X三性别，无neutral SMPL。作者scripts/run_predict.py底部虽然声明gender choices，`args.gender != neutral`实际NotImplementedError；run_predict函数参数与原SMPL wrapper支持传入gender。故本协议只检验另一个male库配置，不能记作默认neutral或原CLI复现。

check_existing_smpl_variant.py独立进程安装权重下载/torch.load拒绝保护，以原24节点树实例化作者HumaniflowModel：原ResNet18、23层级SO3、原配置，seed31随机参数，人工1×18×256×256proxy，原脚本50候选预算。读取本机male标准6890文件直接传给原SMPL wrapper，无复制/改名、无作者文件修改；configs.paths在本进程只解析R1三个原regressor绝对路径，legacy NumPy/inspect兼容只本进程，原模型资产不变。

照原predict_humaniflow的参数调用point和50sample网格路径：父相对23body矩阵、global矩阵、shape_mode或shape_samples、pose2rot=False。输出1/50×6890×3、13776面、90关节。90为45 smplx原节点+9extra+19cocoplus+17H36M，不能叫COCO17或SMPL-H52。原采样旋转正交误差1.19e-7、det误差1.79e-7；同资产独立smplx网格及前45点完全一致，三个独立einsum块差<=1.19e-7m，保存50样本另进程重建及faces也精确。无训练/checkpoint/optimizer/参数变化或网络请求，原FlowMesh/H5/H6/主线未动。

135已有回归测试实际重跑通过/14warning，原male路径由运行器和独立保存重建另验，不混作单元测试覆盖；Ruff/44AST通过，719作者文件存在/字节数一致。初轮Ruff1导入格式诊断修正后执行，原torch.cross warning、SMPL10系数提示及Failed to import pytorch3d保留，未调用渲染函数。新v1及validation保留，NPZ仅本地不提交。没有检测/crop/HRNet/edge-generated输入、世界可视化或原评价，没有默认neutral完整网格或CLI。原默认资源14缺口不变；这是替代资产原库子路径工程证据，不是完整方法或自然人体准确性。

## R3 原RGB→HRNet→proxy→概率参数子路径

原文P05 L207–209、L886及L1100–1101明确Canny边图/HRNet-W48二维热图18×256proxy及ResNet18。协议固定保存LEFT转正frame0/pair1157，不以结果挑帧，不读取PMPose标签/strictXYZ/旧mesh。原no-detector分支对保存全视图运行，输入不是经独立认可的居中单人crop，不能作人物关联或人体观测有效性证据。

新增check_original_proxy.py，原HRNet-W48 config384×288、17×96×72、63,595,745参数，随机seed41；HumaniflowModel seed31，均eval无训练权重。原predict_hrnet函数实际前向；从predict_humaniflow.py选取L74–115十二条连续原AST（hrnet_output到pred），保留原坐标/裁剪/Canny/高斯/visibility/拼接和网络调用。原完整函数在该块前构造PyTorch3D renderer；本次不假造renderer，AST只是源子路径，不声称完整函数/CLI执行。输出proxy1×18×256×256，50候选及原use_shape_mode_for_samples=True。R2调用原库defaultFalse允许shape采样，所以R2是原库body接线，不是发布图像采样设置；现明确补记该区别。

原mode与50候选以现有male标准SMPL经作者wrapper解码6890/13776/90点，不替代neutral。保存的RGB与当前源像素精确，HRNet argmax/confidence独立重建精确，独立proxy仿射坐标差0、高斯差5.96e-8，原Canny对保存crop重跑与边图精确；50参数独立smplx顶点/前45点/面精确，sharedshape50行完全相同。重复proxy与point完全相同，旋转正交/det误差2.98e-7，三模块state_dict不变。没有optimizer、torch.load或网络请求，模型资产/作者/core/主线不改。

必须保留的有效性反例：原HRNet zero/negative热图argmax坐标0、置信值无效，原输入块仍force indices0..6=True，生成7张非零Gaussian；forced对象是COCO面部与肩，不是原SMPL前三关节。Gaussian对NaN坐标返回NaN，乘false仍NaN。原crop匹配aspect时原位改bbox_heights/bbox_widths，复用调用方数组需明确所有权。本次仅记录，不修作者、不引入项目阈值或改主线；有限网络输入不能说明真实观测存在。

六项新测试包括独立非对称Gaussian/argmax/affine控制、zero/negative/NaN、confidence严格大于0.75及head7强制、原AST边界。全模块实际141/15warning、Ruff/46AST通过；原“is”literal SyntaxWarning、meshgrid/cross弃用、Failed to import pytorch3d/SMPL10提示保留。719作者文件存在/字节数（非逐字比较）、11原core/候选/读取路径未变。结果architecture_r3_results_v1.json/NPZ及validation保存，不提交RGB/mesh大NPZ。命令`python -B -m realtime_app.learning_flow_mesh.reproduction.check_original_proxy --output <new.json>`。没有原detector、defaultneutral、renderer/fullfunction/CLI/checkpoint/完整评价，14默认资源缺口不变；输入原路径拓展不等于整篇完成或精度改进。

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
