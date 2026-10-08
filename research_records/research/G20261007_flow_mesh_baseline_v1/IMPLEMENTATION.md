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
