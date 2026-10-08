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
