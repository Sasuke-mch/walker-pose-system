# 固定手部约束全身拟合：详细交接

更新时间：2026-10-01。仓库：`D:\my_works\walker_pose_system`。

本文交接当前固定抓取、VPoser全身优化、脚部约束和448帧可视化链。它是当前工作快照，不覆盖 `AGENTS.md`、`AI_PROGRESS.md`、`VISUALIZATION_PIPELINE.md` 和实验协议。没有重新运行模型；数值来自已有运行产物及此前完成的分析。

## 1. 接手时首先必须知道的结论

**实现、测试和全448帧展示已经完成，但当前全身拟合没有通过验收，不能作为成功的固定腕部结果接入主线。**

当前算法已经保留全身三角化观测、正确的活动VPoser、共享手指PCA、脚底C1/C2、非穿透、腿部时序及Stage2逐顶点静止锚。它确实避免了前一轮身体观测误差大幅恶化；代价是身体优化基本停滞，腕目标没有实现，腿部抖动和穿地没有解决。

当前交付目录是：

```text
research_records/engineering_validation/
  G20261001_annotated_shared_grip_v1/
    active_constructed_grip_v2/
      balanced_body_full448_v1/
```

其中 `full448_assessment.json` 的 `passed=false`、`accepted_for_main_fit=false`。五项门中只有观测退化保护通过。浏览器展示通过表示页面可以正确查看当前数据，不能理解为人体姿态或抓握通过。

## 2. 用户目标及已经明确的工作约束

1. 两手抓紧扶手，参考姿态应固定在**助步器坐标系**，不能固定在世界/地面坐标系。
2. 抬起、放下助步器时腕关节允许目标附近小幅移动；当前优化使用10mm软界和10mm腕误差P95验收门。软界有损失，不等于优化器已经硬保证范围。
3. 优先保证腕部位置，使手—臂—躯干的连接合理；腕部合理后再处理手指包裹和碰撞。手指优化不能替代腕位置求解。
4. 手姿态可以主动构造，原图、人工标注、WiLoR和既有解是参考。构造姿态应标记为设计结果，不能宣称从图像恢复了真实手姿态。
5. 使用PCA约束手指，使用VPoser约束身体；不能解码后覆盖身体关节、关闭姿态修正或贴独立手网格掩盖连接问题。
6. 任何阶段都保留身体各部位观测和脚部约束。用户明确提出Stage2脚相对地面静止，但当前工程地面尚不是独立真值。
7. 不优先重新标定：用户认为标定此前基本正确。发现新证据后才能定位标定问题，不能用重标定替代优化逻辑诊断。
8. 可视化骨架全部黑色实线，标题正式简洁，当前标题为“SMPL-H 固定手部约束拟合”。不做显示平滑、插值补帧或坐标吸附。
9. 保留旧入口与失败产物，可直接回退；不要重构成无法辨别数据来源的单一路线。

本仓库目前有大量已有修改、删除和未跟踪文件，包括其他人体拟合路线。它们不自动属于当前交接链，不能批量暂存、覆盖或清理。每项已验证修改必须本地独立提交；没有新的明确授权，不推送远端。

## 3. 阅读顺序和证据位置

| 文件 | 用途 |
| --- | --- |
| `AGENTS.md` | 长期约束、证据边界和版本管理 |
| `README.md`、`CLAUDE.md` | 仓库导航，任务开始时读取 |
| `AI_PROGRESS.md` | 当前阶段决策与各次实验状态 |
| `VISUALIZATION_PIPELINE.md` | 原图对应、坐标、Stage、真实三角面和显示规范 |
| `research_records/registry/EXPERIMENT_RECORDING_POLICY.md` | 实验可追溯性规则；实验记录不写Git哈希 |
| `research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/EXPERIMENT.md` | 手部/全身接入、逻辑审计、短窗与全帧实验的连续记录 |
| 当前运行的 `protocol.json`、`fit/run_metadata.json` | 冻结门、实际输入、权重与参数化 |
| 当前运行的 `full448_assessment.json` | 数值门和失败结论 |

当前目录没有独立复制一份EXPERIMENT；详细协议继续维护在上表既有实验文档中。历史记录中先出现的Stage2质心锚方案已被后面的schema2逐顶点方案修正，阅读时必须以最新记录和代码为准。

## 4. 当前可靠的数据入口

以下路径相对仓库根目录。当前运行简称 **RUN**，其父目录 `active_constructed_grip_v2` 简称 **BASE**。

| 输入 | 路径/说明 |
| --- | --- |
| 初始人体和不可变参考 | `research_records/engineering_validation/G20261001_fixed_mano_pca448_v1/fit_shared/result.npz`；当前source与pose-reference是同一原始共享VPoser结果 |
| 主动构造抓取 | `BASE/constructed_grasp.json`、`BASE/constructed_grasp.npz` |
| 原二维点 | `research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/{left,right}/pmpose/raw_predictions.json` |
| 同源图像 | 同一 `people_1/input_448pairs/{left_ccw90,right_cw90}/pair_0000.png` 至 `pair_0447.png` |
| 正式标定 | `realtime_app/calibration/results/stereo_fisheye.json`；cam0=LEFT，cam1=RIGHT |
| 静态地面参考 | `research_records/engineering_validation/G20260914_static_ground_reference_v1/estimate_stationary_20pairs_v2/ground_reference.json` |
| 静态助步器 | `research_records/engineering_validation/G20260918_offline_walker_frame_evidence_v1/coarse_complete_model_v2_camera_rail/coarse_walker_model.json` |
| 当前动态地面 | `RUN/scene/scene_transforms.npz`；本轮448帧重新重放生成 |
| 当前脚候选 | `RUN/current_foot_candidates.npz`；原人体参考鞋底经过本轮地面变换重新计算 |
| 表面分区 | `research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json` |
| VPoser | `models/VPoser02_05/V02_05`，checkpoint在其 `snapshots/V02_05_epoch=13_val_loss=0.03.ckpt` |
| SMPL-H/MANO | `third_party/WiLoR/mano_data/models/` 下实际资产 |
| COCO表面回归器 | `models/smpl/J_regressor_coco.npy` |
| Python环境 | `.venv-cuda/Scripts/python.exe` |

**448对图像不等于采集视频的前448帧。** 曾出现这个严重输入错位，不能再次按下标截视频。模型转正图进入几何前必须逆旋转：左转正图向顺时针恢复、右转正图向逆时针恢复到原始1920×1080鱼眼像素。

驱动脚本会读旧 `vposer_body_full448_v1` 元数据，作用是取得静态配置和输入路径；本轮动态地面重新生成，没有把旧动态变换或旧失败人体输出作为当前初始化。完整实际输入以 `fit/run_metadata.json`、`scene/scene_sources.json` 和每一步command JSON为准。

人工轮廓、腕点和WiLoR影响的是前面的抓取构造/参考链。当前全身入口使用已构造的抓取目标、共享PCA和PMPose身体观测，不能说它在每一步都重新直接拟合人工轮廓或WiLoR原始图像结果。

## 5. 已经得到的抓取究竟是什么

`BASE/constructed_grasp.*` 是主动构造的左右手代表性抓取。手指由左右实际MANO/SMPL-H前向计算，保留12维PCA、腕位姿、表面顶点与三角面，不是镜像网格粘贴。

左右腕目标在助步器系中约为：

```text
left  = [ 0.2239034, -0.0075681, 0.8949394] m
right = [-0.2292971, -0.0087240, 0.8914604] m
```

它们相对旧数据参考主动偏移约11.011/9.804mm。左手超过此前旧有界实验10mm范围，是用户允许主动靠近扶手后的设计调整，不能宣称仍满足旧实验边界。

独立构造时两手工程包裹检查通过，最大杆顶点穿透约0.525/0.500mm；该容差不是“绝无穿透”。`accepted_for_grasp_initialization=true` 与 `accepted_for_main_fit=false` 必须同时理解。

**局部构造通过不保证接到身体后仍通过。** 完整SMPL-H的姿态修正和混合蒙皮会改变手的皮肤表面。过去右手接入后有自相交；受控诊断没有找到简单左右参数误映射证据，而是发现局部刚体手形假设不足、精修目标缺自碰撞。当前加入了全帧手自碰撞与表面形状项，最新全帧筛查未检测到非共面自相交，但整体抓取仍失败。

## 6. 当前优化怎样工作

### 6.1 坐标和参数

身体在左相机系以米拟合，原始三角化数组以毫米保存，进入损失前换成米。助步器安装相对相机固定，所以手腕目标在助步器局部系中恒定；助步器相对地面运动时，目标自然随助步器移动。

若 `p_C = R_CW p_W + t_CW`，则行向量实现为 `p_W = (p_C - t_CW) @ R_CW`。地面显示统一使用 `p_G = R_GC p_C + t_GC`，Canvas显示轴为 `[X,-Y,Z]`。不能把显示翻轴写回优化参数。

腕位置取实际SMPL-H关节20/21，身体观测则取表面COCO回归点。这两类“腕点”语义不完全相同，不能直接当作同一个3D点。

每帧身体参数为32维VPoser latent，实时解码全部21个身体旋转。解码器权重冻结且处于评估模式，但梯度经过解码器流向latent；身体不能在解码后覆盖腿或腕旋转。根方向和平移按阶段释放；10维shape固定。左右手各共享12维PCA，最后阶段才优化；PCA均值只加一次。VPoser是全身耦合的，所以“上肢优先”表示约束与释放顺序，不能称为腿参数绝对冻结的独立上肢求解。

### 6.2 四个实现阶段

| 阶段 | 可训练量 | 保持和限制 |
| --- | --- | --- |
| `upper_only` | 全32维latent | 根、平移、PCA冻结；腿8个旋转参考软项，髋膝踝最大移动5mm硬检查；腕位置/朝向/软界权重从0.1爬升到1 |
| `upper_and_torso` | latent、根方向增量、平移增量 | PCA冻结，全身观察与脚项继续参与 |
| `body_wrist_polish` | 同上 | 学习率乘0.25；仍检查逐点预算 |
| `hand_surface_polish` | 左右共享PCA | 身体、根、平移冻结，处理杆穿透、手表面、自碰撞与手指区域 |

当前全帧四阶段各50步，学习率0.003，碰撞候选每25步覆盖全448帧刷新。beta、标定和scene始终冻结。旧默认路线保留；新路线通过 `--full-body --surface-refine --balanced-stages --stage2-static-assumption` 显式启用。

### 6.3 全身观察和保护门

3D观察按左臂、右臂、躯干、左腿、右腿、头六组分别归一化再平均，避免某个部位因点数多完全压过另一部位；肩和髋按设计参与重叠组。3D距离尺度为50mm。二维项使用原始鱼眼投影、逐视图有限性/置信度/边界mask，有效单目信息可以保留。

每个本轮严格接受的3D点都有硬保护：

```text
当前模型到观测的误差 <= 不可变原参考的误差 + 10mm
```

低quality仅降低软权重，不删除硬门。这个规则保护“不要进一步恶化”，**不保证参考误差本来就小，也不保证三角化点是真值**。恢复训练也从原始参考重新计算基准，不能每次resume重新获得10mm预算。

### 6.4 脚部约束并没有删除

- C1：鞋底表面接近地面候选项。当前候选权重由原参考 `sigmoid((0.030-abs(min_z))/0.015)` 生成，不能称为测得触地概率。
- 非穿透：有效脚底表面地下部分受罚。balance路线margin=0，与Stage2锚最低Z=0一致；旧默认正3mm表示留空，不是允许3mm穿透。
- C2：冻结状态机判断的support/support有效相邻帧、同一组鞋底顶点的切向速度约束。状态滞回和拒绝原因保留。当前有效相邻对左右都是0，原因是没有可信静态贴地候选，不能为让损失非零伪造支撑状态。
- 腿部时序：局部旋转相邻变化，以及地面系髋/膝/踝二阶差。采用30fps假设，加速度尺度10m/s²；这些是优化量，不是测量的人体加速度。

当前原参考有明显鞋底地下位置，C1候选权重随距离降低，可能弱化贴地驱动；但非穿透和新Stage2锚仍有梯度。不能把失败简单说成“忘了脚损失”。

### 6.5 Stage2静止参考

该约束单独使用用户静止假设，不冒充C2的测得支撑。先在全序列按精确Stage2标签分连续段，至少3个有效帧才能生成锚，再切窗。

每脚选择最接近该段脚底中心中位数位置的一帧**真实参考脚形**，整体移动XY到段中心，整体抬升Z使最低鞋底为0；脚跟和前掌各选3个不同的低位参考顶点。最终对每一个顶点固定位置并约束相邻速度，目标形状为 `[N,4,3,3]`、顶点身份 `[N,4,3]`，schema_version=2。不能只锚质心，否则脚绕质心转动仍零损失；不能逐顶点坐标独立取中位数，否则会形成并不存在的脚形。

当前位置尺度10mm、速度尺度0.05m/s，权重1/0.1；只计算同一有效段相邻对。本轮115个Stage2帧中114帧有锚、105个有效相邻对。短段和不可用原因保留。

目标Z=0是设计，XY来自原参考，地面来自工程Stage链。Stage2平移仍依赖踝锚，独立外部刚体/标记板轨迹验证尚未完成；不能用该链再证明脚真实静止。

### 6.6 更新接受与回滚

Adam先提出候选更新，整段使用同一缩步系数，从1到1/128最多试8次。只有有限、当前固定目标不增加、观察预算通过、上肢阶段腿移动限制通过、PCA不超过原构造系数±2，才接受。

全部失败时恢复参数、Adam矩和step，异常出口也恢复；接受缩步时保留候选Adam矩，这是当前实现选择。每次尝试的阻断帧和关节保存于 `update_transactions.json`。训练trace是提出更新**之前**的前向值，并记录腕权重爬升，不能把爬升造成的目标变大当作接受了增损失步。

此实现是带保护门的回退更新，**还没有实现约束投影方向或逐帧可行方向求解**。单帧可能阻断全部448帧；回滚后重复同一坏方向也可能持续失败。这是当前主要算法瓶颈。

### 6.7 当前权重

实际完整权重以 `fit/run_metadata.json/coefficients` 为准。关键项：身体3D=1、2D=0.15、身体参考=1、VPoser=0.02、旋转时序=0.2；腕位置=20、朝向=0.1、软界=10；脚C1=1、非穿透=1、C2=0.001；腿旋转时序=0.2、腿加速度=0.05、Stage2位置=1/速度=0.1；手杆穿透=20、自碰撞=50、形状=0.1、PCA参考=0.5、手指区域=3、掌面=2、拇指对向=10。

这些系数乘的是不同归一化尺度的项，不能只比较系数大小就判断重要性。当前腕软界的1mm尺度使实际目标和梯度压倒其他项，见第9节。

## 7. 代码职责和回退边界

以下文件均相对 `realtime_app/`，两个全帧驱动文件在RUN内。

| 文件 | 职责 |
| --- | --- |
| `tools/refine_body_with_constructed_grasp.py` | 新可选入口，输入校验、阶段调度、输出与失败记录 |
| `pose_app/constructed_grasp_refinement.py` | 实际FK、坐标、观察mask、脚项、手表面与全帧碰撞 |
| `pose_app/vposer_grasp_body.py` | 活动VPoser解码、旋转参考与时序 |
| `pose_app/balanced_grasp.py` | 分组观察、不可变预算、腿时序、schema2脚锚、事务更新 |
| `tools/benchmark_balanced_grasp.py` | 冻结短窗control/balanced比较与协议 |
| `tools/audit_constructed_grasp_body.py` | 对实际身体网格审计全部手帧，保留失败 |
| `tools/diagnose_constructed_grasp_deformation.py` | 局部手与完整身体形变差异诊断 |
| `tools/build_grasp_body_canvas_viewer.py` | 当前人体、观察、scene与审计生成网页 |
| `tools/build_surface_contact_canvas_viewer.py` | Canvas共用渲染实现 |
| `RUN/run_full448.py` | 新重放→新脚候选→拟合→审计→网页→结果检查 |
| `RUN/assess_full448.py` | 连续帧、模型形状、有限性、观察门和最终门检查 |

旧 `tools/fit_smplh_wilor_sequence.py` 没有被此路线替代。回退可以使用旧入口和原始共享结果；旧无VPoser/自由身体旋转的抓取接入探针只保留诊断用途，不推荐继续作为成功主线。大输出和模型资产没有随代码全部纳入版本控制，交接工作目录必须同时保留本地数据。

## 8. 重要阶段的完成情况

| 阶段/目录 | 已完成内容 | 当前结论 |
| --- | --- | --- |
| `G20261001_fixed_mano_pca448_v1/fit_shared` | 原始全序列共享手PCA与VPoser参考 | 作为不可变工程参考，非独立真值 |
| `BASE/constructed_grasp.*` | 左右主动构造、实际模型和包裹检查 | 可作抓取初始化；不能直接认证全身 |
| `body_refinement_window60_90_v1/v2` | 旧上肢/躯干接入，腕接近目标 | 右手形变后自相交，失败；身体参数化后来被质疑 |
| `vposer_body_window60_90_v1` | 活动VPoser短窗路线 | 失败诊断保留 |
| `vposer_body_full448_v1` | 前一轮448帧、真实网格/页面 | 腕较近但身体RMS83.119→120.684mm、脚穿地加重，两手均失败 |
| `audit_logic_*` | 输入、冻结参考、碰撞覆盖和接口探针 | 接口检查，不代表求解质量 |
| `balanced_short60_100_v1/v2` | 41帧策略包control/candidate比较 | 观察保护通过；腕/脚/抖动/手形失败 |
| `balanced_logic_audit60_100_v3` | schema2锚、不可变预算、异常回滚等修复 | 47项测试通过，41帧候选仍失败；resume超预算拒绝已验证 |
| `balanced_body_full448_v1` | 修复后的完整448帧及规范展示 | 当前交付；仅观察保护通过，主拟合拒绝 |

用户要求全帧时明确扩大了诊断运行范围，因此全帧运行已执行；这不等于把此前短窗失败的门自动改为通过。

已修复的明确问题包括：原鱼眼2D越界/非有限屏蔽；source/reference/scene帧身份错位拒绝；冻结脚状态不随resume漂移；手碰撞刷新覆盖全帧；零步阶段避免空Adam；原始观察预算不重置；Stage2逐顶点锚与真实参考脚形；异常恢复参数和Adam；Stage2地面0与非穿透留空冲突；无效NaN先mask后平方。修复和接口通过不意味着所有算法问题已经解决。

## 9. 最新全帧结果与原因分析

### 9.1 冻结门和实际结果

| 指标 | 当前结果 | 门/解释 |
| --- | --- | --- |
| 连续帧和真实网格 | 0..447；448×6890顶点；13776三角面 | 形状、有限性和面索引检查通过 |
| 身体加权观察RMS | 83.119→83.096mm | 改善很小，不代表真实3D精度 |
| 最大逐点误差增加 | 9.993mm | 10mm预算通过 |
| 左腕median/P95/max | 97.286/155.193/188.367mm | P95≤10mm失败 |
| 右腕median/P95/max | 84.778/110.646/130.924mm | P95≤10mm失败 |
| 左/右腕朝向median | 40.243/43.043° | 仍明显不匹配目标朝向 |
| 全片最低鞋底Z，左/右 | -116.183/-123.660mm | ≥-3mm失败 |
| 腿地面加速度归一化平方 | 11.536→11.706 | 不增加门失败 |
| 实际手几何通过，左/右 | 0/448、0/448 | 全帧通过门失败 |
| 最大杆顶点穿透，左/右 | 15.714/15.800mm | 抓取不合格 |
| 身体三阶段接受更新 | 6/50、6/50、0/50 | 合计12/150，明显停滞 |
| 共享PCA阶段接受更新 | 50/50 | 手指能够变动，但腕位置无法被该阶段修复 |

两手分别都没有一帧腕误差≤10mm。故当前网页不能称为“腕部基本固定后的全身效果”；实际是固定目标存在、但没有求解到目标的结果。

### 9.2 一个帧如何阻断整段

已完成的事务分析显示：上肢阶段44次最终拒绝主要是腿移动5mm门；躯干阶段44次最终拒绝和联合阶段全部50次最终拒绝，被 **pair283、COCO8右肘** 阻断。

该点参考误差约103.983mm，最终约113.976mm，已经使用约9.993mm预算；quality约0.719，严格门接受，附近281..285三角化轨迹平滑，没有证据允许把它直接删成异常点。

这不是“多跑几轮就必然收敛”的情况。原始误差已经不小，保护门又不允许继续超过基准+10mm，全窗共用缩步导致其他帧也无法前进。现阶段根平移变化中位数仅约4.81mm、身体局部旋转最大变化约1.12°，说明人体基本留在初始解附近。

### 9.3 腕损失规模失衡

最终加权腕软界项约390967，腕位置项约7184，总目标约398824。软界单项占约98%，两腕项合计约99.8%。软界使用超过10mm后的距离除以1mm再平方，并包含mean和max；初始腕误差约100mm时非常大，容易由最坏帧支配方向。

初始平移梯度范数，腕软界约3.71×10⁶，身体3D约0.317，脚非穿透约47.2。梯度范数不能直接等同Adam实际步长，也不能单凭范数证明方向冲突，但这种尺度差异足以要求重新设计接近目标与边界激活方式。单纯再增加腕权重缺少依据。

### 9.4 腿部问题不是本轮才产生

以30fps计算模型地面腿点二阶差，RMS约33.97→34.21m/s²；相机系也约33.42→33.72m/s²，说明抖动不能全归因于地面变换。它很大程度已在原始拟合中存在，本轮几乎未改变身体，当然也没有消除它。

模型膝屈曲相邻变化P95约左8.63°/右8.78°，最大约14.28°/18.62°。这些是模型运动学诊断，不是临床或真实步态结论。

右膝严格接受只有84/448，左膝357/448；quality中位数约0.054/0.277。右膝软观察明显弱，腿又受全身耦合和保护门限制。正确做法是保留可用性/不确定性并分析方向，不能永久排除右膝或靠显示平滑掩盖。

### 9.5 Stage2静止没有得到实现

选定鞋底顶点在Stage2的速度median约0.688m/s、P95约1.859m/s，顶点到静止锚的位置误差median约59.91mm、P95约99.22mm。Stage2速度平方项略减，但远未满足静止要求。

严格接受的原始三角化踝点经当前工程地面变换，在Stage2相邻有效对上速度median约左0.439/右0.521m/s。它说明“当前观测+工程地面”并不可靠地表达静止假设；不能由此直接证明真实脚移动、标定错误或用户判断错误。踝是代理、脚可能转动、地面又依赖踝锚，需要分开验证。

### 9.6 当前手形失败与旧右手自相交要区分

最新全部手帧筛查未检出非共面自相交，不等于排除了共面、切触和连续运动碰撞。当前两手0/448通过主要涉及手指区域离杆、掌面离杆、对向不足和杆穿透。左右手指区域距离门均448帧失败；手腕整体摆放尚未正确，此时继续只解手指不符合优先顺序。

旧右手自相交问题已促进新增碰撞项和实际网格审计，但不能把旧“右手不行”的结论直接套到本轮，也不能把当前失败称为已证明MANO左右转换错误。

## 10. 当前产物清单与显示要求

| RUN下文件 | 内容 |
| --- | --- |
| `fit/result.npz` | 实际模型、latent、21旋转、PCA、raw/模型COCO、质量mask、腕局部位姿、地面顶点等 |
| `fit/run_metadata.json`、`fit/fit_summary.json` | 输入、系数、坐标、阶段指标 |
| `fit/*_parameters.npz` | 四阶段快照，不能与其他运行混合 |
| `fit/observation_guard_reference.npz` | 不可变预测基准、误差和valid |
| `fit/frozen_foot_states.npz` | 冻结脚状态与不可用原因 |
| `fit/stage2_anchors.npz`、`fit/stage2_anchor_assumption.json` | schema2脚锚、段、设计说明 |
| `fit/body_2d_observation_audit.npz` | 逐视图2D状态 |
| `fit/update_transactions.json`、`fit/gradient_audit.json` | 接受/拒绝尝试和初始梯度 |
| `fit/grasp_geometry_audit.json` | 全896手帧实际几何审计 |
| `scene/*` | 当前Stage、地面、静态助步器与来源 |
| `protocol.json`、`*_command.json`、`*.log` | 冻结协议、实际执行和日志 |
| `full448_assessment.json` | 最终五门判断；失败必须保留 |
| `body_viewer_v2.html` | 当前正式交付页面 |
| `viewer_validation.json`、`viewer_frame224.jpg` | 浏览器验证与截图 |

页面包含真实半透明三角面、模型COCO、原始三角化和拒绝点、黑色实线骨架、XY地面/Z高度、实体助步器、双相机/基线/轨迹、脚地下标记、Stage/held/rejected、腕与手形诊断。右膝拒绝点仍保留，不能永久隐藏。

原 `body_viewer.html` 在窄窗口取景裁切人体，仍保留。v2只用全片髋中位数设置固定观察中心，没有逐帧移动坐标；两版数据载荷完全相同检查通过。浏览器已检查0/224/447、播放暂停、旋转重置，控制台无error；这些是已完成的验证记录，不保证本地服务当前仍在运行。

查看地址（服务存在时）：

```text
http://127.0.0.1:8765/research_records/engineering_validation/G20261001_annotated_shared_grip_v1/active_constructed_grip_v2/balanced_body_full448_v1/body_viewer_v2.html
```

HTML也可本地打开。需要服务时先检查8765是否占用，再在仓库根运行 `.venv-cuda/Scripts/python.exe -m http.server 8765 --bind 127.0.0.1`，不要重复启动已有服务。

## 11. 验证与复现方式

### 11.1 测试

逻辑修复时以下47项相关测试通过；最近全帧展示变动另通过5项窗口映射测试、相关脚本编译与diff检查。本次交接文档不重跑耗时拟合，也不把历史测试报告写成刚重新执行。

在仓库根、PowerShell中：

```powershell
& .\.venv-cuda\Scripts\python.exe -m pytest `
  realtime_app/tests/test_balanced_grasp.py `
  realtime_app/tests/test_constructed_grasp_refinement.py `
  realtime_app/tests/test_vposer_grasp_body.py `
  realtime_app/tests/test_audit_grasp_body_pose_prior.py `
  realtime_app/tests/test_smpl_surface_contact.py `
  realtime_app/tests/test_grasp_body_viewer_frame_mapping.py -q
```

回归覆盖不可变预算、逐顶点静止、真实代表帧几何、异常Adam回滚、无效NaN隔离、零高度非穿透及相关输入/参数链。通过这些测试只证明对应实现契约，不证明物理拟合成功。

### 11.2 短窗可重复入口

```powershell
& .\.venv-cuda\Scripts\python.exe realtime_app/tools/benchmark_balanced_grasp.py `
  --reference-metadata research_records/engineering_validation/G20261001_annotated_shared_grip_v1/active_constructed_grip_v2/vposer_body_full448_v1/fit/run_metadata.json `
  --output-root research_records/engineering_validation/G20261001_annotated_shared_grip_v1/active_constructed_grip_v2/balanced_short60_100_repro_v1 `
  --steps 50
```

这是现有策略包比较入口，并不是未来单因素实验协议。必须使用不存在的新输出目录，检查当前脚本默认窗口和参数，不覆盖既有失败目录。

### 11.3 全帧运行注意事项

完整实际命令已经记录于RUN各 `*_command.json`。`run_full448.py` 将自己的所在目录当输出目录，**不要在现有RUN直接重跑**：协议/日志可能先写入，scene随后因已有目录拒绝，导致旧记录被污染。

如以后确需重复原实验，可把 `run_full448.py` 和 `assess_full448.py` 一起复制到BASE下一个全新同级目录，从仓库根用CUDA Python执行新脚本。脚本依赖BASE下已有 `vposer_body_full448_v1` 元数据、原始共享参考及全部模型资产；新目录仍应只读这些输入、重新生成scene。它执行两次网页构建以检查载荷一致，但浏览器交互验证仍需另做。

复现原实验预计仍会失败，不能把重复运行本身当作下一阶段研究。下一阶段应先明确唯一算法变化、冻结control和验收门，再运行短窗。

### 11.4 恢复保护验证

`BASE/balanced_logic_audit60_100_v3/resume_must_reject` 保存超预算恢复拒绝，reason为 `resume_outside_immutable_observation_guard`、最大超出约0.255829m。被拒输入是前一轮失败全身输出，原始参考不变；基准误差与正常运行完全一致。接手者不得改成“恢复时以source重新预算”。

## 12. 尚未实现的下一阶段：按顺序推进

### 第一步：先解更新停滞，不直接再跑448帧

保留全部观察硬门，研究逐帧信赖域或可行方向更新，减少pair283阻断全序列的问题。共享PCA、跨帧时序和根参数存在耦合，不能简单逐帧接受然后宣称全局目标单调；任何局部候选仍须在完整当前目标和保护门下验证。

建议先使用覆盖阻断帧的270..295窗口，并保留60..100作为已有短窗对照。先检查每次实际参数变化、阻断点、可行方向和有效梯度，不能把“没有改变”算成保护成功后的优化成功。这些是建议，尚未实现新求解器。

### 第二步：分阶段接近腕目标和激活微动边界

把“从约100mm接近扶手”与“到位后仅允许约10mm微动”区分。研究尺度归一化、逐步激活边界或约束方向；保留最终目标和验收门，不把最终腕界放宽来让结果通过。只改变一项并比较相同输入和步数。

同时核对实际SMPL-H腕、COCO回归腕、人工透视腕标签与构造腕目标的语义，检查目标是否与臂长/躯干/脚锚存在可行性冲突。用户确认抓紧扶手可以作为先验，但不能自动证明每个构造目标在现有身体shape和观察预算下都可行。

### 第三步：腕到位后再处理手指

先保证实际腕位置和掌面朝向，再允许共享PCA的小范围表面修正。继续实际左右模型、完整身体条件和全帧审计，不回到中立身体局部手刚体近似；不关闭pose correctives。接触区域近距离、对向和无检测相交只是几何代理，尚不等于摩擦锥、力闭合或承重验证。

### 第四步：验证腿时序与Stage2参考的兼容性

分别看相机系模型、地面系模型、原始三角化和冻结锚，不用单一平滑分数替代原因分析。重点保留右膝低可用性，检查脚姿态转动而踝代理变化的影响。Stage2独立外部轨迹验证仍待补充，不能由优化结果反向认证工程地面。

### 第五步：通过短窗后才扩全帧

维持当前冻结门：每个accepted观察退化≤10mm、两腕P95≤10mm、鞋底最低Z≥-3mm、腿加速度不增加、全部实际手帧几何通过。还须有可观察的腕接近和身体更新，不能只保护初始坏解。任何门失败都保存结果和原因、停止自动升级；新的全帧诊断可由用户明确要求，但结论仍按实际门判断。

## 13. 给下一位执行者的交接指令

> 当前重点是修复可行更新和约束尺度，而不是增加运行时间或把右膝/阻断肘点删除。先读权威文档与本handoff，核对工作区和RUN原始JSON。保留原始共享参考、助步器局部腕目标、活动VPoser、PCA、全身观察、脚C1/C2和schema2 Stage2锚。不要使用旧失败全身输出重置观察预算，不读取旧动态地面冒充当前scene，不把局部抓取通过当完整身体通过。先设计一个可验收的短窗单因素修复并记录协议；若没有可行更新证据，停止扩大运行并报告冲突。完成修改后按仓库规则测试、检查差异、本地独立提交，只提交本次文件。

当前交接没有启动下一轮拟合或自动批准主线升级。当前成果的价值是建立了可审计的实现、冻结参考、失败证据和全帧展示；腕部固定、脚物理一致性与腿抖动仍是必须解决的工作。
