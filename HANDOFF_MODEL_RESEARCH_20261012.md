# 模型长期研究交接：2026-10-12

这份文件可作为新窗口的接手指令。工作目录是 `D:\my_works\walker_pose_system`。请先核对当前文件，再从第 8 节的断点继续；不要将交接完成理解为长期研究目标完成。

## 1. 用户目标与当前授权

用户要求：先理解本地模型架构和代码，再持续分析先进论文，判断方法和 tricks 能否改善自己的模型，追求有创造性、优美且一定程度简约的改进，避免堆模块。用户明确纠正过：需要长期任务，而非定时任务。已有协议记录为 continuous_thread_goal；不要重新创建定时自动化，也不要恢复历史定时任务。

做严苛导师，挑战假设。借鉴数量、参数减少、工程跑通、低重投影误差，都不能单独证明创新或有效性。每次只推进一个可检查阶段；先问这个机制解决了什么已有方法不能解决的问题，再决定是否值得实验。

用户已确认额外训练标签和独立噪声校准数据“暂无”。训练与神经 checkpoint 加载继续关闭。真实 SMPL-H/MANO 参数资产可用于已经授权的几何解码检查，这不等于加载训练好的候选生成网络。当前阶段可继续做论文阅读、解析反例、最小算子实现与有限成本验证，不能据此自动开启训练。

不把长期目标缩成“通过几个测试”。最终仍需可证伪的改进、公平对照和独立数据上的有效性；数据缺口未解决前，这部分明确未完成。

## 2. 开始时只读这些权威入口

根目录为 `D:\my_works\walker_pose_system`。以下相对路径全部相对该根目录，除非明确写出其他基目录。

1. `AGENTS.md`、`README.md`、`CLAUDE.md`。
2. `AI_PROGRESS.md`：优先最近 2026-10-12 的研究段落，必要时回查相关阶段。
3. `research_records/registry/EXPERIMENT_RECORDING_POLICY.md`。
4. 研究目录 `research_records/research/G20261007_flow_mesh_baseline_v1` 下的 `EXPERIMENT.md`、`MODEL_ARCHITECTURE.md`、`MODEL_RESULTS.md` 第 24–28 节，以及 `flow_next_longterm_protocol_v1.json`。
5. 论文入口为同目录的 `LITERATURE_ANALYSIS.md`。开头已按对项目的价值和采用状态整理，不必重新从零搜一份泛论文清单。

权威顺序为 AGENTS → AI_PROGRESS → 可视化规范（涉及可视化时）→ 实验记录规则 → 当前局部协议/结果 → 概览。旧实验里关于小时调度的文字只作历史，不覆盖用户的长期任务纠正。

该研究目录没有通用的 `VALIDATION_PROTOCOL.md`；使用 `EXPERIMENT.md` 和相关 JSON 协议。不要为了补齐文件形式新建验证基础设施，也不要扫描全部原始数据。

## 3. 接手时的工作区与中断状态

交接核对时，最近的已提交研究修复是 `fix: keep exact fisheye residual gradients finite`。再往前是高价值论文整理、posterior transport/feedback 阅读、Rodrigues 核替换。用 `git log -6 --oneline` 查实际版本；实验记录不写 Git 哈希。

交接前工作区有 93 条状态项：35 条已跟踪文件修改/删除，58 条未跟踪项，暂存区为空。里面大量属于其他工作，必须全部保留。

与接手断点直接相关的未提交内容：

- `AI_PROGRESS.md` 原有 2 行空行增加，不能顺带提交或覆盖。本交接自己的追加段落应独立暂存。
- 研究目录下 `flow_next_longterm_protocol_v1.json` 有 18 行未提交增加，是 `endpoint_vjp_feasibility_20261012` 的预登记。保留它；它不是已经完成的 VJP 实现。
- 其余 91 条状态涉及现有文档、配置删除、相机/姿态/场景工具、测试、可视化等，不属于这次模型研究交接，不得清理、恢复或顺带提交。

交接核对时没有本研究相关的 Python/pytest 后台进程。没有需要继续等待的旧工具 session。新窗口仍应快速检查一次实际状态；不能把上一轮说“将运行”的计划当成在运行。

本交接只新增本文件并追加自己的进度说明。协议的 18 行在途预登记继续留在工作区。上述状态数是交接时快照，后续以现场 Git 状态为准。

## 4. 正式主线与不可破坏的接口

正式主线入口：`realtime_app/configs/current_mainline.json`。当前选择为 PMPose 原鱼眼观测 → 严格双目 → Stage/地面 → reference25 SMPL-H → WiLoR，360 帧（0–359），pair 1157–1516。

人体结果路径为 `research_records/engineering_validation/G20261005_reference_weight_fit_v1/runs/reference25/fit/result.npz`，`accepted_for_main_fit=false`。不要因为新模型代码可运行就升级该标记。contact 关闭，body_temporal_weight 为 0；不能用平滑或后续模块掩盖前一阶段失败。

场景 mainline 别名当前解析为 `supported_stop_fixed_lag`，0.5 秒固定延迟，依赖“停止时四脚在同一平地”的采集前提；采集集成仍有待完成。它不是本轮研究变量。

必须保留：PnP instance_id 相机身份；cam0=LEFT/cam1=RIGHT；模型可转正但几何必须回原鱼眼像素；严格准入 mask、逐点拒绝原因、max_matches=1；DA3 仅做逐帧 ROI/遮挡诊断。

候选必须保持完整人体的 R/T/beta，一整条候选选择；不能逐关节拼接、平均旋转/形状、重抽失败候选、用上一帧结果掩盖缺失。没有独立三维真值时，仅报告工程或内部一致性，不能声称真实人体精度、接触、步态或临床效果。

## 5. 不要混淆四种模型

| 对象 | 实际状态 | 关键区别 |
|---|---|---|
| 历史 `learning_flow_mesh` | 144019 参数，做过合成训练 | 同 1000 步，FM 110.4661 mm/8 NFE，回归 50.0678 mm/1 NFE；仅合成内部比较，回归更强 |
| 同目录 H8 | 167955 参数，未训练工程版本 | 原生手部条件、因果 prefix8 共享 beta、beta→pose 的 10→64 无偏置映射；不是 next |
| `learning_flow_mesh_next/JointFlow` | 全 52 节点 711443 参数；BODY22 707603 | 当前 ambient 表示；325 维状态，169 维局部切空间；F0 单模型，G0/G1 两个完整模型 |
| `SO3JointFlow` 显式研究候选 | 全 52 节点 711056 参数；BODY22 707216；未训练 | 角速度/右乘群积分，仅少 387 参数；默认工厂、双阶段、正式主线未切换 |

next 共用 width128、4 heads、4 Transformer blocks、54 tokens、GRU128、双相机 typed17 观测及因果时间/age 编码。不要把旧模型的 RGB、父节点图或独立手部记忆描述成 next 已具备。

ambient 全状态为 52×6 + T3 + beta10 = 325；局部方向为 52×3 + 3 + 10 = 169。G0/G1 两个全模型共 1422886 参数。SO3 仍保存矩阵、读 6D 两列，输出 3D 角速度，T/beta 沿原标准化接口；15104 个信息投影参数仍分配但未激活，不能说已删除。

SO3 源为 Haar，路径与推理采用一致的线性时间约定，8 步右乘 Lie Euler。冻结训练端点的 CPU SciPy log 只用于目标准备，不可拿它做 guidance 的可微 log。

## 6. 代码地图与计算成本

代码目录为 `realtime_app/learning_flow_mesh_next`：

| 文件 | 阅读目的 |
|---|---|
| `joint_flow.py` / `so3_flow.py` | 原流、角速度流、CFM/回归、rotation_exp |
| `state_encoder.py` | 因果观测状态、缺失、时间编码 |
| `candidate_store.py` | 固定源、完整候选、Normalization 与绑定、推理计数 |
| `geometry_observation.py` | 原鱼眼投影、NoiseModel、Huber 物理代价；最新零残差修复 |
| `geometry_information.py` | 现有 FD Jacobian/信息；下一步 VJP helper 建议放这里 |
| `input_binding.py` / `tangent.py` | detached 输入快照、右乘局部扰动和尺度、near-pi 边界 |
| `body.py` | 真实 SMPL-H 表面解码及 COCO17 regressor |
| `aggregate.py` / `information_risk.py` | 整候选决策；不得因新梯度改写既有失败语义 |
| `profile.py` | 复用现有成本工具，避免另建 benchmark/audit 框架 |
| `train.py` | 配置、准备和 objective；没有完整真实数据 loader/optimizer/checkpoint 生命周期 |

真实 BodyDecoder：52 节点、6890 顶点、13776 三角面，flat hand mean、无 PCA、10 beta。COCO17 从表面回归，不是直接取 FK 骨架关节。每条候选拥有自己的 T/beta。

旧信息路径全可用时 J 为 68×169，FD = 1 个 anchor + 2×169 = 339 次人体状态解码，默认 step1e-4、chunk16。Lambda=JᵀWJ；sigma10 是代理噪声，置信度不是校准协方差。默认对角块编码会丢交叉耦合，已有 cross-block 扩展停止。

F0 K32/8 为 8 次网络 batch、256 个候选状态；双阶段为 16 次/512 状态，还要计 339 状态 FD 和全部最终解码/决策。不能把 Lambda 或 anchor 当免费输入，也不能仅凭解码调用数量判定实际提速。

## 7. 已完成证据和明确停止的路线

`MODEL_RESULTS.md` 第 26 节与 `flow_next_so3_cost_v2.json` 保存同次固定条件成本：RTX5070Ti Laptop、CUDA float32、seed7、warmup2、每模式8次、6模式循环、真实人体解码+A4。

| 模式 | 完整 p50 ms | 完整 p95 ms | 峰分配增量 bytes |
|---|---:|---:|---:|
| ambient F0 | 33.229600 | 37.743345 | 31074304 |
| SO3 Rodrigues F0 | 38.332850 | 45.692790 | 31074304 |
| SO3 matrix_exp F0 | 43.634950 | 48.255965 | 31074304 |
| ambient regression | 13.786750 | 16.418655 | 961024 |
| SO3 Rodrigues regression | 13.743950 | 16.824375 | 961024 |
| SO3 matrix_exp regression | 15.588300 | 16.979910 | 961024 |

Rodrigues/ambient=1.1535754，通过预先规定的 ≤1.25 门，但仍慢于 ambient，没有显存收益。前一轮 matrix_exp 比值1.2747失败，保留 `flow_next_so3_cost_v1.json`。两轮核优化额度已用完，停止继续调核追分。随机未训练网络、固定观测、未锁负载，不能认证精度或全系统30fps。

其他关键停止点：

- 全 Jacobian AD 替换：两实际 SMPL-H anchor 的 float64 J 相对误差约1.25e-9/1.21e-9，但 vector AD 峰分配106131456 bytes，FD31260672，约3.4倍；状态数量少不等于成本低。见结果第13节。
- 边界反例：原 u=255.99999，AD 在 anchor 有限，但正 FD 邻居 u=256.000812 出界。原 local_information 拒绝该邻居，anchor-only AD 无法保持原契约。直接替换已停止，不能补一堆阈值/回退网络强行过门。见第14节。
- ManiPose：固定 beta 不等于 COCO 表面连线长度恒定。两实际姿态骨架长度近机器精度不变，COCO 连线最多变化6.4614 mm。不要加错误的 COCO 恒骨长损失或额外骨长头。见第21节。
- FrameFlow legacy SO3 log 在部分近 pi 端点失败；冻结端点的时间 JVP 通过。停止直接移植该分支，不能泛称所有导数或 RFM 都不可行。见第22–23节。
- StableHand quality bridge 缺独立质量标签；Flower 的条件/协方差假设和高 NFE 不匹配，均不直接叠加。未来帧依赖网络不能冒充在线因果系统。

## 8. 精确断点：零残差修复完成，端点 VJP 尚未实现

`MODEL_RESULTS.md` 第28节：17个点与原鱼眼投影严格匹配时，旧 Huber 代价为0，但51个坐标导数全 NaN。原因是 `where` 未选中的 sqrt(0) 分支反向产生0×inf。

已提交的修复只让未启用的开方分支输入有限正值，保留原前向 Huber 定义、投影、准入和决策接口。修复后该对照代价与导数严格为0。先有真实失败1项，再修复；相关 information/aggregate 测试11项通过，2份Python Ruff通过。

历史已通过命令如下，仅供复现定位；交接后不要无缘由再跑：

```powershell
.\.venv-cuda\Scripts\python.exe -B -m pytest realtime_app/learning_flow_mesh_next/tests/test_information.py realtime_app/learning_flow_mesh_next/tests/test_aggregate.py -q --tb=short
uv tool run --offline ruff check realtime_app/learning_flow_mesh_next/geometry_observation.py realtime_app/learning_flow_mesh_next/tests/test_information.py
```

现有预登记 `endpoint_vjp_feasibility_20261012` 仍标为 `preregistered_before_checks`；应结合第28节理解：其中零残差检查已完成，其余未执行。协议的该18行尚未提交。

**不存在已完成的 `endpoint_cost_vjp`；它是建议的 helper 名称。没有完整人体端点 VJP 结果，没有反馈成本比较，没有将物理梯度接入神经网络。** `profile.py` 也没有 `--endpoint-feedback-controls` 参数，不能给出虚构的现成运行命令。

## 9. 下一阶段唯一问题与数学定义

问题：在 SO3 生成的晚阶段完整人体端点上，一次反向计算是否能得到169维物理代价方向，并在明确的有限开销内替代昂贵 anchor 信息获取的某一用途？这只是替代假设，不等价于证明 J/Lambda 无用。

8步求解最后一步 s=7/8，使用已经得到的 base velocity 预测：

```text
R_hat = R_s Exp((1-s) omega_base)
aux_hat = aux_s + (1-s) v_base       # aux 是原标准化 T/beta
```

它等于未修改的最后一步端点，因此仅为预测端点不额外调用速度网络。但未来若反馈改变了人体，仍需最终解码；不能把反馈解码冒充修改后人体的免费解码。

对每条完整候选 k，在零标准化扰动 delta_k∈R^169 处：

```text
R(delta)    = R_hat @ Exp(S_pose * delta_pose)
T(delta)    = T_hat + S_T * delta_T
beta(delta) = beta_hat + S_beta * delta_beta
g_k        = d E_k(delta_k) / d delta_k | delta=0
```

S 来自现有 TangentScale，右乘约定必须保持。E 是当前准入原鱼眼像素上的 mean Huber 代价。二次区解析控制为 `g = Jᵀ W r / N_admitted`，不要多乘2，也不要因对候选 batch 求均值误多除 K。

可对独立候选代价求和再做一次 reverse pass；前提是 decoder 行之间独立。真实 SMPL LBS 满足逐样本解码，但不能未来悄悄引入跨候选归一化网络后继续套这个结论。

生成端点、观测、相机、噪声、尺度全部 detach；仅局部 delta 求导。decoder 参数冻结仍可对输入求导；外层 no_grad 需要局部 enable_grad，inference_mode 需单独处理，不能误以为二者一样。

使用现有 Rodrigues rotation_exp，不用 CPU SciPy log 或 near-pi shortest_log 进行可微反馈。输出是标准化坐标下的代价梯度/余切量，尚不是角速度、协方差或后验。它不保留 Lambda 的秩、Schur 或全部交叉耦合，必须承认语义改变。

## 10. 必须保留的失败语义

输入应绑定当前 CandidateStore、当前 raw fish 观测、相机、NoiseModel、TangentScale 和完整52节点 R/T/beta。严格检查 observation_binding；不要照搬旧 local_information 允许 legacy unverified 的宽松路径。

全部 K 个槽位和原因必须保留。无当前观测、绑定缺失/过期、不合法人体数值、准入点投影无效、非有限梯度，均显式 unavailable；梯度使用 NaN 表达不可用，不能填0让它成为可使用的“零纠正”。真实零残差且合法的零梯度应保持 available，两者严格区分。

可以在实际解码前筛出数值合法的计算行，但输出还原全部槽位，不能减少失败率分母、重抽或删除候选。批解码异常要报告受影响槽位，不做隐藏重试。端点有效不代表 FD 邻域有效；不要宣称这保持了原全 Jacobian 准入契约。

## 11. 接手后按顺序完成的最小阶段

1. 核对18行预登记和第28节，保留已有工作。阅读 `geometry_information.py`、`geometry_observation.py`、`tangent.py`、`body.py`、binding 和现有相关测试。先实现最小 helper，不加控制网、第二生成器、密度网或新质量门。
2. 在现有测试文件中加入真正对应风险的检查：刚性解析 `JᵀWr/N`、尺度/符号、mask、不同行梯度不被 K 缩放、无观测/错绑定/非法投影不可用、真实零梯度可用。不要为增加通过数写镜像实现测试。
3. 一次真实 SMPL-H 合法端点 float64 对照：同端点、同准入点、同尺度，VJP 与 FD J 计算的方向相对误差≤1e-4，pose/T/beta 分块有限。FD339状态只用于这个检查，不混入运行时比较。明确端点与 FD 邻域语义区别。
4. 前面通过后扩展现有 `profile.py` 做一次同次配对成本比较：同 SO3 权重/观测/源/Normalization/状态重置，K32、8步、warmup2、每模式8次循环，feedback 与 no-feedback。保存一次必要结果 JSON，不覆盖旧 v1/v2，不新建审计脚本/快照/验收体系。
5. 无反馈为8次网络batch/256状态 + 最终人体1batch/32状态、0 reverse；反馈可行性为同8/256 + 端点人体1batch/32状态和1 reverse，并仍计最终人体1batch/32状态，总2batch/64状态（全部合法时）。实际计数和内存必须报出。
6. 完整 p50 比值≤1.25 才通过当前成本门；同时报告 p95、峰分配增量和失败。超过则停止全 K32 晚反馈晋升，不能减少 K/步数或反复测到好结果。该机制最多两次有依据修订；既有 SO3 核两轮额度已耗尽，不能重置。
7. 把真实结果及边界更新进既有 `MODEL_RESULTS.md`、架构/协议和 `AI_PROGRESS.md`，检查必要改动后独立本地提交。只完成一个可检查阶段；失败如实保留，不以文档包装成功。
8. 只有算子和成本通过，才考虑新的网络整合预登记：同条件 CFM、强回归、zero-feedback、公平容量和全成本。即使通过，也不自动开训练；数据门仍未满足。

这组实验的成功只意味着“该物理方向算子及测得开销可用”，不能写成“模型改进有效”。如果方向仍需复杂控制网才能有效，优先重新审视问题和替代价值，别沿堆模块路线继续补丁。

## 12. 已分析论文与继续阅读重点

完整标题、原文链接、源码范围、采用状态见研究目录 `LITERATURE_ANALYSIS.md`。以下是接续研究最重要的机制关系，不是新做的一轮全文精读结论。

| 论文入口 | 对本项目的实际价值和边界 |
|---|---|
| Flow Matching（2210.02747） | CFM 基础；采用 FM 本身不构成创新 |
| HuManiFlow（2305.06968） | shape→pose 变量依赖；共享形状的使用需区分旧 H8 与 next |
| RFM（2302.03660） | 状态流形与概率路径，影响当前 SO3 候选；内在维数少不证明更快 |
| FMPose（2602.05755v1） | 多候选/聚合参考；原 GT root 相关聚合不能直接用于部署 |
| FMPE（2305.17161） | 单条件联合配对流已有先例，F0 不能凭删 G0/J 自称创新；Dingo loss/sample/divergence/cache/GLU 已读、未运行 |
| SBI-SIM（2410.22573） | 端点代价/梯度反馈，支持当前有限假设；原方法额外约10–11%控制参数，lensing10→19秒，不能当免费收益 |
| On Guidance（2502.02150） | 区分对当前状态、端点以及通过预测器求导；警惕重复条件化、能量目标和完整成本 |
| FrameFlow（2310.05297） | 专用 SO3 实现审查及近 pi 反例；不能混用训练IGSO3、推理Haar和不同日程 |
| Bayesian inverse（2610.02377v1） | 条件传输理论有假设；可信 prior/likelihood/density/MH 条件当前缺失，不用于后验认证 |

FMPE 正文大网络比较与匹配 conditioning 附录必须区分，原样本 NFE 约248、样本+密度约350，不能套作当前8步结果。SBI-SIM 已读方法/zero-control/成本部分，检查当时官方仓库没有可执行原控制实现；这是当时状态，若需当前状态应重新查原站。

Bayesian inverse v1 的解析缩放对照只针对所提取密度公式的符号问题：v=ln2*x、source N(0,1) 应得到 N(0,4)。不能把这个局部发现放大成整篇论文/未公开代码错误。

HMR2、ProHMR、WHAM、VIBE 等已有历史与设计参考。VIBE 的 beta 实际逐帧，之前“clip平均”归因已经纠正。不要把历史论文建议标成 next 当前实现。继续读新文献时优先原文方法/假设/消融和官方代码，使用适用的 cv-paper-reader 技能；不要仅凭摘要写已借鉴，更不要为了凑数量扩大清单。

## 13. 数据与最终效能门

研究目录下 `flow_next_data_manifest_v1.json` 和 `flow_next_noise_data_manifest_v1.json` 的 records 都为空。数据 readiness 为 false；用户没有新增数据。不要把现有生成结果、合成投影、自估置信度当独立标签。

`flow_research_plan_v1.json` 的既有最终门仍需保留，不能被当前算子指标替换：

- G1 对计算匹配 G0：预登记歧义分层 energy score 相对降低至少5%，配对95%CI支持改善；强几何点劣化≤2%。
- 同缓存聚合：相对最强可部署对照，选择误差至少降低5%，配对95%CI，保留率在1个百分点内或匹配风险覆盖。
- 目标90%边际覆盖需87–93%，同时报告区间宽度、CI、子组；不能只拉宽集合换覆盖。
- 无效率增加≤0.5个百分点；没有 oracle、因果泄漏或静默过滤。
- 实时声明需指定硬件全链路 p95≤33.3ms，不能用纯网络计时替代。
- seeds7/17/27；参与者配对bootstrap、参与者内序列块，不把相邻帧当独立人。

这些是原路线最终效能协议，新的反馈替代机制需要另行明确匹配对照；不是马上运行训练的指令。Normalization 仅训练集估计，BODY22 标签必须诚实为部分人体，不能伪造手部/beta 标签。当前完整真实训练 runner 也未完成。

## 14. 环境、命令与运行恢复

实际核对运行环境：PowerShell7.6.6（用户偏好7.6.5，现可用版本继续用）、本地 `.venv-cuda`、Torch2.11.0+cu128、SciPy1.17.1、CUDA可用、NVIDIA GeForce RTX5070Ti Laptop GPU。以接手现场为准，不为确认环境重跑模型。

```powershell
Set-Location 'D:\my_works\walker_pose_system'
git status --short
git diff --cached --name-only
git diff -- research_records/research/G20261007_flow_mesh_baseline_v1/flow_next_longterm_protocol_v1.json
Get-Content AI_PROGRESS.md -Tail 100
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pytest' -and $_.CommandLine -match 'learning_flow_mesh_next|test_information' } | Select-Object ProcessId,CommandLine
```

人体资产从 `realtime_app/learning_flow_mesh/config.json` 的 assets 读取：`third_party/WiLoR/mano_data/models/SMPLH_male.pkl`、`MANO_LEFT.pkl`、`MANO_RIGHT.pkl`（后两者同目录），`models/smpl/J_regressor_coco.npy`。该 config 的模型/训练字段属于旧22节点 baseline，不能整体当 next 配置。

需要解析 PDF 时，已知 bundled Python 为 `C:\Users\毛晨昊\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`，有 pypdf；本地 CUDA venv 不保证有 PDF 包。优先内存读取，不默认下载成额外副本或生成提取报告。

运行恢复规则：工具等待超时不等于进程失败，先看 session/进程和已有结果，不能重复启动。`profile.py` 结果采用排他创建，不覆盖原输出。BodyDecoder 可在 stdout 的 JSON 前打印 WARNING，整段 stdout 解析失败不等于保存结果失败；直接读结果 JSON，不因解析失败重跑测时。报告实际异常，不掩盖失败。

测试文件有 `test_information.py`、`test_aggregate.py`、`test_joint.py`、`test_regression_information.py`、`test_binding.py`、`test_identity_binding.py`、`test_encoding.py` 等；`test_information_binding.py` 不存在。先 `rg --files` 定位，不编造文件名。修改哪个接口就检查对应风险，不默认全量测试。

## 15. 记录、提交与交付

不创建哈希清单、备份、额外快照、审计/验收报告或流程脚本。只保留支撑具体结论所需的输入、代码、参数、结果和失败。禁止把验证材料数量当工作质量。

每个完成且可独立检查的修改要本地提交，禁止 push。提交前看实际 diff/暂存区，只暂存本次文件或代码块；不要 `git add .`。`AI_PROGRESS.md` 已有空行改动，应使用交互分块暂存，或从 diff 中提取唯一的新进度 hunk 用 `git apply --cached`；不能顺手把他人改动一起提交。若不能可靠隔离，保留工作并明确说明阻碍。

提交后检查包含的文件与剩余状态。实验记录不写 Git 哈希，用户交付可以给提交 ID。不要把“本交接已提交”误读成18行预登记或端点 VJP 阶段也已提交。

新窗口第一次进度反馈应明确：已读权威资料；零残差修复完成；端点 VJP 仍未实现；当前只推进算子正确性与有限成本，训练关闭。完成本阶段后给出继续/停止决定及依据，不再扩充证明材料。

请从这里继续：读取当前协议及 `geometry_information.py`，实现并检查最小端点物理代价 VJP。算子对照未通过之前不接神经反馈，成本门未通过之前不晋升；始终围绕一个更简约、具有可验证价值的模型推进。
