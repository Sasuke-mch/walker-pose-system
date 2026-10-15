# PHC+预训练控制器接入与完整448帧实验

## 当前结论（2026-10-15）

已完成官方冻结权重接入、原版PhysX CPU运行、SMPL-H往返转换、站立门、完整448帧有/无假设腕支撑对照及解码分析。当前是三维参考模仿候选，不是二维直接驱动的强化学习估计器；没有训练或微调，没有替换正式主线。

**接入可行，直接用作现有人体估计器的后处理尚不合格。** 假设腕支撑解决了本片跌倒，但二维回投一致性变差，移动助步器阶段脚跟仍滑动。不把448有限帧、参考跟踪通过或网格外观当作真实精度、真实触地或方法完整成功。

## 目标、对照与停止条件

用户要求优先可靠处理完整视频、减少对全段的依赖，并做好详细实现记录。本轮可检查阶段是“已有预训练控制器是否能在本机正确执行，并跟随当前逐帧人体参考”。

先验证网络/观测接口和站立后端；站立10秒全部完成、pelvis不低于0.5m、24刚体参考误差均值不高于0.10m且无外部稳定力，才进入448。完整视频统一448帧、真实时间、同一初始化、同一官方权重和人体XML；最终有/无支撑对照只改变腕外力。初始未启用作者远距离恢复版本单列，不能把它与支撑版视为单变量比较。

二维一致性使用12个四肢关节×两目×448=10752个固定观测槽位；不可回投/不可用槽位在全片≤25px比例中计失败，回投分布另报告有限样本数。无独立多帧标签或3D真值，统计只检验同检测器一致性。脚跟固定目标为移动阶段段内位移不超过10mm；本轮没有实现该硬约束，不能宣称完成它。不做整段目标视频优化，不因失败删帧/插值/重新初始化仿真，也不通过大量训练或调参掩盖本轮失败。

## 来源与环境

- 官方PHC：https://github.com/ZhengyiLuo/PHC ，本地third_party/PHC，未改上游源码。
- 作者移植：https://github.com/ZhengyiLuo/PHC_MJX ，本地third_party/PHC_MJX。README明确under construction，不能当作现成等价PHC+运行包。
- 人体库：https://github.com/ZhengyiLuo/SMPLSim ，本地third_party/SMPLSim，仅查阅，没有把该库接入作为本轮成功条件。
- 权重按PHC/download_data.sh原URL用gdown下载：models/phc/phc_kp_2.pth，Google Drive id=1QVv1qxsN2LnncPna66qSV3OikYfseCnx；phc_comp_kp_2.pth，id=1V1mG5dTXzkONgPiwd97JeKtnFaj7KwQx；standing.pkl，id=1fFauJE0W0nJfihUvjViq9OzmFfHo_rq0。对应约403/247/2.16MB，不入Git。未下载AMASS全库、未重新训练。
- 官方Isaac Gym Preview4包：https://developer.nvidia.com/isaac-gym-preview-4 ，本地third_party/IsaacGym_Preview_4_Package.tar.gz；WSL解压至/home/itachi/walker-phc/isaacgym。遵守其独立许可，不随工程分发。
- PowerShell7.6.6；Windows独立.venv-phc，Python3.12.12、torch2.14.1+cpu、numpy1.26.4、scipy1.14.1、MuJoCo3.16.0、smplx0.1.28、chumpy0.70、Matplotlib3.11.2、joblib1.6.0、gdown6.4.2；作者网络对照额外rl-games1.6.5 --no-deps及其纯网络依赖。requirements-analysis.txt保存直接运行依赖。
- 当前Windows环境有walker_cuda_packages.pth指向已有.venv-cuda/Lib/site-packages作为只读依赖搜索路径；不是完全独立安装证据，原CUDA/RL环境没有改动。模型和MANO/SMPL-H许可资源另行提供。
- 原生运行：WSL Ubuntu-22.04，/home/itachi/miniconda3/envs/walker-phc/bin/python，Python3.8.20、torch1.13.1+cpu、numpy1.23.5、scipy1.10.1、joblib1.4.2、Pillow10.4.0。requirements-native.txt记录直接依赖。IsaacGym必须先于torch导入；不使用gymtorch/GPU扩展。日志中的缺libcuda/WSL代理警告保留，CPU仿真已实际完成。
- 首次conda命令误用全局Windows envs路径产生D:anacondaenvs目录和CondaValueError；改显式-p /home/itachi/miniconda3/envs/walker-phc、CONDA_PKGS_DIRS=/home/itachi/.conda/pkgs后成功。残留目录未作为代码提交或擅自清理。

## 实现对应关系

源码均在realtime_app/learning_phc：contract.py保存24刚体顺序、SMPL索引与upright基旋转；policy.py加载冻结网络、构造作者v7观测与远距离恢复；reference.py将保存的当前SMPL-H姿态转换为原生参考；native.py执行原版IsaacGym PhysX；analyze.py反解完整SMPL-H、回投、脚底和运动曲线；visualize.py重新计算strict及场景并导出页面。physics.py/run.py只保留已失败的MuJoCo迁移诊断，推荐运行入口是native，不是run。

网络是作者3个PNN专家，各574→2048→1536→1024→1024→512→512→69，隐藏层SiLU；composer相同主干输出3，ending_act=False、has_softmax=False。两者各用自身running_mean_std、epsilon=1e-5、归一化裁至±5；composer输出裁至±1，专家动作加权和再裁至±1。参数冻结，无梯度更新。单测直接加载作者网络比较，输出完全相同。

观测574=self358+task216。self为根高度1、根相对位置69、6D旋转144、线速度72、角速度72；task为参考位置差72、参考速度差72、根相对参考位置72，使用24个原生刚体及heading局部系，不以COCO点替代。动作69维是23个三轴关节位置目标，作者尺度一般pi、膝y为5rad，不是力矩。远距离恢复按作者zero_out_far：根距离>0.25m时非根参考/速度改为仿真当前值；根目标距离最多3m。保存的原始参考不改，报告误差仍对原始参考。

PhysX为Z向上、重力9.81、TGS、dt至多1/60和2个内部子步、4次位置/0次速度迭代、地面摩擦1、原XML形状与24个碰撞filter、原PD stiffness/damping。各采集间隔最后不足子步使用真实剩余dt，不把不匀时间按30Hz重采样。原生asset effort约3.4e38；不能将它解释为可信的人体肌肉力矩上限。单人CPU调用没有端到端实时认证。

腕支撑是显式假设：两腕目标固定在首帧相机局部位置，随当前场景变换移动；每仿真子步施加F=900(target-position)-45*world_velocity N，逐侧裁至300N。没有测力数据、实体抓握接触或目标速度阻尼补偿，不能称真实扶手支撑。无支撑版本仅去掉这个外力，其他接口、恢复规则和权重相同。

## 输入与坐标、时序边界

人体参考输入为research_records/engineering_validation/G20261014_smplh_motion448_v1/world_contact_20261015/run_world_unconstrained/result.npz；同目录run_parameters.json包含原始左右PMPose、input_448pairs、标定、模型、首帧标注及capture_pairs路径。本轮明确使用该次保存的世界无接触候选作参考，不等同于正式二维离线主线，也没有读取后续contact_v7网格或混合输出。

场景条件输入为同实验world_contact_20261015/scene/scene_transforms.npz。其supported_stop_fixed_lag含踝锚和名义0.5秒发布延迟，不能称独立世界真值。首帧手工点沿上游初始化使用，hand=腕、foot=踝，不是手指/heel。形状beta从原主线全片预校准并冻结，**不是完全首帧形状估计或完全不依赖全段**。

全部448对顺序、pair_id=0..447，时间由原selection_manifest逐行对齐stereo_pairs.csv左右host-return均值；跨度16.035239秒，不等于曝光/同步真值。参考仅读当前姿态、当前场景、前一帧参考；速度采用真实dt后向差分，起点/断段零初始化。prefix不变性及缺失保留已验证，没有未来姿态、整片高度修正/平滑/重采样。控制在第i帧观测到达后以reference_i从上一状态积分至time_i，不读取i+1；这与作者固定30Hz预览运动设置有区别。

根位置使用实际pelvis joint（R*camera_joint0+t），不能用SMPL transl冒充关节。SMPL全局旋转乘upright基逆、按作者24体顺序重排，原生中性XML骨长生成位置参考；原生身体与固定beta的SMPL-H骨长/皮肤并不相同。两末端手链只取冻结第一指基姿态，不是完整手控制。

反解将原生全局旋转还原SMPL坐标、按父子关系还原21局部旋转，根平移扣除shaped rest pelvis；沿用相同固定beta和原冻结完整手姿，生成6890顶点/13776真实三角面。参考旋转正反转换后的原SMPL-H全448网格最大坐标差1.3113e-6m，确认坐标往返本身没有破坏原网格；不能据此证明仿真骨架和皮肤的物理等价。

## 实际运行与失败保留

| 阶段 | 实际结果 | 决策 |
|---|---|---|
| MuJoCo v1原hinge/隐式PD | 站立均误差1.852m，pelvis最低0.093m | 失败 |
| v2加原碰撞filter | 均误差4.117m，pelvis最低0.093m | 失败 |
| v3旋转向量转Euler | 出现QACC数值警告及可能自动重置 | 失败，有限值不证明正常 |
| v4球关节+stable PD | 均误差1.058m，pelvis最低0.084m | 停止后端迁移 |
| 原版PhysX站立10秒 | 300步，均误差18.84mm，P9543.76mm，pelvis最低0.931m，无外力 | 后端门通过 |
| 448初始无远距离恢复 | 全448有限，315帧低于0.3m | 失败，保留v2 |
| 448原生恢复、无腕支撑 | 全448有限，49帧低于0.3m | 失败对照保留 |
| 448原生恢复、假设腕支撑 | 全448有限，0帧低于0.3m | 只通过本片站立稳定性 |

standing_mujoco_v1..v4、standing_physx_v1、replay448_physx_v2、replay448_physx_recovery、replay448_physx_support分别保存输出和summary。replay448_physx_v1为参考未生成时的启动失败，日志保留，不计为有效实验。prepare_reference初始重复关键字及trusted Stage对象数组读取失败、首次单测纯函数namespace缺失均已修复，原失败日志保留。权重是作者含pickle的checkpoint，明确weights_only=False加载；不扩展到任意不可信文件。

## 完整448结果与解释

| 指标 | 输入SMPL-H | PHC+无腕支撑 | PHC+假设腕支撑 |
|---|---:|---:|---:|
| 仿真输出有限帧 | — | 448/448 | 448/448 |
| 仿真pelvis<0.3m帧数 | — | 49 | 0 |
| 原生24体参考均误差 m | — | 0.4068 | 0.09745 |
| 宽松参考门通过帧占比 | — | 46.65% | 100% |
| 双目回投有限槽位 | 10752 | 8992 | 10752 |
| 回投中位数 px | 23.24 | 168.79 | 74.53 |
| 回投P95 px | 84.28 | 957.75 | 190.44 |
| 全10752槽位≤25px | 53.19% | 4.57% | 8.95% |
| 解码脚底候选最低高度中位数 mm | -36.92 | 38.82 | 30.90 |
| 低于-5mm的单脚帧数/896 | 684 | 2 | 0 |
| Stage2脚跟段内位移P95 mm | 未在此统计 | 795.39 | 363.51 |
| Stage2脚跟段内位移最大 mm | 未在此统计 | 1521.11 | 458.12 |

“宽松参考门”是24体均距离≤0.25m且pelvis≥0.3m，仅操作性参考跟随，不是姿态精度门。两实验无frame0以外重置；跌倒但有限的帧保留fallen_finite。无支撑的1760个不可回投槽位保留在全片分母，不能用8992子集的比例替代。Stage2共115帧、两脚230个段内位移值，每段首个Stage2重建锚，transition保持该段，Stage1清除；这是跟随本次Stage定义的诊断，不是独立触地认证。

假设支撑最大单腕力278.59N，未触及300N裁幅。无/有支撑单帧控制+物理调用中位7.83/8.24ms，P9513.67/13.35ms，不含检测、SMPL-H参考估计、Stage图像重放、解码、渲染或发布等待。不能从这些数字声称端到端实时。

脚底穿地下界改善伴随浮空、脚跟滑移和像素不一致；原生碰撞capsule不等于固定beta解码皮肤，不能称硬地面接触已正确求解。即便输入参考本身有误差，PHC相对同源像素变差仍足以否定“直接套预训练即可改善当前估计”的假设。坐标往返已过，仅排除该类转换错误；体型重定向、动态参考误差、自由根漂移、支撑假设及无二维反馈均仍可能贡献偏差，不能无消融锁定单一原因。

motion_curves.png为同组模型COCO轨迹的离线非均匀三点导数，使用下一采样点，仅用于结果分析，不进入控制器，不冒充其瞬时物理加速度或可因果发布的v/a；端点/断段NaN保留、无平滑。加速度模长P95输入24.99、无支撑92.18、有支撑58.38m/s²，不能称生理运动指标。

## 复现命令

以下在D:/my_works/walker_pose_system运行。输入与输出相对本实验根，已存在输出不会覆盖；复跑应指定新目录。

```powershell
$env:PYTHONPATH='realtime_app'
.venv-phc/Scripts/python.exe -m learning_phc.reference --source research_records/engineering_validation/G20261014_smplh_motion448_v1/world_contact_20261015/run_world_unconstrained/result.npz --scene research_records/engineering_validation/G20261014_smplh_motion448_v1/world_contact_20261015/scene/scene_transforms.npz --xml third_party/PHC/phc/data/assets/mjcf/smpl_0_humanoid.xml --output research_records/engineering_validation/G20261015_phc_pretrained448_v1/reference448.npz
.venv-phc/Scripts/python.exe -m unittest realtime_app.tests.test_phc_pretrained -v
uvx --offline ruff check --select E9,F realtime_app/learning_phc realtime_app/tests/test_phc_pretrained.py
```

原生环境先安装IsaacGym Preview4并按requirements-native.txt装依赖；执行时WSL bash的核心命令如下，先独立站立，再无/有支撑。WSL路径是/mnt/d，不能直接传Windows反斜杠路径。

```bash
cd /mnt/d/my_works/walker_pose_system
export LD_LIBRARY_PATH=/home/itachi/miniconda3/envs/walker-phc/lib
export PYTHONPATH=/home/itachi/walker-phc/isaacgym/python:/mnt/d/my_works/walker_pose_system/realtime_app
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/itachi/miniconda3/envs/walker-phc/bin/python -m learning_phc.native --output research_records/engineering_validation/G20261015_phc_pretrained448_v1/standing_physx_v1
/home/itachi/miniconda3/envs/walker-phc/bin/python -m learning_phc.native --reference research_records/engineering_validation/G20261015_phc_pretrained448_v1/reference448.npz --output research_records/engineering_validation/G20261015_phc_pretrained448_v1/replay448_physx_recovery
/home/itachi/miniconda3/envs/walker-phc/bin/python -m learning_phc.native --reference research_records/engineering_validation/G20261015_phc_pretrained448_v1/reference448.npz --wrist-support --output research_records/engineering_validation/G20261015_phc_pretrained448_v1/replay448_physx_support
```

```powershell
$env:PYTHONPATH='realtime_app'
.venv-phc/Scripts/python.exe -m learning_phc.analyze --root research_records/engineering_validation/G20261015_phc_pretrained448_v1 --source research_records/engineering_validation/G20261014_smplh_motion448_v1/world_contact_20261015/run_world_unconstrained/result.npz --reference research_records/engineering_validation/G20261015_phc_pretrained448_v1/reference448.npz
.venv-phc/Scripts/python.exe -m learning_phc.visualize --root research_records/engineering_validation/G20261015_phc_pretrained448_v1 --view-dir view_v2
```

可视化首个view因frame_times缺signed_host_delta_ms_right_minus_left被场景入口拒绝，未显示旧场景代替；失败日志visualize.log与view/scene.log保留。修复后view_v2从同源原图、当前strict重新运行Stage，只有其R/t、Stage、时序与控制器所用条件场景一致才显示；旧动态输入只用于独立数值比较，不进入新显示计算。原始转正图为1080×1920，叠点按原1920×1080鱼眼坐标转正再缩放，不把540×960预览尺寸当标定尺寸。页面含本轮真实解码网格、黑色原始观测骨架、拒绝原因、模型点、双相机/基线、实体walker、XY地面和0.5m坐标刻度；三个方法均保留448帧，不插值或吸附。浏览器核对结果在本节后续状态记录。

## 验证与下一阶段

7项单测已通过：实际作者网络等值、实际作者v7纯观测函数等值、全局旋转/FK与非均匀步长、非有限观测拒绝、参考prefix不变及断段/缺失、错误时间/有效NaN拒绝、作者远距离恢复不改保存参考。相关Ruff E9/F通过。站立、完整448、同SMPL-H坐标往返及失败全分母已实际执行；不重复训练或全仓测试来装饰结论。

本轮停止于预训练接入与真实片可用性评估。没有证据支持把该候选并入主线。后续应先解决“同体型人体参考→可跟踪物理参考→二维观测闭环”的接口，并把阶段足底约束和显式支撑作为真实对照；先检查参考本身的可行姿态与观测冲突，再决定是否使用现有策略做受限残差/二维条件微调。单片448不能承担训练泛化结论，进入学习阶段前必须另设训练片和冻结验证片。不能靠再换一个随机RL网络、盲目增加训练步数或隐藏跌倒来宣称系统完善。


实际浏览器完成：三种方法页面加载，0/99/143/335/447帧、滑条/数字帧号、播放/暂停、按当前帧切换方法和左右原图叠点已检查；无支撑99帧明确fallen_finite，支撑143帧Stage2显示左右heel315.1/458.1mm，447帧16.035秒。截图support_frame143.jpg、support_frame335.jpg、support_overlay447.jpg留view_v2；metadata.json已记录。页面不因错误大而隐藏有限黑线。修复预览坐标按1080×1920原图缩放，以及数字输入即时换帧；原图缓存只留每目当前图，避免全448张解码图常驻。最终Ruff E9/F与编译通过；7项物理/策略接口单测通过，未重复训练或全仓测试。

## 2026-10-15 效果差原因的冻结输出诊断

用户要求详细分析原因、简明总结。新增learning_phc/diagnose.py，仅分析已保存448输出，不重新训练、不再仿真、不改变原始人体或PHC轨迹。直接重算原source/无支撑/支撑双目回投，并与analysis.json中位数按1e-6px核对，仍用10752全槽位、正深度/缺失保留。将仿真骨盆位置、根朝向或21局部旋转中的一组换回原输入，沿同beta/同完整冻结手姿重建网格并评价；每批检查实际pelvis与所选根位置差≤2e-6m。替换轨迹不是可执行的物理轨迹，只用于几何归因，三项效应非线性、不能相加或解释为修复后控制效果。

| 固定输出中的离线操作 | 无支撑回投中位px | 支撑回投中位px |
|---|---:|---:|
| 原输出 | 168.788 | 74.534 |
| 只恢复输入骨盆位置 | 139.224 | 69.749 |
| 只恢复输入根朝向 | 179.343 | 99.693 |
| 只恢复输入21关节局部旋转 | 188.928 | 114.329 |

只恢复位置后的支撑P95反而190.445→208.905px，全片≤25px8.947→8.938%，不能称整体修复。单独恢复旋转更差说明不同误差可能互相补偿，不能反推原姿态较坏或给各因子分配独立贡献百分比。支撑原生骨盆相对参考距离median72.721/P95122.045mm；对每帧减去整体平移后，24体平均相对位置误差median61.608mm，包含根朝向与局部姿态影响，不等同于单独关节角误差。无支撑相应163.360/P951050.614mm及194.813mm，失败同时含整体漂移和变形。

同源源SMPL-H实际前22关节与相同旋转的作者中性骨架对应点距离median22.720/P9538.209/max48.802mm，包含模型骨长/关节几何差异，比较不是COCO17与内部24点混用。源网格坐标正反往返1.31e-6m并不消除此骨架差别；前者只验证旋转和解码可逆，后者是实际参考重定向造成的几何变化。

精确按每次观测所用p_(i-1)与reference_i计算，原生zero_out_far阈值0.25m在无支撑174/447次观测触发，支撑0次；不是先前简单以同帧p_i比较的170次。恢复时非根目标改为当前仿真点、参考速度改为当前速度，因此原姿态跟踪信息减弱；这是作者恢复机制及失衡后的放大路径，无法解释支撑版残余退化。动作±1裁幅分量比例无支撑0.0616%、支撑0，未支持动作限幅为主要原因，未测得可信生理力矩限幅。

假设相机刚性腕目标与原生参考腕点距离median21.245/P9539.809/max55.821mm；非完全相同任务，腕外力与模仿参考存在可测冲突。当前D项为−45*v_world，没有+45*v_target；按相邻host时间差分推算省略项模长median0/P959.143/max27.917N，仅是条件诊断，尚无加该项的仿真消融。不能把它定为主因或把约279N最大仿真力称真实承重。

真实host间隔min6.982/median39.763/P9561.403/max69.505ms，447间隔中127<20ms、186>45ms；策略每帧更新一次而574观测不含显式dt，与固定30Hz控制有差异。host返回时间不等于曝光时间，差异对结果的因果贡献尚未用统一时钟对照验证。无独立世界轨迹且Stage含踝锚，输入脚底候选684/896脚帧低于−5mm，能说明相对于当前条件地面的接触不一致，不能认定实际人体穿地或证明哪项标定有误。

结论分层：已确定的设计缺口为PHC不直接接收raw2D/二维梯度、不包含本项目heel10mm硬约束、腕外力与训练受力任务有差别、原生碰撞体与解码皮肤并不等价。已测得为参考几何差、整体与相对姿态偏差、支撑目标冲突及恢复机制触发。输入动态可行性、仿真动态接口/时钟、训练域适配仍须有针对性的对照，不能统一归因OOD或训练不足；站立门通过只排除部分基础接口错误，不证明动态论文复现。

优先顺序：先以官方有动态的参考片验证原生动作时序和控制接口；再检查同体型/同骨架的参考与解码、支撑边界；之后才加入保留像素反馈的受限物理纠正及阶段脚底约束。当前不建议直接扩大训练或升级PHC输出替代现有人体。详细数值diagnosis.json；命令：PYTHONPATH=realtime_app，以.venv-phc/Scripts/python.exe -m learning_phc.diagnose --root research_records/engineering_validation/G20261015_phc_pretrained448_v1执行（拒绝覆盖既有diagnosis.json）。本次实际执行完成、日志diagnose.log保留，Ruff E9/F及编译通过；非新准确性验证。

补充接口判断：本权重task216只含参考点位置差、速度差和相对根的位置，没有显式目标关节旋转、原始双目点、相机投影参数、Stage2固定heel目标；self358有当前身体旋转，不能当作已输入目标旋转。目标点跟随与完整SMPL-H网格/局部扭转一致性并非相同目标。当前69维动作是PD位置目标，不是最初设想的广义加速度动作；因此本轮失败不能等价为已完整实现原始二维反馈/固定脚跟强化学习方案后的失败。

由保存的p/r/v/omega_(i-1)及reference_i重建447次实际输入，对官方primitive/composer各自float32 running统计核对：无支撑1.889%的归一化分量越±5（任务分量4.065%），支撑0.286%（任务0.700%）；支撑动作分量无±1裁幅。越界比例只说明截断频率，不能证明训练分布内或排除域差。命令在上述diagnose入口加--observations-only追加已有diagnosis.json，日志diagnose_observations.log保留。无需加载策略做新控制或重新训练。
