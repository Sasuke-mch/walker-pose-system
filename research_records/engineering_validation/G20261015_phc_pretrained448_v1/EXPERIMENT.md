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

## 2026-10-15 逐项修正：只保留有效骨架匹配，不把失败组件叠入推荐配置

用户要求逐一尽可能修正、避免堆砌模块。本阶段固定官方权重、原上游人体、beta、448输入和场景，只改现有reference/native/analyze/visualize，不新增网络、滤波器或训练。按单因素逐项做完整448对照；二维反馈两次合理修订无改善后停止，脚跟约束同时看完整皮肤位移和是否跌倒，不把有限值/平均达标称全片成功。此阶段止于已有控制器的接口/几何/反馈/约束可行性，不声称整个原始RL课题完成。

**保留修改：当前固定beta的rest关节偏移替换作者中性骨架偏移。** matched_body_xml由真实SMPL-H shaped rest J生成24体局部偏移，手端粗链对应左右第一指基（SMPL-H22/37）；关节顺序、PD、动作尺度、原碰撞形状/密度保持。并非完整皮肤碰撞体与惯量重建，不称体型完全物理一致。全448前22关节FK与原SMPL-H内部关节距离median0.0000507mm/max0.000262mm（原22.72/48.80mm），单测另用非零beta和随机姿态与实际SMPL-H前向按2e-6m比较。原来旋转转换可逆但骨长不同的实际缺口已消除。

| 单因素配置 | 回投median/P95 px | 全10752槽≤25px | 跌倒判据帧 | 处置 |
|---|---:|---:|---:|---|
| 原支撑版 | 74.53 / 190.44 | 8.95% | 0 | 原对照 |
| 固定30Hz策略节拍 | 287.02 / 1182.75 | 5.08% | 73 | 拒绝 |
| 支撑阻尼改相对目标速度 | 74.06 / 197.64 | 9.25% | 0 | 尾部变差，不作为精度改进采纳 |
| 匹配固定体型骨架 | 60.55 / 159.30 | 13.76% | 0 | 仅在PHC研究候选内保留 |
| 匹配骨架+二维参考反馈gain1 | 68.43 / 191.07 | 11.28% | 0 | 拒绝 |
| 同反馈gain0.25 | 75.35 / 374.37 | 10.42% | 0 | 第二次失败后停止 |
| 匹配骨架+Stage2刚性heel代理约束 | 70.50 / 176.25 | 8.90% | 0 | 仅接触诊断，不采纳为全片成功 |
| 上述约束改精确SMPL-H皮肤heel | 370.50 / 1207.46 | 2.60% | 179 | 失稳，拒绝 |

匹配骨架相对原PHC回投median降低18.76%、P95降低16.35%，最大误差412.98→367.59px，24体参考均误差97.45→71.72mm，448有限且无跌倒、无frame0后重置。皮肤sole最低高度中位30.90→5.74mm，896脚帧无低于−5mm，但不把该高度当触地真值。相同采集段的内部一致性改进，不是训练泛化或独立精度。Stage2全115帧双heel≤10mm从原11帧变为8帧，不能声称所有指标改善；骨架修正没有解决脚跟固定。推荐配置就是匹配XML+原单次到帧策略+原腕支撑，不打开clock/relative damping/pixel/heel开关。

动态接口先以官方standing姿态初始化，10秒301帧、30Hz两肩连续15度余弦小动作（合成参考，不冒称官方步行片或论文复现），无外力、0跌倒、24体参考均误差26.88mm、pelvis最低0.9315m。probe_dynamic.npz及probe_dynamic_v1记录。固定30Hz尝试在原到帧批次内按固定tick重算动作、tick之间保持动作、物理dt仍真实；没有全片重采样或读取下一到达帧，但当前reference_i被用于该批次上一时刻至time_i，仍是原“到帧后回放上一间隔”语义，非严格实时到达事件驱动。结果失败不能推断作者固定30Hz无效，故不提升默认。相对速度阻尼F=900(target-p)+45(target_velocity-v)，target_velocity只由两次已到达目标后向差分，未证明更好的跟踪。

二维参考反馈是现有reference内一个有界GN步骤：当前仿真身体通过当前源SMPL-H的COCO局部附着点近似投影到raw双目，用25px Huber与置信度、右膝14不监督；75维根/局部旋转增量，根最多0.03m、单关节最多0.12rad，单位高斯增量先验，1/0.5/0.25回溯只接受代理像素下降；将有界增量用于原目标FK，不重置仿真、不修改发布网格，不训练策略。参考速度保留原feed-forward，没有宣称纠正参考严格运动导数。最终评价独立使用完整SMPL-H解码，而非该局部代理；代理下降并未转化为真实输出改善，两版结果保留、默认关闭，不再加第三个补丁或调参网格。

脚跟诊断使用PhysX自身位置attractor（K1e5,D1e3，仅平移、不固定旋转），Stage2第一次以当前仿真heel位置建立双脚锚，transition保持，Stage1释放；没有把参考姿态直接写回状态。首版试图destroy不存在API失败，stage_heel日志保留；修为AXIS_NONE禁用并重用两个attractor，stage_heel_v2完整跑完。其解码真实heel段内median6.96/P9524.24/max31.19mm，双heel同时≤10mm为64/115，51失败，不能把中位<10mm当全片通过。约束作用于代理点，是额外假设力，不是独立接触/力测量。

为消除代理点误差，SMPLHHeelSurface直接实现相同固定beta/手姿的两顶点LBS：原52关节树、pose blendshape、真实heel权重，随当前原生旋转计算3387/6787，避免载入整套新网络或Linux模型库。与已有完整448网格解码逐坐标最大差3.326e-7m；另在随机姿态模型前向单测核对。每个物理外步更新attractor的局部作用点，参考中只导出所需固定数组。结果heel median1.97/P9513.98/max28.38mm、双脚87/115通过，仍28失败且179跌倒，说明点定义正确并不自动构成稳定的皮肤约束控制；停止该动态attractor方案，不以更大的力或放松10mm门宣称修好。没有求得可信地面反力，也不能将冻结脚跟的强假设当现实事实。

验证：原7项策略/观测/旋转/时序/缺失检查继续通过，增加匹配骨架的实际SMPL-H关节/heel LBS前向核对、GN FK有限Jacobian和骨长保持，共9项通过；Ruff E9/F、编译通过。原身体估计、原物理轨迹和正式主线不改；所有失败/新完整输出留corrections_v1。Windows复用.venv-phc，WSL原生环境不换；一度尝试下载Linux smplx/opencv纯运行依赖但未完成，主动停止该下载，最后二维GN与精确heel只使用现有numpy/torch，无新增运行包或模型。

复现：先按原reference命令加--match-shape，输出corrections_v1/reference_matched.npz及同名XML；native使用该reference、该XML和--wrist-support即为所保留配置。其他对照在相同native入口分别追加--control-hz 30、--moving-support-damping、--pixel-feedback（复验加--pixel-feedback-gain .25）或--stage-heel-anchor（精确皮肤加--exact-heel-surface，reference须由当前prepare导出heel数据）。各实验summary.json保留实际输入路径、flags和参数；selection.json列出选中配置、所有分母和拒绝原因。analyze加--runs指定已有run及--output隔离目录，不覆盖原比较。新可视化corrections_view使用--candidate corrections_v1/matched_body --candidate corrections_v1/stage_heel_v2，当前任务重新strict/Stage重放，只有新R/t/Stage/时间与参考条件场景逐项一致才导出；网格不平滑、不插值、不吸附，失败保留。浏览器已实际重新载入核对：修正骨架143/447帧可切换、方法链接保持同帧、脚跟未通过方案143帧显示10.5/10.1mm未达双10mm；当前全片曲线链接指向comparison_all，左右同源原图与投影实际加载。实际浏览器截图corrections_view/matched_frame143.jpg，metadata记录检查范围。初次本地服务连接失败，重启仅127.0.0.1的8772服务后访问通过。

阶段决定：保留一个实际改善的几何修正和必要诊断工具，失败分支仅显式可复现对照，不作为推荐系统的模块栈。原输入23.24px仍明显优于修正PHC的60.55px，故PHC仍不替换人体估计器；可靠二维物理闭环、全115帧10mm双heel、碰撞表面/惯量统一及独立held-out验证均未完成，不用增加网络或训练步数掩盖这些缺口。


## 2026-10-16 继续改进：参考旋转对照失败，二维反馈基点错误已修复

本轮目标：借鉴已有控制研究，定位可复现的缺口，而非叠加网络/脚跟/阻尼模块。固定旧448输入、匹配XML、权重、beta、时间、首帧及原腕外力；先只改PD目标，两个预定权重0.5/1，不网格搜索。protocol在orientation_pd_20261016/protocol.json，运行前门为：448有限、0跌倒、回投median与P95均较matched_body降低至少5%、全10752槽25px比例不降、heel最大位移不增且10mm达标帧不少于8。此门未通过，不能事后改门把次要收益称整体成功。

研究核对：SimPoE（CVPR2021，https://openaccess.thecvf.com/content/CVPR2021/html/Yuan_SimPoE_Simulated_Character_Control_for_3D_Human_Pose_Estimation_CVPR_2021_paper.html）将图像运动学修正和动力学控制联合学习，不是把独立像素优化结果接到冻结控制器就自然成立。PHC官方humanoid_im.py的_action_to_pd_targets提供ref_dof_pos+scale*action残差模式（https://github.com/ZhengyiLuo/PHC/blob/master/phc/env/tasks/humanoid_im.py），但当前权重是绝对目标动作，不能直接改成残差含义冒称复现。PhysHMR（https://arxiv.org/abs/2510.02566，官方https://github.com/fengq1a0/physhmr）把像素提升为空间射线，结合教师蒸馏和RL训练；官方输入还含GVHMR特征，并非现有双鱼眼即插即用。Mimic2DM（https://jiann-li.github.io/mimic2dm/，代码https://github.com/jiann-li/mimic2dm_release）直接以重投影训练并聚合多视图；公开示例为运球/四足及旧IsaacGym栈，未验证本项目人体、视角和助步器。这里只借鉴并核实机制，未宣称复现上述完整方法或完成新策略训练。

旋转对照：上一轮PHC实际关节PD指令相对参考局部旋转差median20.93/P9567.06deg，但该差不等于错误，可能承担平衡补偿。reference_pd_action在SO(3)上将PD指令朝当前参考移动，weight0保持原动作、1为纯参考PD（策略调用0）、0.5保留一半策略目标。只写电机目标，真实p/r仍由PhysX积分，无输出姿态混合/插值/重置。实际单测核对0/1端点、中点旋转距离及共同世界旋转不变性。

| 配置 | 有限/448 | raw回投median/P95 px | 全10752槽≤25px | 跌倒判据帧 | 决定 |
|---|---:|---:|---:|---:|---|
| 输入SMPL-H，无重力仿真 | 448 | 23.24 / 84.28 | 53.19% | 不适用 | 图像一致性对照 |
| 匹配骨架PHC | 448 | 60.55 / 159.30 | 13.76% | 0 | 推荐保持 |
| 纯参考旋转PD | 448 | 592.91 / 1613.83 | 1.15% | 348 | 拒绝 |
| 一半PHC/一半参考旋转目标 | 448 | 67.20 / 190.06 | 11.08% | 0 | 拒绝 |
| 旧有不同基点二维反馈 | 448 | 68.43 / 191.07 | 11.28% | 0 | 含已确认实现错误，仅历史对照 |
| 修复基点后的二维反馈 | 448 | 61.26 / 161.62 | 12.52% | 0 | 修复正确性，但不默认启用 |

纯PD回投有限10424/10752，一半目标10750/10752；缺失/不可投影槽在完整比例中保留为失败，有限分布不冒称全槽。纯PD平均参考位置误差728.80mm、最低pelvis0.105m，腕力饱和1740侧子步；一半目标为87.95mm，原匹配版71.72mm。两版本停止，不再靠PD权重压姿态或增加外力。

**本轮确认并修复了上一轮自己实现的二维反馈错误。** 旧correct先在当前仿真姿态处线性化和回溯，却把所得增量施加到原始参考姿态；内部显示的下降并不对应返回给控制器的那个目标。143帧反例将观测设为参考精确投影，再令仿真整体X偏30mm：旧代理cost12379.59→46.98，却把原本0px的参考改为median30.56/max72.83px，根另移−26.18mm。原test回归实际失败（最大坐标差28.61mm）后再修；before反例/失败日志均保留。旧两反馈版本不能用来证明正确二维反馈机制必然无效。

修补仅统一计算基点：在当前源参考处线性化、同一基点回溯并直接返回该FK，correct(i)不再接收另一仿真姿态，避免混用；没有新增模块、重置状态或修改像素/评估点。合成回归核对：精确参考不被挪动，有偏移观测时返回姿态的实际代理cost必须与记录cost_after相等且下降；真实143帧精确观测反例修后median约1.31e-5px/max5.56e-5px，仅浮点/FK微差，最大身体坐标差6.38e-8m。

随后固定原gain1单独完整448验证，输出pixel_same_base_v2；447个当前帧代理步骤下降，但这仍不保证低层跟踪后像素更好。较旧错误反馈，median68.43→61.26/P95191.07→161.62；较保留版60.55/159.30仍恶化1.16%/1.46%，25px比例也降低，未过预定门。真实heel max578.43→247.32mm（匹配版423.22mm），双heel≤10mm由8/115变12/115，仍103失败；参考均误差74.21mm，原匹配71.72。CPU每步median14.77/P9518.70ms，原匹配6.24/7.79ms；不把不同批次时延当严格性能认证。部分最大误差/脚滑代理改善保留，但不能用它们替代整体门或10mm完整要求。该分支保持显式开关、默认关闭；修复代码保留，原最优配置不变。

运行与复现：复用既有WSL Ubuntu-22.04 walker-phc Python3.8/torch1.13.1 CPU和原PhysX，无新增模型/依赖。native用corrections_v1/reference_matched.npz/XML及--wrist-support，两个旋转对照分别加--reference-pd-weight 1或.5，输出pd_reference/pd_half。修复反馈用corrections_v1/reference_feedback.npz、reference_matched.xml、--wrist-support --pixel-feedback，输出pixel_same_base_v2。初次误传不存在的reference_feedback.xml，在frame0前加载失败，pixel_same_base目录和日志保留；只纠正XML路径后重试，不是另一次算法调参。reference/policy/physics原始结果未改，分析使用analyze --output隔离：analysis保留旋转对照；analysis_feedback保留旧/修后反馈及匹配版，同完整原图10752槽。所有参数/来源/门在protocol，数值与选择在selection.json，完整NPZ/日志本地保留。

验证：原9项及新增旋转目标和同基点回归共11项通过，Ruff E9/F及编译通过。analysis_feedback/comparison.png实际检查，图上所有曲线为原采样值、缺失保留，无平滑/后吸附；没有新三维页面或替换旧最好结果页面。当前任务没有独立3D/接触/泛化真值，所有改善或退化仅限同检测器/同场景条件下的工程证据。

阶段决定：本轮完成一个被反例证明的实现修复，未获得新的整体姿态精度提升；保留匹配骨架PHC，二维反馈/旋转伺服默认关闭，正式人体主线不变。停止“冻结PHC后再补一个伺服”的局部扩展。后续研究应以Mimic2DM/PhysHMR的直接图像条件与联合训练为机制参考，先落实适合现有双鱼眼的观测/动作/训练数据和独立评价，再训练单一控制策略；该学习路线尚未实现，不能把本轮解析GN修补算作新RL网络或论文级创新。
