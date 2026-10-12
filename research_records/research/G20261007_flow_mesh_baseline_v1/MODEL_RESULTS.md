# Flow Matching 模型结果详细记录与解释

更新时间2026-10-12。配套 [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)。本文件读取已保存产物和当前源码；本次没有重跑历史训练、360重放、采样或测速，也未加载神经权重。唯一本轮架构执行是创建随机模块统计参数；当前原帧评价导出新增检查为106项通过。下列数字须按各自实验条件理解，不拼成同一个模型的训练结果。

## 1. 总判断与证据层次

目前不能认定信息条件FM优于同容量回归，也不能认定真实pose/beta恢复准确。旧人工任务中的回归更好；新模型还未训练；cross_block_sketch晋升已停止；整链速度门失败。研究方向保留，但任何优势/创新/真实精度说法均未证实。

训练结果、真实观测上的随机网络重放、真实统计解码的一帧人工几何实验、数学反例和接口测试分别列出。106测试代表这些声明的工程契约覆盖，不是106个真人样本，也不是106次精度评价。结果保留无提升、数值拒绝、模型资源缺口和不适用指标；只有真实预注册对照允许讨论方法收益。

## 2. 历史已训练合成baseline：不能归到next或H8

来源verification_summary.json，training目录与prediction目录均在该JSON明确列出。4组各1000优化步，单candidate，FM8 NFE，regression1 NFE。输入是人工低维运动、SMPL-H前22个运动学关节（非COCO17）合成针孔观测（f1/baseline0.3m）与关节heatmap RGB proxy接口，使用真实6890顶点人体统计解码，但不是实际RGB/鱼眼真人精度。表中loss的目标在FM与回归之间不同，不横向比较loss数值。vertex_consistency是同人工参考下的顶点一致性，单位mm。

| 模型 | 参数 | 初始验证loss | 最终验证loss | 顶点一致性mm | 乱序条件mm | NFE | K |
|---|---:|---:|---:|---:|---:|---:|---:|
| flow | 144019 | 1.57123064995 | 0.0533562041819 | 110.466100276 | 443.439155817 | 8 | 1 |
| regression | 144019 | 0.54326236248 | 0.00326271215454 | 50.0677898526 | 468.930482864 | 1 | 1 |
| no_graph | 135699 | 1.42979085445 | 0.0452832169831 | 99.7431650758 | 453.743338585 | 8 | 1 |
| no_history | 144019 | 1.44004631042 | 0.0796662196517 | 98.2498899102 | 459.533810616 | 8 | 1 |

回归相对FM误差减少54.6759%，FM误差约为回归2.2063倍。因此必须保留同容量回归对照；不能用FM loss下降覆盖这一负面点估计。no_graph参数少8320，不能解释为严格同容量graph收益；no_history同参数但误差也低于FM，历史依赖收益未证实。乱序条件会增大误差，仅支持模型使用条件，不证明该条件具备真实准确性。旧记录无三训练seed跨人物CI，不能把该表升级为总体统计优势。

所有原验证项列于verification_summary.json：1000步日志、合成loss下降、无oracle、旋转/网格合法、人工missing64帧保持invalid、训练与推理重放逐值一致、21行为测试。旋转最大正交误差及det误差均约10^-7，只是SO3数值合法性。以下逐模型原资源路径和误差，保留精度：
```json
{
  "flow": {
    "training": "outputs/flow_v2",
    "prediction": "outputs/fresh_flow",
    "parameters": 144019,
    "steps": 1000,
    "initial_validation_loss": 1.5712306499481201,
    "final_validation_loss": 0.05335620418190956,
    "vertex_consistency_mm": 110.46610027551651,
    "shuffled_condition_mm": 443.43915581703186,
    "nfe": 8,
    "candidates": 1,
    "max_orthogonality_error": 4.76837158203125e-07,
    "max_determinant_error": 4.172325134277344e-07
  },
  "regression": {
    "training": "outputs/regression_v2",
    "prediction": "outputs/fresh_regression",
    "parameters": 144019,
    "steps": 1000,
    "initial_validation_loss": 0.5432623624801636,
    "final_validation_loss": 0.0032627121545374393,
    "vertex_consistency_mm": 50.06778985261917,
    "shuffled_condition_mm": 468.9304828643799,
    "nfe": 1,
    "candidates": 1,
    "max_orthogonality_error": 4.76837158203125e-07,
    "max_determinant_error": 4.76837158203125e-07
  },
  "no_graph": {
    "training": "outputs/flow_no_graph",
    "prediction": "outputs/fresh_no_graph",
    "parameters": 135699,
    "steps": 1000,
    "initial_validation_loss": 1.4297908544540405,
    "final_validation_loss": 0.04528321698307991,
    "vertex_consistency_mm": 99.74316507577896,
    "shuffled_condition_mm": 453.7433385848999,
    "nfe": 8,
    "candidates": 1,
    "max_orthogonality_error": 3.5762786865234375e-07,
    "max_determinant_error": 3.5762786865234375e-07
  },
  "no_history": {
    "training": "outputs/flow_no_history",
    "prediction": "outputs/fresh_no_history",
    "parameters": 144019,
    "steps": 1000,
    "initial_validation_loss": 1.4400463104248047,
    "final_validation_loss": 0.07966621965169907,
    "vertex_consistency_mm": 98.2498899102211,
    "shuffled_condition_mm": 459.53381061553955,
    "nfe": 8,
    "candidates": 1,
    "max_orthogonality_error": 4.172325134277344e-07,
    "max_determinant_error": 4.172325134277344e-07
  }
}
```

## 3. 当前next候选：未训练，不能填真实准确性表

全52节点711443参数，BODY22部分707603；G0/G1各711443，合计1422886，宽128/4heads/4blocks。当前没有训练checkpoint，normalization train mean/scale仍null。预注册seed7/17/27是未来训练要求，不能把用于随机工程检查的初始化种子当作三次训练复现。

S3保存随机G0/G1每池32合法候选，完整状态325维，人工观测下J68×169可用，anchor G0:0:15，sigma10px proxy/calibrated=False，G0/G1各8Euler共16网络batch调用、512candidate state evaluations，J339解码状态/23decoder batches。一次非稳定化CPU等工程运行total2476.8178ms，不能与GPU warmup后p95拼表得出提速倍数。保留v1 Schur rank误导失败及v2修复记录。G0.npz/G1.npz为outputs/flow_next_s3_v1中的固定随机缓存，新版本不重写其原观测provenance。

S4 binding重放只读原cache，不运行generator。A4选G1:0:19；A5/A6/A7选G1:0:3。A5与A6相同不是信息机制改善；A7 delta0同A6只是无时间作用对照。选择ID与risk_gap不是误差，缺独立标签时没有MPJPE/PVE/真实β误差。原几何数据为人工S3 cameras/pixels，不能描述为360真实集的A6精度。

S4最新完整决定记录：
```json
{
  "stage": "S4",
  "cache_dir": "research_records\\research\\G20261007_flow_mesh_baseline_v1\\outputs\\flow_next_s3_v1",
  "generator_executed": false,
  "results": {
    "A4": {
      "selected_id": "G1:0:19",
      "risk_gap": 0.00014057044077275815,
      "ambiguity": true,
      "reasons": []
    },
    "A5": {
      "selected_id": "G1:0:3",
      "risk_gap": 0.0028008390123677174,
      "ambiguity": true,
      "reasons": []
    },
    "A6": {
      "selected_id": "G1:0:3",
      "risk_gap": 0.010903186541566856,
      "ambiguity": true,
      "reasons": []
    },
    "A7": {
      "selected_id": "G1:0:3",
      "risk_gap": 0.010903186541566856,
      "ambiguity": true,
      "reasons": []
    }
  },
  "permutation_same_non_tie_selection": true,
  "A7_delta0_same_A6": true,
  "risk_policy": {
    "lambda0": 1,
    "lambda_g": 0.1,
    "delta": 0,
    "source": "analytic_fixture_not_validation_locked",
    "ambiguity_calibrated": false
  },
  "noise": "proxy-only fixed isotropic sigma=10.0px; not detection-confidence covariance",
  "training_executed": false,
  "checkpoint_loaded": false,
  "boundary": "fixed random candidates engineering decision only; no risk/time improvement claim"
}
```

## 4. 360原观测重放：工程因果性，不是精度

来源flow_next_replay_v1.json与flow_next_validation_v1.json。实际输入architecture_h2_results_v3.npz，frames.json提供原pair1157..1516映射；0..359共360帧。331帧当前观测可用，29帧缺失；K32/Euler8，29×32=928个候选显式no_current_observation拒绝。missing hidden change=0，chunk64与11输出maxerror=0，原source masks/reasons保持。331可发布不是331准确帧；不存在独立真值误差。

原strict点3031，作用是来源/几何重放，不做训练标签。鱼眼投影与项目同源实现maxerror5.684341886080801e-13px，只认证同源公式一致。时间为host_read_return_perf_counter_ns，不能称sensor曝光同步。LEFT/RIGHT PnP ID在replay原记录，当前模型不能用数字camera索引替代。

模型为随机G0，完整输出可数值合法但语义不必是正确人体。此检查证明状态和输入边界，不证明真实手部/体型、姿态重建或物理接触。没有利用29失败帧的插值恢复视觉效果。

## 5. 资源与速度：失败门明确保留

flow_next_profile_v2.json：RTX5070TiLaptop、torch2.11 CUDA、float32，1warmup+5repeats，G0K32+G1K32/Euler8、原记录像素入口。含预处理、G0、J、G1、decode和A6aggregate；不含相机采集与2D detector。不同阶段分位数不可相加当作总分位数。单次records原样存于JSON，以下p50/p95取该记录summary：

| 阶段 | p50 ms | p95 ms |
|---|---:|---:|
| preprocessing_ms | 1.7826000003 | 2.00636000081 |
| G0_ms | 16.8725000003 | 18.9493199992 |
| J_ms | 270.198900002 | 294.67698 |
| G1_ms | 27.5543999996 | 28.5958999993 |
| decode_ms | 5.81450000027 | 7.82279999839 |
| aggregate_ms | 23.6936000001 | 25.3002400015 |
| full_ms | 358.644700002 | 385.408600001 |

full p95=385.408600001ms超过33.3ms，不能声称30fps；仅从旧有限测时样本算倒数约2.59Hz也不是生产稳定fps保证。J占主要开销，正负扰动使解码状态成本大；不能把单模型参数小等同整链成本小。旧peak allocated 90242048bytes，含matched controls的peak 120371712bytes；allocator统计不等于全设备进程总显存。

同总采样对照G0 K64/Euler8评估512状态，没有第二模型/J；其速度远快，说明512采样状态不是总计算匹配。旧wall-time校准G0 K64/Euler189为12096状态，校准不读标签；p50=398.233ms对target358.6447ms不在10%内，matched walltime门失败。不把失败对照用于同耗时优势声明。以下全部对照运行数值含不稳定试验，不能挑选最快一次：
```json
{
  "equal_total_samples_G0_K64_Euler8": [
    {
      "full_ms": 52.28790000182926,
      "generation_ms": 25.79130000231089,
      "K": 64,
      "Euler_steps": 8,
      "network_calls": 8,
      "candidate_state_evaluations": 512
    },
    {
      "full_ms": 52.42609999913839,
      "generation_ms": 26.13550000023679,
      "K": 64,
      "Euler_steps": 8,
      "network_calls": 8,
      "candidate_state_evaluations": 512
    },
    {
      "full_ms": 49.997500002064044,
      "generation_ms": 25.63200000076904,
      "K": 64,
      "Euler_steps": 8,
      "network_calls": 8,
      "candidate_state_evaluations": 512
    },
    {
      "full_ms": 51.5927999986161,
      "generation_ms": 26.72089999759919,
      "K": 64,
      "Euler_steps": 8,
      "network_calls": 8,
      "candidate_state_evaluations": 512
    },
    {
      "full_ms": 50.39840000245022,
      "generation_ms": 25.00979999967967,
      "K": 64,
      "Euler_steps": 8,
      "network_calls": 8,
      "candidate_state_evaluations": 512
    }
  ],
  "walltime_G0_fixed_K64_NFE_calibrated_without_labels": [
    {
      "full_ms": 397.2505000019737,
      "generation_ms": 368.4018999992986,
      "K": 64,
      "Euler_steps": 189,
      "network_calls": 189,
      "candidate_state_evaluations": 12096
    },
    {
      "full_ms": 365.5375999987882,
      "generation_ms": 333.08690000194474,
      "K": 64,
      "Euler_steps": 189,
      "network_calls": 189,
      "candidate_state_evaluations": 12096
    },
    {
      "full_ms": 398.23299999989104,
      "generation_ms": 368.51320000278065,
      "K": 64,
      "Euler_steps": 189,
      "network_calls": 189,
      "candidate_state_evaluations": 12096
    },
    {
      "full_ms": 598.1514999984938,
      "generation_ms": 569.6578999995836,
      "K": 64,
      "Euler_steps": 189,
      "network_calls": 189,
      "candidate_state_evaluations": 12096
    },
    {
      "full_ms": 484.28169999897364,
      "generation_ms": 454.08129999850644,
      "K": 64,
      "Euler_steps": 189,
      "network_calls": 189,
      "candidate_state_evaluations": 12096
    }
  ],
  "runtime_only_calibration": [
    {
      "full_ms": 214.35039999778382,
      "generation_ms": 188.7148999994679,
      "K": 64,
      "Euler_steps": 107,
      "network_calls": 107,
      "candidate_state_evaluations": 6848
    },
    {
      "full_ms": 354.84319999886793,
      "generation_ms": 331.55429999897024,
      "K": 64,
      "Euler_steps": 189,
      "network_calls": 189,
      "candidate_state_evaluations": 12096
    }
  ],
  "target_p50_ms": 358.6447000016051,
  "walltime_control_p50_ms": 398.23299999989104,
  "within_10_percent_p50": false,
  "comparison_scope": "same recorded-pixel input, explicit decode and A5 decision; time profile only, no efficacy"
}
```

这些测速早于输入绑定/metadata序列化改动。当前infer resources.total_ms子阶段不含provenance序列化，整调用profile涵盖整调用；本次未复测，不认证当前版本。也没有训练后的精度/速度联合权衡。

## 6. 数值信息、噪声敏感性与体型失败

step1e-3/1e-4/1e-5/1e-6在同固定cache全部A6=G1:0:3。四个方向的有限差分对AD JVP误差分别约1.2015e-7、1.2008e-9、8.6891e-11、8.7957e-10；不是完整独立J真值。最紧rank阈值与最小step导致nuisance rank变化，保留数值病态。仍用step1e-4，未按有利选择ID调step。

固定J/policy，sigma5/10/20px分别选G1:0:10/3/3，A5也同样切换；不能归因A6信息机制。Λ缩放与λ0补偿后B可逐值不变，只证明尺度混淆，不是校准。G1未重新条件生成，不能将固定cache聚合敏感性描述为generator收益。

Gram Schur反例：完全混淆变量的Gram截断pinv可得到Sβ00=1，但direct whitened-J SVD残余=0。必须拒绝“非零Schur rank10等于10个体型参数真实可辨识”的推断。更详细rank/eigen与各step/sigma记录按下列实际摘要保留：

### flow_next_shape_audit_v1.json

```json
{
  "stage": "longterm_first_svd_shape_audit",
  "input": "3x13 explicitly constructed J; nuisance diag(1,1e-4,0), first shape column exactly aligned with weak nuisance",
  "dtype": "float64",
  "control_schur_shape_00": 1.0,
  "control_nuisance_rank": 1,
  "svd_shape_00": 0.0,
  "svd_audit": {
    "jacobian_rtol": 1e-06,
    "equivalent_gram_rtol": 1e-12,
    "nuisance_rank": 2,
    "nuisance_singular_values": [
      1.0,
      0.0001,
      0.0
    ],
    "residual_shape_eigenvalues": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "residual_shape_energy_fraction": 0.0,
    "shape_signal_present": true,
    "boundary": "threshold-dependent local linear diagnostic; no prior or identifiability certificate"
  },
  "tests": {
    "targeted_suite_passed": 32,
    "ruff": "passed"
  },
  "training_executed": false,
  "weights_loaded": false,
  "accuracy_gain_measured": false,
  "speed_gain_measured": false,
  "parameters_full52": 711443,
  "sources": [
    {
      "url": "https://www.netlib.org/lapack/lug/node27.html",
      "used": "rank-deficient least-squares direct SVD reference; own projection implementation, not claimed novel SVD method"
    },
    {
      "url": "https://gitlab.com/libeigen/eigen/-/blob/master/doc/LeastSquares.dox?ref_type=heads",
      "used": "official source search excerpt on normal-equation condition-number squaring; page body not readable in current browser fetch"
    }
  ],
  "failure_history": [
    "first heartbeat tool call required destination=thread; corrected; no research check failures in this stage"
  ],
  "boundary": "numeric diagnostics only; no genuine shape identifiability, accuracy or speed improvement demonstrated"
}
```

### flow_next_sensitivity_summary_v1.json

```json
{
  "stage": "fixed-cache sensitivity verified",
  "protocol": "flow_next_sensitivity_protocol_v1.json",
  "raw_results": [
    "flow_next_sensitivity_v1.json",
    "flow_next_sensitivity_v2.json"
  ],
  "versions": "v2 adds exact provenance gate and full candidate metadata/state preservation audit; v1 retained; identical numerical conclusions",
  "checks": {
    "targeted_tests_passed": 34,
    "ruff": "passed",
    "candidates_masks_metadata_unchanged": true,
    "all_four_directional_ad_checks_relative_error_below_1e-4": true,
    "same_anchor_all_steps": true
  },
  "findings": {
    "step_relative_directional_errors": {
      "0.001": 1.2014993806283578e-07,
      "0.0001": 1.2007602642139552e-09,
      "1e-05": 8.689130622929368e-11,
      "1e-06": 8.795696742193837e-10
    },
    "step_decisions": {
      "0.001": "G1:0:3",
      "0.0001": "G1:0:3",
      "1e-05": "G1:0:3",
      "1e-06": "G1:0:3"
    },
    "sigma_fixed_policy_decisions": {
      "5.0": "G1:0:10",
      "10.0": "G1:0:3",
      "20.0": "G1:0:3"
    },
    "sigma_A5_decisions": {
      "5.0": "G1:0:10",
      "10.0": "G1:0:3",
      "20.0": "G1:0:3"
    },
    "noise_scale_compensated_metric_error": [
      0.0,
      0.0,
      0.0
    ],
    "tight_rank_warning": "step1e-6/J rtol1e-10 nuisance rank56 vs51 at baseline; tight numerical rank is sensitive even when candidate choice stable"
  },
  "decision": "retain step1e-4 engineering default; no evidence for tuning it to1e-5 for efficacy; do not attribute sigma-induced choice switch to information mechanism because A5 switches too; calibrated noise and validation-locked weights remain required",
  "source": {
    "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.differentiate.jacobian.html",
    "used": "step-size convergence and cancellation/error-estimate guidance; original central difference implementation retained"
  },
  "next_action": "Audit whether G1 information tokenization discards pose/shape coupling: current diagonal blocks can be identical for fully confounded vs separately observable states. Build an explicit counterexample and same-capacity candidate before any training.",
  "training_executed": false,
  "weights_loaded": false,
  "efficacy_revision_count_increment": 0,
  "boundary": "single artificial S3 observation and fixed random caches; not a real dataset sensitivity study or shape/pose accuracy result"
}
```

## 7. 跨块信息候选：同规模，但筛选失败

同对角、异交叉的PSD I与I±C反例：默认diagonal_blocks完全无法区分，cross_block_sketch可区分这两个指定耦合。它不增加learned参数，但增加固定buffer和计算：full52 float64 buffer17576bytes，每velocity额外约102583MAC加上log/copy，尚无整链新耗时。存在非零PSD交叉部分落probe零空间，不能称完整信息传输。

固定S3anchor/J的一帧Frobenius工程筛选，完整观测cross energy占全矩阵32.588193%；projectable cross fraction2.4708667%，unseen97.5291333%，触发预先80%unseen停止条件。固定sketch raw能量比138.250631%不是信息保留率。mask情况下比例在2.165966%到3.098235%附近；left/right/drop_wrists/upper/lower与anisotropic/no-observation逐格见flow_next_visibility_v1.json。全missing一格显式不可用，其余10格可用。

结果决定暂停当前cross_block_sketch晋升，不挑seed、不增加channels。Frobenius重建低不证明训练后预测必差，也不是丢失97.5%Fisher信息/精度；只说明它不满足本次工程压缩目标。它仍保留作数值对照，default不变。这里一次人工几何anchor不能外推真人遮挡分布。

## 8. 独立噪声契约：实际参考仍缺失

noise_calibration.py只允许独立原鱼眼像素参考/命名人工known-reference；拒绝当前fit自回投影。拟合保留bias、centered covariance和未中心化二阶矩；精度proxy使用含bias二阶矩与0.25px²数值floor，任何缺cell不可导出假校准。64帧known residual x=4±1时bias4px，centered variance1.015873px²，二阶矩17px²：忽略bias会夸大白化精度。始终calibrated=False。

256帧ARα0.95的known fixture，255相邻对，pair Pearson约0.9594/0.96026；只同序列相邻原帧且0<dt≤0.5s，无missing/gap桥接，不由lag1反推独立样本数。未建cross-view/time完整covariance；不能用这一人工结果当作真实误差结构。

frozen validation fixture：train seed21、holdout seed23，各2048帧，已知sigma2；fit/holdout使用不同人工group。下表是预声明view0/joint0示例而非挑最高覆盖率cell；34cell实际总体coverage90范围另报。每行holdout重复采用同一噪声，仅mean shift20px或scale×2；没有seed搜索。

| holdout条件 | coverage50 | coverage90 | coverage95 | 90%面积px² | 径向P50/P90/P95/P99 px | quadratic P99 | Gaussian参考log score均值 |
|---|---:|---:|---:|---:|---|---:|---:|
| matched_gaussian_sigma2 | 0.4907226562 | 0.9013671875 | 0.9497070312 | 57.9302356724 | 2.38478425/4.28136032/4.90080086/6.06334132 | 9.19520604413 | 4.23149494005 |
| bias_shift_x20px | 0 | 0 | 0 | 57.9302356724 | 19.9696137/22.5275696/23.1401865/24.8156526 | 160.539869862 | 55.6953735582 |
| scale_doubled_sigma4 | 0.1596679688 | 0.4291992188 | 0.5205078125 | 57.9302356724 | 4.76956851/8.56272064/9.80160172/12.1266826 | 36.7808241765 | 7.25035873596 |

匹配sigma2条件的90覆盖约90.1367%，34cell范围88.3301%–91.5527%；10px对照coverage90=100%，面积1446.756882px²约是匹配proxy57.930236px²的25倍。高覆盖可能只是过宽，不能单看coverage。20pxbias下小proxy覆盖0；scale×2覆盖42.9199%，说明固定统计无法覆盖显著shift，保留失败。

表中Gaussian reference log score不是FM likelihood，ellipse不是CFM置信区间。人工帧独立生成不代表真实视频帧独立，当前不报告真实binomial CI。真实independent raw-pixel reference记录0，真实noise fitting未执行，calibrated始终False；没有完成人体分布校准。各行示例全部bias/面积/对照详细字段如下：

### matched_gaussian_sigma2

```json
{
  "example_view0_joint0": {
    "view": 0,
    "joint": 0,
    "admitted_count": 2048,
    "reference_usable_count": 2048,
    "reference_missing_count": 0,
    "training_count": 2048,
    "available": true,
    "reasons": [],
    "trained_proxy": {
      "coverage": {
        "0.5": 0.49072265625,
        "0.9": 0.9013671875,
        "0.95": 0.94970703125
      },
      "ellipse_area_px2": {
        "0.5": 17.438738593284082,
        "0.9": 57.93023567242692,
        "0.95": 75.36897426571097
      },
      "mean_bias_px": [
        -0.13606063198682924,
        0.002537576863731683
      ],
      "radial_quantile_px": [
        2.384784252733443,
        4.281360318155264,
        4.900800857961595,
        6.063341321201188
      ],
      "quadratic_score_p99": 9.195206044126868,
      "gaussian_reference_log_score_mean": 4.231494940049107,
      "reference_boundary": "nominal zero-centered Gaussian ellipse/log score using a proxy second moment; not FM likelihood",
      "bias_change_from_training_px": [
        -0.15368756430428793,
        0.0368622209406049
      ]
    },
    "isotropic10px_control": {
      "coverage": {
        "0.5": 1.0,
        "0.9": 1.0,
        "0.95": 1.0
      },
      "ellipse_area_px2": {
        "0.5": 435.51721806072044,
        "0.9": 1446.756882483093,
        "0.95": 1882.2741005438127
      },
      "mean_bias_px": [
        -0.13606063198682924,
        0.002537576863731683
      ],
      "radial_quantile_px": [
        2.384784252733443,
        4.281360318155264,
        4.900800857961595,
        6.063341321201188
      ],
      "quadratic_score_p99": 0.367641108476961,
      "gaussian_reference_log_score_mean": 6.483296844837557,
      "reference_boundary": "nominal zero-centered Gaussian ellipse/log score using a proxy second moment; not FM likelihood"
    }
  },
  "all_available_cells": 34,
  "calibrated": false,
  "statistics_updated": false,
  "coverage90_range_all_cells": [
    0.88330078125,
    0.91552734375
  ]
}
```

### bias_shift_x20px

```json
{
  "example_view0_joint0": {
    "view": 0,
    "joint": 0,
    "admitted_count": 2048,
    "reference_usable_count": 2048,
    "reference_missing_count": 0,
    "training_count": 2048,
    "available": true,
    "reasons": [],
    "trained_proxy": {
      "coverage": {
        "0.5": 0.0,
        "0.9": 0.0,
        "0.95": 0.0
      },
      "ellipse_area_px2": {
        "0.5": 17.438738593284082,
        "0.9": 57.93023567242692,
        "0.95": 75.36897426571097
      },
      "mean_bias_px": [
        19.86393936801317,
        0.002537576863731683
      ],
      "radial_quantile_px": [
        19.96961371459355,
        22.52756959197971,
        23.14018650856207,
        24.81565255302141
      ],
      "quadratic_score_p99": 160.53986986219522,
      "gaussian_reference_log_score_mean": 55.695373558231985,
      "reference_boundary": "nominal zero-centered Gaussian ellipse/log score using a proxy second moment; not FM likelihood",
      "bias_change_from_training_px": [
        19.84631243569571,
        0.0368622209406049
      ]
    },
    "isotropic10px_control": {
      "coverage": {
        "0.5": 0.0,
        "0.9": 0.77783203125,
        "0.95": 0.98486328125
      },
      "ellipse_area_px2": {
        "0.5": 435.51721806072044,
        "0.9": 1446.756882483093,
        "0.95": 1882.2741005438127
      },
      "mean_bias_px": [
        19.86393936801317,
        0.002537576863731683
      ],
      "radial_quantile_px": [
        19.96961371459355,
        22.52756959197971,
        23.14018650856207,
        24.81565255302141
      ],
      "quadratic_score_p99": 6.158167694079567,
      "gaussian_reference_log_score_mean": 8.456084718440191,
      "reference_boundary": "nominal zero-centered Gaussian ellipse/log score using a proxy second moment; not FM likelihood"
    }
  },
  "all_available_cells": 34,
  "calibrated": false,
  "statistics_updated": false,
  "coverage90_range_all_cells": [
    0.0,
    0.0
  ]
}
```

### scale_doubled_sigma4

```json
{
  "example_view0_joint0": {
    "view": 0,
    "joint": 0,
    "admitted_count": 2048,
    "reference_usable_count": 2048,
    "reference_missing_count": 0,
    "training_count": 2048,
    "available": true,
    "reasons": [],
    "trained_proxy": {
      "coverage": {
        "0.5": 0.15966796875,
        "0.9": 0.42919921875,
        "0.95": 0.5205078125
      },
      "ellipse_area_px2": {
        "0.5": 17.438738593284082,
        "0.9": 57.93023567242692,
        "0.95": 75.36897426571097
      },
      "mean_bias_px": [
        -0.2721212639736585,
        0.005075153727463366
      ],
      "radial_quantile_px": [
        4.769568505466886,
        8.562720636310528,
        9.80160171592319,
        12.126682642402375
      ],
      "quadratic_score_p99": 36.78082417650747,
      "gaussian_reference_log_score_mean": 7.250358735956615,
      "reference_boundary": "nominal zero-centered Gaussian ellipse/log score using a proxy second moment; not FM likelihood",
      "bias_change_from_training_px": [
        -0.28974819629111714,
        0.039399797804336584
      ]
    },
    "isotropic10px_control": {
      "coverage": {
        "0.5": 0.986328125,
        "0.9": 1.0,
        "0.95": 1.0
      },
      "ellipse_area_px2": {
        "0.5": 435.51721806072044,
        "0.9": 1446.756882483093,
        "0.95": 1882.2741005438127
      },
      "mean_bias_px": [
        -0.2721212639736585,
        0.005075153727463366
      ],
      "radial_quantile_px": [
        4.769568505466886,
        8.562720636310528,
        9.80160171592319,
        12.126682642402375
      ],
      "quadratic_score_p99": 1.470564433907844,
      "gaussian_reference_log_score_mean": 6.60404562215792,
      "reference_boundary": "nominal zero-centered Gaussian ellipse/log score using a proxy second moment; not FM likelihood"
    }
  },
  "all_available_cells": 34,
  "calibrated": false,
  "statistics_updated": false,
  "coverage90_range_all_cells": [
    0.40966796875,
    0.47021484375
  ]
}
```

## 9. 接口和评价失败：避免错误数值升级为效果

冻结信息修复前，同frame/time改mask/pixels/noise/K的四项均fallback=False；修复后四项均fallback=True，并区分observation/noise/rig。same-count mask换位也拒绝reweight；缺核验资源时lambda_g0同样回退。原anchor tensor克隆，不因缓存后续变动改变anchor。该修复只认证绑定可靠性，无精度收益数字。

标准化ES已知两点joint fixture=3.570714214271425，β ES=0；单位对应转换后分数同值，仅非floor激活情况。旧bool CI门接受虚填50%改善，当前无配对记录为not_evaluated；one seed/unmatched block/one subject拒绝，混合效应CI[-1,1]不支持改善。两subject仅数学下限，不是power充分。

原population登记门拒绝train/heldout人/原动作/序列泄漏、wrong stratum/method/budget/source、错误frames、完整block遗漏和wrong training-stat origin。实际登记metadata不验证文件内容/真实独立性；不能填independent字符串就获得认证。

当前FrameScores原帧导出测试106项，12原frame-seed槽→12block-seed槽；实际standardized_distributions/candidate_metrics known fixture进入export。缺一个原帧或原raw ES=NaN各保留一个失败块且evidence=None；不拿legal ES0替代raw。两帧known values2/4的块mean3，缺端点后None，不对剩余帧求均值。input逆序不改block结果。JSON原NaN以显式unavailable_numeric类型标签保留，不伪装有限值。

这组known fixture可得到100%相对数值改善，只是人为constructed complete report，不是G1/模型精度提升。全部完整原reports和失败输入见flow_next_blocks_checks_v1.json，本说明不复制5000行重复候选报告；不可把未复制的失败称为已删去。

## 10. 未执行结果与继续/停止决定

真实MPJPE/PVE/shape/手部/接触误差：未执行。真实heldout coverage/width/paired CI、三个训练seed、跨rig外推：未执行。真实噪声校准和已校准ambiguity：未完成。完整采集+2D+本模块30fps：未满足。next模型训练loss曲线与checkpoint：不存在。真实AMASS资源合法取得及同beta basis审计：未完成。不能填0表示缺失。

继续FM研究方向及同规模强对照，但停止当前sketch晋升，不替换mainline。两次performance revision规则尚未消耗为真实效果改版：工程修复不算性能回合。后续只有新数据/授权或可复现缺陷允许推进相应可验收工作；不能不断叠架构或重复工程通过来宣称长期目标达到。

维护要求：每新实验追加输入/范围/模型/种子/预算/对照/数值/失败/CI单位/资源/结论，单独写未执行和旧版本边界。历史结果不可更名为新模型结果，速度重测不可抹去旧失败。独立数据门缺失时，只能给出工程结论。

## 11. 2026-10-12 双阶段必要性审查与信息条件回归修复

协议flow_next_regression_information_protocol_v1.json先于修改落盘。本轮实际核对pipeline.py、state_encoder.py、geometry_information.py、information_encoding.py、joint_flow.py与train.py。G0候选、anchor与Λ来自同一观测历史和已有rig/noise/surface；条件于这些来源及独立采样随机数，计算Λ不会新增独立测量。显式几何表示仍可能补偿encoder压缩、提供原encoder未完整编码的算子或噪声信息，不能从数据处理关系直接推出G1无用或两者上下文等价。

核对到可复现接口缺口：objective("regression", ..., information=Λ)原先静默丢弃kwargs，regression也没有信息参数。现已透传冻结Λ并拒绝意外选项，复用原投影。新增4个测试：22/52节点默认及零信息逐值相等、完整objective与手算分块loss相等、三个信息projection梯度非零而Λ无梯度且参数未更新、非法信息与拼错参数拒绝。完整next测试实际110 passed；参数仍707603/711443。只做数值反传，没有优化器、训练或checkpoint。首次环境内ruff缺包退出；随后uv tool run --offline ruff对四个改动Python文件通过。未安装新依赖。旧106测试是历史范围，不能用本轮覆盖更新其历史结果。

audit_conditioning.py执行并独占保存flow_next_conditioning_audit_v1.json。固定anchor=0、y=x+eta、W=1，y=-1/+1的曲率均1，梯度+1/-1，修正-1/+1；只证明曲率不包含残差方向，当前context仍读取y，不是整个G1输入不可区分的反例。先验N(0,1)、y=1、噪声方差1，正确后验mean/variance=0.5/0.5，再乘同一似然后变为2/3、1/3，方差相对减少1/3。它警示未来guidance目标的重复条件化；当前A6是动作风险，不据此认定A6已改变生成分布。

论文证据及适配决定：

- [Fast and Faithful](https://arxiv.org/html/2609.12953v1)，实际阅读3.1–3.4、式12–23与附录D：将观测算子置入速度估计，内循环共享权重。其线性算子/高斯噪声前提及多次内部求解，不能迁移为本项目非线性鱼眼算子的后验保证或低成本保证。只保留“单生成器中显式残差/算子”作为待检验方向；官方算法文本已读，未核验作者实现源码、未复现性能。
- [On the Guidance of Flow Matching](https://arxiv.org/html/2502.02150v3)，实际阅读3.1–3.2、式1–3及近似条件，并核对[作者仓库说明](https://github.com/AI4Science-WestlakeU/flow_guidance)。guidance指定的是对基准分布的能量倾斜，不能直接把已有观测条件后验当无条件先验再次加同一似然。作者README训练/评价路径已核对，具体guidance源码尚未核验、未运行。
- 新检索到2610.02377，但HTML读取失败，仅保留待读线索；不计为精读成果，不据摘要作实现决定。

继续/停止决定：停止“两个生成器天然增加独立观测信息”的论证；保留G1为未证实的表示/归纳偏置候选。先补齐强回归对照，再审查单生成器的残差方向、完整耦合及算子成本，未通过数学/合法性/预算门前不新增网络。旧profile的J p50约270.20ms、全链p95约385.41ms来自保存测时，本轮没有重测；逐步调用算子可能更慢。创新不能只写“FM+几何guidance”，已有工作已覆盖此通用机制。下一阶段先核对作者具体实现，并对非线性局部更新构造可失败的对照；真实效能门、三seed独立标签和30fps仍未通过。

## 12. 2026-10-12 作者guidance源码与非线性移植反例

本轮先落flow_next_operator_transfer_protocol_v1.json。通过公开GitHub tree API定位作者路径，实际读取flow_guidance的image/gflow_img/inverse/inverse_problems.py与synthetic/guided_flow/guidance/gradient_guidance.py；范围/函数/链接和web受限后raw读取路径记录于flow_next_operator_sources_v1.json。未导入或运行作者代码。guide_fn明确区分当前状态梯度、终点梯度、经过终点预测器回到当前状态的梯度；PiGDM计算model VJP，PiGDM+省略，MC分支依赖支持样本、path权重与能量归一化，不能冒充A6全候选决策。尺度/时间日程也是成本及超参数。Fast and Faithful附录B式69–80/Algorithm3进一步核对：每次velocity含K次算子求解和K次共享网络，默认K5；只减少独立参数并不意味着网络调用减少。未核验该论文作者源码。

audit_operator_transfer.py调用项目实际geometry_observation.project，CPU float64，明确人工相机K焦距70/c128、D0、R=I/T0、256×256；单点[x,0,1]从x4修正至目标x0。原像素半平方cost4306.591569，局部Jacobian4.117647，完整GN步−22.538900令线性预测残差精确0，却使实际cost升至5637.473602；alpha0.5仍5038.720978，alpha0.25才降至2557.989629。所有这三次点投影仍合法，说明仅合法性门也不能保证非线性下降。原点/目标加三次尝试共5次projection调用，未测wall-time；原微小扰动/J与完整人体解码成本不由此估计。

第二个反例：绕Z的−90°/+90°旋转均通过原six_to_matrix合法门，平均columns6D约为零并被拒绝。欧氏条件均值可能不对应合法人体旋转，不能把xt+(1−t)v的平均终点当作无需检查的可解码人体；不是当前随机模型发生该故障的频率证据。两反例断言通过、Ruff离线通过；结果保存flow_next_operator_transfer_audit_v1.json；没有网络、优化器、训练、checkpoint或正式主线改动。

决定：停止直接移植线性HQS、未经保护的端点GN和“均值人体天然合法”三项方案。单生成器方向仍待验证；增加逐候选/逐步非线性解码、回溯、时间尺度可能反而比一次冻结J更贵，暂不新增此生成模块。下一阶段优先检查现有339状态有限差分能否以实际surface/鱼眼反向微分替换，并比较真实Jacobian、拒绝边界、耗时/显存；维数68对169本身不证明速度优势。方法效果和总体目标仍未完成。


## 13. 2026-10-12 实际SMPL-H局部Jacobian自动微分审计

协议flow_next_jacobian_protocol_v1.json在执行前登记，audit_jacobian.py实际调用配置中的本地SMPLH male/MANO和models/smpl/J_regressor_coco.npy。两个已知人工锚点：T=[0,0,3]，beta全0或全0.15，第二点第3节点标准化x旋转0.2；52旋转节点、169切空间维度，fixture尺度，float64，RTX5070Ti Laptop、Torch2.11+cu128。人工相机直接调用tests.test_aggregate.rig，实际fx300/fy290/c320,240，D=[.03,-.01,.001,0]，基线0.2，640×480，全部17点准入。v2/v3输出camera_source误写D0；数值计算始终调用上述非零畸变rig，未改历史结果，更正见flow_next_jacobian_validation_v1.json，runner文字已修复。H对照仅JᵀJ，没有校准precision白化。

有限差分step1e-4/chunk16，339次状态解码；标量/向量化反向微分各1次前向状态解码，但反向成本仍计入耗时。每法1次warmup、3次同步CUDA计时。两个锚点J相对Frobenius误差约1.25e-9/1.21e-9，H约2.05e-9/1.96e-9，均通过1e-4预注册门。只说明这两个合法锚点的局部导数一致。

| 运行/锚点 | 有限差分中位ms | 标量反向中位ms | 向量反向中位ms | FD再加向量反向单次ms |
|---|---:|---:|---:|---:|
| v2/0 | 384.893 | 764.171 | 102.053 | 424.240 |
| v2/1 | 317.697 | 624.541 | 96.479 | 418.567 |
| v3/0 | 1340.685 | 6048.032 | 791.310 | 2352.846 |
| v3/1 | 1394.817 | 6381.402 | 786.554 | 2144.546 |

额外CUDA allocated峰值两运行一致：FD31260672 bytes、标量反向2579456、向量反向106131456。向量反向增加约3.4倍峰值显存；v3耗时显著慢于v2，未控制/记录功耗及竞争负载，不归因于闭包修复，也不选快版本作为稳定速度承诺。旧float32全链profile不可与本轮float64局部直接比较。

拒绝语义尚未等价：FD检查锚点及338邻域状态，raw AD仅锚点。最后一列为保守FD加AD（还计算FD J），只有单次且并非最优guard实现，不据此证明所有保留guard的AD都更慢。没有执行非法锚点/扰动边界的批量对照，不宣称已通过拒绝门。当前不替换geometry_information.local_information，也未改G0/G1、参数或候选预算。下一阶段先判断邻域拒绝合同能否用更简单且可验证的方式保留；若无法保留并获全成本收益，停止此替换方向，转回单生成器机制审查，不继续打磨数值小差异。

失败保留：首次回归器路径拼错在启动期FileNotFound，见failure_v1；随后B023发现循环闭包未显式绑定，修正后v3实际复跑通过；最终离线Ruff及AST通过。未使用优化器、训练、神经checkpoint或独立真值；本阶段是局部算子工程证据，长期方法有效性尚未完成。


## 14. 2026-10-12 邻域拒绝反例与自动微分捷径停止决定

flow_next_guard_protocol_v1.json先登记，audit_guard.py调用实际project与local_information，CPU float64、52节点、fixture尺度、step1e-4。使用明确的解析rigid_surface而非SMPL-H；人工双目f100/c128、D0、256×256、基线0.2。内部点[0,0,1]和边界点[tan((256−1e-5−128)/100),0,1]分别作为唯一完整候选，其他输入固定。没有优化器或神经网络。

内部点原像素[128,128]，正向平移扰动后u=128.00999999997；自动微分有限且原FD信息可用。边界点x=3.34134876465，原u=255.99999合法，自动微分仍有限，但正向扰动后u=256.00081203408越界，原local_information返回information_unavailable:ValueError:anchor_or_perturbation_projection_invalid。所有断言、Ruff、AST通过，原输出保存在flow_next_guard_audit_v1.json。此为可复现存在性反例，不代表真实人体边界失败频率；也不证明FD数值邻域门本身是最佳统计设计。

决定：关闭“anchor-only AD直接替换FD并保持原拒绝语义”的捷径。精确保留现有探针语义的直接实现仍需评估原扰动；此前保守FD+AD实现未见全成本收益，但并未排除未来有证明的界限或专门优化。当前不新增近边界阈值、回退分支或检查网络来追逐这项局部速度；暂停该性能替换方向，保留原FD。下一阶段回到G0/G1简约性：审查单个生成器如何表达残差方向与算子条件，优先已有条件表示的替换/合并，先核对论文假设与现有信息接口，训练和主线仍冻结。真实方法收益尚未完成。


## 15. 2026-10-12 观测身份在池化中丢失：同参数修复

flow_next_identity_protocol_v1.json在测试前登记。本轮逐行读取state_encoder.py、information_encoding.py及joint_flow.py发现：旧tokens=MLP(feature)+joint_id+view_role，随后对准入槽取mean。对固定同一准入集合，任意槽位特征排列π满足sum MLP(feature_π(i))+sum id_i不变，因此“哪些特征属于哪些关节/视图”无法由这段编码恢复。相同完整历史逐帧做此排列也不能靠GRU恢复身份对应。此结论针对特征包置换与相同掩码/年龄，不能泛称任意物理相机变换都不变。

修复把已有joint/view嵌入移到Linear15→128之后、GELU之前，随后执行原Linear128→128；没有增加网络、参数、候选或额外几何计算。相同权重张量形状不代表新旧推理语义兼容；当前next无训练checkpoint，历史随机数值仍归旧版本。G0/G1和regression共用此修复，后续对照不可一方使用旧错误encoder。

test_identity_binding.py使用CPU float64固定seed7/19/31、全准入34槽、人造不同xy；分别交换0/16关节和两视图xy，其他特征常量。修改前6失败/1通过，旧additive公式的置换等式测试通过，确认不是抽样不足。修改后context最大差：关节2.58e-5至3.47e-5，视图2.64e-4至3.97e-4，均超过预注册1e-8；完整next117测试通过（82.45s），含既有缺失/时间/分块状态和回归控制。Ruff两文件、AST通过，JointFlow52仍711443参数。详细seed数值、初次失败及Windows rg通配符检索失败在flow_next_identity_audit_v1.json。没有训练、优化器或效果测量；这些差值只排除指定结构性碰撞，不证明编码可逆、信息充分或真实准确性。

相关文献本轮读取[Solving physics-constrained inverse problems with conditional flow matching](https://arxiv.org/html/2603.14135v1)第2.2节式18–21及附录A的MLP条件注入描述：以联合样本学习直接条件速度场，不要求每次推理额外求导；目标依赖所用联合数据与条件。本文中这些理想条件不能认证我们128维压缩条件、有限网络和Euler8就是正确后验。仅核对论文文本，未定位并核验该作者实现，不计复现。Deep Sets会议摘要作为集合模型背景读过，不以摘要认证本项目修复；置换缺陷与修复依据为上面的源码代数和实际red/green测试。

决定：先修复条件表达，再评价G1是否提供表示收益。保留双阶段候选但不认定其必要；单生成器直接条件FM是已有方法类别，不称创新。下一阶段核对官方条件FM实现及当前encoder没有显式表达的相机/噪声信息，避免把删除G1造成的输入差异误判为网络数量差异。训练关闭、独立有效性门仍未满足。


## 16. 2026-10-12 外部噪声条件审查与公平对照约束

先登记flow_next_noise_condition_protocol_v1.json，再扩展已有audit_conditioning.py。固定obs、解析rigid_surface、store_fixture完整候选、人工rig与尺度，唯一变量NoiseModel.proxy的sigma1/2；CPU，encoder float32、几何float64。G0 encoder输入没有noise参数，两次context逐值相等；实际local_information的J逐值相等，Λ(sigma2)=Λ(sigma1)/4最大误差0。G1 information_features最大差1.386285502。精确标量例prior N(0,1)、y=1，噪声方差1/4对应后验mean0.5/0.2、variance0.5/0.8，说明同观测不决定不同已知噪声条件下的同一后验。没有训练或候选采样，不证明G1实际校准。固定噪声可隐含在权重中，confidence也不能未经标定充当precision。

结果flow_next_conditioning_audit_v3.json及旧v2保留：v2沿用旧boundary误写no neural model，实际调用了随机未训练encoder；v3修正文案并重新执行。两版数值相同；Ruff与AST通过。旧v1仅标量审查范围不被改写。当前改变的仅审计脚本，不增加生成器输入或参数。

相机信息需区别对待：encoder已有origins/directions，不能说完全没有rig；但未显式输入完整K/D或NoiseModel.precision。G1的Λ依赖surface、rig、scale、anchor及precision，也不是这些来源的可逆编码。PnP ID用于绑定，不自动编码校准参数。当前records=0的独立噪声校准清单仍未满足。

公平对照约束：固定rig/noise训练与测试可比较同encoder的单生成器与双阶段；变化noise实验必须对两边提供同一登记noise条件，否则只能称“架构加额外条件”的组合比较。给单生成器相同冻结Λ的对照须完整计入产生anchor/J的成本，不能把G0开销隐藏；独立共享anchor或无Λ方案是不同控制。参数数、总网络调用、候选数和wall-time不能同时用一个K数宣称匹配，需分别登记。暂不新增noise MLP或把noise偷塞入confidence槽。

官方实现核对：[Meta Flow Matching条件教学代码](https://facebookresearch.github.io/flow_matching/notebooks/standalone_flow_matching.html)，实际阅读conditional段Flow.forward/step、训练及采样：训练和采样均显式提供c，c与t/x在首层前拼接；step为两次速度调用的中点法，不是一次Euler。只读取源码，未运行训练、未复制代码；这是类别标签教学例，不是2603.14135作者代码或人体逆问题复现。该逆问题论文作者仓库本轮检索未定位，不声称不存在。

决定：将输入信息公平性列为后续G0/G1消融前提，继续关闭训练门。下一轮回到先进论文的可迁移机制与官方实现，重点比较单模型条件化与推理时修正的目标分布、合法人体表示和总成本；不继续用条件接口检查代替效能研究。
