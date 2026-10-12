# Flow Matching 模型结果详细记录与解释

更新时间2026-10-12。配套 [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)。各节按实际执行条件持续追加；历史训练、360重放及测速在本阶段未重跑，未加载神经权重。本阶段新增SO3候选的工程检查见第24节，早先106项评价检查保留为当时的范围。下列数字不拼成同一个模型的训练结果。

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


## 17. 2026-10-12 Flower方法与官方实现移植审查

实际读取[Flower v2](https://arxiv.org/html/2509.26287v2)的先验假设、局部Gaussian近似、Algorithm1和第6节式21–23；读取[flower.py](https://raw.githubusercontent.com/mehrsapo/Flower/main/pnpflow/methods/flower.py)、flower_cov.py中model_forward/BtB/cg/solve_ip及flower.yaml。路径、范围、执行边界在flow_next_flower_source_audit_v1.json；GitHub API网页读取失败后公开REST/raw成功。未运行作者包或下载权重。

其核心是无条件先验的终点估计、观测修正、时间推进；线性假设下Gaussian近似成立，非线性扩展提出额外Langevin类采样。当前JointFlow已条件于观测，不可直接当该先验；人体非线性和R6D合法性也不由线性理论覆盖。默认100步×5轨迹为500次网络调用；flower每步一次CG(max50)，cov版两次(max100各)，最后平均图像样本。少一套网络参数不代表总成本更少；本项目需完整人体候选，不能直接平均R6D。此为移植限制，未作速度实测或方法效果复现。

源码CG判断r_norm.sqrt().all()<eps先产生布尔值，并非所有残差小于容差。预注册后CPU float64独立检查eps1e-5：残差[1e-6,1e-6]实际False/期望True；[0,1]实际True/期望False。两个断言通过，保留原数值；仅复现表达式，不声称已测作者整体失败率，也不因此否定论文理论。无外部写入或提交作者仓库。

决定：不移植Flower端点/CG/重噪声机制。保留“区分先验与观测条件后验、计入全部求解成本”的评价原则。本轮没有修改模型。通用线性逆问题路线已出现重复的不适配，下一轮检索直接相关的人体姿态/第一视角条件flow及几何表示，优先寻找能替换当前表示或双阶段的统一机制；不继续为线性solver添加非线性修补模块。训练及正式主线继续冻结。


## 18. 2026-10-12 StableHand质量路径审查与同起点候选

本轮阅读[StableHand v1](https://arxiv.org/html/2605.18553v1)第3.1–3.3节式1–11与实验设置：四通道质量需要误差监督及质量网络，生成模型使用整段时间attention。式8训练起点为q*x1+(1-q)*epsilon，式11推理换成q*y+(1-q)*epsilon。本文的同起点判断只涉及这两条公式，不据此否定实验指标。当前项目缺相应独立质量标签，且因果合同不同，不直接加入质量网络和整段特征。

官方源码实际读[diffusion/flow_matching.py](https://raw.githubusercontent.com/huajian-zeng/stablehand/main/diffusion/flow_matching.py)全部sample函数、sample/infer_clips.py的质量广播及调用，以及docs/enhanced_version.md。发布版以min(q)截断到[0,.95]为t_start，再传有效时间q+(1-q)*t；不是v1的从tau0积分。增强版还有残差腕轨迹和几何模型变化，未读取训练实现，不能假设论文训练公式逐项对应已发权重。未导入作者模型/下载权重或运行复现。

flow_next_quality_bridge_protocol_v1.json先登记精确标量检查。设真值1、观测1.2、epsilon0，q=0/.8/1的训练/推理起点差分别0/.16/.2，严格等于q*(y-x1)。q1原样本速度目标为0，不能把“高质量”当作“无误差”；但未训练网络在偏离训练支撑的点上如何输出未知，不宣称实际模型一定冻结失败。

保留一个简约候选：b=q*y+(1-q)*epsilon；训练路径x_tau=(1-tau)*b+tau*x1，目标速度x1-b，推理同样从b出发。三点解析端点均1，q1时仍有−.2的误差修正目标。这是普通条件桥的具体设定，不称原创理论，也不证明有限网络/数值积分学到正确分布。它改变源分布而不要求另加速度模块；前提是y已在同一完整状态空间、训练取得部署方式的y且标签独立。现有COCO17并不等于325D状态，若再训练一个全状态proposal只为此初始化，可能又回到双模型，所以本轮不接入。完整数值与来源在flow_next_quality_bridge_audit_v1.json。

检索边界：PoseD-Flow项目页可读，paper按钮读取失败、官方仓库URL404，未计正文或源码阅读；FMPose3D已有历史实际审查，本轮不重复。下一阶段先核对现有合法资源能否在不增加生成器、无未来帧与无GT泄漏条件下提供同状态起点，尤其beta/手部未知的处理；没有此前提就放弃质量桥，不为制造起点堆预测模块。训练关闭，未取得效果证据。


## 19. 2026-10-12 部分观测起点资源与随机性边界

核对schema.py、data_firewall.py、candidate_store.py及实际wilor_mano_prior.py：COCO17仅为条件，不能直接当局部旋转/体型；WiLoR convert_record在约定、链序、镜像门后只返回每手15局部旋转，审计明确不迁移全局朝向、平移、形状。现有离线拟合包含全片共享beta/手姿与联合优化，不能直接充作无未来输入的在线起点；flow_next_data_manifest_v1.json仍records为空。本轮没有重跑模型或扫描原始数据。

纠正上一节过强的完整观测前提：若未知分量q=0，b=q*y+(1-q)*epsilon可以只使用部分同空间观测；因此不必先预测全部325D。仍须建立手局部姿态绑定、归一化、可用性及训练同源构造；未知NaN不能靠0*NaN掩盖。部分WiLoR起点也新增了当前next的输入信息，不能与COCO17-only基线混称纯结构消融。

预注册flow_next_bridge_support_protocol_v1.json后计算固定y/q的源协方差diag((1-q)^2)。q=0/.8/.99/1时单维方差1/.04/.0001/0；要得到单位标准差目标，线性增益为1/5/100/不可行。全q1时源是点，确定性ODE不能产生条件多解。部分q1只降低源秩，稠密动力学可向该坐标传播其他维度噪声，不能断言这些输出坐标必然不变；规则Lipschitz流的支撑维度限制也不能直接等同325D R6D中的人体内在维度。解析反例不构成StableHand实测失败或JointFlow精度结论。数值与资源入口在flow_next_bridge_support_audit_v1.json，JSON及断言通过。

决定：不将硬q1冻结作为概率模型默认，不增加质量预测器、完整proposal或噪声补丁。质量桥分支暂停实现准备，普通高斯源条件FM继续作为首要强对照。本次纠正可行性推理，但没有满足独立标签、训练授权与有效性门。下一阶段回到G0/G1简约性，明确能真正判别必要性的最小消融及仍缺数据，避免继续凭随机/解析检查迭代新架构。


## 20. 2026-10-12 无额外数据确认、最小消融与信息回归推理修复

用户直接回复“暂无”，明确未提供额外未登记标签/噪声参考；实际validate_manifest返回training_data_ready=false、records=0，noise清单records也为0，训练/神经checkpoint开关继续false。不能把当前360/448拟合、已有合法人体资产或随机fixture当作这些缺失数据。上轮被用户回复打断，未运行训练进程；本轮接续同一阶段。

消融收敛为Flow/回归×无Λ/有Λ四个最终学习器，细节追加在现有flow_research_plan_v1.json的decisive_generation_ablation_20261012。主变量为信息是否输入或训练目标，种子7/17/27、每seed同初始参数/数据/配方，CFM固定K32/Euler8，回归K1。F_info必须与重新独立训练、配方匹配的F0比较；不能仅推理关闭已训练G1的Λ，也不能只和训练不同的冻结G0比。信息投影15104参数在无Λ条件中不活跃，711443名义参数相同不等于有效容量完全等同。

生成阶段固定raw own-T COCO17标准化ES和A4完整medoid，beta分开，不同时调A6/A7。输入作用隔离时两边同样计入共同G0/J；实际单模型部署另记可省去的G0/J成本，不能用免费缓存Λ。K64单模型与G0K32+G1K32只能说512候选速度评估相同；前者8批网络调用、后者16批且另有FD最多339状态，wall-time待实测。V-statistic ES受K影响，K32为主比较；K64预算对照报告两个K32面板及原K64结果，不能以增加K本身证明分布更好。效能门继承既有5%收益/配对CI/强几何2%恶化及失败、覆盖等标准；最多两轮有意义性能修订，当前未执行，既不能据缺证据认定G1有用，也不能宣称已证无用。

本轮源码审查发现此前第11节的regression/loss透传未覆盖部署infer：candidate_store.infer回归分支仍拒绝information且不传入model.regression。先登记flow_next_regression_inference_protocol_v1.json，再扩展已有test_regression_information.py；修改前22/52两项都在旧拒绝分支失败，2failed/4passed。仅删除information禁止条件、保留K1门并透传Λ。修复后35相关regression/joint/encoding/binding测试通过（2.88s），两文件Ruff与AST通过；不把本轮35称完整next测试。

新测试通过实际infer确认：非均匀Normalization后raw与直接regression逐值相等；换stream/seed仍确定；none与zeroΛ逐值相等；非零Λ实际改变输出；K2、错维度、NaN拒绝；缺观测不发布、原失败原因保留；一候选、一网络速度调用、一状态计数；参数707603/711443及参数值都不变。未用训练优化器、神经checkpoint或真实数据。审计曾读不存在的test_candidates.py，失败保留于validation，没有当作测试执行。结果见flow_next_regression_inference_validation_v1.json。

当前仅有匹配对照的单帧接口，train.py依然没有数据loader/优化器/权重生命周期，完整四法训练评估runner也未实现，不能称效能消融已可运行。后续阅读仍可继续；新机制必须先说明有何可判别优势及需要什么监督，避免无数据时持续扩架构。效能验证等待资源与训练门满足；长期目标未完成。


## 21. 2026-10-12 ManiPose骨架流形审查与真实表面算子反例

按cv-paper-reader作聚焦审查：阅读[ManiPose v2](https://arxiv.org/html/2312.06386v2)3.1/3.2、4.1/B.1、5.1–5.3及Table4、AppendixD；不是全文/全部证明复现。作者模型把2D窗口映射为共享骨长与K组6D旋转，经FK得到根相对骨架；WTA训练最佳头，score预测头的胜出概率。它不是SO3概率流，也不等于解剖合法性。其结构一致性来源于rigid skeleton输出约束，缺完整T/beta/SMPLH表面语义。

实际阅读作者pose_decoder、forward_kinematics、rotation_tools、rmcl_manifold_mix_ste、losses中的WTA/score与config.yaml。weighted_ave在解码坐标上求平均；best_score选一个整骨架，oracle需GT。代码score采用softmax后binary_cross_entropy，不能按多类NLL假定同一实现；论文Algorithm2的local×parent与代码parent×local顺序不同，需要坐标审计，未据此认定作者整体错误。6D转换有硬编码CUDA epsilon且没有本项目退化返回valid门，不移除现有合法性检查。源码读取范围/URL见flow_next_pose_manifold_sources_v1.json，未运行作者包/训练/权重。

公平性：配置T243、旋转宽512/8层、骨长宽128/2层、K5及TTA与项目因果GRU128/Euler8不同；Table4名义34.44M与本项目711443也非同规模。Table4使用oracle，不能替代实际发布输出表现。当前3D关键点数据标签不能凭此成为完整52节点/手/beta标签。未核验当前全局SOTA，不声称复现作者指标。作为多候选强对照有价值，暂不堆其骨长网络或多头分支；CFM是否必要仍须同信息/预算效能比较。

本地反例在执行前登记flow_next_pose_manifold_protocol_v1.json。audit_pose_manifold.py只解码两个完整人工姿态：同beta0、T[0,0,3]，第一组identity；第二组节点18/19绕z±70°、节点4/5绕x60/45°。实际使用已登记本地SMPLH male/MANO和J_regressor_coco.npy，CPU float64、无神经网络/优化器。全部51内部父子骨段与8条COCO四肢连线逐项保存，不筛失败；投影未执行，所有旋转通过现有SO3合法门，网格与点有限。内部骨长最大变化2.77555756e-16米；COCO连线最大变化0.006461397707米（6.4614mm），均通过预注册1e-7/1e-4门。结果见flow_next_pose_manifold_audit_v1.json；Ruff/AST通过。

这证明固定体型且内部骨架刚性的合法人体仍可产生不同COCO表面连线长度。把其当刚体骨段强制恒定会改变既定观测算子的契约，不能把6.46mm称实际人体误差，也不能把骨长损失直接搬来改善重建。原SMPLH已经承担内部骨架约束，再加一个独立骨长预测头未获必要性证据。本轮不修改模型或正式主线。

同时压力测试“任意非退化姿态都使每条骨段的条件均值缩短”的过强用法：对骨段向量d，精确恒等式||E[d]||²=E||d||²−trCov(d)，严格缩短需该骨段Cov非零。两组合法单位边总体分布中，一条边歧义、另一条固定时均值长度为0/1；两条都歧义时0/0。总体姿态非退化不能推出每条边都随机。该解析检查不否定作者实验，也不支持取消本项目参数回归强对照（其损失针对packedR/T/beta，并非自由Cartesian joint MSE）。

决定：不接入COCO骨长恒定损失，保留真实surface与内部FK角色区分，ManiPose作为结构与决策强对照继续记录；本轮取得的是排除不合适约束的算子证据，非生成精度成果。下一阶段对SO3概率路径本身阅读原始RFM与官方实现，先检验合法旋转/可微路径是否足以支持概率与成本结论；不以骨架流形名称代替实质分析。数据门和训练关闭继续有效。


## 22. 2026-10-12 RFM概率路径、SO3局部chart与导数边界

聚焦阅读[Flow Matching on General Geometries v3](https://arxiv.org/html/2302.03660v3)3.1式3–10/Algorithm1、3.2条件/式13、AppendixE；未计完整证明和基准复现。RFM要求源/目标测度、端点一致的条件路径、切向速度与相应metric一致。“简单几何simulation-free”针对训练目标路径，不表示推理无需ODE或速度调用，也不认证有限训练模型的真实后验。

作者riemannian-fm的rfm_loss_fn实际调用random_base、geodesic、timeJVP和manifold.inner；Meta新版GeodesicProbPath同样由log/exp路径作时间JVP。实际读两个仓库的manifold导出、Meta geodesic和RiemannianODESolver源码，读取范围/URL在flow_next_so3_path_sources_v1.json。已查导出没有专门SO3类型，不据此声称理论不支持SO3或其它包没有实现。新版Euler/midpoint/RK4每步为1/2/4次速度调用并带点/切向投影；几何路径的timeJVP不能等同完整人体/鱼眼观测Jacobian成本。没有运行作者包、训练或权重。ar5iv和solver文档读取失败后arxivHTML/官方raw成功，失败记录保留。

可保留的简约方向是替换旋转表示：SO3^52×R3×R10的内在维数169，而当前6D嵌入状态为325；用角速度与群积分可让生成和局部信息采用一致的旋转几何，但需要新的路径/尺度/输入输出/积分合同，不能靠维数宣称参数或耗时减少，更不能据此删除G0/G1/J。合法SO3也不等于解剖/接触合法。本轮未实施新生成器。

在先登记flow_next_so3_path_protocol_v1.json后，audit_so3_path.py用CPU float64调用当前shortest_log/skew。0.4rad普通旋转通过；另一输入π−0.0005rad仍通过SO3合法门，合并后整个log张量明确返回near_pi_rotation_chart。默认margin0.001rad排除的是正测度邻域，不能解释成仅处理概率零的π点。

对假设独立Haar源与固定目标，单位四元数可由标准Gaussian4归一化得到；θ=2acos(|qw|)，角密度(1−cosθ)/π，CDF=(θ−sinθ)/π。因此p近π=(epsilon+sin epsilon)/π=0.0006366197193。独立22/52/16×52/32×52旋转对的整张量拒绝概率分别1.3912%/3.2572%/41.1299%/65.3431%。seed7的200000次抽样出现126次尾部，比例0.00063，与解析值在5binomialSE内。此为假设源的局部门适配证据，不是当前归一化6D源、学到的候选或有相依性的A6两两风险失败率；不能把独立公式直接套到K²两两比较。

第二项失败证据：当前log(Exp(w))在w0前向全0，但对w反向Jacobian九项为NaN，理论应为I。v1/v2用显式nonfinite_NaN保存全部九项，不将有限前向当可微认证。重要的正对照在v2执行前补充登记：固定端点log后，只对Exp(t*skew(omega))的时间求JVP，单位/普通旋转速度均有限，与R(t)*skew(omega)的最大误差0/5.5511e-17。这正是冻源/目标训练路径所需的一种求导，不要求端点log梯度，因此NaN本身不证明RFM训练不可行；它限制的是对端点/当前旋转再求导的用途。

v1/v2原结果与数值/非有限证据保留，Ruff/AST、普通log/概率抽样/timeJVP断言通过。没有优化器、模型推理、人体资产解码、随机模型效果或速度重测。当前决策辅助函数继续保留其近π回退，未把它硬改成全局可微log。停止“只替换6D路径、无改动复用现有局部helper即可获得RFM”的捷径；RFM候选保留，下一阶段审查专用SO3实现对π分支与源测度的处理，尤其避免拒绝/重采样改变训练源；仍需独立数据门，未宣称模型创新或整体完成。

## 23. 2026-10-12 FrameFlow专用SO3实现与匹配源路径门

使用cv-paper-reader聚焦读取 [Fast protein backbone generation with SE(3) flow matching](https://ar5iv.labs.arxiv.org/html/2310.05297) 第2.1/2.2节式2–14、先验/预对齐、第3.1节Table1及第4节讨论；方法图只读图注，不称视觉核验。官方README要求用legacy分支对应该2023短论文；保留其so3_utils.py、interpolant.py、base.yaml和MIT许可，未下载蛋白质数据/权重。源码范围见flow_next_frameflow_so3_sources_v1.json，不把最新main的motif方法混作本轮复现。

论文使用SO3×R3的加性metric和测地路径，以网络预测clean端点，再转换为切向速度。实际_corrupt_rotmats用R0=R1·E、E~IGSO3(1.5)训练，推理_uniform_so3改为Haar；训练linear、推理exp_rate10的指数日程也不同。非Haar目标中心源不等于独立Haar源，源/路径的这些变化不能直接获得匹配条件路径的保证。对Haar噪声乘目标旋转确实不改变其分布，故不能仅凭源码含R1就断言源分布改变。未运行作者IGSO3采样器、密度或训练。蛋白质零中心/全局旋转不变与项目相机锚定米制T不同，IPA、自条件和Kabsch未移植。

先登记flow_next_frameflow_so3_protocol_v1.json，新增audit_frameflow_so3.py。仅AST执行11个完整原纯函数，无修改函数体/模块导入或神经网络。CPU/torch2.11+cu128，float64/32；5个轴×7个角含0/1e-8/0.4/π-0.02/π-0.005/π-0.0005/π，另固定seed7、1024×52独立归一化Gaussian4产生Haar源，目标I。所有53248对保留，没有拒绝、截断或重抽。协议阈值为float64 1e-10、float32 3e-5，包含端点元素误差、正交和行列式。

|检查|float64|float32|
|---|---:|---:|
|作者35固定旋转端点超过合同数|4|2|
|作者固定端点最大矩阵元素误差|0.00801235|0.00801244|
|作者53248源/目标端点超过合同数|359|283|
|作者Haar组最大矩阵元素误差|0.01977859|0.01977851|
|独立控制Haar组最大矩阵元素误差|2.22e-15|3.83e-6|
|独立控制固定/Haar端点及时间JVP合同|通过|通过|

上述是无量纲矩阵数值误差与协议不符的旋转对数，不是人体错误率或真实精度；两dtype阈值不同，不能据计数声称float32更准确。原近π分支取(I+R)/2对角根与行符号，近π仍有近似误差；对[-1,2,-3]/sqrt14、π-0.005的合法旋转，所选全局符号使端点误差约0.008，强于单纯浮点容差问题。原函数t0返回I，冻端点t0.4的单位/普通/近π时间JVP均有限并符合各自闭式导数；不能把这个端点问题扩大成时间JVP失败。

独立控制仅对固定端点用SciPy principal rotvec，R(t)=R0·matrix_exp(t·skew(omega))，omega冻结；其值检查不要求端点梯度。这里的pi分支仍非唯一，CPU准备不能冒充全局光滑/可微guidance算子，也未证明最快。v1/v2完整JSON与同名NPZ保留；仅本地保存约10MB数组，包含所有源、作者端点/log/误差/合同和对照log。初次Ruff的导入、typing和受控AST执行标记已修正并复跑v2，端点计数和最大误差与v1相同；Ruff/AST通过，5项作者docstring SyntaxWarning保留，未修作者源码。检查仅用于回答路径端点是否一致，未另建验收报告或执行全模型测试。

决定：关闭原FrameFlow分支和源/日程tricks的直接移植，不继续修补作者分支；保留SO3角速度及群积分作独立表示因素。下一阶段可预注册同条件F0的SO3版本：相同StateEncoder/Transformer、R/T/beta联合状态、匹配Haar训练/推理源、相同预算及回归对照，先用固定端点控制检验目标/积分。不要同时改变观测信息、加入IPA/自条件或声称已删去G0/G1/J；其效能仍需当前缺失的数据与训练授权。当前JointFlow、正式主线和训练关闭不变。

## 24. 2026-10-12 单模型SO3角速度候选：实现检查通过，效能未测

先在已有flow_next_longterm_protocol_v1.json的so3_representation_factor_20261012登记唯一因素和停止条件，再新增so3_flow.py。SO3JointFlow继承现有core，pose_out由6维缩为3维，保持StateEncoder/Transformer/T/beta。内部合法矩阵列+标准化辅助量组成网络输入，body角速度作右乘群Euler，公开输出转换回绑定Normalization供现有CandidateStore读取。默认欧氏模型、G0/G1/J、数据门与训练工厂保留；单模型本身已有F0先例，不把这一候选视为已删除双阶段的成果。

源使用现有固定候选seed的前4个独立Gaussian分量/节点归一化四元数，训练与采样均Haar、线性日程；不用原FrameFlow的目标中心IGSO3或指数推理。冻结端点由SciPy准备principal omega，不对端点求梯度；网络和群采样仍由Torch反传。角速度MSE按每坐标均值取固定1rad单位，T/beta标准化照旧。强确定性对照使用同一模型和现有regression_loss，CFM/回归优化的量不同，训练损失不能作为优势证据。RFM/FrameFlow已涵盖旋转流机制，本次没有新颖性结论。

本次只运行test_joint.py与直接受objective/infer改动影响的test_regression_information.py：20 passed in 2.82s。既有因果/序列化、无观测不发布测试增加SO3分支；新问题对应检查为22/52节点近π端点和时间导数、非交换源旋转下右更新闭式结果、非均匀输出标准化、CFM/回归/采样梯度、不完整标签拒绝、codec错配与Lambda/非法源拒绝。所有矩阵检查是数学/接口证据，没有新人体训练数据、optimizer、checkpoint、原视频重放或真实mesh解码。首次Ruff仅两处长导入格式错误，修正后四个改动Python文件通过；未为导入换行重复功能测试，也未执行全套或新建验收报告。

|候选|完整52节点参数|部分22节点参数|生成/点预测调用|
|---|---:|---:|---|
|原ambient F0|711443|707603|8步默认Euler或1次回归|
|SO3 F0及其同容量回归|711056|707216|8步默认LieEuler或1次回归|

387参数差异来自输出头，不能称与原ambient严格等容量。SO3内部保存3×3矩阵，matrix_exp及CPU训练目标准备有额外代价，未测内存/耗时，不据169维速度声称更快。当前仍缺独立训练与噪声标签；下一步只核算同观测、同K/NFE和完整候选/解码的实际成本，使用现有资源工具，优先判断这个表示因素是否值得保留。成本通过也不能替代既有heldout效能门，未证明G0/G1/局部信息必要或不必要，长期目标未完成。

## 25. 2026-10-12 SO3与ambient单模型固定预算成本：未通过晋升门

在已有长期协议so3_cost_comparison_20261012中先登记输入、预算、次序和1.25倍停止门。扩展现有profile.py的--representation-controls，不建立另一个测量/验收脚本。RTX5070Ti Laptop、torch2.11+cu128、float32、CPUthreads1；单固定观测由现有许可male SMPLH/MANO的I/beta0/Tz3表面投影到fixture rig，原观测字段保持显式。使用fixture_normalization的0.03尺度，不是训练统计。随机seed7，两模型共同StateEncoder/Transformer等参数复制对齐，姿态头维度不同；相同可用输入，不加载权重或优化。

四模式：ambient/SO3 F0各K32×8步，ambient/SO3回归各K1×1调用；同一A4全候选medoid，所有数值合法人体解码，不用J、几何再评分或二次选择。每模式warmup2、记录8次，四模式按循环次序执行，边界CUDA同步。state每次从None开始，是固定帧成本而非流式部署。外层infer计入候选来源/掩码/发布，实际SMPLH解码与A4另分段；候选与失败理由/调用/decoder计数逐次保存在flow_next_so3_cost_v1.json。没有调整步数、按耗时选样或重复获取更快结果。

|模式|生成外层p50 ms|解码p50 ms|A4决策p50 ms|完整p50 ms|完整p95 ms|
|---|---:|---:|---:|---:|---:|
|ambient F0 K32/8|25.31|5.49|4.32|35.20|39.06|
|SO3 F0 K32/8|34.34|5.38|4.45|44.87|52.49|
|ambient回归 K1/1|7.07|5.37|1.32|14.43|18.93|
|SO3回归 K1/1|8.66|4.85|1.13|15.60|17.22|

分段中位数之和不必等于完整中位数。32/1候选在所有记录中均数值合法、动作准入、实际解码vertices有限；计数F0为8batch/256state速度评估、1batch/32state人体解码，回归各1。来源/原候选/失败数组均保存，不据全有限认定真实人体有效。两F0每次allocator峰增量31,074,304 bytes，两回归961,024 bytes；两模型及surface共驻GPU，该口径不是全进程/驱动显存或单模型独占显存。

SO3/ambient完整p50=1.274720998，超过先定1.25门，因此停止当前实现的成本晋升。生成internal p50为27.05/17.68ms；额外时间主要位于生成段，但未作算子剖析，不能全部归因matrix_exp。两生成p95在这个窄范围也超过33.3ms，capture/2D detector未计入，不支持30fps。回归p95较小不证明相同输出质量，也不能据8次样本的p95微差认为SO3回归稳定更快。测前GPU15%/1885MiB/20.62W，功耗、时钟与其他应用负载未锁定；小样本比值是本轮停止依据，非跨设备稳定性能规律。

实际测量完成，profile.py Ruff通过。控制台含人体层WARNING，外部PowerShell将全部stdout作JSON解析失败；保存JSON完整可读，直接读取结果收尾，没有重跑测时。无新测试/快照/验收报告。保留角速度候选的结构合法性研究价值，但速度优势未成立。下一步最多一次有数学依据的3×3指数核替换，复用已有模型/旋转测试和固定成本比较，若仍不满足成本门则停止这条成本优化，等待独立数据验证。禁止把删去G0/J的运行差异当新表示收益；正式主线、训练关闭及长期目标继续。

## 26. 2026-10-12 Rodrigues闭式旋转核：第二轮通过成本门，停止核优化

按已有长期协议在so3_kernel_revision2_20261012先登记最后一次合理修订。阅读 [PyTorch3D官方rotation_conversions.py](https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/pytorch3d/transforms/rotation_conversions.py) 的axis_angle_to_matrix(fast=True)，实际检验到K²=wwᵀ−θ²I和half-angle sinc实现；没有导入作者包或增加依赖/源码快照。此前FrameFlow已给出Rodrigues公式，该数值优化不是架构创新。PyTorch sinc文档页面读取仅返回redirect，公式/实现依据为实际取得的官方源码。

在so3_flow.py添加rotation_exp，K=skew(w)、q=||w||²：R=(1−Bq)I+AK+Bwwᵀ。regular A=sinc(sqrt(q)/π)，B=sinc(sqrt(q)/(2π))²/2；q<1e-4时使用A=1−q/6+q²/120、B=1/2−q/24+q²/720，regular的sqrt先clamp到1e-4。小角Taylor有截断误差，数值等价只按声明容差检查；不是全精度代数恒等。该做法保留零点有限导数，省去通用matrix_exp；训练路径、sample和regression使用闭式核，显式matrix_exp选项仅保留旧核推理对照，固定端点SciPy log继续保留。网络、源、日程、预算和711056/707216参数未变。

本次仅test_joint.py实际17 passed in6.58s，其中新检查CPUfloat64/CUDAfloat32的0、1e-8、阈值两侧0.0099/0.0101、普通、近π、π、2π、10π矩阵值、正交/行列式；float64零/普通/近πJacobian与独立torch.matrix_exp对照，非有限输入不变成identity。新完整同权重采样/回归对照在3e-5内一致；既有CFM/梯度/端点/因果/发布检查仍通过。检查覆盖值/一阶导数与该范围，不认证全部高阶导数/全局误差界。三份改动Python Ruff通过，没有全套、额外验收报告或重复功能运行。

扩展已有profile.py的--compare-rotation-kernels，同次6模式ambient/Rodrigues/matrix_exp×F0 K32/8及回归K1/1；SO3两核复制完全相同权重，ambient与SO3除不同维姿态头共享初始化。其他输入/Normal/真实maleSMPLH/表面算子/A4/warmup2/每组8次与第25节一致，六模式循环执行，所有48记录与失败槽保存flow_next_so3_cost_v2.json。增加第三个常驻模型，但各模式共用同一allocator基线口径。

|同次模式|生成外层p50 ms|完整p50 ms|完整p95 ms|
|---|---:|---:|---:|
|ambient F0|22.85|33.23|37.74|
|SO3 Rodrigues F0|28.17|38.33|45.69|
|SO3 matrix_exp F0|32.12|43.63|48.26|
|ambient回归|7.44|13.79|16.42|
|SO3 Rodrigues回归|7.48|13.74|16.82|
|SO3 matrix_exp回归|8.12|15.59|16.98|

闭式/ambient完整p50=1.1535754，通过先定1.25成本门；同次闭式相对旧核完整中位数约减少12.2%，但仍慢于ambient约15.4%。只作本次随机固定GPU样本的工程判断，不能将跨v1/v2差值全部归为核收益。所有F0每组32、回归1候选数值准入并真实解码有限，预算/decoder计数正确；F0峰allocator增量仍各31,074,304bytes、回归961,024bytes，没有显存收益。测前GPU16%/1801MiB/21.39W，时钟功耗/竞争负载未锁，三个生成p95仍高于33.3ms且不含检测采集，不支持整系统30fps。控制台WARNING通过读取保存JSON绕过，未触发追加测时。

决定：保留Rodrigues SO3候选供今后独立效能对照，结束本机制两轮核/成本优化。成本通过只排除当前过高开销阻碍，没有准确度、校准或概率表达优越性结论；旧核与v1失败保留，默认欧氏工厂、G0/J/G1和正式主线未晋升。下一阶段转回单生成器/局部信息必要性的新机制与新颖性分析，保留同信息强回归和数据门，不继续靠数值核、减K或堆模块制造收益。

## 27. 2026-10-12 单生成器机制重合与后验认证边界

聚焦读FMPE(2305.17161)方法/conditioning/附录成本及官方Dingo loss/sample/divergence/GLU/cache，2610.02377v1的第3.1–3.4节，SBI-SIM(2410.22573)反馈/控制/zero-control与lensing指标/成本。范围/链接和实际缺口已追加现有LITERATURE_ANALYSIS.md，未运行作者模型或保存新PDF/源码快照。原HTML/PDF网页后续读取不稳定，bundled pypdf直接内存解析正文；未把只含摘要的Zhai/PoseD等升级成已读方法。

本轮改变研究判断的证据：单条件流离线从可信参数—观测joint配对学习、在线只积分网络，是已有FMPE/条件transport机制，当前F0已经属于这一路线。它支持保留单模型强对照，不证明当前训练分布/encoder、先验或噪声充分，也不证明G1无收益。SBI-SIM的反馈能为状态相关物理输入提供依据，但原方案另加约10–11%控制参数，lensing耗时10→19s，不能作为无额外代价的模块直接叠加。其zero-control逻辑应继承，误差/概率及原置信度/precision仍需分列。

在长期协议先登记密度符号问题后仅做一次解析缩放控制：x0~N(0,1)，v=ln2*x，x1=2*x0。x=2的真实logq=-2.1120857138；负散度积分相同；按提取的2610式11加号得-0.7257913526，相对密度4倍、总质量4。没有采样/网络/optimizer。该局部问题与提取的式10不一致；未能进行PDF截图核验、没有作者实现，故不扩大成论文整体或其定理失败，明确停止照抄该符号。Dingo源码用负散度的联合ODE和对应输出，独立支持正确公式，未运行原代码。

TV/KL或Metropolis不是免费后验认证：需要真实定义的prior/likelihood/proposal density，及平稳/大样本/二阶矩等条件。当前raw等权候选/A6风险没有这些接口；合法旋转、有限点、重投影低或32候选不补足它们。小训练MSE的mass-coverage结论也需要额外正则性，不能抄到当前未训练人体模型。继续关闭密度/MH/learned-ranker扩展。

下一可检查阶段：只验证SO3单模型的晚阶段完整参数终点上，真实SMPLH到原鱼眼代价能否用一次VJP返回169维角/T/beta方向，保留无效端点/原观测状态；这是新的反馈输入语义，不冒充原FD邻域合同。只有不增第二生成器、不堆control网络、容量与完整成本可控，才考虑替换现有信息接口并预注册新CFM/回归/零反馈对照。外部noise/operator可用信息必须同条件，未知质量不伪装校准。此处仅方向选择，尚无该物理VJP执行、架构改动、训练或有效性证据，整体目标继续。

## 28. 2026-10-12 原鱼眼Huber成本零残差梯度修复

按现有 endpoint_vjp_feasibility_20261012 的首项前提，使用同一合法点 [0.1,-0.05,3] 扩展为17点、fixture 双鱼眼相机及固定 sigma10px proxy，将投影像素作为观测。新增现有 test_information.py 的精确匹配导数测试；首次实际运行1 failed：代价严格为0，但点坐标的51个导数全部NaN。原因是原未启用Huber线性分支仍对sqrt(0)反传，where的零分支系数不能消除0乘无限导数。

geometry_observation.py 改为直接用平方残差判定二次区间；二次区间内仅给未启用的sqrt分支输入1。前向二次/线性Huber公式不变，不用epsilon制造非零匹配代价，也不把非有限候选改为默认有效。原鱼眼投影、准入掩码、proxy precision和A5/A6决策未改。

实际 test_information.py 与 test_aggregate.py 共11 passed；精确匹配成本和全部导数严格为0，既有投影、缺失、完整候选、FD信息及双阶段接口检查通过。两份修改Python的Ruff通过。没有全套重跑、训练、checkpoint或额外验收文件。该修复是完整端点物理VJP的数值前提，不证明反馈有效或旧FD邻域合同可取消；对应VJP实现与成本检查仍待推进。

## 29. 2026-10-12 完整端点物理代价VJP：方向通过，K32成本未通过

按已有 endpoint_vjp_feasibility_20261012 预登记，在 geometry_information.py 实现 endpoint_cost_vjp。输入严格绑定当前 CandidateStore/原鱼眼Observation、Camera、NoiseModel 和 TangentScale；仅完整52节点，端点与全部外部条件 detach，局部标准化 delta169 求导。右乘 Rodrigues Exp、T/beta 物理尺度与既有定义一致，独立候选的 mean Huber 代价求和后一次 reverse，不除候选数K。输出为代价余切量，不是角速度、协方差或后验；未改变神经网络或原J/Λ接口。

全部候选槽位、ID和已有原因保留。缺失/错误/过期 observation_binding、无当前观测、实际非法R/T/beta、准入点投影无效、非有限代价/梯度均明确 unavailable，梯度NaN；合法精确匹配的梯度0仍available。数值计算行可先筛出，输出还原全部槽位；批解码异常影响原计算槽位，不重试/重抽。局部关闭inference_mode并enable_grad，使外层no_grad/inference_mode及其张量可用；真实SMPL LBS逐行独立是批代价求和成立的前提。端点合法不认证FD邻域合法。

现有 test_information.py 共13 passed in4.02s。新检查独立解析鱼眼Jacobian的 JᵀWr/N、各向异性precision/非均匀T尺度、准入mask、单行与5行方向逐值一致、零梯度、无观测/缺失及过期绑定、负深度/非有限beta/非法旋转、批异常不重试、有限前向而非有限导数。Ruff先后报告测试导入格式、profile多余dict展开，均作等价格式修正；三份改动Python最终通过，未因此重复功能测试或测时。

扩展既有 profile.py --endpoint-feedback-controls。先用真实本地maleSMPLH/MANO及COCO17表面回归器做一个CPU float64端点对照：seed7、各节点Exp(0.03 Gaussian3)、T=[0.04,-0.02,3]米、beta从-0.05到0.05、尺度pose0.1/T[0.2,0.4,0.3]/beta0.08，34准入观测。VJP与同端点中心FD J（step1e-4/chunk16）的 JᵀWr/N 相对误差 **3.9459897e-11**，通过≤1e-4门；pose/T/beta三块均有限且非零。FD339人体状态仅用于正确性，不纳入两种运行模式成本。

方向通过后同次配对成本：RTX5070Ti Laptop、Torch2.11.0+cu128、CUDA float32、CPUthreads1、seed7、K32/8步、warmup2/每模式8次交替次序，同一SO3模型实例/权重/原观测/源/Normalization、每次state=None，真实人体解码+A4。VJP模式只计算旁路方向，没有修正候选或接入控制网；仍保留最终人体解码，以计入未来修正后需要解码的成本。

|同次模式|完整p50 ms|完整p95 ms|峰分配增量 bytes|网络batch/状态|人体batch/状态|reverse|
|---|---:|---:|---:|---|---|---:|
|SO3 F0，无VJP|45.61805|74.93429|30504960|8/256|1/32|0|
|同SO3，加独立端点VJP|81.30155|120.12344|40720384|8/256|2/64|1|

VJP阶段自身p50为34.70410ms；完整p50比 **1.7822233**，超过先定1.25门，停止全K32晚反馈晋升。所有记录32候选数值合法，VJP均available，最终vertices有限，实际计数通过；失败原因数组完整保存于 flow_next_endpoint_vjp_cost_v1.json。该负结果属于计算可行性，不能解释为反馈准确度差；也不能因方向误差小就忽略成本。8次小样本p95与未锁负载只支持本次停止判断，不认证生产延迟或真实人体精度。

测前GPU16%/1921MiB/21.33W。工具等待被用户新消息中断时进程已完成并保存结果，现场检查无相关后台进程，直接读取JSON，没有重复启动/重跑以选较好耗时。用户转向方案解释与架构图，当前停止新增实验，保留独立算子供研究；未开启第二轮修订、训练、checkpoint或默认主线晋升。数据门仍缺，长期效能目标未完成。

## 30. 2026-10-12 长期不足改进D01：移除SO3不可达分配，活跃计算不变

用户要求持续分析现模型不足并逐项改进。建立当前聊天continuous goal，12项按源码确认/实测失败/设计假设/资源限制分级台账见MODEL_ARCHITECTURE第16节，原held-out门、数据暂无、训练关闭、两轮机制限制和VJP成本停止决定继承。没有用台账替代改进实施，也不将暂无RGB、共享beta或局部attention自动判为效果缺陷。

在现有长期协议deficiency_improvement_20261012::first_step先登记D01：SO3所有Λ输入已拒绝，其15104个info_pose/position/shape权重及fixed probes仅常驻、无活跃依赖。竞争解释分别为可无损删除、隐藏依赖和初始化RNG假差异。so3_flow构造新增显式retain_inactive_information默认False，沿原super/pose头构造顺序后删除不可达层、将probes置None；True保留原分配。profile共享初始化仅复制ambient/SO3实际共同keys，避免把已删层视作新模型参数。原JointFlow/G0/G1与信息条件回归保留。

|分配模式|52节点参数|22节点参数|Lambda|活跃计算|
|---|---:|---:|---|---|
|默认精简SO3|695952|692112|仍显式拒绝|与同权重旧模式逐值相同|
|显式原分配SO3|711056|707216|仍显式拒绝|原行为保留|
|原ambient JointFlow|711443|707603|支持现有接口|未改变|

减少15104个参数，约原52节点分配的2.1242%；float32权重张量减少60416bytes。fixed probes为13M元素，默认原float64下52/22为17576/8216bytes，转换dtype后另算；非持久buffers不在state_dict里。这里只证明常驻/序列化权重简化，删除前仍构造原层，未声称更低初始化峰值、运行延迟、显存allocator峰值、人体精度或新颖性；实际有效容量不变。

先运行新对照测试两拓扑，2failed/17deselected是新增legacy构造参数尚未实现的TypeError，不是原模型精度失败。实现后只运行test_joint.py/test_regression_information.py：25passed in7.55s；三份改动Python Ruff通过。新22/52逐值对照检查同seed活跃参数、所有state_dict差异恰为三组信息权重、buffer大小；K32/8在观测→全缺失→恢复三帧的raw/R/T/beta/numerical/actions/IDs/seeds/reasons/网络计数与完整serialized state均一致；两目标loss及每个活跃参数梯度逐值相同，旧不可达权重gradNone；Lambda仍拒绝。严格加载原模式随机state_dict到新模式明确unexpected keys，未隐式吞旧层；这是内存随机模块的键合同检查，没有读取checkpoint、训练或optimizer运行。

实际验证命令：`.\.venv-cuda\Scripts\python.exe -B -m pytest realtime_app/learning_flow_mesh_next/tests/test_joint.py realtime_app/learning_flow_mesh_next/tests/test_regression_information.py -q --tb=short`；`uv tool run --offline ruff check realtime_app/learning_flow_mesh_next/so3_flow.py realtime_app/learning_flow_mesh_next/profile.py realtime_app/learning_flow_mesh_next/tests/test_joint.py`。

此前架构图与SO3成本v1/v2/VJPv1保留原711056容量并在架构文档标为历史分配；未因删参数重跑测时或覆盖结果。D01完成，下一步优先判别D02观测压缩或D08缺失时间处理是否存在新的具体问题，先排除已修复身份碰撞和“context不同就是精度改善”的错误推理。其余设计/效能不足仍未完成，长期目标保持active。

## 31. 2026-10-12 D08局部时间特征修复：先编码incoming age/seen，再保留原状态重置

D01独立完成后继续检查StateEncoder。发现原输入流程先age=where(valid,0,age)、seen=seen|valid，然后构造15维local特征，且只池化tokens[valid]。因此进入GRU的每个准入槽，其local age永远0、seen永远True；mean(age)和seen_fraction的global time分支仅看到重置后的统计。这是两个槽位特征失效的源码问题，不等于证明GRU无法隐式保留任何历史或真实误差已恶化。

先登记长期协议D08_first_check。typed人工TemporalState对照固定identity/rig/frame/clock与incoming hidden，当前所有点准入；只改变一个槽的incoming age或prior_seen，post-reset age/seen/metadata一样。CPUfloat64 seeds7/17/27，两类共6组旧context最大差严格0。第7项真实顺序0秒观测→0.2秒全缺失→0.9秒恢复，原第一次local seen为True，不提供首次出现信息。新检查首次7failed/7deselected，保持原失败说明，不将这些人工状态pair当真实轨迹/精度样本。

修复state_encoder.py仅保存incoming measurement_age/prior_seen用于原特征12/13，输出状态和global timing使用原post-observation age/seen。特征维度15、StateEncoder137216参数及各模型容量不变，原槽位身份绑定、准入池化、GRU更新函数和全缺失hidden冻结不变。首次观测输入age0/prior_seenFalse；缺失后恢复的gap=累计物理dt，previous seenTrue。未知槽gap是序列累计时间，prior_seenFalse明确它不是已知的上次检测时间，不伪造可靠观测历史。

修复后一次test_identity_binding.py/test_joint.py为27passed/6failed in8.22s：6项测试仍错误要求完整返回state（含hidden）相同，而新时间信息本应影响GRU hidden。明确纠正测试为post-reset age/seen/身份/时钟相同、incoming状态不被修改，hidden可随新特征改变；相应协议说明同步。这是测试约束错误，不是再次修模型或调阈值。只重跑这6项：6passed/8deselected in1.52s；其余27通过保持有效，没有重复功能检查。两份改动Python Ruff通过。33相关项分次覆盖包括原关节/视图身份、未来隔离、serialized carry、无观测不发布、两模型CFM/回归与D01计算等价。

实际验证命令：首次红检查为`.\.venv-cuda\Scripts\python.exe -B -m pytest realtime_app/learning_flow_mesh_next/tests/test_identity_binding.py -q --tb=short -k 'freshness or elapsed_age'`；修复后两文件使用相同pytest入口/`-q --tb=short`；测试约束更正后仅该测试文件`-k freshness`。Ruff检查state_encoder.py/test_identity_binding.py，测试调整后只追加该测试文件lint；没有重跑27项或原成本。

结论：local编码现在可以接收再次观测前的槽位时间间隔/首次出现标志。六组context区分超过1e-8是随机网络的输入路径证据，不是缺失人体准确度改善；更广的运动/采样节奏效能尚未执行。既有模型checkpoint/旧hidden不在新语义下混用，原历史成本与图保持历史记录；无训练、权重加载、人体解码或额外测时。D08这一具体局部缺陷完成，长期台账其他假设和独立有效性继续未完成，下一步转D02的信息压缩必要性判别。

## 32. 2026-10-12 D02维数与局部信息判别：不支持仅凭池化扩容

开始核对实际D01/D08提交、当前encoder和已有身份碰撞修复，未重复旧检查。聚焦读取两原始论文：[Deep Sets](https://arxiv.org/pdf/1703.06114)第2.2/3.1节（Theorem2可数域、连续域限定）与[Wagstaff2019](https://proceedings.mlr.press/v97/wagstaff19a/wagstaff19a.pdf)第2/4/4.1节（连续scalar-set sum-decomposition维数条件、不是每个任务都需要通用表达）。NeurIPS一个Reviewed.pdf地址读取失败后用原arXiv正文，无新PDF副本。没有读取/运行作者网络或训练实验；来源范围补入既有LITERATURE_ANALYSIS。

不能从34槽×15特征大于128就推出当前必然丢失关键关系：固定mask/rig/geometry关闭/初始时间状态时，像素68+score34只有102连续维；全特征中身份/掩码是离散项，射线与真实像素/标定也不总是自由独立。这个计数仅针对本次固定输入族，不覆盖所有ray/history/rig条件；论文通用sum表示定理也不直接认证本模型的固定小MLP/GRU。按已有长期协议D02_dimension_check先登记三竞争解释、控制与停止条件，再扩展现有audit_conditioning.py --encoder-dimension，不改生成器/encoder。

数学控制：用已知34槽身份把每槽[u/1000,v/1000,score+1]路由至独占3维，准入集合取mean，共102维。score+1严格正，把真实score0和缺失块0区分；非零第三坐标恢复mask，计数N后乘N恢复准入值。full34、sparse5、zero-score34、empty0四例均恢复mask和值（≤1e-12），empty保持无观测；无身份3维mean在0/16槽交换下相同，而固定身份路由可区分。该控制是限定输入族的表示存在性反例，不是实际网络、最强人体对照或已实现新的encoder。

实际当前StateEncoder检查：CPUfloat64、threads1、seeds7/17/27、137216参数；LEFT像素u=120+19j/v=100+11(j mod7)，RIGHT相对[-8,+6]像素；score分别0.4+0.02j/0.75−0.015j。全部准入、geometry=False、state=None，不宣称点来自真实人体或已同步双目。计算context128对[xy/1000的68维,score34]的Jacobian128×102。全部奇异值保留，rank阈值1e-8/1e-10/1e-12/1e-14均102；固定seed19单位方向与central FD step1e-4对照。

|seed|四档局部秩|smin/smax|FD方向相对误差|
|---|---|---:|---:|
|7|均102|6.5779121e-7|1.6207567e-10|
|17|均102|6.4009348e-7|1.4936306e-10|
|27|均102|6.2981912e-7|1.6491970e-10|

数值检查通过，结果flow_next_encoder_dimension_v1.json保存每个输入/控制/谱；actual Jac有限，方向误差≤1e-4。小奇异值比揭示该初始化/单位范围内的敏感性不均，并不证明训练后的观测丢失、实际误差或float32失败。局部满列秩不等于全局单射，不排除另一输入域、缺失组合、ray/history或真实数据上的不足。没有扩大参数、token或attention，未执行人体解码/J339/网络K32成本，也未添加镜像测试/验收框架。

实际命令：`.\.venv-cuda\Scripts\python.exe -B -m realtime_app.learning_flow_mesh_next.audit_conditioning --encoder-dimension --output research_records/research/G20261007_flow_mesh_baseline_v1/flow_next_encoder_dimension_v1.json`；改动文件Ruff通过。保存JSON独占创建；stdout截断后直接读保存JSON汇总，没有重复运行。决定保留当前mean/GRU，停止只凭维数/均值池化就增加cross-attention/RGB/宽度的提案；D02的真实信息充分性留待独立训练/对照。下一项D10检查真实训练准备接口与资源合同，训练和神经checkpoint继续关闭，长期目标未完成。

## 33. 2026-10-13 D10.1归一化统计固定与来源/dtype往返修复

上轮已按长期协议登记D10_normalization_check，未提交的五份Python修改本轮现场确认、继续收尾；没有把中断视为失败重跑41项。原Normalization.fit在声明train/fitted_reference的人工梯度输入上mean/scale均requires_grad=True，mean_grad_fn=MeanBackward1；to_dict仅mean/nodes/scale/source四字段。另一个float64人工值回读为float32，mean最大绝对差8.2978027e-8。两次最小现场输出为失败证据，未新建JSON/报告，人工声明仅测试合同而非真实参考数据。

保留原拟合mean/std及1e-4 floor，结果detach；新增split/source_kind和float32/float64 dtype保存与回读。fixture()明确engineering_fixture/analytic_fixture；旧四字段从字串或旧数值均不升级，仍legacy_unverified、原float32 loader默认。require_training_statistics仅认可显式train+fitted_reference/independent_reference元数据，其他来源明确拒绝。validate拒绝mean/scale梯度、混合/未知dtype、validation/test统计和不一致声明；它不判断声明是真实还是伪造，独立源/组划分证据仍缺。

相关传播：regression_loss与candidate_store.infer明确验证统计；SO3 output codec同时绑定来源，避免同数值但换split静默调用；候选来源记录normalization_split/source_kind。不改状态维度、源、损失或网络容量。旧工程人工构造未填写新增字段的Normalizer仍可用作legacy工程记录，不能通过新训练统计gate；没有加载neural checkpoint。

上轮三现有测试文件test_firewall.py/test_joint.py/test_regression_information.py实际41passed in7.69s、五文件Ruff通过；覆盖22/52、float32/64值/来源往返、detach、fixture/legacy拒绝、不一致split、dtype、codec和现有生成/回归/因果接口。新增infer.validate后只运行test_firewall -k inference_rejects：2passed/16deselected in3.34s，两文件Ruff通过。41项未重复；合计43相关检查分次覆盖，新增问题对应新增检查，不将其当独立真实验证。

命令均使用`.\.venv-cuda\Scripts\python.exe -B -m pytest`，三文件`-q --tb=short`；新增入口检查`-k inference_rejects`。Ruff使用既有uv offline入口。没有训练、optimizer、人体解码、成本重跑、数据下载或新验收材料。D10.1完成；下一阶段在train.py把逐帧观测历史、显式标签帧绑定和缺失行保留接到现有目标计算，实际数据loader/训练循环/效能仍未完成。

## 34. 2026-10-13 D10.2因果序列目标：内存路径完成，真实训练runner仍缺

先在已有长期协议登记D10_causal_objective，随后扩展train.py三个数据类和sequence_objective，不建立另一个训练/审计框架。问题为原objective只有context batch，无法从该接口核对目标行与物理原帧的绑定及因果历史。新SequenceChunk每次最多32个原帧，LabelFrame显式行绑定；各帧typed观测/标签/统计、frame/time严格递增、身份/序列/rig及目标时钟对齐在编码前检查，坏最后一行不能让前面先forward后才报错。

按原顺序编码全部Observation，labels不进入encoder；仅当前有观测+完整姿态/T/beta标签的行调用原单行CFM/回归。目标tensor detach，incoming及returned TemporalState detach；块内历史观测梯度保留。无观测保留时钟/age/hidden冻结；有观测无完整目标仍更新hidden。返回全部F个原槽、context、loss/NaN、supervised、逐帧原因和原观测rejection_reasons。均值以实际监督帧计数，跨chunk均值按supervised_count加权；如果任何已计算loss非有限，则该chunk aggregate=None，不能静默挑其他有限行。没有optimizer、checkpoint或真实loader；当前F0接口不能冒称已完成G1当前信息路径。

人工5帧100–104、物理时刻[0,.17,.52,.73,1.1]，1号全缺观测、3号beta maskFalse；监督行0/2/4，其余两行仍保留。fixture声明analytic/engineering；另用明确标注synthetic_contract_not_real_data的train来源人工tensor仅测试统计gate，不作为实际训练数据或held-out效果。SO3/ambient、22/52、CFM/回归八对照，随机seed7、源generator17，原fixture Normalization；不解码人体或采集/检测。

首次仅新sequence相关检查21passed/19deselected in4.41s，train.py/test_joint.py Ruff通过；随后补上与原encoder+逐行objective的独立连接对照、部分pose/T及train来源gate/typed chunk检查，相关八项扩充+一项新增9passed/32deselected in2.31s；其余13项未重复。22个新检查分次覆盖全部风险，没有失败或靠调整数值阈值过门。旧19项不因无关新接口默认重跑。

|明确问题|实际证据|边界|
|---|---|---|
|目标是否正确接到原逐帧API|八组合手动encoder/原objective的context/单帧loss逐值相同；无参数更新|只说明连接，不说明真实学习|
|分块是否改变推理值/随机抽样|3+2分块使用同generator/carry，context及loss一致；监督计数2/1加权得到原3帧均值|TBPTT梯度跨块明确截断，不声称全长梯度相同|
|未来信息/标签是否影响过去|变最后两帧像素与目标，前三帧context/loss不变|仅这个声明的因果路径，不认证未来数据loader|
|块内梯度与目标/carry屏障|最后回归loss反传到第0帧及无标签第3帧观测，GRU梯度非零；target/carry无grad，缺观测无输入grad；returned hidden脱图|无optimizer步骤，不称已训练|
|缺失/失败是否隐藏|all-missing/all-partial保留5槽NaN、lossNone、零velocity；第一监督loss inf控制令整个aggregateNone，剩余有限行仍记录|前向接口失败语义，不是全系统失败率|
|对齐能否在forward前拒绝|frame/clock/identity/rig/order/stream/split/statistics/行数/33帧上限十项坏输入均在encoder前拒绝|显式绑定声明不认证实际数据身份或标签真实来源|

命令：`.\.venv-cuda\Scripts\python.exe -B -m pytest realtime_app/learning_flow_mesh_next/tests/test_joint.py -q --tb=short -k sequence`；追加检查只`-k 'causal_sequence_objective or partial_targets'`。Ruff既有offline入口检查train.py与test_joint.py。无需新JSON/快照/验收文档，记录实际条件/失败状态/原接口对应即可。D10.2完成，下一阶段建立实际数据文件到这个接口的读取/标签与来源绑定，并保持无数据时不执行；真实data loader、划分资产真实性、optimizer/checkpoint生命周期、训练授权及独立有效性仍未满足，长期目标未完成。

## 35. 2026-10-13 D10.3注册文件读取：连接完成，数据可用性仍为否

问题：D10.2只有内存观测/目标连接，没有从已注册文件恢复原行与显式标签绑定的路径。先在现有长期协议登记D10_registered_file_reader，再扩展train.py的load_registered_chunks和现有test_firewall.py，不新增训练或审计框架。唯一变量为文件→原Observation/Labels/LabelFrame/SequenceChunk的读取；encoder、目标公式、容量、缺失/失败语义不改。

注册入口先验证整份manifest的原subject/action/sequence跨split隔离和本地许可声明，再只选择train记录；选中record声明identity/PnP/source_kind/nodes。文件header与record、manifest及调用方asset_contract的family/basis/hand_mean等声明必须一致；raw-fisheye、左右身份顺序、cam0/metres、surface operator、时钟类型显式。numeric NPZ使用allow_pickle=False，Boolean masks不作整型转换；全部原frame/clock/标签frame/clock、观测/目标行数、有限可信标签及原拒绝原因检查后，按32个连续原行分块，不按可监督行筛选。

检查输入为pytest临时目录中的明确synthetic_contract_fixture，不作为训练数据登记：22与52节点各35原帧501–535，时钟1000+i×0.07秒；index=2的原行缺观测并保留NaN/原拒绝原因，index=8的原行beta maskFalse且值NaN。人工subject/action/rig、声明参考标签和train统计仅测试接口合同，beta basis明确synthetic_test_basis_not_real_asset。两种节点均得到32+3原行chunk，全部帧/时间/张量/mask/NaN/原因及optional view时钟保持。没有解码真实人体或采集新数据。

15种坏文件分别覆盖subject、beta basis、camera IDs/roles、coordinate frame、format、label顺序/时钟、原clock顺序、无符号frame倒序、整型mask、缺array、错shape、pickle object及可信NaN；均明确拒绝。empty/license/test record/跨split泄漏/source_kind共5种登记失败用np.load spy检查在文件打开前拒绝。5帧文件→现有JointFlow回归因果目标与独立构造内存输入的contexts、state、aggregate逐值相同，逐帧loss按torch.allclose含NaN一致，监督行[True,True,False,True,False]；该检查只认证连接，不认证真实可学习性。

首次22项因train.py新append少一个闭合括号而在import时报SyntaxError，未进入文件读取，不能称为数据条件失败。修复括号与import排序后22passed/18deselected in2.23s；Ruff随后发现nodes类型异常应为TypeError，修正后仅新增“Boolean nodes在开文件前TypeError拒绝”1passed/40deselected in2.18s，两改动文件Ruff通过。提交前发现np.diff对uint64倒序帧回绕：最小[502,501]控制旧递增判断True、直接相邻比较False。frame/time改用直接比较后，只新增无符号倒序拒绝1passed/41deselected in2.52s，Ruff通过。共24个新检查分次覆盖；已通过22项未为保险重跑。命令：`.\.venv-cuda\Scripts\python.exe -B -m pytest realtime_app/learning_flow_mesh_next/tests/test_firewall.py -q --tb=short -k registered`；追加项仅`-k 'registry_failure and nodes'`及`-k unsigned_frame_order`。静态检查使用既有uv tool run --offline ruff check。

现场读真实flow_next_data_manifest_v1.json：records0、training_data_ready=False、truth_status=no_independent_3d_or_scan；实际调用新读取器返回registered_training_data_unavailable。flow_next_noise_data_manifest_v1.json的独立参考records同为0，360帧现有观测不升级为训练/验证真值。真实清单未改，临时文件不成为永久假数据。

D10.3解决文件连接与声明对齐，不能认证标签/资产基/人物/物理时钟真实性或统计实际只来自train；caller asset_contract也只是当前显式声明，未做真实资产认证。optimizer/checkpoint生命周期、G1冻结信息路径、真实资源、训练授权及独立效能均未完成，长期目标未完成。下一阶段D03先分析逐帧beta与序列共享shape的合同/机制：估计随新证据变化不等于真实体型变化；不自动冻结、EMA或把旧beta拼到新pose，也不以平滑作收益证据。

## 36. 2026-10-13 D03体型合同判别：保留逐帧模型，停止直接嫁接冻结

上轮完成注册文件读取，本轮先核对实际candidate_store.py、schema.TemporalState、ambient/SO3 solver、旧shared_shape.py/shape_features.py/beta_conditioning.py和H8协议。当前种子依frame变化，候选beta不进入下一TemporalState；采样是当前帧完整R/T/beta，历史只由观测编码。这确认缺“共享beta整段轨迹的生成合同”，不能从beta随帧变化进一步推断真实精度变差。H8则用前8个认可帧local特征均值经旧head，ready后不再消费shape特征；它与next不是同一个模型。

先登记D03_sequence_shape_contract，再运行一次临时解析控制，未建立新audit JSON/脚本或永久数据。正文聚焦[HuManiFlow](https://arxiv.org/abs/2305.06968)现有sources/P05.txt 3.4、3.5式8–10、3.6式12：Gaussian shape与shape-conditioned ancestor pose因子化，先shape后pose抽样；论文点估计也不一般等于联合mode。在线PDF403、arXiv HTML v2不存在/ar5iv不可访问；已读现有正文而非声称网络获取成功。SMPL论文39.8MB超工具内容上限，改读[作者主页](https://smpl.is.tue.mpg.de/)对identity-dependent和pose-dependent形变的区分，不称SMPL全文复读。没有运行作者网络、下载模型/训练标签或做全局新颖性认证。

输入/控制1：已知维数1的静态隐变量beta~N(0,1)，y_i=beta+epsilon_i，独立epsilon_i~N(0,1)。精确Fraction顺序Bayes更新与闭式mu_t=sum(y)/(t+1)、v_t=1/(t+1)在全部16时刻相同。人为y前8次2、后8次0时，t8均值16/9、方差1/9；t16均值16/17、方差1/17。复用原PrefixSharedShape，人工1→10无偏置head只第一维固定8/9、其他0，16行float64 feature、admitted全True、frame0–15、analytic_scalar_not_human序列；实际前7行beta NaN/readyFalse，count停8，第7–15行第一维维持16/9，误差<1e-12。此仅是原前缀组件控制，不是随机H8或next精度测试。该已知模型下对全部数据积分的posterior-mean平方误差为冻结1/9、更新1/17；它否定“物理beta静态所以冻结估计必然更好”，不把这个风险数值套到COCO/真实人体。

控制2：beta每时刻±1各概率1/2。共享coupling表[[1/2,0],[0,1/2]]和独立表[[1/4,1/4],[1/4,1/4]]，行列边缘完全相同，时间间beta不同的概率分别0与1/2。这证明逐帧边缘/逐帧coverage或beta散度不能单独认证跨帧世界状态耦合。当前frame-specific随机数是可复核采样政策，非生理体型变化证据；复用同seed的context-dependent映射也未保证beta恒定。若前后shape边缘不同，等权固定slot的beta逐一恒定与同时匹配两边缘相矛盾；静态参数过滤需明确权重/重采样/谱系语义。

控制3：抽象联合支持只有(beta,T)=(-1,+1),(+1,-1)，两完整模式均满足beta+T=0；旧beta=-1与新模式T=-1拼接得到绝对残差2，移到原联合支持以外。没有真实相机/米/mm或SMPL解码，不称重投影失败率；它说明正确的条件生成不同于输出拼接，也不保证仅将beta速度清零后原joint flow仍采样目标条件分布。

唯一代码改动是infer缓存temporal_sampling声明当前帧完整候选、cross-frame law未定义、没有显式beta carry、slot不是轨迹。所有参数、随机种子、encoder/solver/损失、R/T/beta tensor和原不可用/发布路径未改。仅扩展原test_nested_independent_frame_noise_and_cache：K8/K32嵌套及跨帧noise不同、raw/IDs缓存一致、声明保存、无声明旧临时缓存不自动升级。实际1passed/40deselected in2.25s，两文件Ruff通过；无失败和全套重跑。命令`.\.venv-cuda\Scripts\python.exe -B -m pytest realtime_app/learning_flow_mesh_next/tests/test_joint.py -q --tb=short -k nested_independent_frame_noise_and_cache`；三个解析控制在一次PowerShell here-string送同Python进程执行，结果stdout，公式/输入/head/代码路径在本节可追溯，不另建流程脚本。

D03合同阶段决定：保留现有逐帧联合模型，停止默认嫁接H8冻结、EMA、跨模式旧beta拼接或固定seed当共享轨迹的提案；没有独立证据时不添加shape容量。后续若提出共享序列模型，必须定义一个beta与全历史R/T的联合目标、因果在线发布政策、配套条件采样训练/原标签来源；对照为同输入/预算逐帧联合及真实实施的因果条件shape政策，完整计算额外采样/初始化/重建成本，不以平滑挑赢家。H8固定8认可帧仅是工程阈值，shape可辨识、真实优劣、序列训练与独立held-out仍缺，当前真实records0/训练关闭。下一项D05审查噪声信息的共同可用性与F0/G1输入公平合同，沿用既有noise检查，不默认接新precision模块；长期目标未完成。

## 37. 2026-10-13 D05同信息回归：完整G0/J推理入口补齐

核对geometry_observation.NoiseModel、StateEncoder、旧conditioning audit第16节、已有regression信息原语、TwoStageInference与四法预注册。外部precision的旧sigma1/2控制已确认，不重复；本轮新发现是pipeline.__call__没有regression选择，R_info只能由调用方手工提供Λ，不能用同一入口保持G0/J获取路径和计数。按长期协议D05_information_regression_pipeline先登记，只改第二阶段选择，不给F0增加噪声条件或改precision来源。

先在test_regression_information.py加2个topology完整路径控制与1个k限制控制，原代码3failed/6deselected in3.61s，全部TypeError unexpected keyword regression，未进入新回归路径。随后pipeline.__call__新增regression=False，True明确要求k1且先拒绝无效预算；执行原G0/local_information，再透传原infer的信息回归。report区分1generator+1regressor或原2generators，记second_stage_objective；R_info的sample-budget匹配字段None，不将确定性点误记为额外抽样预算。默认CFM原采样和所有参数/门未改。

实际输入：CPU float64/threads1/seed7，ambient22/52各两份未训练JointFlow；工程Normalizer mean为节点identity6D/T=[0,0,3]/beta0，scale全0.03，明确engineering_fixture/analytic_fixture；原人工raw-fisheye PnP rig、原observations和proxy sigma10px，非真实标定/标签。surface=原rigid_surface仅复制T到17点，pose/beta不影响这个几何控制；不作为真实SMPL-H或shape可辨识验证。G0 anchorK2/steps2，R_info finalK1/1次；同输入F_info finalK3/steps2。第一帧0/time0，恢复frame1/time0.1，缺失frame2/time0.2。

通过控制：两topology G0 raw/IDs与anchor_id、非零Λ、全部观测state逐值一致；R_info raw等于独立手动g1.encoder→regression→denormalize；carry恢复后下一帧raw逐值相同。R_info网络调用3、候选状态求值5，原F_info调用4/求值10来自原预算；J报告调用/状态逐项等于独立surface收集的实际batch数量/行数，与F_info的J工作一致。没有测时或声称这两个不同候选数的wall-time相等。缺失时保留全部原reason/槽、不可发布，ΛNone、J states0、g1 hidden冻结；k2回归在g0 encoder前拒绝。G0所有参数保持逐值相同且requires_gradFalse，calibratedFalse保存。原CFM双阶段/state检查也通过。

改动后仅相关3新项加原1项：4passed/18deselected in2.09s，pipeline.py/test_regression_information.py Ruff通过。命令：`.\.venv-cuda\Scripts\python.exe -X utf8 -B -m pytest realtime_app/learning_flow_mesh_next/tests/test_regression_information.py realtime_app/learning_flow_mesh_next/tests/test_information.py -q --tb=short -k 'frozen_information_regression_path or two_stage_frozen_g0_and_state_chunk_roundtrip'`。初次仅新文件-k frozen_information_regression_path。未重跑旧noise/J审计、全套测试或K32耗时，也未创建额外JSON/验收报告/脚本。

四法计划R_info部署接口同步为完整TwoStageInference回归入口；状态仍preregistered_not_executed_training_closed。推理接口不是已完成四法训练评估，R_info未来需独立回归训练、使用同一个已冻结G0/源/统计/rig/noise；不能直接拿CFM训练weights调用regression作为强基线。F0/R0与F_info/R_info在变化noise时输入不匹配问题继续存在，比较只能按预注册的输入因素解释；G1接Λ不认证covariance或后验。当前真实training/noise-reference records仍0，无optimizer/checkpoint/人体解码或正式主线改动。D05完整推理准备路径完成、校准/共同外部noise条件及效能未完成；下一项D10.4接G1因果序列目标与逐帧冻结信息来源，默认训练继续关闭，长期目标未完成。

## 38. 2026-10-13 D10.4原帧冻结G0/J到G1序列目标连接

问题与预注册：D10.2序列目标只有F0条件，D05推理已有完整R_info，但还不能把当前逐帧Λ/来源与获取成本连到CFM/回归序列目标。先登记D10_G1_sequence_information，再扩展现有train.py的SequenceInformation、observation-only acquire_sequence_information和可选sequence_objective information。重用infer/local_information/require_information_binding；不另建数据或审计框架，不改core参数/CFM/solver/选候选。第一次4个新22/52×CFM/回归检查ImportError缺acquire_sequence_information，4failed/9deselected in3.65s，尚未执行任何获取；原失败保留。实现后首4passed/9deselected in2.84s，Ruff通过。

实现范围：冻结eval G0只接原typed观测，先验证整块身份/物理时钟/rig/noise/拓扑，不接labels；no_grad逐行获取anchor/J，返回全行Information、原获取资源及脱图carry。信息包绑定实数观测/mask/ray/clock、实际K/D/R/T/PnP和noise/scale，目标编码前检查整块；它只认证声明/当前数值一致，不能认证G0真实训练来源、过去history/decoder资产或数据真实性。当前块内固定rig/noise，变噪声跨块需显式提供新来源；不因此解决F0外部precision输入不足。信息matrix/J/scale/anchor要求冻结，SO3F0信息合同保持拒绝。

实际输入：CPU float64、threads1、seed7，ambient22/52各独立随机g0/g1，g0在获取前eval+requires_gradFalse；Normalizer工程source=synthetic_G1_sequence_not_training_data，identityR/Tz3/beta0 mean、scale全0.03、engineering_fixture/analytic_fixture。复用原5帧100–104、物理时间[0,.17,.52,.73,1.1]，1缺观测、3缺beta；标签为既有人工R/T/beta，非真实参考。原artificial raw-fisheye rig、proxy sigma10px/calibratedFalse；surface=rigid_surface只复制T为17点，不含真实pose/shape依赖或SMPL-H解码。G0 anchorK2/steps2，CFM generator17；目标仅0/2/4三行，但原5行均推进观测与获取状态。

实际连接：CFM/回归×22/52的G1 context/逐帧目标与独立调用原encoder和原objective含本帧Λ逐值相同，aggregate为原3行均值。目标3+2分块同generator/carry得到同contexts/state、loss含NaN allclose；G0获取也按3+2传独立G0 carry后逐帧矩阵/快照和末state相同。扰动最后两帧观测及标签并重新因果获取信息，前3帧contexts/loss保持；整块的前置metadata校验仍会拒绝坏后行，这不等于在线输出在未来任意坏文件下保证可得。

每块实际获取5原帧，G0 10网络调用/20候选状态求值；有观测4帧产生J，独立surface记录batch数/行数与返回计数一致，J states为22节点4×159=636、52节点4×339=1356。目标仅3 velocity calls/3 target states，与G0/J成本分开。部分标签第3行仍有信息与获取成本，不为少监督省略。未运行K32/8成本或测时，不用小预算估算部署速度。信息包重用时computed_in_this_objective=False记录预计算，完整runner须按实际复用/部署边界聚合成本。

失败与反传：14个G1坏输入分别frame/pixels/clock/rig/noise/scale/matrix shape/nonfinite/rows/gradient/counts/无None原因/legacy anchor/SO3，均用encoder spy证明全部原信息在首次G1编码前拒绝。4个获取坏源为未冻结G0、倒序后行、noise rig不符、非typed carry，均在首次G0 encoder前拒绝。原合法缺观测matrixNone保留no_current_observation；另用controlled_surface_unavailable产生所有5行有原原因的matrixNone，目标保留原观测/标签失败和独立信息原因，无Λ回退contexts/loss等于原F0，监督0/2/4不变。回归末行反传到第0/无标签第3观测和G1 info_position/GRU，缺观测无输入梯度；target/G0/J/incomingcarry无grad，returned G0/G1 carry脱图。只验证依赖关系，无optimizer步。

完善typed noise/scale及所有anchor绑定后只覆盖新路径25passed/9deselected in8.86s（含此前4项因新增约束重新覆盖）；追加非有限matrix一项1passed/34deselected in1.85s，其余新25项未重跑。默认F0原值/梯度/全缺失/非有限目标相关11passed/30deselected in2.06s，确保这次连接没有改变default。补获取前后G0权重逐值断言后仅两个22节点CFM/回归连接检查2passed/33deselected in1.97s，Ruff通过。26个新检查与11个既有F0相关检查分次覆盖，记录具体目的，未全套重跑。

命令：`.\.venv-cuda\Scripts\python.exe -X utf8 -B -m pytest realtime_app/learning_flow_mesh_next/tests/test_regression_information.py -q --tb=short -k 'G1_sequence or G1_acquisition'`；追加仅`-k 'G1_sequence_rejects and nonfinite'`，权重断言仅`-k 'G1_sequence_matches and regression and 22'`。原F0文件test_joint.py仅`-k 'causal_sequence_objective or sequence_gradient or sequence_all_missing'`；改动两文件用既有uv offline Ruff。所有人工数组临时内存、无永久数据/新JSON/验收文档。

D10.4当前G0→原帧信息→G1序列CFM/回归准备路径完成。真实训练与独立噪声reference records现场均0，normalization/资产/数据/完整历史真实性、独立标签、训练授权、optimizer/checkpoint与四法实际训练评价仍缺；矩阵可用或依赖检查不证明真实手/shape/后验/精度。下一D10.5推进同组多序列训练准备与执行生命周期的剩余连接，每次只补一个实际接口缺口，训练/神经checkpoint执行仍关闭，长期目标未完成。
