"""Read-only two-capture arm diagnostics using existing observations and fits.

Pixel label disagreement and fit-observation residuals are different quantities.
No detector inference, label correction, point fusion or fitting is performed.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pose_app.body_observations import raw_side, raw_triangulate
from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "research_records/engineering_validation"
JOINTS = {"left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def signed_components(prediction, manual, elbow):
    axis = np.asarray(manual, dtype=float) - np.asarray(elbow, dtype=float)
    length = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or length < 1e-8:
        return None
    axis /= length
    delta = np.asarray(prediction) - np.asarray(manual)
    return {"along_px": float(delta @ axis),
            "absolute_across_px": float(abs(delta @ [-axis[1], axis[0]])),
            "forearm_px": length}


def project(points_m, calibration, camera):
    points = np.asarray(points_m) * 1000
    if camera == "right":
        points = points @ calibration.R_cam0_to_cam1.T + calibration.T_cam0_to_cam1_mm
    K, D = ((calibration.K0, calibration.D0) if camera == "left"
            else (calibration.K1, calibration.D1))
    return fisheye_project_numpy(points.reshape(-1, 3), K, D).reshape(*points.shape[:-1], 2)


def grouped_manual(rows):
    groups = {"all": rows}
    groups.update({c + "_" + h: [r for r in rows if r["camera"] == c and r["hand"] == h]
                   for c in ("left", "right") for h in ("left", "right")})
    return {key: {"error_px": stats([r["error_px"] for r in group]),
                  "along_common_axis_px": stats([r["common_axis"]["along_px"] for r in group
                                                 if r["common_axis"] is not None]),
                  "toward_elbow": sum(r["common_axis"] is not None
                                      and r["common_axis"]["along_px"] < 0 for r in group),
                  "own_axis_toward_elbow": sum(r["own_axis"] is not None
                                              and r["own_axis"]["along_px"] < 0 for r in group)}
            for key, group in groups.items()}


def render_review(report, output):
    def number(v):
        return "—" if v is None else f"{v:.2f}"

    rows = []
    for capture, d in report["datasets"].items():
        for method, groups in d["manual_pixel_disagreement"].items():
            for group, v in groups.items():
                rows.append(f"<tr><td>{capture}</td><td>{method}</td><td>{group}</td>"
                            f"<td>{v['error_px']['n']}</td><td>{number(v['error_px']['mean'])}</td>"
                            f"<td>{number(v['error_px']['median'])}</td>"
                            f"<td>{number(v['along_common_axis_px']['mean'])}</td></tr>")
    body = []
    for capture, d in report["datasets"].items():
        for method, v in d["body_fit_consistency"].items():
            if v["state"] == "unavailable":
                continue
            body.append(f"<tr><td>{capture}</td><td>{method}</td>" + "".join(
                f"<td>{number(v['joint_residuals'][j]['fit_minus_observation_mm']['median'])}</td>"
                for j in JOINTS) + "</tr>")
    text = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>腕肘对照</title>
<style>body{font:16px/1.6 sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;background:#fff;color:#000}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #000;padding:6px;text-align:left;background:#fff}a{color:#000}img{max-width:420px}h1{font-size:26px}</style>
<h1>腕肘对照</h1><p><a href="body.html">最新主线：360帧身体</a> · <a href="arm_comparison.json">完整指标</a> · <a href="arm_records.json">逐点记录</a></p>
<p>旧448帧与新360帧相机布置、外参及腕约束不同，不能作严格A/B。主线人体来自新PMPose冷启动，场景来自mainline→baseline；身体、beta和手形均未修改。</p>
<h2>腕点与人工像素差</h2><p>单位px。group为相机_手别；负的沿轴偏差表示向肘侧。轴由Sapiens2肘至人工腕定义，同时保存各模型自己的肘轴敏感性。WiLoR没有肘。组内重复帧不等于独立样本。</p>
<table><tr><th>数据</th><th>模型</th><th>组</th><th>数量</th><th>平均差</th><th>中位差</th><th>沿轴均值</th></tr>MANUAL_ROWS</table>
<h2>身体拟合与自身三角化观测</h2><p>下表是毫米中位残差，不是解剖精度。PMPose_ablation_baseline和PMPose_ablation_no_wrist来自同一旧数据受控消融，仅关闭腕软参考；新数据同样关闭人工腕项。新Sapiens2初始化失败，没有人体网格。</p>
<table><tr><th>数据</th><th>结果</th><th>左肘</th><th>右肘</th><th>左腕</th><th>右腕</th></tr>BODY_ROWS</table>
<h2>原因判断</h2>
<p>① Sapiens2沿前臂向肘侧偏差在两次数据中仍出现。左相机右手组平均像素差19.22→11.39px，沿轴均值−18.86→−9.61px；说明该视角改善，不能推出整个视频精度提高。</p>
<p>② 新数据frame0左图，两个人工左右腕标签几乎重合在同一只手上；它制造约800px大残差。frame74右图，Sapiens2左右腕都落在图像右侧一只手附近；与人工右腕相差846.82px。frame139右图另有830.43px残差。原始标签和检测保持不变，所有异常仍在统计内。</p>
<p>③ 肘与腕之间有观测/模型长度冲突。PMPose旧三角化前臂中位约190/193mm，模型约248/251mm；新观测约203/180mm，模型约244/247mm。内部关节与COCO回归点仅相差数毫米，不能单独解释数十至上百毫米拟合残差。原因仍可能涉及二维定位、双目对应、标定/同步、模型比例与先验，尚无独立真值作最终归因。</p>
<p>④ 关闭旧数据腕参考后，肘残差110/115→101/100mm，腕残差24/26→63/49mm。腕约束确实改变取舍，但关闭后肘的大残差仍在。新数据同样关闭腕项，肘66/77mm、腕70/46mm；相机和图像条件改变后观测兼容性有差异，不能全部归因某个模块。</p>
<p>⑤ 当前共同地面刚体变换不改变同帧模型与观测的欧氏距离，因此Stage修正不会修复上述像素偏差和相机系腕肘残差；它会影响世界系轨迹和视觉运动，两者需分开评估。</p>
<p>腕位置没有复用旧坐标：PMPose拟合腕到本次左相机光心的左右距离中位，由旧302/495mm变为新412/558mm；原始三角化由321/517mm变为395/545mm。两次相机布置和动作不同，这不能当成同一世界坐标系中的真实抓握后移量。</p>
<p>⑥ 旧人工双目通过5组、新仅1组。与模型同时通过的旧PMPose/Sapiens2/WiLoR为4/0/2，新为1/0/0。新唯一PMPose距离约9.96mm，只是该点的操作性参考一致性，不能据此给三维精度排名。肘没有独立人工标注。</p>
<h2>原图核查</h2><p><a href="../G20261003_wrist_detector_manual_v2/wrist_points/index.html">旧数据腕点</a> · <a href="../G20261003_repositioned360_manual_v1/wrists/index.html">新数据腕点</a></p>
<p>红人工、黄PMPose、绿Sapiens2、蓝WiLoR；原图点位未移动。</p>
<img src="../G20261003_repositioned360_manual_v1/wrists/images/0000_left.png" alt="首帧人工标签近重合"><img src="../G20261003_repositioned360_manual_v1/wrists/images/0074_right.png" alt="Sapiens2双腕近重合">
<h2>下一步</h2><p>优先做单模型左右/前臂链一致性诊断，显式标出双腕重合、跨侧跳变及双目不对应，不自动交换标签或补齐点；在已有人工槽位上验证局部腕定位处理。身体侧先利用既有同数据无腕消融核查长度冲突和肘残差，再做有观测保护的局部上肢更新。Stage、显示平滑和固定腕位移补丁不能代替这些验证。</p></html>"""
    (output / "arm_review.html").write_text(text.replace("MANUAL_ROWS", "".join(rows))
                                           .replace("BODY_ROWS", "".join(body)), encoding="utf-8")


def dataset(label, comparison, fits):
    protocol = read(comparison / "protocol.json")
    labels = read(comparison / "points.json")
    n = protocol.get("sequence_frames", 448)
    observations, geometry = {}, {}
    full = {}
    for method in ("PMPose", "Sapiens2"):
        paths = protocol["inputs"][method]
        observations[method] = {}
        for camera in ("left", "right"):
            rows = raw_side(Path(paths[camera + "_raw"]), camera)
            if sorted(rows) != list(range(n)):
                raise ValueError("noncontiguous observation frames")
            observations[method][camera] = np.stack([rows[i] for i in range(n)])
        cal = load_stereo_fisheye(Path(paths["calibration_dir"]))
        tri = raw_triangulate(observations[method]["left"], observations[method]["right"], cal)
        xyz, el, er, _, dl, dr, accepted, reasons, _, _ = tri
        geometry[method] = tri
        full[method] = {"joints": {}, "scope": "first-instance engineering triangulation; not strict association acceptance"}
        for name, j in JOINTS.items():
            full[method]["joints"][name] = {
                "engineering_accepted": int(accepted[:, j].sum()), "total": n,
                "negative_depth": int(((dl[:, j] <= 0) | (dr[:, j] <= 0)).sum()),
                "rejection_reasons": dict(Counter(str(v) for v in reasons[~accepted[:, j], j])),
                "reprojection_px_accepted": stats(((el[:, j] + er[:, j]) / 2)[accepted[:, j]])}
        for hand, e, w in (("left", 7, 9), ("right", 8, 10)):
            valid = accepted[:, e] & accepted[:, w]
            full[method][hand + "_forearm_length_mm"] = stats(np.linalg.norm(xyz[:, w] - xyz[:, e], axis=1)[valid])
            full[method][hand + "_wrist_camera_distance_mm"] = stats(np.linalg.norm(xyz[:, w], axis=1)[accepted[:, w]])

    records = []
    states = Counter()
    for row in labels:
        states[row["manual"]["state"]] += 1
        if row["manual"]["state"] != "present":
            continue
        f, camera, hand = row["frame_index"], row["camera"], row["hand"]
        elbow = 7 if hand == "left" else 8
        manual = np.array(row["manual"]["raw_px"])
        common = observations["Sapiens2"][camera][f, elbow, :2]
        for method in ("PMPose", "Sapiens2", "WiLoR"):
            measurement = row["methods"][method]
            if measurement["state"] != "compared":
                continue
            prediction = np.array(measurement["prediction_raw_px"])
            own_elbow = observations[method if method != "WiLoR" else "Sapiens2"][camera][f, elbow, :2]
            records.append({"capture": label, "frame": f, "camera": camera, "hand": hand, "method": method,
                            "manual_raw_px": manual.tolist(), "prediction_raw_px": prediction.tolist(),
                            "error_px": float(np.linalg.norm(prediction - manual)),
                            "common_axis": signed_components(prediction, manual, common),
                            "own_axis": signed_components(prediction, manual, own_elbow),
                            "error_over_100px_diagnostic_flag": bool(np.linalg.norm(prediction - manual) > 100)})
    manual_report = {m: grouped_manual([r for r in records if r["method"] == m])
                     for m in ("PMPose", "Sapiens2", "WiLoR")}
    stereo = read(comparison / "stereo_diagnostics.json")
    stereo_report = {"manual_states": dict(Counter(r["manual"]["state"] for r in stereo)),
                     "methods": {m: dict(Counter(r["methods"][m]["state"] for r in stereo))
                                 for m in ("PMPose", "Sapiens2", "WiLoR")}}
    stereo_report["paired_accepted_distance_mm"] = {
        m: stats([r["methods"][m]["manual_distance_mm"] for r in stereo
                  if r["manual"]["state"] == "accepted_manual_reference_geometry"
                  and r["methods"][m]["state"] == "accepted_manual_reference_geometry"
                  and "manual_distance_mm" in r["methods"][m]])
        for m in ("PMPose", "Sapiens2", "WiLoR")}
    body_report = {}
    for method, fit in fits.items():
        detector = method.split("_", 1)[0]
        if not fit.exists():
            body_report[method] = {"state": "unavailable", "reason": "initialization_failed_no_body_result"}
            continue
        with np.load(fit) as a:
            if not np.array_equal(a["pair_id"], np.arange(n)):
                raise ValueError("body frame range mismatch")
            coco, joints = a["predicted_coco"], a["smplh_joints"]
            raw, mask = a["raw_triangulated_points"], a["body_accepted"]
            # Frozen fitter archives store these mm arrays in float32. Match
            # the existing viewer's 0.001 mm replay tolerance, not pixel gates.
            np.testing.assert_allclose(raw, geometry[detector][0], atol=0.001, equal_nan=True)
            np.testing.assert_array_equal(mask, geometry[detector][6])
            residual = coco * 1000 - raw
            item = {"source": str(fit), "state": "engineering_candidate", "joint_residuals": {},
                    "wrist_reference": read(fit.parent / "fit_summary.json")["wrist_reference"]}
            for name, j in JOINTS.items():
                good = mask[:, j]
                native = {7: 18, 8: 19, 9: 20, 10: 21}[j]
                item["joint_residuals"][name] = {
                    "fit_minus_observation_mm": stats(np.linalg.norm(residual[:, j], axis=1)[good]),
                    "mean_signed_camera_xyz_mm": residual[good, j].mean(axis=0).tolist() if good.any() else None,
                    "native_vs_coco_mm": stats(np.linalg.norm(coco[:, j] - joints[:, native], axis=1) * 1000)}
            cal = load_stereo_fisheye(Path(protocol["inputs"][detector]["calibration_dir"]))
            item["projection_residuals"] = {}
            for camera in ("left", "right"):
                px = project(coco, cal, camera)
                observed = observations[detector][camera][..., :2]
                item["projection_residuals"][camera] = {
                    name: stats(np.linalg.norm(px[:, j] - observed[:, j], axis=-1)[mask[:, j]])
                    for name, j in JOINTS.items()}
            for hand, e, w in (("left", 7, 9), ("right", 8, 10)):
                item[hand + "_model_forearm_length_mm"] = stats(np.linalg.norm(coco[:, w] - coco[:, e], axis=1) * 1000)
                item[hand + "_internal_forearm_length_mm"] = stats(np.linalg.norm(joints[:, w + 11] - joints[:, e + 11], axis=1) * 1000)
                item[hand + "_wrist_camera_distance_mm"] = stats(np.linalg.norm(coco[:, w], axis=1) * 1000)
            body_report[method] = item
    return {"frames": n, "comparison": str(comparison), "manual_states": dict(states),
            "manual_pixel_disagreement": manual_report, "manual_stereo": stereo_report,
            "full_sequence_geometry": full, "body_fit_consistency": body_report}, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if (args.output / "arm_comparison.json").exists():
        raise FileExistsError("refuse existing report")
    args.output.mkdir(exist_ok=True, parents=True)
    route = read(ROOT / "realtime_app/configs/current_mainline.json")
    old_fits = {"PMPose": ROOT / route["matched_control"]["result"] / "fit/result.npz",
                "Sapiens2": ROOT / route["body_candidate"]["result"] / "fit/result.npz"}
    ablation = RECORDS / "G20261003_smplh_ablation_v1"
    before = read(ablation / "pmpose_baseline/resolved_options.json")
    after = read(ablation / "pmpose_no_wrist/resolved_options.json")
    changed = {k for k in before if before[k] != after[k]}
    if changed != {"output_dir", "wrist_reference", "wrist_reference_weight"}:
        raise ValueError("wrist ablation contains unexpected option changes")
    for case in ("baseline", "no_wrist"):
        old_fits["PMPose_ablation_" + case] = ablation / ("pmpose_" + case) / "fit/result.npz"
    new = RECORDS / "G20261003_repositioned360_manual_v1"
    specifications = [("old448", RECORDS / "G20261003_wrist_detector_manual_v2", old_fits),
                      ("new360", new / "comparison", {"PMPose": new / "pmpose/fit/result.npz",
                                                       "Sapiens2": new / "sapiens2/fit/result.npz"})]
    result, records = {}, []
    for label, comparison, fits in specifications:
        result[label], rows = dataset(label, comparison, fits)
        records.extend(rows)
    report = {"datasets": result, "wrist_ablation_changed_options": sorted(changed),
              "reviewed_anomalies": [
                  {"capture": "new360", "frame": 0, "camera": "left",
                   "state": "manual_left_right_points_near_coincident",
                   "evidence": "Both label shapes within about3.4px of the same visible hand; label files left unchanged."},
                  {"capture": "new360", "frame": 74, "camera": "right",
                   "state": "Sapiens2_both_wrists_on_one_visible_hand",
                   "evidence": "Original point overlay inspected; right-wrist residual846.82px. Predictions left unchanged."},
                  {"capture": "new360", "frame": 139, "camera": "right",
                   "state": "large_Sapiens2_left_wrist_disagreement",
                   "error_px": 830.4307921624137}],
              "limits": ["Captures differ in cameras/extrinsics, viewpoints, lighting and motion; not controlled A/B.",
                         "Old body uses diagnostic manual wrist prior; new body does not. Fit residual is not detector accuracy.",
                         "No manual elbow labels or independent anatomical 3D truth. Elbow axes are estimated.",
                         "All labels and gross residuals retained; 100px flags are diagnostics, not exclusion gates.",
                         "Within-capture rigid ground transforms preserve fit-observation Euclidean distances.",
                         "No new Sapiens2 body fit after initialization failure; no old mesh substitution."]}
    for name, value in (("arm_comparison.json", report), ("arm_records.json", records)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render_review(report, args.output)
    print(json.dumps({k: v["manual_pixel_disagreement"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
