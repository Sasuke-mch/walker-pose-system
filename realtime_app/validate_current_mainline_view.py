"""Verify exported geometry against selected body and fresh scene artifacts."""
import argparse
import base64
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from pose_app.fisheye_camera import load_stereo_fisheye


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    args = parser.parse_args()
    out = args.view
    protocol = read(out / "protocol.json")
    selected = protocol["selected_body"]
    with np.load(selected["result"], allow_pickle=False) as archive:
        body = {k: archive[k] for k in archive.files}
    with np.load(out / "result.npz", allow_pickle=False) as archive:
        exported = {k: archive[k] for k in archive.files}
    with np.load(out / "scene/scene_transforms.npz", allow_pickle=True) as archive:
        scene = {k: archive[k] for k in archive.files}
    page = (out / "body.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',
                                page, re.DOTALL)[1])
    n = protocol["frames"]

    def decode(key, shape):
        return np.frombuffer(base64.b64decode(data[key]), dtype="<f4").reshape(shape)

    def xf(points):
        return (np.einsum("nij,nvj->nvi", scene["rotation_ground_from_left"], points)
                + scene["translation_ground_from_left_mm"][:, None] / 1000)

    def equal(a, b):
        numeric = np.asarray(a).dtype.kind in "biufc" and np.asarray(b).dtype.kind in "biufc"
        return np.array_equal(a, b, equal_nan=numeric)

    checks = {}
    checks["source_body_all_fields_identical"] = all(
        body[k].dtype == exported[k].dtype and equal(body[k], exported[k]) for k in body)
    checks["mesh_shapes"] = (data["n"] == n and data["nv"] == 6890
                             and np.shape(data["faces"]) == (13776, 3))
    checks["faces_from_selected_body"] = equal(data["faces"], body["faces"])
    for key, array, shape in (("v", body["vertices"], (n, 6890, 3)),
                               ("j", body["predicted_coco"], (n, 17, 3)),
                               ("t", body["raw_triangulated_points"] / 1000, (n, 17, 3))):
        checks[key + "_fresh_transform_exact"] = equal(decode(key, shape), xf(array).astype(np.float32))
    checks["strict_mask_exact"] = equal(np.array(data["acc"]).reshape(n, 17), scene["accepted"])
    checks["ankle_trail_strict_xyz_exact"] = equal(decode("a", (n, 2, 3)),
        xf(scene["raw_triangulated_points_mm"] / 1000)[:, [15, 16]].astype(np.float32))
    checks["ankle_trail_mask_exact"] = equal(np.array(data["av"]).reshape(n, 2), scene["accepted"][:, [15, 16]])
    checks["fit_weights_exact"] = equal(data["fit_weights"], body["body_fit_weights"])
    checks["old_quality_separate_exact"] = equal(data["quality"], body["body_quality"])
    expected_pairs = protocol["source_pair_ids"]
    if isinstance(expected_pairs, dict):
        expected_pairs = list(range(expected_pairs["start"], expected_pairs["end"] + 1))
    elif isinstance(expected_pairs, str):
        first, last = map(int, expected_pairs.split(".."))
        expected_pairs = list(range(first, last + 1))
    checks["source_pair_ids_exact"] = data["source_pair_ids"] == expected_pairs
    stages = Counter(read_row["stage"]["operational"] for read_row in
                     (json.loads(line) for line in (out / "scene/stage1_stage2.jsonl").read_text(encoding="utf-8").splitlines()))
    checks["stage_counts_current"] = Counter(data["stage"]) == stages
    sources = read(out / "scene/scene_sources.json")
    checks["old_scene_not_consumed"] = all(sources[k] is False for k in
        ("old_stage_jsonl_read", "old_dynamic_ground_read", "old_dynamic_walker_pose_read"))
    strict_lines = (out / "first_instance_strict.jsonl").read_text(encoding="utf-8").splitlines()
    strict_count = 0
    for i, line in enumerate(strict_lines):
        for person in json.loads(line)["persons_3d"]:
            for point in person["keypoints_3d"]:
                if point["valid"]:
                    strict_count += 1
                    np.testing.assert_array_equal(scene["raw_triangulated_points_mm"][i, point["index"]], point["xyz"])
    checks["strict_points_count"] = len(strict_lines) == n and int(scene["accepted"].sum()) == strict_count
    checks["strict_accepted_xyz_current_records"] = True
    options = read(selected["resolved_options"])
    cal = load_stereo_fisheye(Path(options["calibration_dir"]))
    c1 = -cal.R_cam0_to_cam1.T @ (cal.T_cam0_to_cam1_mm / 1000)
    checks["both_camera_centers_exact"] = equal(decode("c", (n, 2, 3)),
        xf(np.broadcast_to(np.stack([np.zeros(3), c1]), (n, 2, 3))).astype(np.float32))
    walker = read(out / "scene/static_walker_model.json")
    nodes = (np.array([walker["nodes_walker_mm"][name] for name in data["wn"]]) / 1000
             @ np.array(walker["rotation_left_camera_from_walker"]).T
             + np.array(walker["translation_left_camera_from_walker_mm"]) / 1000)
    checks["all_walker_nodes_exact"] = equal(decode("w", (n, len(nodes), 3)),
        xf(np.broadcast_to(nodes, (n, len(nodes), 3))).astype(np.float32))
    checks["all_walker_edges_exact"] = [[data["wn"].index(a), data["wn"].index(b)]
                                       for a, b in walker["edges"]] == data["we"]
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
                  "stage_counts": dict(stages), "strict_accepted_points": int(scene["accepted"].sum()),
                  "limits": "Source/transform consistency; not independent accuracy or contact truth"}
    (out / "source_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)
                                               + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise ValueError("viewer source consistency failed")


if __name__ == "__main__":
    main()
