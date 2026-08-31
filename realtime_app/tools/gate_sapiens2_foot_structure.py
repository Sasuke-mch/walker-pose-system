#!/usr/bin/env python3
"""Conservatively reject unreliable Sapiens2 foot points; never synthesize them.

An ankle passes only its confidence threshold.  A toe or heel additionally
needs both ankles available and must be no farther from its own anatomical
ankle than from the opposite ankle.  This catches obvious cross-foot swaps in
near-range fisheye views.  It is a quality gate, not a correction method.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def side_of(joint: str) -> str:
    return "left" if joint.startswith("left_") else "right"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.50)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        raise ValueError("confidence must be in [0,1]")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with args.input_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows: grouped[row["image_id"]].append(row)

    output = []
    for image_id, items in grouped.items():
        by_joint = {x["joint_subject_anatomy"]: x for x in items}
        points = {}
        for joint, row in by_joint.items():
            if row["x_px_raw_fisheye"] and row["y_px_raw_fisheye"] and float(row["sapiens2_confidence"]) >= args.confidence:
                points[joint] = np.asarray([float(row["x_px_raw_fisheye"]), float(row["y_px_raw_fisheye"])])
        for joint, row in by_joint.items():
            own = side_of(joint)
            opposite = "right" if own == "left" else "left"
            conf_ok = joint in points
            if joint.endswith("_ankle"):
                passed, reason = conf_ok, "confidence" if conf_ok else "low_confidence"
            else:
                own_ankle, other_ankle = f"{own}_ankle", f"{opposite}_ankle"
                if not conf_ok:
                    passed, reason = False, "low_confidence"
                elif own_ankle not in points or other_ankle not in points:
                    passed, reason = False, "missing_ankle_for_assignment"
                else:
                    own_distance = float(np.linalg.norm(points[joint] - points[own_ankle]))
                    other_distance = float(np.linalg.norm(points[joint] - points[other_ankle]))
                    passed = own_distance <= other_distance
                    reason = "own_ankle_nearest" if passed else "closer_to_opposite_ankle"
            copied = dict(row)
            copied.update({"gate_confidence_threshold": args.confidence, "structural_gate_pass": str(passed).lower(), "structural_gate_reason": reason})
            output.append(copied)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "foot_points_with_structural_gate.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0])); w.writeheader(); w.writerows(output)
    summary = defaultdict(list)
    for row in output: summary[(row["condition"], row["side"], row["joint_subject_anatomy"])].append(row)
    summary_rows = []
    for key, items in sorted(summary.items()):
        summary_rows.append({"condition":key[0],"side":key[1],"joint":key[2],"image_count":len(items),"base_usable_count":sum(x["reference_usable"]=="true" for x in items),"structural_gate_count":sum(x["structural_gate_pass"]=="true" for x in items)})
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0]));w.writeheader();w.writerows(summary_rows)
    meta={"input":str(args.input_csv.resolve()),"confidence":args.confidence,"rule":"ankle confidence; toe/heel confidence plus nearest-own-ankle check","interpretation":"reject-only quality gate; does not correct points or establish ground truth"}
    (args.output_dir/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"rows":len(output),"passed":sum(x["structural_gate_pass"]=="true" for x in output),"output":str(args.output_dir)},ensure_ascii=False))


if __name__ == "__main__": main()
