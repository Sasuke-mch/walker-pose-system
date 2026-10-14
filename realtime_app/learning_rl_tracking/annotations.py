"""Evaluate user frame-zero labels without silently guessing distal-point semantics."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .body import PhysicsBody
from .data import ROOT, load_real
from .run import save_json, stats


def to_raw(point, side):
    u, v = point
    return np.array([1919 - v, u]) if side == "left" else np.array([v, 1079 - u])


def evaluate_labels(output: Path, distal_joints=False):
    folder = ROOT / "research_records/annotation_handoffs/H20261014_people1_pair0000"
    data = load_real()
    body = PhysicsBody(data.initial_fit, data.rotation[0], data.translation[0])
    model = data.project(body.points(), 0)
    labels = {"left_shoulder": 0, "right_shoulder": 1, "left_knee": 8, "right_knee": 9}
    if distal_joints:
        labels.update(
            left_hand=4, leftt_hand=4, right_hand=5, left_foot=10, right_foot=11
        )
    rows = []
    excluded = []
    for side_index, side in enumerate(("left", "right")):
        source = folder / f"labels/{side}_pair_0000.json"
        obj = json.loads(source.read_text(encoding="utf-8"))
        if (obj["imageWidth"], obj["imageHeight"]) != (1080, 1920):
            raise ValueError(
                "labels are expected in the existing upright image coordinates"
            )
        image = cv2.imread(str(folder / f"images/{side}_pair_0000.png"))
        original = cv2.imread(
            str(
                ROOT
                / f"research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/input_448pairs/{'left_ccw90' if side == 'left' else 'right_cw90'}/pair_0000.png"
            )
        )
        if image is None or original is None or not np.array_equal(image, original):
            raise ValueError(
                "annotation image does not match selected input pair0 pixels"
            )
        for shape in obj["shapes"]:
            label = shape["label"]
            if shape["shape_type"] != "point" or len(shape["points"]) != 1:
                raise ValueError("expected single-point annotations")
            if label not in labels:
                excluded.append(
                    dict(
                        side=side,
                        label=label,
                        reason="hand/foot semantics need confirmation; not automatically mapped",
                    )
                )
                continue
            j = labels[label]
            point = to_raw(shape["points"][0], side)
            rows.append(
                dict(
                    side=side,
                    label=label,
                    coco_index=int(j + 5),
                    manual_raw_pixel=point,
                    detector_raw_pixel=data.pixels[0, side_index, j],
                    initializer_raw_pixel=model[side_index, j],
                    detector_error_px=float(
                        np.linalg.norm(data.pixels[0, side_index, j] - point)
                    ),
                    initializer_error_px=float(
                        np.linalg.norm(model[side_index, j] - point)
                    ),
                )
            )
    result = dict(
        label_folder=str(folder),
        frame=0,
        image_identity="upright label PNG matches input pair0 pixels",
        distal_semantics="user confirmed wrist/ankle"
        if distal_joints
        else "unconfirmed, excluded",
        evaluated_points=len(rows),
        excluded_points=excluded,
        points=rows,
        detector_error_px=stats([r["detector_error_px"] for r in rows]),
        initializer_error_px=stats([r["initializer_error_px"] for r in rows]),
        use="independent first-frame check only; no RL training, init refit, reward tuning or sequence-generalization claim",
    )
    save_json(output / "first_frame_annotation_check.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "evaluated_points",
                    "detector_error_px",
                    "initializer_error_px",
                )
            }
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distal-joints", action="store_true")
    args = parser.parse_args()
    evaluate_labels(args.output, args.distal_joints)
