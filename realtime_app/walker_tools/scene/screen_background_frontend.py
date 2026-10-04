"""Selected-frame diagnostic; mask-off runs are never eligible for pose use."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pose_app.body_observations import raw_side
from pose_app.calibration import StereoCalibration
from pose_app.static_background_stereo_frontend import (
    StereoFrontendCriteria,
    extract_frame_stereo_features,
    extract_rectified_frame_features,
    make_local_rectification,
    person_exclusion_feature_mask,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    cal = StereoCalibration.load(a.root / "calibration/stereo_fisheye.json")
    points = {
        s: raw_side(a.raw_root / f"{s}_pmpose_raw.json", s) for s in ["left", "right"]
    }
    report = []
    for i in [0, 85, 123, 200, 275, 346]:
        for width in [480, 960]:
            size = (width, round(width * 1080 / 1920))
            c = cal.for_runtime_sizes(size, size)
            images = {}
            masks = {}
            for side, folder, rotation in [
                ("left", "left_ccw90", cv2.ROTATE_90_CLOCKWISE),
                ("right", "right_cw90", cv2.ROTATE_90_COUNTERCLOCKWISE),
            ]:
                image = cv2.imread(str(a.root / "input" / folder / f"pair_{i:04d}.png"))
                if image is None:
                    raise FileNotFoundError(f"{i} {side}")
                images[side] = cv2.resize(cv2.rotate(image, rotation), size)
                masks[side] = person_exclusion_feature_mask(
                    images[side].shape[:2],
                    [{"keypoints": points[side][i].tolist()}],
                    scale=width / 1920,
                    keypoint_score_threshold=0.25,
                    dilation_fraction=0.055,
                )
            for masked in [True, False]:
                mm = (
                    masks
                    if masked
                    else {s: np.full(size[::-1], 255, np.uint8) for s in images}
                )
                f = extract_frame_stereo_features(
                    images["left"],
                    images["right"],
                    mm["left"],
                    mm["right"],
                    c,
                    StereoFrontendCriteria(),
                )
                report.append(
                    dict(
                        frame=i,
                        width=width,
                        masked=masked,
                        mode="raw",
                        coverage={s: float(np.mean(m == 255)) for s, m in mm.items()},
                        pose_use_allowed=masked,
                        **f.diagnostics,
                    )
                )
            for focal in [0.35, 0.45, 0.65]:
                rect = make_local_rectification(c, size, focal_px=width * focal)
                f = extract_rectified_frame_features(
                    images["left"],
                    images["right"],
                    masks["left"],
                    masks["right"],
                    rect,
                    StereoFrontendCriteria(),
                )
                report.append(
                    dict(
                        frame=i,
                        width=width,
                        masked=True,
                        mode="rectified",
                        focal_fraction=focal,
                        **f.diagnostics,
                    )
                )
                if i == 123 and width == 480 and focal == 0.65:
                    raw = np.hstack([images["left"], images["right"]])
                    warped = np.hstack(
                        [
                            cv2.remap(
                                images["left"],
                                rect.map_left_x,
                                rect.map_left_y,
                                cv2.INTER_LINEAR,
                            ),
                            cv2.remap(
                                images["right"],
                                rect.map_right_x,
                                rect.map_right_y,
                                cv2.INTER_LINEAR,
                            ),
                        ]
                    )
                    cv2.imwrite(
                        str(a.output / "background_frame123.jpg"),
                        np.vstack([raw, warped]),
                    )
    (a.output / "frontend_diagnosis.json").write_text(
        json.dumps(
            {
                "inputs": vars_as_paths(a),
                "frames": [0, 85, 123, 200, 275, 346],
                "screen_is_exploratory": True,
                "results": report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def vars_as_paths(a):
    return {k: str(v.resolve()) for k, v in vars(a).items()}


if __name__ == "__main__":
    main()
