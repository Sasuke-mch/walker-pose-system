from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EDGES = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
    (2, 4), (0, 5), (0, 6),
)

NAMES = {
    "bboxmaskpose": "YOLO26x + BBoxMaskPose (compatibility)",
    "pmpose": "YOLO26x + PMPose-b",
    "probpose": "YOLO26x + ProbPose-s",
    "sapiens2": "YOLO26x + Sapiens2-0.4B",
    "yolo26x_pose": "YOLO26x-pose",
}

COLORS = {
    "valid": (40, 220, 40),
    "high_reprojection_error": (0, 165, 255),
    "low_2d_score": (130, 130, 130),
    "negative_depth": (255, 0, 255),
    "association_failed": (255, 255, 0),
    "other": (30, 30, 230),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize saved pose predictions and stereo geometry decisions."
    )
    parser.add_argument("--left-video", required=True, type=Path)
    parser.add_argument("--right-video", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(NAMES),
        choices=list(NAMES),
    )
    parser.add_argument("--keypoint-threshold", type=float, default=0.25)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--contact-frames", type=int, nargs="*", default=[0, 20, 40, 59])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing geometry JSONL: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise RuntimeError(f"No records in {path}")
    if [int(row["pair_id"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"pair_id must be continuous 0..N-1: {path}")
    return rows


def clip_point(x, y, width, height):
    return (
        int(round(np.clip(x, 0, width - 1))),
        int(round(np.clip(y, 0, height - 1))),
    )


def statuses_for_person(record, side, person_id):
    for stereo_person in record.get("persons_3d", []):
        if stereo_person.get(f"{side}_person_id") != person_id:
            continue
        return {
            int(point["index"]): (
                "valid" if point.get("valid") else point.get("reason") or "other"
            )
            for point in stereo_person.get("keypoints_3d", [])
        }
    return {}


def draw_person(image, person, statuses, threshold, associated):
    height, width = image.shape[:2]
    bbox = person.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1 = clip_point(bbox[0], bbox[1], width, height)
        x2, y2 = clip_point(bbox[2], bbox[3], width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
        text = f"person {person.get('person_id', '?')}  score={person.get('pose_score', 0):.2f}"
        cv2.putText(
            image, text, (x1 + 4, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )

    keypoints = person.get("keypoints", [])
    visible = []
    for point in keypoints[:17]:
        if len(point) < 3:
            visible.append(False)
            continue
        x, y, score = map(float, point[:3])
        visible.append(score >= threshold and 0 <= x < width and 0 <= y < height)

    for a, b in EDGES:
        if a >= len(visible) or b >= len(visible) or not visible[a] or not visible[b]:
            continue
        sa = statuses.get(a, "association_failed" if not associated else "other")
        sb = statuses.get(b, "association_failed" if not associated else "other")
        color = COLORS["valid"] if sa == sb == "valid" else COLORS["high_reprojection_error"]
        pa = clip_point(keypoints[a][0], keypoints[a][1], width, height)
        pb = clip_point(keypoints[b][0], keypoints[b][1], width, height)
        cv2.line(image, pa, pb, color, 2, cv2.LINE_AA)

    for index, point in enumerate(keypoints[:17]):
        if len(point) < 3:
            continue
        x, y, score = map(float, point[:3])
        if not (0 <= x < width and 0 <= y < height):
            continue
        if score < threshold:
            color, radius = COLORS["low_2d_score"], 2
        else:
            state = statuses.get(index, "association_failed" if not associated else "other")
            color, radius = COLORS.get(state, COLORS["other"]), 5
        cv2.circle(image, clip_point(x, y, width, height), radius, color, -1, cv2.LINE_AA)


def draw_side(image, record, side, threshold):
    painted = image.copy()
    associated_ids = {
        item[f"{side}_person_id"]
        for item in record.get("persons_3d", [])
        if f"{side}_person_id" in item
    }
    for person in record.get(side, {}).get("persons", []):
        person_id = person.get("person_id")
        draw_person(
            painted,
            person,
            statuses_for_person(record, side, person_id),
            threshold,
            person_id in associated_ids,
        )
    return painted


def summary(record):
    people = record.get("persons_3d", [])
    if not people:
        return "associated=0  valid_3D=0/17"
    valid = sum(int(item.get("valid_keypoints", 0)) for item in people)
    cost = min(float(item.get("association_cost", 0)) for item in people)
    reproj = [
        float(item["mean_reprojection_error_px"])
        for item in people
        if item.get("mean_reprojection_error_px") is not None
    ]
    reproj_text = f"{float(np.mean(reproj)):.2f}px" if reproj else "n/a"
    return f"associated={len(people)}  valid_3D={valid}/17  epi_cost={cost:.4f}  reproj={reproj_text}"


def compose(left, right, record, model, panel_width, threshold):
    left = draw_side(left, record, "left", threshold)
    right = draw_side(right, record, "right", threshold)

    source_height, source_width = left.shape[:2]
    panel_height = int(round(panel_width * source_height / source_width))
    left = cv2.resize(left, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (panel_width, panel_height), interpolation=cv2.INTER_AREA)

    header_height = 56
    canvas = np.zeros((header_height + panel_height, panel_width * 2, 3), dtype=np.uint8)
    canvas[header_height:, :panel_width] = left
    canvas[header_height:, panel_width:] = right

    title = f"{NAMES[model]} | pair {record['pair_id']:03d} | {summary(record)}"
    legend = "green=valid_3D  orange=high_reprojection  gray=low_2D_score  cyan=association_failed"
    cv2.putText(canvas, title, (16, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, legend, (16, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def read_pair(left_cap, right_cap, index):
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok_left, left = left_cap.read()
    ok_right, right = right_cap.read()
    if not ok_left or not ok_right:
        raise RuntimeError(f"Cannot read raw stereo pair {index}")
    return left, right


def write_contact_sheet(path, frames):
    rows = []
    for model, frame in frames.items():
        width = 960
        height = int(round(width * frame.shape[0] / frame.shape[1]))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        row = np.zeros((height + 34, width, 3), dtype=np.uint8)
        row[34:] = frame
        cv2.putText(row, NAMES[model], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise RuntimeError(f"Cannot write: {path}")


def main():
    args = parse_args()
    if args.panel_width < 320 or args.panel_width % 2:
        raise ValueError("--panel-width must be an even integer >= 320")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {
        model: load_records(args.geometry_root / model / "offline_stereo_results.jsonl")
        for model in args.models
    }
    count = min(len(rows) for rows in records.values())
    if args.limit is not None:
        count = min(count, args.limit)
    if count <= 0:
        raise RuntimeError("No selected frames")

    left_cap = cv2.VideoCapture(str(args.left_video))
    right_cap = cv2.VideoCapture(str(args.right_video))
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError("Cannot open one or both raw stereo videos")

    try:
        fps = left_cap.get(cv2.CAP_PROP_FPS) or 30.0
        if min(
            int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        ) < count:
            raise RuntimeError("Raw videos contain fewer frames than geometry results")

        contact_indices = {item for item in args.contact_frames if 0 <= item < count}
        contact_cache = {item: {} for item in contact_indices}
        outputs = []

        for model in args.models:
            output = args.output_dir / f"{model}_stereo_geometry.mp4"
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Output exists: {output}")

            first_left, first_right = read_pair(left_cap, right_cap, 0)
            first = compose(
                first_left, first_right, records[model][0],
                model, args.panel_width, args.keypoint_threshold,
            )
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (first.shape[1], first.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create MP4: {output}")

            try:
                for index in range(count):
                    left, right = read_pair(left_cap, right_cap, index)
                    frame = compose(
                        left, right, records[model][index],
                        model, args.panel_width, args.keypoint_threshold,
                    )
                    writer.write(frame)
                    if index in contact_cache:
                        contact_cache[index][model] = frame
            finally:
                writer.release()

            outputs.append(output.name)
            print(f"Wrote {output}")

        for index, frames in contact_cache.items():
            output = args.output_dir / f"comparison_pair_{index:03d}.jpg"
            write_contact_sheet(output, frames)
            outputs.append(output.name)
            print(f"Wrote {output}")

        manifest = {
            "left_video": str(args.left_video),
            "right_video": str(args.right_video),
            "geometry_root": str(args.geometry_root),
            "models": args.models,
            "frames": count,
            "fps": fps,
            "keypoint_threshold": args.keypoint_threshold,
            "outputs": outputs,
        }
        (args.output_dir / "visualization_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        left_cap.release()
        right_cap.release()


if __name__ == "__main__":
    main()