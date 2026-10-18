#!/usr/bin/env python3
"""Extract offline walker-frame evidence from LabelMe floor exclusions.

The output is deliberately an evidence catalogue, not a completed walker model.
Narrow non-floor slots recovered by horizontal closing of manual floor masks are
treated as possible camera-attached tube occlusions.  Explicit walker polygons
and independently audited stereo keypoints are preserved as stronger sources.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


SIDES = ("left", "right")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def polygon_mask(label: dict[str, Any], name: str) -> np.ndarray:
    height, width = int(label["imageHeight"]), int(label["imageWidth"])
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in label.get("shapes", []):
        if shape.get("label") != name or shape.get("shape_type") != "polygon":
            continue
        points = np.rint(np.asarray(shape.get("points", []), dtype=np.float64)).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)
    return mask


def _axis_from_pixels(component: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ys, xs = np.nonzero(component)
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = np.mean(points, axis=0)
    centered = points - center
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    if direction[1] < 0:
        direction = -direction
    projected = centered @ direction
    low, high = np.percentile(projected, [3.0, 97.0])
    verticality = abs(float(direction[1]))
    return center + low * direction, center + high * direction, verticality


def floor_notch_mask(floor_mask: np.ndarray, close_width: int) -> np.ndarray:
    """Recover narrow horizontal gaps inside labelled floor support."""
    if close_width < 3 or close_width % 2 == 0:
        raise ValueError("close_width must be odd and at least 3")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 5))
    closed = cv2.morphologyEx(
        floor_mask,
        cv2.MORPH_CLOSE,
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    recovered = cv2.bitwise_and(closed, cv2.bitwise_not(floor_mask))
    recovered[: int(0.38 * floor_mask.shape[0])] = 0
    return recovered


def component_candidates(
    notch_mask: np.ndarray,
    image: np.ndarray,
    floor_mask: np.ndarray,
    walker_mask: np.ndarray,
    person_mask: np.ndarray,
    *,
    minimum_height: int,
    maximum_width: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(notch_mask, connectivity=8)
    accepted_mask = np.zeros_like(notch_mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    records: list[dict[str, Any]] = []
    for component_id in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[component_id]]
        component = labels == component_id
        axis_start, axis_end, verticality = _axis_from_pixels(component)
        aspect = height / max(width, 1)
        dark_fraction = float(np.mean(gray[component] <= 105))
        walker_overlap = float(np.mean(walker_mask[component] > 0))
        person_overlap = float(np.mean(person_mask[component] > 0))
        dilated = cv2.dilate(component.astype(np.uint8), np.ones((5, 5), np.uint8))
        floor_boundary_support = float(np.mean(floor_mask[dilated > 0] > 0))
        reasons = []
        if height < minimum_height:
            reasons.append("too_short")
        if width > maximum_width:
            reasons.append("too_wide")
        if aspect < 1.25:
            reasons.append("not_slender")
        if verticality < 0.62:
            reasons.append("not_vertical")
        if person_overlap > 0.20 and walker_overlap < 0.20:
            reasons.append("explicit_person_overlap")
        status = "candidate" if not reasons else "rejected"
        if status == "candidate":
            accepted_mask[component] = 255
        records.append(
            {
                "status": status,
                "reasons": reasons,
                "bbox_xywh": [x, y, width, height],
                "area_px": area,
                "aspect_height_over_width": aspect,
                "axis_start_upright_px": axis_start.tolist(),
                "axis_end_upright_px": axis_end.tolist(),
                "axis_verticality": verticality,
                "dark_pixel_fraction": dark_fraction,
                "floor_boundary_support": floor_boundary_support,
                "explicit_walker_overlap_fraction": walker_overlap,
                "explicit_person_overlap_fraction": person_overlap,
                "source": "manual_floor_horizontal_gap",
            }
        )
    return records, accepted_mask


def draw_preview(
    image: np.ndarray,
    floor_mask: np.ndarray,
    walker_mask: np.ndarray,
    records: list[dict[str, Any]],
    title: str,
) -> np.ndarray:
    canvas = image.copy()
    green = np.zeros_like(canvas)
    green[:, :, 1] = floor_mask
    canvas = cv2.addWeighted(canvas, 1.0, green, 0.18, 0.0)
    if np.any(walker_mask):
        contours, _ = cv2.findContours(walker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 220, 255), 4, cv2.LINE_AA)
    for record in records:
        start = tuple(np.rint(record["axis_start_upright_px"]).astype(int))
        end = tuple(np.rint(record["axis_end_upright_px"]).astype(int))
        color = (255, 0, 255) if record["status"] == "candidate" else (80, 80, 180)
        cv2.line(canvas, start, end, color, 5, cv2.LINE_AA)
        x, y, width, height = record["bbox_xywh"]
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        title,
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def make_contact_sheet(paths: list[Path], output: Path, columns: int = 4) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    thumb_width, thumb_height = 270, 480
    thumbs = [cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA) for image in images]
    rows = (len(thumbs) + columns - 1) // columns
    sheet = np.zeros((rows * thumb_height, columns * thumb_width, 3), dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row, column = divmod(index, columns)
        sheet[row * thumb_height : (row + 1) * thumb_height, column * thumb_width : (column + 1) * thumb_width] = thumb
    cv2.imwrite(str(output), sheet)


def cluster_persistent_axes(
    records: list[dict[str, Any]],
    side: str,
    image_height: int,
    *,
    minimum_unique_frames: int = 3,
    maximum_reference_offset_px: float = 55.0,
    maximum_slope_difference: float = 0.20,
) -> list[dict[str, Any]]:
    """Merge repeated near-vertical axes in the fixed camera image plane."""
    reference_y = 0.78 * image_height
    prepared = []
    for record in records:
        if record.get("side") != side or record.get("persistence_status") != "persistent_candidate":
            continue
        start = np.asarray(record["axis_start_upright_px"], dtype=np.float64)
        end = np.asarray(record["axis_end_upright_px"], dtype=np.float64)
        delta = end - start
        if abs(delta[1]) < 1e-6:
            continue
        slope = float(delta[0] / delta[1])
        reference_x = float(start[0] + slope * (reference_y - start[1]))
        prepared.append((record, slope, reference_x))
    prepared.sort(key=lambda item: (-item[0]["temporal_support_frames_maximum"], item[2]))

    clusters: list[dict[str, Any]] = []
    for record, slope, reference_x in prepared:
        compatible = []
        for index, cluster in enumerate(clusters):
            offset = abs(reference_x - float(np.mean(cluster["reference_xs"])))
            slope_delta = abs(slope - float(np.mean(cluster["slopes"])))
            if offset <= maximum_reference_offset_px and slope_delta <= maximum_slope_difference:
                compatible.append((offset + 120.0 * slope_delta, index))
        if compatible:
            cluster = clusters[min(compatible)[1]]
        else:
            cluster = {"records": [], "slopes": [], "reference_xs": []}
            clusters.append(cluster)
        cluster["records"].append(record)
        cluster["slopes"].append(slope)
        cluster["reference_xs"].append(reference_x)

    hypotheses = []
    for cluster in clusters:
        pair_ids = sorted({int(record["pair_id"]) for record in cluster["records"]})
        if len(pair_ids) < minimum_unique_frames:
            continue
        endpoints = np.asarray(
            [
                point
                for record in cluster["records"]
                for point in (record["axis_start_upright_px"], record["axis_end_upright_px"])
            ],
            dtype=np.float64,
        )
        center = np.median(endpoints, axis=0)
        centered = endpoints - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        if direction[1] < 0:
            direction = -direction
        projection = centered @ direction
        low, high = np.percentile(projection, [5.0, 95.0])
        start, end = center + low * direction, center + high * direction
        hypotheses.append(
            {
                "side": side,
                "axis_start_upright_px": start.tolist(),
                "axis_end_upright_px": end.tolist(),
                "unique_frame_count": len(pair_ids),
                "pair_ids": pair_ids,
                "member_count": len(cluster["records"]),
                "reference_x_px_at_y_0p78h": float(np.mean(cluster["reference_xs"])),
                "dx_over_dy": float(np.mean(cluster["slopes"])),
                "source": "clustered_persistent_manual_floor_notches",
                "semantic_identity": "possible_walker_tube_not_class_verified",
            }
        )
    return sorted(hypotheses, key=lambda row: row["reference_x_px_at_y_0p78h"])


def audited_anchors(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    audit = read_json(path)
    anchors = []
    for pair in audit.get("pairs", []):
        usable = pair.get("state") == "within_declared_annotation_tolerance"
        anchors.append(
            {
                "label": pair.get("label"),
                "status": "candidate" if usable else "rejected",
                "reason": None if usable else pair.get("state"),
                "xyz_left_camera_mm": pair.get("xyz_left_camera_mm"),
                "mean_reprojection_error_px": pair.get("mean_reprojection_error_px"),
                "ray_gap_mm": pair.get("ray_gap_mm"),
                "semantic_identity": "unassigned_walker_landmark",
            }
        )
    return {
        "source": str(path.resolve()),
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "anchors": anchors,
        "candidate_count": sum(anchor["status"] == "candidate" for anchor in anchors),
        "rejected_count": sum(anchor["status"] == "rejected" for anchor in anchors),
        "interpretation": "Geometry-compatible manually paired points; topology and physical landmark names remain unset.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stereo-keypoint-audit", type=Path)
    parser.add_argument("--close-width", type=int, default=81)
    parser.add_argument("--minimum-height", type=int, default=35)
    parser.add_argument("--maximum-width", type=int, default=150)
    args = parser.parse_args()

    root = args.label_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    preview_root = output / "previews"
    preview_root.mkdir()

    all_records: list[dict[str, Any]] = []
    explicit_walker_records: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    side_preview_paths: dict[str, list[Path]] = {side: [] for side in SIDES}
    side_candidate_masks: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    state_counts: Counter[str] = Counter()
    for side in SIDES:
        label_dir = root / "labels" / side
        image_dir = root / "images" / side
        side_preview = preview_root / side
        side_preview.mkdir()
        for label_path in sorted(label_dir.glob("pair_*.json")):
            label = read_json(label_path)
            image_path = image_dir / f"{label_path.stem}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            floor = polygon_mask(label, "floor_eligible")
            walker = polygon_mask(label, "walker")
            person = polygon_mask(label, "person")
            pair_id = int(label_path.stem.split("_")[-1])
            for shape_index, shape in enumerate(label.get("shapes", [])):
                if shape.get("label") == "walker" and shape.get("shape_type") == "polygon":
                    explicit_walker_records.append(
                        {
                            "pair_id": pair_id,
                            "side": side,
                            "shape_index": shape_index,
                            "points_upright_px": shape.get("points", []),
                            "source": "manual_explicit_walker_polygon",
                        }
                    )
            notch = floor_notch_mask(floor, args.close_width)
            candidates, accepted_mask = component_candidates(
                notch,
                image,
                floor,
                walker,
                person,
                minimum_height=args.minimum_height,
                maximum_width=args.maximum_width,
            )
            for index, candidate in enumerate(candidates):
                candidate.update({"pair_id": pair_id, "side": side, "candidate_id": index})
                state_counts[candidate["status"]] += 1
                all_records.append(candidate)
            side_candidate_masks[side].append(accepted_mask)
            preview = draw_preview(
                image,
                floor,
                walker,
                candidates,
                f"{side} pair {pair_id:04d} | magenta=candidate red=rejected yellow=walker",
            )
            preview_path = side_preview / f"{label_path.stem}.jpg"
            cv2.imwrite(str(preview_path), preview)
            preview_paths.append(preview_path)
            side_preview_paths[side].append(preview_path)

    consensus_outputs = {}
    support_maps: dict[str, np.ndarray] = {}
    for side, masks in side_candidate_masks.items():
        if not masks:
            continue
        support = np.sum(np.stack([mask > 0 for mask in masks]), axis=0).astype(np.uint8)
        support_maps[side] = support
        heat = cv2.applyColorMap(np.uint8(np.clip(support / max(len(masks), 1) * 255, 0, 255)), cv2.COLORMAP_TURBO)
        support_path = output / f"{side}_floor_notch_support.png"
        cv2.imwrite(str(support_path), heat)
        consensus_outputs[side] = {
            "annotated_frame_count": len(masks),
            "maximum_pixel_support": int(np.max(support)),
            "support_image": str(support_path),
        }

    persistence_counts: Counter[str] = Counter()
    for record in all_records:
        support = support_maps[record["side"]]
        start = np.asarray(record["axis_start_upright_px"], dtype=np.float64)
        end = np.asarray(record["axis_end_upright_px"], dtype=np.float64)
        samples = np.linspace(start, end, 64)
        xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, support.shape[1] - 1)
        ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, support.shape[0] - 1)
        counts = support[ys, xs]
        median_support = float(np.median(counts))
        record["temporal_support_frames_median"] = median_support
        record["temporal_support_frames_maximum"] = int(np.max(counts))
        if record["status"] != "candidate":
            persistence = "rejected_geometry"
        elif median_support >= 3.0 or record["explicit_walker_overlap_fraction"] >= 0.20:
            persistence = "persistent_candidate"
        else:
            persistence = "low_support_candidate"
        record["persistence_status"] = persistence
        persistence_counts[persistence] += 1

    fixed_axis_hypotheses: dict[str, list[dict[str, Any]]] = {}
    persistent_overlay_paths = {}
    for side in SIDES:
        base_image_path = root / "images" / side / "pair_0000.png"
        canvas = cv2.imread(str(base_image_path), cv2.IMREAD_COLOR)
        if canvas is None:
            continue
        hypotheses = cluster_persistent_axes(all_records, side, canvas.shape[0])
        fixed_axis_hypotheses[side] = hypotheses
        for index, hypothesis in enumerate(hypotheses, start=1):
            start = tuple(np.rint(hypothesis["axis_start_upright_px"]).astype(int))
            end = tuple(np.rint(hypothesis["axis_end_upright_px"]).astype(int))
            cv2.line(canvas, start, end, (255, 0, 255), 7, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"H{index}:{hypothesis['unique_frame_count']}f",
                start,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        path = output / f"{side}_persistent_floor_notch_axes.jpg"
        cv2.imwrite(str(path), canvas)
        persistent_overlay_paths[side] = str(path)

    fixed_axis_path = output / "fixed_2d_axis_hypotheses.json"
    fixed_axis_path.write_text(
        json.dumps(
            {
                "coordinate_frame": "per_side_upright_camera_image",
                "hypotheses": fixed_axis_hypotheses,
                "interpretation": "Repeated floor-notch axes only; correspondence between left and right views is not assigned yet.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    anchor_output = audited_anchors(args.stereo_keypoint_audit)
    if anchor_output is not None:
        (output / "stereo_anchor_candidates.json").write_text(
            json.dumps(anchor_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    records_path = output / "floor_notch_candidates.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in all_records),
        encoding="utf-8",
    )
    explicit_path = output / "explicit_walker_polygons.jsonl"
    explicit_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in explicit_walker_records),
        encoding="utf-8",
    )
    contact_sheet = output / "floor_notch_contact_sheet.jpg"
    make_contact_sheet(preview_paths, contact_sheet)
    side_contact_sheets = {}
    for side, paths in side_preview_paths.items():
        path = output / f"{side}_floor_notch_contact_sheet.jpg"
        make_contact_sheet(paths, path)
        side_contact_sheets[side] = str(path)
    summary = {
        "schema_version": "annotated_walker_evidence_v1",
        "label_root": str(root),
        "frame_count": sum(len(masks) for masks in side_candidate_masks.values()),
        "state_counts": dict(state_counts),
        "persistence_counts": dict(persistence_counts),
        "explicit_walker_polygon_count": len(explicit_walker_records),
        "explicit_walker_polygon_frames": sorted({record["pair_id"] for record in explicit_walker_records}),
        "floor_notch_method": {
            "operation": "horizontal morphological closing minus manual floor mask",
            "close_width_px": args.close_width,
            "minimum_height_px": args.minimum_height,
            "maximum_width_px": args.maximum_width,
            "interpretation": "Possible narrow occluders only; person, annotation topology and image borders can create false positives.",
        },
        "per_side_consensus": consensus_outputs,
        "persistent_axes_overlays": persistent_overlay_paths,
        "fixed_2d_axis_hypothesis_counts": {
            side: len(rows) for side, rows in fixed_axis_hypotheses.items()
        },
        "stereo_anchor_candidates": (
            {"candidate_count": anchor_output["candidate_count"], "rejected_count": anchor_output["rejected_count"]}
            if anchor_output is not None
            else None
        ),
        "model_fit_ready": False,
        "model_fit_blocker": "walker topology and semantic landmark correspondence await the user-provided reference model",
        "records": str(records_path),
        "explicit_walker_polygons": str(explicit_path),
        "fixed_2d_axis_hypotheses": str(fixed_axis_path),
        "contact_sheet": str(contact_sheet),
        "side_contact_sheets": side_contact_sheets,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
