"""Build an upright visual-audit pack for Sapiens2 lower-body pseudo-labels."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


JOINT_ORDER = ("left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle")
LEG_CHAINS = (("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle"))
COMPARE_MODELS = {"M1_PMPose_raw_fisheye", "M1_BBoxMaskPose_raw_fisheye", "M1_ProbPose_raw_fisheye"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", required=True, type=Path)
    parser.add_argument("--selection-review-csv", required=True, type=Path)
    parser.add_argument("--relative-error-csv", required=True, type=Path)
    parser.add_argument("--upright-input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def raw_to_upright(side: str, x_raw: float, y_raw: float) -> tuple[float, float]:
    if side == "left":
        return y_raw, 1919.0 - x_raw
    return 1079.0 - y_raw, x_raw


def load_font(size: int):
    for candidate in (Path(r"C:\Windows\Fonts\consola.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def add_reason(reasons: dict[str, set[str]], image_id: str, reason: str) -> None:
    reasons.setdefault(image_id, set()).add(reason)


def render_overlay(source: Path, destination: Path, image_row: dict, points: dict[str, dict], reason: str) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    mapped = {}
    for joint, row in points.items():
        if not row["x_px_raw_fisheye"] or not row["y_px_raw_fisheye"]:
            continue
        mapped[joint] = raw_to_upright(image_row["side"], float(row["x_px_raw_fisheye"]), float(row["y_px_raw_fisheye"]))

    for chain in LEG_CHAINS:
        chain_points = [mapped[name] for name in chain if name in mapped]
        if len(chain_points) == 3:
            draw.line(chain_points, fill="white", width=14)
            draw.line(chain_points, fill="black", width=7)
        for joint in chain:
            if joint not in mapped:
                continue
            x, y = mapped[joint]
            usable = points[joint]["reference_usable"].lower() == "true"
            if usable:
                draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="white")
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="black")
            else:
                draw.line((x - 10, y - 10, x + 10, y + 10), fill="black", width=5)
                draw.line((x - 10, y + 10, x + 10, y - 10), fill="black", width=5)

    font = load_font(25)
    title = (
        f"{image_row['image_id']} | {image_row['condition']} | "
        f"ankle_min={float(image_row['min_ankle_confidence']):.3f} | "
        f"ambiguous={image_row['target_ambiguous']} | {reason}"
    )
    banner_height = 44
    draw.rectangle((0, 0, image.width, banner_height), fill="white")
    draw.text((12, 8), title, fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=92)


def make_contact_sheets(images: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_width, panel_height = 405, 720
    for sheet_index in range(0, len(images), 4):
        batch = images[sheet_index:sheet_index + 4]
        sheet = Image.new("RGB", (panel_width * 2, panel_height * 2), "white")
        for index, path in enumerate(batch):
            panel = Image.open(path).convert("RGB")
            panel.thumbnail((panel_width, panel_height), Image.Resampling.LANCZOS)
            x = (index % 2) * panel_width + (panel_width - panel.width) // 2
            y = (index // 2) * panel_height + (panel_height - panel.height) // 2
            sheet.paste(panel, (x, y))
        sheet.save(output_dir / f"sheet_{sheet_index // 4 + 1:02d}.jpg", quality=92)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    reference_rows = read_csv(args.reference_csv.resolve())
    selection_rows = read_csv(args.selection_review_csv.resolve())
    error_rows = read_csv(args.relative_error_csv.resolve())
    selection_by_id = {row["image_id"]: row for row in selection_rows}
    points_by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in reference_rows:
        points_by_id[row["image_id"]][row["joint_subject_anatomy"]] = row

    ankle_errors: dict[str, list[float]] = defaultdict(list)
    model_ankle_usable: dict[str, int] = defaultdict(int)
    for row in error_rows:
        if row["condition_name"] not in COMPARE_MODELS or "ankle" not in row["joint"]:
            continue
        if row["reference_usable"].lower() != "true" or row["model_usable"].lower() != "true":
            continue
        ankle_errors[row["image_id"]].append(float(row["error_px"]))
        model_ankle_usable[row["image_id"]] += 1

    audit_rows = []
    for image_id in sorted(points_by_id):
        points = points_by_id[image_id]
        selection = selection_by_id[image_id]
        ankles = [points[name] for name in ("left_ankle", "right_ankle") if name in points]
        confidences = [float(row["sapiens2_confidence"]) for row in points.values()]
        ankle_confidences = [float(row["sapiens2_confidence"]) for row in ankles]
        errors = ankle_errors.get(image_id, [])
        audit_rows.append({
            "image_id": image_id,
            "file_name": selection["file_name"],
            "side": selection["side"],
            "condition": selection["condition"],
            "target_ambiguous": selection["needs_manual_target_review"].lower(),
            "candidate_instance_count": selection["candidate_instance_count"],
            "usable_joint_count": sum(row["reference_usable"].lower() == "true" for row in points.values()),
            "min_joint_confidence": min(confidences),
            "min_ankle_confidence": min(ankle_confidences),
            "usable_ankle_count": sum(row["reference_usable"].lower() == "true" for row in ankles),
            "comparison_ankle_sample_count": len(errors),
            "comparison_ankle_error_median_px": median(errors) if errors else "",
        })

    reasons: dict[str, set[str]] = {}
    for row in audit_rows:
        if row["target_ambiguous"] == "true":
            add_reason(reasons, row["image_id"], "target_ambiguous")

    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in audit_rows:
        strata[(row["condition"], row["side"])].append(row)
    for rows in strata.values():
        for row in sorted(rows, key=lambda item: float(item["min_ankle_confidence"]))[:2]:
            add_reason(reasons, row["image_id"], "low_ankle_confidence")
        with_error = [row for row in rows if row["comparison_ankle_error_median_px"] != ""]
        for row in sorted(with_error, key=lambda item: float(item["comparison_ankle_error_median_px"]), reverse=True)[:1]:
            add_reason(reasons, row["image_id"], "large_model_disagreement")

    output.mkdir(parents=True)
    audit_fields = list(audit_rows[0].keys())
    write_csv(output / "all_120_image_audit_manifest.csv", audit_rows, audit_fields)
    selected_rows = []
    rendered = []
    row_by_id = {row["image_id"]: row for row in audit_rows}
    for image_id in sorted(reasons):
        row = row_by_id[image_id]
        reason = "+".join(sorted(reasons[image_id]))
        row_out = dict(row)
        row_out["selection_reason"] = reason
        selected_rows.append(row_out)
        rotation_dir = "left_ccw90" if row["side"] == "left" else "right_cw90"
        source = args.upright_input_root.resolve() / rotation_dir / row["file_name"]
        destination = output / "review_images" / f"{image_id}.jpg"
        render_overlay(source, destination, row, points_by_id[image_id], reason)
        rendered.append(destination)
    write_csv(output / "review_selection.csv", selected_rows, audit_fields + ["selection_reason"])
    make_contact_sheets(rendered, output / "contact_sheets")
    print(f"all_images={len(audit_rows)}")
    print(f"review_images={len(selected_rows)}")
    print(f"contact_sheets={(len(rendered) + 3) // 4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
