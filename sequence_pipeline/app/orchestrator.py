from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LoadedConfig
from .images import scan_images, write_manifest
from .launcher import ModelLauncher, StageResult
from .normalize import normalize_raw
from .preflight import run_preflight
from .utils import write_json


def parse_models(config: LoadedConfig, raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return [k for k in config.model_order if config.model(k).get("enabled", True)]
    selected = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [x for x in selected if x not in config.models or x == "yolo26_detector"]
    if unknown:
        raise ValueError("Unknown or non-selectable model(s): " + ", ".join(unknown))
    return selected


def _write_stage_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_dir / "run_summary.json", {"schema_version": "1.0", "stages": results})
    with (summary_dir / "stages.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "status", "message", "outputs"])
        writer.writeheader()
        for item in results:
            writer.writerow({
                "stage": item["stage"],
                "status": item["status"],
                "message": item["message"],
                "outputs": " | ".join(item.get("outputs", [])),
            })


def run_pipeline(*, config: LoadedConfig, input_dir: Path, output_dir: Path, selected: list[str], weight_overrides: dict[str, str], options: dict[str, Any]) -> int:
    images = scan_images(input_dir)
    if not images:
        raise RuntimeError(f"No supported images found in {input_dir}")

    report = run_preflight(config, selected, weight_overrides, check_docker=not options.get("dry_run"), input_dir=input_dir)
    report.print()
    if report.errors:
        print("\nPreflight failed. No model was started.")
        return 2

    run_name = options.get("run_name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    if run_dir.exists() and not options.get("overwrite") and not options.get("skip_existing"):
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "summary", "detections"):
        (run_dir / name).mkdir(exist_ok=True)

    write_manifest(input_dir, images, run_dir / "manifest.json")
    write_json(run_dir / "resolved_config.json", {"project_root": str(config.project_root), "selected_models": selected, "defaults": config.defaults, "models": {k: config.model(k) for k in (["yolo26_detector"] + selected)}, "weight_overrides": weight_overrides, "runtime_options": options})


    stages = []
    if any(config.model(k).get("requires_detector") for k in selected):
        stages.append("yolo26_detector")
    stages.extend(selected)

    context = {
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "device": options.get("device") or config.defaults.get("device", "0"),
        "imgsz": options.get("imgsz") or config.defaults.get("imgsz", 1280),
        "det_conf": options.get("det_conf") if options.get("det_conf") is not None else config.defaults.get("det_conf", 0.05),
        "pose_conf": options.get("pose_conf") if options.get("pose_conf") is not None else config.defaults.get("pose_conf", 0.05),
        "det_iou": options.get("det_iou") if options.get("det_iou") is not None else config.defaults.get("det_iou", 0.70),
        "probpose_score_thr": config.defaults.get("probpose_score_thr", 0.20),
        "sapiens_batch_size": config.defaults.get("sapiens_batch_size", 2),
        "sapiens_kpt_thr": config.defaults.get("sapiens_kpt_thr", 0.30),
        "save_vis": bool(options.get("save_vis")),
        "dry_run": bool(options.get("dry_run")),
        "overwrite": bool(options.get("overwrite")),
        "skip_existing": bool(options.get("skip_existing")),
    }

    records: list[dict[str, Any]] = []
    for stage in stages:
        print(f"\n========== {stage} ==========")
        launcher = ModelLauncher(config, stage, context, weight_overrides.get(stage))
        result = launcher.run()
        rec = {"stage": stage, "status": result.status, "message": result.message, "outputs": [str(p) for p in result.expected_outputs], "command": result.command}
        records.append(rec)
        _write_stage_summary(run_dir, records)

        if result.status == "success" and not options.get("save_vis"):
            vis_dir = run_dir / stage / "visualizations"
            if vis_dir.exists():
                shutil.rmtree(vis_dir)

        if result.status == "success" and stage not in ("yolo26_detector", "yolo26x_pose"):
            raw = run_dir / stage / "raw_predictions.json"
            common = run_dir / stage / "common_predictions.json"
            if raw.exists():
                ok, message = normalize_raw(raw, common, stage, "YOLO26x", input_dir)
                rec["normalization"] = {"success": ok, "message": message, "output": str(common) if ok else None}
                _write_stage_summary(run_dir, records)

        if result.status == "failed" and not options.get("continue_on_error"):
            print("Stopping because a stage failed. Use --continue-on-error to continue.")
            return 1

    print(f"\nRun directory: {run_dir}")
    print(f"Input images: {len(images)}")
    failures = [r for r in records if r["status"] == "failed"]
    print(f"Failed stages: {len(failures)}")
    return 1 if failures else 0
