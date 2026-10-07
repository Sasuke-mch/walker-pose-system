"""Recreate the frozen synthetic test and label-free missing-input case."""
import argparse
import json
from pathlib import Path

import numpy as np

from .body import BodyDecoder
from .synthetic import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    decoder = BodyDecoder(**config["assets"]).to(config["device"])
    fresh = generate(decoder, 16, 4, args.seed, "test", config["device"])
    metadata = json.loads(str(fresh["metadata"].item()))
    metadata["subject_ids"] = [f"fresh_subject_{i}" for i in range(16)]
    metadata["sequence_ids"] = [f"fresh_sequence_{i}" for i in range(16)]
    metadata["evaluation_usage"] = "Generated after architecture/budgets frozen; no subsequent tuning"
    fresh["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(output / "fresh_test.npz", **fresh)
    missing = {k: value.copy() for k, value in fresh.items()
               if k not in ("rotations", "pose_mask", "betas", "shape_mask", "translation", "translation_mask", "metadata")}
    for key in ("view_mask", "observation_mask", "geometry_mask", "image_mask"):
        missing[key][:] = False
    for key in ("xy", "confidence", "ray_origins", "ray_directions", "images"):
        missing[key][:] = np.nan
    metadata["label_source"] = "none_all_missing_engineering_case"
    missing["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(output / "all_missing_input.npz", **missing)
    print(json.dumps({"output": str(output.resolve()), "seed": args.seed, "sequences": 16, "frames_each": 4}))


if __name__ == "__main__":
    main()
