"""Validate selected-pair host timestamps without treating them as exposure sync."""

import csv
import json
import math
from pathlib import Path

import numpy as np


def load_capture_pair_times(
    input_pairs: Path, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Read selected capture pairs and return host-return midpoint times.

    Shared by the causal stereo baseline and the PHC/RL/viewer consumers. The
    ``selection_manifest.csv`` is checked against the capture ``stereo_pairs.csv``
    by frame identity; both cameras must increase strictly; the integer
    nanosecond origin is subtracted before converting to float, then each
    midpoint is expressed in seconds. The returned ``skew`` is
    ``right - left`` in milliseconds and reflects host read-return timing, not
    exposure sync. This is a separate adapter from :func:`load_pair_timing`,
    which reads a different JSON input; the file format and return structure of
    the two are intentionally not unified.
    """
    with (input_pairs / "selection_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        selection = list(csv.DictReader(stream))
    if len(selection) != count or [int(row["sequence_index"]) for row in selection] != list(range(count)):
        raise ValueError("selection manifest does not match prediction frame count")
    metadata = json.loads((input_pairs / "metadata.json").read_text(encoding="utf-8"))
    capture_path = Path(metadata["source_session"]) / "stereo_pairs.csv"
    capture_metadata = json.loads((capture_path.parent / "metadata.json").read_text(encoding="utf-8"))
    if capture_metadata["timestamp_type"] != "host_read_return_perf_counter_ns":
        raise ValueError("unexpected capture timestamp definition")
    with capture_path.open(encoding="utf-8-sig", newline="") as stream:
        capture = list(csv.DictReader(stream))
    left_ns, right_ns, pair_ids = [], [], []
    for row in selection:
        source = capture[int(row["source_pair_index"])]
        if any(row[f"{side}_frame_id"] != source[f"{side}_frame_id"] for side in ("left", "right")):
            raise ValueError("selected frame does not match capture identity")
        left_ns.append(int(source["left_host_return_timestamp_ns"]))
        right_ns.append(int(source["right_host_return_timestamp_ns"]))
        pair_ids.append(int(source["pair_id"]))
    left, right = np.asarray(left_ns, dtype=np.int64), np.asarray(right_ns, dtype=np.int64)
    if np.any(np.diff(left) <= 0) or np.any(np.diff(right) <= 0):
        raise ValueError("host-return timestamps must increase strictly")
    # Subtract the integer origin before float conversion to preserve ns differences.
    times = (left + right - int(left[0] + right[0])) / 2e9
    return times, (right - left) / 1e6, np.asarray(pair_ids), capture_path


def load_pair_timing(path, count, *, input_pair_dir=None):
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(rows) != count or [r["index"] for r in rows] != list(range(count)):
        raise ValueError("timestamp frame identities differ from observations")
    values = []
    previous = None
    for row in rows:
        if input_pair_dir is not None:
            for side, folder in [("left", "left_ccw90"), ("right", "right_cw90")]:
                expected = (
                    input_pair_dir / folder / f"pair_{row['index']:04d}.png"
                ).resolve()
                if Path(row[f"{side}_image"]).resolve() != expected:
                    raise ValueError(
                        "timestamp image identity differs from selected replay image"
                    )
        left = float(row["left_host_return_timestamp_ns"]) * 1e-9
        right = float(row["right_host_return_timestamp_ns"]) * 1e-9
        if not math.isfinite(left + right) or (
            previous is not None and (left <= previous[0] or right <= previous[1])
        ):
            raise ValueError(
                "host timestamps must be finite and strictly increasing on both cameras"
            )
        if (
            abs(
                (right - left) * 1000
                - float(row["signed_host_delta_ms_right_minus_left"])
            )
            > 1e-4
        ):
            raise ValueError("timestamp skew metadata mismatch")
        values.append((left, right))
        previous = (left, right)
    origin = sum(values[0]) / 2
    return [(l - origin, r - origin) for l, r in values]
