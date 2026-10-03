"""Validate selected-pair host timestamps without treating them as exposure sync."""

import json
import math
from pathlib import Path


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
