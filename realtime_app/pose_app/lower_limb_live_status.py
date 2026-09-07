"""Append-only online observations for the already accepted stereo stream.

This module is deliberately downstream-only: it cannot change detection,
association, or triangulation.  The complete T1--T4 archive is still built at
the end of a run; this writer makes the current direct-observation state
available while a run is in progress.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .fixed_coordinate import RigidTransform, load_coordinate_transform, transform_trajectory_record
from .gait_candidates import derive_gait_candidates
from .lower_limb_kinematics import derive_frame_kinematics
from .lower_limb_trajectory import LOWER_LIMB_JOINTS, normalize_stereo_record


class LowerLimbLiveStatusWriter:
    """Write one downstream status record for every completed stereo pair."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        coordinate_transform_path: str | Path | None = None,
        allow_test_coordinate_transform: bool = False,
    ) -> None:
        self.output_path = Path(output_path).resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output_path.open("w", encoding="utf-8", buffering=1)
        self._last_pair_id: int | None = None
        self._kinematics_tail: list[dict[str, Any]] = []
        self._pairs = 0
        self._direct_point_count = 0
        self._events: list[dict[str, Any]] = []

        self.transform: RigidTransform | None = None
        if coordinate_transform_path is not None:
            self.transform = load_coordinate_transform(
                coordinate_transform_path,
                allow_test_transform=allow_test_coordinate_transform,
            )

    def consume(self, stereo_record: Mapping[str, Any]) -> dict[str, Any]:
        """Consume one just-written stereo result and append its live status."""

        trajectory = normalize_stereo_record(dict(stereo_record))
        pair_id = int(trajectory["pair_id"])
        if self._last_pair_id is not None and pair_id <= self._last_pair_id:
            raise ValueError("live lower-limb status requires strictly increasing pair_id values")
        self._last_pair_id = pair_id

        kinematics = derive_frame_kinematics(trajectory)
        self._kinematics_tail.append(kinematics)
        self._kinematics_tail = self._kinematics_tail[-3:]
        newly_confirmed = (
            derive_gait_candidates(self._kinematics_tail)["events"]
            if len(self._kinematics_tail) == 3
            else []
        )
        self._events.extend(newly_confirmed)

        observed = [
            name
            for _, name in LOWER_LIMB_JOINTS
            if trajectory["points"][name]["observed_3d"]
        ]
        missing_or_rejected = [
            {
                "joint_name": name,
                "reason": trajectory["points"][name]["reason"],
            }
            for _, name in LOWER_LIMB_JOINTS
            if not trajectory["points"][name]["observed_3d"]
        ]
        t3: dict[str, Any]
        if self.transform is None:
            t3 = {
                "status": "not_configured",
                "reason": "no_coordinate_transform_supplied",
            }
        else:
            transformed = transform_trajectory_record(trajectory, self.transform)
            t3 = {
                "status": self.transform.status,
                "transform_id": self.transform.transform_id,
                "target_coordinate_frame": self.transform.target_frame,
                "points": transformed["points"],
            }

        output = {
            "schema": "lower_limb_live_status_v1",
            "pair_id": pair_id,
            "pair_timestamp_sec": trajectory["pair_timestamp_sec"],
            "source_observation_policy": "direct_accepted_stereo_only",
            "t1_trajectory": {
                "frame_status": trajectory["frame_status"],
                "direct_observed_joint_names": observed,
                "missing_or_rejected_joints": missing_or_rejected,
            },
            "t2_kinematics": kinematics["metrics"],
            "t3_coordinates": t3,
            "t4_noncontact_candidates": {
                "newly_confirmed_candidates": newly_confirmed,
                "accepted_contact_events": 0,
                "interpretation": (
                    "A candidate is emitted only after its following pair arrives; it is not a "
                    "contact, support, swing, step, or gait-cycle label."
                ),
            },
            "interpretation": (
                "Online downstream observation only. It neither changes nor validates the "
                "upstream 2-D or 3-D result."
            ),
        }
        self._handle.write(json.dumps(output, ensure_ascii=False, allow_nan=False) + "\n")
        self._pairs += 1
        self._direct_point_count += len(observed)
        return output

    def close(self, *, completed: bool) -> dict[str, Any]:
        """Close the JSONL and write a compact status summary exactly once."""

        if self._handle is None:
            raise RuntimeError("live lower-limb status writer is already closed")
        self._handle.close()
        self._handle = None
        summary = {
            "status": "complete" if completed else "interrupted_or_failed",
            "pairs": self._pairs,
            "last_pair_id": self._last_pair_id,
            "direct_observed_lower_limb_points": self._direct_point_count,
            "newly_confirmed_noncontact_candidates": len(self._events),
            "accepted_contact_events": 0,
            "coordinate_transform_status": (
                "not_configured" if self.transform is None else self.transform.status
            ),
            "output_jsonl": str(self.output_path),
            "interpretation": (
                "Append-only online state only. The end-of-run T1--T4 archive remains "
                "the complete per-run record; no gait parameter or accuracy claim is made."
            ),
        }
        summary_path = self.output_path.with_name("lower_limb_live_status_summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return summary
