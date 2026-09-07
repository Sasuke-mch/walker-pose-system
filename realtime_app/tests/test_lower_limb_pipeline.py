from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_app.lower_limb_pipeline import build_lower_limb_pipeline


def stereo_record(pair_id: int) -> dict:
    keypoints = []
    coordinates = {
        "left_hip": [0.0, 0.0, 1000.0],
        "right_hip": [100.0, 0.0, 1000.0],
        "left_knee": [10.0 + pair_id, 0.0, 700.0],
        "right_knee": [110.0 - pair_id, 0.0, 700.0],
        "left_ankle": [0.0, 0.0, 400.0],
        "right_ankle": [100.0, 0.0, 400.0],
    }
    for index, name in ((11, "left_hip"), (12, "right_hip"), (13, "left_knee"), (14, "right_knee"), (15, "left_ankle"), (16, "right_ankle")):
        keypoints.append({"index": index, "name": name, "valid": True, "xyz": coordinates[name], "reason": None})
    return {
        "pair_id": pair_id,
        "pair_timestamp_sec": pair_id / 30.0,
        "coordinate_frame": "left_camera",
        "length_unit": "millimeter",
        "persons_3d": [{"keypoints_3d": keypoints}],
    }


class LowerLimbPipelineTests(unittest.TestCase):
    def test_builds_t1_to_t4_below_one_directory_without_target_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "stereo_results.jsonl"
            source.write_text(
                "".join(json.dumps(stereo_record(pair_id)) + "\n" for pair_id in range(3)),
                encoding="utf-8",
            )
            output = root / "lower_limb_pipeline"
            summary = build_lower_limb_pipeline(source, output)

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["frames"], 3)
            self.assertEqual(summary["stages"]["t3_coordinates"]["status"], "not_configured")
            self.assertTrue((output / "t1_trajectory" / "lower_limb_trajectory.jsonl").is_file())
            self.assertTrue((output / "t2_kinematics" / "lower_limb_kinematics.jsonl").is_file())
            self.assertTrue((output / "t3_coordinates" / "coordinate_normalization_summary.json").is_file())
            self.assertTrue((output / "t4_noncontact_candidates" / "gait_event_candidates.csv").is_file())
            self.assertTrue((output / "pipeline_metadata.json").is_file())
            parameters = json.loads(
                (output / "t4_noncontact_candidates" / "gait_parameter_candidates.json").read_text(encoding="utf-8")
            )
            self.assertFalse(any(value["available"] for value in parameters.values()))

    def test_refuses_to_overwrite_pipeline_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "stereo_results.jsonl"
            source.write_text(json.dumps(stereo_record(0)) + "\n", encoding="utf-8")
            output = root / "lower_limb_pipeline"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                build_lower_limb_pipeline(source, output)


if __name__ == "__main__":
    unittest.main()
