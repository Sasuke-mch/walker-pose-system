import unittest

import numpy as np

from pose_app.local_ground_state import (
    ImuRelativeRotation,
    LocalGroundPlane,
    PropagationCriteria,
    SupportCriteria,
    VisualRelativePose,
    WalkerSupportTemplate,
    audit_walker_support,
    propagate_plane_with_visual_imu,
)


IDENTITY = np.eye(3).tolist()


class LocalGroundStateTests(unittest.TestCase):
    def direct_plane(self):
        return LocalGroundPlane.from_mapping(
            {
                "ground_state": {
                    "status": "direct",
                    "coordinate_frame": "left_camera",
                    "length_unit": "millimeter",
                    "plane": {
                        "normal_left_camera": [0.0, 1.0, 0.0],
                        "offset_mm": 1000.0,
                        "normal_orientation": "toward_camera",
                    },
                }
            }
        )

    def template(self):
        return WalkerSupportTemplate.from_mapping(
            {
                "schema": "walker_support_template_v1",
                "status": "measured_locked",
                "configuration_id": "height_01",
                "walker_coordinate_frame": "walker",
                "length_unit": "millimeter",
                "rotation_left_camera_from_walker": IDENTITY,
                "translation_left_camera_from_walker_mm": [0.0, 0.0, 0.0],
                "support_points_walker_mm": [
                    {"support_point_id": "a", "xyz_walker_mm": [0.0, -1000.0, 0.0]},
                    {"support_point_id": "b", "xyz_walker_mm": [100.0, -1000.0, 0.0]},
                    {"support_point_id": "c", "xyz_walker_mm": [0.0, -1000.0, 100.0]},
                ],
                "measurement_evidence": {"method": "fixture", "capture_session": "s", "evidence_id": "e"},
            }
        )

    def test_direct_contract_and_support_template_only_provide_geometry_condition(self):
        result = audit_walker_support(
            self.direct_plane(), self.template(),
            SupportCriteria.from_mapping(
                {"criterion_id": "fixture", "support_compatibility_distance_mm": 5.0, "minimum_lift_clearance_mm": 20.0}
            ),
        )
        self.assertEqual(result["status"], "support-compatible")
        self.assertIn("not a contact", result["interpretation"])

    def test_pair_zero_is_valid_and_imu_never_supplies_translation(self):
        visual = VisualRelativePose.from_mapping(
            {
                "status": "accepted_static_background", "from_pair_id": 0, "to_pair_id": 1,
                "timestamp_from_ns": 1, "timestamp_to_ns": 10_000_001,
                "rotation_current_from_previous": IDENTITY,
                "translation_current_from_previous_mm": [0.0, 10.0, 0.0],
            }
        )
        imu = ImuRelativeRotation.from_mapping(
            {
                "status": "accepted_rotation_only", "from_pair_id": 0, "to_pair_id": 1,
                "rotation_current_from_previous": IDENTITY,
                "gravity_current_left_camera_unit": [0.0, 1.0, 0.0],
                "imu_to_left_camera_extrinsics_status": "measured_locked",
                "camera_imu_time_alignment_status": "verified",
            }
        )
        state, audit = propagate_plane_with_visual_imu(
            self.direct_plane(), visual, imu,
            PropagationCriteria.from_mapping(
                {"criterion_id": "fixture", "maximum_propagation_interval_ms": 20.0,
                 "maximum_visual_imu_rotation_disagreement_deg": 1.0, "maximum_plane_gravity_angle_deg": 1.0}
            ),
        )
        self.assertEqual(state.status, "propagated")
        self.assertAlmostEqual(state.offset_mm, 990.0)
        self.assertEqual(audit["status"], "propagated")
        with self.assertRaisesRegex(ValueError, "acceleration-integrated height"):
            ImuRelativeRotation.from_mapping(
                {
                    "status": "accepted_rotation_only", "from_pair_id": 0, "to_pair_id": 1,
                    "rotation_current_from_previous": IDENTITY,
                    "gravity_current_left_camera_unit": [0.0, 1.0, 0.0],
                    "imu_to_left_camera_extrinsics_status": "measured_locked",
                    "camera_imu_time_alignment_status": "verified",
                    "height_from_acceleration_mm": 12.0,
                }
            )


if __name__ == "__main__":
    unittest.main()
