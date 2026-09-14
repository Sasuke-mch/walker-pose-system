"""Tests for the strict-association lower-foot observation archive."""

from tools.build_sapiens2_lower_foot_observation_sequence import (
    expected_association,
    point_record,
)


def test_expected_association_does_not_select_between_multiple_people() -> None:
    empty = expected_association({"persons_3d": []})
    assert empty["status"] == "no_accepted_stereo_person"
    try:
        expected_association({"file_name": "pair.png", "persons_3d": [{}, {}]})
    except RuntimeError as error:
        assert "at most one" in str(error)
    else:
        raise AssertionError("multiple people must be rejected")


def test_point_record_keeps_invalid_observation_missing() -> None:
    accepted = point_record(
        {"valid": True, "xyz": [1.0, 2.0, 3.0], "reason": None}, coordinate_key="xyz"
    )
    rejected = point_record(
        {"valid_at_reprojection_gate": False, "xyz_left_camera": [1.0, 2.0, 3.0], "reason": "high_reprojection_error"},
        coordinate_key="xyz_left_camera",
    )
    assert accepted["valid"] is True and accepted["xyz_left_camera_mm"] == [1.0, 2.0, 3.0]
    assert rejected["valid"] is False and rejected["xyz_left_camera_mm"] is None
    assert rejected["reason"] == "high_reprojection_error"
