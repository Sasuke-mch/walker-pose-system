"""Unit tests for association-inherited Sapiens2 distal-foot replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.evaluate_sapiens2_foot_stereo import (
    FOOT_POINTS,
    load_association,
    select_instance,
    unassociated_points,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_association_loader_preserves_accepted_ids_and_missing_state(tmp_path: Path) -> None:
    path = tmp_path / "association.jsonl"
    _write_jsonl(
        path,
        [
            {
                "file_name": "accepted.png",
                "persons_3d": [
                    {
                        "left_person_id": 1,
                        "right_person_id": 0,
                        "association_cost": 0.15,
                        "common_keypoints": 12,
                    }
                ],
            },
            {"file_name": "rejected.png", "persons_3d": []},
        ],
    )

    association = load_association(path)

    assert association["accepted.png"] == {
        "status": "accepted_upstream_association",
        "left_person_id": 1,
        "right_person_id": 0,
        "association_cost": 0.15,
        "common_keypoints": 12,
    }
    assert association["rejected.png"]["status"] == "no_accepted_stereo_person"
    assert association["rejected.png"]["left_person_id"] is None


def test_explicit_association_id_overrides_independent_score_selection() -> None:
    instances = [
        {"bbox_score_from_yolo26x": 0.1, "keypoints308": []},
        {"bbox_score_from_yolo26x": 0.9, "keypoints308": []},
    ]
    image = {"file_name": "pair.png", "instances": instances}

    assert select_instance(image, selected_person_id=0)[0] is instances[0]
    assert select_instance(image)[0] is instances[1]
    with pytest.raises(RuntimeError, match="not present"):
        select_instance(image, selected_person_id=2)


def test_unassociated_points_cannot_become_triangulation_evidence() -> None:
    points = unassociated_points()

    assert len(points) == len(FOOT_POINTS)
    assert set(points) == set(FOOT_POINTS.values())
    assert all(point["valid_at_reprojection_gate"] is False for point in points.values())
    assert all(point["reason"] == "no_accepted_stereo_person" for point in points.values())
    assert all(point["xyz_left_camera"] is None for point in points.values())
