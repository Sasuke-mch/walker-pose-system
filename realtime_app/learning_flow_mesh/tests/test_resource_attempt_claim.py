"""A consumed network attempt must survive crashes and concurrent claims."""
import json

import pytest

from realtime_app.learning_flow_mesh.reproduction.check_wham_resources import (
    claim_attempt,
)


def test_crash_after_claim_refuses_retry_without_overwriting_evidence(tmp_path):
    path = tmp_path / "attempt.json"
    claim_attempt(path, "https://example.test/original")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        claim_attempt(path, "https://example.test/other")
    assert path.read_bytes() == before
    assert json.loads(before)["status"] == "attempt_claimed_before_GET"


def test_preexisting_non_json_record_still_blocks_attempt(tmp_path):
    path = tmp_path / "attempt.json"
    path.write_text("partial process record", encoding="utf-8")
    with pytest.raises(FileExistsError):
        claim_attempt(path, "https://example.test/original")
    assert path.read_text(encoding="utf-8") == "partial process record"
