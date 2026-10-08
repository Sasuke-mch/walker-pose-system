"""R15 reject large/binary responses and repeated network attempts."""
import json

import pytest

from realtime_app.learning_flow_mesh.reproduction import (
    check_motionbert_directory as check,
)


def test_small_identity_html_headers_accepted():
    check.validate_headers(200, {"Content-Type": "text/html; charset=utf-8", "Content-Length": "12"}, 12)


@pytest.mark.parametrize("status,headers", [
    (403, {"Content-Type": "text/html"}),
    (200, {"Content-Type": "application/zip"}),
    (200, {"Content-Type": "text/html", "Content-Encoding": "gzip"}),
    (200, {"Content-Type": "text/html", "Content-Length": "404347"}),
    (200, {"Content-Type": "text/html", "Content-Length": "-1"}),
    (200, {"Content-Type": "text/html", "Content-Length": "unknown"}),
])
def test_unpermitted_headers_rejected_before_body(status, headers):
    with pytest.raises(ValueError):
        check.validate_headers(status, headers, 262144)


def test_existing_claim_stops_before_network(monkeypatch, tmp_path):
    (tmp_path / "architecture_r15_protocol.json").write_text(json.dumps({"source": "unused"}))
    (tmp_path / "architecture_r15_attempt.json").write_text("retained attempt")
    monkeypatch.setattr(check, "BASE", tmp_path)
    monkeypatch.setattr(check.requests, "Session", lambda: pytest.fail("network session created after existing claim"))
    with pytest.raises(FileExistsError):
        check.run()
    assert (tmp_path / "architecture_r15_attempt.json").read_text() == "retained attempt"
