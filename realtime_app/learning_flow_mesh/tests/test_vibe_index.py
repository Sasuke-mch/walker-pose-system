"""R11 no-fullarchive fallback and exact suffix boundaries."""
import pytest

from realtime_app.learning_flow_mesh.reproduction.check_vibe_index import validate_range


def test_whole_archive_or_html_status_rejected_before_body_read():
    for status in (200, 301, 403, 404, 500):
        with pytest.raises(ValueError, match="no fullzip fallback"):
            validate_range(status, {})


def test_bounded_suffix_range_and_archive_endpoint_required():
    assert validate_range(206, {"Content-Range": "bytes 34464-99999/100000"}) == (34464, 99999, 100000)
    assert validate_range(206, {"Content-Range": "bytes 0-9/10"}) == (0, 9, 10)
    for value in ("bytes 0-65536/65537", "bytes 10-19/21", "bytes 9-8/9", "bytes 0-9/*", ""):
        with pytest.raises(ValueError):
            validate_range(206, {"Content-Range": value})


def test_encoded_transport_rejected():
    with pytest.raises(ValueError, match="encoded"):
        validate_range(206, {"Content-Range": "bytes 0-9/10", "Content-Encoding": "gzip"})
