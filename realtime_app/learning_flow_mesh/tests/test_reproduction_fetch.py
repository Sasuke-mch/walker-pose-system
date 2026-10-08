"""Prevent source archives/cached provenance from escaping or overwriting data."""
import io
import json
import zipfile

import pytest

from realtime_app.learning_flow_mesh.reproduction import fetch


@pytest.mark.parametrize("member", ["repo/../outside.py", "repo/a/../../outside.py", "repo/C:/outside.py", "repo/a\\outside.py"])
def test_unsafe_archive_member_rejected(member):
    with pytest.raises(ValueError, match="unsafe"):
        fetch.selected_path(member)


def test_acquire_checks_all_members_before_writing(tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("repo/source.py", "x = 1")
        z.writestr("repo/../../outside.py", "escape")
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(archive.getvalue()))
    result = fetch.acquire({"id": "X", "name": "Test", "repo": "author/repo", "branch": "main"}, tmp_path)
    assert result["status"] == "source_acquisition_failed"
    assert not (tmp_path / "Test").exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_unfinished_snapshot_is_preserved(tmp_path):
    target = tmp_path / "Test"
    target.mkdir()
    source = target / "source.py"
    source.write_text("existing user file", encoding="utf-8")
    result = fetch.acquire({"id": "X", "name": "Test", "repo": "author/repo", "branch": "main"}, tmp_path)
    assert result["status"] == "source_acquisition_failed"
    assert source.read_text() == "existing user file"


def test_missing_cached_source_is_not_reported_as_archived(tmp_path):
    target = tmp_path / "Test"
    target.mkdir()
    (target / "SOURCE_SNAPSHOT.json").write_text(json.dumps({
        "archive_url": "https://codeload.github.com/author/repo/zip/refs/heads/main",
        "files": [{"path": "missing.py", "bytes": 8}]}))
    result = fetch.acquire({"id": "X", "name": "Test", "repo": "author/repo", "branch": "main"}, tmp_path)
    assert result["status"] == "source_acquisition_failed"
    assert "incomplete" in result["error"]
