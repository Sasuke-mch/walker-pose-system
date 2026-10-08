"""Resource bounds, no overwrite, exact original schemas and linear semantics."""
import io
import pickle
import tarfile

import numpy as np
import pytest

from realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources import (
    MAX_FILE,
    ArrayUnpickler,
    ByteLimitError,
    LimitedReader,
    check_network_budget,
    fetch_archive,
    fetch_small,
    install_resource,
    normalized_member,
    validate_resource,
    verify_local_resources,
)


def array_bytes(matrix):
    stream = io.BytesIO()
    np.save(stream, matrix, allow_pickle=False)
    return stream.getvalue()


def matrix():
    value = np.zeros((9, 6890), np.float64)
    value[:, 0], value[:, 1] = -.2, 1.2
    return value


def test_byte_bound_and_deadline(monkeypatch):
    source = io.BytesIO(b"abcdef")
    reader = LimitedReader(source, 5)
    assert reader.read(2) == b"ab" and reader.read(100) == b"cde"
    with pytest.raises(ByteLimitError):
        reader.read(1)
    assert source.tell() == 5
    other = LimitedReader(io.BytesIO(b"x"))
    monkeypatch.setattr("realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources.time.monotonic",
                        lambda: other.begin + 21)
    with pytest.raises(TimeoutError):
        other.read(1)


@pytest.mark.parametrize("name", ["../x", "/absolute", "data/../../x", "C:/x", "data\\x"])
def test_selected_archive_member_traversal_rejected(name):
    member = tarfile.TarInfo(name)
    member.size = 1
    with pytest.raises(ValueError):
        normalized_member(member)


def test_selected_member_requires_regular_bounded_file():
    member = tarfile.TarInfo("./data/example.npy")
    member.size = 1
    assert normalized_member(member) == "data/example.npy"
    for kind, size in ((tarfile.SYMTYPE, 1), (tarfile.DIRTYPE, 1), (tarfile.REGTYPE, MAX_FILE + 1)):
        member.type, member.size = kind, size
        with pytest.raises(ValueError):
            normalized_member(member)


def test_original_regressor_linear_map_allows_negative_weights():
    result = validate_resource(array_bytes(matrix()), "npy_regressor", 9)
    assert result["negative_weight_count"] == 9
    assert result["row_sum_min"] == 1
    assert result["artificial_vertex_gradient_finite_nonzero"]
    assert result["linear_translation_response_max_error"] < 1e-10


@pytest.mark.parametrize("fault", ["shape", "int", "nan"])
def test_invalid_regressor_schema_rejected(fault):
    value = matrix()
    if fault == "shape":
        value = value[:8]
    elif fault == "int":
        value = value.astype(np.int64)
    else:
        value[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_resource(array_bytes(value), "npy_regressor", 9)


def test_install_existing_exact_and_different_bytes_without_overwrite(tmp_path):
    path = tmp_path / "original.npy"
    data = array_bytes(matrix())
    assert install_resource(data, path, "npy_regressor", 9)["installation"] == "new_original_resource"
    assert install_resource(data, path, "npy_regressor", 9)["installation"] == "existing_exact_bytes"
    changed = matrix()
    changed[:, 1] = 1.3
    with pytest.raises(FileExistsError):
        install_resource(array_bytes(changed), path, "npy_regressor", 9)
    assert path.read_bytes() == data


class NonArrayObject:
    def __reduce__(self):
        return str, ("should not be called by the restricted loader",)


def test_pickle_other_globals_rejected():
    with pytest.raises(ValueError, match="non-array pickle global"):
        ArrayUnpickler(io.BytesIO(pickle.dumps(NonArrayObject()))).load()


def test_mean_initialization_preserves_near_collinear_root_without_gate_relaxation():
    pose = np.tile([1., 0, 0, 0, 1, 0], (24, 1))
    pose[0] = [1, 0, 0, -1, 0, 1.5e-7]
    buffer = io.BytesIO()
    np.savez(buffer, pose=pose.reshape(144), shape=np.zeros(10), cam=np.zeros(3))
    result = validate_resource(buffer.getvalue(), "mean")
    assert result["existing_project_codec_invalid_init_nodes"] == [0]
    assert result["original_decoder_min_determinant"] > .99999
    assert result["valid_nodes_original_vs_adapter_max_error"] == 0


class Response:
    def __init__(self, data, headers=None):
        self.raw = io.BytesIO(data)
        self.headers = {} if headers is None else headers

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_archive_stops_before_large_unrequested_payload_and_retains_small_resource(monkeypatch, tmp_path):
    data = array_bytes(matrix())
    member = tarfile.TarInfo("data/J.npy")
    member.size = len(data)
    huge = tarfile.TarInfo("data/unrequested_checkpoint.bin")
    huge.size = MAX_FILE + 100
    # Valid headers; the large payload is deliberately absent and never read.
    archive = member.tobuf() + data + b"\0" * ((-len(data)) % 512) + huge.tobuf() + b"\0" * 1024
    monkeypatch.setattr("realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources.requests.get",
                        lambda *args, **kwargs: Response(archive))
    destination = tmp_path / "J.npy"
    result = fetch_archive("https://example.invalid/fixture", {
        "data/J.npy": (destination, "npy_regressor", 9),
        "data/not_yet_visited.npy": (tmp_path / "missing.npy", "npy_regressor", 9)})
    assert result["status"] == "bounded_archive_stopped"
    assert len(result["installed"]) == 1 and destination.read_bytes() == data
    assert "unrequested large member" in result["error"]
    assert not (tmp_path / "missing.npy").exists()


def test_raw_declared_oversize_never_consumed(monkeypatch, tmp_path):
    response = Response(b"x", {"Content-Length": str(MAX_FILE + 1)})
    monkeypatch.setattr("realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources.requests.get",
                        lambda *args, **kwargs: response)
    with pytest.raises(ByteLimitError):
        fetch_small("https://example.invalid/fixture", tmp_path / "unused.npy", "npy_regressor", 9)
    assert response.raw.tell() == 0 and not (tmp_path / "unused.npy").exists()


@pytest.mark.parametrize("names,expected", [
    (["data/one.npy"], "archive_eof_missing_selected"),
    (["data/one.npy", "./data/one.npy"], "bounded_archive_stopped"),
    (["././data/one.npy", "data/two.npy"], "all_selected_acquired"),
])
def test_archive_completion_counts_unique_selected_members(monkeypatch, tmp_path, names, expected):
    data = array_bytes(matrix())
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    monkeypatch.setattr("realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources.requests.get",
                        lambda *args, **kwargs: Response(buffer.getvalue()))
    result = fetch_archive("https://example.invalid/fixture", {
        "data/one.npy": (tmp_path / "one.npy", "npy_regressor", 9),
        "data/two.npy": (tmp_path / "two.npy", "npy_regressor", 9)})
    assert result["status"] == expected
    assert result["all_selected_acquired"] == (expected == "all_selected_acquired")
    assert result["missing_members"] == ([] if result["all_selected_acquired"] else ["data/two.npy"])
    if len(names) == 2 and names[-1] == "./data/one.npy":
        assert "duplicate selected" in result["error"]
        assert len(result["installed"]) == 1


def test_network_budget_rejects_third_attempt_before_any_request(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr("realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources.BASE", tmp_path)
    for version in (1, 2):
        (tmp_path / f"architecture_resources_results_v{version}.json").write_text(
            json.dumps({"raw_attempts": [{"url": "https://example.invalid/fixture"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        check_network_budget()


def test_deadline_expired_during_read_is_rejected(monkeypatch):
    import realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources as module

    ticks = iter([0, 0, 21])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    reader = LimitedReader(io.BytesIO(b"abc"))
    with pytest.raises(TimeoutError, match="during read"):
        reader.read()
    assert reader.consumed == 3


def test_retained_local_resources_never_issue_network_requests(monkeypatch):
    import realtime_app.learning_flow_mesh.reproduction.fetch_auxiliary_resources as module

    if not (module.BASE / "architecture_resources_results_v2.json").is_file():
        pytest.skip("local R1 acquisition evidence not available")

    def forbidden(*args, **kwargs):
        raise AssertionError("local verification attempted a network request")

    monkeypatch.setattr(module.requests, "get", forbidden)
    report = verify_local_resources()
    assert len(report["verified"]) == 7
    assert report["network_requests_executed"] == 0
    assert all(item["raw_bytes_equal"] and item["arrays_and_dtypes_equal"]
               for item in report["specific_official_mean_fixture_comparison"])
