"""R1 primary auxiliary resources; no models, checkpoints or training data."""
import argparse
import io
import json
import pickle
import tarfile
import time
import traceback
from pathlib import Path, PurePosixPath

import numpy as np
import requests
import torch
from smplx.lbs import vertices2joints

from ..interfaces import SixDLayout, decode_rotation
from .audit_mesh_resources import mesh_resource_audit
from .check_interfaces import original_function
from .run_original import ROOT

MAX_FILE = 8 * 1024 * 1024
MAX_STREAM = 16 * 1024 * 1024
TIMEOUT = 20
BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")


class ByteLimitError(OSError):
    pass


class LimitedReader:
    def __init__(self, source, maximum=MAX_STREAM):
        self.source = source
        self.maximum = maximum
        self.consumed = 0
        self.begin = time.monotonic()

    def read(self, size=-1):
        if time.monotonic() - self.begin > TIMEOUT:
            raise TimeoutError("bounded resource stream exceeded total20s")
        remaining = self.maximum - self.consumed
        if remaining <= 0:
            raise ByteLimitError("resource stream byte limit reached")
        if size < 0:
            size = remaining
        data = self.source.read(min(size, remaining))
        self.consumed += len(data)
        if time.monotonic() - self.begin > TIMEOUT:
            raise TimeoutError("bounded resource stream exceeded total20s during read")
        return data


class ArrayUnpickler(pickle.Unpickler):
    """Only NumPy numeric-array globals for the official regressor pickle."""
    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"), ("numpy", "dtype"),
                   ("numpy.core.multiarray", "_reconstruct"),
                   ("numpy._core.multiarray", "_reconstruct"),
                   ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}
        if (module, name) not in allowed:
            raise ValueError(f"non-array pickle global rejected: {module}.{name}")
        return super().find_class(module, name)


def normalized_member(member):
    name = member.name
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("unsafe archive member name")
    if not member.isfile() or member.size < 1 or member.size > MAX_FILE:
        raise ValueError("selected resource must be a bounded regular file")
    return path.as_posix()


def numeric_array(value, shape):
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(f"original resource requires finite floating array {shape}")
    return value


def validate_resource(data, kind, rows=None):
    if kind == "mean":
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            for name, shape in (("pose", (144,)), ("shape", (10,)), ("cam", (3,))):
                numeric_array(archive[name], shape)
            pose = torch.tensor(archive["pose"].reshape(24, 6), dtype=torch.float64)
            original = original_function(ROOT / "HMR2/hmr2/utils/geometry.py", "rot6d_to_rotmat")(pose)
            adapted, valid = decode_rotation(pose, SixDLayout.PACKED_COLUMNS)
            first, second = pose[:, :3], pose[:, 3:]
            axis = first / first.norm(dim=-1, keepdim=True)
            perpendicular = second - (second * axis).sum(-1, keepdim=True) * axis
            identity = torch.eye(3, dtype=original.dtype)
            return {"kind": kind, "keys": archive.files,
                    "shapes": {key: list(archive[key].shape) for key in ("pose", "shape", "cam")},
                    "rotation_layout": "HMR2 published packed-column decoder; raw file unchanged",
                    "HMR2_mean_loader": "hmr2/models/heads/smpl_head.py:49-55; ProHMR FCHead only uses shape/cam",
                    "mean_pose_orthogonal_residual_min": float(perpendicular.norm(dim=-1).min()),
                    "existing_project_codec_invalid_init_nodes": (~valid).nonzero().flatten().tolist(),
                    "original_decoder_orthogonality_max_error": float((original.transpose(-1, -2) @ original - identity).abs().max()),
                    "original_decoder_min_determinant": float(torch.linalg.det(original).min()),
                    "valid_nodes_original_vs_adapter_max_error": float((original[valid] - adapted[valid]).abs().max()),
                    "boundary": "A learned head initialization file, not a direct pose observation. Original F.normalize eps differs from project degeneracy gate; neither file nor gate changed."}
    if kind == "pickle_regressor":
        matrix = ArrayUnpickler(io.BytesIO(data), encoding="latin1").load()
    elif kind == "npy_regressor":
        matrix = np.load(io.BytesIO(data), allow_pickle=False)
    else:
        raise ValueError("unknown explicit auxiliary resource kind")
    numeric_array(matrix, (rows, 6890))
    # Existing official vertices2joints on artificial vertices; no mesh/truth claim.
    random = np.random.default_rng(113)
    vertices = random.normal(0, .1, (2, 6890, 3))
    shift = np.array([.3, -.2, .7])
    tensor = torch.tensor(matrix, dtype=torch.float64)
    output = vertices2joints(tensor, torch.tensor(vertices))
    expected = np.einsum("jv,bvc->bjc", matrix, vertices)
    difference = float(np.max(np.abs(output.numpy() - expected)))
    shifted = vertices2joints(tensor, torch.tensor(vertices + shift))
    translation = matrix.sum(-1)[None, :, None] * shift
    translation_error = float(np.max(np.abs((shifted - output).numpy() - translation)))
    assert max(difference, translation_error) < 1e-10
    probe = torch.tensor(vertices, requires_grad=True)
    vertices2joints(tensor, probe).square().mean().backward()
    assert torch.isfinite(probe.grad).all() and probe.grad.abs().sum() > 0
    return {"kind": kind, "shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "negative_weight_count": int((matrix < 0).sum()),
            "row_sum_min": float(matrix.sum(-1).min()), "row_sum_max": float(matrix.sum(-1).max()),
            "original_vertices2joints_vs_independent_einsum_max_error": difference,
            "linear_translation_response_max_error": translation_error,
            "artificial_vertex_gradient_finite_nonzero": True,
            "boundary": "resource/linear map check only; negative weights allowed, no real joint accuracy"}


def install_resource(data, destination, kind, rows):
    report = validate_resource(data, kind, rows)
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != data:
            raise FileExistsError("different existing resource; never overwrite")
        state = "existing_exact_bytes"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(data)
        state = "new_original_resource"
    return dict(report, path=destination.as_posix(), bytes=len(data), installation=state)


def fetch_small(url, destination, kind, rows):
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        if int(response.headers.get("Content-Length", "0")) > MAX_FILE:
            raise ByteLimitError("raw auxiliary file declared oversized")
        reader = LimitedReader(response.raw, MAX_FILE)
        data = reader.read()
        # Refuse silent truncation of unknown-length resources.
        if len(data) == MAX_FILE:
            raise ByteLimitError("raw auxiliary file at byte boundary")
        result = install_resource(data, destination, kind, rows)
        return dict(result, url=url, transferred_bytes=reader.consumed)


def fetch_archive(url, targets):
    inventory, installed = [], []
    acquired = set()
    if not targets:
        raise ValueError("explicit nonempty resource allowlist required")
    reader = None
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()
            reader = LimitedReader(response.raw)
            with tarfile.open(fileobj=reader, mode="r|*") as archive:
                for member in archive:
                    name = member.name
                    while name.startswith("./"):
                        name = name[2:]
                    inventory.append({"name": name, "bytes": member.size, "regular": member.isfile()})
                    if name in targets:
                        checked = normalized_member(member)
                        if checked in acquired:
                            raise ValueError(f"duplicate selected archive member: {checked}")
                        destination, kind, rows = targets[checked]
                        content = archive.extractfile(member).read(MAX_FILE + 1)
                        if len(content) != member.size or len(content) > MAX_FILE:
                            raise ByteLimitError("selected member size inconsistent")
                        installed.append(dict(install_resource(content, destination, kind, rows), archive_member=checked))
                        acquired.add(checked)
                        if acquired == set(targets):
                            break
                    elif member.isfile() and member.size > MAX_FILE:
                        raise ByteLimitError("stop before consuming unrequested large member")
        status = "all_selected_acquired" if acquired == set(targets) else "archive_eof_missing_selected"
        error = None
    except (OSError, ValueError, requests.RequestException, tarfile.TarError) as exception:
        status, error = "bounded_archive_stopped", f"{type(exception).__name__}: {exception}"
    return {"url": url, "status": status, "error": error, "inventory": inventory,
            "wanted_members": sorted(targets), "acquired_members": sorted(acquired),
            "missing_members": sorted(set(targets) - acquired),
            "all_selected_acquired": acquired == set(targets),
            "installed": installed, "transferred_bytes": reader.consumed if reader is not None else 0,
            "boundary": "stopped before large members or byte/time cap; unvisited members not proved absent"}


def check_network_budget():
    """Count retained acquisition evidence before allowing another network run."""
    counts = {}
    for path in BASE.glob("architecture_resources_results_v*.json"):
        previous = json.loads(path.read_text(encoding="utf-8"))
        for item in previous.get("raw_attempts", []) + previous.get("archive_attempts", []):
            url = item["url"]
            counts[url] = counts.get(url, 0) + 1
    if any(count >= 2 for count in counts.values()):
        raise RuntimeError("R1 retained network-attempt budget exhausted; use --local-only")


def run():
    check_network_budget()
    before = mesh_resource_audit()
    attempts = []
    human = ROOT / "HuManiFlow/model_files"
    for name, rows in (("J_regressor_extra.npy", 9), ("J_regressor_h36m.npy", 17), ("cocoplus_regressor.npy", 19)):
        url = "https://raw.githubusercontent.com/akashsengupta1997/HuManiFlow/main/model_files/" + name
        try:
            result = dict(fetch_small(url, human / name, "npy_regressor", rows), status="verified_original_auxiliary")
        except (OSError, ValueError, requests.RequestException) as error:
            result = {"url": url, "status": "resource_failed", "error": f"{type(error).__name__}: {error}"}
        attempts.append(result)
        print(json.dumps(result), flush=True)
    archives = []
    for url, root, prefix in (
            ("https://visiondata.cis.upenn.edu/prohmr/data.tar.gz", ROOT / "ProHMR/data", "data/"),
            ("https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz", Path.home() / ".cache/4DHumans/data", "data/")):
        targets = {prefix + "smpl_mean_params.npz": (root / "smpl_mean_params.npz", "mean", None),
                   prefix + "SMPL_to_J19.pkl": (root / "SMPL_to_J19.pkl", "pickle_regressor", 19)}
        result = fetch_archive(url, targets)
        archives.append(result)
        print(json.dumps(result), flush=True)
    # Source audit records the independently observed Drive transport failure,
    # not a resource absence verdict. No model/account downloads are attempted.
    after = mesh_resource_audit()
    return {"status": "auxiliary_acquisition_and_default_gap_audit_completed", "protocol": "architecture_resources_protocol.json",
            "training_executed": False, "original_mesh_forward_executed": False,
            "raw_attempts": attempts, "archive_attempts": archives,
            "before_missing": sum(not r["available"] for p in before["author_mesh_paths"] for r in p["requirements"]),
            "after_missing": sum(not r["available"] for p in after["author_mesh_paths"] for r in p["requirements"]),
            "after_default_audit": after, "neutral_models_acquired": False,
            "boundary": "Small original resources validated only; default neutral/license, full original input and mesh/evaluation remain separate"}


def verify_local_resources():
    """Recheck acquired bytes against retained records without network requests."""
    previous_path = BASE / "architecture_resources_results_v2.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    records = previous["raw_attempts"] + [
        item for archive in previous["archive_attempts"] for item in archive["installed"]]
    verified = []
    for item in records:
        path = Path(item["path"])
        data = path.read_bytes()
        if len(data) != item["bytes"]:
            raise ValueError(f"retained resource byte-count mismatch: {path}")
        rows = item.get("shape", [None])[0]
        report = validate_resource(data, item["kind"], rows)
        verified.append(dict(report, path=path.as_posix(), bytes=len(data),
                             provenance_record=previous_path.as_posix()))
    means = [ROOT / "ProHMR/data/smpl_mean_params.npz",
             Path.home() / ".cache/4DHumans/data/smpl_mean_params.npz"]
    fixture = ROOT / "smpl_mean_params_mmpose.npz"
    comparisons = []
    for path in means:
        with np.load(path, allow_pickle=False) as official, np.load(fixture, allow_pickle=False) as old:
            keys_equal = set(official.files) == set(old.files)
            arrays_equal = keys_equal and all(
                official[key].dtype == old[key].dtype and np.array_equal(official[key], old[key])
                for key in official.files)
        comparisons.append({"official_path": path.as_posix(), "fixture_path": fixture.as_posix(),
                            "keys_equal": keys_equal, "arrays_and_dtypes_equal": arrays_equal,
                            "raw_bytes_equal": path.read_bytes() == fixture.read_bytes()})
    historical = [{"source": "user-supplied takeover attachment, R1 historical WHAM preflight",
                   "operation": "HEAD redirect before this takeover; not repeated in local verification",
                   "outcome": "12s ReadTimeout on drive.usercontent.google.com",
                   "archive_body_downloaded": False,
                   "boundary": "Historical report, not current network evidence or absence/license verdict"}]
    return {"status": "local_auxiliary_resources_verified", "protocol": "architecture_resources_protocol.json",
            "network_requests_executed": 0, "training_executed": False,
            "original_mesh_forward_executed": False, "verified": verified,
            "specific_official_mean_fixture_comparison": comparisons,
            "historical_preflight_records": historical,
            "retained_acquisition_records": ["architecture_resources_results_v1.json", "architecture_resources_results_v2.json"],
            "boundary": "Local auxiliary interfaces only; no original mesh/full-method or accuracy validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain prior acquisition evidence; choose new output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = verify_local_resources() if args.local_only else run()
    except Exception as error:  # noqa: BLE001 -- preserve resource-stage failure
        result = {"status": "resource_acquisition_stage_failed", "training_executed": False,
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
