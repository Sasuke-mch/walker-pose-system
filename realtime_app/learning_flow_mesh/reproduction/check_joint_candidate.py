"""H3: complete saved360 observation-to-mesh plumbing, fresh random weights only."""
import argparse
import copy
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.candidate import (
    CausalMeshCandidate,
    OutputStatus,
    decode_available,
    saved_contract_conditions,
)
from realtime_app.learning_flow_mesh.interfaces import JointSchema, require_joint_schema
from realtime_app.learning_flow_mesh.model import FlowMesh, ModelConfig
from realtime_app.pose_app.smpl_coco_observation import (
    load_coco17_regressor,
    regress_coco17_torch,
)

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")
INPUT = BASE / "architecture_h2_results_v3.npz"
TOLERANCE = 2e-5


def slice_batch(batch, start, end):
    return {key: value[:, start:end] for key, value in batch.items()}


def stream(candidate, batch, ids, admitted, names, size):
    pieces, state = [], None
    for start in range(0, ids.shape[1], size):
        stop = min(start + size, ids.shape[1])
        result = candidate.step(slice_batch(batch, start, stop), ids[:, start:stop], names,
                                admitted[:, start:stop], state)
        state = result.state
        pieces.append(result.tensors)
    return {key: torch.cat([piece[key] for piece in pieces], dim=1) for key in pieces[0]}


def difference(a, b):
    if not torch.equal(torch.isfinite(a), torch.isfinite(b)):
        raise AssertionError("finite/unavailable masks differ")
    valid = torch.isfinite(a)
    return float((a[valid] - b[valid]).abs().max()) if valid.any() else 0.


def parity(first, other):
    result = {}
    for key, value in first.items():
        if value.is_floating_point():
            result[key] = difference(value, other[key])
            assert result[key] <= TOLERANCE, (key, result[key])
        else:
            assert torch.equal(value, other[key]), key
    return result


def run(output, device):
    start = time.perf_counter()
    torch.manual_seed(19)
    torch.set_num_threads(2)
    sidecar = json.loads((BASE / "architecture_h2_results_v3.json").read_text(encoding="utf-8"))
    assert sidecar["status"] == "passed" and sidecar["saved_replay"]["strict_mask_and_reason_parity"]
    assert sidecar["saved_replay"]["ray_reference"] == "cam0"
    assert sidecar["saved_replay"]["ray_origin_unit"] == "meter"
    with np.load(INPUT, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    assert np.array_equal(arrays["frame_ids"], np.arange(360))
    assert np.array_equal(arrays["original_pair_ids"], np.arange(1157, 1517))
    assert int(arrays["strict_gate"].sum()) == 3031
    batch, admitted, ids = saved_contract_conditions(arrays, schema=JointSchema.COCO17_SURFACE,
                                                     frame="cam0", unit="meter", device=device)
    before_arrays = {key: value.copy() for key, value in arrays.items()}
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**config["assets"]).to(device)
    cfg = replace(ModelConfig(**config["model"]), observation_joints=17)
    model = FlowMesh(cfg, decoder.parents).to(device).eval()
    before_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    regressor = load_coco17_regressor("models/smpl/J_regressor_coco.npy")
    names = ("20260927_201716_220_pmpose360",)
    candidate = CausalMeshCandidate(model, schema=JointSchema.COCO17_SURFACE, calibration_frames=8)
    control = CausalMeshCandidate(model, schema=JointSchema.COCO17_SURFACE, calibration_frames=1)
    assert candidate.shape.head is control.shape.head is model.encoder.shape_head
    with torch.no_grad():
        first = candidate.step(batch, ids, names, admitted)
        base = control.step(batch, ids, names, admitted)
        result = first.tensors
        assert torch.equal(result["raw_state_diagnostic"], base.tensors["raw_state_diagnostic"])
        assert torch.equal(result["raw_translation_diagnostic_m"], base.tensors["raw_translation_diagnostic_m"])
        original_beta = model.encoder(batch)["betas"]
        assert difference(base.tensors["betas"][:, 0], original_beta) <= TOLERANCE
        assert admitted[0, 0] and result["shape_ready"][0].nonzero()[0].item() == 7
        assert not result["shape_ready"][0, :7].any()
        assert torch.isnan(result["betas"][~result["shape_ready"]]).all()
        beta = result["betas"][0, result["shape_ready"][0]]
        assert torch.equal(beta, beta[:1].expand_as(beta))
        missing = ~result["frame_observed"]
        assert int(missing.sum()) == 29 and not result["available"][missing].any()
        assert torch.isnan(result["rotations"][~result["available"]]).all()
        assert torch.isnan(result["translation_m"][~result["available"]]).all()
        shape_delta = float((result["betas"][:, 7] - base.tensors["betas"][:, 7]).norm())
        mesh = decode_available(decoder, result, regressor)
        control_mesh = decode_available(decoder, base.tensors, regressor)
        require_joint_schema(mesh["coco17_surface_m"], actual=JointSchema.COCO17_SURFACE,
                             expected=JointSchema.COCO17_SURFACE)
        for key in ("vertices", "kinematic_joints", "coco17_surface_m"):
            assert torch.isfinite(mesh[key][result["available"]]).all()
            assert torch.isnan(mesh[key][~result["available"]]).all()
        reconstruction = regress_coco17_torch(mesh["vertices"][result["available"]], regressor)
        surface_error = difference(reconstruction, mesh["coco17_surface_m"][result["available"]])
        assert surface_error == 0
        stream_checks = {}
        for size in (1, 11, 64):
            other = stream(candidate, batch, ids, admitted, names, size)
            stream_checks[str(size)] = parity(result, other)
            # One complete streaming run also verifies the full body/surface path.
            if size == 11:
                other_mesh = decode_available(decoder, other, regressor)
                stream_checks[str(size)]["mesh_max_error_m"] = difference(mesh["vertices"], other_mesh["vertices"])
                stream_checks[str(size)]["coco_surface_max_error_m"] = difference(mesh["coco17_surface_m"], other_mesh["coco17_surface_m"])
                assert stream_checks[str(size)]["mesh_max_error_m"] <= TOLERANCE
                assert stream_checks[str(size)]["coco_surface_max_error_m"] <= TOLERANCE
                del other_mesh
            del other
        future = {key: value.clone() for key, value in batch.items()}
        # Future point changes are marked as raw-pixel conditions only, not geometry.
        future["xy"][:, 120:] += 0.2
        future["confidence"][:, 120:] *= 0.5
        altered = candidate.step(future, ids, names, admitted).tensors
        future_checks = parity({key: value[:, :120] for key, value in result.items()},
                               {key: value[:, :120] for key, value in altered.items()})
        assert difference(result["betas"], altered["betas"]) == 0
        del altered
        source_gate = torch.as_tensor(arrays["strict_gate"], device=device)[None]
        assert torch.equal(batch["geometry_mask"], source_gate[:, :, None].expand(-1, -1, 2, -1))
        # Diagnostic XYZ must have no route into selected neural input keys.
        changed = copy.deepcopy(arrays)
        changed["strict_xyz_m_diagnostic"][:] = 99999
        changed_batch, changed_admitted, changed_ids = saved_contract_conditions(
            changed, schema=JointSchema.COCO17_SURFACE, frame="cam0", unit="meter", device=device)
        assert torch.equal(changed_admitted, admitted) and torch.equal(changed_ids, ids)
        for key, value in batch.items():
            if value.is_floating_point():
                assert difference(value, changed_batch[key]) == 0
            else:
                assert torch.equal(value, changed_batch[key])
        # Rebuild the accepted first/last actual mesh from saved parameters.
        indexes = result["available"][0].nonzero().flatten()[[0, -1]]
        direct = decoder(result["rotations"][:, indexes], result["betas"][:, indexes[0]],
                         result["translation_m"][:, indexes])
        rebuild_error = difference(direct["vertices"], mesh["vertices"][:, indexes])
        assert rebuild_error <= TOLERANCE
        assert decoder.faces.shape == (13776, 3)
    for key, value in before_model.items():
        assert torch.equal(model.state_dict()[key], value), key
    for key, value in before_arrays.items():
        assert np.array_equal(arrays[key], value, equal_nan=True) if value.dtype.kind in "fc" else np.array_equal(arrays[key], value), key
    status_counts = {status.name.lower(): int((result["status"] == status).sum()) for status in OutputStatus}
    report = {"status": "isolated_H3_real_input_mesh_contract_passed", "protocol": "architecture_h3_protocol.json",
              "training_executed": False, "checkpoint_loaded": False, "weights": "fresh_random_seed19",
              "source_contract": INPUT.as_posix(), "source_paths": sidecar["saved_replay"]["source_paths"],
              "frames": 360, "physical_pair_range": [1157, 1516], "input_schema": JointSchema.COCO17_SURFACE.value,
              "model_config": cfg.to_dict(), "model_parameter_count": sum(p.numel() for p in model.parameters()),
              "neural_input_keys": list(batch), "strict_xyz_excluded_from_conditions": True,
              "strict_accepted_preserved": int(arrays["strict_gate"].sum()),
              "unmatched_frames_preserved": int(missing.sum()), "source_arrays_unchanged": True,
              "shape_admission": "any immutable strict accepted stereo COCO point, operational not shape accuracy",
              "first_ready_frame": 7, "beta_constant_after_ready": True, "same_shape_head": True,
              "shape_K8_vs_K1_l2_not_accuracy": shape_delta, "pose_and_translation_same_in_K1_K8": True,
              "K1_original_first_frame_beta_max_error": difference(base.tensors["betas"][:, 0], original_beta),
              "status_counts": status_counts, "model_state_unchanged": True, "candidates": 1, "Euler_nfe_per_frame": 8,
              "batch_stream_checks": stream_checks, "future_before120_checks": future_checks,
              "topology": {"vertices": 6890, "faces": 13776, "rotation_nodes": 52, "surface_observations": 17},
              "coco_surface_reconstruction_max_error_m": surface_error, "saved_parameter_mesh_rebuild_max_error_m": rebuild_error,
              "assets": config["assets"], "regressor": "models/smpl/J_regressor_coco.npy",
              "total_check_seconds_not_latency": time.perf_counter() - start,
              "boundary": "Real conditions and random structure/internal consistency only. No accuracy, world/ground/contact, original-paper performance or realtime claim. Existing GRU still consumes null frames; shared pooling not proven unbiased/identifiable."}
    output_arrays = {"source__" + key: value for key, value in arrays.items()}
    output_arrays.update({key: value.cpu().numpy() for key, value in result.items()})
    output_arrays.update({key: value.cpu().numpy() for key, value in mesh.items()})
    output_arrays.update({"control_K1__" + key: value.cpu().numpy() for key, value in base.tensors.items()})
    output_arrays["control_K1__coco17_surface_m"] = control_mesh["coco17_surface_m"].cpu().numpy()
    output_arrays["metadata"] = np.array(json.dumps(report, ensure_ascii=False))
    np.savez_compressed(output.with_suffix(".npz"), **output_arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("preserve earlier result, choose a new output version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.output, args.device)
    except Exception as error:  # noqa: BLE001 -- retain actual candidate failures
        result = {"status": "isolated_H3_check_failed", "training_executed": False,
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
