"""H5 real360 composition and cost checks; random weights, no training."""
import argparse
import json
import platform
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.candidate import (
    CausalMeshCandidate,
    OutputStatus,
    decode_available,
    saved_contract_conditions,
)
from realtime_app.learning_flow_mesh.hand_candidate import (
    NativeHandConditioner,
    NativeHandMeshCandidate,
    disabled_native_hands,
    saved_native_hands,
    slice_native_hands,
)
from realtime_app.learning_flow_mesh.interfaces import JointSchema
from realtime_app.learning_flow_mesh.model import FlowMesh, ModelConfig
from realtime_app.learning_flow_mesh.reproduction.check_joint_candidate import (
    difference,
    parity,
)
from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")


def slice_inputs(batch, start, end):
    return {key: slice_native_hands(value, start, end) if key == "native_hands" else value[:, start:end]
            for key, value in batch.items()}


def stream(candidate, batch, ids, admitted, size):
    pieces, state = [], None
    for start in range(0, ids.shape[1], size):
        end = min(start + size, ids.shape[1])
        piece = candidate.step(slice_inputs(batch, start, end), ids[:, start:end],
                               ("repositioned360_H5",), admitted[:, start:end], state)
        pieces.append(piece.tensors)
        state = piece.state
    return {key: torch.cat([piece[key] for piece in pieces], 1) for key in pieces[0]}


def timed(function, samples=50):
    for _ in range(5):
        function()
    values = []
    for _ in range(samples):
        begin = time.perf_counter()
        function()
        values.append((time.perf_counter() - begin) * 1000)
    return {"samples": samples, "warmup": 5, "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95))}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def run(output):
    torch.set_num_threads(2)
    source = load_arrays(BASE / "architecture_h2_results_v3.npz")
    hand_source = load_arrays(BASE / "architecture_h4_results_v3.npz")
    hand_report = json.loads((BASE / "architecture_h4_results_v3.json").read_text(encoding="utf-8"))
    assert hand_report["status"] == "native_hand_source_and_causal_token_contract_passed"
    assert np.array_equal(source["frame_ids"], np.arange(360))
    assert np.array_equal(source["original_pair_ids"], np.arange(1157, 1517))
    for key, value in source.items():
        old = hand_source["source__" + key]
        assert np.array_equal(value, old, equal_nan=True) if value.dtype.kind in "fc" else np.array_equal(value, old)
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**config["assets"])
    batch, admitted, ids = saved_contract_conditions(source, schema=JointSchema.COCO17_SURFACE,
                                                     frame="cam0", unit="meter")
    hands = saved_native_hands(hand_source, ids, batch["view_mask"], decoder.parents)
    for name, field in (("tokens", "tokens"), ("token_present", "present"), ("hand_admitted", "admitted")):
        stored, rebuilt = hand_source[name], getattr(hands, field).numpy()
        assert np.array_equal(stored, rebuilt, equal_nan=True) if stored.dtype.kind in "fc" else np.array_equal(stored, rebuilt)
    torch.manual_seed(19)
    model = FlowMesh(replace(ModelConfig(**config["model"]), observation_joints=17), decoder.parents).eval()
    torch.manual_seed(29)
    conditioner = NativeHandConditioner(model.cfg.width, model.cfg.heads).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    branch_before = {key: value.clone() for key, value in conditioner.state_dict().items()}
    settings = {"schema": JointSchema.COCO17_SURFACE, "calibration_frames": 8, "steps": 8, "noise_seed": 97}
    candidate = NativeHandMeshCandidate(model, conditioner, **settings)
    control = CausalMeshCandidate(model, **settings)
    augmented = dict(batch, native_hands=hands)
    disabled = dict(batch, native_hands=disabled_native_hands(hands))
    names = ("repositioned360_H5",)
    regressor = torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")))
    with torch.no_grad():
        original = control.step(batch, ids, names, admitted)
        ablated = candidate.step(disabled, ids, names, admitted)
        baseline = original.tensors
        zero_checks = parity(baseline, ablated.tensors)
        assert all(value == 0 for value in zero_checks.values())
        whole = candidate.step(augmented, ids, names, admitted)
        result = whole.tensors
        for key in ("betas", "shape_ready", "translation_m", "raw_translation_diagnostic_m",
                    "frame_observed", "shape_admitted", "available", "status"):
            parity({key: baseline[key]}, {key: result[key]})
            if baseline[key].is_floating_point():
                assert difference(baseline[key], result[key]) == 0
        assert torch.equal(original.state.hidden, whole.state.hidden)
        assert torch.equal(original.state.initial_noise, whole.state.initial_noise)
        with np.load(BASE / "architecture_h3_results_v2.npz", allow_pickle=False) as prior:
            old_checks = parity(baseline, {key: torch.tensor(prior[key]) for key in baseline})
            assert all(value == 0 for value in old_checks.values())
        encoded, _features = control._encode(batch, ids, None)
        delta = conditioner(encoded["condition"], hands, view_available=batch["view_mask"])
        assert torch.equal(delta[:, :, :22], torch.zeros_like(delta[:, :, :22]))
        no_hand_frame = ~hands.present.any(dim=(-1, -2, -3))
        assert torch.equal(delta[no_hand_frame], torch.zeros_like(delta[no_hand_frame]))
        raw_change = difference(result["raw_state_diagnostic"], baseline["raw_state_diagnostic"])
        assert raw_change > 1e-6
        assert difference(result["raw_state_diagnostic"][no_hand_frame], baseline["raw_state_diagnostic"][no_hand_frame]) == 0
        generated = result["raw_rotations_diagnostic"][:, :, 22:].reshape(1, 360, 1, 2, 15, 3, 3).expand(-1, -1, 2, -1, -1, -1, -1)
        source_distance = float((generated[hands.admitted] - hands.local_rotations[hands.admitted]).abs().max())
        assert source_distance > 1e-5  # source is a condition, not a hard replacement
        mesh = decode_available(decoder, result, regressor)
        control_mesh = decode_available(decoder, baseline, regressor)
        for key in ("vertices", "kinematic_joints", "coco17_surface_m"):
            assert torch.isfinite(mesh[key][result["available"]]).all()
            assert torch.isnan(mesh[key][~result["available"]]).all()
        chunks = {}
        for size in (1, 11, 64):
            other = stream(candidate, augmented, ids, admitted, size)
            chunks[str(size)] = parity(result, other)
            if size == 11:
                other_mesh = decode_available(decoder, other, regressor)
                chunks[str(size)]["vertices_max_error_m"] = difference(mesh["vertices"], other_mesh["vertices"])
                chunks[str(size)]["coco_surface_max_error_m"] = difference(mesh["coco17_surface_m"], other_mesh["coco17_surface_m"])
                assert max(chunks[str(size)].values()) <= 2e-5
        future = {key: value.clone() for key, value in batch.items()}
        future["xy"][:, 120:] += .2
        future["confidence"][:, 120:] *= .5
        changed_hands = dict(hand_source)
        rotations = hand_source["source_native_rotations"].copy()
        # Non-symmetric proper rotation changes future native conditions only.
        rotations[120:] = Rotation.from_rotvec([.4, -.15, .2]).as_matrix() @ rotations[120:]
        changed_hands["source_native_rotations"] = rotations
        future["native_hands"] = saved_native_hands(changed_hands, ids, batch["view_mask"], decoder.parents)
        altered = candidate.step(future, ids, names, admitted).tensors
        future_checks = parity({key: value[:, :120] for key, value in result.items()},
                               {key: value[:, :120] for key, value in altered.items()})
        assert all(value == 0 for value in future_checks.values())
        assert difference(result["betas"], altered["betas"]) == 0
        first8 = candidate.step(slice_inputs(augmented, 0, 8), ids[:, :8], names, admitted[:, :8])
        one = slice_inputs(augmented, 8, 9)
        empty_one = slice_inputs(disabled, 8, 9)
        condition_one, _ = control._encode(slice_inputs(batch, 8, 9), ids[:, 8:9], first8.state.hidden)
        costs = {"branch_real_frame8": timed(lambda: conditioner(condition_one["condition"], one["native_hands"])),
                 "branch_missing_frame8": timed(lambda: conditioner(condition_one["condition"], empty_one["native_hands"])),
                 "H3_candidate_frame8_8NFE": timed(lambda: control.step(slice_inputs(batch, 8, 9), ids[:, 8:9], names, admitted[:, 8:9], first8.state)),
                 "H5_candidate_frame8_8NFE": timed(lambda: candidate.step(one, ids[:, 8:9], names, admitted[:, 8:9], first8.state))}
        indexes = result["available"][0].nonzero().flatten()[[0, -1]]
        rebuilt = decoder(result["rotations"][:, indexes], result["betas"][:, indexes[0]],
                          result["translation_m"][:, indexes])
        rebuild_error = difference(rebuilt["vertices"], mesh["vertices"][:, indexes])
        assert rebuild_error < 2e-5
    for key, value in before.items():
        assert torch.equal(model.state_dict()[key], value), key
    for key, value in branch_before.items():
        assert torch.equal(conditioner.state_dict()[key], value), key
    assert int(source["strict_gate"].sum()) == 3031
    report = {"status": "isolated_native_hand_condition_branch_contract_passed", "protocol": "architecture_h5_protocol.json",
              "training_executed": False, "checkpoint_loaded": False, "frames": 360, "physical_pair_range": [1157, 1516],
              "model_seed": 19, "hand_branch_seed": 29, "noise_seed": 97, "candidates": 1, "Euler_nfe": 8,
              "model_config": model.cfg.to_dict(), "base_parameters": sum(p.numel() for p in model.parameters()),
              "new_hand_parameters": sum(p.numel() for p in conditioner.parameters()), "weights_unchanged": True,
              "source_body": "architecture_h2_results_v3.json/npz", "source_hands": "architecture_h4_results_v3.json/npz",
              "camera_order": list(conditioner.camera_order), "source_hand_view_slots": int(hands.source_present.sum()),
              "admitted_hand_view_slots": int(hands.admitted.sum()), "source_fields_preserved": len(source),
              "zero_hand_control_H3_parity": zero_checks, "saved_previous_H3_parity": old_checks,
              "shape_translation_history_noise_unchanged": True, "shared_beta_first_ready_frame": 7,
              "status_counts": {s.name.lower(): int((result["status"] == s).sum()) for s in OutputStatus},
              "nonhand_condition_residual_exact_zero": True, "all_missing_hand_condition_exact_zero": True,
              "raw_flow_state_max_change_not_accuracy": raw_change, "generated_vs_native_max_difference_not_error": source_distance,
              "stream_checks": chunks, "future_before120_checks": future_checks,
              "saved_parameter_mesh_rebuild_max_error_m": rebuild_error,
              "mesh_vertex_max_change_m_not_accuracy": difference(mesh["vertices"], control_mesh["vertices"]),
              "CPU_two_thread_cost_checks": costs,
              "cost_boundary": "50warmed repetitions CPU eager frame8 only; excludes detection, raw/strict geometry, source preparation, real body decode and Stage. Not end-to-end FPS or realtime certification.",
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "threads": 2, "dtype": "float32"},
              "boundary": "Untrained condition-branch wiring/internal consistency only, not pose/shape/contact accuracy. Original method resources/input/evaluation gaps and H3 null-GRU/pooling limitations remain."}
    arrays = {"source__" + key: value for key, value in source.items()}
    arrays.update({"hand_source__" + key: value for key, value in hand_source.items() if not key.startswith("source__")})
    arrays.update({key: value.numpy() for key, value in result.items()})
    arrays.update({key: value.numpy() for key, value in mesh.items()})
    arrays.update({"control_H3__" + key: value.numpy() for key, value in baseline.items()})
    arrays["control_H3__coco17_surface_m"] = control_mesh["coco17_surface_m"].numpy()
    arrays["hand_condition_delta"] = delta.numpy()
    arrays["metadata"] = np.array(json.dumps(report, ensure_ascii=False))
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain previous outputs; use a new version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.output)
    except Exception as error:  # noqa: BLE001 -- preserve actual architecture/source failures
        report = {"status": "isolated_native_hand_branch_failed", "training_executed": False,
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
