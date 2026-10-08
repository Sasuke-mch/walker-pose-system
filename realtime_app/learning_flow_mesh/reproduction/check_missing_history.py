"""H6 full360 original-vs-frozen history audit, no training or weight loading."""
import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.candidate import (
    OutputStatus,
    decode_available,
    saved_contract_conditions,
)
from realtime_app.learning_flow_mesh.hand_candidate import (
    NativeHandConditioner,
    NativeHandMeshCandidate,
    saved_native_hands,
)
from realtime_app.learning_flow_mesh.interfaces import JointSchema
from realtime_app.learning_flow_mesh.missing_history import (
    MissingHistoryMeshCandidate,
    freeze_missing_transitions,
)
from realtime_app.learning_flow_mesh.model import FlowMesh, ModelConfig
from realtime_app.learning_flow_mesh.reproduction.check_hand_candidate import (
    load_arrays,
    slice_inputs,
    stream,
)
from realtime_app.learning_flow_mesh.reproduction.check_joint_candidate import (
    difference,
    parity,
)
from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")


def hidden_trace(candidate, batch, ids):
    pieces, hidden = [], None
    for index in range(ids.shape[1]):
        encoded, _ = candidate._encode(slice_inputs(batch, index, index + 1), ids[:, index:index + 1], hidden)
        hidden = encoded["hidden"]
        pieces.append(hidden[-1].clone())
    return torch.stack(pieces, 1)


def run(output):
    torch.set_num_threads(2)
    source = load_arrays(BASE / "architecture_h2_results_v3.npz")
    hands_source = load_arrays(BASE / "architecture_h4_results_v3.npz")
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**config["assets"])
    body, admitted, ids = saved_contract_conditions(source, schema=JointSchema.COCO17_SURFACE,
                                                   frame="cam0", unit="meter")
    hands = saved_native_hands(hands_source, ids, body["view_mask"], decoder.parents)
    batch = dict(body, native_hands=hands)
    torch.manual_seed(19)
    model = FlowMesh(replace(ModelConfig(**config["model"]), observation_joints=17), decoder.parents).eval()
    torch.manual_seed(29)
    conditioner = NativeHandConditioner(model.cfg.width, model.cfg.heads).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    hand_before = {key: value.clone() for key, value in conditioner.state_dict().items()}
    settings = {"schema": JointSchema.COCO17_SURFACE, "calibration_frames": 8, "steps": 8, "noise_seed": 97}
    control = NativeHandMeshCandidate(model, conditioner, **settings)
    disabled = MissingHistoryMeshCandidate(model, conditioner, missing_policy="update_all", **settings)
    candidate = MissingHistoryMeshCandidate(model, conditioner, **settings)
    names = ("repositioned360_H5",)
    regressor = torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")))
    with torch.no_grad():
        original = control.step(batch, ids, names, admitted)
        switched_off = disabled.step(batch, ids, names, admitted)
        disabled_parity = parity(original.tensors, switched_off.tensors)
        assert all(value == 0 for value in disabled_parity.values())
        old = load_arrays(BASE / "architecture_h5_results_v3.npz")
        old_parity = parity(original.tensors, {key: torch.tensor(old[key]) for key in original.tensors})
        assert all(value == 0 for value in old_parity.values())
        result = candidate.step(batch, ids, names, admitted)
        for key in ("available", "status", "shape_admitted", "shape_ready", "frame_observed", "frame_ids"):
            assert torch.equal(result.tensors[key], original.tensors[key])
        chunks = {}
        for size in (1, 11, 64):
            checks = parity(result.tensors, stream(candidate, batch, ids, admitted, size))
            assert all(value <= 2e-5 for value in checks.values())
            chunks[str(size)] = checks
        future = {key: value.clone() for key, value in body.items()}
        future["xy"][:, 120:] += .4
        tokens = hands.tokens.clone()
        tokens[:, 120:, ..., :6] += .2
        future["native_hands"] = replace(hands, tokens=tokens)
        altered = candidate.step(future, ids, names, admitted)
        future_checks = parity({key: value[:, :120] for key, value in result.tensors.items()},
                               {key: value[:, :120] for key, value in altered.tensors.items()})
        assert all(value == 0 for value in future_checks.values())
        original_trace = hidden_trace(control, batch, ids)
        frozen_trace = hidden_trace(candidate, batch, ids)
        missing = ~result.tensors["frame_observed"]
        assert int(missing.sum()) == 29
        zeros = original_trace.new_zeros(1, 1, model.cfg.width)
        original_previous = torch.cat((zeros, original_trace[:, :-1]), 1)
        frozen_previous = torch.cat((zeros, frozen_trace[:, :-1]), 1)
        frozen_missing_delta = difference(frozen_trace[missing], frozen_previous[missing])
        original_missing_delta = difference(original_trace[missing], original_previous[missing])
        assert frozen_missing_delta == 0 and original_missing_delta > 0
        duration_checks = {}
        generator = torch.Generator().manual_seed(103)
        hidden = torch.randn(1, 1, model.cfg.width, generator=generator)
        recovery = torch.randn(1, 1, model.cfg.width, generator=generator)
        expected, expected_hidden = model.encoder.history.forward(recovery, hidden)
        for length in (1, 8, 64):
            pooled = torch.cat((torch.zeros(1, length, model.cfg.width), recovery), 1)
            presence = torch.zeros(1, length + 1, dtype=torch.bool)
            presence[:, -1] = True
            recovered, final = freeze_missing_transitions(model.encoder.history, pooled, presence, hidden)
            assert torch.equal(final, expected_hidden) and torch.equal(recovered[:, -1:], expected)
            duration_checks[str(length)] = {"recovery_hidden_vs_compressed_observation_max_error": difference(final, expected_hidden),
                                           "time_signal_provided": False}
        # Frame numbers validate identity/order; changing elapsed frame IDs has no model effect.
        sparse_ids = ids * 10
        sparse_hands = replace(hands, frame_ids=sparse_ids)
        sparse = candidate.step(dict(body, native_hands=sparse_hands), sparse_ids, names, admitted)
        sparse_checks = parity({k: v for k, v in result.tensors.items() if k != "frame_ids"},
                               {k: v for k, v in sparse.tensors.items() if k != "frame_ids"})
        assert all(value == 0 for value in sparse_checks.values())
        mesh = decode_available(decoder, result.tensors, regressor)
        base_mesh = decode_available(decoder, original.tensors, regressor)
        available = result.tensors["available"]
        assert torch.isfinite(mesh["vertices"][available]).all()
        assert torch.isnan(mesh["vertices"][~available]).all()
        indexes = available[0].nonzero().flatten()[[0, -1]]
        rebuilt = decoder(result.tensors["rotations"][:, indexes], result.tensors["betas"][:, indexes[0]],
                          result.tensors["translation_m"][:, indexes])
        rebuild_error = difference(rebuilt["vertices"], mesh["vertices"][:, indexes])
        assert rebuild_error < 2e-5
    assert not model.encoder.history._forward_hooks
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(torch.equal(hand_before[key], value) for key, value in conditioner.state_dict().items())
    report = {"status": "H6_missing_history_contract_passed", "protocol": "architecture_h6_protocol.json",
              "whole_goal_complete": False, "training_executed": False, "checkpoint_loaded": False,
              "parameter_updates": False, "new_parameters": 0, "frames": 360, "physical_pair_range": [1157, 1516],
              "model_seed": 19, "hand_seed": 29, "noise_seed": 97, "Euler_nfe": 8, "shape_calibration_frames": 8,
              "source_body": "architecture_h2_results_v3.npz", "source_hands": "architecture_h4_results_v3.npz",
              "disabled_policy_H5_exact_parity": disabled_parity, "saved_H5_exact_parity": old_parity,
              "source_body_fields_preserved": len(source), "source_hand_slots": int(hands.source_present.sum()),
              "admitted_hand_slots": int(hands.admitted.sum()), "strict_points": int(source["strict_gate"].sum()),
              "status_counts": {status.name.lower(): int((result.tensors["status"] == status).sum()) for status in OutputStatus},
              "shape_first_ready_frame": int(ids[result.tensors["shape_ready"]].min()),
              "frozen_missing_hidden_max_change": frozen_missing_delta, "control_missing_hidden_max_change": original_missing_delta,
              "hidden_max_change_between_policies_not_accuracy": difference(original_trace, frozen_trace),
              "beta_max_change_between_policies_not_accuracy": difference(original.tensors["betas"], result.tensors["betas"]),
              "mesh_max_change_m_not_accuracy": difference(base_mesh["vertices"], mesh["vertices"]),
              "chunk_checks": chunks, "future_before120_checks": future_checks,
              "long_missing_duration_checks": duration_checks, "sparse_frame_ids_no_time_response": sparse_checks,
              "independent_parameter_mesh_rebuild_max_error_m": rebuild_error,
              "elapsed_time_limit": "Frozen history uses observation clock only. Missing1/8/64 and frame IDs scaled10 give identical recovery; no physical-time motion prediction.",
              "harness_limit": "Scoped eager GRU output hook runs original batched path then repeats GRU per frame; no concurrency/export or latency claim.",
              "boundary": "Random untrained state/causal and real decoder wiring only, not accuracy improvement or recommended deployment policy."}
    arrays = {"source__" + key: value for key, value in source.items()}
    arrays.update({"hand_source__" + key: value for key, value in hands_source.items() if not key.startswith("source__")})
    arrays.update({key: value.numpy() for key, value in result.tensors.items()})
    arrays.update({"control_H5__" + key: value.numpy() for key, value in original.tensors.items()})
    arrays.update({key: value.numpy() for key, value in mesh.items()})
    arrays["control_hidden_trace"] = original_trace.numpy()
    arrays["frozen_hidden_trace"] = frozen_trace.numpy()
    arrays["metadata"] = np.array(json.dumps(report, ensure_ascii=False))
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain prior evidence and choose new version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args.output)
    except Exception as error:  # noqa: BLE001 -- retain actual candidate failures
        report = {"status": "H6_missing_history_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if report["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
