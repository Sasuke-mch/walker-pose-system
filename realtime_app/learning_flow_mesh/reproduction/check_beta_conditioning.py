"""H8 complete saved360 ablation, streaming, beta sensitivity and real mesh."""
import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from realtime_app.learning_flow_mesh.beta_conditioning import (
    BetaConditionedFlow,
    BetaConditionedMeshCandidate,
)
from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.candidate import (
    OutputStatus,
    decode_available,
    saved_contract_conditions,
)
from realtime_app.learning_flow_mesh.hand_candidate import (
    NativeHandConditioner,
    saved_native_hands,
)
from realtime_app.learning_flow_mesh.interfaces import JointSchema
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
from realtime_app.learning_flow_mesh.shape_features import ShapeFeatureMeshCandidate
from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")


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
    base = FlowMesh(replace(ModelConfig(**config["model"]), observation_joints=17), decoder.parents).eval()
    torch.manual_seed(29)
    hand = NativeHandConditioner(base.cfg.width, base.cfg.heads).eval()
    torch.manual_seed(127)
    wrapper = BetaConditionedFlow(base, enabled=False).eval()
    before = {key: value.clone() for key, value in wrapper.state_dict().items()}
    hand_before = {key: value.clone() for key, value in hand.state_dict().items()}
    options = {"schema": JointSchema.COCO17_SURFACE, "calibration_frames": 8, "steps": 8, "noise_seed": 97}
    original = ShapeFeatureMeshCandidate(base, hand, **options)
    candidate = BetaConditionedMeshCandidate(wrapper, hand, **options)
    names = ("repositioned360_H5",)
    regressor = torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")))
    with torch.no_grad():
        control = original.step(batch, ids, names, admitted)
        disabled = candidate.step(batch, ids, names, admitted)
        disabled_checks = parity(control.tensors, disabled.tensors)
        old = load_arrays(BASE / "architecture_h7_results_v1.npz")
        old_checks = parity(control.tensors, {key: torch.tensor(old[key]) for key in control.tensors})
        assert all(value == 0 for value in disabled_checks.values())
        assert all(value == 0 for value in old_checks.values())
        wrapper.enabled = True
        result = candidate.step(batch, ids, names, admitted)
        untouched = {key: value for key, value in control.tensors.items() if key not in
                     ("rotations", "raw_rotations_diagnostic", "raw_state_diagnostic")}
        unchanged = parity(untouched, result.tensors)
        assert all(value == 0 for value in unchanged.values())
        assert torch.equal(control.state.hidden, result.state.hidden)
        assert torch.equal(control.state.initial_noise, result.state.initial_noise)
        assert torch.equal(control.tensors["raw_state_diagnostic"][:, :7], result.tensors["raw_state_diagnostic"][:, :7])
        chunk_checks = {}
        for size in (1, 11, 64):
            checks = parity(result.tensors, stream(candidate, batch, ids, admitted, size))
            assert all(value <= 2e-5 for value in checks.values())
            chunk_checks[str(size)] = checks
        future = {key: value.clone() for key, value in body.items()}
        future["xy"][:, 120:] += .4
        tokens = hands.tokens.clone()
        tokens[:, 120:, ..., :6] += .2
        future["native_hands"] = replace(hands, tokens=tokens)
        altered = candidate.step(future, ids, names, admitted)
        future_checks = parity({key: value[:, :120] for key, value in result.tensors.items()},
                               {key: value[:, :120] for key, value in altered.tensors.items()})
        assert all(value == 0 for value in future_checks.values())
        mesh = decode_available(decoder, result.tensors, regressor)
        base_mesh = decode_available(decoder, control.tensors, regressor)
        available = result.tensors["available"]
        assert torch.isfinite(mesh["vertices"][available]).all() and torch.isnan(mesh["vertices"][~available]).all()
        indices = available[0].nonzero().flatten()[[0, -1]]
        rebuilt = decoder(result.tensors["rotations"][:, indices], result.state.shape.betas,
                          result.tensors["translation_m"][:, indices])
        rebuild_error = difference(rebuilt["vertices"], mesh["vertices"][:, indices])
        assert rebuild_error < 2e-5
        encoded, _features = candidate._encode(slice_inputs(batch, 0, 16), ids[:, :16], None)
        readout = dict(encoded, betas=result.tensors["betas"][:, :16], shape_available=result.tensors["shape_ready"][:, :16])
        noise = result.state.initial_noise.expand(-1, 16, -1, -1)
        base_endpoint = wrapper.transport(readout, steps=8, initial=noise)
        changed_endpoint = wrapper.transport(dict(readout, betas=readout["betas"] + .25), steps=8, initial=noise)
        beta_sensitivity = difference(base_endpoint["state"], changed_endpoint["state"])
        assert beta_sensitivity > 0 and torch.isfinite(changed_endpoint["state"]).all()
        assert torch.equal(base_endpoint["state"][:, :7], changed_endpoint["state"][:, :7])
    beta = readout["betas"].clone().requires_grad_()
    endpoint = wrapper.transport(dict(readout, betas=beta), steps=8, initial=noise)
    beta_grad, projection_grad = torch.autograd.grad(endpoint["state"].square().mean(), (beta, wrapper.projection.weight))
    assert beta_grad[:, :7].eq(0).all() and beta_grad[:, 7:].abs().sum() > 0
    assert torch.isfinite(beta_grad).all() and torch.isfinite(projection_grad).all() and projection_grad.abs().sum() > 0
    assert all(value.grad is None for value in wrapper.parameters())
    assert all(torch.equal(before[key], value) for key, value in wrapper.state_dict().items())
    assert all(torch.equal(hand_before[key], value) for key, value in hand.state_dict().items())
    total_params = sum(p.numel() for p in wrapper.parameters()) + sum(p.numel() for p in hand.parameters())
    report = {"status": "H8_beta_velocity_condition_contract_passed", "protocol": "architecture_h8_protocol.json",
              "training_executed": False, "checkpoint_loaded": False, "network_requests": 0, "whole_goal_complete": False,
              "model_seed": 19, "hand_seed": 29, "conditioning_seed": 127, "noise_seed": 97,
              "Euler_nfe": 8, "shape_K": 8, "frames": 360, "physical_pair_range": [1157, 1516],
              "added_parameters": wrapper.projection.weight.numel(), "total_parameters": total_params,
              "parameters_unchanged": True, "disabled_H7_exact": disabled_checks, "saved_H7_exact": old_checks,
              "unchanged_beta_translation_masks": unchanged, "hidden_noise_exact": True, "first7_rawstate_exact": True,
              "source_body_fields_preserved": len(source), "source_native_slots": int(hands.source_present.sum()),
              "admitted_native_slots": int(hands.admitted.sum()), "strict_points": int(source["strict_gate"].sum()),
              "status_counts": {status.name.lower(): int((result.tensors["status"] == status).sum()) for status in OutputStatus},
              "chunk_checks": chunk_checks, "future_before120_checks": future_checks,
              "random_pose_state_delta_not_accuracy": difference(control.tensors["raw_state_diagnostic"], result.tensors["raw_state_diagnostic"]),
              "random_mesh_delta_m_not_accuracy": difference(base_mesh["vertices"], mesh["vertices"]),
              "beta_only_endpoint_sensitivity_first16": beta_sensitivity,
              "beta_endpoint_gradient_l1_first16": float(beta_grad.abs().sum()),
              "projection_gradient_l1_first16": float(projection_grad.abs().sum()), "unknown_prefix_beta_gradient_exact0": True,
              "independent_mesh_rebuild_max_error_m": rebuild_error, "vertices_shape": list(mesh["vertices"].shape),
              "faces_shape": list(mesh["faces"].shape),
              "boundary": "Added640 random projection parameters; activebeta dependency and fullsaved360engineering contracts, not trainedaccuracy, capacitymatched gain, hierarchicalSO3 or deployment. H6 elapsedtime/eagerhooks limits persist."}
    arrays = {"source__" + key: value for key, value in source.items()}
    arrays.update({"hand_source__" + key: value for key, value in hands_source.items() if not key.startswith("source__")})
    arrays.update({key: value.numpy() for key, value in result.tensors.items()})
    arrays.update({"control_H7__" + key: value.numpy() for key, value in control.tensors.items()})
    arrays.update({key: value.numpy() for key, value in mesh.items()})
    arrays.update({"beta_probe_gradient": beta_grad.numpy(), "projection_probe_gradient": projection_grad.numpy()})
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain prior results; select a new version")
    try:
        report = run(args.output)
    except Exception as error:  # noqa: BLE001 -- keep actual candidate failures
        report = {"status": "H8_beta_velocity_condition_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, allow_nan=False), flush=True)
    if report["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
