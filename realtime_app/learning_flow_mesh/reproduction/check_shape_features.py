"""H7 full360 shape source ablation and recurrent information attribution."""
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
    stream,
)
from realtime_app.learning_flow_mesh.reproduction.check_joint_candidate import (
    difference,
    parity,
)
from realtime_app.learning_flow_mesh.shape_features import ShapeFeatureMeshCandidate
from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")


def information_attribution(model):
    """Controlled original-GRU input: two observed but nonadmitted early frames."""
    generator = torch.Generator().manual_seed(109)
    pooled = torch.randn(1, 10, model.cfg.width, generator=generator, requires_grad=True)
    observed = torch.ones(1, 10, dtype=torch.bool)
    history, _ = freeze_missing_transitions(model.encoder.history, pooled, observed)
    local_beta = model.encoder.shape_head(pooled[:, 2:].mean(1))
    history_beta = model.encoder.shape_head(history[:, 2:].mean(1))
    local_grad = torch.autograd.grad(local_beta.square().sum(), pooled)[0]
    history_grad = torch.autograd.grad(history_beta.square().sum(), pooled)[0]
    assert local_grad[:, :2].eq(0).all() and history_grad[:, :2].abs().sum() > 0
    assert local_grad[:, 2:].abs().sum() > 0 and torch.isfinite(local_grad).all()
    assert torch.isfinite(history_grad).all()
    changed = pooled.detach().clone()
    changed[:, :2] += .5
    with torch.no_grad():
        changed_history, _ = freeze_missing_transitions(model.encoder.history, changed, observed)
        changed_local_beta = model.encoder.shape_head(changed[:, 2:].mean(1))
        changed_history_beta = model.encoder.shape_head(changed_history[:, 2:].mean(1))
    assert torch.equal(local_beta, changed_local_beta)
    assert difference(history_beta, changed_history_beta) > 0
    values = {"local_gradient_l1_by_frame": local_grad.abs().sum(-1)[0].tolist(),
              "history_gradient_l1_by_frame": history_grad.abs().sum(-1)[0].tolist(),
              "nonadmitted_early_local_beta_delta": difference(local_beta, changed_local_beta),
              "nonadmitted_early_history_beta_delta": difference(history_beta, changed_history_beta),
              "fixture": "Two observed but unadmitted frames, followed by eight admitted; gradient norm is sensitivity, not statistical information/accuracy"}
    arrays = {"attribution__pooled": pooled.detach().numpy(),
              "attribution__local_gradient": local_grad.numpy(),
              "attribution__history_gradient": history_grad.numpy()}
    return values, arrays


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
    branch = NativeHandConditioner(model.cfg.width, model.cfg.heads).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    hand_before = {key: value.clone() for key, value in branch.state_dict().items()}
    settings = {"schema": JointSchema.COCO17_SURFACE, "calibration_frames": 8, "steps": 8, "noise_seed": 97}
    control = MissingHistoryMeshCandidate(model, branch, **settings)
    history = ShapeFeatureMeshCandidate(model, branch, shape_feature_source="history", **settings)
    local = ShapeFeatureMeshCandidate(model, branch, **settings)
    names = ("repositioned360_H5",)
    regressor = torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")))
    captured = []

    def capture(_module, args):
        captured.append(args[0].clone())

    with torch.no_grad():
        original = control.step(batch, ids, names, admitted)
        history_readout = history.step(batch, ids, names, admitted)
        disabled_parity = parity(original.tensors, history_readout.tensors)
        old = load_arrays(BASE / "architecture_h6_results_v1.npz")
        old_parity = parity(original.tensors, {key: torch.tensor(old[key]) for key in original.tensors})
        assert all(value == 0 for value in disabled_parity.values())
        assert all(value == 0 for value in old_parity.values())
        hook = model.encoder.history.register_forward_pre_hook(capture)
        try:
            _encoded, pooled = local._encode(batch, ids, None)
        finally:
            hook.remove()
        assert len(captured) == 1 and torch.equal(pooled, captured[0])
        result = local.step(batch, ids, names, admitted)
        nonbeta = {key: value for key, value in original.tensors.items() if key != "betas"}
        unchanged = parity(nonbeta, result.tensors)
        assert all(value == 0 for value in unchanged.values())
        assert torch.equal(original.state.hidden, result.state.hidden)
        assert torch.equal(original.state.initial_noise, result.state.initial_noise)
        calibration = admitted[0].nonzero().flatten()[:8]
        expected_beta = model.encoder.shape_head(pooled[:, calibration].mean(1))
        independent_error = difference(result.state.shape.betas, expected_beta)
        assert independent_error < 2e-5
        ready = result.tensors["shape_ready"]
        frozen_beta = result.tensors["betas"][ready]
        assert torch.equal(frozen_beta, frozen_beta[:1].expand_as(frozen_beta))
        chunks = {}
        for size in (1, 11, 64):
            checks = parity(result.tensors, stream(local, batch, ids, admitted, size))
            assert all(value <= 2e-5 for value in checks.values())
            chunks[str(size)] = checks
        future = {key: value.clone() for key, value in body.items()}
        future["xy"][:, 120:] += .4
        future_tokens = hands.tokens.clone()
        future_tokens[:, 120:, ..., :6] += .2
        future["native_hands"] = replace(hands, tokens=future_tokens)
        altered = local.step(future, ids, names, admitted)
        future_checks = parity({key: value[:, :120] for key, value in result.tensors.items()},
                               {key: value[:, :120] for key, value in altered.tensors.items()})
        assert all(value == 0 for value in future_checks.values())
        changed = dict(batch, xy=body["xy"].clone())
        changed["xy"][:, 0] += .8
        _, changed_pooled = local._encode(changed, ids, None)
        _, base_history = control._encode(batch, ids, None)
        _, changed_history = control._encode(changed, ids, None)
        assert torch.equal(pooled[:, 1:], changed_pooled[:, 1:])
        later_history_delta = difference(base_history[:, 1:], changed_history[:, 1:])
        assert later_history_delta > 0
        mesh = decode_available(decoder, result.tensors, regressor)
        base_mesh = decode_available(decoder, original.tensors, regressor)
        available = result.tensors["available"]
        assert torch.isfinite(mesh["vertices"][available]).all()
        assert torch.isnan(mesh["vertices"][~available]).all()
        indices = available[0].nonzero().flatten()[[0, -1]]
        rebuilt = decoder(result.tensors["rotations"][:, indices], result.state.shape.betas,
                          result.tensors["translation_m"][:, indices])
        rebuild_error = difference(rebuilt["vertices"], mesh["vertices"][:, indices])
        assert rebuild_error < 2e-5
        parameter_count = sum(value.numel() for value in model.parameters()) + sum(value.numel() for value in branch.parameters())
    attribution, attribution_arrays = information_attribution(model)
    assert not model.encoder.history._forward_hooks and not model.encoder.history._forward_pre_hooks
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(torch.equal(hand_before[key], value) for key, value in branch.state_dict().items())
    assert all(value.grad is None for value in model.parameters())
    report = {"status": "H7_shape_information_source_contract_passed", "protocol": "architecture_h7_protocol.json",
              "whole_goal_complete": False, "training_executed": False, "checkpoint_loaded": False,
              "network_requests": 0, "new_parameters": 0, "parameter_count": parameter_count,
              "parameters_unchanged": True, "frames": 360, "physical_pair_range": [1157, 1516],
              "model_seed": 19, "hand_seed": 29, "noise_seed": 97, "Euler_nfe": 8, "shape_calibration_frames": 8,
              "source_body": "architecture_h2_results_v3.npz", "source_hands": "architecture_h4_results_v3.npz",
              "history_source_H6_exact_parity": disabled_parity, "saved_H6_exact_parity": old_parity,
              "local_vs_history_all_nonbeta_tensors_exact": unchanged, "hidden_noise_exact": True,
              "local_features_equal_original_GRU_inputs": True, "independent_beta_mean_max_error": independent_error,
              "shape_calibration_frame_ids": ids[:, calibration].tolist(),
              "source_body_fields_preserved": len(source), "source_hand_slots": int(hands.source_present.sum()),
              "admitted_hand_slots": int(hands.admitted.sum()), "strict_points": int(source["strict_gate"].sum()),
              "status_counts": {status.name.lower(): int((result.tensors["status"] == status).sum()) for status in OutputStatus},
              "beta_max_change_not_accuracy": difference(original.tensors["betas"], result.tensors["betas"]),
              "mesh_max_change_m_not_accuracy": difference(base_mesh["vertices"], mesh["vertices"]),
              "chunk_checks": chunks, "future_before120_checks": future_checks,
              "earlier_frame_perturb_later_local_features_delta": 0,
              "earlier_frame_perturb_later_history_features_delta": later_history_delta,
              "controlled_gradient_attribution": attribution,
              "independent_mesh_rebuild_error_m": rebuild_error, "vertices_shape": list(mesh["vertices"].shape),
              "faces_shape": list(mesh["faces"].shape),
              "harness_limit": "Read-only eager GRU input hook plus inheritedH6 output intervention; not concurrent/export/deployment or latency evidence",
              "boundary": "Removes recurrent history reuse from shape feature pool, not statistical dependence or shape error. Untrained engineering ablation; no accuracy preference. H6 time blindness retained."}
    arrays = {"source__" + key: value for key, value in source.items()}
    arrays.update({"hand_source__" + key: value for key, value in hands_source.items() if not key.startswith("source__")})
    arrays.update({key: value.numpy() for key, value in result.tensors.items()})
    arrays.update({"control_H6__" + key: value.numpy() for key, value in original.tensors.items()})
    arrays.update({key: value.numpy() for key, value in mesh.items()})
    arrays.update(attribution_arrays)
    arrays["local_shape_features"] = pooled.numpy()
    arrays["history_shape_features"] = base_history.numpy()
    arrays["metadata"] = np.array(json.dumps(report, ensure_ascii=False, allow_nan=False))
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
    except Exception as error:  # noqa: BLE001 -- preserve actual experiment failures
        report = {"status": "H7_shape_information_source_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)
    if report["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
