"""R13 isolate beta/translation dependencies without modifying the generator."""
import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.candidate import (
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
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**config["assets"])
    source = load_arrays(BASE / "architecture_h2_results_v3.npz")
    hand_source = load_arrays(BASE / "architecture_h4_results_v3.npz")
    body, admitted, ids = saved_contract_conditions(source, schema=JointSchema.COCO17_SURFACE,
                                                   frame="cam0", unit="meter")
    hands = saved_native_hands(hand_source, ids, body["view_mask"], decoder.parents)
    batch = slice_inputs(dict(body, native_hands=hands), 0, 16)
    ids, admitted = ids[:, :16], admitted[:, :16]
    torch.manual_seed(19)
    model = FlowMesh(replace(ModelConfig(**config["model"]), observation_joints=17), decoder.parents).eval()
    torch.manual_seed(29)
    branch = NativeHandConditioner(model.cfg.width, model.cfg.heads).eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    hand_before = {key: value.clone() for key, value in branch.state_dict().items()}
    candidate = ShapeFeatureMeshCandidate(model, branch, schema=JointSchema.COCO17_SURFACE,
                                         calibration_frames=8, steps=8, noise_seed=97)
    regressor = torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")))
    with torch.no_grad():
        control = candidate.step(batch, ids, ("repositioned360_H5",), admitted)
        old = load_arrays(BASE / "architecture_h7_results_v1.npz")
        old_parity = parity(control.tensors, {key: torch.tensor(old[key][:, :16]) for key in control.tensors})
        encoded, _features = candidate._encode(batch, ids, None)
        readout = dict(encoded, betas=control.tensors["betas"], shape_available=control.tensors["shape_ready"])
        noise = control.state.initial_noise.expand(-1, 16, -1, -1)
        original = model.transport(readout, steps=8, initial=noise)
        assert torch.equal(original["state"], control.tensors["raw_state_diagnostic"])
        changed_beta = dict(readout, betas=readout["betas"] + .25)
        beta_only = model.transport(changed_beta, steps=8, initial=noise)
        shift = torch.tensor([.1, -.2, .3])
        changed_T = dict(readout, translation=readout["translation"] + shift)
        T_only = model.transport(changed_T, steps=8, initial=noise)
        for value in (beta_only, T_only):
            assert torch.equal(value["state"], original["state"])
            assert torch.equal(value["rotations"], original["rotations"])
        assert torch.equal(beta_only["translation"], original["translation"])
        assert difference(T_only["betas"], original["betas"]) == 0
        base_mesh = decode_available(decoder, control.tensors, regressor)
        beta_tensors = dict(control.tensors, betas=changed_beta["betas"])
        beta_mesh = decode_available(decoder, beta_tensors, regressor)
        translated = dict(control.tensors, translation_m=control.tensors["translation_m"] + shift)
        T_mesh = decode_available(decoder, translated, regressor)
        available = control.tensors["available"]
        beta_vertex_change = difference(base_mesh["vertices"], beta_mesh["vertices"])
        T_error = float(((T_mesh["vertices"] - base_mesh["vertices"])[available] - shift).abs().max())
        assert beta_vertex_change > 0 and T_error < 2e-5
        for value in (base_mesh, beta_mesh, T_mesh):
            assert torch.isfinite(value["vertices"][available]).all()
            assert torch.isnan(value["vertices"][~available]).all()
    # A missing autograd edge differs from a saturated zero derivative.
    beta_leaf = control.tensors["betas"].clone().requires_grad_()
    state = model.transport(dict(readout, betas=beta_leaf), steps=8, initial=noise)["state"]
    beta_to_state = torch.autograd.grad(state.square().mean(), beta_leaf, allow_unused=True)[0]
    assert beta_to_state is None
    gradient_mesh = decode_available(decoder, dict(control.tensors, betas=beta_leaf), regressor)
    beta_gradient = torch.autograd.grad(gradient_mesh["vertices"][available].square().mean(), beta_leaf)[0]
    assert torch.isfinite(beta_gradient).all() and beta_gradient[available].abs().sum() > 0
    assert beta_gradient[~available].eq(0).all()
    assert all(value.grad is None for value in model.parameters())
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(torch.equal(hand_before[key], value) for key, value in branch.state_dict().items())
    report = {"status": "R13_parameter_coupling_verified", "training_executed": False,
              "checkpoint_loaded": False, "network_requests": 0, "whole_goal_complete": False,
              "frames": 16, "frame_range": [0, 15], "physical_pair_range": [1157, 1172],
              "model_seed": 19, "hand_seed": 29, "noise_seed": 97, "shape_K": 8, "Euler_nfe": 8,
              "source_body": "architecture_h2_results_v3.npz", "source_hands": "architecture_h4_results_v3.npz",
              "saved_H7_prefix_checks": old_parity, "available_frames": int(available.sum()),
              "beta_only_delta": .25, "beta_only_rawstate_rotations_translation_exact": True,
              "T_only_shift_m": shift.tolist(), "T_only_rawstate_rotations_beta_exact": True,
              "state_autograd_beta_edge": "absent_None_allow_unused_not_saturation",
              "beta_mesh_gradient_l1": float(beta_gradient[available].abs().sum()),
              "beta_mesh_max_change_m_not_accuracy": beta_vertex_change,
              "translation_mesh_shift_max_error_m": T_error, "unavailable_meshes_retained_NaN": True,
              "parameters_unchanged": True,
              "finding": "Current conditional velocity consumes condition/state/time only. Beta and translation are pass-through readouts to mesh, not velocity conditions. Learned correlation through shared encoder is not direct beta conditioning.",
              "boundary": "16-frame intervention/dependency audit only, not new360 validation or improvedmesh/pose accuracy. Beta/body positive control prevents conflating absent flowdependency with irrelevant beta asset."}
    np.savez_compressed(output.with_suffix(".npz"), frame_ids=ids.numpy(), available=available.numpy(),
                        control_betas=control.tensors["betas"].numpy(), beta_only=changed_beta["betas"].numpy(),
                        control_state=original["state"].numpy(), beta_only_state=beta_only["state"].numpy(),
                        T_only_state=T_only["state"].numpy(), control_rotations=original["rotations"].numpy(),
                        control_translation=control.tensors["translation_m"].numpy(), T_only_translation=translated["translation_m"].numpy(),
                        control_vertices=base_mesh["vertices"].numpy(), beta_only_vertices=beta_mesh["vertices"].numpy(),
                        T_only_vertices=T_mesh["vertices"].numpy(), beta_mesh_gradient=beta_gradient.numpy(),
                        faces=decoder.faces.numpy())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain prior evidence; select a new version")
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 -- retain actual audit failure
        result = {"status": "R13_parameter_coupling_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
