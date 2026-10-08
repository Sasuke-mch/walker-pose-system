"""H4: real native hand-source audit and causal condition contract, no training."""
import argparse
import ast
import json
import pickle
import platform
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from realtime_app.learning_flow_mesh.body import BodyDecoder
from realtime_app.learning_flow_mesh.hand_conditions import (
    FINGER_ORDER,
    SIDE_ORDER,
    HandConvention,
    native_hand_tokens,
)
from realtime_app.learning_flow_mesh.interfaces import SixDLayout, decode_rotation
from realtime_app.pose_app.wilor_mano_prior import (
    audit_assets,
    encode_pca,
    read_parameter_view,
)

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")
MAIN = Path("research_records/engineering_validation/G20261005_reference_weight_fit_v1/runs/reference25")


def original_shared_initialization(coeff, weights, hand_index):
    """Run the exact reviewed offline assignment; not a new online initializer."""
    path = Path("realtime_app/pose_app/smplh_fitting/pipeline.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Attribute)
                and node.test.attr == "shared_hand_pose"):
            selected.extend(statement for statement in node.body
                            if isinstance(statement, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "initial"
                                    for target in statement.targets))
    if len(selected) != 1:
        raise ValueError("original shared initial assignment missing or ambiguous")
    scope = {"coeff": coeff, "w": weights, "h": hand_index}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 -- exact reviewed source assignment
    return scope["initial"], {"path": path.as_posix(), "line": selected[0].lineno}


def encode_tokens(rotations, weights, ids, views, parents):
    return native_hand_tokens(torch.tensor(rotations, dtype=torch.float32)[None],
                              torch.tensor(weights, dtype=torch.float32)[None],
                              source_frame_ids=ids, target_frame_ids=ids, view_available=views,
                              convention=HandConvention.SMPLH_PARENT_LOCAL_NATIVE,
                              side_order=SIDE_ORDER, parents=parents)


def run(output):
    torch.set_num_threads(2)
    options = json.loads((MAIN / "resolved_options.json").read_text(encoding="utf-8"))
    expected = json.loads((MAIN / "fit/wilor_mano_parameter_audit.json").read_text(encoding="utf-8"))
    with np.load(BASE / "architecture_h2_results_v3.npz", allow_pickle=False) as archive:
        source = {key: archive[key].copy() for key in archive.files}
    assert np.array_equal(source["frame_ids"], np.arange(360))
    assert np.array_equal(source["original_pair_ids"], np.arange(1157, 1517))
    # Original body reader's legacy absolute package import is scoped to this CLI.
    sys.path.insert(0, str(Path("realtime_app").resolve()))
    from pose_app.body_observations import raw_side

    body = {side: raw_side(Path(options[f"{side}_raw"]), side) for side in SIDE_ORDER}
    rotations, weights, audits = [], [], []
    for view, side in enumerate(SIDE_ORDER):
        observations = np.stack([body[side][i] for i in range(360)])
        # On H2-selected frames the original body reader must select the same person.
        matched = source["selected_people"][:, view] >= 0
        assert (source["selected_people"][matched, view] == 0).all()
        assert np.array_equal(observations[matched, :, :2], source["raw_xy"][matched, view])
        # pipeline.py casts the raw body reader before box/wrist association.
        # Keep the float64 source comparison above, then replay its exact dtype.
        observations = observations.astype(np.float32)
        r, w, audit = read_parameter_view(Path(options[f"wilor_{side}"]), observations, (1920, 1080))
        assert audit == expected[f"{side}_camera"], side
        rotations.append(r)
        weights.append(w)
        audits.append(audit)
    rotations = np.stack(rotations, axis=1)
    weights = np.stack(weights, axis=1)
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**config["assets"])  # installs existing legacy loader compatibility
    files = [options[key] for key in ("smplh_model", "mano_left", "mano_right", "canonical_mano_right")]
    asset_data = []
    for path in files:
        with Path(path).open("rb") as handle:
            asset_data.append(pickle.load(handle, encoding="latin1"))
    asset_report = audit_assets(*asset_data)
    assert asset_report == expected["assets"]
    ids = torch.tensor(source["frame_ids"].astype(np.int64))[None]
    views = torch.tensor(source["selected_people"] >= 0)[None]
    token = encode_tokens(rotations, weights, ids, views, decoder.parents)
    restored, valid = decode_rotation(token.tokens[..., :6], SixDLayout.PACKED_COLUMNS)
    assert torch.equal(valid, token.present)
    reconstruction = float((restored[token.present] - token.local_rotations[token.present]).abs().max())
    assert reconstruction < 1e-5
    assert torch.isnan(token.tokens[~token.present]).all()
    assert not token.admitted[~views].any()
    assert np.array_equal(source["strict_gate"].sum(), 3031)
    # This probe changes future native hypotheses only, never source files.
    future = rotations.copy()
    perturbation = Rotation.from_rotvec([.4, -.15, .2]).as_matrix()
    future[120:] = perturbation @ future[120:]
    altered = encode_tokens(future, weights, ids, views, decoder.parents)
    torch.testing.assert_close(token.tokens[:, :120], altered.tokens[:, :120], atol=0, rtol=0, equal_nan=True)
    disagreement, offline_dependency = {}, {}
    original_assignment = None
    for h, side in enumerate(SIDE_ORDER):
        paired = token.admitted[0, :, 0, h].numpy() & token.admitted[0, :, 1, h].numpy()
        delta = rotations[paired, 0, h].swapaxes(-1, -2) @ rotations[paired, 1, h]
        angles = np.degrees(Rotation.from_matrix(delta.reshape(-1, 3, 3)).magnitude()).reshape(-1, 15)
        disagreement[side] = {"paired_frames": int(paired.sum()), "median_deg": float(np.median(angles)),
                              "p95_deg": float(np.percentile(angles, 95)),
                              "statistical_unit": "pooled 15 native joint rotations over paired admitted frames",
                              "per_finger_p95_deg": {finger: float(np.percentile(angles[:, 3*f:3*f+3], 95))
                                                     for f, finger in enumerate(FINGER_ORDER)},
                              "boundary": "cross-view model consistency, not independent error"}
        mano = asset_data[1 + h]
        components = np.asarray(mano["hands_components"], np.float32)[:options["hand_pca_comps"]]
        mean = np.asarray(mano["hands_mean"], np.float32)
        scale = np.maximum(np.linalg.norm(components, axis=1), 1e-6)
        coeff, _errors = encode_pca(rotations[:, :, h], components, mean, scale)
        future_coeff, _ = encode_pca(future[:, :, h], components, mean, scale)
        initial, original_assignment = original_shared_initialization(coeff, weights, h)
        changed, _ = original_shared_initialization(future_coeff, weights, h)
        change = float(np.max(np.abs(changed - initial)))
        assert change > 1e-6
        offline_dependency[side] = {"future_changes_from_frame": 120,
                                    "shared_PCA_initializer_max_coefficient_change": change,
                                    "boundary": "artificial information dependency; original mainline is intentionally offline"}
    counts = {}
    for view, camera in enumerate(SIDE_ORDER):
        counts[camera] = {side: {"original_source_accepted": int((weights[:, view, h] > 0).sum()),
                                "H3_input_admitted": int(token.admitted[0, :, view, h].sum()),
                                "source_accepted_but_H2_view_unavailable": int(((weights[:, view, h] > 0) & ~views[0, :, view].numpy()).sum())}
                         for h, side in enumerate(SIDE_ORDER)}
    report = {"status": "native_hand_source_and_causal_token_contract_passed", "protocol": "architecture_h4_protocol.json",
              "training_executed": False, "body_fit_run": False, "detector_run": False,
              "source_options": (MAIN / "resolved_options.json").as_posix(),
              "source_paths": [options[f"wilor_{side}"] for side in SIDE_ORDER],
              "mainline_audit_exact_parity": True, "asset_audit_exact_parity": True,
              "association_body_dtype": "float32 matching pipeline.py; float64 raw H2 pixel parity checked first",
              "frames": 360, "physical_pair_range": [1157, 1516], "source_acceptance_and_candidate_scope": counts,
              "source_reason_counts": {camera: dict(Counter(row["reason"] for row in audits[v])) for v, camera in enumerate(SIDE_ORDER)},
              "strict_3031_gate_and_rejected_states_preserved": True,
              "tokens_shape": list(token.tokens.shape), "native_nodes": token.node_ids.tolist(),
              "anatomical_sides": list(SIDE_ORDER), "finger_order": list(FINGER_ORDER),
              "confidence_kind": "detector_box_not_joint", "source_zero_weight_identity_never_admitted": True,
              "float32_rotation_roundtrip_max_error": reconstruction, "per_frame_future_prefix_difference": 0.,
              "original_offline_initial_assignment": original_assignment, "offline_future_dependency": offline_dependency,
              "view_disagreement": disagreement, "assets": asset_report,
              "global_orientation_shape_camera_projected_points_transferred": False,
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "dtype": "float32"},
              "boundary": "Native hand model hypotheses and typed input only. No learned hand branch, training, accuracy/contact/world-position or realtime claim; H3/core/mainline unchanged."}
    arrays = {"source__" + key: value for key, value in source.items()}
    arrays.update(source_native_rotations=rotations, source_box_confidence=weights,
                  source_native_audit_json=np.array(json.dumps(audits, ensure_ascii=False)),
                  tokens=token.tokens.numpy(), token_present=token.present.numpy(), local_rotations=token.local_rotations.numpy(),
                  source_native_present=token.source_present.numpy(), hand_admitted=token.admitted.numpy(),
                  native_frame_ids=token.frame_ids.numpy(), node_ids=token.node_ids.numpy(),
                  metadata=np.array(json.dumps(report, ensure_ascii=False)))
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("preserve earlier result, select a new output version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 -- preserve source/contract failures
        result = {"status": "native_hand_contract_failed", "training_executed": False,
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
