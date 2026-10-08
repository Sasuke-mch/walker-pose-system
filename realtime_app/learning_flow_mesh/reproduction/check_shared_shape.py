"""Verify H1 with unchanged real FlowMesh encoder/head and real SMPL-H assets.

Artificial proxy observations, random weights and geometry contracts only.
No training, real accuracy claim or runtime-mainline change.
"""
import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from ..body import BodyDecoder
from ..data import OBSERVATION_KEYS, observed_inputs
from ..model import FlowMesh, ModelConfig
from ..shared_shape import PrefixSharedShape, encode_with_global_features
from ..synthetic import generate


@torch.no_grad()
def run(output):
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    torch.set_num_threads(2)
    body = BodyDecoder(**config["assets"])
    arrays = generate(body, count=2, frames=13, seed=31, split="test", device="cpu")
    source_metadata = json.loads(str(arrays["metadata"].item()))
    # The generator includes broadcast views (notably camera origins). Own the
    # storage before injecting missing frames, or one write can corrupt a clip.
    batch = {key: torch.from_numpy(value).clone() for key, value in arrays.items()
             if key in OBSERVATION_KEYS + ("images", "image_features")}
    # Explicit all-missing probes; preserve frame positions instead of deleting them.
    missing = torch.zeros(2, 13, dtype=torch.bool)
    missing[0, 1] = True
    missing[1, :3] = True
    for key in OBSERVATION_KEYS + ("images", "image_features"):
        if key in batch:
            batch[key][missing] = False if batch[key].dtype == torch.bool else torch.nan
    torch.manual_seed(19)
    model = FlowMesh(ModelConfig(**config["model"]), body.parents).eval()
    snapshot = {key: value.clone() for key, value in model.state_dict().items()}
    candidate = PrefixSharedShape(model.encoder.shape_head, 8)
    names = tuple(source_metadata["sequence_ids"])
    frame_ids = torch.arange(13).expand(2, -1).clone()
    with torch.no_grad():
        direct = model.encoder(observed_inputs(batch))
        encoded, global_features = encode_with_global_features(model.encoder, observed_inputs(batch))
        assert torch.equal(direct["condition"], encoded["condition"])
        assert torch.equal(direct["translation"], encoded["translation"])
        # Presence is used only for this controlled fixture, not as a physical quality gate.
        admitted = encoded["frame_observed"]
        calibration = candidate(global_features, admitted, frame_ids, names)
        stream_betas, stream_ready, hidden, state = [], [], None, None
        for frame in range(13):
            one = {k: v[:, frame:frame+1] for k, v in observed_inputs(batch).items()}
            e, g = encode_with_global_features(model.encoder, one, hidden)
            hidden = e["hidden"]
            step = candidate(g, e["frame_observed"], frame_ids[:, frame:frame+1], names, state)
            state = step.state
            stream_betas.append(step.betas)
            stream_ready.append(step.ready)
        streamed = torch.cat(stream_betas, dim=1)
        assert torch.equal(calibration.ready, torch.cat(stream_ready, dim=1))
        torch.testing.assert_close(calibration.betas, streamed, atol=1e-6, rtol=1e-6, equal_nan=True)
        future = {k: v.clone() for k, v in observed_inputs(batch).items()}
        future["xy"][:, 11:] += 0.3
        future["images"][:, 11:] = 1 - future["images"][:, 11:]
        altered, changed_global = encode_with_global_features(model.encoder, future)
        future_shape = candidate(changed_global, altered["frame_observed"], frame_ids, names)
        torch.testing.assert_close(calibration.betas, future_shape.betas, atol=0, rtol=0, equal_nan=True)
        assert torch.isnan(calibration.betas[~calibration.ready]).all()
        # Pose transport stays exactly the original network (shape is separate here).
        transported = model.transport(encoded, steps=8, seed=7)
        assert transported["rotation_valid"].all()
        mesh_checks = []
        for row in range(2):
            start = int(calibration.ready[row].nonzero()[0])
            mesh = body(transported["rotations"][row:row+1, start:],
                        calibration.betas[row:row+1, start], encoded["translation"][row:row+1, start:])
            assert torch.isfinite(mesh["vertices"]).all()
            available_beta = calibration.betas[row, calibration.ready[row]]
            assert torch.equal(available_beta, available_beta[:1].expand_as(available_beta))
            mesh_checks.append({"sequence_id": names[row], "first_available_frame": start,
                                "vertices_shape": list(mesh["vertices"].shape), "finite": True})
        assert all(torch.equal(snapshot[key], value) for key, value in model.state_dict().items())
    ready_difference = float((calibration.betas[calibration.ready] - streamed[calibration.ready]).abs().max())
    # First row's first frame is present; this control isolates only the readout.
    first_control = PrefixSharedShape(model.encoder.shape_head, 1)(
        global_features[:1], admitted[:1], frame_ids[:1], names[:1])
    # Same head and same GEMM batch shape reproduce the original formula exactly.
    expected_single = model.encoder.shape_head(global_features[:1, 0])
    torch.testing.assert_close(first_control.betas[:, 0], expected_single, atol=0, rtol=0)
    torch.testing.assert_close(model.encoder.shape_head(global_features[:, 0]), encoded["betas"], atol=0, rtol=0)
    first_control_error = float((first_control.betas[:, 0] - encoded["betas"][:1]).abs().max())
    # GEMM B=1 versus B=2 roundoff is recorded, not reported as zero or ignored.
    assert first_control_error < 1e-6
    beta_change = float((calibration.state.betas[:1] - encoded["betas"][:1]).norm())
    npz_path = output.with_suffix(".npz")
    if npz_path.exists():
        raise FileExistsError(npz_path)
    np.savez_compressed(npz_path, betas=calibration.betas.detach().numpy(),
                        ready=calibration.ready.numpy(), admitted=admitted.numpy(),
                        frame_ids=frame_ids.numpy(), global_features=global_features.numpy(),
                        source_metadata=np.array(json.dumps(source_metadata)))
    return {"status": "isolated_H1_structure_passed", "training_executed": False,
            "weights": "random_initialization_seed19", "fixture": source_metadata,
            "fixture_missing_frames_by_row": [[1], [0, 1, 2]], "calibration_valid_frames": 8,
            "same_original_shape_head": candidate.head is model.encoder.shape_head,
            "baseline_parameters_unchanged": True, "original_condition_translation_unchanged": True,
            "stream_batch_beta_max_difference": ready_difference,
            "future_probe_max_beta_difference": 0., "unavailable_beta_entries_are_nan": True,
            "missing_frames_excluded_from_count": True, "frozen_beta_constant": True,
            "mesh_checks": mesh_checks, "first_frame_control_matches_original_beta": True,
            "first_frame_control_different_GEMM_batch_max_error": first_control_error,
            "candidate_vs_first_frame_beta_l2_structure_difference_not_accuracy": beta_change,
            "pose_transport_nfe_unchanged": 8, "artifact": npz_path.as_posix(),
            "boundary": "Random synthetic structure only. Prefix pooling changes readout, not proven beta accuracy. Eager hook adapter not compiled/concurrent deployment; no mainline integration or retroactive meshes for unavailable prefix."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 -- retain candidate failures with a nonzero exit
        result = {"status": "isolated_H1_structure_failed", "training_executed": False,
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
