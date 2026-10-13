"""No-weight engineering replay from existing same-source H2 pixel records."""
import argparse
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from realtime_app.learning_flow_mesh.rotations import matrix_to_six

from .candidate_store import infer
from .data_firewall import Normalization
from .evaluate import online_metrics
from .joint_flow import JointFlow
from .schema import Observation, TemporalState, Topology

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")
SAVED = Path("research_records/engineering_validation/G20261003_repositioned360_cold_smplh_v2")


def load_saved_observations(source=BASE / "architecture_h2_results_v3.npz", mapping=SAVED / "frames.json",
                            protocol=SAVED / "protocol.json", *, device="cpu", dtype=torch.float32):
    metadata = json.loads(Path(protocol).read_text(encoding="utf-8"))
    physical = metadata["metadata"]["calibration_camera_mapping"]["camera_registry_resolution"]
    camera_ids = tuple(physical[side]["expected_instance_id"] for side in ("left", "right"))
    frames = json.loads(Path(mapping).read_text(encoding="utf-8"))
    names = ("frame_ids", "original_pair_ids", "raw_xy", "confidence", "observation_mask",
             "geometry_mask", "strict_gate", "selected_people", "strict_reasons_json",
             "quality_flags_json", "point_reasons_json", "ray_reasons_json",
             "ray_origins_m", "ray_directions_cam0")
    # Label-like diagnostic XYZ is deliberately never read from the archive.
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in names}
    ids = arrays["frame_ids"]
    if len(frames) != len(ids) or [int(f["index"]) for f in frames] != ids.tolist():
        raise ValueError("same-source frame mapping mismatch")
    if [int(f["pair_id"]) for f in frames] != arrays["original_pair_ids"].tolist():
        raise ValueError("original pair provenance mismatch")
    if not np.array_equal(arrays["geometry_mask"], np.broadcast_to(arrays["strict_gate"][:, None], (len(ids), 2, 17))):
        raise ValueError("immutable H2 strict geometry masks changed")
    clock = metadata["metadata"]["timestamp_type"]
    if clock != "host_read_return_perf_counter_ns":
        raise ValueError("unreviewed clock convention")
    observations = []
    def tensor(value):
        return torch.as_tensor(value, device=device, dtype=torch.bool if value.dtype == np.bool_ else dtype)
    for i, frame in enumerate(frames):
        point = json.loads(str(arrays["point_reasons_json"][i]))
        ray = json.loads(str(arrays["ray_reasons_json"][i]))
        strict = json.loads(str(arrays["strict_reasons_json"][i]))
        reasons = tuple(tuple(json.dumps({"point": point[v][j], "ray": ray[v][j],
                                         "strict": strict[j], "quality_flags": json.loads(str(arrays["quality_flags_json"][i, j]))})
                              for j in range(17)) for v in range(2))
        view_times = tuple(int(frame[f"{side}_host_return_timestamp_ns"])/1e9 for side in ("left", "right"))
        # H2 observation_mask is per-view finite, in-bounds pixel support.
        # Its strict stereo gate belongs only to geometry; failing triangulation
        # must not erase usable 2D evidence in either associated view.
        obs = Observation(tensor(arrays["raw_xy"][i]), tensor(arrays["confidence"][i]),
                          tensor(arrays["observation_mask"][i]),
                          tensor(arrays["observation_mask"][i].copy()),
                          tensor(arrays["geometry_mask"][i]),
                          torch.tensor(arrays["selected_people"][i] >= 0, device=device, dtype=torch.bool),
                          tensor(arrays["ray_origins_m"][i]), tensor(arrays["ray_directions_cam0"][i]),
                          reasons, camera_ids, "repositioned360_person", "capture20260927_pairs1157_1516",
                          int(ids[i]), sum(view_times)/2, str(Path(source).resolve()),
                          timestamp_kind=clock, view_timestamps=view_times).validate()
        observations.append(obs)
    if any(b.timestamp <= a.timestamp for a, b in pairwise(observations)):
        raise ValueError("physical host clock must increase; cannot replace it with FPS")
    return observations, {"source": str(Path(source).resolve()), "mapping": str(Path(mapping).resolve()),
                          "protocol": str(Path(protocol).resolve()), "camera_ids": list(camera_ids),
                          "timestamp_kind": clock, "clock_warning": metadata["metadata"]["timestamp_warning"],
                          "original_pairs": [int(arrays["original_pair_ids"][0]), int(arrays["original_pair_ids"][-1])],
                          "strict_points": int(arrays["strict_gate"].sum())}


def fixture_normalization(topology, *, device="cpu", dtype=torch.float32):
    mean = torch.cat((matrix_to_six(torch.eye(3)).repeat(topology.nodes), torch.tensor([0., 0., 3.]), torch.zeros(10)))
    return Normalization(mean.to(device=device, dtype=dtype), torch.ones(topology.dimensions, device=device, dtype=dtype)*.03,
                         topology, "analytic_fixture_identity_R_Tz3_scale0.03_not_train_statistics")


def replay(observations, model, normalization, *, k=32, steps=8, chunk=64, cache_dir=None):
    state, stores, missing_hidden_error = None, [], 0.0
    with torch.no_grad():
        for i, obs in enumerate(observations):
            previous = state
            store, state = infer(model, normalization, obs, state, k=k, steps=steps)
            if not obs.admitted().any() and previous is not None:
                missing_hidden_error = max(missing_hidden_error, float((state.hidden-previous.hidden).abs().max()))
            if cache_dir is not None:
                store.save(cache_dir / f"frame_{obs.frame:04d}.npz")
            stores.append(store)
            if (i+1) % chunk == 0:
                state = TemporalState.from_dict(json.loads(json.dumps(state.to_dict())), device=obs.xy.device, dtype=obs.xy.dtype)
    return stores, state, missing_hidden_error


def run(*, device="cpu", k=32, steps=8, cache_dir=None, chunk_check=False):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = JointFlow().to(device).eval()
    observations, provenance = load_saved_observations(device=device)
    normal = fixture_normalization(Topology(), device=device)
    stores, state, error = replay(observations, model, normal, k=k, steps=steps, cache_dir=cache_dir)
    counts = Counter(r for store in stores for reasons in store.reasons for r in reasons)
    observed = [bool(obs.admitted().any()) for obs in observations]
    published = [bool(store.actions.any()) for store in stores]
    report = {"scope": "random initialized G0 engineering replay; no learned weights or accuracy",
              "frames": len(observations), "k": k, "steps": steps, "source": provenance,
              "observed_frames": sum(observed), "unobserved_frames": len(observed)-sum(observed),
              "action_available_frames": sum(published), "candidate_rejection_counts": dict(counts),
              "missing_hidden_max_change": error, "final_state": state.to_dict(),
              "online": online_metrics([o.timestamp for o in observations], observed, published),
              "all_observation_reasons_retained": all(store.provenance["rejection_reasons"] == obs.rejection_reasons
                                                       for store, obs in zip(stores, observations)),
              "cache_dir": str(cache_dir) if cache_dir else None, "training_executed": False,
              "checkpoint_loaded": False, "mainline_changed": False}
    if chunk_check:
        second, _, _ = replay(observations, model, normal, k=k, steps=steps, chunk=11)
        difference = max(float((a.raw-b.raw).abs().max()) for a, b in zip(stores, second))
        report["chunk64_vs_chunk11_max_error"] = difference
        if difference > 1e-6:
            raise AssertionError("chunk/state replay mismatch")
    if error != 0 or any(pub and not obs for pub, obs in zip(published, observed)):
        raise AssertionError("missing/publication invariant violated")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--chunk-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(device=args.device, k=args.k, steps=args.steps, cache_dir=args.cache_dir, chunk_check=args.chunk_check)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
