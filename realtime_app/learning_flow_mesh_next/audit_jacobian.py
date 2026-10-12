"""Real licensed surface/fisheye derivative audit without neural generation."""
import argparse
import json
import time
from pathlib import Path

import torch

from .body import SurfaceDecoder
from .geometry_information import local_parameters
from .geometry_observation import project
from .schema import Topology
from .tangent import TangentScale
from .tests.test_aggregate import rig


def run(device):
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    torch.set_num_threads(1)
    dtype = torch.float64
    assets = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text())["assets"]
    decoder = BodyDecoder(**assets).to(device=device, dtype=dtype)
    regressor = torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")),
                             device=device, dtype=dtype)
    surface = SurfaceDecoder(decoder, regressor)
    cameras = rig()
    scale = TangentScale.fixture(Topology(52)).values.to(device=device, dtype=dtype)
    m = len(scale)
    report = {"device": device, "dtype": "float64", "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
              "assets": decoder.asset_paths, "regressor": "models/smpl/J_regressor_coco.npy",
              "camera_source": "tests.test_aggregate.rig artificial fisheye D=[.03,-.01,.001,0]",
              "information_comparison": "unweighted J.T@J only; not calibrated precision-weighted information",
              "anchors": [], "training": False, "checkpoint_loaded": False,
              "boundary": "actual surface on artificial anchors; no efficacy or whole-chain speed evidence"}

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for index in range(2):
        rotations = torch.eye(3, device=device, dtype=dtype).repeat(52, 1, 1)
        anchor = {"rotations": rotations, "translation": torch.tensor([0., 0., 3.], device=device, dtype=dtype),
                  "beta": torch.full((10,), index*.15, device=device, dtype=dtype)}
        if index:
            displacement = torch.zeros(1, m, device=device, dtype=dtype)
            displacement[0, 9] = .2
            anchor["rotations"] = local_parameters(anchor, displacement, scale)[0][0].detach()
        evaluations = 0

        def pixels(displacement, anchor=anchor):
            nonlocal evaluations
            evaluations += len(displacement)
            r, t, b = local_parameters(anchor, displacement, scale)
            points = surface(r, t, b)["coco17"]
            projected = [project(points, camera) for camera in cameras]
            if not all(valid.all() for _, valid in projected):
                raise ValueError("anchor_or_perturbation_projection_invalid")
            return torch.stack([uv for uv, _ in projected], 1).flatten(1)

        zero = torch.zeros(m, device=device, dtype=dtype)

        def finite(step=1e-4, zero=zero, pixels=pixels):
            with torch.no_grad():
                pixels(zero[None])
                basis = torch.eye(m, device=device, dtype=dtype)*step
                out = torch.cat([pixels(part) for part in torch.cat((basis, -basis)).split(16)])
                return ((out[:m]-out[m:])/(2*step)).T

        def reverse(vectorized, zero=zero, pixels=pixels):
            with torch.enable_grad():
                return torch.autograd.functional.jacobian(
                    lambda delta: pixels(delta[None])[0], zero,
                    create_graph=False, vectorize=vectorized)

        results, values = {}, {}
        for name, method in (("finite", finite), ("reverse_scalar", lambda: reverse(False)),
                             ("reverse_vectorized", lambda: reverse(True))):
            try:
                method()  # Warmup; no timing selection.
                records = []
                for _ in range(3):
                    evaluations = 0
                    sync()
                    baseline = torch.cuda.memory_allocated() if device == "cuda" else None
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    value = method()
                    sync()
                    records.append({"ms": (time.perf_counter()-start)*1000,
                                    "decoder_states": evaluations,
                                    "peak_extra_cuda_bytes": torch.cuda.max_memory_allocated()-baseline
                                    if device == "cuda" else None})
                values[name] = value.detach()
                results[name] = {"records": records, "finite": bool(torch.isfinite(value).all()),
                                 "median_ms": sorted(r["ms"] for r in records)[1]}
            except (RuntimeError, ValueError) as error:
                results[name] = {"failure": f"{type(error).__name__}:{error}"}
        reference = values.get("finite")
        if reference is not None:
            H = reference.T@reference
            for name, value in values.items():
                results[name]["J_relative_frobenius"] = float((value-reference).norm()/reference.norm())
                results[name]["H_relative_frobenius"] = float((value.T@value-H).norm()/H.norm())
        # AD derivative omits neighborhood guards; exact existing guard needs the same probes.
        if "reverse_vectorized" in values:
            start = time.perf_counter()
            finite()
            reverse(True)
            sync()
            results["reverse_vectorized_with_existing_neighborhood_guard"] = {
                "single_ms": (time.perf_counter()-start)*1000,
                "note": "runs existing probe evaluation plus reverse Jacobian; cannot count raw AD as guard-equivalent"}
        report["anchors"].append({"index": index, "beta": index*.15,
                                  "local_node3_standardized_rotation_x": .2 if index else 0,
                                  "results": results})
    report["numerical_gate"] = all(
        a["results"][name].get("finite", False)
        and a["results"][name].get("J_relative_frobenius", 1) < 1e-4
        and a["results"][name].get("H_relative_frobenius", 1) < 1e-4
        for a in report["anchors"] for name in ("reverse_scalar", "reverse_vectorized"))
    report["default_replacement_authorized"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.device)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
