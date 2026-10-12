"""Audit official FrameFlow rotation functions against fixed-endpoint controls."""
import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .tangent import shortest_log, skew

FUNCTIONS = (
    "_broadcast_identity", "skew_matrix_exponential_map", "rotvec_to_rotmat",
    "rotmat_to_rotvec", "angle_from_rotmat", "vector_to_skew_matrix",
    "skew_matrix_to_vector", "rot_transpose", "rot_mult", "calc_rot_vf", "geodesic_t",
)


def author_functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in selected} != set(FUNCTIONS):
        raise ValueError("official rotation function snapshot is incomplete")
    namespace = {"torch": torch, "np": np, "Tuple": tuple}
    # Only the inspected pure FunctionDef nodes are compiled; no module imports.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    ranges = {node.name: [node.lineno, node.end_lineno] for node in selected}
    return namespace, ranges


def finite_json(value):
    if isinstance(value, torch.Tensor):
        return finite_json(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nonfinite_NaN" if math.isnan(value) else "nonfinite_Infinity"
    return value


def contracts(point, target, tolerance):
    error = (point-target).abs().flatten(-2).amax(-1)
    orthogonal = (point.transpose(-1, -2)@point-torch.eye(3, dtype=point.dtype)).abs().flatten(-2).amax(-1)
    determinant = (torch.linalg.det(point)-1).abs()
    finite = torch.isfinite(point).all(-1).all(-1)
    passed = finite & (error <= tolerance) & (orthogonal <= tolerance) & (determinant <= tolerance)
    return {"endpoint_max": float(error.max()), "orthogonality_max": float(orthogonal.max()),
            "determinant_max": float(determinant.max()), "all_passed": bool(passed.all()),
            "failed_flat_indices": (~passed).flatten().nonzero().flatten().tolist()}, passed, error


def fixed_endpoint_log(rotations):
    # An independent detached CPU target-preparation control. This is not a
    # differentiable endpoint operator and is never used by the current model.
    vectors = Rotation.from_matrix(rotations.detach().cpu().double().numpy().reshape(-1, 3, 3)).as_rotvec()
    return torch.from_numpy(vectors.copy()).to(rotations).reshape(*rotations.shape[:-2], 3)


def run(source):
    torch.set_num_threads(1)
    official, ranges = author_functions(source)
    axes = torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [1., -2, 3], [-1., 2, -3]], dtype=torch.float64)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    theta = torch.tensor([0., 1e-8, .4, math.pi-.02, math.pi-.005, math.pi-.0005, math.pi], dtype=torch.float64)
    vectors = (axes[:, None]*theta[None, :, None]).reshape(-1, 3)
    targets64 = torch.matrix_exp(skew(vectors))
    generator = torch.Generator().manual_seed(7)
    quaternion = torch.randn(1024, 52, 4, generator=generator, dtype=torch.float64)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    # SciPy uses xyzw; the sampled Gaussian is invariant to this permutation.
    haar64 = torch.from_numpy(Rotation.from_quat(quaternion.numpy().reshape(-1, 4)).as_matrix()).reshape(1024, 52, 3, 3)
    arrays = {"source_quaternion_xyzw": quaternion.numpy(), "haar_rotations64": haar64.numpy()}
    results = []
    for dtype in (torch.float64, torch.float32):
        name = str(dtype).split(".")[-1]
        tolerance = 1e-10 if dtype == torch.float64 else 3e-5
        targets = targets64.to(dtype)
        identity = torch.eye(3, dtype=dtype).expand_as(targets)
        original_log = official["rotmat_to_rotvec"](targets)
        reference_log = fixed_endpoint_log(targets)
        original_endpoint = official["geodesic_t"](1., targets, identity)
        reference_endpoint = torch.matrix_exp(skew(reference_log))
        original_start, _, _ = contracts(official["geodesic_t"](0., targets, identity), identity, tolerance)
        reference_start, _, _ = contracts(torch.matrix_exp(0*skew(reference_log)), identity, tolerance)
        original, original_pass, original_error = contracts(original_endpoint, targets, tolerance)
        reference, reference_pass, _ = contracts(reference_endpoint, targets, tolerance)
        rows = [{"axis": axes[index//len(theta)].tolist(), "angle_rad": float(theta[index % len(theta)]),
                 "target": target, "author_log": original_log[index],
                 "author_endpoint": original_endpoint[index], "author_endpoint_error": float(original_error[index]),
                 "author_contract_passed": bool(original_pass[index]),
                 "reference_contract_passed": bool(reference_pass[index])}
                for index, target in enumerate(targets)]
        time_controls = []
        for case, index in (("identity", 0), ("regular", 2), ("near_pi", 5)):
            for implementation, omega in (("author", original_log[index:index+1].detach()),
                                           ("fixed_endpoint_control", reference_log[index:index+1].detach())):
                def path(time, omega=omega, implementation=implementation):
                    if implementation == "author":
                        return official["rotvec_to_rotmat"](time*omega)
                    return torch.matrix_exp(time*skew(omega))
                point, velocity = torch.func.jvp(path, (theta.new_tensor(.4).to(dtype),),
                                                (theta.new_tensor(1.).to(dtype),))
                difference = float((velocity-point@skew(omega)).abs().max())
                finite = bool(torch.isfinite(velocity).all())
                time_controls.append({"case": case, "implementation": implementation,
                                      "point": point, "time_velocity": velocity,
                                      "closed_form_max_error": difference,
                                      "passed": finite and difference <= tolerance})
        haar = haar64.to(dtype)
        fixed_target = torch.eye(3, dtype=dtype).expand_as(haar)
        original_haar_log = official["calc_rot_vf"](haar, fixed_target)
        original_haar_endpoint = official["geodesic_t"](1., fixed_target, haar, rot_vf=original_haar_log)
        haar_original, haar_pass, haar_error = contracts(original_haar_endpoint, fixed_target, tolerance)
        reference_haar_log = fixed_endpoint_log(haar.transpose(-1, -2))
        haar_reference, _, _ = contracts(haar@torch.matrix_exp(skew(reference_haar_log)), fixed_target, tolerance)
        try:
            shortest_log(haar)
            project_rejection = None
        except ValueError as error:
            project_rejection = str(error)
        arrays.update({name+"_author_haar_log": original_haar_log.numpy(),
                       name+"_author_haar_endpoint": original_haar_endpoint.numpy(),
                       name+"_author_haar_error": haar_error.numpy(),
                       name+"_author_haar_contract": haar_pass.numpy(),
                       name+"_reference_haar_log": reference_haar_log.numpy()})
        reference_times = [item["passed"] for item in time_controls
                           if item["implementation"] == "fixed_endpoint_control"]
        results.append({"dtype": name, "tolerance": tolerance, "fixed_case_count": len(rows),
                        "author_start": original_start, "reference_start": reference_start,
                        "author_fixed": original, "reference_fixed": reference, "cases": rows,
                        "time_controls": time_controls, "author_Haar": haar_original,
                        "reference_Haar": haar_reference, "current_local_chart_rejection": project_rejection,
                        "reference_contract_passed": reference_start["all_passed"] and reference["all_passed"] and haar_reference["all_passed"] and all(reference_times)})
    return finite_json({"source_path": str(source), "function_ranges": ranges,
                        "execution": "complete original FunctionDef nodes only, not full author package/model",
                        "device": "cpu", "torch_version": torch.__version__, "training": False,
                        "Haar_count": 1024*52, "seed": 7, "resampled_or_dropped": 0, "results": results,
                        "boundary": "fixed endpoints/time derivatives only; no endpoint gradient, speed, calibration or efficacy claim"}), arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.output.with_suffix(".npz")
    if args.output.exists() or archive.exists():
        raise FileExistsError("refusing to overwrite existing audit evidence")
    result, arrays = run(args.source)
    with archive.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    summary = [{"dtype": row["dtype"], "author_fixed_failed": len(row["author_fixed"]["failed_flat_indices"]),
                "author_fixed_max_error": row["author_fixed"]["endpoint_max"],
                "author_Haar_failed": len(row["author_Haar"]["failed_flat_indices"]),
                "author_Haar_max_error": row["author_Haar"]["endpoint_max"],
                "reference_Haar_max_error": row["reference_Haar"]["endpoint_max"],
                "reference_contract_passed": row["reference_contract_passed"]}
               for row in result["results"]]
    print(json.dumps(summary))
    if not all(row["reference_contract_passed"] for row in result["results"]):
        raise AssertionError("fixed-endpoint control failed; saved results must not be marked passed")


if __name__ == "__main__":
    main()
