"""Inventory exact author mesh prerequisites; never silently substitute assets."""
import argparse
import inspect
import json
import subprocess
from pathlib import Path

import smplx
import yaml

from .run_original import ROOT


def prerequisite(path, role):
    path = Path(path)
    return {"path": path.as_posix(), "role": role, "available": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None}


def mesh_resource_audit():
    neutral_name = "SMPL_NEUTRAL.pkl"
    plans = []
    pro_path = ROOT / "ProHMR/prohmr/configs/prohmr.yaml"
    pro = yaml.safe_load(pro_path.read_text(encoding="utf-8"))["SMPL"]
    assert pro["GENDER"] == "neutral"
    plans.append(("P04", [pro_path], [
        (ROOT / "ProHMR" / pro["MODEL_PATH"] / neutral_name, "released neutral SMPL"),
        (ROOT / "ProHMR" / pro["JOINT_REGRESSOR_EXTRA"], "published OpenPose plus J19 regressor"),
        (ROOT / "ProHMR" / pro["MEAN_PARAMS"], "original mean parameters, release equivalence required")]))
    hmr_path = ROOT / "HMR2/hmr2/configs_hydra/experiment/default.yaml"
    hmr = yaml.safe_load(hmr_path.read_text(encoding="utf-8"))["SMPL"]
    assert hmr["GENDER"] == "neutral"
    cache = Path.home() / ".cache/4DHumans/data"
    plans.append(("P07", [hmr_path], [
        (cache / "smpl" / neutral_name, "released neutral SMPL in configured user cache"),
        (cache / "SMPL_to_J19.pkl", "published J19 regressor"),
        (cache / "smpl_mean_params.npz", "original release mean parameters")]))
    human = ROOT / "HuManiFlow"
    plans.append(("P05", [human / "configs/paths.py", human / "models/smpl.py", human / "scripts/run_predict.py"], [
        (human / "model_files/smpl" / neutral_name, "default neutral SMPL; CLI gender alternatives do not reproduce default"),
        (human / "model_files/J_regressor_extra.npy", "extra joints"),
        (human / "model_files/cocoplus_regressor.npy", "COCO-plus joints"),
        (human / "model_files/J_regressor_h36m.npy", "H36M evaluation joints")]))
    vibe = ROOT / "VIBE"
    plans.append(("P09", [vibe / "lib/core/config.py", vibe / "lib/models/smpl.py", vibe / "lib/models/spin.py"], [
        (vibe / "data/vibe_data" / neutral_name, "default smplx.SMPL neutral model"),
        (vibe / "data/vibe_data/J_regressor_extra.npy", "extra joints for original 49-point output"),
        (vibe / "data/vibe_data/smpl_mean_params.npz", "original iterative initialization")]))
    wham = ROOT / "WHAM"
    plans.append(("P10", [wham / "configs/constants.py", wham / "lib/models/smpl.py", wham / "lib/models/__init__.py"], [
        (wham / "dataset/body_models/smpl" / neutral_name, "build_body_model default neutral model"),
        (wham / "dataset/body_models/J_regressor_wham.npy", "WHAM COCO/extended landmarks"),
        (wham / "dataset/body_models/J_regressor_h36m.npy", "evaluation joints"),
        (wham / "dataset/body_models/J_regressor_feet.npy", "original foot geometry for trajectory/refinement")]))
    mb = ROOT / "MotionBERT"
    plans.append(("P12", [mb / "docs/mesh.md", mb / "lib/utils/utils_smpl.py", mb / "lib/model/model_mesh.py"], [
        (mb / "data/mesh" / neutral_name, "default neutral SMPL explicitly required in mesh documentation"),
        (mb / "data/mesh/J_regressor_extra.npy", "extra landmark joints"),
        (mb / "data/mesh/J_regressor_h36m_correct.npy", "specific corrected H36M regressor, not arbitrary COCO17"),
        (mb / "data/mesh/smpl_mean_params.npz", "original mesh-head initialization")]))
    checked = []
    for paper, sources, items in plans:
        if not all(path.is_file() for path in sources):
            raise FileNotFoundError(f"original prerequisite source missing: {paper}")
        required = [prerequisite(path, role) for path, role in items]
        checked.append({"id": paper, "source_anchors": [p.as_posix() for p in sources],
                        "default_mesh_ready": all(x["available"] for x in required), "requirements": required})
    # Targeted name inventory only; no raw imagery/datasets or licensed model contents scanned.
    found = subprocess.run(["rg", "--files", "--hidden", "--no-ignore", "models", "third_party",
        "--glob", "*SMPL_NEUTRAL*", "--glob", "*smpl_neutral*", "--glob", "*basicModel_neutral*",
        "--glob", "*SMPL_to_J19*", "--glob", "*J_regressor*", "--glob", "*cocoplus_regressor*"],
        capture_output=True, text=True, check=False)
    if found.returncode not in (0, 1):
        raise RuntimeError(f"resource name inventory failed: {found.stderr}")
    signature = inspect.signature(smplx.SMPL.__init__)
    assert signature.parameters["gender"].default == "neutral"
    local_assets = [prerequisite(p, role) for p, role in (
        ("models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl", "local licensed male SMPL, not neutral"),
        ("models/smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl", "local licensed female SMPL, not neutral"),
        ("third_party/WiLoR/mano_data/models/SMPLH_male.pkl", "local licensed SMPL-H, different parameterization"),
        ("models/smplx/SMPLX_NEUTRAL.pkl", "SMPL-X neutral, different vertex/parameter topology"),
        ("models/smpl/J_regressor_coco.npy", "project COCO17 surface regressor, not original regressors"),
        (ROOT / "smpl_mean_params_mmpose.npz", "public initialization fixture; original bundle equivalence unconfirmed"))]
    fixture = ROOT / "smpl_mean_params_mmpose.npz"
    comparisons = []
    for original in (ROOT / "ProHMR/data/smpl_mean_params.npz", cache / "smpl_mean_params.npz"):
        if fixture.is_file() and original.is_file():
            comparisons.append({"official_path": original.as_posix(), "fixture_path": fixture.as_posix(),
                                "raw_bytes_equal": original.read_bytes() == fixture.read_bytes()})
    if len(comparisons) == 2 and all(item["raw_bytes_equal"] for item in comparisons):
        local_assets[-1]["role"] = "public fixture byte-identical to acquired ProHMR/HMR2 means only; no automatic substitution"
    return {"status": "resource_inventory_completed_with_gaps", "training_executed": False,
            "specific_fixture_byte_comparisons": comparisons,
            "author_mesh_paths": checked, "smplx_default_gender_verified": "neutral",
            "targeted_resource_names_found": found.stdout.splitlines(), "existing_alternatives": local_assets,
            "automatic_substitutions": [], "mesh_loading_or_weights_loading_executed": False,
            "scope": "Six original SMPL-dependent default mesh paths only; readiness checks prerequisites, not full method completion",
            "neutral_acquisition_reference": (ROOT / "WHAM/fetch_demo_data.sh").as_posix(),
            "neutral_acquisition_policy": "Author script requires official SMPLify account/license. No credential lookup or authenticated download attempted",
            "remaining": "Original neutral asset/regressors/means, input preprocessing and full evaluation; independent other methods continue"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = mesh_resource_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for paper in report["author_mesh_paths"]:
        print(paper["id"], "default mesh ready:", paper["default_mesh_ready"],
              "missing prerequisites:", sum(not x["available"] for x in paper["requirements"]))


if __name__ == "__main__":
    main()
