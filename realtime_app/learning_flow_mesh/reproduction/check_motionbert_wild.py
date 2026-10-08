"""R8 original MotionBERT wild preprocessing, clip/TTA and pixel outputs."""
import argparse
import ast
import importlib
import json
import math
import sys
import traceback
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .run_original import ROOT
from .run_probability import forbid_weights, install_download_guard


def compile_nodes(path, nodes, scope):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 -- exact author AST


def original_reader():
    root = ROOT / "MotionBERT"
    sys.path.insert(0, str(root.resolve()))
    utils = importlib.import_module("lib.utils.utils_data")
    source = root / "lib/data/dataset_wild.py"
    scope = {"np": np, "json": json, "math": math, "Dataset": torch.utils.data.Dataset, "crop_scale": utils.crop_scale}
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    compile_nodes(source, nodes, scope)
    return scope


def run(output):
    torch.set_num_threads(2)
    torch.manual_seed(61)
    install_download_guard()
    torch.load = forbid_weights
    root = ROOT / "MotionBERT"
    sys.path.insert(0, str(root.resolve()))
    utils = importlib.import_module("lib.utils.utils_data")
    scope = original_reader()
    read = scope["read_input"]
    frames, joints = np.arange(248)[:, None], np.arange(26)[None, :]
    xy = np.stack((400 + 3 * joints + .3 * frames, 300 + 5 * joints + .2 * frames), -1)
    detections = np.concatenate((xy, np.full((248, 26, 1), .8)), axis=-1)
    records = [{"idx": 7, "image_id": str(247 - i), "keypoints": row.reshape(-1).tolist()}
               for i, row in enumerate(detections)]
    fixture = output.with_name(output.name.replace("_results_", "_input_"))
    with fixture.open("x", encoding="utf-8") as handle:
        json.dump(records, handle)
    dataset = scope["WildDetDataset"](fixture, clip_len=243, vid_size=(1920, 1080), scale_range=None, focus=7)
    assert len(dataset) == 2 and [len(dataset[i]) for i in range(2)] == [243, 5]
    mapping = [19, 12, 14, 16, 11, 13, 15, None, 18, 0, 17, 5, 7, 9, 6, 8, 10]
    expected = np.stack([((detections[:, 18] + detections[:, 19]) / 2 if j is None else detections[:, j])
                         for j in mapping], axis=1)
    expected[..., :2] = (expected[..., :2] - [960, 540]) / 540
    assert np.array_equal(dataset.vid_all, expected.astype(np.float32))
    pixel_input = dataset.vid_all.copy()
    cropped = read(fixture, None, [1, 1], 7)
    changed = detections.copy()
    changed[-1, 0, :2] += 4000
    # Exact original conversion and crop; only the future endpoint changed.
    changed_h36m = scope["halpe2h36m"](changed)
    future_crop = utils.crop_scale(changed_h36m, [1, 1]).astype(np.float32)
    crop_prefix_delta = float(np.abs(cropped[:120] - future_crop[:120]).max())
    future_pixel = changed_h36m.copy()
    future_pixel[..., :2] = (future_pixel[..., :2] - [960, 540]) / 540
    pixel_prefix_delta = float(np.abs(pixel_input[:120] - future_pixel[:120].astype(np.float32)).max())
    assert crop_prefix_delta > 0 and pixel_prefix_delta == 0
    # Parser preserves JSON record order and treats people as successive records.
    duplicate_path = output.with_name(output.name.replace("_results_", "_duplicate_input_"))
    with duplicate_path.open("x", encoding="utf-8") as handle:
        json.dump([records[2], {**records[0], "idx": 9}, records[1]], handle)
    mixed = read(duplicate_path, (1920, 1080), None, None)
    focused = read(duplicate_path, (1920, 1080), None, 7)
    assert np.array_equal(mixed[0], pixel_input[2]) and np.array_equal(mixed[2], pixel_input[1])
    assert focused.shape == (2, 17, 3)
    try:
        read(duplicate_path, (1920, 1080), None, 99)
    except (ValueError, IndexError) as error:
        empty_error = f"{type(error).__name__}: {error}"
    else:
        raise AssertionError("original empty focus unexpectedly accepted")
    zero_h36m = scope["halpe2h36m"](detections)
    zero_h36m[..., 2] = 0
    zero_crop = utils.crop_scale(zero_h36m, [1, 1])
    assert np.count_nonzero(zero_crop) == 0
    nan_h36m = scope["halpe2h36m"](detections)
    nan_h36m[0, 0, 0] = np.nan
    nan_crop = utils.crop_scale(nan_h36m, [1, 1])
    config = root / "configs/pose3d/MB_ft_h36m_global_lite.yaml"
    args = SimpleNamespace(**yaml.safe_load(config.read_text()))
    learning_source = root / "lib/utils/learning.py"
    factory = next(n for n in ast.parse(learning_source.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == "load_backbone")
    factory_scope = {"DSTformer": importlib.import_module("lib.model.DSTformer").DSTformer,
                     "partial": partial, "nn": torch.nn}
    compile_nodes(learning_source, [factory], factory_scope)
    model = factory_scope["load_backbone"](args).eval()
    if torch.cuda.is_available():
        model.cuda()
    device = next(model.parameters()).device
    before = {k: v.clone() for k, v in model.state_dict().items()}
    inference_source = root / "infer_wild.py"
    inference_tree = ast.parse(inference_source.read_text())
    loop = next(n for n in inference_tree.body if isinstance(n, ast.With))
    pixel_restore = next(n for n in reversed(inference_tree.body) if isinstance(n, ast.If))
    joins = [n for n in inference_tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "results_all" for t in n.targets)
             and isinstance(n.value, ast.Call)]
    assert len(joins) == 2

    def infer(batches):
        symbols = {"torch": torch, "np": np, "args": args, "model_pos": model, "tqdm": iter,
                   "test_loader": batches, "flip_data": utils.flip_data, "results_all": [],
                   "opts": SimpleNamespace(pixel=True), "vid_size": (1920, 1080)}
        compile_nodes(inference_source, [loop, *joins, pixel_restore], symbols)
        return symbols["results_all"]

    batches = [torch.from_numpy(dataset[i]).unsqueeze(0) for i in range(2)]
    result = infer(batches)
    repeat = infer(batches)
    assert result.shape == (248, 17, 3) and np.isfinite(result).all() and np.array_equal(result, repeat)
    references = []
    with torch.no_grad():
        for batch in batches:
            x = batch.to(device)
            a = model(x)
            b = model(utils.flip_data(x))
            index = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]
            restored = b[:, :, index].clone()
            restored[..., 0] *= -1
            y = ((a + restored) / 2).cpu().numpy()[0]
            y[0, 0, 2] = 0
            y *= 540
            y[..., :2] += [960, 540]
            references.append(y)
        independent = np.concatenate(references)
        independent_error = float(np.abs(result - independent).max())
        assert independent_error < 2e-5
        zero_output = infer([torch.zeros(1, 5, 17, 3)])
        nan_batch = batches[1].clone()
        nan_batch[0, 0, 0, 0] = torch.nan
        nan_output = infer([nan_batch])
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    np.savez_compressed(output.with_suffix(".npz"), pixel_input=pixel_input, crop_input=cropped,
                        output_image_relative=result, zero_input_output=zero_output, nan_input_output=nan_output)
    return {"status": "R8_original_wild_input_inference_verified", "training_executed": False, "checkpoint_loaded": False,
            "config": str(config), "parameters": sum(p.numel() for p in model.parameters()),
            "source_AST_loop_lines": [loop.lineno, loop.end_lineno], "source_pixel_restore_lines": [pixel_restore.lineno, pixel_restore.end_lineno],
            "scope": "Published default wild global_lite branch; original factory/dataset/loop, synthetic HALPE26 only, not earlier fullsize config",
            "record_count": 248, "clip_lengths": [243, 5], "independent_mapping_pixel_exact": True,
            "independent_TTA_root_restore_max_error": independent_error, "repeat_exact": True, "state_dict_unchanged": True,
            "root_zero_frame_indices": [0, 243], "other_root_z_nonzero_count": int(np.count_nonzero(result[:, 0, 2])),
            "future_changed_frame": 247, "future_fullsequence_crop_prefix_max_delta": crop_prefix_delta,
            "future_pixel_normalization_prefix_max_delta": pixel_prefix_delta,
            "parser_image_id_ignored_record_order_retained": True, "parser_unfocused_people_become_sequence_records": True,
            "focused_duplicate_parser_shape": list(focused.shape), "empty_focus_error": empty_error,
            "all_confidence_zero_crop_returns_all_zero": True,
            "NaN_crop_nonfinite_values": int(np.count_nonzero(~np.isfinite(nan_crop))),
            "zero_input_output_allfinite": bool(np.isfinite(zero_output).all()),
            "zero_input_output_nonzero_values": int(np.count_nonzero(zero_output)),
            "NaN_network_nonfinite_output_values": int(np.count_nonzero(~np.isfinite(nan_output))),
            "adaptations": ["Compile exact source AST to bypass ipdb/training/weight/video/renderer imports", "Logger progress iter instead of tqdm only; CUDA transfer branch retained"],
            "boundary": "Original default wild inference subpath engineering on synthetic HALPE26. Output scale equals imagepixels, not metricdepth, rootrelative skeleton or SMPL-H. Complete detector/videoCLI/neutralmesh/datasetaccuracy not executed. Sequencecrop uses future observations; originalparser is not identity/frame association."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "_results_" not in args.output.name:
        parser.error("Output must contain _results_ to keep versioned inputs separate")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("retain previous evidence; choose new version")
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 -- retain exact original failures
        result = {"status": "R8_failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
