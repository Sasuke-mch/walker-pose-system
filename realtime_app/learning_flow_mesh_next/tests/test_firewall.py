import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from ..data_firewall import Labels, Normalization, validate_manifest
from ..schema import Observation, Topology


def observation(frame=0, timestamp=0.0):
    return Observation(torch.ones(2, 17, 2), torch.ones(2, 17),
                       torch.ones(2, 17, dtype=torch.bool), torch.ones(2, 17, dtype=torch.bool),
                       torch.ones(2, 17, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                       torch.zeros(2, 17, 3), torch.tensor([0., 0., 1.]).expand(2, 17, 3),
                       tuple(tuple("" for _ in range(17)) for _ in range(2)),
                       ("PnP_LEFT_fixture", "PnP_RIGHT_fixture"), "person", "sequence",
                       frame, timestamp, "analytic_fixture")


def labels(nodes=52):
    return Labels(torch.eye(3).expand(2, nodes, 3, 3), torch.zeros(2, 3), torch.zeros(2, 10),
                  torch.ones(2, nodes, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                  torch.ones(2, dtype=torch.bool), Topology(nodes), "analytic_fixture", "engineering_fixture")


def manifest():
    return {"source_path": "pending", "license_status": "missing", "family": "SMPL-H", "gender": "male",
            "beta_basis": "local_asset_basis_pending_audit", "beta_dimensions": 10,
            "rotation_convention": "local_matrix_packed_columns6d", "hand_mean": "flat",
            "dmpl": "excluded_explicitly", "label_scope": "pending", "noise_source": "proxy-only",
            "records": [], "truth_status": "no_independent_3d_or_scan"}


def test_no_gt_or_untyped_fields():
    with pytest.raises(TypeError):
        replace(observation(), gt_root=torch.zeros(3))
    with pytest.raises(TypeError):
        Observation(**{**vars(observation()), "betas": torch.zeros(10)})


def test_missing_nan_preserved_present_nan_rejected():
    obs = observation()
    missing = replace(obs, present=torch.zeros_like(obs.present), xy=torch.full_like(obs.xy, torch.nan))
    missing.validate()
    with pytest.raises(ValueError, match="nonfinite"):
        replace(obs, xy=missing.xy).validate()


@pytest.mark.parametrize("nodes,dim,tangent", [(52, 325, 169), (22, 145, 79)])
def test_topology_and_partial_scope(nodes, dim, tangent):
    target = labels(nodes)
    assert target.packed()[0].shape == (2, dim)
    assert target.topology.tangent_dimensions == tangent
    partial = replace(target, beta_mask=torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="incomplete"):
        partial.packed()
    assert torch.isnan(partial.packed(require_complete=False)[0][:, -10:]).all()
    with pytest.raises(ValueError, match="training data"):
        Normalization.fit(target, source_path="fixture")


def test_split_and_family_firewall():
    m = manifest()
    assert not validate_manifest(m)["training_data_ready"]
    with pytest.raises(ValueError, match="SMPL-X"):
        validate_manifest({**m, "family": "SMPL-X"})
    records = [{"subject": "s", "original_action": "a", "sequence": "q", "rig": "r", "source_path": "x", "split": s}
               for s in ("train", "test")]
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest({**m, "records": records})


@pytest.mark.parametrize("nodes", [22, 52])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fitted_statistics_are_frozen_and_roundtrip_declared_origin(nodes, dtype):
    # Artificial tensors with declared train/reference metadata test the
    # contract only; they are not independently acquired reference data.
    target = labels(nodes)
    target = replace(target, rotations=target.rotations.to(dtype), beta=target.beta.to(dtype),
                     translation=torch.tensor([[0., 0., 3.], [.123456789, .05, 3.123456789]],
                                              dtype=dtype, requires_grad=True),
                     split="train", source_kind="fitted_reference")
    expected, _ = target.packed()
    fitted = Normalization.fit(target, source_path="declared_contract_fixture")
    assert torch.equal(fitted.mean, expected.mean(0))
    assert torch.equal(fitted.scale, expected.std(0, unbiased=False).clamp_min(1e-4))
    assert not fitted.mean.requires_grad and not fitted.scale.requires_grad
    assert fitted.mean.grad_fn is None and fitted.scale.grad_fn is None
    assert target.translation.grad is None
    assert fitted.require_training_statistics() is fitted
    restored = Normalization.from_dict(fitted.to_dict())
    assert restored.split == "train" and restored.source_kind == "fitted_reference"
    assert restored.mean.dtype == restored.scale.dtype == dtype
    assert torch.equal(fitted.mean, restored.mean) and torch.equal(fitted.scale, restored.scale)
    assert restored.require_training_statistics() is restored


def test_fixture_and_legacy_statistics_do_not_claim_train_origin():
    fixture = Normalization.fixture(Topology())
    assert fixture.split == "engineering_fixture" and fixture.source_kind == "analytic_fixture"
    current = fixture.to_dict()
    legacy = {key: current[key] for key in ("mean", "scale", "nodes", "source")}
    restored = Normalization.from_dict(legacy)
    assert restored.split == restored.source_kind == "legacy_unverified"
    assert restored.mean.dtype == torch.float32
    assert torch.equal(restored.normalize(torch.ones(325)), fixture.normalize(torch.ones(325)))
    for value in (fixture, restored):
        with pytest.raises(ValueError, match="declared training statistics"):
            value.require_training_statistics()
    with pytest.raises(ValueError, match="provenance"):
        Normalization.from_dict({**legacy, "split": "train"})


@pytest.mark.parametrize("split,kind", [("validation", "fitted_reference"), ("test", "independent_reference"),
                                        ("train", "analytic_fixture"), ("engineering_fixture", "fitted_reference")])
def test_nontrain_and_conflicting_statistics_provenance_rejected(split, kind):
    fixture = Normalization.fixture(Topology())
    with pytest.raises(ValueError, match="provenance"):
        replace(fixture, split=split, source_kind=kind).validate()


def test_gradient_bearing_or_unsupported_statistics_rejected():
    fixture = Normalization.fixture(Topology())
    with pytest.raises(ValueError, match="must be frozen"):
        replace(fixture, mean=fixture.mean.clone().requires_grad_()).validate()
    with pytest.raises(ValueError, match="dtype"):
        Normalization.from_dict({**fixture.to_dict(), "dtype": "float16"})


def test_so3_codec_and_candidate_provenance_bind_statistics_origin():
    from ..candidate_store import infer
    from ..so3_flow import SO3JointFlow

    torch.set_num_threads(1)
    normal = Normalization.fixture(Topology())
    model = SO3JointFlow(normal).eval()
    with torch.no_grad():
        store, _ = infer(model, normal, observation(), k=1, regression=True)
    assert store.provenance["normalization_split"] == "engineering_fixture"
    assert store.provenance["normalization_source_kind"] == "analytic_fixture"
    relabeled = replace(normal, split="legacy_unverified", source_kind="legacy_unverified")
    assert torch.equal(relabeled.mean, normal.mean) and relabeled.source == normal.source
    with pytest.raises(ValueError, match="normalization mismatch"):
        model.validate_normalization(relabeled)


@pytest.mark.parametrize("geometry", ["ambient", "SO3"])
def test_inference_rejects_unfrozen_statistics_before_encoding(geometry):
    from ..candidate_store import infer
    from ..joint_flow import JointFlow
    from ..so3_flow import SO3JointFlow

    normal = Normalization.fixture(Topology())
    model = JointFlow() if geometry == "ambient" else SO3JointFlow(normal)
    unsafe = replace(normal, mean=normal.mean.clone().requires_grad_())
    with pytest.raises(ValueError, match="must be frozen"):
        infer(model, unsafe, observation(), k=1, steps=1)


def registered_file_fixture(tmp_path, nodes=52, count=35):
    """Temporary declared-reference files are synthetic schema tests only."""
    record = {"subject": "synthetic_subject", "original_action": "synthetic_action", "sequence": "synthetic_sequence",
              "rig": "fixture_rig", "source_path": "sequence.npz", "split": "train", "identity": "fixture_person",
              "camera_ids": ["PnP_LEFT_fixture", "PnP_RIGHT_fixture"], "source_kind": "fitted_reference", "nodes": nodes}
    declared = {**manifest(), "license_status": "locally_authorized", "source_path": "synthetic_contract_fixture",
                "beta_basis": "synthetic_test_basis_not_real_asset", "hand_mean": "flat",
                "truth_status": "synthetic_contract_fixture_not_independent_data",
                "label_scope": "complete52" if nodes == 52 else "BODY22_partial", "records": [record]}
    body_keys = ("family", "gender", "beta_basis", "beta_dimensions", "rotation_convention", "hand_mean", "dmpl")
    asset = {key: declared[key] for key in body_keys}
    metadata = {**asset, **{key: declared[key] for key in ("label_scope", "noise_source", "truth_status")},
                **{key: value for key, value in record.items() if key != "source_path"},
                "format": "flow_next_sequence_v1", "camera_roles": ["LEFT", "RIGHT"], "pixel_domain": "raw_fisheye",
                "units": "metres", "operator": "COCO17_SURFACE_REGRESSOR_6890", "coordinate_frame": "cam0_LEFT",
                "timestamp_kind": "analytic_fixture_clock"}
    frames = np.arange(501, 501+count, dtype=np.int64)
    times = 1000+np.arange(count, dtype=np.float64)*.07
    reasons = [[['' for _ in range(17)] for _ in range(2)] for _ in range(count)]
    reasons[2][0][0] = "original_missing_fixture"
    arrays = {"frame_ids": frames, "label_frame_ids": frames.copy(), "timestamps": times,
              "label_timestamps": times.copy(), "view_timestamps": times[:, None]+np.array([-.001, .001]),
              "xy": np.ones((count, 2, 17, 2), dtype=np.float64)*100,
              "confidence": np.ones((count, 2, 17), dtype=np.float64)*.8,
              "present": np.ones((count, 2, 17), dtype=bool), "quality": np.ones((count, 2, 17), dtype=bool),
              "geometry": np.zeros((count, 2, 17), dtype=bool), "views": np.ones((count, 2), dtype=bool),
              "origins": np.zeros((count, 2, 17, 3), dtype=np.float64),
              "directions": np.tile([0., 0., 1.], (count, 2, 17, 1)),
              "rotations": np.tile(np.eye(3), (count, nodes, 1, 1)),
              "translation": np.column_stack((np.arange(count)*.01, np.zeros(count), np.ones(count)*3)),
              "beta": np.zeros((count, 10), dtype=np.float64), "pose_mask": np.ones((count, nodes), dtype=bool),
              "translation_mask": np.ones(count, dtype=bool), "beta_mask": np.ones(count, dtype=bool),
              "rejection_reasons_json": np.array([json.dumps(value) for value in reasons]),
              "metadata_json": np.array(json.dumps(metadata))}
    arrays["present"][2] = False
    arrays["xy"][2] = arrays["confidence"][2] = np.nan
    unknown = min(8, count-1)
    arrays["beta_mask"][unknown] = False
    arrays["beta"][unknown] = np.nan
    selected = arrays["beta_mask"]
    training_contract_labels = Labels(*(torch.from_numpy(arrays[key][selected].copy()) for key in
                                       ("rotations", "translation", "beta", "pose_mask", "translation_mask", "beta_mask")),
                                     Topology(nodes), "fitted_reference", "train")
    normal = Normalization.fit(training_contract_labels, source_path="synthetic_statistics_contract_not_real_data")
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(declared), encoding="utf-8")
    np.savez(tmp_path/"sequence.npz", **arrays)
    return path, asset, normal, arrays, declared, metadata


@pytest.mark.parametrize("nodes", [22, 52])
def test_registered_reader_preserves_every_original_row_and_missing_state(tmp_path, nodes):
    from ..train import load_registered_chunks

    path, asset, normal, arrays, _, _ = registered_file_fixture(tmp_path, nodes)
    chunks, report = load_registered_chunks(path, 0, normal, asset_contract=asset, dtype=torch.float64)
    assert [len(chunk.observations) for chunk in chunks] == [32, 3]
    observations = [obs for chunk in chunks for obs in chunk.observations]
    assert [obs.frame for obs in observations] == arrays["frame_ids"].tolist()
    assert [obs.timestamp for obs in observations] == arrays["timestamps"].tolist()
    for index, obs in enumerate(observations):
        assert obs.source == str((tmp_path/"sequence.npz").resolve())
        assert obs.view_timestamps == tuple(arrays["view_timestamps"][index])
        for key in ("xy", "confidence", "present", "quality", "geometry", "views", "origins", "directions"):
            torch.testing.assert_close(getattr(obs, key), torch.from_numpy(arrays[key][index]), rtol=0, atol=0, equal_nan=True)
        assert obs.rejection_reasons == tuple(tuple(row) for row in json.loads(arrays["rejection_reasons_json"][index]))
    for key in ("rotations", "translation", "beta", "pose_mask", "translation_mask", "beta_mask"):
        values = torch.cat([getattr(chunk.labels, key) for chunk in chunks])
        torch.testing.assert_close(values, torch.from_numpy(arrays[key]), rtol=0, atol=0, equal_nan=True)
    assert not observations[2].admitted().any() and torch.isnan(chunks[0].labels.beta[8]).all()
    assert report["original_frames"] == 35 and report["chunks"] == 2
    assert not report["authenticity_verified"] and not report["training_executed"] and not report["checkpoint_loaded"]


@pytest.mark.parametrize("fault", ["subject", "asset_basis", "camera_ids", "camera_roles", "coordinate_frame", "format",
                                  "label_order", "label_clock", "clock_order", "unsigned_frame_order", "integer_mask", "missing_array", "shape", "pickle", "trusted_nan"])
def test_registered_reader_rejects_mismatched_source_contracts(tmp_path, fault):
    from ..train import load_registered_chunks

    path, asset, normal, arrays, _, metadata = registered_file_fixture(tmp_path)
    if fault in ("subject", "camera_ids", "camera_roles", "coordinate_frame", "format"):
        metadata[fault] = {"subject": "other", "camera_ids": ["other_left", "other_right"],
                           "camera_roles": ["RIGHT", "LEFT"], "coordinate_frame": "world", "format": "unknown"}[fault]
        arrays["metadata_json"] = np.array(json.dumps(metadata))
    elif fault == "asset_basis":
        asset = {**asset, "beta_basis": "other_basis"}
    elif fault == "label_order":
        arrays["label_frame_ids"] = arrays["label_frame_ids"][::-1].copy()
    elif fault == "label_clock":
        arrays["label_timestamps"][1] += .001
    elif fault == "clock_order":
        arrays["timestamps"][1] = arrays["timestamps"][0]
        arrays["label_timestamps"] = arrays["timestamps"].copy()
    elif fault == "unsigned_frame_order":
        arrays["frame_ids"] = arrays["frame_ids"][::-1].astype(np.uint64)
        arrays["label_frame_ids"] = arrays["frame_ids"].copy()
    elif fault == "integer_mask":
        arrays["present"] = arrays["present"].astype(np.uint8)
    elif fault == "missing_array":
        del arrays["beta_mask"]
    elif fault == "shape":
        arrays["translation"] = arrays["translation"][:-1]
    elif fault == "pickle":
        arrays["xy"] = arrays["xy"].astype(object)
    else:
        arrays["translation"][0, 0] = np.nan
    np.savez(tmp_path/"sequence.npz", **arrays)
    with pytest.raises(ValueError):
        load_registered_chunks(path, 0, normal, asset_contract=asset)


@pytest.mark.parametrize("fault", ["empty", "license", "test_record", "leakage", "source_kind", "nodes"])
def test_registered_reader_registry_failure_precedes_archive_open(tmp_path, monkeypatch, fault):
    from ..train import load_registered_chunks

    path, asset, normal, _, declared, _ = registered_file_fixture(tmp_path)
    record = declared["records"][0]
    if fault == "empty":
        declared["records"] = []
    elif fault == "license":
        declared["license_status"] = "missing"
    elif fault == "test_record":
        declared["records"] = [{**record, "split": "test"}, {**record, "subject": "s2", "original_action": "a2", "sequence": "q2"}]
    elif fault == "leakage":
        declared["records"].append({**record, "split": "test"})
    elif fault == "source_kind":
        record["source_kind"] = "analytic_fixture"
    else:
        record["nodes"] = True
    path.write_text(json.dumps(declared), encoding="utf-8")
    opened = []
    def must_not_open(*args, **kwargs):
        opened.append(1)
        raise AssertionError("registry gate must run before archive open")
    monkeypatch.setattr(np, "load", must_not_open)
    with pytest.raises(TypeError if fault == "nodes" else ValueError):
        load_registered_chunks(path, 0, normal, asset_contract=asset)
    assert not opened


def test_registered_file_to_causal_objective_matches_memory_fixture(tmp_path):
    from ..joint_flow import JointFlow
    from ..train import (
        LabelFrame,
        SequenceChunk,
        load_registered_chunks,
        sequence_objective,
    )

    torch.set_num_threads(1)
    torch.manual_seed(7)
    path, asset, normal, arrays, declared, _ = registered_file_fixture(tmp_path, count=5)
    chunks, _ = load_registered_chunks(path, 0, normal, asset_contract=asset, dtype=torch.float64)
    record = declared["records"][0]
    independent_obs = tuple(replace(observation(int(frame), float(time)),
                                   **{key: torch.from_numpy(arrays[key][i].copy()) for key in
                                      ("xy", "confidence", "present", "quality", "geometry", "views", "origins", "directions")},
                                   identity=record["identity"], sequence=record["sequence"])
                            for i, (frame, time) in enumerate(zip(arrays["frame_ids"], arrays["timestamps"])))
    direct_labels = Labels(*(torch.from_numpy(arrays[key].copy()) for key in
                             ("rotations", "translation", "beta", "pose_mask", "translation_mask", "beta_mask")),
                           Topology(), "fitted_reference", "train")
    frames = tuple(LabelFrame(obs.identity, obs.sequence, obs.frame, obs.timestamp, obs.camera_ids) for obs in independent_obs)
    memory = SequenceChunk(independent_obs, direct_labels, frames)
    model = JointFlow().double().eval()
    with torch.no_grad():
        from_file = sequence_objective("regression", model, chunks[0], normal)
        in_memory = sequence_objective("regression", model, memory, normal)
    assert torch.equal(from_file.contexts, in_memory.contexts)
    assert torch.allclose(from_file.frame_losses, in_memory.frame_losses, equal_nan=True)
    assert from_file.state.to_dict() == in_memory.state.to_dict()
    assert from_file.supervised.tolist() == [True, True, False, True, False]
    assert torch.equal(from_file.loss, in_memory.loss)
