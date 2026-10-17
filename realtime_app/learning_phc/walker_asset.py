"""Offline free-root compound walker asset; no simulator imports."""

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PARAMETER_UNITS = {"mass_kg": "kg", "com_m": "m", "inertia_kg_m2": "kg*m^2",
                   "tube_radius_m": "m", "pad_radius_m": "m", "pad_height_m": "m"}


def _array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("finite array with shape %s required" % (shape,))
    return result


def _rotation(value):
    r = _array(value, (3, 3))
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5, rtol=0) or abs(np.linalg.det(r) - 1) > 1e-5:
        raise ValueError("proper orthonormal rotation required")
    return r


def _transform(r, t):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = r, t
    return result.tolist()


def _fmt(value):
    return " ".join(format(float(x), ".12g") for x in value)


def _bounds(shapes):
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for shape in shapes:
        a, b, radius = np.asarray(shape["a"]), np.asarray(shape["b"]), shape["radius"]
        length = np.linalg.norm(b - a)
        axis = (b - a) / length
        extent = length / 2 * np.abs(axis) + radius * np.sqrt(np.maximum(0, 1 - axis ** 2))
        center = (a + b) / 2
        low, high = np.minimum(low, center - extent), np.maximum(high, center + extent)
    return {"min": low.tolist(), "max": high.tolist(), "size": (high - low).tolist()}


def build_walker_asset(source, *, mass_kg, com_m, inertia_kg_m2, tube_radius_m,
                       pad_radius_m, pad_height_m, sources, source_path):
    """Return (URDF str, metadata dict). Required inertia is about COM in walker axes."""
    values = {"mass_kg": mass_kg, "com_m": com_m, "inertia_kg_m2": inertia_kg_m2,
              "tube_radius_m": tube_radius_m, "pad_radius_m": pad_radius_m, "pad_height_m": pad_height_m}
    for key in ("mass_kg", "tube_radius_m", "pad_radius_m", "pad_height_m"):
        if isinstance(values[key], bool) or not math.isfinite(float(values[key])) or float(values[key]) <= 0:
            raise ValueError(key + " must be positive and finite")
    com = _array(com_m, (3,))
    xx, yy, zz, xy, xz, yz = _array(inertia_kg_m2, (6,))
    inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    principal = np.linalg.eigvalsh(inertia)
    if principal[0] <= 0 or principal[-1] > principal[0] + principal[1] + 1e-10 * principal[-1]:
        raise ValueError("inertia must be positive definite and obey principal triangle relations")
    if set(sources) != set(PARAMETER_UNITS):
        raise ValueError("per-parameter provenance required")
    for item in sources.values():
        if item.get("kind") not in {"synthetic_fixture", "estimated", "measured"} or not item.get("source", "").strip():
            raise ValueError("explicit parameter kind/source required")
    nodes = {name: _array(value, (3,)) / 1000 for name, value in source["nodes_walker_mm"].items()}
    edges, feet, handles = source["edges"], source["foot_nodes"], source["handle_segments"]
    if len(nodes) != 15 or len(edges) != 9 or len(feet) != 4 or len(set(feet)) != 4:
        raise ValueError("15 nodes, 9 rods and 4 unique feet required")
    if any(len(edge) != 2 or any(name not in nodes for name in edge) or edge[0] == edge[1] for edge in edges):
        raise ValueError("invalid edge")
    keys = [frozenset(edge) for edge in edges]
    if len(set(keys)) != 9 or any("front_top_center" in edge for edge in edges):
        raise ValueError("duplicate rod or semantic anchor used as geometry")
    if set(handles) != {"left", "right"} or any(len(v) != 2 or frozenset(v) not in keys for v in handles.values()):
        raise ValueError("handles must reuse existing rods")
    if any(name not in nodes or abs(nodes[name][2]) > 1e-9 for name in feet):
        raise ValueError("feet must be canonical pad-bottom centers at z=0")
    rods = []
    for index, edge in enumerate(edges):
        a, b = (nodes[name].copy() for name in edge)
        if edge[0] in feet or edge[1] in feet:
            low, high = (a, b) if edge[0] in feet else (b, a)
            if np.linalg.norm(high[:2] - low[:2]) > 1e-9:
                raise ValueError("foot rods must be vertical")
            low[2] += pad_height_m
            if high[2] <= low[2]:
                raise ValueError("pad leaves no upright length")
        if np.linalg.norm(b - a) <= 1e-9:
            raise ValueError("positive rod length required")
        rods.append({"name": "rod_%02d" % index, "a": a.tolist(), "b": b.tolist(), "radius": float(tube_radius_m)})
    pads = [{"name": "pad_" + name, "a": nodes[name].tolist(),
             "b": (nodes[name] + [0, 0, pad_height_m]).tolist(), "radius": float(pad_radius_m)} for name in feet]
    root = ET.Element("robot", name="walker_compound_candidate")
    link = ET.SubElement(root, "link", name="walker")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=_fmt(com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=format(float(mass_kg), ".12g"))
    ET.SubElement(inertial, "inertia", **{k: format(float(v), ".12g") for k, v in
                  zip(("ixx", "iyy", "izz", "ixy", "ixz", "iyz"), inertia_kg_m2)})
    for shape in rods + pads:
        a, b = np.asarray(shape["a"]), np.asarray(shape["b"])
        d = b - a
        rpy = [0, math.atan2(math.hypot(d[0], d[1]), d[2]), math.atan2(d[1], d[0])]
        for tag in ("collision", "visual"):
            element = ET.SubElement(link, tag, name=shape["name"])
            ET.SubElement(element, "origin", xyz=_fmt((a + b) / 2), rpy=_fmt(rpy))
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "cylinder", radius=format(shape["radius"], ".12g"),
                          length=format(float(np.linalg.norm(d)), ".12g"))
    r = _rotation(source["rotation_left_camera_from_walker"])
    t = _array(source["translation_left_camera_from_walker_mm"], (3,)) / 1000
    camera = {"T_C_from_W": _transform(r, t), "T_W_from_C": _transform(r.T, -r.T @ t),
              "kind": "estimated", "source": source_path + ": fitted left-camera mount", "unit": "m",
              "right_full_transform": None, "right_orientation_status": "unknown; center is insufficient"}
    right = source.get("camera_centers_left_camera_mm", {}).get("right")
    if right is not None:
        camera["right_optical_center_in_left_camera_m"] = (_array(right, (3,)) / 1000).tolist()
    cache = {}
    for ck, rk, tk in (("nodes_initial_ground_mm", "rotation_initial_ground_from_walker", "translation_initial_ground_from_walker_mm"),
                       ("nodes_left_camera_mm", "rotation_left_camera_from_walker", "translation_left_camera_from_walker_mm")):
        if ck in source:
            try:
                cr, ct = _rotation(source[rk]), _array(source[tk], (3,))
                residual = {name: float(np.linalg.norm(cr @ nodes[name] * 1000 + ct - _array(value, (3,))))
                            for name, value in source[ck].items() if name in nodes}
                cache[ck] = {"max_mm": max(residual.values(), default=0), "per_node_mm": residual, "used": False}
            except (ValueError, KeyError, TypeError) as error:
                cache[ck] = {"diagnostic_unavailable": str(error), "used": False}
    provenance = {key: {"kind": sources[key]["kind"], "source": sources[key]["source"],
                        "value": np.asarray(values[key]).tolist(), "unit": unit}
                  for key, unit in PARAMETER_UNITS.items()}
    metadata = {"status": "asset_generated_unvalidated", "source_path": source_path,
                "contains_synthetic_fixture": any(v["kind"] == "synthetic_fixture" for v in sources.values()),
                "geometry_kind": "estimated", "canonical_nodes_m": {k: v.tolist() for k, v in nodes.items()},
                "parameter_provenance": provenance, "inertia_convention": "about COM, expressed in walker axes",
                "simulator_loaded": False, "inertia_readback_verified": False, "free_root_intent": True,
                "collision_count": 13, "rods": rods, "pads": pads,
                "handles_m": {side: [nodes[name].tolist() for name in segment] for side, segment in handles.items()},
                "foot_pad_bottom_centers_m": {name: nodes[name].tolist() for name in feet},
                "tube_collision_bounds_m": _bounds(rods), "all_collision_bounds_m": _bounds(rods + pads),
                "left_camera_mount": camera, "cached_node_residuals_mm": cache,
                "dimension_boundary": "coarse canonical centerlines; collision outer dimensions are not catalog dimensions",
                "scope": "offline asset only; no Gym loading, grasp, contact or real-load validation"}
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n", metadata


def write_asset(output, urdf, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "walker.urdf").write_text(urdf, encoding="utf-8")
    (output / "asset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
