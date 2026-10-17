import copy
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from learning_phc.walker_asset import PARAMETER_UNITS, build_walker_asset, write_asset


def fixture():
    nodes = {}
    for side, x in (("right", 225.0), ("left", -225.0)):
        for end, y in (("front", -150.0), ("rear", 150.0)):
            for level, z in (("foot", 0.0), ("mid", 445.0), ("top", 840.0)):
                nodes[end + "_" + side + "_" + level] = [x, y, z]
        nodes["front_" + side + "_rail"] = [x, -150.0, 675.0]
    nodes["front_top_center"] = [0.0, -150.0, 675.0]
    edges = [[end + "_" + side + "_foot", end + "_" + side + "_top"]
             for side in ("right", "left") for end in ("front", "rear")]
    edges += [["front_left_rail", "front_right_rail"], ["front_right_top", "rear_right_top"],
              ["front_left_top", "rear_left_top"], ["front_right_mid", "rear_right_mid"],
              ["front_left_mid", "rear_left_mid"]]
    source = {"nodes_walker_mm": nodes, "edges": edges,
              "handle_segments": {"left": ["front_right_top", "rear_right_top"], "right": ["front_left_top", "rear_left_top"]},
              "foot_nodes": ["front_right_foot", "front_left_foot", "rear_right_foot", "rear_left_foot"],
              "rotation_left_camera_from_walker": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
              "translation_left_camera_from_walker_mm": [100, 200, 300],
              "camera_centers_left_camera_mm": {"right": [20, 0, 0]}}
    parameters = dict(mass_kg=4.0, com_m=[0, 0, 0.42], inertia_kg_m2=[0.25, 0.2, 0.15, 0, 0, 0],
                      tube_radius_m=0.01, pad_radius_m=0.02, pad_height_m=0.02,
                      sources={k: {"kind": "synthetic_fixture", "source": "named test fixture; not actual equipment"}
                               for k in PARAMETER_UNITS}, source_path="synthetic_fixture")
    return source, parameters


class WalkerAssetTests(unittest.TestCase):
    def test_required_parameters_and_inertia(self):
        source, p = fixture()
        missing = dict(p)
        del missing["mass_kg"]
        with self.assertRaises(TypeError):
            build_walker_asset(source, **missing)
        for tensor in ([0, 0, 0, 0, 0, 0], [2, 2, 2, 3, 0, 0], [4, 1, 1, 0, 0, 0]):
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                build_walker_asset(source, **dict(p, inertia_kg_m2=tensor))
        build_walker_asset(source, **dict(p, inertia_kg_m2=[0.3, 0.3, 0.2, 0.02, 0.01, -0.01]))
        with self.assertRaises(ValueError):
            build_walker_asset(source, **dict(p, sources={}))

    def test_single_link_free_root_and_no_duplicate_geometry(self):
        source, p = fixture()
        urdf, metadata = build_walker_asset(source, **p)
        root = ET.fromstring(urdf)
        self.assertEqual(len(root.findall("link")), 1)
        self.assertEqual(len(root.findall("joint")), 0)
        self.assertEqual(len(root.findall("link/collision")), 13)
        self.assertEqual(len(root.findall(".//sphere")), 0)
        self.assertEqual(root.find("link/inertial/origin").get("xyz"), "0 0 0.42")
        self.assertEqual(root.find("link/inertial/origin").get("rpy"), "0 0 0")
        self.assertEqual(float(root.find("link/inertial/mass").get("value")), 4.0)
        self.assertEqual(metadata["collision_count"], 13)

    def test_cylinder_endpoints_pad_bottom_and_outer_dimensions(self):
        source, p = fixture()
        urdf, metadata = build_walker_asset(source, **p)
        elements = {x.get("name"): x for x in ET.fromstring(urdf).findall("link/collision")}
        for shape in metadata["rods"] + metadata["pads"]:
            element = elements[shape["name"]]
            center = np.fromstring(element.find("origin").get("xyz"), sep=" ")
            roll, pitch, yaw = np.fromstring(element.find("origin").get("rpy"), sep=" ")
            self.assertEqual(roll, 0)
            axis = np.array([math.cos(yaw) * math.sin(pitch), math.sin(yaw) * math.sin(pitch), math.cos(pitch)])
            length = float(element.find("geometry/cylinder").get("length"))
            np.testing.assert_allclose(center - length / 2 * axis, shape["a"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(center + length / 2 * axis, shape["b"], atol=1e-10, rtol=0)
        for pad in metadata["pads"]:
            self.assertEqual(pad["a"][2], 0)
            self.assertEqual(pad["b"][2], 0.02)
        for rod in metadata["rods"][:4]:
            self.assertAlmostEqual(min(rod["a"][2], rod["b"][2]), 0.02)
        self.assertAlmostEqual(metadata["tube_collision_bounds_m"]["size"][0], 0.47)
        self.assertAlmostEqual(metadata["all_collision_bounds_m"]["size"][0], 0.49)
        self.assertEqual(metadata["all_collision_bounds_m"]["min"][2], 0)

    def test_same_source_left_side_camera_inverse_and_cache_not_used(self):
        source, p = fixture()
        baseline, _ = build_walker_asset(source, **p)
        source = copy.deepcopy(source)
        source["nodes_left_camera_mm"] = {"front_right_foot": [999, 999, 999]}
        source["nodes_initial_ground_mm"] = {"front_right_foot": [0, 0, 0]}
        source["rotation_initial_ground_from_walker"] = [[0, 0, 0]] * 3
        source["translation_initial_ground_from_walker_mm"] = [0, 0, 0]
        urdf, metadata = build_walker_asset(source, **p)
        self.assertEqual(urdf, baseline)
        self.assertGreater(metadata["cached_node_residuals_mm"]["nodes_left_camera_mm"]["max_mm"], 0)
        self.assertIn("diagnostic_unavailable", metadata["cached_node_residuals_mm"]["nodes_initial_ground_mm"])
        self.assertEqual(metadata["handles_m"]["left"][0], [0.225, -0.15, 0.84])
        mount = metadata["left_camera_mount"]
        t = np.asarray(mount["T_W_from_C"])
        np.testing.assert_allclose(t[:3, 3], [-0.2, 0.1, -0.3], atol=1e-12)
        np.testing.assert_allclose(t @ np.asarray(mount["T_C_from_W"]), np.eye(4), atol=1e-12)
        self.assertIsNone(mount["right_full_transform"])
        self.assertEqual(mount["right_optical_center_in_left_camera_m"], [0.02, 0.0, 0.0])

    def test_provenance_unvalidated_status_and_no_overwrite(self):
        source, p = fixture()
        urdf, metadata = build_walker_asset(source, **p)
        self.assertEqual(metadata["status"], "asset_generated_unvalidated")
        self.assertTrue(metadata["contains_synthetic_fixture"])
        self.assertEqual(metadata["geometry_kind"], "estimated")
        self.assertFalse(metadata["simulator_loaded"])
        self.assertFalse(metadata["inertia_readback_verified"])
        self.assertEqual(metadata["parameter_provenance"]["mass_kg"]["unit"], "kg")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic_fixture"
            write_asset(output, urdf, metadata)
            self.assertEqual(set(x.name for x in output.iterdir()), {"walker.urdf", "asset_metadata.json"})
            with self.assertRaises(FileExistsError):
                write_asset(output, urdf, metadata)


if __name__ == "__main__":
    unittest.main()
