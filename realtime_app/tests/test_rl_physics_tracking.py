"""Tests of named RL contracts: skin map, timing, anchors and information boundary."""

from pathlib import Path
import pickle
import sys
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from learning_rl_tracking.body import PhysicsBody, wxyz  # noqa: E402
from learning_rl_tracking.data import ROOT, load_real  # noqa: E402
from learning_rl_tracking.environment import PixelTrackingEnv  # noqa: E402
from learning_rl_tracking.run import load_prepared  # noqa: E402


class RLContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_real()

    def body(self):
        d = self.dataset
        return PhysicsBody(d.initial_fit, d.rotation[0], d.translation[0])

    def test_exact_full_surface_model_multiple_poses(self):
        import torch
        import smplx
        from smplx.utils import Struct

        torch.set_num_threads(1)
        b = self.body()
        root = ROOT / "third_party/WiLoR/mano_data/models"
        with (root / "SMPLH_male.pkl").open("rb") as stream:
            asset = pickle.load(stream, encoding="latin1")
        for side, name in (("l", "LEFT"), ("r", "RIGHT")):
            with (root / f"MANO_{name}.pkl").open("rb") as stream:
                mano = pickle.load(stream, encoding="latin1")
            asset["hands_components" + side] = np.asarray(
                mano["hands_components"], np.float32
            )
            asset["hands_mean" + side] = np.asarray(mano["hands_mean"], np.float32)
        model = smplx.SMPLHLayer(
            str(root / "SMPLH_male.pkl"),
            data_struct=Struct(**asset),
            use_pca=False,
            flat_hand_mean=True,
        )
        reg = np.load(ROOT / "models/smpl/J_regressor_coco.npy")[5:17]
        rng = np.random.default_rng(7)
        for amplitude in (0.0, 0.04, 0.2):
            rotations = (
                b.initial_rotations
                @ Rotation.from_rotvec(rng.normal(0, amplitude, (21, 3))).as_matrix()
            )
            b.data.qpos[b.qpos_indices] = [wxyz(r) for r in rotations]
            mujoco.mj_forward(b.model, b.data)
            root_r = b.data.xmat[b.body_ids[0]].reshape(3, 3)
            transl = b.data.qpos[:3] - b.joints_rest[0]
            with torch.inference_mode():
                result = model(
                    betas=torch.tensor(b.beta[None], dtype=torch.float32),
                    transl=torch.tensor(transl[None], dtype=torch.float32),
                    global_orient=torch.tensor(root_r[None], dtype=torch.float32),
                    body_pose=torch.tensor(rotations[None], dtype=torch.float32),
                    left_hand_pose=torch.tensor(
                        b.hands[None, :15], dtype=torch.float32
                    ),
                    right_hand_pose=torch.tensor(
                        b.hands[None, 15:], dtype=torch.float32
                    ),
                )
            expected = reg @ result.vertices[0].numpy()
            np.testing.assert_allclose(b.points(), expected, atol=8e-6, rtol=0)

    def test_smpl_joint_order_and_first_frame_only(self):
        b = self.body()
        d = self.dataset
        with np.load(d.initial_fit) as z:
            expected = z["smplh_joints"][0, :22] @ d.rotation[0].T + d.translation[0]
        np.testing.assert_allclose(b.data.xpos[b.body_ids], expected, atol=2e-6, rtol=0)
        self.assertLess(b.initial_mapping_max_m, 2e-5)

    def test_actual_dt_remainder_and_hard_heel_velocity(self):
        b = self.body()
        b.enable_heel(True)
        start = b.data.time
        ok, _, reason = b.advance(b.target, 0.039763)
        self.assertTrue(ok, reason)
        self.assertAlmostEqual(b.data.time - start, 0.039763, places=11)
        self.assertLess(b.max_heel_error, 0.001)
        self.assertLess(np.linalg.norm(b._heel_jacobian() @ b.data.qvel), 1e-5)
        self.assertGreater(b.max_projection_impulse, 0)
        b.enable_heel(False)
        self.assertIsNone(b.heel_anchor)

    def test_no_future_pixel_leak(self):
        d = self.dataset
        b = self.body()
        e = PixelTrackingEnv(d, b)
        first, _ = e.reset()
        saved = d.pixels[2:].copy()
        d.pixels[2:] += 9000
        try:
            np.testing.assert_array_equal(e.observation(), first)
        finally:
            d.pixels[2:] = saved
        self.assertEqual(first.shape, (370,))

    def test_missing_pixels_do_not_get_full_reward(self):
        d = self.dataset
        b = self.body()
        e = PixelTrackingEnv(d, b)
        e.reset()
        saved = d.valid[1].copy()
        d.valid[1] = False
        try:
            _, reward, terminated, truncated, info = e.step(np.zeros(63))
            self.assertEqual(info["observed_slots"], 0)
            self.assertLessEqual(reward, 0)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
        finally:
            d.valid[1] = saved

    def test_synthetic_split_and_no_cross_clip_transition(self):
        root = (
            ROOT / "research_records/engineering_validation/G20261014_rl_physics448_v1"
        )
        if not (root / "synthetic.npz").exists():
            self.skipTest("requires prepared experiment inputs")
        _, d, s = load_prepared(root)
        self.assertFalse(set(s["train_clips"]) & set(s["test_clips"]))
        e = PixelTrackingEnv(d, self.body(), synthetic=s, training=True)
        e.reset(seed=5)
        clip = s["clip_id"][e.index]
        self.assertIn(clip, s["train_clips"])
        last = int(np.flatnonzero(s["clip_id"] == clip)[-1])
        e.index = last
        _, _, _, truncated, _ = e.step(np.zeros(63))
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
