"""Named PHC transfer risks: exact author inference/observation and body axes."""

import ast
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'realtime_app'))
from learning_phc.physics import PHCBody
from learning_phc.policy import FrozenPHC, observation_v7, recovery_reference
from learning_phc.reference import retarget


def author_function(path, name, namespace):
    """Load the actual pure function, excluding its heavyweight simulator imports."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class PHCPretrainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phc = ROOT / 'third_party/PHC'
        if not (cls.phc / 'phc').exists() or not (ROOT / 'models/phc/phc_kp_2.pth').exists():
            raise unittest.SkipTest('requires the actual author source/checkpoints, not fake weights')
        sys.path.insert(0, str(cls.phc))
        torch.set_num_threads(1)

    def test_frozen_inference_matches_author_loaders(self):
        from phc.learning.network_loader import load_pnn, load_mcp_mlp
        primitive_path, composer_path = [ROOT / 'models/phc' / n
                                        for n in ('phc_kp_2.pth', 'phc_comp_kp_2.pth')]
        ours = FrozenPHC(primitive_path, composer_path)
        primitive = torch.load(primitive_path, map_location='cpu', weights_only=False)
        composer = torch.load(composer_path, map_location='cpu', weights_only=False)
        author_pnn = load_pnn(primitive, num_prim=3, has_lateral=False, activation='silu')
        # Official CLI ending_act=False removes the activation added by loader.
        author_composer = load_mcp_mlp(composer, activation='silu', mlp_name='composer')[:-1]
        rng = np.random.default_rng(34)
        obs = torch.from_numpy(rng.normal(size=(8, 574)).astype(np.float32))
        with torch.inference_mode():
            def normalized(checkpoint):
                n = checkpoint['running_mean_std']
                return ((obs - n['running_mean'].float()) /
                        torch.sqrt(n['running_var'].float() + 1e-5)).clamp(-5, 5)
            weights = author_composer(normalized(composer)).clamp(-1, 1)
            _, experts = author_pnn(normalized(primitive))
            expected = (weights[..., None] * torch.stack(experts, dim=1)).sum(1).clamp(-1, 1)
            actual, actual_weights = ours(obs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_weights, weights, atol=0, rtol=0)
        self.assertTrue(all(not p.requires_grad for p in ours.parameters()))

    def test_observation_matches_author_functions(self):
        from phc.utils import isaacgym_torch_utils as utils
        namespace = dict(torch=torch, torch_utils=utils, Tensor=torch.Tensor)
        namespace.update({name: getattr(utils, name) for name in
                          ('quat_mul', 'quat_to_tan_norm', 'quat_conjugate')})
        self_obs = author_function(self.phc / 'phc/utils/isaacgym_humanoid_funcs.py',
                                   'compute_humanoid_observations_smpl_max', namespace)
        task_obs = author_function(self.phc / 'phc/env/tasks/humanoid_im.py',
                                   'compute_imitation_observations_v7', namespace)
        rng = np.random.default_rng(74)
        for _ in range(5):
            p, v, w, rp, rv = [rng.normal(size=(24, 3)).astype(np.float32) for _ in range(5)]
            r = Rotation.random(24, random_state=rng)
            pt, vt, wt, rpt, rvt = [torch.from_numpy(a)[None] for a in (p, v, w, rp, rv)]
            rt = torch.from_numpy(r.as_quat().astype(np.float32))[None]
            expected = torch.cat([
                self_obs(pt, rt, vt, wt, torch.zeros(1), torch.zeros(1), True, True, True, False, False),
                task_obs(pt[:, 0], rt[:, 0], pt, vt, rpt, rvt, 1, True)], dim=-1)
            np.testing.assert_allclose(observation_v7(p, r.as_matrix(), v, w, rp, rv),
                                       expected.numpy()[0], atol=4e-6, rtol=2e-5)

    def test_global_rotation_mapping_and_variable_dt(self):
        body = PHCBody(self.phc / 'phc/data/assets/mjcf/smpl_0_humanoid.xml')
        root = np.array([.2, -.1, 1.1])
        rotation = Rotation.random(24, random_state=24).as_matrix()
        body.reset(body.from_global_rotations(root, rotation))
        p, r, v, w = body.state()
        np.testing.assert_allclose(p[0], root, atol=1e-12)
        np.testing.assert_allclose(r, rotation, atol=1e-12)
        np.testing.assert_array_equal(v, np.zeros((24, 3)))
        np.testing.assert_array_equal(w, np.zeros((24, 3)))
        for dt in (.0069816, .0695054, .017):
            before = body.data.time
            self.assertTrue(body.advance(np.zeros(69), dt))
            self.assertAlmostEqual(body.data.time - before, dt, places=10)
        self.assertEqual(body.model.nv, 75)
        self.assertEqual(body.model.nu, 69)

    def test_unavailable_observations_rejected(self):
        zero = np.zeros((24, 3))
        bad = zero.copy()
        bad[4, 1] = np.nan
        with self.assertRaises(ValueError):
            observation_v7(bad, np.tile(np.eye(3), (24, 1, 1)), zero, zero, zero, zero)

    def test_reference_is_prefix_invariant_and_respects_segments(self):
        time = np.array([0, .03, .08, .12, .19])
        pelvis = np.c_[time*.4, time*0, np.ones(5)]
        root = Rotation.from_euler('x', np.repeat(np.pi/2, 5)).as_matrix()
        local = np.tile(np.eye(3), (5, 21, 1, 1))
        hand = np.tile(np.eye(3), (2, 1, 1))
        valid, segment = np.ones(5, bool), np.array([0, 0, 0, 1, 1])
        xml = self.phc/'phc/data/assets/mjcf/smpl_0_humanoid.xml'
        full = retarget(time, pelvis, root, local, hand, valid, segment, xml)
        prefix = retarget(time[:3], pelvis[:3], root[:3], local[:3], hand, valid[:3], segment[:3], xml)
        for key in prefix:
            np.testing.assert_array_equal(full[key][:3], prefix[key])
        np.testing.assert_allclose(full['reference_velocity_m_s'][1:3, :, 0], .4, atol=1e-12)
        np.testing.assert_array_equal(full['reference_velocity_m_s'][3], np.zeros((24, 3)))
        np.testing.assert_allclose(full['reference_position_m'][:, 0], pelvis, atol=0)
        valid[2] = False
        invalid = retarget(time, pelvis, root, local, hand, valid, segment, xml)
        self.assertTrue(np.isnan(invalid['reference_position_m'][2]).all())
        self.assertTrue(np.isnan(invalid['reference_velocity_m_s'][2]).all())

    def test_reference_rejects_invalid_time_and_available_nan(self):
        time = np.array([0., .02])
        pelvis = np.zeros((2, 3))
        root = np.tile(np.eye(3), (2, 1, 1))
        local = np.tile(np.eye(3), (2, 21, 1, 1))
        hand = np.tile(np.eye(3), (2, 1, 1))
        valid, segment = np.ones(2, bool), np.zeros(2, int)
        xml = self.phc/'phc/data/assets/mjcf/smpl_0_humanoid.xml'
        with self.assertRaises(ValueError):
            retarget(np.array([0., 0.]), pelvis, root, local, hand, valid, segment, xml)
        pelvis[1, 0] = np.nan
        with self.assertRaises(ValueError):
            retarget(time, pelvis, root, local, hand, valid, segment, xml)

    def test_author_far_recovery_does_not_modify_saved_reference(self):
        position, velocity = np.zeros((24, 3)), np.ones((24, 3))
        target = np.ones((24, 3))
        target[0] = [4, 0, 0]
        target_velocity = np.ones((24, 3))*2
        recovered, recovered_velocity = recovery_reference(position, velocity, target, target_velocity)
        np.testing.assert_array_equal(target[0], [4, 0, 0])
        np.testing.assert_allclose(recovered[0], [3, 0, 0])
        np.testing.assert_array_equal(recovered[1:], position[1:])
        np.testing.assert_array_equal(recovered_velocity, velocity)


if __name__ == '__main__':
    unittest.main()
