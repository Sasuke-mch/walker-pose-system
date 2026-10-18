"""Direct-image risks: exact surface, fisheye rays, no future3D, masks and PPO."""

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'realtime_app'))
from learning_phc.image_policy import pixel_rays, ImageObservation, ImageActorCritic
from pose_app.fisheye_camera import fisheye_project_numpy

EXPERIMENT=ROOT/'research_records/engineering_validation/G20261015_phc_pretrained448_v1'


class ImageGeometryTests(unittest.TestCase):
    def test_fisheye_rays_roundtrip_and_axis(self):
        k=np.array([[500.,0,960.],[0,510.,540.],[0,0,1.]])
        d=np.array([.03,-.01,.001,0.])
        points=np.array([[0.,0.,1.],[.4,-.3,1.],[-.8,.7,1.],[2.,1.,1.]])
        rays=pixel_rays(fisheye_project_numpy(points,k,d),k,d)
        np.testing.assert_allclose(rays,points/np.linalg.norm(points,axis=1,keepdims=True),atol=2e-14)
        self.assertTrue(np.isnan(pixel_rays(np.array([[5000.,5000.]]),k,np.zeros(4))).all())

    def test_squashed_policy_log_probability_and_gradient(self):
        torch.manual_seed(7)
        policy=ImageActorCritic()
        x=torch.randn(8,574)
        normal,value=policy.distribution(x)
        raw=normal.sample()
        loss=-(policy.log_probability(normal,raw)*torch.linspace(-1,1,8)).mean()+value.square().mean()
        loss.backward()
        gradients=[p.grad for p in policy.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0.)
        self.assertTrue((policy.action(x).abs()<=1).all())


@unittest.skipUnless((EXPERIMENT/'direct_image_20261016/inference_packet.npz').exists(),
                     'local licensed SMPL-H packet required')
class ImagePacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(EXPERIMENT/'direct_image_20261016/inference_packet.npz') as z:
            cls.packet={k:z[k].copy() for k in z.files}
        with np.load(EXPERIMENT/'corrections_v1/matched_body/rollout.npz') as z:
            cls.states={k:z[k].copy() for k in ('position_m','rotation_world','velocity_m_s','angular_velocity_rad_s')}

    def state(self,i):
        return tuple(self.states[k][i] for k in ('position_m','rotation_world','velocity_m_s','angular_velocity_rad_s'))

    def test_exact_reduced_surface_matches_full_saved_decode(self):
        observation=ImageObservation(self.packet)
        with np.load(EXPERIMENT/'corrections_v1/matched_body/decoded_smplh.npz') as z:
            for i in (0,143,447):
                actual=observation.points(*self.state(i)[:2])
                np.testing.assert_allclose(actual,z['world_predicted_coco'][i,5:17],atol=8e-7,rtol=0)

    def test_inference_rejects_privileged_future_3d(self):
        corrupted=dict(self.packet,reference_position_m=np.zeros((448,24,3)))
        with self.assertRaisesRegex(ValueError,'privileged'):
            ImageObservation(corrupted)
        with np.load(EXPERIMENT/'direct_image_20261016/teacher_train_only.npz') as z:
            self.assertTrue(all(len(z[k])==256 for k in z.files))

    def test_future_pixels_and_camera_do_not_change_current_features(self):
        packet={k:v.copy() for k,v in self.packet.items()}
        before=ImageObservation(packet).features(143,*self.state(142))
        packet['pixels'][144:]=np.nan
        packet['camera_rotation'][144:]=np.nan
        packet['camera_translation_m'][144:]=np.nan
        np.testing.assert_array_equal(ImageObservation(packet).features(143,*self.state(142)),before)

    def test_missing_and_evaluation_knee_slots_zero_without_hiding_other_points(self):
        packet={k:v.copy() for k,v in self.packet.items()}
        observation=ImageObservation(packet)
        before=observation.features(143,*self.state(142))
        self.assertTrue(np.isfinite(before).all())
        np.testing.assert_array_equal(before[358:].reshape(2,12,9)[:,9],0)
        packet['pixels'][143,:,9]=[100,100,1]
        np.testing.assert_array_equal(observation.features(143,*self.state(142)),before)
        packet['pixels'][143,0,0]=np.nan
        after=observation.features(143,*self.state(142))
        np.testing.assert_array_equal(after[358:].reshape(2,12,9)[0,0],0)
        np.testing.assert_array_equal(after[358:].reshape(2,12,9)[1],before[358:].reshape(2,12,9)[1])

    def test_exact_observations_give_unit_image_reward_and_behind_camera_is_failure(self):
        packet={k:v.copy() for k,v in self.packet.items()}
        observation=ImageObservation(packet)
        state=self.state(143)
        packet['pixels'][143,:,:,:2]=observation.project(observation.points(*state[:2]),143)
        packet['pixels'][143,:,:,2]=1
        _,image,_=observation.reward(143,*state[:2],np.zeros(69))
        self.assertAlmostEqual(image,1.,places=12)
        packet['camera_translation_m'][143]=state[0][0]+packet['camera_rotation'][143]@np.array([0,0,10.])
        _,image,_=observation.reward(143,*state[:2],np.zeros(69))
        self.assertEqual(image,0.)


if __name__=='__main__':
    unittest.main()
