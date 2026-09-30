import unittest
import numpy as np
from tools.audit_fixed_mano_grip import crossing_pairs, capsule


class GeometryScreenTests(unittest.TestCase):
    def test_triangle_crossing_is_detected(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[.2,.2,-1],[.2,.2,1],[.8,.2,1.]])
        self.assertEqual(crossing_pairs(v,np.array([[0,1,2],[3,4,5]])),[[0,1]])

    def test_separated_triangles_do_not_cross(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[.2,.2,1],[1,.2,1],[.8,.8,1.]])
        self.assertEqual(crossing_pairs(v,np.array([[0,1,2],[3,4,5]])),[])

    def test_adjacent_mesh_faces_are_excluded(self):
        v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]])
        self.assertEqual(crossing_pairs(v,np.array([[0,1,2],[0,1,3]])),[])

    def test_capsule_uses_segment_ends_and_signed_radius(self):
        p=np.array([[.5,0,0],[.5,.2,0],[2,0,0]])
        np.testing.assert_allclose(capsule(p,np.array([0.,0,0]),np.array([1.,0,0]),.1),[-.1,.1,.9])


if __name__=='__main__':unittest.main()
