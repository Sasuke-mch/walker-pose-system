import numpy as np
from pose_app.grip_mode_search import search_grip_modes

def test_search_is_invariant_to_common_translation():
    rng=np.random.default_rng(3); base=rng.normal(size=(30,3))*.01; n=8
    ends=np.zeros((n,2,3)); ends[:,0]=[-.2,0,0]; ends[:,1]=[.2,0,0]
    palm=np.repeat(base[None],n,axis=0)+[0,.03,.02]
    a=search_grip_modes(palm,ends); shift=np.stack([np.array([i*.2,.1,0]) for i in range(n)])
    b=search_grip_modes(palm+shift[:,None],ends+shift[:,None]); assert a['modes'][0]['mode']==b['modes'][0]['mode']
