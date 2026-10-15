"""Author PHC SMPL body ordering and upright change of basis."""

import numpy as np
from scipy.spatial.transform import Rotation

NAMES = ('Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee',
         'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head',
         'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand',
         'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand')
SMPL_INDICES = np.array([0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15,
                        13, 16, 18, 20, 22, 14, 17, 19, 21, 23])
BASE_ROTATION = Rotation.from_quat([.5, .5, .5, .5]).as_matrix()
COLLISION_FILTERS = [0, 0, 7, 16, 12, 0, 56, 2, 33, 128, 0, 192,
                     0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
