"""Author PHC SMPL body ordering and upright change of basis."""

import xml.etree.ElementTree as ET

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


def motor_effort_limits(xml, joint_names):
    """Read this author's scalar, symmetric MJCF motor torque bounds.

    Gym's position-drive importer does not retain these actuator limits.
    Restoring them preserves the author's fixture, not measured human strength.
    Unsupported actuator forms fail instead of silently inventing limits.
    """
    tree = ET.parse(xml)
    default = tree.find("default/motor")
    defaults = {} if default is None else default.attrib
    limits = {}
    for motor in tree.findall("actuator/motor"):
        attrs = {**defaults, **motor.attrib}
        name = attrs.get("joint")
        if name in limits or attrs.get("class") or attrs.get("ctrllimited") != "true":
            raise ValueError("unique, explicitly limited scalar motor required")
        gear = np.fromstring(attrs.get("gear", "1"), sep=" ")
        bounds = np.fromstring(attrs.get("ctrlrange", ""), sep=" ")
        if gear.shape != (1,) or bounds.shape != (2,):
            raise ValueError("scalar gear and two control bounds required")
        torque = np.sort(gear[0]*bounds)
        if (not np.isfinite(torque).all() or torque[0] >= 0 or torque[1] <= 0
                or not np.isclose(torque[0], -torque[1], rtol=1e-6, atol=1e-8)):
            raise ValueError("finite symmetric torque interval required")
        limits[name] = torque[1]
    if len(set(joint_names)) != len(joint_names) or set(limits) != set(joint_names):
        raise ValueError("actuators must match every requested joint exactly once")
    return np.array([limits[name] for name in joint_names], dtype=np.float32)
