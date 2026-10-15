"""Causal, explicit SMPL-H -> author neutral SMPL reference retargeting.

Only q_i, the frozen shape convention, scene_i and q_{i-1} enter reference_i.
No full-clip height correction, time resampling, smoothing or future derivative.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import BASE_ROTATION, NAMES, SMPL_INDICES

BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def retarget(time_s, root_position, root_rotation, body_rotation, hand_rotation,
             available, segment, xml):
    time_s = np.asarray(time_s)
    if time_s.ndim != 1 or not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0):
        raise ValueError('strictly increasing finite capture time required')
    count = len(time_s)
    if (np.shape(root_position) != (count, 3) or np.shape(root_rotation) != (count, 3, 3)
            or np.shape(body_rotation) != (count, 21, 3, 3)
            or np.shape(hand_rotation) != (2, 3, 3)
            or np.shape(available) != (count,) or np.shape(segment) != (count,)):
        raise ValueError('SMPL-H reference shapes do not agree')
    tree = ET.parse(xml)
    offsets = [np.zeros(3)]
    parents = [-1]
    parent_lookup = {child.get('name'): parent.get('name') for parent in tree.findall('.//body')
                     for child in parent.findall('body')}
    for name in NAMES[1:]:
        offsets.append(np.fromstring(tree.find(f'.//body[@name="{name}"]').get('pos'), sep=' '))
        parents.append(NAMES.index(parent_lookup[name]))
    offsets = np.asarray(offsets)
    positions = np.full((count, 24, 3), np.nan)
    rotations = np.full((count, 24, 3, 3), np.nan)
    velocity = np.full_like(positions, np.nan)
    for i in range(count):
        if not available[i]:
            continue
        if not all(np.isfinite(a).all() for a in (root_position[i], root_rotation[i], body_rotation[i])):
            raise ValueError('available source frame contains nonfinite model parameters')
        global_smpl = [root_rotation[i]]
        for j in range(1, 22):
            global_smpl.append(global_smpl[BODY_PARENTS[j]] @ body_rotation[i, j-1])
        # The released policy has two coarse terminal hand links. Use the first
        # native SMPL-H finger-base orientation only to initialize those links;
        # this does not infer or control all finger articulation.
        global_smpl += [global_smpl[20] @ hand_rotation[0], global_smpl[21] @ hand_rotation[1]]
        rotations[i] = np.asarray(global_smpl)[SMPL_INDICES] @ BASE_ROTATION.T
        positions[i, 0] = root_position[i]
        for j in range(1, 24):
            positions[i, j] = positions[i, parents[j]] + rotations[i, parents[j]] @ offsets[j]
        velocity[i] = 0.
        if i and available[i-1] and segment[i] == segment[i-1]:
            velocity[i] = (positions[i]-positions[i-1])/(time_s[i]-time_s[i-1])
    return dict(timestamp_s=time_s, reference_position_m=positions,
                reference_rotation=rotations, reference_velocity_m_s=velocity,
                available=np.asarray(available, bool), segment_id=np.asarray(segment))


def prepare(source, scene, xml, output):
    if output.exists():
        raise FileExistsError(output)
    # The trusted local scene export stores Stage labels as an object array.
    with np.load(source, allow_pickle=False) as fitted, np.load(scene, allow_pickle=True) as world:
        ids = fitted['pair_id']
        if ids.tolist() != list(range(448)):
            raise ValueError('expected all448 source IDs in order, including unavailable states')
        time_s = fitted['timestamp_s']
        np.testing.assert_allclose(time_s, world['pair_timestamp_sec'], atol=1e-7, rtol=0)
        rotation, translation = world['rotation_ground_from_left'], world['translation_ground_from_left_mm']/1000
        # Source stores native joints in camera coordinates, but model root
        # orientations in both camera and world. Use the actual pelvis joint,
        # not SMPL transl, because SMPL rotates about shaped rest pelvis J0.
        pelvis = np.einsum('nij,nj->ni', rotation, fitted['smplh_joints'][:, 0])+translation
        root_rotation = fitted['world_root_rotation']
        np.testing.assert_allclose(root_rotation, rotation @ fitted['root_rotation'], atol=1e-6, rtol=0)
        hand = np.stack([Rotation.from_rotvec(fitted[key][0].reshape(15, 3)[0]).as_matrix()
                         for key in ('left_hand_pose', 'right_hand_pose')])
        result = retarget(time_s, pelvis, root_rotation, fitted['body_rotations'], hand,
                          fitted['pose_available'], fitted['segment_id'], xml)
        result.update(pair_id=ids, betas=fitted['betas'], source_pelvis_world_m=pelvis,
                      camera_rotation=rotation, camera_translation_m=translation,
                      stage=world['stage'].astype(str))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    metadata = dict(source=str(source.resolve()), scene=str(scene.resolve()), xml=str(xml.resolve()),
        output=str(output.resolve()), frames=448, duration_s=float(time_s[-1]-time_s[0]),
        source_model='complete causal SMPL-H state estimator; frozen full-clip pre-calibrated beta',
        scene_semantics='frozen same-source Stage conditional ground, nominal0.5s publication lag; not independent world truth',
        retarget='preserve current world root/pelvis and joint rotations; author neutral body lengths',
        terminal_hands='coarse2 links initialized from first finger-base rotations, not full hands',
        reference_velocity='backward difference within segment, zero on initialization/reinitialization',
        future_reference_frames=0, height_fix=False, smoothing=False,
        accuracy='retargeted reference is not independent3D truth; beta calibration and conditional scene disclosed')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.scene, args.xml, args.output)


if __name__ == '__main__':
    main()
