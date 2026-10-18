"""Display a saved PhysHMR run: same-run neutral SMPL and reference joints."""

import argparse
import base64
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .contract import BASE_ROTATION, SMPL_INDICES


def export(run, model_path, output):
    import smplx
    from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility

    if output.exists():
        raise FileExistsError(f'Preserve the existing visualization: {output}')
    _install_legacy_smpl_pickle_compatibility()
    torch.set_num_threads(2)
    with np.load(run / 'trajectory.npz', allow_pickle=False) as saved:
        position = saved['position'].copy()
        reference = saved['reference_position'].copy()
        quaternion = saved['rotation_xyzw'].copy()
        termination = saved['termination'].copy()
    runtime = json.loads((run / 'runtime.json').read_text(encoding='utf-8'))
    count = len(position)
    if position.shape != (count, 24, 3) or reference.shape != position.shape:
        raise ValueError('Expected all saved frames and 24 native body origins')
    finite = (np.isfinite(position).all((1, 2)) & np.isfinite(reference).all((1, 2))
              & np.isfinite(quaternion).all((1, 2)))
    # This exporter does not substitute geometry for an unavailable state.
    if not finite.all():
        raise ValueError(f'Unavailable frames retained in trajectory: {np.flatnonzero(~finite)}')
    model = smplx.SMPL(str(model_path), gender='neutral').eval()
    model.requires_grad_(False)
    world = Rotation.from_quat(quaternion.reshape(-1, 4)).as_matrix()
    world = world.reshape(count, 24, 3, 3)[:, np.argsort(SMPL_INDICES)] @ BASE_ROTATION
    local = np.stack([world[:, model.parents[j]].transpose(0, 2, 1) @ world[:, j]
                      for j in range(1, 24)], axis=1)
    with torch.no_grad():
        result = model(global_orient=torch.tensor(world[:, :1], dtype=torch.float32),
                       body_pose=torch.tensor(local, dtype=torch.float32),
                       betas=torch.zeros(count, 10), pose2rot=False)
    origin = result.joints[:, :1].numpy()
    vertices = result.vertices.numpy() - origin + position[:, :1]
    decoded = result.joints[:, :24].numpy() - origin + position[:, :1]
    discrepancy = np.linalg.norm(decoded[:, SMPL_INDICES] - position, axis=-1)
    # Catch wrong body order/basis/shape, allowing sub-mm PhysX constraint error.
    if discrepancy.max() > .001:
        raise ValueError(f'SMPL/native joint mismatch: {discrepancy.max()} m')
    faces = np.asarray(model.faces, dtype=np.uint32).reshape(-1)
    if vertices.shape != (count, 6890, 3) or faces.size != 41328 or faces.max() >= 6890:
        raise ValueError('Unexpected SMPL topology')
    native_parents = np.full(24, -1, dtype=int)
    inverse = np.argsort(SMPL_INDICES)
    for j in range(1, 24):
        native_parents[j] = inverse[int(model.parents[SMPL_INDICES[j]])]
    global_error = np.linalg.norm(position - reference, axis=-1).mean(-1) * 1000
    relative_error = np.linalg.norm((position - position[:, :1])
                                    - (reference - reference[:, :1]), axis=-1).mean(-1) * 1000

    def packed(array, dtype='<f4'):
        return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode('ascii')

    metadata = {
        'run': run.as_posix(), 'trajectory': (run / 'trajectory.npz').as_posix(),
        'model': model_path.as_posix(), 'source_key': runtime['source_key'],
        'frames': count, 'dt': runtime['control_dt_s'], 'vertices': 6890, 'faces': 13776,
        'native_joint_roundtrip_max_m': float(discrepancy.max()),
        'native_joint_roundtrip_mean_m': float(discrepancy.mean()),
        'reference': 'Exact saved same-time author body_pos_gt; input reference, not independent truth',
        'coordinates': 'Saved PhysX world meters; Z up; display map [X,-Y,Z]',
        'ground': 'Author simulation plane Z=0, not measured walker ground',
        'frame_policy': 'Every saved step, no interpolation/smoothing/centering by frame',
        'scope': 'Official AIST clip, not walker 448; no matching walker/cameras/Stage data',
    }
    data = dict(metadata, mesh=packed(vertices), faces_u32=packed(faces, '<u4'),
                position=packed(position), reference=packed(reference),
                parents=native_parents.tolist(), termination=termination.tolist(),
                global_error_mm=global_error.tolist(), relative_error_mm=relative_error.tolist())
    output.mkdir(parents=True)
    with gzip.open(output / 'data.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, separators=(',', ':'))
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    here = Path(__file__).parent
    shutil.copyfile(here / 'physhmr_view.html', output / 'index.html')
    shutil.copyfile(here / 'physhmr_view.js', output / 'viewer.js')
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.run.resolve(), args.model.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
