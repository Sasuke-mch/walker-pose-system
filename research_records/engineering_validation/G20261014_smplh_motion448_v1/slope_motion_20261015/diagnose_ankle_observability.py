"""Free-point stereo height profiles, diagnostic only; never body fit inputs."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import prefix_derivative_feedback as shared  # noqa: E402
from pose_app.fisheye_camera import fisheye_project_numpy  # noqa: E402


def main():
    source = OUT.parent/'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    p = json.loads((source/'run_parameters.json').read_text(encoding='utf-8'))
    cal = shared.load_stereo_fisheye(Path(p['inputs']['calibration_dir']))
    left = shared._load_predictions(Path(p['inputs']['left_predictions']), 'left')
    right = shared._load_predictions(Path(p['inputs']['right_predictions']), 'right')
    saved = np.load(source/'result.npz', allow_pickle=False)
    scene = np.load(OUT/'scene/scene_transforms.npz', allow_pickle=False)
    rotations = scene['rotation_ground_from_left']
    translations = scene['translation_ground_from_left_mm']/1000
    records = [json.loads(s) for s in (OUT/'first_instance_strict.jsonl').read_text().splitlines()]
    rows = []
    for i in (35, 37, 48, 59, 61):
        raw = records[i]['persons_3d'][0]['keypoints_3d']
        R, t = rotations[i], translations[i]

        def project(world):
            camera = (world-t) @ R
            other = camera @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm/1000
            return np.r_[fisheye_project_numpy(camera, cal.K0, cal.D0),
                         fisheye_project_numpy(other, cal.K1, cal.D1)]

        for j in (15, 16):
            target = np.r_[left[i, j, :2], right[i, j, :2]]
            initial = R @ (np.array(raw[j]['xyz'])/1000)+t
            fit = least_squares(lambda w: project(w)-target, initial,
                                xtol=1e-12, ftol=1e-12, gtol=1e-9, max_nfev=200)
            point = fit.x
            model = R @ saved['predicted_coco'][i, j]+t
            baseline_height = (rotations[35] @ saved['predicted_coco'][35, j]+translations[35])[2]
            eps = 1e-5
            J = np.stack([(project(point+axis*eps)-project(point-axis*eps))/(2*eps)
                          for axis in np.eye(3)], axis=1)
            _, singular, vectors = np.linalg.svd(J, full_matrices=False)
            weak = vectors[-1]
            diff = model-point
            profiles = []
            for name, height in [('free_optimum', point[2]), ('model_height', model[2]),
                                 ('model_frame35_height', baseline_height)]:
                result = least_squares(lambda xy: project(np.r_[xy, height])-target,
                    point[:2], xtol=1e-12, ftol=1e-12, gtol=1e-9, max_nfev=200)
                errors = (project(np.r_[result.x, height])-target).reshape(2, 2)
                profiles.append({'name': name, 'height_mm': float(height*1000),
                    'view_error_px': np.linalg.norm(errors, axis=1).tolist(),
                    'squared_pixel_cost': float(np.sum(errors**2)), 'optimizer_success': bool(result.success)})
            rows.append({'frame': i, 'joint': j, 'strict_accepted': bool(raw[j]['valid']),
                'raw_view_targets_px': target.reshape(2, 2).tolist(),
                'free_point_world_mm': (point*1000).tolist(), 'model_point_world_mm': (model*1000).tolist(),
                'model_minus_free_world_mm': (diff*1000).tolist(),
                'model_view_error_px': np.linalg.norm((project(model)-target).reshape(2, 2), axis=1).tolist(),
                'singular_values_px_per_m': singular.tolist(), 'weak_direction_world': weak.tolist(),
                'model_error_weak_direction_abs_cosine': float(abs(diff@weak)/max(np.linalg.norm(diff), 1e-12)),
                'linearized_sigma8_world_height_std_mm': float(np.sqrt(np.linalg.inv(J.T@J)[2, 2])*8*1000),
                'height_profiles': profiles, 'free_optimizer_success': bool(fit.success)})
    report = {'question': 'Can stereo pixels distinguish the model ankle height from a lower-height alternative?',
        'frames': [35, 37, 48, 59, 61], 'pixel_sigma_for_sensitivity_only': 8,
        'scope': 'Independent XYZ and fixed-height XY optimization of ankle observations only;diagnostic,not an anatomical/body optimization or truth',
        'limits': ['sigma8 local linearization is sensitivity scale,not calibrated uncertainty',
                   'frame35 model ankle height is a counterfactual reference,not ground contact truth',
                   'No diagnostic point,profile or height is fed back to the tracker'], 'rows': rows}
    (OUT/'ankle_observability_diagnostic.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for row in rows:
        print(row['frame'], row['joint'], 'delta height', row['model_minus_free_world_mm'][2],
              'sigma8height', row['linearized_sigma8_world_height_std_mm'],
              'weakcos', row['model_error_weak_direction_abs_cosine'],
              'profiles', [(v['name'], np.round(v['view_error_px'], 2).tolist()) for v in row['height_profiles']], flush=True)


if __name__ == '__main__':
    main()
