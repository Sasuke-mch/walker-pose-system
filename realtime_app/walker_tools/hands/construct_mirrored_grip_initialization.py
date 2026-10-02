"""Construct mirrored hand parameters from an opposite-side engineering grasp.

This exports initialization parameters, never fabricates a target-hand mesh.
The downstream solver must generate and check the real SMPL-H target surface.
"""

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import PROJECT_ROOT as _tool_project_root
_tool_prepare_imports()
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np


def mirrored_left_parameters(right_pose, right_rotation, right_wrist):
    reflection=np.diag([-1.,1.,1.])
    pose=np.asarray(right_pose).reshape(15,3)*[1,-1,-1]
    rotation=reflection@np.asarray(right_rotation)@reflection
    wrist=reflection@np.asarray(right_wrist)
    return pose.reshape(1,45),rotation,wrist


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--grip','--right-grip',dest='right_grip',type=Path,required=True)
    ap.add_argument('--source-side',choices=('left','right'),default='right')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise ValueError('refuse_existing_output')
    z=np.load(args.right_grip,allow_pickle=False)
    source=args.source_side
    target='left' if source=='right' else 'right'
    pose,R,t=mirrored_left_parameters(z[f'{source}_hand_pose'],z[f'{source}_rotation_walker_from_wrist'],z[f'{source}_wrist_walker_m'])
    root=_tool_project_root
    sys.path.insert(0,str(root/'realtime_app'))
    from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility
    _install_legacy_smpl_pickle_compatibility()
    with (root/f'third_party/WiLoR/mano_data/models/MANO_{target.upper()}.pkl').open('rb') as f:
        mano=pickle.load(f,encoding='latin1')
    components=np.asarray(mano['hands_components'])[:z[f'{source}_hand_pca'].shape[1]]
    normalized=components/np.linalg.norm(components,axis=1,keepdims=True)
    coeff=np.linalg.lstsq(normalized.T,pose.ravel()-mano['hands_mean'],rcond=None)[0][None]
    error=float(np.max(np.abs(coeff@normalized+mano['hands_mean']-pose)))
    if error>1e-5:raise ValueError('mirrored_pose_not_representable_in_target_PCA')
    arrays={f'{source}_{k}':z[f'{source}_{k}'] for k in ('hand_pca','hand_pose','rotation_walker_from_wrist','wrist_walker_m')}
    arrays.update({f'{target}_hand_pca':coeff,f'{target}_hand_pose':pose,
                   f'{target}_rotation_walker_from_wrist':R,f'{target}_wrist_walker_m':t})
    arrays.update(mirror_pca_max_error=np.asarray(error),scope=np.asarray('constructed_initialization_not_observed_target_hand'),
                  source=np.asarray(str(args.right_grip.resolve())),source_side=np.asarray(source))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**arrays)
    print('Saved bilateral construction initialization; actual target mesh requires forward model.')


if __name__=='__main__':main()
