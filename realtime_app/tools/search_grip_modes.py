"""Search plausible fixed grip modes without using a previous pose solution."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from pose_app.grip_mode_search import search_grip_modes, refine_grip_modes
from pose_app.global_hand_handle_pose import load_static_handle_ends, transform_ground_to_walker

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--result',type=Path,required=True); ap.add_argument('--walker-model',type=Path,required=True); ap.add_argument('--vertex-sets',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    z=np.load(args.result,allow_pickle=False); model=json.loads(args.walker_model.read_text(encoding='utf-8')); sets=json.loads(args.vertex_sets.read_text(encoding='utf-8'))['sets']
    n=z['vertices'].shape[0]; R=np.asarray(model['rotation_left_camera_from_walker'],float); t=np.asarray(model['translation_left_camera_from_walker_mm'],float)/1000
    R=np.repeat(R[None],n,axis=0); t=np.repeat(t[None],n,axis=0); ends=load_static_handle_ends(args.walker_model,n)
    out={'schema_version':'grip_mode_search_v1','status':'engineering_candidate','uses_previous_pose_solution':False,'relative_frame':'walker_rigid_frame','hands':{}}
    for s,j in (('left',0),('right',1)):
        idx=np.asarray(sets[f'{s}_palm_surface_candidate']['palm_fingers'],int); palm=transform_ground_to_walker(z['vertices'][:,idx],R,t)
        out['hands'][s]=search_grip_modes(palm,ends[:,j])
        out['hands'][s]['continuous_refinement']=refine_grip_modes(palm,ends[:,j],starts=24,bootstrap=0)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())
