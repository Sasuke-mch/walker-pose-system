import json,re
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
z=np.load(p/'fit/result.npz'); s=json.loads((p/'fit/fit_summary.json').read_text(encoding='utf8'))
initial=s['checkpoints']['initial']; final=list(s['checkpoints'].values())[-1]
a=json.loads((p/'fit/grasp_geometry_audit.json').read_text(encoding='utf8'))
t=json.loads((p/'fit/update_transactions.json').read_text(encoding='utf8'))['updates']
guard=np.load(p/'fit/observation_guard_reference.npz')
error=np.linalg.norm(z['predicted_coco']-z['raw_triangulated_points']/1000,axis=-1)
excess=float((error-guard['baseline_error_m'])[guard['valid']].max()*1000)
assert np.array_equal(z['pair_id'],np.arange(448))
assert z['vertices'].shape==(448,6890,3) and z['faces'].shape==(13776,3)
assert z['faces'].min()>=0 and z['faces'].max()<6890
assert all(np.isfinite(z[key]).all() for key in ('vertices','predicted_coco','vposer_latent','body_rotation_matrices'))
old=re.search(r'<script id="data" type="application/json">(.*?)</script>',(p/'body_viewer.html').read_text(encoding='utf8'),re.S).group(1)
new=re.search(r'<script id="data" type="application/json">(.*?)</script>',(p/'body_viewer_v2.html').read_text(encoding='utf8'),re.S).group(1)
assert old==new
gates=dict(observations=excess<=10.002,wrists=all(r['p95']<=10 for r in final['wrist_error_mm'].values()),
    feet=all(v>=-3 for v in final['sole_minimum_z_mm'].values()),
    leg_temporal=final['terms']['leg_ground_acceleration']<=initial['terms']['leg_ground_acceleration'],
    hands=all(r['passed_geometry_proxy'] for r in a['records']))
report=dict(frames=448,finite=True,maximum_observation_regression_mm=excess,
    checkpoints=s['checkpoints'],hand_geometry=a['summary'],
    updates={k:dict(accepted=sum(r['accepted'] for r in t if r['stage']==k),total=sum(r['stage']==k for r in t)) for k in s['checkpoints'] if k!='initial'},
    viewer_payload_unchanged=True,gates=gates,passed=all(gates.values()),accepted_for_main_fit=False)
(p/'full448_assessment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(initial_rms_mm=initial['body_3d_rms_mm'],final_rms_mm=final['body_3d_rms_mm'],
    wrists=final['wrist_error_mm'],feet=final['sole_minimum_z_mm'],hand_geometry=a['summary'],
    updates=report['updates'],max_regression_mm=excess,gates=gates),ensure_ascii=False,indent=2))
