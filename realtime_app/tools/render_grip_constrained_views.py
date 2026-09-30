"""Render observed versus constrained grip surfaces in the walker frame."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def disp(p):
    p=np.asarray(p,float); return np.stack((p[...,0],-p[...,1],p[...,2]),axis=-1)

def rod(ax,a,b,r=.014,c='#4d5963'):
    a=np.asarray(a); b=np.asarray(b); d=b-a; L=np.linalg.norm(d)
    if L<1e-9:return
    d=d/L; ref=np.array([0.,0.,1.]) if abs(d[2])<.9 else np.array([1.,0.,0.])
    u=np.cross(d,ref);u/=np.linalg.norm(u);v=np.cross(d,u);t=np.linspace(0,2*np.pi,20)
    X=np.array([a+d*z+r*(np.cos(t)[:,None]*u+np.sin(t)[:,None]*v) for z in [0,L]])
    ax.plot_surface(X[...,0],X[...,1],X[...,2],color=c,alpha=.9,linewidth=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--search',type=Path,required=True);ap.add_argument('--walker-model',type=Path,required=True);ap.add_argument('--vertex-sets',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    z=np.load(args.result,allow_pickle=False); model=json.loads(args.walker_model.read_text(encoding='utf-8')); sets=json.loads(args.vertex_sets.read_text(encoding='utf-8'))['sets']; search=json.loads(args.search.read_text(encoding='utf-8'))
    R=np.asarray(model['rotation_left_camera_from_walker'],float);t=np.asarray(model['translation_left_camera_from_walker_mm'],float)/1000;n=z['vertices'].shape[0]; names=list(model['nodes_walker_mm']); nodes={k:np.asarray(v,float)/1000 for k,v in model['nodes_walker_mm'].items()}; edges=model['edges']; handles=model['handle_segments']
    views={'front':(18,-72),'side':(12,18),'top':(78,-72)}
    for side,j in [('left',0),('right',1)]:
        idx=np.asarray(sets[f'{side}_palm_surface_candidate']['palm_fingers'],int); p=z['vertices'][:,idx]; pw=np.einsum('ij,nvj->nvi',R.T,p-t); observed=np.median(pw,axis=0); sr=search['hands'][side]['continuous_refinement']; optimized=np.asarray(sr['shared_surface_walker_m']); ends=np.asarray([nodes[x] for x in handles[side]])
        allp=np.concatenate([observed,optimized,np.asarray(list(nodes.values()))]); q=disp(allp); lo,hi=q.min(0),q.max(0); c=(lo+hi)/2; span=max(hi-lo)*1.35;c[2]=max(c[2],.75)
        for name,(elev,azim) in views.items():
            fig=plt.figure(figsize=(11,8),dpi=180);ax=fig.add_subplot(111,projection='3d')
            for a,b in [(nodes[a],nodes[b]) for a,b in edges]:rod(ax,disp(a),disp(b))
            eo=disp(ends);ax.plot(eo[:,0],eo[:,1],eo[:,2],color='#111111',lw=5,label='handle')
            oo=disp(observed); xx=disp(optimized);ax.scatter(oo[:,0],oo[:,1],oo[:,2],s=3,c='#f39c12',alpha=.65,label='observed median palm');ax.scatter(xx[:,0],xx[:,1],xx[:,2],s=4,c='#c62828',alpha=.85,label='constrained palm')
            for a,b in zip(oo[::max(1,len(oo)//40)],xx[::max(1,len(xx)//40)]):ax.plot([a[0],b[0]],[a[1],b[1]],[a[2],b[2]],c='#777777',lw=.45,alpha=.4)
            ax.set_xlim(c[0]-span/2,c[0]+span/2);ax.set_ylim(c[1]-span/2,c[1]+span/2);ax.set_zlim(max(0,c[2]-span/2),c[2]+span/2);ax.set_xlabel('X walker (m)');ax.set_ylabel('-Y walker (m)');ax.set_zlabel('Z walker (m)');ax.view_init(elev,azim);d=sr['diagnostics'];ax.set_title(f'{side} grip search | {name} | observed vs constrained\nmedian/P95 deviation {d["observation_error_median_mm"]:.1f}/{d["observation_error_p95_mm"]:.1f} mm; within 3 mm {100*d["within_3mm_vertex_fraction"]:.1f}%');ax.legend(loc='upper left');ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(args.output_dir/f'grip_{side}_{name}.png',facecolor='white');plt.close(fig)
if __name__=='__main__':main()
