"""Paper-style architecture comparison; schematic modules, real neutral SMPL-H."""
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from realtime_app.learning_flow_mesh.body import BodyDecoder

OUT = Path("research_records/research/G20261007_flow_mesh_baseline_v1/architecture_figure_v2")
C = {"old": "#edf0f3", "body": "#e2eef8", "history": "#e8e5f4", "shape": "#f7e9ce",
     "hand": "#dceee6", "flow": "#e1eaf3", "edge": "#303b49", "new": "#a16622"}


def box(ax, x, y, w, h, label, color="body", small=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                              facecolor=C[color], edgecolor=C["edge"], linewidth=1.1))
    ax.text(x+w/2, y+h-2.5, label, ha="center", va="top", fontsize=10, weight="bold")
    if small:
        ax.text(x+w/2, y+2.0, small, ha="center", va="bottom", fontsize=8.0, color=C["edge"])


def arrow(ax, points, color=None, dashed=False):
    color = color or C["edge"]
    for a, b in pairwise(points[:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.25,
                ls="--" if dashed else "-")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=11, lw=1.25, color=color,
                                linestyle="--" if dashed else "-"))


def tokens(ax, x, y, n=5, color="#789dbc", width=6, height=5):
    for k in range(n):
        ax.add_patch(Rectangle((x+k*1.3, y+k*.55), width, height,
                              fc="white", ec=color, lw=1))
        for row in range(3):
            ax.plot([x+k*1.3+.5, x+k*1.3+width-.5],
                    [y+k*.55+1+row*1.3]*2, color=color, lw=.65)


def graph(ax, x, y, scale=1, color=None):
    pts=np.array([[0,4],[0,2],[0,0],[-2,2],[2,2],[-3,0],[3,0],[-1,-2],[1,-2],[-1,-4],[1,-4]])*scale+[x,y]
    for a,b in [(0,1),(1,2),(1,3),(1,4),(3,5),(4,6),(2,7),(2,8),(7,9),(8,10)]:
        ax.plot(pts[[a,b],0],pts[[a,b],1],color=color or C["edge"],lw=1.0)
    ax.scatter(pts[:,0],pts[:,1],s=9,color=color or "#617f9e",zorder=3)


def gru(ax, x, y):
    for k in range(3):
        ax.add_patch(FancyBboxPatch((x+k*5.4,y),4.5,5,boxstyle="round,pad=.1",fc="white",ec="#786c9e"))
        ax.text(x+k*5.4+2.25,y+2.5,"h",ha="center",va="center",fontsize=11)
        if k<2:arrow(ax,[(x+k*5.4+4.5,y+2.5),(x+(k+1)*5.4,y+2.5)])


def mesh_image():
    import json
    config=json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text())
    decoder=BodyDecoder(**config["assets"])
    with torch.no_grad():
        out=decoder(torch.eye(3).expand(1,1,52,3,3),torch.zeros(1,10),torch.zeros(1,1,3))
    v=out["vertices"][0,0].numpy()
    f=out["faces"].numpy()
    # Front orthographic view of the actual licensed neutral-pose male asset.
    tri=v[f]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    shade=.52+.34*np.abs(normals@np.array([.25,.4,.88]))
    order=np.argsort(tri[:,:,2].mean(axis=1))
    fig,ax=plt.subplots(figsize=(3,3.6),dpi=170)
    ax.add_collection(PolyCollection(tri[order,:,:2],facecolors=plt.cm.Greys(1-shade[order]),
                                      edgecolors=(.15,.2,.25,.11),linewidths=.1))
    ax.autoscale();ax.set_aspect("equal");ax.axis("off")
    fig.patch.set_alpha(0);ax.patch.set_alpha(0);fig.tight_layout(pad=0)
    path=OUT/"smplh_neutral_pose.png";fig.savefig(path,transparent=True);plt.close(fig)
    return plt.imread(path)


def render():
    OUT.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none","pdf.fonttype":42})
    mesh=mesh_image()
    fig,ax=plt.subplots(figsize=(16,10),dpi=240)
    fig.subplots_adjust(left=.025,right=.985,top=.97,bottom=.035)
    ax.set_xlim(0,160);ax.set_ylim(0,100);ax.axis("off")
    ax.text(1,98,"Conditional SMPL-H reconstruction: baseline and H8",fontsize=18,weight="bold")
    ax.text(1,93.8,"(a) Original baseline",fontsize=12,weight="bold")
    box(ax,2,77,19,12,"Observations","old", "xy / score / rays")
    graph(ax,8,82,.45);tokens(ax,13,80,2,width=3,height=3)
    box(ax,27,77,22,12,"Condition encoder","old","52 learned queries")
    tokens(ax,32,80,4,height=3)
    box(ax,55,77,23,12,"Causal GRU","old","all-frame update")
    gru(ax,59,80)
    box(ax,88,77,28,12,"Pose flow","old","graph + attention / Euler")
    graph(ax,94,82,.45);tokens(ax,101,80,4,height=3)
    box(ax,130,75,27,16,"SMPL-H","old","pose + shape + T")
    ax.imshow(mesh,extent=(141,153,77,86),zorder=4)
    for a,b in [(21,27),(49,55),(78,88),(116,130)]:arrow(ax,[(a,83),(b,83)])
    box(ax,58,63,19,8,"First-frame β","shape")
    box(ax,92,63,17,8,"Translation T","old")
    arrow(ax,[(66,77),(66,71)])
    arrow(ax,[(77,67),(84,67),(84,72),(124,72),(124,79),(130,79)])
    arrow(ax,[(75,77),(75,73),(100,73),(100,71)])
    arrow(ax,[(109,67),(127,67),(127,76),(130,76)])
    ax.text(3,65,"β enters the body decoder only",fontsize=9,color="#6d7279")
    ax.plot([1,158],[59,59],color="#b6bdc5",lw=.8)
    ax.text(1,55.5,"(b) Current H8 research architecture",fontsize=12,weight="bold")

    box(ax,2,28,19,18,"Body inputs","body","COCO17 / 2 views")
    graph(ax,8,36,.85);graph(ax,15,36,.85)
    box(ax,27,28,22,18,"Cross-attention","body","52 × 64 features")
    tokens(ax,30,33,6,width=7,height=6)
    ax.text(38,32,"Q ← observations",fontsize=7.3,ha="center")
    box(ax,56,28,25,18,"Local + GRU context","history","H6: carry / missing freeze")
    gru(ax,60,34)
    box(ax,105,26,26,20,"Pose velocity","flow","2 spatial blocks / 8 Euler")
    graph(ax,111,34,.8);tokens(ax,118,31,4,width=5,height=6)
    ax.text(118,29.7,"52 × 6D → rotations",fontsize=7.5,ha="center")
    ax.add_patch(Circle((94,37),2.1,fc="white",ec=C["edge"],lw=1.2))
    ax.text(94,37,"+",ha="center",va="center",fontsize=14)
    arrow(ax,[(21,37),(27,37)])
    arrow(ax,[(49,37),(56,37)])
    arrow(ax,[(81,37),(91.8,37)])
    arrow(ax,[(96.2,37),(105,37)])

    box(ax,31,48,45,6,"Local features → K=8 pool → β head","shape")
    arrow(ax,[(43,46),(43,48)],color=C["new"])
    box(ax,86,48,26,6,"H8: β projection 10→64","shape")
    arrow(ax,[(76,51),(86,51)],color=C["new"])
    arrow(ax,[(94,48),(94,39.2)],color=C["new"])
    ax.text(82,44,"ready-gated",fontsize=7.7,color=C["new"])

    box(ax,27,7,23,14,"Native hand tokens","hand","2 views × 2 hands × 15")
    tokens(ax,32,10,4,color="#638f7b",width=5,height=4)
    box(ax,57,7,25,14,"Hand cross-attention","hand","H5: hand-query residual")
    graph(ax,68,12.5,.4,color="#638f7b")
    arrow(ax,[(50,14),(57,14)],color="#638f7b")
    arrow(ax,[(82,14),(94,14),(94,34.8)],color="#638f7b")
    arrow(ax,[(73,28),(73,24),(66,24),(66,21)],color="#638f7b",dashed=True)

    box(ax,139,21,19,31,"SMPL-H decoder","body","6,890 vertices")
    ax.imshow(mesh,extent=(140,157,25,46),zorder=4)
    arrow(ax,[(131,37),(139,37)])
    arrow(ax,[(80,51),(80,57),(135,57),(135,46),(139,46)],color=C["new"])
    ax.text(127,57.8,"shared β",fontsize=7.8,color=C["new"])
    box(ax,106,8,24,9,"Translation head","history","T from history features")
    arrow(ax,[(80,28),(85,28),(85,12.5),(106,12.5)])
    arrow(ax,[(130,12.5),(135,12.5),(135,24),(139,24)])
    box(ax,139,5,19,10,"Publish gate","old","ready / observed / valid")
    arrow(ax,[(148.5,21),(148.5,15)])
    ax.text(107,22,"noise z₀ + flow time τ",fontsize=8.5,ha="center")
    arrow(ax,[(112,23),(112,26)])
    ax.text(2,1,"Added branches: hand conditions (H5), shared-shape calibration (H7), shape-to-pose coupling (H8).  Backbone and spatial flow blocks retained.",fontsize=8.5)
    for suffix in ("png","svg","pdf"):
        fig.savefig(OUT/f"architecture_comparison.{suffix}",dpi=240,facecolor="white")
    plt.close(fig)
    print(str(OUT.resolve()))


if __name__=="__main__":
    render()
