"""Draw a source-checked, editable paper figure for the implemented baseline."""
import argparse
import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "research_records/research/G20261007_flow_mesh_baseline_v1"
COLORS = {"ink": "#253142", "blue": "#edf4fb", "green": "#edf6f2",
          "orange": "#fff3e8", "purple": "#f2eef8", "gray": "#f6f7f9"}


def draw(output, language):
    zh = language == "zh"
    def label(en, cn):
        return cn if zh else en
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
                         "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(18, 10.5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 18), ylim=(0, 10.5))
    ax.axis("off")
    ink = COLORS["ink"]
    texts = []

    def text(x, y, s, size=11, weight="normal", align="center", **kwargs):
        artist = ax.text(x, y, s, ha=align, va="center", fontsize=size,
                         weight=weight, color=ink, linespacing=1.55, **kwargs)
        texts.append(artist)
        return artist

    def box(x, y, w, h, s, color="gray", size=11, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                   fc=COLORS[color], ec=ink, lw=1.1,
                                   linestyle="--" if dashed else "-"))
        text(x + w / 2, y + h / 2, s, size)

    def arrow(points, dashed=False, color=ink):
        for p, q in pairwise(points[:-1]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.15,
                    ls="--" if dashed else "-", solid_capstyle="round")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=11, lw=1.15, color=color,
                                    linestyle="--" if dashed else "-",
                                    shrinkA=0, shrinkB=0))

    def panel(x, y, w, h, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                                   fc="white", ec="#b8bec6", lw=0.8))
        text(x + 0.16, y + h - .24, title, 12.2, "bold", "left")

    text(.45, 10.08, label("Conditional Flow Matching for SMPL-H Reconstruction",
                          "条件 Flow Matching 人体模型架构"), 20, "bold", "left")
    text(.45, 9.7, label("Implemented research baseline  |  52 rotation nodes  |  causal conditioning",
                         "已实现的研究 baseline  ·  52 个旋转节点  ·  因果条件编码"),
         11, align="left")
    panel(.4, 5.2, 17.2, 4.1, label("(a) Conditional generation and differentiable body decoding",
                                   "(a) 条件生成与可微人体解码"))
    box(.65, 7.75, 2.5, .92,
        label("RGB / supplied features\nSmall CNN or feature projection",
              "RGB / 已有图像特征\n小型 CNN 或特征投影"), "blue", 10)
    box(.65, 6.25, 2.5, 1.04,
        label("2D points + confidence\nRay origins + directions\nPresence / geometry masks",
              "二维点 + 置信度\n射线原点 + 方向\n存在 / 几何掩码"), "blue", 10)
    text(1.9, 5.83, label("V views, J = 22 observed joints", "V 个视图，J = 22 个观测关节"), 9)

    box(3.6, 7.26, 3.15, 1.5,
        label("Condition encoder\n52 learned queries + cross-attention\nMean pool → causal GRU",
              "条件编码器\n52 个查询 + 交叉注意力\n均值池化 → 单向 GRU"), "blue", 10.5)
    arrow([(3.15, 8.22), (3.6, 8.22)])
    arrow([(3.15, 6.79), (3.35, 6.79), (3.35, 7.62), (3.6, 7.62)])
    ax.add_patch(Circle((7.19, 8.0), .22, fc=COLORS["green"], ec=ink, lw=1))
    text(7.19, 8.0, r"$c_f$", 12)
    arrow([(6.75, 8.0), (6.95, 8.0)])
    arrow([(7.42, 8.0), (7.85, 8.0)])
    text(5.18, 6.95, label("52 × 64 node features + broadcast history",
                          "52 × 64 节点特征 + 广播历史特征"), 9)
    box(3.6, 5.72, 3.15, .78,
        label("Shape head: shared β from initial frame\nTranslation head: t per frame (metres)",
              "形状头：首帧 β，全序列固定\n平移头：逐帧 t（米）"), "green", 9.5)
    arrow([(5.18, 7.26), (5.18, 7.05), (5.18, 6.5)])

    text(9.39, 8.86, r"$x_\tau$     +     $\tau \in [0,1]$", 12)
    arrow([(9.39, 8.66), (9.39, 8.42)])
    box(7.85, 7.25, 3.1, 1.16,
        label("Velocity field  vθ(xτ, τ, cf)\nState + time + condition\n2 spatial blocks → 6D velocity",
              "速度场  vθ(xτ, τ, cf)\n状态 + 时间 + 观测条件\n2 层空间模块 → 6D 速度"), "orange", 10.5)
    box(7.85, 5.72, 3.1, .78,
        label("Gaussian start → Euler · 8 steps\nOne candidate; shared noise across frames",
              "高斯初值 → Euler 积分 · 8 步\n单候选；序列各帧共享噪声"), "orange", 9.5)
    arrow([(9.39, 7.25), (9.39, 6.5)])
    text(10.8, 6.86, r"$x_{k+1}=x_k+v_\theta\!/8$", 10)
    arrow([(7.85, 6.11), (7.52, 6.11), (7.52, 7.48), (7.85, 7.48)], True)
    arrow([(10.95, 6.11), (11.35, 6.11)])
    box(11.35, 5.72, 2.18, .78,
        label("6D → SO(3)\n52 local rotation matrices",
              "6D → SO(3)\n52 个局部旋转矩阵"), "purple", 9.5)
    text(12.44, 7.0, label("Root 1 + body 21\nLeft hand 15 + right hand 15",
                           "根 1 + 身体 21\n左手 15 + 右手 15"), 9.3)
    text(12.44, 8.14, r"$x_1\in\mathbb{R}^{52\times6}$", 13)
    text(12.44, 7.68, label("Euclidean 6D transport", "欧氏 6D 空间输运"), 9)
    arrow([(13.53, 6.11), (13.98, 6.11)])
    box(13.98, 5.72, 2.15, .78,
        label("SMPL-H\nFixed, differentiable layer",
              "SMPL-H\n固定参数的可微人体层"), "purple", 10)
    arrow([(6.75, 6.1), (7.05, 6.1), (7.05, 5.4), (15.05, 5.4), (15.05, 5.72)])
    text(14.44, 5.4, r"$\beta,\ t_f$", 10,
         bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    arrow([(16.13, 6.11), (16.87, 6.11), (16.87, 6.67)])

    # Read a real saved prediction; no rerun or substitute body illustration.
    with np.load(RECORD / "outputs/fresh_flow/prediction.npz") as data:
        vertices = data["vertices"][0, 0].copy()
        faces = data["faces"].copy()
        assert vertices.shape == (6890, 3) and faces.shape == (13776, 3)
        assert np.isfinite(vertices).all()
    vertices -= vertices.mean(axis=0)
    projected = vertices[:, [0, 1]]
    projected *= min(1.82 / np.ptp(projected[:, 1]), 1.52 / np.ptp(projected[:, 0]))
    projected += np.array([16.8, 7.83])
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    intensity = .5 + .35 * np.abs(normals[:, 2]) / np.maximum(np.linalg.norm(normals, axis=1), 1e-10)
    order = np.argsort(vertices[faces, 2].mean(axis=1))
    rgba = np.column_stack((intensity, intensity, intensity, np.ones(len(faces))))
    mesh = PolyCollection(projected[faces[order]], facecolors=rgba[order],
                          edgecolors="none", rasterized=True, zorder=3)
    ax.add_collection(mesh)
    text(15.23, 8.7, label("3D body surface", "三维人体表面"), 11, "bold")
    text(15.22, 8.17, label("6,890 vertices\n13,776 faces\n52 kinematic joints",
                          "6,890 顶点\n13,776 三角面\n52 运动学关节"), 10)

    panel(.4, 1.34, 7.72, 3.55,
          label("(b) Spatial block · repeated twice", "(b) 空间模块 · 重复 2 次"))
    text(4.2, 4.22, r"$H = c_f + W_x x_\tau + e(\tau)$", 12)
    box(.78, 2.73, 1.1, .75, "Layer\nNorm", "gray", 10)
    arrow([(.58, 3.11), (.78, 3.11)])
    arrow([(1.88, 3.11), (2.18, 3.11), (2.18, 3.64), (2.53, 3.64)])
    arrow([(2.18, 3.11), (2.18, 2.74), (2.53, 2.74)])
    box(2.53, 3.28, 1.98, .71, label("Self-attention\nGlobal joint relations",
                                    "自注意力\n全局关节关系"), "orange", 9.5)
    box(2.53, 2.38, 1.98, .71, label("Tree graph · optional\nLocal kinematic relations",
                                    "运动树图 · 可切换\n局部运动学关系"), "green", 9.5, True)
    arrow([(4.51, 3.64), (4.77, 3.64), (4.77, 3.11), (5.14, 3.11)])
    arrow([(4.51, 2.74), (4.77, 2.74), (4.77, 3.11)])
    ax.add_patch(Circle((5.31, 3.11), .17, fc="white", ec=ink))
    text(5.31, 3.11, "+", 14)
    arrow([(.62, 3.11), (.62, 4.05), (5.31, 4.05), (5.31, 3.29)])
    arrow([(5.48, 3.11), (5.88, 3.11)])
    box(5.88, 2.73, 1.42, .75, label("Norm + MLP\nResidual add",
                                    "Norm + MLP\n残差相加"), "orange", 9.5)
    arrow([(7.3, 3.11), (7.76, 3.11)])
    text(4.2, 1.91, label("Row-normalized parent–child graph with self-loops; frame-local attention.",
                         "父子运动树含自环并按行归一化；注意力只在同一帧内计算。"), 9)
    text(4.2, 1.61, label("Width 64 · 4 heads · output head 64 → 6 per rotation node",
                         "宽度 64 · 4 个注意力头 · 每个旋转节点输出 6 维速度"), 9)

    panel(8.48, 1.34, 9.12, 3.55,
          label("(c) Training: linear conditional flow matching", "(c) 训练：线性条件 Flow Matching"))
    for x, s in [(9.13, r"$x_0$"), (10.64, r"$x_\tau$"), (12.2, r"$x_1^*$")]:
        ax.add_patch(Circle((x, 3.93), .075, fc=ink))
        text(x, 4.2, s, 12)
    arrow([(9.24, 3.93), (10.52, 3.93)])
    arrow([(10.75, 3.93), (12.08, 3.93)])
    text(9.13, 3.55, label("Gaussian noise", "高斯噪声"), 9)
    text(12.2, 3.55, label("Labeled 6D state", "有标签的 6D 状态"), 9)
    text(10.68, 3.13, r"$x_\tau=(1-\tau)x_0+\tau x_1^*$", 12)
    text(10.68, 2.59, r"$u^*=x_1^*-x_0$", 12)
    arrow([(12.6, 3.15), (13.2, 3.15)])
    box(13.2, 2.69, 3.95, 1.32,
        label("Masked velocity matching\nLcfm = mean ‖vθ − u*‖²\nOnly labeled pose nodes contribute",
              "带标签掩码的速度匹配\nLcfm = mean ‖vθ − u*‖²\n只有已标注的姿态节点参与"), "orange", 10.5)
    text(13.03, 2.03, r"$L=L_{\mathrm{CFM}}+0.1L_\beta+L_t+0.02L_{\mathrm{mesh}}$", 12)
    text(13.03, 1.64, label("Mesh loss uses a one-step endpoint estimate; full labels required.",
                           "网格损失使用单步终点估计；仅完整标签帧参与。"), 9)

    text(.5, .9, label("Solid arrows: data flow. Dashed loop: sampling feedback. Dashed box: optional graph. f: frame; τ: flow time.",
                       "实线：数据流。虚线回路：采样反馈。虚线框：可切换图分支。f：视频帧；τ：流时间。"),
         10, align="left")
    text(.5, .48, label("Research baseline; synthetic validation only. Stage, ground and contact remain in the separate engineering pipeline.",
                        "研究 baseline，当前仅完成合成验证；Stage、地面与接触仍属于独立工程流程。"),
         9.5, align="left")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = [a.get_text() for a in texts if not bounds.contains(*a.get_window_extent(renderer).p0)
               or not bounds.contains(*a.get_window_extent(renderer).p1)]
    if outside:
        raise ValueError(f"text outside canvas: {outside}")
    base = output / f"architecture_{language}"
    for ext in ("png", "svg", "pdf"):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=240, facecolor="white")
        if ext == "svg":
            # Preserve XML geometry/text while removing backend trailing spaces.
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
                            + "\n", encoding="utf-8")
    plt.close(fig)
    return {"language": language, "texts_checked": len(texts), "outside_canvas": outside,
            "mesh_source": "outputs/fresh_flow/prediction.npz, sequence0/frame0",
            "vertices": 6890, "faces": 13776, "svg_text": "editable"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((RECORD / "config_learning1000.json").read_text(encoding="utf-8"))
    for key, value in {"observation_joints": 22, "width": 64, "heads": 4,
                       "layers": 2, "use_graph": True, "use_history": True}.items():
        if config["model"][key] != value:
            raise ValueError(f"figure needs updating for model.{key}")
    for key, value in {"sample_steps": 8, "shape_weight": .1,
                       "translation_weight": 1, "mesh_weight": .02}.items():
        if config["training"][key] != value:
            raise ValueError(f"figure needs updating for training.{key}")
    args.output.mkdir(parents=True, exist_ok=True)
    checks = [draw(args.output, language) for language in ("en", "zh")]
    (args.output / "figure_validation.json").write_text(
        json.dumps({"checks": checks, "configuration": "config_learning1000.json",
                    "configuration_contract": "22 observations, width64, heads4, blocks2, graph/history enabled; 8 Euler steps",
                    "scope": "diagram export, not new model evaluation"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
