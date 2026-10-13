"""Render the current JointFlow architecture as an editable report figure."""
import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research_records/research/G20261007_flow_mesh_baseline_v1/figures/jointflow_architecture_20261013"
MESH = ROOT / "research_records/research/G20261007_flow_mesh_baseline_v1/architecture_figure_v4_cn/smplh_neutral_pose.png"


def draw(prefix=DEFAULT):
    """Schematic only: no model execution or invented experimental results."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
    })
    fig, ax = plt.subplots(figsize=(22, 13.7))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 22), ylim=(0, 13.7), aspect="equal")
    ax.axis("off")
    ink, muted, line = "#1E3443", "#5B6D79", "#CDD8DD"
    teal, pale = "#147F88", "#EFF8F8"
    amber, warm = "#A66B28", "#FBF7F0"
    violet = "#77639B"

    def text(x, y, value, size=12, color=ink, weight="normal", ha="left", **kwargs):
        return ax.text(x, y, value, fontsize=size, color=color, weight=weight,
                       ha=ha, va="center", linespacing=1.55, **kwargs)

    def box(x, y, w, h, face="white", edge=line, lw=1, radius=.11, dashed=False):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle=f"round,pad=0,rounding_size={radius}",
                               fc=face, ec=edge, lw=lw,
                               linestyle=(0, (5, 4)) if dashed else "solid")
        ax.add_patch(patch)
        return patch

    def arrow(points, color=teal, dashed=False, lw=1.6):
        style = (0, (4, 3)) if dashed else "solid"
        for p, q in pairwise(points[:-1]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=style,
                    solid_capstyle="round")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=12, color=color, lw=lw,
                                    linestyle=style, shrinkA=0, shrinkB=0))

    def pill(x, y, w, label, color=teal, face=pale, size=11):
        box(x, y, w, .34, face, edge=face, radius=.12)
        text(x + w/2, y+.17, label, size, color, ha="center")

    def card(x, w, number, title, subtitle):
        box(x, 8.35, w, 3.12)
        ax.plot([x+.2, x+.62], [11.16, 11.16], color=teal, lw=3)
        text(x+.2, 10.84, title, 16, weight="bold")
        text(x+w-.2, 11.13, number, 11, muted, ha="right")
        text(x+.2, 8.64, subtitle, 10.8, muted)

    def skeleton(x, y, scale=1, color=teal):
        nodes = [(0, .9), (-.16, .78), (.16, .78), (-.3, .48), (.3, .48),
                 (-.34, .17), (.34, .17), (-.13, .28), (.13, .28),
                 (-.15, -.1), (.15, -.1), (-.18, -.45), (.18, -.45),
                 (-.06, .96), (.06, .96), (-.13, .92), (.13, .92)]
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
                 (1, 7), (2, 8), (7, 8), (7, 9), (8, 10), (9, 11), (10, 12),
                 (0, 13), (0, 14), (13, 15), (14, 16)]
        for i, j in edges:
            ax.plot([x+nodes[i][0]*scale, x+nodes[j][0]*scale],
                    [y+nodes[i][1]*scale, y+nodes[j][1]*scale], color=color, lw=1.1)
        for u, v in nodes:
            ax.add_patch(Circle((x+u*scale, y+v*scale), .025*scale, fc=color, ec="none"))

    # Title and the main left-to-right generation path.
    text(.65, 13.12, "MODEL ARCHITECTURE  /  JOINTFLOW", 11, teal, "bold")
    text(.65, 12.64, "从双鱼眼观测到完整人体候选", 28, weight="bold")
    text(.67, 12.12, "因果观测条件  ·  姿态 / 位置 / 体型联合生成  ·  按需使用几何信息", 13.5, muted)
    text(21.35, 13.08, "2026.10.13", 11, muted, ha="right")
    pill(18.39, 12.55, 2.96, "SMPLH52 · 当前实现", size=11.5)

    card(.65, 3.48, "01", "双鱼眼观测", "二维支持与严格几何支持分开")
    for x, label in ((.9, "LEFT"), (2.39, "RIGHT")):
        box(x, 9.22, 1.24, 1.03, face=pale, edge=pale)
        skeleton(x+.62, 9.68, .61)
        text(x+.62, 9.34, label, 9, teal, ha="center")
    text(.87, 8.99, "原像素 · 分数 · 射线 · 掩码 · 时间", 10.6)
    pill(.88, 10.30, 2.98, "2 个视图 × COCO17", size=11)

    card(4.63, 3.64, "02", "因果条件编码", "当前观测 + 已发生的历史")
    text(4.85, 10.39, "34 槽 × 15 维特征", 12.4, teal, "bold")
    text(4.85, 9.98, "关节 / 视图身份 + MLP", 11.8)
    arrow([(6.44, 9.79), (6.44, 9.56)], lw=1.1)
    box(4.9, 9.01, 3.09, .48, face=pale, edge=pale)
    text(6.445, 9.25, "准入池化 → GRU 128", 12, teal, ha="center", weight="bold")
    # Past-state input is the only recurrence in this architecture view.
    arrow([(5.08, 11.48), (5.08, 11.73), (7.9, 11.73), (7.9, 11.48)], muted, True, 1)
    text(6.49, 11.91, "历史 h、age、seen；物理 Δt", 9.6, muted, ha="center")

    card(8.77, 4.12, "03", "联合 Flow 生成", "每个候选同时包含完整 R、T、β")
    pill(8.99, 10.29, 1.35, "32 份噪声", size=10.5)
    arrow([(10.45, 10.46), (10.86, 10.46)], lw=1.1)
    pill(10.98, 10.29, 1.67, "8 步 Euler", size=10.5)
    box(9.02, 9.48, 3.6, .57, face=teal, edge=teal)
    text(10.82, 9.765, "共享速度网络  vθ", 14, "white", "bold", "center")
    text(10.83, 9.12, "325D 状态 → 32 个完整候选", 12.2, teal, "bold", "center")

    card(13.39, 3.76, "04", "完整候选决策", "保留全部候选及原失败原因")
    for i, alpha in enumerate((.35, .58, 1.0)):
        xx = 13.66+i*.12
        yy = 9.57-i*.095
        p = box(xx, yy, 1.17, .61, face="white", edge=teal)
        p.set_alpha(alpha)
    text(14.48, 9.68, "R / T / β", 11, teal, ha="center")
    arrow([(15.08, 9.69), (15.46, 9.69)], lw=1.1)
    box(15.56, 9.36, 1.3, .65, face=pale, edge=teal)
    text(16.21, 9.685, "同一个 k*", 11.2, teal, "bold", "center")
    text(13.64, 10.34, "完整状态风险 / medoid", 12.2)
    text(13.65, 9.01, "可选：加入当前观测几何成本", 10.7, amber)

    card(17.65, 3.7, "05", "SMPL-H 人体", "选中 R、T、β → LBS 人体解码")
    mesh = plt.imread(MESH)
    ax.imshow(mesh, extent=(18.60, 19.98, 9.00, 10.66), zorder=2)
    text(17.9, 9.45, "6890\n顶点", 11.5, teal, "bold")
    text(20.2, 9.45, "17\n表面点", 11.2, teal)
    text(19.5, 8.99, "网格示意", 8.5, muted, ha="center")
    for left, right in ((4.13, 4.63), (8.27, 8.77), (12.89, 13.39), (17.15, 17.65)):
        arrow([(left+.055, 9.71), (right-.06, 9.71)])
    text(8.52, 9.99, r"$c_t$", 12, teal, ha="center")

    # Detailed velocity field: same graph at every integration step.
    box(.65, 1.64, 12.24, 5.83, face=pale, edge="#D3E7E6")
    text(.94, 7.08, "A", 13, teal, "bold")
    text(1.34, 7.08, "速度网络展开", 17, weight="bold")
    text(12.58, 7.07, "每一步共享同一套参数", 11.2, muted, ha="right")
    text(1.03, 6.49, r"联合状态 $z_s$", 12, teal, "bold")
    text(1.03, 6.16, "标准化后的 325 维", 10.8, muted)
    for yy, title, detail, color in (
        (5.25, "姿态", "52 × 6D", teal),
        (4.42, "平移", "1 × 3D", amber),
        (3.59, "体型", "1 × 10D", violet),
    ):
        box(1.0, yy, 2.12, .64, edge=color)
        ax.add_patch(Rectangle((1.0, yy+.10), .045, .44, color=color))
        text(1.19, yy+.32, title, 12, color, "bold")
        text(2.95, yy+.32, detail, 11.4, ha="right")
        arrow([(3.16, yy+.32), (3.83, yy+.32)], color, lw=1.15)
    text(3.48, 6.07, "线性投影", 10.2, muted, ha="center")
    for i in range(9):
        box(3.93+i*.072, 3.89+i*.12, .6, 1.52, face="white", edge=teal, radius=.025, lw=.65)
    text(4.55, 3.61, "52 + 1 + 1", 11.2, teal, "bold", "center")
    text(4.55, 3.26, "54 tokens × 128", 11.3, ha="center")
    arrow([(5.24, 4.89), (5.84, 4.89)], lw=1.3)
    ax.add_patch(Circle((6.05, 4.89), .19, fc="white", ec=teal, lw=1.25))
    text(6.05, 4.90, "+", 18, teal, ha="center")
    pill(5.48, 6.20, 6.87, r"context $c_t$  +  token ID  +  共同流时间 $s$", size=11.4)
    arrow([(6.05, 6.15), (6.05, 5.13)], lw=1.1)
    arrow([(6.28, 4.89), (6.70, 4.89)], lw=1.2)
    for i in reversed(range(4)):
        box(6.85+i*.085, 3.95+i*.09, 2.17, 1.64,
            face=teal if i == 0 else "white", edge=teal, radius=.09)
    text(7.935, 5.10, "Transformer", 14, "white", "bold", "center")
    text(7.935, 4.70, "4 层 · 4 头", 13, "white", ha="center")
    text(7.935, 4.32, "width 128  /  FFN 256", 9.6, "white", ha="center")
    ax.plot([9.36, 9.69], [4.89, 4.89], color=teal, lw=1.3)
    ax.plot([9.69, 9.69], [3.955, 5.555], color=teal, lw=1.1)
    for yy, val, color in ((5.27, "姿态头  52 × 6", teal),
                           (4.47, "平移头  1 × 3", amber),
                           (3.67, "体型头  1 × 10", violet)):
        box(10.04, yy, 2.27, .57, edge=color)
        text(11.175, yy+.285, val, 11.8, color, ha="center")
        arrow([(9.69, yy+.285), (9.98, yy+.285)], color, lw=1.1)
    text(11.17, 3.32, r"拼接 → 325D 速度 $v_\theta$", 10.8, ha="center")
    ax.plot([.99, 12.54], [2.9, 2.9], color="#CBDDDD", lw=.8)
    text(1.04, 2.44, "共同更新", 11.5, teal, "bold")
    text(3.0, 2.43, r"$z_{s+\Delta s}=z_s+\Delta s\,v_\theta(z_s,s,c_t\,[,\Lambda_t])$", 17)
    text(12.46, 2.42, "Δs = 1/8", 12, teal, ha="right")
    text(1.04, 1.98, "姿态、位置与体型在同一流时间下交互更新；最终反标准化并将 6D 旋转转换为矩阵。", 10.6, muted)

    # Optional geometric paths are distinct from the unconditional data path.
    box(13.39, 1.64, 7.96, 5.83, face=warm, edge="#DCCBAF", dashed=True)
    text(13.69, 7.08, "B", 13, amber, "bold")
    text(14.09, 7.08, "按需几何分支", 17, weight="bold")
    text(21.05, 7.08, "启用时才计算", 11, amber, ha="right")
    text(13.73, 6.46, "候选几何评分", 12, amber, "bold")
    text(13.73, 6.11, "当前候选 → 表面 COCO17 → 原鱼眼投影", 11.4)
    box(13.74, 5.39, 4.12, .47, edge="#DCCBAF")
    text(15.8, 5.625, "准入像素 + rig + 像素精度 W", 11, ha="center")
    arrow([(17.98, 5.625), (18.48, 5.625)], amber, True, 1.2)
    box(18.61, 5.28, 2.31, .69, edge=amber)
    text(19.765, 5.625, "Huber 成本 Eobs", 11.7, amber, "bold", "center")
    # Two distinct outlets: Eobs to decision, Lambda to generator.
    arrow([(20.98, 5.72), (21.52, 5.72), (21.52, 7.87), (15.27, 7.87), (15.27, 8.30)], amber, True, 1.2)
    ax.plot([13.72, 21.03], [5.02, 5.02], color="#E1D5C3", lw=.8)
    text(13.73, 4.65, "信息条件研究分支  G0 → J → G1", 12.3, amber, "bold")
    for x, w, label in ((13.75, 1.78, "冻结 G0\n完整候选锚点"),
                         (16.09, 1.98, "局部有限差分\nJ : 68 × 169*"),
                         (18.62, 2.28, "$\\Lambda = J^{\\mathsf{T}} W J$\n169 × 169")):
        box(x, 3.34, w, .88, edge="#DCCBAF")
        text(x+w/2, 3.78, label, 11.4, ha="center", color=amber)
    arrow([(15.57, 3.78), (16.02, 3.78)], amber, True, 1.1)
    arrow([(18.11, 3.78), (18.55, 3.78)], amber, True, 1.1)
    text(13.74, 2.96, "* 满 34 个准入点时；缺失观测对应行不参与。", 10.2, muted)
    text(13.74, 2.60, "G1 使用 Λ 条件；固定编码每次采样仅计算一次。", 11, amber)
    text(13.74, 2.24, "G0 / G1 各有自己的观测编码器与因果状态。", 10.8, muted)
    text(13.74, 1.90, "Λ 由同帧像素、rig 和 W 计算，非后验协方差。", 10.2, muted)
    arrow([(20.95, 3.70), (21.74, 3.70), (21.74, 7.64), (11.66, 7.64), (11.66, 8.30)], amber, True, 1.0)

    # Footer: explain line encodings and the scope without an audit-style checklist.
    arrow([(.71, 1.11), (1.18, 1.11)], lw=1.3)
    text(1.31, 1.11, "单模型生成路径", 10.4, muted)
    arrow([(4.07, 1.11), (4.54, 1.11)], amber, True, 1.2)
    text(4.69, 1.11, "可选几何 / 信息条件", 10.4, muted)
    text(8.24, 1.11, "SO(3) 流为独立表示对照；本图展开默认 6D 联合流。", 10.4, muted)
    ax.plot([.66, 21.35], [.76, .76], color=line, lw=.8)
    text(.68, .40, "实现结构示意 · 当前模型未训练，尚无真实精度结论。", 10.4, muted)
    text(21.32, .40, "当前无准入观测：不发布人体；保留缺失与失败状态。", 10.4, muted, ha="right")

    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        path = prefix.with_suffix("." + ext)
        fig.savefig(path, dpi=200, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT)
    draw(parser.parse_args().output_prefix)
