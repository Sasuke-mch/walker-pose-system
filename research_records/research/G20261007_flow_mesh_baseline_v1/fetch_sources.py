"""Archive publicly available primary-paper HTML; download is not a reading claim."""
import argparse
import concurrent.futures
import json
from html.parser import HTMLParser
from pathlib import Path

import requests

PAPERS = [
    ("P01", "Flow Matching for Generative Modeling", "2210.02747", "flow"),
    ("P02", "Flow Matching on General Geometries", "2302.03660", "rotation"),
    ("P03", "FMPose3D: monocular 3D pose estimation via flow matching", "2602.05755v1", "flow_pose"),
    ("P04", "Probabilistic Modeling for Human Mesh Recovery", "2108.11944", "probabilistic_mesh"),
    ("P05", "HuManiFlow: Ancestor-Conditioned Normalising Flows on SO(3) Manifolds for Human Pose and Shape Distribution Estimation", "2305.06968", "rotation_mesh"),
    ("P06", "Score-Guided Diffusion for 3D Human Recovery", "2403.09623", "diffusion_mesh"),
    ("P07", "Humans in 4D: Reconstructing and Tracking Humans with Transformers", "2305.20091", "image_mesh"),
    ("P08", "Learning to Reconstruct 3D Human Pose and Shape via Model-fitting in the Loop", "1909.12828", "pseudo_supervision"),
    ("P09", "VIBE: Video Inference for Human Body Pose and Shape Estimation", "1912.05656", "video_mesh"),
    ("P10", "WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion", "2312.07531", "video_world"),
    ("P11", "World-Grounded Human Motion Recovery via Gravity-View Coordinates", "2409.06662", "video_world"),
    ("P12", "MotionBERT: A Unified Perspective on Learning Human Motion Representations", "2210.06551", "temporal"),
    ("P13", "3D human pose estimation in video with temporal convolutions and semi-supervised training", "1811.11742", "causal"),
    ("P14", "MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network", "2310.16288", "graph_temporal"),
    ("P15", "Multiple View Geometry Transformers for 3D Human Pose Estimation", "2311.10983", "multiview"),
    ("P16", "Learnable Triangulation of Human Pose", "1905.05754", "multiview"),
    ("P17", "End-to-End Human Pose and Mesh Reconstruction with Transformers", "2012.09760", "vertices"),
    ("P18", "Mesh Graphormer", "2104.00272", "graph_mesh"),
    ("P19", "Probabilistic Human Mesh Recovery in 3D Scenes from Egocentric Views", "2304.06024", "occlusion_mesh"),
    ("P20", "Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision", "2503.06089", "fisheye_mesh"),
]


class PaperText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "math":
            self.parts.append(" " + dict(attrs).get("alttext", "") + " ")
        if tag in {"script", "style", "math"}:
            self.skip += 1
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "tr", "figcaption", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "math"}:
            self.skip = max(0, self.skip - 1)
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "tr", "figcaption"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def lines(self):
        return [" ".join(line.split()) for line in "".join(self.parts).splitlines() if line.strip()]


def fetch(paper, root):
    pid, title, aid, category = paper
    result = {"id": pid, "title": title, "arxiv_id": aid, "category": category,
              "primary_url": f"https://arxiv.org/abs/{aid}", "reading_status": "not_read"}
    target = root / "sources" / f"{pid}.txt"
    if target.exists():
        result.update(source_status="archived", text_path=str(target.relative_to(root)),
                      lines=len(target.read_text(encoding="utf-8").splitlines()))
        return result
    attempts = []
    for base in ["https://arxiv.org/html/", "https://ar5iv.labs.arxiv.org/html/"]:
        url = base + aid
        try:
            response = requests.get(url, timeout=40)
            response.raise_for_status()
            parser = PaperText()
            parser.feed(response.text)
            lines = parser.lines()
            if sum(map(len, lines)) < 12000:
                raise ValueError("Not a full paper: text too short")
            html_path = target.with_suffix(".html")
            html_path.write_text(response.text, encoding="utf-8")
            target.write_text("\n".join(f"L{i + 1:04d}: {line}" for i, line in enumerate(lines)), encoding="utf-8")
            result.update(source_status="archived", fetched_url=url,
                          text_path=str(target.relative_to(root)), lines=len(lines))
            break
        except (requests.RequestException, ValueError, OSError) as error:
            attempts.append({"url": url, "error": str(error)})
    else:
        result["source_status"] = "failed"
    result["source_attempts"] = attempts
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    (root / "sources").mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda paper: fetch(paper, root), PAPERS))
    registry = root / "papers.json"
    previous = {p["id"]: p for p in json.loads(registry.read_text(encoding="utf-8"))} if registry.exists() else {}
    for record in records:
        old = previous.get(record["id"], {})
        # Fetching again must not erase an already completed reading audit.
        for key in ("reading_status", "read_anchor_lines", "analysis", "reading_scope",
                    "fetched_url", "lines", "source_attempts"):
            if key in old and (key not in record or key == "reading_status"):
                record[key] = old[key]
    (root / "papers.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: p.get(k) for k in ["id", "source_status", "lines", "source_attempts"]} for p in records], ensure_ascii=False))


if __name__ == "__main__":
    main()
