from __future__ import annotations

from pathlib import Path
from .utils import natural_key, write_json

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scan_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(str(input_dir))
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
    return sorted(images, key=natural_key)


def write_manifest(input_dir: Path, images: list[Path], output_path: Path) -> None:
    write_json(output_path, {
        "schema_version": "1.0",
        "input_dir": str(input_dir.resolve()),
        "num_frames": len(images),
        "frames": [
            {
                "frame_index": i,
                "frame_id": f"frame_{i:06d}",
                "file_name": p.name,
                "absolute_path": str(p.resolve()),
                "timestamp": None,
                "camera_id": None,
            }
            for i, p in enumerate(images)
        ],
    })
