"""Archive author source/config files locally without installing or training models.

External files are not redistributed. Each snapshot keeps its originating URL,
download time, selected relative filenames and byte sizes. Model/data binaries
are excluded. Never overwrite a previously acquired snapshot.
"""
import argparse
import concurrent.futures
import io
import json
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DEFAULT_RECORD = Path("research_records/research/G20261007_flow_mesh_baseline_v1")
EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".cfg", ".sh"}
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_FILE = 2 * 1024 * 1024


def selected_path(name):
    """Validate the archive member before creating any filesystem path."""
    parts = PurePosixPath(name).parts
    if len(parts) < 2 or any(x in {"..", "."} or ":" in x or "\\" in x for x in parts):
        raise ValueError(f"unsafe archive member: {name}")
    relative = PurePosixPath(*parts[1:])
    if any(p in {".git", "__pycache__", ".github", "node_modules"} for p in relative.parts):
        return None
    if relative.suffix.lower() in EXTENSIONS or relative.name.lower().startswith(("license", "copying")):
        return relative
    return None


def acquire(paper, root):
    destination = root / paper["name"]
    url = f"https://codeload.github.com/{paper['repo']}/zip/refs/heads/{paper['branch']}"
    record = {"id": paper["id"], "name": paper["name"], "repo_url": f"https://github.com/{paper['repo']}",
              "archive_url": url, "branch": paper["branch"], "source_path": destination.as_posix(),
              "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "status": "pending"}
    try:
        if destination.exists():
            provenance = destination / "SOURCE_SNAPSHOT.json"
            if provenance.is_file():
                saved = json.loads(provenance.read_text(encoding="utf-8"))
                if saved["archive_url"] != url:
                    raise ValueError("snapshot provenance mismatch")
                for entry in saved["files"]:
                    cached = destination / Path(*PurePosixPath(entry["path"]).parts)
                    if not cached.resolve().is_relative_to(destination.resolve()):
                        raise ValueError("cached provenance path escaped root")
                    if not cached.is_file() or cached.stat().st_size != entry["bytes"]:
                        raise ValueError("cached source files incomplete or modified; preserve and inspect")
                return saved
            raise FileExistsError(f"incomplete snapshot exists; preserve and inspect {destination}")
        request = urllib.request.Request(url, headers={"User-Agent": "walker-architecture-research"})
        with urllib.request.urlopen(request, timeout=40) as response:
            archive_bytes = response.read(MAX_ARCHIVE + 1)
        if len(archive_bytes) > MAX_ARCHIVE:
            raise ValueError("archive exceeds source-download budget")
        selected = []
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative = selected_path(member.filename)
                if relative is not None and member.file_size <= MAX_FILE:
                    selected.append((relative, archive.read(member)))
        if not selected:
            raise ValueError("no source/config/license members found")
        destination.mkdir(parents=True, exist_ok=False)
        root_resolved = destination.resolve()
        for relative, data in selected:
            target = destination / Path(*relative.parts)
            if not target.resolve().is_relative_to(root_resolved):
                raise ValueError("resolved archive path escaped snapshot root")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        files = [{"path": str(relative), "bytes": len(data)} for relative, data in selected]
        record.update(status="source_archived", files=files, source_file_count=len(files),
                      selected_bytes=sum(x["bytes"] for x in files),
                      license_paths=[x["path"] for x in files if PurePosixPath(x["path"]).name.lower().startswith(("license", "copying"))],
                      scope="Source/config snapshot only; no model/data binaries extracted, no import or forward yet")
        (destination / "SOURCE_SNAPSHOT.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        record.update(status="source_acquisition_failed", error=f"{type(error).__name__}: {error}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--paper", action="append", help="Selected paper ID; default all ten")
    parser.add_argument("--output", type=Path, required=True, help="New acquisition report; never overwrite")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = json.loads((args.record / "architecture_research.json").read_text(encoding="utf-8"))
    papers = [p for p in state["papers"] if args.paper is None or p["id"] in args.paper]
    if not papers or (args.paper and set(args.paper) != {p["id"] for p in papers}):
        raise ValueError("unknown or empty paper selection")
    root = Path(state["source_root"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda p: acquire(p, root), papers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"training_executed": False, "papers": records}, indent=2), encoding="utf-8")
    for record in records:
        print(record["id"], record["name"], record["status"], record.get("source_file_count", 0), flush=True)
    if any(record["status"] != "source_archived" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
