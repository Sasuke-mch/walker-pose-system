"""R11 inspect official ZIP metadata with one bounded suffix request, no models."""
import argparse
import io
import json
import re
import time
import traceback
import zipfile
from pathlib import Path

import requests

from .audit_mesh_resources import mesh_resource_audit
from .fetch_auxiliary_resources import BASE, LimitedReader

MAX_TAIL = 65536


def validate_range(status, headers):
    """Reject whole archives before reading their bodies."""
    if status != 206:
        raise ValueError(f"suffix range required206, got{status}; no fullzip fallback")
    if headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("encoded archive transport rejected")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("Content-Range", ""))
    if match is None:
        raise ValueError("numeric Content-Range required")
    start, end, total = map(int, match.groups())
    if not 0 <= start <= end == total - 1 or end - start + 1 > MAX_TAIL:
        raise ValueError("invalid or excessive suffix range")
    return start, end, total


def run(output):
    protocol = json.loads((BASE / "architecture_r11_protocol.json").read_text(encoding="utf-8"))
    before = mesh_resource_audit()
    claim = BASE / "architecture_r11_attempt.json"
    with claim.open("x", encoding="utf-8") as handle:
        json.dump({"status": "claimed_before_GET", "url": protocol["source"],
                   "GET_invocations": 1, "prior_same_source_requests": 0}, handle, indent=2)
        handle.write("\n")
    reader = None
    start = time.monotonic()
    result = {"status": "not_acquired", "inventory": [], "installed": [], "body_bytes": 0}
    try:
        with requests.get(protocol["source"], stream=True, timeout=20,
                          headers={"Range": f"bytes=-{MAX_TAIL}", "Accept-Encoding": "identity"}) as response:
            result.update(http_status=response.status_code, final_url=response.url,
                          content_range=response.headers.get("Content-Range"))
            offset, end, total = validate_range(response.status_code, response.headers)
            response.raw.decode_content = False
            reader = LimitedReader(response.raw, maximum=MAX_TAIL + 1)
            reader.begin = start
            data = reader.read(end - offset + 1)
            if len(data) != end - offset + 1:
                raise ValueError("incomplete suffix response")
            if time.monotonic() - start > 20:
                raise TimeoutError("request elapsed deadline exceeded after read")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                result["inventory"] = [{"name": entry.filename, "uncompressed_bytes": entry.file_size,
                                        "compressed_bytes": entry.compress_size}
                                       for entry in archive.infolist()]
            result.update(status="bounded_ZIP_index_only", archive_total_bytes=total)
    except Exception as error:  # noqa: BLE001 -- retain exact transport/index failure
        result.update(status="bounded_index_request_failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        result["body_bytes"] = reader.consumed if reader else 0
    after = mesh_resource_audit()
    audit = BASE / "architecture_full_path_resources_v5.json"
    if audit.exists():
        raise FileExistsError("retain previous resource audit")
    audit.write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "R11_bounded_VIBE_index_audited", "protocol": "architecture_r11_protocol.json",
            "training_executed": False, "checkpoint_loaded": False, "whole_goal_complete": False,
            "GET_invocations": 1, "same_source_budget_remaining": 1, "attempt_claim": str(claim),
            "elapsed_seconds": time.monotonic() - start, "archive_request": result,
            "before_missing": sum(not q["available"] for p in before["author_mesh_paths"] for q in p["requirements"]),
            "after_missing": sum(not q["available"] for p in after["author_mesh_paths"] for q in p["requirements"]),
            "after_audit": str(audit),
            "selected_members_status": {name: "not_acquired" for name in protocol["selected_future_members"]},
            "boundary": "ZIP index only, not auxiliary acquisition or fullVIBE. Bodybytes do not count HTTP/TLS redirects. Existingclaim blocks retry; any second source request requires distinct preregistered stage."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (BASE / "architecture_full_path_resources_v5.json").exists():
        raise FileExistsError("retain existing evidence and source attempt")
    report = run(args.output)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
