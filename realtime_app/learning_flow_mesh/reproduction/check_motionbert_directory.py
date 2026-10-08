"""R15 bounded primary directory metadata; no binary assets fetched."""
import argparse
import json
import time
import traceback
from pathlib import Path

import requests

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")


def validate_headers(status, headers, cap):
    if status != 200:
        raise ValueError(f"directory HTTP200 required, received{status}")
    if headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "text/html":
        raise ValueError("only HTML directory metadata permitted")
    if headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("encoded body rejected")
    size = headers.get("Content-Length")
    if size is not None and (not size.isdecimal() or int(size) > cap):
        raise ValueError("invalid or excessive declared body size")


def run():
    protocol = json.loads((BASE / "architecture_r15_protocol.json").read_text(encoding="utf-8"))
    with (BASE / "architecture_r15_attempt.json").open("x", encoding="utf-8") as handle:
        json.dump({"source": protocol["source"], "direct_GET_invocations": 1,
                   "prior_documentation_attempts": 1, "status": "claimed_before_GET"}, handle, indent=2)
    result = {"status": "not_acquired", "body_bytes_read": 0, "installed": [], "filename_mentions": []}
    begin = time.monotonic()
    cap = protocol["max_html_body_bytes"]
    session = requests.Session()
    session.max_redirects = protocol["redirect_limit"]
    try:
        with session.get(protocol["source"], stream=True, timeout=protocol["deadline_seconds"],
                         headers={"Accept": "text/html", "Accept-Encoding": "identity"}) as response:
            result.update(http_status=response.status_code, final_url=response.url,
                          redirects=[{"status": r.status_code, "url": r.url} for r in response.history],
                          headers={k: response.headers.get(k) for k in ("Content-Type", "Content-Length", "Content-Encoding")})
            validate_headers(response.status_code, response.headers, cap)
            if time.monotonic() - begin > protocol["deadline_seconds"]:
                raise TimeoutError("headers exceeded stage deadline")
            response.raw.decode_content = False
            data = bytearray()
            while len(data) <= cap:
                chunk = response.raw.read(min(16384, cap + 1 - len(data)))
                data.extend(chunk)
                result["body_bytes_read"] = len(data)
                if time.monotonic() - begin > protocol["deadline_seconds"]:
                    raise TimeoutError("body exceeded stage deadline")
                if not chunk:
                    break
            if len(data) > cap:
                raise ValueError("directory body cap exceeded")
            target = BASE / "architecture_r15_directory.html"
            with target.open("xb") as handle:
                handle.write(data)
            text = data.decode("utf-8", errors="replace")
            result.update(status="bounded_HTML_directory_received", local_body=str(target),
                          filename_mentions=[n for n in protocol["wanted"] if n in text])
    except Exception as error:  # noqa: BLE001 -- retain source access failure
        result.update(status="directory_access_failed", error=f"{type(error).__name__}: {error}",
                      traceback=traceback.format_exc())
    finally:
        session.close()
    return {"status": "R15_official_MotionBERT_directory_audited", "protocol": "architecture_r15_protocol.json",
            "directory": result, "elapsed_seconds": time.monotonic() - begin,
            "direct_GET_invocations": 1, "prior_documentation_attempts": 1,
            "conservative_same_source_budget_remaining": 0, "resource_files_acquired": 0,
            "default_gaps_before": 14, "default_gaps_after": 14,
            "training_executed": False, "checkpoint_loaded": False, "whole_goal_complete": False,
            "boundary": "Official directory reachability and text only, never proof of selected auxiliary bytes or complete method."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain prior report")
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
