"""R7 single bounded WHAM auxiliary request with persistent attempt claim."""
import argparse
import json
import time
from pathlib import Path

from .audit_mesh_resources import mesh_resource_audit
from .fetch_auxiliary_resources import BASE, fetch_archive
from .run_original import ROOT


def claim_attempt(path, url):
    """Exclusive creation retains a consumed attempt even after a process crash."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump({"status": "attempt_claimed_before_GET", "url": url,
                   "R7_GET_invocations": 1, "prior_historical_HEAD_report": 1}, handle, indent=2)
        handle.write("\n")


def run(output):
    protocol = json.loads((BASE / "architecture_r7_protocol.json").read_text(encoding="utf-8"))
    assert protocol["id"] == "R7" and not protocol["training_allowed"] and not protocol["checkpoint_allowed"]
    before = mesh_resource_audit()
    claim = BASE / "architecture_r7_attempt.json"
    claim_attempt(claim, protocol["source"])
    folder = ROOT / "WHAM/dataset/body_models"
    targets = {name: (folder / Path(name).name, spec["kind"], spec.get("rows"))
               for name, spec in protocol["selection"].items()}
    start = time.monotonic()
    result = fetch_archive(protocol["source"], targets)
    after = mesh_resource_audit()
    audit = BASE / "architecture_full_path_resources_v4.json"
    if audit.exists():
        raise FileExistsError("retain previous audit; one GET has already been consumed")
    audit.write_text(json.dumps(after, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "R7_bounded_resource_request_audited", "protocol": "architecture_r7_protocol.json",
            "network_GET_invocations": 1, "prior_HEAD_handoff_report": 1, "attempt_claim": str(claim),
            "elapsed_seconds": time.monotonic() - start, "training_executed": False,
            "checkpoint_loaded": False, "archive_request": result,
            "before_missing": sum(not r["available"] for p in before["author_mesh_paths"] for r in p["requirements"]),
            "after_missing": sum(not r["available"] for p in after["author_mesh_paths"] for r in p["requirements"]),
            "after_audit": str(audit),
            "boundary": "Original auxiliary transport/schema only, not WHAM body/network completion. Missing or unvisited members retained; no WHAM joint regressor row semantics guessed. Source budget now exhausted, no retry."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (BASE / "architecture_full_path_resources_v4.json").exists():
        raise FileExistsError("retain previous records; no acquisition retry")
    result = run(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
