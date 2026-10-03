"""Run public test suites from their required working directories."""
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    for directory in (root / "realtime_app", root / "sequence_pipeline"):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
             "--disable-warnings", "--tb=short"], cwd=directory,
        )
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
