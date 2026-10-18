"""Run the realtime_app test suite.

Each test module runs in its own subprocess so that module-level imports
(e.g. torch in the R4 alternative-body ray factor tests) can never pollute
the ``sys.modules`` of unrelated modules -- this is what keeps the
learned-stereo benchmark's "no torch in the main process" contract intact.

Aggregates the per-module counts into one final summary and returns a
non-zero exit code if any module failed.
"""

from __future__ import annotations

import compileall
import re
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    if not compileall.compile_dir(root, quiet=1):
        return 1
    test_dir = root / "tests"
    modules = sorted(p.stem for p in test_dir.glob("test_*.py"))
    failed_modules: list[str] = []
    total_tests = 0
    total_skips = 0
    for stem in modules:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", f"tests.{stem}", "-v"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        output = proc.stdout + proc.stderr
        match = re.search(r"Ran (\d+) tests?", output)
        ran_count = int(match.group(1)) if match else None
        if ran_count == 0:
            # A few legacy modules use pytest-style top-level functions.  They
            # must be executed, not silently accepted as an empty unittest module.
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_dir / f"{stem}.py"), "-q"],
                cwd=str(root), capture_output=True, text=True,
            )
            output = proc.stdout + proc.stderr
            pytest_passed = re.search(r"(\d+) passed", output)
            ran_count = int(pytest_passed.group(1)) if pytest_passed else None
        sys.stdout.write(output)
        sys.stdout.flush()
        if proc.returncode != 0 or ran_count in (None, 0):
            failed_modules.append(stem)
            continue
        total_tests += ran_count
        skip_match = re.search(r"skipped=(\d+)", output)
        if skip_match:
            total_skips += int(skip_match.group(1))

    print("=" * 70)
    print(
        f"aggregate: {total_tests} tests, {len(failed_modules)} failed modules, "
        f"{total_skips} skipped across {len(modules)} modules"
    )
    if failed_modules:
        print(f"failed modules: {', '.join(failed_modules)}")
    return 1 if failed_modules else 0


if __name__ == "__main__":
    raise SystemExit(main())
