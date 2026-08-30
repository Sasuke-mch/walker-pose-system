from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mount:
    host: Path
    container: str
    mode: str = "rw"


def build_command(*, image: str, workdir: str, mounts: list[Mount], env: dict[str, str], inner: list[str], shm_size: str) -> list[str]:
    cmd = ["docker", "run", "--gpus", "all", "--rm", "--user", "root", f"--shm-size={shm_size}"]
    for key, value in env.items():
        cmd.extend(["-e", f"{key}={value}"])
    for m in mounts:
        cmd.extend(["-v", f"{m.host.resolve()}:{m.container}:{m.mode}"])
    cmd.extend(["-w", workdir, image])
    cmd.extend(inner)
    return cmd


def printable(cmd: list[str]) -> str:
    # list2cmdline is closer to Windows command-line quoting than shlex, but
    # shlex is more readable in logs. The actual execution does not use a shell.
    return subprocess.list2cmdline(cmd)


def _console_line(value: str, *, end: str = "\n") -> None:
    """Relay UTF-8 container logs without crashing legacy Windows consoles.

    The log file retains the original UTF-8 text.  A GBK stdout may not encode
    a Unicode symbol emitted by a third-party container; rendering an escaped
    fallback is preferable to aborting the entire model run.
    """
    try:
        print(value, end=end)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        fallback = value.encode(encoding, errors="backslashreplace").decode(encoding)
        print(fallback, end=end)


def run_streaming(cmd: list[str], log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = printable(cmd)
    _console_line(shown)
    if dry_run:
        log_path.write_text(shown + "\n", encoding="utf-8")
        return 0

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            _console_line(line, end="")
            log.write(line)
        return process.wait()
