from __future__ import annotations

import shlex
import subprocess
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


def run_streaming(cmd: list[str], log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = printable(cmd)
    print(shown)
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
            print(line, end="")
            log.write(line)
        return process.wait()
