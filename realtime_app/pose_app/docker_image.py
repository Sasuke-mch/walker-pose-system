from __future__ import annotations

from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class DockerImageResolution:
    configured: str
    runtime_reference: str
    image_id: str
    matched_by: str


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def resolve_docker_image(configured: str) -> DockerImageResolution:
    """Resolve an image tag to a runnable image reference.

    Some Docker Desktop/CLI combinations can list a tag but fail to resolve the
    same tag through ``docker image inspect`` or ``docker run``.  In that case,
    locate the exact repository:tag in ``docker image ls`` and use its immutable
    image ID for subsequent commands.
    """
    configured = configured.strip()
    if not configured:
        raise RuntimeError("Docker镜像配置为空。")

    direct = _run(["docker", "image", "inspect", configured, "--format", "{{.Id}}"])
    if direct.returncode == 0 and direct.stdout.strip():
        image_id = direct.stdout.strip().splitlines()[-1].strip()
        return DockerImageResolution(configured, configured, image_id, "inspect")

    listed = _run(
        [
            "docker",
            "image",
            "ls",
            "--no-trunc",
            "--format",
            "{{.Repository}}:{{.Tag}}|{{.ID}}",
        ]
    )
    if listed.returncode != 0:
        raise RuntimeError(
            "无法读取Docker镜像列表。\n" + (listed.stdout or direct.stdout or "")
        )

    matches: list[str] = []
    for raw_line in listed.stdout.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        tag, image_id = line.split("|", 1)
        if tag.strip() == configured and image_id.strip():
            matches.append(image_id.strip())

    if not matches:
        details = (direct.stdout or "").strip()
        suffix = f"\nDocker返回：{details}" if details else ""
        raise RuntimeError(f"找不到Docker镜像：{configured}{suffix}")

    image_id = matches[0]
    by_id = _run(["docker", "image", "inspect", image_id, "--format", "{{.Id}}"])
    if by_id.returncode != 0:
        raise RuntimeError(
            f"镜像标签已列出但镜像ID无法读取：{configured} -> {image_id}\n"
            + (by_id.stdout or "")
        )
    canonical_id = by_id.stdout.strip().splitlines()[-1].strip() or image_id
    return DockerImageResolution(configured, canonical_id, canonical_id, "image-list")
