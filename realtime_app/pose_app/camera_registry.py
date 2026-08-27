from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol


CameraBackend = Literal["auto", "dshow", "msmf"]


class CameraRegistryError(RuntimeError):
    """Raised when the physical cameras cannot be resolved unambiguously."""


class EnumeratedCamera(Protocol):
    index: int
    name: str
    path: str


@dataclass(frozen=True)
class RegisteredCamera:
    key: str
    role: Literal["left", "right"]
    calibration: str
    friendly_name: str
    instance_id: str
    usb_location: str | None
    acpi_port: str | None


@dataclass(frozen=True)
class ResolvedCamera:
    key: str
    role: Literal["left", "right"]
    index: int
    backend: Literal["dshow", "msmf"]
    expected_instance_id: str
    device_name: str
    device_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedStereoCameras:
    registry_path: Path
    backend: Literal["dshow", "msmf"]
    left: ResolvedCamera
    right: ResolvedCamera

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_path": str(self.registry_path),
            "backend": self.backend,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


def backend_candidates(backend: CameraBackend) -> tuple[Literal["dshow", "msmf"], ...]:
    """Return the common-backend fallback order used by the live camera path.

    MSMF is intentionally tried first.  With the physical cam0/cam1 mapping
    resolved from the registry, this project measures roughly 32 FPS per
    camera and low host-side pair skew on MSMF; DirectShow is the fallback.
    """

    if backend == "auto":
        return ("msmf", "dshow")
    if backend in {"dshow", "msmf"}:
        return (backend,)
    raise ValueError("backend must be one of: auto, dshow, msmf.")


def _identifier_token(value: str) -> str:
    """Normalize PnP IDs and DirectShow paths for strict containment matching."""

    return "".join(character for character in value.casefold() if character.isalnum())


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraRegistryError(f"{field} must be a non-empty string.")
    return value.strip()


def load_camera_registry(path: str | Path) -> tuple[RegisteredCamera, RegisteredCamera]:
    """Load and validate the two calibrated physical camera identities.

    ``cam0`` is the calibrated left camera and ``cam1`` is the calibrated
    right camera.  Older registry files with ``role: null`` remain accepted;
    the key provides the unambiguous legacy convention.
    """

    registry_path = Path(path).resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"Camera registry does not exist: {registry_path}")

    raw = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise CameraRegistryError("Camera registry root must be a JSON object.")

    cameras: list[RegisteredCamera] = []
    for key, expected_role in (("cam0", "left"), ("cam1", "right")):
        entry = raw.get(key)
        if not isinstance(entry, dict):
            raise CameraRegistryError(f"Camera registry must contain object {key!r}.")

        role = entry.get("role") or expected_role
        if role != expected_role:
            raise CameraRegistryError(
                f"{key}.role must be {expected_role!r}; got {role!r}. "
                "The calibration convention is cam0=left and cam1=right."
            )

        device = entry.get("device")
        if not isinstance(device, dict):
            raise CameraRegistryError(f"{key}.device must be a JSON object.")

        cameras.append(
            RegisteredCamera(
                key=key,
                role=expected_role,
                calibration=_require_string(entry.get("calibration"), f"{key}.calibration"),
                friendly_name=_require_string(
                    device.get("friendly_name"), f"{key}.device.friendly_name"
                ),
                instance_id=_require_string(
                    device.get("instance_id"), f"{key}.device.instance_id"
                ),
                usb_location=(
                    str(device["usb_location"]).strip()
                    if device.get("usb_location") is not None
                    else None
                ),
                acpi_port=(
                    str(device["acpi_port"]).strip()
                    if device.get("acpi_port") is not None
                    else None
                ),
            )
        )

    return cameras[0], cameras[1]


def _enumerate_windows_cameras(
    backend: Literal["dshow", "msmf"],
) -> list[EnumeratedCamera]:
    try:
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError as exc:
        raise CameraRegistryError(
            "Camera-identity resolution requires cv2-enumerate-cameras. "
            "Install it with: python -m pip install cv2-enumerate-cameras==1.3.3"
        ) from exc

    backend_api = cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF
    try:
        return list(enumerate_cameras(backend_api))
    except Exception as exc:
        raise CameraRegistryError(
            f"Could not enumerate Windows cameras through backend={backend!r}."
        ) from exc


def _describe_devices(devices: Iterable[EnumeratedCamera]) -> str:
    values = []
    for device in devices:
        values.append(
            f"index={getattr(device, 'index', None)}, "
            f"name={getattr(device, 'name', None)!r}, "
            f"path={getattr(device, 'path', None)!r}"
        )
    return "[" + "; ".join(values) + "]"


def _resolve_one(
    registered: RegisteredCamera,
    devices: list[EnumeratedCamera],
    backend: Literal["dshow", "msmf"],
) -> ResolvedCamera:
    expected_token = _identifier_token(registered.instance_id)
    matches = [
        device
        for device in devices
        if expected_token in _identifier_token(str(getattr(device, "path", "")))
    ]
    if not matches:
        raise CameraRegistryError(
            f"{registered.key} ({registered.role}) was not found by exact PnP identity. "
            f"Expected instance_id={registered.instance_id!r}; "
            f"enumerated devices={_describe_devices(devices)}"
        )
    if len(matches) != 1:
        raise CameraRegistryError(
            f"{registered.key} ({registered.role}) matched {len(matches)} devices; "
            "refusing to guess. "
            f"matches={_describe_devices(matches)}"
        )

    device = matches[0]
    try:
        index = int(getattr(device, "index"))
    except (TypeError, ValueError) as exc:
        raise CameraRegistryError(
            f"{registered.key} returned an invalid OpenCV index: {getattr(device, 'index', None)!r}"
        ) from exc

    return ResolvedCamera(
        key=registered.key,
        role=registered.role,
        index=index,
        backend=backend,
        expected_instance_id=registered.instance_id,
        device_name=str(getattr(device, "name", "")),
        device_path=str(getattr(device, "path", "")),
    )


def resolve_stereo_cameras(
    registry_path: str | Path,
    backend: CameraBackend = "auto",
    *,
    enumerate_devices: Callable[[Literal["dshow", "msmf"]], list[EnumeratedCamera]]
    | None = None,
) -> ResolvedStereoCameras:
    """Resolve cam0/left and cam1/right to indices for one concrete backend.

    With ``backend='auto'`` this resolves the first backend whose registry
    identities are both visible.  The caller should still try the returned
    pair for opening and, if that fails, retry the next backend candidate.
    ``run_stereo.py`` performs that full open-time fallback.
    """

    left_registered, right_registered = load_camera_registry(registry_path)
    enumerator = enumerate_devices or _enumerate_windows_cameras
    failures: list[str] = []

    for candidate in backend_candidates(backend):
        try:
            devices = enumerator(candidate)
            left = _resolve_one(left_registered, devices, candidate)
            right = _resolve_one(right_registered, devices, candidate)
            if left.index == right.index:
                raise CameraRegistryError(
                    "cam0/left and cam1/right resolved to the same OpenCV index; "
                    "refusing to start."
                )
            return ResolvedStereoCameras(
                registry_path=Path(registry_path).resolve(),
                backend=candidate,
                left=left,
                right=right,
            )
        except CameraRegistryError as exc:
            failures.append(f"{candidate}: {exc}")

    raise CameraRegistryError(
        "Could not resolve both registered cameras. " + " | ".join(failures)
    )
