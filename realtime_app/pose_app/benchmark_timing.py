"""Module-level CPU wall-clock timing shared by the ground benchmark tools.

This module exists so that every later comparison (matcher, plane fitter,
temporal helper) can be timed with one identical, inspectable recorder instead
of ad-hoc ``time.time()`` calls sprinkled through experiment scripts.

Design rules kept deliberately narrow:

* every duration is a CPU wall-clock interval measured with
  :func:`time.perf_counter`;
* when a CUDA device is visible, ``torch.cuda.synchronize()`` is called before
  the start timestamp and again before the end timestamp, so an asynchronous
  GPU kernel cannot be timed as zero and cannot leak its cost into the next
  segment;
* the raw per-sample durations of every module are retained, not just the
  summary, so a caller can print a per-frame table;
* an empty collector summarises to an empty mapping instead of raising or
  dividing by zero;
* no new third-party dependency is introduced.  ``torch`` is optional and is
  only imported to decide whether a CUDA synchronisation is needed.

Environment constraint (stated, not hidden)
-------------------------------------------

The CUDA probe above is the default behaviour, but it is not always usable *in
the same process* as this project's numeric stack.  This machine's NumPy is an
Anaconda MKL build (``mkl-sdl``) that ships its own ``libiomp5md.dll``, and the
locally installed ``torch`` ships a second copy.  Whichever copy initialises
first wins and the other copy aborts the whole process with

    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
    already initialized.

That abort is not a Python exception and cannot be caught.  Measured on this
host it fires on the first ``np.linalg.eigh`` / ``np.linalg.svd`` call after
``import torch`` (see
``research_records/engineering_validation/G20260912_modular_ground_benchmark_v1/EXPERIMENT.md``).
A CPU-only caller that performs NumPy linear algebra therefore passes
``gpu_synchronize=False``: no torch import, no CUDA synchronisation, and the
recorded timings stay plain CPU wall-clock.  The unsafe
``KMP_DUPLICATE_LIB_OK=TRUE`` workaround is deliberately not used, because it is
documented to risk silent numerical corruption.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import time
from typing import Any, Iterator


__all__ = [
    "SUMMARY_FIELDS",
    "TimingCollector",
    "cuda_is_available",
    "synchronize_if_cuda",
]


SUMMARY_FIELDS = ("count", "mean_ms", "median_ms", "p90_ms", "p95_ms", "min_ms", "max_ms")

_TORCH_RESOLVED = False
_TORCH_MODULE: Any = None


def _torch() -> Any:
    """Return the ``torch`` module once, or ``None`` when it is unavailable."""
    global _TORCH_RESOLVED, _TORCH_MODULE
    if not _TORCH_RESOLVED:
        try:
            import torch  # noqa: PLC0415 - optional dependency, resolved lazily
        except Exception:  # pragma: no cover - depends on the local environment
            _TORCH_MODULE = None
        else:
            _TORCH_MODULE = torch
        _TORCH_RESOLVED = True
    return _TORCH_MODULE


def cuda_is_available() -> bool:
    """True only when an optional ``torch`` build reports a usable CUDA device."""
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive, environment dependent
        return False


def synchronize_if_cuda() -> bool:
    """Block on the current CUDA stream when one exists.

    Returns ``True`` when a synchronisation was actually performed.  Any
    environment failure is reported as ``False`` rather than raised: measuring a
    CPU-only run must never fail because of the accelerator probe.
    """
    torch = _torch()
    if torch is None:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.synchronize()
    except Exception:  # pragma: no cover - defensive, environment dependent
        return False
    return True


def percentile_ms(samples: list[float], percent: float) -> float:
    """Linear-interpolation percentile of already-sorted-to-be samples.

    Uses the same rule as ``numpy.percentile``'s default ("linear"): the rank is
    ``(n - 1) * p / 100`` and the result interpolates between the two
    neighbouring order statistics.  Implemented locally so that this module has
    no numeric dependency.
    """
    if not samples:
        raise ValueError("percentile of an empty sample set is undefined")
    if not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be within [0, 100]")
    ordered = sorted(float(value) for value in samples)
    rank = (len(ordered) - 1) * (percent / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class TimingCollector:
    """Collect per-sample millisecond durations for named pipeline modules.

    ``gpu_synchronize`` selects the CUDA policy:

    * ``None`` (default) -- probe ``torch.cuda.is_available()`` and synchronise
      when a CUDA device is visible;
    * ``False`` -- never import ``torch`` and never synchronise; required by
      CPU-only callers on hosts where torch and the numeric stack cannot share a
      process (see the module docstring);
    * ``True`` -- always attempt a synchronisation on both sides of every
      measured segment.
    """

    def __init__(self, *, gpu_synchronize: bool | None = None) -> None:
        self._samples: dict[str, list[float]] = {}
        self._gpu_synchronized = False
        self._gpu_synchronize = gpu_synchronize

    @property
    def gpu_synchronize_policy(self) -> bool | None:
        """The configured policy: ``None`` means auto-detect."""
        return self._gpu_synchronize

    def _should_synchronize(self) -> bool:
        if self._gpu_synchronize is None:
            return cuda_is_available()
        return bool(self._gpu_synchronize)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Time the enclosed block and record it under ``name``.

        A CUDA synchronise is issued before the start timestamp and before the
        end timestamp whenever the collector's policy asks for one, in both
        cases outside the measured interval's endpoints so that pending GPU work
        is charged to the segment that launched it.
        """
        synchronized = self._should_synchronize()
        self._gpu_synchronized = self._gpu_synchronized or synchronized
        start = time.perf_counter()
        try:
            yield
        finally:
            if synchronized:
                synchronize_if_cuda()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record_ms(name, elapsed_ms)

    def record_ms(self, name: str, elapsed_ms: float) -> None:
        """Append one already-measured duration in milliseconds."""
        value = float(elapsed_ms)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}: elapsed_ms must be a finite non-negative number")
        self._samples.setdefault(str(name), []).append(value)

    def samples_ms(self, name: str) -> list[float]:
        """Every recorded duration for one module, in insertion order."""
        return list(self._samples.get(name, []))

    def module_names(self) -> list[str]:
        return list(self._samples)

    def total_samples(self) -> int:
        return int(sum(len(values) for values in self._samples.values()))

    @property
    def gpu_synchronized(self) -> bool:
        """True when at least one measured segment was CUDA-synchronised."""
        return bool(self._gpu_synchronized)

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Per-module statistics; an empty collector returns an empty mapping."""
        summary: dict[str, dict[str, float | int]] = {}
        for name, values in self._samples.items():
            if not values:
                continue
            summary[name] = {
                "count": int(len(values)),
                "mean_ms": float(sum(values) / len(values)),
                "median_ms": float(percentile_ms(values, 50.0)),
                "p90_ms": float(percentile_ms(values, 90.0)),
                "p95_ms": float(percentile_ms(values, 95.0)),
                "min_ms": float(min(values)),
                "max_ms": float(max(values)),
            }
        return summary
