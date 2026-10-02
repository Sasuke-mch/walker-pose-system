"""Compatibility entry; implementation: walker_tools.pose2d.benchmark_mask2former_online_latency."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from walker_tools._compat import export_legacy as _export_legacy
_export_legacy(__name__, globals(), "walker_tools.pose2d.benchmark_mask2former_online_latency")
