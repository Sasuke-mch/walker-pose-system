"""Compatibility entry; implementation: walker_tools.pose2d.evaluate_sapiens2_temporal_optical_flow."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from walker_tools._compat import export_legacy as _export_legacy
_export_legacy(__name__, globals(), "walker_tools.pose2d.evaluate_sapiens2_temporal_optical_flow")
