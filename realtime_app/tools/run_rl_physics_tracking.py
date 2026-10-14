"""Compatibility entry for the registered isolated physics/PPO tool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walker_tools._compat import export_legacy

export_legacy(__name__, globals(), "walker_tools.body.run_rl_physics_tracking")
