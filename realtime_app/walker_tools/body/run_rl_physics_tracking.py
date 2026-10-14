"""Isolated physics/PPO experiment; registration does not promote its results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from walker_tools._compat import export_legacy

export_legacy(__name__, globals(), "learning_rl_tracking.run")
