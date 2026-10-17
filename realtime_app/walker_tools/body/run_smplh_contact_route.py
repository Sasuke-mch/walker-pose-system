"""Expose the autograd/contact route through the existing command catalog."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from walker_tools._compat import export_legacy

export_legacy(__name__, globals(), 'tools.run_smplh_contact_route')
