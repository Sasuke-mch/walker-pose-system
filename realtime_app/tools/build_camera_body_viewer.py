"""Forward the camera-coordinate diagnostic to its visualization module."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walker_tools._compat import export_legacy

export_legacy(__name__, globals(), 'walker_tools.visualization.build_camera_body_viewer')
