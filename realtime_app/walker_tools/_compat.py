"""Compatibility for historical script paths and flat tool imports."""

from __future__ import annotations

import importlib
import runpy
import sys
from typing import Any

from pose_app.project_paths import APP_ROOT

LEGACY_TOOLS = APP_ROOT / "tools"


def prepare_imports() -> None:
    """Retain imports used by historical tools without changing the CWD."""
    for directory in (APP_ROOT, LEGACY_TOOLS):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)


def export_legacy(name: str, namespace: dict[str, Any], target: str) -> None:
    """Forward execution and alias imports to one implementation module.

    Module aliasing preserves monkeypatches and module-level state. Copying the
    namespace also supports callers using spec_from_file_location/exec_module.
    Private helpers remain available, as they were at the historical path.
    """
    prepare_imports()
    if name == "__main__":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
        return
    module = importlib.import_module(target)
    namespace.update({key: value for key, value in vars(module).items()
                      if key not in {"__name__", "__loader__", "__package__",
                                     "__spec__", "__file__", "__cached__"}})
    sys.modules[name] = module
