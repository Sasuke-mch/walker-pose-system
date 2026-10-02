"""Command migration contracts: old calls, new calls and shared module state."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from pose_app.project_paths import APP_ROOT
from walker_tools.catalog import command_path, commands


def run_python(arguments, cwd):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-B", *map(str, arguments)], cwd=cwd,
                          env=environment, capture_output=True, text=True,
                          encoding="utf-8", timeout=40)


def test_catalog_resolves_every_legacy_entry_without_optional_imports():
    for name, filename in commands().items():
        path = command_path(name)
        assert path.is_file()
        assert path == command_path(filename) == command_path(Path(filename).stem)
        shim = APP_ROOT / "tools" / filename
        assert shim.is_file()
        assert "walker_tools." + name in shim.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        command_path("../../run_stereo")


def test_listing_does_not_import_numerical_or_model_packages():
    source = f"""
import sys
sys.path.insert(0, {str(APP_ROOT)!r})
from walker_tools.catalog import commands
assert commands()
assert not {{'torch', 'cv2', 'numpy', 'transformers'}}.intersection(sys.modules)
"""
    result = run_python(["-c", source], tempfile.gettempdir())
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", [
    "body.run_sapiens_cold_smplh",
    "body.fit_smplh_wilor_sequence",
    "body.refine_body_with_constructed_grasp",
    "body.audit_constructed_grasp_body",
    "hands.run_wilor_sequence",
    "visualization.build_grasp_body_canvas_viewer",
    "scene.replay_realtime_stage_walker",
    "capture.capture_stereo",
    "calibration.calibrate_stereo_fisheye",
    "pose2d.build_continuous_foot_inclusive_roi",
])
def test_cli_help_identical_at_old_new_and_dispatch_paths_from_foreign_cwd(name):
    filename = commands()[name]
    invocations = [
        [APP_ROOT / "tools" / filename, "--help"],
        [command_path(name), "--help"],
        [APP_ROOT / "run_tool.py", name, "--help"],
    ]
    results = [run_python(args, tempfile.gettempdir()) for args in invocations]
    for result in results:
        assert result.returncode == 0, result.stderr
    assert results[0].stdout == results[1].stdout == results[2].stdout


def test_legacy_imports_alias_canonical_module_including_private_helpers():
    from walker_tools._compat import prepare_imports
    prepare_imports()
    canonical = importlib.import_module("walker_tools.pose2d.build_continuous_foot_inclusive_roi")
    bare = importlib.import_module("build_continuous_foot_inclusive_roi")
    qualified = importlib.import_module("tools.build_continuous_foot_inclusive_roi")
    assert bare is qualified is canonical
    assert bare.foot_inclusive_box is canonical.foot_inclusive_box
    # A caller patching a historical module must patch the implementation too.
    sentinel = object()
    try:
        bare._layout_sentinel = sentinel
        assert canonical._layout_sentinel is sentinel
    finally:
        del bare._layout_sentinel


def test_dispatcher_retains_cwd_and_arguments(tmp_path, monkeypatch):
    import run_tool
    seen = {}
    def execute(path, run_name):
        seen.update(path=path, name=run_name, argv=list(sys.argv), cwd=Path.cwd())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_tool.runpy, "run_path", execute)
    previous = sys.argv
    run_tool.main(["run_sapiens_cold_smplh.py", "--run", "relative input", "--joint-steps", "300"])
    assert seen["cwd"] == tmp_path
    assert seen["argv"][1:] == ["--run", "relative input", "--joint-steps", "300"]
    assert seen["name"] == "__main__"
    assert sys.argv is previous
