"""Guard the established ROI rule and online/offline dependency boundary."""

import ast
import subprocess
import sys

import pytest

from pose_app.person_roi import RULE, expansion_fraction, foot_inclusive_box
from pose_app.project_paths import APP_ROOT


def box(value):
    return foot_inclusive_box(value, 1000, 800, RULE["threshold"], RULE["power"])


@pytest.mark.parametrize("width", [1., 200., 369., 370.])
def test_far_and_threshold_boxes_are_preserved(width):
    source = [100., 100., 100. + width, 500.]
    result, metrics = box(source)
    assert result == source
    assert metrics["growth"] == 0.
    assert source == [100., 100., 100. + width, 500.]


def test_close_box_prefers_foot_context_and_respects_image_boundary():
    result, metrics = box([100., 100., 900., 500.])
    assert result[0] < 100. and result[2] > 900.
    assert result[1] < 100. and result[3] == 799.
    assert 0 <= result[0] < result[2] <= 999.
    assert metrics["bottom_pad"] > metrics["side_pad"] > metrics["top_pad"]


def test_growth_is_continuous_at_threshold_and_bounded():
    assert expansion_fraction(.37, .37, .75) == 0.
    assert 0. < expansion_fraction(.370000001, .37, .75) < .000001
    assert expansion_fraction(1., .37, .75) == 1.
    assert expansion_fraction(2., .37, .75) == 1.


def test_online_offline_and_legacy_use_one_roi_function_and_rule():
    from pose_app import http_client
    from walker_tools.pose2d import build_continuous_foot_inclusive_roi as tool
    from tools import build_continuous_foot_inclusive_roi as legacy
    assert http_client.foot_inclusive_box is tool.foot_inclusive_box is legacy.foot_inclusive_box is foot_inclusive_box
    assert http_client.RULE is tool.RULE is legacy.RULE is RULE


def test_online_http_client_does_not_load_offline_command_modules():
    code = (
        f"import sys; sys.path.insert(0, {str(APP_ROOT)!r}); "
        "import pose_app.http_client; "
        "assert not any(n == 'tools' or n.startswith(('tools.', 'walker_tools')) for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_runtime_modules_do_not_import_offline_tool_packages():
    violations = []
    for path in (APP_ROOT / "pose_app").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level
                     else [])
            if any(name.split(".")[0] in {"tools", "walker_tools"} for name in names):
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations, violations
