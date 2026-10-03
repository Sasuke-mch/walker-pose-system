from pathlib import Path
import unittest

from pose_app.project_paths import APP_ROOT, PROJECT_ROOT, project_path, repo_root


class ProjectPathTests(unittest.TestCase):
    def test_root_is_anchored_to_module_location(self) -> None:
        self.assertEqual(repo_root(), PROJECT_ROOT)
        self.assertEqual(APP_ROOT.name, "realtime_app")
        self.assertEqual(PROJECT_ROOT, Path(__file__).resolve().parents[2])

    def test_project_path_does_not_depend_on_current_directory(self) -> None:
        expected = PROJECT_ROOT / "research_records" / "registry"
        self.assertEqual(project_path("research_records", "registry"), expected)
        self.assertTrue(project_path("README.md").is_file())


if __name__ == "__main__":
    unittest.main()
