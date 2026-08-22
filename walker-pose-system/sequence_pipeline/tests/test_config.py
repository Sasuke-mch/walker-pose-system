import unittest
from pathlib import Path
from app.config import load_config

class TestConfig(unittest.TestCase):
    def test_load(self):
        root=Path(__file__).resolve().parents[1]
        c=load_config(root)
        self.assertIn("yolo26x_pose", c.models)
        self.assertEqual(c.project_root, root.parent.resolve())
if __name__ == "__main__": unittest.main()
