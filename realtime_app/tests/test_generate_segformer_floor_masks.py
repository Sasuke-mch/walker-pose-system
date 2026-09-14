from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from generate_segformer_floor_masks import find_floor_class_id  # noqa: E402


class SegformerFloorCandidateTests(unittest.TestCase):
    def test_finds_the_unique_floor_semantic_class(self):
        self.assertEqual(find_floor_class_id({0: "wall", 3: "floor", 12: "person"}), 3)

    def test_refuses_missing_or_ambiguous_floor_class(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_floor_class_id({0: "wall"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            find_floor_class_id({3: "floor", 4: "Floor"})


if __name__ == "__main__":
    unittest.main()
