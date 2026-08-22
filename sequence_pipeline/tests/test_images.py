import tempfile, unittest
from pathlib import Path
from app.images import scan_images

class TestImages(unittest.TestCase):
    def test_natural_order(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            for name in ("frame10.jpg","frame2.jpg","frame1.jpg","x.txt"):
                (p/name).write_bytes(b"")
            self.assertEqual([x.name for x in scan_images(p)], ["frame1.jpg","frame2.jpg","frame10.jpg"])
if __name__ == "__main__": unittest.main()
