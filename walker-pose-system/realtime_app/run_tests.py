from __future__ import annotations
import compileall
from pathlib import Path
import unittest

def main()->int:
    root=Path(__file__).resolve().parent
    if not compileall.compile_dir(root,quiet=1):return 1
    result=unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(root/"tests"),pattern="test_*.py"))
    return 0 if result.wasSuccessful() else 1
if __name__=="__main__":raise SystemExit(main())
