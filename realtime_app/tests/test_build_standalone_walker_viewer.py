from __future__ import annotations

import json
import unittest

from tools.build_standalone_walker_viewer import build_standalone_html


class StandaloneWalkerViewerTests(unittest.TestCase):
    def test_build_embeds_css_javascript_and_motion_without_fetch_only_dependency(self) -> None:
        index = (
            '<html><head>\n  <link rel="stylesheet" href="./styles.css">\n</head><body>\n'
            '  <script type="module" src="./app.js"></script>\n</body></html>'
        )
        motion = json.dumps({"frame_count": 1, "frames": [{"pair": 0}]})
        html = build_standalone_html(index, "body { color: red; }", "const ready = true;", motion)
        self.assertIn('<script id="motion-data" type="application/json">', html)
        self.assertIn("body { color: red; }", html)
        self.assertIn("const ready = true;", html)
        self.assertNotIn('src="./app.js"', html)
        self.assertNotIn('href="./styles.css"', html)

    def test_build_rejects_inconsistent_frame_count(self) -> None:
        index = (
            '<html><head>\n  <link rel="stylesheet" href="./styles.css">\n</head><body>\n'
            '  <script type="module" src="./app.js"></script>\n</body></html>'
        )
        with self.assertRaises(ValueError):
            build_standalone_html(index, "", "", '{"frame_count":2,"frames":[]}')


if __name__ == "__main__":
    unittest.main()
