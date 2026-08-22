from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from pose_app.config import load_config
from pose_app.http_client import PMPosePipelineClient
from pose_app.schema import InferenceResult, PersonPose
from pose_app.sources import ImageDirectorySource, VideoSource, natural_key


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"测试图片编码失败：{path}")
    encoded.tofile(str(path))


class Tests(unittest.TestCase):
    def test_config(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config.example.json")
        self.assertEqual(config.project_root, root.parent.resolve())
        self.assertEqual(config.model.pose_threshold, 0.40)
        self.assertEqual(config.pmpose.pose_threshold, 0.30)
        self.assertEqual(config.detector.host_port, 18081)

    def test_natural_sort(self):
        names = [Path("10.jpg"), Path("2.jpg"), Path("1.jpg")]
        self.assertEqual([p.name for p in sorted(names, key=natural_key)], ["1.jpg", "2.jpg", "10.jpg"])

    def test_person(self):
        person = PersonPose.from_dict({"bbox": [1,2,3,4], "keypoints": [[1,2,.5]], "pose_score": .8})
        self.assertEqual(len(person.keypoints), 17)
        result = InferenceResult(0,0,64,48,"test",1,2,[person],stage_times_ms={"pose_ms":1})
        self.assertEqual(result.to_dict()["stage_times_ms"]["pose_ms"], 1)


    def test_pmpose_pipeline_client(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        responses = [
            {"model_ms": 10.0, "detections": [{"bbox": [1, 2, 20, 40], "score": 0.9}]},
            {
                "model": "YOLO26x + PMPose-b",
                "model_ms": 20.0,
                "image_width": 64,
                "image_height": 48,
                "persons": [{
                    "person_id": 0,
                    "bbox": [1, 2, 20, 40],
                    "bbox_score": 0.9,
                    "pose_score": 0.8,
                    "keypoints": [[3, 4, 0.7]] * 17,
                }],
            },
        ]
        client = PMPosePipelineClient("http://det", "http://pose", 30, 90)
        with patch("pose_app.http_client._json_request", side_effect=responses) as request:
            result = client.infer(image, 7, 0.5, 2)
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result.model_name, "YOLO26x + PMPose-b")
        self.assertEqual(result.model_ms, 30.0)
        self.assertEqual(result.stage_times_ms["detector_ms"], 10.0)
        self.assertEqual(len(result.persons), 1)

    def test_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            image = np.zeros((48,64,3), dtype=np.uint8)
            for index in [1,2,10]:
                write_image(directory / f"{index}.jpg", image)
            images = ImageDirectorySource(directory,25)
            self.assertEqual(images.total_frames,3)
            self.assertIsNotNone(images.read())
            video = directory / "test.mp4"
            writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),10,(64,48))
            self.assertTrue(writer.isOpened())
            for _ in range(3): writer.write(image)
            writer.release()
            source = VideoSource(video)
            self.assertIsNotNone(source.read())
            source.close()


if __name__ == "__main__":
    unittest.main()

