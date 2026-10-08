"""Render saved PMPose COCO-17 detections on their paired upright images."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EDGES = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / 'PMPose_2D_448frames.mp4'
    if video.exists():
        raise FileExistsError(video)
    records = {}
    for side in ('left', 'right'):
        path = args.source / f'c3_predictions/{side}/pmpose/raw_predictions.json'
        records[side] = json.loads(path.read_text(encoding='utf-8'))['images']
        assert len(records[side]) == 448
        assert [r['file_name'] for r in records[side]] == [f'pair_{i:04d}.png' for i in range(448)]
    width, height = 608, 1080
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'avc1'), 30, (width * 2, height + 60))
    if not writer.isOpened():
        raise RuntimeError('Cannot initialize H.264 video writer')
    counts = {side: {'people': 0, 'low_score': 0, 'unavailable': 0} for side in records}
    try:
        for i in range(448):
            canvas = np.zeros((height + 60, width * 2, 3), dtype=np.uint8)
            for side, rotation, offset in (('left', 'left_ccw90', 0), ('right', 'right_cw90', width)):
                row = records[side][i]
                image = cv2.imread(str(args.source / 'input_448pairs' / rotation / row['file_name']))
                if image is None:
                    raise FileNotFoundError(row['file_name'])
                h, w = image.shape[:2]
                for person in row['keypoints']:
                    points = np.asarray(person, dtype=float)[:17]
                    assert points.shape == (17, 3)
                    counts[side]['people'] += 1
                    valid = np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
                    high = valid & (points[:, 2] >= 0.25)
                    counts[side]['low_score'] += int((valid & ~high).sum())
                    counts[side]['unavailable'] += int((~valid).sum())
                    for a, b in EDGES:
                        if high[a] and high[b]:
                            cv2.line(image, tuple(np.rint(points[a, :2]).astype(int)), tuple(np.rint(points[b, :2]).astype(int)), (0, 220, 255), 5, cv2.LINE_AA)
                    for j, point in enumerate(points):
                        if valid[j]:
                            xy = tuple(np.rint(point[:2]).astype(int))
                            color = (30, 245, 40) if high[j] else (150, 150, 150)
                            cv2.circle(image, xy, 9, (0, 0, 0), -1, cv2.LINE_AA)
                            cv2.circle(image, xy, 6, color, -1, cv2.LINE_AA)
                            cv2.putText(image, str(j), (xy[0]+10, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
                canvas[60:, offset:offset+width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                cv2.putText(canvas, f'{side.upper()}  PMPose COCO-17', (offset+12, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, f'Frame {i+1}/448 | green: score >= 0.25 | gray: lower score', (offset+12, 49), cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1, cv2.LINE_AA)
            writer.write(canvas)
            if i in (0, 224, 447):
                cv2.imwrite(str(args.output / f'preview_{i:04d}.jpg'), canvas)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    fps = capture.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (height+60, width*2, 3)
        decoded += 1
    capture.release()
    assert decoded == 448, decoded
    metadata = {'source': str(args.source.resolve()), 'images': 'input_448pairs/{left_ccw90,right_cw90}/pair_0000.png..pair_0447.png',
                'detections': 'c3_predictions/{left,right}/pmpose/raw_predictions.json',
                'coordinates': 'saved upright model-input pixels; first 17 keypoints are COCO-17',
                'output': str(video.resolve()), 'frames': decoded, 'fps': fps, 'resolution': [width*2, height+60],
                'score_display_threshold': 0.25, 'counts': counts,
                'scope': 'saved raw 2D detections; all frames/persons retained; low scores gray; no smoothing or interpolation'}
    (args.output / 'render_validation.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
