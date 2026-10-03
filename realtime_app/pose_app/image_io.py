"""OpenCV image codecs with support for Unicode file paths on Windows."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    name = str(path)
    if name.isascii():
        return cv2.imread(name, flags)
    try:
        encoded = np.fromfile(name, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def write_image(path, image):
    name = str(path)
    if name.isascii():
        return cv2.imwrite(name, image)
    success, encoded = cv2.imencode(Path(name).suffix, image)
    if not success:
        return False
    try:
        encoded.tofile(name)
    except OSError:
        return False
    return True
