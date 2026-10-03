"""Frozen classical low-light front-end transforms and their pixel-level contract.

Scope
-----
This module defines the four preprocessing groups compared by
``tools/benchmark_lowlight_floor_preprocessing.py`` ("raw", "gamma_0p6",
"clahe_lab", "gamma_0p6_then_clahe_lab") and the exact contract every transform
must satisfy.  Every transform is an ordinary classical image operation.  No
learned low-light model is downloaded, trained, fine-tuned or selected with
label feedback, and the parameter values are fixed constants in this file; no
caller can override them.

Boundary
--------
The transforms are applied only to the semantic network's input image.  Stereo
matching, disparity, mask-directed rectification, triangulation, RANSAC and the
fisheye calibration continue to consume the raw upright images, and no enhanced
pixel is ever converted back into a calibrated ray.  Masks produced from an
enhanced image keep the ``automatic_candidate`` status and the
``provided_unvalidated`` identity evidence, so the strict stereo tool can never
turn them into an accepted ground state, a foot height, a contact judgement, a
support judgement or a gait measurement.  Every number this pipeline produces is
internal candidate evidence, not ground truth and not a physical accuracy
measurement.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from .image_io import read_image


RAW = "raw"
GAMMA_0P6 = "gamma_0p6"
CLAHE_LAB = "clahe_lab"
GAMMA_0P6_THEN_CLAHE_LAB = "gamma_0p6_then_clahe_lab"

METHODS: tuple[str, ...] = (RAW, GAMMA_0P6, CLAHE_LAB, GAMMA_0P6_THEN_CLAHE_LAB)

# Fixed parameters.  They are module constants on purpose: the comparison may
# not grow a fifth group and may not search extra gamma or CLAHE values.
GAMMA_EXPONENT = 0.6
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Fixed frame set of the frozen people_1 crop.
ALL_PAIR_IDS: tuple[int, ...] = (0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440)
DEV_PAIR_IDS: tuple[int, ...] = (0, 80, 160, 240, 320, 400)
HOLDOUT_PAIR_IDS: tuple[int, ...] = (40, 120, 200, 280, 360, 440)

# Method selection rule constants.
CONSERVATIVE_RAW_MARGIN = 0.01

# Floating-point tolerance for the fixed comparison against the conservative
# margin.  Without it an exactly boundary margin such as 0.81 - 0.80 would
# evaluate slightly above 0.01 and silently stop being "at most the margin".
MARGIN_COMPARISON_EPSILON = 1e-9

# Provenance labels that a candidate mask derived from an automatic network may
# carry.  ``direct`` is unreachable for these masks by construction.
MASK_STATUS_AUTOMATIC_CANDIDATE = "automatic_candidate"
MASK_IDENTITY_PROVIDED_UNVALIDATED = "provided_unvalidated"
MASK_IDENTITY_MANUALLY_AUDITED = "manually_audited"
SAFE_PLANE_FIELD = "unaccepted_candidate_plane"

# Constants of the frozen candidate chain.  They are inherited from the frozen
# chain source rather than from a command line, and the funnel check in the
# benchmark tool aborts if they do not reproduce the frozen candidate set.
CHAIN_MINIMUM_DEPTH_MM = 250.0
CHAIN_MAXIMUM_DEPTH_MM = 8000.0
CHAIN_MAXIMUM_PHOTOMETRIC_DIFFERENCE = 45


class TransformContractError(RuntimeError):
    """Raised when a preprocessing transform breaks the pixel contract."""


class OutOfBoundsExplanationError(RuntimeError):
    """Raised when an explanation string makes a claim outside the evidence."""


def pair_name(pair_id: int) -> str:
    return f"pair_{int(pair_id):04d}.png"


def split_of(pair_id: int) -> str:
    if pair_id in DEV_PAIR_IDS:
        return "dev"
    if pair_id in HOLDOUT_PAIR_IDS:
        return "holdout"
    raise ValueError(f"pair id {pair_id} is outside the fixed 12-pair frame set")


def _require_bgr_uint8(image: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TransformContractError(f"{label}: expected uint8, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise TransformContractError(f"{label}: expected a three-channel BGR image, got shape {array.shape}")
    return array


def gamma_0p6(image: np.ndarray) -> np.ndarray:
    """Per-channel ``255 * (input / 255) ** 0.6`` written as uint8."""
    source = _require_bgr_uint8(image, "gamma input")
    scaled = source.astype(np.float32) / np.float32(255.0)
    values = np.float32(255.0) * np.power(scaled, np.float32(GAMMA_EXPONENT))
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)


def clahe_lab(image: np.ndarray) -> np.ndarray:
    """OpenCV CLAHE on the LAB L channel only, with the fixed clip and grid."""
    source = _require_bgr_uint8(image, "clahe input")
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    equalised = clahe.apply(lightness)
    merged = cv2.merge((equalised, channel_a, channel_b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_preprocessing(method: str, image: np.ndarray) -> np.ndarray:
    """Return the preprocessing group's output for one BGR upright image."""
    if method not in METHODS:
        raise ValueError(f"unsupported preprocessing method {method!r}; expected one of {METHODS}")
    source = _require_bgr_uint8(image, f"{method} input")
    if method == RAW:
        return source.copy()
    if method == GAMMA_0P6:
        return gamma_0p6(source)
    if method == CLAHE_LAB:
        return clahe_lab(source)
    return clahe_lab(gamma_0p6(source))


def method_parameters(method: str) -> dict[str, Any]:
    """Return the fixed, locked parameter set of one preprocessing group."""
    if method == RAW:
        return {"pixel_transform": "none"}
    if method == GAMMA_0P6:
        return {"pixel_transform": "per_bgr_channel_power", "gamma_exponent": GAMMA_EXPONENT}
    if method == CLAHE_LAB:
        return {
            "pixel_transform": "opencv_clahe_on_lab_lightness_only",
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "clahe_tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
        }
    if method == GAMMA_0P6_THEN_CLAHE_LAB:
        return {
            "pixel_transform": "per_bgr_channel_power_then_opencv_clahe_on_lab_lightness_only",
            "gamma_exponent": GAMMA_EXPONENT,
            "clahe_clip_limit": CLAHE_CLIP_LIMIT,
            "clahe_tile_grid_size": list(CLAHE_TILE_GRID_SIZE),
        }
    raise ValueError(f"unsupported preprocessing method {method!r}; expected one of {METHODS}")


def assert_transform_contract(original: np.ndarray, transformed: np.ndarray, method: str) -> dict[str, Any]:
    """Verify the pixel contract and return the measured checks.

    The contract is what makes the comparison meaningful: the enhanced image is
    the semantic network's input only, and it must therefore stay in the same
    pixel grid as the original upright image so that a mask produced from it
    addresses exactly the same coordinates.
    """
    source = _require_bgr_uint8(original, f"{method} original")
    output = _require_bgr_uint8(transformed, f"{method} output")
    if output.shape != source.shape:
        raise TransformContractError(
            f"{method}: output shape {output.shape} differs from the original shape {source.shape}"
        )
    if not np.all(np.isfinite(output.astype(np.float64))):
        raise TransformContractError(f"{method}: output contains non-finite values")
    identical = bool(np.array_equal(source, output))
    if method == RAW and not identical:
        raise TransformContractError("raw must be a pixel-exact identity transform")
    darker = int(np.count_nonzero(output.astype(np.int16) < source.astype(np.int16)))
    brighter = int(np.count_nonzero(output.astype(np.int16) > source.astype(np.int16)))
    return {
        "method": method,
        "dtype": str(output.dtype),
        "shape_height_width_channels": [int(value) for value in output.shape],
        "same_shape_as_original": True,
        "same_three_bgr_channels": True,
        "all_values_finite": True,
        "pixel_exact_identity": identical,
        "pixels_darker_than_original": darker,
        "pixels_brighter_than_original": brighter,
        "coordinate_alignment": (
            "same upright grid, same top-left origin and same pixel count, so a mask produced from this "
            "image addresses exactly the original upright coordinates without any crop, rotation or scale"
        ),
    }


def load_binary_mask_exact(path: str | Any, expected_shape: tuple[int, int]) -> np.ndarray:
    """Read a binary mask that must already be in the expected pixel grid.

    Unlike the frozen tool's loader this never resizes: a mask produced in a
    different grid is a contract violation, not something to be silently
    resampled onto the original coordinates.
    """
    image = read_image(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise TransformContractError(f"cannot read mask: {path}")
    if image.shape != tuple(expected_shape):
        raise TransformContractError(
            f"{path}: mask shape {image.shape} differs from the expected upright shape {tuple(expected_shape)}; "
            "refusing to resample a mask onto different coordinates"
        )
    return image > 0


def gray_region_statistics(image: np.ndarray, region: np.ndarray) -> dict[str, Any]:
    """Descriptive gray statistics of one region. They are diagnostics only."""
    source = _require_bgr_uint8(image, "statistics input")
    mask = np.asarray(region, dtype=bool)
    if mask.shape != source.shape[:2]:
        raise TransformContractError("region shape must match the image")
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    pixels = gray[mask]
    if pixels.size == 0:
        return {
            "pixel_count": 0,
            "mean_gray_0_255": None,
            "median_gray_0_255": None,
            "p10_gray_0_255": None,
            "p90_gray_0_255": None,
            "reason": "empty_region",
        }
    return {
        "pixel_count": int(pixels.size),
        "mean_gray_0_255": float(np.mean(pixels)),
        "median_gray_0_255": float(np.median(pixels)),
        "p10_gray_0_255": float(np.percentile(pixels, 10)),
        "p90_gray_0_255": float(np.percentile(pixels, 90)),
    }


# ---------------------------------------------------------------------------
# Explanation-language guard
# ---------------------------------------------------------------------------
#
# Every human-readable explanation this pipeline emits is checked against the
# evidence boundary before it can be written.  A sentence that mentions one of
# these subjects must also carry an explicit negation, so that a boundary
# sentence such as "these records are not ground truth" stays allowed while an
# assertive claim such as "the enhanced front-end improved the real ground
# accuracy" is refused.

OUT_OF_BOUNDS_ROOT_TERMS: tuple[str, ...] = (
    "真实精度提高",
    "真实精度提升",
    "真实地面",
    "接触",
    "落地",
    "支撑",
    "足地高度",
    "步态事件",
    "ground truth",
    "foot contact",
    "support phase",
    "ground accuracy",
    "physical accuracy",
)

NEGATION_MARKERS: tuple[str, ...] = (
    "不得",
    "不能",
    "不可",
    "不输出",
    "不构成",
    "不代表",
    "不作为",
    "不进入",
    "不允许",
    "不应",
    "不是",
    "不把",
    "不将",
    "不含",
    "不等于",
    "禁止",
    "未",
    "没有",
    "无法",
    "没",
    "无",
    "非",
    "not ",
    "no ",
    "never",
    "cannot",
    "without",
    "does not",
    "do not",
    "is not",
    "are not",
    "must not",
    "none",
    "cannot be",
)

_SENTENCE_SPLIT = "。；;.!?"
_WHITESPACE = " \t\r\n\f\v"


def normalise_prose(text: str) -> str:
    """Collapse wrapped lines so a sentence is not split by source formatting."""
    output: list[str] = []
    previous_was_space = False
    for character in str(text):
        if character in _WHITESPACE:
            if not previous_was_space:
                output.append(" ")
            previous_was_space = True
        else:
            output.append(character)
            previous_was_space = False
    return "".join(output).strip()


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for character in normalise_prose(text):
        if character in _SENTENCE_SPLIT:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
        else:
            current.append(character)
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def explanation_violations(text: str) -> list[str]:
    """Return the sentences that assert something outside the evidence."""
    violations: list[str] = []
    for sentence in split_sentences(text):
        lowered = sentence.casefold()
        for term in OUT_OF_BOUNDS_ROOT_TERMS:
            if term.casefold() not in lowered:
                continue
            if any(marker.casefold() in lowered for marker in NEGATION_MARKERS):
                break
            violations.append(sentence)
            break
    return violations


def assert_explanation_text(text: str, label: str = "explanation") -> None:
    violations = explanation_violations(text)
    if violations:
        raise OutOfBoundsExplanationError(
            f"{label}: refusal to emit a claim outside the evidence boundary: {violations[:3]}"
        )


def audit_payload_texts(payload: Any, label: str = "payload", _path: str = "") -> int:
    """Recursively check every string in a payload. Returns the checked count."""
    checked = 0
    if isinstance(payload, str):
        assert_explanation_text(payload, f"{label}{_path}")
        return 1
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            checked += audit_payload_texts(value, label, f"{_path}.{key}")
        return checked
    if isinstance(payload, (list, tuple, set)):
        for index, value in enumerate(payload):
            checked += audit_payload_texts(value, label, f"{_path}[{index}]")
    return checked


# Every explanation string that is not derived from the numbers is collected
# here so the unit test can validate the whole vocabulary in one place.
EXPLANATION_TEXTS: tuple[str, ...] = (
    "Fixed classical low-light preprocessing groups applied to the semantic network input only.",
    "The raw upright left and right images, the formal fisheye calibration and the frozen disparity, "
    "rectification, triangulation and RANSAC parameters are used for every geometric number; the enhanced "
    "image is never used for geometry and no enhanced pixel is converted into a calibrated ray.",
    "Candidate masks come from an automatic network, so they carry the automatic_candidate status and the "
    "provided_unvalidated identity evidence and can never produce an accepted ground state.",
    "No accepted ground state, no foot height, no contact judgement, no support judgement and no gait "
    "measurement is produced; the plane records are unaccepted candidate planes.",
    "Brightness statistics are explanatory diagnostics only. They are not correctness evidence, and a "
    "brighter region is not a better floor candidate.",
    "Pixel agreement with the manual holdout labels is a two-dimensional image-space measurement on a small "
    "fixed frame set. It is not ground truth, not a generalisation result and not a physical accuracy "
    "measurement.",
    "Internal consistency of a fitted plane is not ground truth either: a plane can be internally stable and "
    "still not be the floor.",
    "The manual floor_eligible masks are the condition of this comparison. They are not a physical ground "
    "reference and they are not ground truth.",
    "Method selection uses only the development split; the holdout split is reported for all four groups and "
    "is never used to re-select or re-tune a group.",
    "If no group separates from raw on the holdout split, the honest conclusion is that the classical "
    "enhancement front-end was not shown to be effective on this data, and the next step is exposure or "
    "illumination at capture time rather than more parameter scanning.",
)


def deduplicated_terms() -> tuple[str, ...]:
    seen: list[str] = []
    for term in (*OUT_OF_BOUNDS_ROOT_TERMS, *NEGATION_MARKERS):
        if term not in seen:
            seen.append(term)
    return tuple(seen)


def assert_all_explanations() -> int:
    """Validate the module vocabulary. Returns the number of checked strings."""
    checked = 0
    assert_explanation_text(__doc__ or "", "module docstring")
    checked += 1
    for index, text in enumerate(EXPLANATION_TEXTS):
        assert_explanation_text(text, f"EXPLANATION_TEXTS[{index}]")
        checked += 1
    return checked


def terms_not_in_use(extra_texts: Iterable[str]) -> list[str]:
    """Report lexicon terms that appear outside the guard's own vocabulary.

    Used by the tests to show the guard is actually reachable and not dead code.
    """
    return [term for term in deduplicated_terms() if any(term in text for text in extra_texts)]
