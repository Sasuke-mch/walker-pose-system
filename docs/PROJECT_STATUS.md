# Project status

Last reviewed: 2026-08-28

## What is verified

- The realtime application accepts camera, video, side-by-side video, and recorded stereo-capture replay inputs.
- Two fisheye cameras have a ChArUco stereo calibration at 1920 × 1080. The calibrated coordinate system is `left_camera` and the length unit is mm.
- Live capture records host-side `VideoCapture.read()` return timestamps and produces an explicit one-to-one `stereo_pairs.csv`; formal replay restores that pairing rather than matching equal AVI frame numbers.
- The runtime supports camera-identity resolution, rotation normalization, person association, fisheye triangulation, positive-depth checks, and reprojection-error rejection.
- `python realtime_app/run_tests.py` passes 45 unit tests on the reviewed Windows environment. GitHub Actions runs the same suite on Python 3.11 and 3.12 for future changes.

## Formal replay baseline

The authoritative experiment registry is `research_records/registry/experiment_registry.csv`; raw assets and rendered outputs are deliberately local-only.

| Experiment | Input | Change from B0 | Result |
| --- | --- | --- | --- |
| E20260827-B0 | R20260826-01 approach, 389 true CSV pairs | Baseline | 15.3368 valid 3D points/pair; 3.216 FPS |
| E20260827-L1 | Same 389 pairs | Strict local virtual perspective | 15.3933 valid 3D points/pair; 0.920 FPS; not a useful trade-off |
| E20260827-U1 | Same 389 pairs | Model-input fisheye undistortion | 9.1003 valid 3D points/pair; negative result |

The timestamp deltas recorded for B0 are host read-return deltas, not camera exposure synchronisation errors.

## Evidence limits

- A higher number of valid 3D points is coverage, not proof of greater 3D accuracy.
- No rigorous external 3D ground-truth evaluation has been completed.
- M0 is a five-model engineering screen with consistent 60-pair inputs; it is not a real-world 3D accuracy comparison.
- COCO-17 has ankle points but no toe or heel points. The current system does not reconstruct complete foot anatomy or validated gait events.
- Local virtual perspective and global model-input undistortion should not be retuned without a new pre-registered ablation and an explicit B0 comparison.

## Current next step

Build a lower-limb/foot quality evaluation set that separates failures in 2D detection, left-right association, triangulation, and reprojection gating. A future foot-specific model must define the target landmarks (at least ankle, heel, and toe) before its performance can be meaningfully evaluated.

## Repository boundaries

The public repository contains source code, test code, portable configuration templates, compact experiment registry metadata, and small qualitative examples. It deliberately excludes model weights, third-party source trees, raw captures, videos, JSONL outputs, and local device bindings. See [`third_party/README.md`](../third_party/README.md) and [`realtime_app/docs/CAMERA_REGISTRY.md`](../realtime_app/docs/CAMERA_REGISTRY.md).
