#!/usr/bin/env python3
"""Intelligent visual-quality scoring for extracted video frames.

Combines sharpness and exposure with face-based intelligence from
``face_intelligence.py``: real eye-open/closed detection (Eye Aspect Ratio
from facial landmarks, with a Haar-cascade fallback), head-pose/facing-camera
scoring, face size, face position, and group-photo awareness (every face in
the frame is scored, not just the largest).

No artificial video length or file-size limit is imposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from face_intelligence import FaceIntelligence, GroupResult


@dataclass
class QualityResult:
    face_count: int
    largest_face_ratio: float
    face_score: float
    eye_count: int  # eyes open on the best/largest face (0, 1, or 2)
    eye_visibility_score: float
    face_position_score: float
    facing_camera_score: float
    group_score: float  # rewards multiple people with eyes open, facing camera
    sharpness_score: float
    exposure_score: float
    total_score: float
    method: str  # "landmark" or "haar_fallback" -- which detector produced this


def _exposure(gray) -> float:
    mean = float(gray.mean())
    return max(0.0, 1.0 - abs(mean - 128.0) / 128.0)


def _sharpness_score(gray) -> float:
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(1.0, sharpness / 500.0)


def analyse_with(fi: FaceIntelligence, path: Path) -> QualityResult:
    """Analyse one image using an already-loaded FaceIntelligence instance.
    Prefer this over ``analyse_image`` when processing many frames, since it
    avoids reloading the face model/cascades on every single call."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness_score = _sharpness_score(gray)
    exposure_score = _exposure(gray)

    group: GroupResult = fi.analyse(path)
    best = group.best_face

    if best is not None:
        total = (
            0.35 * min(1.0, best.area_ratio * 8.0)
            + 0.20 * sharpness_score
            + 0.10 * exposure_score
            + 0.20 * (best.eyes_open_count / 2.0)
            + 0.10 * best.facing_camera_score
            + 0.05 * best.position_score
        )
        result = QualityResult(
            face_count=group.face_count,
            largest_face_ratio=best.area_ratio,
            face_score=min(1.0, best.area_ratio * 8.0),
            eye_count=best.eyes_open_count,
            eye_visibility_score=best.eyes_open_count / 2.0,
            face_position_score=best.position_score,
            facing_camera_score=best.facing_camera_score,
            group_score=group.group_score,
            sharpness_score=sharpness_score,
            exposure_score=exposure_score,
            total_score=total,
            method=group.method,
        )
    else:
        total = 0.15 * sharpness_score + 0.85 * exposure_score
        result = QualityResult(
            face_count=0,
            largest_face_ratio=0.0,
            face_score=0.0,
            eye_count=0,
            eye_visibility_score=0.0,
            face_position_score=0.0,
            facing_camera_score=0.0,
            group_score=0.0,
            sharpness_score=sharpness_score,
            exposure_score=exposure_score,
            total_score=total,
            method=group.method,
        )
    return result


def analyse_image(path: Path) -> QualityResult:
    """Standalone convenience wrapper -- loads its own FaceIntelligence
    instance. For batch processing many frames, use ``analyse_with`` with a
    single shared instance instead (much faster)."""
    fi = FaceIntelligence()
    return analyse_with(fi, path)
