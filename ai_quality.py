#!/usr/bin/env python3
"""Intelligent visual-quality scoring for extracted video frames.

The module is deliberately dependency-light. It uses OpenCV's built-in
Haar cascade for face detection and combines face presence, face size,
sharpness and exposure into a ranking score. It does not impose any video
length or file-size limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class QualityResult:
    face_count: int
    largest_face_ratio: float
    face_score: float
    sharpness_score: float
    exposure_score: float
    total_score: float


CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def _exposure(gray) -> float:
    mean = float(gray.mean())
    return max(0.0, 1.0 - abs(mean - 128.0) / 128.0)


def analyse_image(path: Path) -> QualityResult:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    faces = cv2.CascadeClassifier(CASCADE).detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    largest_ratio = 0.0
    for x, y, fw, fh in faces:
        largest_ratio = max(largest_ratio, (fw * fh) / float(w * h))

    face_score = min(1.0, largest_ratio * 8.0) if len(faces) else 0.0
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(1.0, sharpness / 500.0)
    exposure_score = _exposure(gray)

    if len(faces):
        total = 0.55 * face_score + 0.30 * sharpness_score + 0.15 * exposure_score
    else:
        total = 0.15 * sharpness_score + 0.85 * exposure_score

    return QualityResult(
        face_count=len(faces),
        largest_face_ratio=largest_ratio,
        face_score=face_score,
        sharpness_score=sharpness_score,
        exposure_score=exposure_score,
        total_score=total,
    )
