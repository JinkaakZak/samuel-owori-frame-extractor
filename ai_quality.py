#!/usr/bin/env python3
"""Intelligent visual-quality scoring for extracted video frames.

Uses OpenCV's built-in Haar cascades for face and eye detection and combines
face size, face position, eye visibility, sharpness and exposure into a
ranking score. Eye detection is a heuristic: it estimates visible eyes but
is not a medical-grade or deep-learning eye-state classifier.

No artificial video length or file-size limit is imposed.
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
    eye_count: int
    eye_visibility_score: float
    face_position_score: float
    sharpness_score: float
    exposure_score: float
    total_score: float


FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE = cv2.data.haarcascades + "haarcascade_eye.xml"


def _exposure(gray) -> float:
    mean = float(gray.mean())
    return max(0.0, 1.0 - abs(mean - 128.0) / 128.0)


def _position_score(x: int, y: int, fw: int, fh: int, width: int, height: int) -> float:
    """Score how naturally positioned the largest face is in the frame."""
    face_cx = (x + fw / 2.0) / width
    face_cy = (y + fh / 2.0) / height
    dx = face_cx - 0.5
    dy = face_cy - 0.5
    distance = (dx * dx + dy * dy) ** 0.5
    return max(0.0, 1.0 - distance / 0.7072)


def analyse_image(path: Path) -> QualityResult:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    face_detector = cv2.CascadeClassifier(FACE_CASCADE)
    eye_detector = cv2.CascadeClassifier(EYE_CASCADE)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    largest_ratio = 0.0
    largest_face = None
    for face in faces:
        x, y, fw, fh = face
        ratio = (fw * fh) / float(w * h)
        if ratio > largest_ratio:
            largest_ratio = ratio
            largest_face = face

    face_score = min(1.0, largest_ratio * 8.0) if len(faces) else 0.0

    eye_count = 0
    eye_visibility_score = 0.0
    face_position_score = 0.0

    if largest_face is not None:
        x, y, fw, fh = largest_face
        face_position_score = _position_score(x, y, fw, fh, w, h)

        roi = gray[y:y + fh, x:x + fw]
        eyes = eye_detector.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(max(10, fw // 12), max(10, fh // 12)),
        )
        eye_count = min(2, len(eyes))
        eye_visibility_score = {0: 0.0, 1: 0.55, 2: 1.0}[eye_count]

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = min(1.0, sharpness / 500.0)
    exposure_score = _exposure(gray)

    if len(faces):
        total = (
            0.40 * face_score
            + 0.22 * sharpness_score
            + 0.13 * exposure_score
            + 0.15 * eye_visibility_score
            + 0.10 * face_position_score
        )
    else:
        total = 0.15 * sharpness_score + 0.85 * exposure_score

    return QualityResult(
        face_count=len(faces),
        largest_face_ratio=largest_ratio,
        face_score=face_score,
        eye_count=eye_count,
        eye_visibility_score=eye_visibility_score,
        face_position_score=face_position_score,
        sharpness_score=sharpness_score,
        exposure_score=exposure_score,
        total_score=total,
    )
