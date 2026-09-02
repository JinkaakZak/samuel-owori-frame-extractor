#!/usr/bin/env python3
"""True facial-landmark eye-state and head-pose intelligence.

Upgrades the previous Haar-cascade "1 or 2 eyes visible" heuristic with:

  * MediaPipe FaceLandmarker (478-point face mesh, incl. iris landmarks)
  * Eye-Aspect-Ratio (EAR) based open/closed eye classification per eye
  * A lightweight head-yaw estimate (facing-camera score) from landmark
    geometry -- no camera calibration required
  * Multi-face support for group photos: every detected face is scored,
    not just the largest one

If MediaPipe or its model file is unavailable (e.g. no internet on first
run, since the ~3.5MB model is downloaded from Google on first use), this
module falls back automatically to the original Haar-cascade heuristic from
``ai_quality.py`` so the pipeline never hard-crashes -- it just runs with
less precise eye-state detection until the model can be downloaded.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

# Standard MediaPipe FaceMesh landmark indices used for Eye Aspect Ratio.
# (These are the widely-used 6-point EAR sets mapped onto the 468/478-point mesh.)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# A few stable landmarks used for a lightweight, calibration-free yaw estimate.
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

EAR_OPEN_THRESHOLD = 0.19  # below this, an eye is considered closed


@dataclass
class FaceQuality:
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixels
    area_ratio: float
    left_ear: float | None
    right_ear: float | None
    eyes_open_count: int  # 0, 1, or 2 -- how many eyes are open (None-safe)
    facing_camera_score: float  # 0 (profile/turned away) .. 1 (facing camera)
    position_score: float
    face_quality: float  # combined 0..1 score for this one face


@dataclass
class GroupResult:
    faces: list[FaceQuality] = field(default_factory=list)
    method: str = "landmark"  # "landmark" or "haar_fallback"

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def best_face(self) -> FaceQuality | None:
        return max(self.faces, key=lambda f: f.face_quality, default=None)

    @property
    def group_score(self) -> float:
        """Rewards frames where MORE people have eyes open and face the camera --
        useful for group photos where a single dominant face isn't the point."""
        if not self.faces:
            return 0.0
        good = sum(
            1
            for f in self.faces
            if f.eyes_open_count >= 2 and f.facing_camera_score > 0.5
        )
        avg_quality = sum(f.face_quality for f in self.faces) / len(self.faces)
        coverage = good / len(self.faces)
        return 0.6 * avg_quality + 0.4 * coverage


def _position_score(x: int, y: int, fw: int, fh: int, width: int, height: int) -> float:
    face_cx = (x + fw / 2.0) / width
    face_cy = (y + fh / 2.0) / height
    dx = face_cx - 0.5
    dy = face_cy - 0.5
    distance = (dx * dx + dy * dy) ** 0.5
    return max(0.0, 1.0 - distance / 0.7072)


def _ear(landmarks, indices, w, h) -> float:
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    vert1 = np.hypot(p2[0] - p6[0], p2[1] - p6[1])
    vert2 = np.hypot(p3[0] - p5[0], p3[1] - p5[1])
    horiz = np.hypot(p1[0] - p4[0], p1[1] - p4[1])
    if horiz == 0:
        return 0.0
    return float((vert1 + vert2) / (2.0 * horiz))


def _facing_camera_score(landmarks, w, h) -> float:
    """Calibration-free yaw proxy: compares how centred the nose tip is
    between the two cheek landmarks. 1.0 = perfectly centred (facing camera),
    lower values = head turned to one side."""
    nose = landmarks[NOSE_TIP]
    left = landmarks[LEFT_CHEEK]
    right = landmarks[RIGHT_CHEEK]
    nose_x = nose.x * w
    left_x = left.x * w
    right_x = right.x * w
    span = right_x - left_x
    if abs(span) < 1e-6:
        return 0.0
    center = (left_x + right_x) / 2.0
    offset_ratio = abs(nose_x - center) / (abs(span) / 2.0)
    return max(0.0, 1.0 - min(1.0, offset_ratio))


class FaceIntelligence:
    """Loads the MediaPipe FaceLandmarker once and reuses it across frames.
    Falls back to the legacy Haar-cascade heuristic if unavailable."""

    def __init__(self) -> None:
        self._landmarker = None
        self._method = "haar_fallback"
        self._face_cascade = None
        self._eye_cascade = None
        self._try_load_landmarker()
        if self._landmarker is None:
            self._load_haar_fallback()

    def _try_load_landmarker(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not MODEL_PATH.exists():
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

            base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=8,  # supports group photos
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            self._vision = vision
            self._method = "landmark"
        except Exception:
            # No internet for the model download, unsupported platform, etc.
            # We fall back below rather than crashing the whole pipeline.
            self._landmarker = None

    def _load_haar_fallback(self) -> None:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self._face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self._eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    @property
    def method(self) -> str:
        return self._method

    def analyse(self, image_path: Path) -> GroupResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = image.shape[:2]

        if self._landmarker is not None:
            return self._analyse_landmark(image, w, h)
        return self._analyse_haar(image, w, h)

    def _analyse_landmark(self, image, w: int, h: int) -> GroupResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        faces: list[FaceQuality] = []
        for landmarks in result.face_landmarks:
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            fw, fh = x1 - x0, y1 - y0
            area_ratio = (fw * fh) / float(w * h)

            left_ear = _ear(landmarks, LEFT_EYE, w, h)
            right_ear = _ear(landmarks, RIGHT_EYE, w, h)
            eyes_open = sum(
                1 for ear in (left_ear, right_ear) if ear >= EAR_OPEN_THRESHOLD
            )
            facing = _facing_camera_score(landmarks, w, h)
            position = _position_score(int(x0), int(y0), int(fw), int(fh), w, h)
            face_score = min(1.0, area_ratio * 8.0)

            quality = (
                0.35 * face_score
                + 0.30 * (eyes_open / 2.0)
                + 0.20 * facing
                + 0.15 * position
            )

            faces.append(
                FaceQuality(
                    bbox=(int(x0), int(y0), int(fw), int(fh)),
                    area_ratio=area_ratio,
                    left_ear=left_ear,
                    right_ear=right_ear,
                    eyes_open_count=eyes_open,
                    facing_camera_score=facing,
                    position_score=position,
                    face_quality=quality,
                )
            )

        return GroupResult(faces=faces, method="landmark")

    def _analyse_haar(self, image, w: int, h: int) -> GroupResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        faces: list[FaceQuality] = []
        for x, y, fw, fh in detected:
            area_ratio = (fw * fh) / float(w * h)
            roi = gray[y : y + fh, x : x + fw]
            eyes = self._eye_cascade.detectMultiScale(
                roi,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(max(10, fw // 12), max(10, fh // 12)),
            )
            eyes_open = min(2, len(eyes))  # heuristic: detected == open (approx.)
            position = _position_score(x, y, fw, fh, w, h)
            face_score = min(1.0, area_ratio * 8.0)
            quality = (
                0.40 * face_score
                + 0.30 * (eyes_open / 2.0)
                + 0.30 * position
            )
            faces.append(
                FaceQuality(
                    bbox=(x, y, fw, fh),
                    area_ratio=area_ratio,
                    left_ear=None,
                    right_ear=None,
                    eyes_open_count=eyes_open,
                    facing_camera_score=0.5,  # unknown under this fallback
                    position_score=position,
                    face_quality=quality,
                )
            )
        return GroupResult(faces=faces, method="haar_fallback")
