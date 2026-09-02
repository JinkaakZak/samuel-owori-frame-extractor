#!/usr/bin/env python3
"""Smart ranking layer for the video photo extractor.

Keeps the existing extraction engine intact while adding visual intelligence:
face detection, eye-state (open/closed), head-pose/facing-camera scoring,
sharpness, exposure, and group-photo awareness. No artificial video length or
file-size limit is imposed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from ai_quality import QualityResult, analyse_with
from face_intelligence import FaceIntelligence
from extractor import FrameResult, perceptual_key


def analyse_candidates(candidates: list[FrameResult], work_dir: Path) -> dict[str, QualityResult]:
    """Analyse every extracted candidate and return results keyed by image filename.

    ``work_dir`` is required: FrameResult.image stores only the bare filename
    (e.g. "frame_000000000_000000.000.jpg"), so callers must tell us which
    folder those files actually live in -- otherwise every image lookup fails.
    A single FaceIntelligence instance is loaded once and reused across every
    frame, since loading the face model per-frame would be very slow.
    """
    fi = FaceIntelligence()
    results: dict[str, QualityResult] = {}
    for candidate in candidates:
        image_path = work_dir / candidate.image
        results[str(candidate.image)] = analyse_with(fi, image_path)
    return results


def smart_score(candidate: FrameResult, quality: QualityResult) -> float:
    """Combine the extractor score with the visual-intelligence score.
    When more than one face is present, blend in the group score too, so a
    frame with several people looking at the camera can outrank a frame with
    only one great single-person shot."""
    base = 0.45 * candidate.score + 0.55 * quality.total_score
    if quality.face_count > 1:
        base = 0.7 * base + 0.3 * quality.group_score
    return base


def select_smart(
    candidates: list[FrameResult],
    quality_results: dict[str, QualityResult],
    work_dir: Path,
    max_photos: int = 100,
    similarity_threshold: float = 0.015,
) -> list[FrameResult]:
    """Select high-quality, visually diverse frames using intelligent ranking.

    ``work_dir`` is required for the same reason as in ``analyse_candidates``:
    FrameResult.image is a bare filename, not a full path.
    """
    ranked = sorted(
        candidates,
        key=lambda c: smart_score(c, quality_results[str(c.image)]),
        reverse=True,
    )

    selected: list[FrameResult] = []
    keys = []

    for candidate in ranked:
        image_path = work_dir / candidate.image
        key = perceptual_key(image_path)
        if key is None:
            continue
        if any(((key.astype(float) - old.astype(float)) ** 2).mean() < similarity_threshold for old in keys):
            continue
        selected.append(candidate)
        keys.append(key)
        if len(selected) >= max_photos:
            break

    return selected


def copy_selected(selected: list[FrameResult], work_dir: Path, selected_dir: Path) -> None:
    """Copy selected candidate images into the final output folder.

    ``work_dir`` is required for the same reason as above.
    """
    selected_dir.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(selected, start=1):
        source = work_dir / candidate.image
        destination = selected_dir / f"photo_{index:04d}.jpg"
        shutil.copy2(source, destination)


def write_smart_report(
    output_dir: Path,
    selected: list[FrameResult],
    quality_results: dict[str, QualityResult],
) -> Path:
    """Write the intelligent ranking data to a JSON report."""
    report = []
    for candidate in selected:
        quality = quality_results[str(candidate.image)]
        report.append(
            {
                **asdict(candidate),
                "image": str(candidate.image),
                "ai_quality": asdict(quality),
                "smart_score": smart_score(candidate, quality),
            }
        )

    path = output_dir / "smart_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
