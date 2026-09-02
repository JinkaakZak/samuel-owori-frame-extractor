#!/usr/bin/env python3
"""Smart ranking layer for the video photo extractor.

Keeps the existing extraction engine intact while adding visual intelligence:
face detection, face size, sharpness and exposure scoring. No artificial video
length or file-size limit is imposed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ai_quality import QualityResult, analyse_image
from extractor import FrameResult, perceptual_key


def analyse_candidates(candidates: list[FrameResult]) -> dict[str, QualityResult]:
    """Analyse every extracted candidate and return results keyed by image path."""
    results: dict[str, QualityResult] = {}
    for candidate in candidates:
        results[str(candidate.image)] = analyse_image(Path(candidate.image))
    return results


def smart_score(candidate: FrameResult, quality: QualityResult) -> float:
    """Combine the extractor score with the visual-intelligence score."""
    return 0.45 * candidate.score + 0.55 * quality.total_score


def select_smart(
    candidates: list[FrameResult],
    quality_results: dict[str, QualityResult],
    max_photos: int = 100,
    similarity_threshold: float = 0.015,
) -> list[FrameResult]:
    """Select high-quality, visually diverse frames using intelligent ranking."""
    ranked = sorted(
        candidates,
        key=lambda c: smart_score(c, quality_results[str(c.image)]),
        reverse=True,
    )

    selected: list[FrameResult] = []
    keys = []

    for candidate in ranked:
        key = perceptual_key(candidate.image)
        if any(((key.astype(float) - old.astype(float)) ** 2).mean() < similarity_threshold for old in keys):
            continue
        selected.append(candidate)
        keys.append(key)
        if len(selected) >= max_photos:
            break

    return selected


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
