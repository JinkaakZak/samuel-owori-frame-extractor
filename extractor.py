#!/usr/bin/env python3
"""Video -> intelligent photo extractor.

No artificial file-size or duration limit is imposed by this application.
Processing capacity depends on the computer, available storage, FFmpeg, and
video codec. The tool samples frames, scores sharpness/exposure, and removes
near-duplicates before saving the best candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class FrameResult:
    timestamp: float
    source_frame: int
    sharpness: float
    brightness: float
    exposure_score: float
    score: float
    image: str


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("FFmpeg and FFprobe must be installed and available on PATH.")


def video_info(video: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(video)
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
    duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
    fps_text = stream.get("avg_frame_rate", "0/1")
    n, d = fps_text.split("/")
    fps = float(n) / float(d) if float(d) else 0.0
    return {"duration": duration, "fps": fps, "width": stream.get("width"), "height": stream.get("height")}


def score_frame(image) -> tuple[float, float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    # Prefer well-exposed images around the middle of the luminance range.
    exposure_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    sharp_norm = min(1.0, math.log1p(sharpness) / math.log1p(1500.0))
    score = 0.82 * sharp_norm + 0.18 * exposure_score
    return sharpness, brightness, exposure_score, score


def extract_candidates(video: Path, work_dir: Path, sample_every: float) -> list[FrameResult]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install dependencies with: pip install -r requirements.txt")

    work_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    next_time = 0.0
    results: list[FrameResult] = []
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_no / fps
        if timestamp + 1e-9 >= next_time:
            sharpness, brightness, exposure, score = score_frame(frame)
            filename = f"frame_{frame_no:09d}_{timestamp:010.3f}.jpg"
            path = work_dir / filename
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            results.append(FrameResult(timestamp, frame_no, sharpness, brightness, exposure, score, filename))
            next_time += sample_every
        frame_no += 1

    cap.release()
    return results


def perceptual_key(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    return image.astype("float32")


def select_best(results: list[FrameResult], work_dir: Path, output_dir: Path, max_photos: int, similarity_threshold: float) -> list[FrameResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[FrameResult] = []
    keys = []
    for item in sorted(results, key=lambda x: x.score, reverse=True):
        if len(selected) >= max_photos:
            break
        key = perceptual_key(work_dir / item.image)
        if key is None:
            continue
        duplicate = False
        for previous in keys:
            mse = float(((key - previous) ** 2).mean())
            normalized = mse / (255.0 ** 2)
            if normalized < similarity_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        keys.append(key)
        destination = output_dir / item.image
        shutil.copy2(work_dir / item.image, destination)
        selected.append(item)

    selected.sort(key=lambda x: x.timestamp)
    return selected


def write_report(output_dir: Path, info: dict, selected: list[FrameResult]) -> None:
    report = {
        "video": info,
        "selected_count": len(selected),
        "frames": [asdict(x) for x in selected],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(selected[0]).keys() if selected else ["timestamp", "source_frame", "sharpness", "brightness", "exposure_score", "score", "image"])
        writer.writeheader()
        writer.writerows(asdict(x) for x in selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sharp, well-exposed, non-duplicate photos from any video.")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--sample-every", type=float, default=1.0, help="Sample one frame every N seconds")
    parser.add_argument("--max-photos", type=int, default=100, help="Maximum selected photos")
    parser.add_argument("--similarity-threshold", type=float, default=0.015, help="Lower values remove more near-duplicates")
    args = parser.parse_args()

    if args.sample_every <= 0:
        raise SystemExit("--sample-every must be greater than zero")
    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    require_ffmpeg()
    info = video_info(args.video)
    work_dir = args.output / "_frames"
    selected_dir = args.output / "selected_photos"

    print(f"Video: {args.video}")
    print(f"Duration: {info['duration']:.2f}s | FPS: {info['fps']:.2f} | Resolution: {info['width']}x{info['height']}")
    print(f"Sampling every {args.sample_every}s ...")
    candidates = extract_candidates(args.video, work_dir, args.sample_every)
    print(f"Analysed {len(candidates)} candidate frames.")
    selected = select_best(candidates, work_dir, selected_dir, args.max_photos, args.similarity_threshold)
    write_report(args.output, info, selected)
    print(f"Selected {len(selected)} photos.")
    print(f"Photos: {selected_dir}")
    print(f"Reports: {args.output / 'report.json'} and {args.output / 'report.csv'}")


if __name__ == "__main__":
    main()
