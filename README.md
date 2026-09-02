# Video → Intelligent Photo Extractor

A reusable tool for extracting high-quality still photos from long videos, livestream recordings, memorial videos, weddings, interviews, and events.

## Design goals

- No artificial video duration limit.
- No artificial GB/file-size limit.
- Video files stay on the local computer; they are **not** stored in GitHub.
- FFmpeg/FFprobe are used for video metadata and processing.
- OpenCV scores frame sharpness and exposure.
- Near-duplicate frames are filtered.
- Selected photos are copied to `output/selected_photos`.
- A JSON and CSV report records timestamps and quality scores.

## Current version

This is the first working engine. It deliberately uses a lightweight local pipeline so it can handle long recordings without requiring the entire video to be uploaded to GitHub.

## Install

1. Install FFmpeg and make sure `ffmpeg` and `ffprobe` work in PowerShell.
2. Create a Python virtual environment if desired.
3. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

## Run

From the repository folder:

```powershell
python extractor.py "C:\path\to\your\video.mp4"
```

Example:

```powershell
python extractor.py "C:\Videos\Samuel Frobisher Owori.mp4" --sample-every 1 --max-photos 100
```

## Useful options

```text
--sample-every 1.0             Analyse one frame every second
--max-photos 100               Maximum number of selected photos
--similarity-threshold 0.015   Near-duplicate sensitivity
--output output                Output directory
```

For a long event recording, reduce `--sample-every` to find more moments. For example, `0.5` analyses two frames per second.

## Output

```text
output/
├── selected_photos/
├── report.csv
├── report.json
└── _frames/
```

The `_frames` folder contains sampled working frames and can be deleted after selection if storage becomes an issue.

## Roadmap

1. Better scene-change detection.
2. Face detection and face-quality scoring.
3. Person/subject detection.
4. Closed-eye and motion-blur filtering.
5. Intelligent burst selection around important moments.
6. Contact-sheet preview.
7. Desktop GUI with video picker and progress bar.
8. Pause/resume and crash recovery.
9. Batch processing of multiple videos.
10. Optional AI ranking for memorial, wedding, and event photography.
