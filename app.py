#!/usr/bin/env python3
"""Desktop interface for the intelligent video photo extractor."""

from __future__ import annotations

import os
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import extractor
import smart_pipeline


class ExtractorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Intelligent Video Photo Extractor")
        self.geometry("820x620")
        self.minsize(760, 560)

        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(Path.cwd() / "output"))
        self.sample_var = tk.StringVar(value="1.0")
        self.max_photos_var = tk.StringVar(value="100")
        self.threshold_var = tk.StringVar(value="0.015")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=20)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="INTELLIGENT VIDEO PHOTO EXTRACTOR", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(root, text="Video → analyse frames → AI quality ranking → duplicate removal → photos", font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 18))

        files = ttk.LabelFrame(root, text="Files", padding=14)
        files.pack(fill="x", pady=(0, 12))

        ttk.Label(files, text="Video").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(files, textvariable=self.video_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(files, text="Browse…", command=self.choose_video).grid(row=0, column=2)

        ttk.Label(files, text="Output folder").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(files, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(files, text="Browse…", command=self.choose_output).grid(row=1, column=2)
        files.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(root, text="Analysis settings", padding=14)
        settings.pack(fill="x", pady=(0, 12))

        ttk.Label(settings, text="Sample every (seconds)").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(settings, textvariable=self.sample_var, width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(settings, text="Lower = more frames analysed").grid(row=0, column=2, sticky="w", padx=15)

        ttk.Label(settings, text="Maximum photos").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(settings, textvariable=self.max_photos_var, width=12).grid(row=1, column=1, sticky="w")
        ttk.Label(settings, text="No artificial video length or GB limit").grid(row=1, column=2, sticky="w", padx=15)

        ttk.Label(settings, text="Duplicate threshold").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Entry(settings, textvariable=self.threshold_var, width=12).grid(row=2, column=1, sticky="w")
        ttk.Label(settings, text="Lower = stricter duplicate removal").grid(row=2, column=2, sticky="w", padx=15)

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(0, 12))
        self.start_button = ttk.Button(actions, text="START ANALYSIS", command=self.start)
        self.start_button.pack(side="left")
        ttk.Button(actions, text="Open Output", command=self.open_output).pack(side="left", padx=8)

        self.progress = ttk.Progressbar(root, variable=self.progress_var, maximum=100, mode="indeterminate")
        self.progress.pack(fill="x", pady=(5, 8))
        ttk.Label(root, textvariable=self.status_var).pack(anchor="w")

        log_frame = ttk.LabelFrame(root, text="Activity", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.m4v *.mts *.m2ts"), ("All files", "*.*")],
        )
        if path:
            self.video_var.set(path)
            self.log_message(f"Selected video: {path}")

    def choose_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    def log_message(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self) -> None:
        video = Path(self.video_var.get().strip())
        output = Path(self.output_var.get().strip())
        try:
            sample = float(self.sample_var.get())
            max_photos = int(self.max_photos_var.get())
            threshold = float(self.threshold_var.get())
            if sample <= 0 or max_photos <= 0 or threshold < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid settings", "Enter valid positive values for sampling and maximum photos, and a non-negative duplicate threshold.")
            return

        if not video.exists() or not video.is_file():
            messagebox.showerror("Video not found", "Please select a valid video file.")
            return

        self.start_button.configure(state="disabled")
        self.progress.start(10)
        self.status_var.set("Analysing video…")
        self.log_message("Starting intelligent analysis…")

        thread = threading.Thread(target=self._run, args=(video, output, sample, max_photos, threshold), daemon=True)
        thread.start()

    def _run(self, video: Path, output: Path, sample: float, max_photos: int, threshold: float) -> None:
        try:
            extractor.require_ffmpeg()
            info = extractor.video_info(video)
            self.after(0, self.log_message, f"Duration: {info['duration']:.2f}s | FPS: {info['fps']:.2f} | Resolution: {info['width']}x{info['height']}")
            self.after(0, self.log_message, f"Sampling every {sample} second(s)…")

            work_dir = output / "_frames"
            selected_dir = output / "selected_photos"
            candidates = extractor.extract_candidates(video, work_dir, sample)
            self.after(0, self.log_message, f"Extracted {len(candidates)} candidate frames.")

            self.after(0, self.log_message, "Running visual intelligence: faces, sharpness and exposure…")
            quality_results = smart_pipeline.analyse_candidates(candidates)
            face_frames = sum(1 for result in quality_results.values() if result.face_count > 0)
            self.after(0, self.log_message, f"Visual analysis complete. Faces detected in {face_frames}/{len(candidates)} frames.")

            selected = smart_pipeline.select_smart(candidates, quality_results, max_photos, threshold)
            smart_pipeline.copy_selected(selected, selected_dir)
            smart_pipeline.write_smart_report(output, selected, quality_results)
            extractor.write_report(output, info, selected)
            self.after(0, self._finished, len(selected), selected_dir, output)
        except Exception as exc:
            self.after(0, self._failed, str(exc))

    def _finished(self, count: int, selected_dir: Path, output: Path) -> None:
        self.progress.stop()
        self.start_button.configure(state="normal")
        self.status_var.set(f"Complete — {count} intelligent selections")
        self.log_message(f"Complete. Photos: {selected_dir}")
        self.log_message(f"Reports: {output / 'report.json'}, {output / 'report.csv'} and {output / 'smart_report.json'}")
        messagebox.showinfo("Analysis complete", f"Selected {count} photos.\n\nOutput: {selected_dir}")

    def _failed(self, error: str) -> None:
        self.progress.stop()
        self.start_button.configure(state="normal")
        self.status_var.set("Error")
        self.log_message(f"ERROR: {error}")
        messagebox.showerror("Extraction failed", error)

    def open_output(self) -> None:
        path = Path(self.output_var.get().strip())
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except AttributeError:
            subprocess.Popen(["xdg-open", str(path)])


if __name__ == "__main__":
    ExtractorApp().mainloop()
