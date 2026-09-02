#!/usr/bin/env python3
"""Thumbnail-based manual review window for AI-selected photos."""

from __future__ import annotations

import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


class PhotoReview(tk.Toplevel):
    """Let the operator approve/reject selected photos before final delivery."""

    def __init__(self, parent: tk.Misc, photo_dir: Path) -> None:
        super().__init__(parent)
        self.title("Review AI-Selected Photos")
        self.geometry("1100x760")
        self.minsize(850, 600)
        self.photo_dir = photo_dir
        self.files = sorted(photo_dir.glob("*.jpg"))
        self.vars = [tk.BooleanVar(value=True) for _ in self.files]
        self._images: list[ImageTk.PhotoImage] = []
        self._build()

    def _build(self) -> None:
        header = ttk.Frame(self, padding=12)
        header.pack(fill="x")
        self.count_var = tk.StringVar()
        ttk.Label(header, text="AI PHOTO REVIEW", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(header, textvariable=self.count_var).pack(anchor="w", pady=(3, 0))

        actions = ttk.Frame(header)
        actions.pack(anchor="e", pady=(8, 0))
        ttk.Button(actions, text="Select All", command=self.select_all).pack(side="left", padx=4)
        ttk.Button(actions, text="Clear All", command=self.clear_all).pack(side="left", padx=4)
        ttk.Button(actions, text="FINALIZE APPROVED PHOTOS", command=self.finalize).pack(side="left", padx=4)

        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        grid = ttk.Frame(canvas)
        grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=grid, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        columns = 4
        for index, path in enumerate(self.files):
            try:
                image = Image.open(path).convert("RGB")
                image.thumbnail((240, 160), Image.Resampling.LANCZOS)
                thumb = ImageTk.PhotoImage(image)
                self._images.append(thumb)
            except Exception:
                continue

            card = ttk.Frame(grid, padding=8, relief="ridge")
            card.grid(row=index // columns, column=index % columns, sticky="nsew", padx=5, pady=5)
            ttk.Label(card, image=thumb).pack()
            ttk.Checkbutton(card, text=path.name, variable=self.vars[index]).pack(anchor="w", pady=(5, 0))

        self._update_count()
        for column in range(columns):
            grid.columnconfigure(column, weight=1)

    def _update_count(self) -> None:
        approved = sum(var.get() for var in self.vars)
        self.count_var.set(f"{approved} of {len(self.files)} photos currently approved")

    def select_all(self) -> None:
        for var in self.vars:
            var.set(True)
        self._update_count()

    def clear_all(self) -> None:
        for var in self.vars:
            var.set(False)
        self._update_count()

    def finalize(self) -> None:
        approved = [path for path, var in zip(self.files, self.vars) if var.get()]
        if not approved:
            messagebox.showwarning("Nothing selected", "Select at least one photo.")
            return

        final_dir = self.photo_dir.parent / "final_photos"
        final_dir.mkdir(parents=True, exist_ok=True)
        for old in final_dir.glob("*.jpg"):
            old.unlink()

        for index, source in enumerate(approved, start=1):
            shutil.copy2(source, final_dir / f"photo_{index:04d}.jpg")

        messagebox.showinfo("Review complete", f"{len(approved)} photos finalized.\n\n{final_dir}")
        self.destroy()
