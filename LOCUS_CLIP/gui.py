"""
Clipping Studio -- a small GUI front-end for the workspace.

Pick a video, (optionally) keep a section and strip its silences, then burn
styled captions -- all in one click. It drives the existing command-line tools
(trimmer/trim.py and subtitler/subtitle.py) as subprocesses.

Run it with the project venv:  run_gui.bat   (or  .venv\\Scripts\\pythonw gui.py)
"""

import glob
import os
import shutil
import subprocess
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

ROOT = os.path.dirname(os.path.abspath(__file__))
SUBTITLE_PY = os.path.join(ROOT, "subtitler", "subtitle.py")
TRIM_PY = os.path.join(ROOT, "trimmer", "trim.py")
OUTPUT_DIR = os.path.join(ROOT, "output_video")

# Always drive the tools with the project venv's Python (has faster-whisper).
# The venv lives at the repo root (one level up from this LOCUS_CLIP folder);
# fall back to a venv alongside this file, then to the current interpreter.
_venv_candidates = (
    os.path.join(os.path.dirname(ROOT), ".venv", "Scripts", "python.exe"),
    os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
)
PYTHON = next((p for p in _venv_candidates if os.path.exists(p)), sys.executable)

# Don't pop up console windows for child processes when launched via pythonw.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _find_exe(name):
    """Locate ffprobe even when it isn't on PATH (the GUI process often lacks it)."""
    found = shutil.which(name)
    if found:
        return found
    for base in (os.environ.get("LOCALAPPDATA", ""), os.environ.get("ProgramFiles", "")):
        if not base:
            continue
        hits = glob.glob(os.path.join(base, "Microsoft", "WinGet", "Packages",
                                      "Gyan.FFmpeg*", "**", f"{name}.exe"), recursive=True)
        if hits:
            return hits[0]
    return name


FFPROBE = _find_exe("ffprobe")

# Silence-removal presets -> (threshold dB, min-silence s, padding s)
AGGRO = {
    "Aggressive": ("-35", "0.3", "0.08"),
    "Balanced":   ("-30", "0.5", "0.15"),
    "Gentle":     ("-27", "0.8", "0.2"),
}

STYLES = ["karaoke", "words", "segments"]
PLACEMENTS = ["Middle", "Between middle & lower", "Lower", "Custom (px from bottom)"]


# --- pure helpers (unit-testable, no GUI) -----------------------------------
def trimmed_path(input_path):
    name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(OUTPUT_DIR, f"{name}_trimmed.mp4")


def subtitled_path(input_path):
    name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(OUTPUT_DIR, f"{name}_subtitled.mp4")


def build_trim_cmd(inp, start, end, remove_silences, aggro):
    cmd = [PYTHON, TRIM_PY, inp]
    if start:
        cmd += ["--start", start]
    if end:
        cmd += ["--end", end]
    if remove_silences:
        thr, ms, pad = AGGRO[aggro]
        cmd += ["--threshold", thr, "--min-silence", ms, "--padding", pad]
    else:
        cmd += ["--keep-silences"]
    return cmd


def build_subtitle_cmd(inp, style, font, size, bold, box, placement, custom_px, height):
    cmd = [PYTHON, SUBTITLE_PY, inp, "--style", style, "--theme", "white-box",
           "--font", font, "--font-size", str(size)]
    cmd += ["--bold"] if bold else []
    cmd += ["--box"] if box else ["--no-box"]
    if placement == "Middle":
        cmd += ["--position", "middle"]
    elif placement == "Lower":
        cmd += ["--position", "bottom", "--margin-v", "70"]
    elif placement.startswith("Between"):
        cmd += ["--position", "bottom", "--margin-v", "240"]
    else:  # Custom px from bottom -> convert to the tool's 720-tall coord space
        mv = max(0, round(int(custom_px) * 720 / max(1, height)))
        cmd += ["--position", "bottom", "--margin-v", str(mv)]
    return cmd


def probe_height(path):
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True, creationflags=_NO_WINDOW,
        )
        return int(out.stdout.strip())
    except Exception:
        return 1080


def list_fonts():
    """System fonts (via subtitle.py) with our per-user fonts pinned on top.

    Per-user installs (Satoshi, TeXGyreHeros) live in HKCU and don't show up in
    subtitle.py --list-fonts (which reads HKLM), so pin them manually.
    """
    fonts = []
    try:
        out = subprocess.run(
            [PYTHON, SUBTITLE_PY, "--list-fonts"],
            capture_output=True, encoding="utf-8", errors="replace",
            check=True, creationflags=_NO_WINDOW,
        )
        fonts = [f for f in out.stdout.splitlines() if f.strip()]
    except Exception:
        fonts = ["Arial", "Impact", "Bebas Neue"]
    pinned = ["Satoshi", "TeXGyreHeros"]  # TeXGyreHeros = free Helvetica clone
    ordered = pinned + [f for f in fonts if f not in pinned]
    return ordered


# --- GUI --------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Clipping Studio")
        root.geometry("640x760")
        self.q = queue.Queue()
        self.running = False

        pad = {"padx": 8, "pady": 4}
        main = ttk.Frame(root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # --- Input ---
        f_in = ttk.LabelFrame(main, text="1. Input video", padding=8)
        f_in.grid(row=0, column=0, sticky="ew", pady=6)
        f_in.columnconfigure(0, weight=1)
        self.input_var = tk.StringVar()
        ttk.Entry(f_in, textvariable=self.input_var).grid(row=0, column=0, sticky="ew", **pad)
        ttk.Button(f_in, text="Browse...", command=self.browse).grid(row=0, column=1, **pad)

        # --- Trim ---
        f_trim = ttk.LabelFrame(main, text="2. Keep a section (optional) + remove silences", padding=8)
        f_trim.grid(row=1, column=0, sticky="ew", pady=6)
        ttk.Label(f_trim, text="Start").grid(row=0, column=0, **pad)
        self.start_var = tk.StringVar()
        ttk.Entry(f_trim, textvariable=self.start_var, width=12).grid(row=0, column=1, **pad)
        ttk.Label(f_trim, text="End").grid(row=0, column=2, **pad)
        self.end_var = tk.StringVar()
        ttk.Entry(f_trim, textvariable=self.end_var, width=12).grid(row=0, column=3, **pad)
        ttk.Label(f_trim, text="(mm:ss, e.g. 0:40 - blank = whole video)").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=8)
        self.remove_sil_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_trim, text="Remove silences", variable=self.remove_sil_var).grid(
            row=2, column=0, columnspan=2, sticky="w", **pad)
        ttk.Label(f_trim, text="Tightness").grid(row=2, column=2, **pad)
        self.aggro_var = tk.StringVar(value="Aggressive")
        ttk.Combobox(f_trim, textvariable=self.aggro_var, values=list(AGGRO),
                     state="readonly", width=12).grid(row=2, column=3, **pad)

        # --- Captions ---
        f_cap = ttk.LabelFrame(main, text="3. Caption style", padding=8)
        f_cap.grid(row=2, column=0, sticky="ew", pady=6)
        f_cap.columnconfigure(1, weight=1)
        ttk.Label(f_cap, text="Mode").grid(row=0, column=0, sticky="w", **pad)
        self.style_var = tk.StringVar(value="karaoke")
        ttk.Combobox(f_cap, textvariable=self.style_var, values=STYLES,
                     state="readonly", width=14).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(f_cap, text="Font").grid(row=1, column=0, sticky="w", **pad)
        self.font_var = tk.StringVar(value="Satoshi")
        self.font_box = ttk.Combobox(f_cap, textvariable=self.font_var, values=["Satoshi"], width=28)
        self.font_box.grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(f_cap, text="Size").grid(row=2, column=0, sticky="w", **pad)
        self.size_var = tk.IntVar(value=48)
        ttk.Spinbox(f_cap, from_=10, to=120, textvariable=self.size_var, width=6).grid(
            row=2, column=1, sticky="w", **pad)
        self.bold_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_cap, text="Bold", variable=self.bold_var).grid(row=3, column=0, sticky="w", **pad)
        self.box_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f_cap, text="Background box", variable=self.box_var).grid(
            row=3, column=1, sticky="w", **pad)

        # --- Placement ---
        f_pos = ttk.LabelFrame(main, text="4. Placement", padding=8)
        f_pos.grid(row=3, column=0, sticky="ew", pady=6)
        self.place_var = tk.StringVar(value="Between middle & lower")
        for i, name in enumerate(PLACEMENTS):
            ttk.Radiobutton(f_pos, text=name, variable=self.place_var, value=name,
                            command=self._toggle_custom).grid(row=i, column=0, sticky="w", padx=8, pady=2)
        self.custom_px_var = tk.IntVar(value=200)
        self.custom_entry = ttk.Spinbox(f_pos, from_=0, to=2000, textvariable=self.custom_px_var, width=8)
        self.custom_entry.grid(row=3, column=1, sticky="w", padx=8)
        ttk.Label(f_pos, text="px from bottom").grid(row=3, column=2, sticky="w")
        self._toggle_custom()

        # --- Run + log ---
        bar = ttk.Frame(main)
        bar.grid(row=4, column=0, sticky="ew", pady=8)
        bar.columnconfigure(1, weight=1)
        self.run_btn = ttk.Button(bar, text="Generate", command=self.on_generate)
        self.run_btn.grid(row=0, column=0, padx=4)
        self.status = ttk.Label(bar, text="Ready.")
        self.status.grid(row=0, column=1, sticky="w", padx=8)
        self.clear_btn = ttk.Button(bar, text="Clear output", command=self.clear_output)
        self.clear_btn.grid(row=0, column=2, padx=4)
        self.open_btn = ttk.Button(bar, text="Open output folder",
                                   command=lambda: os.startfile(OUTPUT_DIR))
        self.open_btn.grid(row=0, column=3, padx=4)

        self.log = scrolledtext.ScrolledText(main, height=12, wrap="word", state="disabled")
        self.log.grid(row=5, column=0, sticky="nsew", pady=6)
        main.rowconfigure(5, weight=1)

        # populate fonts in the background so startup stays snappy
        threading.Thread(target=self._load_fonts, daemon=True).start()
        self.root.after(100, self._poll)

    # --- small UI callbacks ---
    def _toggle_custom(self):
        state = "normal" if self.place_var.get().startswith("Custom") else "disabled"
        self.custom_entry.configure(state=state)

    def _load_fonts(self):
        fonts = list_fonts()
        self.q.put(("fonts", fonts))

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            initialdir=os.path.join(ROOT, "input_videos"),
            filetypes=[("Video", "*.mp4 *.mov *.mkv *.webm *.m4v *.avi"), ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)

    def _log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    # --- run pipeline ---
    def on_generate(self):
        if self.running:
            return
        inp = self.input_var.get().strip().strip('"')
        if not inp or not os.path.isfile(inp):
            messagebox.showerror("Clipping Studio", "Please choose a valid input video.")
            return
        if self.place_var.get().startswith("Custom"):
            try:
                int(self.custom_px_var.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Clipping Studio", "Custom placement needs a number of pixels.")
                return

        opts = dict(
            inp=inp,
            start=self.start_var.get().strip(),
            end=self.end_var.get().strip(),
            remove_sil=self.remove_sil_var.get(),
            aggro=self.aggro_var.get(),
            style=self.style_var.get(),
            font=self.font_var.get().strip() or "Satoshi",
            size=int(self.size_var.get()),
            bold=self.bold_var.get(),
            box=self.box_var.get(),
            placement=self.place_var.get(),
            custom_px=self.custom_px_var.get(),
        )
        self.running = True
        self.run_btn.configure(state="disabled")
        self.clear_btn.configure(state="disabled")
        self.status.configure(text="Working...")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        threading.Thread(target=self._worker, args=(opts,), daemon=True).start()

    def _run_step(self, label, cmd):
        self.q.put(("log", f"\n=== {label} ===\n$ " + subprocess.list2cmdline(cmd)))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace", bufsize=1,
                                creationflags=_NO_WINDOW)
        for line in proc.stdout:
            self.q.put(("log", line.rstrip()))
        proc.wait()
        return proc.returncode

    def _worker(self, o):
        try:
            need_trim = bool(o["start"] or o["end"] or o["remove_sil"])
            sub_input = o["inp"]

            if need_trim:
                self.q.put(("status", "Trimming / removing silences..."))
                code = self._run_step("Trim + silence removal",
                                       build_trim_cmd(o["inp"], o["start"], o["end"],
                                                      o["remove_sil"], o["aggro"]))
                if code != 0:
                    self.q.put(("error", "Trimming failed (see log)."))
                    return
                sub_input = trimmed_path(o["inp"])
                if not os.path.isfile(sub_input):
                    self.q.put(("error", "Trimmed file was not produced."))
                    return

            self.q.put(("status", "Transcribing + burning captions (this can take a bit)..."))
            height = probe_height(sub_input)
            code = self._run_step("Captioning",
                                   build_subtitle_cmd(sub_input, o["style"], o["font"], o["size"],
                                                      o["bold"], o["box"], o["placement"],
                                                      o["custom_px"], height))
            if code != 0:
                self.q.put(("error", "Captioning failed (see log)."))
                return

            final = subtitled_path(sub_input)
            self.q.put(("done", final))
        except Exception as e:  # noqa: BLE001 - surface anything to the log
            self.q.put(("error", f"{type(e).__name__}: {e}"))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "fonts":
                    self.font_box.configure(values=payload)
                elif kind == "error":
                    self._finish(f"Error: {payload}")
                    messagebox.showerror("Clipping Studio", payload)
                elif kind == "done":
                    self._log(f"\nDone -> {payload}")
                    self._finish("Done.")
                    self.open_btn.configure(state="normal")
                    messagebox.showinfo("Clipping Studio", f"Finished!\n\n{payload}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finish(self, status):
        self.running = False
        self.run_btn.configure(state="normal")
        self.clear_btn.configure(state="normal")
        self.status.configure(text=status)

    def clear_output(self):
        if self.running:
            return
        try:
            files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)
                     if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
        except FileNotFoundError:
            files = []
        if not files:
            messagebox.showinfo("Clipping Studio", "Output folder is already empty.")
            return
        total_mb = sum(os.path.getsize(p) for p in files) / (1024 * 1024)
        if not messagebox.askyesno(
                "Clear output folder",
                f"Permanently delete {len(files)} file(s) ({total_mb:.1f} MB) "
                f"from output_video?\n\nThis cannot be undone."):
            return
        removed, freed, failed = 0, 0, []
        for p in files:
            try:
                sz = os.path.getsize(p)
                os.remove(p)
                removed += 1
                freed += sz
            except OSError as e:
                failed.append((os.path.basename(p), e))
        msg = f"Cleared output: removed {removed} file(s), freed {freed / 1024 / 1024:.1f} MB."
        self._log(msg)
        self.status.configure(text=msg)
        if failed:
            names = "\n".join(f"- {n}: {getattr(e, 'strerror', e)}" for n, e in failed)
            messagebox.showwarning(
                "Clipping Studio",
                f"Couldn't delete {len(failed)} file(s) — likely open in a player:\n\n{names}")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
