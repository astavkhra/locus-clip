# Clipping workspace

Two tools for turning raw footage into captioned clips, plus a GUI that chains them.

## Quick start (GUI)

Double-click **`run_gui.bat`** (or run `.venv\Scripts\pythonw gui.py`).

**Clipping Studio** lets you, in one window:
1. Pick an input video.
2. Optionally keep a section (`start`–`end`) and remove its silences.
3. Choose caption style (karaoke / words / sentences), font, size, bold, box.
4. Place captions: Middle, Between middle & lower, Lower, or a custom
   pixels-from-bottom value.

It runs the trim, then the captioning, and drops the result in `output_video/`.
Use **Clear output** to wipe `output_video/` (with confirmation) once you've
saved/uploaded your clips, so results don't pile up.

## Layout

```
clipping/
  input_videos/    <- source videos
  output_video/    <- all results (trimmed / subtitled) land here
  subtitler/       <- caption tool (subtitle.py)  - see its README
  trimmer/         <- silence-trim tool (trim.py) - see its README
  gui.py, run_gui.bat
  .venv/           <- shared Python environment (faster-whisper + CUDA)
```

## Command line

The GUI is optional — each tool works on its own:

```bat
trimmer\trim.bat "input_videos\talk.mp4" --start 0:40 --end 2:40
subtitler\subtitle.bat "output_video\talk_trimmed.mp4" --style karaoke --font Satoshi
```

See `subtitler/README.md` and `trimmer/README.md` for all options.
