# Auto-Subtitle Burner

Transcribes a video with **faster-whisper** (GPU) and burns clean, styled
captions into it with **ffmpeg**. Tuned for an RTX 3050 (4 GB) + English.

## Setup (already done on this machine)

- shared venv at `../.venv` with `faster-whisper`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`
- ffmpeg on PATH
- Default model: `large-v3-turbo` (float16) — best English accuracy/speed for 4 GB VRAM

To recreate elsewhere (from the workspace root):
```
python -m venv .venv
.venv\Scripts\python -m pip install -r subtitler\requirements.txt
```

## Usage

```bat
REM basic (uses Arial, distil-large-v3 on GPU)
subtitle video.mp4

REM custom font + size
subtitle video.mp4 --font "Bebas Neue" --font-size 28 --bold

REM keep the editable .ass subtitle file
subtitle video.mp4 --keep-subs

REM list installed fonts (use the exact name with --font)
subtitle --list-fonts
```

## Workspace layout

```
clipping/
  input_videos/    <- put source videos here
  output_video/    <- burned + trimmed results land here automatically
  subtitler/       <- this tool (subtitle.py, subtitle.bat, README, requirements)
  trimmer/         <- silence-trimming subproject
  .venv/           <- shared Python environment
```

Output goes to `output_video/<name>_subtitled.mp4` by default. Use `-o path.mp4`
to override. With `--keep-subs`, the editable `.ass` is saved next to the output
video in `output_video/`.

```bat
subtitle "input_videos\myclip.mp4"
```

## Common options

| Option | Default | Notes |
|--------|---------|-------|
| `--style` | words | `words` = word-groups; `karaoke` = words highlight as spoken; `segments` = full sentences |
| `--theme` | classic | color preset (see below) |
| `--box` / `--no-box` | per theme | opaque background box behind text |
| `--highlight-color` | per theme | karaoke highlight color `#RRGGBB` |
| `--words-per-caption` | 3 | max words shown at once (words/karaoke) |
| `--max-caption-duration` | 2.5 | max seconds a caption stays up |
| `--max-gap` | 0.7 | split when the pause between words exceeds this (s) |
| `--font` | Arial | family name from `--list-fonts` |
| `--font-size` | 24 | |
| `--color` / `--outline-color` | #FFFFFF / #000000 | `#RRGGBB` |
| `--outline` / `--shadow` | 2.0 / 0.6 | thickness / depth |
| `--position` | bottom | bottom / top / middle |
| `--margin-v` | 90 | distance from edge (720p units); higher = further up |
| `--model` | large-v3-turbo | e.g. `distil-large-v3`, `large-v3` |
| `--compute-type` | float16 | `int8_float16` to save VRAM |
| `--crf` | 18 | lower = higher quality/bigger file |

### Color themes (`--theme`)

`classic` (white text, black outline, no box), `white-box` (white on black box),
`yellow-box`, `black-box` (black text, white box), `red-box`, `blue-box`,
`green-box`, `purple-box`, `pink-box`, `mint`, `neon`, `hormozi` (white text +
green karaoke highlight), `sunset`. Any explicit `--color` / `--outline-color` /
`--highlight-color` / `--box` overrides the theme.

```bat
REM boxed white-on-black, punchy one word at a time
subtitle "clip.mp4" --theme white-box --words-per-caption 1

REM karaoke: words light up as spoken, green highlight
subtitle "clip.mp4" --style karaoke --theme hormozi

REM custom combo: yellow text, blue box
subtitle "clip.mp4" --box --color "#FFE600" --outline-color "#0353A4"
```

### VAD / anti-hallucination

| Option | Default | Notes |
|--------|---------|-------|
| `--vad-threshold` | 0.5 | higher = stricter (drops more non-speech) |
| `--vad-min-silence` | 500 | min silence (ms) that splits speech |
| `--vad-pad` | 200 | padding (ms) kept around speech |
| `--vad-max-speech` | 8.0 | max seconds per segment; shorter = better punctuation/accuracy |
| `--hallucination-silence` | 2.0 | skip silent gaps longer than this (s) |

> **Accuracy tip:** the default `large-v3-turbo` + capped `--vad-max-speech`
> gives the best results. If a clip has music over dialogue, the single biggest
> remaining win is isolating the vocals first (Demucs) — ask to add it.

Runs on GPU by default and falls back to CPU automatically if CUDA is unavailable.
