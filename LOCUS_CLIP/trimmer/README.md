# Silence Trimmer

Cut an optional section out of a video, then remove the silent gaps from it
(jump-cut style). Pure ffmpeg — no GPU, no extra Python dependencies.

## Usage

```bat
REM remove silences from the whole video
trim video.mp4

REM only a section (14:12 to 15:32), then remove silences within it
trim video.mp4 --start 14:12 --end 15:32

REM see what would be cut, without encoding
trim video.mp4 --start 14:12 --end 15:32 --dry-run

REM just trim the section, keep the silences
trim video.mp4 --start 14:12 --end 15:32 --keep-silences
```

Output goes to the shared `../output_video/<name>_trimmed.mp4` by default
(override with `-o`). Times accept `SS`, `MM:SS`, or `HH:MM:SS` (with optional
decimals).

## How it works

1. If a section is given, extract `[start, end]` accurately (re-encoded).
2. `ffmpeg silencedetect` finds every quiet gap.
3. Invert to the loud "keep" segments, pad each one, merge overlaps.
4. Trim + concat the keep segments into the final clip.

Prints a summary like `1:20.00 -> 0:51.30 (removed 0:28.70 across 22 cuts)`.

## Options

| Option | Default | Notes |
|--------|---------|-------|
| `--start` / `--end` | whole video | section to keep before de-silencing |
| `--threshold` | -35 (dB) | silence level; less-negative cuts more |
| `--min-silence` | 0.3 | only cut gaps longer than this (s) |
| `--padding` | 0.08 | audio kept around each segment (s); raise for smoother cuts |
| `--min-segment` | 0.05 | drop kept segments shorter than this (s) |
| `--keep-silences` | off | section-only trim, don't de-silence |
| `--dry-run` | off | report only, no encoding |
| `--crf` / `--preset` | 18 / medium | output quality/speed |

Defaults are tuned "aggressive" (tight, punchy). For a gentler edit try
`--threshold -27 --min-silence 0.8 --padding 0.2`.

> Note: if a clip has continuous background music, there's no true silence to
> detect, so nothing gets cut — that's expected.
