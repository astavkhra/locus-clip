"""
Silence trimmer: cut an optional section out of a video, then remove the
silent gaps from it (jump-cut style). Pure ffmpeg -- no GPU, no extra deps.

Usage:
    python trim.py input.mp4                          # whole video, remove silences
    python trim.py input.mp4 --start 14:12 --end 15:32
    python trim.py input.mp4 --start 14:12 --end 15:32 -o clip.mp4
    python trim.py input.mp4 --dry-run                # just report, don't encode

Time formats accepted: SS, MM:SS, HH:MM:SS (optionally with .decimals).
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Trimmed clips go to the shared output_video/ folder at the workspace root
# (the parent of this trimmer/ directory).
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, "output_video")


def _find_exe(name):
    """Locate ffmpeg/ffprobe even when they aren't on PATH (e.g. under the GUI)."""
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
    return name  # last resort; will raise a clear error if truly missing


FFMPEG = _find_exe("ffmpeg")
FFPROBE = _find_exe("ffprobe")


# --- time helpers -----------------------------------------------------------
def parse_time(text):
    """'14:12' / '1:02:03' / '83.5' -> seconds (float)."""
    text = text.strip()
    if not re.fullmatch(r"[\d:.]+", text):
        raise argparse.ArgumentTypeError(f"bad time: {text!r}")
    parts = text.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError(f"bad time: {text!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def fmt_dur(seconds):
    seconds = max(0, seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h:d}:{m:02d}:{s:05.2f}"
    return f"{m:d}:{s:05.2f}"


# --- ffprobe helpers --------------------------------------------------------
def probe_duration(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", os.path.abspath(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def has_audio(path):
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0",
         os.path.abspath(path)],
        capture_output=True, text=True, check=True,
    )
    return bool(out.stdout.strip())


# --- core steps -------------------------------------------------------------
def extract_section(src, start, end, dst):
    """Accurately cut [start, end] from src into dst (re-encoded)."""
    cmd = [FFMPEG, "-y", "-ss", f"{start:.3f}", "-i", os.path.abspath(src),
           "-t", f"{end - start:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-c:a", "aac", "-b:a", "192k", os.path.abspath(dst)]
    subprocess.run(cmd, check=True, capture_output=True)


_SIL_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SIL_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(path, threshold_db, min_silence):
    """Return list of (start, end) silent intervals via ffmpeg silencedetect."""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostats", "-i", os.path.abspath(path),
         "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    log = proc.stderr
    starts = [float(m) for m in _SIL_START.findall(log)]
    ends = [float(m) for m in _SIL_END.findall(log)]
    silences = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None  # trailing silence to EOF
        silences.append((s, e))
    return silences


def compute_keeps(silences, duration, padding, min_segment):
    """Invert silences -> kept (loud) segments, padded and merged."""
    keeps, cursor = [], 0.0
    for s, e in silences:
        if s > cursor:
            keeps.append((cursor, s))
        cursor = duration if e is None else max(cursor, e)
    if cursor < duration:
        keeps.append((cursor, duration))

    # pad each kept segment, then merge overlaps created by padding
    padded = []
    for a, b in keeps:
        padded.append((max(0.0, a - padding), min(duration, b + padding)))
    merged = []
    for a, b in padded:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= min_segment]


def cut_segments(src, keeps, dst, opts):
    """Keep only `keeps` segments of src, concatenated, into dst."""
    lines = []
    for i, (s, e) in enumerate(keeps):
        lines.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];")
        lines.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}];")
    concat_in = "".join(f"[v{i}][a{i}]" for i in range(len(keeps)))
    lines.append(f"{concat_in}concat=n={len(keeps)}:v=1:a=1[outv][outa]")

    # Write the graph to a script file to dodge Windows command-line length limits.
    fd, script = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    try:
        cmd = [FFMPEG, "-y", "-i", os.path.abspath(src),
               "-filter_complex_script", script,
               "-map", "[outv]", "-map", "[outa]",
               "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf),
               "-c:a", "aac", "-b:a", "192k", os.path.abspath(dst)]
        subprocess.run(cmd, check=True)
    finally:
        os.remove(script)


# --- CLI --------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Trim a section and/or remove silences from a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="input video file")
    p.add_argument("-o", "--output", help="output (default: <input>_trimmed.mp4)")
    p.add_argument("--start", type=parse_time, help="section start (e.g. 14:12)")
    p.add_argument("--end", type=parse_time, help="section end (e.g. 15:32)")

    # silence removal (defaults = 'aggressive')
    p.add_argument("--threshold", type=float, default=-35.0,
                   help="silence level in dB (higher/less-negative = more gets cut)")
    p.add_argument("--min-silence", type=float, default=0.3,
                   help="only cut silent gaps longer than this (seconds)")
    p.add_argument("--padding", type=float, default=0.08,
                   help="keep this much audio around each kept segment (seconds)")
    p.add_argument("--min-segment", type=float, default=0.05,
                   help="drop kept segments shorter than this (seconds)")
    p.add_argument("--keep-silences", action="store_true",
                   help="only cut the section; do NOT remove silences")

    p.add_argument("--preset", default="medium", help="x264 preset")
    p.add_argument("--crf", type=int, default=18, help="x264 quality (lower=better)")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be cut without encoding")

    opts = p.parse_args()

    if not os.path.isfile(opts.input):
        p.error(f"file not found: {opts.input}")

    in_name = os.path.splitext(os.path.basename(opts.input))[0]
    output = opts.output or os.path.join(OUTPUT_DIR, f"{in_name}_trimmed.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    full_dur = probe_duration(opts.input)
    start = opts.start if opts.start is not None else 0.0
    end = opts.end if opts.end is not None else full_dur
    if end > full_dur:
        end = full_dur
    if start < 0 or start >= end:
        p.error(f"invalid section: {fmt_dur(start)} -> {fmt_dur(end)} (video is {fmt_dur(full_dur)})")

    has_section = opts.start is not None or opts.end is not None
    if has_section:
        print(f"Section: {fmt_dur(start)} -> {fmt_dur(end)}  ({fmt_dur(end - start)})")

    tmp = None
    try:
        # 1. Isolate the working clip (the section, or the whole video).
        if has_section:
            fd, tmp = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            print("Extracting section...")
            extract_section(opts.input, start, end, tmp)
            work = tmp
        else:
            work = opts.input

        work_dur = probe_duration(work)

        # 2. Optionally remove silences.
        if opts.keep_silences:
            keeps = [(0.0, work_dur)]
            print("Keeping silences (section-only trim).")
        else:
            if not has_audio(work):
                p.error("no audio stream -> cannot detect silence "
                        "(use --keep-silences to just trim the section)")
            print(f"Detecting silence (noise={opts.threshold}dB, min={opts.min_silence}s)...")
            silences = detect_silences(work, opts.threshold, opts.min_silence)
            keeps = compute_keeps(silences, work_dur, opts.padding, opts.min_segment)
            if not keeps:
                print("Everything was below the silence threshold; nothing to keep.",
                      file=sys.stderr)
                sys.exit(1)

        kept_dur = sum(b - a for a, b in keeps)
        removed = work_dur - kept_dur
        print(f"\n{fmt_dur(work_dur)} -> {fmt_dur(kept_dur)}  "
              f"(removed {fmt_dur(removed)} across {len(keeps) - 1} cuts, "
              f"{len(keeps)} segments kept)")

        if opts.dry_run:
            print("\n[dry run] no file written.")
            return

        # 3. Cut & stitch.
        if len(keeps) == 1 and keeps[0] == (0.0, work_dur) and has_section:
            # Section-only, no silence cutting: the extracted temp IS the result.
            # shutil.move (not os.replace) so it works across drives (temp on C:,
            # output on D:).
            if os.path.exists(output):
                os.remove(output)
            shutil.move(tmp, output)
            tmp = None
        else:
            print("\nCutting and stitching...")
            cut_segments(work, keeps, output, opts)

        print(f"\nDone -> {output}")
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    main()
