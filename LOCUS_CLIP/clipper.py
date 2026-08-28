"""
LOCUS_CLIP -- local, batch "Opus-Clips-style" auto-clipper.

Pipeline:  long video
  -> transcribe ONCE with faster-whisper (reusing subtitler/subtitle.py)
  -> Gemini picks the strongest standalone moments (returns segment INDICES,
     never raw timestamps, so it can't hallucinate times)
  -> per clip: cut + reframe to 9:16 + burn captions in a single NVENC pass
  -> a batch of short vertical clips in output_video/<video_name>/

Usage:
    python clipper.py "input_videos/some talk.mp4"
    python clipper.py talk.mp4 --clips 8 --reframe blur --min-score 7
    python clipper.py talk.mp4 --dry-run        # select only, don't render
    python clipper.py talk.mp4 --no-cache       # force re-transcription

Needs GEMINI_API_KEY in LOCUS_CLIP/.env (see .env.example).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output_video")
CACHE_DIR = os.path.join(ROOT, ".cache")

# Reuse the subtitle tool's transcription + caption + styling code.
sys.path.insert(0, os.path.join(ROOT, "subtitler"))
import subtitle as st  # noqa: E402  (transcribe, build_captions, build_ass, THEMES, FFMPEG...)

GEMINI_MODEL = "gemini-3.6-flash"


# --- default options for the reused subtitle.py functions -------------------
def default_opts(theme="classic", style="words", font="Arial", font_size=52):
    """Build an argparse-style namespace with subtitle.py's defaults.

    Font size is bumped from subtitle.py's 24 because captions sit on a tall
    9:16 canvas here; the .ass PlayResY stays 720 so this is in 720p units.
    """
    o = SimpleNamespace(
        # transcription
        model="large-v3-turbo", language="en", device="auto",
        compute_type="float16", beam_size=5,
        vad_threshold=0.5, vad_min_silence=500, vad_pad=200,
        vad_max_speech=8.0, hallucination_silence=2.0,
        # caption grouping
        style=style, words_per_caption=3, max_caption_duration=2.5, max_gap=0.7,
        # styling
        theme=theme, font=font, font_size=font_size, bold=True,
        color=None, outline_color=None, highlight_color=None, box=None,
        outline=0.0, shadow=0.6, position="bottom", margin_v=150, max_chars=28,
        # encode (only used by subtitle.py's burn(); we render ourselves)
        preset="medium", crf=18,
        aspect=9 / 16,
    )
    # Resolve theme colors exactly like subtitle.py's main() does.
    t_text, t_back, t_hl, t_box = st.THEMES[o.theme]
    o.color = o.color or t_text
    o.outline_color = o.outline_color or t_back
    o.highlight_color = o.highlight_color or t_hl
    o.box = t_box if o.box is None else o.box
    return o


# --- transcription (with a JSON cache so re-runs are instant) ---------------
def _cache_path(video_path, model):
    h = hashlib.md5(
        f"{os.path.abspath(video_path)}|{os.path.getmtime(video_path)}|{model}".encode()
    ).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")


def transcribe_cached(video_path, opts, use_cache=True):
    cache = _cache_path(video_path, opts.model)
    if use_cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            raw = json.load(f)
        print(f"Loaded cached transcript ({len(raw)} segments) <- {os.path.basename(cache)}")
        return raw
    raw = st.transcribe(video_path, opts)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)
    print(f"Cached transcript -> {os.path.basename(cache)}")
    return raw


# --- clip selection (the "intelligence": Gemini over the transcript) --------
def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


SYSTEM_INSTRUCTION = """You are a video editor selecting short-form vertical clips from a long video transcript.

You receive a numbered transcript; each line has an index and a time range.
Select the strongest STANDALONE moments to cut into shorts.

Rules:
- Each clip must be SELF-CONTAINED: start at the beginning of a thought and end
  at its natural resolution. Never start mid-sentence or cut off a payoff.
- Target length 20-60 seconds (estimate from the time ranges).
- The FIRST line of a clip should hook the viewer: a question, bold claim,
  surprising statement, or "here's the thing" moment.
- Clips must NOT overlap.
- Return only genuinely strong moments; do NOT pad to a count. If only 3 are
  good, return 3.
- Score each clip 1-10 for how well it would perform as a standalone short.
- start_line and end_line are transcript INDICES. Never invent timestamps."""


def select_clips(raw, opts, max_clips, min_score, min_len, max_len):
    from google import genai
    from google.genai import types
    from pydantic import BaseModel
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("No GEMINI_API_KEY found in LOCUS_CLIP/.env")

    class Clip(BaseModel):
        start_line: int
        end_line: int
        title: str
        reason: str
        score: int

    class ClipSet(BaseModel):
        clips: list[Clip]

    transcript = "\n".join(
        f"[{i}] ({_fmt(s['start'])}-{_fmt(s['end'])}) {s['text']}"
        for i, s in enumerate(raw)
    )
    prompt = (
        f"Transcript ({len(raw)} lines):\n{transcript}\n\n"
        f"Select up to {max_clips} clips."
    )

    client = genai.Client(api_key=key)
    print(f"Asking {GEMINI_MODEL} to pick clips from {len(raw)} segments...")
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ClipSet,
            temperature=0.4,
        ),
    )
    picks = resp.parsed.clips if resp.parsed else []

    # Reconcile INDICES -> real timestamps (every number here comes from whisper).
    n = len(raw)
    accepted = []
    for c in picks:
        if not (0 <= c.start_line <= c.end_line < n):
            continue                                   # malformed / hallucinated index
        start, end = raw[c.start_line]["start"], raw[c.end_line]["end"]
        dur = end - start
        if dur < min_len or dur > max_len:
            continue                                   # length gate
        if c.score < min_score:
            continue                                   # quality gate
        if any(start < a["end"] and end > a["start"] for a in accepted):
            continue                                   # overlap guard
        accepted.append({
            "start": start, "end": end, "dur": dur,
            "title": c.title, "reason": c.reason, "score": c.score,
        })
    accepted.sort(key=lambda a: a["score"], reverse=True)
    return accepted[:max_clips]


# --- render one clip: cut + reframe + captions in a single ffmpeg pass ------
def _clip_captions(captions, start, end):
    """Filter captions to [start,end] and rebase their times to clip-zero."""
    out = []
    for c in captions:
        if c["end"] <= start or c["start"] >= end:
            continue
        nc = {
            "start": max(0.0, c["start"] - start),
            "end": min(end, c["end"]) - start,
            "text": c["text"],
        }
        if c.get("words"):
            nc["words"] = [
                {"word": w["word"],
                 "start": max(0.0, w["start"] - start),
                 "end": max(0.0, w["end"] - start)}
                for w in c["words"]
            ]
        out.append(nc)
    return out


def _reframe_chain(mode, ass_name, fontsdir):
    """filter_complex producing [v]: reframe (crop|blur|none) then burn subs."""
    subs = f"subtitles={ass_name}:fontsdir='{fontsdir}'"
    if mode == "crop":
        return f"[0:v]crop=ih*9/16:ih,scale=1080:1920,setsar=1,{subs}[v]"
    if mode == "blur":
        return (
            "[0:v]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,gblur=sigma=20[bgb];"
            "[fg]scale=1080:-2:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,{subs}[v]"
        )
    return f"[0:v]{subs}[v]"   # none: keep source frame, just burn subs


def render_clip(video, clip, captions, opts, out_path, reframe, use_nvenc=True):
    caps = _clip_captions(captions, clip["start"], clip["end"])
    # 9:16 output -> tell build_ass the canvas is vertical so text isn't stretched.
    opts.aspect = 9 / 16 if reframe in ("crop", "blur") else opts.aspect

    ass_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(ass_dir, exist_ok=True)
    fd, ass_path = tempfile.mkstemp(suffix=".ass", dir=ass_dir, text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(st.build_ass(caps, opts))
    ass_name = os.path.basename(ass_path)
    fontsdir = st.WINDOWS_FONTS_DIR.replace("\\", "/").replace(":", r"\:")

    if use_nvenc:
        venc = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    else:
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]

    cmd = [
        st.FFMPEG, "-y",
        "-ss", f"{clip['start']:.3f}", "-i", os.path.abspath(video),
        "-t", f"{clip['dur']:.3f}",
        "-filter_complex", _reframe_chain(reframe, ass_name, fontsdir),
        "-map", "[v]", "-map", "0:a?",
        *venc, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        os.path.abspath(out_path),
    ]
    try:
        subprocess.run(cmd, cwd=ass_dir, check=True, capture_output=True, text=True)
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)


def nvenc_available():
    """True if h264_nvenc can actually open (driver new enough), not just listed."""
    try:
        subprocess.run(
            [st.FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc=size=256x256:duration=0.1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            check=True, capture_output=True, text=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def _safe(name):
    keep = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip()
    return (keep or "clip")[:50]


# --- CLI --------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Auto-cut a long video into vertical shorts.")
    p.add_argument("input", help="input video file")
    p.add_argument("--clips", type=int, default=10, help="max clips to produce")
    p.add_argument("--reframe", choices=["blur", "crop", "none"], default="blur")
    p.add_argument("--min-score", type=int, default=7, help="drop clips below this score (1-10)")
    p.add_argument("--min-len", type=float, default=12.0, help="drop clips shorter than this (s)")
    p.add_argument("--max-len", type=float, default=75.0, help="drop clips longer than this (s)")
    p.add_argument("--theme", default="classic", choices=sorted(st.THEMES))
    p.add_argument("--style", default="words", choices=["words", "karaoke", "segments"])
    p.add_argument("--font", default="Arial")
    p.add_argument("--no-cache", action="store_true", help="force re-transcription")
    p.add_argument("--no-nvenc", action="store_true", help="use CPU x264 instead of NVENC")
    p.add_argument("--dry-run", action="store_true", help="select clips but don't render")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        p.error(f"file not found: {args.input}")

    opts = default_opts(theme=args.theme, style=args.style, font=args.font)

    # 1. Transcribe once (cached).
    raw = transcribe_cached(args.input, opts, use_cache=not args.no_cache)
    if not raw:
        sys.exit("No speech transcribed; nothing to clip.")

    # 2. Select clips (Gemini -> reconciled timestamps).
    clips = select_clips(raw, opts, args.clips, args.min_score, args.min_len, args.max_len)
    if not clips:
        sys.exit("Gemini returned no clips passing the length/score gates.")

    print(f"\n=== {len(clips)} clip(s) selected ===")
    for i, c in enumerate(clips, 1):
        print(f"{i:2d}. [{_fmt(c['start'])}-{_fmt(c['end'])}] {c['dur']:4.0f}s "
              f"score={c['score']}  {c['title']}")
        print(f"      {c['reason']}")

    if args.dry_run:
        print("\n[dry run] no files rendered.")
        return

    # 3. Build the full caption list once, then render each clip.
    captions = st.build_captions(raw, opts)
    base = _safe(os.path.splitext(os.path.basename(args.input))[0])
    out_dir = os.path.join(OUTPUT_DIR, base)
    os.makedirs(out_dir, exist_ok=True)

    # Pick the encoder: honor --no-nvenc, else use NVENC only if it can truly open
    # (an out-of-date GPU driver blocks it, so we probe rather than assume).
    use_nvenc = not args.no_nvenc and nvenc_available()
    if not args.no_nvenc and not use_nvenc:
        print("  (NVENC unavailable -- likely an out-of-date GPU driver; using CPU x264)")

    print(f"\nRendering to {out_dir} ({'NVENC' if use_nvenc else 'x264'}, reframe={args.reframe})...")
    for i, c in enumerate(clips, 1):
        out_path = os.path.join(out_dir, f"clip{i:02d}_{_safe(c['title'])}.mp4")
        print(f"  [{i}/{len(clips)}] {_fmt(c['start'])}-{_fmt(c['end'])} -> {os.path.basename(out_path)}")
        try:
            render_clip(args.input, c, captions, opts, out_path,
                        args.reframe, use_nvenc=use_nvenc)
        except subprocess.CalledProcessError as e:
            print(f"      FFMPEG FAILED:\n{(e.stderr or '')[-800:]}")
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
