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

# Reuse the subtitle tool's transcription + caption + styling code, and the
# trimmer's silence detection.
sys.path.insert(0, os.path.join(ROOT, "subtitler"))
sys.path.insert(0, os.path.join(ROOT, "trimmer"))
import subtitle as st  # noqa: E402  (transcribe, build_captions, build_ass, THEMES, FFMPEG...)
import trim as tr      # noqa: E402  (detect_silences, compute_keeps)
import autoframe       # noqa: E402  (build_track, write_sendcmd -- face-tracking reframe)

GEMINI_MODEL = "gemini-3.6-flash"
YUNET_MODEL = os.path.join(ROOT, "models", "face_detection_yunet.onnx")

# Silence-trim tuning (mirrors trim.py's defaults) for --tighten.
SIL_PADDING = 0.08      # keep this much audio around each kept segment (s)
SIL_MIN_SEGMENT = 0.05  # drop kept segments shorter than this (s)


# --- default options for the reused subtitle.py functions -------------------
def default_opts(theme="classic", style="words", font="Arial", font_size=52, margin_v=150):
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
        outline=0.0, shadow=0.6, position="bottom", margin_v=margin_v, max_chars=28,
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


def select_clips(raw, opts, max_clips, min_score, min_len, max_len, focus=None):
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

    system = SYSTEM_INSTRUCTION
    if focus:
        # Layer the user's angle on top of the structural rules, which still hold.
        system += (f"\n\nADDITIONAL FOCUS -- prioritize clips matching this intent, "
                   f"and score them by how well they fit it: {focus}")

    client = genai.Client(api_key=key)
    print(f"Asking {GEMINI_MODEL} to pick clips from {len(raw)} segments"
          + (f" (focus: {focus})" if focus else "") + "...")
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
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


def _reframe_chain(mode, ass_name, fontsdir, src="0:v"):
    """filter fragment producing [v]: reframe (crop|blur|none) then burn subs,
    reading from pad `src` (0:v for a plain clip, or the concat output vc)."""
    subs = f"subtitles={ass_name}:fontsdir='{fontsdir}'"
    if mode == "crop":
        return f"[{src}]crop=ih*9/16:ih,scale=1080:1920,setsar=1,{subs}[v]"
    if mode == "blur":
        return (
            f"[{src}]split=2[bg][fg];"
            "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,gblur=sigma=20[bgb];"
            "[fg]scale=1080:-2:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,{subs}[v]"
        )
    return f"[{src}]{subs}[v]"   # none: keep source frame, just burn subs


# --- silence tightening (jump-cut) -----------------------------------------
def keeps_for_clip(all_silences, start, end):
    """Loud sub-segments to keep within source window [start,end], returned in
    CLIP-RELATIVE time (0..dur). Reuses trim.py's compute_keeps (padding+merge)."""
    dur = end - start
    rel = []
    for s, e in all_silences:
        e = end if e is None else e            # trailing silence runs to clip end
        s2, e2 = max(s, start), min(e, end)
        if e2 > s2:
            rel.append((s2 - start, e2 - start))
    return tr.compute_keeps(rel, dur, SIL_PADDING, SIL_MIN_SEGMENT)


def _make_remap(keeps):
    """Return f(clip_time) -> tightened_time: total kept content lying before t.
    Monotonic; a time inside a removed gap maps to the end of the prior keep."""
    def remap(t):
        out = 0.0
        for a, b in keeps:
            if t >= b:
                out += b - a
            elif t <= a:
                break
            else:
                out += t - a
                break
        return out
    return remap


def _tighten_captions(captions, start, end, keeps):
    """Rebase captions to clip-zero, then remap through the kept segments so
    they stay in sync after the silent gaps are removed."""
    remap = _make_remap(keeps)
    out = []
    for c in captions:
        if c["end"] <= start or c["start"] >= end:
            continue
        ns = remap(max(0.0, c["start"] - start))
        ne = remap(min(end, c["end"]) - start)
        if ne - ns < 0.05:                     # fell into a removed gap
            continue
        nc = {"start": ns, "end": ne, "text": c["text"]}
        if c.get("words"):
            nc["words"] = [
                {"word": w["word"],
                 "start": remap(max(0.0, w["start"] - start)),
                 "end": remap(max(0.0, w["end"] - start))}
                for w in c["words"]
            ]
        out.append(nc)
    return out


def render_clip(video, clip, captions, opts, out_path, reframe, use_nvenc=True,
                keeps=None, src_w=1920, src_h=1080):
    """Render one clip. If `keeps` is given (clip-relative loud segments), the
    silent gaps are jump-cut out and captions are remapped to match."""
    # 9:16 output -> tell build_ass the canvas is vertical so text isn't stretched.
    opts.aspect = 9 / 16 if reframe in ("crop", "blur") else opts.aspect

    if keeps is not None:
        caps = _tighten_captions(captions, clip["start"], clip["end"], keeps)
    else:
        caps = _clip_captions(captions, clip["start"], clip["end"])

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

    # Build the filtergraph. Without tightening: reframe straight off the clip.
    # With tightening: trim+concat the kept segments, then reframe the result.
    script_path = cmd_path = None
    if keeps is not None:
        parts = []
        for i, (a, b) in enumerate(keeps):
            parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}];")
            parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}];")
        concat_in = "".join(f"[v{i}][a{i}]" for i in range(len(keeps)))
        parts.append(f"{concat_in}concat=n={len(keeps)}:v=1:a=1[vc][ac];")
        parts.append(_reframe_chain(reframe, ass_name, fontsdir, src="vc"))
        graph = "\n".join(parts)
        audio_map = "[ac]"
    elif reframe == "track":
        # Face-follow: plan a moving crop, then drive it with a sendcmd script.
        plan = autoframe.build_track(st.FFMPEG, video, clip["start"], clip["dur"],
                                     src_w, src_h, YUNET_MODEL)
        subs = f"subtitles={ass_name}:fontsdir='{fontsdir}'"
        if plan is None:
            print("      (no face detected; falling back to center crop)")
            graph = _reframe_chain("crop", ass_name, fontsdir)
        else:
            cfd, cmd_path = tempfile.mkstemp(suffix=".cmd", dir=ass_dir, text=True)
            os.close(cfd)
            autoframe.write_sendcmd(plan["commands"], cmd_path)
            cw, ch, x0 = plan["crop_w"], plan["crop_h"], plan["commands"][0][1]
            graph = (f"[0:v]sendcmd=f='{os.path.basename(cmd_path)}',"
                     f"crop={cw}:{ch}:{x0}:0,scale=1080:1920,setsar=1,{subs}[v]")
            print(f"      tracked: face in {plan['n_hits']}/{plan['n_samples']} "
                  f"sampled frames, {plan['n_cuts']} scene-cut(s)")
        audio_map = "0:a?"
    else:
        graph = _reframe_chain(reframe, ass_name, fontsdir)
        audio_map = "0:a?"

    # A long tighten graph can blow the Windows command-line limit, so pass it
    # via a script file (same trick trim.py uses).
    if keeps is not None:
        sfd, script_path = tempfile.mkstemp(suffix=".txt", dir=ass_dir, text=True)
        with os.fdopen(sfd, "w", encoding="utf-8") as f:
            f.write(graph)
        filter_args = ["-filter_complex_script", os.path.basename(script_path)]
    else:
        filter_args = ["-filter_complex", graph]

    cmd = [
        st.FFMPEG, "-y",
        "-ss", f"{clip['start']:.3f}", "-i", os.path.abspath(video),
        "-t", f"{clip['dur']:.3f}",
        *filter_args,
        "-map", "[v]", "-map", audio_map,
        *venc, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        os.path.abspath(out_path),
    ]
    try:
        subprocess.run(cmd, cwd=ass_dir, check=True, capture_output=True, text=True)
    finally:
        for p in (ass_path, script_path, cmd_path):
            if p and os.path.exists(p):
                os.remove(p)


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
    p.add_argument("--reframe", choices=["blur", "crop", "none", "track"], default="blur",
                   help="'track' follows the speaker's face with a moving 9:16 crop")
    p.add_argument("--min-score", type=int, default=7, help="drop clips below this score (1-10)")
    p.add_argument("--min-len", type=float, default=12.0, help="drop clips shorter than this (s)")
    p.add_argument("--max-len", type=float, default=75.0, help="drop clips longer than this (s)")
    p.add_argument("--theme", default="classic", choices=sorted(st.THEMES))
    p.add_argument("--style", default="words", choices=["words", "karaoke", "segments"])
    p.add_argument("--font", default="Arial")
    p.add_argument("--font-size", type=int, default=52,
                   help="caption size in 720p units (scales ~2.67x on the 1920-tall output)")
    p.add_argument("--margin-v", type=int, default=150,
                   help="caption distance from the bottom edge, in 720p units")
    p.add_argument("--focus", default=None,
                   help="steer selection toward an angle/audience, e.g. "
                        "\"controversial takes for crypto skeptics\"")
    p.add_argument("--tighten", action="store_true",
                   help="jump-cut silent gaps out of each clip (captions stay synced)")
    p.add_argument("--silence-threshold", type=float, default=-35.0,
                   help="silence level in dB for --tighten (higher = cuts more)")
    p.add_argument("--min-silence", type=float, default=0.3,
                   help="only cut silent gaps longer than this for --tighten (s)")
    p.add_argument("--no-cache", action="store_true", help="force re-transcription")
    p.add_argument("--no-nvenc", action="store_true", help="use CPU x264 instead of NVENC")
    p.add_argument("--dry-run", action="store_true", help="select clips but don't render")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        p.error(f"file not found: {args.input}")

    # v1: face-tracking doesn't yet support the tightened (non-linear) timeline.
    if args.reframe == "track" and args.tighten:
        print("  (note: --tighten isn't supported with --reframe track yet; ignoring --tighten)")
        args.tighten = False

    opts = default_opts(theme=args.theme, style=args.style, font=args.font,
                        font_size=args.font_size, margin_v=args.margin_v)

    # 1. Transcribe once (cached).
    raw = transcribe_cached(args.input, opts, use_cache=not args.no_cache)
    if not raw:
        sys.exit("No speech transcribed; nothing to clip.")

    # 2. Select clips (Gemini -> reconciled timestamps).
    clips = select_clips(raw, opts, args.clips, args.min_score, args.min_len, args.max_len,
                         focus=args.focus)
    if not clips:
        sys.exit("Gemini returned no clips passing the length/score gates.")

    # Optional: detect silences once on the source, then compute per-clip keeps.
    if args.tighten:
        print("Detecting silence for --tighten...")
        all_sil = tr.detect_silences(args.input, args.silence_threshold, args.min_silence)
        for c in clips:
            c["keeps"] = keeps_for_clip(all_sil, c["start"], c["end"])
            c["tight_dur"] = sum(b - a for a, b in c["keeps"])

    print(f"\n=== {len(clips)} clip(s) selected ===")
    for i, c in enumerate(clips, 1):
        if args.tighten:
            saved = c["dur"] - c["tight_dur"]
            dur_str = f"{c['dur']:4.0f}s -> {c['tight_dur']:4.0f}s (-{saved:.0f}s)"
        else:
            dur_str = f"{c['dur']:4.0f}s"
        print(f"{i:2d}. [{_fmt(c['start'])}-{_fmt(c['end'])}] {dur_str} "
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

    src_w, src_h = st.probe_video_size(args.input)   # needed for track-crop math

    tstr = ", tighten" if args.tighten else ""
    print(f"\nRendering to {out_dir} ({'NVENC' if use_nvenc else 'x264'}, reframe={args.reframe}{tstr})...")
    for i, c in enumerate(clips, 1):
        out_path = os.path.join(out_dir, f"clip{i:02d}_{_safe(c['title'])}.mp4")
        print(f"  [{i}/{len(clips)}] {_fmt(c['start'])}-{_fmt(c['end'])} -> {os.path.basename(out_path)}")
        try:
            render_clip(args.input, c, captions, opts, out_path,
                        args.reframe, use_nvenc=use_nvenc,
                        keeps=c.get("keeps") if args.tighten else None,
                        src_w=src_w, src_h=src_h)
        except subprocess.CalledProcessError as e:
            print(f"      FFMPEG FAILED:\n{(e.stderr or '')[-800:]}")
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
