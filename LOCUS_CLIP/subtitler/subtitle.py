"""
Auto-subtitle burner: transcribe a video with faster-whisper (GPU) and burn
clean, styled captions into the video with ffmpeg.

Tuned for: NVIDIA RTX 3050 (4 GB) + English via distil-large-v3.

Usage:
    python subtitle.py input.mp4
    python subtitle.py input.mp4 --font "Bebas Neue" --font-size 26
    python subtitle.py input.mp4 -o out.mp4 --keep-subs
    python subtitle.py --list-fonts
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import textwrap

# Windows consoles default to cp1252 and crash when printing non-Latin-1 chars
# (e.g. a full-width | in a filename). Force UTF-8 output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# HuggingFace caching uses symlinks by default; on Windows without admin/Developer
# Mode that raises WinError 1314 mid-download, which we'd otherwise swallow as a
# silent CPU fallback. Force plain copies so any model downloads cleanly to GPU.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# Workspace layout: burned videos go to output_video/ at the workspace root (the
# parent of this script's subtitler/ folder), regardless of where the input is.
# A kept .ass (--keep-subs) is written alongside the output video.
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

# --- 1. Register the CUDA DLLs that live inside this venv --------------------
# On Windows the nvidia-cublas / nvidia-cudnn wheels drop their DLLs into
# site-packages/nvidia/*/bin, but they are NOT on the default DLL search path.
# We must add them before importing faster_whisper, or CUDA load fails.
def _register_cuda_dlls():
    base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    binpaths = glob.glob(os.path.join(base, "*", "bin"))
    for binpath in binpaths:
        try:
            os.add_dll_directory(binpath)
        except OSError:
            pass
    # CTranslate2 loads cuBLAS with a legacy LoadLibrary call that ignores
    # add_dll_directory and searches PATH, so prepend the bin dirs to PATH too.
    if binpaths:
        os.environ["PATH"] = os.pathsep.join(binpaths + [os.environ.get("PATH", "")])


_register_cuda_dlls()


# --- 2. Font helpers --------------------------------------------------------
WINDOWS_FONTS_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")


def list_system_fonts():
    """Return {display_name: filename} from the Windows font registry."""
    fonts = {}
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        )
        i = 0
        while True:
            try:
                name, filename, _ = winreg.EnumValue(key, i)
            except OSError:
                break
            i += 1
            # Names look like "Arial (TrueType)" / "Bebas Neue (TrueType)"
            clean = name
            for suffix in (" (TrueType)", " (OpenType)", " (All res)"):
                if clean.endswith(suffix):
                    clean = clean[: -len(suffix)]
            fonts[clean] = filename
    except Exception as e:
        print(f"Could not read font registry: {e}", file=sys.stderr)
    return fonts


# --- 3. ASS subtitle generation --------------------------------------------
def fmt_time(seconds):
    """Seconds -> ASS timestamp H:MM:SS.cc (centiseconds)."""
    if seconds < 0:
        seconds = 0
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def rgb_to_ass(color):
    """'#RRGGBB' -> ASS '&HAABBGGRR' (AA=00 fully opaque)."""
    color = color.lstrip("#")
    r, g, b = color[0:2], color[2:4], color[4:6]
    return f"&H00{b}{g}{r}".upper()


# Color-combo presets: name -> (text, background/outline, karaoke-highlight, boxed)
# "background" is the box fill when boxed, or the glyph outline when not boxed.
THEMES = {
    "classic":     ("#FFFFFF", "#000000", "#FFD400", False),  # white text, black outline
    "white-box":   ("#FFFFFF", "#000000", "#FFD400", True),   # white on black box (requested)
    "yellow-box":  ("#FFE600", "#000000", "#FFFFFF", True),
    "black-box":   ("#111111", "#FFFFFF", "#E50914", True),   # black text, white box
    "red-box":     ("#FFFFFF", "#C1121F", "#FFE600", True),
    "blue-box":    ("#FFFFFF", "#0353A4", "#FFD400", True),
    "green-box":   ("#FFFFFF", "#1B7F3B", "#FFE600", True),
    "purple-box":  ("#FFFFFF", "#6A0DAD", "#FFD400", True),
    "pink-box":    ("#FFFFFF", "#D6336C", "#FFFFFF", True),
    "mint":        ("#0B3D2E", "#B8F2D6", "#0B7A4B", True),   # dark text, mint box
    "neon":        ("#39FF14", "#000000", "#FFFFFF", True),   # neon green on black
    "hormozi":     ("#FFFFFF", "#000000", "#00E676", False),  # white + green highlight (great for karaoke)
    "sunset":      ("#FFF3B0", "#9D0208", "#FFFFFF", True),
}


def karaoke_text(words, group_end):
    """Build ASS \\k karaoke markup: words fill to the highlight color as spoken."""
    parts = []
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else group_end
        dur_cs = max(1, round((nxt - w["start"]) * 100))
        parts.append(f"{{\\k{dur_cs}}}{w['word'].strip()}")
    return " ".join(parts)


def wrap_caption(text, max_chars):
    """Wrap a caption to at most 2 lines, breaking on words."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False)
    # keep to 2 lines max; merge overflow into the second line
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return "\\N".join(lines)


def build_ass(captions, opts):
    """Build a full .ass document string from the caption list."""
    text_col = rgb_to_ass(opts.color)
    back_col = rgb_to_ass(opts.outline_color)      # box fill (boxed) or glyph outline
    hl_col = rgb_to_ass(opts.highlight_color)      # karaoke fill target
    align = {"bottom": 2, "top": 8, "middle": 5}[opts.position]

    # BorderStyle 3 = opaque box (OutlineColour = box fill); 1 = outline + shadow.
    border_style = 3 if opts.box else 1
    outline_w = opts.outline if opts.outline else (4 if opts.box else 2)

    # In karaoke, unsung text is SecondaryColour and it fills to PrimaryColour as
    # each word is spoken, so Primary = highlight, Secondary = base text color.
    if opts.style == "karaoke":
        primary, secondary = hl_col, text_col
    else:
        primary, secondary = text_col, text_col

    # Keep a 720-tall coordinate system (so font-size/margins mean the same on
    # any clip) but match the video's aspect ratio so text isn't stretched.
    play_y = 720
    play_x = round(play_y * opts.aspect)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_x}
PlayResY: {play_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{opts.font},{opts.font_size},{primary},{secondary},{back_col},&H64000000,{-1 if opts.bold else 0},0,0,0,100,100,0,0,{border_style},{outline_w},{opts.shadow},{align},40,40,{opts.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for cap in captions:
        if opts.style == "karaoke" and cap.get("words"):
            text = karaoke_text(cap["words"], cap["end"])
        else:
            text = wrap_caption(cap["text"], opts.max_chars)
        if not text:
            continue
        lines.append(
            f"Dialogue: 0,{fmt_time(cap['start'])},{fmt_time(cap['end'])},"
            f"Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


# --- 4. Transcription -------------------------------------------------------
def transcribe(video_path, opts):
    from faster_whisper import WhisperModel

    device, compute = opts.device, opts.compute_type
    if device == "auto":
        try:
            print(f"Loading '{opts.model}' on GPU (CUDA, {opts.compute_type})...")
            model = WhisperModel(opts.model, device="cuda", compute_type=opts.compute_type)
            device = "cuda"
        except Exception as e:
            print(f"  GPU load failed ({e}); falling back to CPU (int8).")
            model = WhisperModel(opts.model, device="cpu", compute_type="int8")
            device, compute = "cpu", "int8"
    else:
        print(f"Loading '{opts.model}' on {device} ({compute})...")
        model = WhisperModel(opts.model, device=device, compute_type=compute)

    print("Transcribing (this streams as it goes)...")
    segments_iter, info = model.transcribe(
        video_path,
        language=opts.language,
        beam_size=opts.beam_size,
        word_timestamps=True,
        # Better VAD: stricter Silero settings to drop music-only stretches.
        vad_filter=True,
        vad_parameters=dict(
            threshold=opts.vad_threshold,
            min_silence_duration_ms=opts.vad_min_silence,
            speech_pad_ms=opts.vad_pad,
            # Cap segment length: stops speech+music merging into long blobs,
            # which is what makes big models drop casing/punctuation & misread.
            max_speech_duration_s=opts.vad_max_speech,
        ),
        # Anti-hallucination: don't feed prior text (stops repetition loops),
        # and actively skip long silent/music gaps where phantoms appear.
        condition_on_previous_text=False,
        hallucination_silence_threshold=opts.hallucination_silence,
        no_speech_threshold=0.6,
    )
    if info.language:
        print(f"  Detected language: {info.language} (p={info.language_probability:.2f})")

    raw = []
    for seg in segments_iter:
        txt = seg.text.strip()
        if not txt:
            continue
        words = [
            {"start": w.start, "end": w.end, "word": w.word}
            for w in (seg.words or [])
            if w.word.strip()
        ]
        raw.append({"start": seg.start, "end": seg.end, "text": txt, "words": words})
        print(f"  [{fmt_time(seg.start)} -> {fmt_time(seg.end)}] {txt}")
    return raw


# --- 4b. Group words into short, tightly-synced captions --------------------
_SENTENCE_END = (".", "!", "?", "…")


def build_captions(raw, opts):
    """Turn raw whisper output into the final caption list per --style."""
    if opts.style == "segments":
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in raw]

    # Word-group style: a few words per caption, synced to word timestamps.
    words = [w for seg in raw for w in seg["words"]]
    if not words:  # model gave no word timings; fall back to segments
        print("  (no word timestamps returned; using segment timing)")
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in raw]

    captions, cur = [], []

    def flush():
        if not cur:
            return
        text = "".join(w["word"] for w in cur).strip()
        if text:
            captions.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "text": text,
                "words": list(cur),
            })

    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            dur = cur[-1]["end"] - cur[0]["start"]
            if gap >= opts.max_gap or dur >= opts.max_caption_duration \
                    or len(cur) >= opts.words_per_caption:
                flush()
                cur = []
        cur.append(w)
        # Hard break right after sentence-ending punctuation for clean phrasing.
        if w["word"].strip().endswith(_SENTENCE_END):
            flush()
            cur = []
    flush()
    return captions


# --- 5. Burn with ffmpeg ----------------------------------------------------
def probe_audio_codec(video_path):
    """Return the first audio stream's codec name, or None if there's no audio."""
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0",
             os.path.abspath(video_path)],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except subprocess.CalledProcessError:
        return None


def probe_video_size(video_path):
    """Return (width, height) of the first video stream, or (1280, 720)."""
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
             os.path.abspath(video_path)],
            capture_output=True, text=True, check=True,
        )
        w, h = out.stdout.strip().split("x")
        return int(w), int(h)
    except (subprocess.CalledProcessError, ValueError):
        return 1280, 720


def burn(video_path, ass_path, output_path, opts):
    # ffmpeg's subtitles filter mangles Windows paths (drive colon, backslashes),
    # so run ffmpeg from the .ass directory and reference it by bare filename.
    ass_dir = os.path.dirname(os.path.abspath(ass_path))
    ass_name = os.path.basename(ass_path)
    fontsdir = WINDOWS_FONTS_DIR.replace("\\", "/").replace(":", r"\:")

    vf = f"subtitles={ass_name}:fontsdir='{fontsdir}'"

    # MP4 only reliably plays AAC/MP3 audio. Sources are often Opus (YouTube AV1)
    # which copies into MP4 fine but plays silent in most players/editors, so
    # transcode anything that isn't already MP4-friendly to AAC.
    codec = probe_audio_codec(video_path)
    if codec is None:
        audio_args = ["-an"]
        print("  (no audio stream found in source)")
    elif codec in ("aac", "mp3"):
        audio_args = ["-c:a", "copy"]
    else:
        audio_args = ["-c:a", "aac", "-b:a", "192k"]
        print(f"  (transcoding {codec} audio -> AAC for MP4 compatibility)")

    cmd = [
        FFMPEG, "-y",
        "-i", os.path.abspath(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf),
        *audio_args,
        os.path.abspath(output_path),
    ]
    print("\nBurning subtitles with ffmpeg...")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=ass_dir, check=True)


# --- 6. CLI -----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Auto-generate and burn subtitles into a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="input video file")
    p.add_argument("-o", "--output", help="output video (default: <input>_subtitled.mp4)")
    p.add_argument("--list-fonts", action="store_true", help="list installed system fonts and exit")

    # model / transcription
    p.add_argument("--model", default="large-v3-turbo", help="whisper model")
    p.add_argument("--language", default="en", help="language code (en); '' = auto-detect")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute-type", default="float16", help="float16 / int8_float16 / int8")
    p.add_argument("--beam-size", type=int, default=5)

    # style
    p.add_argument("--theme", default="classic", choices=sorted(THEMES),
                   help="color preset: " + ", ".join(sorted(THEMES)))
    p.add_argument("--font", default="Arial", help="font family name (see --list-fonts)")
    p.add_argument("--font-size", type=int, default=24)
    p.add_argument("--bold", action="store_true")
    # Color flags default to None so a chosen --theme fills them in; an explicit
    # value always overrides the theme.
    p.add_argument("--color", default=None, help="text color #RRGGBB (overrides theme)")
    p.add_argument("--outline-color", default=None,
                   help="box/outline color #RRGGBB (overrides theme)")
    p.add_argument("--highlight-color", default=None,
                   help="karaoke highlight color #RRGGBB (overrides theme)")
    p.add_argument("--box", action=argparse.BooleanOptionalAction, default=None,
                   help="draw an opaque box behind text (--box / --no-box; default from theme)")
    p.add_argument("--outline", type=float, default=0.0,
                   help="outline/box-padding thickness (0 = auto per style)")
    p.add_argument("--shadow", type=float, default=0.6, help="shadow depth")
    p.add_argument("--position", default="bottom", choices=["bottom", "top", "middle"])
    p.add_argument("--margin-v", type=int, default=90,
                   help="vertical margin from edge (720p units; higher = further from bottom)")
    p.add_argument("--max-chars", type=int, default=42, help="max chars per caption line")

    # caption grouping (word-timestamp style)
    p.add_argument("--style", default="words", choices=["words", "karaoke", "segments"],
                   help="'words' = word-groups; 'karaoke' = words highlight as spoken; "
                        "'segments' = full sentences")
    p.add_argument("--words-per-caption", type=int, default=3,
                   help="max words shown at once in 'words' style")
    p.add_argument("--max-caption-duration", type=float, default=2.5,
                   help="max seconds a single caption stays on screen")
    p.add_argument("--max-gap", type=float, default=0.7,
                   help="split caption when the pause between words exceeds this (s)")

    # VAD / anti-hallucination
    p.add_argument("--vad-threshold", type=float, default=0.5,
                   help="Silero VAD speech probability threshold (higher = stricter)")
    p.add_argument("--vad-min-silence", type=int, default=500,
                   help="min silence (ms) that splits speech")
    p.add_argument("--vad-pad", type=int, default=200,
                   help="padding (ms) kept around detected speech")
    p.add_argument("--vad-max-speech", type=float, default=8.0,
                   help="max seconds per speech segment; shorter = better punctuation")
    p.add_argument("--hallucination-silence", type=float, default=2.0,
                   help="skip silent gaps longer than this (s) to cut phantom captions")

    # encode
    p.add_argument("--preset", default="medium", help="x264 preset")
    p.add_argument("--crf", type=int, default=18, help="x264 quality (lower=better)")
    p.add_argument("--keep-subs", action="store_true", help="keep the generated .ass file")

    opts = p.parse_args()

    # Resolve theme, letting any explicitly-passed color/box flag win over it.
    t_text, t_back, t_hl, t_box = THEMES[opts.theme]
    if opts.color is None:
        opts.color = t_text
    if opts.outline_color is None:
        opts.outline_color = t_back
    if opts.highlight_color is None:
        opts.highlight_color = t_hl
    if opts.box is None:
        opts.box = t_box

    if opts.list_fonts:
        for name in sorted(list_system_fonts(), key=str.lower):
            print(name)
        return

    if not opts.input:
        p.error("input video is required (or use --list-fonts)")
    if not os.path.isfile(opts.input):
        p.error(f"file not found: {opts.input}")
    if opts.language == "":
        opts.language = None

    in_name = os.path.splitext(os.path.basename(opts.input))[0]
    output = opts.output or os.path.join(OUTPUT_DIR, f"{in_name}_subtitled.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    # The .ass is a working file next to the output; kept only with --keep-subs.
    ass_path = os.path.join(os.path.dirname(os.path.abspath(output)), f"{in_name}.ass")

    vw, vh = probe_video_size(opts.input)
    opts.aspect = vw / vh
    print(f"Video: {vw}x{vh} ({'vertical' if vw < vh else 'landscape'})")

    raw = transcribe(opts.input, opts)
    if not raw:
        print("No speech detected; nothing to burn.", file=sys.stderr)
        sys.exit(1)

    captions = build_captions(raw, opts)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(build_ass(captions, opts))
    print(f"\nWrote subtitles: {ass_path} ({len(captions)} captions, style={opts.style})")

    burn(opts.input, ass_path, output, opts)

    if not opts.keep_subs:
        try:
            os.remove(ass_path)
        except OSError:
            pass

    print(f"\nDone -> {output}")


if __name__ == "__main__":
    main()
