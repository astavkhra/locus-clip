"""
Content-aware vertical reframing for LOCUS_CLIP.

Follows the (single) speaker's face with a moving 9:16 crop window instead of
padding with a blurred background. v1 scope: single-speaker, horizontal-only
pan, scene-cut aware.

The only degree of freedom for a 16:9 -> 9:16 crop is the horizontal position
(the window is full height), so tracking reduces to a 1-D signal x(t):
  detect face center per sampled frame  (OpenCV YuNet)
  -> fill gaps, snap at scene cuts, smooth with a deadzone
  -> emit ffmpeg `sendcmd` commands that pan the crop's x over time.
"""

import os
import re
import subprocess
import numpy as np
import cv2

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def ensure_model(path):
    """Return the YuNet model path, downloading it once if missing."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        print("Downloading face-detection model (YuNet, ~230KB)...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _face_centers(ffmpeg, video, start, dur, w, h, model_path,
                  sample_fps=8, score_thresh=0.6):
    """Sample the clip at `sample_fps` and return [(clip_time, center_x|None)].

    Frames are piped from ffmpeg (handles odd filenames + only decodes the
    frames we sample) as raw BGR, then run through YuNet."""
    det = cv2.FaceDetectorYN.create(model_path, "", (w, h), score_thresh)
    det.setInputSize((w, h))
    proc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", os.path.abspath(video),
         "-vf", f"fps={sample_fps}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE,
    )
    frame_bytes = w * h * 3
    samples, i = [], 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape((h, w, 3)).copy()
        _, faces = det.detect(frame)
        cx = None
        if faces is not None and len(faces):
            best = max(faces, key=lambda f: f[2] * f[3])   # most prominent face
            cx = float(best[0] + best[2] / 2.0)
        samples.append((i / sample_fps, cx))
        i += 1
    proc.stdout.close()
    proc.wait()
    return samples


def _scene_cuts(ffmpeg, video, start, dur, threshold=0.4):
    """Clip-relative timestamps where the shot changes."""
    out = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", os.path.abspath(video),
         "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return sorted(float(m) for m in re.findall(r"pts_time:([\d.]+)", out.stderr))


def build_track(ffmpeg, video, start, dur, w, h, model_path,
                sample_fps=8, alpha=0.25, deadzone_frac=0.03, out_ar=9 / 16):
    """Plan a moving crop. Returns dict(crop_w, crop_h, commands=[(t, x_left)])
    or None if no face is ever found (caller should fall back to a center crop)."""
    model_path = ensure_model(model_path)
    samples = _face_centers(ffmpeg, video, start, dur, w, h, model_path, sample_fps)
    if not samples or all(cx is None for _, cx in samples):
        return None

    crop_w = min(w, int(round(h * out_ar)))
    crop_h = h
    half = crop_w / 2.0
    max_x = max(0, w - crop_w)
    deadzone = w * deadzone_frac
    cuts = _scene_cuts(ffmpeg, video, start, dur)

    # Fill undetected frames with the last known center (holds position).
    filled, last = [], w / 2.0
    for t, cx in samples:
        if cx is None:
            cx = last
        else:
            last = cx
        filled.append((t, cx))

    # Smooth into a crop-x trajectory; snap (reset) at each scene cut.
    commands, s, cut_ptr, prev_t = [], None, 0, None
    for t, cx in filled:
        reset = s is None
        while cut_ptr < len(cuts) and cuts[cut_ptr] <= t:
            if prev_t is None or cuts[cut_ptr] > prev_t:
                reset = True
            cut_ptr += 1
        if reset:
            s = cx
        elif abs(cx - s) > deadzone:
            s += alpha * (cx - s)
        x_left = int(round(min(max(s - half, 0), max_x)))
        commands.append((t, x_left))
        prev_t = t

    n_hits = sum(1 for _, cx in samples if cx is not None)
    return {"crop_w": crop_w, "crop_h": crop_h, "commands": commands,
            "n_hits": n_hits, "n_samples": len(samples), "n_cuts": len(cuts)}


def write_sendcmd(commands, path):
    """Write a ffmpeg sendcmd script that pans crop x over time."""
    lines = [f"{t:.3f} crop x {x};" for t, x in commands]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
