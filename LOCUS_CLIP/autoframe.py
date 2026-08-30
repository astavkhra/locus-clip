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


def _make_saliency():
    """Static spectral-residual saliency detector, or None if unavailable."""
    if not hasattr(cv2, "saliency"):
        return None
    try:
        return cv2.saliency.StaticSaliencySpectralResidual_create()
    except Exception:
        return None


def _saliency_center_x(sal, frame, w):
    """Horizontal center-of-mass of visual saliency (peaks emphasized), or None.
    Used only when no face is present, so volatile shots follow the focal point
    instead of freezing on a stale position."""
    if sal is None:
        return None
    try:
        ok, smap = sal.computeSaliency(frame)
    except Exception:
        return None
    if not ok or smap is None:
        return None
    smap = smap.astype(np.float32)
    smap *= smap                                   # emphasize salient peaks
    col = smap.sum(axis=0)
    total = float(col.sum())
    if total <= 0:
        return None
    wm = col.shape[0]
    cx_map = float((np.arange(wm, dtype=np.float32) * col).sum() / total)
    return cx_map / wm * w


def _face_centers(ffmpeg, video, start, dur, w, h, model_path,
                  sample_fps=8, score_thresh=0.6):
    """Sample the clip at `sample_fps` and return [(clip_time, center_x, kind)].

    kind is 'face' (YuNet, authoritative), 'sal' (saliency fallback when no
    face), or None (nothing found). Frames are piped from ffmpeg (handles odd
    filenames + only decodes the frames we sample) as raw BGR."""
    det = cv2.FaceDetectorYN.create(model_path, "", (w, h), score_thresh)
    det.setInputSize((w, h))
    sal = _make_saliency()
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
        if faces is not None and len(faces):
            best = max(faces, key=lambda f: f[2] * f[3])   # most prominent face wins
            cx, kind = float(best[0] + best[2] / 2.0), "face"
        else:
            cx = _saliency_center_x(sal, frame, w)         # fall back to focal point
            kind = "sal" if cx is not None else None
        samples.append((i / sample_fps, cx, kind, False))
        i += 1
    proc.stdout.close()
    proc.wait()
    return samples


def _mouth_roi(face, w, h):
    """Rectangle around the mouth from YuNet's two mouth-corner landmarks."""
    rcx, rcy, lcx, lcy = face[10], face[11], face[12], face[13]
    mx, my = (rcx + lcx) / 2.0, (rcy + lcy) / 2.0
    mw = max(10.0, abs(lcx - rcx))
    hw, hh = mw * 0.9, mw * 0.7
    x0, x1 = int(max(0, mx - hw)), int(min(w, mx + hw))
    y0, y1 = int(max(0, my - hh)), int(min(h, my + hh))
    return x0, y0, x1, y1


def _speaker_centers(ffmpeg, video, start, dur, w, h, model_path,
                     sample_fps=10, score_thresh=0.6):
    """Active-speaker tracking: per frame, pick the face whose mouth is moving
    most (frame-diff in the mouth region), with hysteresis so the crop cuts to a
    new speaker only once they clearly take over. Returns
    [(t, center_x, kind, is_switch)]; is_switch marks a speaker change (snap)."""
    det = cv2.FaceDetectorYN.create(model_path, "", (w, h), score_thresh)
    det.setInputSize((w, h))
    sal = _make_saliency()
    proc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", os.path.abspath(video),
         "-vf", f"fps={sample_fps}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE,
    )
    frame_bytes = w * h * 3
    tracks = []           # {id, cx, cy, ema, seen} -- one per followed face
    next_id, active_id, lead_id, lead_n = 0, None, None, 0
    dist_thr = 0.12 * w
    SWITCH_HOLD = 3       # challenger must lead this many frames to steal focus
    samples, prev_gray, i = [], None, 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape((h, w, 3)).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, faces = det.detect(frame)
        faces = [] if faces is None else list(faces)

        # Measure mouth motion per detected face (vs previous frame's pixels).
        dets = []
        for f in faces:
            cx = float(f[0] + f[2] / 2.0)
            cy = float(f[1] + f[3] / 2.0)
            act = 0.0
            if prev_gray is not None:
                x0, y0, x1, y1 = _mouth_roi(f, w, h)
                if x1 > x0 and y1 > y0:
                    a = gray[y0:y1, x0:x1].astype(np.int16)
                    b = prev_gray[y0:y1, x0:x1].astype(np.int16)
                    act = float(np.abs(a - b).mean())
            dets.append({"cx": cx, "cy": cy, "act": act})

        # Associate detections with existing tracks (greedy nearest center).
        for tr in tracks:
            tr["seen"] = False
        for d in dets:
            best, bestdist = None, dist_thr
            for tr in tracks:
                if tr["seen"]:
                    continue
                dist = abs(tr["cx"] - d["cx"]) + abs(tr["cy"] - d["cy"])
                if dist < bestdist:
                    best, bestdist = tr, dist
            if best is None:
                best = {"id": next_id, "cx": d["cx"], "cy": d["cy"], "ema": 0.0, "seen": True}
                tracks.append(best)
                next_id += 1
            best["cx"], best["cy"], best["seen"] = d["cx"], d["cy"], True
            best["ema"] = 0.5 * best["ema"] + 0.5 * d["act"]
        # Keep seen tracks; decay unseen ones and drop when they fade out.
        kept = []
        for t in tracks:
            if t["seen"]:
                kept.append(t)
            else:
                t["ema"] *= 0.6
                if t["ema"] > 1.0:
                    kept.append(t)
        tracks = kept

        present = [t for t in tracks if t["seen"]]
        is_switch = False
        cx = None
        if present:
            best = max(present, key=lambda t: t["ema"])
            cur = next((t for t in present if t["id"] == active_id), None)
            if cur is None:
                active_id, is_switch = best["id"], (active_id is not None)  # forced switch
                lead_id, lead_n = None, 0
            elif best["id"] != cur["id"] and best["ema"] > cur["ema"] * 1.25:
                lead_n = lead_n + 1 if best["id"] == lead_id else 1
                lead_id = best["id"]
                if lead_n >= SWITCH_HOLD:
                    active_id, is_switch, lead_n = best["id"], True, 0
            else:
                lead_id, lead_n = None, 0
            act_tr = next((t for t in present if t["id"] == active_id), best)
            cx = act_tr["cx"]
        else:
            cx = _saliency_center_x(sal, frame, w)

        kind = "face" if present else ("sal" if cx is not None else None)
        samples.append((i / sample_fps, cx, kind, is_switch))
        prev_gray = gray
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
                sample_fps=8, alpha=0.25, deadzone_frac=0.03, out_ar=9 / 16,
                multi_speaker=False):
    """Plan a moving crop. Returns dict(crop_w, crop_h, commands=[(t, x_left)])
    or None if no face is ever found (caller should fall back to a center crop).
    With multi_speaker, the crop cuts to whoever is actively speaking."""
    model_path = ensure_model(model_path)
    if multi_speaker:
        samples = _speaker_centers(ffmpeg, video, start, dur, w, h, model_path)
    else:
        samples = _face_centers(ffmpeg, video, start, dur, w, h, model_path, sample_fps)
    if not samples or all(cx is None for _, cx, _, _ in samples):
        return None

    crop_w = min(w, int(round(h * out_ar)))
    crop_h = h
    half = crop_w / 2.0
    max_x = max(0, w - crop_w)
    deadzone = w * deadzone_frac
    cuts = _scene_cuts(ffmpeg, video, start, dur)

    # Fill any still-empty frames (no face AND no saliency) with the last center.
    filled, last = [], w / 2.0
    for t, cx, _, sw in samples:
        if cx is None:
            cx = last
        else:
            last = cx
        filled.append((t, cx, sw))

    # Smooth into a crop-x trajectory; snap (reset) at scene cuts AND speaker
    # switches (a switch should be a hard cut to the new face, not a pan).
    commands, s, cut_ptr, prev_t, n_switch = [], None, 0, None, 0
    for t, cx, sw in filled:
        reset = s is None
        while cut_ptr < len(cuts) and cuts[cut_ptr] <= t:
            if prev_t is None or cuts[cut_ptr] > prev_t:
                reset = True
            cut_ptr += 1
        if sw:
            reset = True
            n_switch += 1
        if reset:
            s = cx
        elif abs(cx - s) > deadzone:
            s += alpha * (cx - s)
        x_left = int(round(min(max(s - half, 0), max_x)))
        commands.append((t, x_left))
        prev_t = t

    n_face = sum(1 for _, _, k, _ in samples if k == "face")
    n_sal = sum(1 for _, _, k, _ in samples if k == "sal")
    return {"crop_w": crop_w, "crop_h": crop_h, "commands": commands,
            "n_face": n_face, "n_sal": n_sal, "n_samples": len(samples),
            "n_cuts": len(cuts), "n_switch": n_switch}


def write_sendcmd(commands, path):
    """Write a ffmpeg sendcmd script that pans crop x over time."""
    lines = [f"{t:.3f} crop x {x};" for t, x in commands]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
