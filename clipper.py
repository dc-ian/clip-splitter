#!/usr/bin/env python3
"""
clipper.py - Phase 1 + 2: split one long video into 15-30 s clips,
cutting at natural pauses and scene changes.

Usage:
    python clipper.py input.mp4                    # smart cuts (default)
    python clipper.py input.mp4 --plan-only        # preview the cut list, no encoding
    python clipper.py input.mp4 --mode fixed       # Phase 1 behaviour
    python clipper.py input.mp4 -o clips --clip-len 30 --min-len 15

Phase 3 (transcript cuts, needs: pip install faster-whisper):
  * local Whisper (int8 on CPU, float16 on CUDA, VAD on), one model instance, run in
    the background while the other scans run; finished and freed before any export
  * cuts land on sentence ends; no cut ever passes through a spoken word
  * transcript cached to transcript.json (word timestamps), reused on every re-clip
  * each clip's text is saved in manifest.json
  * --skip-transcription or a missing faster-whisper falls back to Phase 2 behaviour

Phase 2 (smart cuts):
  * adaptive noise floor -> silencedetect (min pause 0.4 s)
  * scene-change detection (ffmpeg scene filter, runs on a small 5 fps copy)
  * audio-energy dips as a last-resort signal for music / constant noise
  * scored cut picker with a fallback ladder:
        sentence end > silence > scene change > energy dip > word gap > hard cut (with fade)
  * hard-cut clips are flagged in the manifest and the end report
  * analysis cached to analysis.json, so re-runs are instant

Phase 1 features (still present):
  * ffprobe validation with clear errors
  * VFR detection + one-time normalization to constant frame rate
  * free-disk-space check
  * precise cuts (-ss before -i, re-encode) with hardware encoder auto-detect
  * JSON manifest written after every clip, so a crashed run resumes
  * temp/work files cleaned up on success (kept on crash so resume is fast)

Requires: Python 3.8+, ffmpeg and ffprobe on PATH. No other dependencies.
"""
import argparse
import bisect
import contextlib
import gc
import json
import math
import re
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

# Encoders tried in order. The first one that passes a real test encode wins.
ENCODERS = [
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21"]),
    ("h264_qsv", ["-c:v", "h264_qsv", "-global_quality", "21"]),
    ("h264_videotoolbox", ["-c:v", "h264_videotoolbox", "-q:v", "60"]),
    ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]),
]
DISK_FACTOR = 3.0  # normalized copy + clips ~ 2-3x source size
MANIFEST_NAME = "manifest.json"
ANALYSIS_NAME = "analysis.json"
TRANSCRIPT_NAME = "transcript.json"
RMS_WIN = 0.1  # seconds per audio-energy window
VERT_W, VERT_H = 1080, 1920          # 9:16 output size
SIDECAR_VER = 1                      # bump if the .srt layout ever changes
QUEUE_NAME = "clipper_queue.json"
QUEUE_MAX_CRASHES = 3                # an item that kills the app this many times is marked failed
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".ts", ".wmv"}
LEGACY_SIG = {"fast": False, "reframe": "off", "burn": False}   # what clips exported before Phase 5 look like


class ClipperError(Exception):
    """Expected, user-facing failure (bad file, no disk space, ...)."""


class Cancelled(ClipperError):
    """The user pressed Cancel. The message always starts with 'Cancelled'."""


# ----------------------------------------------------------------- helpers
_CANCEL = None   # threading.Event of the running job (set by run_job); polled while ffmpeg runs
# Windows: never pop up a console window for every ffmpeg/ffprobe call (matters for --noconsole builds)
_NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _run_cancellable(cmd, check=True, cwd=None, text=True):
    """subprocess.run() that notices Cancel within ~0.25 s and kills the child process."""
    kw = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, **_NO_WINDOW)
    if text:
        kw.update(text=True, errors="replace")
    proc = subprocess.Popen(cmd, **kw)
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if _CANCEL is not None and _CANCEL.is_set():
                    raise Cancelled("Cancelled.")
    except BaseException:
        proc.kill()
        with contextlib.suppress(Exception):
            proc.communicate()
        raise
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, out, err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def run(cmd, check=True, cwd=None):
    return _run_cancellable(cmd, check=check, cwd=cwd, text=True)


def need_tool(name):
    if shutil.which(name) is None:
        raise ClipperError(f"'{name}' not found on PATH. Install FFmpeg first: https://ffmpeg.org")


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def parse_rate(text):
    try:
        f = Fraction(text)
        return float(f)
    except (ValueError, ZeroDivisionError):
        return 0.0


# ------------------------------------------------------------------ ingest
def probe(path):
    """Validate the file and return the metadata we need."""
    r = run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=False,
    )
    if r.returncode != 0:
        raise ClipperError(f"ffprobe could not read this file:\n{r.stderr.strip() or 'unknown error'}")
    info = json.loads(r.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ClipperError("No video stream found in this file.")
    duration = float(info.get("format", {}).get("duration") or video.get("duration") or 0)
    if duration <= 0:
        raise ClipperError("Could not determine the video duration (file may be truncated).")
    width, height = video.get("width"), video.get("height")
    rot = 0
    with contextlib.suppress(TypeError, ValueError):
        rot = int(float((video.get("tags") or {}).get("rotate", 0)))
    for sd in video.get("side_data_list") or []:
        if "rotation" in sd:
            with contextlib.suppress(TypeError, ValueError):
                rot = int(float(sd["rotation"]))
    if abs(rot) % 180 == 90 and width and height:      # phone video stored sideways
        width, height = height, width
    r_fps = parse_rate(video.get("r_frame_rate", "0/1"))
    avg_fps = parse_rate(video.get("avg_frame_rate", "0/1"))
    # VFR when the two rates disagree by more than ~1%
    is_vfr = r_fps > 0 and avg_fps > 0 and abs(r_fps - avg_fps) / max(r_fps, avg_fps) > 0.01
    return {
        "duration": duration,
        "codec": video.get("codec_name"),
        "width": width,
        "height": height,
        "fps": avg_fps or r_fps or 30.0,
        "is_vfr": is_vfr,
        "has_audio": audio is not None,
        "size_bytes": int(info.get("format", {}).get("size") or os.path.getsize(path)),
    }


def check_disk_space(out_dir, size_bytes):
    need = int(size_bytes * DISK_FACTOR)
    free = shutil.disk_usage(out_dir).free
    if free < need:
        raise ClipperError(
            f"Not enough free disk space: need about {need / 1e9:.1f} GB, "
            f"have {free / 1e9:.1f} GB in '{out_dir}'."
        )


# ---------------------------------------------------------------- encoders
def pick_encoder(forced=None):
    """Return (name, args). Verifies the encoder with a real 1-frame test."""
    listed = run(["ffmpeg", "-hide_banner", "-encoders"], check=False).stdout
    candidates = [e for e in ENCODERS if forced in (None, e[0])]
    if forced and not candidates:
        raise ClipperError(f"Unknown encoder '{forced}'. Choose from: {', '.join(e[0] for e in ENCODERS)}")
    for name, args in candidates:
        if name not in listed:
            continue
        test = run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2:r=10",
             *args, "-pix_fmt", "yuv420p", "-f", "null", "-"],
            check=False,
        )
        if test.returncode == 0:
            return name, args
    raise ClipperError("No working H.264 encoder found (not even libx264).")


# ----------------------------------------------------------- normalization
def normalize(src, dst, fps, enc_args, has_audio):
    """Convert a VFR file to constant frame rate. Writes to .part then renames."""
    part = dst.with_name(dst.name + ".part")
    cfr = round(fps) if abs(fps - round(fps)) < 0.05 else round(fps, 3)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", f"fps={cfr}", *enc_args, "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-b:a", "160k"] if has_audio else ["-an"]
    cmd += ["-f", "mp4", str(part)]
    try:
        r = run(cmd, check=False)
    except Cancelled:
        part.unlink(missing_ok=True)
        raise
    if r.returncode != 0:
        part.unlink(missing_ok=True)
        raise ClipperError(f"Normalization failed:\n{r.stderr.strip()}")
    part.replace(dst)


# ------------------------------------------------------------ segmentation
def plan_segments(duration, clip_len, min_len):
    """
    Fixed-length plan. If the last clip would be shorter than min_len, the last two
    clips are re-balanced into two equal halves so every clip stays inside
    [min_len, clip_len]. (Falls back to merging them if that is impossible.)
    """
    segs, t = [], 0.0
    while t < duration - 0.01:
        end = min(t + clip_len, duration)
        segs.append([round(t, 3), round(end, 3)])
        t = end
    if len(segs) > 1 and (segs[-1][1] - segs[-1][0]) < min_len:
        tail = segs.pop()
        prev = segs[-1]
        combined = tail[1] - prev[0]
        if combined / 2 >= min_len:
            mid = round(prev[0] + combined / 2, 3)
            prev[1] = mid
            segs.append([mid, tail[1]])
        else:
            prev[1] = tail[1]
    return segs


# ---------------------------------------------------------------- analysis
def say(msg):
    """print() that stays on one line when two background threads talk at once."""
    with _PRINT_LOCK:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()


def pass_rms(src):
    """Audio energy per RMS_WIN seconds -> list of (time, dB). One cheap ffmpeg pass."""
    n = int(8000 * RMS_WIN)
    af = (f"asetnsamples=n={n}:p=0,astats=metadata=1:reset=1,"
          "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-")
    r = run(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "8000",
             "-af", af, "-f", "null", "-"], check=False)
    series, t = [], None
    for line in r.stdout.splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        if "RMS_level=" in line and t is not None:
            v = line.split("=", 1)[1].strip()
            try:
                db = float(v)
            except ValueError:
                db = -90.0
            series.append((t, max(-90.0, min(0.0, db))))
    return series


def pack_rms(series):
    """Compact form of the loudness series for analysis.json (uniform RMS_WIN steps)."""
    if not series:
        return None
    return {"t0": round(series[0][0], 3), "step": RMS_WIN, "db": [round(d, 1) for _, d in series]}


def noise_floor_threshold(series, override=None):
    """Silence threshold relative to THIS video's noise floor (not a fixed -30 dB)."""
    if override is not None:
        return float(override), None
    if not series:
        return -35.0, None
    vals = sorted(db for _, db in series)
    floor = vals[int(len(vals) * 0.10)]  # 10th percentile ~ quietest normal moments
    median = vals[len(vals) // 2]
    # floor + 6 dB, but never closer than 12 dB to the typical level (protects constant music/noise)
    thr = min(floor + 6.0, median - 12.0)
    return max(-55.0, min(-25.0, thr)), floor


def pass_silence(src, thresh_db, min_dur):
    r = run(["ffmpeg", "-nostats", "-v", "info", "-i", str(src), "-vn",
             "-af", f"silencedetect=n={thresh_db:.1f}dB:d={min_dur}", "-f", "null", "-"], check=False)
    out, start = [], None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start: (-?[0-9.]+)", line)
        if m:
            start = max(0.0, float(m.group(1)))
            continue
        m = re.search(r"silence_end: ([0-9.]+)", line)
        if m and start is not None:
            out.append([round(start, 3), round(float(m.group(1)), 3)])
            start = None
    return out


def pass_scenes(src, thresh):
    """Scene cuts, measured on a tiny 5 fps copy so a 1 hr file takes minutes, not an hour."""
    vf = f"fps=5,scale=320:-2,select='gt(scene,{thresh})',showinfo"
    r = run(["ffmpeg", "-nostats", "-v", "info", "-i", str(src), "-an", "-vf", vf, "-f", "null", "-"], check=False)
    times = [float(m.group(1)) for m in re.finditer(r"pts_time:([0-9.]+)", r.stderr)]
    return sorted(set(round(t, 3) for t in times))


def find_energy_dips(series, min_depth_db=6.0):
    """Local loudness minima: natural 'breathing points' even in music or constant noise."""
    if len(series) < 50:
        return []
    ts = [t for t, _ in series]
    db = [d for _, d in series]
    sm = [sum(db[max(0, i - 1):i + 2]) / len(db[max(0, i - 1):i + 2]) for i in range(len(db))]  # ~0.3 s
    pre = [0.0]
    for v in sm:
        pre.append(pre[-1] + v)
    half = int(2.5 / RMS_WIN)       # +-2.5 s context
    nb = int(0.5 / RMS_WIN)         # must be the minimum within +-0.5 s
    dips = []
    for i in range(len(sm)):
        lo, hi = max(0, i - half), min(len(sm), i + half + 1)
        local_mean = (pre[hi] - pre[lo]) / (hi - lo)
        depth = local_mean - sm[i]
        if depth < min_depth_db:
            continue
        if sm[i] > min(sm[max(0, i - nb):i + nb + 1]) + 1e-9:
            continue
        t = ts[i] + RMS_WIN / 2
        if dips and t - dips[-1][0] < 1.0:           # keep only the deeper of two close dips
            if depth > dips[-1][1]:
                dips[-1] = [t, depth]
            continue
        dips.append([t, depth])
    return [[round(t, 3), round(d, 1)] for t, d in dips]


def analyze(src, meta, args, cache_path, fp):
    """Run the three passes in parallel; cache the result."""
    key = dict(fp, silence_min=args.silence_min, scene_thresh=args.scene_thresh, silence_db=args.silence_db)
    if cache_path.exists() and not args.reanalyze:
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("key") == key:
                say("Analysis: reusing cached result.")
                return cached
        except json.JSONDecodeError:
            pass
    say("Analysis: scanning audio and video (runs in parallel)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_scene = pool.submit(pass_scenes, src, args.scene_thresh)
        series, silences, dips, thr, floor = [], [], [], None, None
        if meta["has_audio"]:
            series = pass_rms(src)
            thr, floor = noise_floor_threshold(series, args.silence_db)
            silences = pass_silence(src, thr, args.silence_min)
            dips = find_energy_dips(series)
        scenes = f_scene.result()
    result = {"key": key, "noise_floor_db": floor, "silence_threshold_db": thr,
              "silences": silences, "scenes": scenes, "dips": dips, "rms": pack_rms(series)}
    cache_path.write_text(json.dumps(result))
    fl = f"{floor:.1f}" if floor is not None else "n/a"
    th = f"{thr:.1f}" if thr is not None else "n/a"
    say(f"  noise floor {fl} dB -> silence threshold {th} dB")
    say(f"  found {len(silences)} pauses, {len(scenes)} scene changes, {len(dips)} energy dips "
          f"({fmt_time(time.time() - t0)})")
    return result


# ------------------------------------------------------------ transcription
_PRINT_LOCK = threading.Lock()
_WHISPER_LOCK = threading.Lock()  # never more than ONE model instance alive at a time
ABBREV = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "jr.", "sr."}


def whisper_available():
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def pick_device(want):
    if want in ("cpu", "cuda"):
        return want
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def get_transcript(src, args, cache_path, work_dir, fp, duration):
    """
    Transcribe with faster-whisper (local). Returns {"words": [...], "segments": [...]} or None.
    One model instance, run sequentially, freed before the export stage. Cached to disk
    (the cache does NOT depend on clip settings, so re-clipping never re-transcribes).
    """
    key = dict(fp, model=args.whisper_model, language=args.language)
    if cache_path.exists() and not args.reanalyze:
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("key") == key:
                say("Transcript: reusing cached result.")
                return cached
        except json.JSONDecodeError:
            pass
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        say("Transcript: faster-whisper not installed, skipping (pip install faster-whisper).")
        return None
    wav = work_dir / "audio16k.wav"
    try:
        r = run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(wav)], check=False)
        if r.returncode != 0:
            raise ClipperError(f"audio extraction failed: {r.stderr.strip()}")
        device = pick_device(args.device)
        compute = "float16" if device == "cuda" else "int8"
        say(f"Transcript: model '{args.whisper_model}' on {device} ({compute}), VAD on...")
        t0 = time.time()
        with _WHISPER_LOCK:
            model = WhisperModel(args.whisper_model, device=device, compute_type=compute)
            segments, info = model.transcribe(
                str(wav), language=args.language, vad_filter=True, word_timestamps=True,
                beam_size=5 if device == "cuda" else 1)
            words, segs, last_pct = [], [], -1
            for sg in segments:
                if _CANCEL is not None and _CANCEL.is_set():
                    raise Cancelled("Cancelled.")
                segs.append({"s": round(sg.start, 3), "e": round(sg.end, 3), "text": sg.text.strip()})
                for w in (sg.words or []):
                    text = w.word.strip()
                    if text:
                        words.append({"w": text, "s": round(w.start, 3), "e": round(w.end, 3)})
                pct = int(sg.end / duration * 100) // 10 * 10 if duration else 0
                if pct != last_pct:
                    last_pct = pct
                    say(f"  transcribing... {min(pct, 100)}%")
            lang = getattr(info, "language", None)
            del model, segments  # free RAM/VRAM before the export stage
            gc.collect()
        result = {"key": key, "language": lang, "words": words, "segments": segs}
        cache_path.write_text(json.dumps(result))
        say(f"  transcript: {len(words)} words in {fmt_time(time.time() - t0)}")
        return result
    except Cancelled:
        raise
    except Exception as e:  # missing model download, out of memory, ...
        say(f"Transcript: failed ({type(e).__name__}: {e}). Continuing without it.")
        return None
    finally:
        wav.unlink(missing_ok=True)


def sentence_gaps(words):
    """-> (sentence_cuts, gap_cuts): lists of (cut_time, gap_len). Cuts sit in the pause AFTER the word."""
    sent, gaps = [], []
    for i, w in enumerate(words):
        nxt = words[i + 1]["s"] if i + 1 < len(words) else w["e"] + 0.6
        gap = max(0.0, nxt - w["e"])
        cut = w["e"] + min(gap * 0.5, 0.35)
        token = w["w"].lower().rstrip("\"')]”’")
        if token.endswith((".", "?", "!", "…")) and token not in ABBREV:
            sent.append((cut, gap))
        elif gap >= 0.15:
            gaps.append((cut, gap))
    return sent, gaps


def snap_out_of_word(cut, words, starts, lo, hi):
    """Never cut through a spoken word: move the cut to the nearest gap that is still inside [lo, hi]."""
    if not words:
        return cut, False
    i = bisect.bisect_right(starts, cut) - 1
    if i < 0 or not (words[i]["s"] < cut < words[i]["e"]):
        return cut, False
    options = [words[i]["s"] - 0.03, words[i]["e"] + 0.03]
    options = [o for o in options if lo <= o <= hi]
    if not options:
        return cut, False
    return min(options, key=lambda o: abs(o - cut)), True


def text_between(words, start, end):
    return " ".join(w["w"] for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.05)


# ------------------------------------------------------------ smart planner
def build_candidates(an, words=None):
    """Turn raw detections into scored cut candidates: [time, kind, score]."""
    c = []
    for s, e in an["silences"]:
        c.append([(s + e) / 2, "silence", 70 + min(e - s, 1.5) * 10])   # 74 .. 85
    for t in an["scenes"]:
        c.append([t, "scene", 60.0])
    for t, depth in an["dips"]:
        c.append([t, "dip", 40 + min(depth, 12.0) / 12.0 * 10])          # 40 .. 50
    if words:
        sent, gaps = sentence_gaps(words)
        for t, gap in sent:
            c.append([t, "sentence", 100.0])                              # best signal
        for t, gap in gaps:
            c.append([t, "word", 20 + min(gap, 0.4) / 0.4 * 10])         # 20 .. 30: between two words
    c.sort()
    times = [x[0] for x in c]
    # two different signals within 0.5 s agree with each other -> boost
    for i, (t, kind, score) in enumerate(c):
        lo, hi = bisect.bisect_left(times, t - 0.5), bisect.bisect_right(times, t + 0.5)
        if any(c[j][1] != kind for j in range(lo, hi)):
            c[i][2] = score + 12
    return c


def plan_smart(duration, an, clip_len, min_len, words=None):
    """Greedy: from t, pick the best-scoring candidate in [t+min, t+max]; else hard cut."""
    cands = build_candidates(an, words)
    times = [x[0] for x in cands]
    starts = [w["s"] for w in words] if words else []
    ideal = (min_len + clip_len) / 2
    segs, t = [], 0.0
    while t < duration - 0.01:
        if duration - t <= clip_len:                       # last clip: take the rest
            segs.append({"start": round(t, 3), "end": round(duration, 3), "signal": "end", "score": 0.0})
            break
        lo = t + min_len
        hi = min(t + clip_len, duration - min_len)         # never leave a tail shorter than min_len
        if hi < lo:
            hi = lo
        i0, i1 = bisect.bisect_left(times, lo), bisect.bisect_right(times, hi)
        best = None
        for t_c, kind, score in cands[i0:i1]:
            adj = score - 0.3 * abs((t_c - t) - ideal)     # tiny pull toward mid-length clips
            if best is None or adj > best[0]:
                best = (adj, t_c, kind, score)
        snapped = False
        if best:
            _, cut, kind, score = best
            if kind not in ("sentence", "word"):           # those already sit in a pause between words
                cut, snapped = snap_out_of_word(cut, words, starts, lo, hi)
        else:
            cut, kind, score = hi, "hard", 0.0
            cut, snapped = snap_out_of_word(cut, words, starts, lo, hi)
        seg = {"start": round(t, 3), "end": round(cut, 3), "signal": kind, "score": round(score, 1)}
        if snapped:
            seg["snapped"] = True
        segs.append(seg)
        t = cut
    return segs


# ======================================================================= PHASE 5
# Shareable clips: captions, 9:16 reframing, highlight scoring. (Batch queue is at the bottom.)

def has_filter(name):
    """True when this FFmpeg build has the named filter (e.g. 'ass' needs libass)."""
    out = run(["ffmpeg", "-hide_banner", "-filters"], check=False).stdout
    return re.search(rf"^\s*\S+\s+{re.escape(name)}\s", out, re.M) is not None


# ------------------------------------------------------------------ render options
def render_options(args):
    """Everything about HOW a clip is rendered, read from args (all optional, safe defaults)."""
    cap = getattr(args, "captions", "off") or "off"
    return {
        "fast": bool(getattr(args, "fast", False)),
        "captions": cap,
        "burn": cap in ("burn", "both"),
        "sidecar": cap in ("sidecar", "both"),
        "cap_style": getattr(args, "caption_style", "words"),
        "cap_words": int(getattr(args, "caption_words", 0) or 0),
        "cap_font": getattr(args, "caption_font", "Arial"),
        "cap_scale": float(getattr(args, "caption_scale", 1.0) or 1.0),
        "cap_color": getattr(args, "caption_color", "#FFD400"),
        "cap_pos": getattr(args, "caption_pos", "bottom"),
        "cap_upper": bool(getattr(args, "caption_upper", False)),
        "reframe": getattr(args, "vertical", "off") or "off",
    }


def render_signature(args):
    """Small dict stored on every exported clip. If it no longer matches the current options,
    the clip is out of date and gets re-encoded; if it matches, the clip is left alone."""
    o = render_options(args)
    sig = {"fast": o["fast"], "reframe": o["reframe"], "burn": o["burn"]}
    if o["burn"]:
        sig.update(style=o["cap_style"], words=o["cap_words"], font=o["cap_font"], scale=o["cap_scale"],
                   color=o["cap_color"], pos=o["cap_pos"], upper=o["cap_upper"])
    return sig


def wants_sidecar(args):
    return render_options(args)["sidecar"]


def sidecar_path(out_dir, c):
    return Path(out_dir) / Path(c["file"]).with_suffix(".srt").name


def clip_state(c, out_dir, sig=None, want_sidecar=False):
    """'ok' (nothing to do) | 'encode' (needs (re-)encoding) | 'sidecar' (only the .srt is missing)."""
    if not (c.get("done") and (Path(out_dir) / c["file"]).exists()):
        return "encode"
    if sig is not None and (c.get("render") if c.get("render") is not None else LEGACY_SIG) != sig:
        return "encode"
    if want_sidecar and not (c.get("sidecar") == SIDECAR_VER and sidecar_path(out_dir, c).exists()):
        return "sidecar"
    return "ok"


# ---------------------------------------------------------------------- captions
def load_words(out_dir, fp=None):
    """Word timestamps from the cached Phase 3 transcript (None if missing or for another video)."""
    p = Path(out_dir) / TRANSCRIPT_NAME
    if not p.exists():
        return None
    try:
        tr = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if fp:
        key = tr.get("key", {})
        if any(key.get(f) != fp.get(f) for f in ("source", "size", "mtime")):
            return None
    return sorted(tr.get("words") or [], key=lambda w: w["s"]) or None


def clip_words(words, starts, start, end):
    """The words that belong to a clip, with times RELATIVE to the clip start."""
    out = []
    for w in words[bisect.bisect_left(starts, start - 0.5):]:
        if w["s"] > end:
            break
        if start <= (w["s"] + w["e"]) / 2 <= end:
            s = max(w["s"], start) - start
            e = min(w["e"], end) - start
            out.append({"w": w["w"], "s": s, "e": max(e, s + 0.05)})
    return out


def caption_chunks(words, max_words, max_chars, max_dur=4.0, gap_break=0.7):
    """Group words into on-screen captions: break at sentence ends, long pauses, or size limits."""
    chunks, cur = [], []
    for w in words:
        if cur:
            chars = sum(len(x["w"]) + 1 for x in cur) + len(w["w"])
            last = cur[-1]["w"].rstrip("\"')]”’").lower()
            ends = last.endswith((".", "?", "!", "…")) and last not in ABBREV
            if (ends or w["s"] - cur[-1]["e"] > gap_break or len(cur) >= max_words
                    or chars > max_chars or w["e"] - cur[0]["s"] > max_dur):
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def _ass_time(t):
    cs = max(0, int(round(t * 100)))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _srt_time(t):
    ms = max(0, int(round(t * 1000)))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _hex_rgb(hex_str, default):
    h = (hex_str or default).lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        h = default
    return h[0:2], h[2:4], h[4:6]


def write_ass(path, rel_words, out_w, out_h, o):
    """Burn-in subtitles. Style 'words' highlights the word being spoken (TikTok look);
    'phrase' shows each chunk as plain static text."""
    portrait = out_h > out_w
    size = max(12, round(out_h * (0.038 if portrait else 0.055) * o["cap_scale"]))
    outline = max(2, round(size * 0.075))
    align = {"bottom": 2, "middle": 5, "top": 8}.get(o["cap_pos"], 2)
    margin_v = round(out_h * {"bottom": 0.17 if portrait else 0.06, "middle": 0.0, "top": 0.10}.get(o["cap_pos"], 0.1))
    margin_h = round(out_w * (0.08 if portrait else 0.05))
    max_words = o["cap_words"] or (4 if portrait else 7)
    max_chars = 22 if portrait else 42
    r, g, b = _hex_rgb(o["cap_color"], "FFD400")
    hl = f"&H{b}{g}{r}&".upper()

    def prep(t):
        t = t.upper() if o["cap_upper"] else t
        return t.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")

    chunks = caption_chunks(rel_words, max_words, max_chars)
    ev = []
    for ci, ch in enumerate(chunks):
        nxt = chunks[ci + 1][0]["s"] if ci + 1 < len(chunks) else None
        if o["cap_style"] == "phrase":
            s, e = ch[0]["s"], ch[-1]["e"] + 0.25
            if nxt is not None:
                e = min(e, nxt)
            e = max(e, s + 0.1)
            ev.append((s, e, " ".join(prep(x["w"]) for x in ch)))
            continue
        for k, w in enumerate(ch):
            s = w["s"]
            if k + 1 < len(ch):
                e = ch[k + 1]["s"] if ch[k + 1]["s"] - w["e"] <= 0.5 else w["e"] + 0.15
            else:
                e = w["e"] + 0.25
                if nxt is not None:
                    e = min(e, nxt)
            e = max(e, s + 0.08)
            text = " ".join(("{\\1c" + hl + "}" + prep(x["w"]) + "{\\r}") if j == k else prep(x["w"])
                            for j, x in enumerate(ch))
            ev.append((s, e, text))
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {out_w}", f"PlayResY: {out_h}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{o['cap_font']},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,"
        f"{outline},1,{align},{margin_h},{margin_h},{margin_v},1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    lines += [f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cap,,0,0,0,,{t}" for s, e, t in ev]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_srt(path, rel_words):
    """Sidecar subtitles for uploading next to the video (YouTube, TikTok, Facebook all accept .srt)."""
    chunks = caption_chunks(rel_words, 12, 42, max_dur=5.5, gap_break=0.8)
    out = []
    for i, ch in enumerate(chunks, 1):
        s, e = ch[0]["s"], ch[-1]["e"] + 0.2
        if i < len(chunks):
            e = min(e, chunks[i][0]["s"])
        out.append(f"{i}\n{_srt_time(s)} --> {_srt_time(max(e, s + 0.1))}\n{' '.join(x['w'] for x in ch)}\n")
    Path(path).write_text("\n".join(out), encoding="utf-8")


# --------------------------------------------------------------------- 9:16 reframing
TRACK_FPS = 4        # motion samples per second
TRACK_COLS = 128     # horizontal resolution of the motion profile


def motion_center(row):
    """One motion profile (mean frame-to-frame change per column, 0..255) -> where the action is,
    as 0..1 of the frame width, or None when nothing clearly moves. Subtracting the median column
    ignores noise and whole-frame changes (camera moves, scene cuts), which move every column alike."""
    base = sorted(row)[len(row) // 2]
    w = [max(0, v - base - 3) for v in row]
    total = sum(w)
    if total < 20:
        return None
    return sum((i + 0.5) * x for i, x in enumerate(w)) / total / len(row)


def sample_motion(source, start, dur):
    """-> [(t_relative, x of the main motion as 0..1 of the width, or None)] every 1/TRACK_FPS s.
    One cheap FFmpeg pass: tiny grayscale frames, differenced against the previous sample, squashed
    to a single row of per-column energy. No OpenCV, no numpy, no model files. In a static-camera
    talking-head or podcast shot, the motion IS the speaker."""
    vf = (f"fps={TRACK_FPS},scale={TRACK_COLS}:72:flags=area,format=gray,"
          f"tblend=all_mode=difference,scale={TRACK_COLS}:1:flags=area")
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", str(source),
           "-an", "-vf", vf, "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    p = _run_cancellable(cmd, check=False, text=False)
    if p.returncode != 0:
        return []
    data = p.stdout
    # tblend has no 'previous frame' for the first sample, so output k describes time (k + 1) / fps
    return [((k + 1) / TRACK_FPS, motion_center(list(data[k * TRACK_COLS:(k + 1) * TRACK_COLS])))
            for k in range(len(data) // TRACK_COLS)]


def crop_keys(samples, crop_frac, dz_frac=0.18, move_s=0.8, min_hold=1.0, max_keys=60):
    """
    Turn motion samples into a calm virtual-camera path: [(t, f)] where f in 0..1 is how far the crop
    window sits along its travel (0 = far left, 1 = far right). The camera holds still while the
    speaker stays inside a dead zone and glides over `move_s` seconds when they really move.
    Too little motion to trust -> centered.
    """
    travel = 1.0 - crop_frac
    found = [(t, x) for t, x in samples if x is not None]
    if travel <= 0.001 or len(found) < max(2, int(0.15 * len(samples))):
        return [(0.0, 0.5)]
    # fill gaps with the nearest sample that saw motion, then median-filter (9 samples ~ 2 s)
    ft = [t for t, _ in found]
    filled = []
    for t, x in samples:
        if x is None:
            i = bisect.bisect_left(ft, t)
            cands = [found[j] for j in (i - 1, i) if 0 <= j < len(found)]
            x = min(cands, key=lambda tx: abs(tx[0] - t))[1]
        filled.append((t, x))
    xs = [x for _, x in filled]
    sm = [sorted(xs[max(0, i - 4):i + 5])[len(xs[max(0, i - 4):i + 5]) // 2] for i in range(len(xs))]

    def f_of(p):
        return min(1.0, max(0.0, (p - crop_frac / 2) / travel))

    dz = dz_frac * crop_frac
    while True:
        p = sm[0]
        keys = [(0.0, f_of(p))]
        last_end = 0.0
        for i in range(1, len(sm)):
            t = filled[i][0]
            if abs(sm[i] - p) > dz and t >= last_end + min_hold:
                dest = sm[min(len(sm) - 1, i + 5)]           # aim at where the subject SETTLES (~1.2 s on)
                keys.append((t, f_of(p)))                    # hold until now ...
                keys.append((t + move_s, f_of(dest)))        # ... then one smooth glide
                p, last_end = dest, t + move_s
        if len(keys) <= max_keys or dz > crop_frac:
            return keys[:max_keys]
        dz *= 1.5


def piecewise_expr(keys):
    """ffmpeg expression in t for a piecewise-linear path through keys [(t, value)]."""
    expr = f"{keys[-1][1]:.4f}"
    for (t0, f0), (t1, f1) in reversed(list(zip(keys, keys[1:]))):
        if t1 - t0 < 1e-3:
            continue
        seg = f"{f0:.4f}" if abs(f1 - f0) < 1e-4 else f"{f0:.4f}+({f1 - f0:.4f})*(t-{t0:.3f})/{t1 - t0:.3f}"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return expr


def reframe_crop_filter(keys):
    """Crop a 9:16 window out of the frame (x follows `keys`), then scale to 1080x1920."""
    return (f"crop=w='trunc(ih*9/16/2)*2':h=ih:x='(iw-ow)*({piecewise_expr(keys)})':y=0,"
            f"scale={VERT_W}:{VERT_H}:flags=lanczos,setsar=1")


def reframe_blur_filter():
    """Whole frame kept, centered on a blurred, zoomed copy of itself (nothing is cropped away)."""
    w4, h4 = VERT_W // 4, VERT_H // 4
    return (f"split[rf_a][rf_b];"
            f"[rf_a]scale={w4}:{h4}:force_original_aspect_ratio=increase,crop={w4}:{h4},boxblur=8:2,"
            f"scale={VERT_W}:{VERT_H}[rf_bg];"
            f"[rf_b]scale={VERT_W}:-2[rf_fg];[rf_bg][rf_fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def follow_keys_for_clip(c, source, meta):
    """Auto-pan path for one clip; remembered in the manifest so re-exports do not re-analyse."""
    start, end = c["start"], c["end"]
    tr = c.get("track")
    if tr and abs(tr["start"] - start) < 1e-3 and abs(tr["end"] - end) < 1e-3:
        return [tuple(k) for k in tr["keys"]]
    samples = sample_motion(source, start, end - start)
    crop_w = int(meta["height"] * 9 / 16) // 2 * 2
    keys = crop_keys(samples, crop_w / meta["width"])
    c["track"] = {"start": start, "end": end, "keys": [[round(t, 3), round(f, 4)] for t, f in keys]}
    return keys


def prepare_clip_render(c, cut_source, meta, ro, words, wstarts, work_dir, vertical_on):
    """-> (extra video filters, ass file name or None) for one clip."""
    extra, ass_name = [], None
    out_w, out_h = meta["width"], meta["height"]
    if vertical_on:
        out_w, out_h = VERT_W, VERT_H
        if ro["reframe"] == "blur":
            extra.append(reframe_blur_filter())
        else:
            forced = c.get("crop_x")
            if forced is not None:
                keys = [(0.0, min(1.0, max(0.0, forced / 100.0)))]
            elif ro["reframe"] == "follow":
                keys = follow_keys_for_clip(c, cut_source, meta)
            else:
                keys = [(0.0, 0.5)]
            extra.append(reframe_crop_filter(keys))
    if ro["burn"] and words:
        rel = clip_words(words, wstarts, c["start"], c["end"])
        if rel:
            ass_name = Path(c["file"]).stem + ".ass"
            write_ass(work_dir / ass_name, rel, out_w, out_h, ro)
    return extra, ass_name


def write_sidecar(c, out_dir, words, wstarts):
    rel = clip_words(words, wstarts, c["start"], c["end"]) if words else []
    write_srt(sidecar_path(out_dir, c), rel)      # an empty file when the clip has no speech
    c["sidecar"] = SIDECAR_VER


# --------------------------------------------------------------- highlight scoring
HOOK_WORDS = {
    "secret", "mistake", "mistakes", "never", "always", "nobody", "everyone", "actually", "truth", "crazy",
    "insane", "wow", "amazing", "shocking", "biggest", "worst", "best", "why", "free", "stop", "warning",
    "danger", "problem", "hack", "trick", "lesson", "rule", "million", "billion", "important", "fail",
    "failed", "wrong", "wait", "listen", "imagine", "believe", "honestly", "literally", "unbelievable",
    # Filipino / Taglish
    "grabe", "totoo", "sikreto", "mali", "huwag", "bakit", "paano", "importante", "pera", "sana", "talaga",
}
HOOK_PHRASES = ("how to", "what if", "the truth", "did you know", "here's the", "you won't believe",
                "the problem is", "the reason", "number one", "game changer", "let me tell you")
HOOK_START = re.compile(r"^(what if|imagine|did you know|here'?s|here is|the (biggest|worst|best|only|real|first)"
                        r"|never |stop |why |how |you |let me|the truth|listen|look,)", re.I)
BOUNDARY_Q = {"sentence": 1.0, "end": 1.0, "silence": 0.7, "scene": 0.6, "manual": 0.6,
              "fixed": 0.4, "dip": 0.4, "word": 0.3, "hard": 0.0}
SCORE_WEIGHTS = {"hook": 0.25, "energy": 0.20, "dyn": 0.15, "pace": 0.15, "density": 0.10,
                 "boundary": 0.10, "visual": 0.05}
SCORE_LABELS = {"hook": "hook words", "energy": "loudness", "dyn": "dynamics", "pace": "pace",
                "density": "speech density", "boundary": "clean cuts", "visual": "visual change"}


def parse_keywords(text):
    return {k.strip().lower() for k in (text or "").split(",") if k.strip()}


def hook_score(text, extra=()):
    """0..1: how 'hooky' the words are (questions, emphasis, power words, a strong opening line)."""
    t = text.lower().strip()
    if not t:
        return 0.0
    kws = HOOK_WORDS | set(extra)
    hits = {w for w in re.findall(r"[\w']+", t) if w in kws} | {p for p in HOOK_PHRASES if p in t}
    first = re.split(r"(?<=[.?!])\s+", t, maxsplit=1)[0]
    s = (0.17 * min(t.count("?"), 3) + 0.12 * min(t.count("!"), 3) + 0.12 * min(len(hits), 4)
         + (0.25 if HOOK_START.match(first) else 0.0) + (0.07 if re.search(r"\d", t) else 0.0)
         + (0.08 if len(re.findall(r"\b(you|your|you're)\b", t)) >= 2 else 0.0))
    return min(1.0, s)


def _pct_ranks(vals):
    s = sorted(vals)
    n = len(s)
    return [(bisect.bisect_left(s, v) + bisect.bisect_right(s, v)) / 2 / n for v in vals]


def _load_analysis(out_dir, src=None):
    """analysis.json, with the loudness series filled in if it came from an older version."""
    p = Path(out_dir) / ANALYSIS_NAME
    if not p.exists():
        return None
    try:
        an = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    if not an.get("rms") and src and Path(src).is_file():
        series = pass_rms(src)
        an["rms"] = pack_rms(series)
        with contextlib.suppress(OSError):
            p.write_text(json.dumps(an))
    return an


def score_clips(manifest, out_dir, extra_keywords=(), src=None):
    """
    Give every clip a 0-100 'highlight' score from signals we already have, all local and fast:
      hook words / questions / strong opening line   (transcript)
      loudness and loudness range (energetic speech)  (audio)
      speaking pace and how much of the clip is speech (transcript)
      clean start/end boundaries, and visual changes  (planner / scene cuts)
    Continuous signals are turned into percentile ranks within THIS video, so 'loud' means loud for
    this recording. A missing signal (no transcript, no audio) is skipped, not counted as zero.
    Stores h_score, h_rank and h_why on each clip.
    """
    out_dir = Path(out_dir)
    src = src or manifest.get("fingerprint", {}).get("source")
    an = _load_analysis(out_dir, src)
    words = load_words(out_dir, manifest.get("fingerprint"))
    starts = [w["s"] for w in words] if words else []
    rms = (an or {}).get("rms")
    scenes = (an or {}).get("scenes") or []
    clips = sorted(manifest["clips"], key=lambda c: c["start"])
    feats = []
    for i, c in enumerate(clips):
        length = max(0.1, c["end"] - c["start"])
        f = {}
        if rms and rms.get("db"):
            i0 = max(0, int((c["start"] - rms["t0"]) / rms["step"]))
            i1 = min(len(rms["db"]), int((c["end"] - rms["t0"]) / rms["step"]) + 1)
            seg = sorted(rms["db"][i0:i1])
            if len(seg) >= 5:
                f["energy"] = sum(seg[len(seg) // 2:]) / len(seg[len(seg) // 2:])
                f["dyn"] = seg[int(0.95 * (len(seg) - 1))] - seg[int(0.25 * (len(seg) - 1))]
        if words is not None:
            cw = clip_words(words, starts, c["start"], c["end"])
            speech = sum(min(w["e"] - w["s"], 1.0) for w in cw)
            f["density"] = min(1.0, speech / length)
            f["pace"] = len(cw) / max(speech, 1.0)
            f["hook"] = hook_score(" ".join(w["w"] for w in cw), extra_keywords)
        if an is not None:
            n_sc = bisect.bisect_right(scenes, c["end"]) - bisect.bisect_left(scenes, c["start"])
            f["visual"] = min(4, n_sc) / 4
        prev_sig = clips[i - 1]["signal"] if i else "end"
        f["boundary"] = (BOUNDARY_Q.get(prev_sig, 0.4) + BOUNDARY_Q.get(c["signal"], 0.4)) / 2
        feats.append(f)
    for key in ("energy", "dyn", "pace", "density"):          # raw -> percentile within this video
        idx = [i for i, f in enumerate(feats) if key in f]
        for i, r in zip(idx, _pct_ranks([feats[i][key] for i in idx])):
            feats[i][key] = r
    for c, f in zip(clips, feats):
        wsum = sum(SCORE_WEIGHTS[k] for k in f)
        c["h_score"] = round(100.0 * sum(SCORE_WEIGHTS[k] * v for k, v in f.items()) / wsum, 1) if wsum else 0.0
        c["h_why"] = {SCORE_LABELS[k]: round(v, 2) for k, v in f.items()}
    for rank, c in enumerate(sorted(clips, key=lambda c: -c["h_score"]), 1):
        c["h_rank"] = rank
    return manifest


def select_highlights(manifest, out_dir, top=None, pct=None, min_score=None):
    """
    Keep only the best clips; the rest are marked 'auto_dropped' (not exported, restorable).
    top = best N clips, pct = best P percent of clips, min_score = at least this score; combine freely.
    Clips YOU deleted by hand stay deleted. Re-running with different numbers is safe.
    Returns (kept, total).
    """
    pool = [c for c in manifest["clips"] if not (c.get("deleted") and not c.get("auto_dropped"))]
    if any(c.get("h_score") is None for c in pool):
        raise ClipperError("These clips have no highlight scores yet.")
    ranked = sorted(pool, key=lambda c: -c["h_score"])
    n = len(ranked)
    limit = n
    if top is not None:
        limit = min(limit, max(1, int(top)))
    if pct is not None:
        limit = min(limit, max(1, math.ceil(n * float(pct) / 100.0)))
    keep = {c["index"] for c in ranked[:limit] if min_score is None or c["h_score"] >= float(min_score)}
    for c in pool:
        if c["index"] in keep:
            if c.get("auto_dropped"):
                c["deleted"], c["auto_dropped"], c["done"] = False, False, False
        elif not c.get("deleted"):
            c["deleted"], c["auto_dropped"], c["done"] = True, True, False
            _drop_file(out_dir, c)
    save_job(out_dir, manifest)
    return len(keep), n


def show_all_clips(manifest, out_dir):
    """Undo select_highlights: bring back every auto-dropped clip."""
    for c in manifest["clips"]:
        if c.get("auto_dropped"):
            c["deleted"], c["auto_dropped"], c["done"] = False, False, False
    save_job(out_dir, manifest)


# ---------------------------------------------------------------- manifest
def load_manifest(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            print("  (manifest unreadable, starting fresh)")
    return None


def save_manifest(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)  # atomic: a crash never leaves a half-written manifest


def fingerprint(src):
    st = os.stat(src)
    return {
        "source": str(Path(src).resolve()),
        "size": st.st_size,
        "mtime": int(st.st_mtime),
    }


def plan_fingerprint(src, args):
    return dict(fingerprint(src), mode=args.mode, clip_len=args.clip_len, min_len=args.min_len,
                silence_min=args.silence_min, scene_thresh=args.scene_thresh,
                silence_db=args.silence_db, fade=args.fade,
                transcribe=not args.skip_transcription, whisper_model=args.whisper_model,
                language=args.language)


# --------------------------------------------------------------------- cut
def cut_clip(source, c, enc_args, has_audio, out_path, fast=False, extra_vf=None, ass_name=None, cwd=None):
    """
    Precise cut. -ss BEFORE -i = fast seek; because we re-encode, ffmpeg still
    trims to the exact frame, and the encoder starts a fresh keyframe at 0.
    Clips touching a hard cut get a short fade so the jump is not abrupt.
    extra_vf = 9:16 reframing filters, ass_name = subtitle file (relative to cwd) to burn in.
    """
    dur = c["end"] - c["start"]
    source, out_path = Path(source).resolve(), Path(out_path).resolve()   # cwd may differ from ours
    if fast:
        # FAST mode: stream copy. Near-instant, but video can only start on a keyframe, so the
        # real start/length can be off by a few seconds, and no fades are possible.
        part = out_path.with_name(out_path.name + ".part")
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{c['start']:.3f}", "-i", str(source),
               "-t", f"{dur:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero",
               "-movflags", "+faststart", "-f", "mp4", str(part)]
        try:
            r = run(cmd, check=False)
        except Cancelled:
            part.unlink(missing_ok=True)
            raise
        if r.returncode != 0:
            part.unlink(missing_ok=True)
            raise ClipperError(f"Cut failed at {fmt_time(c['start'])}:\n{r.stderr.strip()}")
        part.replace(out_path)
        return
    fade = c.get("fade", 0.0)
    vf, af = list(extra_vf or []), []
    if ass_name:                      # burned in before the fades, so captions fade with the picture
        vf.append(f"ass=filename='{ass_name}'")
    if fade > 0:
        if c.get("fade_in"):
            vf.append(f"fade=t=in:st=0:d={fade}")
            af.append(f"afade=t=in:st=0:d={fade}")
        if c.get("fade_out"):
            vf.append(f"fade=t=out:st={max(0.0, dur - fade):.3f}:d={fade}")
            af.append(f"afade=t=out:st={max(0.0, dur - fade):.3f}:d={fade}")
    part = out_path.with_name(out_path.name + ".part")
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{c['start']:.3f}", "-i", str(source), "-t", f"{dur:.3f}"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += [*enc_args, "-pix_fmt", "yuv420p"]     # 10-bit / 4:2:2 sources would not play everywhere
    if has_audio:
        if af:
            cmd += ["-af", ",".join(af)]
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", "-f", "mp4", str(part)]
    try:
        r = run(cmd, check=False, cwd=cwd)
    except Cancelled:
        part.unlink(missing_ok=True)
        raise
    if r.returncode != 0:
        part.unlink(missing_ok=True)
        raise ClipperError(f"Cut failed at {fmt_time(c['start'])}:\n{r.stderr.strip()}")
    part.replace(out_path)


# ----------------------------------------------------------------- reporting
def summarize(clips):
    by = {}
    for c in clips:
        by[c["signal"]] = by.get(c["signal"], 0) + 1
    order = ["sentence", "silence", "scene", "dip", "word", "hard", "end", "fixed"]
    parts = [f"{by[k]} {k}" for k in order if k in by]
    return ", ".join(parts)


def print_plan(clips):
    print(f"\nPlan: {len(clips)} clips  ({summarize(clips)})")
    hard = [c for c in clips if c["signal"] == "hard"]
    if hard:
        print(f"  {len(hard)} clips end on a hard cut (no pause/scene/dip found) - worth a quick look:")
        for c in hard[:10]:
            print(f"    {c['file']}  ends at {fmt_time(c['end'])}")
        if len(hard) > 10:
            print(f"    ... and {len(hard) - 10} more (see manifest.json)")


# ------------------------------------------------------- review / edit API
# Used by the desktop UI (clipper_ui.py). Every function edits manifest.json, so the next
# export only re-encodes what changed.
MIN_EDIT_LEN = 1.0


def load_job(out_dir):
    return load_manifest(Path(out_dir) / MANIFEST_NAME)


def save_job(out_dir, manifest):
    save_manifest(Path(out_dir) / MANIFEST_NAME, manifest)


def live_clips(manifest):
    return sorted((c for c in manifest["clips"] if not c.get("deleted")), key=lambda c: c["start"])


def pending_clips(manifest, out_dir, sig=None, want_sidecar=False):
    """Clips that still need work: encoding (never exported, or exported with other render options)
    or just their .srt. Pass sig=render_signature(args) to also catch clips that are out of date."""
    return [c for c in live_clips(manifest) if clip_state(c, out_dir, sig, want_sidecar) != "ok"]


def _find(manifest, index):
    for c in manifest["clips"]:
        if c["index"] == index:
            return c
    raise ClipperError(f"No clip #{index}.")


def _drop_file(out_dir, c):
    for name in (c["file"], c["file"] + ".part", sidecar_path(out_dir, c).name):
        (Path(out_dir) / name).unlink(missing_ok=True)


def _invalidate(c, rescore=False):
    """Forget everything derived from the clip's old cut points."""
    for k in ("render", "sidecar") + (("h_score", "h_rank", "h_why") if rescore else ()):
        c.pop(k, None)


def refresh_text(manifest, out_dir):
    tp = Path(out_dir) / TRANSCRIPT_NAME
    if not tp.exists():
        return
    try:
        words = json.loads(tp.read_text()).get("words", [])
    except json.JSONDecodeError:
        return
    for c in manifest["clips"]:
        c["text"] = text_between(words, c["start"], c["end"])


def edit_trim(manifest, out_dir, index, start, end, min_len=15.0, max_len=30.0):
    """Change a clip's start/end. Returns a list of warnings (never blocks, except invalid times)."""
    c = _find(manifest, index)
    start, end = round(float(start), 3), round(float(end), 3)
    dur = manifest.get("duration")
    if start < 0 or end <= start:
        raise ClipperError("End must be after start.")
    if dur and end > dur + 0.01:
        raise ClipperError(f"End is past the end of the video ({fmt_time(dur)}).")
    if end - start < MIN_EDIT_LEN:
        raise ClipperError(f"A clip must be at least {MIN_EDIT_LEN:.0f} second long.")
    warnings = []
    length = end - start
    if length < min_len or length > max_len:
        warnings.append(f"Clip is {length:.1f}s, outside your {min_len:.0f}-{max_len:.0f}s range.")
    for o in live_clips(manifest):
        if o["index"] != index and min(end, o["end"]) - max(start, o["start"]) > 0.05:
            warnings.append(f"Overlaps clip #{o['index']}.")
    c.update(start=start, end=end, signal="manual", score=0.0, done=False,
             fade=0.0, fade_in=False, fade_out=False)
    c.pop("snapped", None)
    _invalidate(c, rescore=True)
    _drop_file(out_dir, c)
    refresh_text(manifest, out_dir)
    save_job(out_dir, manifest)
    return warnings


def edit_delete(manifest, out_dir, index):
    c = _find(manifest, index)
    c["deleted"], c["done"] = True, False
    c.pop("auto_dropped", None)             # a hand-made delete is never undone by highlight selection
    _drop_file(out_dir, c)
    save_job(out_dir, manifest)


def edit_restore(manifest, out_dir, index):
    c = _find(manifest, index)
    c["deleted"], c["done"] = False, False
    c.pop("auto_dropped", None)
    save_job(out_dir, manifest)


def edit_merge_next(manifest, out_dir, index, max_len=30.0):
    """Join a clip with the next live clip. Returns a list of warnings."""
    c = _find(manifest, index)
    live = live_clips(manifest)
    pos = next((i for i, x in enumerate(live) if x["index"] == index), None)
    if pos is None:
        raise ClipperError("That clip is deleted; restore it first.")
    if pos + 1 >= len(live):
        raise ClipperError("This is the last clip; there is nothing to merge with.")
    n = live[pos + 1]
    c.update(end=n["end"], signal="manual", score=0.0, done=False, fade_out=n.get("fade_out", False))
    c["fade"] = n.get("fade", 0.0) if c["fade_out"] else (c.get("fade", 0.0) if c.get("fade_in") else 0.0)
    c.pop("snapped", None)
    _invalidate(c, rescore=True)
    _drop_file(out_dir, c)
    _drop_file(out_dir, n)
    manifest["clips"].remove(n)
    refresh_text(manifest, out_dir)
    save_job(out_dir, manifest)
    length = c["end"] - c["start"]
    return [f"Merged clip is {length:.1f}s (longer than {max_len:.0f}s)."] if length > max_len else []


def edit_reexport(manifest, out_dir, index):
    c = _find(manifest, index)
    c["done"] = False
    _invalidate(c)
    c.pop("track", None)                  # also re-detect the speaker
    _drop_file(out_dir, c)
    save_job(out_dir, manifest)


def edit_set_crop(manifest, out_dir, index, pct):
    """Pin the 9:16 crop window of one clip: 0 = far left, 50 = center, 100 = far right.
    None (or a negative number) goes back to automatic."""
    c = _find(manifest, index)
    if pct is None or float(pct) < 0:
        c.pop("crop_x", None)
    else:
        c["crop_x"] = round(min(100.0, float(pct)), 1)
    c["done"] = False
    _invalidate(c)
    _drop_file(out_dir, c)
    save_job(out_dir, manifest)


def cleanup_work(out_dir):
    shutil.rmtree(Path(out_dir) / ".work", ignore_errors=True)


def purge_clips(out_dir):
    """Delete exported clips and their .srt files (used before planning again with new settings)."""
    d = Path(out_dir)
    if d.is_dir():
        for pattern in ("clip_*.mp4", "clip_*.mp4.part", "clip_*.srt"):
            for p in d.glob(pattern):
                with contextlib.suppress(OSError):
                    p.unlink()


# -------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description="Split a long video into 15-30 s clips at natural boundaries.")
    ap.add_argument("input", nargs="*", help="video file(s) or folder(s); several = batch queue")
    ap.add_argument("-o", "--out", default=None,
                    help="output folder (default: <input name>_clips). With several inputs: parent folder for all of them")
    ap.add_argument("--mode", choices=["smart", "fixed"], default="smart",
                    help="smart = cut at pauses/scenes (default); fixed = every --clip-len seconds")
    ap.add_argument("--clip-len", type=float, default=30.0, help="longest clip in seconds (default 30)")
    ap.add_argument("--min-len", type=float, default=15.0, help="shortest clip in seconds (default 15)")
    ap.add_argument("--silence-min", type=float, default=0.4, help="shortest pause that counts, seconds (default 0.4)")
    ap.add_argument("--silence-db", type=float, default=None, help="force silence threshold in dB (default: auto)")
    ap.add_argument("--scene-thresh", type=float, default=0.3, help="scene-change sensitivity 0-1, lower = more cuts (default 0.3)")
    ap.add_argument("--fade", type=float, default=0.25, help="fade length at hard cuts, seconds (default 0.25, 0 = off)")
    ap.add_argument("--skip-transcription", action="store_true", help="do not run Whisper (pause/scene cuts only)")
    ap.add_argument("--whisper-model", default="base", help="tiny, base (default), small, medium ... (bigger = slower)")
    ap.add_argument("--language", default=None, help="speech language code, e.g. en, tl (default: auto-detect)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="where Whisper runs (default auto)")
    ap.add_argument("--fast", action="store_true",
                    help="FAST export: stream copy, near-instant but cuts snap to keyframes (off by seconds), no fades")
    ap.add_argument("--plan-only", action="store_true", help="analyse and print the cut list, do not encode")
    ap.add_argument("--reanalyze", action="store_true", help="ignore the cached analysis")
    ap.add_argument("--encoder", choices=[e[0] for e in ENCODERS], help="force an encoder")
    ap.add_argument("--no-normalize", action="store_true", help="skip VFR normalization")
    ap.add_argument("--restart", action="store_true", help="ignore any saved progress")
    g = ap.add_argument_group("Phase 5: captions, 9:16, highlights")
    g.add_argument("--captions", choices=["off", "burn", "sidecar", "both"], default="off",
                   help="burn = hard-coded into the video, sidecar = clip_0001.srt next to it, both")
    g.add_argument("--caption-style", choices=["words", "phrase"], default="words",
                   help="words = highlight the spoken word (default), phrase = plain static text")
    g.add_argument("--caption-words", type=int, default=0, help="max words on screen at once (default auto: 4 vertical / 7 wide)")
    g.add_argument("--caption-font", default="Arial", help="font family for burned captions (default Arial)")
    g.add_argument("--caption-scale", type=float, default=1.0, help="caption size multiplier (default 1.0)")
    g.add_argument("--caption-color", default="#FFD400", help="highlight colour for the spoken word (default #FFD400)")
    g.add_argument("--caption-pos", choices=["bottom", "middle", "top"], default="bottom")
    g.add_argument("--caption-upper", action="store_true", help="UPPERCASE captions")
    g.add_argument("--vertical", choices=["off", "center", "follow", "blur"], default="off",
                   help="9:16 output: center = middle crop, follow = auto-pan to where the action is "
                        "(the speaker, in a talking-head shot), blur = whole frame over a blurred background")
    g.add_argument("--highlights", type=int, default=None, metavar="N", help="export only the best N clips")
    g.add_argument("--highlights-pct", type=float, default=None, metavar="P", help="export only the best P percent of clips")
    g.add_argument("--min-score", type=float, default=None, help="with the above: also require this highlight score (0-100)")
    g.add_argument("--highlight-keywords", default="", help="extra comma-separated words that make a clip more engaging")
    q = ap.add_argument_group("Batch queue (several videos, crash-safe)")
    q.add_argument("--queue-file", default=None, help=f"queue file (default ./{QUEUE_NAME}); giving it turns batch mode on")
    q.add_argument("--resume-queue", action="store_true", help="continue the queue file without adding anything")
    q.add_argument("--retry-failed", action="store_true", help="put failed queue items back in line")
    q.add_argument("--queue-status", action="store_true", help="print the queue and exit")
    return ap


def expand_inputs(items):
    """Files and folders -> a de-duplicated list of video files."""
    files = []
    for it in items or []:
        p = Path(it)
        if p.is_dir():
            files += sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in VIDEO_EXT)
        elif p.is_file():
            files.append(p)
        else:
            raise ClipperError(f"Not found: {it}")
    seen, out = set(), []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def main():
    ap = build_parser()
    args = ap.parse_args()
    try:
        batch = (args.queue_status or args.resume_queue or args.queue_file or len(args.input) > 1
                 or any(Path(i).is_dir() for i in args.input))
        if batch:
            sys.exit(run_batch_cli(args))
        if not args.input:
            ap.error("give a video file (or use --resume-queue / --queue-status)")
        run_job(args)
    except ClipperError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command again to resume.")
        sys.exit(130)


def run_job(args, progress=None, cancel=None):
    """progress(event: dict) and cancel (threading.Event) are optional hooks for the UI.
    Cancel is honoured everywhere: during analysis, transcription, normalization and every clip."""
    global _CANCEL
    _CANCEL = cancel
    try:
        return _run_job(args, progress, cancel)
    finally:
        _CANCEL = None


def _run_job(args, progress=None, cancel=None):
    emit = progress or (lambda e: None)
    inp = args.input[0] if isinstance(args.input, (list, tuple)) and args.input else args.input
    if not inp or isinstance(inp, (list, tuple)):
        raise ClipperError("No input video given.")
    src = Path(inp)
    if not src.is_file():
        raise ClipperError(f"File not found: {src}")
    if not (0 < args.min_len <= args.clip_len):
        raise ClipperError("--min-len must be > 0 and <= --clip-len.")
    if args.mode == "smart" and args.clip_len < 2 * args.min_len:
        print("Note: --clip-len < 2 x --min-len, so smart cuts have very little room to choose from.")
    ro = render_options(args)
    hl_wanted = any(getattr(args, k, None) is not None for k in ("highlights", "highlights_pct", "min_score"))
    if ro["fast"] and (ro["burn"] or ro["reframe"] != "off"):
        raise ClipperError("Fast mode copies the video untouched, so it cannot burn in captions or reframe to 9:16. "
                           "Use Precise mode (sidecar .srt captions work in both).")
    if hl_wanted and args.mode != "smart":
        raise ClipperError("Highlight selection needs --mode smart.")
    need_tool("ffmpeg")
    need_tool("ffprobe")
    if ro["burn"] and not has_filter("ass"):
        raise ClipperError("This FFmpeg build cannot draw subtitles (no libass). Install a 'full' FFmpeg build "
                           "(for Windows: gyan.dev 'full'), or use sidecar captions instead.")

    out_dir = Path(args.out) if args.out else src.with_name(src.stem + "_clips")
    work_dir = out_dir / ".work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    print(f"Input : {src}")
    meta = probe(src)
    print(
        f"        {meta['width']}x{meta['height']} {meta['codec']}, {meta['fps']:.2f} fps, "
        f"{fmt_time(meta['duration'])}, {'VFR' if meta['is_vfr'] else 'constant fps'}"
    )
    if not meta["has_audio"]:
        print("        (no audio track: only scene changes can be used for cuts)")
    vertical_on = bool(ro["reframe"] != "off" and meta["width"] and meta["height"]
                       and meta["width"] / meta["height"] > 0.6)
    if ro["reframe"] != "off" and not vertical_on:
        print("Note: this video is already vertical, so 9:16 reframing is skipped.")

    check_disk_space(out_dir, meta["size_bytes"])
    enc_name, enc_args = pick_encoder(args.encoder)
    print(f"Encoder: {enc_name}")

    manifest_path = out_dir / MANIFEST_NAME
    fp = plan_fingerprint(src, args)
    manifest = None if args.restart else load_manifest(manifest_path)
    if manifest and manifest.get("fingerprint") != fp:
        print("Existing manifest is for a different file or settings, starting fresh.")
        manifest = None

    # ---- constant-frame-rate source (only built when actually needed)
    state = {"source": src}

    def prepare_source():
        if state.get("ready"):
            return state["source"]
        if meta["is_vfr"] and not args.no_normalize:
            norm = work_dir / "normalized.mp4"
            if norm.exists():
                print("Normalized copy found, reusing it.")
            else:
                print("Variable frame rate detected, normalizing (one-time step)...")
                t0 = time.time()
                normalize(src, norm, meta["fps"], enc_args, meta["has_audio"])
                print(f"  done in {fmt_time(time.time() - t0)}")
            state["source"] = norm
        elif meta["is_vfr"]:
            print("Warning: VFR file and --no-normalize set; audio may drift.")
        state["ready"] = True
        return state["source"]

    # ---- plan (analysis only runs when there is no saved plan)
    if manifest is None:
        cut_source = prepare_source()
        words = None
        if args.mode == "smart":
            tr_future = None
            pool = ThreadPoolExecutor(max_workers=1)
            if meta["has_audio"] and not args.skip_transcription:
                # Whisper runs in the background WHILE the pause/scene scans run
                tr_future = pool.submit(get_transcript, cut_source, args, out_dir / TRANSCRIPT_NAME,
                                        work_dir, fingerprint(src), meta["duration"])
            try:
                an = analyze(cut_source, meta, args, out_dir / ANALYSIS_NAME, fingerprint(src))
                if tr_future is not None:
                    tr = tr_future.result()  # transcription must be FINISHED before any export starts
                    words = tr["words"] if tr else None
            finally:
                # on Cancel, wait for the transcription thread (it stops within a segment) so it can
                # never keep running behind the next job
                pool.shutdown(wait=bool(cancel is not None and cancel.is_set()))
            segs = plan_smart(meta["duration"], an, args.clip_len, args.min_len, words)
        else:
            segs = [{"start": s, "end": e, "signal": "fixed", "score": 0.0}
                    for s, e in plan_segments(meta["duration"], args.clip_len, args.min_len)]
        clips = []
        for i, sg in enumerate(segs):
            clips.append(dict(sg, index=i + 1, file=f"clip_{i + 1:04d}.mp4", done=False,
                              fade=0.0, fade_in=False, fade_out=False,
                              text=text_between(words, sg["start"], sg["end"]) if words else ""))
        # fades: a hard cut fades OUT the clip before it and fades IN the clip after it
        if args.fade > 0:
            for i, c in enumerate(clips):
                if c["signal"] == "hard":
                    c["fade"] = c["fade_out"] = args.fade
                    if i + 1 < len(clips):
                        clips[i + 1]["fade"] = args.fade
                        clips[i + 1]["fade_in"] = True
        manifest = {"fingerprint": fp, "encoder": enc_name, "duration": meta["duration"],
                    "summary": summarize(clips), "clips": clips}
        if args.mode == "smart":
            try:
                score_clips(manifest, out_dir, parse_keywords(getattr(args, "highlight_keywords", "")), src)
            except Exception as e:  # scoring is a bonus; never lose the plan over it
                print(f"  (highlight scoring skipped: {type(e).__name__}: {e})")
        save_manifest(manifest_path, manifest)
    clips = manifest["clips"]

    # ---- highlights: keep only the best clips (safe to repeat; scores are stored in the manifest)
    if hl_wanted:
        if any(c.get("h_score") is None for c in clips):
            score_clips(manifest, out_dir, parse_keywords(getattr(args, "highlight_keywords", "")), src)
        kept, total_clips = select_highlights(manifest, out_dir, getattr(args, "highlights", None),
                                              getattr(args, "highlights_pct", None), getattr(args, "min_score", None))
        print(f"Highlights: keeping the best {kept} of {total_clips} clips.")

    emit({"stage": "plan_ready", "manifest": manifest, "out_dir": str(out_dir)})
    print_plan([c for c in clips if not c.get("deleted")])
    if args.plan_only:
        print(f"\nPlan saved to {manifest_path}. (--plan-only: nothing encoded.)")
        if not getattr(args, "keep_work", False):   # the UI keeps the normalized copy for the export step
            shutil.rmtree(work_dir, ignore_errors=True)
        return manifest

    sig = render_signature(args)
    live = [c for c in clips if not c.get("deleted")]
    pending = [c for c in live if clip_state(c, out_dir, sig, ro["sidecar"]) != "ok"]
    done_before = len(live) - len(pending)
    if done_before:
        print(f"\nResuming: {done_before}/{len(live)} clips already finished.")

    # ---- captions need word timestamps (cached Phase 3 transcript, or transcribe now)
    words, wstarts = None, []
    if pending and (ro["burn"] or ro["sidecar"]):
        words = load_words(out_dir, manifest.get("fingerprint"))
        if words is None:
            if not whisper_available():
                raise ClipperError("Captions need a transcript: install faster-whisper "
                                   "(pip install faster-whisper) and make sure transcription is not skipped.")
            print("Captions need a transcript; transcribing now...")
            tr = get_transcript(prepare_source(), args, out_dir / TRANSCRIPT_NAME, work_dir,
                                fingerprint(src), meta["duration"])
            words = tr["words"] if tr else None
            if not words:
                raise ClipperError("No speech was found, so there is nothing to caption.")
        wstarts = [w["s"] for w in words]

    # ---- cut
    fast = ro["fast"]
    need_encode = any(clip_state(c, out_dir, sig, ro["sidecar"]) == "encode" for c in pending)
    cut_source = prepare_source() if need_encode else src
    total = len(live)
    t_start = time.time()
    finished_now = 0
    what = []
    if vertical_on:
        what.append({"center": "9:16 center crop", "follow": "9:16 auto-pan", "blur": "9:16 blurred background"}[ro["reframe"]])
    if ro["burn"]:
        what.append(f"burned captions ({ro['cap_style']})")
    if ro["sidecar"]:
        what.append(".srt files")
    print(f"\nExporting {len(pending)} clip(s) in {'FAST (stream copy)' if fast else 'PRECISE (re-encode)'} mode"
          + (f" + {', '.join(what)}" if what else "") + ".")
    emit({"stage": "export_start", "total": total, "done": done_before})
    for c in pending:
        if cancel is not None and cancel.is_set():
            raise Cancelled("Cancelled. Press Export again to continue where you left off.")
        if clip_state(c, out_dir, sig, ro["sidecar"]) == "encode":
            extra_vf, ass_name = prepare_clip_render(c, cut_source, meta, ro, words, wstarts, work_dir, vertical_on)
            cut_clip(cut_source, c, enc_args, meta["has_audio"], out_dir / c["file"], fast=fast,
                     extra_vf=extra_vf, ass_name=ass_name, cwd=work_dir)
            if ass_name:
                (work_dir / ass_name).unlink(missing_ok=True)
            c["done"], c["render"] = True, sig
        if ro["sidecar"]:
            write_sidecar(c, out_dir, words, wstarts)
        save_manifest(manifest_path, manifest)  # progress saved after EVERY clip
        finished_now += 1
        done_total = done_before + finished_now
        eta = (time.time() - t_start) / finished_now * (total - done_total)
        dur = c["end"] - c["start"]
        print(f"  [{done_total:>4}/{total}] {c['file']}  {fmt_time(c['start'])} +{dur:4.1f}s  {c['signal']:<7} ETA {fmt_time(eta)}")
        emit({"stage": "clip_done", "done": done_total, "total": total, "index": c["index"]})

    # ---- clean up (only reached on success; on crash the work folder is kept for fast resume)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(f"\nDone: {total} clips in '{out_dir}'.")
    emit({"stage": "finished", "total": total})
    return manifest


# ------------------------------------------------------------------- batch queue
# One JSON file lists the videos and the settings each was added with. It is rewritten atomically on
# every state change. A crash leaves at most one item 'running'; the next start puts it back in line and
# run_job resumes it from its own manifest.json (finished clips are never redone). One crash-looping file
# cannot block the rest: after QUEUE_MAX_CRASHES interruptions it is marked failed.
QUEUE_SKIP_OPTS = {"input", "out", "queue_file", "resume_queue", "retry_failed", "queue_status",
                   "plan_only", "keep_work", "restart"}


def load_queue(path):
    path = Path(path)
    if path.exists():
        try:
            q = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(q, dict) and isinstance(q.get("items"), list):
                return q
        except (json.JSONDecodeError, OSError):
            pass
        with contextlib.suppress(OSError):          # unreadable: keep a copy instead of overwriting it
            path.replace(path.with_name(path.name + ".corrupt"))
    return {"version": 1, "items": []}


def save_queue(path, q):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(q, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic: a crash never leaves a half-written queue


def queue_options(args):
    return {k: v for k, v in vars(args).items() if k not in QUEUE_SKIP_OPTS}


def args_for_item(item):
    ns = build_parser().parse_args([item["input"]])
    ns.__dict__.update(item.get("options", {}))
    ns.input, ns.out = [item["input"]], item["out"]
    return ns


def queue_add(q, files, args, out_parent=None):
    """Add videos with the current settings. Returns (added, skipped) lists of readable messages."""
    opts = queue_options(args)
    added, skipped = [], []
    for f in files:
        f = Path(f).resolve()
        out = (Path(out_parent) if out_parent else f.parent) / (f.stem + "_clips")
        out = out.resolve()
        dup = next((i for i in q["items"] if i["input"] == str(f) and i["out"] == str(out)), None)
        if dup is None:
            q["items"].append({"id": uuid.uuid4().hex[:8], "input": str(f), "out": str(out), "options": opts,
                               "status": "queued", "detail": "", "crashes": 0, "added": time.time()})
            added.append(f.name)
        elif dup["status"] == "running":
            skipped.append(f"{f.name}: running right now")
        elif dup["status"] == "done" and dup.get("options") == opts:
            skipped.append(f"{f.name}: already done")
        elif dup["status"] == "queued" and dup.get("options") == opts:
            skipped.append(f"{f.name}: already waiting")
        else:   # failed, or the settings changed: line it up again (finished clips are reused, others redone)
            dup.update(options=opts, status="queued", detail="", crashes=0)
            added.append(f"{f.name} (re-queued)")
    return added, skipped


def queue_retry_failed(q):
    n = 0
    for it in q["items"]:
        if it["status"] == "failed":
            it.update(status="queued", detail="", crashes=0)
            n += 1
    return n


def queue_remove(q, item_id):
    q["items"] = [i for i in q["items"] if not (i["id"] == item_id and i["status"] != "running")]


def sweep_partials(out_dir):
    d = Path(out_dir)
    if d.is_dir():
        for p in list(d.glob("*.part")) + list(d.glob("*.tmp")):
            with contextlib.suppress(OSError):
                p.unlink()


def queue_recover(q):
    """After a crash/kill: 'running' items go back in line (or fail, if they keep killing the app)."""
    for it in q["items"]:
        if it["status"] == "running":
            it["crashes"] = it.get("crashes", 0) + 1
            sweep_partials(it["out"])
            if it["crashes"] >= QUEUE_MAX_CRASHES:
                it.update(status="failed", detail=f"Interrupted {it['crashes']} times in a row; skipped.")
            else:
                it.update(status="queued", detail="Interrupted last time; resuming.")
    return q


def run_queue(queue_path, progress=None, cancel=None):
    """Process queued items one after another. Failures are recorded and the queue moves on."""
    emit = progress or (lambda e: None)
    qpath = Path(queue_path)
    q = queue_recover(load_queue(qpath))
    save_queue(qpath, q)
    total = len(q["items"])
    counts = {"done": 0, "failed": 0}
    while True:
        if cancel is not None and cancel.is_set():
            break
        item = next((i for i in q["items"] if i["status"] == "queued"), None)
        if item is None:
            break
        item.update(status="running", started=time.time(), detail="")
        save_queue(qpath, q)
        pos = sum(1 for i in q["items"] if i["status"] in ("done", "failed", "running"))
        say(f"\n=== [{pos}/{total}] {Path(item['input']).name} ===")
        emit({"stage": "queue_item_start", "id": item["id"], "input": item["input"], "pos": pos, "of": total})
        try:
            args = args_for_item(item)
            manifest = run_job(args, progress=lambda e, _id=item["id"]: emit(dict(e, item=_id)), cancel=cancel)
            item.update(status="done", finished=time.time(), crashes=0,
                        detail=f"{len(live_clips(manifest))} clips")
            counts["done"] += 1
        except ClipperError as e:
            if str(e).startswith("Cancelled"):
                item.update(status="queued", detail="Cancelled; continues where it stopped.")
                save_queue(qpath, q)
                break
            item.update(status="failed", finished=time.time(), detail=" ".join(str(e).split())[:300])
            counts["failed"] += 1
            say(f"  FAILED: {item['detail']}")
        except KeyboardInterrupt:
            item.update(status="queued", detail="Interrupted; continues where it stopped.")
            save_queue(qpath, q)
            raise
        except Exception as e:  # one bad file must not take the whole batch down
            item.update(status="failed", finished=time.time(), detail=f"{type(e).__name__}: {e}"[:300])
            counts["failed"] += 1
            say(f"  FAILED: {item['detail']}")
        save_queue(qpath, q)
        emit({"stage": "queue_item_end", "id": item["id"], "status": item["status"]})
    for it in q["items"]:                      # temp cleanup across the whole batch
        if it["status"] == "done":
            cleanup_work(it["out"])
            sweep_partials(it["out"])
    emit({"stage": "queue_finished", **counts})
    return q


def print_queue(q, qpath):
    items = q["items"]
    print(f"\nQueue {qpath}: {len(items)} item(s)")
    for it in items:
        print(f"  {it['status']:<8} {Path(it['input']).name}" + (f"   ({it['detail']})" if it.get("detail") else ""))


def run_batch_cli(args):
    qpath = Path(args.queue_file or QUEUE_NAME)
    q = load_queue(qpath)
    if args.queue_status:
        print_queue(q, qpath)
        return 0
    files = expand_inputs(args.input)
    if files:
        added, skipped = queue_add(q, files, args, out_parent=args.out)
        for a in added:
            print(f"Queued: {a}")
        for s in skipped:
            print(f"Skipped ({s})")
    if args.retry_failed:
        print(f"Re-queued {queue_retry_failed(q)} failed item(s).")
    save_queue(qpath, q)
    if not any(i["status"] in ("queued", "running") for i in q["items"]):
        print("Nothing waiting in the queue.")
        print_queue(q, qpath)
        return 0
    run_queue(qpath)
    q = load_queue(qpath)
    print_queue(q, qpath)
    return 1 if any(i["status"] == "failed" for i in q["items"]) else 0


if __name__ == "__main__":
    main()