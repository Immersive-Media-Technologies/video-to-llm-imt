#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# V2L-IMT (video2llm.py) — Copyright (c) 2026 Immersive Media Technologies.
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License:
# free for NON-COMMERCIAL use, modification and sharing; any commercial or monetised use needs a
# separate written license from Immersive Media Technologies. Attribution is mandatory and must
# stay visible in every copy and derivative work:
#     V2L-IMT by Immersive Media Technologies — https://github.com/Immersive-Media-Technologies
# Full text and terms: LICENSE, NOTICE and THIRD-PARTY-NOTICES.md next to this file. Provided "as is",
# without warranty.
"""
V2L-IMT (video2llm.py) — let an AI agent or a chat model watch a video.

Most models do not take video; the ones that do take a small file. video2llm turns a video into a
*lane* the model can follow: frames with [mm:ss] tags, the words spoken at those moments between
them, and a header that tells the model what it is looking at, how to ask for more detail and what
that detail costs. The technique, as Gemini does it: an overview at 1 frame per second; where that
is not enough, every frame of just that moment, on request; sounds other than speech, on request too.

What it does, step by step (all files land in <video>_frames/ next to the video, cached per file):

  1. probe          — ffprobe: duration, frame size, fps, audio track, file size.
  2. work copy      — ONLY if the short side is above 1080 px (4K and the like): a 1080p copy to cut
                      frames from. Nothing is ever upscaled.
  3. overview       — --frames 1 (default): 1 frame per second over the whole video, 768 px on the
                      long side (≈448 Claude tokens a frame in either orientation), names mm-ss.jpg.
  4. frame by frame — --frames all --start A --end B: every frame from the frame at A to the frame at
                      B, both included, 512 px on the long side, names mm-ss.cc.jpg; up to ~120
                      frames per request (4 s at 30 fps), never thinned. Written beside the overview.
  5. speech         — faster-whisper with its voice filter, exactly as in Deep Artisan; unsure of the
                      language → the system language. Only speech by default.
  6. sounds         — on request only (--sounds --start A --end B): sounds other than speech (music,
                      laughter, a slam, an engine, "Indistinct voices") from the PANNs AudioSet
                      classifier as ONNX on ONNX Runtime (CPU), for just that moment.
  7. lane           — lane.md (for agents: frame links + the commands for more detail), and on request
                      lane.pdf (Claude / Gemini chats), sheets/ (contact sheets + text — ChatGPT and any
                      chat), payload.<provider>.json (OpenAI / Anthropic / Gemini APIs), --native (the
                      video itself, fitted into a 19 MB Gemini request).
  8. ask            — optional (--ask "question"): send the lane to the API and print the answer.

Requirements: Python 3.9+, ffmpeg + ffprobe; `pip install faster-whisper` for speech (it also brings
ONNX Runtime + numpy for the sounds); `pip install fpdf2` for PDF and sheets. Works on macOS, Windows
and Linux. Models are downloaded once, only from this project's own release (models-v1). No other
network use except the optional --ask.

The video pipeline of Deep Artisan (Immersive Media Technologies), trimmed to what runs anywhere.
IMT Non-Commercial License — non-commercial use only, attribution required (LICENSE).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

VERSION = "0.7.2"

# ── constants (same numbers as the Deep Artisan pipeline) ────────────────────────────────────────
NATIVE_LIMIT_BYTES = 19 * 1024 * 1024  # Gemini request limit 20 MB minus headroom
AUDIO_KBPS = 64
FRAME_WIDTH = 768   # overview frames fit in 768×768 (long side 768): ≈448 Claude tokens in either orientation
FRAME_FPS = 1
OVERVIEW_FPS_MAX = 4  # --fps: a denser overview (2–4 a second) adds frames to the same folder, [00:12.25] between [00:12] and [00:13]
DENSE_MAX_FRAMES = 120  # frame-by-frame: every frame of up to 120/fps s per request (4 s at 30 fps), one message
DENSE_MAX_FPS = 100.0   # frame names carry 1/100 s
DENSE_WIDTH = 512   # frame-by-frame frames fit in 512×512
AT_WIDTH = 1024     # single frames asked for by time (a slide, an interface, text on screen): readable, ≈ 800 Claude tokens
AT_MAX = 40         # moments per --frames at request
SPARSE_PER_MESSAGE = 30  # frames of the 1 fps lane per message (API request bodies)
LANE_MD_FRAMES = 150     # frames of the 1 fps lane per lane.md: 2½ minutes, 50 sheets — an agent opens them one by one
                         # anyway, so a longer lane only saves it the extra runs; a longer video comes in parts
MANIFEST_NAME = "video2llm.json"

# Folder names: English by default; --names ru gives the names Deep Artisan uses, so a folder
# prepared here is recognisable there.
NAMES = {
    "en": {
        "sparse": "frames_1fps_{w}px",
        "dense": "frames_all{span}_{fps}fps_{w}px",
        "at": "frames_at_{w}px",
        "native": "native",
        "work": "work-1080p.mp4",
        "audio": "audio.wav",
    },
    "ru": {
        "sparse": "кадры 1 в сек ({w}px)",
        "dense": "все кадры{span} ({fps} в сек, {w}px)",
        "at": "кадры по запросу ({w}px)",
        "native": "native",
        "work": "work-1080p.mp4",
        "audio": "audio.wav",
    },
}

Progress = Callable[[str, int, str], None]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def make_progress(quiet: bool) -> Progress:
    last = {"key": ""}

    def progress(phase: str, pct: int, note: str = "") -> None:
        if quiet:
            return
        key = f"{phase}:{pct}:{note}"
        if key == last["key"]:
            return
        last["key"] = key
        tail = f" — {note}" if note else ""
        end = "\n" if pct >= 100 or not sys.stderr.isatty() else "\r"
        print(f"  [{phase}] {pct:3d}%{tail}".ljust(60), end=end, file=sys.stderr, flush=True)

    return progress


# ── external tools ───────────────────────────────────────────────────────────────────────────────
def find_tool(name: str, override: Optional[str]) -> str:
    cand = override or os.environ.get(f"VIDEO2LLM_{name.upper()}") or shutil.which(name)
    if cand and (Path(cand).exists() or shutil.which(cand)):
        return cand
    hint = {
        "darwin": "brew install ffmpeg",
        "win32": "winget install Gyan.FFmpeg   (or: choco install ffmpeg)",
    }.get(sys.platform, "sudo apt install ffmpeg")
    sys.exit(f"{name} not found. Install it ({hint}) or pass --{name} /path/to/{name}.")


def run(cmd: List[str], on_line: Optional[Callable[[str], None]] = None) -> subprocess.CompletedProcess:
    if on_line is None:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    out_lines: List[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        out_lines.append(line)
        on_line(line.rstrip("\n"))
    _, err = p.communicate()
    return subprocess.CompletedProcess(cmd, p.returncode, "".join(out_lines), err)


class Tools:
    def __init__(self, ffmpeg: str, ffprobe: str):
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self._vfr: Optional[List[str]] = None

    def vfr(self) -> List[str]:
        """Keep frames as selected (no duplicates): -fps_mode vfr on ffmpeg 5.1+, -vsync vfr before it."""
        if self._vfr is None:
            m = re.search(r"version\s+n?(\d+)\.(\d+)", run([self.ffmpeg, "-hide_banner", "-version"]).stdout or "")
            old = bool(m) and (int(m.group(1)), int(m.group(2))) < (5, 1)
            self._vfr = ["-vsync", "vfr"] if old else ["-fps_mode", "vfr"]
        return self._vfr

    def probe(self, path: Path) -> Dict[str, Any]:
        r = run([self.ffprobe, "-v", "error", "-show_entries",
                 "format=duration,bit_rate:stream=codec_type,width,height,r_frame_rate:stream_tags=rotate"
                 ":stream_side_data=rotation:stream_disposition=attached_pic",
                 "-of", "json", str(path)])
        if r.returncode != 0:
            sys.exit(f"ffprobe failed: {r.stderr.strip()[:300]}")
        j = json.loads(r.stdout or "{}")
        streams = j.get("streams") or []
        # a sound file's cover art is a "video" stream too (attached_pic) — it is not a picture that moves
        v = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), {})
        # phones store portrait video as landscape + a rotation; ffmpeg turns the frames, the size must follow
        rot = 0
        try:
            rot = int(float((v.get("tags") or {}).get("rotate") or 0)) or next(
                (int(float(sd["rotation"])) for sd in (v.get("side_data_list") or []) if "rotation" in sd), 0)
        except (TypeError, ValueError):
            rot = 0
        w, h = int(v.get("width") or 0), int(v.get("height") or 0)
        if abs(rot) % 180 == 90:
            w, h = h, w
        num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
        try:
            fps = float(num) / float(den) if den and float(den) else float(num or 0)
        except ValueError:
            fps = 0.0
        return {
            "duration": float((j.get("format") or {}).get("duration") or 0),
            "width": w,
            "height": h,
            "fps": fps,
            "bitrate_kbps": round(float((j.get("format") or {}).get("bit_rate") or 0) / 1000),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "has_video": bool(v),  # False — a sound file: no frames, the lane is the words and the sounds
            "bytes": path.stat().st_size,
        }

    def ffmpeg_run(self, args: List[str], duration: float, phase: str, progress: Progress) -> None:
        def on_line(line: str) -> None:
            m = re.match(r"^out_time_ms=(\d+)", line)
            if m and duration > 0:
                progress(phase, min(99, round(int(m.group(1)) / 1e6 / duration * 100)))

        r = run([self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-nostdin", "-progress", "pipe:1", *args], on_line)
        if r.returncode != 0:
            sys.exit(f"ffmpeg failed ({phase}): {r.stderr.strip()[:400]}")


# ── geometry / planning ──────────────────────────────────────────────────────────────────────────
def short_side(p: Dict[str, Any]) -> int:
    w, h = p.get("width") or 0, p.get("height") or 0
    return min(w, h) if w and h else (h or w)


def scale_filter(p: Dict[str, Any], max_short: int) -> str:
    """Short side ≤ max_short, never upscale, keep proportions (even dimensions for H.264)."""
    return f"scale=-2:'min({max_short},ih)'" if (p.get("width") or 0) > (p.get("height") or 0) else f"scale='min({max_short},iw)':-2"


def shell_arg(path: Any) -> str:
    """A path for the commands in the header: ~ for the home folder (shorter, and no user name in the text), quoted
    only when needed; Windows gets double quotes."""
    text = str(path)
    if os.name == "nt":
        return f'"{text}"'
    home = str(Path.home())
    if text.startswith(home + os.sep):
        return "~/" + shlex.quote(text[len(home) + 1:])
    return shlex.quote(text)


def span_tag(a: float, b: float, names: Dict[str, str]) -> str:
    """'_00-12_00-14' in file and folder names (ASCII: no spaces, no dashes to escape); with --names ru the
    Deep Artisan style ' 00-12–00-14'."""
    return f" {fmt_t(a)}–{fmt_t(b)}" if names is NAMES["ru"] else f"_{fmt_t(a)}_{fmt_t(b)}"


# The overview takes, for every whole second, the first frame at or after it — so the frame tagged [00:12]
# is the frame at 12.000 s, the same frame the frame-by-frame lane tags [00:12.00]. (ffmpeg's fps=1 would
# take the frame nearest to 12.5 s.)
OVERVIEW_SELECT = "select='isnan(prev_selected_t)+gte(floor(t),floor(prev_selected_t)+1)'"
OVERVIEW_CUT = 2  # manifest marker: caches cut the older way (fps=1) are redone


def overview_select(fps: int) -> str:
    """The overview at `fps` a second: for every 1/fps s the first frame at or after it — [00:12.25] is the frame
    at 12.250 s. At 1 a second this is OVERVIEW_SELECT."""
    if fps <= 1:
        return OVERVIEW_SELECT
    return f"select='isnan(prev_selected_t)+gte(floor(t*{fps}),floor(prev_selected_t*{fps})+1)'"


def tcode_frac(sec: float) -> str:
    """mm:ss for a whole second, mm:ss.cc for a frame between two seconds (a 2–4 a second overview)."""
    cc = int(round((sec % 1) * 100))
    return tcode(sec) if cc == 0 or cc == 100 else f"{tcode(math.floor(sec))}.{cc:02d}"


def overview_frames_at(frames: List[Dict[str, Any]], fps: int) -> List[Dict[str, Any]]:
    """The frames of the folder that belong to an overview at `fps` a second (the folder may hold a denser cut)."""
    if fps <= 1:
        return [f for f in frames if abs(f["t"] - round(f["t"])) < 1e-6]
    return [f for f in frames if abs(f["t"] * fps - round(f["t"] * fps)) < 1e-6]


def fit_filter(size: int) -> str:
    """Scale so the long side is `size` px (never up): a portrait frame costs what a landscape one does."""
    return (f"scale='if(gte(iw,ih),min({size},iw),-2)':'if(gte(iw,ih),-2,min({size},ih))'")


def lane_ok(frames: List[Dict[str, Any]], p: Dict[str, Any], size: int) -> bool:
    """Cached frames were cut at this size (an older cut at 768 px width is redone)."""
    if not frames:
        return False
    wh = jpeg_size(frames[0]["file"])
    want = min(size, max(p.get("width") or size, p.get("height") or size))
    return bool(wh) and abs(max(wh) - want) <= 2


def needs_work_copy(p: Dict[str, Any]) -> bool:
    return short_side(p) > 1080


def plan_native(p: Dict[str, Any]) -> Dict[str, Any]:
    """Fit the video into NATIVE_LIMIT_BYTES for models that take video.
    Budget B = 19 MB·8 / duration − audio. H.264 ladder: 1080p 3500/2500 kbps, 720p 2000/1500.
    1) 1080p@3500 if B ≥ 3500 (≈ up to 43 s); 2) 1080p@B if B ≥ 2500 (≈ 60 s); 3) 720p@B if B ≥ 1500
    (≈ 101 s); 4) otherwise equal parts of ≈74 s at 720p@2000. Never below 720p — UI and text in the
    frame become unreadable; a file already within budget (≤19 MB, ≤1080p, ≤30 fps) goes as is."""
    short = short_side(p)
    src_h = min(short or 1080, 1080)
    dur = max(1.0, float(p["duration"]))
    if p["bytes"] <= NATIVE_LIMIT_BYTES and short <= 1080 and p["fps"] <= 30.5:
        return {"height": short, "kbps": p["bitrate_kbps"], "parts": [{"start": 0.0, "dur": float(p["duration"])}], "reencode": False}
    budget = NATIVE_LIMIT_BYTES * 8 / 1000 / dur - AUDIO_KBPS
    if budget >= 3500:
        return {"height": src_h, "kbps": 3500, "parts": [{"start": 0.0, "dur": dur}], "reencode": True}
    if budget >= 2500:
        return {"height": src_h, "kbps": math.floor(budget), "parts": [{"start": 0.0, "dur": dur}], "reencode": True}
    h720 = min(src_h, 720)
    if budget >= 1500:
        return {"height": h720, "kbps": math.floor(budget), "parts": [{"start": 0.0, "dur": dur}], "reencode": True}
    part_len = math.floor(NATIVE_LIMIT_BYTES * 8 / 1000 / (2000 + AUDIO_KBPS))  # ≈ 74 s
    n = math.ceil(dur / part_len)
    even = dur / n  # equal parts, not a 3-second tail
    parts = [{"start": round(i * even, 2), "dur": round(even, 2)} for i in range(n)]
    return {"height": h720, "kbps": 2000, "parts": parts, "reencode": True}


def fmt_t(sec: float) -> str:
    s = max(0, int(round(sec)))
    h, m, r = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}-{m:02d}-{r:02d}" if h else f"{m:02d}-{r:02d}"


def tcode(sec: float) -> str:
    return fmt_t(sec).replace("-", ":")


def parse_tcode(s: str) -> float:
    parts = [float(x) for x in s.replace("-", ":").split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def frame_time_from_name(name: str) -> Optional[float]:
    m = re.match(r"^(\d\d)-(\d\d)(?:-(\d\d))?(?:\.(\d\d))?\.jpg$", name)
    if not m:
        return None
    a, b, c, cc = m.groups()
    t = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    return t + (int(cc) / 100 if cc else 0)


# ── transcript ───────────────────────────────────────────────────────────────────────────────────
def read_wav_mono16k(wav: Path):
    """16-bit PCM wav (as ffmpeg wrote it) → float32 numpy array in [-1, 1]."""
    import wave
    import numpy as np  # type: ignore  # comes with faster-whisper / onnxruntime
    with wave.open(str(wav), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM wav")
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    a = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a


def os_language() -> Optional[str]:
    """Two-letter UI language of the user's system ('ru', 'en' …) — the fallback when Whisper is unsure
    which language a few seconds of audio are in. None — unknown."""
    cands: List[str] = []
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        if os.environ.get(var):
            cands.append(os.environ[var])
    try:
        if sys.platform == "darwin":  # GUI-launched Python on macOS often has no LANG — ask the system
            r = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True, text=True, timeout=5)
            m = re.search(r'"?([A-Za-z]{2,3})[-_]?', r.stdout.replace("(", " "))
            if m:
                cands.append(m.group(1))
        elif sys.platform == "win32":
            import ctypes
            import locale
            lid = ctypes.windll.kernel32.GetUserDefaultUILanguage()  # type: ignore[attr-defined]
            if locale.windows_locale.get(lid):
                cands.append(locale.windows_locale[lid])
    except Exception:  # noqa: BLE001 — best effort
        pass
    try:
        import locale
        loc = locale.getlocale()[0]
        if loc:
            cands.append(loc)
    except Exception:  # noqa: BLE001
        pass
    for c in cands:
        code = re.split(r"[-_.@]", c.strip())[0].lower()
        if re.fullmatch(r"[a-z]{2,3}", code) and code not in ("c", "posix"):
            return code
    return None


# Accept a segment of the supplementary pass only when Whisper itself is confident it is speech and words:
SUPP_MAX_NO_SPEECH = 0.4
SUPP_MIN_AVG_LOGPROB = -0.8
SUPP_MAX_COMPRESSION = 2.4


def _overlaps(a0: float, a1: float, spans: List[List[float]], margin: float = 0.0) -> bool:
    return any(a0 < b1 + margin and a1 > b0 - margin for b0, b1 in spans)


def transcribe(wav: Path, model_name: str, language: Optional[str], progress: Progress, scan=None,
               mode: str = "hybrid") -> Optional[Dict[str, Any]]:
    """Speech → segments, in two passes.

    1. The main pass is exactly what Deep Artisan does: Whisper with its own voice filter (Silero VAD),
       segments Whisper itself doubts are speech (no_speech ≥ 0.6) dropped. On real footage — a teacher
       over children's noise and laughter — this is the most accurate; without the filter Whisper "hears"
       words in laughter and crowd noise.
    2. Only with --voice-rescue (off by default: on real footage it doubled the time and added a handful of
       unverifiable lines): where the sound model hears a voice the main pass left empty — typically a voice
       under loud music, which Silero drops — Whisper runs once more on just those spans (no filter,
       temperature 0) and a segment is kept only if Whisper is confident it is speech (no_speech < 0.4,
       avg log-prob > −0.8, not repetitive).
    Language: --language if given; otherwise Whisper's detection, and when it is unsure (< 0.5) the system UI
    language. Returns {"segments", "language", "language_source", "gate", "voice"} — `voice` are the spans
    where the sound model hears a voice; None if faster-whisper is not installed."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        log("  faster-whisper is not installed — transcript skipped (pip install faster-whisper)")
        return None
    try:  # Whisper's temperature fallback samples; one fixed seed = the same transcript on every run of a video
        import ctranslate2  # type: ignore
        ctranslate2.set_random_seed(0)
    except Exception:
        pass
    hybrid = mode == "hybrid" and scan is not None
    path = whisper_model_path(model_name, progress)
    if path is None:
        return None
    progress("transcript", 0, f"faster-whisper {model_name}")
    model = WhisperModel(path, device="cpu", compute_type="int8")
    base_kw: Dict[str, Any] = dict(condition_on_previous_text=False, word_timestamps=True, hallucination_silence_threshold=2.0)
    # The wav is ours (16 kHz mono PCM): decode it with the stdlib and hand faster-whisper the samples —
    # this sidesteps its PyAV decoder, whose version pinning breaks on newer Pythons.
    audio = read_wav_mono16k(wav)
    lang_src = "option"
    lang: Optional[str] = language if language and language != "auto" else None
    if lang is None:
        lang_src = "detected"
    kw = dict(base_kw, vad_filter=True, **({"language": lang} if lang else {}))
    segments, info = model.transcribe(audio, **kw)  # language detection runs here, segments are lazy
    if lang is None and float(getattr(info, "language_probability", 1.0)) < 0.5:
        sys_lang = os_language()
        if sys_lang and sys_lang != info.language:
            guessed = f"{info.language} {float(info.language_probability):.2f}"
            try:
                segments, info = model.transcribe(audio, **dict(kw, language=sys_lang))
                lang_src = f"system language; Whisper guessed {guessed}"
            except ValueError:  # Whisper does not know this language code
                pass
    lang = getattr(info, "language", None)
    out: List[Dict[str, Any]] = []
    for sgm in segments:
        text = (sgm.text or "").strip()
        if text and float(sgm.no_speech_prob) < 0.6:
            out.append({"start": round(float(sgm.start), 2), "end": round(float(sgm.end), 2), "text": text})
    n_main = len(out)

    voice: List[List[float]] = []
    n_supp = 0
    if hybrid:
        voice = speech_regions(scan)
        spans = [[x["start"], x["end"]] for x in out]
        todo = [r for r in voice if not _overlaps(r[0], r[1], spans, margin=0.5)]
        if todo:
            progress("transcript", 50, f"{len(todo)} voice span(s) the main pass left empty")
            # one greedy decode (no temperature fallback — sampling is where invented words come from); Whisper's
            # own skip rules are off so every candidate reaches our stricter filter below
            sup, _ = model.transcribe(audio, **dict(base_kw, vad_filter=False, temperature=0.0, language=lang,
                                                    no_speech_threshold=None, log_prob_threshold=None,
                                                    compression_ratio_threshold=None,
                                                    clip_timestamps=[x for r in todo for x in r]))
            for sgm in sup:
                text = (sgm.text or "").strip()
                ok = bool(text and re.search(r"\w", text) and float(sgm.no_speech_prob) < SUPP_MAX_NO_SPEECH
                          and float(sgm.avg_logprob) > SUPP_MIN_AVG_LOGPROB
                          and float(sgm.compression_ratio) < SUPP_MAX_COMPRESSION)
                if os.environ.get("VIDEO2LLM_DEBUG"):
                    log(f"    {'KEEP' if ok else 'drop'} [{tcode(sgm.start)}–{tcode(sgm.end)}] no_speech {sgm.no_speech_prob:.2f} "
                        f"logprob {sgm.avg_logprob:.2f} compr {sgm.compression_ratio:.2f} «{text}»")
                if ok:
                    out.append({"start": round(float(sgm.start), 2), "end": round(float(sgm.end), 2), "text": text,
                                "pass": "voice-under-noise"})
                    n_supp += 1
    out.sort(key=lambda x: x["start"])
    del segments, model  # release CTranslate2 thread pools now, not during interpreter shutdown
    gate = "silero+sound-model" if hybrid else "silero"
    progress("transcript", 100, f"{len(out)} segments ({n_main} main" + (f", {n_supp} under noise" if hybrid else "")
             + f"), language {lang} ({lang_src})")
    return {"segments": out, "language": lang, "language_source": lang_src, "gate": gate, "voice": voice}


def indistinct_voices(voice: List[List[float]], segs: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Where the sound model hears a voice but no words were transcribed (crowd, chatter, a distant voice) — a tag,
    so the model knows people are talking even though the words are not there. Spans already explained by laughter
    are left to the laughter tag."""
    laugh = [[e["start"], e["end"]] for e in events
             if e["label"] in ("Laughter", "Chuckle, chortle", "Giggle", "Snicker", "Belly laugh", "Baby laughter")]
    spans = [[x["start"], x["end"]] for x in segs]
    out: List[Dict[str, Any]] = []
    for v0, v1 in voice:
        if v1 - v0 < 1.0 or _overlaps(v0, v1, spans) or _overlaps(v0, v1, laugh):
            continue
        out.append({"start": round(v0, 1), "end": round(v1, 1), "label": "Indistinct voices", "score": None})
    return out


# ── sound events (PANNs / AudioSet, run with ONNX Runtime) ──────────────────────────────────────
# The model is PANNs CNN14-DecisionLevelMax (Kong et al., 2019, CC BY 4.0), converted once to ONNX
# (export_panns_onnx.py next to this script; the spectrogram front-end is inside the graph, the 527
# AudioSet class names and the attribution are in the file's metadata). It runs on ONNX Runtime —
# the same package faster-whisper already installs — so no PyTorch is needed. Torch and ONNX give
# the same probabilities (max difference 4.5e-7 on real audio).
SOUNDS_MODEL_NAME = "panns_cnn14_dlmax.onnx"
SOUNDS_MODEL_SHA256 = "9ec11d29b372dff6561bd31d7174e52a1246dee697748e30044ab769046bab71"
# Everything the script downloads lives in this project's own release "models-v1" — nothing is fetched
# from anywhere else (faster-whisper would otherwise pull its models from Hugging Face).
MODELS_URL = "https://github.com/Immersive-Media-Technologies/video-to-llm-imt/releases/download/models-v1/"
SOUNDS_MODEL_URL = MODELS_URL + "panns_cnn14_dlmax.onnx"
# Whisper models in CTranslate2 form (OpenAI Whisper weights, MIT; converted by Systran for faster-whisper,
# MIT), one zip per size: config.json, model.bin, tokenizer.json, vocabulary.txt.
WHISPER_MODELS = {
    "base": ("faster-whisper-base.zip", "d3186dd7bd618fa07463150c7bd2999fd1692d6132dcf1afe3d1d9e8d21b01ba", 141),
    "small": ("faster-whisper-small.zip", "fab3200cff302cf3759651187db3e479392d5cb155faf618c59c7216abaf1f59", 464),
}
WHISPER_FILES = {"config.json", "model.bin", "tokenizer.json", "vocabulary.txt"}
SOUNDS_SR = 32000
SOUND_CONTEXT = 3.0  # s of audio around a requested moment that the classifier hears too
# Whisper covers the words; these AudioSet classes would only duplicate the transcript.
SPEECH_LABELS = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
                 "Conversation", "Narration, monologue", "Speech synthesizer", "Babbling"}
# Acoustic-scene labels that say nothing about what happened.
SCENE_LABELS = {"Inside, small room", "Inside, large room or hall", "Inside, public space", "Outside, urban or manmade",
                "Outside, rural or natural", "Sound effect", "Reverberation", "Echo"}


def cache_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "video2llm"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "video2llm"


def download(url: str, dest: Path, progress: Progress, phase: str, sha256: Optional[str] = None) -> None:
    import hashlib
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": f"video2llm/{VERSION}"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); h.update(chunk); done += len(chunk)
            if total:
                progress(phase, min(99, round(done / total * 100)), f"downloading {dest.name} ({total >> 20} MB)")
    if sha256 and h.hexdigest() != sha256:
        tmp.unlink()
        raise OSError(f"checksum mismatch for {dest.name}")
    tmp.replace(dest)


def whisper_model_path(name: str, progress: Progress) -> Optional[str]:
    """--whisper-model base|small → cached copy → download from this project's release (sha256-checked);
    or a folder with a CTranslate2 Whisper model of any other size."""
    own = Path(name).expanduser()
    if own.is_dir():
        return str(own)
    if name not in WHISPER_MODELS:
        log(f"  --whisper-model {name}: this project hosts {' and '.join(WHISPER_MODELS)}; for another size pass the "
            "path to a folder with a CTranslate2 Whisper model")
        return None
    folder = cache_dir() / "whisper" / name
    if all((folder / f).is_file() for f in WHISPER_FILES):
        return str(folder)
    fname, sha, mb = WHISPER_MODELS[name]
    z = cache_dir() / "whisper" / fname
    try:
        download(os.environ.get("VIDEO2LLM_MODELS_URL", MODELS_URL) + fname, z, progress, "whisper model", sha)
        with zipfile.ZipFile(z) as zf:
            members = [n for n in zf.namelist() if n in WHISPER_FILES]  # plain names only, nothing outside the folder
            zf.extractall(folder, members)
        z.unlink()
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as e:
        log(f"  whisper model download failed ({e}) — transcript skipped; unpack {fname} into {folder} or pass "
            "--whisper-model PATH")
        return None
    return str(folder)


def sounds_model_path(override: Optional[str], progress: Progress) -> Optional[Path]:
    """--sounds-model / VIDEO2LLM_SOUNDS_MODEL → cached copy → download (sha256-checked)."""
    explicit = override or os.environ.get("VIDEO2LLM_SOUNDS_MODEL")
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        log(f"  sounds model not found at {p} — sound events skipped")
        return None
    cached = cache_dir() / SOUNDS_MODEL_NAME
    if cached.is_file():
        return cached
    url = os.environ.get("VIDEO2LLM_SOUNDS_URL") or SOUNDS_MODEL_URL
    try:
        download(url, cached, progress, "sounds", SOUNDS_MODEL_SHA256)
        return cached
    except (urllib.error.URLError, OSError) as e:
        log(f"  sounds model download failed ({e}) — sound events skipped; "
            f"put {SOUNDS_MODEL_NAME} into {cache_dir()} or pass --sounds-model PATH")
        return None


def read_wav_chunks(wav: Path, seconds: float):
    """16-bit PCM wav → float32 chunks of `seconds` (bounded memory for long videos)."""
    import wave
    import numpy as np  # type: ignore
    with wave.open(str(wav), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM wav")
        ch, n = w.getnchannels(), int(w.getframerate() * seconds)
        while True:
            raw = w.readframes(n)
            if not raw:
                break
            a = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
            yield a.reshape(-1, ch).mean(axis=1) if ch > 1 else a


# Voice for the speech gate: speech in all its forms, shouting and singing (Whisper transcribes sung words too).
VOICE_LABELS = SPEECH_LABELS | {"Shout", "Yell", "Children shouting", "Screaming", "Whispering", "Singing", "Choir",
                                "Child singing", "Male singing", "Female singing", "Chant", "Rapping", "Humming"}


def panns_scan(wav32: Path, progress: Progress, model_override: Optional[str] = None):
    """One pass of the sound model over the audio → (probs (T, 527) at 0.1 s, labels, duration s) or None.
    The same pass feeds the sound tags and the speech gate."""
    try:
        import numpy as np  # type: ignore
        import onnxruntime as ort  # type: ignore
    except ImportError:
        log("  onnxruntime is not installed — sound model skipped (pip install onnxruntime; "
            "faster-whisper brings it along)")
        return None
    model = sounds_model_path(model_override, progress)
    if model is None:
        return None
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    labels: List[str] = json.loads(sess.get_modelmeta().custom_metadata_map.get("video2llm.labels", "[]"))
    if len(labels) != 527:
        log(f"  {model.name} carries no AudioSet labels — sound model skipped")
        return None
    progress("sounds", 0, "PANNs CNN14 (AudioSet) on ONNX Runtime")
    total = max(1, wav32.stat().st_size - 44) / 2 / SOUNDS_SR  # seconds, mono 16-bit
    rows: List[Any] = []
    done = 0.0
    # chunks of 30 s; the model outputs 100 frames/s — averaged down to 10 per second right away
    for piece in read_wav_chunks(wav32, 30):
        n_samples = len(piece)
        if n_samples < SOUNDS_SR // 2:  # < 0.5 s tail — pad to a full second
            piece = np.pad(piece, (0, SOUNDS_SR - n_samples))
        fw = sess.run(["framewise"], {"audio": piece[None, :].astype("float32")})[0][0]  # (T, 527)
        keep = min(fw.shape[0], int(round(n_samples / SOUNDS_SR * 100)))  # drop padding frames
        keep = max(10, (keep // 10) * 10)
        fw = fw[:keep]
        rows.append(fw.reshape(-1, 10, fw.shape[1]).mean(axis=1))
        done += n_samples / SOUNDS_SR
        progress("sounds", min(99, round(done / total * 100)))
    del sess  # release ONNX Runtime threads now, not during interpreter shutdown
    return np.concatenate(rows, axis=0), labels, done


def speech_regions(scan, threshold: float = 0.3, pad: float = 0.3, gap: float = 0.5, min_len: float = 0.5) -> List[List[float]]:
    """Where a voice is heard (max over VOICE_LABELS ≥ threshold for ≥ min_len), padded and merged."""
    probs, labels, dur = scan
    idx = [i for i, lb in enumerate(labels) if lb in VOICE_LABELS]
    voice = probs[:, idx].max(axis=1)
    on = voice >= threshold
    raw: List[List[float]] = []
    start: Optional[int] = None
    for t in range(len(on) + 1):
        o = t < len(on) and bool(on[t])
        if o and start is None:
            start = t
        elif not o and start is not None:
            if (t - start) * 0.1 >= min_len:
                raw.append([max(0.0, round(start * 0.1 - pad, 2)), min(round(dur, 2), round(t * 0.1 + pad, 2))])
            start = None
    merged: List[List[float]] = []
    for r in raw:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def sound_events(scan, threshold: float = 0.15) -> List[Dict[str, Any]]:
    """Non-speech sound events with time spans: [{start, end, label, score}].
    CNN14-DecisionLevelMax is a frame-level tagger with modest absolute probabilities (a clean 440 Hz tone
    scores ≈0.25 as "Sine wave"), so the threshold is low and, per moment, only the two strongest
    classes count — that keeps "White noise" and drops the "Static / Waterfall" tail."""
    import numpy as np  # type: ignore
    probs, labels, dur = scan
    probs = probs.copy()
    step = 0.1
    skip = SPEECH_LABELS | SCENE_LABELS
    skip_idx = [i for i, lb in enumerate(labels) if lb in skip]
    probs[:, skip_idx] = 0.0
    # per moment only the two strongest classes survive
    rank = np.argsort(-probs, axis=1)
    mask = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(mask, rank[:, :2], True, axis=1)
    probs = np.where(mask, probs, 0.0)
    events: List[Dict[str, Any]] = []
    for c, label in enumerate(labels):
        if label in skip:
            continue
        col = probs[:, c]
        active = col >= threshold
        if not active.any():
            continue
        start: Optional[int] = None
        for t in range(len(active) + 1):
            on = t < len(active) and active[t]
            if on and start is None:
                start = t
            elif not on and start is not None:
                if (t - start) * step >= 0.3:  # shorter blips are noise
                    events.append({"start": round(start * step, 1), "end": round(min(t * step, dur), 1),
                                   "label": label, "score": round(float(col[start:t].max()), 2)})
                start = None
    # merge the same label across gaps < 0.5 s
    events.sort(key=lambda e: (e["label"], e["start"]))
    merged: List[Dict[str, Any]] = []
    for e in events:
        if merged and merged[-1]["label"] == e["label"] and e["start"] - merged[-1]["end"] < 0.5:
            merged[-1]["end"] = e["end"]; merged[-1]["score"] = max(merged[-1]["score"], e["score"])
        else:
            merged.append(dict(e))
    merged.sort(key=lambda e: (e["start"], -e["score"]))
    return merged


def sounds_note(scan, threshold: float) -> str:
    """What else the classifier heard in a requested moment — so "nothing found" reads as "could not be identified",
    not as silence: how much of it is speech, and the strongest other class with its probability."""
    probs, labels, _ = scan
    sp = [i for i, lb in enumerate(labels) if lb in SPEECH_LABELS]
    speech = float((probs[:, sp].max(axis=1) >= 0.3).mean()) if len(probs) else 0.0
    skip = SPEECH_LABELS | SCENE_LABELS
    best = max(((float(probs[:, i].max()), lb) for i, lb in enumerate(labels) if lb not in skip), default=(0.0, "—"))
    found_any = best[0] >= threshold
    return (("(" if found_any else "(no other sounds could be identified above the threshold — which does not mean there "
             "were none: quiet sounds under speech can be missed; ")
            + f"speech is heard in {round(speech * 100)}% of this moment; strongest other class: {best[1]} "
              + (f"{best[0]:.2f}" if best[0] >= 0.01 else "<0.01") + f", threshold {threshold})")


def sound_line(e: Dict[str, Any]) -> str:
    a, b = tcode(e["start"]), tcode(e["end"])
    return f"[{a}–{b}] (sound: {e['label']})" if a != b else f"[{a}] (sound: {e['label']})"


# ── where the files go ───────────────────────────────────────────────────────────────────────────
# Default: <video>_frames next to the video, on every OS. When that folder cannot be written (a read-only
# share, a mounted image, a locked card, Windows "Controlled folder access" …) the lane goes to the
# user's Videos folder instead: ~/Movies/video2llm on macOS, the Videos known folder on Windows
# (wherever it was moved to), XDG_VIDEOS_DIR or ~/Videos on Linux.
def videos_dir() -> Path:
    home = Path.home()
    if os.name == "nt":
        try:
            import ctypes
            import uuid
            fid = (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID("18989B1D-99B5-455B-841C-AB7C74E4DDFC").bytes_le)  # FOLDERID_Videos
            ptr = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(ptr)) == 0:
                found = Path(ptr.value)
                ctypes.windll.ole32.CoTaskMemFree(ptr)
                return found
        except Exception:
            pass
        return home / "Videos"
    if sys.platform == "darwin":
        return home / "Movies"
    xdg = os.environ.get("XDG_VIDEOS_DIR")
    return Path(os.path.expandvars(xdg)).expanduser() if xdg else home / "Videos"


def writable_dir(d: Path) -> bool:
    """True if a file can actually be created in `d` (created if missing). A real write, not
    os.access — ACLs, network shares and Windows folder protection only show up on write."""
    try:
        d.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".video2llm-", dir=str(d))
        os.close(fd)
        os.unlink(probe)
        return True
    except OSError:
        return False


def fallback_out_dir(source: Path) -> Path:
    """<Videos>/video2llm/<name>_<6 hex of the path>_frames — the hash keeps two clips with the same
    name from different folders apart."""
    import hashlib
    tag = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:6]
    base = videos_dir()
    root = (base if base.is_dir() else Path.home()) / "video2llm"
    for cand in (root, Path(tempfile.gettempdir()) / "video2llm"):
        d = cand / f"{source.stem}_{tag}_frames"
        if writable_dir(d):
            return d
    sys.exit(f"cannot write next to the video nor to {root} — pass --out with a folder you can write to")


# ── captions beside the video (a site's subtitles, or any .vtt / .srt the user puts there) ──
_TS_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")


def _ts(s: str) -> float:
    m = _TS_RE.search(s)
    if not m:
        return 0.0
    h, mi, se, frac = m.groups()
    return (int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(frac.ljust(3, "0")) / 1000.0)


def parse_captions(text: str) -> List[Dict[str, Any]]:
    """WebVTT or SRT → [{start, end, text}]. YouTube's automatic captions roll: every cue repeats the
    previous line and adds the new one word by word (<00:00:01.500><c> word</c>) — tags are dropped and
    a line already given by the previous cue is not given again, so each line comes once, at the
    time it first appears. Manual subtitles pass through as they are."""
    out: List[Dict[str, Any]] = []
    prev_lines: List[str] = []
    cue_start = cue_end = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf, prev_lines
        if cue_start is None:
            buf = []
            return
        lines = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", ln)).strip() for ln in buf]
        lines = [ln for ln in lines if ln and not ln.startswith("WEBVTT") and not re.fullmatch(r"\d+", ln)]
        new = [ln for ln in lines if ln not in prev_lines]
        if new and cue_end is not None and cue_end > cue_start:
            if out and out[-1]["text"] == " ".join(new):
                out[-1]["end"] = max(out[-1]["end"], cue_end)
            else:
                out.append({"start": cue_start, "end": cue_end, "text": " ".join(new)})
        if lines:
            prev_lines = lines
        buf = []

    for raw in text.replace("\r\n", "\n").split("\n"):
        line = raw.rstrip("\r")
        if "-->" in line:
            flush()
            a, b = line.split("-->", 1)
            cue_start, cue_end = _ts(a), _ts(b)
            continue
        if line == "":   # a cue ends at an EMPTY line; YouTube puts a line of one space INSIDE its cues
            flush()
            cue_start = cue_end = None
            continue
        if cue_start is None:
            continue  # header, NOTE, a cue number
        buf.append(line)
    flush()
    # the rolling style leaves many 10 ms cues: give each line a span up to the next line
    for i, seg in enumerate(out):
        if seg["end"] - seg["start"] < 0.5 and i + 1 < len(out):
            seg["end"] = max(seg["end"], min(out[i + 1]["start"], seg["start"] + 8.0))
    return out


def captions_beside(source: Path, language: Optional[str], prefer: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """A subtitle file next to the video — `<name>.<lang>.vtt` / `.srt` as yt-dlp writes them, or plain
    `<name>.vtt` / `.srt`. Which one, when there are several: --language first; then, with the site's
    `<name>.info.json` beside the file (yt-dlp), a MANUAL track in the video's own language, a manual one
    in a preferred language (--captions: the chat's language, the system's UI language), the automatic
    track in the original language (`*-orig`), an automatic one in a preferred language (the site's
    machine translation — last, it is the least faithful); without the info file: --language, `*-orig`,
    the preferred languages, ru, en, any. {file, language, segments, kind} or None."""
    stem = source.stem
    cands: List[Path] = []
    for f in source.parent.iterdir():
        if not f.is_file() or f.suffix.lower() not in (".vtt", ".srt"):
            continue
        if f.stem == stem or (f.stem.startswith(stem + ".") and "." not in f.stem[len(stem) + 1:]):
            cands.append(f)
    if not cands:
        return None
    manual: set = set()
    auto: set = set()
    original = ""
    info = source.with_name(stem + ".info.json")
    if info.is_file():
        try:
            j = json.loads(info.read_text(encoding="utf-8", errors="replace"))
            manual = {str(k).lower() for k in (j.get("subtitles") or {}).keys()}
            auto = {str(k).lower() for k in (j.get("automatic_captions") or {}).keys()}
            original = str(j.get("language") or "").lower()
        except (OSError, ValueError):
            pass
    pref = [x.lower() for x in (prefer or []) if x]

    def lang_of(f: Path) -> str:
        return f.stem[len(stem) + 1:].lower() if f.stem != stem else ""

    def base(lg: str) -> str:
        return lg.replace("-orig", "").split("-")[0]

    def rank(f: Path) -> tuple:
        lg = lang_of(f)
        want = (language or "").lower()
        if want and want != "auto" and base(lg) == base(want):
            r = 0
        elif lg.endswith("-orig"):
            r = 3 if (manual or auto) else 1
        elif lg in manual and original and base(lg) == original:
            r = 1
        elif lg in manual and any(base(lg) == base(x) for x in pref):
            r = 2
        elif lg in manual:
            r = 4
        elif lg in auto and original and base(lg) == original:
            r = 3
        elif any(base(lg) == base(x) for x in pref):
            r = 5 if (manual or auto) else 2
        elif base(lg) in ("ru", "en"):
            r = 6
        else:
            r = 7
        return (r, 0 if f.suffix.lower() == ".vtt" else 1, lg)
    f = sorted(cands, key=rank)[0]
    try:
        segs = parse_captions(f.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    lg = lang_of(f)
    kind = ("manual" if lg in manual else "automatic" if (lg in auto or lg.endswith("-orig")) else "file")
    return {"file": f, "language": base(lg) or None, "segments": segs, "kind": kind}


URL_RE = re.compile(r"^https?://", re.I)


def yt_dlp_command(override: Optional[str]) -> Optional[List[str]]:
    """The yt-dlp to run: --yt-dlp / VIDEO2LLM_YT_DLP (a path, or a command line like "py -m yt_dlp") / the
    `yt-dlp` on PATH / the yt_dlp module of this Python. None when there is none."""
    cand = override or os.environ.get("VIDEO2LLM_YT_DLP") or shutil.which("yt-dlp")
    if cand and (Path(cand).exists() or shutil.which(cand)):
        return [cand]
    if cand and " " in cand.strip():  # a command line: "py -m yt_dlp", "/opt/venv/bin/python -m yt_dlp"
        parts = [x.strip("\"'") for x in shlex.split(cand, posix=(os.name != "nt"))]
        if parts and (Path(parts[0]).exists() or shutil.which(parts[0])):
            return parts
    import importlib.util
    if importlib.util.find_spec("yt_dlp") is not None:
        return [sys.executable, "-m", "yt_dlp"]
    return None


def download_dir() -> Path:
    """<Videos>/video2llm/downloads (or VIDEO2LLM_DOWNLOADS) — the file a URL points to lands here; its
    lane folder goes next to it as for any local file (<name>_frames)."""
    env = os.environ.get("VIDEO2LLM_DOWNLOADS")
    if env:
        d = Path(env).expanduser()
        if writable_dir(d):
            return d
        sys.exit(f"cannot write to VIDEO2LLM_DOWNLOADS {d}")
    base = videos_dir()
    root = (base if base.is_dir() else Path.home()) / "video2llm" / "downloads"
    for cand in (root, Path(tempfile.gettempdir()) / "video2llm" / "downloads"):
        if writable_dir(cand):
            return cand
    sys.exit(f"cannot write to {root} — nowhere to download the video")


def yt_dlp_args(exe: List[str], url: str, dest: Path, ffmpeg: str, id_file: Path, path_file: Path,
                language: Optional[str] = None, prefer: Optional[List[str]] = None, subs: bool = True,
                skip_download: bool = False) -> List[str]:
    """One video (no playlist), the best streams up to 1080p merged into an mp4 — the lane never needs
    more; the site's subtitles beside it as .vtt (manual ones in the asked language / ru / en, and the
    automatic track in the original language — `*-orig`) — they stand in for Whisper; the id and the
    final path are written to files (the console is progress only). `skip_download` (0.7.2, the words of a
    link): the info file and the subtitles only — the path file then names the file the video WOULD get,
    so the same run later lands on the same name."""
    langs: List[str] = []
    for x in ([language] if language and language != "auto" else []) + list(prefer or []) + ["en"]:
        if x and x not in langs:
            langs.append(x)
    # exact codes, not patterns: `ru.*` also matched YouTube's translated tracks (ru-en, …), and a dozen
    # subtitle requests got a 429; a subtitle that fails must not take the video down with it (-i);
    # the info file says which tracks are manual and what the video's language is (captions_beside)
    return exe + ["--no-playlist", "--no-warnings", "--newline", "--progress", "--ignore-errors", "--no-abort-on-error",
                  "-f", "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b", "-S", "res:1080,vcodec:h264,ext:mp4:m4a",
                  "--merge-output-format", "mp4", "--ffmpeg-location", str(Path(ffmpeg).resolve().parent),
                  "--write-info-json", "--no-clean-info-json",
                  *(["--write-subs", "--write-auto-subs", "--sub-langs", ",".join(langs) + ",.*-orig,-live_chat",
                     "--sub-format", "vtt/srt/best", "--convert-subs", "vtt"] if subs else []),
                  *(["--skip-download"] if skip_download else []),
                  "-o", str(dest / "%(title).60B [%(id)s].%(ext)s"),
                  "--print-to-file", "%(id)s", str(id_file),
                  "--print-to-file", "filename" if skip_download else "after_move:filepath", str(path_file),
                  "--", url]


def fetch_words(url: str, tools: "Tools", progress: Progress, override: Optional[str] = None,
                language: Optional[str] = None, prefer: Optional[List[str]] = None, out: Optional[Path] = None,
                whisper_model: str = "small", transcript: bool = True) -> Tuple[Path, Path, Dict[str, Any]]:
    """0.7.2 — a link, the words only: the site's info file and subtitles land where the video would (its
    name is known without downloading it), nothing else is fetched. No subtitles on the site → the sound
    alone (a few MB) → Whisper, the wav kept for the video's own run later. Writes the link manifest into
    the lane folder (`<name>_frames` beside the future file): the same manifest prepare() picks up when the
    video is downloaded, so the words are never made twice. Returns (the future file, the lane folder, the
    manifest)."""
    exe = yt_dlp_command(override)
    if not exe:
        hint = {"darwin": "brew install yt-dlp", "win32": "winget install yt-dlp"}.get(sys.platform, "pip install yt-dlp")
        sys.exit(f"a video URL needs yt-dlp. Install it ({hint}, or: pip install yt-dlp) or pass --yt-dlp /path/to/yt-dlp.")
    dest = download_dir()
    progress("words", 0, url)
    with tempfile.TemporaryDirectory(prefix="video2llm-dl-") as td:
        id_file, path_file = Path(td) / "id.txt", Path(td) / "path.txt"
        try:
            r = run(yt_dlp_args(exe, url, dest, tools.ffmpeg, id_file, path_file, language, prefer, skip_download=True))
        except OSError as e:
            sys.exit(f"yt-dlp could not be run ({' '.join(exe)}): {e}")
        got = path_file.read_text(encoding="utf-8").strip().splitlines()[-1] if path_file.exists() and path_file.read_text(encoding="utf-8").strip() else ""
    if not got:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        sys.exit("yt-dlp could not read the link" + (": " + tail[-1] if tail else "") + f" ({url})")
    planned = dest / Path(got).name  # the name the video will get; info.json and the .vtt already sit beside it
    out_dir = (out or planned.with_name(f"{planned.stem}_frames")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    info = read_info_json(planned)
    progress("words", 60, str(info.get("title") or planned.stem))
    cap = captions_beside(planned, language if language != "auto" else None, prefer)
    segs: Optional[List[Dict[str, Any]]] = None
    tr_meta: Dict[str, Any] = {}
    if cap is not None:
        segs = cap["segments"]
        tr_meta = {"language": cap["language"], "language_source": "captions", "gate": "captions",
                   "source": "captions", "kind": cap["kind"], "file": cap["file"].name}
        progress("words", 100, f"{len(segs)} lines from {cap['file'].name} ({cap['kind']} captions, no download)")
    elif transcript:
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is None:
            log("  the site has no captions and faster-whisper is not installed — no words (pip install faster-whisper)")
            progress("words", 100, "no captions")
        else:
            log("  the site has no captions — fetching the sound alone for Whisper")
            wav = out_dir / NAMES["en"]["audio"]
            if not wav.exists():
                with tempfile.TemporaryDirectory(prefix="video2llm-au-") as td:
                    last = [-1]

                    def on_line(line: str) -> None:
                        m = re.search(r"\[download\]\s+([\d.]+)%", line)
                        if m:
                            pct = int(float(m.group(1)))
                            if pct != last[0]:
                                last[0] = pct
                                progress("audio", pct)
                    r = run(exe + ["--no-playlist", "--no-warnings", "--newline", "--progress", "-f", "ba/b",
                                   "--ffmpeg-location", str(Path(tools.ffmpeg).resolve().parent),
                                   "-o", str(Path(td) / "audio-src.%(ext)s"), "--", url], on_line)
                    src = next((f for f in Path(td).iterdir() if f.is_file() and f.stem == "audio-src"), None)
                    if src is None:
                        tail = (r.stderr or "").strip().splitlines()
                        sys.exit("yt-dlp could not fetch the sound" + (": " + tail[-1] if tail else "") + f" ({url})")
                    dur = float(info.get("duration") or 0)
                    tools.ffmpeg_run(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)], dur, "audio", progress)
                progress("audio", 100)
            res = transcribe(wav, whisper_model, language, progress, None, "silero")
            if res is not None:
                segs = res["segments"]
                tr_meta = {k: res[k] for k in ("language", "language_source", "gate")}
    else:
        progress("words", 100, "no captions")
    if segs is not None:
        (out_dir / "transcript.txt").write_text("\n".join(seg_line(x) for x in segs) + "\n", encoding="utf-8")
    dur = float(info.get("duration") or 0)
    p = {"duration": dur, "width": int(info.get("width") or 0), "height": int(info.get("height") or 0),
         "fps": float(info.get("fps") or 0), "bitrate_kbps": 0, "has_audio": True, "has_video": True, "bytes": 0}
    old = {}
    try:
        old = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    m = {
        "tool": "video2llm", "version": VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(planned), "source_mtime": None, "dir": str(out_dir), "probe": p, "work": str(planned),
        "frames": [], "overview_cut": None, "overview_fps": FRAME_FPS, "dense": None,
        "transcript": segs, "transcript_meta": tr_meta, "sound_spans": [], "native": None, "native_plan": None,
        "link": {"url": url, "webpage_url": info.get("webpage_url") or url, "title": info.get("title"),
                 "site": info.get("extractor_key") or info.get("extractor"), "uploader": info.get("uploader") or info.get("channel"),
                 "downloaded": False, "lecture": lecture_hint(planned, p, segs)},
    }
    if planned.is_file():  # fetched before: the video is already here — the words lane still says so
        m["link"]["downloaded"] = True
        if old.get("source_mtime") is not None:  # a full manifest exists: keep it, refresh the words only
            old["transcript"], old["transcript_meta"], old["link"] = segs, tr_meta, m["link"]
            m = old
    (out_dir / MANIFEST_NAME).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return planned, out_dir, m


def download_video(url: str, tools: "Tools", progress: Progress, override: Optional[str] = None,
                   language: Optional[str] = None, prefer: Optional[List[str]] = None) -> Path:
    """A URL on the command line (YouTube, Vimeo, a direct link — whatever yt-dlp knows): download it
    once into <Videos>/video2llm/downloads and go on as with a local file. A video downloaded before is
    found by its id and not fetched again."""
    exe = yt_dlp_command(override)
    if not exe:
        hint = {"darwin": "brew install yt-dlp", "win32": "winget install yt-dlp"}.get(sys.platform, "pip install yt-dlp")
        sys.exit(f"a video URL needs yt-dlp. Install it ({hint}, or: pip install yt-dlp) or pass --yt-dlp /path/to/yt-dlp.")
    dest = download_dir()
    with tempfile.TemporaryDirectory(prefix="video2llm-dl-") as td:
        id_file, path_file = Path(td) / "id.txt", Path(td) / "path.txt"
        progress("download", 0, url)
        last = [-1]

        def on_line(line: str) -> None:
            m = re.search(r"\[download\]\s+([\d.]+)%", line)
            if m:
                pct = int(float(m.group(1)))
                if pct != last[0]:
                    last[0] = pct
                    progress("download", pct)
        try:
            r = run(yt_dlp_args(exe, url, dest, tools.ffmpeg, id_file, path_file, language, prefer), on_line)
            # a subtitle the site refuses (YouTube answers 429 on its caption endpoint after a few requests)
            # must not take the video down: once more without subtitles — the lane then runs Whisper
            if r.returncode != 0 and "subtitles" in (r.stderr or "") and not path_file.exists():
                log("  captions: the site refused the subtitles (" + ((r.stderr or "").strip().splitlines() or [""])[-1][:120]
                    + ") — downloading the video without them; the words come from Whisper")
                r = run(yt_dlp_args(exe, url, dest, tools.ffmpeg, id_file, path_file, language, prefer, subs=False), on_line)
        except OSError as e:   # a broken shebang, a venv moved elsewhere, no permission
            sys.exit(f"yt-dlp could not be run ({' '.join(exe)}): {e}")
        vid = id_file.read_text(encoding="utf-8").strip().splitlines()[-1] if id_file.exists() and id_file.read_text(encoding="utf-8").strip() else ""
        got = path_file.read_text(encoding="utf-8").strip().splitlines()[-1] if path_file.exists() and path_file.read_text(encoding="utf-8").strip() else ""
    file = Path(got) if got and Path(got).is_file() else None
    if file is None and vid:  # downloaded earlier: yt-dlp skips the file and the after_move hook with it
        cand = sorted((f for f in dest.iterdir() if f.is_file() and f.stem.endswith(f" [{vid}]") and not f.name.endswith(".part")),
                      key=lambda f: f.stat().st_mtime)
        file = cand[-1] if cand else None
    if file is None:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        sys.exit("yt-dlp could not download the video" + (": " + tail[-1] if tail else "") + f" ({url})")
    progress("download", 100, file.name)
    return file


_CLOUD_NAMES = (("yandex.disk", "Yandex Disk"), ("yandexdisk", "Yandex Disk"), ("onedrive", "OneDrive"),
                ("dropbox", "Dropbox"), ("google drive", "Google Drive"), ("googledrive", "Google Drive"),
                ("my drive", "Google Drive"), ("icloud", "iCloud Drive"), ("pcloud", "pCloud"), ("pclouddrive", "pCloud"),
                ("mega", "MEGA"),
                ("nextcloud", "Nextcloud"), ("owncloud", "ownCloud"), ("box sync", "Box"), ("sync.com", "Sync.com"))


def cloud_synced(d: Path) -> Optional[str]:
    """The sync service a folder belongs to, if it looks like one (by path) — the frames and the audio
    would be uploaded there. A heuristic: known folder names, OneDrive's own variables, macOS iCloud
    Drive and File Provider roots, iCloud "Desktop & Documents"."""
    try:
        d = d.resolve()
    except OSError:
        pass
    parts = d.parts
    for i, part in enumerate(parts):
        if part == "Mobile Documents":
            return "iCloud Drive"
        if part == "CloudStorage" and i + 1 < len(parts):  # macOS File Provider: ~/Library/CloudStorage/<Service-account>
            svc = parts[i + 1].split("-")[0]
            return next((name for marker, name in _CLOUD_NAMES if svc.lower() == marker.replace(" ", "")), svc)
        low = part.lower()
        for marker, name in _CLOUD_NAMES:
            if low == marker or low.startswith(marker + " ") or low.startswith(marker + "-") or low.startswith(marker + "."):
                return name
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        root = os.environ.get(var)
        if root:
            try:
                d.relative_to(Path(root).resolve())
                return "OneDrive"
            except ValueError:
                pass
    if sys.platform == "darwin":
        icloud = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
        for name in ("Desktop", "Documents"):
            try:
                d.relative_to(Path.home() / name)
                if (icloud / name).is_dir():
                    return "iCloud Drive (Desktop & Documents)"
            except ValueError:
                pass
    return None


def stale_cleanup(out_dir: Path) -> None:
    """The video changed since the lane was cut (same name, new content): drop what video2llm made
    there so nothing old is reused. Only its own files — frames named as timecodes, its wavs, copies,
    parts, texts and lanes; anything else in the folder stays."""
    own = {"audio.wav", "audio-32k.wav", "work-1080p.mp4", "transcript.txt", "sounds.txt"}
    for f in out_dir.iterdir():
        try:
            if f.is_file() and (f.name in own or re.fullmatch(r"lane([ _][^/\\]+)?(-\d\d)?\.(md|pdf|json)|payload\.\w+([ _][^/\\]+)?\.json"
                                                             r"|sounds([ _][^/\\]+)?\.txt|audio-32k([ _][^/\\]+)?\.wav", f.name)):
                f.unlink()
            elif f.is_dir() and re.fullmatch(r"sheets([ _].+)?", f.name):
                _clear_sheets(f)
                if not any(f.iterdir()):
                    f.rmdir()
            elif f.is_dir():
                for g in f.iterdir():
                    if g.is_file() and (frame_time_from_name(g.name) is not None or re.fullmatch(r"(part-\d\d|native)\.mp4", g.name)):
                        g.unlink()
                    elif g.is_dir() and g.name == "sheets":  # lane sheets made from those frames
                        for h in g.iterdir():
                            if h.is_file() and re.fullmatch(r"sheet_[\d.-]+_[\d.-]+\.jpg", h.name):
                                h.unlink()
        except OSError:
            pass


# ── preparation ──────────────────────────────────────────────────────────────────────────────────
def list_frames(folder: Path) -> List[Dict[str, Any]]:
    out = []
    if folder.exists():
        for f in folder.iterdir():
            t = frame_time_from_name(f.name)
            if t is not None:
                out.append({"t": t, "file": str(f)})
    return sorted(out, key=lambda x: x["t"])


def rename_sequence(folder: Path, fps: float, dense: bool, offset: float = 0.0) -> None:
    """ffmpeg writes %06d.jpg; rename to timecodes (first frame is t=offset)."""
    for f in sorted(folder.glob("[0-9][0-9][0-9][0-9][0-9][0-9].jpg")):
        idx = int(f.stem) - 1
        t = offset + idx / fps
        cc = int(round((t % 1) * 100)) % 100
        # a denser overview (2–4 a second) names the frames between two seconds mm-ss.cc, the whole seconds as before
        name = f"{fmt_t(math.floor(t))}.{cc:02d}.jpg" if (dense or cc) else f"{fmt_t(t)}.jpg"
        target = folder / name
        if target.exists():
            target.unlink()
        f.rename(target)


def jpeg_size(path: str) -> Optional[List[int]]:
    """[width, height] from a JPEG's SOF marker (no imaging library needed)."""
    try:
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                b = f.read(1)
                while b and b != b"\xff":
                    b = f.read(1)
                while b == b"\xff":
                    b = f.read(1)
                if not b:
                    return None
                marker = b[0]
                size = int.from_bytes(f.read(2), "big")
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    f.read(1)
                    h, w = int.from_bytes(f.read(2), "big"), int.from_bytes(f.read(2), "big")
                    return [w, h]
                f.seek(size - 2, 1)
    except OSError:
        return None


def claude_image_tokens(w: float, h: float) -> int:
    """Anthropic's published estimate for one image: ⌈w/28⌉ × ⌈h/28⌉ visual tokens, after the image is
    scaled to a long edge of at most 1568 px and at most 1568 tokens (standard tier). Other providers
    count differently; the exact figure comes back only with the response."""
    k = min(1.0, 1568 / max(w, h, 1))
    w, h = w * k, h * k
    while math.ceil(w / 28) * math.ceil(h / 28) > 1568:
        w, h = w * 0.98, h * 0.98
    return math.ceil(w / 28) * math.ceil(h / 28)


def p_has_audio(m: Dict[str, Any]) -> bool:
    return bool((m.get("probe") or {}).get("has_audio"))


def p_has_video(p: Dict[str, Any]) -> bool:
    """A picture that moves; False for a sound file (mp3, wav, m4a …). Manifests from before 0.7.1 have no flag:
    a width stands in."""
    return bool(p.get("has_video", p.get("width")))


def frame_rate_limits(p: Dict[str, Any]) -> Dict[str, float]:
    """Frame-by-frame at this video's rate: the rate used (≤100), the source rate, the longest span per request."""
    src = p.get("fps") or 30.0
    fps = min(src, DENSE_MAX_FPS)
    return {"src": src, "fps": fps, "max_span": DENSE_MAX_FRAMES / fps}


def dense_span(p: Dict[str, Any], start: float, end: Optional[float]) -> Dict[str, Any]:
    """--frames all: every frame from the frame at `start` to the frame at `end`, both included (default:
    to the end of the video). Never thinned: a span longer than one request takes (120 frames — 4 s at
    30 fps) is refused with the pieces to ask for instead."""
    dur = p["duration"]
    lim = frame_rate_limits(p)
    fps, max_span = lim["fps"], lim["max_span"]
    s = max(0.0, min(start, dur))
    e = dur if end is None else max(s, min(end, dur))
    if e - s < 0.5 / fps and end is not None and e >= dur:
        sys.exit(f"--frames all: {tcode(s)} is at the end of the video — check --start / --end")
    if e - s > max_span + 1e-3:
        pieces, t = [], s
        while t < e - 1e-3 and len(pieces) < 4:
            pieces.append(f"--start {tcode(t)} --end {tcode(min(t + max_span, e))}")
            t += max_span
        sys.exit(f"--frames all: every frame from {tcode(s)} to {tcode(e)} is ~{int((e - s) * fps) + 1} frames; one request "
                 f"takes up to {max_span:.1f} s (~{DENSE_MAX_FRAMES} frames at {fps:.0f} per second). Ask in pieces: "
                 + "; ".join(pieces) + (" …" if t < e - 1e-3 else "") + " — or name a shorter moment.")
    return {"start": s, "end": e, "fps": fps, "every": fps >= lim["src"] - 0.01, "whole": s < 0.05 and e > dur - 0.05,
            "to_end": e >= dur - 1e-3}


def prepare(source: Path, out_dir: Path, tools: Tools, names: Dict[str, str], *, frames_mode: str,
            span: Optional[List[Optional[float]]] = None, transcript: bool, force_whisper: bool = False,
            captions_prefer: Optional[List[str]] = None,
            sounds: bool, sound_threshold: float, native: bool, whisper_model: str, language: Optional[str],
            progress: Progress, sounds_model: Optional[str] = None, voice_rescue: bool = False,
            overview_fps: int = FRAME_FPS) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    mtime = source.stat().st_mtime
    m: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = {"source_mtime": None}
        if m.get("link") and m.get("source_mtime") is None:
            # 0.7.2: the words of this link came first (fetch_words); the video file is here now — keep the
            # words (and the wav Whisper read), cut the rest as for any new file
            m = {"transcript": m.get("transcript"), "transcript_meta": m.get("transcript_meta"),
                 "link": dict(m["link"], downloaded=True)}
        elif m.get("source_mtime") != mtime or (m.get("probe") or {}).get("bytes", source.stat().st_size) != source.stat().st_size:
            stale_cleanup(out_dir)  # a different video under the same name: cut everything again
            m = {}

    progress("probe", 0)
    p = m.get("probe") or tools.probe(source)
    dur = max(0.5, p["duration"])
    progress("probe", 100, f"sound file, {tcode(dur)}" if not p_has_video(p) else
             f"{p['width']}×{p['height']}, {tcode(dur)}, {p['fps']:.2f} fps, {'audio' if p['has_audio'] else 'no audio'}")

    # 2) which frames: --frames 1 → the 1 fps lane over the whole video; --frames all → every frame of a span
    dense: Optional[Dict[str, Any]] = m.get("dense")
    sparse_dir = out_dir / names["sparse"].format(w=FRAME_WIDTH)
    frames = list_frames(sparse_dir)
    if frames_mode == "all":
        sp = dense_span(p, (span or [0.0, None])[0] or 0.0, (span or [0.0, None])[1])
        rate = f"{sp['fps']:.2f}".rstrip("0").rstrip(".")
        log(f"  frames: frame by frame — every frame from {tcode(sp['start'])} to {tcode(sp['end'])}, both included, "
            f"at {rate} per second" + ("" if sp["every"] else f" (of {p['fps']:.0f})") + f", {DENSE_WIDTH}px on the long side")
    elif frames_mode == "none" and not p_has_video(p):
        log("  frames: none — a sound file: the words with their time spans, and the sounds on request (--sounds)")
    elif frames_mode == "words":
        log("  frames: none — the words lane: the transcript, the picture on request")
    elif frames_mode == "none":
        log("  frames: none — a lecture lane: the transcript, and single frames on request (--frames at --times mm:ss,…)")
    elif frames_mode == "at":
        log(f"  frames: single frames at the moments asked for, {AT_WIDTH}px on the long side")
    else:
        log(f"  frames: {overview_fps} per second over the whole video, {FRAME_WIDTH}px on the long side — frame by frame for a moment: "
            f"--frames all --start mm:ss --end mm:ss (up to {frame_rate_limits(p)['max_span']:.1f} s)")

    # 3) work copy — only above 1080p by the short side
    work = source
    if needs_work_copy(p):
        work = out_dir / names["work"]
        if not work.exists():
            tools.ffmpeg_run(["-i", str(source), "-vf", scale_filter(p, 1080), "-c:v", "libx264", "-preset", "veryfast",
                              "-b:v", "3500k", "-maxrate", "3800k", "-bufsize", "7000k", "-pix_fmt", "yuv420p",
                              "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-movflags", "+faststart", str(work)], dur, "work copy", progress)
        progress("work copy", 100, "1080p")

    cut_fps = int(m.get("overview_fps") or FRAME_FPS)  # how dense the folder's overview already is
    if frames_mode in ("none", "at", "words"):
        pass  # no overview: a lecture lane (the transcript) or single frames by time — cut after prepare()
    elif frames_mode != "all":
        # the overview lane, 768 px, names = timecodes; --fps 2–4 adds the frames between the seconds to the same
        # folder (mm-ss.cc.jpg) — the 1 a second lane keeps reading only the whole seconds
        sparse_dir.mkdir(exist_ok=True)
        whole = overview_frames_at(frames, FRAME_FPS)
        if (len(whole) < math.floor(dur) or not lane_ok(frames, p, FRAME_WIDTH)
                or m.get("overview_cut") != OVERVIEW_CUT):
            for old in sparse_dir.glob("*.jpg"):
                if frame_time_from_name(old.name) is not None:
                    old.unlink()
            frames, cut_fps = [], FRAME_FPS
        want_fps = max(overview_fps, cut_fps if frames else FRAME_FPS)
        if not frames or (overview_fps > cut_fps):
            tools.ffmpeg_run(["-i", str(work), "-vf", f"{overview_select(want_fps)},{fit_filter(FRAME_WIDTH)}", *tools.vfr(),
                              "-q:v", "4", str(sparse_dir / "%06d.jpg")], dur, "frames", progress)
            rename_sequence(sparse_dir, want_fps, dense=False)  # whole seconds are rewritten with the same frames
            frames = list_frames(sparse_dir)
            cut_fps = want_fps
        progress("frames", 100, f"{len(overview_frames_at(frames, overview_fps))} frames at {overview_fps}/s")
    else:
        # every frame of the span, 512 px; its own folder per span (the whole video keeps Deep Artisan's name)
        rate = f"{sp['fps']:.1f}".rstrip("0").rstrip(".")
        tag = "" if sp["whole"] else span_tag(sp["start"], sp["end"], names)
        dense_dir = out_dir / names["dense"].format(span=tag, fps=rate, w=DENSE_WIDTH)
        same = (dense and dense.get("incl") and dense.get("dir") == str(dense_dir) and abs(dense.get("start", 0.0) - sp["start"]) < 1e-3
                and abs(dense.get("end", dur) - sp["end"]) < 1e-3 and lane_ok(list_frames(dense_dir), p, DENSE_WIDTH))
        if not same:
            dense_dir.mkdir(exist_ok=True)
            for old in dense_dir.glob("*.jpg"):
                if frame_time_from_name(old.name) is not None:
                    old.unlink()
            # the frame at --end is included: half a frame past it, the next one is not
            length = sp["end"] - sp["start"] + (0 if sp["to_end"] else 0.5 / sp["fps"])
            tools.ffmpeg_run(["-ss", f"{sp['start']:.3f}", "-t", f"{length:.4f}", "-i", str(work),
                              "-vf", f"fps={sp['fps']:.4f},{fit_filter(DENSE_WIDTH)}", "-q:v", "5", str(dense_dir / "%06d.jpg")],
                             length, "all frames", progress)
            rename_sequence(dense_dir, sp["fps"], dense=True, offset=sp["start"])
        dense = dict(sp, width=DENSE_WIDTH, dir=str(dense_dir), frames=list_frames(dense_dir), incl=True)
        progress("all frames", 100, f"{len(dense['frames'])} frames at {rate}/s")
        if dense["frames"]:
            wh = jpeg_size(dense["frames"][0]["file"]) or [DENSE_WIDTH, DENSE_WIDTH]
            log(f"  frame by frame: {len(dense['frames'])} frames ≈ {len(dense['frames']) * claude_image_tokens(*wh):,} image "
                "tokens by Anthropic's formula for Claude (other models count differently)")

    # 4) speech: Whisper as in Deep Artisan (its own voice filter, doubtful lines dropped). Other sounds only on
    #    request, for a moment (--sounds --start … --end): the sound model runs on just that span.
    segs: Optional[List[Dict[str, Any]]] = m.get("transcript")
    tr_meta: Dict[str, Any] = m.get("transcript_meta") or {}
    sound_spans: List[Dict[str, Any]] = m.get("sound_spans") or []
    want_tr = transcript and p["has_audio"] and segs is None
    if want_tr and not force_whisper:
        cap = captions_beside(source, language if language != "auto" else None, captions_prefer)
        if cap is not None:
            segs = cap["segments"]
            tr_meta = {"language": cap["language"], "language_source": "captions", "gate": "captions",
                       "source": "captions", "kind": cap["kind"], "file": cap["file"].name}
            (out_dir / "transcript.txt").write_text("\n".join(seg_line(x) for x in segs) + "\n", encoding="utf-8")
            progress("transcript", 100, f"{len(segs)} segments from {cap['file'].name} ({cap['kind']} captions, no Whisper)")
            want_tr = False
    if want_tr:
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is None:  # say so before extracting audio for nothing
            log("  faster-whisper is not installed — transcript skipped (pip install faster-whisper)")
            want_tr = False
    scan_whole = None
    if p["has_audio"] and want_tr and voice_rescue:
        wav32 = out_dir / "audio-32k.wav"
        if not wav32.exists():
            tools.ffmpeg_run(["-i", str(work), "-vn", "-ac", "1", "-ar", str(SOUNDS_SR), "-c:a", "pcm_s16le", "-f", "wav",
                              str(wav32)], dur, "audio 32k", progress)
        scan_whole = panns_scan(wav32, progress, sounds_model)
    if want_tr:
        wav = out_dir / names["audio"]
        if not wav.exists():
            tools.ffmpeg_run(["-i", str(work), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)], dur, "audio", progress)
            progress("audio", 100)
        res = transcribe(wav, whisper_model, language, progress, scan_whole, "hybrid" if voice_rescue else "silero")
        if res is not None:
            segs = res["segments"]
            tr_meta = {k: res[k] for k in ("language", "language_source", "gate")}
            (out_dir / "transcript.txt").write_text("\n".join(seg_line(x) for x in segs) + "\n", encoding="utf-8")
    elif transcript and not p["has_audio"]:
        progress("transcript", 100, "no audio track")
    if sounds and not p["has_audio"]:
        log("  sounds: the video has no audio track")
    elif sounds:
        s0 = max(0.0, min((span or [0.0, None])[0] or 0.0, dur))
        e_req = (span or [0.0, None])[1]
        # the last anchor stands for its whole second: up to the next overview frame
        s1 = dur if e_req is None else max(s0, min(dur, e_req + 1.0 / FRAME_FPS))
        whole = s0 < 0.05 and s1 > dur - 0.05
        tag = "" if whole else span_tag(s0, e_req if e_req is not None else s1, names)
        same_span = lambda x: abs(x["start"] - s0) < 1e-3 and abs(x["end"] - s1) < 1e-3
        found = next((x for x in sound_spans if same_span(x) and x.get("threshold") == sound_threshold), None)
        if found is None:
            # the classifier needs context: a moment is heard with SOUND_CONTEXT s around it (a slam in a
            # one-second cut scores far lower than in the whole video), then only what falls inside is kept
            c0, c1 = max(0.0, s0 - SOUND_CONTEXT), min(dur, s1 + SOUND_CONTEXT)
            wav32 = out_dir / f"audio-32k{tag}.wav"
            if not wav32.exists() or abs(wav32.stat().st_size - 44 - (c1 - c0) * SOUNDS_SR * 2) > SOUNDS_SR:
                tools.ffmpeg_run((["-ss", f"{c0:.3f}", "-t", f"{c1 - c0:.3f}"] if not whole else []) +
                                 ["-i", str(work), "-vn", "-ac", "1", "-ar", str(SOUNDS_SR), "-c:a", "pcm_s16le", "-f", "wav",
                                  str(wav32)], c1 - c0, "audio 32k", progress)
            scan = panns_scan(wav32, progress, sounds_model)
            if scan is not None and not whole:
                probs, labels, _ = scan
                a, b = int(round((s0 - c0) * 10)), int(round((s1 - c0) * 10))  # 10 rows per second
                scan = (probs[a:b], labels, s1 - s0)
            if scan is not None:
                shift = lambda e: dict(e, start=round(e["start"] + s0, 1), end=round(e["end"] + s0, 1))
                events = [shift(e) for e in sound_events(scan, sound_threshold)]
                regions = [[round(r0 + s0, 2), round(r1 + s0, 2)] for r0, r1 in speech_regions(scan)]
                if segs is not None:
                    events = sorted(events + indistinct_voices(regions, segs, events), key=lambda e: (e["start"], e["label"]))
                found = {"start": round(s0, 3), "end": round(s1, 3), "events": events, "threshold": sound_threshold,
                         "note": sounds_note(scan, sound_threshold)}
                sound_spans = sorted([x for x in sound_spans if not same_span(x)] + [found], key=lambda x: x["start"])
                head = (f"# sounds other than speech, {tcode(s0)} to {tcode(s1)} (through the whole second of the last "
                        f"anchor), AudioSet classifier, threshold {sound_threshold}")
                body = [sound_line(e) for e in events] + [found["note"]]
                (out_dir / f"sounds{tag}.txt").write_text(head + "\n" + "\n".join(body) + "\n", encoding="utf-8")
        if found is not None:
            progress("sounds", 100, f"{len(found['events'])} sound tag(s) for {tcode(s0)}–{tcode(s1)}")
            log(f"  sounds {tcode(s0)}–{tcode(s1)} (written to sounds{tag}.txt):")
            for e in found["events"] or []:
                log(f"    {sound_line(e)}")
            if found.get("note"):
                log(f"    {found['note']}")

    # 5) native parts on request
    native_parts: Optional[List[Dict[str, Any]]] = m.get("native")
    plan = plan_native(p)
    if native and not native_parts:
        native_parts = []
        if not plan["reencode"]:
            native_parts.append({"file": str(source), "start": 0.0, "dur": dur, "bytes": p["bytes"]})
        else:
            ndir = out_dir / names["native"]
            ndir.mkdir(exist_ok=True)
            for i, part in enumerate(plan["parts"]):
                f = ndir / (f"part-{i + 1:02d}.mp4" if len(plan["parts"]) > 1 else "native.mp4")
                if not f.exists():
                    fps_f = ",fps=30" if p["fps"] > 30.5 else ""
                    tools.ffmpeg_run(["-ss", str(part["start"]), "-t", str(part["dur"]), "-i", str(source),
                                      "-vf", f"{scale_filter(p, plan['height'])}{fps_f}", "-c:v", "libx264", "-preset", "veryfast",
                                      "-b:v", f"{plan['kbps']}k", "-maxrate", f"{round(plan['kbps'] * 1.1)}k", "-bufsize", f"{plan['kbps'] * 2}k",
                                      "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-movflags", "+faststart", str(f)],
                                     part["dur"], f"native {i + 1}/{len(plan['parts'])}", progress)
                native_parts.append({"file": str(f), "start": part["start"], "dur": part["dur"], "bytes": f.stat().st_size})
        progress("native", 100, f"{len(native_parts)} part(s), {plan['height']}p @ {plan['kbps']} kbps")

    m = {
        "tool": "video2llm", "version": VERSION, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(source), "source_mtime": mtime, "dir": str(out_dir), "probe": p, "work": str(work),
        "frames": frames, "overview_cut": OVERVIEW_CUT if frames else m.get("overview_cut"), "overview_fps": cut_fps, "dense": dense,
        "transcript": segs, "transcript_meta": tr_meta, "sound_spans": sound_spans,
        "native": native_parts, "native_plan": plan,
        "link": m.get("link"),  # 0.7.2: the file came from a link — the lanes name the site and the address
    }
    manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    progress("done", 100, str(out_dir))
    return m


# ── the lane: what the model gets ───────────────────────────────────────────────────────────────
def seg_line(s: Dict[str, Any]) -> str:
    a, b = tcode(s.get("start", 0)), tcode(s.get("end", s.get("start", 0)))
    return f"[{a}–{b}] «{s['text'].strip()}»" if a != b else f"[{a}] «{s['text'].strip()}»"


GAP_TEXT = "no clear speech recognised — voices may still be heard; sound tags can be asked for"


def timeline_items(m: Dict[str, Any]) -> List[Any]:
    """Spoken segments and sound events on one timeline: [(start, line, entry), …] sorted by time; `entry` is
    {kind: speech | sound | gap, a, b, text} for lanes that place each line under the frame it belongs to."""
    items = [(s.get("start", 0), seg_line(s), {"kind": "speech", "a": s.get("start", 0), "b": s.get("end", s.get("start", 0)),
                                               "text": s["text"].strip()}) for s in (m.get("transcript") or [])]
    segs = m.get("transcript")
    if segs is not None and (m.get("probe") or {}).get("has_audio"):
        # say where no speech was recognised, so the model does not read a gap as missing material
        dur = m["probe"]["duration"]
        edges = [0.0] + [x for sg in segs for x in (sg.get("start", 0), sg.get("end", 0))] + [dur]
        for a, b in zip(edges[0::2], edges[1::2]):
            if b - a >= (5.0 if b >= dur - 0.05 or a <= 0.05 else 10.0):
                items.append((a + 0.001, f"[{tcode(a)}–{tcode(b)}] ({GAP_TEXT})", {"kind": "gap", "a": a, "b": b, "text": GAP_TEXT}))
    seen = set()
    for x in (m.get("sound_spans") or []):  # overlapping requests: each tag once
        for e in x["events"]:
            key = (e["start"], e["end"], e["label"])
            if key not in seen:
                seen.add(key)
                items.append((e["start"], sound_line(e), {"kind": "sound", "a": e["start"], "b": e["end"], "text": e["label"]}))
    return sorted(items, key=lambda x: x[0])


def tdec(sec: float) -> str:
    """mm:ss.d — a tenth of a second, cut down (not rounded), so 7.95 s stays in the frame [00:07]."""
    t = math.floor(sec * 10 + 1e-6) / 10
    return f"{tcode(math.floor(t))}.{int(round((t % 1) * 10)) % 10}"


def entry_line(ftag: str, e: Dict[str, Any]) -> str:
    """One line under a sheet: the frame it begins at, what is heard, its own time span (to a tenth)."""
    a, b = tdec(e["a"]), tdec(e["b"])
    span = f"{a}–{b}" if a != b else a
    if e["kind"] == "speech":
        return f"[{ftag}] → «{e['text']}» (speech {span})"
    if e["kind"] == "sound":
        return f"[{ftag}] → {e['text']} (sound {span})"
    return f"[{ftag}] → {e['text']} ({span})"


def fit_dims(w: float, h: float, size: int) -> List[float]:
    """A frame fitted by its long side into size×size, never up."""
    k = min(size, max(w, h, 1)) / max(w, h, 1)
    return [w * k, h * k]


class Rerun:
    """How the model asks for more of this video. By default a shell command — this script with the same
    settings (`cmd`); with --agent-tool NAME the header tells an agent harness that runs the script itself
    (Deep Artisan) to call that tool: {"video": …, "what": "next" | "frames" | "sounds", "start", "end"}."""

    def __init__(self, cmd: str, video: Path, tool: Optional[str] = None, auto: bool = False):
        self.cmd, self.video, self.tool, self.auto = cmd, video, tool, auto  # auto: the lane was chosen by --frames auto

    def _call(self, what: str, start: str, end: Optional[str] = None) -> str:
        args = {"video": str(self.video), "what": what, "start": start}
        if end is not None:
            args["end"] = end
        return f"call {self.tool} with {json.dumps(args, ensure_ascii=False)}"

    def frames(self, start: str = "mm:ss", end: str = "mm:ss") -> str:
        return self._call("frames", start, end) if self.tool else f"run {self.cmd} --frames all --start {start} --end {end}"

    def sounds(self, start: str = "mm:ss", end: str = "mm:ss") -> str:
        return self._call("sounds", start, end) if self.tool else f"run {self.cmd} --sounds --start {start} --end {end}"

    def next(self, at: float) -> str:  # the next part of the overview must stay the overview — not be decided again
        return self._call("next", tcode(at)) if self.tool else f"run {self.cmd}{' --frames 1' if self.auto else ''} --start {tcode(at)}"

    def at(self, times: Optional[List[str]] = None) -> str:
        """Single frames at the moments named: {what: "at", times: [...]} for the tool, --frames at --times for the command."""
        ts = times or ["mm:ss", "mm:ss"]
        if self.tool:
            return f"call {self.tool} with " + json.dumps({"video": str(self.video), "what": "at", "times": ts}, ensure_ascii=False)
        return f"run {self.cmd} --frames at --times {','.join(ts)}"

    def lecture(self) -> str:
        """The lecture lane of the same video: {what: "lecture"} for the tool, --frames none for the command."""
        if self.tool:
            return f"call {self.tool} with " + json.dumps({"video": str(self.video), "what": "lecture"}, ensure_ascii=False)
        return f"run {self.cmd} --frames none"

    def next_file(self, at: float) -> str:
        """"it writes lane_from_mm-ss.md, " for the command; the tool returns the lane itself."""
        return "" if self.tool else f"it writes lane_from_{fmt_t(at)}.md, "

    def writes(self, name: str) -> str:
        return "" if self.tool else f"it writes {name}; "

    def these(self) -> str:
        return f"the {self.tool} tool" if self.tool else "these commands"

    def cannot(self) -> str:
        return (f"If the {self.tool} tool is not available to you, tell the user which request you need." if self.tool else
                "If you cannot run commands, give the user the command and ask for what it writes.")

    def cannot_next(self, more_txt: str) -> str:
        return (f"if the tool is not available, ask the user to send {more_txt}." if self.tool else
                f"if you cannot run commands, ask the user to send {more_txt}.")


def lane_blocks(m: Dict[str, Any], label: str, start: float, max_frames: Optional[int], dense: bool,
                end: Optional[float] = None, kind: str = "message", part: Optional[Dict[str, Any]] = None,
                rerun: Optional["Rerun"] = None, per_sheet: int = 0, fps_overview: int = FRAME_FPS,
                sheet_tokens: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Header + frames interleaved with the words spoken around each frame; the rest of the
    transcript as text at the end (cheap — the model knows all the speech). `kind` says what the
    header describes: "message" (images in a request or lane.md), "pdf" (pages; part = {index, parts, files})
    or "sheets" (contact sheets; part = {first, last, total, batch, batches}). `per_sheet` > 1: lane.md shows
    the frames that many to an image (agent_sheets) and the header says so. `fps_overview` 2–4: the denser
    overview (--fps). `sheet_tokens` {"over", "dense"}: Claude tokens of one frame as it sits on its sheet when
    --sheet-frames scales the frames — the costs in the header follow. The header is a short list of rules:
    the model picks the moments itself, fetches the rest
    of the overview itself, asks the user only yes or no with the cost before a frame-by-frame or sound request,
    and uses no other tool on the video."""
    p = m["probe"]
    dur = p["duration"]
    d = m.get("dense") if dense else None
    use_dense = bool(d and d.get("frames"))
    lane = d["frames"] if use_dense else overview_frames_at(m.get("frames") or [], fps_overview)
    fps = d["fps"] if use_dense else fps_overview
    width = d["width"] if use_dense else FRAME_WIDTH
    limit = max_frames or (DENSE_MAX_FRAMES + 1 if use_dense else SPARSE_PER_MESSAGE)  # +1: both anchors are included
    win = [f for f in lane if f["t"] >= start - 1e-6 and (end is None or f["t"] <= end + 1e-6)][:limit]  # --end is included
    to_sec = (win[-1]["t"] + 1 / fps) if win else start

    def tag(t: float) -> str:
        if not use_dense:
            return tcode_frac(t)  # [00:12], and [00:12.25] between two seconds in a denser overview
        # hundredths are rounded; the frame number in the video (counted from 0) is exact
        return f"{tcode(math.floor(t))}.{int(round((t % 1) * 100)):02d} #{int(round(t * (p.get('fps') or fps)))}"

    spans = m.get("sound_spans") or []
    span_txt = ", ".join(f"{tcode(x['start'])}–{tcode(x['end'])}" for x in spans)
    sounds_note = (f"; sounds other than speech, detected by an AudioSet classifier, are tagged (sound: …) for {span_txt} only — "
                   "the moments they were asked for" if spans else "; only speech is transcribed — other sounds are not listed")
    if not p["has_audio"]:
        audio_note = "no audio"
    elif m.get("transcript") is not None and (m.get("transcript_meta") or {}).get("source") == "captions":
        kind = (m.get("transcript_meta") or {}).get("kind")
        audio_note = ("with audio (the words are the site's own captions"
                      + (" — automatic ones, they can mishear; read them with the frames" if kind == "automatic" else
                         ", written by people" if kind == "manual" else " — read them with the frames")
                      + ("" if m.get("transcript") else ": the caption file is empty") + sounds_note + ")")
    elif m.get("transcript") is not None:
        audio_note = ("with audio (automatic speech transcript — it can mishear words, read it with the frames"
                      + ("" if m.get("transcript") else ": no words were recognised") + sounds_note + ")")
    else:
        audio_note = "with audio but no transcript" + (f" (sound tags for {span_txt})" if spans else "")
    what = f"[{label}: video \"{Path(m['source']).name}\", {tcode(dur)}, {p['width']}×{p['height']}, {audio_note}.\n"
    lim = frame_rate_limits(p)
    max_span = f"{lim['max_span']:.1f}"
    wh = jpeg_size(win[0]["file"]) if win else None
    pw, ph = (wh if wh else [p["width"], p["height"]]) if not use_dense else [p["width"], p["height"]]
    over = claude_image_tokens(*fit_dims(pw, ph, FRAME_WIDTH)) if pw and ph else 0
    per_dense = (claude_image_tokens(*wh) if use_dense and wh else
                 claude_image_tokens(*fit_dims(pw, ph, DENSE_WIDTH)) if pw and ph else 0)
    if sheet_tokens:  # --sheet-frames: the frames sit scaled on their sheets — the costs follow
        over = sheet_tokens.get("over") or over
        per_dense = sheet_tokens.get("dense") or per_dense
    per_sec = round(per_dense * lim["fps"])
    example = per_dense * (int(round(2 * lim["fps"])) + 1)  # [00:12]–[00:14], both anchors included
    cap = per_dense * (DENSE_MAX_FRAMES + 1)
    tok = (lambda n: f"{n:,} tokens") if per_dense else (lambda n: "N tokens")
    tag_form = ("[mm:ss.cc #frame] (#n = the frame's number in the video, counted from 0)" if use_dense else
                "[mm:ss] (the frames between two seconds: [mm:ss.cc])" if fps > 1 else "[mm:ss]")

    # ── what the images are ──
    if kind == "pdf":
        view = ("The pages show consecutive frames of one moving picture, in time order — left to right, then top to bottom, "
                "page after page; treat them as one moving picture, not as separate photos. Each frame is captioned with its "
                "timestamp above it; under each row of frames are the words spoken and the sounds heard in those seconds, "
                "each line with its own time span.")
    elif kind == "sheets":
        view = ("The attached images are contact sheets of consecutive frames of one moving picture, in time order — left to "
                "right, then top to bottom, sheet after sheet (each sheet has its number and time span along the top); treat "
                "them as one moving picture, not as separate photos. Each frame has its timestamp above it.")
    elif per_sheet > 1:
        view = (f"The images are contact sheets of consecutive frames of one moving picture, in time order — {per_sheet} frames "
                f"to a sheet, left to right, then top to bottom, each with its tag {tag_form} above it; treat them as one moving "
                "picture, not as separate photos. Under each sheet are the words and sounds of its frames: each line starts "
                "with the tag of the frame it begins at, then what is heard and its own time span. Open the sheets in order, "
                "several in one step where your tools allow it.")
    else:
        view = (f"The images are consecutive frames of one moving picture, in time order, each tagged {tag_form}; treat them "
                "as one moving picture, not as separate photos. The words and sounds of each frame follow it.")

    rules: List[str] = []
    fps_txt = "1 frame per second" if fps_overview <= 1 else f"{fps_overview} frames per second"
    if use_dense:
        view += (f" This is frame by frame: every frame from the overview frame [{tcode(d['start'])}] to [{tcode(d['end'])}], "
                 f"both included — {len(lane)} frames at {fps:.2f}".rstrip("0").rstrip(".") + " per second"
                 + ("" if d.get("every") else f" (of {lim['src']:.0f} in the video)") + f", {width}px on the long side; "
                 f"[{tag(d['start'])}] is the same frame as the overview frame [{tcode(d['start'])}]"
                 + (f"; these frames ≈ {per_dense * len(win):,} image tokens by Anthropic's formula for Claude" if per_dense else "")
                 + ".")
        rules.append(f"Look at fine motion, flicker and the changes from frame to frame. The whole video at {fps_txt} "
                     "is the overview lane (lane.md).")
        rules.append("Another moment frame by frame, or the sounds of a moment: you choose it, then ask the user — yes or no "
                     f"with the cost (a second frame by frame ≈ {tok(per_sec)}; sounds come with every frame where they happen, "
                     f"up to ≈ {tok(cap)})"
                     + (f"; after the yes, {rerun.frames()} (up to {max_span} s) or {rerun.sounds()}" if rerun else
                        "; after the yes, the user runs video2llm with --frames all (or --sounds) --start mm:ss --end mm:ss")
                     + ". Ask before every such request.")
    else:
        if kind == "pdf":
            n, i, files = part["parts"], part["index"], part["files"]
            unit = "This file" + (f" (part {i + 1} of {n}: {files[0]} … {files[-1]})" if n > 1 else "")
            more_here = i < n - 1
            more_txt = "the next PDF file"
        elif kind == "sheets":
            unit = f"This message (sheets {part['first']:02d}–{part['last']:02d} of {part['total']:02d})"
            more_here = part["batch"] < part["batches"]
            more_txt = (f"the files of message-{part['batch'] + 1:02d} (sheets {part['last'] + 1:02d}–{part['total']:02d} come in "
                        "the next messages)")
        else:
            unit, more_here, more_txt = "This lane", False, ""
        r1 = (f"The video is shown at {fps_txt} — the normal view, enough for what happens and who says what. "
              f"{unit} carries {tcode(start)}–{tcode(min(to_sec, dur))}.")
        tail = to_sec < dur and tcode(to_sec) != tcode(dur)
        nxt_cost = (f", ≈ {over * min(LANE_MD_FRAMES, int(math.ceil((dur - to_sec) * fps_overview))):,} tokens a part"
                    if over and tail else "")
        nxt_file = rerun.next_file(to_sec) if rerun else ""
        if tail:
            if more_here:
                r1 += (f" The rest ({tcode(to_sec)}–{tcode(dur)}) is not here. When your answer concerns it (a question about "
                       "the whole video does), "
                       + (f"get it yourself first, without asking the user — {rerun.next(to_sec)} ({nxt_file}the next part"
                          f"{nxt_cost}; repeat until the end); {rerun.cannot_next(more_txt)}" if rerun
                          else f"ask the user to send {more_txt}."))
            elif rerun:
                r1 += (f" Frames {tcode(to_sec)}–{tcode(dur)} are not here: when your answer concerns them (a question about "
                       f"the whole video does), get them yourself first, without asking the user — {rerun.next(to_sec)} "
                       f"({nxt_file}the next part{nxt_cost}; repeat until the end).")
            else:
                r1 += (f" Frames {tcode(to_sec)}–{tcode(dur)} are not here: when your answer concerns them, ask the user for "
                       f"the next part (from {tcode(to_sec)}).")
        r1 += (" Open every sheet the question concerns — all of them for a question about the whole video, in order, not a "
               "sample — and never describe from the transcript alone what you have not seen in the frames.")
        rules.append(r1)
        rules.append("You choose the moments, by what you see and hear in the lane — the user does not know the timecodes. "
                     "Ask the user only for a yes or a no, with the cost.")
        rules.append(f"Frame by frame — when {fps_txt} cannot answer (something happens between two frames: fast "
                     "motion, a gesture, a flicker, a cut), do not guess; ask, e.g. \"To answer this I need every frame from "
                     f"[00:12] to [00:14] — about {tok(example)}. Shall I?\" The anchors are two overview frames by their tags, "
                     f"both included (00:12.5 narrows it); one request covers up to {max_span} s (~{DENSE_MAX_FRAMES} frames at "
                     f"{lim['fps']:.0f} per second), a longer moment goes in consecutive pieces. "
                     + (f"After the yes, {rerun.frames('00:12', '00:14')} — {rerun.writes('lane_00-12_00-14.md')}look at its images."
                        if rerun else
                        "After the yes, the user runs video2llm with --frames all --start 00:12 --end 00:14 and sends you what "
                        "it writes."))
        if p["has_audio"]:
            rules.append("Sounds other than speech (music, a slam, laughter, a crowd, an engine) are not in the lane. When the "
                         "answer depends on them, ask, e.g. \"To answer this I need the sounds from [00:40] to [00:44] with the "
                         f"frames where they happen — up to about {tok(cap)}. Shall I?\" One sound request may cover any span, a "
                         f"whole minute too (it covers the whole second of the last anchor); the {max_span} s limit is only on the "
                         "frames that come with it — they are taken where a sound is found. "
                         + (f"After the yes, {rerun.sounds('00:40', '00:44')} — it gives the tags and "
                            f"{rerun.writes('lane_sounds_00-40_00-44.md')}every frame where each sound happens: look at them to "
                            "see what makes it." if rerun else
                            "After the yes, the user runs video2llm with --sounds --start 00:40 --end 00:44 and sends you what "
                            "it writes.")
                         + " If nothing besides speech is found, tell the user that no other sounds could be identified there "
                           "— not that there were none.")
        rules.append(f"Ask before every request of rules 3{' and 4' if p['has_audio'] else ''} — a second or follow-up one "
                     "too, and also when the user's question is itself about a sound or an exact moment: the yes is for the "
                     "tokens. The rest of the overview (rule 1) needs no yes.")
    if rerun:
        rules.append(f"Use only {rerun.these()} for this video. Do not search for or open the video file, do not run ffmpeg, "
                     "ffprobe or any other tool on it, do not read video2llm.json, and do not crop, zoom or combine frames "
                     "yourself — open only the images the lanes link to."
                     # 08.10: with the sheets inline (--agent-tool) the model does not know where the frames are; asked for one
                     # frame by name it searched the whole home folder for two minutes
                     + (f" The overview frames themselves are the files {shell_arg(Path(lane[0]['file']).parent)}/mm-ss.jpg "
                        f"({FRAME_WIDTH}px) — read one from there only when the user asks for that frame as a file."
                        if rerun.tool and lane and not use_dense else "")
                     + (f" For a slide, an interface or text on screen at a few exact moments, single larger frames "
                        f"({AT_WIDTH}px) are cheaper than frame by frame: {rerun.at(['03:12', '07:40'])} — up to 6 without asking, "
                        "more with the cost." if not use_dense else "")
                     + " If a detail is too small to tell, say so.")
        rules.append(rerun.cannot())
        if not use_dense:
            rules.append("If the words show that this is something taught, shown or explained — a lecture, a tutorial, a how-to, "
                         "a course lesson, a webinar, a talk, a review or a breakdown of a tool or a technique — stop reading "
                         f"frames of a talking person: {rerun.lecture()} gives the LECTURE lane (the whole transcript, no frames, "
                         "single frames where the screen matters on request) and, after your answer, the question about a "
                         "Guide / Notes of it. Footage to watch (a film, a clip, a vlog, home video) stays here.")
    else:
        rules.append("Do not cut frames from the video or analyse its sound some other way.")
    cost = ""
    if over and not use_dense:
        cost = (f"\nCost, by Anthropic's formula for Claude (other models count differently; the exact figure comes only with "
                f"the response): an overview frame ≈ {over:,} tokens, a frame-by-frame one ≈ {per_dense:,} ({DENSE_WIDTH}px on "
                f"the long side), a second frame by frame ≈ {per_sec:,} — {per_sec / max(1, over * fps_overview):.0f}× the same "
                f"second in the overview; the frames here ≈ {over * len(win):,}.")
    if kind == "pdf":
        close = ("\nThe whole transcript and sound list is here too, before and after the frames. If you can read only the text "
                 "of this PDF and not the frames (some chat apps pass PDFs as text only), say so instead of guessing what is "
                 "on screen.")
    elif kind == "sheets":
        close = ("\nBelow is the whole transcript and sound list, each line with its time span — match it to the frames by "
                 "time. If some sheets did not come through, say which instead of guessing.")
    else:
        close = ""
    head = (what + view + "\nRules:\n" + "\n".join(f"{k + 1}. {r}" for k, r in enumerate(rules)) + cost + close + "]")
    out: List[Dict[str, Any]] = [{"type": "text", "text": head, "role": "head"}]
    # one timeline: spoken segments and sound events, each placed after the frame of the second it starts in
    items = timeline_items(m)
    si = 0
    before = []
    while si < len(items) and items[si][0] < start:
        before.append(items[si][1]); si += 1
    if before:
        out.append({"type": "text", "text": f"[transcript and sounds before {tcode(start)}]\n" + "\n".join(before), "role": "before"})
    for f in win:
        lines, ents = [], []
        while si < len(items) and items[si][0] < f["t"] + 1 / fps:
            lines.append(items[si][1]); ents.append(items[si][2]); si += 1
        out.append({"type": "text", "text": f"[{tag(f['t'])}]", "role": "tag"})
        out.append({"type": "image", "path": f["file"], "t": f["t"], "tag": tag(f["t"]), "lines": lines, "entries": ents})
        if lines:
            out.append({"type": "text", "text": "\n".join(lines), "role": "lines"})
    rest = [x[1] for x in items[si:]]
    if rest:
        out.append({"type": "text", "text": f"[transcript and sounds after {tcode(to_sec)}]\n" + "\n".join(rest), "role": "rest"})
    return out


def native_blocks(m: Dict[str, Any], label: str, part_index: int) -> List[Dict[str, Any]]:
    parts = m.get("native") or []
    if not parts:
        sys.exit("no native parts — run with --native first")
    i = min(part_index, len(parts) - 1)
    part = parts[i]
    p = m["probe"]
    head = (f"[{label}: video \"{Path(m['source']).name}\", {tcode(p['duration'])}, {p['width']}×{p['height']}"
            + (f"; split into {len(parts)} parts for the request limit — this message carries part {i + 1} "
               f"({tcode(part['start'])}–{tcode(part['start'] + part['dur'])}); ask the user for the next part when you need it"
               if len(parts) > 1 else "") + "]")
    return [{"type": "text", "text": head}, {"type": "video", "path": part["file"], "mime": "video/mp4"}]


def b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def cut_every_frame(tools: "Tools", work: Path, out_dir: Path, names: Dict[str, str], p: Dict[str, Any], s: float,
                    e: float, progress: Progress) -> Dict[str, Any]:
    """Every frame from s to e (both included), 512 px on the long side, named by time — reused when already cut."""
    fps = min(p.get("fps") or 30.0, DENSE_MAX_FPS)
    rate = f"{fps:.1f}".rstrip("0").rstrip(".")
    folder = out_dir / names["dense"].format(span=span_tag(s, e, names), fps=rate, w=DENSE_WIDTH)
    frames = [f for f in list_frames(folder) if s - 1e-3 <= f["t"] <= e + 0.5 / fps]
    want = int(math.floor((e - s) * fps + 1e-6)) + 1
    if len(frames) < want or not lane_ok(frames, p, DENSE_WIDTH):
        folder.mkdir(exist_ok=True)
        for old in folder.glob("*.jpg"):
            if frame_time_from_name(old.name) is not None:
                old.unlink()
        length = e - s + 0.5 / fps
        tools.ffmpeg_run(["-ss", f"{s:.3f}", "-t", f"{length:.4f}", "-i", str(work), "-vf",
                          f"fps={fps:.4f},{fit_filter(DENSE_WIDTH)}", "-q:v", "5", str(folder / "%06d.jpg")],
                         length, "frames of the sounds", progress)
        rename_sequence(folder, fps, dense=True, offset=s)
        frames = list_frames(folder)
    return {"start": s, "end": e, "fps": fps, "width": DENSE_WIDTH, "dir": str(folder), "frames": frames}


def sound_check(m: Dict[str, Any], tools: "Tools", out_dir: Path, names: Dict[str, str], label: str, s0: float,
                s1: float, threshold: float, progress: Progress, rerun: Optional["Rerun"] = None,
                sheets: bool = True) -> Optional[Dict[str, Any]]:
    """A sound request answered with its source in view: the tags of the moment and every frame where each sound
    happens (events ± 0.3 s, up to one request's worth of frames), speech lines in place — lane_sounds_a_b.md."""
    p = m["probe"]
    span = next((x for x in m.get("sound_spans") or [] if abs(x["start"] - s0) < 1e-3 and abs(x["end"] - s1) < 1e-3
                 and x.get("threshold") == threshold), None)
    if span is None:
        return None
    lim = frame_rate_limits(p)
    fps = lim["fps"]
    windows: List[List[float]] = []
    for ev in sorted((x for x in span["events"] if x["label"] != "Indistinct voices"), key=lambda x: x["start"]):
        a, b = max(s0, ev["start"] - 0.3), min(s1, ev["end"] + 0.3)
        if windows and a <= windows[-1][1] + 0.3:
            windows[-1][1] = max(windows[-1][1], b)
        else:
            windows.append([a, b])
    kept, skipped, budget = [], [], lim["max_span"]
    if not p_has_video(p):
        windows = []  # a sound file: nothing to show, the tags are the answer
    for a, b in windows:
        if budget <= 0.05:
            skipped.append([a, b])
            continue
        if b - a > budget:
            skipped.append([a + budget, b])
            b = a + budget
        kept.append([a, b])
        budget -= b - a
    cuts = [cut_every_frame(tools, Path(m.get("work") or m["source"]), out_dir, names, p, a, b, progress) for a, b in kept]
    n_frames = sum(len(c["frames"]) for c in cuts)
    wh = jpeg_size(cuts[0]["frames"][0]["file"]) if cuts and cuts[0]["frames"] else None
    per = claude_image_tokens(*wh) if wh else 0
    grid = lane_grid(*wh) if (wh and sheets and pillow_ok()) else [1, 1]
    n_sheet = grid[0] * grid[1]

    def tag(t: float) -> str:
        return f"{tcode(math.floor(t))}.{int(round((t % 1) * 100)):02d} #{int(round(t * (p.get('fps') or fps)))}"

    tag_form = "[mm:ss.cc #frame] (#n = the frame's number in the video, counted from 0)"
    other = ("Another moment frame by frame, or the sounds of another moment: you choose it, then ask the user — yes or no "
             f"with the cost (a second frame by frame ≈ {per * round(fps):,} tokens)"
             + (f"; after the yes, {rerun.frames()} (up to {lim['max_span']:.1f} s) or {rerun.sounds()}" if rerun else "")
             + ". Ask before every such request.")
    only = (f"Use only {rerun.these()} for this video. Do not search for or open the video file, do not run ffmpeg, ffprobe "
            "or any other tool on it or on its sound, and do not crop, zoom or combine frames yourself — open only the "
            "images this lane links to." if rerun else "Do not cut frames or analyse the sound some other way.")
    if cuts:
        view = (f"Each moment comes as contact sheets of consecutive frames, {n_sheet} to a sheet, left to right, then top to "
                f"bottom, each frame with its tag {tag_form} above it; under each sheet are the words heard in its frames, "
                "each line starting with the tag of the frame it begins at." if n_sheet > 1 else
                f"Each frame is tagged {tag_form}; the words heard in it follow it.")
        head = (f"[{label}: video \"{Path(m['source']).name}\" — sound check of {tcode(s0)}–{tcode(s1)}: the sounds other "
                "than speech an AudioSet classifier heard there, each with every frame of the moment it happens in "
                f"({fps:.0f} per second, {DENSE_WIDTH}px on the long side). {view} Speech lines are the automatic transcript "
                f"and can mishear words. {len(cuts)} moment(s), {n_frames} frames"
                + (f" ≈ {per * n_frames:,} image tokens by Anthropic's formula for Claude" if per else "")
                + (". No frames for " + ", ".join(f"{tcode(a)}–{tcode(b)}" for a, b in skipped)
                   + " (over one request) — ask for them separately" if skipped else "") + ". " + (span.get("note") or "")
                + "\nRules:\n1. Say what makes each sound only from what the frames show; when they do not show it, say so."
                f"\n2. {other}\n3. {only}"
                + (f"\n4. {rerun.cannot()}" if rerun else "")
                + "]")
    else:
        head = (f"[{label}: sound file \"{Path(m['source']).name}\" — sound check of {tcode(s0)}–{tcode(s1)}: the sounds "
                "other than speech an AudioSet classifier heard there, with their times. " + (span.get("note") or "")
                + f"\nRules:\n1. Say what is heard from the tags and the words; a sound file has no picture to show. {only}]"
                if not p_has_video(p) else
                f"[{label}: video \"{Path(m['source']).name}\" — sound check of {tcode(s0)}–{tcode(s1)}: "
                + (span.get("note") or "nothing besides speech was identified") + " No frames: there is no sound to show the "
                "source of; the overview frames of this moment stand. Tell the user that no other sounds could be "
                f"identified there — not that there were none.\nRules:\n1. {other}\n2. {only}]")
    items = [x for x in timeline_items(m) if s0 - 1e-3 <= x[0] <= s1]
    out: List[Dict[str, Any]] = [{"type": "text", "text": head, "role": "head"}]
    tags = [sound_line(e) for e in span["events"]]
    if tags:
        out.append({"type": "text", "text": "[sound tags of this moment]\n" + "\n".join(tags), "role": "before"})
    for c in cuts:
        out.append({"type": "text", "text": f"[frames {tag(c['start'])} – {tag(c['end'])}]", "role": "before"})
        lane = c["frames"]
        for i, f in enumerate(lane):
            nxt = lane[i + 1]["t"] if i + 1 < len(lane) else f["t"] + 1 / fps
            here = [x for x in items if f["t"] - 1e-3 <= x[0] < nxt and x[2]["kind"] != "sound"]
            lines = [x[1] for x in here]
            out.append({"type": "text", "text": f"[{tag(f['t'])}]", "role": "tag"})
            out.append({"type": "image", "path": f["file"], "t": f["t"], "tag": tag(f["t"]), "lines": lines,
                        "entries": [x[2] for x in here]})
            if lines:
                out.append({"type": "text", "text": "\n".join(lines), "role": "lines"})
    if n_sheet > 1:
        out = agent_sheets(out, grid)
    path = out_dir / f"lane_sounds{span_tag(s0, s1 - 1.0 / FRAME_FPS if s1 - s0 >= 1 else s1, names)}.md"
    write_markdown(out, "", path)
    return {"path": path, "moments": len(cuts), "frames": n_frames, "tokens": per * n_frames, "skipped": skipped}


# ── single frames by time, and the lecture lane (user decision 08.10.2026) ────────────────────────────
def cut_frames_at(tools: "Tools", work: Path, out_dir: Path, names: Dict[str, str], times: List[float],
                  progress: Progress) -> List[Dict[str, Any]]:
    """One frame at each moment, AT_WIDTH on the long side, in <out>/frames_at_1024px/mm-ss.cc.jpg — cut once."""
    folder = out_dir / names["at"].format(w=AT_WIDTH)
    folder.mkdir(exist_ok=True)
    out: List[Dict[str, Any]] = []
    for i, t in enumerate(times):
        f = folder / f"{fmt_t(math.floor(t))}.{int(round((t % 1) * 100)):02d}.jpg"
        if not f.exists():
            r = run([tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(work),
                     "-frames:v", "1", "-vf", fit_filter(AT_WIDTH), "-q:v", "3", str(f)])
            if r.returncode != 0 or not f.exists():
                log(f"  frame at {tcode(t)}: ffmpeg failed — {(r.stderr or '').strip().splitlines()[-1:] or ''}")
                continue
        out.append({"t": t, "file": str(f)})
        progress("frames at", int((i + 1) / len(times) * 100))
    progress("frames at", 100, f"{len(out)} frame(s)")
    return out


def read_info_json(source: Path) -> Dict[str, Any]:
    """yt-dlp's `<name>.info.json` beside a downloaded video, or {}."""
    f = source.with_name(source.stem + ".info.json")
    try:
        return json.loads(f.read_text(encoding="utf-8", errors="replace")) if f.is_file() else {}
    except (OSError, ValueError):
        return {}


_LECTURE_WORDS = re.compile(
    r"\b(lecture|lesson|tutorial|how[- ]?to|guide|walkthrough|course|class|webinar|workshop|masterclass|explained|"
    r"introduction to|intro to|crash course|step[- ]by[- ]step|setup|install|configure|"
    r"лекци|урок|туториал|инструкци|гайд|курс|вебинар|мастер[- ]класс|разбор|обучени|настройк|установк|как (?:сделать|настроить|установить|работать)|"
    r"objašnj|erklärt|anleitung|kurs|clase|lección|cómo|tuto)", re.I)


def lecture_hint(source: Path, p: Dict[str, Any], segs: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Does this look like a lecture / tutorial — a talking person and things shown on a screen — rather than
    footage to watch? From the site's info (category, title, description, tags, chapters) and the speech
    density. {hint, score, reasons} — a suggestion for the user, never a switch."""
    info = read_info_json(source)
    reasons: List[str] = []
    score = 0
    cats = " ".join(str(x) for x in (info.get("categories") or [])).lower()
    if any(c in cats for c in ("education", "howto", "science & technology")):
        score += 2; reasons.append(f"category: {cats.strip()}")
    text = " ".join(str(info.get(k) or "") for k in ("title", "description")) + " " + " ".join(str(x) for x in (info.get("tags") or []))
    m = _LECTURE_WORDS.search(text[:4000])
    if m:
        score += 2; reasons.append(f"title/description: «{m.group(0)}»")
    else:  # a local file: its name is all the title there is
        m = _LECTURE_WORDS.search(source.stem.replace("_", " ").replace("-", " "))
        if m:
            score += 2; reasons.append(f"file name: «{m.group(0)}»")
    ch = info.get("chapters") or []
    if len(ch) >= 3:
        score += 1; reasons.append(f"{len(ch)} chapters")
    dur = float(p.get("duration") or 0)
    if dur >= 8 * 60:
        score += 1; reasons.append(f"{tcode(dur)} long")
    if segs:
        spoken = sum(max(0.0, float(x.get("end", 0)) - float(x.get("start", 0))) for x in segs)
        if dur and spoken / dur >= 0.8:  # someone talking nearly all the time: a lecture even without a site's word
            score += 2; reasons.append(f"speech {spoken / dur:.0%} of the time")
        elif dur and spoken / dur >= 0.6:
            score += 1; reasons.append(f"speech {spoken / dur:.0%} of the time")
    return {"hint": score >= 3, "score": score, "reasons": reasons}


def _lines_around(items: List[Any], t: float, before: float = 4.0, after: float = 4.0) -> List[str]:
    return [x[1] for x in items if x[2]["kind"] != "sound" and x[2]["a"] <= t + after and x[2]["b"] >= t - before]


def frames_at_blocks(m: Dict[str, Any], label: str, frames: List[Dict[str, Any]], rerun: Optional["Rerun"] = None) -> List[Dict[str, Any]]:
    """The frames asked for by time, one image each, the words spoken around each under it."""
    p = m["probe"]
    items = timeline_items(m)
    wh = jpeg_size(frames[0]["file"]) if frames else None
    per = claude_image_tokens(*wh) if wh else 0
    head = (f"[{label}: video \"{Path(m['source']).name}\", {tcode(p['duration'])} — the {len(frames)} frame(s) you asked for, "
            f"one per moment, {AT_WIDTH}px on the long side, each tagged [mm:ss.cc] with the words spoken around it"
            + (f"; ≈ {per * len(frames):,} image tokens by Anthropic's formula for Claude" if per else "") + ".\nRules:\n"
            "1. A frame is the picture at exactly that moment: if it caught a cut, a transition or a blur, ask for the same "
            f"moment a second earlier or later ({rerun.at(['mm:ss']) if rerun else 'with --frames at --times mm:ss'}).\n"
            "2. Say what the frame shows and read what is on it (text, labels, values) — that is what it was asked for; a "
            "frame of the speaker alone shows nothing — ask for another moment.\n"
            + (f"3. {rerun.cannot()}" if rerun else "3. If you cannot run commands, give the user the command.") + "]")
    out: List[Dict[str, Any]] = [{"type": "text", "text": head, "role": "head"}]
    for f in frames:
        tag = f"{tcode(math.floor(f['t']))}.{int(round((f['t'] % 1) * 100)):02d}"
        lines = _lines_around(items, f["t"])
        out.append({"type": "text", "text": f"[{tag}]", "role": "tag"})
        out.append({"type": "image", "path": f["file"], "t": f["t"], "tag": tag, "lines": lines, "entries": []})
        if lines:
            out.append({"type": "text", "text": "\n".join(lines), "role": "lines"})
    return out


def document_offer_rule(rerun: Optional["Rerun"]) -> str:
    """The one question after the answer: a document of this video — PDF, web, both or none. Never «publishing»."""
    how = (
        "call the `video_guide` tool with the structure (title, intro, kind, sections: heading, text, the moments mm:ss whose "
        "frames illustrate it, a caption each) — the host builds the page and returns its address; ask nothing else"
        if rerun and rerun.tool else
        "write the spec as JSON ({title, intro, lang, kind, source: {file, title, url}, sections: [{heading, text, frames, captions}]}) "
        "to a file and run `" + (f"{shell_arg(sys.executable)} {shell_arg(Path(__file__).resolve())}" if rerun else "video2llm")
        + " guide --spec spec.json`, then `… link <the folder it printed>` — "
        "the document goes to the user's own Neocities site (free); the first time it opens a page on this computer where they "
        "sign up and paste their API key (Neocities: Profile → Settings → Manage Site Settings → API Key) — tell them that, "
        "and never ask for a key in the chat")
    return ("When you have answered, decide whether this video or recording is MATERIAL TO KEEP — something taught, shown or explained: a "
            "lecture, a tutorial, a how-to, a walkthrough, a course lesson, a webinar, a talk, a review or a breakdown of a tool "
            "or a technique, an explainer. Only then ask the user in ONE line whether to make the Guide (a step-by-step "
            "instruction, kind guide: numbered steps) or the Notes (a talk, kind notes) of it — name the one that fits — yes or "
            "no; on no, just go on. A film, a clip, a vlog, a sketch, a music video, an ad, a stream of someone playing or "
            "chatting, home footage: do not offer it at all. It is one page of their own with the frames that matter (none for a sound file), at a web "
            "address, with a «Download PDF» button on it: never ask about a PDF separately. Do not build anything before they "
            "answer. After it: " + how + ". Never build the page or the "
            "PDF by hand, and never call this «publishing» — it is the user's link, nothing is announced anywhere.")


def lecture_blocks(m: Dict[str, Any], label: str, rerun: Optional["Rerun"] = None) -> List[Dict[str, Any]]:
    """The lecture lane: no frames — the whole transcript with time spans (and the site's chapters), and the
    rules: read it, decide where the SCREEN matters, ask for single frames at exactly those moments.
    A sound file (no picture) gets the audio lane: the same words, the sounds on request, no frames at all."""
    p = m["probe"]
    if not p_has_video(p):
        return audio_blocks(m, label, rerun)
    dur = p["duration"]
    info = read_info_json(Path(m["source"]))
    chapters = info.get("chapters") or []
    tm = m.get("transcript_meta") or {}
    src = ("the site's own captions" + (" (written by people)" if tm.get("kind") == "manual" else " (automatic — they can mishear)" if tm.get("kind") == "automatic" else "")
           if tm.get("source") == "captions" else "an automatic speech transcript — it can mishear words")
    per = claude_image_tokens(*fit_dims(p["width"], p["height"], AT_WIDTH)) if p.get("width") and p.get("height") else 0
    title = str(info.get("title") or "").strip()
    rules = [
        "This is a lecture, a tutorial or an instruction: most of the picture is a person talking, so NO frames come with "
        "this lane — the words below are the whole of it" + (f" ({src})" if src else "") + ". Read them first.",
        "Then decide where the SCREEN matters: a slide, a diagram, an interface, code, a table, a formula, text on screen, "
        "a demonstration — and also whatever is simply being talked about (a button, a panel, a chart) that is clearer "
        "seen than described. Ask for exactly those moments, one frame each: "
        + (rerun.at(["03:12", "07:40"]) if rerun else "run video2llm with --frames at --times 03:12,07:40")
        + " — any number, 3 or 30, by the sense of what is said and shown ("
        + (f"≈ {per:,} tokens a frame, " if per else "") + f"up to {AT_MAX} a request); frames of the speaker alone are "
        "not needed; no need to ask the user before this — the lecture mode is their choice.",
        "Answer from the words AND the frames: what is said, what is shown, what is written on the screen. When something "
        "is not visible in the frames you got, say so or ask for another moment.",
        document_offer_rule(rerun),
    ]
    if p.get("has_audio"):
        rules.append("Sounds other than speech, or every frame of a short moment, are available as for any video: ask the user "
                     "with the cost first (" + (f"{rerun.frames()} / {rerun.sounds()}" if rerun else "--frames all / --sounds") + ").")
    if rerun:
        rules.append(f"Use only {rerun.these()} for this video; do not open the video file or run anything else on it. {rerun.cannot()}")
    head = (f"[{label}: video \"{Path(m['source']).name}\"" + (f" — «{title}»" if title else "") + f", {tcode(dur)}, {p['width']}×{p['height']}, "
            "LECTURE MODE.\nRules:\n" + "\n".join(f"{k + 1}. {r}" for k, r in enumerate(rules)) + "]")
    out: List[Dict[str, Any]] = [{"type": "text", "text": head, "role": "head"}]
    if chapters:
        out.append({"type": "text", "role": "before", "text": "[chapters, from the site]\n" + "\n".join(
            f"[{tcode(float(c.get('start_time') or 0))}] {str(c.get('title') or '').strip()}" for c in chapters)})
    items = timeline_items(m)
    lines = [x[1] for x in items]
    out.append({"type": "text", "role": "rest", "text": "[the words, with their time spans]\n" + ("\n".join(lines) if lines else "(no words were recognised)")})
    return out


def audio_blocks(m: Dict[str, Any], label: str, rerun: Optional["Rerun"] = None) -> List[Dict[str, Any]]:
    """The audio lane (0.7.1): a sound file — a podcast, a recording of a talk, a voice memo, music. No picture
    exists, so no frames are offered; the words with their time spans, the sounds other than speech on request."""
    p = m["probe"]
    dur = p["duration"]
    info = read_info_json(Path(m["source"]))
    chapters = info.get("chapters") or []
    tm = m.get("transcript_meta") or {}
    src = ("the site's own captions" if tm.get("source") == "captions" else "an automatic speech transcript — it can mishear words")
    title = str(info.get("title") or "").strip()
    rules = [
        "This is a SOUND FILE: there is no picture, so no frames exist and none can be asked for — the words below are "
        f"the whole of it ({src}). Read them first; answer from what is said and when.",
        "Sounds other than speech (music, noises, applause, a door, a car) for a moment: ask the user with the cost first, "
        "a yes or no — then " + (rerun.sounds() if rerun else "run video2llm with --sounds --start mm:ss --end mm:ss")
        + " (free, on this computer; up to a few minutes a request). Do not guess at a sound from the words.",
        document_offer_rule(rerun),
    ]
    if rerun:
        rules.append(f"Use only {rerun.these()} for this file; do not open the file or run anything else on it. {rerun.cannot()}")
    head = (f"[{label}: sound file \"{Path(m['source']).name}\"" + (f" — «{title}»" if title else "") + f", {tcode(dur)}, "
            "AUDIO.\nRules:\n" + "\n".join(f"{k + 1}. {r}" for k, r in enumerate(rules)) + "]")
    out: List[Dict[str, Any]] = [{"type": "text", "text": head, "role": "head"}]
    if chapters:
        out.append({"type": "text", "role": "before", "text": "[chapters, from the site]\n" + "\n".join(
            f"[{tcode(float(c.get('start_time') or 0))}] {str(c.get('title') or '').strip()}" for c in chapters)})
    lines = [x[1] for x in timeline_items(m)]
    out.append({"type": "text", "role": "rest", "text": "[the words, with their time spans]\n" + ("\n".join(lines) if lines else "(no words were recognised)")})
    return out


def link_blocks(m: Dict[str, Any], label: str, rerun: Optional["Rerun"] = None) -> List[Dict[str, Any]]:
    """The words lane (0.7.2): a link the user gave — the site's captions (or a transcript of the sound), no
    frames, the video not even downloaded. The rules: decide from the user's message what they want of it,
    ask one question when the message does not say, and ask for the picture only where it is needed (the
    first such request downloads the video). The same lane for a downloaded file asked with --frames words."""
    p = m["probe"]
    dur = float(p.get("duration") or 0)
    link = m.get("link") or {}
    info = read_info_json(Path(m["source"]))
    chapters = info.get("chapters") or []
    tm = m.get("transcript_meta") or {}
    src = ("the site's own captions" + (" (written by people)" if tm.get("kind") == "manual" else " (automatic — they can mishear)" if tm.get("kind") == "automatic" else "")
           if tm.get("source") == "captions" else "an automatic speech transcript — it can mishear words")
    title = str(link.get("title") or info.get("title") or "").strip()
    site = str(link.get("site") or "").strip()
    url = str(link.get("webpage_url") or link.get("url") or "").strip()
    downloaded = bool(link.get("downloaded")) or Path(m["source"]).is_file()
    hint = (link.get("lecture") or {})
    per = claude_image_tokens(*fit_dims(p["width"], p["height"], AT_WIDTH)) if p.get("width") and p.get("height") else 0
    rules = [
        "This is a LINK the user gave, not a file: only the WORDS came" + (f" ({src})" if src else "") + " — the video "
        + ("is on this computer but no frames were cut" if downloaded else "itself was not downloaded") + " and no frame comes with this lane. Read the words first.",
        "Decide from the user's message what they want of the link — an answer, a summary, a check of what is said, "
        "a step from it, the words at a moment, something to do with what it teaches — and do that. When the message "
        "does not say what to do with it, ask ONE short question and stop: do not summarise it unasked.",
        "The picture, only where the task needs it SEEN (a slide, an interface, code, a demonstration, what is shown at a "
        "moment the words point at): " + (rerun.at(["03:12", "07:40"]) if rerun else "run video2llm with --frames at --times 03:12,07:40")
        + f" gives one frame per moment ({AT_WIDTH}px" + (f", ≈ {per:,} tokens each" if per else "") + f", up to {AT_MAX} a request); "
        + (rerun.next(0.0) if rerun else "run video2llm with --start 0:00") + " gives the 1 fps overview from a moment"
        + (", " + rerun.frames() + " / " + rerun.sounds() + " every frame / the sounds of a moment" if rerun else "")
        + ". " + ("" if downloaded else "The first such request DOWNLOADS the video (up to 1080p — a minute or more for a long one): say so in a "
                  "line and go on. ")
        + "Up to 6 frames with `at` need no yes; the overview, every frame and the sounds — ask yes or no with the cost first.",
        document_offer_rule(rerun),
    ]
    if hint.get("hint"):
        rules.append("The site files it as something taught or explained (" + "; ".join(hint.get("reasons") or []) + ") — a sign for rule 4, not a decision.")
    if rerun:
        rules.append(f"Use only {rerun.these()} for this video; do not open the link in a browser, do not download or run anything else on it. {rerun.cannot()}")
    head = (f"[{label}: link" + (f" — «{title}»" if title else "") + (f" ({site})" if site else "") + (f", {tcode(dur)}" if dur else "")
            + (f", {p['width']}×{p['height']}" if p.get("width") and p.get("height") else "") + (f"\n{url}" if url else "")
            + "\nWORDS ONLY.\nRules:\n" + "\n".join(f"{k + 1}. {r}" for k, r in enumerate(rules)) + "]")
    out: List[Dict[str, Any]] = [{"type": "text", "text": head, "role": "head"}]
    if chapters:
        out.append({"type": "text", "role": "before", "text": "[chapters, from the site]\n" + "\n".join(
            f"[{tcode(float(c.get('start_time') or 0))}] {str(c.get('title') or '').strip()}" for c in chapters)})
    lines = [x[1] for x in timeline_items(m)]
    out.append({"type": "text", "role": "rest", "text": "[the words, with their time spans]\n" + ("\n".join(lines) if lines else "(no words: the site has no captions and no speech was recognised)")})
    return out


# ── lane sheets: an agent opens one image per 3 (overview) or 6 (frame by frame) frames ───────────────
# Same pixels per frame, same tokens within a few per cent, a third to a sixth of the reads. Each frame keeps
# its own size and its tag above it; a sheet stays inside Claude's limits (long side ≤ 1568 px, ≤ 1568 tokens).
LANE_CAP_PX = 24  # the tag strip above each frame
LANE_GAP_PX = 4


def pillow_ok() -> bool:
    import importlib.util
    return importlib.util.find_spec("PIL") is not None


def sheet_fit(tw: int, th: int, cols: int, rows: int) -> float:
    """The scale (≤ 1) of the frames on a cols × rows sheet that keeps the sheet within Claude's limits
    (1568 px on the long side, 1568 tokens); 1 = every frame at its own size. The tag strips are not scaled."""
    s = 1.0
    for _ in range(60):
        w = cols * tw * s + (cols - 1) * LANE_GAP_PX
        h = rows * (LANE_CAP_PX + th * s) + (rows - 1) * LANE_GAP_PX
        if max(w, h) <= 1568 and math.ceil(w / 28) * math.ceil(h / 28) <= 1568:
            return s
        s *= 0.98
    return s


def lane_grid(tw: int, th: int, want: int = 0) -> List[int]:
    """The most frames per image that keep every frame at its own size and the sheet within Claude's limits,
    at no more than ~6 % extra tokens over the frames sent one by one; [1, 1] = no sheets. `want` > 1
    (--sheet-frames): exactly that many frames to a sheet, in the grid that scales them the least — for a chat
    that counts the files of a message."""
    if want > 1:
        best, best_s = [want, 1], 0.0
        for cols in range(1, want + 1):
            if want % cols:
                continue
            rows = want // cols
            s = sheet_fit(tw, th, cols, rows)
            if s > best_s + 1e-9:
                best, best_s = [cols, rows], s
        return best
    single = claude_image_tokens(tw, th)
    best, best_n, best_tok = [1, 1], 1, single
    for cols in range(1, 9):
        for rows in range(1, 9):
            n = cols * rows
            w = cols * tw + (cols - 1) * LANE_GAP_PX
            h = rows * (LANE_CAP_PX + th) + (rows - 1) * LANE_GAP_PX
            if w > 1568 or h > 1568:
                continue
            tok = math.ceil(w / 28) * math.ceil(h / 28)
            if tok > 1568 or tok > n * single * 1.06:
                continue
            if n > best_n or (n == best_n and tok < best_tok):
                best, best_n, best_tok = [cols, rows], n, tok
    return best


def lane_sheet_plan(m: Dict[str, Any], dense: bool, enabled: bool = True, want: int = 0) -> List[int]:
    """The grid lane.md will use for this lane's frames — [1, 1] when sheets are off or Pillow is missing."""
    frames = ((m.get("dense") or {}).get("frames") if dense else m.get("frames")) or []
    if not (enabled and frames and pillow_ok()):
        return [1, 1]
    wh = jpeg_size(frames[0]["file"])
    return lane_grid(*wh, want=want) if wh else [1, 1]


def sheet_frame_tokens(wh: Optional[List[int]], grid: List[int]) -> int:
    """Claude tokens of one frame as it sits on a sheet of this grid (scaled when the grid asks for it)."""
    if not wh:
        return 0
    cols, rows = grid
    s = sheet_fit(wh[0], wh[1], cols, rows)
    w = cols * wh[0] * s + (cols - 1) * LANE_GAP_PX
    h = rows * (LANE_CAP_PX + wh[1] * s) + (rows - 1) * LANE_GAP_PX
    return max(1, round(claude_image_tokens(w, h) / (cols * rows)))


def _render_lane_sheet(cells: List[Dict[str, Any]], cols: int, path: Path, font: Any) -> None:
    from PIL import Image, ImageDraw  # type: ignore
    with Image.open(cells[0]["path"]) as im:
        tw, th = im.size
    used = min(cols, len(cells))
    rows = math.ceil(len(cells) / cols)
    s = sheet_fit(tw, th, cols, rows)  # a sheet with more frames than fit at their own size → the frames are scaled
    if s < 1:
        tw, th = max(1, round(tw * s)), max(1, round(th * s))
    img = Image.new("RGB", (used * tw + (used - 1) * LANE_GAP_PX,
                            rows * (LANE_CAP_PX + th) + (rows - 1) * LANE_GAP_PX), "white")
    dr = ImageDraw.Draw(img)
    for j, c in enumerate(cells):
        x = (j % cols) * (tw + LANE_GAP_PX)
        y = (j // cols) * (LANE_CAP_PX + th + LANE_GAP_PX)
        dr.text((x + 4, y + 2), f"[{c['tag']}]", fill=(30, 30, 30), font=font)
        with Image.open(c["path"]) as fr:
            fr = fr.convert("RGB")
            if fr.size != (tw, th):
                fr = fr.resize((tw, th), Image.LANCZOS)
            img.paste(fr, (x, y + LANE_CAP_PX))
    img.save(path, "JPEG", quality=85, optimize=True)


def agent_sheets(blocks: List[Dict[str, Any]], grid: List[int]) -> List[Dict[str, Any]]:
    """Frame blocks → sheet blocks: consecutive frames, cols × rows to an image, written to a "sheets" folder
    inside the frames' own folder (sheet_<first>_<last>.jpg), each followed by the words and sounds of its frames —
    every line starting with the tag of the frame it begins at."""
    cols, rows = grid
    per = cols * rows
    if per <= 1:
        return blocks
    from PIL import ImageFont  # type: ignore
    fonts = pdf_fonts(None)
    try:
        font = ImageFont.truetype(fonts["bold"] or fonts["regular"], 18) if (fonts["bold"] or fonts["regular"]) else ImageFont.load_default(18)
    except Exception:
        try:
            font = ImageFont.load_default(18)
        except TypeError:  # Pillow < 10.1
            font = ImageFont.load_default()
    out: List[Dict[str, Any]] = []
    run: List[Dict[str, Any]] = []

    def flush() -> None:
        for i in range(0, len(run), per):
            chunk = run[i:i + per]
            folder = Path(chunk[0]["path"]).parent / "sheets"
            folder.mkdir(exist_ok=True)
            path = folder / f"sheet_{Path(chunk[0]['path']).stem}_{Path(chunk[-1]['path']).stem}.jpg"
            try:
                fresh = path.stat().st_mtime >= max(Path(c["path"]).stat().st_mtime for c in chunk)
            except OSError:
                fresh = False
            if not fresh:
                _render_lane_sheet(chunk, cols, path, font)
            span = f"[{chunk[0]['tag']}]" + (f"–[{chunk[-1]['tag']}]" if len(chunk) > 1 else "")
            out.append({"type": "image", "path": str(path), "t": chunk[0]["t"], "tag": f"frames {span}", "sheet": True,
                        "frames": [c["tag"] for c in chunk]})
            lines = [entry_line(c["tag"], e) for c in chunk for e in c.get("entries") or []]
            if lines:
                out.append({"type": "text", "text": "\n".join(lines), "role": "lines"})
        run.clear()

    for b in blocks:
        if b["type"] == "image":
            run.append(b)
        elif b.get("role") in ("tag", "lines"):
            continue
        else:
            flush()
            out.append(b)
    flush()
    return out


def write_markdown(blocks: List[Dict[str, Any]], prompt: str, path: Path) -> None:
    """lane.md for an agent, and the same lane as lane.json for a program (Deep Artisan) that puts the images
    into the model's message itself: text blocks with their role, image blocks with the file, the tags of the
    frames on it and the lines that belong under it."""
    lines = []
    for b in blocks:
        if b["type"] == "text":
            lines.append(b["text"])
        elif b["type"] == "image":
            # <…> form: the folder names carry spaces and parentheses
            rel = os.path.relpath(b["path"], path.parent).replace(os.sep, "/")
            lines.append(f"![{b.get('tag') or tcode(b['t'])}](<{rel}>)")
        else:
            rel = os.path.relpath(b["path"], path.parent).replace(os.sep, "/")
            lines.append(f"[video part](<{rel}>)")
    if prompt:
        lines.append(prompt)
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    plain = []
    for b in blocks:
        if b["type"] == "text":
            plain.append({"type": "text", "role": b.get("role") or "text", "text": b["text"]})
        elif b["type"] == "image":
            plain.append({"type": "image", "path": b["path"], "t": b.get("t"), "tag": b.get("tag") or tcode(b["t"]),
                          "frames": b.get("frames") or [b.get("tag") or tcode(b["t"])], "sheet": bool(b.get("sheet"))})
        else:
            plain.append({"type": "video", "path": b["path"]})
    path.with_suffix(".json").write_text(json.dumps({"tool": "video2llm", "version": VERSION, "lane": path.name,
                                                     "blocks": plain}, ensure_ascii=False, indent=1), encoding="utf-8")


def to_openai(blocks: List[Dict[str, Any]], prompt: str, model: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for b in blocks:
        if b["type"] == "text":
            content.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(b['path'])}", "detail": "low"}})
        else:
            content.append({"type": "text", "text": "[video file omitted — this API takes frames, not video; use the frame lane]"})
    if prompt:
        content.append({"type": "text", "text": prompt})
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def to_anthropic(blocks: List[Dict[str, Any]], prompt: str, model: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for b in blocks:
        if b["type"] == "text":
            content.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(b["path"])}})
        else:
            content.append({"type": "text", "text": "[video file omitted — this API takes frames, not video; use the frame lane]"})
    if prompt:
        content.append({"type": "text", "text": prompt})
    return {"model": model, "max_tokens": 4096, "messages": [{"role": "user", "content": content}]}


def to_gemini(blocks: List[Dict[str, Any]], prompt: str) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []
    for b in blocks:
        if b["type"] == "text":
            parts.append({"text": b["text"]})
        elif b["type"] == "image":
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64(b["path"])}})
        else:
            parts.append({"inline_data": {"mime_type": b["mime"], "data": b64(b["path"])}})
    if prompt:
        parts.append({"text": prompt})
    return {"contents": [{"role": "user", "parts": parts}]}


# ── PDF: the same lane as one file to drop into a chat ───────────────────────────────────────────
# Claude reads the images of a PDF up to 100 pages (text only above that) and takes 32 MB per API
# request; Gemini takes 1000 pages / 50 MB. ChatGPT reads images inside PDFs only on Enterprise —
# elsewhere it gets the text (header + transcript), and the header tells the model to say so.
PDF_MAX_BYTES = 28 * 1024 * 1024
PDF_MAX_PAGES = 90
PDF_MARGIN, PDF_GUTTER, PDF_ROW_GAP = 10.0, 6.0, 4.0  # mm
PDF_TEXT_PT, PDF_TAG_PT = 9, 8
PDF_LINE_MM, PDF_TAG_MM = 4.2, 4.6


def _pdf_font_candidates() -> List[List[Optional[str]]]:
    """[regular, bold] pairs of system TrueType fonts that carry Latin, Cyrillic and Greek."""
    if sys.platform == "darwin":
        sup = "/System/Library/Fonts/Supplemental"
        return [[f"{sup}/Arial.ttf", f"{sup}/Arial Bold.ttf"], [f"{sup}/Arial Unicode.ttf", None],
                ["/Library/Fonts/Arial Unicode.ttf", None]]
    if os.name == "nt":
        d = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        return [[str(d / "arial.ttf"), str(d / "arialbd.ttf")], [str(d / "segoeui.ttf"), str(d / "segoeuib.ttf")],
                [str(d / "tahoma.ttf"), str(d / "tahomabd.ttf")]]
    out: List[List[Optional[str]]] = []
    for r in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF", "/usr/local/share/fonts"):
        out.append([f"{r}/DejaVuSans.ttf", f"{r}/DejaVuSans-Bold.ttf"])
    out.append(["/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"])
    out.append(["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"])
    return out


def pdf_fonts(override: Optional[str]) -> Dict[str, Optional[str]]:
    """Regular + bold + a fallback for glyphs the regular one lacks (e.g. Arial Unicode on a Mac)."""
    found = [[r, b if b and Path(b).is_file() else None] for r, b in _pdf_font_candidates() if r and Path(r).is_file()]
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            sys.exit(f"--pdf-font: not a file: {p}")
        return {"regular": str(p), "bold": None, "fallback": found[0][0] if found else None}
    if not found:
        return {"regular": None, "bold": None, "fallback": None}
    reg = found[0][0]
    fb = next((r for r, _ in found[1:] if Path(r).resolve() != Path(reg).resolve()), None)
    return {"regular": reg, "bold": found[0][1], "fallback": fb}


def _missing_glyphs(text: str, font_path: str) -> bool:
    try:
        from fontTools.ttLib import TTFont  # comes with fpdf2
        cmap = TTFont(font_path, lazy=True).getBestCmap() or {}
    except Exception:
        return False
    return any(ord(ch) > 32 and ord(ch) not in cmap for ch in set(text) if ch not in "\n\r\t")


class _PdfLane:
    """Lays the lane out: caption [mm:ss] above each frame, the speech and sound lines of the row's
    seconds in a full-width band under each row (so text extraction keeps them in order). Pages are as tall as their content (no blank space for the renderer to shrink): landscape
    and square frames 2×2 per page, very wide ones 1×3, portrait ones 3 across. A short header and the
    transcript before the frames sit on top of the first page; long text gets its own A4 pages."""

    PRE_MAX = 60.0  # mm of header/transcript text that may share a page with frames

    def __init__(self, FPDF: Any, fonts: Dict[str, Optional[str]], aspect: float, use_fallback: bool):
        self.FPDF, self.fonts, self.use_fallback = FPDF, fonts, use_fallback
        if aspect < 0.8:
            self.pw, self.cols, self.rows_pp = 297.0, 3, 1
        elif aspect >= 2.0:
            self.pw, self.cols, self.rows_pp = 210.0, 1, 3
        else:
            self.pw, self.cols, self.rows_pp = 210.0, 2, 2
        self.std_h = 297.0 if self.pw < 250 else 210.0  # A4 for text-only pages
        self.body_w = self.pw - 2 * PDF_MARGIN
        self.col_w = (self.body_w - (self.cols - 1) * PDF_GUTTER) / self.cols
        self.img_w, self.img_h = self.col_w, self.col_w / aspect

    @property
    def per_page(self) -> int:
        return self.cols * self.rows_pp

    def new_doc(self) -> Any:
        pdf = self.FPDF(orientation="P", unit="mm", format=(self.pw, self.std_h))
        pdf.set_auto_page_break(False)
        pdf.set_margins(PDF_MARGIN, PDF_MARGIN, PDF_MARGIN)
        pdf.add_font("lane", "", self.fonts["regular"])
        pdf.add_font("lane", "B", self.fonts["bold"] or self.fonts["regular"])
        if self.use_fallback and self.fonts["fallback"]:
            pdf.add_font("lanefb", "", self.fonts["fallback"])
            pdf.set_fallback_fonts(["lanefb"])
        return pdf

    def text_h(self, pdf: Any, text: str, w: float) -> float:
        if not text:
            return 0.0
        if pdf.page == 0:
            pdf.add_page()
        pdf.set_font("lane", "", PDF_TEXT_PT)
        return len(pdf.multi_cell(w, PDF_LINE_MM, text, dry_run=True, output="LINES", align="L")) * PDF_LINE_MM

    def blocks_h(self, pdf: Any, texts: List[str]) -> float:
        return sum(self.text_h(pdf, t, self.body_w) + 2.5 for t in texts)

    def row_h(self, pdf: Any, row: List[Dict[str, Any]]) -> float:
        th = self.text_h(pdf, "\n".join(ln for c in row for ln in c["lines"]), self.body_w)
        return PDF_TAG_MM + self.img_h + (1.5 + th if th else 0.0) + PDF_ROW_GAP

    def lines(self, pdf: Any, lines: List[str], x: float, w: float, flow: bool) -> None:
        """Speech in near-black, sound tags in grey; `flow` lets long text run onto the next page."""
        pdf.set_auto_page_break(flow, PDF_MARGIN)
        pdf.set_font("lane", "", PDF_TEXT_PT)
        for ln in lines:
            pdf.set_text_color(95 if "] (sound: " in ln else (70 if ln.startswith("[transcript") else 15))
            pdf.set_x(x)
            pdf.multi_cell(w, PDF_LINE_MM, ln, new_x="LEFT", new_y="NEXT", align="L")
        pdf.set_auto_page_break(False)

    def text_pages(self, pdf: Any, texts: List[str]) -> None:
        pdf.add_page(format=(self.pw, self.std_h))
        pdf.set_xy(PDF_MARGIN, PDF_MARGIN)
        for t in texts:
            self.lines(pdf, t.split("\n"), PDF_MARGIN, self.body_w, flow=True)
            pdf.ln(2.5)

    def render(self, blocks: List[Dict[str, Any]], prompt: str, meta: Dict[str, str], path: Path) -> int:
        pdf = self.new_doc()
        pdf.set_title(meta["title"])
        pdf.set_creator(f"video2llm {VERSION} — Immersive Media Technologies")
        if meta.get("lang"):
            pdf.set_lang(meta["lang"])
        measure = self.new_doc()
        pre = [b["text"] for b in blocks if b["type"] == "text" and b.get("role") in ("head", "before")]
        tail = [b["text"] for b in blocks if b["type"] == "text" and b.get("role") == "rest"] + ([prompt] if prompt else [])
        pre_h, tail_h = self.blocks_h(measure, pre), self.blocks_h(measure, tail)
        pre_inline, tail_inline = pre_h <= self.PRE_MAX, tail_h <= self.PRE_MAX
        if not pre_inline:
            self.text_pages(pdf, pre)
        cells = [b for b in blocks if b["type"] == "image"]
        rows = [cells[i:i + self.cols] for i in range(0, len(cells), self.cols)]
        pages = [rows[i:i + self.rows_pp] for i in range(0, len(rows), self.rows_pp)]
        for pi, page_rows in enumerate(pages):
            heights = [self.row_h(measure, r) for r in page_rows]
            top = pre_h if pi == 0 and pre_inline else 0.0
            bottom = tail_h if pi == len(pages) - 1 and tail_inline else 0.0
            pdf.add_page(format=(self.pw, 2 * PDF_MARGIN + top + sum(heights) - PDF_ROW_GAP + bottom))
            pdf.set_xy(PDF_MARGIN, PDF_MARGIN)
            if top:
                for t in pre:
                    self.lines(pdf, t.split("\n"), PDF_MARGIN, self.body_w, flow=False)
                    pdf.ln(2.5)
            y = PDF_MARGIN + top
            for row, rh in zip(page_rows, heights):
                for k, c in enumerate(row):
                    x = PDF_MARGIN + k * (self.col_w + PDF_GUTTER)
                    pdf.set_xy(x, y)
                    pdf.set_font("lane", "B", PDF_TAG_PT)
                    pdf.set_text_color(70)
                    pdf.cell(self.img_w, PDF_TAG_MM, f"[{c['tag']}]")
                    pdf.image(c["path"], x, y + PDF_TAG_MM, self.img_w, self.img_h)
                    pdf.set_draw_color(205)
                    pdf.set_line_width(0.2)
                    pdf.rect(x, y + PDF_TAG_MM, self.img_w, self.img_h)
                # the row's speech and sound lines as one full-width band: text extractors keep it in order
                band = [ln for c in row for ln in c["lines"]]
                if band:
                    pdf.set_xy(PDF_MARGIN, y + PDF_TAG_MM + self.img_h + 1.5)
                    self.lines(pdf, band, PDF_MARGIN, self.body_w, flow=False)
                y += rh
            if bottom:
                pdf.set_xy(PDF_MARGIN, y)
                for t in tail:
                    self.lines(pdf, t.split("\n"), PDF_MARGIN, self.body_w, flow=False)
                    pdf.ln(2.5)
        if tail and not tail_inline:
            self.text_pages(pdf, tail)
        pdf.output(str(path))
        return pdf.pages_count


def write_pdf(m: Dict[str, Any], label: str, start: float, end: Optional[float], max_frames: Optional[int], dense: bool,
              prompt: str, out_dir: Path, font: Optional[str], max_pages: int = PDF_MAX_PAGES,
              base: str = "lane", rerun: Optional["Rerun"] = None, fps_overview: int = FRAME_FPS) -> List[Dict[str, Any]]:
    """The lane from `start` to the end (or `max_frames`) as lane.pdf, or lane-01.pdf … when it does
    not fit one file. Needs fpdf2 (pip install fpdf2); without it the PDF is skipped."""
    try:
        from fpdf import FPDF  # type: ignore
        from PIL import Image  # type: ignore  # comes with fpdf2
        import logging
        logging.getLogger("fpdf").setLevel(logging.ERROR)
        logging.getLogger("fontTools").setLevel(logging.ERROR)
    except ImportError:
        log("  pdf: skipped — needs fpdf2 (pip install fpdf2)")
        return []
    fonts = pdf_fonts(font)
    if not fonts["regular"]:
        log("  pdf: skipped — no TrueType font with Cyrillic found; pass --pdf-font /path/to/font.ttf")
        return []
    full = lane_blocks(m, label, start, max_frames or 10 ** 9, dense, end=end, kind="pdf", fps_overview=fps_overview,
                       part={"index": 0, "parts": 2, "files": ["lane-01.pdf", "lane-02.pdf"]}, rerun=rerun)
    cells = [b for b in full if b["type"] == "image"]
    if not cells:
        log("  pdf: skipped — no frames in the window")
        return []
    with Image.open(cells[0]["path"]) as im:
        aspect = im.size[0] / max(1, im.size[1])
    all_text = "\n".join(b["text"] for b in full if b["type"] == "text") + ("\n" + prompt if prompt else "")
    lay = _PdfLane(FPDF, fonts, aspect, use_fallback=_missing_glyphs(all_text, fonts["regular"]))
    # pages per part: the frame pages plus, when the transcript is long, the A4 pages its text takes
    probe = lay.new_doc()
    items = [ln for b in full if b["type"] == "text" and b.get("role") in ("before", "lines", "rest") for ln in b["text"].split("\n")]
    text_h = lay.text_h(probe, "\n".join(items), lay.body_w) + lay.text_h(probe, full[0]["text"], lay.body_w)
    text_pages = math.ceil(text_h / (lay.std_h - 2 * PDF_MARGIN)) if text_h > lay.PRE_MAX else 0
    per_part = lay.per_page * max(1, max_pages - 2 * text_pages)
    parts: List[List[Dict[str, Any]]] = [[]]
    size = 0
    for c in cells:
        b = Path(c["path"]).stat().st_size
        if parts[-1] and (len(parts[-1]) >= per_part or size + b > PDF_MAX_BYTES):
            parts.append([])
            size = 0
        parts[-1].append(c)
        size += b
    n = len(parts)
    files = [f"{base}.pdf"] if n == 1 else [f"{base}-{i + 1:02d}.pdf" for i in range(n)]
    for old in list(out_dir.glob(f"{base}-[0-9][0-9].pdf")) + [out_dir / f"{base}.pdf"]:
        if old.exists() and old.name not in files:
            old.unlink()
    lang = (m.get("transcript_meta") or {}).get("language")
    out = []
    for i, part in enumerate(parts):
        blocks = lane_blocks(m, label, part[0]["t"], len(part), dense, end=end, kind="pdf", fps_overview=fps_overview,
                             part={"index": i, "parts": n, "files": files}, rerun=rerun)
        path = out_dir / files[i]
        title = f"video2llm — {Path(m['source']).name}" + (f" (part {i + 1} of {n})" if n > 1 else "")
        pages = lay.render(blocks, prompt, {"title": title, "lang": lang or ""}, path)
        step = 1 / m["dense"]["fps"] if dense and m.get("dense") else 1 / FRAME_FPS
        span = f"{tcode(part[0]['t'])}–{tcode(min(part[-1]['t'] + step, m['probe']['duration']))}"
        out.append({"path": path, "pages": pages, "bytes": path.stat().st_size, "span": span, "frames": len(part)})
        if pages > 100 or path.stat().st_size > 30 * 1024 * 1024:
            log(f"  pdf: {path.name} has {pages} pages / {path.stat().st_size / 1e6:.1f} MB — above what Claude reads with images")
    return out


# ── contact sheets: the lane as a few JPGs for chats that take images but not PDFs (ChatGPT) ──────
SHEETS_PER_MESSAGE = 9  # + the text file = 10 attachments, the smallest chat limit (Gemini; Claude takes 20)
SHEET_NOTE = "about this video.txt"


def _sheet_grid(aspect: float, override: Optional[str]) -> List[int]:
    if override:
        mm = re.fullmatch(r"\s*(\d+)\s*[x×*]\s*(\d+)\s*", override)
        if not mm or not (1 <= int(mm.group(1)) <= 8 and 1 <= int(mm.group(2)) <= 8):
            sys.exit(f"--sheet-grid: expected COLSxROWS like 3x2, got {override!r}")
        return [int(mm.group(1)), int(mm.group(2))]
    if aspect < 0.8:
        return [3, 1]
    if aspect >= 2.0:
        return [1, 3]
    return [2, 2]


def _clear_sheets(folder: Path) -> None:
    """Drop the sheets and notes video2llm wrote there before (own files only)."""
    if not folder.is_dir():
        return
    for f in list(folder.iterdir()):
        try:
            if f.is_file() and (re.fullmatch(r"sheet-\d\d+\.jpg", f.name) or f.name == SHEET_NOTE):
                f.unlink()
            elif f.is_dir() and re.fullmatch(r"message-\d\d", f.name):
                _clear_sheets(f)
                if not any(f.iterdir()):
                    f.rmdir()
        except OSError:
            pass


def write_sheets(m: Dict[str, Any], label: str, start: float, end: Optional[float], max_frames: Optional[int], dense: bool,
                 prompt: str, out_dir: Path, font: Optional[str], grid: Optional[str], per_message: int,
                 folder_name: str = "sheets", rerun: Optional["Rerun"] = None, fps_overview: int = FRAME_FPS) -> Optional[Dict[str, Any]]:
    """sheets/sheet-01.jpg … — frames on a grid, each with its [mm:ss] above it and the sheet's number and
    time span along the top — plus "about this video.txt" (header + the whole transcript and sounds).
    More sheets than one message takes → sheets/message-01/, message-02/ …, each with its own text file."""
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore  # pip install pillow (fpdf2 brings it)
    except ImportError:
        log("  sheets: skipped — needs Pillow (pip install pillow; fpdf2 brings it)")
        return None
    full = lane_blocks(m, label, start, max_frames or 10 ** 9, dense, end=end, fps_overview=fps_overview)
    cells = [b for b in full if b["type"] == "image"]
    if not cells:
        log("  sheets: skipped — no frames in the window")
        return None
    with Image.open(cells[0]["path"]) as im:
        tw, th = im.size
    cols, rows = _sheet_grid(tw / max(1, th), grid)
    per = cols * rows
    sheets = [cells[i:i + per] for i in range(0, len(cells), per)]
    per_message = max(1, per_message)
    batches = [list(range(i, min(i + per_message, len(sheets)))) for i in range(0, len(sheets), per_message)]

    fonts = pdf_fonts(font)
    scale = max(0.6, tw / 768)

    def ttf(path: Optional[str], px: int) -> Any:
        try:
            return ImageFont.truetype(path, px) if path else ImageFont.load_default(px)
        except Exception:
            return ImageFont.load_default()

    f_cap = ttf(fonts["bold"] or fonts["regular"], round(24 * scale))
    f_head = ttf(fonts["regular"], round(26 * scale))
    mg, gap, cap, headh = round(12 * scale), round(10 * scale), round(34 * scale), round(46 * scale)
    W = 2 * mg + cols * tw + (cols - 1) * gap
    root = out_dir / folder_name
    root.mkdir(exist_ok=True)
    _clear_sheets(root)
    name = Path(m["source"]).name
    step = 1 / m["dense"]["fps"] if dense and m.get("dense") else 1 / FRAME_FPS
    total = len(sheets)
    for bi, batch in enumerate(batches):
        folder = root if len(batches) == 1 else root / f"message-{bi + 1:02d}"
        folder.mkdir(exist_ok=True)
        for k in batch:
            sh = sheets[k]
            r_used = math.ceil(len(sh) / cols)
            H = mg + headh + r_used * (cap + th) + (r_used - 1) * gap + mg
            img = Image.new("RGB", (W, H), "white")
            dr = ImageDraw.Draw(img)
            t0, t1 = sh[0]["t"], min(sh[-1]["t"] + step, m["probe"]["duration"])
            dr.text((mg, mg + round(6 * scale)), f"{label} · {name} · sheet {k + 1:02d} of {total:02d} · "
                    f"{tcode(t0)}–{tcode(t1)}", fill=(30, 30, 30), font=f_head)
            for j, c in enumerate(sh):
                x = mg + (j % cols) * (tw + gap)
                y = mg + headh + (j // cols) * (cap + th + gap)
                dr.text((x, y + round(4 * scale)), f"[{c['tag']}]", fill=(60, 60, 60), font=f_cap)
                with Image.open(c["path"]) as fr:
                    fr = fr.convert("RGB")
                    if fr.size != (tw, th):
                        fr = fr.resize((tw, th))
                    img.paste(fr, (x, y + cap))
            img.save(folder / f"sheet-{k + 1:02d}.jpg", "JPEG", quality=85, optimize=True)
        first, last = batch[0], batch[-1]
        b_start = sheets[first][0]["t"]
        b_count = sum(len(sheets[k]) for k in batch)
        head = lane_blocks(m, label, b_start, b_count, dense, end=end, kind="sheets", fps_overview=fps_overview,
                           part={"first": first + 1, "last": last + 1, "total": total, "batch": bi + 1, "batches": len(batches)},
                           rerun=rerun)[0]["text"]
        lines = [x[1] for x in timeline_items(m)]
        text = head + "\n\n" + ("\n".join(lines) if lines else "(no speech or sounds)") + ("\n\n" + prompt if prompt else "") + "\n"
        (folder / SHEET_NOTE).write_text(text, encoding="utf-8")
    return {"dir": root, "sheets": total, "batches": len(batches), "grid": f"{cols}x{rows}", "frames": len(cells)}


DEFAULT_MODELS = {"openai": "gpt-5.5", "anthropic": "claude-sonnet-5-5", "gemini": "gemini-3.1-flash"}


def ask(provider: str, body: Dict[str, Any], model: str) -> str:
    """Send the payload with the key from the environment; print the model's text."""
    env = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    key = os.environ.get(env)
    if not key:
        sys.exit(f"{env} is not set")
    if provider == "openai":
        url, headers = "https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {key}"}
    elif provider == "anthropic":
        url, headers = "https://api.anthropic.com/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url, headers = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", {"x-goog-api-key": key}
    headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.exit(f"{provider} HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:800]}")
    if provider == "openai":
        return j["choices"][0]["message"]["content"]
    if provider == "anthropic":
        return "".join(c.get("text", "") for c in j.get("content", []))
    return "".join(pt.get("text", "") for pt in j["candidates"][0]["content"]["parts"])


# ── CLI ──────────────────────────────────────────────────────────────────────────────────────────
# ───────────────────────────── guide: an illustrated page and a PDF from a spec ─────────────────────────────
# `video2llm guide --spec guide.json [--out DIR] [--pdf] [--json]`
# The model writes the spec (title, intro, sections with a heading, the text and the moments whose frames
# illustrate them); this builds <out>/index.html + img/ — one page, the frames open full screen on a click — and a PDF
# when asked, printed by a Chrome / Chromium / Edge / Brave already on the computer. Nothing is uploaded here;
# `video2llm link` does that, only when the user asked for a web address.
GUIDE_HOME = "guides"                # ~/.cache/video2llm/guides/<title>/ — where a document is built by default (--out to choose)
GUIDE_MAX_FRAMES = 60
REPO_URL = "https://github.com/Immersive-Media-Technologies/video-to-llm-imt"
TERMS_URL = REPO_URL + "/blob/main/TERMS.md"

_TIME_RE = re.compile(r"^\d{1,2}(?::\d\d){1,2}(?:\.\d{1,2})?$")
_INLINE_B = re.compile(r"\*\*([^*]+)\*\*")
_INLINE_C = re.compile(r"`([^`]+)`")
_LIST_RE = re.compile(r"^\s*(?:[-*•]\s+|(\d+)[.)]\s+)(.*)$")


def esc_html(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _inline(t: str) -> str:
    return _INLINE_C.sub(r"<code>\1</code>", _INLINE_B.sub(r"<b>\1</b>", esc_html(t)))


def guide_text_html(text: str) -> str:
    """The smallest markdown: paragraphs, `- ` and `1. ` lists, **bold**, `code`."""
    out: List[str] = []
    lst: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal lst
        if lst:
            out.append(f"<{lst['kind']}>" + "".join(f"<li>{i}</li>" for i in lst["items"]) + f"</{lst['kind']}>")
            lst = None

    for para in re.split(r"\n{2,}", text.replace("\r\n", "\n")):
        for line in para.split("\n"):
            m = _LIST_RE.match(line)
            if m:
                kind = "ol" if m.group(1) else "ul"
                if not lst or lst["kind"] != kind:
                    flush()
                    lst = {"kind": kind, "items": []}
                lst["items"].append(_inline(m.group(2)))
            elif line.strip():
                flush()
                out.append(f"<p>{_inline(line.strip())}</p>")
        flush()
    return "\n".join(out)


def guide_slug(title: str) -> str:
    t = re.sub(r"\s+", " ", re.sub(r'[\\/:*?"<>|]+', " ", title.strip())).strip()
    return (t or "guide")[:80]


def web_slug(title: str) -> str:
    """ASCII folder name for a web address: letters, digits, dashes; Cyrillic transliterated the simple way."""
    tr = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                  ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u",
                   "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))
    s = "".join(tr.get(c, c) for c in title.lower())
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] or "guide"


IMT_LOGO_PNG = "iVBORw0KGgoAAAANSUhEUgAAAE8AAAAoCAQAAAAVzG/vAAAICElEQVR42sVYW2xdRxVde+Zc2w12ajdJE6IArWMVQigtpaUFuXlHQpF4SKD0BwkFPkBUBX7oBwi5QeonEpRCZSFIPyBqHZCgagmNSOyQKJQSGmKZPJ3UjovzcPy27+OcmVl8nPuY83D4gJK58sc93nf2nrXXXnvPAVKLATDz44iVKGTtUzEh548CVPg/ryDzxEJaNgWAHwqB6CgABYfbuaiA0U4bko7esuT4ZoD6dqOn4No+pRQqoj3sVHTj8luAjx0V3r1UU2z+cw0UX2Fmlf94O7DLoCeWYi7iZTgPGwsdvuRb9ai9bvgjrY9ZQv73IWkUZzsP/DfYB8Difr5ra/FQLVdBTnHoDCYOThq8EzH9QfAIDBwEFoIAtcRHsEizkrAQFOpPI9gcH7VlEJiXgPj/mfDE/SfxoAjXfyjoBKEQoAAY6yb0HAO0B3eiENOhfjBWbSJOqAXXJO1BW8omcRIJrJ06USvDTHilrpZCOmKMyTxF2Kju5d2K0EB01bxWOlQ6/c61g8VHddfyuzqbdrTsKayHhQbhoIHoinm1fGh+aOzGeHFVYVX7qq7WTxf2FNbmBSgOyp1/+iJFXI7mjX/AlNJMMHZ0o98xqIHSy2R4YerJwY4sAv2tc8+TDEmyPDi953hb1ubcysX9JE2GdhFZfCFmdw7hp76U+pElK2d7lHipBSbaStcXfhi7paamolQ/ioEAmH+BDKdmv93XFO/s2/TpkwUAWPhdToCGnP4i0J8fXnEfnY286CK6xV7/NBTg0n1ju6puJa/zUL2xfGbfla5q+Hk2mjK0xszQJdqTI6PShXVU/QEVld/jRdBbCIdjxPzTzDyRB3a+24zIyy3F6cU4nb638tFcWaYId21QcT02Cknb8sSJZDuLEVqi7dQPS7Vka6p7nPsTvpzULwV7fuKDzbpiq0Xpn2b+qexpKm/GCc3bvk9T33rIonAJG2rgencqVyStMc4444w1rjTiIwW9A3CSHKTMUSArABRq4W4rVtxSSe5R1EKJbVSOxKFpMYFQrFpaixYtGlrc4aCeWjvUGjyG5JyngMX+eCt/9WmxsK8s++TGplWzV2TI00Q//W4vBjtWbwyWT56XSxkbAcL3xEqcDZwiAEoHPahvbEpB7cho9uTKdHKpgDMrFp+N3iZJ42Z/kJ2jKcDb9xR/Fl4nyag0/dX0xMMAmNmdq30139NDa3zmPZNTR0fSrqmAazvCkeomERmFw3cnj0BNmdgeTlWVMyQrIz0BJWETAPPPpjy6+ieiKx9pVK4DCttSTCBg+pMjPJVyI/euOBgEiFygBBqEcU0AxKeAsPiNQgfKaIYCQHCtCBNZcEBTd8qj+IoSDlTDo4gbvlt/LMu84kCKeYruzm1BgAgFVZ2kOfzrcb9HUsT2twQPg2iCAI5K2cGvRdQNoaGIG1ytPp7yWMFiLXinwkMe86Y+l+SBdWR0rb81nTZgcX+c1FqPXHg+Kdt9Grj6Cde4rkTk/FNJG2pg6jMJj4acf5IdXMkOtrOD7cl29lySB9aQpd8nCU0B/tAcjnolZMjpzyd7JANg4WlvN2fc2EeTHGYALPzItyFNdO7eJcQTEp5O1W1Ezn4ri8v4wx4ujowWLq1OIayA8ut1ZCxZGe4tSNbjKdLaxugRnu5R3nAh3nYXu2yYbs/WvfNg9syz3/HObMjysZSNAIMd0RRrh4jI4r5UahUw2mlCJgiw8Fy2t6v4b+XjqgDrVY6D2JETZ1KF4YBmv74J2CPZglr7SNABV7USoHw4c11Fe7cuwNR3EiA8DAxkDUGgZXvmdoHoxO6QuqH1FHGn2tWjXjgKWDySOoIAzdvrQwShbTh7PDVUEAh2JL5rszD1F2CLzfIOQ03hSHaQmt6TrbabOz07R4Y3jrelmCdA6c0a86wly39HQpAhQF9TeDlZYOU/573DUQcUcNf9hfdnBikzfyx1ZgHu2NZ45hxg3+ieTyCshGPrCvfXEFYOsAMgUvW/+cP6Hs8jgShNkji8TgW0bYHApvrFuV9cTl1ILKC3NpinGPMlofsKWNatW+o8lpz0K6Blq/I9CmCP9wf/VFTUDDwUqYHiwVRrjsjiT5Op7VHAhXVRyas2Z931B7K1vfjzem07Mpo9syIrPMVXY2VtJHfqs0uMjBeWR5NMvpIy5M0vZMU2MWFYMrzUW0iiJ+gJKufrrDJk+XBWeE7cEf4r7kv1vVzl0swTIw+NPDSxbfabN79LVT/QxLacQap4dm2MWKKz9HqaF5GlF1NdRQGjG41NqNn3cjRvvYlSgCRW8Se13wRAy9bqmN8QFW1PbxhPM69P680efQUoZZnnOjZpBVOdhHT+OGtbdQDGI6fnNS4VC5n9be03AVDYkh2kogEAGsaffcfvC7rq1Rbr2bFb6pmDiq6O/SN7kSpM2opuFqaLCoBDYCbnG+8SL6yL5lJAW/LmzlTaAmDu643UWkuWTiX1jAKcXFa5WqdKRJZ+k30vSOlR5be8qSd5DXel1xpsVSseD9rqDag6wZmZ8ZNZXApeZ8nTMyhg3QNNa3wFrRzJXnag9rr5ZyAIYOHqiSccLAwk7PcoVPpVSlQMWX49W20nl0XjHsqGnNyVRXjh+/6QZM34hrxeQAUsfCWczCuLqOKLVWC3RKD/DoUK5YHkCH9Awa58kO8Nq2wmlTZzo39LIewAtyWq3rLolKpc6r0oyLxrgjgq+eX5gbV7CjtVF9tFS5lzbtyeNX+dG3jfYKMo/w226eGbb9mh2wAAAABJRU5ErkJggg=="  # the white «imt» mark, 79×40, for the footer (inverted on a light page)


def guide_html(spec: Dict[str, Any], sections: List[Dict[str, Any]], *, pdf_name: Optional[str], duration: Optional[float] = None) -> str:
    """The page: one column, the frames as figures with a timecode chip, a lightbox on click, a tight print layout.
    kind «guide» (a sequence of steps) numbers the sections on the left margin; «notes» (a talk) does not."""
    lang = spec.get("lang") or "en"
    ru = lang.lower().startswith("ru")
    title, intro, src = spec["title"], spec.get("intro") or "", spec.get("source") or {}
    brand = spec.get("brand") or "Immersive Media Technologies"
    kind = spec.get("kind") or "guide"
    made = ("Сделано с V2L-IMT" if ru else "Made with V2L-IMT")
    t_src, t_terms, t_contents, t_pdf, t_close = (("Источник", "условия", "Содержание", "Скачать PDF", "Закрыть") if ru
                                                   else ("Source", "terms", "Contents", "Download PDF", "Close"))
    meta = []
    if src.get("title") or src.get("url"):
        name = esc_html(src.get("title") or src.get("url"))
        meta.append(f'<span>{t_src}: ' + (f'<a href="{esc_html(src["url"])}">{name}</a>' if src.get("url") else name) + "</span>")
    if duration:
        meta.append(f'<span class="tc">{esc_html(tcode(duration))}</span>')
    # the PDF is made by the reader's browser (print → save as PDF, into their downloads folder); a file printed with
    # --pdf is linked instead
    meta.append(f'<a class="pdf" href="{esc_html(pdf_name)}" download>{t_pdf}</a>' if pdf_name
                else f'<button class="pdf" type="button" onclick="window.print()">{t_pdf}</button>')
    def heading(s: Dict[str, Any]) -> str:  # a guide numbers its steps itself — a «1.» the model put in front is dropped
        h = s["heading"]
        return re.sub(r"^\s*\d{1,2}\s*[.)]\s+", "", h) if kind == "guide" else h
    toc = ""
    if len(sections) >= 4:
        toc = f'<nav class="toc"><p>{t_contents}</p><ol>' + "".join(
            f'<li><a href="#s{i}">{esc_html(heading(s))}</a></li>' for i, s in enumerate(sections, 1)) + "</ol></nav>"
    body = []
    n = 0
    for i, s in enumerate(sections, 1):
        figs = ""
        if s["images"]:
            cells = []
            for im in s["images"]:
                n += 1
                cap = esc_html(im["caption"]) if im.get("caption") else ""
                cells.append(f'<figure><a href="{esc_html(im["src"])}" data-i="{n}" data-cap="{cap}" data-t="{esc_html(im["time"])}">'
                             f'<img src="{esc_html(im["src"])}" alt="{cap or esc_html(im["time"])}" loading="lazy"><span class="tc">{esc_html(im["time"])}</span></a>'
                             + (f"<figcaption>{cap}</figcaption>" if cap else "") + "</figure>")
            figs = f'<div class="figs n{min(len(cells), 3)}">' + "".join(cells) + "</div>"
        num = f'<span class="num" aria-hidden="true">{i}.</span>' if kind == "guide" else ""
        body.append(f'<section id="s{i}">\n<h2>{num}{esc_html(heading(s))}</h2>\n{guide_text_html(s["text"])}\n{figs}\n</section>')
    foot = (f'<a href="{REPO_URL}">{made}</a> · <img class="imt" src="data:image/png;base64,{IMT_LOGO_PNG}" alt="" width="40" height="20">'
            f'{esc_html(brand)} · <a href="{TERMS_URL}">{t_terms[0].upper() + t_terms[1:]}</a>')
    return f"""<!doctype html>
<html lang="{esc_html(lang)}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex"><title>{esc_html(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<style>
:root{{--paper:#fbfbfa;--ink:#1c1d21;--ink2:#5c5f66;--rule:#e4e4e2;--acc:#b8780f;--pdf:#008080;--chip:#111214;--chipink:#f2d38a;--code:#f0efeb}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--paper:#131417;--ink:#ececea;--ink2:#9a9ca3;--rule:#2a2b30;--acc:#e3b04b;--pdf:#008080;--chip:#0a0a0b;--chipink:#f2d38a;--code:#1e1f24}}}}
*{{box-sizing:border-box}}html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--paper);color:var(--ink);font:17px/1.6 "IBM Plex Sans",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-feature-settings:"ss01"}}
main{{max-width:46rem;margin:0 auto;padding:3.5rem 1.25rem 5rem}}
header h1{{font-size:clamp(2.3rem,6vw,3.3rem);line-height:1.1;font-weight:600;letter-spacing:-.015em;margin:0 0 .9rem}}
.meta{{display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;color:var(--ink2);font-size:.9rem;margin:0 0 2rem}}
.meta a,.meta .pdf{{color:var(--acc);text-decoration:none;border:0;border-bottom:1px solid transparent;background:none;padding:0;font:inherit;cursor:pointer}}.meta a:hover{{border-color:var(--acc)}}.meta .pdf{{color:var(--pdf)}}.meta .pdf:hover{{border-color:var(--pdf)}}.meta .pdf:focus-visible{{outline:2px solid var(--acc);outline-offset:3px}}
.tc{{font:500 .78rem/1 "IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.02em;font-variant-numeric:tabular-nums}}
.meta .tc{{align-self:center;padding:.2rem .45rem;border:1px solid var(--rule);border-radius:4px}}
.lead{{font-size:1.1rem;line-height:1.55;color:var(--ink);margin:0 0 2rem}}.lead p{{margin:0 0 .75rem}}
.toc{{margin:0 0 2.5rem;padding:.2rem 0 .2rem 1.1rem;border-left:3px solid var(--acc);font-size:.95rem}}
.toc p{{margin:0 0 .4rem;color:var(--ink2)}}.toc ol{{margin:0;padding-left:1.4rem;color:var(--ink)}}.toc li{{margin:.15rem 0}}.toc a{{color:inherit;text-decoration:none}}.toc li:hover{{color:var(--acc)}}
section{{position:relative;margin:2.75rem 0 0;padding-top:1.5rem;border-top:1px solid var(--rule)}}
h2{{font-size:1.35rem;line-height:1.3;font-weight:600;margin:0 0 .75rem;letter-spacing:-.005em}}
.num{{display:inline-block;width:2.6rem;margin-left:-2.6rem;color:var(--acc);font-variant-numeric:tabular-nums}}
@media (max-width:52rem){{.num{{width:auto;margin:0 .5rem 0 0}}}}
p{{margin:0 0 .9rem}}ul,ol{{margin:0 0 .9rem;padding-left:1.4rem}}li{{margin:.2rem 0}}
code{{font:.88em "IBM Plex Mono",ui-monospace,monospace;background:var(--code);padding:.08em .35em;border-radius:4px}}
.figs{{display:grid;gap:1rem;margin:1.25rem 0 .5rem}}.figs.n2,.figs.n3{{grid-template-columns:1fr 1fr}}
@media (max-width:40rem){{.figs.n2,.figs.n3{{grid-template-columns:1fr}}}}
figure{{margin:0}}figure a{{position:relative;display:block;border-radius:8px;overflow:hidden;background:#000;outline:none}}
figure a:focus-visible{{box-shadow:0 0 0 3px var(--acc)}}
figure img{{display:block;width:100%;height:auto;cursor:zoom-in}}
figure .tc{{position:absolute;left:.5rem;bottom:.5rem;padding:.3rem .45rem;border-radius:4px;background:var(--chip);color:var(--chipink)}}
figcaption{{font-size:.9rem;line-height:1.45;color:var(--ink2);margin:.45rem .1rem 0}}
footer{{margin-top:4rem;padding-top:1rem;border-top:1px solid var(--rule);color:var(--ink2);font-size:.82rem}}footer a{{color:var(--ink2)}}
footer .imt{{height:.95em;width:auto;vertical-align:-.12em;margin-right:.35em;opacity:.85}}@media (prefers-color-scheme: light){{:root:not([data-theme=dark]) footer .imt{{filter:invert(1)}}}}
/* lightbox */
.lb{{position:fixed;inset:0;background:rgba(8,8,10,.94);display:none;z-index:9;flex-direction:column;align-items:center;justify-content:center;padding:3.5rem 1rem 2.5rem}}
.lb.on{{display:flex}}.lb img{{max-width:100%;max-height:calc(100vh - 7.5rem);object-fit:contain;border-radius:4px;box-shadow:0 20px 60px rgba(0,0,0,.6)}}
.lb .cap{{margin-top:.9rem;color:#d9d9d6;font-size:.95rem;text-align:center;max-width:60rem}}.lb .cap .tc{{color:var(--chipink);margin-right:.6rem}}
.lb button{{position:absolute;background:none;border:0;color:#fff;cursor:pointer;font:400 2.2rem/1 "IBM Plex Sans",sans-serif;padding:.4rem .7rem;border-radius:8px;opacity:.75}}
.lb button:hover,.lb button:focus-visible{{opacity:1;background:rgba(255,255,255,.1);outline:none}}
.lb .x{{top:.6rem;right:.9rem}}.lb .p{{left:.4rem;top:50%;transform:translateY(-50%)}}.lb .n{{right:.4rem;top:50%;transform:translateY(-50%)}}
@media (max-width:40rem){{.lb .p,.lb .n{{display:none}}}}
@media (prefers-reduced-motion:no-preference){{.lb img{{animation:lbin .18s ease-out}}@keyframes lbin{{from{{opacity:0;transform:scale(.97)}}to{{opacity:1;transform:none}}}}}}
/* print: tight — 11pt, two frames to a row, a frame no taller than a third of the page */
@page{{size:A4;margin:12mm}}
@media print{{body{{background:#fff;color:#1c1d21;font-size:11pt;line-height:1.4}}main{{padding:0;max-width:none}}.toc,.pdf,.lb{{display:none!important}}
header h1{{font-size:22pt;margin-bottom:4pt}}.meta{{font-size:9pt;margin-bottom:10pt}}.lead{{font-size:11pt;margin-bottom:10pt}}
section{{margin-top:12pt;padding-top:8pt;break-inside:auto}}h2{{font-size:14pt;margin-bottom:6pt;break-after:avoid}}.num{{width:auto;margin:0 .4em 0 0}}
p,ul,ol{{margin-bottom:6pt}}.figs{{display:grid;grid-template-columns:1fr 1fr;gap:5mm;margin:6pt 0 8pt}}.figs.n1{{grid-template-columns:1fr}}.figs.n1 a{{max-width:80%;margin:0 auto}}
figure{{break-inside:avoid}}figure a{{border-radius:3px}}figure img{{max-height:62mm;width:auto;max-width:100%;margin:0 auto}}figure .tc{{font-size:7.5pt}}figcaption{{font-size:8.5pt;margin-top:3pt;text-align:center}}
footer{{margin-top:14pt;font-size:8pt}}footer .imt{{filter:invert(1)}}}}
</style></head><body><main>
<header><h1>{esc_html(title)}</h1>{f'<p class="meta">{" ".join(meta)}</p>' if meta else ""}</header>
{f'<div class="lead">{guide_text_html(intro)}</div>' if intro else ""}
{toc}
{chr(10).join(body)}
<footer>{foot}</footer>
</main>
<div class="lb" id="lb" role="dialog" aria-modal="true" aria-label="{esc_html(title)}"><button class="x" aria-label="{t_close}">×</button><button class="p" aria-label="‹">‹</button><button class="n" aria-label="›">›</button><img alt=""><p class="cap"></p></div>
<script>
(function(){{var L=[].slice.call(document.querySelectorAll('figure a')),lb=document.getElementById('lb'),im=lb.querySelector('img'),cap=lb.querySelector('.cap'),k=-1,last=null;
function show(i){{k=(i+L.length)%L.length;var a=L[k];im.src=a.getAttribute('href');im.alt=a.dataset.cap||a.dataset.t;
cap.innerHTML='<span class="tc">'+a.dataset.t+'</span>'+(a.dataset.cap||'');lb.classList.add('on');document.body.style.overflow='hidden';lb.querySelector('.x').focus();}}
function hide(){{lb.classList.remove('on');document.body.style.overflow='';im.src='';if(last)last.focus();}}
L.forEach(function(a,i){{a.addEventListener('click',function(e){{e.preventDefault();last=a;show(i);}});}});
lb.querySelector('.x').onclick=hide;lb.querySelector('.p').onclick=function(e){{e.stopPropagation();show(k-1);}};lb.querySelector('.n').onclick=function(e){{e.stopPropagation();show(k+1);}};
lb.addEventListener('click',function(e){{if(e.target===lb||e.target===im)hide();}});
document.addEventListener('keydown',function(e){{if(!lb.classList.contains('on'))return;if(e.key==='Escape')hide();else if(e.key==='ArrowLeft')show(k-1);else if(e.key==='ArrowRight')show(k+1);}});}})();
</script></body></html>
"""


def find_browser(override: Optional[str] = None) -> Optional[str]:
    """A Chromium-family browser that can print a page to PDF: Chrome, Chromium, Edge, Brave (env VIDEO2LLM_BROWSER first)."""
    cand: List[str] = [x for x in [override, os.environ.get("VIDEO2LLM_BROWSER")] if x]
    if sys.platform == "darwin":
        for app, exe in (("Google Chrome", "Google Chrome"), ("Chromium", "Chromium"), ("Microsoft Edge", "Microsoft Edge"),
                         ("Brave Browser", "Brave Browser"), ("Arc", "Arc"), ("Vivaldi", "Vivaldi")):
            for root in ("/Applications", str(Path.home() / "Applications")):
                cand.append(f"{root}/{app}.app/Contents/MacOS/{exe}")
    elif os.name == "nt":
        for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
            if root:
                cand += [rf"{root}\Google\Chrome\Application\chrome.exe", rf"{root}\Microsoft\Edge\Application\msedge.exe",
                         rf"{root}\BraveSoftware\Brave-Browser\Application\brave.exe", rf"{root}\Chromium\Application\chrome.exe"]
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "brave-browser", "chrome", "msedge"):
        w = shutil.which(name)
        if w:
            cand.append(w)
    for c in cand:
        if c and Path(c).is_file():
            return c
    return None


def html_to_pdf(html: Path, pdf: Path, browser: Optional[str] = None) -> bool:
    exe = find_browser(browser)
    if not exe:
        return False
    tmp_profile = Path(tempfile.mkdtemp(prefix="v2l-pdf-"))
    try:
        r = run([exe, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                 f"--user-data-dir={tmp_profile}", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
                 "--virtual-time-budget=4000", f"--print-to-pdf={pdf}", html.resolve().as_uri()])
        return r.returncode == 0 and pdf.is_file() and pdf.stat().st_size > 1000
    finally:
        shutil.rmtree(tmp_profile, ignore_errors=True)


def guide_default_out(title: str) -> Path:
    return cache_dir() / GUIDE_HOME / guide_slug(title)


def guide_main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(prog="video2llm guide",
                                 description="An illustrated document of a video from a spec the model wrote: index.html + img/ — "
                                             "one page, the frames open full screen on a click (+ a PDF with --pdf, printed by the "
                                             "Chrome / Chromium / Edge / Brave on this computer). Nothing is uploaded — see `video2llm link`.")
    ap.add_argument("--spec", required=True, help="JSON: {title, intro?, lang?, kind?: guide | notes (guide numbers the sections), "
                                                  "source: {file, title?, url?}, brand?, sections: [{heading, text, frames?: [mm:ss…], captions?: […]}]}")
    ap.add_argument("--out", help="output folder (default: the user cache, video2llm/guides/<title>/)")
    ap.add_argument("--pdf", action="store_true", help="also print <title>.pdf here (needs a Chromium-family browser); without it the page's "
                                                   "«Download PDF» button prints through the reader's own browser")
    ap.add_argument("--browser", help="the browser executable for the PDF (default: found; env VIDEO2LLM_BROWSER)")
    ap.add_argument("--ffmpeg", help="path to ffmpeg")
    ap.add_argument("--json", action="store_true", help="print {dir, html, pdf, frames} as JSON")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"--spec: {e}")
    title = str(spec.get("title") or "").strip()
    secs = spec.get("sections")
    if not title or not isinstance(secs, list) or not secs:
        sys.exit("--spec: title and a non-empty sections list are required")
    src = spec.get("source") or {}
    video = Path(str(src.get("file") or "")).expanduser()
    progress = make_progress(a.quiet)
    times: List[str] = []
    for s in secs:
        if not isinstance(s, dict) or not str(s.get("heading") or "").strip() or not isinstance(s.get("text"), str):
            sys.exit("--spec: every section needs a heading and a text")
        for x in s.get("frames") or []:
            x = str(x).strip()
            if not _TIME_RE.match(x):
                sys.exit(f"--spec: frames: {x} — moments are mm:ss or mm:ss.cc")
            if x not in times:
                times.append(x)
    if len(times) > GUIDE_MAX_FRAMES:
        sys.exit(f"--spec: up to {GUIDE_MAX_FRAMES} frames in one document")
    if times and not video.is_file():
        sys.exit(f"--spec: source.file is not a video file: {video}")
    out_dir = Path(a.out).expanduser().resolve() if a.out else guide_default_out(title)
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    if not writable_dir(out_dir):
        sys.exit(f"cannot write to {out_dir}")
    # the frames: cut once, 1024px on the long side, into img/mm-ss.cc.jpg (a frame cut earlier for `--frames at` is reused)
    files: Dict[str, Path] = {}
    if times:
        ffmpeg = find_tool("ffmpeg", a.ffmpeg)
        at_dir = video.with_name(f"{video.stem}_frames") / NAMES["en"]["at"].format(w=AT_WIDTH)
        at_dir_ru = video.with_name(f"{video.stem}_frames") / NAMES["ru"]["at"].format(w=AT_WIDTH)
        for i, x in enumerate(times):
            t = parse_tcode(x)
            name = f"{fmt_t(math.floor(t))}.{int(round((t % 1) * 100)):02d}.jpg"
            dest = out_dir / "img" / name
            if not dest.is_file():
                ready = next((d / name for d in (at_dir, at_dir_ru) if (d / name).is_file()), None)
                if ready:
                    shutil.copyfile(ready, dest)
                else:
                    r = run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                             "-frames:v", "1", "-vf", fit_filter(AT_WIDTH), "-q:v", "3", str(dest)])
                    if r.returncode != 0 or not dest.is_file():
                        log(f"  frame at {x}: ffmpeg failed")
                        continue
            files[x] = dest
            progress("frames", int((i + 1) / len(times) * 100))
    sections: List[Dict[str, Any]] = []
    for s in secs:
        images = []
        caps = s.get("captions") or []
        for i, x in enumerate(s.get("frames") or []):
            x = str(x).strip()
            if x in files:
                images.append({"src": f"img/{files[x].name}", "time": x, "caption": str(caps[i]) if i < len(caps) and caps[i] else None})
        sections.append({"heading": str(s["heading"]).strip(), "text": s["text"], "images": images})
    if not src.get("title") and video.is_file():
        info = read_info_json(video)
        src = {**src, "title": info.get("title") or video.name, "url": src.get("url") or info.get("webpage_url")}
        spec["source"] = src
    pdf_name = f"{guide_slug(title)}.pdf" if a.pdf else None
    duration: Optional[float] = None
    if video.is_file():
        try:
            r = run([find_tool("ffprobe", None), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)])
            duration = float((r.stdout or "").strip() or 0) or None
        except (OSError, ValueError, SystemExit):
            duration = float(read_info_json(video).get("duration") or 0) or None
    html_path = out_dir / "index.html"
    html_path.write_text(guide_html(spec, sections, pdf_name=pdf_name, duration=duration), encoding="utf-8")
    result: Dict[str, Any] = {"dir": str(out_dir), "html": str(html_path), "frames": len(files), "sections": len(sections)}
    log(f"video2llm {VERSION} — guide «{title}»: {len(sections)} section(s), {len(files)} frame(s)")
    log(f"  wrote {html_path}")
    if a.pdf:
        pp = out_dir / pdf_name
        if html_to_pdf(html_path, pp, a.browser):
            result["pdf"] = str(pp)
            log(f"  wrote {pp} ({pp.stat().st_size / 1e6:.1f} MB)")
        else:
            result["pdf"] = None
            result["pdf_note"] = "no Chrome / Chromium / Edge / Brave found to print the PDF — install one or pass --browser"
            log("  PDF skipped: " + result["pdf_note"])
    if a.json:
        print(json.dumps(result, ensure_ascii=False), flush=True)


# ───────────────────────────── link: a web address for the document, the user's own ─────────────────────────────
# `video2llm link DIR [--github] [--json]`
# The document goes to the user's own Neocities site (free: 1 GB, HTML / images / PDF allowed; a site lives as long as
# the account): sign-up takes a minute in the browser, the API key is pasted once into a small local page this script
# opens (http://127.0.0.1:…) — never into a chat — and is kept in ~/.config/video2llm/. --github instead: the user's
# GitHub Pages (gh when logged in, else a saved token, else the device flow — a link and a code, the sign-in happens in
# the browser). The address is the user's, for their own use; this script never calls it publishing.
NEOCITIES_API = "https://neocities.org/api"
GITHUB_API = "https://api.github.com"
GITHUB_CLIENT_ID = os.environ.get("VIDEO2LLM_GITHUB_CLIENT_ID", "")  # IMT's OAuth app (device flow); gh stands in when present
GITHUB_REPO_NAME = "v2l-guides"
LINK_EXCLUDE = {".DS_Store", "Thumbs.db"}
KEY_WAIT_SEC = 15 * 60


def config_dir() -> Path:
    base = Path(os.environ.get("APPDATA", Path.home() / ".config")) if os.name == "nt" else Path.home() / ".config"
    d = base / "video2llm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _http(method: str, url: str, *, data: Optional[bytes] = None, headers: Optional[Dict[str, str]] = None,
          timeout: float = 60) -> Tuple[int, bytes, Dict[str, str]]:
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": f"video2llm/{VERSION}", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def _json(method: str, url: str, body: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
          timeout: float = 60) -> Tuple[int, Any]:
    h = {"Accept": "application/json", **(headers or {})}
    data = None
    if body is not None:
        h["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    st, raw, _ = _http(method, url, data=data, headers=h, timeout=timeout)
    try:
        return st, json.loads(raw.decode("utf-8", "replace")) if raw else {}
    except ValueError:
        return st, {"raw": raw[:300].decode("utf-8", "replace")}


def link_files(folder: Path) -> List[Tuple[str, Path]]:
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name not in LINK_EXCLUDE and not p.name.startswith("."):
            out.append((p.relative_to(folder).as_posix(), p))
    return out


def links_log(entry: Dict[str, Any]) -> None:
    """Every address this computer made, in ~/.config/video2llm/links.json."""
    f = config_dir() / "links.json"
    try:
        rows = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else []
    except (OSError, ValueError):
        rows = []
    rows.append({"at": time.strftime("%Y-%m-%d %H:%M"), **entry})
    f.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def _secret_file(name: str) -> Path:
    return config_dir() / name


def _read_secret(name: str) -> Optional[str]:
    try:
        v = _secret_file(name).read_text(encoding="utf-8").strip()
        return v or None
    except OSError:
        return None


def _write_secret(name: str, value: str) -> None:
    f = _secret_file(name)
    f.write_text(value.strip() + "\n", encoding="utf-8")
    try:
        os.chmod(f, 0o600)
    except OSError:
        pass


def ask_key_in_browser(service: str, steps: List[str], check: Callable[[str], Optional[str]], where: str = "") -> str:
    """A key is pasted into a page on this computer (http://127.0.0.1:<port>/), never into a chat: the page shows the
    steps, takes the key, `check` verifies it (returns an error text or None) and it is saved by the caller."""
    import http.server, socketserver, threading, webbrowser, urllib.parse
    result: Dict[str, Any] = {}
    page = ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc_html(service)} — V2L-IMT</title><style>body{{font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:560px;"
            "margin:48px auto;padding:0 24px;background:#141416;color:#eee}a{color:#e0a03c}input{width:100%;font:16px monospace;padding:10px;"
            "border-radius:8px;border:1px solid #444;background:#1e1e22;color:#eee}button{margin-top:12px;font:16px -apple-system,sans-serif;"
            "padding:10px 18px;border-radius:8px;border:0;background:#e0a03c;color:#111}ol li{margin:6px 0}.err{color:#f66}</style></head><body>"
            f"<h2>{esc_html(service)}</h2><ol>" + "".join(f"<li>{s}</li>" for s in steps) + "</ol>"
            "<form method='post'><input name='key' placeholder='API key' autofocus autocomplete='off'><button>Save on this computer</button></form>"
            "{err}<p style='color:#888;font-size:13px'>The key stays in ~/.config/video2llm/ on this computer — it is not sent anywhere else.</p></body></html>")

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _send(self, body: str, code: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(page.replace("{err}", ""))

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            form = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
            key = (form.get("key") or [""])[0].strip()
            err = check(key) if key else "no key"
            if err:
                self._send(page.replace("{err}", f"<p class='err'>{esc_html(err)}</p>"))
                return
            result["key"] = key
            self._send("<!doctype html><html><body style='font:16px -apple-system,sans-serif;background:#141416;color:#eee;padding:48px'>"
                       "<h2>Saved. You can close this tab and go back.</h2></body></html>")

    with socketserver.TCPServer(("127.0.0.1", 0), H) as srv:
        port = srv.server_address[1]
        url = f"http://127.0.0.1:{port}/"
        th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
        print(f"\n{service}: open {url} in your browser, follow the steps there and paste the key"
              + (f" — {where}" if where else "") + " (waits up to 15 minutes)\n", flush=True)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
        deadline = time.time() + KEY_WAIT_SEC
        while "key" not in result and time.time() < deadline:
            time.sleep(0.5)
        srv.shutdown()
    if "key" not in result:
        raise RuntimeError(f"{service}: the key was not entered in time — run the command again")
    return result["key"]


def _multipart(parts: List[Tuple[str, str, bytes]]) -> Tuple[bytes, str]:
    import mimetypes, uuid
    bnd = "----v2l" + uuid.uuid4().hex
    buf = bytearray()
    for name, filename, data in parts:
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        buf += (f"--{bnd}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode()
        buf += data + b"\r\n"
    buf += f"--{bnd}--\r\n".encode()
    return bytes(buf), f"multipart/form-data; boundary={bnd}"


def neocities_check(key: str) -> Optional[str]:
    st, d = _json("GET", f"{NEOCITIES_API}/info", None, {"Authorization": f"Bearer {key}"})
    return None if st == 200 and d.get("result") == "success" else "Neocities did not accept this key"


def neocities_key() -> Tuple[str, str]:
    """The saved key, or the one the user pastes into the local page; returns (key, sitename)."""
    key = _read_secret("neocities.key")
    if key and neocities_check(key) is None:
        pass
    else:
        key = ask_key_in_browser("Neocities", [
            "<a href='https://neocities.org/signup' target='_blank'>Sign up</a> (a minute: a site name, a password, an e-mail) — "
            "or <a href='https://neocities.org/signin' target='_blank'>sign in</a> if you have a site",
            "The key: <b>Profile</b> (top right) → <a href='https://neocities.org/settings' target='_blank'>Settings</a> → your site → "
            "<b>Manage Site Settings</b> → <b>API Key</b> → Generate",
            "Paste it below"], neocities_check, where="the key is in Neocities: Profile → Settings → Manage Site Settings → API Key")
        _write_secret("neocities.key", key)
    st, d = _json("GET", f"{NEOCITIES_API}/info", None, {"Authorization": f"Bearer {key}"})
    return key, str((d.get("info") or {}).get("sitename") or "")


def neocities_link(folder: Path, slug: str, progress: Progress, title: str = "") -> Dict[str, Any]:
    key, sitename = neocities_key()
    if not sitename:
        raise RuntimeError("neocities: the site name is unknown — check the key in Settings")
    files = link_files(folder)
    batch: List[Tuple[str, str, bytes]] = []
    size = 0
    sent = 0

    def flush() -> None:
        nonlocal batch, size
        if not batch:
            return
        body, ctype = _multipart(batch)
        st, raw, _ = _http("POST", f"{NEOCITIES_API}/upload", data=body,
                           headers={"Authorization": f"Bearer {key}", "Content-Type": ctype}, timeout=300)
        try:
            d = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            d = {}
        if st != 200 or d.get("result") != "success":
            raise RuntimeError(f"neocities: upload failed ({st}: {d.get('message') or raw[:200].decode('utf-8', 'replace')})")
        batch, size = [], 0

    for i, (rel, p) in enumerate(files):
        data = p.read_bytes()
        if size + len(data) > 60 * 1024 * 1024:  # the request limit is 100 MB
            flush()
        batch.append((f"{slug}/{rel}", p.name, data)); size += len(data); sent += 1
        progress("upload", int((i + 1) / len(files) * 100))
    flush()
    url = f"https://{sitename}.neocities.org/{slug}/"
    out = {"provider": "neocities", "url": url, "site": f"https://{sitename}.neocities.org/", "files": sent,
           "note": "the address works at once; it stays as long as the Neocities account does"}
    links_log({"kind": "neocities", **out, "dir": str(folder)})
    return out


# — GitHub —
def _gh_cli_token() -> Optional[str]:
    gh = shutil.which("gh")
    if not gh:
        return None
    r = run([gh, "auth", "token"])
    t = (r.stdout or "").strip()
    return t if r.returncode == 0 and t else None


def _saved_token() -> Optional[str]:
    f = config_dir() / "github.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("token") if f.is_file() else None
    except (OSError, ValueError):
        return None


def _save_token(token: str) -> None:
    f = config_dir() / "github.json"
    f.write_text(json.dumps({"token": token}), encoding="utf-8")
    try:
        os.chmod(f, 0o600)
    except OSError:
        pass


def github_device_flow(progress: Progress) -> str:
    """The sign-in that never sees a password: a link and a code are printed, the user opens the browser, signs in or
    signs up there, enters the code; the token comes back here and is saved for the next time."""
    if not GITHUB_CLIENT_ID:
        raise RuntimeError("github: no way to sign in — install GitHub CLI (`gh auth login`) or set VIDEO2LLM_GITHUB_CLIENT_ID")
    st, d = _json("POST", "https://github.com/login/device/code", {"client_id": GITHUB_CLIENT_ID, "scope": "public_repo"})
    if st != 200 or not d.get("device_code"):
        raise RuntimeError(f"github: device code not issued ({st}: {str(d)[:200]})")
    print(f"\nOpen {d['verification_uri']} and enter the code  {d['user_code']}  (sign in or sign up there; this waits up to 15 minutes)\n",
          flush=True)
    interval = int(d.get("interval") or 5)
    deadline = time.time() + min(int(d.get("expires_in") or 900), 900)
    while time.time() < deadline:
        time.sleep(interval)
        st, r = _json("POST", "https://github.com/login/oauth/access_token",
                      {"client_id": GITHUB_CLIENT_ID, "device_code": d["device_code"], "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})
        if r.get("access_token"):
            _save_token(r["access_token"])
            return r["access_token"]
        err = r.get("error")
        if err == "slow_down":
            interval += 5
        elif err in ("expired_token", "access_denied"):
            raise RuntimeError(f"github: {err.replace('_', ' ')}")
    raise RuntimeError("github: the code was not entered in time — run the command again")


def github_token(progress: Progress) -> str:
    for t in (_gh_cli_token(), _saved_token()):
        if t:
            st, u = _json("GET", f"{GITHUB_API}/user", None, {"Authorization": f"Bearer {t}"})
            if st == 200:
                return t
    return github_device_flow(progress)


def github_link(folder: Path, slug: str, progress: Progress, title: str = "") -> Dict[str, Any]:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("github: git is not installed")
    token = github_token(progress)
    auth = {"Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
    st, user = _json("GET", f"{GITHUB_API}/user", None, auth)
    if st != 200:
        raise RuntimeError(f"github: not signed in ({st})")
    login, uid = user["login"], user.get("id")
    st, repo = _json("GET", f"{GITHUB_API}/repos/{login}/{GITHUB_REPO_NAME}", None, auth)
    if st == 404:
        st, repo = _json("POST", f"{GITHUB_API}/user/repos",
                         {"name": GITHUB_REPO_NAME, "description": "Documents of videos, made with V2L-IMT", "auto_init": True,
                          "has_issues": False, "has_wiki": False, "has_projects": False}, auth)
        if st not in (200, 201):
            raise RuntimeError(f"github: the repository was not created ({st}: {str(repo)[:200]})")
        time.sleep(2)
    elif st != 200:
        raise RuntimeError(f"github: {st}: {str(repo)[:200]}")
    branch = repo.get("default_branch") or "main"
    work = cache_dir() / "github" / login / GITHUB_REPO_NAME
    remote = f"https://github.com/{login}/{GITHUB_REPO_NAME}.git"
    remote_auth = f"https://x-access-token:{token}@github.com/{login}/{GITHUB_REPO_NAME}.git"
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def g(*args: str, check: bool = True) -> subprocess.CompletedProcess:
        r = subprocess.run([git, "-C", str(work), *args], capture_output=True, text=True, env=env)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {args[0]}: {(r.stderr or r.stdout).strip()[:300].replace(token, '***')}")
        return r

    if not (work / ".git").is_dir():
        work.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([git, "clone", "-q", "--depth", "50", remote_auth, str(work)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError(f"git clone: {(r.stderr or r.stdout).strip()[:300].replace(token, '***')}")
        g("remote", "set-url", "origin", remote)
    else:
        g("fetch", "-q", remote_auth, branch, check=False)
        g("reset", "-q", "--hard", "FETCH_HEAD", check=False)
    g("config", "user.name", login)
    g("config", "user.email", f"{uid}+{login}@users.noreply.github.com" if uid else f"{login}@users.noreply.github.com")
    dest = work / slug
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(folder, dest, ignore=shutil.ignore_patterns(".*", "Thumbs.db"))
    (work / ".nojekyll").write_text("", encoding="utf-8")
    # the root page: a plain list of the documents (titles from each index.html)
    rows = []
    for d in sorted(p for p in work.iterdir() if p.is_dir() and not p.name.startswith(".") and (p / "index.html").is_file()):
        m = re.search(r"<title>(.*?)</title>", (d / "index.html").read_text(encoding="utf-8", errors="replace"), re.S)
        rows.append(f'<li><a href="{esc_html(d.name)}/">{esc_html(m.group(1).strip() if m else d.name)}</a></li>')
    (work / "index.html").write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>Documents</title><style>body{font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:720px;margin:48px auto;"
        "padding:0 24px;color:#1d1d1f}@media(prefers-color-scheme:dark){body{background:#141416;color:#eee}a{color:#e0a03c}}"
        "li{margin:6px 0}footer{margin-top:40px;font-size:12px;color:#888}</style></head><body><h1>Documents</h1><ul>"
        + "".join(rows) + f'</ul><footer><a href="{REPO_URL}">Made with V2L-IMT</a></footer></body></html>', encoding="utf-8")
    g("add", "-A")
    changed = bool(g("status", "--porcelain").stdout.strip())
    if changed:
        g("commit", "-q", "-m", f"{title or slug} ({slug})")
        g("push", "-q", remote_auth, f"HEAD:{branch}")
    st, _ = _json("GET", f"{GITHUB_API}/repos/{login}/{GITHUB_REPO_NAME}/pages", None, auth)
    if st == 404:
        st, d = _json("POST", f"{GITHUB_API}/repos/{login}/{GITHUB_REPO_NAME}/pages",
                      {"build_type": "legacy", "source": {"branch": branch, "path": "/"}}, auth)
        if st not in (200, 201):
            log(f"  github: Pages not switched on by the API ({st}) — repo Settings → Pages → {branch} / root")
    url = f"https://{login}.github.io/{GITHUB_REPO_NAME}/{slug}/"
    out = {"provider": "github", "url": url, "repo": f"https://github.com/{login}/{GITHUB_REPO_NAME}",
           "unchanged": not changed, "note": "the page is live within a minute or two after the first push"}
    links_log({"kind": "open", **out, "dir": str(folder)})
    return out


def link_main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(prog="video2llm link",
                                 description="A web address for a document made by `video2llm guide` — the user's own link, for their "
                                             "own use, on their own Neocities site (free; sign-up in the browser takes a minute; the API "
                                             "key is pasted once into a local page, never into a chat). --github: their GitHub Pages "
                                             f"instead (gh, a saved token, or the device flow) — repo {GITHUB_REPO_NAME}.")
    ap.add_argument("dir", help="the document folder (index.html + img/ …)")
    ap.add_argument("--github", action="store_true", help="the user's GitHub Pages instead of Neocities")
    ap.add_argument("--slug", help="the page's folder on the site (default: from the title, latin)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    folder = Path(a.dir).expanduser().resolve()
    if not (folder / "index.html").is_file():
        sys.exit(f"{folder}: no index.html — make the document with `video2llm guide` first")
    progress = make_progress(a.quiet)
    m = re.search(r"<title>(.*?)</title>", (folder / "index.html").read_text(encoding="utf-8", errors="replace"), re.S)
    title = (m.group(1).strip() if m else folder.name)
    slug = a.slug or web_slug(title)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,78}", slug):
        sys.exit("--slug: latin letters, digits and dashes")
    try:
        out = github_link(folder, slug, progress, title) if a.github else neocities_link(folder, slug, progress, title)
        log(f"  link: {out['url']}" + (" — unchanged" if out.get("unchanged") else ""))
    except RuntimeError as e:
        if a.json:
            print(json.dumps({"error": str(e)}, ensure_ascii=False), flush=True)
        sys.exit(f"✗ {e}")
    if a.json:
        print(json.dumps(out, ensure_ascii=False), flush=True)
    else:
        print(out["url"])



def rerun_cmd(a: argparse.Namespace, source: Path, out_dir: Path) -> str:
    """The command the lane header gives for the next run: this script, the same file, the same settings that
    shape the files (a follow-up run without them would cut into other folders)."""
    return (("& " if os.name == "nt" else "")  # PowerShell runs a quoted path only with the call operator
            + f"{shell_arg(sys.executable)} {shell_arg(Path(__file__).resolve())} {shell_arg(source)}"
            + (f" --out {shell_arg(out_dir)}" if a.out else "")
            # the settings that shape the files: a follow-up run without them would cut into other folders
            + (f" --names {a.names}" if a.names != "en" else "") + (" --single-frames" if a.single_frames else "")
            + (f" --sheet-frames {a.sheet_frames}" if a.sheet_frames else "")
            + (f" --fps {a.fps}" if a.fps != FRAME_FPS else "") + (f" --max-frames {a.max_frames}" if a.max_frames else "")
            + (f" --language {a.language}" if a.language != "auto" else "") + (f" --captions {a.captions}" if a.captions else "")
            + (f" --whisper-model {shell_arg(a.whisper_model)}" if a.whisper_model != "small" else "")
            + (f" --sound-threshold {a.sound_threshold:g}" if a.sound_threshold != 0.15 else "")
            + (f" --ffmpeg {shell_arg(a.ffmpeg)}" if a.ffmpeg else "") + (f" --ffprobe {shell_arg(a.ffprobe)}" if a.ffprobe else "")
            + (f" --sounds-model {shell_arg(a.sounds_model)}" if a.sounds_model else ""))


def main(argv: Optional[List[str]] = None) -> None:
    args0 = sys.argv[1:] if argv is None else argv
    if args0 and args0[0] == "guide":   # `video2llm guide --spec …` — the document (the page, the PDF)
        return guide_main(args0[1:])
    if args0 and args0[0] == "link":    # `video2llm link DIR` — a web address for it, on the user's own site
        return link_main(args0[1:])
    ap = argparse.ArgumentParser(prog="video2llm", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Examples:\n"
                                        "  video2llm clip.mp4                        # 1 frame per second + speech → clip_frames/lane.md\n"
                                        "  video2llm clip.mp4 --frames all --start 0:05 --end 0:09   # every frame of 0:05–0:09\n"
                                        "  video2llm clip.mp4 --format pdf           # the whole video as lane.pdf — Claude / Gemini chats\n"
                                        "  video2llm clip.mp4 --format sheets        # contact sheets + text — ChatGPT and any chat\n"
                                        "  video2llm clip.mp4 --format anthropic --prompt \"What goes wrong at 0:05?\"\n"
                                        "  video2llm clip.mp4 --native --format gemini   # the video itself, fitted to 19 MB\n"
                                        "  video2llm clip.mp4 --start 0:30          # the next 30 frames of a long video\n"
                                        "  video2llm https://youtu.be/…              # a video by URL (yt-dlp): its captions stand in for Whisper\n"
                                        "  ANTHROPIC_API_KEY=… video2llm clip.mp4 --format anthropic --ask \"Describe the camera moves\"\n")
    ap.add_argument("video", help="video file (mp4, mov, m4v, webm, mpeg, avi, 3gp …) or a URL (YouTube, Vimeo, a direct link — "
                                  "anything yt-dlp knows; downloaded once into <Videos>/video2llm/downloads, up to 1080p)")
    g = ap.add_argument_group("which frames")
    g.add_argument("--frames", choices=["auto", "1", "all", "none", "at", "words"], default="auto",
                   help="auto (default) = the script decides by the material: a URL gets the WORDS lane (the site's captions, the "
                        "video not downloaded — see words); a file that is a lecture, a tutorial, a review, a talk (the file "
                        "name, the chapters, the length, the share of speech — the reasons are printed) gets the LECTURE "
                        "lane, footage to watch gets the overview; "
                        "words = the words only: no frames, the video of a URL is NOT downloaded (the site's captions, or the sound "
                        "alone → Whisper); the lane tells the model to decide from the user's message what to do and to ask for "
                        "the picture only where it is needed — the first such request downloads the video; "
                        "1 = one frame per second over the whole video, 768px — what happens, who says what; "
                        f"all = frame by frame, {DENSE_WIDTH}px — every frame from --start to --end, both included (flicker, "
                        f"glitches, fast motion); up to {DENSE_MAX_FRAMES} frames per request (4 s at 30 fps, 2 s at 60), "
                        "never thinned — a longer span is refused with the pieces to ask for; "
                        "none = the LECTURE lane: no frames, the whole transcript (and the site's chapters) with the rules to ask "
                        f"for single frames where the screen matters; at = single frames at --times, {AT_WIDTH}px")
    g.add_argument("--times", metavar="T,T,…", help=f"with --frames at: the moments, mm:ss or mm:ss.cc, comma-separated (up to {AT_MAX})")
    g.add_argument("--all-frames", action="store_true", help=argparse.SUPPRESS)  # old name of --frames all
    g.add_argument("--fps", type=int, default=FRAME_FPS, choices=range(1, OVERVIEW_FPS_MAX + 1), metavar="N",
                   help=f"a denser overview, 2–{OVERVIEW_FPS_MAX} frames a second (default 1): the frames between the seconds are "
                        "added to the same folder as [mm:ss.cc]; costs that many times the tokens of the 1 a second lane")
    g.add_argument("--start", default="0:00", help="the first frame, mm:ss or h:mm:ss, may have a fraction: 0:12.5 (default 0:00)")
    g.add_argument("--end", help="the last frame, included (default: the end of the video)")
    g.add_argument("--max-frames", type=int,
                   help=f"frames per lane.md (default {LANE_MD_FRAMES} at 1 per second) and per API request body (default "
                        f"{SPARSE_PER_MESSAGE}); {DENSE_MAX_FRAMES} with --frames all; pdf and sheets take the whole span")
    g = ap.add_argument_group("what to write")
    g.add_argument("--format", action="append", choices=["md", "pdf", "sheets", "openai", "anthropic", "gemini", "all"],
                   help="md (default): lane.md linking contact sheets of the frames (--single-frames: one image per frame) — for "
                        "agents; pdf: the whole span in lane.pdf — "
                        "Claude / Gemini chats (needs fpdf2); sheets: JPG contact sheets + a text file — ChatGPT and any chat "
                        "(needs Pillow); openai / anthropic / gemini: request bodies; all — repeatable")
    g.add_argument("--prompt", default="", help="your question, appended after the lane")
    g.add_argument("--label", default="Video 1", help="how the video is named in the header (default \"Video 1\")")
    g.add_argument("--pdf-font", help="TrueType font for PDF and sheet text (default: Arial / Segoe UI / DejaVu Sans)")
    g.add_argument("--pdf-pages", type=int, default=PDF_MAX_PAGES,
                   help=f"pages per PDF before it is split (default {PDF_MAX_PAGES}; Claude looks at the images of up to 100)")
    g.add_argument("--sheet-grid", metavar="COLSxROWS", help="frames per sheet (default 2x2; portrait 3x1; very wide 1x3)")
    g.add_argument("--sheets-per-message", type=int, default=SHEETS_PER_MESSAGE,
                   help=f"sheets per chat message (default {SHEETS_PER_MESSAGE} + the text file = 10 files, Gemini's limit)")
    g.add_argument("--single-frames", action="store_true",
                   help="lane.md: one image per frame instead of contact sheets (3 overview frames or 6 frame-by-frame "
                        "ones to an image — same pixels per frame, a third to a sixth of the images to open)")
    g.add_argument("--sheet-frames", type=int, metavar="N", choices=range(2, 13),
                   help="lane.md: exactly N frames (2–12) to a contact sheet; more than fit at their own size → the frames are "
                        "scaled so the sheet stays within Claude's limits (6 overview frames ≈ 75 %% of their size), the costs "
                        "in the header follow — for a chat that counts the files of a message (claude.ai: 20)")
    g.add_argument("--native", action="store_true", help="the video file itself, fitted into 19 MB, goes into the Gemini payload")
    g.add_argument("--part", type=int, default=1, help="which native part goes into the Gemini payload (default 1)")
    g.add_argument("--model", help="model id for the payload / --ask (defaults: " + ", ".join(f"{k} {v}" for k, v in DEFAULT_MODELS.items()) + ")")
    g.add_argument("--ask", metavar="QUESTION", help="send the lane to the API (provider from --format, key from env) and print the answer")
    g = ap.add_argument_group("speech and sounds")
    g.add_argument("--no-transcript", action="store_true", help="skip faster-whisper (the words)")
    g.add_argument("--force-whisper", action="store_true",
                   help="transcribe with Whisper even when a subtitle file sits beside the video (<name>.<lang>.vtt / .srt — "
                        "a site's captions downloaded with the URL, or your own); by default such a file stands in for Whisper")
    g.add_argument("--sounds", action="store_true",
                   help="sounds other than speech (music, laughter, a slam, a crowd …) for the --start…--end moment "
                        "(the whole video without them), by the PANNs AudioSet model; off by default — only speech. "
                        "--end here covers the whole second of that anchor. Alone it prints the tags and writes sounds_mm-ss_mm-ss.txt; "
                        "with --frames all they go into that lane too")
    g.add_argument("--no-sounds", action="store_true", help=argparse.SUPPRESS)  # the old default; sounds are opt-in now
    g.add_argument("--voice-rescue", action="store_true",
                   help="second, strict Whisper pass where the sound model hears a voice the main pass dropped "
                        "(a voice under loud music); slower, off by default")
    g.add_argument("--sounds-model", help=f"path to {SOUNDS_MODEL_NAME} (default: downloaded once into the user cache)")
    g.add_argument("--sound-threshold", type=float, default=0.15, help="PANNs probability a sound needs to be tagged (default 0.15; raise to 0.3 for fewer, surer tags)")
    g.add_argument("--whisper-model", default=os.environ.get("VIDEO2LLM_WHISPER", "small"),
                   help="small (default, as in Deep Artisan) or base (faster) — downloaded once from this project's release; "
                        "or the path to a folder with any other CTranslate2 Whisper model")
    g.add_argument("--language", default="auto", help="transcript language code (default: auto-detect)")
    g.add_argument("--captions", metavar="LANGS",
                   help="with a URL: the subtitle languages to fetch and prefer, comma-separated, e.g. \"es,en\" — your "
                        "chat's language first (default: this computer's UI language); the video's own language and ru/en "
                        "are always taken too, and a manual track in the video's own language is read before a "
                        "machine-translated one")
    g = ap.add_argument_group("other")
    g.add_argument("--out", help="output folder (default: <video>_frames next to the video)")
    g.add_argument("--names", choices=["en", "ru"], default="en", help="language of the folder names (ru = as in Deep Artisan)")
    g.add_argument("--agent-tool", metavar="NAME",
                   help="the lane is for an agent harness that runs this script itself: the header tells the model to call "
                        "the tool NAME with {video, what: next | frames | sounds, start, end} instead of giving it commands")
    g.add_argument("--ffmpeg", help="path to ffmpeg"); g.add_argument("--ffprobe", help="path to ffprobe")
    g.add_argument("--download-only", action="store_true",
                   help="with a URL: download the video and its captions, print the file's path to stdout and stop (for a program "
                        "that runs the rest itself)")
    g.add_argument("--json", action="store_true",
                   help="with --download-only: print JSON instead — {file, title, duration, language, lecture: {hint, score, reasons}}; "
                        "the lecture hint says whether the video looks like a lecture / tutorial (the site's category, title, "
                        "chapters, length) — a suggestion to use --frames none")
    g.add_argument("--yt-dlp", help="yt-dlp for a video URL: a path or a command line (\"py -m yt_dlp\"); default: the one on PATH "
                                    "or the yt_dlp module of this Python")
    g.add_argument("--quiet", action="store_true")
    g.add_argument("--version", action="version",
                   version=f"V2L-IMT {VERSION} — © 2026 Immersive Media Technologies, IMT Non-Commercial License (non-commercial use only, "
                           "attribution required; see LICENSE)")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # a Windows pipe in a legacy code page must not crash on "≈" or "→"
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    if a.all_frames:
        a.frames = "all"

    tools = Tools(find_tool("ffmpeg", a.ffmpeg), find_tool("ffprobe", a.ffprobe))
    progress = make_progress(a.quiet)
    captions_prefer = [x.strip() for x in (a.captions or "").split(",") if x.strip()] or ([os_language()] if os_language() else [])
    link_m: Optional[Dict[str, Any]] = None  # 0.7.2: the words lane of a link — there is no file to cut
    out_dir: Optional[Path] = None
    if URL_RE.match(a.video.strip()):
        log(f"video2llm {VERSION} — {a.video.strip()}")
        if a.frames in ("auto", "words") and not a.download_only:
            # 0.7.2: a link gets the words only — the video is downloaded when a frame is asked for
            source, out_dir, link_m = fetch_words(a.video.strip(), tools, progress, a.yt_dlp, a.language, captions_prefer,
                                                  Path(a.out).expanduser().resolve() if a.out else None, a.whisper_model,
                                                  not a.no_transcript)
            a.frames = "words"
        else:
            source = download_video(a.video.strip(), tools, progress, a.yt_dlp, a.language, captions_prefer).resolve()
        if a.download_only:
            if a.json:
                info = read_info_json(source)
                try:
                    pr = tools.probe(source)
                except Exception:  # noqa: BLE001 — the hint is best effort
                    pr = {"duration": float(info.get("duration") or 0)}
                cap = captions_beside(source, None, captions_prefer)
                print(json.dumps({"file": str(source), "title": info.get("title"), "duration": pr.get("duration"),
                                  "language": info.get("language"), "captions": (cap or {}).get("kind"),
                                  "lecture": lecture_hint(source, pr, (cap or {}).get("segments"))}, ensure_ascii=False), flush=True)
            else:
                print(str(source), flush=True)
            return
    else:
        if a.download_only:
            sys.exit("--download-only needs a URL")
        source = Path(a.video).expanduser().resolve()
        if not source.is_file():
            # 0.7.2: the name a link's video WOULD get (its words came first — fetch_words): the words lane again
            # from the manifest, or, when frames are asked for, the download now and the usual run
            lane_dir = Path(a.out).expanduser().resolve() if a.out else source.with_name(f"{source.stem}_frames")
            lm: Dict[str, Any] = {}
            try:
                lm = json.loads((lane_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            url = str((lm.get("link") or {}).get("url") or "")
            if url and lm.get("source_mtime") is None:
                if a.frames in ("auto", "words"):
                    link_m, out_dir, a.frames = lm, lane_dir, "words"
                else:
                    log(f"video2llm {VERSION} — {url}")
                    log("  the video of this link is not on this computer yet — downloading it")
                    source = download_video(url, tools, progress, a.yt_dlp, a.language, captions_prefer).resolve()
                    if not a.out and source.with_name(f"{source.stem}_frames") != lane_dir:
                        a.out = str(lane_dir)  # the words and the wav live here already
    if link_m is None:
        try:
            with open(source, "rb"):
                pass
        except PermissionError:
            sys.exit(f"no permission to read {source}"
                     + (" — macOS asks once per folder: allow your terminal in System Settings → Privacy & Security → "
                        "Files and Folders (or Full Disk Access)" if sys.platform == "darwin" else ""))
        except OSError:
            sys.exit(f"not a file: {source}")
        if not source.is_file():
            sys.exit(f"not a file: {source}")
    else:
        assert out_dir is not None
        names = NAMES[a.names]
        rerun = Rerun(rerun_cmd(a, source, out_dir), source, a.agent_tool)
        md_blocks = link_blocks(link_m, a.label, rerun)
        path = out_dir / ("lane ссылка.md" if names is NAMES["ru"] else "lane_link.md")
        write_markdown(md_blocks, a.ask or a.prompt, path)
        lk = link_m.get("link") or {}
        log(f"  words lane: {len(link_m.get('transcript') or [])} line(s), no frames"
            + ("" if lk.get("downloaded") else " — the video is not downloaded"))
        log(f"  wrote {path}")
        if a.json:
            print(json.dumps({"file": str(source), "dir": str(out_dir), "lane": str(path), "url": lk.get("url"),
                              "title": lk.get("title"), "site": lk.get("site"), "duration": (link_m.get("probe") or {}).get("duration"),
                              "captions": (link_m.get("transcript_meta") or {}).get("kind"),
                              "lines": len(link_m.get("transcript") or []), "downloaded": bool(lk.get("downloaded")),
                              "lecture": lk.get("lecture")}, ensure_ascii=False), flush=True)
        return
    try:
        start = parse_tcode(a.start)
        end = parse_tcode(a.end) if a.end else None
    except ValueError:
        sys.exit("--start / --end: expected mm:ss or h:mm:ss, like 0:05 or 1:02:30")
    if end is not None and end < start:  # equal is one anchor: one frame, or for sounds its whole second
        sys.exit("--end must not be before --start")
    at_times: List[float] = []
    if a.frames == "at":
        try:
            at_times = [parse_tcode(x.strip()) for x in (a.times or "").split(",") if x.strip()]
        except ValueError:
            sys.exit("--times: expected mm:ss or mm:ss.cc, comma-separated, like 3:12,7:40.5")
        if not at_times:
            sys.exit("--frames at needs --times mm:ss,mm:ss,…")
        if len(at_times) > AT_MAX:
            sys.exit(f"--times: up to {AT_MAX} moments a request — ask for the rest separately")
    log(f"video2llm {VERSION} — {source.name}")
    try:
        pr0 = tools.probe(source)
    except Exception:  # noqa: BLE001
        pr0 = {"duration": float(read_info_json(source).get("duration") or 0)}
    # 0.7.1: a sound file (mp3, wav, m4a …) has no picture: the audio lane — the words and the sounds, no frames,
    # whatever --frames asked for
    audio_only = "has_video" in pr0 and not p_has_video(pr0)
    if audio_only:
        if a.frames not in ("auto", "none"):
            log(f"  a sound file: no frames to cut (--frames {a.frames} ignored) — the lane is the words and the sounds")
        a.frames = "none"
    # --frames auto: a lecture / tutorial / review (by the site's info, before any cutting) → the lecture lane; otherwise
    # the overview is cut and the decision is taken again with the speech share (a local file has no site info)
    auto_hint: Optional[Dict[str, Any]] = None
    if a.frames == "auto":
        auto_hint = lecture_hint(source, pr0, None)
        a.frames = "none" if auto_hint["hint"] else "1"
        if auto_hint["hint"]:
            log(f"  auto: a lecture / tutorial ({'; '.join(auto_hint['reasons'])}) → the lecture lane; --frames 1 for the overview")
    if a.out:
        out_dir = Path(a.out).expanduser().resolve()
        if not writable_dir(out_dir):
            sys.exit(f"cannot write to --out {out_dir}")
    else:
        out_dir = source.with_name(f"{source.stem}_frames")
        if not writable_dir(out_dir):
            out_dir = fallback_out_dir(source)
            log(f"  [out] the video's folder is read-only — the lane goes to {out_dir}")
    cloud = cloud_synced(out_dir)
    if cloud:
        log(f"  [out] note: {out_dir} is synced by {cloud} — the frames and the audio will be uploaded there too; "
            "--out <local folder> keeps them on this computer")
    names = NAMES[a.names]
    m = prepare(source, out_dir, tools, names, frames_mode=a.frames, span=[start, end], transcript=not a.no_transcript,
                force_whisper=a.force_whisper, captions_prefer=captions_prefer,
                sounds=a.sounds, sound_threshold=a.sound_threshold, native=a.native, whisper_model=a.whisper_model,
                language=a.language, progress=progress, sounds_model=a.sounds_model, voice_rescue=a.voice_rescue,
                overview_fps=a.fps)

    if auto_hint is not None and a.frames == "1":  # the second look, with the words: a talking person most of the time
        auto_hint = lecture_hint(source, m["probe"], m.get("transcript"))
        if auto_hint["hint"]:
            a.frames = "none"
            log(f"  auto: a lecture / tutorial after all ({'; '.join(auto_hint['reasons'])}) → the lecture lane; --frames 1 for the overview")
        else:
            log("  auto: footage to watch → the overview (--frames none for the lecture lane)")
    dense = a.frames == "all"
    if a.frames in ("none", "at", "words"):
        blocks = []   # these lanes are built below, after the rerun command is known
    else:
        blocks = lane_blocks(m, a.label, start, a.max_frames, dense, end=end, fps_overview=a.fps)
    # frame-by-frame output is written beside the overview, never over it: lane.md stays the 1 fps lane
    d = m.get("dense") or {}
    if dense:
        suffix = (" all" if names is NAMES["ru"] else "_all") if d.get("whole") else span_tag(d.get("start", 0), d.get("end", 0), names)
    elif start > 0 or end is not None:  # a later part of the overview goes beside the first, not over it
        suffix = span_tag(start, end, names) if end is not None else (
            f" from {fmt_t(start)}" if names is NAMES["ru"] else f"_from_{fmt_t(start)}")
    else:
        suffix = ""
    rerun = Rerun(rerun_cmd(a, source, out_dir), source, a.agent_tool, auto=auto_hint is not None)
    if a.frames == "at":
        dur0 = m["probe"]["duration"]
        times = sorted({round(min(max(0.0, t), max(0.0, dur0 - 0.05)), 2) for t in at_times})  # time order, each moment once
        cut = cut_frames_at(tools, Path(m.get("work") or m["source"]), out_dir, names, times, progress)
        md_blocks = frames_at_blocks(m, a.label, cut, rerun)
        tagged = "_".join(fmt_t(t) for t in times[:2]) + (f"_+{len(times) - 2}" if len(times) > 2 else "")
        path = out_dir / (f"lane at {tagged}.md" if names is NAMES["ru"] else f"lane_at_{tagged}.md")
        write_markdown(md_blocks, a.ask or a.prompt, path)
        wh = jpeg_size(cut[0]["file"]) if cut else None
        log(f"  frames at {len(cut)} moment(s)" + (f" ≈ {claude_image_tokens(*wh) * len(cut):,} image tokens" if wh else ""))
        log(f"  wrote {path}")
        return
    if a.frames in ("none", "words") and not (a.sounds and not a.format):  # a sound request on this lane falls through to the sound check
        if a.frames == "words":  # 0.7.2: the words lane of a file that is here (a link downloaded, or any file)
            md_blocks = link_blocks(m, a.label, rerun)
            path = out_dir / ("lane ссылка.md" if names is NAMES["ru"] else "lane_link.md")
            write_markdown(md_blocks, a.ask or a.prompt, path)
            log(f"  words lane: {len(m.get('transcript') or [])} line(s), no frames")
            log(f"  wrote {path}")
            return
        md_blocks = lecture_blocks(m, a.label, rerun)
        if audio_only:
            path = out_dir / ("lane звук.md" if names is NAMES["ru"] else "lane_audio.md")
        else:
            path = out_dir / ("lane лекция.md" if names is NAMES["ru"] else "lane_lecture.md")
        write_markdown(md_blocks, a.ask or a.prompt, path)
        log(f"  {'audio' if audio_only else 'lecture'} lane: {len(m.get('transcript') or [])} line(s), no frames")
        log(f"  wrote {path}")
        return
    # a sound request alone (--sounds without --frames all / --format) only prints and saves the tags
    formats = set(a.format or ([] if a.sounds and not dense else ["md"]))
    if "all" in formats:
        formats = {"md", "pdf", "sheets", "openai", "anthropic", "gemini"}
    if a.ask:
        providers = [f for f in formats if f in DEFAULT_MODELS]
        if len(providers) != 1:
            sys.exit("--ask needs exactly one provider in --format (openai | anthropic | gemini)")
    prompt = a.ask or a.prompt
    written = []
    if "md" in formats:
        want = a.sheet_frames or 0
        grid = lane_sheet_plan(m, dense, enabled=not a.single_frames, want=want)
        sheet_tokens = None
        if want > 1:  # the frames are scaled on their sheets — the header's costs follow, for both lanes
            def _wh(fr: List[Dict[str, Any]]) -> Optional[List[int]]:
                return jpeg_size(fr[0]["file"]) if fr else None
            over_wh = _wh(m.get("frames") or [])
            dense_wh = _wh(d.get("frames") or []) or (
                [int(x) for x in fit_dims(m["probe"]["width"], m["probe"]["height"], DENSE_WIDTH)] if over_wh else None)
            sheet_tokens = {"over": sheet_frame_tokens(over_wh, lane_grid(*over_wh, want=want)) if over_wh else 0,
                            "dense": sheet_frame_tokens(dense_wh, lane_grid(*dense_wh, want=want)) if dense_wh else 0}
        md_blocks = lane_blocks(m, a.label, start, a.max_frames or (None if dense else LANE_MD_FRAMES), dense, end=end,
                                rerun=rerun, per_sheet=grid[0] * grid[1], fps_overview=a.fps, sheet_tokens=sheet_tokens)
        md_blocks = agent_sheets(md_blocks, grid)
        path = out_dir / f"lane{suffix}.md"; write_markdown(md_blocks, prompt, path); written.append(path)
    payloads: Dict[str, Dict[str, Any]] = {}
    for prov in ("openai", "anthropic", "gemini"):
        if prov in formats:
            model = a.model or DEFAULT_MODELS[prov]
            if prov == "gemini":
                body = to_gemini(native_blocks(m, a.label, a.part - 1) if a.native else blocks, prompt)
            else:
                body = to_openai(blocks, prompt, model) if prov == "openai" else to_anthropic(blocks, prompt, model)
            payloads[prov] = body
            path = out_dir / f"payload.{prov}{suffix}.json"
            path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8"); written.append(path)
    n_img = sum(1 for b in blocks if b["type"] == "image")
    if "md" in formats:
        imgs = [b for b in md_blocks if b["type"] == "image"]
        n_fr = sum(len(b.get("frames") or [1]) for b in imgs)
        log(f"  lane.md: {n_fr} frame(s)" + (f" on {len(imgs)} sheet(s)" if any(b.get("sheet") for b in imgs) else ""))
    extra = ((f", transcript {len(m['transcript'])} segment(s)" if m.get("transcript") else "")
             + (f", sound tags for {len(m['sound_spans'])} moment(s)" if m.get("sound_spans") else ""))
    if formats & {"openai", "anthropic", "gemini"}:
        log(f"  request bodies: {n_img} frame(s) each" + extra
            + ("; the Gemini payload carries the video file" if a.native and "gemini" in formats else ""))
    elif "md" in formats and extra:
        log("  " + extra[2:])
    for w in written:
        log(f"  wrote {w}")
    if "pdf" in formats:
        for r in write_pdf(m, a.label, start, end, a.max_frames, dense, prompt, out_dir,
                           a.pdf_font, max(1, a.pdf_pages), base=f"lane{suffix}", rerun=rerun, fps_overview=a.fps):
            log(f"  wrote {r['path']} — {r['span']}, {r['frames']} frames, {r['pages']} pages, {r['bytes'] / 1e6:.1f} MB")
    if "sheets" in formats:
        r = write_sheets(m, a.label, start, end, a.max_frames, dense, prompt, out_dir, a.pdf_font, a.sheet_grid,
                         a.sheets_per_message, folder_name=f"sheets{suffix}", rerun=rerun, fps_overview=a.fps)
        if r:
            log(f"  wrote {r['dir']}{os.sep} — {r['frames']} frames on {r['sheets']} sheet(s), {r['grid']}"
                + (f", in {r['batches']} message folders (message-01 …): attach one folder's files per message"
                   if r["batches"] > 1 else f" + \"{SHEET_NOTE}\": attach them all to one message"))
    if a.sounds and not dense and not a.format and p_has_audio(m):
        e_req = end
        s1 = m["probe"]["duration"] if e_req is None else min(m["probe"]["duration"], e_req + 1.0 / FRAME_FPS)
        r = sound_check(m, tools, out_dir, names, a.label, max(0.0, start), s1, a.sound_threshold, progress,
                        rerun=rerun, sheets=not a.single_frames)
        if r:
            log(f"  wrote {r['path']} — " + (f"{r['frames']} frames of {r['moments']} moment(s) where the sounds happen"
                                             + (f" ≈ {r['tokens']:,} image tokens" if r["tokens"] else "")
                                             if r["frames"] else "no frames: nothing besides speech to show"))
    if a.ask:
        prov = next(iter(payloads))
        log(f"  asking {prov} ({a.model or DEFAULT_MODELS[prov]}) …")
        print(ask(prov, payloads[prov], a.model or DEFAULT_MODELS[prov]))


if __name__ == "__main__":
    main()
    # CTranslate2 (faster-whisper) and ONNX Runtime both own native thread pools; on macOS their teardown
    # during interpreter shutdown can race and abort with "recursive_mutex lock failed" after all the work
    # is done (seen once in ~30 runs). Everything is written and flushed by now — leave without finalizers.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
