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
from typing import Any, Callable, Dict, List, Optional

VERSION = "0.3.0"

# ── constants (same numbers as the Deep Artisan pipeline) ────────────────────────────────────────
NATIVE_LIMIT_BYTES = 19 * 1024 * 1024  # Gemini request limit 20 MB minus headroom
AUDIO_KBPS = 64
FRAME_WIDTH = 768   # overview frames fit in 768×768 (long side 768): ≈448 Claude tokens in either orientation
FRAME_FPS = 1
OVERVIEW_FPS_MAX = 4  # --fps: a denser overview (2–4 a second) adds frames to the same folder, [00:12.25] between [00:12] and [00:13]
DENSE_MAX_FRAMES = 120  # frame-by-frame: every frame of up to 120/fps s per request (4 s at 30 fps), one message
DENSE_MAX_FPS = 100.0   # frame names carry 1/100 s
DENSE_WIDTH = 512   # frame-by-frame frames fit in 512×512
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
        "native": "native",
        "work": "work-1080p.mp4",
        "audio": "audio.wav",
    },
    "ru": {
        "sparse": "кадры 1 в сек ({w}px)",
        "dense": "все кадры{span} ({fps} в сек, {w}px)",
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
                 ":stream_side_data=rotation",
                 "-of", "json", str(path)])
        if r.returncode != 0:
            sys.exit(f"ffprobe failed: {r.stderr.strip()[:300]}")
        j = json.loads(r.stdout or "{}")
        streams = j.get("streams") or []
        v = next((s for s in streams if s.get("codec_type") == "video"), {})
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
            span: Optional[List[Optional[float]]] = None, transcript: bool,
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
        if m.get("source_mtime") != mtime or (m.get("probe") or {}).get("bytes", source.stat().st_size) != source.stat().st_size:
            stale_cleanup(out_dir)  # a different video under the same name: cut everything again
            m = {}

    progress("probe", 0)
    p = m.get("probe") or tools.probe(source)
    dur = max(0.5, p["duration"])
    progress("probe", 100, f"{p['width']}×{p['height']}, {tcode(dur)}, {p['fps']:.2f} fps, {'audio' if p['has_audio'] else 'no audio'}")

    # 2) which frames: --frames 1 → the 1 fps lane over the whole video; --frames all → every frame of a span
    dense: Optional[Dict[str, Any]] = m.get("dense")
    sparse_dir = out_dir / names["sparse"].format(w=FRAME_WIDTH)
    frames = list_frames(sparse_dir)
    if frames_mode == "all":
        sp = dense_span(p, (span or [0.0, None])[0] or 0.0, (span or [0.0, None])[1])
        rate = f"{sp['fps']:.2f}".rstrip("0").rstrip(".")
        log(f"  frames: frame by frame — every frame from {tcode(sp['start'])} to {tcode(sp['end'])}, both included, "
            f"at {rate} per second" + ("" if sp["every"] else f" (of {p['fps']:.0f})") + f", {DENSE_WIDTH}px on the long side")
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
    if frames_mode != "all":
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

    def __init__(self, cmd: str, video: Path, tool: Optional[str] = None):
        self.cmd, self.video, self.tool = cmd, video, tool

    def _call(self, what: str, start: str, end: Optional[str] = None) -> str:
        args = {"video": str(self.video), "what": what, "start": start}
        if end is not None:
            args["end"] = end
        return f"call {self.tool} with {json.dumps(args, ensure_ascii=False)}"

    def frames(self, start: str = "mm:ss", end: str = "mm:ss") -> str:
        return self._call("frames", start, end) if self.tool else f"run {self.cmd} --frames all --start {start} --end {end}"

    def sounds(self, start: str = "mm:ss", end: str = "mm:ss") -> str:
        return self._call("sounds", start, end) if self.tool else f"run {self.cmd} --sounds --start {start} --end {end}"

    def next(self, at: float) -> str:
        return self._call("next", tcode(at)) if self.tool else f"run {self.cmd} --start {tcode(at)}"

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
                     "yourself — open only the images the lanes link to. If a detail is too small to tell, say so.")
        rules.append(rerun.cannot())
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
        head = (f"[{label}: video \"{Path(m['source']).name}\" — sound check of {tcode(s0)}–{tcode(s1)}: "
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
def main(argv: Optional[List[str]] = None) -> None:
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
                                        "  ANTHROPIC_API_KEY=… video2llm clip.mp4 --format anthropic --ask \"Describe the camera moves\"\n")
    ap.add_argument("video", help="video file (mp4, mov, m4v, webm, mpeg, avi, 3gp …)")
    g = ap.add_argument_group("which frames")
    g.add_argument("--frames", choices=["1", "all"], default="1",
                   help="1 = one frame per second over the whole video, 768px (default) — what happens, who says what; "
                        f"all = frame by frame, {DENSE_WIDTH}px — every frame from --start to --end, both included (flicker, "
                        f"glitches, fast motion); up to {DENSE_MAX_FRAMES} frames per request (4 s at 30 fps, 2 s at 60), "
                        "never thinned — a longer span is refused with the pieces to ask for")
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
    g = ap.add_argument_group("other")
    g.add_argument("--out", help="output folder (default: <video>_frames next to the video)")
    g.add_argument("--names", choices=["en", "ru"], default="en", help="language of the folder names (ru = as in Deep Artisan)")
    g.add_argument("--agent-tool", metavar="NAME",
                   help="the lane is for an agent harness that runs this script itself: the header tells the model to call "
                        "the tool NAME with {video, what: next | frames | sounds, start, end} instead of giving it commands")
    g.add_argument("--ffmpeg", help="path to ffmpeg"); g.add_argument("--ffprobe", help="path to ffprobe")
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

    source = Path(a.video).expanduser().resolve()
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
    try:
        start = parse_tcode(a.start)
        end = parse_tcode(a.end) if a.end else None
    except ValueError:
        sys.exit("--start / --end: expected mm:ss or h:mm:ss, like 0:05 or 1:02:30")
    if end is not None and end < start:  # equal is one anchor: one frame, or for sounds its whole second
        sys.exit("--end must not be before --start")
    log(f"video2llm {VERSION} — {source.name}")
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
    tools = Tools(find_tool("ffmpeg", a.ffmpeg), find_tool("ffprobe", a.ffprobe))
    progress = make_progress(a.quiet)
    m = prepare(source, out_dir, tools, names, frames_mode=a.frames, span=[start, end], transcript=not a.no_transcript,
                sounds=a.sounds, sound_threshold=a.sound_threshold, native=a.native, whisper_model=a.whisper_model,
                language=a.language, progress=progress, sounds_model=a.sounds_model, voice_rescue=a.voice_rescue,
                overview_fps=a.fps)

    dense = a.frames == "all"
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
    cmd = (("& " if os.name == "nt" else "")  # PowerShell runs a quoted path only with the call operator
           + f"{shell_arg(sys.executable)} {shell_arg(Path(__file__).resolve())} {shell_arg(source)}"
           + (f" --out {shell_arg(out_dir)}" if a.out else "")
           # the settings that shape the files: a follow-up run without them would cut into other folders
           + (f" --names {a.names}" if a.names != "en" else "") + (" --single-frames" if a.single_frames else "")
           + (f" --sheet-frames {a.sheet_frames}" if a.sheet_frames else "")
           + (f" --fps {a.fps}" if a.fps != FRAME_FPS else "") + (f" --max-frames {a.max_frames}" if a.max_frames else "")
           + (f" --language {a.language}" if a.language != "auto" else "")
           + (f" --whisper-model {shell_arg(a.whisper_model)}" if a.whisper_model != "small" else "")
           + (f" --sound-threshold {a.sound_threshold:g}" if a.sound_threshold != 0.15 else "")
           + (f" --ffmpeg {shell_arg(a.ffmpeg)}" if a.ffmpeg else "") + (f" --ffprobe {shell_arg(a.ffprobe)}" if a.ffprobe else "")
           + (f" --sounds-model {shell_arg(a.sounds_model)}" if a.sounds_model else ""))
    rerun = Rerun(cmd, source, a.agent_tool)
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
