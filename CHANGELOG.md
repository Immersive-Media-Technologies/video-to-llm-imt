# Changelog

## [0.6.0] — 2026-10-08

- **The lecture lane — `--frames none`.** A lecture, a tutorial, an instruction is a talking person: a stream of
  frames is wasted tokens. This lane carries no frames — the whole transcript (the site's captions when there
  are any) with the time spans and the site's chapters, and the rules: read it, decide where the SCREEN matters
  (a slide, a diagram, an interface, code, text on screen, a demonstration — or simply what is being talked
  about, clearer seen than described) and ask for single frames at exactly those moments, any number, 3 or 30;
  then offer the user a written guide with those frames as illustrations.
- **Single frames by time — `--frames at --times 3:12,7:40.5`:** one frame per moment, 1024px on the long side
  (readable text), in `frames_at_1024px/`, `lane_at_*.md` + `.json` with the words spoken around each frame;
  up to 40 a request. The normal lane mentions it for a few exact moments (cheaper than frame by frame).
- **`--download-only --json`** prints `{file, title, duration, language, captions, lecture: {hint, score, reasons}}` —
  the lecture hint says whether the video looks like a lecture / tutorial (the site's category, title and
  description, chapters, length, how much of the time is speech): a suggestion, never a switch.

## [0.5.1] — 2026-10-08

- **Which captions:** `--captions LANGS` (default: this computer's UI language) names the subtitle languages to
  fetch and prefer; the video's own language and ru/en are always taken too. yt-dlp's `info.json` beside the
  file says which tracks are manual and what the video's language is, so the choice is: a manual track in the
  video's language, a manual one in a preferred language, the automatic original (`*-orig`), and the site's
  machine translation into a preferred language last — it is the least faithful. The header says whether the
  captions were written by people or are automatic. No captions at all → Whisper as before.

## [0.5.0] — 2026-10-08

- **The site's captions stand in for Whisper.** With a URL, yt-dlp also saves the subtitles beside the file
  (`<name>.<lang>.vtt`: manual ones in the asked language / ru / en, and the automatic track in the original
  language — `*-orig`); the lane takes them instead of running Whisper — nothing to transcribe, and the words
  are the site's own. YouTube's rolling automatic captions (every cue repeating the previous line) come out
  once each, at the time a line first appears. The same for **any `.vtt` / `.srt` beside a local video**
  (`clip.ru.srt`, `clip.srt`): it is used; `--language` picks the file; `--force-whisper` ignores them.
  The header says the words are captions.
- **`--download-only`** — with a URL: download the video and its captions, print the file's path and stop
  (for a program that runs the rest itself — Deep Artisan).
- yt-dlp: exact subtitle codes instead of patterns (`ru.*` also matched translated tracks and a dozen requests
  got a 429), `--ignore-errors` so a failed subtitle does not take the video down; a yt-dlp that cannot be run
  (a moved venv) is reported in one line.

## [0.4.1] — 2026-10-08

- `--yt-dlp` / `VIDEO2LLM_YT_DLP` also take a command line (`"py -m yt_dlp"`, a venv's Python with the module).
- Tests: the stand-in yt-dlp is run as a command line — on Windows a `.bat` let `cmd` read the `<` of
  `[height<=1080]` as a redirection (CI, windows-latest).

## [0.4.0] — 2026-10-08

- **A video by URL** — `video2llm https://youtu.be/…`: YouTube, Vimeo, a direct link, anything
  [yt-dlp](https://github.com/yt-dlp/yt-dlp) knows (optional: `pip install yt-dlp` / `brew install yt-dlp` /
  `winget install yt-dlp`, or `--yt-dlp PATH`). One video (no playlist), the best streams up to 1080p merged into
  an mp4 by ffmpeg, downloaded once into `<Videos>/video2llm/downloads/`; the lane goes next to that file and the
  rerun commands name the file, not the URL. The same URL again is found by its id and not fetched twice.
- **`--agent-tool`: the header names the frame files** (`…/frames_1fps_768px/mm-ss.jpg`) — an agent asked for one
  frame as a file reads it from there instead of searching the disk (a live case: two minutes of `find`).

## [0.3.0] — 2026-10-08

- **`--sheet-frames N`** — exactly N frames (2–12) to a contact sheet in `lane.md`: more than fit at their own
  size are scaled so the sheet stays within Claude's limits (6 overview frames ≈ 74 % of their size, 1505 tokens
  a sheet instead of 448 a frame), in the grid that scales them the least (3 × 2 portrait, 2 × 3 landscape); the
  costs in the header follow. For a chat that counts the files of a message — claude.ai takes 20 and has an
  upload quota; Deep Artisan uses it on the Anthropic subscription channel (20 sheets × 6 = 120 frames a part).

## [0.2.0] — 2026-10-08

- **A denser overview on request** — `--fps 2 | 3 | 4`: the frames between the seconds join the same folder
  (`[00:12.25]`), the 1 a second lane keeps reading only the whole seconds; the cost scales with the density.
- **lane.json beside every lane.md** — the lane as data (image paths, the tags of each sheet's frames, the
  lines under it) for a program that puts the images into the model's message itself.
- **`--agent-tool NAME`** — the header asks the model to call that tool (`{video, what, start, end}`) instead
  of giving it commands: for an agent harness that runs the script itself (Deep Artisan).
- The cost of the next part in the header follows the density; a rerun keeps `--fps` and `--max-frames`;
  stale `lane*.json` files are cleaned up with the `.md` ones.

## [0.1.0] — 2026-10-07

First public version.

- **Overview** — the whole video at 1 frame per second, 768 px on the long side (≈ 448 Claude tokens a
  frame in either orientation), with the speech placed between the frames, and a header that tells the
  model it is looking at one moving picture, how to ask for more detail and what that detail costs.
- **Frame by frame on request** — every frame between two overview frames named as anchors (both
  included), never thinned, up to ~120 frames per request (4 s at 30 fps); written beside the overview.
- **Sounds on request, with their source in view** — sounds other than speech for one moment, from a
  local AudioSet classifier (PANNs CNN14 as ONNX, no PyTorch), together with every frame where each sound
  happens; speech only by default.
- **Contact sheets for agents** — `lane.md` links 3 overview frames (6 frame by frame) to an image, each at
  its own size, with the words and sounds of each frame under the sheet: a third of the images to open at
  the same tokens. `--single-frames` for one image per frame.
- **A header of short rules** — the model picks the moments itself, fetches the rest of the overview
  itself, asks the user only yes or no with the cost before every frame-by-frame or sound request, and
  uses no other tool on the video.
- **Outputs for every route** — `lane.md` for agents (with the ready commands for this video),
  `lane.pdf` for Claude and Gemini chats, contact sheets + text for ChatGPT and any chat, request bodies
  for the OpenAI / Anthropic / Gemini APIs, the video itself fitted into a 19 MB Gemini request.
- **Ready instructions for agents** — `agents/`: the same instruction as `CLAUDE.md`, `AGENTS.md`,
  `GEMINI.md`, `QWEN.md`, a Cursor rule and a Claude skill (`SKILL.md`), with where each goes in the README.
- **Speech** — faster-whisper with the voice filter and the doubtful-line rule of Deep Artisan.
- **Downloads only from this repository** — Whisper small / base and the sound model come from the
  release `models-v1`, checksum-verified.
- **The same transcript on every run** — Whisper's temperature fallback samples; the random seed is fixed.
- macOS, Windows and Linux; end-to-end tests in `tests/`. Verified on macOS (Apple Silicon), Linux and
  Windows 11 (a Parallels VM on Apple Silicon, x64 Python under Windows' emulation).
