# Changelog

## [0.7.1] — 2026-10-09

- **Sound files.** mp3, wav, m4a, aac, ogg, opus, flac, aiff — whatever ffmpeg reads — get the audio lane
  (`lane_audio.md`): the words with their time spans (captions beside the file, or Whisper), the sounds of a
  moment on request (`--sounds`), no frames at all; `--frames 1 | all | at` on a sound file is ignored with a
  note. A cover picture inside an mp3 is not taken for a video stream. The one question (the Guide / the
  Notes) is asked of a recording too — a podcast or a talk makes Notes without frames.
- **The lecture hint for local files.** A file has no site to ask, so its name counts as the title (`tutorial`,
  `урок`, `how-to` …) and speech 80 % of the time or more counts double: a 10-minute talking-head tutorial
  takes the lecture lane by itself; a film with dialogue (60–70 %) stays footage.
- **`--sounds` on the lecture lane.** A sound request on a video that `--frames auto` put on the lecture lane
  now brings the sound check, not the lecture lane again.

## [0.7.0] — 2026-10-08

- **Notes — `video2llm guide --spec guide.json`:** the Guide or the Notes of a video from the structure the model wrote
  (title, intro, `kind` guide | notes, sections with the moments whose frames illustrate them, captions): the frames are
  cut (1024 px, or reused from `--frames at`) and one page is written — the frames under a timecode chip, full screen on
  a click, a guide's steps numbered on the margin, a contents list from four sections, a **Download PDF** button that
  prints through the reader's browser (tight: two frames to a row, nothing breaks across pages). The page says «Made
  with V2L-IMT · Immersive Media Technologies» and links the terms. `--pdf` prints a file here for a program that wants
  one; by default there is no file and no extra folder — the page is the document.
- **`video2llm link DIR`** — the page's web address on the user's own Neocities site (one multipart request; the API key
  is pasted once into a page on 127.0.0.1 the script opens, never into a chat, and kept in `~/.config/video2llm/`;
  every address made is listed in `links.json` there). `--github`: the user's GitHub Pages instead (gh when logged in,
  a saved token, or the device flow — a link and a code, the sign-in in the browser).
- **The lecture lane ends with one question:** make the Guide / the Notes? — the model names the fitting form and builds
  nothing before the answer; in `--agent-tool` mode it calls the host's `video_guide`. The word «publishing» is gone:
  it is the user's link, nothing is announced anywhere.
- **`--frames auto` is the default:** the script decides by the material — the site's category, the title and
  description, the chapters, the length, the share of speech (a local file: the last two) — and says why; a lecture,
  a tutorial, a review gets the lecture lane, footage to watch the overview. The overview's rules tell the model to
  switch lanes itself (`--frames none`; the `video` tool's `what: lecture` in Deep Artisan) when the words show it is
  a lecture after all. The lecture lane's rules name what counts as material to keep (and what does not — a film, a
  clip, a vlog, a sketch, an ad, a stream).
- `agents/` is positioned as a section to add to the instructions an agent already has (AGENTS.md / CLAUDE.md / an
  existing SKILL.md), updated for links, the auto lane and Notes.
- `TERMS.md`: what the tool is responsible for when a document gets an address, and what the user is.

## [0.6.1] — 2026-10-08

- **A subtitle the site refuses no longer takes the video down.** YouTube answers `429` to too many caption
  requests (a pattern like `ru.*` used to match every machine-translated track); the download is now asked for
  exact codes only (the asked languages, `en`, the original `*-orig`) with `--ignore-errors`, and when yt-dlp
  still fails because of the subtitles, the video is fetched once more without them and Whisper transcribes as
  before. A yt-dlp that cannot start (a moved venv, a broken shebang) reports as such instead of a traceback.
- In agent-tool mode the lecture lane names the host's `video_guide` tool for the written guide, so the model
  does not build the page and the PDF by hand.

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
