<p align="center"><img src="assets/banner.jpg" alt="V2L-IMT — Video to LLM by Immersive Media Technologies" width="100%"></p>

# V2L-IMT — let an AI agent or a chat model watch a video

Video to LLM · one Python file · macOS, Windows, Linux · Immersive Media Technologies

Most models do not take video, and the ones that do take a small file. `video2llm.py` turns a video into
a **lane** a model can follow: frames tagged `[mm:ss]`, the words spoken at those moments placed between
them, and a header that tells the model what it is looking at, how to ask for more detail and what that
detail will cost. It works the way Gemini watches video: **an overview at one frame per second; where that
is not enough, every frame of just that moment — on request.**

It is the video pipeline of Deep Artisan, trimmed to one file that runs anywhere.

## How it behaves

**By default: one frame per second, and the speech.** `python3 video2llm.py clip.mp4` cuts the whole
video at 1 fps (768 px on the long side), transcribes the speech locally with Whisper and writes
`clip_frames/lane.md`: the frames in time order, each tagged `[mm:ss]`, with the words spoken under them.
The model reads that first; for most questions — what happens, who does what, who says what — it is
enough, and nothing more is asked of you.

**When the overview cannot answer, the model asks for every frame of one moment — and names the price.**
Something that happens between two frames (a fast gesture, a cut, a flicker, the exact frame a hand
touches something) is not visible at 1 fps. The model does not guess, and does not ask you for
timecodes: it picks the moment itself, from what it has seen and heard, and asks one question —
*"To answer this I need every frame from [00:12] to [00:14] — about 12,749 tokens. Shall I?"* After your
yes it runs the script itself with `--frames all --start 00:12 --end 00:14`, gets every frame of those
seconds (never thinned: 30 or 60 a second, 512 px) in `lane_00-12_00-14.md`, and answers from them. One
request covers up to 4 s (about 120 frames); a longer moment is asked for in consecutive pieces, each with
its price.

**Sounds other than speech — the same way.** Only speech is transcribed by default. When the answer
depends on what else is heard (music, a slam, laughter, a crowd, an engine), the model asks for the sounds
of that moment, with the cost — *"To answer this I need the sounds from [01:53] to [01:58] with the frames
where they happen — up to about 25,289 tokens. Shall I?"* After your yes it runs `--sounds --start 01:53
--end 01:58`: a local sound classifier tags what it hears — `[01:55–01:58] (sound: Laughter)` — and every
frame of the moments where a sound is found comes with the tags (`lane_sounds_01-53_01-58.md`), so the
model can say what makes the sound: who laughs, what slams. A sound request may cover any span, a whole
minute too; only the frames that come with it are capped like a frame-by-frame request. When nothing
besides speech is found, the model is told to say that no other sounds could be identified — not that
there were none.

**The price is known before the question.** The lane's header carries the cost of an overview frame, a
frame-by-frame frame and a second of them, computed from this video's real frame size by Anthropic's
formula for Claude (other models count differently; the exact figure comes only with the response), so the
model can name the tokens when it asks. The rest of the overview of a long video (it comes in parts of
2½ minutes) the model fetches by itself, without asking — only the extras cost a yes.

**You only say yes or no.** You never type a timecode and never run ffmpeg; neither does the model — it is
told to use nothing on the video but the script's commands: no ffmpeg of its own, no crops or zooms, no
reading of the tool's working files. A model that cannot run commands (a chat app) gives you the exact
command to run and reads what it writes.

## How the frames reach the model

A model reads images differently depending on where you talk to it, so the script writes the same frames
in four forms. All carry the same header, the same tags and the same words under the frames.

**Agents — `lane.md` with contact sheets (the default).** An agent with a file tool (Claude Code, Claude
Cowork, Codex CLI, Gemini CLI, Qwen Code, Cursor, Antigravity …) opens `lane.md`, which links one image per
three overview frames: a contact sheet — the three frames side by side, each at its own size, each with its
tag above it — and under the sheet the words and sounds of its frames, every line starting with the tag of
the frame it begins at: `[00:07] → «И он находится на первом плане…» (speech 00:07.5–00:11.6)`.
Frame-by-frame and sound lanes pack six frames to a sheet. A sheet stays under Claude's limits (1568 px on
the long side, 1568 tokens), so nothing is downscaled: the same pixels as the single frames, for Claude
the same tokens within a few per cent, and a third (a sixth) of the images to open — 50 reads instead of
149 for 2½ minutes, and the reads are most of an agent's time. GPT counts a sheet cheaper than three
frames; Gemini counts an overview sheet about a third dearer and a frame-by-frame sheet a third cheaper.
`--single-frames` goes back to one image per frame. Every lane also carries the exact commands for this
video, so the agent runs the next request itself.

**Claude and Gemini chats — `--format pdf`.** A chat app without a shell takes attachments. Claude reads
every page of a PDF as text and as an image (PDFs up to 100 pages), Gemini up to 1000 pages — so
`lane.pdf` carries the whole video in one file: the frames with their tags, the words between them, the
header on the first page. Gemini also takes the video file itself up to 5 minutes (`--native --format gemini`).

**ChatGPT and any other chat — `--format sheets`.** ChatGPT looks at images inside PDFs only on Enterprise,
and chats cap attachments (Claude 20 images per message, Gemini 10 files) while a minute of video is 60
frames. `sheets/` holds the contact sheets as plain images plus `about this video.txt` — the header and the
whole transcript, each line with its time span — in folders `message-01`, `message-02` … of 9 sheets each:
attach one folder per message (`--sheets-per-message` changes the count).

**An API key — `--format anthropic | openai | gemini`.** A ready request body with the frames inlined and
the header without the commands (a model behind an API cannot run them); `--ask "…"` sends it.

| Where you ask | What to make | How the frames get to the model | Checked by us |
|---|---|---|---|
| **Claude Code** (CLI, desktop, IDE) | `lane.md` (default) | the agent runs the script, reads `lane.md`, opens the contact sheets with its file tool (as images), runs the frame-by-frame and sound commands itself | Opus 5.5 ✅ (agent run, 07.10) |
| **Claude Cowork** | `lane.md` | the same, inside the folder you connected | not yet |
| **Qwen Code** | `lane.md` | the agent runs the script and opens the sheets; needs a vision-capable Qwen model | Qwen3.8 Max ✅ (agent run, 07.10) |
| **Codex CLI**, **Gemini CLI**, Cursor, Antigravity and other agents that read images from disk | `lane.md` | the same route: run the script, read `lane.md`, open the sheets | GPT-6 Astra ✅, Gemini 3.8 Flash ✅, Grok 4.7 ✅ (agent runs, 07.10) |
| **Claude** (claude.ai, desktop chat) | `--format pdf` | attach `lane.pdf`: Claude reads every page as text **and** image | not yet |
| **Gemini** (app / web) | `--format pdf` | attach `lane.pdf`; for clips up to 5 min Gemini also takes the video file itself | not yet |
| **ChatGPT** (Plus / Pro / Team) | `--format sheets` | attach the files of one `message-NN` folder per message | not yet |
| **Any chat** with image upload (Grok, Qwen Chat, …) | `--format sheets` | the same contact sheets + text | not yet |
| **An API key** (OpenAI, Anthropic, Gemini) | `--format anthropic` / `openai` / `gemini` | a ready request body with the frames inlined; `--ask "…"` sends it | ✅ |

**Agent runs, 07.10.2026** — five agents (Opus 5.5, GPT-6 Astra, Gemini 3.8 Flash, Qwen3.8 Max, Grok 4.7) each
got the video file and the one-line instruction from *Install*, in the Deep Artisan chat (an agent harness
with file and shell tools), and the same three questions about a 2½-minute lesson. All five ran the script
themselves, read the sheets and answered the first question from them (2–12 minutes, most of it opening
sheets). For "at which frame does she first touch the theatre" all five named the moment themselves, asked
with the cost, waited for the yes, ran the frame-by-frame command and answered with a frame number (`#896
[00:29.87]`, the first frame after a cut — or `#0`, for the first model in the video). For "what sounds
besides speech" all five asked first, ran the sound command and named the laughter at 01:55–01:58 and the boy
in the yellow T-shirt who laughs, from the frames that came with it. Nobody ran ffmpeg or cut frames of
their own (in an earlier round, before the rules, two of them did).

## In detail

- **A denser overview.** `--fps 2`, `3` or `4` adds the frames between the seconds to the same folder
  (`[00:12.25]`, `[00:12.50]`, `[00:12.75]` between `[00:12]` and `[00:13]`), at that many times the tokens of
  the 1 a second lane; the 1 a second lane keeps reading only the whole seconds, nothing is cut twice. The
  frame-by-frame and sound requests work the same in either density. A whole 200k context holds about 1:18
  of video at 4 a second, 1:44 at 3, 2:36 at 2, 5:12 at 1 (768 px frames, ≈ 448 tokens each for Claude).
- **For a program, not an agent.** Every `lane*.md` has a `lane*.json` twin — the same header, sheets and lines
  as data (image paths, the tags on each sheet, the lines under it) — for an app that puts the images into the
  model's message itself; `--agent-tool NAME` makes the header ask the model to call that tool
  (`{"video", "what": "next" | "frames" | "sounds", "start", "end"}`) instead of giving it commands.
  This is how Deep Artisan uses the script.
- **Fewer files for a chat that counts them.** `--sheet-frames 6` puts six frames on every sheet instead of the
  three (portrait) or four that fit at their own size: the frames are scaled so the sheet stays within Claude's
  limits (≈ 74 % of their size, 1505 tokens a sheet instead of 448 a frame), the header's costs follow. claude.ai
  takes 20 files a message and has an upload quota — Deep Artisan sends 20 such sheets (120 frames) a part there.
- **Anchors and frame numbers.** A frame-by-frame request is anchored on two overview frames by their tags,
  both included. The overview frame `[00:12]` is the first frame at or after 12.000 s, and the
  frame-by-frame lane tags that same frame `[00:12.00 #360]` — time and the frame's number in the video,
  counted from 0 — so an answer can name the exact frame. A fraction (`--end 0:12.5`) narrows a request.
- **Sounds.** A local AudioSet classifier (PANNs CNN14 as ONNX — no PyTorch) hears 3 s around the moment
  for context and tags 527 classes: good at music, applause, laughter, engines, animals; it does not name a
  song or a person. When it finds nothing besides speech, the output says so honestly — *no other sounds
  could be identified (quiet sounds under speech can be missed); speech in 100% of the moment; strongest
  other class <0.01*. The model writes that in the user's language; the tool's own text is English.
- **What it costs**, by Anthropic's published formula for Claude (⌈w/28⌉ × ⌈h/28⌉ tokens per image),
  computed from the real frame size:

  | 720×1280 video, 30 fps | Claude tokens (estimate) |
  |---|---|
  | one overview frame (432×768) | ≈ 448 |
  | one frame-by-frame frame (288×512) | ≈ 209 |
  | one second frame by frame | ≈ 6,270 — about 14× the same second in the overview |
  | 4 seconds frame by frame (121 frames) | ≈ 25,300 |

  Frame counts and sizes are exact; token figures are estimates — GPT and Gemini count images by their
  own rules, and the exact figure comes only with the response.

## Why this one

Compared with the projects closest to it, from their READMEs as of 2026-10-07:

| | V2L-IMT (this repo) | claude-real-video | watch-cli | mathiaschu/watch | peepshow | mcp-video-analyzer | llm-video-frames | Video2LLM (DiogoNeves) | Video-to-LLM-Context-Extractor |
|---|---|---|---|---|---|---|---|---|---|
| Form | one Python file (CLI) | CLI + MCP + skill + web | CLI + skill + MCP | Claude Code skill | CLI + plugins | MCP server | `llm` plugin | Python scripts | Electron app |
| Frames | 1 fps overview | scenes + dedup, ≥ 1 fps floor | 8 evenly spaced | auto, ≤ 2 fps, ≤ 100 | scenes + dedup, ≤ 40 | scenes + dedup, ≤ 60 | fixed fps | 10 fps, 20 max | intervals |
| **Every frame of a chosen moment** | **✅ on request, anchored on two overview frames, never thinned** | ❌ (dedup always on) | ❌ | ❌ (2 fps cap) | ❌ | partly (2–30 frame burst) | ❌ | ❌ | ❌ |
| **Token cost given to the model before it asks** | **✅ per frame, per second, per message** | in the docs | price per video (API) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Speech | local Whisper (Deep Artisan filters) | local Whisper / captions | hosted API (key) | captions / local Whisper | whisper.cpp / cloud | captions / Whisper / API | ❌ | ❌ | Google cloud |
| Sounds other than speech | **✅ local, free, on request per moment** | paid tier only | via Gemini API | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Chat outputs | **PDF (Claude / Gemini) + contact sheets (ChatGPT, any chat)** + API bodies | grid sheets, viewer | — | — | — | — | — | one image | PDF |
| Downloads at run time | **only from this repository** | Whisper models from their hosts | API calls | Whisper models from their hosts | Whisper models | models / APIs | — | OpenAI API | Google API |
| Video URLs (YouTube etc.) | ❌ local files | ✅ | ✅ | ✅ | direct URLs | ✅ | ❌ | ❌ | ❌ |
| Platforms named | macOS, Windows, Linux | macOS, Windows, Linux | macOS, Linux | macOS, Linux, Windows | macOS, Windows, Linux | macOS, Windows, Linux | macOS | — | Windows, macOS, Linux |

What is ours and verified: the overview + frame-by-frame technique with anchors; the cost in the header;
sounds on request with a local model and no PyTorch; PDF and contact-sheet outputs chosen for how each chat
really reads attachments; speech exactly as in Deep Artisan (Whisper with its voice filter, doubtful lines
dropped — compared line by line on real footage); every model downloaded from this repository only.
What others do and we do not: download from video sites, scene detection and de-duplication, an MCP server.

## Stack

One file, `video2llm.py`, on Python 3.9 or newer, with no server, no daemon and no PyTorch:

| Part | What it does | How it gets there |
|---|---|---|
| **ffmpeg / ffprobe** | probes the video, cuts the frames (1 fps at 768 px; every frame at 512 px), extracts the audio | system package: brew / winget / apt |
| **faster-whisper** (Whisper on CTranslate2, int8) | the speech: Whisper *small* by default, *base* for speed, with Deep Artisan's voice filter and its rule for doubtful lines | `pip install faster-whisper`; the model weights come from this repository's release |
| **PANNs CNN14** as ONNX on **ONNX Runtime** | the sounds other than speech, 527 AudioSet classes, 3 s of context around the moment | ONNX Runtime and numpy arrive with faster-whisper; the model comes from this repository's release |
| **Pillow** | the contact sheets (3 overview frames or 6 frame-by-frame ones to an image) | `pip install pillow`; without it the lane links one image per frame |
| **fpdf2** | `--format pdf` | optional, `pip install fpdf2` |
| the standard library | the lane, the manifest, the downloads with sha256 checks, the API request bodies | — |

Everything runs on the CPU; the frames and the lane are written next to the video, the models go to
`~/.cache/video2llm` (Windows: `%LOCALAPPDATA%\video2llm`) once. The only network use is that one-time
download from this repository; `--format … --ask` sends a request to the API whose key you give.

## Install

Everything below runs on macOS, Windows and Linux alike — the same script, the same files for the apps.
On Windows write `python` instead of `python3`, and `~` is `%USERPROFILE%`.

### 1. The script and what it needs

**macOS**

```sh
brew install ffmpeg
pip3 install faster-whisper pillow   # speech (also brings ONNX Runtime + numpy for the sounds) + contact sheets
pip3 install fpdf2                   # optional: --format pdf
```

**Windows** (PowerShell)

```powershell
winget install Python.Python.3.12 Gyan.FFmpeg    # if you have no Python yet; then open a new PowerShell
pip install faster-whisper pillow
pip install fpdf2               # optional: --format pdf
```

Windows on ARM (a Parallels VM on a Mac, a Snapdragon laptop): install the **x64** Python from
[python.org](https://www.python.org/downloads/windows/), not the ARM64 one — the speech engine
(CTranslate2, which faster-whisper uses) ships no ARM64 build for Windows, and the x64 one runs through
Windows' built-in emulation. Everything works; speech recognition is a few times slower than native.

**Linux**

```sh
sudo apt install ffmpeg python3-pip      # or your distribution's packages
pip3 install faster-whisper pillow
pip3 install fpdf2                       # optional: --format pdf
```

Then download `video2llm.py` from the [latest release](../../releases/latest) (or clone this repository),
put it anywhere and run `python3 video2llm.py video.mp4` (Windows: `python`). If ffmpeg is
not on PATH: `--ffmpeg PATH --ffprobe PATH` or `VIDEO2LLM_FFMPEG` / `VIDEO2LLM_FFPROBE`.

**What it downloads — once, and only from this repository's release `models-v1`** (checksum-verified,
into `~/.cache/video2llm`, on Windows `%LOCALAPPDATA%\video2llm`):

| File | When | Size |
|---|---|---|
| `faster-whisper-small.zip` — Whisper small (default) | the first transcript | 464 MB |
| `faster-whisper-base.zip` — Whisper base (`--whisper-model base`, faster) | only if you choose it | 141 MB |
| `panns_cnn14_dlmax.onnx` — the sound classifier | only the first time sounds are asked for | 312 MB |

Any other Whisper size: `--whisper-model /path/to/a/ctranslate2-whisper-folder`. The Python packages
(faster-whisper and its dependencies) take 219 MB; Pillow 15 MB, fpdf2 with fontTools 23 MB more. Without
Pillow the lane falls back to one image per frame.

### 2. Your app

An agent does not know the script exists until you tell it. The [`agents/`](agents/) folder holds one
instruction, [`agents/AGENTS.md`](agents/AGENTS.md) — run the script instead of opening the video, read
`lane.md`, open every sheet, ask with the cost before `--frames all` / `--sounds`, never ffmpeg, allow the
first run its time — plus the same text as a Claude skill and as a Cursor rule. Put it where your app reads it
(the table; [`agents/README.md`](agents/README.md) says the same per file) and replace `/path/to/video2llm.py`
with where you put the script. Chats need nothing installed — only the right format to attach.

| App | Do this |
|---|---|
| **Claude Code** (CLI, desktop, IDE) | the line `@agents/AGENTS.md` in the project's `CLAUDE.md` — or append the text of `agents/AGENTS.md` to `CLAUDE.md` / `~/.claude/CLAUDE.md` for every project |
| **Claude Code**, as a skill | put `agents/SKILL.md` and `video2llm.py` into `~/.claude/skills/video2llm/` — found whenever a video comes up, no path to edit |
| **Claude Cowork** | the same two files in a folder `video2llm`, zipped, at *Customize → Skills* in the Claude app — or say the line below in the chat; keep the video inside the folder you connected |
| **Claude** app / claude.ai (chat) | nothing to install: `python3 video2llm.py clip.mp4 --format pdf`, attach `lane.pdf` |
| **Deep Artisan** | nothing to install: attach the video — the app runs this script itself, puts the sheets into the model's message (`lane.json`) and executes the model's `video` requests (`--agent-tool`); the chip's toggle turns 1 a second into 2–4 (`--fps`); on the Anthropic subscription channel six frames to a sheet (`--sheet-frames 6`) |
| **Cursor** | `agents/.cursor/rules/video2llm.mdc` into the project's `.cursor/rules/` (Cursor reads `AGENTS.md` in the project root too) |
| **Codex CLI** | `agents/AGENTS.md` into the project root, or `~/.codex/AGENTS.md` |
| **Antigravity** | `agents/AGENTS.md` into the project root, or `~/.gemini/AGENTS.md` |
| **Gemini CLI** | the same `agents/AGENTS.md`, saved as `GEMINI.md` in the project root or `~/.gemini/GEMINI.md` |
| **Qwen Code** | the same `agents/AGENTS.md`, saved as `QWEN.md` in the project root or `~/.qwen/QWEN.md` |
| **Gemini** app / web | nothing to install: `--format pdf`, attach `lane.pdf` — or the video itself, up to 5 min |
| **ChatGPT** (desktop, web) | nothing to install: `--format sheets`, attach the files of one `message-NN` folder per message |
| **Any chat** with image upload | the same `--format sheets` |
| **An API key** | `--format anthropic` / `openai` / `gemini`, `--ask "…"` with the key in `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` |

Or just say it in the chat, once per conversation:

```
To watch a video, run: python3 /path/to/video2llm.py <video>   — then read the lane.md it writes and follow its rules.
```

Either way the agent runs the script itself, reads the sheets, fetches the rest of the overview itself and
asks you before anything that costs extra. The files add the two things an agent cannot learn from a lane
it has not made yet: not to open the video or run ffmpeg on its own, and to allow the first run its
10–20 s of speech recognition (the very first run on a machine also downloads the speech model) — every
later run on the same video is under a second.

## Use

```sh
python3 video2llm.py clip.mp4                                        # overview + speech → clip_frames/lane.md (sheets of 3 frames)
python3 video2llm.py clip.mp4 --frames all --start 0:12 --end 0:14   # every frame from 0:12 to 0:14
python3 video2llm.py clip.mp4 --sounds --start 2:14 --end 2:22       # sounds of that moment + the frames where they happen
python3 video2llm.py clip.mp4 --format pdf                           # lane.pdf for Claude / Gemini chats
python3 video2llm.py clip.mp4 --format sheets                        # contact sheets + text for ChatGPT / any chat
python3 video2llm.py clip.mp4 --format anthropic --ask "What goes wrong at 0:05?"   # key from ANTHROPIC_API_KEY
python3 video2llm.py long.mp4 --start 2:30                           # the next part of a long video → lane_from_02-30.md
python3 video2llm.py clip.mp4 --single-frames                        # lane.md with one image per frame
python3 video2llm.py clip.mp4 --fps 3                                # a denser overview: 3 frames a second, added to the same folder
python3 video2llm.py clip.mp4 --agent-tool video                     # for a harness with its own tool: the header says "call video with {…}"
python3 video2llm.py clip.mp4 --sheet-frames 6                       # six frames to a sheet (scaled to fit) — half the files for a chat that counts them
python3 video2llm.py clip.mp4 --native --format gemini               # the video itself, fitted into 19 MB
```

```
video2llm 0.3.0 — IMG_2056.MOV
  [probe] 100% — 720×1280, 02:29, 30.00 fps, audio
  frames: 1 per second over the whole video, 768px on the long side — frame by frame for a moment: --frames all --start mm:ss --end mm:ss (up to 4.0 s)
  [frames] 100% — 149 frames at 1/s
  [transcript] 100% — 24 segments (24 main), language ru (detected)
  [done] 100% — /…/IMG_2056_frames
  lane.md: 149 frame(s) on 50 sheet(s)
  transcript 24 segment(s)
  wrote /…/IMG_2056_frames/lane.md
```

A frame-by-frame segment, a later part and sound tags are written **beside** the overview, never over it:
`lane_00-12_00-14.md`, `lane_00-12_00-14.pdf`, `sheets_00-12_00-14/`, `lane_from_00-30.md`,
`sounds_02-14_02-22.txt` (plain ASCII names — nothing to escape in a shell).
`--format` is repeatable or `all`; `--help` lists everything, grouped.

**Timing** (Apple M-series laptop, CPU): a 2½-minute clip — 11–12 s for the overview and speech once the
Whisper model is cached; a frame-by-frame request — about a second; a sound request — under a second.
Whisper takes most of the time; a video already processed is not transcribed again (its manifest is kept).

## Where the files go

Next to the video, on every system: `~/Downloads/clip.mp4` → `~/Downloads/clip_frames/`; `--out DIR`
puts them anywhere. If the video's folder cannot be written (a network share, a mounted image, Windows
*Controlled folder access*), the lane goes to your Videos folder — `~/Movies/video2llm/` on macOS,
`Videos\video2llm\` on Windows, `~/Videos/video2llm/` on Linux — and the script says where. In a folder
synced by iCloud Drive, OneDrive, Dropbox, Google Drive or Yandex Disk it works, but warns that the frames
will be uploaded too. On macOS, if the terminal was refused access to a folder, the script says where to
allow it. Agents read the frames only where they are allowed to — keep the lane inside the folder you
opened or connected.

```
clip_frames/
  frames_1fps_768px/                    00-00.jpg 00-01.jpg …          the overview
    sheets/                             sheet_00-00_00-02.jpg …        3 frames to an image, linked from lane.md
  frames_all_00-12_00-14_30fps_512px/   00-12.00.jpg 00-12.03.jpg …    frame by frame (+ sheets/, 6 to an image)
  audio.wav  transcript.txt             speech (16 kHz) and its lines
  sounds_02-14_02-22.txt                sound tags of a moment (on request)
  lane_sounds_02-14_02-22.md            those tags + every frame where each sound happens
  lane.md  lane.pdf  sheets/            the overview: for agents / Claude & Gemini chats / any chat
  lane_00-12_00-14.md …                 a frame-by-frame segment, beside the overview
  payload.<provider>.json               request bodies (--format openai | anthropic | gemini)
  video2llm.json                        manifest; a changed video is cut again
```

## What the model is told

The first text block of every lane, PDF or sheet set — a short list of rules (a 2½-minute clip; the
commands carry this video's real paths, the home folder written as `~`):

> [Video 1: video "IMG_2056.MOV", 02:29, 720×1280, with audio (automatic speech transcript — it can mishear
> words, read it with the frames; only speech is transcribed — other sounds are not listed).
> The images are contact sheets of consecutive frames of one moving picture, in time order — 3 frames to a
> sheet, … each with its tag [mm:ss] above it … Under each sheet are the words and sounds of its frames:
> each line starts with the tag of the frame it begins at, then what is heard and its own time span.
> Rules:
> 1. The video is shown at 1 frame per second — the normal view, enough for what happens and who says what.
>    This lane carries 00:00–02:29. Open every sheet the question concerns — all of them for a question
>    about the whole video, in order, not a sample — and never describe from the transcript alone what you
>    have not seen in the frames. (In a longer video: Frames 02:30–… are not here: when your answer
>    concerns them — a question about the whole video does — get them yourself first, without asking the
>    user — run `~/…/video2llm.py ~/Downloads/long.mp4 --start 02:30` …)
> 2. You choose the moments, by what you see and hear in the lane — the user does not know the timecodes.
>    Ask the user only for a yes or a no, with the cost.
> 3. Frame by frame — when one frame per second cannot answer (something happens between two frames: fast
>    motion, a gesture, a flicker, a cut), do not guess; ask, e.g. "To answer this I need every frame from
>    [00:12] to [00:14] — about 12,749 tokens. Shall I?" … After the yes, run `… --frames all --start 00:12
>    --end 00:14` — it writes lane_00-12_00-14.md; look at its images.
> 4. Sounds other than speech (music, a slam, laughter, a crowd, an engine) are not in the lane. When the
>    answer depends on them, ask, e.g. "To answer this I need the sounds from [00:40] to [00:44] with the
>    frames where they happen — up to about 25,289 tokens. Shall I?" One sound request may cover any span,
>    a whole minute too; the 4.0 s limit is only on the frames that come with it … After the yes, run
>    `… --sounds --start 00:40 --end 00:44` … If nothing besides speech is found, tell the user that no
>    other sounds could be identified there — not that there were none.
> 5. Ask before every request of rules 3 and 4 — a second or follow-up one too, and also when the user's
>    question is itself about a sound or an exact moment: the yes is for the tokens. The rest of the
>    overview (rule 1) needs no yes.
> 6. Use only these commands for this video. Do not search for or open the video file, do not run ffmpeg,
>    ffprobe or any other tool on it, do not read video2llm.json, and do not crop, zoom or combine frames
>    yourself — open only the images the lanes link to. If a detail is too small to tell, say so.
> 7. If you cannot run commands, give the user the command and ask for what it writes.
> Cost, by Anthropic's formula for Claude (…): an overview frame ≈ 448 tokens, a frame-by-frame one ≈ 209
> (512px on the long side), a second frame by frame ≈ 6,270 — 14× the same second in the overview; the
> frames here ≈ 66,752.]

Each rule answers something models did in our tests: one described the second half of a video from the
transcript; one asked the user which seconds to look at; two cut frames with their own ffmpeg, cropped and
built spectrograms; several ran a second paid request without asking. Where no speech was recognised for a
while, the transcript says so — `[01:18–02:29] (no clear speech recognised — voices may still be heard;
sound tags can be asked for)` — so a model does not read the gap as missing material.

The frame-by-frame and sound lanes carry their own short rules (look at the motion; say what makes a sound
only from what the frames show; ask before the next request; nothing but these commands). The PDF and the
sheets carry the same header worded for pages and sheets — the next part comes as the next message's files
when the model cannot run commands — and ask the model to say so if it sees only text or if some sheets did
not come through. The API request bodies carry the header without the commands (a model behind an API cannot
run them).

## Limits worth knowing

- Local video files only — no downloading from video sites.
- 150 overview frames (2½ minutes, 50 sheets, ≈ 67,000 Claude tokens) per lane; a longer video comes in
  parts, and the model fetches the next one itself with `--start` — the header names the price of a part.
  A 10-minute video watched whole is four parts, ≈ 270,000 tokens: the overview at 768 px is the deliberate
  default (it reads faces and text), not the cheapest view. API request bodies carry 30 frames each. PDF and
  sheets take the whole video.
- Speech is recognised exactly as in Deep Artisan, with Whisper small: children's or unclear speech gets
  misheard words, and speech under noise may be left out (the header tells the model to read the
  transcript together with the frames). On unclear speech Whisper picks one of several readings; the
  random seed is fixed, so a video gives the same transcript on every run on the same machine (another
  machine can differ). A larger CTranslate2 Whisper model can be passed with `--whisper-model PATH`. When
  Whisper is unsure of the language on a few seconds of audio, the system's UI language is used;
  `--language ru` overrides both.
- Sound tags are AudioSet classes (527, English): good at music, applause, laughter, engines, animals;
  they do not name a song or a person. `--sound-threshold` trades recall for precision.
- ChatGPT reads images inside PDFs only on Enterprise — elsewhere use `--format sheets`.
- Token figures in the header are estimates for Claude.

## Platforms

macOS, Windows and Linux — the same code; paths, fonts, the Videos folder and cloud-sync detection are
handled per system. Verified by us:

- **macOS** (Apple Silicon; Python 3.9 and 3.14) — everything, end to end.
- **Linux** (a clean container, Python 3.13) — overview, frame by frame, lane.md, contact sheets, API
  bodies, the fallback to `~/Videos/video2llm` when the video's folder is read-only, cloud-folder
  detection. Speech and PDF were not run there (packages not installed in that container).
- **Windows 11** (a Parallels VM on Apple Silicon — Windows on ARM running the x64 Python 3.12 through
  Windows' own emulation; ffmpeg 9 from gyan.dev) — everything, end to end: the tests, the speech model
  downloaded and checksum-verified into `%LOCALAPPDATA%\video2llm`, a 2½-minute clip (149 frames on 50
  sheets, 25 speech segments, language detected), frame by frame (61 frames, 4 s), sounds (the sound model
  downloaded; laughter found at 01:55–01:58 with 97 frames, 9 s), the PDF (51 pages) and the contact sheets;
  a second run on the same video 1 s. Only speech recognition is slower under the emulation: a new
  2½-minute video takes about 2 minutes there instead of 10–20 s (see Install).

## License

[IMT Non-Commercial License](LICENSE) — the same as every Immersive Media Technologies project:

1. non-commercial use only — commercial or monetised use needs a written license from Immersive Media
   Technologies;
2. attribution is mandatory, visible to users, in the form
   `V2L-IMT by Immersive Media Technologies — https://github.com/Immersive-Media-Technologies`;
3. no warranty.

Notices for the models and tools it downloads: [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Release history

Every version is on the [Releases](../../releases) page.

- v0.3.0 — `--sheet-frames N`: more frames to a sheet, scaled to Claude's limits — half the files for a chat
  that counts them (claude.ai); the header's costs follow.
- v0.2.0 — a denser overview on request (`--fps 2 | 3 | 4`, the frames join the same folder), `lane.json`
  beside every `lane.md` for a program that builds the model's message itself, `--agent-tool NAME` for a
  harness with its own tool (how Deep Artisan runs the script).
- v0.1.0 — first public version: overview + frame-by-frame on request with anchors and cost, sounds on
  request, lane.md / PDF / contact sheets / API bodies, models in this repository's release `models-v1`.
