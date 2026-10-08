# Changelog

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
