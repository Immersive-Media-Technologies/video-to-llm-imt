# Changelog

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
