# agents/ — tell your agent about video2llm

An agent does not know the script exists until its instructions say so. This folder is **one section to add to
the instructions your agent already has** — not a file to replace them: paste it into the AGENTS.md / CLAUDE.md /
GEMINI.md your project reads, or into the SKILL.md of a skill you already keep; the same text is also here as a
skill of its own and as a Cursor rule. It says: a video file or a video link → run the script, read the lane it
chose (the overview, or the lecture lane for a tutorial), open every image, ask with the cost before `--frames
all` / `--sounds`, offer the Guide / Notes only for material to keep, never ffmpeg. Replace `/path/to/video2llm.py`
with where you put the script.

| File | For | Where it goes |
|---|---|---|
| [`AGENTS.md`](AGENTS.md) | **Codex CLI**, **Antigravity**, **Cursor** (reads AGENTS.md too) | project root, or `~/.codex/AGENTS.md` / `~/.gemini/AGENTS.md` |
| the same `AGENTS.md` | **Claude Code** (CLI, desktop, IDE) | the line `@agents/AGENTS.md` in the project's `CLAUDE.md` — or its text appended to `CLAUDE.md` / `~/.claude/CLAUDE.md` |
| the same `AGENTS.md`, saved as `GEMINI.md` | **Gemini CLI** | project root, or `~/.gemini/GEMINI.md` |
| the same `AGENTS.md`, saved as `QWEN.md` | **Qwen Code** | project root, or `~/.qwen/QWEN.md` |
| [`SKILL.md`](SKILL.md) | **Claude Code** and **Claude Cowork** as a skill | `~/.claude/skills/video2llm/` together with `video2llm.py` (Cowork: the folder zipped, *Customize → Skills*) |
| [`.cursor/rules/video2llm.mdc`](.cursor/rules/video2llm.mdc) | **Cursor** as a rule | the project's `.cursor/rules/` |

Chats (Claude, Gemini, ChatGPT, …) need none of this — they get `lane.pdf` or the contact sheets, see the
[README](../README.md#2-your-app).
