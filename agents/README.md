# agents/ — tell your agent about video2llm

One instruction, in the three forms agents read. The text is the same in all of them: run the script instead of
opening the video, read `lane.md`, open every sheet, ask with the cost before `--frames all` / `--sounds`, never
ffmpeg, allow the first run its time. Replace `/path/to/video2llm.py` with where you put the script.

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
