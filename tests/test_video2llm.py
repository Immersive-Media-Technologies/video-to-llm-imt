"""End-to-end checks of video2llm.py on a generated clip (runs on Linux, macOS and Windows in CI).

Needs ffmpeg on PATH. fpdf2 (PDF, contact sheets) and faster-whisper (speech) are optional: the
tests that need them are skipped without them.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "video2llm.py"
FFMPEG = shutil.which("ffmpeg")
sys.path.insert(0, str(ROOT))
import video2llm  # noqa: E402

pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")


def run(*args, env=None, ok=True):
    e = dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {}))
    r = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=e, timeout=900)
    if ok:
        assert r.returncode == 0, r.stderr[-3000:]
    return r


def has(module):
    import importlib.util
    return importlib.util.find_spec(module) is not None


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """6 s, 720×1280 at 30 fps with a tone — a space in the name on purpose."""
    d = tmp_path_factory.mktemp("clip")
    v = d / "portrait clip.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=30:duration=6",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(v)], check=True)
    return v


def frames_dir(clip):
    return clip.with_name(f"{clip.stem}_frames")


def test_overview_fits_the_long_side_and_tells_the_model(clip):
    run(clip, "--no-transcript")
    out = frames_dir(clip)
    jpgs = sorted((out / "frames_1fps_768px").glob("*.jpg"))
    assert [p.name for p in jpgs] == [f"00-0{i}.jpg" for i in range(6)]
    assert video2llm.jpeg_size(str(jpgs[0])) == [432, 768]
    head = (out / "lane.md").read_text(encoding="utf-8").split("\n\n")[0]
    assert "consecutive frames of one moving picture" in head
    assert "every frame from [00:12] to [00:14]" in head          # how to anchor a frame-by-frame request
    assert "≈ 448 tokens" in head                                  # the cost, from the real frame size
    assert "Shall I?" in head                                      # propose the moment and its cost, then wait
    assert "--frames all --start 00:12 --end 00:14" in head        # the ready command, run after the yes
    assert "the user does not know the timecodes" in head          # the model picks the moments, the user says yes
    assert "never describe from the transcript alone" in head
    assert "do not run ffmpeg" in head                             # only the tool's own commands


def test_lane_shows_frames_three_to_a_sheet(clip):
    out = frames_dir(clip)
    lane = (out / "lane.md").read_text(encoding="utf-8")
    links = [ln for ln in lane.splitlines() if ln.startswith("![")]
    assert links == ["![frames [00:00]–[00:02]](<frames_1fps_768px/sheets/sheet_00-00_00-02.jpg>)",
                     "![frames [00:03]–[00:05]](<frames_1fps_768px/sheets/sheet_00-03_00-05.jpg>)"]
    sheet = out / "frames_1fps_768px" / "sheets" / "sheet_00-00_00-02.jpg"
    w, h = video2llm.jpeg_size(str(sheet))
    assert (w, h) == (1304, 792)                                   # every frame at its own 432×768
    assert video2llm.claude_image_tokens(w, h) <= 1.06 * 3 * 448   # the same tokens as three frames, within 6 %
    run(clip, "--no-transcript", "--single-frames")
    lane = (out / "lane.md").read_text(encoding="utf-8")
    assert sum(1 for ln in lane.splitlines() if ln.startswith("![")) == 6


def test_a_short_lane_names_the_next_part_and_the_model_fetches_it(clip):
    out = frames_dir(clip)
    run(clip, "--no-transcript", "--max-frames", "3")
    lane = (out / "lane.md").read_text(encoding="utf-8")
    assert "This lane carries 00:00–00:03" in lane
    assert "Frames 00:03–00:06 are not here" in lane and "without asking the user" in lane
    assert "--start 00:03" in lane and "lane_from_00-03.md" in lane
    assert sum(1 for ln in lane.splitlines() if ln.startswith("![")) == 1
    run(clip, "--no-transcript", "--start", "0:03")
    nxt = (out / "lane_from_00-03.md").read_text(encoding="utf-8")
    assert "This lane carries 00:03–00:06" in nxt and "not here" not in nxt
    assert "sheet_00-03_00-05.jpg" in nxt
    run(clip, "--no-transcript")                                     # the whole lane again for the tests below


def test_a_denser_overview_adds_frames_to_the_same_folder(clip):
    """--fps 4: the frames between the seconds join the 1 a second frames in the same folder as mm-ss.cc.jpg;
    a 1 a second run afterwards reads only the whole seconds and cuts nothing again."""
    out = frames_dir(clip)
    folder = out / "frames_1fps_768px"
    run(clip, "--no-transcript", "--fps", "4")
    names = sorted(p.name for p in folder.glob("*.jpg"))
    assert "00-01.25.jpg" in names and "00-01.50.jpg" in names and "00-01.75.jpg" in names and "00-01.jpg" in names
    assert len(names) == 24
    lane = (out / "lane.md").read_text(encoding="utf-8")
    head = lane.split("\n\n")[0]
    assert "shown at 4 frames per second" in head and "[00:12.25]" not in head
    assert "(the frames between two seconds: [mm:ss.cc])" in head
    assert "--fps 4" in head                                       # the follow-up commands keep the density
    assert "[00:01.25]" in lane and "sheet_00-00_00-00.50.jpg" in lane
    m = json.loads((out / "video2llm.json").read_text(encoding="utf-8"))
    assert m["overview_fps"] == 4 and len(m["frames"]) == 24
    before = {p.name: p.stat().st_mtime for p in folder.glob("*.jpg")}
    run(clip, "--no-transcript")                                     # back to 1 a second: nothing is cut again
    assert {p.name: p.stat().st_mtime for p in folder.glob("*.jpg")} == before
    lane = (out / "lane.md").read_text(encoding="utf-8")
    assert "shown at 1 frame per second" in lane and "[00:01.25]" not in lane
    assert sum(1 for ln in lane.splitlines() if ln.startswith("![")) == 2


def test_lane_json_mirrors_lane_md_for_a_program(clip):
    out = frames_dir(clip)
    j = json.loads((out / "lane.json").read_text(encoding="utf-8"))
    assert j["lane"] == "lane.md" and j["blocks"][0]["role"] == "head"
    imgs = [b for b in j["blocks"] if b["type"] == "image"]
    assert [b["frames"] for b in imgs] == [["00:00", "00:01", "00:02"], ["00:03", "00:04", "00:05"]]
    assert all(b["sheet"] and Path(b["path"]).is_file() for b in imgs)


def test_sheet_frames_packs_more_frames_to_a_sheet_scaled_to_the_limits(clip):
    """--sheet-frames 6 (a chat that counts files): six frames to a sheet, scaled so the sheet stays within
    Claude's limits; the header's costs follow the scale."""
    out = frames_dir(clip)
    run(clip, "--no-transcript", "--sheet-frames", "6")
    j = json.loads((out / "lane.json").read_text(encoding="utf-8"))
    imgs = [b for b in j["blocks"] if b["type"] == "image"]
    assert [b["frames"] for b in imgs] == [["00:00", "00:01", "00:02", "00:03", "00:04", "00:05"]]
    w, h = video2llm.jpeg_size(imgs[0]["path"])
    assert max(w, h) <= 1568 and video2llm.claude_image_tokens(w, h) <= 1568
    assert (w, h) == (965, 1186)                                    # 3 × 2, every frame at 319×567 (74 %)
    head = j["blocks"][0]["text"]
    assert "6 frames to a sheet" in head
    assert video2llm.sheet_frame_tokens([432, 768], [3, 2]) == 251  # 1505 tokens a sheet ÷ 6, instead of 448 a frame
    assert video2llm.lane_grid(432, 768, want=6) == [3, 2] and video2llm.lane_grid(768, 432, want=6) == [2, 3]
    assert video2llm.lane_grid(432, 768, want=4) == [2, 2] and video2llm.sheet_fit(432, 768, 2, 2) > 0.9
    run(clip, "--no-transcript")                                     # the whole lane again for the tests below


def test_agent_tool_header_calls_the_tool_instead_of_commands(clip):
    out = frames_dir(clip)
    run(clip, "--no-transcript", "--agent-tool", "video", "--max-frames", "3")
    head = (out / "lane.md").read_text(encoding="utf-8").split("\n\n")[0]
    assert '"what": "frames", "start": "00:12", "end": "00:14"' in head and "call video with" in head
    assert '"what": "next", "start": "00:03"' in head
    assert "python" not in head.lower() and "--frames all" not in head
    assert "Use only the video tool" in head and "If the video tool is not available" in head
    run(clip, "--no-transcript")                                     # the whole lane again for the tests below


def test_words_and_sounds_sit_under_the_sheet_of_their_frame(clip):
    m = json.loads((frames_dir(clip) / "video2llm.json").read_text(encoding="utf-8"))
    m["transcript"] = [{"start": 1.4, "end": 2.6, "text": " hello "}]
    m["sound_spans"] = [{"start": 3.0, "end": 5.0, "threshold": 0.15, "note": "",
                         "events": [{"start": 4.2, "end": 4.9, "label": "Laughter"}]}]
    blocks = video2llm.agent_sheets(video2llm.lane_blocks(m, "Video 1", 0.0, None, False, rerun=video2llm.Rerun("RUN", clip), per_sheet=3), [3, 1])
    texts = [b["text"] for b in blocks if b["type"] == "text"]
    assert "(sound: …) for 00:03–00:05 only" in texts[0]
    assert "[00:01] → «hello» (speech 00:01.4–00:02.6)" in texts
    assert "[00:04] → Laughter (sound 00:04.2–00:04.9)" in texts
    assert [b["tag"] for b in blocks if b["type"] == "image"] == ["frames [00:00]–[00:02]", "frames [00:03]–[00:05]"]


def test_sound_lane_shows_every_frame_where_the_sound_happens(clip):
    out = frames_dir(clip)
    m = json.loads((out / "video2llm.json").read_text(encoding="utf-8"))
    m["transcript"] = []
    m["sound_spans"] = [{"start": 1.0, "end": 4.0, "threshold": 0.15, "note": "speech is heard in 0% of this moment",
                         "events": [{"start": 2.0, "end": 2.5, "label": "Slam"}]}]
    tools = video2llm.Tools(video2llm.find_tool("ffmpeg", None), video2llm.find_tool("ffprobe", None))
    r = video2llm.sound_check(m, tools, out, video2llm.NAMES["en"], "Video 1", 1.0, 4.0, 0.15,
                              video2llm.make_progress(True), rerun=video2llm.Rerun("RUN", clip))
    assert r["moments"] == 1 and r["frames"] == 34                  # 1.7–2.8 s: the slam ± 0.3 s, every frame
    lane = r["path"].read_text(encoding="utf-8")
    assert r["path"].name == "lane_sounds_00-01_00-03.md"
    assert "[00:02–00:02] (sound: Slam)" in lane or "[00:02] (sound: Slam)" in lane
    assert "Say what makes each sound only from what the frames show" in lane
    assert sum(1 for ln in lane.splitlines() if ln.startswith("![")) == 6   # 34 frames, 6 to a sheet
    assert "[frames 00:01.70 #51 – 00:02.80 #84]" in lane


def test_lane_grid_keeps_frames_whole_and_inside_the_limits():
    assert video2llm.lane_grid(432, 768) == [3, 1]                 # portrait overview
    assert video2llm.lane_grid(768, 432) == [1, 3]                 # landscape overview
    assert video2llm.lane_grid(288, 512) == [3, 2]                 # portrait frame by frame
    assert video2llm.lane_grid(512, 288)[0] * video2llm.lane_grid(512, 288)[1] == 6
    assert video2llm.lane_grid(768, 768) == [1, 1]                 # square: no sheet fits two whole frames


def test_frame_by_frame_is_every_frame_between_two_anchors(clip):
    out = frames_dir(clip)
    before = hashlib.sha256((out / "lane.md").read_bytes()).hexdigest()
    run(clip, "--frames", "all", "--start", "0:01", "--end", "0:02", "--no-transcript")
    dense = out / "frames_all_00-01_00-02_30fps_512px"
    names = sorted(p.name for p in dense.glob("*.jpg"))
    assert len(names) == 31 and names[0] == "00-01.00.jpg" and names[-1] == "00-02.00.jpg"
    assert video2llm.jpeg_size(str(dense / names[0])) == [288, 512]
    lane = (out / "lane_00-01_00-02.md").read_text(encoding="utf-8")
    assert sum(1 for ln in lane.splitlines() if ln.startswith("![")) == 6          # 31 frames, 6 to a sheet
    assert "6 frames to a sheet" in lane and "Ask before every such request" in lane
    assert hashlib.sha256((out / "lane.md").read_bytes()).hexdigest() == before   # the overview is untouched


def test_a_span_longer_than_one_request_is_refused_with_pieces(clip):
    r = run(clip, "--frames", "all", "--start", "0:00", "--end", "0:06", "--no-transcript", ok=False)
    assert r.returncode != 0 and "Ask in pieces" in r.stderr


def test_payloads(clip):
    run(clip, "--format", "anthropic", "--format", "openai", "--format", "gemini", "--no-transcript")
    out = frames_dir(clip)
    body = json.loads((out / "payload.anthropic.json").read_text(encoding="utf-8"))
    images = [c for c in body["messages"][0]["content"] if c["type"] == "image"]
    assert len(images) == 6
    assert (out / "payload.openai.json").is_file() and (out / "payload.gemini.json").is_file()


@pytest.mark.skipif(not has("PIL"), reason="Pillow is not installed")
def test_contact_sheets(clip):
    run(clip, "--format", "sheets", "--no-transcript")
    sheets = frames_dir(clip) / "sheets"
    assert sorted(p.name for p in sheets.glob("*.jpg")) == ["sheet-01.jpg", "sheet-02.jpg"]
    note = (sheets / "about this video.txt").read_text(encoding="utf-8")
    assert note.startswith("[Video 1:") and "contact sheets" in note


@pytest.mark.skipif(not has("fpdf"), reason="fpdf2 is not installed")
def test_pdf(clip):
    run(clip, "--format", "pdf", "--no-transcript")
    pdf = frames_dir(clip) / "lane.pdf"
    assert pdf.is_file() and pdf.read_bytes()[:5] == b"%PDF-"


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions and a non-root user")
def test_read_only_folder_falls_back_to_videos(clip, tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    v = ro / "clip.mp4"
    shutil.copy(clip, v)
    ro.chmod(0o555)
    home = tmp_path / "home"
    home.mkdir()
    try:
        r = run(v, "--no-transcript", env={"HOME": str(home), "XDG_VIDEOS_DIR": str(home / "Videos")})
    finally:
        ro.chmod(0o755)
    assert "read-only" in r.stderr
    assert list(home.rglob("lane.md"))


def test_videos_folder_exists_or_is_named():
    d = video2llm.videos_dir()
    assert isinstance(d, Path) and d.name in ("Videos", "Movies") or d.is_dir()


def test_cloud_folders_are_recognised():
    assert video2llm.cloud_synced(Path("/Users/me/Library/CloudStorage/OneDrive-Personal/x")) == "OneDrive"
    assert video2llm.cloud_synced(Path("/home/me/Dropbox/x")) == "Dropbox"
    assert video2llm.cloud_synced(Path("/home/me/Videos/x")) is None


def test_claude_token_formula():
    assert video2llm.claude_image_tokens(432, 768) == 448
    assert video2llm.claude_image_tokens(288, 512) == 209
    assert video2llm.claude_image_tokens(3840, 2160) <= 1568


@pytest.mark.skipif(not has("faster_whisper") or not os.environ.get("VIDEO2LLM_TEST_SPEECH"),
                    reason="speech test: faster-whisper and VIDEO2LLM_TEST_SPEECH=1")
def test_speech_model_comes_from_this_project(clip, tmp_path):
    cache = tmp_path / "cache"
    r = run(clip, "--whisper-model", "base", "--out", tmp_path / "out",
            env={"XDG_CACHE_HOME": str(cache), "LOCALAPPDATA": str(cache)})
    assert "download failed" not in r.stderr, r.stderr[-2000:]
    assert any(p.name == "model.bin" for p in cache.rglob("model.bin"))
    assert (tmp_path / "out" / "transcript.txt").is_file()


def test_overview_frames_sit_on_whole_seconds(tmp_path):
    """Frame n of a clip whose luma is n: the overview frame [00:0k] must be frame 30·k, not 30·k + 14."""
    v = tmp_path / "index.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=black:s=64x64:r=30:d=4,format=gray,geq=lum='N*2'",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(v)], check=True)
    run(v, "--no-transcript")
    PIL = pytest.importorskip("PIL.Image")
    jpgs = sorted((tmp_path / "index_frames" / "frames_1fps_768px").glob("*.jpg"))
    lum = [round(PIL.open(p).convert("L").getpixel((32, 32)) / 2) for p in jpgs]
    assert [abs(x - 30 * k) <= 1 for k, x in enumerate(lum)] == [True] * len(lum), lum


def test_stale_cleanup_drops_the_lane_sheets_too(clip):
    out = frames_dir(clip)
    sheets = out / "frames_1fps_768px" / "sheets"
    assert any(sheets.glob("sheet_*.jpg"))
    video2llm.stale_cleanup(out)
    assert not any(sheets.glob("sheet_*.jpg")) and not any((out / "frames_1fps_768px").glob("*.jpg"))
    assert not list(out.glob("lane*.md"))


def test_agent_files_carry_one_instruction():
    """agents/: ONE instruction — AGENTS.md (Codex / Antigravity / Cursor; Claude Code imports it, Gemini CLI and Qwen
    Code get it under their own file name), the Cursor rule and the Claude skill carry the same text after their front
    matter (the skill names the script's place inside the skill folder). No per-app copies that differ only by name."""
    agents = ROOT / "agents"

    def body(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            text = text.split("---\n", 2)[2]
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)  # the header comment (one line or several)
        text = text.replace("~/.claude/skills/video2llm/video2llm.py", "/path/to/video2llm.py")
        text = text.replace("   (the copy of video2llm.py in this skill's folder)", "")
        return text.strip()

    plain = [agents / "AGENTS.md"]
    for dup in ("CLAUDE.md", "GEMINI.md", "QWEN.md"):
        assert not (agents / dup).exists(), f"{dup}: one AGENTS.md, saved under the app's name by the user"
    others = [agents / ".cursor" / "rules" / "video2llm.mdc", agents / "SKILL.md"]
    ref = body(plain[0])
    for path in plain + others:
        assert path.is_file(), path
        assert body(path) == ref, path
    for must in ("python3 /path/to/video2llm.py <video>", "lane.md", "--frames all", "--sounds", "ffmpeg", "yes or no"):
        assert must in ref, must
    skill = (agents / "SKILL.md").read_text(encoding="utf-8")
    front = skill.split("---\n", 2)[1]
    assert "name: video2llm" in front
    desc = next(line for line in front.splitlines() if line.startswith("description: "))[len("description: "):]
    assert len(desc) <= 200, len(desc)
    cursor = (agents / ".cursor" / "rules" / "video2llm.mdc").read_text(encoding="utf-8")
    assert cursor.startswith("---\n") and "alwaysApply: true" in cursor.split("---\n", 2)[1]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("agents/AGENTS.md", "@agents/AGENTS.md", "GEMINI.md", "QWEN.md", "SKILL.md", "video2llm.mdc", "agents/README.md"):
        assert name in readme, name
    guide = (agents / "README.md").read_text(encoding="utf-8")
    for app in ("Claude Code", "Codex CLI", "Antigravity", "Gemini CLI", "Qwen Code", "Cursor", "Cowork"):
        assert app in guide, app


FAKE_YT_DLP = """\
# a stand-in for yt-dlp: copies the clip named in VIDEO2LLM_TEST_CLIP to the -o folder as
# "Title [abc123].mp4" and writes the --print-to-file outputs the way yt-dlp does
import os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("-o") + 1]).parent
prints = {}
for i, a in enumerate(args):
    if a == "--print-to-file":
        prints[args[i + 1]] = Path(args[i + 2])
dest = out / "Title [abc123].mp4"
prints["%(id)s"].write_text("abc123\\n", encoding="utf-8")
if dest.exists():      # yt-dlp skips a file it downloaded before and the after_move hook with it
    print("[download] " + str(dest) + " has already been downloaded")
    sys.exit(0)
shutil.copy(os.environ["VIDEO2LLM_TEST_CLIP"], dest)
if "--write-subs" in args:   # the site's captions beside the file, as yt-dlp writes them: a manual English
    # track (the video's language) and YouTube's machine translation into Russian; the info file tells which is which
    (out / "Title [abc123].en.vtt").write_text(
        "WEBVTT\\n\\n00:00:00.500 --> 00:00:02.000\\nhello from the captions\\n\\n00:00:03.000 --> 00:00:05.000\\nsecond line\\n",
        encoding="utf-8")
    (out / "Title [abc123].ru.vtt").write_text(
        "WEBVTT\\n\\n00:00:00.500 --> 00:00:02.000\\nprivet iz perevoda\\n", encoding="utf-8")
    (out / "Title [abc123].info.json").write_text(
        '{"language": "en", "subtitles": {"en": []}, "automatic_captions": {"en-orig": [], "ru": []}}', encoding="utf-8")
print("[download]  50.0% of 1.00MiB")
print("[download] 100.0% of 1.00MiB")
prints["after_move:filepath"].write_text(str(dest) + "\\n", encoding="utf-8")
"""


def fake_yt_dlp(tmp_path):
    """The stand-in above as the yt-dlp to run — a command line (this Python + the script), the same on
    every platform; a .bat on Windows would let cmd read the `<` of `[height<=1080]` as a redirection."""
    py = tmp_path / "fake_yt_dlp.py"
    py.write_text(FAKE_YT_DLP, encoding="utf-8")
    return {"VIDEO2LLM_YT_DLP": f'"{sys.executable}" "{py}"'}


def test_a_url_is_downloaded_with_yt_dlp_then_treated_as_a_file(clip, tmp_path):
    env = fake_yt_dlp(tmp_path)
    env["VIDEO2LLM_TEST_CLIP"] = str(clip)
    env["VIDEO2LLM_DOWNLOADS"] = str(tmp_path / "downloads")   # this test's folder, not the user's Videos
    out = tmp_path / "lane"
    # no --no-transcript: the captions stand in; --captions ru asks for Russian, but the manual English track —
    # the video's own language — is read before the site's machine translation
    r = run("https://example.com/watch?v=abc123", "--captions", "ru", "--out", out, env=env)
    assert "[download] 100%" in r.stderr
    assert "(manual captions, no Whisper)" in r.stderr and "faster-whisper" not in r.stderr
    head = (out / "lane.md").read_text(encoding="utf-8")
    assert 'video "Title [abc123].mp4"' in head
    assert "site's own captions, written by people" in head and "hello from the captions" in head and "second line" in head
    assert "privet iz perevoda" not in head
    args = video2llm.yt_dlp_args(["x"], "u", tmp_path, FFMPEG, tmp_path / "i", tmp_path / "p", None, ["es"])
    assert args[args.index("--sub-langs") + 1] == "es,en,.*-orig,-live_chat" and "--write-info-json" in args
    assert "--write-subs" not in video2llm.yt_dlp_args(["x"], "u", tmp_path, FFMPEG, tmp_path / "i", tmp_path / "p", subs=False)
    assert "Title [abc123].mp4" in head.split("run ")[1]   # the rerun command names the downloaded file, not the URL
    assert "--write-subs" in video2llm.yt_dlp_args(["x"], "u", tmp_path, FFMPEG, tmp_path / "i", tmp_path / "p")
    # --download-only: the path on stdout, nothing else written
    r3 = run("https://example.com/watch?v=abc123", "--download-only", "--quiet", env=env)
    assert r3.stdout.strip().endswith("Title [abc123].mp4") and Path(r3.stdout.strip()).is_file()
    # the same URL again: yt-dlp reports the file as downloaded already and video2llm finds it by its id
    r2 = run("https://example.com/watch?v=abc123", "--no-transcript", "--out", out, env=env)
    assert "[download] 100%" in r2.stderr and "Title [abc123].mp4" in r2.stderr and (out / "lane.md").exists()


def test_yt_dlp_args_ask_for_one_mp4_up_to_1080p(tmp_path):
    args = video2llm.yt_dlp_args(["yt-dlp"], "https://x/y", tmp_path, "/usr/bin/ffmpeg", tmp_path / "id", tmp_path / "p")
    assert args[0] == "yt-dlp" and args[-1] == "https://x/y" and args[-2] == "--"
    assert "--no-playlist" in args and "--merge-output-format" in args and args[args.index("--merge-output-format") + 1] == "mp4"
    assert "[height<=1080]" in args[args.index("-f") + 1]
    assert args[args.index("-o") + 1].startswith(str(tmp_path))
    assert "after_move:filepath" in args


def test_no_yt_dlp_says_how_to_install_it(tmp_path):
    env = {"PATH": str(tmp_path), "VIDEO2LLM_YT_DLP": ""}
    if has("yt_dlp"):
        pytest.skip("the yt_dlp module is installed in this Python — it would be used")
    # ffmpeg must still be found: point at it explicitly
    r = run("https://example.com/v", "--ffmpeg", FFMPEG, "--ffprobe", shutil.which("ffprobe"), "--out", tmp_path / "o",
            env=env, ok=False)
    assert r.returncode != 0 and "yt-dlp" in r.stderr and ("pip install yt-dlp" in r.stderr or "install yt-dlp" in r.stderr)


def test_agent_tool_header_names_the_frame_files(clip):
    run(clip, "--no-transcript", "--agent-tool", "video")
    head = (frames_dir(clip) / "lane.md").read_text(encoding="utf-8")
    assert "frames_1fps_768px" in head and "/mm-ss.jpg" in head and "only when the user asks for that frame as a file" in head


YOUTUBE_ROLLING_VTT = """WEBVTT
Kind: captions
Language: en

00:00:00.160 --> 00:00:02.390 align:start position:0%
 
so<00:00:00.640><c> today</c><00:00:00.880><c> we</c><00:00:01.199><c> look</c>

00:00:02.390 --> 00:00:02.400 align:start position:0%
so today we look
 

00:00:02.400 --> 00:00:04.549 align:start position:0%
so today we look
at<00:00:02.720><c> the</c><00:00:03.040><c> fence</c>

00:00:04.549 --> 00:00:04.559 align:start position:0%
at the fence
 

00:00:04.559 --> 00:00:06.000 align:start position:0%
at the fence
and<00:00:05.000><c> laugh</c>
"""


def test_youtube_rolling_captions_come_out_once_each():
    segs = video2llm.parse_captions(YOUTUBE_ROLLING_VTT)
    assert [s["text"] for s in segs] == ["so today we look", "at the fence", "and laugh"]
    assert segs[0]["start"] == pytest.approx(0.16) and segs[1]["start"] == pytest.approx(2.4)
    assert segs[0]["end"] >= 2.0   # a span up to the next line, not the 10 ms of a rolling cue
    srt = "1\n00:00:01,000 --> 00:00:02,000\nPrivet\n\n2\n00:00:02,500 --> 00:00:03,000\nDanya!\n"
    assert [(s["start"], s["text"]) for s in video2llm.parse_captions(srt)] == [(1.0, "Privet"), (2.5, "Danya!")]


def test_a_subtitle_file_beside_the_video_stands_in_for_whisper(clip, tmp_path):
    v = tmp_path / "with subs.mp4"
    shutil.copy(clip, v)
    (tmp_path / "with subs.ru.srt").write_text("1\n00:00:01,000 --> 00:00:02,500\nПривет, Даня!\n", encoding="utf-8")
    (tmp_path / "with subs.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,500\nHi Danya!\n", encoding="utf-8")
    r = run(v, "--captions", "ru")   # transcript on, no Whisper: the file beside the video is used — the preferred one
    assert "captions, no Whisper" in r.stderr
    head = (v.with_name("with subs_frames") / "lane.md").read_text(encoding="utf-8")
    assert "Привет, Даня!" in head and "Hi Danya!" not in head and "site's own captions" in head
    r2 = run(v, "--language", "en", "--out", tmp_path / "en_lane")   # --language picks the file
    assert "Hi Danya!" in (tmp_path / "en_lane" / "lane.md").read_text(encoding="utf-8")


def test_single_frames_at_moments(clip, tmp_path):
    out = tmp_path / "at"
    r = run(clip, "--no-transcript", "--frames", "at", "--times", "1:00,0:02.5,4", "--out", out, "--agent-tool", "video")
    assert "frames at 3 moment(s)" in r.stderr
    md = next(out.glob("lane_at_*.md"))
    head = md.read_text(encoding="utf-8")
    assert "the 3 frame(s) you asked for" in head and "[00:02.50]" in head and "[00:04.00]" in head
    assert '"what": "at"' in head and "Rules:" in head
    files = sorted((out / "frames_at_1024px").glob("*.jpg"))
    assert [f.name for f in files] == ["00-02.50.jpg", "00-04.00.jpg", "00-05.95.jpg"]   # 1:00 is past the end → the last frame
    assert video2llm.jpeg_size(str(files[0]))[1] == 1024   # portrait: the long side is 1024
    lane = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert [b["tag"] for b in lane["blocks"] if b["type"] == "image"] == ["00:02.50", "00:04.00", "00:05.95"]
    r2 = run(clip, "--no-transcript", "--frames", "at", "--out", out, ok=False)
    assert r2.returncode != 0 and "--times" in r2.stderr


def test_lecture_lane_has_the_words_and_no_frames(clip, tmp_path):
    v = tmp_path / "lecture.mp4"
    shutil.copy(clip, v)
    (tmp_path / "lecture.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,500\nOpen the Settings panel\n\n2\n00:00:03,000 --> 00:00:04,000\nand click Export\n", encoding="utf-8")
    (tmp_path / "lecture.info.json").write_text(json.dumps({"title": "Export tutorial", "categories": ["Education"], "language": "en",
                                                             "subtitles": {"en": []}, "chapters": [{"start_time": 0, "title": "Intro"}, {"start_time": 3, "title": "Export"}]}), encoding="utf-8")
    r = run(v, "--frames", "none", "--agent-tool", "video")
    assert "lecture lane: 2 line(s), no frames" in r.stderr
    out = v.with_name("lecture_frames")
    head = (out / "lane_lecture.md").read_text(encoding="utf-8")
    assert "LECTURE MODE" in head and "«Export tutorial»" in head and "NO frames" in head
    assert "Open the Settings panel" in head and "and click Export" in head and "[chapters, from the site]" in head
    assert '"what": "at", "times": ["03:12", "07:40"]' in head and "written by people" in head
    assert "with the video_guide tool" in head   # the harness's own tool, named only in --agent-tool mode
    assert not (out / "frames_1fps_768px").exists() or not list((out / "frames_1fps_768px").glob("*.jpg"))
    hint = video2llm.lecture_hint(v, {"duration": 6.0}, [{"start": 1, "end": 2.5}, {"start": 3, "end": 4}])
    assert hint["hint"] is True and "category" in " ".join(hint["reasons"])
    assert video2llm.lecture_hint(clip, {"duration": 6.0}, None)["hint"] is False
