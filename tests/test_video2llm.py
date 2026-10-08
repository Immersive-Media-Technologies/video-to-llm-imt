"""End-to-end checks of video2llm.py on a generated clip (runs on Linux, macOS and Windows in CI).

Needs ffmpeg on PATH. fpdf2 (PDF, contact sheets) and faster-whisper (speech) are optional: the
tests that need them are skipped without them.
"""
import hashlib
import json
import os
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
    """agents/: every app's file is the same instruction — the plain ones word for word, the Cursor rule and the
    Claude skill after their front matter (the skill names the script's place inside the skill folder)."""
    agents = ROOT / "agents"

    def body(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            text = text.split("---\n", 2)[2]
        text = "\n".join(line for line in text.splitlines() if not line.startswith("<!--"))
        text = text.replace("~/.claude/skills/video2llm/video2llm.py", "/path/to/video2llm.py")
        text = text.replace("   (the copy of video2llm.py in this skill's folder)", "")
        return text.strip()

    plain = [agents / n for n in ("CLAUDE.md", "AGENTS.md", "GEMINI.md", "QWEN.md")]
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
    for name in ("agents/", "CLAUDE.md", "AGENTS.md", "GEMINI.md", "QWEN.md", "SKILL.md", "video2llm.mdc"):
        assert name in readme, name
