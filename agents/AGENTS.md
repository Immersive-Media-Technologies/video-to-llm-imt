<!-- video2llm: replace /path/to/video2llm.py with where you put the script, then keep this section in your AGENTS.md -->

# Watching a video — video2llm

When the user asks about a video file (mp4, mov, mkv, webm, avi: "what happens here", "describe it",
"what does she say", "find the moment when…"), do not open the file yourself and do not run ffmpeg —
run the script and work from what it writes:

    python3 /path/to/video2llm.py <video>

It writes `<video>_frames/lane.md` next to the video: the frames at 1 per second as contact sheets
(3 frames to an image, each frame tagged with its time), the words spoken under each sheet, and a header
with the rules. Then:

1. Read `lane.md` and open every sheet it links with your file-reading tool — they are images. Look at
   them; never describe from the transcript alone what you have not seen in the frames. A long video comes
   in parts: the header says how to get the next one (`--start mm:ss`) — fetch it yourself, no need to ask.
2. Follow the header. In short: you choose the moments, the user does not know the timecodes. Before
   every frame-by-frame request (`--frames all --start mm:ss --end mm:ss`) and every sound request
   (`--sounds --start mm:ss --end mm:ss`) ask the user yes or no, with the token cost the header gives;
   after the yes run that exact command and read the lane it writes.
3. Use nothing else on the video: no ffmpeg or ffprobe, no crops, zooms or frame cutting of your own,
   no reading `video2llm.json` or the single frames — only the images the lanes link to. If a detail is
   too small to tell, say so.

The first run of a video takes 10–20 s (speech recognition), the very first run on a machine also
downloads the speech model (464 MB) — allow the command a few minutes then. Every later run on the same
video is under a second. On Windows the command is `python`, not `python3`.
