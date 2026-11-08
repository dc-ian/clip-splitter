# Clip Splitter

Split one long video into short clips (default 15–30 seconds) that end at natural boundaries instead of mid-sentence. Use it from the command line or from a desktop window.

Clips can optionally get captions, be reframed to 9:16 for vertical platforms, and be ranked so you only export the best ones.

## Features

- **Smart cuts.** Each cut is chosen from the best signal available, in this order: sentence end, pause, scene change, loudness dip, gap between words, and finally a hard cut with a short fade. No cut ever lands inside a spoken word.
- **Speech-aware (optional).** Local [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcription finds sentence ends and powers captions. Everything runs on your machine.
- **Desktop UI.** Drop a video, analyze, review the clip list, preview, trim, merge or delete clips, then export. Only changed clips are re-encoded.
- **Captions.** Burned into the video (with the spoken word highlighted, or plain phrases), as `.srt` files next to each clip, or both.
- **9:16 reframing.** Center crop, auto-pan that follows the speaker, or the full frame over a blurred background.
- **Highlight scoring.** Clips are scored from hook words, loudness, pace, speech density and cut quality, so you can export only the best N.
- **Crash-safe.** Progress is saved after every clip. Run the same command again and it resumes. Cancel works at every stage.
- **Batch queue.** Process many videos one after another. A file that keeps crashing is skipped after 3 tries instead of blocking the rest.
- **Handles awkward files.** Variable-frame-rate video is normalized once, phone videos stored sideways are detected, and the output is always `yuv420p` so clips play everywhere.
- **Hardware encoding.** NVENC, Quick Sync or VideoToolbox are used when they work, with a fallback to `libx264`.

## Requirements

| What | Needed for |
|---|---|
| Python 3.8 or newer | everything |
| [FFmpeg](https://ffmpeg.org) and `ffprobe` on your `PATH` | everything |
| `PySide6` | the desktop UI |
| `faster-whisper` | sentence-end cuts and captions (optional) |

Burned-in captions also need an FFmpeg build with `libass`. On Windows use the **full** build (for example from gyan.dev). Without it, use `.srt` captions instead.

## Install

```bash
git clone https://github.com/YOUR-USERNAME/clip-splitter.git
cd clip-splitter
pip install -r requirements.txt
```

Check that FFmpeg is found:

```bash
ffmpeg -version
ffprobe -version
```

To skip speech features, remove `faster-whisper` from `requirements.txt` first, or install only what you need (`pip install PySide6`).

## Desktop UI

```bash
python clipper_ui.py
```

1. Drop a video onto the window, or press **Open video**.
2. Set the shortest and longest clip length. Turn Whisper on for sentence-end cuts.
3. Press **1. Analyze**. The first run with Whisper downloads a model, so it can take a while.
4. Review the list. Orange rows ended on a hard cut, so check those first. Click a clip to preview it, then trim, merge, delete or restore as needed.
5. Choose the export options (below), then press **2. Export clips**.

**Export options**

- **Precise** (default) re-encodes for frame-accurate cuts. **Fast** copies the stream, which is near-instant but cuts snap to keyframes and can be seconds off, with no fades. Burned captions and 9:16 need Precise.
- **Captions:** off, burned into the video, sidecar `.srt` files, or both.
- **9:16 vertical:** center crop, auto-pan to the speaker, or blurred background.
- **Best clips:** export only the top N by highlight score. The others show as "Not in best" and can be restored.

Good to know:

- If you change a plan setting (clip lengths, Whisper options) after analyzing, Export turns off until you press Analyze again. Replanning discards your manual edits and deletes the clips already exported in that folder, and the app asks before doing it.
- The Status column shows **Out of date** when a clip was exported with different options. Export redoes only those clips.
- Cancel works during Analyze as well as Export. Finished clips are kept.

## Command line

```bash
# smart cuts (default)
python clipper.py input.mp4

# preview the cut list without encoding anything
python clipper.py input.mp4 --plan-only

# fixed-length clips instead of smart cuts
python clipper.py input.mp4 --mode fixed

# custom length and output folder
python clipper.py input.mp4 -o clips --clip-len 30 --min-len 15

# captions burned in + .srt files, vertical 9:16 following the speaker
python clipper.py input.mp4 --captions both --vertical follow

# only the best 10 clips
python clipper.py input.mp4 --highlights 10

# near-instant keyframe cuts
python clipper.py input.mp4 --fast
```

By default the output goes to `<video name>_clips` next to the video.

### Batch queue

Give several files or a folder and they are queued and processed one by one:

```bash
python clipper.py video1.mp4 video2.mp4 folder_of_videos/ --queue-file my_queue.json
python clipper.py --queue-status --queue-file my_queue.json     # see progress
python clipper.py --resume-queue --queue-file my_queue.json     # continue after a stop
python clipper.py --retry-failed --queue-file my_queue.json     # put failed items back in line
```

### All options

Run `python clipper.py --help` for the full list. The most useful ones:

| Option | What it does |
|---|---|
| `--min-len`, `--clip-len` | Shortest and longest clip in seconds (default 15 and 30) |
| `--skip-transcription` | Don't run Whisper; use pauses and scene changes only |
| `--whisper-model` | `tiny`, `base` (default), `small`, `medium`. Bigger is slower and more accurate |
| `--language` | Speech language code such as `en` or `tl` (default: auto-detect) |
| `--device` | `auto`, `cpu` or `cuda` for Whisper |
| `--captions` | `off`, `burn`, `sidecar`, `both` |
| `--caption-style` | `words` (highlight the spoken word) or `phrase` |
| `--vertical` | `off`, `center`, `follow`, `blur` |
| `--highlights N` / `--highlights-pct P` | Export only the best N clips or best P percent |
| `--highlight-keywords` | Extra comma-separated words that make a clip score higher |
| `--fast` | Stream-copy export (keyframe-accurate only) |
| `--encoder` | Force `h264_nvenc`, `h264_qsv`, `h264_videotoolbox` or `libx264` |
| `--reanalyze`, `--restart` | Ignore the cached analysis or saved progress |

## What you get

```
my_video_clips/
├── clip_0001.mp4
├── clip_0001.srt        (if sidecar captions are on)
├── clip_0002.mp4
├── ...
├── manifest.json        the clip list, cut reasons, scores, and export progress
├── analysis.json        cached pause / scene / loudness analysis
└── transcript.json      cached word timestamps (if Whisper was used)
```

The cached files make re-clipping instant. Delete them, or use `--reanalyze`, to start over.

## How the cuts are chosen

1. The audio is scanned for the video's own noise floor, then for pauses, loudness dips and scene changes. With Whisper on, word timestamps add sentence ends and word gaps.
2. From the start of each clip, the planner looks in the window between the minimum and maximum length and picks the best-scoring cut point. Two different signals agreeing on the same moment boost that point.
3. If nothing suitable exists, it makes a hard cut and fades the audio and picture around it. These clips are flagged in the list and in the end report.

## Build a standalone app (optional)

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile clipper_ui.py
```

FFmpeg is not bundled. It still has to be installed on the machine that runs the app.

## Troubleshooting

- **`'ffmpeg' not found on PATH`**: install FFmpeg and reopen your terminal so the `PATH` change takes effect.
- **"This FFmpeg build cannot draw subtitles"**: you need a build with `libass` for burned captions. Use a "full" build, or choose sidecar `.srt` captions.
- **"Captions need a transcript"**: install `faster-whisper` and make sure Whisper is turned on. If no speech was found, there is nothing to caption.
- **Export button is greyed out**: either every clip is already exported with the current options, or you changed a plan setting and need to press Analyze again.
- **First Analyze seems stuck**: with Whisper on, the first run downloads the model. Later runs reuse it.
- **Fast mode gives clips of the wrong length**: expected. Fast cuts can only start on keyframes. Use Precise for exact cuts.

## License

Add a `LICENSE` file before publishing. If you're unsure which to choose, [choosealicense.com](https://choosealicense.com) walks you through it.
