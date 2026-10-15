# ContentClipper 🎬⚡

**ContentClipper** is an automated, AI-driven video intelligence pipeline that transforms long-form videos into high-retention, monetization-ready short-form content (YouTube Shorts, TikTok, Instagram Reels).

The system operates in a two-stage pipeline:
1. **Extraction & Scoring (`SmartViralClipper.py`)**: Ingests videos, eliminates dead air, transcribes speech with word-level timestamps using OpenAI Whisper, scores segments using multi-factor viral heuristics, and exports candidate clips with synced captions.
2. **Post-Processing & Retention Enhancement (`PostProcessing.py`)**: Converts clips to vertical 9:16 format, applies retention zooms with OpenCV face tracking, layers hook headlines and TTS commentary, and renders using Apple Silicon hardware-accelerated encoders (`h264_videotoolbox` / `hevc_videotoolbox`).

---

## Architecture Pipeline

```text
Long-Form Video (YouTube URL or Local File)
                     │
                     ▼
         [ Stage 1: SmartViralClipper ]
         ├─ Vectorized NumPy Silence Stripping
         ├─ Whisper Speech-to-Text & Word Timestamps
         ├─ Multi-Factor Viral Scoring (Energy + Density + Sentiment)
         └─ Initial 9:16 Export with Burned-In Captions
                     │
                     ▼
             viral_clips/  +  clip_metadata.json
                     │
                     ▼
         [ Stage 2: PostProcessing Engine ]
         ├─ Face Detection & Smart Dynamic Zooming (OpenCV)
         ├─ Attention Hook Banners & Engagement Overlays
         ├─ AI Hook Commentary (gTTS) & Audio Ducking
         ├─ Loop Audio Transitions & Color Grading
         └─ VideoToolbox Hardware Encoding (HEVC/H.264)
                     │
                     ▼
        viral_clips_enhanced/  +  enhanced_metadata.json
```

---

## Key Features

- **High-Performance Audio Ingestion**: Audio is loaded directly into RAM as a NumPy array; all volume and energy queries run via array slicing without redundant disk I/O.
- **Accurate Subtitles**: Whisper AI timestamping aligned via binary search (`bisect`), rendered with caching for instant frame lookups.
- **Viral Potential Scoring**: Evaluates audio amplitude dynamics, speech tempo/density, and emotional or punchline markers.
- **Smart Framing & Retention Zoom**: Converts horizontal footage into 9:16 vertical shorts, using Haar cascade face detection to automatically center speakers during dynamic zooms.
- **AI Hook Commentary**: Generates contextual voiceover intros using `gTTS`, smoothly ducking primary audio.
- **Apple Silicon Hardware Acceleration**: Uses `h264_videotoolbox` and `hevc_videotoolbox` for fast, power-efficient exports on macOS M-series chips.

---

## Prerequisites

1. **Python 3.9+**
2. **FFmpeg**: Must be available on your system path.
   - **macOS** (Homebrew):
     ```bash
     brew install ffmpeg
     ```
   - **Ubuntu / Debian**:
     ```bash
     sudo apt update && sudo apt install ffmpeg
     ```

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone git@github.com:manoj7maxi/ContentClipper.git
cd ContentClipper

pip install -r requirements.txt
```

---

## Usage

### Stage 1: Generate Viral Clips

Download and extract the top viral moments from any YouTube video or local media file:

```bash
python SmartViralClipper.py "https://www.youtube.com/watch?v=EXAMPLE_ID"
```

#### Common Stage 1 Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `viral_clips` | Destination folder for candidate clips |
| `--top-n` | `3` | Number of top viral clips to extract |
| `--min-duration` | `15.0` | Minimum clip duration (seconds) |
| `--max-duration` | `90.0` | Maximum clip duration (seconds) |
| `--silence-db` | `-40` | Silence threshold in dB for cutoffs |
| `--words-per-phrase` | `5` | Caption chunk size |
| `--caption-fontsize` | `52` | Subtitle font size |
| `--no-captions` | `False` | Disable burning captions into video |
| `--keep-aspect` | `False` | Preserve original aspect ratio instead of 9:16 |
| `--preset` | `fast` | x264 preset (`ultrafast`, `fast`, `medium`, etc.) |
| `--bitrate` | `6M` | Video target bitrate |

---

### Stage 2: Post-Processing & Enhancement

Enhance the extracted clips with dynamic zooms, hook headlines, AI audio commentary, and hardware encoding:

```bash
python PostProcessing.py
```

#### Common Stage 2 Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir` | `viral_clips` | Directory containing clips from Stage 1 |
| `--output-dir` | `viral_clips_enhanced`| Target directory for enhanced shorts |
| `--metadata` | `viral_clips/clip_metadata.json` | Path to metadata generated in Stage 1 |
| `--hevc` | `False` | Export using HEVC (H.265) for higher compression |
| `--bitrate` | `auto` | Target video bitrate (e.g. `6M`, `8M`, `10M`) |
| `--no-commentary` | `False` | Disable AI hook commentary intro |
| `--no-zoom` | `False` | Disable dynamic retention zoom effect |
| `--no-label` | `False` | Disable monetization headline overlays |
| `--enable-loop` | `False` | Create a seamless loop crossfade at clip boundaries |
| `--square` | `False` | Also export 1:1 aspect ratio versions |

---

## Directory Structure

```text
├── SmartViralClipper.py             # Stage 1 extraction & viral scoring script
├── PostProcessing.py               # Stage 2 video post-processing & enhancement
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignores large video formats (*.mp4, etc.)
├── viral_clips/
│   └── clip_metadata.json          # Metadata & transcript timestamps from Stage 1
└── viral_clips_enhanced/
    └── enhanced_metadata.json      # Final metrics & enhancement summary from Stage 2
```

---

## License

MIT
