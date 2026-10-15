# ContentClipper 🎬⚡

**ContentClipper** is an automated, AI-driven video intelligence pipeline that transforms long-form videos into high-retention, monetization-ready short-form content (YouTube Shorts, TikTok, Instagram Reels).

Equipped with an integrated **Clip Quality Engine (`ClipQualityEngine.py`)**, ContentClipper evaluates candidate clips across 8 weighted criteria, continuously tracks protagonists in vertical 9:16 framing, synchronizes cuts to musical beats, masters audio to broadcast standards (-14 LUFS), and outputs comprehensive quality reports.

---

## Architecture Pipeline

```text
Long-Form Video (YouTube URL or Local File)
                     │
                     ▼
         [ Stage 1: SmartViralClipper ]
         ├─ Vectorized NumPy Silence Stripping
         ├─ Whisper Speech-to-Text & Word Timestamps
         ├─ Sliding-Window Candidate Generation (Sentence Boundary Alignment)
         ├─ Clip Quality Engine Evaluation (8 Weighted Criteria)
         ├─ Non-Maximum Suppression (NMS) to eliminate overlaps
         ├─ Protagonist Tracking for 9:16 Vertical Cropping
         ├─ Audio Boundary Micro-Fades
         └─ Export with Burned-In Captions
                     │
                     ▼
             viral_clips/  +  clip_metadata.json  +  quality_report.md
                     │
                     ▼
         [ Stage 2: PostProcessing Engine ]
         ├─ Continuous Protagonist Centering (OpenCV)
         ├─ Beat Detection & Beat-Synchronized Pattern Interrupt Flash
         ├─ Attention Hook Banners & Engagement Overlays
         ├─ AI Hook Commentary (gTTS) & Audio Ducking
         ├─ Audio Loudness Mastering (-14 LUFS) & Soft Limiting
         ├─ Seamless Loop Audio & Video Transitions
         └─ VideoToolbox Hardware Encoding (HEVC/H.264)
                     │
                     ▼
        viral_clips_enhanced/  +  enhanced_metadata.json  +  quality_report.md
```

---

## Clip Quality Engine (`ClipQualityEngine.py`)

Every candidate clip is evaluated from 0–100 using 8 weighted criteria:

| Criterion | Weight | What It Analyzes |
|:---|:---:|:---|
| **Hook Strength** | **20%** | Opening 1–2s vocal onset energy, trigger words, questions, absence of weak fillers (`"because"`, `"so"`). |
| **Emotional Impact** | **15%** | High-arousal sentiment words, acoustic pitch/energy dynamic range, expressive punctuation, visual motion. |
| **Protagonist Visibility** | **15%** | Percentage of frames where the primary protagonist is detected, centered, and prominent. |
| **Pacing** | **15%** | Words-per-second (target: 2.3–3.8 wps), elimination of awkward dead-air pauses ($>1.2$s). |
| **Story Continuity** | **10%** | Clean opening thought, complete sentence termination (`.`, `?`, `!`), narrative arc suitability. |
| **Audio Synchronization** | **10%** | Cut and transition alignment to musical beats and speech cadence. |
| **Visual Composition** | **10%** | Rule-of-thirds/centering, vertical 9:16 framing, adherence to mobile UI safe margins. |
| **Ending / Rewatchability** | **5%** | Punchline/dramatic payoff, natural audio cushion, seamless loop potential. |

### Quality Tiers & Technical Health
- **PRIORITY ($\ge 80$)**: Highest viral potential; recommended for immediate publishing.
- **REVIEW ($65 - 79$)**: Solid candidate with minor pacing or composition adjustments suggested.
- **RECONSIDER ($< 65$)**: Lower-scoring candidate; preserved as a fallback.
- **Technical Status (`PASS` / `FAIL`)**: Technical issues (silent audio, invalid aspect ratio, video clipping) are validated independently from engagement scores.

---

## Key Features

1. **Intelligent Hook Detection**: Analyzes opening moments, removing leading silence and weak sentence starters.
2. **Continuous Protagonist Tracking**: Uses multi-frame face detection with exponential moving average (EMA) smoothing and deadband inertia to keep the main character centered in 9:16 vertical video without jitter.
3. **Music & Beat Synchronization**: SciPy-based onset and novelty detection snaps cut transitions and visual flashes to musical beats.
4. **Professional Audio Mastering**: Loudness normalization targeting -14 LUFS, soft peak limiting (max -1 dBFS), high-pass rumble filtering ($>80$ Hz), and smooth 50–150ms boundary fades.
5. **Non-Maximum Suppression (NMS)**: Eliminates duplicate clips with overlapping time windows.
6. **Detailed Quality Reporting**: Automatically generates `quality_report.json` and human-readable `quality_report.md` summarizing why each clip was selected and which improvements were applied.
7. **Automatic Language Detection & Translation**: Automatically detects any spoken language in the audio via Whisper mel spectrograms and translates non-English audio to fluent, well-synced English captions with safe mobile 9:16 margins.
8. **Complete Thought Boundary Closure**: Prevents abrupt cutoffs by rejecting candidate clips that end on dangling words (conjunctions, prepositions, filler words) and requiring terminal punctuation or natural pauses. Appends an intelligent 0.75s speech cushion and smooth audio/video fadeouts.
9. **Apple Silicon Hardware Acceleration**: Uses `h264_videotoolbox` and `hevc_videotoolbox` for high-speed, power-efficient exports on macOS M-series chips.

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

Clone the repository and install dependencies:

```bash
git clone git@github.com:manoj7maxi/ContentClipper.git
cd ContentClipper

pip install -r requirements.txt
```

---

## Usage

### Stage 1: Generate Viral Clips with Quality Engine

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
| `--words-per-phrase` | `4` | Number of words shown together per subtitle phrase |
| `--caption-fontsize` | `44` | Subtitle base font size (auto-scaled for mobile) |
| `--no-captions` | `False` | Disable burning captions into video |
| `--no-translate` | `False` | Disable automatic translation to English captions for foreign language videos |
| `--keep-aspect` | `False` | Preserve original aspect ratio instead of 9:16 |
| `--preset` | `fast` | x264 preset (`ultrafast`, `fast`, `medium`, etc.) |
| `--bitrate` | `6M` | Video target bitrate |

---

### Stage 2: Post-Processing & Cinematic Enhancement

Enhance the extracted clips with beat-synchronized flashes, hook headlines, AI audio commentary, and hardware encoding:

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

## Running Automated Tests

Run the test suite to verify scoring weights, protagonist tracking, beat detection, and audio mastering:

```bash
python test_quality_engine.py
```

---

## Directory Structure

```text
├── ClipQualityEngine.py            # 8-criteria scoring, protagonist tracking, beat sync, audio mastering
├── SmartViralClipper.py             # Stage 1 extraction, NMS selection & viral scoring script
├── PostProcessing.py               # Stage 2 video post-processing & enhancement
├── test_quality_engine.py          # Comprehensive test suite
├── requirements.txt                # Python dependencies
├── .gitignore                      # Ignores large video formats (*.mp4, etc.)
├── viral_clips/
│   ├── clip_metadata.json          # Extended metadata with quality metrics
│   └── quality_report.md           # Markdown quality assessment report
└── viral_clips_enhanced/
    ├── enhanced_metadata.json      # Final metrics & enhancement summary
    └── quality_report.md           # Comprehensive quality report for enhanced clips
```

---

## License

MIT
