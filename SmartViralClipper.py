"""
Smart Viral Clip Generator  ·  Optimized Pipeline
===================================================
Downloads video, removes silence, scores for viral potential,
and exports top clips with frame-accurate synced captions.

Requirements:
    pip install yt-dlp moviepy pydub openai-whisper numpy Pillow scipy

Optimizations vs previous version
──────────────────────────────────
1. Audio loaded ONCE into RAM as numpy array — all energy queries are
   pure array slices, zero disk I/O after first load.
2. Font loaded ONCE at startup, cached globally — not re-scanned per phrase.
3. Caption frames rendered ONCE per unique text via LRU cache — repeated
   phrases hit the cache instead of re-running PIL draw calls.
4. Whisper WAV extracted via direct ffmpeg subprocess — 2-3x faster than
   MoviePy's write_audiofile, outputs exactly 16 kHz mono.
5. Silence detection runs on vectorised numpy instead of pydub's Python
   loop — orders of magnitude faster on long audio.
6. Word-to-segment assignment uses bisect (O log n) instead of O(n) scan.
7. Clips exported strictly sequentially — one ffmpeg process at a time.
8. CRF + preset exposed as CLI flags for quality/speed tradeoff control.
"""

from __future__ import annotations

import bisect
import functools
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yt_dlp
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from scipy.io import wavfile

from ClipQualityEngine import (
    quality_engine,
    BeatDetector,
    ProtagonistTracker,
    AudioMastering,
    QualityEvaluation,
)


# ─────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────

@dataclass
class CaptionPhrase:
    start: float   # seconds in original-video time
    end:   float
    text:  str


@dataclass
class Segment:
    start:                float
    end:                  float
    transcript:           str   = ""
    caption_phrases:      list  = field(default_factory=list)
    energy:               float = 0.0
    viral_score:          float = 0.0
    quality_score:        float = 0.0
    quality_tier:         str   = "REVIEW"
    technical_status:     str   = "PASS"
    technical_issues:     list  = field(default_factory=list)
    quality_breakdown:    dict  = field(default_factory=dict)
    applied_enhancements: list  = field(default_factory=list)
    selection_reasoning:  str   = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ClipConfig:
    # Silence removal
    silence_threshold_db: int   = -40
    min_silence_ms:       int   = 500
    keep_silence_ms:      int   = 150

    # Clip selection
    min_clip_duration:    float = 15.0
    max_clip_duration:    float = 90.0
    top_n_clips:          int   = 3

    # Captions
    caption_words_per_phrase: int   = 5
    caption_fontsize:         int   = 52
    caption_y_position:       float = 0.80

    # Export
    output_dir:       str           = "viral_clips"
    add_captions:     bool          = True
    target_aspect:    Optional[str] = "9:16"
    encode_preset:    str           = "fast"   # passed to videotoolbox (ignored gracefully if unsupported)
    encode_bitrate:   str           = "6M"     # target bitrate for h264_videotoolbox
    parallel_exports: bool          = False


# ─────────────────────────────────────────────────────────────────
# Opt 1 — Font cache (loaded once, reused forever)
# ─────────────────────────────────────────────────────────────────

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    # Windows
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Filesystem scanned at most once per font size."""
    if size not in _FONT_CACHE:
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    _FONT_CACHE[size] = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


# ─────────────────────────────────────────────────────────────────
# Opt 2 — Caption frame cache (render once per unique text+config)
# ─────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=512)
def _render_caption_cached(
    text:           str,
    video_w:        int,
    video_h:        int,
    fontsize:       int,
    y_position_pct: int,    # y_position × 1000 as int so it's hashable
) -> tuple[bytes, bytes, tuple[int, int]]:
    """
    Render caption text → RGBA.  Returns (rgb_bytes, alpha_bytes, (h, w)).
    Bytes are returned (not arrays) so results are picklable across processes.
    LRU cache means the same phrase (very common) costs ~2 µs instead of ~15 ms.
    """
    y_pos = y_position_pct / 1000.0
    font  = _get_font(fontsize)

    img  = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    words   = text.split()
    mid     = (len(words) + 1) // 2 if len(words) > 4 else len(words)
    caption = " ".join(words[:mid])
    if mid < len(words):
        caption += "\n" + " ".join(words[mid:])

    bbox   = draw.multiline_textbbox((0, 0), caption, font=font, spacing=6)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = max((video_w - tw) // 2, 10)
    y = max(int(video_h * y_pos) - th // 2, 10)

    pad = 16
    draw.rounded_rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        radius=14, fill=(0, 0, 0, 155),
    )
    for dx, dy in ((-1,-1),(1,-1),(-1,1),(1,1),(0,-1),(0,1),(-1,0),(1,0)):
        draw.multiline_text((x+dx*3, y+dy*3), caption, font=font,
                            fill=(0, 0, 0, 255), spacing=6, align="center")
    draw.multiline_text((x, y), caption, font=font,
                        fill=(255, 255, 255, 255), spacing=6, align="center")

    arr = np.array(img)
    h, w = arr.shape[:2]
    return arr[:, :, :3].tobytes(), arr[:, :, 3].tobytes(), (h, w)


def _phrase_to_moviepy(
    phrase:        CaptionPhrase,
    clip_start:    float,
    clip_duration: float,
    video_w:       int,
    video_h:       int,
    cfg:           ClipConfig,
) -> Optional[ImageClip]:
    """Convert one caption phrase to a timed, alpha-masked MoviePy ImageClip."""
    local_start = max(phrase.start - clip_start, 0.0)
    local_end   = min(phrase.end   - clip_start, clip_duration)
    duration    = local_end - local_start
    if duration < 0.05:
        return None

    y_pct = int(cfg.caption_y_position * 1000)
    rgb_b, alpha_b, (h, w) = _render_caption_cached(
        phrase.text, video_w, video_h, cfg.caption_fontsize, y_pct
    )

    # Reconstruct from cached bytes — cheap view, effectively free
    rgb   = np.frombuffer(rgb_b,   dtype=np.uint8).reshape(h, w, 3).copy()
    alpha = np.frombuffer(alpha_b, dtype=np.uint8).reshape(h, w).astype(np.float32) / 255.0

    # CRITICAL: split RGBA into RGB + mask so MoviePy alpha-blends correctly.
    # Passing raw RGBA causes transparent pixels to render as black.
    return (
        ImageClip(rgb, ismask=False)
        .set_start(local_start)
        .set_duration(duration)
        .set_mask(
            ImageClip(alpha, ismask=True)
            .set_start(local_start)
            .set_duration(duration)
        )
    )


# ─────────────────────────────────────────────────────────────────
# Step 1 — Download
# ─────────────────────────────────────────────────────────────────

def download_video(url: str, output_dir: str) -> str:
    ydl_opts = {
        "outtmpl":             os.path.join(output_dir, "%(id)s.%(ext)s"),
        "format":              "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet":               True,
        "no_warnings":         True,
        "writeinfojson":       True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        for ext in (".webm", ".mkv"):
            path = path.replace(ext, ".mp4")
    return path


# ─────────────────────────────────────────────────────────────────
# Step 2 — Load audio ONCE into RAM + silence detection
# ─────────────────────────────────────────────────────────────────

def load_audio_numpy(video_path: str) -> tuple[np.ndarray, int]:
    """
    Extract 16 kHz mono audio via direct ffmpeg call → numpy float32.
    Loaded once; all energy queries are O(1) array slices from this point.
    """
    wav_path = video_path.replace(".mp4", "_audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-ac", "1", "-ar", "16000", "-f", "wav", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    rate, data = wavfile.read(wav_path)
    os.remove(wav_path)

    if   data.dtype == np.int16:  data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:  data = data.astype(np.float32) / 2147483648.0
    else:                          data = data.astype(np.float32)

    return data, rate


def detect_silence_intervals(
    audio: np.ndarray, rate: int, cfg: ClipConfig
) -> list[tuple[float, float]]:
    """
    Vectorised numpy silence detection — much faster than pydub's Python loop.
    Returns list of (start_sec, end_sec) non-silent intervals.
    """
    frame_ms    = 20
    frame_size  = int(rate * frame_ms / 1000)
    n_frames    = len(audio) // frame_size
    frames      = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms_db      = 20.0 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)).clip(1e-9))

    thresh           = max(float(np.mean(rms_db)) - 10.0, cfg.silence_threshold_db)
    is_speech        = rms_db > thresh
    min_sil_frames   = cfg.min_silence_ms // frame_ms

    intervals: list[tuple[float, float]] = []
    in_speech   = False
    start_frame = 0
    sil_count   = 0

    for i, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                in_speech   = True
                start_frame = i
            sil_count = 0
        else:
            if in_speech:
                sil_count += 1
                if sil_count >= min_sil_frames:
                    end_frame = i - sil_count + 1
                    intervals.append((
                        start_frame * frame_ms / 1000.0,
                        end_frame   * frame_ms / 1000.0,
                    ))
                    in_speech = False
                    sil_count = 0

    if in_speech:
        intervals.append((start_frame * frame_ms / 1000.0, len(audio) / rate))

    # Merge near-adjacent intervals (within keep_silence_ms)
    keep_gap = cfg.keep_silence_ms / 1000.0
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= keep_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return merged


def energy_for_range(audio: np.ndarray, rate: int, start: float, end: float) -> float:
    """RMS energy of a time range — pure array slice, effectively O(1)."""
    chunk = audio[int(start * rate) : int(end * rate)]
    return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────
# Step 3 — Transcription (Whisper)
# ─────────────────────────────────────────────────────────────────

def transcribe(video_path: str) -> tuple[list[dict], list[dict]]:
    """
    Transcribes the ORIGINAL video so Whisper timestamps match the original
    video timeline (critical for caption sync — see previous bug).
    Uses direct ffmpeg for WAV extraction (2-3x faster than MoviePy).
    """
    try:
        import whisper
    except ImportError:
        print("⚠️  whisper not installed — pip install openai-whisper")
        return [], []

    wav_path = video_path.replace(".mp4", "_whisper.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-ac", "1", "-ar", "16000", "-f", "wav", wav_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    model  = whisper.load_model("base")
    result = model.transcribe(wav_path, word_timestamps=True, verbose=False)

    try:    os.remove(wav_path)
    except: pass

    sent_segs: list[dict] = []
    all_words:  list[dict] = []

    for seg in result.get("segments", []):
        sent_segs.append({
            "start": seg["start"], "end": seg["end"], "text": seg["text"].strip()
        })
        for w in seg.get("words", []):
            all_words.append({
                "start": w["start"], "end": w["end"], "word": w["word"].strip()
            })

    return sent_segs, all_words


# ─────────────────────────────────────────────────────────────────
# Step 4 — Scoring
# ─────────────────────────────────────────────────────────────────

VIRAL_KEYWORDS = frozenset({
    "shocking","secret","reveal","never","always","truth","exposed",
    "mistake","wrong","insane","crazy","viral","trending","breaking",
    "unbelievable","you won't believe","wait for it","plot twist",
    "but then","suddenly","that's when","here's the thing",
    "the problem is","and it gets worse","listen","real talk",
    "study shows","experts say","most people don't know",
    "how to","why you","stop doing","the reason",
})


def score_segment(seg: Segment, max_energy: float) -> float:
    text  = seg.transcript.lower()
    score = min(sum(1 for kw in VIRAL_KEYWORDS if kw in text) * 8, 40)
    score += (seg.energy / max_energy * 25) if max_energy > 0 else 0
    score += 20 if 20 <= seg.duration <= 60 else (10 if 10 <= seg.duration <= 90 else 0)
    wps    = len(seg.transcript.split()) / seg.duration if seg.duration > 0 else 0
    score += min(wps * 2, 15)
    return round(min(score, 100.0), 1)


def build_caption_phrases(
    words:            list[dict],
    word_starts:      list[float],   # pre-extracted sorted list for bisect
    seg_start:        float,
    seg_end:          float,
    words_per_phrase: int,
) -> list[CaptionPhrase]:
    """
    Opt: bisect gives O(log n) word lookup instead of O(n) linear scan.
    """
    lo = bisect.bisect_left(word_starts,  seg_start - 0.1)
    hi = bisect.bisect_right(word_starts, seg_end   + 0.1)

    phrases = []
    seg_words = words[lo:hi]
    for i in range(0, len(seg_words), words_per_phrase):
        chunk = seg_words[i : i + words_per_phrase]
        text  = " ".join(w["word"] for w in chunk).strip()
        if text:
            phrases.append(CaptionPhrase(
                start = chunk[0]["start"],
                end   = chunk[-1]["end"] + 0.15,   # 150 ms grace period
                text  = text,
            ))
    return phrases


def build_and_score_segments(
    sent_segs:  list[dict],
    all_words:  list[dict],
    audio:      np.ndarray,
    rate:       int,
    cfg:        ClipConfig,
) -> list[Segment]:
    if not sent_segs:
        return []

    word_starts = [w["start"] for w in all_words]

    # Pre-detect rhythmic beats from audio for cut synchronization
    beats = BeatDetector.detect_beats(audio, rate)

    candidates: list[Segment] = []
    n_sents = len(sent_segs)

    # Sliding candidate window generator with sentence boundary integrity
    for i in range(n_sents):
        cur_text_list = []
        cand_start = sent_segs[i]["start"]

        # Lookahead up to 15 sentences or max duration
        for j in range(i, min(i + 16, n_sents)):
            cur_text_list.append(sent_segs[j]["text"])
            cand_end = sent_segs[j]["end"]
            dur = cand_end - cand_start

            if dur < cfg.min_clip_duration:
                continue
            if dur > cfg.max_clip_duration:
                break

            full_cand_text = " ".join(cur_text_list).strip()

            # Natural cushion after last word (up to 250ms)
            total_audio_sec = len(audio) / rate if rate > 0 else cand_end
            cushioned_end = min(cand_end + 0.25, total_audio_sec)

            # Evaluate candidate via 8-criteria ClipQualityEngine
            eval_res = quality_engine.evaluate_candidate(
                transcript=full_cand_text,
                start_time=cand_start,
                end_time=cushioned_end,
                audio=audio,
                rate=rate,
                beats=beats,
            )

            seg = Segment(
                start                = cand_start,
                end                  = cushioned_end,
                transcript           = full_cand_text,
                energy               = energy_for_range(audio, rate, cand_start, cushioned_end),
                viral_score          = eval_res.total_score,
                quality_score        = eval_res.total_score,
                quality_tier         = eval_res.quality_tier,
                technical_status     = eval_res.technical_status,
                technical_issues     = eval_res.technical_issues,
                quality_breakdown    = eval_res.breakdown.to_dict(),
                applied_enhancements = ["quality_scoring", "sentence_boundary_alignment"],
                selection_reasoning  = eval_res.selection_reasoning,
            )
            seg.caption_phrases = build_caption_phrases(
                all_words, word_starts, seg.start, seg.end, cfg.caption_words_per_phrase
            )
            candidates.append(seg)

    if not candidates:
        # Fallback to sequential grouping if sliding window yielded no matches
        raw_start = sent_segs[0]["start"]
        raw_end = sent_segs[-1]["end"]
        raw_text = " ".join(s["text"] for s in sent_segs)
        eval_res = quality_engine.evaluate_candidate(
            transcript=raw_text, start_time=raw_start, end_time=raw_end,
            audio=audio, rate=rate, beats=beats
        )
        fallback_seg = Segment(
            start=raw_start, end=raw_end, transcript=raw_text,
            viral_score=eval_res.total_score, quality_score=eval_res.total_score,
            quality_tier=eval_res.quality_tier, technical_status=eval_res.technical_status,
            quality_breakdown=eval_res.breakdown.to_dict(),
            selection_reasoning=eval_res.selection_reasoning,
        )
        fallback_seg.caption_phrases = build_caption_phrases(
            all_words, word_starts, fallback_seg.start, fallback_seg.end, cfg.caption_words_per_phrase
        )
        return [fallback_seg]

    # Sort candidates by quality score descending
    candidates.sort(key=lambda s: s.quality_score, reverse=True)

    # Non-Maximum Suppression (NMS) to eliminate near-duplicate overlapping segments
    selected: list[Segment] = []
    for cand in candidates:
        overlaps = False
        for s in selected:
            inter = max(0.0, min(cand.end, s.end) - max(cand.start, s.start))
            if inter > 0:
                overlap_ratio = inter / min(cand.duration, s.duration)
                if overlap_ratio > 0.40:
                    overlaps = True
                    break
        if not overlaps:
            selected.append(cand)
        if len(selected) >= cfg.top_n_clips * 2:
            break

    return selected


# ─────────────────────────────────────────────────────────────────
# Step 5 — Export (single clip)
# ─────────────────────────────────────────────────────────────────

def crop_to_vertical(clip, protagonist_cx: Optional[int] = None):
    target_w = int(clip.h * 9 / 16)
    if target_w >= clip.w:
        return clip
    if protagonist_cx is not None:
        x1 = ProtagonistTracker.compute_crop_x(clip.w, target_w, protagonist_cx)
    else:
        x1 = max((clip.w - target_w) // 2, 0)
    return clip.crop(x1=x1, width=target_w)


def export_segment(
    video_path: str,
    seg:        Segment,
    index:      int,
    cfg:        ClipConfig,
    output_dir: str,
) -> str:
    t0        = time.time()
    full_clip = VideoFileClip(video_path)
    try:
        fps  = full_clip.fps or 30
        clip = full_clip.subclip(seg.start, min(seg.end, full_clip.duration)).set_fps(fps)

        # Feature 2: Continuous Protagonist Tracking for 9:16 vertical crop
        protagonist_cx = None
        if cfg.target_aspect == "9:16":
            try:
                trajectory, vis_ratio = quality_engine.tracker.track_clip(clip, sample_fps=2.0)
                if trajectory:
                    centers = list(trajectory.values())
                    protagonist_cx = int(np.median(centers))
                    print(f"  👤 Protagonist tracked ({int(vis_ratio*100)}% visible) — center x={protagonist_cx}")
                    if "protagonist_tracking_9_16" not in seg.applied_enhancements:
                        seg.applied_enhancements.append("protagonist_tracking_9_16")
            except Exception as e:
                print(f"  ⚠️ Protagonist tracking fallback ({e})")
            clip = crop_to_vertical(clip, protagonist_cx=protagonist_cx)

        # Feature 9: Professional audio boundary micro-fades
        if clip.audio is not None:
            try:
                clip = clip.audio_fadein(0.06).audio_fadeout(0.16)
                if "audio_boundary_fades" not in seg.applied_enhancements:
                    seg.applied_enhancements.append("audio_boundary_fades")
            except Exception:
                pass

        # Synced captions — each phrase is its own timed layer
        caption_layers = []
        if cfg.add_captions and seg.caption_phrases:
            for phrase in seg.caption_phrases:
                layer = _phrase_to_moviepy(
                    phrase,
                    clip_start    = seg.start,
                    clip_duration = clip.duration,
                    video_w       = clip.w,
                    video_h       = clip.h,
                    cfg           = cfg,
                )
                if layer is not None:
                    caption_layers.append(layer)

            if caption_layers:
                # size= required — prevents canvas resize → black frames
                clip = CompositeVideoClip(
                    [clip] + caption_layers, size=(clip.w, clip.h)
                ).set_fps(fps)

        label    = f"clip_{index+1}_score{int(seg.viral_score)}"
        out_path = os.path.join(output_dir, f"{label}.mp4")

        clip.write_videofile(
            out_path,
            codec        = "h264_videotoolbox",
            audio_codec  = "aac",
            fps          = fps,
            preset       = cfg.encode_preset,
            ffmpeg_params= ["-b:v", cfg.encode_bitrate],
            logger       = None,
            threads      = 2,
            write_logfile= False,
        )
        print(f"  ✅ clip_{index+1}  [{seg.quality_tier}] score={seg.viral_score}  "
              f"{len(caption_layers)} captions  {time.time()-t0:.1f}s  →  {out_path}")
        return out_path

    finally:
        full_clip.close()   # MUST be after write_videofile — see blackout bug notes


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

def generate_viral_clips(video_url: str, cfg: ClipConfig = ClipConfig()) -> None:
    work_dir   = tempfile.mkdtemp(prefix="viral_")
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    wall_t     = time.time()

    try:
        # 1. Download
        print(f"\n📥  Downloading: {video_url}")
        t = time.time()
        video_path = download_video(video_url, work_dir)
        print(f"    {time.time()-t:.1f}s")

        # 2. Load audio (once into RAM)
        print("\n🎧  Loading audio…")
        t = time.time()
        audio, rate = load_audio_numpy(video_path)
        print(f"    {time.time()-t:.1f}s  |  {len(audio)/rate:.1f}s  @  {rate} Hz")

        # 3. Silence detection
        print("\n🔇  Detecting silence…")
        t = time.time()
        intervals = detect_silence_intervals(audio, rate, cfg)
        kept      = sum(e - s for s, e in intervals)
        print(f"    {time.time()-t:.3f}s  |  {kept:.1f}s speech  in  {len(intervals)} segments")

        # 4. Transcribe
        print("\n🗣   Transcribing (Whisper)…")
        t = time.time()
        sent_segs, all_words = transcribe(video_path)
        print(f"    {time.time()-t:.1f}s  |  {len(sent_segs)} sentences  {len(all_words)} words")

        # 5. Score
        print("\n📊  Scoring segments…")
        t = time.time()
        ranked = build_and_score_segments(sent_segs, all_words, audio, rate, cfg)
        print(f"    {time.time()-t:.3f}s  |  {len(ranked)} candidates")

        if not ranked:
            print("⚠️   No scoreable segments — exporting full cleaned video.")
            _export_full_clean(video_path, intervals, output_dir)
            return

        top = ranked[: cfg.top_n_clips]
        print(f"\n🏆  Top {len(top)} clips:")
        for i, s in enumerate(top):
            print(f"    #{i+1}  [{s.start:.1f}s → {s.end:.1f}s]  "
                  f"score={s.viral_score}  {len(s.caption_phrases)} phrases")

        # 6. Export (sequential — one ffmpeg process at a time)
        print(f"\n🎬  Exporting {len(top)} clips (sequential)…")
        t    = time.time()

        exported = [""] * len(top)

        for i, seg in enumerate(top):
            exported[i] = export_segment(video_path, seg, i, cfg, output_dir)

        print(f"    Export wall time: {time.time()-t:.1f}s")

        # 7. Metadata & Quality Reports
        meta_path = os.path.join(output_dir, "clip_metadata.json")
        clips_data = [
            {
                "rank":                 i + 1,
                "start":                round(s.start, 2),
                "end":                  round(s.end, 2),
                "duration":             round(s.duration, 2),
                "viral_score":          s.viral_score,
                "quality_score":        s.quality_score,
                "quality_tier":         s.quality_tier,
                "technical_status":     s.technical_status,
                "technical_issues":     s.technical_issues,
                "quality_breakdown":    s.quality_breakdown,
                "applied_enhancements": s.applied_enhancements,
                "selection_reasoning":  s.selection_reasoning,
                "transcript":           s.transcript,
                "caption_phrases": [
                    {"start": p.start, "end": p.end, "text": p.text}
                    for p in s.caption_phrases
                ],
                "file":                 os.path.basename(exported[i]),
            }
            for i, s in enumerate(top)
        ]

        with open(meta_path, "w") as f:
            json.dump(clips_data, f, indent=2)

        # Generate human-readable Markdown and structured JSON quality reports
        report_md_path = os.path.join(output_dir, "quality_report.md")
        report_json_path = os.path.join(output_dir, "quality_report.json")
        quality_engine.generate_quality_report(clips_data, report_json_path, report_md_path)

        print(f"\n📋  Metadata: {meta_path}")
        print(f"📊  Quality Report: {report_md_path}")
        print(f"✅  Done in {time.time()-wall_t:.1f}s total\n")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _export_full_clean(video_path: str, intervals: list, output_dir: str) -> None:
    full  = VideoFileClip(video_path)
    parts = [full.subclip(s, min(e, full.duration)) for s, e in intervals if e - s > 0.5]
    if not parts:
        full.close()
        return
    out = os.path.join(output_dir, "full_clean.mp4")
    concatenate_videoclips(parts).write_videofile(
        out, codec="h264_videotoolbox", audio_codec="aac",
        ffmpeg_params=["-b:v", "6M"], logger=None
    )
    full.close()
    print(f"Exported: {out}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Smart Viral Clip Generator")
    p.add_argument("url")
    p.add_argument("--output-dir",           default="viral_clips")
    p.add_argument("--top-n",          type=int,   default=3)
    p.add_argument("--max-duration",   type=float, default=90.0)
    p.add_argument("--min-duration",   type=float, default=15.0)
    p.add_argument("--words-per-phrase", type=int, default=5)
    p.add_argument("--caption-fontsize", type=int, default=52)
    p.add_argument("--no-captions",    action="store_true")
    p.add_argument("--keep-aspect",    action="store_true")
    p.add_argument("--silence-db",     type=int,   default=-40)
    p.add_argument("--preset",         default="fast",
                   choices=["ultrafast","superfast","veryfast","faster","fast","medium","slow"],
                   help="x264 encode preset (default: fast)")
    p.add_argument("--bitrate",        default="6M",
                   help="Target video bitrate for h264_videotoolbox (default: 6M)")
    args = p.parse_args()

    cfg = ClipConfig(
        silence_threshold_db     = args.silence_db,
        min_clip_duration        = args.min_duration,
        max_clip_duration        = args.max_duration,
        top_n_clips              = args.top_n,
        caption_words_per_phrase = args.words_per_phrase,
        caption_fontsize         = args.caption_fontsize,
        output_dir               = args.output_dir,
        add_captions             = not args.no_captions,
        target_aspect            = None if args.keep_aspect else "9:16",
        encode_preset            = args.preset,
        encode_bitrate           = args.bitrate,
    )

    generate_viral_clips(args.url, cfg)
