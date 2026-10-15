"""
post_process_m3.py — Viral Clip Post-Processing Enhancement Engine
===================================================================
Reads existing clips from viral_clips/ and clip_metadata.json.
Transforms them into high-retention, monetization-safer, platform-
optimized Shorts. Optimized for Apple Silicon M3 Pro.

Does NOT:  download videos · run Whisper · rescore segments · touch generator
Does ONLY: read → enhance → export

Requirements:
    pip install moviepy pydub gtts opencv-python numpy Pillow scipy

Usage:
    python post_process_m3.py
    python post_process_m3.py --hevc --bitrate 8M
    python post_process_m3.py --no-commentary --no-zoom --square
    python post_process_m3.py --enable-loop --no-label
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import AudioArrayClip
from pydub import AudioSegment


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

@dataclass
class PostConfig:
    # I/O
    input_dir:      str = "viral_clips"
    output_dir:     str = "viral_clips_enhanced"
    metadata_file:  str = "viral_clips/clip_metadata.json"

    # Encoder (Apple Silicon)
    hevc:           bool = False          # True = hevc_videotoolbox, False = h264_videotoolbox
    bitrate:        str  = "6M"           # auto-adjusted for motion; override via CLI
    threads:        int  = 2             # fixed — one Media Engine at a time
    pixel_format:   str  = "yuv420p"

    # Features
    commentary:     bool = True
    audio_duck:     bool = True
    label:          bool = True
    zoom:           bool = True
    enable_loop:    bool = False
    square:         bool = False          # export additional 1:1 version

    # Hook
    hook_max_secs:  float = 7.0
    duck_level:     float = 0.30          # original audio volume during commentary

    # Label text
    label_text:     str = "Commentary & Analysis"
    source_text:    str = "Source: Original YouTube Video"


# ─────────────────────────────────────────────────────────────────
# Font cache — loaded once, never reloaded
# ─────────────────────────────────────────────────────────────────

_FONT_CACHE: dict[tuple, ImageFont.FreeTypeFont] = {}

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Return cached font — filesystem scanned at most once per (size, bold) pair."""
    key = (size, bold)
    if key not in _FONT_CACHE:
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    _FONT_CACHE[key] = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
        else:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


# ─────────────────────────────────────────────────────────────────
# Hook generation
# ─────────────────────────────────────────────────────────────────

HOOK_TEMPLATES = {
    "curiosity": [
        "You won't believe what happens here…",
        "This moment changes everything. Watch.",
        "Most people miss this completely.",
        "Here's what nobody talks about.",
        "This is why everyone is sharing this clip.",
    ],
    "shock": [
        "This is absolutely insane.",
        "I had to rewatch this three times.",
        "Nobody saw this coming.",
        "This changes everything we thought we knew.",
        "Wait until you see what happens next.",
    ],
    "authority": [
        "Here's our analysis of this viral moment.",
        "Breaking this down so you understand why it matters.",
        "This is our commentary on a key moment.",
        "Let's unpack why this is so significant.",
        "Our take on what you're about to see.",
    ],
    "question": [
        "What would you do in this situation?",
        "Did you catch what just happened here?",
        "Can you spot what makes this so powerful?",
        "How does this make you feel?",
        "What do you think happens next?",
    ],
}


def generate_hook_text(transcript: str) -> tuple[str, str]:
    """
    Generate a hook sentence from transcript context.
    Returns (hook_text, style_used).
    Randomizes style for variety across clips.
    """
    style = random.choice(list(HOOK_TEMPLATES.keys()))
    hook  = random.choice(HOOK_TEMPLATES[style])

    # If transcript has enough words, prepend a context anchor
    words = transcript.split()
    if len(words) >= 6:
        # Use first 4 meaningful words as a micro-teaser
        snippet = " ".join(words[:4]).strip(".,!?")
        if style in ("curiosity", "shock"):
            hook = f'"{snippet}…" — {hook}'

    return hook, style


def generate_hook_audio(text: str, max_secs: float, tmp_dir: str) -> Optional[str]:
    """
    Synthesise hook voiceover with gTTS.
    Returns path to WAV file, or None if gTTS unavailable.
    Clips audio to max_secs automatically.
    """
    try:
        from gtts import gTTS
    except ImportError:
        print("  ⚠️  gTTS not installed — skipping commentary voice (pip install gtts)")
        return None

    mp3_path = os.path.join(tmp_dir, "hook.mp3")
    wav_path = os.path.join(tmp_dir, "hook.wav")

    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(mp3_path)

    # Convert to WAV and trim to max_secs
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio[: int(max_secs * 1000)]
    audio.export(wav_path, format="wav")
    os.remove(mp3_path)

    return wav_path


# ─────────────────────────────────────────────────────────────────
# Audio ducking
# ─────────────────────────────────────────────────────────────────

def apply_audio_ducking(
    original_audio,        # MoviePy AudioClip
    hook_audio_path: str,
    duck_level: float,
    fade_secs: float = 0.3,
) -> CompositeAudioClip:
    """
    Reduce original audio to duck_level during hook commentary,
    with smooth fade in/out. Restores to 100% after hook ends.
    Returns a CompositeAudioClip ready to attach to the video.
    """
    hook_clip   = AudioFileClip(hook_audio_path)
    hook_dur    = hook_clip.duration
    fade_secs   = min(fade_secs, hook_dur / 3)

    # Build volume envelope for original audio as numpy array
    total_dur   = original_audio.duration
    sample_rate = 44100
    n_samples   = int(total_dur * sample_rate)
    envelope    = np.ones(n_samples, dtype=np.float32)

    hook_end    = hook_dur
    fade_in_end = int(fade_secs * sample_rate)
    duck_start  = fade_in_end
    duck_end    = int(hook_end * sample_rate)
    fade_out_end= min(duck_end + int(fade_secs * sample_rate), n_samples)

    # Ramp down
    envelope[:fade_in_end] = np.linspace(1.0, duck_level, fade_in_end)
    # Hold duck
    envelope[duck_start:duck_end] = duck_level
    # Ramp back up
    if duck_end < fade_out_end:
        envelope[duck_end:fade_out_end] = np.linspace(duck_level, 1.0,
                                                        fade_out_end - duck_end)

    def make_ducked_frame(t):
        # MoviePy may pass either a scalar float or a numpy array of timestamps.
        # np.atleast_1d ensures we always work with an array so indexing and
        # broadcasting work correctly whether t is scalar or vectorised.
        t_arr   = np.atleast_1d(np.asarray(t, dtype=np.float64))
        indices = (t_arr * sample_rate).astype(np.int64)
        indices = np.clip(indices, 0, n_samples - 1)   # always in-bounds
        gains   = envelope[indices]                     # (n_times,)

        frame = original_audio.get_frame(t)             # (n_times, n_ch) or (n_ch,)
        frame = np.atleast_2d(frame)                    # normalise to 2-D

        # gains[:, np.newaxis] → (n_times, 1) broadcasts cleanly over channels
        return (frame * gains[:, np.newaxis]).astype(np.float32)

    from moviepy.audio.AudioClip import AudioClip
    ducked = AudioClip(make_ducked_frame, duration=total_dur, fps=sample_rate)

    return CompositeAudioClip([ducked, hook_clip])


# ─────────────────────────────────────────────────────────────────
# Label overlay (monetization safety)
# ─────────────────────────────────────────────────────────────────

def render_label_frame(
    text: str,
    video_w: int,
    video_h: int,
    font_size: int = 28,
    position: str = "top",       # "top" or "bottom"
    bg_alpha: int = 180,
) -> np.ndarray:
    """
    Render a persistent semi-transparent label bar.
    Returns RGBA numpy array (H, W, 4). Cached by caller via ImageClip.
    """
    font = get_font(font_size)
    img  = Image.new("RGBA", (video_w, video_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    bbox   = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad    = 10

    bar_h = th + pad * 2
    bar_y = 0 if position == "top" else video_h - bar_h

    draw.rectangle([0, bar_y, video_w, bar_y + bar_h], fill=(0, 0, 0, bg_alpha))
    x = (video_w - tw) // 2
    y = bar_y + pad
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    return np.array(img)


def make_label_clip(text: str, video_w: int, video_h: int,
                    duration: float, position: str = "top") -> ImageClip:
    """Build a full-duration persistent label ImageClip (alpha-masked)."""
    rgba  = render_label_frame(text, video_w, video_h, position=position)
    rgb   = rgba[:, :, :3]
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0

    return (
        ImageClip(rgb, ismask=False)
        .set_duration(duration)
        .set_mask(ImageClip(alpha, ismask=True).set_duration(duration))
    )


# ─────────────────────────────────────────────────────────────────
# Zoom effect (retention boost) — FFmpeg filter, not Python frames
# ─────────────────────────────────────────────────────────────────

def apply_zoom_effect(clip, zoom_ratio: float = 1.03, zoom_duration: float = 3.0):
    """
    Zoom is now handled entirely inside FFmpeg via a zoompan filter injected
    in export_clip(). This function intentionally returns the clip unchanged
    so that no Python/PIL per-frame processing occurs.

    The zoom filter expression used at export time:
        scale=iw*1.03:ih*1.03, zoompan=z='min(zoom+0.0010,1.03)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'
    replicates the original 3% Ken Burns zoom over 3 seconds without any
    Python frame loop, PIL resize, or numpy array manipulation.
    """
    return clip   # zoom injected as ffmpeg vf filter at export time


# ─────────────────────────────────────────────────────────────────
# Pattern interrupt flash
# ─────────────────────────────────────────────────────────────────

def make_flash_clip(video_w: int, video_h: int, duration: float = 0.15) -> ColorClip:
    """Brief white flash at 3 seconds — pattern interrupt to reset attention."""
    return ColorClip(size=(video_w, video_h), color=[255, 255, 255]).set_duration(duration)


# ─────────────────────────────────────────────────────────────────
# Aspect ratio enforcement
# ─────────────────────────────────────────────────────────────────

def ensure_9_16(clip, face_detect: bool = True):
    """
    Ensure clip is 9:16. If already correct, returns as-is (no re-encode waste).
    Attempts face-detection centre-crop if OpenCV available.
    Falls back to geometric centre crop. Always validates crop bounds.
    """
    w, h     = clip.w, clip.h
    target_w = int(h * 9 / 16)

    # ── Safety guard (Bug 3) ─────────────────────────────────────
    # If target_w is invalid (zero, negative, or wider than source),
    # cropping would produce a 1-pixel or nonsense result — skip entirely.
    if target_w <= 0 or target_w > w:
        return clip

    if w == target_w:
        return clip   # already 9:16 — no crop needed

    # ── Face detection (Bug 2) ────────────────────────────────────
    if face_detect:
        try:
            import cv2
            sample_t  = clip.duration * 0.25
            frame_rgb = clip.get_frame(sample_t)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade      = cv2.CascadeClassifier(cascade_path)
            faces        = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                face_cx = fx + fw // 2
                # Clamp x1 so crop window always fits within frame (Bug 2)
                x1 = face_cx - target_w // 2
                x1 = max(0, min(x1, w - target_w))   # never negative, never overflows

                # Final sanity check — reject if resulting width would be wrong
                if x1 >= 0 and (x1 + target_w) <= w and target_w > 1:
                    print(f"  👤 Face detected — smart crop at x={x1}")
                    return clip.crop(x1=x1, width=target_w)
                else:
                    print(f"  ⚠️  Face crop invalid (x1={x1}, target_w={target_w}) — centre crop fallback")

        except ImportError:
            pass   # OpenCV not installed — fall through
        except Exception as e:
            print(f"  ⚠️  Face detection failed ({e}) — using centre crop")

    # ── Geometric centre crop fallback ────────────────────────────
    x1 = max((w - target_w) // 2, 0)
    x1 = min(x1, w - target_w)   # ensure right edge never exceeds frame
    return clip.crop(x1=x1, width=target_w)


def make_square(clip):
    """Centre-crop to 1:1 for Instagram feed."""
    size = min(clip.w, clip.h)
    x1   = (clip.w - size) // 2
    y1   = (clip.h - size) // 2
    return clip.crop(x1=x1, y1=y1, width=size, height=size)


# ─────────────────────────────────────────────────────────────────
# Seamless loop ending
# ─────────────────────────────────────────────────────────────────

def apply_loop_ending(clip, crossfade_secs: float = 0.5):
    """
    Cross-fade last crossfade_secs into the first crossfade_secs to
    create a seamless loop for TikTok/Reels repeat plays.
    """
    if clip.duration <= crossfade_secs * 2:
        return clip

    tail  = clip.subclip(clip.duration - crossfade_secs).crossfadeout(crossfade_secs)
    head  = clip.subclip(0, crossfade_secs).crossfadein(crossfade_secs)
    body  = clip.subclip(0, clip.duration - crossfade_secs)

    return concatenate_videoclips([body, tail])


# ─────────────────────────────────────────────────────────────────
# Motion-adaptive bitrate
# ─────────────────────────────────────────────────────────────────

def estimate_motion(clip, n_samples: int = 6) -> float:
    """
    Sample n_samples frame pairs and compute mean absolute difference.
    Returns 0.0 (static) → 1.0 (high motion). Fast — no optical flow.
    """
    step     = clip.duration / (n_samples + 1)
    diffs    = []
    prev     = None

    for i in range(1, n_samples + 1):
        t     = step * i
        frame = clip.get_frame(t).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(frame - prev))) / 255.0)
        prev = frame

    return float(np.mean(diffs)) if diffs else 0.0


def adaptive_bitrate(clip, cfg: PostConfig) -> str:
    """
    Choose bitrate between 6M and 10M based on estimated motion.
    High motion clips need more bits to avoid blocking artefacts.
    """
    if cfg.bitrate != "auto":
        return cfg.bitrate   # user override — respect it

    motion = estimate_motion(clip)
    if motion > 0.15:
        br = "10M"
    elif motion > 0.07:
        br = "8M"
    else:
        br = "6M"

    print(f"  📊 Motion={motion:.3f} → bitrate={br}")
    return br


# ─────────────────────────────────────────────────────────────────
# Core export
# ─────────────────────────────────────────────────────────────────

def _build_zoom_filter(
    clip_w: int,
    clip_h: int,
    fps: float,
    zoom_ratio: float = 1.03,
    zoom_duration: float = 3.0,
) -> str:
    """
    Build an ffmpeg video-filter string that replicates the original Ken Burns
    zoom entirely inside the ffmpeg process — no Python frame loop.

    Strategy — two chained filters:
    1. scale: upscale the frame to zoom_ratio size so zoompan has pixels to
       work with and never samples outside the frame boundary.
    2. zoompan: animate zoom from 1.0 → zoom_ratio over zoom_duration seconds,
       then hold at zoom_ratio. Centre-crop back to original dimensions.

    zoompan parameters:
      z  — zoom expression per output frame.
           Increments by (zoom_ratio-1)/zoom_frames each frame until reaching
           zoom_ratio, then holds. 'zoom' is zoompan's built-in running value.
      d  — output frame duration (d=1 = every input frame, no slowdown).
      x/y — centre-crop: always take from the centre of the scaled frame.
      s  — output size locked to original clip dimensions.
      fps — must match the clip fps so zoompan frame counting is correct.
    """
    zoom_frames = int(fps * zoom_duration)
    increment   = (zoom_ratio - 1.0) / max(zoom_frames, 1)

    # Upscale step: give zoompan headroom to crop into
    scale_filter = f"scale={int(clip_w * zoom_ratio)}:{int(clip_h * zoom_ratio)}"

    # zoompan zoom expression: ramp up then hold
    z_expr = f"'min(zoom+{increment:.6f},{zoom_ratio:.6f})'"
    x_expr = f"'iw/2-(iw/zoom/2)'"
    y_expr = f"'ih/2-(ih/zoom/2)'"
    size   = f"{clip_w}x{clip_h}"

    zoompan_filter = (
        f"zoompan=z={z_expr}:d=1"
        f":x={x_expr}:y={y_expr}"
        f":s={size}:fps={fps:.3f}"
    )

    return f"{scale_filter},{zoompan_filter}"


def _ffmpeg_write_no_zoom(
    composite: VideoFileClip,
    out_path:  str,
    bitrate:   str,
    codec:     str,
    fps:       float,
    cfg:       PostConfig,
) -> None:
    """
    Standard MoviePy export path — used when zoom is disabled or as the
    intermediate step before zoom is applied by a direct ffmpeg call.
    No -vf flag, no pipe-blocking filters.
    """
    ffmpeg_params: list = [
        "-b:v",      str(bitrate),
        "-pix_fmt",  str(cfg.pixel_format),
        "-movflags", "+faststart",
    ]

    assert all(isinstance(p, str) for p in ffmpeg_params), (
        f"ffmpeg_params contains a non-string value: {ffmpeg_params}"
    )

    composite.write_videofile(
        out_path,
        codec         = codec,
        audio_codec   = "aac",
        fps           = fps,
        ffmpeg_params = ffmpeg_params,
        logger        = None,
        threads       = cfg.threads,
        write_logfile = False,
    )


def _ffmpeg_apply_zoom(
    input_path:  str,
    out_path:    str,
    bitrate:     str,
    codec:       str,
    fps:         float,
    clip_w:      int,
    clip_h:      int,
    cfg:         PostConfig,
) -> None:
    """
    Apply the zoompan filter via a direct ffmpeg subprocess call using
    a FILE-BASED input (not a pipe). This is the only way zoompan works
    correctly — it requires seekable, random-access input to accumulate
    the frame sequence it needs for the zoom ramp.

    MoviePy feeds frames through a pipe (pipe: input). zoompan blocks
    on pipe input because it cannot seek back to re-read frames for the
    zoom accumulation — causing the process to appear frozen. By writing
    the composited clip to a temp file first and then calling ffmpeg
    directly with that file as input, we give zoompan the seekable stream
    it requires while keeping all MoviePy layers (labels, audio, flash)
    fully intact in the intermediate file.

    Two-stage zoom pipeline:
        Stage 1: MoviePy → temp_file (all layers, no zoom, no pipe block)
        Stage 2: ffmpeg  temp_file → out_path (zoom filter, file input)
    """
    vf = _build_zoom_filter(clip_w, clip_h, fps)

    cmd = [
        "ffmpeg", "-y",
        "-i",        input_path,          # file-based input — zoompan can seek
        "-vf",       vf,                  # zoom filter chain
        "-c:v",      codec,               # h264_videotoolbox / hevc_videotoolbox
        "-b:v",      str(bitrate),
        "-pix_fmt",  str(cfg.pixel_format),
        "-movflags", "+faststart",
        "-c:a",      "aac",               # preserve audio from intermediate file
        "-threads",  str(cfg.threads),
        out_path,
    ]

    print(f"  🔍 Zoom cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ❌ ffmpeg zoom failed (rc={result.returncode}):")
        print(result.stderr[-1000:])   # last 1000 chars of ffmpeg stderr
        raise RuntimeError(f"ffmpeg zoom pass failed for {input_path}")

    print(f"  ✅ Zoom pass complete → {out_path}")


def export_clip(
    composite: VideoFileClip,
    out_path:  str,
    bitrate:   str,
    cfg:       PostConfig,
    zoom:      bool = False,
) -> None:
    """
    Single sequential export — one ffmpeg Media Engine call at a time.
    Uses h264_videotoolbox or hevc_videotoolbox depending on cfg.hevc.

    When zoom=False:
        MoviePy → out_path  (single pass, standard pipe path)

    When zoom=True:
        Stage 1 — MoviePy → tmp_no_zoom.mp4   (all layers composited, no zoom)
        Stage 2 — ffmpeg  tmp_no_zoom.mp4 → out_path  (zoompan filter, file input)

    zoompan MUST have file-based input. Feeding it via MoviePy's pipe causes
    FFmpeg to block indefinitely because zoompan cannot seek back through a
    pipe to accumulate the frame sequence required for the zoom ramp.
    """
    codec = "hevc_videotoolbox" if cfg.hevc else "h264_videotoolbox"
    fps   = composite.fps or 30

    if not zoom:
        # Fast path — single MoviePy export, no intermediate file
        _ffmpeg_write_no_zoom(composite, out_path, bitrate, codec, fps, cfg)
        return

    # Zoom path — two-stage: composite → temp file, then ffmpeg zoom pass
    tmp_path = out_path.replace(".mp4", "_nozoom_tmp.mp4")
    try:
        print(f"  🎞  Stage 1/2: compositing layers → temp file…")
        _ffmpeg_write_no_zoom(composite, tmp_path, bitrate, codec, fps, cfg)

        print(f"  🔍 Stage 2/2: applying FFmpeg zoompan filter (file input)…")
        _ffmpeg_apply_zoom(
            input_path = tmp_path,
            out_path   = out_path,
            bitrate    = bitrate,
            codec      = codec,
            fps        = fps,
            clip_w     = composite.w,
            clip_h     = composite.h,
            cfg        = cfg,
        )
    finally:
        # Always clean up the intermediate file, even on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"  🗑  Removed temp file: {os.path.basename(tmp_path)}")


# ─────────────────────────────────────────────────────────────────
# Per-clip pipeline
# ─────────────────────────────────────────────────────────────────

def process_clip(
    clip_path:    str,
    meta:         dict,
    cfg:          PostConfig,
    tmp_dir:      str,
    clip_index:   int,
) -> dict:
    """
    Full enhancement pipeline for one clip.
    Returns enhanced metadata dict.
    """
    t0        = time.time()
    clip_name = Path(clip_path).stem
    print(f"\n{'─'*55}")
    print(f"  🎬 [{clip_index+1}] {clip_name}")
    print(f"{'─'*55}")

    # ── Load source clip ────────────────────────────────────────
    clip = VideoFileClip(clip_path)
    fps  = clip.fps or 30

    enhanced_meta = {
        "source_file":   os.path.basename(clip_path),
        "hook_text":     None,
        "hook_style":    None,
        "voice_duration": None,
        "encoding_mode": "hevc_videotoolbox" if cfg.hevc else "h264_videotoolbox",
        "bitrate_used":  None,
        "features":      [],
    }

    # ── Feature 5: Smart crop to 9:16 ──────────────────────────
    clip = ensure_9_16(clip)
    print(f"  📐 Aspect: {clip.w}×{clip.h}")

    # ── Feature 4: Zoom (first 3 seconds) ──────────────────────
    if cfg.zoom:
        # Zoom is applied as an FFmpeg filter at export time (see export_clip).
        # apply_zoom_effect() is called here only to keep the pipeline flow
        # intact and to register the feature in metadata — it returns clip unchanged.
        apply_zoom_effect(clip, zoom_ratio=1.03, zoom_duration=3.0)
        enhanced_meta["features"].append("zoom_3pct")

    # ── Feature 1: AI Hook Commentary ──────────────────────────
    hook_audio_path = None
    hook_dur        = 0.0
    hook_text       = None
    hook_style      = None

    if cfg.commentary:
        transcript = meta.get("transcript", "")
        hook_text, hook_style = generate_hook_text(transcript)
        print(f"  🎙  Hook [{hook_style}]: \"{hook_text}\"")

        hook_audio_path = generate_hook_audio(hook_text, cfg.hook_max_secs, tmp_dir)
        if hook_audio_path:
            hook_audio  = AudioFileClip(hook_audio_path)
            hook_dur    = hook_audio.duration
            hook_audio.close()
            print(f"  🔊 Voice duration: {hook_dur:.2f}s")
            enhanced_meta["hook_text"]      = hook_text
            enhanced_meta["hook_style"]     = hook_style
            enhanced_meta["voice_duration"] = round(hook_dur, 2)
            enhanced_meta["features"].append("hook_commentary")

    # ── Feature 2: Audio Ducking ────────────────────────────────
    if cfg.commentary and cfg.audio_duck and hook_audio_path and clip.audio:
        print(f"  🔉 Ducking original audio to {int(cfg.duck_level*100)}% during hook…")
        try:
            ducked_audio = apply_audio_ducking(
                clip.audio, hook_audio_path, cfg.duck_level
            )
            clip = clip.set_audio(ducked_audio)
            enhanced_meta["features"].append("audio_ducking")
        except Exception as e:
            print(f"  ⚠️  Audio ducking failed ({e}) — using original audio")

    # ── Feature 4: Pattern interrupt flash at 3s ───────────────
    if cfg.zoom and clip.duration > 3.2:
        flash = (
            make_flash_clip(clip.w, clip.h, duration=0.12)
            .set_start(3.0)
        )
        clip  = CompositeVideoClip([clip, flash], size=(clip.w, clip.h)).set_fps(fps)
        enhanced_meta["features"].append("pattern_interrupt")

    # ── Feature 3: Monetization labels ─────────────────────────
    layers = [clip]

    if cfg.label:
        top_label = make_label_clip(
            cfg.label_text, clip.w, clip.h, clip.duration, position="top"
        )
        bot_label = make_label_clip(
            cfg.source_text, clip.w, clip.h, clip.duration, position="bottom"
        )
        layers += [top_label, bot_label]
        enhanced_meta["features"].append("monetization_labels")
        print(f"  🏷  Labels added: '{cfg.label_text}' | '{cfg.source_text}'")

    if len(layers) > 1:
        clip = CompositeVideoClip(layers, size=(clip.w, clip.h)).set_fps(fps)

    # ── Feature 5: Seamless loop ────────────────────────────────
    if cfg.enable_loop:
        print("  🔁 Applying seamless loop crossfade…")
        clip = apply_loop_ending(clip)
        enhanced_meta["features"].append("seamless_loop")

    # ── Feature 6: Adaptive bitrate ────────────────────────────
    bitrate = adaptive_bitrate(clip, cfg)
    enhanced_meta["bitrate_used"] = bitrate

    # ── Export 9:16 ─────────────────────────────────────────────
    out_name = f"{clip_name}_enhanced.mp4"
    out_path = os.path.join(cfg.output_dir, out_name)
    print(f"  💾 Exporting 9:16 → {out_name}  [{bitrate}]")
    export_clip(clip, out_path, bitrate, cfg, zoom=cfg.zoom)
    enhanced_meta["output_9_16"] = out_name

    # ── Feature 6: Optional 1:1 square export ──────────────────
    if cfg.square:
        sq_clip     = make_square(VideoFileClip(out_path))   # re-read exported
        sq_name     = f"{clip_name}_enhanced_square.mp4"
        sq_path     = os.path.join(cfg.output_dir, sq_name)
        print(f"  🟥 Exporting 1:1 square → {sq_name}")
        export_clip(sq_clip, sq_path, bitrate, cfg, zoom=False)  # zoom already baked in
        sq_clip.close()
        enhanced_meta["output_1_1"] = sq_name

    clip.close()
    elapsed = time.time() - t0
    print(f"  ✅ Done in {elapsed:.1f}s")
    return enhanced_meta


# ─────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────

def run(cfg: PostConfig) -> None:
    wall_t = time.time()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Read metadata ───────────────────────────────────────────
    if not os.path.exists(cfg.metadata_file):
        print(f"❌  Metadata not found: {cfg.metadata_file}")
        print("    Run smart_viral_generator.py first.")
        return

    with open(cfg.metadata_file) as f:
        clips_meta: list[dict] = json.load(f)

    print(f"\n{'═'*55}")
    print(f"  🚀  Post-Processing Enhancement Engine")
    print(f"  📂  Input:  {cfg.input_dir}/")
    print(f"  📂  Output: {cfg.output_dir}/")
    print(f"  🎛   Encoder: {'hevc_videotoolbox' if cfg.hevc else 'h264_videotoolbox'}")
    print(f"  📦  Clips found in metadata: {len(clips_meta)}")
    print(f"{'═'*55}")

    tmp_dir = tempfile.mkdtemp(prefix="pp_m3_")
    all_enhanced_meta = []

    try:
        for i, meta in enumerate(clips_meta):
            clip_file = meta.get("file", "")
            clip_path = os.path.join(cfg.input_dir, clip_file)

            if not os.path.exists(clip_path):
                print(f"\n  ⚠️  Clip not found, skipping: {clip_path}")
                continue

            try:
                enhanced = process_clip(clip_path, meta, cfg, tmp_dir, i)
                # Merge original metadata fields with enhanced fields
                merged = {**meta, **enhanced, "original_metadata": meta}
                all_enhanced_meta.append(merged)
            except Exception as e:
                print(f"\n  ❌  Failed on {clip_file}: {e}")
                import traceback
                traceback.print_exc()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Feature 8: Save enhanced metadata ──────────────────────
    meta_out = os.path.join(cfg.output_dir, "enhanced_metadata.json")
    with open(meta_out, "w") as f:
        json.dump(all_enhanced_meta, f, indent=2)

    total = time.time() - wall_t
    print(f"\n{'═'*55}")
    print(f"  📋  Enhanced metadata: {meta_out}")
    print(f"  ✅  All clips processed in {total:.1f}s")
    print(f"{'═'*55}\n")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> PostConfig:
    p = argparse.ArgumentParser(
        description="post_process_m3.py — Viral Clip Post-Processing Engine (Apple Silicon M3)"
    )
    p.add_argument("--input-dir",     default="viral_clips",
                   help="Folder containing generated clips (default: viral_clips)")
    p.add_argument("--output-dir",    default="viral_clips_enhanced",
                   help="Folder for enhanced output clips")
    p.add_argument("--metadata",      default="viral_clips/clip_metadata.json",
                   help="Path to clip_metadata.json from generator")

    # Encoder
    p.add_argument("--hevc",          action="store_true",
                   help="Use hevc_videotoolbox instead of h264 (smaller files, same quality)")
    p.add_argument("--bitrate",       default="auto",
                   help="Video bitrate e.g. 6M, 8M, 10M (default: auto based on motion)")

    # Feature toggles
    p.add_argument("--no-commentary", action="store_true", help="Disable AI hook commentary")
    p.add_argument("--no-zoom",       action="store_true", help="Disable retention zoom effect")
    p.add_argument("--no-label",      action="store_true", help="Disable monetization labels")
    p.add_argument("--enable-loop",   action="store_true", help="Add seamless loop crossfade")
    p.add_argument("--square",        action="store_true", help="Also export 1:1 square version")

    args = p.parse_args()

    return PostConfig(
        input_dir     = args.input_dir,
        output_dir    = args.output_dir,
        metadata_file = args.metadata,
        hevc          = args.hevc,
        bitrate       = args.bitrate,
        commentary    = not args.no_commentary,
        audio_duck    = not args.no_commentary,   # duck only if commentary is on
        label         = not args.no_label,
        zoom          = not args.no_zoom,
        enable_loop   = args.enable_loop,
        square        = args.square,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run(cfg)
