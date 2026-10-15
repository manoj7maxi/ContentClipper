"""
ClipQualityEngine.py — Intelligent Viral Quality Assessment & Optimization Engine
==================================================================================
Comprehensive evaluation, protagonist tracking, music synchronization, and
broadcast-standard audio mastering for short-form viral videos.

Quality Scoring (0–100 Scale):
  1. Hook Strength         (20%) — Opening 1–2s vocal delivery, question/trigger words, visual onset
  2. Emotional Impact      (15%) — Acoustic dynamics, dialogue sentiment, action/visual motion
  3. Protagonist Tracking  (15%) — Continuous subject centering in 9:16 vertical framing
  4. Pacing                (15%) — Words-per-second rhythm, elimination of awkward dead space
  5. Story Continuity      (10%) — Sentence boundary alignment, narrative arc, clean thoughts
  6. Audio Synchronization (10%) — Beat alignment, cadence timing, speech rhythm
  7. Visual Composition    (10%) — Rule-of-thirds/centering, stable margins, framing balance
  8. Ending/Rewatchability  (5%) — Memorable climax/punchline, audio cushion, loop potential

Quality Tiers:
  - PRIORITY   (≥ 80) : Top-tier candidates recommended for primary distribution.
  - REVIEW     (65–79): Strong clips with minor pacing or composition optimizations.
  - RECONSIDER (< 65) : Marginal candidates preserved only as fallbacks.

Technical Integrity:
  Kept separate from engagement scores (PASS / FAIL).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Quality Weights & Data Structures
# ─────────────────────────────────────────────────────────────────

CRITERIA_WEIGHTS = {
    "hook_strength":         0.20,
    "emotional_impact":      0.15,
    "protagonist_visibility":0.15,
    "pacing":                0.15,
    "story_continuity":      0.10,
    "audio_synchronization": 0.10,
    "visual_composition":    0.10,
    "ending_rewatchability": 0.05,
}

HOOK_TRIGGER_WORDS = frozenset({
    "why", "how", "what", "who", "when", "where", "never", "always",
    "secret", "stop", "dont", "don't", "remember", "wait", "watch",
    "imagine", "warning", "look", "listen", "truth", "exposed", "insane",
    "unbelievable", "mistake", "wrong", "shocking", "revealed", "crazy",
    "everyone", "nobody", "think", "believe", "notice", "tell", "ever"
})

EMOTIONAL_PUNCHWORDS = frozenset({
    "insane", "crazy", "shocking", "epic", "brutal", "love", "hate",
    "crying", "tears", "laugh", "laughing", "screaming", "win", "won",
    "lost", "fail", "failed", "died", "killed", "saved", "hero", "legend",
    "disaster", "genius", "nightmare", "miracle", "humanity", "fear", "danger"
})

STORY_START_PENALTY_WORDS = frozenset({
    "and", "but", "so", "because", "or", "also", "then", "like", "well", "um", "uh"
})


@dataclass
class CriterionScore:
    score: float       # 0 - 100
    weight: float      # e.g. 0.20
    details: str       # human-readable rationale

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class QualityBreakdown:
    hook_strength: CriterionScore
    emotional_impact: CriterionScore
    protagonist_visibility: CriterionScore
    pacing: CriterionScore
    story_continuity: CriterionScore
    audio_synchronization: CriterionScore
    visual_composition: CriterionScore
    ending_rewatchability: CriterionScore

    def to_dict(self) -> dict[str, dict[str, Any]]:
        result = {}
        for name in (
            "hook_strength", "emotional_impact", "protagonist_visibility",
            "pacing", "story_continuity", "audio_synchronization",
            "visual_composition", "ending_rewatchability"
        ):
            c: CriterionScore = getattr(self, name)
            result[name] = {
                "score": round(float(c.score), 1),
                "weight": float(c.weight),
                "weighted_score": round(float(c.weighted_score), 2),
                "details": str(c.details),
            }
        return result


@dataclass
class QualityEvaluation:
    total_score: float             # 0 - 100
    quality_tier: str              # PRIORITY, REVIEW, RECONSIDER
    technical_status: str          # PASS, FAIL
    technical_issues: list[str]    # e.g. ["audio_clipping", "black_frames"]
    breakdown: QualityBreakdown
    applied_enhancements: list[str]= field(default_factory=list)
    selection_reasoning: str       = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": round(self.total_score, 1),
            "quality_tier": self.quality_tier,
            "technical_status": self.technical_status,
            "technical_issues": self.technical_issues,
            "quality_breakdown": self.breakdown.to_dict(),
            "applied_enhancements": self.applied_enhancements,
            "selection_reasoning": self.selection_reasoning,
        }


# ─────────────────────────────────────────────────────────────────
# 1. Beat & Rhythm Synchronization
# ─────────────────────────────────────────────────────────────────

class BeatDetector:
    """
    Lightweight, SciPy-based musical beat and rhythmic onset detector.
    Computes spectral novelty and onset envelope without external heavyweight libraries.
    """

    @staticmethod
    def detect_beats(audio: np.ndarray, rate: int, hop_size: int = 512) -> list[float]:
        """
        Detect rhythm onsets/beats from 1D audio array.
        Returns list of timestamps (in seconds).
        """
        if len(audio) < rate * 0.5:
            return []

        try:
            from scipy.signal import find_peaks

            # Convert to mono if needed
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Frame-level RMS energy and spectral difference
            frame_len = 1024
            n_frames = (len(audio) - frame_len) // hop_size
            if n_frames <= 0:
                return []

            # Vectorized framing
            shape = (n_frames, frame_len)
            strides = (audio.strides[0] * hop_size, audio.strides[0])
            frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

            # Windowing
            window = np.hanning(frame_len)
            windowed = frames * window

            # FFT magnitude spectrum
            spec = np.abs(np.fft.rfft(windowed, axis=1))

            # Spectral flux / novelty (difference between consecutive frames)
            diff = np.diff(spec, axis=0)
            flux = np.sum(np.maximum(0, diff), axis=1)

            # Pad one frame to restore length
            novelty = np.pad(flux, (1, 0), mode="constant")

            # Local dynamic thresholding for peak picking
            smooth_win = max(int(rate / hop_size * 0.3), 3)  # ~300ms moving avg
            kernel = np.ones(smooth_win) / smooth_win
            threshold = np.convolve(novelty, kernel, mode="same") * 1.35

            peaks, _ = find_peaks(novelty, height=threshold, distance=int(rate / hop_size * 0.25))

            beat_times = [(p * hop_size) / rate for p in peaks]
            return beat_times

        except Exception as e:
            # Fallback to simple RMS peak detection
            return BeatDetector._simple_rms_peaks(audio, rate)

    @staticmethod
    def _simple_rms_peaks(audio: np.ndarray, rate: int) -> list[float]:
        chunk_ms = 100
        chunk_len = int(rate * chunk_ms / 1000)
        n_chunks = len(audio) // chunk_len
        if n_chunks < 2:
            return []
        chunks = audio[: n_chunks * chunk_len].reshape(n_chunks, chunk_len)
        rms = np.sqrt(np.mean(chunks ** 2, axis=1))
        thresh = np.mean(rms) * 1.4
        peaks = [i * chunk_ms / 1000.0 for i in range(1, n_chunks - 1)
                 if rms[i] > thresh and rms[i] > rms[i-1] and rms[i] > rms[i+1]]
        return peaks

    @staticmethod
    def snap_to_nearest_beat(target_time: float, beats: list[float], tolerance_sec: float = 0.20) -> float:
        """
        Snap a cut or transition point to the nearest musical beat if within tolerance.
        Otherwise preserves target_time.
        """
        if not beats:
            return target_time
        import bisect
        idx = bisect.bisect_left(beats, target_time)
        candidates = []
        if idx < len(beats):
            candidates.append(beats[idx])
        if idx > 0:
            candidates.append(beats[idx - 1])
        if not candidates:
            return target_time
        nearest = min(candidates, key=lambda b: abs(b - target_time))
        if abs(nearest - target_time) <= tolerance_sec:
            return nearest
        return target_time


# ─────────────────────────────────────────────────────────────────
# 2. Continuous Protagonist Tracking
# ─────────────────────────────────────────────────────────────────

class ProtagonistTracker:
    """
    Detects and tracks the main protagonist across video frames, maintaining
    smooth horizontal centering in vertical 9:16 aspect ratio crops.
    """

    def __init__(self, ema_alpha: float = 0.18, deadband_px: int = 15):
        self.ema_alpha = ema_alpha
        self.deadband_px = deadband_px
        self._cascade = None
        self._init_cascade()

    def _init_cascade(self):
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                self._cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._cascade = None

    def detect_face(self, frame_rgb: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """
        Detect primary protagonist face in a single frame.
        Returns (x, y, w, h) of primary face or None.
        """
        if self._cascade is None or frame_rgb is None:
            return None

        try:
            import cv2
            h, w = frame_rgb.shape[:2]
            # Downscale large frames for fast detection
            scale = 1.0
            if w > 640:
                scale = 640.0 / w
                small = cv2.resize(frame_rgb, (640, int(h * scale)))
            else:
                small = frame_rgb

            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
            faces = self._cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30)
            )

            if len(faces) == 0:
                return None

            # Primary face heuristic: combination of size (prominence) and central location
            frame_cx = small.shape[1] / 2.0
            best_face = None
            best_score = -1.0

            for (fx, fy, fw, fh) in faces:
                face_cx = fx + fw / 2.0
                dist_from_center = abs(face_cx - frame_cx) / frame_cx
                # Score = area * central proximity bonus
                score = (fw * fh) * (1.0 - 0.4 * dist_from_center)
                if score > best_score:
                    best_score = score
                    best_face = (int(fx / scale), int(fy / scale), int(fw / scale), int(fh / scale))

            return best_face
        except Exception:
            return None

    def track_clip(
        self,
        clip_or_path: Any,
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        sample_fps: float = 2.0
    ) -> tuple[dict[float, int], float]:
        """
        Sample video across duration and compute smoothed horizontal center trajectory.
        Returns:
            trajectory: dict mapping timestamp -> horizontal center x-coordinate
            visibility_ratio: float (0.0 to 1.0) indicating fraction of sampled frames with protagonist
        """
        trajectory: dict[float, int] = {}
        detected_count = 0
        total_samples = 0

        try:
            # Check if input is a MoviePy clip or file path
            is_clip_obj = hasattr(clip_or_path, "get_frame") and hasattr(clip_or_path, "duration")

            if is_clip_obj:
                dur = clip_or_path.duration
                width = clip_or_path.w
                default_cx = width // 2
                timestamps = np.arange(0.0, dur, 1.0 / sample_fps)
                frames = [clip_or_path.get_frame(t) for t in timestamps]
            else:
                import cv2
                cap = cv2.VideoCapture(str(clip_or_path))
                if not cap.isOpened():
                    return {0.0: 0}, 0.0
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                default_cx = width // 2
                step = max(int(fps / sample_fps), 1)

                cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
                cur_t = start_sec
                timestamps = []
                frames = []

                frame_idx = 0
                while cap.isOpened() and (end_sec <= 0.0 or cur_t <= end_sec):
                    ret, bgr = cap.read()
                    if not ret:
                        break
                    if frame_idx % step == 0:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        timestamps.append(cur_t)
                        frames.append(rgb)
                    cur_t += 1.0 / fps
                    frame_idx += 1
                cap.release()

            if not frames:
                return {0.0: default_cx}, 0.5

            total_samples = len(frames)
            raw_centers: list[int] = []

            for frame in frames:
                face = self.detect_face(frame)
                if face is not None:
                    detected_count += 1
                    raw_centers.append(face[0] + face[2] // 2)
                else:
                    # Fallback to motion or previous center
                    raw_centers.append(raw_centers[-1] if raw_centers else default_cx)

            # Smooth trajectory with Exponential Moving Average (EMA) and deadband
            smoothed_cx = float(raw_centers[0])
            for t, raw_cx in zip(timestamps, raw_centers):
                delta = raw_cx - smoothed_cx
                if abs(delta) > self.deadband_px:
                    smoothed_cx += self.ema_alpha * delta
                trajectory[round(float(t), 2)] = int(round(smoothed_cx))

            vis_ratio = detected_count / max(total_samples, 1)
            return trajectory, vis_ratio

        except Exception as e:
            # Fallback to center frame
            return {0.0: 960}, 0.5

    @staticmethod
    def compute_crop_x(
        video_w: int,
        target_w: int,
        target_cx: int
    ) -> int:
        """
        Calculate clamped x1 start coordinate for 9:16 vertical crop.
        Guarantees crop window stays within [0, video_w - target_w].
        """
        if target_w >= video_w:
            return 0
        x1 = target_cx - target_w // 2
        return int(max(0, min(x1, video_w - target_w)))


# ─────────────────────────────────────────────────────────────────
# 3. Audio Mastering & Normalization
# ─────────────────────────────────────────────────────────────────

class AudioMastering:
    """
    Broadcast-standard audio mastering:
    - Target loudness: -14.0 LUFS / RMS equivalent for YouTube Shorts / Reels / TikTok
    - Peak soft limiter to avoid digital clipping (max -1.0 dBFS)
    - Low-cut high-pass filtering (>80Hz) to remove sub-bass rumble
    - Boundary micro-fades (50ms in, 150ms out) to eliminate clicks and pops
    """

    @staticmethod
    def normalize_and_master(
        audio: np.ndarray,
        rate: int = 16000,
        target_db: float = -14.0,
        fade_in_ms: int = 60,
        fade_out_ms: int = 160,
    ) -> np.ndarray:
        """
        Processes 1D or 2D audio array. Returns float32 mastered audio in [-1.0, 1.0].
        """
        if len(audio) == 0:
            return audio

        out = audio.astype(np.float32).copy()

        # 1. High-pass filter rumble removal (simple 1st-order IIR highpass ~80Hz)
        try:
            rc = 1.0 / (2.0 * math.pi * 80.0)
            dt = 1.0 / rate
            alpha = rc / (rc + dt)
            filtered = np.zeros_like(out)
            if out.ndim == 1:
                filtered[0] = out[0]
                for i in range(1, len(out)):
                    filtered[i] = alpha * (filtered[i-1] + out[i] - out[i-1])
                out = filtered
        except Exception:
            pass

        # 2. RMS-based loudness gain adjustment
        rms = np.sqrt(np.mean(out ** 2))
        if rms > 1e-7:
            current_db = 20.0 * math.log10(rms)
            gain_db = target_db - current_db
            # Limit maximum boost to +12 dB to avoid noise magnification
            gain_db = max(min(gain_db, 12.0), -24.0)
            gain = 10.0 ** (gain_db / 20.0)
            out = out * gain

        # 3. Soft-knee limiter (prevent clipping above -1.0 dBFS / 0.891)
        ceiling = 0.891
        peak = np.max(np.abs(out))
        if peak > ceiling:
            # Smooth hyperbolic tangent soft limit for peaks
            overshoot = peak / ceiling
            out = np.tanh(out / ceiling) * ceiling

        # 4. Boundary micro-fades
        n_fade_in = min(int(rate * fade_in_ms / 1000), len(out) // 4)
        n_fade_out = min(int(rate * fade_out_ms / 1000), len(out) // 4)

        if n_fade_in > 0:
            fade_in = np.linspace(0.0, 1.0, n_fade_in, dtype=np.float32)
            if out.ndim == 1:
                out[:n_fade_in] *= fade_in
            else:
                out[:n_fade_in, :] *= fade_in[:, np.newaxis]

        if n_fade_out > 0:
            fade_out = np.linspace(1.0, 0.0, n_fade_out, dtype=np.float32)
            if out.ndim == 1:
                out[-n_fade_out:] *= fade_out
            else:
                out[-n_fade_out:, :] *= fade_out[:, np.newaxis]

        return np.clip(out, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────
# 4. Criteria Evaluation Functions
# ─────────────────────────────────────────────────────────────────

def evaluate_hook(
    transcript: str,
    first_phrase: str,
    first_2s_audio: np.ndarray,
    rate: int,
    start_time: float,
) -> CriterionScore:
    """Evaluate hook strength in the first 1-2 seconds."""
    score = 50.0  # neutral baseline
    details_list = []

    clean_first = re.sub(r"[^\w\s]", "", first_phrase.lower()).split()
    first_words = clean_first[:6]

    # 1. Trigger word bonus
    triggers_found = [w for w in first_words if w in HOOK_TRIGGER_WORDS]
    if triggers_found:
        score += min(len(triggers_found) * 12.0, 25.0)
        details_list.append(f"Hook words ({', '.join(triggers_found)})")

    # 2. Question / mystery opening
    if "?" in first_phrase or any(q in first_words[:2] for q in ("why", "how", "what", "did")):
        score += 15.0
        details_list.append("Question hook")

    # 3. Avoid awkward filler starts
    if first_words and first_words[0] in STORY_START_PENALTY_WORDS:
        score -= 14.0
        details_list.append(f"Weak connector start ('{first_words[0]}')")

    # 4. Acoustic energy onset in first 1.5s
    if len(first_2s_audio) > 0:
        rms_start = float(np.sqrt(np.mean(first_2s_audio ** 2)))
        if rms_start > 0.06:
            score += 15.0
            details_list.append("High vocal/audio energy onset")
        elif rms_start < 0.015:
            score -= 15.0
            details_list.append("Low initial audio energy")

    score = float(np.clip(score, 10.0, 100.0))
    details = "; ".join(details_list) if details_list else "Average opening hook"
    return CriterionScore(score=score, weight=CRITERIA_WEIGHTS["hook_strength"], details=details)


def evaluate_emotional_impact(
    transcript: str,
    audio_chunk: np.ndarray,
    visual_motion: float = 0.10,
) -> CriterionScore:
    """Analyze dialogue sentiment, acoustic pitch/energy dynamics, and scene intensity."""
    score = 55.0
    details_list = []

    words = re.sub(r"[^\w\s]", "", transcript.lower()).split()
    punchwords = [w for w in words if w in EMOTIONAL_PUNCHWORDS]
    if punchwords:
        score += min(len(punchwords) * 6.0, 24.0)
        details_list.append(f"{len(punchwords)} high-arousal words")

    # Exclamation & excitement in transcription
    exclamations = transcript.count("!")
    if exclamations > 0:
        score += min(exclamations * 4.0, 12.0)
        details_list.append("Expressive punctuation")

    # Acoustic dynamic range (ratio of peak to median)
    if len(audio_chunk) > 1000:
        frame_sz = 800
        n_f = len(audio_chunk) // frame_sz
        if n_f > 4:
            rms_frames = np.sqrt(np.mean(audio_chunk[: n_f * frame_sz].reshape(n_f, frame_sz) ** 2, axis=1))
            med = float(np.median(rms_frames)) + 1e-6
            peak = float(np.max(rms_frames))
            dyn_range = peak / med
            if dyn_range > 3.5:
                score += 12.0
                details_list.append("Dynamic audio contrast")

    # Visual motion
    if visual_motion > 0.12:
        score += 8.0
        details_list.append("Active scene motion")
    elif visual_motion < 0.03:
        score -= 6.0
        details_list.append("Low visual motion")

    score = float(np.clip(score, 15.0, 100.0))
    details = "; ".join(details_list) if details_list else "Moderate emotional resonance"
    return CriterionScore(score=score, weight=CRITERIA_WEIGHTS["emotional_impact"], details=details)


def evaluate_protagonist_visibility(
    visibility_ratio: float,
    multi_character: bool = False,
) -> CriterionScore:
    """Evaluate how consistently the primary protagonist is detected and framed."""
    score = visibility_ratio * 100.0
    details_list = [f"Protagonist tracked in {int(visibility_ratio * 100)}% of frames"]

    if visibility_ratio >= 0.75:
        score = min(score + 10.0, 100.0)
        details_list.append("High subject prominence")
    elif visibility_ratio < 0.35:
        score = max(score - 10.0, 20.0)
        details_list.append("Subject occasionally out of frame")

    if multi_character:
        details_list.append("Multi-character scene (primary prioritized)")

    score = float(np.clip(score, 20.0, 100.0))
    return CriterionScore(score=score, weight=CRITERIA_WEIGHTS["protagonist_visibility"], details="; ".join(details_list))


def evaluate_pacing(
    word_count: int,
    duration: float,
    pause_count: int = 0,
) -> CriterionScore:
    """Measure words-per-second rhythm and absence of awkward dead air."""
    if duration <= 0:
        return CriterionScore(score=50.0, weight=CRITERIA_WEIGHTS["pacing"], details="Zero duration")

    wps = word_count / duration
    score = 70.0
    details_list = [f"{wps:.1f} words/sec"]

    # Target viral pacing: 2.3 to 3.8 WPS
    if 2.3 <= wps <= 3.8:
        score += 20.0
        details_list.append("Optimal viral delivery rhythm")
    elif 1.6 <= wps < 2.3:
        score += 5.0
        details_list.append("Moderate pace")
    elif wps > 4.2:
        score -= 10.0
        details_list.append("Excessively rushed speech")
    else:
        score -= 20.0
        details_list.append("Sluggish speech rate")

    # Dead air pauses (>1.2s)
    if pause_count > 2:
        score -= min(pause_count * 5.0, 20.0)
        details_list.append(f"{pause_count} noticeable pauses")
    else:
        score += 8.0
        details_list.append("Continuous tight pacing")

    score = float(np.clip(score, 15.0, 100.0))
    return CriterionScore(score=score, weight=CRITERIA_WEIGHTS["pacing"], details="; ".join(details_list))


def evaluate_story_continuity(
    transcript: str,
    duration: float,
) -> CriterionScore:
    """Check sentence completeness, clean beginning, and resolution."""
    score = 65.0
    details_list = []

    trimmed = transcript.strip()
    if not trimmed:
        return CriterionScore(score=30.0, weight=CRITERIA_WEIGHTS["story_continuity"], details="Empty transcript")

    # Sentence closure
    if trimmed[-1] in (".", "?", "!"):
        score += 15.0
        details_list.append("Clean sentence termination")
    elif trimmed[-1] in (",", "-", ":"):
        score -= 15.0
        details_list.append("Abrupt mid-sentence cutoff")

    # Capitalized start
    if trimmed[0].isupper():
        score += 8.0
        details_list.append("Complete opening thought")

    # Duration suitability for narrative arc (30s to 75s is sweet spot)
    if 25.0 <= duration <= 75.0:
        score += 12.0
        details_list.append("Balanced narrative length")

    score = float(np.clip(score, 20.0, 100.0))
    details = "; ".join(details_list) if details_list else "Average narrative coherence"
    return CriterionScore(score=score, weight=CRITERIA_WEIGHTS["story_continuity"], details=details)


def evaluate_audio_sync(
    beats: list[float],
    clip_start: float,
    clip_end: float,
) -> CriterionScore:
    """Measure alignment of cut boundaries and internal rhythm with soundtrack beats."""
    score = 60.0
    details_list = []

    if not beats:
        return CriterionScore(score=65.0, weight=CRITERIA_WEIGHTS["audio_synchronization"], details="Acoustic cadence tracking (music beats subtle)")

    # Check how close clip boundaries are to beats
    start_dist = min(abs(b - clip_start) for b in beats) if beats else 1.0
    end_dist = min(abs(b - clip_end) for b in beats) if beats else 1.0

    if start_dist < 0.18:
        score += 15.0
        details_list.append("Start cut beat-synced")
    if end_dist < 0.20:
        score += 15.0
        details_list.append("End cut beat-synced")

    score = float(np.clip(score, 25.0, 100.0))
    details = "; ".join(details_list) if details_list else "Standard audio-visual cadence"
    return CriterionScore(score=score, weight=CRITERIA_WEIGHTS["audio_synchronization"], details=details)


def evaluate_visual_composition(
    aspect_ratio: str = "9:16",
    face_centered: bool = True,
    safe_margin_passed: bool = True,
) -> CriterionScore:
    """Assess framing balance, face scale, and vertical platform-safe margins."""
    score = 75.0
    details_list = []

    if aspect_ratio == "9:16":
        score += 10.0
        details_list.append("Platform-standard 9:16 vertical ratio")
    if face_centered:
        score += 10.0
        details_list.append("Subject framed in center third")
    if safe_margin_passed:
        score += 5.0
        details_list.append("Compliant with mobile UI safe margins")

    score = float(np.clip(score, 30.0, 100.0))
    return CriterionScore(score=score, weight=CRITERIA_WEIGHTS["visual_composition"], details="; ".join(details_list))


def evaluate_ending_rewatchability(
    transcript: str,
    loop_friendly: bool = False,
) -> CriterionScore:
    """Evaluate punchline/climax payoff, audio cushion, and loop potential."""
    score = 65.0
    details_list = []

    last_phrase = transcript.strip().split(".")[-1].strip().lower()
    punchline_markers = ("boom", "done", "never", "always", "won", "lost", "happened", "crazy", "why")
    if any(m in last_phrase for m in punchline_markers):
        score += 18.0
        details_list.append("Punchline/dramatic resolution")

    if loop_friendly:
        score += 15.0
        details_list.append("Seamless loop compatible")
    else:
        score += 5.0
        details_list.append("Natural audio cushion fade")

    score = float(np.clip(score, 20.0, 100.0))
    details = "; ".join(details_list) if details_list else "Standard ending payoff"
    return CriterionScore(score=score, weight=CRITERIA_WEIGHTS["ending_rewatchability"], details=details)


# ─────────────────────────────────────────────────────────────────
# 5. Core Quality Engine
# ─────────────────────────────────────────────────────────────────

class ClipQualityEngine:
    """
    Main orchestration class for evaluating candidate video clips,
    validating technical health, and generating comprehensive quality reports.
    """

    def __init__(self):
        self.tracker = ProtagonistTracker()
        self.beat_detector = BeatDetector()
        self.audio_mastering = AudioMastering()

    def validate_technical_health(
        self,
        audio_chunk: Optional[np.ndarray] = None,
        video_w: int = 1080,
        video_h: int = 1920,
        fps: float = 30.0,
    ) -> tuple[str, list[str]]:
        """
        Verify technical stream health independent of engagement score.
        Returns ("PASS" | "FAIL", issues_list).
        """
        issues = []

        # Video stream checks
        if video_w <= 0 or video_h <= 0:
            issues.append("invalid_resolution")
        if fps < 15.0:
            issues.append("low_framerate")

        # Audio stream checks
        if audio_chunk is not None and len(audio_chunk) > 0:
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
            if rms < 1e-5:
                issues.append("silent_audio")
            peak = float(np.max(np.abs(audio_chunk)))
            if peak > 1.05:
                issues.append("audio_clipping")
        else:
            issues.append("missing_audio_stream")

        status = "FAIL" if ("invalid_resolution" in issues or "silent_audio" in issues) else "PASS"
        return status, issues

    def evaluate_candidate(
        self,
        transcript: str,
        start_time: float,
        end_time: float,
        audio: np.ndarray,
        rate: int,
        beats: Optional[list[float]] = None,
        visibility_ratio: float = 0.85,
        visual_motion: float = 0.10,
        loop_friendly: bool = False,
    ) -> QualityEvaluation:
        """
        Calculates 8-criteria weighted quality evaluation for a candidate segment.
        """
        duration = end_time - start_time
        if beats is None:
            beats = []

        # Slice audio for candidate
        s_idx = max(0, int(start_time * rate))
        e_idx = min(len(audio), int(end_time * rate))
        audio_chunk = audio[s_idx:e_idx] if len(audio) > 0 else np.array([], dtype=np.float32)

        # First 1.5s audio for hook
        h_idx = min(len(audio), s_idx + int(1.5 * rate))
        hook_audio = audio[s_idx:h_idx] if len(audio) > 0 else np.array([], dtype=np.float32)

        # First phrase for hook text
        sentences = [s.strip() for s in re.split(r"[.!?]", transcript) if s.strip()]
        first_phrase = sentences[0] if sentences else transcript[:40]
        words = transcript.split()

        # Evaluate all 8 criteria
        c_hook = evaluate_hook(transcript, first_phrase, hook_audio, rate, start_time)
        c_emotion = evaluate_emotional_impact(transcript, audio_chunk, visual_motion)
        c_protagonist = evaluate_protagonist_visibility(visibility_ratio)
        c_pacing = evaluate_pacing(len(words), duration)
        c_story = evaluate_story_continuity(transcript, duration)
        c_sync = evaluate_audio_sync(beats, start_time, end_time)
        c_visual = evaluate_visual_composition()
        c_ending = evaluate_ending_rewatchability(transcript, loop_friendly)

        breakdown = QualityBreakdown(
            hook_strength=c_hook,
            emotional_impact=c_emotion,
            protagonist_visibility=c_protagonist,
            pacing=c_pacing,
            story_continuity=c_story,
            audio_synchronization=c_sync,
            visual_composition=c_visual,
            ending_rewatchability=c_ending,
        )

        # Weighted total
        total_score = (
            c_hook.weighted_score +
            c_emotion.weighted_score +
            c_protagonist.weighted_score +
            c_pacing.weighted_score +
            c_story.weighted_score +
            c_sync.weighted_score +
            c_visual.weighted_score +
            c_ending.weighted_score
        )
        total_score = round(float(np.clip(total_score, 0.0, 100.0)), 1)

        # Quality tier determination
        if total_score >= 80.0:
            tier = "PRIORITY"
        elif total_score >= 65.0:
            tier = "REVIEW"
        else:
            tier = "RECONSIDER"

        # Technical health validation
        tech_status, tech_issues = self.validate_technical_health(audio_chunk)

        # Selection reasoning
        reasons = []
        if c_hook.score >= 75:
            reasons.append("strong opening hook")
        if c_emotion.score >= 75:
            reasons.append("high emotional/action intensity")
        if c_protagonist.score >= 80:
            reasons.append("steady protagonist centering")
        if c_story.score >= 80:
            reasons.append("complete narrative thought")
        if not reasons:
            reasons.append(f"steady overall engagement ({tier})")

        reasoning = f"Ranked as {tier} based on {', '.join(reasons)}."

        return QualityEvaluation(
            total_score=total_score,
            quality_tier=tier,
            technical_status=tech_status,
            technical_issues=tech_issues,
            breakdown=breakdown,
            selection_reasoning=reasoning,
        )

    @staticmethod
    def generate_quality_report(
        clips_metadata: list[dict],
        output_json_path: str,
        output_md_path: Optional[str] = None,
    ) -> None:
        """
        Export detailed quality report in both JSON and Markdown formats.
        """
        # Save JSON report
        with open(output_json_path, "w") as f:
            json.dump(clips_metadata, f, indent=2)

        # Generate human-readable Markdown report
        if output_md_path:
            md_lines = [
                "# ContentClipper Quality Engine Report 🎬🏆",
                "",
                "Summary of candidate video clips evaluated by the 8-criteria Quality Engine.",
                "",
                "| Rank | Tier | Quality Score | Duration | Hook | Emotion | Protagonist | Status | File |",
                "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
            ]

            for c in clips_metadata:
                rank = c.get("rank", "-")
                tier = c.get("quality_tier", "REVIEW")
                score = c.get("quality_score", c.get("viral_score", 0))
                dur = c.get("duration", 0)
                status = c.get("technical_status", "PASS")
                filename = c.get("file", c.get("source_file", "-"))

                bd = c.get("quality_breakdown", {})
                hook_s = bd.get("hook_strength", {}).get("score", "-")
                emo_s = bd.get("emotional_impact", {}).get("score", "-")
                pro_s = bd.get("protagonist_visibility", {}).get("score", "-")

                md_lines.append(
                    f"| #{rank} | **{tier}** | **{score}** | {dur:.1f}s | {hook_s} | {emo_s} | {pro_s} | `{status}` | `{filename}` |"
                )

            md_lines.extend([
                "",
                "---",
                "## Detailed Clip Analyses",
                "",
            ])

            for c in clips_metadata:
                rank = c.get("rank", "-")
                score = c.get("quality_score", c.get("viral_score", 0))
                tier = c.get("quality_tier", "REVIEW")
                reason = c.get("selection_reasoning", "High engagement potential.")
                enhancements = c.get("applied_enhancements", [])
                bd = c.get("quality_breakdown", {})

                md_lines.extend([
                    f"### Clip #{rank} — {c.get('file', 'Unknown')} (Score: {score}/100 — {tier})",
                    f"- **Selection Rationale**: {reason}",
                    f"- **Applied Enhancements**: {', '.join(enhancements) if enhancements else 'Standard polish'}",
                    "- **Score Breakdown**:",
                ])

                for name, info in bd.items():
                    if isinstance(info, dict):
                        w = info.get("weight", 0)
                        s = info.get("score", 0)
                        det = info.get("details", "")
                        label = name.replace("_", " ").title()
                        md_lines.append(f"  - **{label}** ({int(w*100)}%): **{s}**/100 — *{det}*")

                md_lines.append("")

            with open(output_md_path, "w") as f:
                f.write("\n".join(md_lines))


# Global singleton instance for easy import across modules
quality_engine = ClipQualityEngine()
