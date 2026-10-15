"""
test_quality_engine.py — Comprehensive Test Suite for ClipQualityEngine
=======================================================================
Validates the 8-criteria quality scoring, protagonist tracking,
beat detection, audio mastering, and report generation.
"""

import json
import os
import tempfile
import unittest
import numpy as np

from ClipQualityEngine import (
    CRITERIA_WEIGHTS,
    AudioMastering,
    BeatDetector,
    ClipQualityEngine,
    ProtagonistTracker,
    quality_engine,
)


class TestClipQualityEngine(unittest.TestCase):

    def test_weights_sum_to_one(self):
        """Verify the 8 criteria weights sum exactly to 1.0."""
        total_weight = sum(CRITERIA_WEIGHTS.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        self.assertEqual(len(CRITERIA_WEIGHTS), 8)

    def test_criteria_evaluation_and_tiers(self):
        """Verify scoring logic, bounds [0, 100], and tier assignments."""
        engine = ClipQualityEngine()
        sample_rate = 16000
        # 30 seconds of synthetic audio with speech-like energy
        audio = (np.sin(np.linspace(0, 440 * 2 * np.pi * 30, 30 * sample_rate)) * 0.15).astype(np.float32)

        # High viral candidate
        high_transcript = (
            "Why is this legendary football kicker going up against an insane robot? "
            "Wait until you see what happens next. This moment changes everything. "
            "Nobody thought it was possible to win."
        )
        res_high = engine.evaluate_candidate(
            transcript=high_transcript,
            start_time=0.0,
            end_time=30.0,
            audio=audio,
            rate=sample_rate,
            visibility_ratio=0.92,
        )

        self.assertGreaterEqual(res_high.total_score, 0.0)
        self.assertLessEqual(res_high.total_score, 100.0)
        self.assertEqual(res_high.quality_tier, "PRIORITY")
        self.assertEqual(res_high.technical_status, "PASS")
        self.assertEqual(len(res_high.technical_issues), 0)

        # Check breakdown structure
        bd = res_high.breakdown.to_dict()
        self.assertEqual(len(bd), 8)
        for criterion in CRITERIA_WEIGHTS:
            self.assertIn(criterion, bd)
            self.assertIn("score", bd[criterion])
            self.assertIn("weight", bd[criterion])
            self.assertIn("details", bd[criterion])

        # Low candidate (awkward start, sluggish, cut off mid-sentence)
        low_transcript = "so um uh because and then like..."
        res_low = engine.evaluate_candidate(
            transcript=low_transcript,
            start_time=0.0,
            end_time=30.0,
            audio=audio * 0.001,  # near silent
            rate=sample_rate,
            visibility_ratio=0.10,
        )
        self.assertIn(res_low.quality_tier, ("REVIEW", "RECONSIDER"))

    def test_technical_health_separation(self):
        """Technical issues like silent audio should trigger FAIL regardless of engagement."""
        engine = ClipQualityEngine()
        silent_audio = np.zeros(16000 * 5, dtype=np.float32)
        status, issues = engine.validate_technical_health(silent_audio, video_w=1080, video_h=1920)
        self.assertEqual(status, "FAIL")
        self.assertIn("silent_audio", issues)

    def test_beat_detector(self):
        """Detect beats from audio and verify snap_to_nearest_beat."""
        rate = 16000
        # Create audio with distinct beats at 1.0s, 2.0s, 3.0s
        audio = np.zeros(rate * 4, dtype=np.float32)
        for beat_sec in (1.0, 2.0, 3.0):
            idx = int(beat_sec * rate)
            audio[idx : idx + 300] = 0.8

        beats = BeatDetector.detect_beats(audio, rate)
        self.assertGreater(len(beats), 0)

        # Snap test
        snapped = BeatDetector.snap_to_nearest_beat(2.05, beats, tolerance_sec=0.20)
        self.assertAlmostEqual(snapped, 2.0, delta=0.15)

    def test_protagonist_crop_bounds(self):
        """Ensure protagonist crop computation stays strictly within image boundaries."""
        # Frame 1920x1080 -> 9:16 target width is 607
        video_w = 1920
        target_w = 607

        # Center on left edge -> should clamp to 0
        x_left = ProtagonistTracker.compute_crop_x(video_w, target_w, target_cx=100)
        self.assertEqual(x_left, 0)

        # Center on right edge -> should clamp to video_w - target_w
        x_right = ProtagonistTracker.compute_crop_x(video_w, target_w, target_cx=1900)
        self.assertEqual(x_right, video_w - target_w)

        # Center in middle
        x_mid = ProtagonistTracker.compute_crop_x(video_w, target_w, target_cx=960)
        self.assertGreater(x_mid, 0)
        self.assertLess(x_mid, video_w - target_w)

    def test_audio_mastering(self):
        """Validate broadcast loudness normalization, soft limiting, and fades."""
        rate = 16000
        audio = (np.sin(np.linspace(0, 1000 * 2 * np.pi, rate * 2)) * 0.05).astype(np.float32)

        mastered = AudioMastering.normalize_and_master(audio, rate=rate, target_db=-14.0)
        self.assertEqual(len(mastered), len(audio))
        self.assertLessEqual(np.max(np.abs(mastered)), 1.0)

        # Verify boundary micro-fades
        self.assertAlmostEqual(mastered[0], 0.0, places=2)
        self.assertAlmostEqual(mastered[-1], 0.0, places=2)

    def test_quality_report_generation(self):
        """Generate JSON and Markdown quality reports and verify content."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "quality_report.json")
            md_path = os.path.join(tmp_dir, "quality_report.md")

            sample_data = [
                {
                    "rank": 1,
                    "file": "clip_1_score85.mp4",
                    "duration": 45.2,
                    "quality_score": 85.0,
                    "quality_tier": "PRIORITY",
                    "technical_status": "PASS",
                    "selection_reasoning": "High hook strength and consistent protagonist centering.",
                    "applied_enhancements": ["protagonist_tracking_9_16", "beat_synced_cuts"],
                    "quality_breakdown": {
                        "hook_strength": {"score": 90.0, "weight": 0.20, "details": "Strong opening"},
                        "emotional_impact": {"score": 85.0, "weight": 0.15, "details": "High energy"},
                    }
                }
            ]

            ClipQualityEngine.generate_quality_report(sample_data, json_path, md_path)

            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))

            with open(json_path) as f:
                loaded = json.load(f)
                self.assertEqual(len(loaded), 1)
                self.assertEqual(loaded[0]["quality_score"], 85.0)

            with open(md_path) as f:
                content = f.read()
                self.assertIn("ContentClipper Quality Engine Report", content)
                self.assertIn("PRIORITY", content)
                self.assertIn("clip_1_score85.mp4", content)


if __name__ == "__main__":
    unittest.main()
