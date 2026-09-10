"""
Unit tests for the local pipeline pieces (classifier + local image gen).
Run with: python -m unittest tests/test.py
No remote API calls required for these to pass.
"""

import os
import sys
import unittest

# Add the project root to sys.path so `app` can be imported from tests/test.py.
########################### Failure Setup Start ######################################
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
########################### Failure Setup Done #######################################

from model_inference import classify_audio, generate_image_local, FALLBACK_PROMPTS


SAMPLE_AUDIO = os.path.join(os.path.dirname(__file__), "..", "sample.wav")


class TestLocalPipeline(unittest.TestCase):

    def test_classify_audio_returns_genre_and_confidence(self):
        if not os.path.exists(SAMPLE_AUDIO):
            self.skipTest(
                "sample.wav not present - add a short test clip to run this test"
            )

        genre, confidence = classify_audio(SAMPLE_AUDIO)

        self.assertIsInstance(genre, str)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_classify_audio_handles_none(self):
        genre, confidence = classify_audio(None)

        self.assertEqual(genre, "unknown")
        self.assertEqual(confidence, 0.0)

    def test_generate_image_local_returns_image(self):
        prompt = FALLBACK_PROMPTS["jazz"]
        image = generate_image_local(prompt)

        self.assertIsNotNone(image)


if __name__ == "__main__":
    unittest.main()
