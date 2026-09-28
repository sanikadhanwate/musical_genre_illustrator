"""
Unit tests for the local pipeline pieces (classifier + local image gen).
Run with: pytest test_app.py
No remote API calls required for these to pass.
"""

import os
import pytest
from app import classify_audio, generate_image_local, FALLBACK_PROMPTS

SAMPLE_AUDIO = os.path.join(os.path.dirname(__file__), "sample.wav")


@pytest.mark.skipif(
    not os.path.exists(SAMPLE_AUDIO),
    reason="sample.wav not present - add a short test clip to run this test",
)
def test_classify_audio_returns_genre_and_confidence():
    genre, confidence = classify_audio(SAMPLE_AUDIO)
    assert isinstance(genre, str)
    assert 0.0 <= confidence <= 1.0


def test_classify_audio_handles_none():
    genre, confidence = classify_audio(None)
    assert genre == "unknown"
    assert confidence == 0.0


def test_generate_image_local_returns_image():
    prompt = FALLBACK_PROMPTS["jazz"]
    image = generate_image_local(prompt)
    assert image is not None
