"""Integration test for the Zonos TTS endpoint."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import requests


def test_tts_integration(server_url: str, assets_dir: Path, server_process) -> None:
    """Verify TTS endpoint returns audio when given text and reference audio."""
    reference_audio = assets_dir / "voice_002.mp3"
    assert reference_audio.exists(), f"Reference audio not found: {reference_audio}"

    text = "Hello from the Zonos integration test."
    seed = 12345
    data = {
        "text": text,
        "seed": seed,
        "happiness": 0.3077,
        "sadness": 0.0256,
        "disgust": 0.0256,
        "fear": 0.0256,
        "surprise": 0.0256,
        "anger": 0.0256,
        "other": 0.2564,
        "neutral": 0.3077,
        "expressiveness": 0.5,
        "speaking_rate": 0.375,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        ref_copy = Path(temp_dir) / "reference.mp3"
        shutil.copy(reference_audio, ref_copy)

        with open(ref_copy, "rb") as audio_file:
            files = {"reference_audio_file": ("reference.mp3", audio_file, "audio/mpeg")}
            response = requests.post(
                f"{server_url}/tts",
                data=data,
                files=files,
                timeout=180,
            )

        assert response.status_code == 200, f"Request failed: {response.status_code} - {response.text}"

        output_path = Path(temp_dir) / "tts_output.mp3"
        output_path.write_bytes(response.content)

        assert output_path.exists(), "Output file was not created"
        assert output_path.stat().st_size > 0, "Output file is empty"
        assert output_path.stat().st_size > 500, "Output file is too small to be valid audio"
