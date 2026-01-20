#!/usr/bin/env bash
set -euo pipefail

WARMUP_MARKER="/tmp/ct_zonos_warmup_done"

if [[ ! -f "$WARMUP_MARKER" ]]; then
  python3 - <<'PY'
import math
import struct
import wave

from tts_processor import process_tts

audio_path = "/tmp/ct_zonos_warmup.wav"
rate = 24000
duration = 0.2
freq = 440.0
amp = 0.1
frames = int(rate * duration)

with wave.open(audio_path, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(rate)
    for i in range(frames):
        sample = int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate))
        wav.writeframes(struct.pack("<h", sample))

process_tts(
    "warmup",
    audio_path,
    "/tmp/ct_zonos_warmup.mp3",
    1,
    [0.1] * 8,
    0.5,
    0.5,
)
PY
  touch "$WARMUP_MARKER"
fi

exec python3 -m CT_generic_server_client.server --port "${ZONOS_PORT:-8189}"
