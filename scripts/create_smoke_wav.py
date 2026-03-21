from __future__ import annotations

import math
import wave
from pathlib import Path


def main() -> None:
    path = Path("tmp_smoke.wav")
    framerate = 16000
    samples = []
    for i in range(framerate):
        if i < framerate // 4:
            value = int(8000 * math.sin(2 * math.pi * 440 * i / framerate))
        else:
            value = 0
        samples.append(value)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(framerate)
        wav_file.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))

    print(path)


if __name__ == "__main__":
    main()

