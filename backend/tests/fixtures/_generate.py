"""Generate deterministic audio fixtures. Run once: python _generate.py."""
from pathlib import Path

import numpy as np
import soundfile as sf


HERE = Path(__file__).parent
SR = 16000


def write_silence(seconds: float = 1.0) -> None:
    samples = np.zeros(int(SR * seconds), dtype=np.int16)
    sf.write(HERE / "silence.wav", samples, SR, subtype="PCM_16")


def write_sine(freq: float = 440.0, seconds: float = 1.0) -> None:
    t = np.arange(int(SR * seconds), dtype=np.float64) / SR
    sig = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)
    sf.write(HERE / "sine_440.wav", sig, SR, subtype="PCM_16")


if __name__ == "__main__":
    write_silence()
    write_sine()
    print(f"wrote: {HERE}/silence.wav, {HERE}/sine_440.wav")
