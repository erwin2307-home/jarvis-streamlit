"""Voice activity detection tests."""
import numpy as np

from jarvis.vad import FakeVad, SpeechEndDetector


async def test_fake_vad_returns_marker():
    vad = FakeVad(speech_mask=[False, False, True, True, False])
    out = []
    for _ in range(5):
        out.append(await vad.is_speech(np.zeros(480, dtype=np.int16)))
    assert out == [False, False, True, True, False]


async def test_speech_end_detector_signals_after_silence():
    """Detector fires speech_end after `silence_frames` consecutive non-speech post-speech."""
    det = SpeechEndDetector(silence_frames=3)
    # Speech starts
    assert det.update(True) == "speech_start"
    assert det.update(True) is None
    assert det.update(False) is None  # 1 silence
    assert det.update(False) is None  # 2 silence
    assert det.update(False) == "speech_end"  # 3 silence → end


async def test_speech_end_no_signal_without_speech():
    det = SpeechEndDetector(silence_frames=3)
    for _ in range(10):
        assert det.update(False) is None
