"""Audio I/O tests using fake streams (no real device)."""
import numpy as np

from jarvis.audio_io import FakeMicStream, FakeSpeaker, rms_dbfs


async def test_fake_mic_stream_yields_expected_frames():
    pcm = np.zeros(16000, dtype=np.int16)  # 1 second silence @ 16k
    mic = FakeMicStream(pcm, frame_size=480)   # 30ms frames
    frames = []
    async for frame in mic.frames():
        frames.append(frame)
    assert len(frames) == 16000 // 480
    assert frames[0].dtype == np.int16
    assert frames[0].shape == (480,)


async def test_fake_speaker_collects_output():
    spk = FakeSpeaker()
    await spk.play(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    await spk.play(np.array([0.4, 0.5], dtype=np.float32))
    all_audio = spk.collected()
    assert np.allclose(all_audio, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_rms_dbfs_silence_is_minus_infinity():
    silent = np.zeros(1024, dtype=np.int16)
    assert rms_dbfs(silent) < -90  # treat -inf-ish as very low


def test_rms_dbfs_full_scale_is_zero():
    full = np.full(1024, 32767, dtype=np.int16)
    assert abs(rms_dbfs(full)) < 1


def test_fake_mic_drain_is_noop():
    pcm = np.zeros(960, dtype=np.int16)
    mic = FakeMicStream(pcm, frame_size=480)
    assert mic.drain() == 0
