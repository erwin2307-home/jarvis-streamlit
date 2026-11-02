"""Wake-word detection tests."""
import numpy as np

from jarvis.wake import FakeWakeWord


async def test_fake_wake_triggers_on_marker():
    """FakeWakeWord fires when a frame whose first sample equals `marker_value` arrives."""
    marker = 12345
    wake = FakeWakeWord(threshold=0.5, marker_value=marker)
    silent = np.zeros(480, dtype=np.int16)
    marker_frame = np.full(480, marker, dtype=np.int16)

    assert not (await wake.detect(silent))
    assert not (await wake.detect(silent))
    assert (await wake.detect(marker_frame))


async def test_fake_wake_no_trigger_silence():
    """FakeWakeWord never fires on 20 consecutive silent frames."""
    wake = FakeWakeWord(threshold=0.5)
    for _ in range(20):
        assert not (await wake.detect(np.zeros(480, dtype=np.int16)))
    # Empty frame also never triggers.
    assert not (await wake.detect(np.empty(0, dtype=np.int16)))
