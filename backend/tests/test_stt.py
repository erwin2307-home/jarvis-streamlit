"""STT tests using fake transcriber."""
import numpy as np

from jarvis.stt import FakeStt, strip_lone_surrogates


async def test_fake_stt_returns_scripted_text():
    stt = FakeStt(text="hallo Jarvis")
    pcm = np.zeros(16000, dtype=np.int16)
    result = await stt.transcribe(pcm)
    assert result == "hallo Jarvis"


async def test_fake_stt_empty_returns_empty():
    stt = FakeStt(text="")
    assert (await stt.transcribe(np.zeros(16000, dtype=np.int16))) == ""


def test_strip_lone_surrogates_removes_unencodable_char():
    # \udce4 is a lone UTF-16 low surrogate — no valid UTF-8 encoding.
    # Reproduces an actual crash seen from faster-whisper on Windows:
    # "'utf-8' codec can't encode character '\\udce4': surrogates not allowed"
    bad = "Ja, Frankie ist\udce4 gleiche."
    assert strip_lone_surrogates(bad) == "Ja, Frankie ist gleiche."


def test_strip_lone_surrogates_leaves_normal_text_untouched():
    assert strip_lone_surrogates("wie spät ist es") == "wie spät ist es"
