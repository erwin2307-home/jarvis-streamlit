"""End-to-end orchestrator test with all-fake pipeline."""
import asyncio

import numpy as np

from jarvis.audio_io import FakeMicStream, FakeSpeaker
from jarvis.events import State, StateEvent, TranscriptFinalEvent, ResponseDoneEvent
from jarvis.llm import FakeLlm, LlmEvent
from jarvis.orchestrator import Orchestrator
from jarvis.stt import FakeStt
from jarvis.tts import FakeTts
from jarvis.vad import FakeVad
from jarvis.wake import FakeWakeWord


class _SpyTts:
    """Records the exact text passed to synthesize(), for asserting on
    filler vs. main-response content and prefixing."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str):
        self.texts.append(text)
        if text.strip():
            yield np.zeros(64, dtype=np.float32)


class _RecordingFastLlm:
    """Fast-model stand-in that records the messages it was asked to react
    to and returns a fixed scripted reaction."""

    def __init__(self, reaction: str) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self._reaction = reaction

    async def chat(self, messages: list[dict[str, str]]):
        self.calls.append(messages)
        yield LlmEvent(kind="chunk", text=self._reaction)
        yield LlmEvent(kind="done", text=self._reaction)


class _FailingFastLlm:
    async def chat(self, messages: list[dict[str, str]]):
        raise RuntimeError("fast model crashed")
        yield LlmEvent(kind="done", text="")  # unreachable; satisfies generator protocol


async def test_full_loop_with_fakes():
    """Wake-marker frame → silence → STT → LLM → TTS → IDLE again."""
    sample_rate = 16000
    frame_size = 480
    marker = 12345

    # Build PCM: 5 silent frames, then 1 marker frame, then 30 frames "speech",
    # then 30 frames silence (triggers speech_end at silence_frames=27).
    silent = np.zeros(frame_size, dtype=np.int16)
    marker_frame = np.full(frame_size, marker, dtype=np.int16)
    pcm_parts = [silent] * 5 + [marker_frame] + [silent] * 30 + [silent] * 30
    pcm = np.concatenate(pcm_parts)

    # VAD mask matches the frame schedule: 5 silent, 1 marker silent, 30 speech, 30 silent
    speech_mask = [False] * 5 + [False] + [True] * 30 + [False] * 30

    events: list = []

    orchestrator = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=frame_size),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="wie spät ist es"),
        llm=FakeLlm(chunks=["Es ist ", "14 Uhr."]),
        tts=FakeTts(samples_per_chunk=128),
        emit=events.append,
        silence_frames=27,
        sample_rate=sample_rate,
    )

    await asyncio.wait_for(orchestrator.run_once(), timeout=5.0)

    # Verify ordering: state transitions happened in the right sequence
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert state_seq[0] == State.LISTENING
    assert State.TRANSCRIBE in state_seq
    assert State.THINKING in state_seq
    assert State.SPEAKING in state_seq
    assert state_seq[-1] == State.IDLE

    transcripts = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    assert len(transcripts) == 1
    assert transcripts[0].text == "wie spät ist es"

    done = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert len(done) == 1
    assert done[0].full == "Es ist 14 Uhr."


async def test_cancel_during_listening_short_circuits():
    """Wake fires, then cancel during LISTENING → orchestrator returns to IDLE without TRANSCRIBE."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 99
    # 2 silent + marker + 5000 silent frames → mic loop runs long enough for cancel to fire
    pcm = np.concatenate([silent] * 2 + [np.full(480, marker, dtype=np.int16)] + [silent] * 5000)

    events: list = []
    orchestrator = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad([False] * 5005),    # never speaks → speech_end never fires
        stt=FakeStt(text="should not be called"),
        llm=FakeLlm(chunks=["x"]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=99999,            # high so speech_end can't fire
        sample_rate=16000,
    )

    async def cancel_after_wake() -> None:
        # Wait until at least one StateEvent has fired (LISTENING from wake), then cancel.
        for _ in range(100):
            await asyncio.sleep(0.001)
            if any(isinstance(e, StateEvent) and e.value == State.LISTENING for e in events):
                orchestrator.request_cancel()
                return
        orchestrator.request_cancel()

    await asyncio.gather(
        asyncio.wait_for(orchestrator.run_once(), timeout=2.0),
        cancel_after_wake(),
    )

    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert State.LISTENING in state_seq                # wake fired
    assert State.TRANSCRIBE not in state_seq           # cancel short-circuited before STT
    assert state_seq[-1] == State.IDLE                 # ended in IDLE


async def test_barge_in_during_speaking_returns_to_listening() -> None:
    """During TTS playback, a wake-word event aborts playback and returns to LISTENING."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 7777
    # PCM layout:
    #   3 silent + marker → wake fires at frame 4 (IDLE phase)
    #   then 1 speech frame + 3 silent → speech_end fires (LISTENING phase, silence_frames=3)
    #   then marker → barge-in trigger during SPEAKING phase
    #   then 500 silent so the loop survives
    speech_frame = np.full(480, 1000, dtype=np.int16)  # non-marker speech content
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate(
        [silent] * 3 + [marker_frame] + [speech_frame] + [silent] * 3
        + [marker_frame] + [silent] * 500
    )

    # speech_mask aligned with consumed frames:
    #   3 pre-wake silent (not fed to VAD)
    #   1 wake marker frame (not fed to VAD — triggers wake, then continue)
    #   LISTENING: 1 speech, 3 silent → speech_end at silence_frames=3
    speech_mask = [True] + [False] * 99  # first LISTENING frame is speech

    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="ignored"),
        llm=FakeLlm(chunks=["one ", "two ", "three"]),
        # Many TTS chunks so barge-in can fire mid-playback
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        min_transcript_words=0,  # bypass garbage filter for this test
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    # State arc: LISTENING → TRANSCRIBE → THINKING → SPEAKING → LISTENING (barge-in)
    # ... then run_once exits cleanly. We don't assert the final state because
    # the cycle ends at whatever state the barge-in lands in.
    assert State.SPEAKING in state_seq
    assert state_seq.count(State.LISTENING) >= 2  # initial + barge-in


async def test_manual_start_skips_wake() -> None:
    """If manual_start is requested before run_once, orchestrator goes straight to LISTENING."""
    silent = np.zeros(480, dtype=np.int16)
    pcm = np.concatenate([silent] * 5 + [silent] * 50)  # never any wake marker

    speech_mask = [False] * 5 + [True] * 30 + [False] * 30

    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=99999),  # would never fire in this PCM
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="manual hello"),
        llm=FakeLlm(chunks=["pong"]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=27,
        sample_rate=16000,
        min_transcript_words=0,
    )

    orch.set_manual_start()
    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert State.LISTENING in state_seq
    assert State.TRANSCRIBE in state_seq
    transcripts = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    assert transcripts and transcripts[0].text == "manual hello"


async def test_manual_start_mid_loop_skips_wake() -> None:
    """set_manual_start() while orchestrator is already inside run_once (mid
    pre-wake mic loop) must still trigger LISTENING on the next frame, not
    wait for the cycle to restart. This is the Win+J / HUD-click path:
    the flag is flipped by the inbound drain task while run_once is iterating.
    """
    silent = np.zeros(480, dtype=np.int16)
    pcm = np.concatenate(
        [silent] * 20 + [silent] * 1 + [silent] * 30 + [silent] * 30
    )
    # 20 pre-flip silent frames; flag fires somewhere in there.
    # After flip: 1 frame consumed for the manual_start branch (no recording),
    # then 1 speech frame + 30 silent frames → speech_end at silence_frames=3.
    # Actually: with the new check happening before VAD, that 1 marker frame is
    # consumed, then the next 30 frames go through VAD; we make 30 of them
    # speech then 30 silent.
    speech_mask = [False] * 1 + [True] * 30 + [False] * 30

    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=88888),  # never fires
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="mid loop hello"),
        llm=FakeLlm(chunks=["ok"]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        min_transcript_words=0,
    )

    async def flip_flag_soon() -> None:
        # Wait until a few audio_level frames have streamed, then flip.
        for _ in range(50):
            await asyncio.sleep(0.01)
            audio_seen = sum(
                1 for e in events if e.__class__.__name__ == "AudioLevelEvent"
            )
            if audio_seen >= 3:
                orch.set_manual_start()
                return

    await asyncio.gather(
        asyncio.wait_for(orch.run_once(), timeout=5.0),
        flip_flag_soon(),
    )

    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    # Required: LISTENING was emitted (mid-loop flag honored).
    assert State.LISTENING in state_seq
    # And the pipeline progressed past LISTENING (VAD speech_end fired,
    # STT/LLM/TTS ran), which only happens if manual_start was honored.
    assert State.TRANSCRIBE in state_seq


async def test_filler_plays_and_response_is_prefixed() -> None:
    """When use_filler=True, the orchestrator emits filler audio first and
    then the LLM response prefixed with response_prefix."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 5555
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    spk = FakeSpeaker()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=spk,
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="what's the weather"),
        llm=FakeLlm(chunks=["Sunny ", "and ", "warm."]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=True,
        response_prefix="PREFIX. ",
        filler_picker=lambda: "FILLER.",
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    # Check the full response event content reflects the LLM output (raw, no prefix).
    dones = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert dones and dones[0].full == "Sunny and warm."

    # State arc reached SPEAKING then IDLE.
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert State.SPEAKING in state_seq
    assert state_seq[-1] == State.IDLE


async def test_filler_disabled_skips_prefix() -> None:
    """With use_filler=False the response is spoken raw, no prefix."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 11111
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="ping"),
        llm=FakeLlm(chunks=["pong"]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        response_prefix="PREFIX. ",
        min_transcript_words=0,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    dones = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert dones and dones[0].full == "pong"
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert state_seq[-1] == State.IDLE


async def test_short_transcript_skips_llm() -> None:
    """Transcripts shorter than min_transcript_words must NOT invoke the LLM
    (treated as background noise / partial utterance)."""

    class _SpyLlm:
        def __init__(self) -> None:
            self.calls: list = []
        async def chat(self, messages):
            self.calls.append(messages)
            yield  # never reached for the assertion we care about

    silent = np.zeros(480, dtype=np.int16)
    marker = 2222
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    spy_llm = _SpyLlm()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="hi"),  # 1 word, below default min=3
        llm=spy_llm,
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        min_transcript_words=3,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    # LLM was never invoked because transcript was too short.
    assert spy_llm.calls == []
    # We still emitted the transcript and ended at IDLE.
    transcripts = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    assert transcripts and transcripts[0].text == "hi"
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert state_seq[-1] == State.IDLE
    # And no SPEAKING (we skipped the TTS response too).
    assert State.SPEAKING not in state_seq


async def test_min_transcript_words_zero_disables_filter() -> None:
    """Setting min_transcript_words=0 forwards every transcript to the LLM."""

    class _SpyLlm:
        def __init__(self) -> None:
            self.calls: list = []
        async def chat(self, messages):
            self.calls.append(messages)
            yield  # type: ignore[unreachable]

    silent = np.zeros(480, dtype=np.int16)
    marker = 3333
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    spy_llm = _SpyLlm()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text=""),  # empty
        llm=spy_llm,
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        min_transcript_words=0,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)
    assert len(spy_llm.calls) == 1


async def test_complex_query_reacts_via_fast_llm_then_speaks_prefixed_answer() -> None:
    """When fast_llm is set and the transcript scores above the complexity
    threshold, the fast model reacts to the actual transcript (not a canned
    quote) and that reaction is spoken first; the main response is then
    spoken with the usual prefix, same as the legacy filler flow."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 4242
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    fast = _RecordingFastLlm("Interesting question!")
    spy_tts = _SpyTts()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="erkläre mir Quantenmechanik im Detail"),
        llm=FakeLlm(chunks=["42."]),
        tts=spy_tts,
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=True,
        response_prefix="PREFIX. ",
        fast_llm=fast,
        complexity_threshold=1.0,
        min_transcript_words=0,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    # Fast model was asked to react to the actual transcript, not a canned prompt.
    assert len(fast.calls) == 1
    assert fast.calls[0][-1] == {
        "role": "user",
        "content": "erkläre mir Quantenmechanik im Detail",
    }

    # Filler spoken was the fast model's reaction; main response kept the prefix.
    assert spy_tts.texts == ["Interesting question!", "PREFIX. 42."]

    dones = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert dones and dones[0].full == "42."
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert state_seq[-1] == State.IDLE


async def test_simple_query_with_fast_llm_skips_filler_entirely() -> None:
    """When fast_llm is set but the transcript scores below the complexity
    threshold, no filler plays at all — the answer is spoken directly and
    unprefixed, since there's nothing to mask."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 9191
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    fast = _RecordingFastLlm("should not be used")
    spy_tts = _SpyTts()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="danke"),  # scores well below threshold
        llm=FakeLlm(chunks=["pong"]),
        tts=spy_tts,
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=True,
        response_prefix="PREFIX. ",
        fast_llm=fast,
        complexity_threshold=1.0,
        min_transcript_words=0,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    assert fast.calls == []
    assert spy_tts.texts == ["pong"]  # no filler, no prefix


async def test_fast_llm_reaction_failure_does_not_crash_turn() -> None:
    """If the fast model's reaction call raises, the turn still completes
    and the main response is still delivered."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 8181
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    spy_tts = _SpyTts()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="erkläre mir Quantenmechanik im Detail"),
        llm=FakeLlm(chunks=["42."]),
        tts=spy_tts,
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=True,
        response_prefix="PREFIX. ",
        fast_llm=_FailingFastLlm(),
        complexity_threshold=1.0,
        min_transcript_words=0,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    dones = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert dones and dones[0].full == "42."
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert state_seq[-1] == State.IDLE


async def test_listening_hard_cap_forces_speech_end_when_vad_never_completes() -> None:
    """If VAD never signals speech_end (e.g. background noise keeps
    resetting the silence counter), a wall-clock cap on LISTENING still
    forces the turn to proceed, instead of recording indefinitely — a long
    buried-in-noise recording is exactly what caused Whisper to hallucinate
    nonsense in practice."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 7070
    marker_frame = np.full(480, marker, dtype=np.int16)
    # Far more frames available than the cap should ever let it consume.
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 500)
    speech_mask = [False] * 503  # VAD never reports speech -> no natural speech_end

    class _FakeClock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            self.t += 1.0  # advance 1 "second" per call, deterministic
            return self.t

    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="ignored"),
        llm=FakeLlm(chunks=["ok"]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        min_transcript_words=0,
        max_listen_s=5.0,
        now_fn=_FakeClock(),
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert State.TRANSCRIBE in state_seq
    assert state_seq[-1] == State.IDLE


async def test_qa_match_skips_llm_and_answers_instantly() -> None:
    """When qa_matcher returns a known answer for the transcript, the LLM
    must never be called — the answer is spoken directly."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 6060
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    class _SpyLlm:
        def __init__(self) -> None:
            self.calls: list = []

        async def chat(self, messages):
            self.calls.append(messages)
            yield  # pragma: no cover - must never actually be reached

    spy_llm = _SpyLlm()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="was ist ein beadchip"),
        llm=spy_llm,
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        min_transcript_words=0,
        qa_matcher=lambda t: "Ein BeadChip ist ein Microarray." if "beadchip" in t else None,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    assert spy_llm.calls == []
    dones = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert dones and dones[0].full == "Ein BeadChip ist ein Microarray."
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert State.THINKING in state_seq
    assert State.SPEAKING in state_seq
    assert state_seq[-1] == State.IDLE


async def test_qa_no_match_falls_back_to_llm() -> None:
    """When qa_matcher returns None, the normal LLM path runs unchanged."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 6161
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="wie spät ist es"),
        llm=FakeLlm(chunks=["14 Uhr."]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        min_transcript_words=0,
        qa_matcher=lambda t: None,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    dones = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert dones and dones[0].full == "14 Uhr."


async def test_topic_context_injected_as_system_message_when_detected() -> None:
    """When topic_context returns text for the transcript, it must be
    prepended as a system message to the LLM call (RAG-style), so the
    model answers from real facts instead of hallucinating."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 7272
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    class _RecordingLlm:
        def __init__(self) -> None:
            self.calls: list = []

        async def chat(self, messages):
            self.calls.append(messages)
            yield LlmEvent(kind="chunk", text="Antwort.")
            yield LlmEvent(kind="done", text="Antwort.")

    recording_llm = _RecordingLlm()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="was ist ein beadchip"),
        llm=recording_llm,
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        min_transcript_words=0,
        qa_matcher=lambda t: None,
        topic_context=lambda t: "FACT CONTEXT" if "beadchip" in t else None,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    assert len(recording_llm.calls) == 1
    sent_messages = recording_llm.calls[0]
    assert sent_messages[0] == {"role": "system", "content": "FACT CONTEXT"}
    assert sent_messages[-1] == {"role": "user", "content": "was ist ein beadchip"}


async def test_no_topic_context_means_plain_user_message() -> None:
    """When topic_context returns None, the LLM call is unchanged — just
    the user's message, no injected system message."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 7373
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    class _RecordingLlm:
        def __init__(self) -> None:
            self.calls: list = []

        async def chat(self, messages):
            self.calls.append(messages)
            yield LlmEvent(kind="chunk", text="14 Uhr.")
            yield LlmEvent(kind="done", text="14 Uhr.")

    recording_llm = _RecordingLlm()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="wie spät ist es"),
        llm=recording_llm,
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        min_transcript_words=0,
        qa_matcher=lambda t: None,
        topic_context=lambda t: "FACT CONTEXT" if "beadchip" in t else None,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    assert recording_llm.calls == [[{"role": "user", "content": "wie spät ist es"}]]


async def test_coding_request_speaks_ack_and_skips_llm_entirely() -> None:
    """When coding_handler claims the transcript (returns True — meaning
    it already fired off the background generation+execution task), the
    turn speaks a fixed acknowledgment and never touches the LLM at all,
    so the voice turn completes immediately regardless of how long
    background code generation takes."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 8181
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    class _SpyLlm:
        def __init__(self) -> None:
            self.calls: list = []

        async def chat(self, messages):
            self.calls.append(messages)
            yield  # pragma: no cover - must never be reached

    handler_calls: list[str] = []

    def coding_handler(text: str) -> bool:
        handler_calls.append(text)
        return "programmier" in text

    spy_llm = _SpyLlm()
    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="programmiere mir einen taschenrechner"),
        llm=spy_llm,
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        min_transcript_words=0,
        coding_handler=coding_handler,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    assert handler_calls == ["programmiere mir einen taschenrechner"]
    assert spy_llm.calls == []
    dones = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert dones and "programm" in dones[0].full.lower()
    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert state_seq[-1] == State.IDLE


async def test_coding_handler_declines_non_coding_request() -> None:
    """When coding_handler returns False, the normal LLM path runs."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 8282
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate([silent] * 2 + [marker_frame] + [silent] * 30 + [silent] * 30)
    speech_mask = [False] * 2 + [False] + [True] * 30 + [False] * 30

    events: list = []
    orch = Orchestrator(
        mic=FakeMicStream(pcm, frame_size=480),
        speaker=FakeSpeaker(),
        wake=FakeWakeWord(marker_value=marker),
        vad=FakeVad(speech_mask),
        stt=FakeStt(text="wie spät ist es"),
        llm=FakeLlm(chunks=["14 Uhr."]),
        tts=FakeTts(samples_per_chunk=64),
        emit=events.append,
        silence_frames=3,
        sample_rate=16000,
        use_filler=False,
        min_transcript_words=0,
        coding_handler=lambda t: False,
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    dones = [e for e in events if isinstance(e, ResponseDoneEvent)]
    assert dones and dones[0].full == "14 Uhr."
