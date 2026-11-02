"""Pipeline orchestrator: glues wake/vad/stt/llm/tts into the state machine."""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Protocol

import numpy as np
import numpy.typing as npt
from loguru import logger

from jarvis.audio_io import rms_dbfs
from jarvis.events import (
    AudioLevelEvent,
    ErrorEvent,
    Event,
    ResponseChunkEvent,
    ResponseDoneEvent,
    State,
    StateEvent,
    TranscriptFinalEvent,
)
from jarvis.llm import complexity_score
from jarvis.state_machine import Trigger, transition
from jarvis.vad import SpeechEndDetector


# Asks the fast local model to react to the transcript, not answer it — the
# real answer is being computed on the slower/cloud path concurrently.
_REACTION_SYSTEM_PROMPT = (
    "Reagiere auf Deutsch, kurz und natürlich, auf das, was der Nutzer "
    "gerade gesagt hat — höchstens ein kurzer Satz (unter 10 Wörter). "
    "Beantworte die Frage nicht, eine ausführliche Antwort folgt gleich "
    "separat."
)

# Spoken immediately when coding_handler claims a transcript — the actual
# generation + execution happens in a detached background task that
# outlives this turn, so this ack is all the user hears right now.
_CODING_ACK = (
    "Alles klar, ich programmiere das jetzt im Hintergrund. Ein Fenster "
    "öffnet sich gleich."
)


EmitFn = Callable[[Event], None]


class _MicLike(Protocol):
    def frames(self) -> Any: ...
    def drain(self) -> int: ...


class _SpeakerLike(Protocol):
    async def play(self, samples: npt.NDArray[Any]) -> None: ...
    async def drain(self) -> None: ...


class _WakeLike(Protocol):
    async def detect(self, frame_int16: npt.NDArray[np.int16]) -> bool: ...


class _VadLike(Protocol):
    async def is_speech(self, frame_int16: npt.NDArray[np.int16]) -> bool: ...


class _SttLike(Protocol):
    async def transcribe(self, pcm_int16: npt.NDArray[np.int16]) -> str: ...


class _LlmLike(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> Any: ...


class _TtsLike(Protocol):
    def synthesize(self, text: str) -> Any: ...


class Orchestrator:
    def __init__(
        self,
        *,
        mic: _MicLike,
        speaker: _SpeakerLike,
        wake: _WakeLike,
        vad: _VadLike,
        stt: _SttLike,
        llm: _LlmLike,
        tts: _TtsLike,
        emit: EmitFn,
        silence_frames: int = 27,
        sample_rate: int = 16000,
        use_filler: bool = True,
        response_prefix: str = "Hier ist, was ich gefunden habe. ",
        filler_picker: Callable[[], str] | None = None,
        min_transcript_words: int = 3,
        fast_llm: _LlmLike | None = None,
        complexity_threshold: float = 1.0,
        max_listen_s: float = 15.0,
        now_fn: Callable[[], float] | None = None,
        qa_matcher: Callable[[str], str | None] | None = None,
        topic_context: Callable[[str], str | None] | None = None,
        coding_handler: Callable[[str], bool] | None = None,
    ) -> None:
        self._mic = mic
        self._speaker = speaker
        self._wake = wake
        self._vad = vad
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._emit = emit
        self._silence_frames = silence_frames
        self._sample_rate = sample_rate
        self._state = State.IDLE
        self._cancel = False
        self._manual_start = False
        self._use_filler = use_filler
        self._response_prefix = response_prefix
        self._min_transcript_words = min_transcript_words
        self._fast_llm = fast_llm
        self._complexity_threshold = complexity_threshold
        self._max_listen_s = max_listen_s
        self._now: Callable[[], float] = now_fn or time.monotonic
        self._qa_matcher = qa_matcher
        self._topic_context = topic_context
        self._coding_handler = coding_handler
        # Picker injectable so tests can pin a known filler.
        if filler_picker is not None:
            self._pick_filler = filler_picker
        else:
            from jarvis.filler import random_filler as _rf
            self._pick_filler = _rf

    def request_cancel(self) -> None:
        self._cancel = True

    def set_manual_start(self) -> None:
        """Skip wake-word detection on the next run_once cycle."""
        self._manual_start = True

    def _go(self, trigger: Trigger) -> None:
        new = transition(self._state, trigger)
        if new != self._state:
            self._state = new
            self._emit(StateEvent(value=new))

    async def run_once(self) -> None:
        """One full IDLE → IDLE cycle. Returns when speak finishes or cancel fires."""
        self._cancel = False

        try:
            # Drop any audio buffered while we were transcribing/thinking/
            # speaking in the previous cycle — those frames are stale and
            # often contain our own TTS playback echo, which can fool the
            # wake-word detector. Cheap O(n) on the queue.
            dropped = self._mic.drain()
            if dropped:
                logger.info(f"mic drain: dropped {dropped} stale frames")

            # 1. IDLE — listen for wake word
            recorded: list[npt.NDArray[np.int16]] = []
            wake_seen = False
            end_detector = SpeechEndDetector(silence_frames=self._silence_frames)
            t_wake: float | None = None
            t_speech_end: float | None = None
            t_listen_start: float | None = None

            # Honor manual-start request: skip wake-word and go straight to LISTENING.
            if self._manual_start:
                self._manual_start = False
                wake_seen = True
                t_wake = time.monotonic()
                t_listen_start = self._now()
                self._go(Trigger.MANUAL_START)

            async for frame in self._mic.frames():
                if self._cancel:
                    self._go(Trigger.CANCEL)
                    return

                self._emit(AudioLevelEvent(rms=rms_dbfs(frame), fft=[]))

                # Mid-loop manual_start (e.g. HUD click / Win+J during IDLE):
                # skip wake-word detection on the next frame.
                if self._manual_start and not wake_seen:
                    self._manual_start = False
                    wake_seen = True
                    t_listen_start = self._now()
                    self._go(Trigger.MANUAL_START)
                    continue

                if not wake_seen:
                    if await self._wake.detect(frame):
                        wake_seen = True
                        t_wake = time.monotonic()
                        t_listen_start = self._now()
                        self._go(Trigger.WAKE_DETECTED)
                    continue

                # In LISTENING: record + run VAD
                recorded.append(frame)
                speech = await self._vad.is_speech(frame)
                signal = end_detector.update(speech)
                if signal == "speech_end":
                    t_speech_end = time.monotonic()
                    break
                if (
                    t_listen_start is not None
                    and (self._now() - t_listen_start) >= self._max_listen_s
                ):
                    # VAD never completed a clean speech_end — likely
                    # background noise keeps resetting the silence counter.
                    # Force the cutoff rather than recording indefinitely;
                    # an unbounded recording just buries the real question
                    # in noise and makes STT hallucinate on the result.
                    logger.warning(
                        f"LISTENING exceeded max_listen_s={self._max_listen_s}s "
                        "without a clean speech_end — forcing cutoff"
                    )
                    t_speech_end = time.monotonic()
                    break

            if not wake_seen:
                self._go(Trigger.CANCEL)
                return

            # 2. TRANSCRIBE
            self._go(Trigger.SPEECH_END)
            pcm = np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.int16)
            audio_sec = pcm.size / max(1, self._sample_rate)

            # If filler is enabled, kick it off as soon as we leave LISTENING.
            # It plays through the speaker while STT + LLM run; the orchestrator
            # awaits it before starting the main response so the audio stays
            # serialised on a single OutputStream.
            #
            # With no fast_llm configured, this is the legacy canned-phrase
            # filler and can start immediately (masks STT latency too). With
            # a fast_llm configured, the filler becomes a reaction to the
            # transcript instead — it can only start once STT has produced
            # that transcript, so it's deferred below.
            filler_task: asyncio.Task[None] | None = None
            if self._use_filler and self._fast_llm is None:
                filler_phrase = self._pick_filler()
                logger.info(f"filler: {filler_phrase!r}")
                filler_task = asyncio.create_task(self._speak_text(filler_phrase))

            t0_stt = time.monotonic()
            text = await self._stt.transcribe(pcm)
            stt_ms = (time.monotonic() - t0_stt) * 1000
            self._emit(TranscriptFinalEvent(text=text))
            logger.info(
                f"timing: STT {stt_ms:.0f}ms on {audio_sec:.1f}s audio "
                f"({stt_ms / max(audio_sec*1000, 1):.2f}x rt) → {text!r}"
            )

            # Garbage filter: short / empty transcripts are usually background
            # noise, TV chatter, or a stray cough. Drop them before paying
            # for an LLM round-trip; let the filler finish (gives the user a
            # cue that we heard something) and return to IDLE.
            word_count = len(text.split())
            if word_count < self._min_transcript_words:
                logger.info(
                    f"transcript dropped: only {word_count} words "
                    f"(min {self._min_transcript_words}) — likely noise"
                )
                if filler_task is not None:
                    try:
                        await filler_task
                    except Exception:
                        pass
                    await self._speaker.drain()
                # Skip THINKING/SPEAKING entirely; go back to IDLE.
                self._state = State.IDLE
                self._emit(StateEvent(value=State.IDLE))
                return

            # Coding fast-path: coding_handler itself decides whether this
            # is a code-generation request AND, if so, has already fired
            # off the background generation+execution task (fire-and-
            # forget — outlives this turn). Takes top priority: speak a
            # fixed ack and skip the LLM entirely, same as a QA instant
            # match below, so the turn completes immediately regardless
            # of how long generation takes in the background.
            is_coding_request = (
                self._coding_handler(text) if self._coding_handler is not None else False
            )

            # QA fast-path: a known, pre-researched question answers
            # instantly, skipping the LLM entirely. Checked here (after
            # STT) since it needs the actual transcript; takes priority
            # over the fast_llm reaction decision below — there's nothing
            # to mask when the answer is already known.
            qa_answer = (
                _CODING_ACK
                if is_coding_request
                else (self._qa_matcher(text) if self._qa_matcher is not None else None)
            )

            # Deferred filler decision for the fast_llm case: complex queries
            # get a reaction to the actual transcript (masks the slower/cloud
            # answer's latency); simple queries get no filler at all — the
            # fast model's own answer is quick enough already.
            if (
                qa_answer is None
                and self._use_filler
                and self._fast_llm is not None
                and complexity_score(text) >= self._complexity_threshold
            ):
                filler_task = asyncio.create_task(self._react_and_speak(text))

            # 3. THINKING — collect full response. We can't TTS-stream it while
            # the filler is still playing (the speaker is a single stream and
            # would interleave audio), so we buffer the LLM output, wait for
            # filler to drain, then synthesise the prefixed answer.
            self._go(Trigger.TRANSCRIPT_READY)
            if qa_answer is not None:
                logger.info(f"qa_lookup: instant match for {text!r}")
                full_response = qa_answer
                llm_total_ms = 0.0
                llm_ttft_ms = 0.0
                self._emit(ResponseChunkEvent(text=qa_answer))
                self._emit(ResponseDoneEvent(full=full_response))
            else:
                # RAG-style safety net: a loosely topic-matching question
                # (one that missed every QA_PAIRS trigger) still gets the
                # full reference knowledge injected as context, so the LLM
                # answers from real facts instead of hallucinating.
                context = self._topic_context(text) if self._topic_context is not None else None
                messages: list[dict[str, str]] = [{"role": "user", "content": text}]
                if context:
                    messages = [{"role": "system", "content": context}, *messages]
                logger.info(
                    f"thinking: sending {len(text)} chars to LLM"
                    + (" (+ topic context)" if context else "")
                )
                full_response = ""
                t0_llm = time.monotonic()
                t_first_token: float | None = None
                async for ev in self._llm.chat(messages):
                    if self._cancel:
                        self._go(Trigger.CANCEL)
                        if filler_task is not None:
                            filler_task.cancel()
                        return
                    if ev.kind == "chunk":
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        full_response += ev.text
                        self._emit(ResponseChunkEvent(text=ev.text))
                    elif ev.kind == "done":
                        full_response = ev.text or full_response
                        self._emit(ResponseDoneEvent(full=full_response))
                llm_total_ms = (time.monotonic() - t0_llm) * 1000
                llm_ttft_ms = (
                    (t_first_token - t0_llm) * 1000 if t_first_token is not None else -1
                )
                logger.info(
                    f"timing: LLM ttft {llm_ttft_ms:.0f}ms, total {llm_total_ms:.0f}ms, "
                    f"resp {len(full_response)} chars"
                )

            # Wait for filler audio to finish (synthesis + buffer drain) before
            # starting the main response. Without this, two TTS streams would
            # hit the speaker at once.
            if filler_task is not None:
                try:
                    await filler_task
                except Exception as exc:
                    logger.warning(f"filler task error (ignored): {exc!r}")
                await self._speaker.drain()

            # 4. SPEAKING — main response with prefix, playback concurrent with
            # wake detection (barge-in).
            self._go(Trigger.RESPONSE_COMPLETE)
            stripped = full_response.strip()
            if not stripped:
                # LLM yielded nothing — common when cloud rejected the query
                # and local fallback also returned empty. Speak a fallback so
                # the user always gets some audio after the filler.
                logger.warning("LLM returned empty response; using fallback message")
                stripped = "I'm afraid I don't have an answer for that."
            # Gate on whether a filler actually played this turn, not just
            # the static use_filler flag — a simple query with fast_llm
            # configured skips the filler entirely, so the prefix (which
            # exists to smooth the handoff from filler audio) would be a
            # non-sequitur there.
            if filler_task is not None and self._response_prefix:
                spoken_text = self._response_prefix + stripped
            else:
                spoken_text = stripped
            logger.info(f"speaking: {len(spoken_text)} chars: {spoken_text[:80]!r}")
            t0_tts = time.monotonic()
            t_first_audio: float | None = None
            chunks_played = 0
            barge_in = asyncio.Event()

            async def play_tts() -> None:
                nonlocal t_first_audio, chunks_played
                async for chunk in self._tts.synthesize(spoken_text):
                    if self._cancel or barge_in.is_set():
                        return
                    if t_first_audio is None:
                        t_first_audio = time.monotonic()
                    chunks_played += 1
                    await self._speaker.play(chunk)
                    await asyncio.sleep(0)  # yield so watch_for_barge_in can run

            async def watch_for_barge_in() -> None:
                async for frame in self._mic.frames():
                    if self._cancel:
                        return
                    if await self._wake.detect(frame):
                        barge_in.set()
                        return

            play_task = asyncio.create_task(play_tts())
            watch_task = asyncio.create_task(watch_for_barge_in())
            _done, pending = await asyncio.wait(
                {play_task, watch_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in pending:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

            tts_ttfa_ms = (
                (t_first_audio - t0_tts) * 1000 if t_first_audio is not None else -1
            )
            tts_total_ms = (time.monotonic() - t0_tts) * 1000

            if barge_in.is_set():
                # Wake fired mid-playback → go back to LISTENING via state machine,
                # then reset to IDLE so the next run_once cycle starts clean.
                logger.info(
                    f"timing: TTS ttfa {tts_ttfa_ms:.0f}ms, total {tts_total_ms:.0f}ms "
                    f"({chunks_played} chunks, barge-in)"
                )
                self._go(Trigger.WAKE_DETECTED)
                self._state = State.IDLE
                self._emit(StateEvent(value=State.IDLE))
                return

            # Wait for the speaker buffer to fully drain before returning IDLE,
            # otherwise the next run_once starts mic-listening while the tail
            # of our own response is still playing and the wake-word detector
            # picks up TTS echo as input.
            await self._speaker.drain()
            logger.info(
                f"timing: TTS ttfa {tts_ttfa_ms:.0f}ms, total {tts_total_ms:.0f}ms "
                f"({chunks_played} chunks)"
            )

            # End-of-turn summary: helpful for spotting which stage dominates.
            if t_wake is not None and t_speech_end is not None:
                listen_ms = (t_speech_end - t_wake) * 1000
                full_turn_ms = (time.monotonic() - t_wake) * 1000
                logger.info(
                    f"timing: TURN listen {listen_ms:.0f}ms + STT {stt_ms:.0f}ms + "
                    f"LLM-ttft {llm_ttft_ms:.0f}ms + TTS-ttfa {tts_ttfa_ms:.0f}ms "
                    f"= total {full_turn_ms:.0f}ms"
                )

            # 5. IDLE
            self._go(Trigger.PLAYBACK_DONE)

        except asyncio.CancelledError:
            # Propagate cancellation; don't masquerade as an error event.
            self._state = State.IDLE
            self._emit(StateEvent(value=State.IDLE))
            raise
        except Exception as exc:
            self._emit(ErrorEvent(stage=self._state.value, msg=str(exc), retriable=False))
            self._state = State.ERROR
            self._emit(StateEvent(value=State.ERROR))
            # always return to IDLE
            self._state = State.IDLE
            self._emit(StateEvent(value=State.IDLE))

    async def _speak_text(self, text: str) -> None:
        """Synthesise + play a short utterance through the speaker. Used for
        filler acknowledgements that should not be interruptable."""
        if not text.strip():
            return
        async for chunk in self._tts.synthesize(text):
            if self._cancel:
                return
            await self._speaker.play(chunk)

    async def _react_and_speak(self, transcript: str) -> None:
        """Ask the fast model to react to the transcript and speak that
        reaction. Used instead of a canned filler phrase for complex
        queries, so the user hears something relevant to what they asked
        while the slower/cloud backend computes the real answer. Failures
        here are swallowed — a missing reaction just means silence until
        the real answer, not a broken turn."""
        if self._fast_llm is None:
            return
        reaction = ""
        try:
            async for ev in self._fast_llm.chat(
                [
                    {"role": "system", "content": _REACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ]
            ):
                if ev.kind == "chunk":
                    reaction += ev.text
                elif ev.kind == "done":
                    reaction = ev.text or reaction
        except Exception as exc:
            logger.warning(f"reaction generation failed (ignored): {exc!r}")
            return
        logger.info(f"reaction: {reaction!r}")
        await self._speak_text(reaction)
