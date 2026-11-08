"""FastAPI app + WebSocket route. Holds orchestrator lifecycle."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable, Coroutine

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

from jarvis import __version__
from jarvis.config import get_settings
from jarvis.events import (
    CancelEvent,
    ConfirmToolEvent,
    ErrorEvent,
    Event,
    ManualStartEvent,
    parse_inbound,
)

InboundEvent = ManualStartEvent | CancelEvent | ConfirmToolEvent
InboundHandler = Callable[[InboundEvent], Coroutine[Any, Any, None]]


class WsHub:
    """Tracks connected clients and broadcasts events."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def remove(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(json.dumps(payload))
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)


def create_app(start_orchestrator: bool = True) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Jarvis", version=__version__)
    hub = WsHub()
    app.state.hub = hub
    app.state.start_orchestrator = start_orchestrator

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await hub.add(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    ev = parse_inbound(raw)
                except (ValueError, json.JSONDecodeError) as exc:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "stage": "ws",
                                "msg": f"parse failed: {exc}",
                                "retriable": True,
                            }
                        )
                    )
                    continue
                # Phase 1: route inbound events through orchestrator hook (set up below)
                handler: InboundHandler | None = getattr(
                    app.state, "inbound_handler", None
                )
                if handler is not None:
                    await handler(ev)
        except WebSocketDisconnect:
            pass
        finally:
            await hub.remove(ws)

    @app.on_event("startup")
    async def _startup() -> None:
        if not app.state.start_orchestrator:
            return
        # Wire real orchestrator only when explicitly enabled. Tests use
        # start_orchestrator=False so heavy models never load.
        from jarvis.orchestrator import Orchestrator
        from jarvis.qa_topics import fastpath_answer, topic_context as qa_topic_context
        from jarvis.coding import (
            decide_coding_action,
            extract_python_code,
            new_id,
            progress_path_for,
            run_script_in_new_window,
            save_launcher,
            save_script,
            write_progress,
        )
        from jarvis.audio_io import RealMicStream, RealSpeaker
        from jarvis.wake import RealWakeWord
        from jarvis.vad import RealVad
        from jarvis.stt import RealStt
        from jarvis.llm import (
            HybridLlm,
            Llm,
            OpenAiCompatLlm,
            RealLlm,
            load_deepseek_key,
            load_groq_key,
            select_cloud_provider,
        )
        from jarvis.tts import EdgeTts, RealTts, Tts

        def emit(ev: Event) -> None:
            asyncio.create_task(hub.broadcast(ev.model_dump(mode="json")))

        inbound_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()
        app.state.inbound_queue = inbound_queue

        async def inbound_handler(ev: InboundEvent) -> None:
            await app.state.inbound_queue.put(ev)
            logger.info(f"inbound queued: {ev.type}")

        app.state.inbound_handler = inbound_handler

        async def _orch_loop() -> None:
            # Build heavy components once. Mic stream stays open across cycles.
            # Sample rate must match whichever TTS backend is active:
            # edge-tts/Katja outputs 24000 Hz, Piper outputs 22050 Hz.
            speaker = RealSpeaker(sample_rate=settings.tts_sample_rate)
            wake = RealWakeWord(
                model_path=settings.wake_model_path,
                threshold=settings.wake_threshold,
            )
            vad = RealVad(
                threshold=settings.vad_threshold, sample_rate=settings.sample_rate
            )
            stt = RealStt(
                model_name=settings.whisper_model,
                compute_type=settings.whisper_compute,
                language=settings.whisper_lang,
                initial_prompt=settings.whisper_initial_prompt,
            )
            local_llm: Llm = RealLlm(
                host=settings.llm_host,
                model=settings.llm_model,
                system_prompt=settings.llm_system_prompt,
                timeout_s=settings.llm_timeout_s,
                max_tokens=settings.llm_max_tokens,
            )
            # Separate instance (same model/host) with a small token cap —
            # the reaction-filler call needs to stay fast, which the full
            # answer's token budget doesn't guarantee (a small model won't
            # reliably stop itself early just because the prompt asked it to).
            reaction_llm: Llm = RealLlm(
                host=settings.llm_host,
                model=settings.llm_model,
                timeout_s=settings.llm_timeout_s,
                max_tokens=settings.reaction_max_tokens,
            )

            # Three separate Ollama model loads (small/medium/large), not
            # one — a code-generation request starts on the fast "small"
            # tier and only loads a bigger one if the upgrade trigger
            # actually fires, so most requests never pay for anything
            # bigger than coding_model_small.
            _CODING_SYSTEM_PROMPT = (
                "Du bist ein Python-Coding-Assistent. Antworte "
                "AUSSCHLIESSLICH mit einem vollständigen, lauffähigen "
                "Python-Skript in einem einzigen ```python ... ``` "
                "Codeblock. Kein Fließtext davor oder danach. Nutze "
                "für grafische Oberflächen tkinter (Standardbibliothek)."
            )
            coding_models = {
                "small": settings.coding_model_small,
                "medium": settings.coding_model_medium,
                "large": settings.coding_model,
            }
            coding_llms: dict[str, Llm] = {
                tier: RealLlm(
                    host=settings.llm_host,
                    model=model_name,
                    system_prompt=_CODING_SYSTEM_PROMPT,
                    timeout_s=settings.llm_timeout_s,
                    max_tokens=settings.coding_max_tokens,
                )
                for tier, model_name in coding_models.items()
            }
            # Mutable across calls within this orchestrator's lifetime —
            # tracks the tier + prompt text so an upgrade trigger
            # ("größeres Modell") knows what to re-run and at what tier.
            # Reset to coding_default_tier on every NEW coding request;
            # only the upgrade trigger bumps it, and only for a re-run of
            # that same last request.
            coding_state: dict[str, str | None] = {
                "tier": settings.coding_default_tier,
                "last_prompt": None,
            }

            def _dispatch_coding(text: str, tier: str) -> None:
                model_name = coding_models[tier]
                coding_llm = coding_llms[tier]
                script_id = new_id()
                progress_path = progress_path_for(script_id, settings.coding_scripts_dir)
                write_progress(
                    progress_path,
                    status="generating",
                    tokens=0,
                    max_tokens=settings.coding_max_tokens,
                    model=model_name,
                )
                launcher_path = save_launcher(progress_path, settings.coding_scripts_dir, script_id)
                pid = run_script_in_new_window(launcher_path)
                logger.info(f"coding: launcher started (pid {pid}, tier={tier}, model={model_name})")

                async def _generate_and_run() -> None:
                    try:
                        full = ""
                        async for ev in coding_llm.chat([{"role": "user", "content": text}]):
                            if ev.kind == "chunk":
                                full += ev.text
                                # chars/4 ~= tokens: a rough estimate for the
                                # progress bar, not an exact token count —
                                # Ollama's stream only gives text chunks.
                                write_progress(
                                    progress_path,
                                    status="generating",
                                    tokens=len(full) // 4,
                                    max_tokens=settings.coding_max_tokens,
                                    model=model_name,
                                )
                            elif ev.kind == "done":
                                full = ev.text or full
                        code = extract_python_code(full)
                        if code is None:
                            logger.warning(
                                f"coding: model produced no code block for {text!r}"
                            )
                            write_progress(
                                progress_path,
                                status="error",
                                error="Kein gültiger Code-Block erkannt.",
                            )
                            return
                        path = save_script(code, settings.coding_scripts_dir)
                        write_progress(progress_path, status="done", script_path=str(path))
                        logger.info(f"coding: generated {path} (tier={tier})")
                    except Exception as exc:
                        logger.warning(f"coding: background task failed (ignored): {exc!r}")
                        write_progress(progress_path, status="error", error=str(exc))

                asyncio.create_task(_generate_and_run())

            def handle_coding_request(text: str) -> bool:
                """Returns True (and has already fired a detached
                background task) if this transcript is a code-generation
                request OR an upgrade-tier request for the last one;
                False otherwise, so the caller falls through to the
                normal QA/LLM path. Generation + execution happen in a
                task that outlives this turn — intentionally not awaited
                here."""
                last_prompt = coding_state["last_prompt"]
                decision = decide_coding_action(
                    text,
                    str(coding_state["tier"]),
                    last_prompt if isinstance(last_prompt, str) else None,
                    str(settings.coding_default_tier),
                )
                if decision is None:
                    return False
                prompt, tier = decision
                coding_state["tier"] = tier
                coding_state["last_prompt"] = prompt
                try:
                    _dispatch_coding(prompt, tier)
                except Exception as exc:
                    logger.warning(f"coding: dispatch prelude failed, falling through: {exc!r}")
                    return False
                return True

            # Pre-warm: tell Ollama to load the model into RAM now. First real
            # call would otherwise pay a 20-60 s model-load cost. We fire a
            # tiny one-token chat as a background task so startup isn't blocked.
            async def _prewarm_local() -> None:
                try:
                    async for _ in local_llm.chat(
                        [{"role": "user", "content": "hi"}]
                    ):
                        pass
                    logger.info("LLM pre-warm done")
                except Exception as exc:
                    logger.warning(f"LLM pre-warm failed (orchestrator will retry): {exc}")

            asyncio.create_task(_prewarm_local())
            # Prefer Groq (far lower latency, OpenAI-compatible) over DeepSeek
            # when both keys are present; local-only if neither is set or
            # JARVIS_USE_CLOUD=0 forces local regardless of keys.
            groq_key = load_groq_key() if settings.use_cloud else ""
            ds_key = load_deepseek_key() if settings.use_cloud else ""
            provider = select_cloud_provider(groq_key=groq_key, deepseek_key=ds_key)
            cloud_llm: Llm | None = None
            cloud_model = ""
            if provider == "groq":
                cloud_model = settings.groq_model
                cloud_llm = OpenAiCompatLlm(
                    api_key=groq_key,
                    model=settings.groq_model,
                    system_prompt=settings.llm_system_prompt,
                    base_url=settings.groq_base_url,
                    timeout_s=settings.llm_timeout_s,
                    max_tokens=settings.llm_max_tokens,
                )
            elif provider == "deepseek":
                cloud_model = settings.deepseek_model
                cloud_llm = OpenAiCompatLlm(
                    api_key=ds_key,
                    model=settings.deepseek_model,
                    system_prompt=settings.llm_system_prompt,
                    base_url=settings.deepseek_base_url,
                    timeout_s=settings.llm_timeout_s,
                    max_tokens=settings.llm_max_tokens,
                )
            if cloud_llm is not None:
                logger.info(
                    f"LLM routing: hybrid (local={settings.llm_model}, "
                    f"cloud={cloud_model}, threshold={settings.complexity_threshold})"
                )
            elif not settings.use_cloud:
                logger.info(
                    f"LLM routing: local only ({settings.llm_model}) — "
                    "JARVIS_USE_CLOUD=0 (forced local)"
                )
            else:
                logger.info(
                    f"LLM routing: local only ({settings.llm_model}) — "
                    "no Groq/DeepSeek key found"
                )
            from typing import cast as _cast
            from jarvis.llm import RouteStrategy as _RouteStrategy
            llm: Llm = HybridLlm(
                local=local_llm,
                cloud=cloud_llm,
                threshold=settings.complexity_threshold,
                strategy=_cast(_RouteStrategy, settings.route_strategy),
            )
            tts: Tts
            if settings.tts_backend == "edge":
                tts = EdgeTts(voice=settings.tts_edge_voice)
            else:
                tts = RealTts(
                    model_path=settings.tts_model_path,
                    length_scale=settings.tts_length_scale,
                )
            silence_frame_count = int(
                settings.silence_ms * settings.sample_rate / 1000 / 512
            )

            async with RealMicStream(sample_rate=settings.sample_rate) as mic:
                orch = Orchestrator(
                    mic=mic,
                    speaker=speaker,
                    wake=wake,
                    vad=vad,
                    stt=stt,
                    llm=llm,
                    tts=tts,
                    emit=emit,
                    silence_frames=silence_frame_count,
                    sample_rate=settings.sample_rate,
                    use_filler=settings.use_filler,
                    response_prefix=settings.response_prefix,
                    min_transcript_words=settings.min_transcript_words,
                    # Only wire the fast-reaction path when there's actually
                    # a slower cloud backend whose latency it's masking —
                    # with no cloud, local would just race against itself.
                    fast_llm=reaction_llm if cloud_llm is not None else None,
                    complexity_threshold=settings.complexity_threshold,
                    max_listen_s=settings.max_listen_s,
                    qa_matcher=fastpath_answer,
                    topic_context=qa_topic_context,
                    coding_handler=handle_coding_request,
                )

                async def drain_inbound() -> None:
                    while True:
                        ev = await app.state.inbound_queue.get()
                        if isinstance(ev, CancelEvent):
                            orch.request_cancel()
                        elif isinstance(ev, ManualStartEvent):
                            orch.set_manual_start()

                drain = asyncio.create_task(drain_inbound())
                try:
                    while True:
                        try:
                            await orch.run_once()
                        except Exception as exc:
                            logger.exception("orchestrator crashed")
                            await hub.broadcast(
                                ErrorEvent(
                                    stage="orchestrator", msg=str(exc), retriable=True
                                ).model_dump(mode="json")
                            )
                            await asyncio.sleep(1.0)
                finally:
                    drain.cancel()

        app.state.orch_task = asyncio.create_task(_orch_loop())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        task: asyncio.Task[None] | None = getattr(app.state, "orch_task", None)
        if task is not None:
            task.cancel()

    return app


# Controlled by JARVIS_START_ORCHESTRATOR env var. Default OFF so test imports
# don't trigger model loads; scripts/dev.sh sets it to 1.
app = create_app(start_orchestrator=os.getenv("JARVIS_START_ORCHESTRATOR", "0") == "1")
