# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Jarvis: offline-first voice assistant, persona-named "Jeanny" in its system prompt (`settings.llm_system_prompt`) — repo/code identifiers still say "jarvis" throughout, don't rename them to match. Wake-word → VAD → STT → LLM → TTS pipeline, driven by a Python/FastAPI backend and a Tauri (React + Three.js) desktop HUD frontend connected over a single WebSocket. Currently Phase 1 + 1.5 (core voice loop, barge-in, hotkey trigger). No tools/system-control, no vision, no conversation memory yet — those are later phases. "Offline-first" is now an overstatement for TTS specifically: the default voice backend (`edge-tts`) needs internet — see `settings.tts_backend`.

There's a second, separate surface: `backend/src/jarvis/remote_api.py` + `streamlit_app.py` (repo root) — a text/voice chat API meant to be tunneled to the internet, with a thin Streamlit Cloud frontend. See "Remote API + Streamlit frontend" below before touching either file; it has different security constraints than the main app.

## Commands

### Backend (`backend/`, Python 3.11, managed with `uv`)
```bash
cd backend && uv sync                      # install deps (or: pip install -e ".[dev]")
cd backend && uv run pytest                # run all tests
cd backend && uv run pytest tests/test_orchestrator.py -v          # single file
cd backend && uv run pytest tests/test_orchestrator.py::test_name  # single test
cd backend && uv run mypy src/             # type check (strict mode, see pyproject.toml)
cd backend && uv run ruff check src/       # lint
```

### Frontend (`frontend/`, Node 20+)
```bash
cd frontend && npm install
cd frontend && npm test                    # vitest run
cd frontend && npm run test:watch          # vitest watch mode
cd frontend && npx tsc --noEmit            # type check
cd frontend && npm run tauri dev           # Tauri dev window (needs MSVC Build Tools + WebView2 on Windows)
```

### Run full stack
```bash
scripts/dev.sh     # starts backend (uvicorn, JARVIS_START_ORCHESTRATOR=1) + `npm run tauri dev` together
```
Requires Ollama running (`ollama serve`, model `llama3.2:1b` pulled) and models downloaded via `scripts/download_models.sh` (OpenWakeWord ONNX model + Piper voice, into `models/`, gitignored).

Both dev launchers (`scripts/dev.sh` and `scripts/start-jarvis.ps1`) default `JARVIS_USE_CLOUD=0` (local-only), which is the opposite of `Settings`' own default (`use_cloud=True`, cloud-preferred when a key is present). This is intentional — override with `JARVIS_USE_CLOUD=1` before launching if you want cloud routing during dev.

Backend alone, without loading heavy models (used by tests/imports): `JARVIS_START_ORCHESTRATOR` unset or `0`. Set to `1` to actually wire up mic/wake/VAD/STT/LLM/TTS on startup (`app.py`'s `_startup` hook).

### Remote API + Streamlit frontend (one-click: `scripts/start-streamlit-remote.ps1`)
```bash
cd backend && JARVIS_REMOTE_API_KEY="pick-a-long-random-secret" uv run uvicorn jarvis.remote_api:app --host 0.0.0.0 --port 8800
cloudflared tunnel --url http://localhost:8800     # or: ngrok http 8800
streamlit run streamlit_app.py                     # local preview; prod deploy is Streamlit Community Cloud
```
`remote_api.py` refuses to start without `JARVIS_REMOTE_API_KEY` set — it's designed to sit behind a public tunnel, so "no key configured" fails closed rather than starting wide open. The one-click script sources this key from a gitignored `remote_api_key.txt` at repo root instead of an inline env var — generate that file first (see `STREAMLIT_SETUP.md` for the full tunnel/deploy walkthrough).

## Architecture

### Backend: state machine + orchestrator

The voice loop is a strict state machine (`state_machine.py`): `IDLE → LISTENING → TRANSCRIBE → THINKING → SPEAKING → IDLE`, with `CANCEL` universally returning to `IDLE` and barge-in (wake word detected again during `SPEAKING`) jumping back to `LISTENING`. The table-driven `transition()` function is pure and has no I/O — all actual work happens in `orchestrator.py`'s `Orchestrator.run_once()`, which runs one full IDLE→IDLE cycle and drives the state machine as it goes through:

1. **IDLE** — read mic frames, run wake-word detection (or skip straight to LISTENING on manual-start).
2. **LISTENING** — record frames, run VAD + `SpeechEndDetector` (silence-based) until speech end.
3. **TRANSCRIBE** — STT on the recorded PCM, then a filler decision (see "Fast/slow filler" below): either a canned phrase (started *before* STT, masking STT latency too), a transcript-aware reaction from the fast model (started *after* STT, since it needs the text), or no filler at all. Whichever filler plays runs concurrently via a separate task; both audio paths share one `Speaker` stream, so the filler must fully drain before the main response starts.
4. **THINKING** — full LLM response is buffered (not streamed to speaker) because of the single-stream constraint above; `ResponseChunkEvent`/`ResponseDoneEvent` still stream to the frontend as they arrive.
5. **SPEAKING** — TTS playback races against a concurrent wake-word watcher (`watch_for_barge_in`) via `asyncio.wait(..., FIRST_COMPLETED)`; whichever fires first wins (normal completion → IDLE, wake detected → barge-in → LISTENING).

**Fast/slow filler (`Orchestrator._react_and_speak`):** when `fast_llm` is configured (app.py only wires this when a cloud backend is also active — no point racing local against itself), the orchestrator re-scores the transcript with `complexity_score()` against `complexity_threshold` *independently* of whatever `self._llm` (the `HybridLlm`) does internally. Simple queries get no filler — the main answer (routed to local by `HybridLlm`'s own `complexity` strategy) is fast enough already, and skipping the filler also skips `response_prefix` (gated on `filler_task is not None`, not the static `use_filler` flag). Complex queries get `fast_llm` asked to react to the actual transcript (not answer it — see `_REACTION_SYSTEM_PROMPT`) and that reaction is spoken as the filler, while `self._llm` computes the real answer (routed to cloud) concurrently. A `fast_llm` reaction failure is swallowed — the turn still completes with the real answer, just without a filler.

**QA fast-path (`qa_lookup.py` + `qa_beadchip.py`):** checked right after STT, before the filler decision — takes priority over everything else except the coding fast-path below. `Orchestrator`'s optional `qa_matcher: Callable[[str], str | None]` is called with the transcript; a match skips both the filler *and* the LLM entirely (`self._llm.chat()` never runs), answering with the pre-written text immediately. Matching is plain keyword overlap (`qa_lookup.best_match`, no embeddings) against a hand-curated dataset — currently `qa_beadchip.BEADCHIP_QA` (Illumina BeadChip/Infinium assay reference questions, mirrors `~/Desktop/Illumina_BeadChip_Referenz/`). `best_match` strips German stopwords and requires a minimum absolute keyword overlap (`min_overlap`, default 2) before scoring — without both, a short entry reduced to one content keyword by stopword-stripping trivially "wins" any query containing that word, regardless of real relevance (this was a real bug: vague questions kept matching the shortest formamide entry purely on "was"/"ist"). To add another topic's Q&A set, write a new `tuple[QaEntry, ...]` via `qa_lookup.make_entry()` and union it into app.py's wiring.

**Coding fast-path (`coding.py`):** highest priority, checked before the QA match. `Orchestrator`'s optional `coding_handler: Callable[[str], bool]` is called with the transcript; if it returns `True` it has *already* fired a detached `asyncio.create_task` (fire-and-forget — the task outlives the turn and isn't awaited anywhere) and the orchestrator just speaks a fixed ack and skips the LLM, so the voice turn completes immediately regardless of how long generation takes. The background task (defined in `app.py`'s `handle_coding_request`) calls a dedicated small model (`settings.coding_model`, default `qwen2.5-coder:1.5b` — a separate Ollama load from the voice-response model, so code generation doesn't change what model answers normal questions), extracts the fenced ` ```python ` block (`coding.extract_python_code`), saves it under `settings.coding_scripts_dir` (default `~/Desktop/Jarvis_Scripts/`, filenames always generated via `uuid4`, never derived from model output or speech) and launches it in its own console window (`coding.run_script_in_new_window`, `CREATE_NEW_CONSOLE | BELOW_NORMAL_PRIORITY_CLASS` — the lowered priority is the only concrete mitigation against the generated script's CPU use starving the voice pipeline on this machine's limited cores; asyncio-level backgrounding alone doesn't prevent OS-level CPU contention between separate processes). **No confirmation step before running — explicit user choice, auto-run accepted in exchange for speed; Jarvis executes arbitrary generated code with the user's own privileges, no sandbox.**

All components the orchestrator depends on (mic, speaker, wake, VAD, STT, LLM, TTS) are structural `Protocol`s, each with a `Real*` implementation (wraps the actual library) and a fake/test double — this is what makes `Orchestrator` unit-testable without hardware or models. Wire-up of the real implementations only happens in `app.py`'s FastAPI `startup` hook, gated by `JARVIS_START_ORCHESTRATOR`, so importing `jarvis.app` (as tests do) never loads models.

**TTS backend (`tts.py`):** two implementations behind the same `Tts` protocol, picked by `settings.tts_backend` ("edge" default, "piper" fallback) in `app.py`. `EdgeTts` (Microsoft's free neural voices via the `edge-tts` package, no API key) buffers the full MP3 response before decoding — individual streamed MP3 chunks aren't independently decodable — then sub-chunks the decoded PCM for playback; needs internet, no local model file. `RealTts` is the original Piper path, fully offline but the only German female Piper voices (eva_k/kerstin/ramona) are low/x-low quality tier, which is why `edge-tts` is now the default. **`settings.tts_sample_rate` must match whichever backend is active** (`RealSpeaker` is constructed with it) — edge-tts/Katja outputs 24000 Hz, Piper outputs 22050 Hz; mismatch plays audio at the wrong pitch/speed.

### Remote API (`remote_api.py`) + Streamlit frontend (`streamlit_app.py`)

Deliberately separate FastAPI app from `app.py` — not a mode switch on the same server. Reuses `HybridLlm`/STT/TTS (same config, same routing) but drops everything that's unsafe or pointless to expose publicly: **no coding fast-path** (auto-running generated code on an internet-reachable endpoint would be remote code execution), **no WebSocket**, no mic/speaker/wake/VAD — just two HTTP routes, `/chat` (JSON text) and `/voice` (upload WAV, get back transcript + reply + base64 WAV), both gated by a static `X-API-Key` header checked against `settings.remote_api_key`. Time queries and timer requests (`is_time_query`/`is_timer_query`) are regex-detected and answered deterministically here, independent of anything in `qa_lookup.py` — small local models reliably hallucinate on these otherwise (e.g. `llama3.2:1b` asked "wie spät ist es" answered with London's weather despite the injected time context), so these bypass the LLM entirely rather than trusting it to read the system message. The BeadChip QA fast-path (`beadchip_fastpath_answer`) is *not* independent the same way: it calls the same `qa_lookup.best_match` keyword-overlap matcher against the same `qa_beadchip.BEADCHIP_QA` dataset that `app.py` wires into the main voice app's `qa_matcher`, only adding `reagent_answer()` (`qa_beadchip.py`) as an extra bypass for bare reagent abbreviations that `best_match` wouldn't catch.

Voice-set timers: the countdown and alarm playback happen **client-side** in the Streamlit voice widget (plain JS inside a `components.html` iframe) — this API has no per-user session state and the browser tab is what's actually open, so the server's job is only to parse a duration out of speech deterministically (`parse_timer_seconds`, German number-words + units) and hand back `timer_seconds`/`timer_name` for the widget to run. `streamlit_app.py` itself never runs a model — it's a thin client that calls the tunneled `remote_api` URL (pasted into the sidebar, since the tunnel URL changes every restart) directly from the browser for voice, and through Streamlit's own server for typed chat (`st.chat_input` → `/chat`). These two paths keep separate transcripts: the iframe has no channel back into Streamlit's Python session state.

### LLM routing (`llm.py`)

`HybridLlm` wraps a local Ollama backend and an optional cloud backend (`OpenAiCompatLlm` — generic OpenAI-compatible client, works for Groq or DeepSeek). The cloud backend is only constructed if a key is found: Groq preferred (`JARVIS_GROQ_API_KEY` or `Groq_api_key.txt`), else DeepSeek (`JARVIS_DEEPSEEK_API_KEY` or `Deepseek_api_key.txt`), both files gitignored, resolved in `app.py`'s startup hook. No cloud key at all → local-only (`llama3.2:1b`, chosen for CPU speed since it's mostly an emergency fallback). Two routing strategies, set via `JARVIS_ROUTE_STRATEGY`:
- `cloud_first`: always try the configured cloud backend (Groq if present, else DeepSeek); fall back to local only if it fails *before* yielding a first chunk (avoids duplicating partial output).
- `complexity` (default): score the query (`complexity_score()` — keyword + length heuristics, German + English) against `JARVIS_COMPLEXITY_THRESHOLD`; simple → local, complex → cloud.

Every call gets a German-localized current-time system message injected (`realtime_context()`), so time/date questions don't need a separate tool.

### Events contract (`events.py`)

Outbound (backend → frontend) and inbound (frontend → backend) WebSocket messages are pydantic models in a discriminated union (`Event` / `InboundEvent`, keyed on `type`). The frontend's `InboundEvent` type in `frontend/src/store.ts` must be kept in sync by hand with `events.py` — there's no shared schema generation. `app.py`'s single `/ws` route broadcasts outbound events to all connected clients (`WsHub`) and routes inbound events through an `inbound_handler` set up in the startup hook, which pushes onto an `asyncio.Queue` drained by the orchestrator loop (`CancelEvent` → `request_cancel()`, `ManualStartEvent` → `set_manual_start()`).

### Frontend

`store.ts` (zustand) holds all pipeline state and the `handleEvent` reducer for inbound WS events; `ws.ts` owns the WebSocket connection. `hud/` contains the Reactor Core HUD (Three.js ring via `@react-three/fiber`/`drei`) plus waveform and status-corner overlays, all reading from the store. `frontend/src-tauri/src/main.rs` registers a global `Win+J` shortcut that emits a Tauri `hotkey` event the frontend listens for to send a `manual_start` inbound event — this is the "skip wake-word" fast path also triggered by clicking the HUD.

### Config (`config.py`)

All runtime settings are env-var driven (`JARVIS_*` prefix) with defaults, loaded once via `get_settings()` (lru_cache). No config files — check this module for the full list rather than grepping for magic strings elsewhere.

### Plugin scaffold (`plugin/`)

`Tool.register()` decorator + a global `Registry` for OpenAI-style function-calling tools (`helpers.py` is empty). Nothing calls into this yet — no tool is registered anywhere, `HybridLlm`/`RealLlm` don't pass tool schemas to the model. This is Phase 2 scaffolding only; don't assume registered tools exist or are wired into the orchestrator.

## Project docs

- `docs/superpowers/specs/2026-05-14-jarvis-hud-design.md` — HUD design spec
- `docs/superpowers/plans/2026-05-14-jarvis-phase1-core.md`, `.../2026-05-14-jarvis-phase1-5-finish.md` — phase plans (written by the `superpowers` skill set; `.original.md` siblings are pre-compression full-prose backups, not meant to be edited)
- `README.md` — setup prerequisites and the Phase 1.5 manual smoke-test runbook
- `STREAMLIT_SETUP.md` — tunnel + Streamlit Community Cloud deploy walkthrough for `remote_api.py`/`streamlit_app.py`
