# Jarvis HUD — Design Spec

**Date:** 2026-05-14
**Status:** Design approved by user, ready for implementation plan
**Scope vision:** Vollvision (C) — Voice-Chat + System/Web/Files + Calendar/Mail + Vision, phasiert in 3 Releases.

## 1. Goal

Offline-fähiger persönlicher Sprachassistent mit cinematic Vollbild-HUD im Iron-Man-Stil ("Reactor Core"). Wake-Word-getriggert, lokales LLM, lokale STT/TTS. CPU/iGPU Hardware-Constraint. Endziel: System-Steuerung + Web/File-Suche + Calendar/Mail + Vision via Plugins.

### Non-Goals

- Mobile-Versionen.
- Multi-User / Multi-Tenant.
- Cloud-only Modi (Cloud nur als optionaler Fallback in Phase 3 für Vision-Multimodal).

## 2. Constraints

- **Hardware:** CPU / iGPU. Keine dedicated GPU vorausgesetzt. → LLM ≤ 3B Parameter Q4_K_M, Whisper-base CPU, Piper TTS, kein LLaVA on-device.
- **Privacy:** Default 100% offline. Externe APIs (Web-Such-Provider, Cloud-LLM) sind Opt-in pro Tool.
- **OS:** Windows 11 primär (Entwicklungssystem). Linux/macOS Best-effort.
- **Sprache:** Deutsch primär (Whisper multilingual unterstützt en-Fallback).

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  FRONTEND · Tauri Window · Vollbild HUD                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐  │
│  │ HUD Renderer │ │ Audio Visual │ │ State Store (zustand)│  │
│  │ React+Three  │ │ Web Audio FFT│ │ idle/listen/think/.. │  │
│  └──────────────┘ └──────────────┘ └──────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                     ↕ WebSocket ws://127.0.0.1:8765 (JSON)
┌──────────────────────────────────────────────────────────────┐
│  BACKEND · Python FastAPI · asyncio Orchestrator             │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ Wake Word   │→ │ VAD         │→ │ STT         │           │
│  │ OpenWakeWord│  │ Silero      │  │ faster-     │           │
│  │ "jarvis"    │  │             │  │ whisper-base│           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                            ↓                 │
│                     ┌──────────────────────────────┐         │
│                     │ LLM Engine · Ollama          │         │
│                     │ Llama-3.2-3B-Instruct Q4_K_M │         │
│                     │ OpenAI-compat tool-calling   │         │
│                     └──────────────────────────────┘         │
│                                            ↓                 │
│  ┌──────────────────────────────────────────────────┐        │
│  │ Tool Router · Plugin Registry                    │        │
│  │  system · web · files · calendar · mail · vision │        │
│  └──────────────────────────────────────────────────┘        │
│                                            ↓                 │
│                     ┌─────────────┐  ┌─────────────┐         │
│                     │ TTS · Piper │→ │ Audio Out   │         │
│                     │ thorsten    │  │ sounddevice │         │
│                     └─────────────┘  └─────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Depends on |
|---|---|---|
| `frontend/` (Tauri) | Render HUD, capture user clicks, forward WS events to view | WS connection to backend |
| `backend/app.py` | FastAPI app, WebSocket route, lifecycle | Orchestrator |
| `backend/orchestrator.py` | State machine, pipeline asyncio loop | wake/vad/stt/llm/tts modules |
| `backend/wake.py` | OpenWakeWord wrapper, emits `wake_detected` event | mic capture |
| `backend/vad.py` | Silero VAD, emits `speech_start`/`speech_end` | mic stream |
| `backend/stt.py` | faster-whisper, batch transcribe buffered audio | audio buffer |
| `backend/llm.py` | Ollama client, tool-call protocol, streaming responses | Ollama server |
| `backend/tts.py` | Piper inference, streaming chunk emission | text input |
| `backend/audio_io.py` | mic capture + speaker playback (sounddevice), level meter | OS audio |
| `backend/plugin/` | Plugin loader, registry, contract decorator | individual plugin modules |
| `plugins/*.py` | Individual tools (web_search, sys_volume, etc.) | plugin contract |

Each component testable in isolation with mocked input.

## 4. State Machine

```
IDLE ──"jarvis" detected──→ LISTENING
LISTENING ──silence 800ms──→ TRANSCRIBE
LISTENING ──user "jarvis" (barge-in)──→ LISTENING (cancel previous)
TRANSCRIBE ──transcript ready──→ THINKING
THINKING ──tool_call──→ THINKING (loop, executing tool)
THINKING ──response complete──→ SPEAKING
SPEAKING ──playback done──→ IDLE
SPEAKING ──user wakes──→ LISTENING (cancel TTS)
ANY ──error──→ IDLE (emit error event, HUD shows red ring)
```

Manual triggers:
- HUD mic-click → directly to LISTENING (skip wake).
- Hotkey `Win+J` → directly to LISTENING.

## 5. WebSocket Event Schema

Backend → Frontend. JSON messages, one per line.

| Type | Payload | When |
|---|---|---|
| `state` | `{value: "idle"\|"listening"\|"transcribe"\|"thinking"\|"speaking"\|"error"}` | State transition |
| `audio_level` | `{rms: float, fft: float[N]}` | ~30Hz while LISTENING or SPEAKING |
| `transcript_partial` | `{text: string}` | During TRANSCRIBE (if partial supported) |
| `transcript_final` | `{text: string}` | TRANSCRIBE → THINKING |
| `tool_call` | `{name: string, args: object, call_id: string}` | LLM invoked tool |
| `tool_result` | `{call_id: string, ok: bool, summary: string}` | Tool finished |
| `response_chunk` | `{text: string}` | LLM streaming |
| `response_done` | `{full: string}` | THINKING → SPEAKING |
| `error` | `{stage: string, msg: string, retriable: bool}` | Any failure |

Frontend → Backend:
| Type | Payload | Effect |
|---|---|---|
| `manual_start` | `{}` | Skip wake, force LISTENING |
| `cancel` | `{}` | Abort current pipeline, → IDLE |
| `confirm_tool` | `{call_id: string, approved: bool}` | Response to confirm-required tool |

## 6. Plugin Contract

```python
# plugins/web_search.py
from jarvis.plugin import Tool, ToolResult

@Tool.register(
    name="web_search",
    description="Search web for current info",
    schema={
        "q": {"type": "string", "required": True},
        "k": {"type": "int", "default": 5}
    },
    confirm=False,        # True → voice confirmation required before exec
    timeout_s=10,
)
async def web_search(q: str, k: int = 5) -> ToolResult:
    results = await ddg.search(q, max_results=k)
    return ToolResult(ok=True, data=results, summary=f"{len(results)} Ergebnisse")
```

Rules:
- Plugins auto-discovered from `plugins/*.py` at startup.
- `Tool.register` stores metadata in `Registry`. `Registry.json_schemas()` returns OpenAI-compatible function specs for LLM tool-calling.
- `confirm=True` → orchestrator pauses, emits `tool_call` with `requires_confirm: true`, waits for `confirm_tool` event from frontend (voice or click). Used for destructive ops (shutdown, file delete, mail send).
- `timeout_s` enforced via asyncio. On timeout: `ToolResult(ok=False, summary="timeout")`.
- Plugin can import shared utilities from `jarvis.plugin.helpers`.

## 7. Phasen-Roadmap

### Phase 1 — Core (~3-4 Wochen)

**Goal:** End-to-end Voice-Loop. "Jarvis, hallo" → spricht zurück.

Deliverables:
- Tauri shell + Reactor-Core-HUD (idle ring, listening pulse, speaking waveform, thinking spinner)
- FastAPI + WebSocket bus
- Wake (OpenWakeWord "jarvis") + VAD (Silero) + STT (faster-whisper-base, int8 CPU)
- LLM via Ollama (Llama-3.2-3B-Instruct Q4_K_M)
- TTS (Piper, `de_DE-thorsten-medium`) with streaming chunks
- State machine + event protocol
- Tool-Registry framework (no tools yet)
- Manual trigger: HUD mic-click + `Win+J` hotkey

Exit criteria:
- E2E: Wake-word → STT → LLM (no tools) → TTS playback.
- HUD reflects each state.
- Barge-in works (interrupting TTS by saying "Jarvis").
- Unit + integration tests pass.

### Phase 2 — Tools (~3-4 Wochen)

**Goal:** Useful actions. "Jarvis, öffne Spotify und such Foo Fighters."

Plugins to ship:
- `sys_volume` — set/get volume (pycaw)
- `sys_app_launch` — open known apps (whitelisted dict)
- `sys_shell` — run shell command (confirm=True)
- `web_search` — DDG search + readability extraction
- `web_fetch` — fetch URL → markdown
- `files_search` — Everything-CLI integration
- `files_read` — read text/PDF, return summary
- `calendar_list` / `calendar_create` — Outlook COM or Google Calendar API (user picks at config)
- `mail_search` / `mail_send` — IMAP read + SMTP send (confirm=True for send)

Additional:
- HUD tool-panel: shows running tool + result
- Confirm-flow UI + voice
- Plugin hot-reload (file-watch → reimport)

Exit criteria:
- All listed plugins functional with tests.
- Tool-calling loop works (LLM can chain multiple tool calls).
- Destructive tools require + receive confirmation.

### Phase 3 — Vision + Polish (~4-6 Wochen)

**Goal:** Visual awareness + distributable build.

Deliverables:
- `vision_screen` — screenshot → CLIP embedding + OCR (Tesseract)
- `vision_describe` — combine CLIP labels + OCR text → LLM describes
- Optional Cloud-LLM-Fallback (Anthropic/OpenAI API) for multimodal queries — explicit opt-in via settings
- Conversation memory (SQLite, last N exchanges, semantic recall via embeddings)
- HUD chat-log overlay (toggleable)
- Settings UI (model picker, voice picker, plugin toggles)
- System tray icon + autostart option
- Installer: MSI (Wix) or NSIS

Exit criteria:
- Distributable installer builds clean.
- "Jarvis, was ist auf meinem Bildschirm?" produces sensible answer.
- Memory persists across restarts.

## 8. Error Handling

Per-stage error events propagate to frontend. HUD shows pulsing red ring + voice announcement.

| Stage | Failure | Behavior |
|---|---|---|
| wake | mic permission denied | Error event, retry prompt, fallback to manual-only mode |
| vad | mic disconnect | Reconnect attempt 3x, then error |
| stt | whisper exception | Speak "Verstanden habe ich nichts. Bitte wiederhol." → IDLE |
| llm | ollama unreachable | Speak "LLM nicht erreichbar." Retry start-ollama suggestion. |
| tool | timeout / exception | LLM informed of `ok: false`, continues response |
| tts | piper failure | Display text only, log error, continue |

Each error includes `retriable` flag. Orchestrator never crashes — always returns to IDLE.

## 9. Testing Strategy

- **Unit:** Each module tested in isolation. Mock audio I/O via fixture WAVs.
- **Integration:** Fixture audio → STT → mocked LLM → TTS → assert output audio. One per phase.
- **Plugin tests:** Each plugin has own test file. Mock external APIs (DDG, Outlook, etc.).
- **State machine:** Explicit transition tests for all paths.
- **E2E:** Playwright on Tauri HUD + scripted backend event injection.

Test pyramid: Unit > Plugin > Integration > E2E.

## 10. Risks

| Risk | Mitigation |
|---|---|
| **CPU latency 5-10s for full loop** | Streaming TTS + LLM chunks → user hears partial output ASAP. HUD shows activity continuously. |
| **Wake-word false positives** | Tunable threshold, optional model retraining on user voice samples, "tap mic" alternative. |
| **Plugin security (system tools)** | `confirm=True` for destructive ops, allowlist of safe apps, subprocess sandboxing. |
| **Vision on CPU unusable** | Phase 3 uses CLIP+OCR (not multimodal LLM). LLaVA only via cloud opt-in. |
| **Ollama not running** | Backend health-check on start, auto-launch `ollama serve` subprocess, surface error if fails. |
| **Three.js perf on iGPU** | Limit particles, fallback to CSS-only animation if FPS drops. Detect WebGL availability. |

## 11. Repository Structure

```
jarvis/
├── frontend/                # Tauri app
│   ├── src/
│   │   ├── App.tsx
│   │   ├── hud/             # Reactor Core components
│   │   ├── store.ts         # zustand state
│   │   └── ws.ts            # WS client
│   ├── src-tauri/
│   └── package.json
├── backend/
│   ├── app.py               # FastAPI entrypoint
│   ├── orchestrator.py
│   ├── wake.py
│   ├── vad.py
│   ├── stt.py
│   ├── llm.py
│   ├── tts.py
│   ├── audio_io.py
│   ├── plugin/
│   │   ├── __init__.py      # Tool decorator + Registry
│   │   └── helpers.py
│   └── tests/
├── plugins/                 # User-discoverable tools (Phase 2+)
│   ├── sys_volume.py
│   ├── web_search.py
│   └── ...
├── models/                  # Downloaded models (gitignored)
│   ├── whisper-base/
│   ├── piper-de-thorsten/
│   └── openwakeword-jarvis.tflite
├── docs/superpowers/
│   └── specs/
│       └── 2026-05-14-jarvis-hud-design.md
├── pyproject.toml
└── README.md
```

## 12. Open Questions (defer to plan phase)

- Konkretes Wake-Word-Modell: OpenWakeWord-Built-in vs. custom-trained on user-voice samples
- Calendar provider default: Outlook vs Google (user wählt bei first-run)
- Mail send confirm: voice-only ("ja sende") vs HUD-button
- Conversation memory: full transcript vs summarized

## 13. Definition of Done (project)

- Alle 3 Phasen ausgeliefert.
- E2E-Tests grün.
- README + Quickstart-Doku.
- Installer für Windows funktioniert auf frischer VM.
- Mindestens 5 Plugins funktionsfähig.
- User kann offline kompletten Voice-Loop nutzen ohne externes Netzwerk.
