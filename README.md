# Jarvis — Offline Voice Assistant

Phase 1 + 1.5 complete: Wake-word → STT → local LLM → TTS pipeline with Reactor Core HUD, barge-in support, hotkey + click triggers, real inbound event routing.

## Setup

### One-time prerequisites
1. **Python 3.11+** (or use uv: `winget install astral-sh.uv` then `uv python install 3.11`)
2. **Node 20+** + npm
3. **Rust toolchain** — `winget install Rustlang.Rustup` (puts `cargo` in `~/.cargo/bin`)
4. **MSVC Build Tools 2022** with "Desktop development with C++" workload — needed for cargo link step. Install interactively from <https://visualstudio.microsoft.com/downloads/> → Build Tools → select C++ workload. (winget silent install does not reliably trigger the workload.)
5. **WebView2** — preinstalled on Windows 11; verify with reg query under `HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}`.
6. **Ollama** — `winget install Ollama.Ollama`, then start the Ollama service (auto-starts on install).

### Project setup
1. `cd backend && uv sync` (or `pip install -e ".[dev]"`)
2. `scripts/download_models.sh` — fetches OpenWakeWord "hey_jarvis" model + Piper German voice (~62MB total).
3. `ollama pull llama3.2:1b` (~1.3GB) and `ollama pull qwen2.5-coder:1.5b` (~1GB, for the voice-triggered code-generation feature — see Known Limitations).
4. `cd frontend && npm install`
5. `scripts/dev.sh` to start backend + Tauri frontend together.

## Tests
- Backend: `cd backend && uv run pytest` (54 tests)
- Backend types: `cd backend && uv run mypy src/`
- Frontend: `cd frontend && npm test` (11 tests)
- Frontend types: `cd frontend && npx tsc --noEmit`

## Design + Plans
- Spec: `docs/superpowers/specs/2026-05-14-jarvis-hud-design.md`
- Phase 1 plan: `docs/superpowers/plans/2026-05-14-jarvis-phase1-core.md`
- Phase 1.5 plan: `docs/superpowers/plans/2026-05-14-jarvis-phase1-5-finish.md`

## Phase 1.5 Smoke Test Runbook

This is the human-driven verification step that completes Phase 1.5. All automated work is done; what remains is to actually run the system end-to-end with a microphone.

### Pre-flight checklist

- [ ] Ollama service running. Verify: `curl http://localhost:11434/api/version` returns JSON.
- [ ] `ollama list` shows `llama3.2:1b`.
- [ ] `models/openwakeword/hey_jarvis_v0.1.tflite` exists (~1.3 MB).
- [ ] TTS: default is `edge-tts` (Katja voice, needs internet, no setup) — `pip`/`uv` dependency only. Piper (`models/piper/de_DE-kerstin-low.onnx` + `.onnx.json`) is the offline fallback, set `JARVIS_TTS_BACKEND=piper` to use it.
- [ ] MSVC link.exe available on PATH (try `cargo check` from `frontend/src-tauri/`).
- [ ] Microphone connected + selected as default input device.
- [ ] Speakers/headphones connected + selected as default output.

### Run

```bash
scripts/dev.sh
```

Expected:
- Backend logs `Started server process` + model load messages (10-20s for first run).
- Tauri window opens; HUD shows "DISCONNECTED" briefly, then "ONLINE".

### Test cases

| # | Action | Expected |
|---|---|---|
| 1 | Say "Jarvis" | Ring brightens cyan; LISTENING state visible |
| 2 | Say "Jarvis, wie spät ist es?" | Transcript appears bottom-left; response streams bottom-right; audio plays in Thorsten voice |
| 3 | While Jarvis is speaking, say "Jarvis" | TTS aborts; ring flips back to cyan (barge-in works) |
| 4 | Press `Win+J` | Pipeline starts at LISTENING (skip-wake fast-path); backend log shows `inbound queued: manual_start` |
| 5 | Click anywhere on HUD | Same as Win+J |
| 6 | Press `Win+J` then stay silent | After ~800ms silence: TRANSCRIBE → STT returns empty → LLM gets empty input → "Ich verstehe dich nicht" or similar |

### Recording outcome

After running the tests, append your findings here:

```
Verified end-to-end loop on: [DATE]
Hardware: [CPU model, RAM, OS version]
Latency wake→speech-end: [X] seconds
LLM tokens/sec: [Y]
Known issues observed:
  - [list any false wake-word triggers, audio dropouts, latency spikes, etc.]
```

## Known Limitations (carry-over Phase 1)

- **MSVC compiler:** `cargo check` / `cargo build` of `frontend/src-tauri/` fails with linker errors until MSVC Build Tools 2022 (Desktop C++ workload) is installed. Plan Phase 1.5 Task 0 documents the manual install.
- **CPU-only inference:** Llama 3.2 1B + faster-whisper-base run on CPU. Expect 5-10s end-to-end latency per turn. Phase 2 may add streaming TTS for faster first-chunk audio.
- **TTS now needs internet by default:** switched to `edge-tts` (Microsoft's free neural voices) because Piper's only German female voices (eva_k, kerstin, ramona) are low/x-low quality tier — rejected as not good enough. `edge-tts` buffers the full response before playback (no true incremental streaming like Piper had), so there's a fixed per-turn delay proportional to response length. Set `JARVIS_TTS_BACKEND=piper` for the fully-offline fallback.
- **Low-VRAM GPUs crash Ollama's GPU offload:** on GPUs with <4GB VRAM (e.g. GeForce 940MX, 2GB), Ollama auto-offloads layers to GPU and the compute-buffer allocation fails (`llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers`, model load returns HTTP 500). Fix: start Ollama with `OLLAMA_LLM_LIBRARY=cpu ollama serve` to force the CPU backend.
- **No tools yet (except code generation):** Phase 1 + 1.5 ship only the voice loop. System control, web search, files, calendar, mail are Phase 2. The one exception: saying something like "programmiere mir X" triggers a dedicated small coding model (`qwen2.5-coder:1.5b`) that writes a Python script, saves it to `~/Desktop/Jarvis_Scripts/`, and runs it in its own console window — **auto-run, no confirmation, no sandbox**. This was an explicit user choice (speed over safety); see `coding.py` and `CLAUDE.md`.
- **No vision yet:** Phase 3.
- **No conversation memory:** Each turn is independent. Phase 3.
- **First wake-word run downloads tflite-runtime + ONNX runtime:** ~30-60s extra on cold start.

## Phase 1.5 Completion Status

| Component | State |
|---|---|
| Phase 1 core voice loop | ✓ merged to `main` (commit `50462bb`) |
| Phase 1.5 inbound queue routing | ✓ commit `a14ca77` |
| Phase 1.5 barge-in pipeline | ✓ commit `00930a3` (+ `6477a8f` cleanup) |
| Phase 1.5 model download script | ✓ commit `7ef67f6` |
| Phase 1.5 dev.sh pipeline-start env | ✓ commit `dcccd33` |
| Phase 1.5 Tauri Win+J hotkey | ✓ commit `a638743` (code; needs MSVC to build) |
| Phase 1.5 manual_start fast-path | ✓ commit `5f74007` |
| Phase 1.5 manual hardware smoke | ⏳ pending user (runbook above) |
