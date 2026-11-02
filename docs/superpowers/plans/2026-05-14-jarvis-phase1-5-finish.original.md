# Jarvis Phase 1.5 — Finish Core Voice Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Phase 1 gaps so the voice loop actually runs end-to-end on real hardware: Rust toolchain + Tauri hotkey, inbound event routing, fixed `_orch_loop`, model downloads, manual smoke verification.

**Architecture:** Targeted fixes + completions for items left out of Phase 1 either by scope decision (Tasks 17/19/20 of old plan) or flagged by the final code review (inbound stub, orchestrator re-instantiation, barge-in pipeline, dev.sh pipeline start, frontend error event). No new features — just finishing.

**Tech Stack:** Same as Phase 1 — Python 3.11, FastAPI, asyncio, Tauri 2 / Rust, React, faster-whisper, Ollama, Piper.

**Spec reference:** `docs/superpowers/specs/2026-05-14-jarvis-hud-design.md`
**Phase 1 plan reference:** `docs/superpowers/plans/2026-05-14-jarvis-phase1-core.md`
**Phase 1 final review:** Conducted 2026-05-14 by Sonnet reviewer — verdict: ready to merge with 4 Important issues for Phase 2 (now Phase 1.5).

**Out-of-scope for Phase 1.5:** Plugin discovery + tools (Phase 2 proper plan), Vision (Phase 3), Conversation memory, Installer.

---

## File Structure (changes only)

```
jarvis/
├── backend/
│   └── src/jarvis/
│       ├── app.py                  # MODIFY: real inbound queue routing
│       └── orchestrator.py         # MODIFY: orchestrator instantiated once, concurrent wake during SPEAKING
├── frontend/
│   └── src-tauri/
│       └── src/main.rs             # MODIFY: Win+J hotkey emits to webview
├── scripts/
│   ├── download_models.sh          # CREATE: pull openwakeword jarvis.tflite + piper voice
│   └── dev.sh                      # MODIFY: factory mode so backend starts the pipeline
└── README.md                       # MODIFY: Phase 1.5 smoke notes section
```

No new modules. All work in existing files.

---

## Task 0: Rust Toolchain Setup

**Files:** None (system install)

This task installs Rust + cargo so subsequent Tauri builds work. Manual verification step.

- [ ] **Step 0.1: Install Rust via rustup**

On Windows, the recommended path is `rustup-init.exe` from rust-lang.org, but on this system `winget` is available. Try:

```bash
winget install --id Rustlang.Rustup --silent --accept-package-agreements --accept-source-agreements
```

If `winget` is not in PATH for the current shell or fails, fall back to:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
```

(Use the `bash` tool — on Windows Git Bash the shell installer works.)

- [ ] **Step 0.2: Verify rustc + cargo**

```bash
rustc --version
cargo --version
```

Expected: both print versions ≥ 1.80. If PATH is stale, source `~/.cargo/env` or open a new shell.

- [ ] **Step 0.3: Install Tauri prerequisites (Windows)**

Tauri 2 on Windows needs Microsoft Edge WebView2 (usually preinstalled on Windows 11) and the MSVC build tools. Verify WebView2:

```bash
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv
```

Expected: returns a version string (any version). If absent, install from <https://developer.microsoft.com/en-us/microsoft-edge/webview2/>.

MSVC build tools (Visual Studio Build Tools 2022 with "Desktop development with C++"):

```bash
winget install --id Microsoft.VisualStudio.2022.BuildTools --silent --accept-package-agreements --accept-source-agreements --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
```

This install is ~5-7 GB and ~5-10 minutes. Status BLOCKED if elevation is required and not granted.

- [ ] **Step 0.4: Smoke test cargo from frontend/src-tauri**

```bash
cd frontend/src-tauri
cargo check
```

Expected: cargo resolves Tauri 2 dependencies and reports any compile-stage type errors. First run takes 3-5 minutes (downloads tauri + tauri-build + serde + tokio crates). Resulting `target/` directory is gitignored.

If the version mismatches reported by `cargo check` are about Tauri itself (e.g. `tauri-build` not found), they are upstream issues — read the error and report BLOCKED.

- [ ] **Step 0.5: Commit (none required for this task; system install only)**

Nothing to commit. Proceed to Task 1.

---

## Task 1: Inbound Event Queue in app.py

**Files:**
- Modify: `backend/src/jarvis/app.py`
- Modify: `backend/tests/test_app_inbound.py` (currently absent — create)

Phase 1's `inbound_handler` was a stub that only logs. This task replaces it with a real `asyncio.Queue` that `_orch_loop` drains, routing `cancel` to `orchestrator.request_cancel()` and `manual_start` to a future "skip-wake" hook.

- [ ] **Step 1.1: Write failing test in `backend/tests/test_app_inbound.py`**

```python
"""Test that inbound WS events reach the orchestrator queue."""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from jarvis.app import create_app
from jarvis.events import CancelEvent, ManualStartEvent


def test_manual_start_pushed_to_queue() -> None:
    app = create_app(start_orchestrator=False)
    queue: asyncio.Queue = asyncio.Queue()
    app.state.inbound_queue = queue

    async def handler(ev: object) -> None:
        await app.state.inbound_queue.put(ev)
    app.state.inbound_handler = handler

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "manual_start"}))
        ws.send_text(json.dumps({"type": "cancel"}))
        ws.close()

    drained: list[object] = []
    while not queue.empty():
        drained.append(queue.get_nowait())

    assert any(isinstance(ev, ManualStartEvent) for ev in drained)
    assert any(isinstance(ev, CancelEvent) for ev in drained)


def test_garbage_does_not_pollute_queue() -> None:
    app = create_app(start_orchestrator=False)
    queue: asyncio.Queue = asyncio.Queue()
    app.state.inbound_queue = queue

    async def handler(ev: object) -> None:
        await app.state.inbound_queue.put(ev)
    app.state.inbound_handler = handler

    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json")
        # Drain the error frame from the server
        msg = ws.receive_json()
        assert msg["type"] == "error"
        ws.close()

    assert queue.empty()
```

- [ ] **Step 1.2: Run test, expect PASS (test is already aligned with current stub)**

```bash
cd backend
uv run pytest tests/test_app_inbound.py -v
```

The current `inbound_handler` stub will not push to the queue, so `test_manual_start_pushed_to_queue` will FAIL because the manually-installed `handler` is replaced by the stub during startup. Confirm the failure mode then proceed.

Actually: `start_orchestrator=False` means the `_startup` hook does NOT replace `app.state.inbound_handler`. The handler we set manually IS the one used. So this test should PASS as-is against current code, validating that the WS layer correctly invokes the handler. Run and confirm.

```
Expected: both tests PASS (current WS plumbing already routes to whatever handler is set on app.state).
```

If they pass, this test now locks in the contract. Continue.

- [ ] **Step 1.3: Replace `_orch_loop` and `inbound_handler` in `backend/src/jarvis/app.py`**

Find the `@app.on_event("startup")` block. Replace the `def emit(ev) ...` definition and everything below it inside the function (up to and including `app.state.inbound_handler = inbound_handler`) with:

```python
        def emit(ev: Event) -> None:
            asyncio.create_task(hub.broadcast(ev.model_dump(mode="json")))

        app.state.inbound_queue = asyncio.Queue()

        async def inbound_handler(ev: ManualStartEvent | CancelEvent | ConfirmToolEvent) -> None:
            await app.state.inbound_queue.put(ev)
            logger.info(f"inbound queued: {ev.type}")

        app.state.inbound_handler = inbound_handler

        async def _orch_loop() -> None:
            speaker = RealSpeaker()
            wake = RealWakeWord(
                model_path=settings.wake_model_path,
                threshold=settings.wake_threshold,
            )
            vad = RealVad(threshold=settings.vad_threshold, sample_rate=settings.sample_rate)
            stt = RealStt(
                model_name=settings.whisper_model,
                compute_type=settings.whisper_compute,
                language=settings.whisper_lang,
            )
            llm = RealLlm(
                host=settings.llm_host,
                model=settings.llm_model,
                system_prompt=settings.llm_system_prompt,
            )
            tts = RealTts(model_path=settings.tts_model_path)
            silence_frame_count = int(
                settings.silence_ms * settings.sample_rate / 1000 / 480
            )

            async with RealMicStream(sample_rate=settings.sample_rate) as mic:
                orch = Orchestrator(
                    mic=mic, speaker=speaker, wake=wake, vad=vad,
                    stt=stt, llm=llm, tts=tts, emit=emit,
                    silence_frames=silence_frame_count,
                    sample_rate=settings.sample_rate,
                )

                async def drain_inbound() -> None:
                    while True:
                        ev = await app.state.inbound_queue.get()
                        if isinstance(ev, CancelEvent):
                            orch.request_cancel()
                        # ManualStartEvent: orchestrator does not have a skip-wake
                        # fast path yet (Phase 2). Logged via inbound_handler above.

                drain = asyncio.create_task(drain_inbound())
                try:
                    while True:
                        try:
                            await orch.run_once()
                        except Exception as exc:
                            logger.exception("orchestrator crashed")
                            await hub.broadcast(
                                ErrorEvent(stage="orchestrator", msg=str(exc), retriable=True)
                                .model_dump(mode="json")
                            )
                            await asyncio.sleep(1.0)
                finally:
                    drain.cancel()

        app.state.orch_task = asyncio.create_task(_orch_loop())
```

Note three intentional changes vs. the original `_orch_loop`:
1. `Orchestrator` is constructed **once** outside the `while True` loop (final-review Important #2).
2. A separate `drain_inbound` task routes `CancelEvent` → `orch.request_cancel()`.
3. `inbound_queue` + `inbound_handler` are created **before** the orchestrator task starts, so the WS endpoint sees them as soon as startup runs.

Also add `Event` and `ConfirmToolEvent` to the existing import block:

```python
from jarvis.events import CancelEvent, ConfirmToolEvent, ErrorEvent, Event, ManualStartEvent, parse_inbound
```

- [ ] **Step 1.4: Run full backend test suite**

```bash
cd backend
uv run pytest -v
```

Expected: 52 passed (50 from Phase 1 + 2 new in `test_app_inbound.py`). No regressions.

- [ ] **Step 1.5: mypy clean**

```bash
uv run mypy src/
```

Expected: `Success: no issues found in 14 source files`.

- [ ] **Step 1.6: Commit**

```bash
git add backend/src/jarvis/app.py backend/tests/test_app_inbound.py
git commit -m "feat(app): real inbound queue routing + single-instance orchestrator"
```

---

## Task 2: Barge-In — Concurrent Wake Detection During SPEAKING

**Files:**
- Modify: `backend/src/jarvis/orchestrator.py`
- Modify: `backend/tests/test_orchestrator.py`

State machine has `(SPEAKING, WAKE_DETECTED) → LISTENING`. But Phase 1 orchestrator's SPEAKING phase is a linear `async for chunk in tts.synthesize(...)` — it blocks until TTS finishes. This task makes wake detection run concurrently with TTS playback so the user can interrupt.

Design: the SPEAKING phase becomes two concurrent tasks. Task A plays TTS chunks. Task B reads mic frames and runs wake.detect. First one to "finish" wins:
- A completes naturally → PLAYBACK_DONE → IDLE.
- B detects wake → cancel A, transition to LISTENING, return.

Mic stream is shared between IDLE (pre-wake) and SPEAKING (barge-in) phases. Since `mic` is injected at `__init__`, both phases use the same iterator. The iterator yields the same way; only the consumer changes.

- [ ] **Step 2.1: Write failing test in `backend/tests/test_orchestrator.py`**

Append after the existing tests:

```python
async def test_barge_in_during_speaking_returns_to_listening() -> None:
    """During TTS playback, a wake-word event aborts playback and returns to LISTENING."""
    silent = np.zeros(480, dtype=np.int16)
    marker = 7777
    # PCM layout:
    #   3 silent + marker → wake fires at frame 4
    #   then 3 silent (will be consumed during LISTENING → speech_end=3)
    #   then marker → barge-in trigger during SPEAKING phase
    #   then 500 silent so the loop survives
    marker_frame = np.full(480, marker, dtype=np.int16)
    pcm = np.concatenate(
        [silent] * 3 + [marker_frame] + [silent] * 3 + [marker_frame] + [silent] * 500
    )

    # speech_mask: 3 silent + 1 silent (marker frame) + 3 silent post-wake; rest unused
    speech_mask = [False] * 100

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
    )

    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    # State arc: LISTENING → TRANSCRIBE → THINKING → SPEAKING → LISTENING (barge-in)
    # ... then run_once exits cleanly. We don't assert the final state because
    # the cycle ends at whatever state the barge-in lands in.
    assert State.SPEAKING in state_seq
    assert state_seq.count(State.LISTENING) >= 2  # initial + barge-in
```

- [ ] **Step 2.2: Run test, expect FAIL**

```bash
cd backend
uv run pytest tests/test_orchestrator.py::test_barge_in_during_speaking_returns_to_listening -v
```

Expected: FAIL — current SPEAKING phase ignores mic frames.

- [ ] **Step 2.3: Update `run_once` SPEAKING phase in `backend/src/jarvis/orchestrator.py`**

Find the existing SPEAKING block:

```python
            # 4. SPEAKING
            self._go(Trigger.RESPONSE_COMPLETE)
            async for chunk in self._tts.synthesize(full_response):
                if self._cancel:
                    self._go(Trigger.CANCEL)
                    return
                await self._speaker.play(chunk)

            # 5. IDLE
            self._go(Trigger.PLAYBACK_DONE)
```

Replace with:

```python
            # 4. SPEAKING — TTS playback concurrent with wake detection (barge-in)
            self._go(Trigger.RESPONSE_COMPLETE)
            barge_in = asyncio.Event()

            async def play_tts() -> None:
                async for chunk in self._tts.synthesize(full_response):
                    if self._cancel or barge_in.is_set():
                        return
                    await self._speaker.play(chunk)

            async def watch_for_barge_in() -> None:
                async for frame in self._mic.frames():
                    if self._cancel:
                        return
                    if await self._wake.detect(frame):
                        barge_in.set()
                        return

            play_task = asyncio.create_task(play_tts())
            watch_task = asyncio.create_task(watch_for_barge_in())
            done, pending = await asyncio.wait(
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

            if barge_in.is_set():
                # Wake fired mid-playback → go back to LISTENING.
                # State stays at SPEAKING here; transition() supports SPEAKING→LISTENING via WAKE_DETECTED.
                self._go(Trigger.WAKE_DETECTED)
                return

            # 5. IDLE
            self._go(Trigger.PLAYBACK_DONE)
```

Note: after barge-in we `return` from `run_once`. The outer `_orch_loop` will call `run_once` again, starting a fresh IDLE → LISTENING cycle. The mic iterator's position carries over because it's the same shared `RealMicStream` instance.

- [ ] **Step 2.4: Run test, expect PASS**

```bash
uv run pytest tests/test_orchestrator.py -v
```

Expected: all orchestrator tests PASS (3: full_loop, cancel_during_listening, barge_in_during_speaking).

- [ ] **Step 2.5: Full backend suite + mypy**

```bash
uv run pytest
uv run mypy src/
```

Expected: 53 passed (52 + 1 new test). mypy clean.

- [ ] **Step 2.6: Commit**

```bash
git add backend/src/jarvis/orchestrator.py backend/tests/test_orchestrator.py
git commit -m "feat(orchestrator): barge-in support during SPEAKING phase"
```

---

## Task 3: Model Download Script

**Files:**
- Create: `scripts/download_models.sh`

- [ ] **Step 3.1: Create `scripts/download_models.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"

mkdir -p "$MODELS/openwakeword" "$MODELS/piper"

# OpenWakeWord "jarvis" model
WAKE_URL="https://github.com/dscripka/openWakeWord/raw/main/openwakeword/resources/models/jarvis.tflite"
if [ ! -f "$MODELS/openwakeword/jarvis.tflite" ]; then
  echo "Downloading wake-word model..."
  curl -fL "$WAKE_URL" -o "$MODELS/openwakeword/jarvis.tflite"
else
  echo "Wake-word model already present."
fi

# Piper voice (German, Thorsten)
PIPER_ONNX="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
PIPER_JSON="${PIPER_ONNX}.json"
if [ ! -f "$MODELS/piper/de_DE-thorsten-medium.onnx" ]; then
  echo "Downloading Piper voice ONNX..."
  curl -fL "$PIPER_ONNX" -o "$MODELS/piper/de_DE-thorsten-medium.onnx"
  echo "Downloading Piper voice config..."
  curl -fL "$PIPER_JSON" -o "$MODELS/piper/de_DE-thorsten-medium.onnx.json"
else
  echo "Piper voice already present."
fi

echo
echo "Models in place:"
ls -lh "$MODELS/openwakeword/" "$MODELS/piper/"
echo
echo "Note: Whisper model auto-downloads on first faster-whisper run."
echo "Don't forget: ollama pull llama3.2:3b"
```

- [ ] **Step 3.2: Make executable**

```bash
chmod +x scripts/download_models.sh
```

(On Windows + Git Bash this sets the git executable bit but no NTFS effect — still required for the commit.)

- [ ] **Step 3.3: Run script**

```bash
scripts/download_models.sh
```

Expected output (approximate):
```
Downloading wake-word model...
[curl progress] ~100KB
Downloading Piper voice ONNX...
[curl progress] ~62MB
Downloading Piper voice config...
[curl progress] ~5KB

Models in place:
models/openwakeword/jarvis.tflite       100K
models/piper/de_DE-thorsten-medium.onnx 62M
models/piper/de_DE-thorsten-medium.onnx.json 5.0K
```

If curl reports 404 for the wake-word URL (the openwakeword repo path may shift over time), report BLOCKED with the URL and let the controller decide on an alternate model. The model file itself is gitignored regardless.

- [ ] **Step 3.4: Pull Ollama model**

```bash
ollama pull llama3.2:3b
```

Expected: ~2GB download. If ollama is not installed: install via `winget install --id Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements` first, then start the service (`ollama serve` in a separate terminal, or it should auto-start as a Windows service after install). Pulling without the service running fails with "could not connect to ollama".

- [ ] **Step 3.5: Verify ollama health**

```bash
ollama list
```

Expected: shows `llama3.2:3b` with size ~2GB.

- [ ] **Step 3.6: Commit (script only, models gitignored)**

```bash
git add scripts/download_models.sh
git commit -m "chore: model download script for wake-word + piper voice"
```

---

## Task 4: dev.sh Starts Pipeline

**Files:**
- Modify: `scripts/dev.sh`
- Modify: `backend/src/jarvis/app.py` (add env-driven start_orchestrator)

Current `app.py` exports module-level `app = create_app(start_orchestrator=False)` so uvicorn import doesn't trigger model loads. Result: `uvicorn jarvis.app:app` runs WS server but no pipeline. `dev.sh` uses this exact pattern. We fix it via env var.

- [ ] **Step 4.1: Modify the bottom of `backend/src/jarvis/app.py`**

Find the final line:

```python
app = create_app(start_orchestrator=False)
```

Replace with:

```python
import os as _os

# Controlled by JARVIS_START_ORCHESTRATOR env var. Default OFF so test imports
# don't trigger model loads; scripts/dev.sh sets it to 1.
app = create_app(start_orchestrator=_os.getenv("JARVIS_START_ORCHESTRATOR", "0") == "1")
```

- [ ] **Step 4.2: Modify `scripts/dev.sh`**

Replace entire file:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

command -v uvicorn >/dev/null 2>&1 || { echo "uvicorn missing — run: cd backend && uv sync" >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm missing — install Node 20+" >&2; exit 1; }
command -v ollama >/dev/null 2>&1 || echo "warning: ollama not on PATH — voice loop will error out on LLM step" >&2

(cd "$ROOT/backend" && JARVIS_START_ORCHESTRATOR=1 uv run uvicorn jarvis.app:app --port 8765) &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run tauri dev) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
```

Changes vs Phase 1:
- `command -v` preflight checks for uvicorn / npm / ollama.
- Backend started via `uv run uvicorn` so the venv is active.
- `JARVIS_START_ORCHESTRATOR=1` flips the pipeline on.

- [ ] **Step 4.3: Verify backend still test-clean (start_orchestrator default OFF)**

```bash
cd backend
uv run pytest -v
uv run mypy src/
```

Expected: still 53 passed (no tests set the env var). mypy clean.

- [ ] **Step 4.4: Smoke check the dev launcher path (without actually running tauri)**

```bash
JARVIS_START_ORCHESTRATOR=1 timeout 5 uv run uvicorn jarvis.app:app --port 8765 || true
```

On Git Bash on Windows `timeout` is GNU coreutils style: if not available, run without timeout and Ctrl-C after a few seconds. We just want to confirm the process starts loading models and doesn't crash on import.

Expected: logs show:
- `INFO:     Started server process`
- Eventually loguru messages about wake/whisper/piper model loading
- May error on `ollama` if not running — that's expected for this smoke

If it errors on `from openwakeword.model import Model`, the model file path is wrong — verify `models/openwakeword/jarvis.tflite` exists from Task 3.

- [ ] **Step 4.5: Commit**

```bash
git add backend/src/jarvis/app.py scripts/dev.sh
git commit -m "feat(dev): env-controlled orchestrator start + dev.sh preflight checks"
```

---

## Task 5: Tauri Hotkey Integration (Win+J)

**Files:**
- Modify: `frontend/src-tauri/src/main.rs`
- Modify: `frontend/src-tauri/Cargo.toml` (verify deps, may already be correct from Phase 1 Task 14)
- Modify: `frontend/src/App.tsx`

Currently `main.rs` is the stub from Task 14 — it just runs Tauri with the global-shortcut plugin loaded but no shortcut registered. This task registers Win+J and emits a `hotkey` event to the webview. The webview frontend already imported `@tauri-apps/api` in Task 14's deps but hasn't subscribed. We add the listener.

- [ ] **Step 5.1: Replace `frontend/src-tauri/src/main.rs`**

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Emitter, Manager};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

fn main() {
    let shortcut = Shortcut::new(Some(Modifiers::SUPER), Code::KeyJ);

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, sc, event| {
                    if sc == &shortcut && event.state() == ShortcutState::Pressed {
                        if let Some(win) = app.get_webview_window("main") {
                            let _ = win.emit("hotkey", "manual_start");
                        }
                    }
                })
                .build(),
        )
        .setup(move |app| {
            app.global_shortcut().register(shortcut)?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

The window label `"main"` is the default from `tauri.conf.json` (no explicit label set → Tauri uses `"main"`).

- [ ] **Step 5.2: Verify `Cargo.toml` deps match (no changes expected)**

Open `frontend/src-tauri/Cargo.toml` and confirm the `[dependencies]` block contains:

```toml
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tauri = { version = "2", features = [] }
tauri-plugin-global-shortcut = "2"
```

If `serde` or `serde_json` are missing (Phase 1 included them, but verify), add them.

- [ ] **Step 5.3: cargo check from src-tauri**

```bash
cd frontend/src-tauri
cargo check
```

Expected: cargo compiles the project. Any rust-side errors point to API drift in `tauri-plugin-global-shortcut` (the 2.x API has had small breaking changes between minor versions) — read the error and adjust the import names accordingly. If types like `Modifiers::SUPER` are unresolved, check the installed version's docs.

- [ ] **Step 5.4: Update `frontend/src/App.tsx`**

Replace the current Phase 1 App.tsx with:

```tsx
import { useEffect, useRef } from "react";
import { HudShell } from "./hud/HudShell";
import { useJarvisStore } from "./store";
import { JarvisWs } from "./ws";

export function App() {
  const wsRef = useRef<JarvisWs | null>(null);

  useEffect(() => {
    const ws = new JarvisWs({
      url: "ws://127.0.0.1:8765/ws",
      onEvent: (ev) => useJarvisStore.getState().handleEvent(ev),
      onConnected: () => useJarvisStore.getState().setConnected(true),
      onDisconnected: () => useJarvisStore.getState().setConnected(false),
    });
    wsRef.current = ws;
    ws.connect();

    let unlisten: (() => void) | null = null;
    (async () => {
      try {
        const tauriEvent = await import("@tauri-apps/api/event");
        unlisten = await tauriEvent.listen("hotkey", () => {
          ws.send({ type: "manual_start" });
        });
      } catch {
        // Plain browser (vitest, vite dev without Tauri) — silently skip.
      }
    })();

    return () => {
      ws.close();
      unlisten?.();
    };
  }, []);

  const onClickCore = () => wsRef.current?.send({ type: "manual_start" });

  return (
    <div onClick={onClickCore} style={{ cursor: "pointer", height: "100vh" }}>
      <HudShell />
    </div>
  );
}
```

Changes vs Phase 1's App.tsx:
- `useRef` to hold the WS instance for click handler access.
- Tauri event listener subscribes to `"hotkey"` events emitted from `main.rs`.
- Click anywhere on the HUD shell sends `manual_start`.

- [ ] **Step 5.5: Frontend type check + tests**

```bash
cd frontend
npx tsc --noEmit
npm test
```

Expected: tsc clean, 11/11 tests still pass (no new tests in this task; the hotkey path is tested via integration only).

- [ ] **Step 5.6: Commit**

```bash
git add frontend/src-tauri/src/main.rs frontend/src/App.tsx
git commit -m "feat(frontend): Win+J global hotkey + HUD-click both send manual_start"
```

---

## Task 6: ManualStartEvent Skip-Wake Fast Path

**Files:**
- Modify: `backend/src/jarvis/orchestrator.py`
- Modify: `backend/tests/test_orchestrator.py`

Phase 1 routed `ManualStartEvent` only to a logger. With Task 1 it now lands in the queue but is ignored. This task wires it: when a `ManualStartEvent` is consumed before `run_once` enters the mic loop, the orchestrator skips wake detection and goes straight to LISTENING.

Implementation: add a `manual_start_pending` flag to Orchestrator. `_orch_loop` calls `orch.set_manual_start()` when `ManualStartEvent` is drained. `run_once` checks the flag before wake-detection.

- [ ] **Step 6.1: Write failing test in `backend/tests/test_orchestrator.py`**

Append:

```python
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
    )

    orch.set_manual_start()
    await asyncio.wait_for(orch.run_once(), timeout=5.0)

    state_seq = [e.value for e in events if isinstance(e, StateEvent)]
    assert State.LISTENING in state_seq
    assert State.TRANSCRIBE in state_seq
    transcripts = [e for e in events if isinstance(e, TranscriptFinalEvent)]
    assert transcripts and transcripts[0].text == "manual hello"
```

- [ ] **Step 6.2: Run test, expect FAIL**

```bash
cd backend
uv run pytest tests/test_orchestrator.py::test_manual_start_skips_wake -v
```

Expected: FAIL — `set_manual_start` not defined.

- [ ] **Step 6.3: Update `Orchestrator` in `backend/src/jarvis/orchestrator.py`**

Add a flag and setter. Find `__init__`:

```python
        self._state = State.IDLE
        self._cancel = False
```

Replace with:

```python
        self._state = State.IDLE
        self._cancel = False
        self._manual_start = False
```

Add a new method right after `request_cancel`:

```python
    def set_manual_start(self) -> None:
        """Skip wake-word detection on the next run_once cycle."""
        self._manual_start = True
```

In `run_once`, find the IDLE pre-wake mic loop. Currently:

```python
            async for frame in self._mic.frames():
                if self._cancel:
                    self._go(Trigger.CANCEL)
                    return

                self._emit(AudioLevelEvent(rms=rms_dbfs(frame), fft=[]))

                if not wake_seen:
                    if await self._wake.detect(frame):
                        wake_seen = True
                        self._go(Trigger.WAKE_DETECTED)
                    continue
```

Replace with:

```python
            # Honor manual-start request: skip wake-word and go straight to LISTENING.
            if self._manual_start:
                self._manual_start = False
                wake_seen = True
                self._go(Trigger.MANUAL_START)

            async for frame in self._mic.frames():
                if self._cancel:
                    self._go(Trigger.CANCEL)
                    return

                self._emit(AudioLevelEvent(rms=rms_dbfs(frame), fft=[]))

                if not wake_seen:
                    if await self._wake.detect(frame):
                        wake_seen = True
                        self._go(Trigger.WAKE_DETECTED)
                    continue
```

- [ ] **Step 6.4: Wire `_orch_loop` in `backend/src/jarvis/app.py`**

Find the `drain_inbound` coroutine inside `_orch_loop` (added in Task 1). Replace with:

```python
                async def drain_inbound() -> None:
                    while True:
                        ev = await app.state.inbound_queue.get()
                        if isinstance(ev, CancelEvent):
                            orch.request_cancel()
                        elif isinstance(ev, ManualStartEvent):
                            orch.set_manual_start()
```

- [ ] **Step 6.5: Run test, expect PASS**

```bash
uv run pytest tests/test_orchestrator.py -v
```

Expected: all 4 orchestrator tests PASS. Run full suite:

```bash
uv run pytest
uv run mypy src/
```

Expected: 54 passed, mypy clean.

- [ ] **Step 6.6: Commit**

```bash
git add backend/src/jarvis/orchestrator.py backend/src/jarvis/app.py backend/tests/test_orchestrator.py
git commit -m "feat(orchestrator): manual_start fast-path skips wake detection"
```

---

## Task 7: Manual Smoke Test Runbook

**Files:**
- Modify: `README.md`

This is the gated verification step that Phase 1 deferred. It's a runbook executed by the human, not a unit test.

- [ ] **Step 7.1: Run the dev launcher**

In a terminal:

```bash
cd /c/Users/Erwin/Desktop/projekte/jarvis
scripts/dev.sh
```

Expected:
- Backend logs: `INFO: Started server process`, model load messages over ~10-20s.
- Frontend: Tauri window opens with HUD; "DISCONNECTED" briefly then "ONLINE".

- [ ] **Step 7.2: Speak the wake word**

Say clearly: **"Jarvis"**.

Expected:
- Backend log shows wake-word fired.
- HUD ring color shifts from idle cyan to bright cyan (LISTENING).
- VAD detects silence → STT runs → THINKING (purple) → SPEAKING (amber).

- [ ] **Step 7.3: Test full conversational loop**

Say: **"Jarvis, wie spät ist es?"**.

Expected:
- Transcript appears in bottom-left corner of HUD.
- LLM response streams to bottom-right corner.
- Audio plays through speakers in Thorsten voice.

Record: end-to-end latency in seconds.

- [ ] **Step 7.4: Test barge-in**

While Jarvis is speaking, say **"Jarvis"** again.

Expected: TTS cuts off, ring flips back to cyan (LISTENING). New utterance follows.

- [ ] **Step 7.5: Test Win+J hotkey**

Press `Win+J` (Super + J).

Expected: Equivalent to a wake-word — pipeline starts at LISTENING. Backend log shows `inbound queued: manual_start`.

- [ ] **Step 7.6: Test HUD click**

Click anywhere on the HUD.

Expected: Same as hotkey — pipeline starts at LISTENING.

- [ ] **Step 7.7: Document outcomes in `README.md`**

Append a new section to `README.md`:

```markdown
## Phase 1.5 Smoke Notes

Verified end-to-end loop on: [DATE]
Hardware: [SPEC: CPU model, RAM, OS version]
Latency (wake → speech end): [Xs]
LLM model: llama3.2:3b
TTS voice: de_DE-thorsten-medium

Known issues observed:
- [list any glitches, false wake-word triggers, audio dropouts, etc.]

Resolved during Phase 1.5:
- inbound_handler stub → real queue routing
- barge-in supported during SPEAKING
- Win+J hotkey + HUD click trigger manual_start
- model download script + Ollama setup
```

Fill `[DATE]`, `[SPEC]`, `[Xs]` with real values from the smoke run. If any of the steps 7.1-7.6 fail, document the failure mode under "Known issues observed" instead of marking the task DONE — that becomes the entry point for Phase 2's first bug-fix task.

- [ ] **Step 7.8: Commit + tag**

```bash
git add README.md
git commit -m "docs: phase 1.5 smoke notes"
git tag -a phase-1.5 -m "Phase 1.5 — core voice loop verified end-to-end"
```

---

## Self-Review Checklist

| Phase 1 final-review item | Addressed by |
|---|---|
| Important #1: inbound_handler stub | Task 1 + Task 6 |
| Important #2: orchestrator re-instantiation per cycle | Task 1 (single instance) |
| Important #3: frontend error event (was fixed at merge) | Already in `main` |
| Important #4: `lru_cache` settings risk | Deferred — not blocking |
| Minor #5: Waveform animation | Deferred to Phase 2 polish |
| Minor #6: CSP null | Deferred to Phase 3 (installer) |
| Minor #7: dev.sh pipeline disable | Task 4 |
| Minor #8: SpeechEndDetector unused signal | Documented, kept for telemetry |
| Phase 1 Task 17 (Tauri hotkey) | Task 5 |
| Phase 1 Task 18 (inbound queue) | Task 1 + Task 6 |
| Phase 1 Task 19 (model download) | Task 3 |
| Phase 1 Task 20 (manual smoke) | Task 7 |
| Phase 1 spec: barge-in | Task 2 |
| Phase 1 spec: HUD mic click | Task 5 (App.tsx onClick) |

All Phase 1 exit criteria from spec §7 now reachable on real hardware.

**Placeholder scan:** no TBD / TODO / "implement later". All code blocks contain full content. All file paths are exact.

**Type consistency:** `Orchestrator.set_manual_start()` is referenced consistently across Tasks 6 (definition), 6 (test), and 6 (app.py drain_inbound). `_manual_start` flag is initialized in `__init__` (Task 6 Step 6.3) and consumed in `run_once` body (Task 6 Step 6.3). `ManualStartEvent` import added to `app.py` in Task 1.

---

## Definition of Done

- Task 7 smoke run produced a real entry in README "Phase 1.5 Smoke Notes" with measured latency.
- `git tag phase-1.5` exists.
- All Phase 1 deliverables from spec §7 verified working on real hardware.
- Branch is `phase-1.5-finish` (created by executor); merge approach decided by user.
