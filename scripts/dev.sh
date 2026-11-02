#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve uv: PATH first, then known install locations as fallbacks
# (winget-installed uv requires a new shell to land on PATH).
UV_BIN="$(command -v uv 2>/dev/null || true)"
if [ -z "$UV_BIN" ]; then
  # Try ~/.local/bin (rustup-style installer), then winget packages dir
  for candidate in \
    "$HOME/.local/bin/uv.exe" \
    "$HOME/.local/bin/uv" \
    "$HOME"/AppData/Local/Microsoft/WinGet/Packages/astral-sh.uv_*/uv.exe; do
    if [ -x "$candidate" ]; then
      UV_BIN="$candidate"
      break
    fi
  done
fi
if [ -z "$UV_BIN" ]; then
  echo "uv missing — install: winget install astral-sh.uv" >&2
  exit 1
fi
(cd "$ROOT/backend" && "$UV_BIN" run --quiet python -c "import uvicorn" 2>/dev/null) \
  || { echo "uvicorn not installed in backend venv — run: cd backend && $UV_BIN sync" >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm missing — install Node 20+" >&2; exit 1; }

# Ollama presence on PATH is necessary but not sufficient. Voice loop hits
# "All connection attempts failed" if the service is not actually listening.
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
if ! curl -sf "${OLLAMA_HOST}/api/version" >/dev/null 2>&1; then
  echo "warning: Ollama service not reachable at ${OLLAMA_HOST}." >&2
  echo "         Start it in another terminal: 'ollama serve'  (or launch the tray app)." >&2
  echo "         On a low-VRAM GPU (<4GB), GPU offload can crash model load with" >&2
  echo "         'failed to allocate compute pp buffers' — start with:" >&2
  echo "         OLLAMA_LLM_LIBRARY=cpu ollama serve" >&2
  echo "         Backend will still start; LLM step will error until the service is up." >&2
fi

(cd "$ROOT/backend" && JARVIS_START_ORCHESTRATOR=1 JARVIS_USE_CLOUD="${JARVIS_USE_CLOUD:-0}" "$UV_BIN" run uvicorn jarvis.app:app --port 8765) &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run tauri dev) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
