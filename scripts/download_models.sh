#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"

mkdir -p "$MODELS/openwakeword" "$MODELS/piper"

# OpenWakeWord "hey jarvis" model.
# Use ONNX variant — tflite-runtime has no Windows wheel, so OpenWakeWord
# falls back to onnxruntime on Windows. The .onnx file is the runnable
# trigger model; pre-/post-processing models (melspectrogram, embedding)
# are auto-fetched on first OpenWakeWord init via openwakeword.utils.
WAKE_URL="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx"
if [ ! -f "$MODELS/openwakeword/hey_jarvis_v0.1.onnx" ]; then
  echo "Downloading wake-word model (ONNX)..."
  curl -fL "$WAKE_URL" -o "$MODELS/openwakeword/hey_jarvis_v0.1.onnx"
else
  echo "Wake-word model already present."
fi

# OpenWakeWord ships helper models (melspectrogram, embedding) only in
# tflite format by default. Fetch the ONNX variants into the package's
# resources directory so the ONNX runtime can load them on Windows.
echo "Ensuring OpenWakeWord ONNX helper models are downloaded..."
python -c "import openwakeword.utils; openwakeword.utils.download_models()" \
  || echo "  (skip: python/openwakeword not on PATH; rerun after 'cd backend && uv sync')"

# Piper voice — de_DE-ramona-low: German female voice.
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/ramona/low"
PIPER_NAME="de_DE-ramona-low"
if [ ! -f "$MODELS/piper/${PIPER_NAME}.onnx" ]; then
  echo "Downloading Piper voice ONNX (${PIPER_NAME})..."
  curl -fL "${PIPER_BASE}/${PIPER_NAME}.onnx" -o "$MODELS/piper/${PIPER_NAME}.onnx"
  echo "Downloading Piper voice config..."
  curl -fL "${PIPER_BASE}/${PIPER_NAME}.onnx.json" -o "$MODELS/piper/${PIPER_NAME}.onnx.json"
else
  echo "Piper voice already present."
fi

echo
echo "Models in place:"
ls -lh "$MODELS/openwakeword/" "$MODELS/piper/"
echo
echo "Note: Whisper model auto-downloads on first faster-whisper run."
echo "Don't forget: ollama pull llama3.2:1b"
