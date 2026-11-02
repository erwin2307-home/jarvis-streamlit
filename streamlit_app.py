"""Thin Streamlit chat+voice frontend for Jarvis.

Deployed to Streamlit Community Cloud. Talks to the real LLM (and STT/TTS)
running on your own machine (jarvis/backend/src/jarvis/remote_api.py) over
a public tunnel URL (ngrok/cloudflared) you paste into the sidebar — this
app itself never runs any model.

Voice turns (recording -> /voice -> transcript+reply+TTS audio) run
entirely inside one self-contained JS widget (see VOICE_WIDGET_HTML
below), talking to the tunnel URL directly from the browser. This is a
deliberate split from the typed-text path (st.chat_input -> /chat),
which goes through Streamlit's own Python server as usual: a plain
components.html iframe has no channel back into Streamlit's session
state, so the voice widget keeps its own small transcript rather than
pretending to merge into the Python-side chat history below it.
"""
from __future__ import annotations

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Jarvis", page_icon="🤖")


def _default(key: str, fallback: str = "") -> str:
    try:
        return str(st.secrets[key])
    except Exception:
        return fallback


VOICE_WIDGET_HTML = r"""
<div id="jarvis-root" tabindex="0">
  <style>
    html, body { margin: 0; background: #05070c; }
    #jarvis-root {
      display: flex; flex-direction: column; align-items: center;
      gap: 14px; padding: 18px 0 10px; outline: none;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #ring-wrap {
      width: 180px; height: 180px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      position: relative;
    }
    #ring-glow {
      position: absolute; inset: 0; border-radius: 50%;
      background: radial-gradient(circle at 50% 50%, #2b8cff33 0%, transparent 72%);
      transition: background 0.3s ease;
      animation: glow-pulse 3.2s ease-in-out infinite;
    }
    #ring-core {
      width: 118px; height: 118px; border-radius: 50%;
      border: 5px solid #2b8cff;
      border-top-color: rgba(43,140,255,0.15);
      box-shadow: 0 0 28px #2b8cffaa, 0 0 60px #2b8cff44, inset 0 0 20px #2b8cff66;
      transition: border-color 0.25s ease, box-shadow 0.25s ease;
      animation: core-pulse 3.2s ease-in-out infinite;
    }
    #jarvis-root.listening #ring-core {
      border-color: #5fd4ff;
      box-shadow: 0 0 44px #5fd4ffcc, 0 0 90px #5fd4ff66, inset 0 0 26px #5fd4ff99;
      animation: core-pulse-fast 0.7s ease-in-out infinite;
    }
    #jarvis-root.listening #ring-glow {
      background: radial-gradient(circle at 50% 50%, #5fd4ff55 0%, transparent 72%);
      animation: glow-pulse-fast 0.7s ease-in-out infinite;
    }
    #jarvis-root.thinking #ring-core {
      border-color: #ffb020;
      box-shadow: 0 0 36px #ffb020aa, 0 0 70px #ffb02044, inset 0 0 20px #ffb02066;
      animation: core-spin 1s linear infinite;
    }
    #jarvis-root.speaking #ring-core {
      border-color: #2bffa0;
      box-shadow: 0 0 40px #2bffa0cc, 0 0 80px #2bffa066, inset 0 0 22px #2bffa099;
      animation: core-pulse-fast 0.5s ease-in-out infinite;
    }
    @keyframes glow-pulse { 0%,100% { opacity: 0.7; } 50% { opacity: 1; } }
    @keyframes glow-pulse-fast { 0%,100% { opacity: 0.8; } 50% { opacity: 1; } }
    @keyframes core-pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
    @keyframes core-pulse-fast { 0%,100% { transform: scale(1); } 50% { transform: scale(1.12); } }
    @keyframes core-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    #status-line { color: #9fb3d9; font-size: 13px; letter-spacing: 0.02em; min-height: 18px; }
    #hint-line { color: #51607f; font-size: 11px; }
    #transcript {
      width: 100%; max-width: 480px; max-height: 160px; overflow-y: auto;
      font-size: 13px; color: #cdd9f5; padding: 0 8px;
    }
    #transcript .you { color: #5fd4ff; }
    #transcript .jarvis { color: #9fe8c0; }
    #transcript div { margin: 3px 0; }
  </style>
  <div id="ring-wrap">
    <div id="ring-glow"></div>
    <div id="ring-core"></div>
  </div>
  <div id="status-line">Ring anklicken, dann Leertaste halten zum Sprechen</div>
  <div id="hint-line"></div>
  <div id="transcript"></div>

  <script>
  (function () {
    const API_URL = "__API_URL__";
    const API_KEY = "__API_KEY__";
    const root = document.getElementById("jarvis-root");
    const statusLine = document.getElementById("status-line");
    const transcriptEl = document.getElementById("transcript");

    let mediaStream = null;
    let recorder = null;
    let chunks = [];
    let spaceDown = false;

    function setState(state) {
      root.classList.remove("listening", "thinking", "speaking");
      if (state) root.classList.add(state);
      const labels = {
        idle: "Leertaste halten zum Sprechen",
        listening: "Höre zu…",
        thinking: "Denke…",
        speaking: "Spreche…",
      };
      statusLine.textContent = labels[state] || labels.idle;
    }

    function addLine(cls, text) {
      const div = document.createElement("div");
      div.className = cls;
      div.textContent = (cls === "you" ? "Du: " : "Jarvis: ") + text;
      transcriptEl.appendChild(div);
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }

    function floatTo16BitPCM(view, offset, input) {
      for (let i = 0; i < input.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, input[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
    }
    function writeString(view, offset, str) {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    }
    function audioBufferToWav(buffer) {
      const data = buffer.numberOfChannels > 1 ? mixDown(buffer) : buffer.getChannelData(0);
      const sampleRate = buffer.sampleRate;
      const dataSize = data.length * 2;
      const arr = new ArrayBuffer(44 + dataSize);
      const view = new DataView(arr);
      writeString(view, 0, "RIFF");
      view.setUint32(4, 36 + dataSize, true);
      writeString(view, 8, "WAVE");
      writeString(view, 12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(view, 36, "data");
      view.setUint32(40, dataSize, true);
      floatTo16BitPCM(view, 44, data);
      return new Blob([view], { type: "audio/wav" });
    }
    function mixDown(buffer) {
      const ch0 = buffer.getChannelData(0);
      const out = new Float32Array(ch0.length);
      for (let c = 0; c < buffer.numberOfChannels; c++) {
        const d = buffer.getChannelData(c);
        for (let i = 0; i < d.length; i++) out[i] += d[i] / buffer.numberOfChannels;
      }
      return out;
    }

    async function ensureStream() {
      if (!mediaStream) {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      return mediaStream;
    }

    async function startRecording() {
      if (!API_URL || !API_KEY) {
        statusLine.textContent = "API-URL/API-Key fehlen (Sidebar ausfüllen)";
        return;
      }
      try {
        await ensureStream();
      } catch (e) {
        statusLine.textContent = "Mikrofon-Zugriff verweigert: " + e;
        return;
      }
      chunks = [];
      recorder = new MediaRecorder(mediaStream);
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.start();
      setState("listening");
    }

    function stopRecordingAndSend() {
      if (!recorder || recorder.state === "inactive") return;
      recorder.onstop = async () => {
        setState("thinking");
        try {
          const blob = new Blob(chunks, { type: chunks[0] ? chunks[0].type : "audio/webm" });
          const arrayBuf = await blob.arrayBuffer();
          const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const decoded = await audioCtx.decodeAudioData(arrayBuf);
          const wavBlob = audioBufferToWav(decoded);

          const form = new FormData();
          form.append("audio", wavBlob, "rec.wav");

          const resp = await fetch(API_URL.replace(/\/$/, "") + "/voice", {
            method: "POST",
            headers: { "X-API-Key": API_KEY },
            body: form,
          });

          if (!resp.ok) {
            const detail = await resp.text();
            statusLine.textContent = "Fehler (" + resp.status + "): " + detail.slice(0, 120);
            setState("idle");
            return;
          }
          const body = await resp.json();
          addLine("you", body.transcript);
          addLine("jarvis", body.reply);
          setState("speaking");

          if (body.audio_base64) {
            const audio = new Audio("data:audio/wav;base64," + body.audio_base64);
            audio.onended = () => setState("idle");
            audio.onerror = () => setState("idle");
            audio.play().catch(() => setState("idle"));
          } else {
            setState("idle");
          }
        } catch (err) {
          statusLine.textContent = "Fehler: " + err;
          setState("idle");
        }
      };
      recorder.stop();
    }

    root.addEventListener("keydown", (e) => {
      if (e.code === "Space" && !spaceDown) {
        spaceDown = true;
        e.preventDefault();
        startRecording();
      }
    });
    root.addEventListener("keyup", (e) => {
      if (e.code === "Space") {
        spaceDown = false;
        e.preventDefault();
        stopRecordingAndSend();
      }
    });
    root.addEventListener("click", () => root.focus());
    setState("idle");
  })();
  </script>
</div>
"""


def render_voice_widget(api_url: str, api_key: str) -> None:
    html = VOICE_WIDGET_HTML.replace("__API_URL__", api_url.replace('"', "")).replace(
        "__API_KEY__", api_key.replace('"', "")
    )
    components.html(html, height=380)


with st.sidebar:
    st.header("Verbindung")
    api_url = st.text_input(
        "API-URL (Tunnel)",
        value=st.session_state.get("api_url", _default("API_URL")),
        placeholder="https://xxxx.trycloudflare.com",
        help="Die öffentliche Tunnel-URL deines lokalen Rechners "
        "(cloudflared/ngrok). Ändert sich bei jedem Neustart des Tunnels.",
    )
    api_key = st.text_input(
        "API-Key",
        value=st.session_state.get("api_key", _default("API_KEY")),
        type="password",
    )
    st.session_state["api_url"] = api_url
    st.session_state["api_key"] = api_key

    if st.button("Text-Verlauf löschen"):
        st.session_state["messages"] = []
        st.rerun()

st.title("🤖 Jarvis")

render_voice_widget(api_url, api_key)

st.caption(
    "Sprache läuft direkt im Widget oben (Leertaste halten). "
    "Getippte Nachrichten laufen über den Chat unten."
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("…oder Nachricht an Jarvis tippen")

if prompt:
    if not api_url or not api_key:
        st.error("API-URL und API-Key in der Sidebar eintragen.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Jarvis denkt…"):
            try:
                resp = requests.post(
                    f"{api_url.rstrip('/')}/chat",
                    json={
                        "message": prompt,
                        "history": st.session_state["messages"][:-1],
                    },
                    headers={"X-API-Key": api_key},
                    timeout=120,
                )
                if resp.status_code == 401:
                    reply = "⚠️ API-Key falsch oder fehlt."
                elif resp.ok:
                    reply = resp.json()["reply"]
                else:
                    reply = f"⚠️ Fehler vom Server ({resp.status_code}): {resp.text[:300]}"
            except requests.RequestException as exc:
                reply = (
                    f"⚠️ Konnte den Tunnel nicht erreichen: {exc!r}\n\n"
                    "Läuft dein Rechner + Tunnel noch? Stimmt die URL?"
                )
        st.markdown(reply)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
