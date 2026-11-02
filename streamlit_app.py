"""Thin Streamlit chat+voice frontend for Jarvis.

Deployed to Streamlit Community Cloud. Talks to the real LLM (and STT/TTS)
running on your own machine (jarvis/backend/src/jarvis/remote_api.py) over
a public tunnel URL (ngrok/cloudflared) you paste into the sidebar — this
app itself never runs any model.
"""
from __future__ import annotations

import base64
import hashlib

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Jarvis", page_icon="🤖")


def _default(key: str, fallback: str = "") -> str:
    try:
        return str(st.secrets[key])
    except Exception:
        return fallback


_RING_COLORS = {
    "idle": ("#2b6cff", "pulse 3s ease-in-out infinite"),
    "thinking": ("#ffb020", "spin 1.1s linear infinite"),
    "speaking": ("#2bff9e", "pulse 0.6s ease-in-out infinite"),
}


def render_ring(state: str) -> None:
    color, anim = _RING_COLORS.get(state, _RING_COLORS["idle"])
    components.html(
        f"""
        <div style="display:flex;justify-content:center;padding:12px 0;">
          <div style="
              width:140px;height:140px;border-radius:50%;
              background:radial-gradient(circle at 50% 50%, {color}22 0%, transparent 70%);
              display:flex;align-items:center;justify-content:center;
              animation: {anim};
          ">
            <div style="
                width:96px;height:96px;border-radius:50%;
                border:4px solid {color};
                border-top-color:transparent;
                box-shadow:0 0 24px {color}aa, inset 0 0 16px {color}55;
            "></div>
          </div>
        </div>
        <style>
          @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 0.85; }}
            50% {{ transform: scale(1.08); opacity: 1; }}
          }}
          @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
          }}
          body {{ margin: 0; background: transparent; }}
        </style>
        """,
        height=170,
    )


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

    if st.button("Verlauf löschen"):
        st.session_state["messages"] = []
        st.session_state["jarvis_state"] = "idle"
        st.rerun()

st.title("🤖 Jarvis")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "jarvis_state" not in st.session_state:
    st.session_state["jarvis_state"] = "idle"
if "last_audio_hash" not in st.session_state:
    st.session_state["last_audio_hash"] = None
if "pending_audio" not in st.session_state:
    st.session_state["pending_audio"] = None  # (bytes, sample_rate) to autoplay once


def _require_connection() -> bool:
    if not api_url or not api_key:
        st.error("API-URL und API-Key in der Sidebar eintragen.")
        return False
    return True


def _run_voice_turn(wav_bytes: bytes) -> None:
    st.session_state["jarvis_state"] = "thinking"
    with st.spinner("Jarvis hört zu & denkt…"):
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/voice",
                files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
                headers={"X-API-Key": api_key},
                timeout=180,
            )
        except requests.RequestException as exc:
            st.error(f"Konnte den Tunnel nicht erreichen: {exc!r}")
            st.session_state["jarvis_state"] = "idle"
            return

    if resp.status_code == 401:
        st.error("API-Key falsch oder fehlt.")
        st.session_state["jarvis_state"] = "idle"
        return
    if resp.status_code == 400:
        st.warning("Keine Sprache erkannt — nochmal versuchen.")
        st.session_state["jarvis_state"] = "idle"
        return
    if not resp.ok:
        st.error(f"Fehler vom Server ({resp.status_code}): {resp.text[:300]}")
        st.session_state["jarvis_state"] = "idle"
        return

    body = resp.json()
    st.session_state["messages"].append({"role": "user", "content": body["transcript"]})
    st.session_state["messages"].append({"role": "assistant", "content": body["reply"]})
    st.session_state["jarvis_state"] = "speaking"

    audio_bytes = base64.b64decode(body["audio_base64"]) if body["audio_base64"] else b""
    st.session_state["pending_audio"] = audio_bytes or None


# --- Voice input (push-to-talk) ---
audio_value = st.audio_input("🎤 Sprich mit Jarvis", sample_rate=16000)
if audio_value is not None:
    raw = audio_value.getvalue()
    audio_hash = hashlib.md5(raw).hexdigest()
    if audio_hash != st.session_state["last_audio_hash"]:
        st.session_state["last_audio_hash"] = audio_hash
        if _require_connection():
            _run_voice_turn(raw)

render_ring(st.session_state["jarvis_state"])

if st.session_state["pending_audio"]:
    st.audio(st.session_state["pending_audio"], format="audio/wav", autoplay=True)
    st.session_state["pending_audio"] = None

# --- Chat history ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Text input (alternative to voice) ---
prompt = st.chat_input("…oder Nachricht an Jarvis tippen")

if prompt:
    if _require_connection():
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
