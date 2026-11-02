"""Thin Streamlit chat frontend for Jarvis.

Deployed to Streamlit Community Cloud. Talks to the real LLM running on
your own machine (jarvis/backend/src/jarvis/remote_api.py) over a public
tunnel URL (ngrok/cloudflared) you paste into the sidebar — this app
itself never runs any model.
"""
from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="Jarvis", page_icon="🤖")


def _default(key: str, fallback: str = "") -> str:
    try:
        return str(st.secrets[key])
    except Exception:
        return fallback


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
        st.rerun()

st.title("🤖 Jarvis")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Nachricht an Jarvis…")

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
