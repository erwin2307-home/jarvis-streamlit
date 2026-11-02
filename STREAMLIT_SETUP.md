# Jarvis via Streamlit Cloud + Tunnel

Thin chat UI on Streamlit Community Cloud, talking to the real LLM running
on your own machine through a public tunnel. Your machine does all the
work (Ollama + optional Groq/DeepSeek cloud routing); Streamlit Cloud only
renders the chat.

Scope: **text chat only**. No voice, no coding fast-path (auto-running
generated scripts on an internet-reachable endpoint would be remote code
execution — deliberately left out of `remote_api.py`), no QA fast-path.

## 1. Run the local API

```bash
cd backend
uv sync --all-extras
JARVIS_REMOTE_API_KEY="pick-a-long-random-secret" uv run uvicorn jarvis.remote_api:app --host 0.0.0.0 --port 8800
```

Requires Ollama running (`ollama serve`) with `llm_model` pulled (see
`backend/src/jarvis/config.py` for the model name / Groq/DeepSeek key
file locations — same config as the main voice app).

## 2. Tunnel it

Cloudflare Tunnel (free, no account needed for a quick tunnel):

```bash
cloudflared tunnel --url http://localhost:8800
```

It prints a `https://xxxx.trycloudflare.com` URL. This URL changes every
time you restart the tunnel — that's why the Streamlit app takes it as a
sidebar field, not a hardcoded constant.

(ngrok works the same way: `ngrok http 8800`.)

## 3. Push this folder to GitHub

```bash
git init
git add .
git commit -m "Jarvis Streamlit frontend"
git remote add origin <your-repo-url>
git push -u origin main
```

`Deepseek_api_key.txt` / `Groq_api_key.txt` / `.streamlit/secrets.toml`
are gitignored — check `git status` before pushing if unsure.

## 4. Deploy on Streamlit Community Cloud

1. share.streamlit.io → New app → pick this repo, branch `main`, main
   file `streamlit_app.py`.
2. App settings → Secrets → paste:
   ```toml
   API_URL = "https://xxxx.trycloudflare.com"
   API_KEY = "the same secret you set as JARVIS_REMOTE_API_KEY"
   ```
3. Deploy. The sidebar fields pre-fill from secrets but can be overridden
   live (useful since the tunnel URL changes on every restart — update it
   in the sidebar without redeploying).

## Notes

- Your machine + Ollama + tunnel must all be running for the app to
  answer. If you shut your PC down, the Streamlit app just shows a
  connection error.
- The API key is the only thing standing between the public tunnel URL
  and your local LLM — treat it like a password, rotate it if you
  suspect it leaked.
