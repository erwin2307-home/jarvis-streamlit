"""Thin Streamlit chat+voice frontend for Jeanny.

Deployed to Streamlit Community Cloud. Talks to the real LLM (and STT/TTS)
running on your own machine (jarvis/backend/src/jarvis/remote_api.py) over
a public tunnel URL (ngrok/cloudflared) configured in st.secrets — this
app itself never runs any model.

Voice turns (recording -> /voice -> transcript+reply+TTS audio) run
entirely inside one self-contained JS widget (see VOICE_WIDGET_HTML
below), talking to the tunnel URL directly from the browser. This is a
deliberate split from the typed-text path (st.chat_input -> /chat),
which goes through Streamlit's own Python server as usual: a plain
components.html iframe has no channel back into Streamlit's session
state, so the voice widget keeps its own small transcript rather than
pretending to merge into the Python-side chat history below it.

The widget is embedded via a SHORT-LIVED SESSION TOKEN, never the real
JARVIS_REMOTE_API_KEY: this page is public once deployed to Streamlit
Community Cloud, and the iframe's HTML (including whatever credential
is baked into it) is visible in plain text to any visitor via view-source
— no interaction needed. Python mints the token through POST /session-token
(a normal server-side call, so the real key never reaches the browser for
this either) and caches it in session_state keyed by (api_url, api_key),
so the embedded HTML stays byte-identical across Streamlit reruns — that
stability is what keeps the iframe (and its live countdown timers) from
being torn down and recreated every time something elsewhere on the page
triggers a rerun.

API_KEY and API_URL are BOTH read ONLY from st.secrets — there is
deliberately no sidebar widget for either, editable or otherwise.

For API_KEY: a text_input prefilled with a secret still ships that value
to the browser to render the widget (Streamlit has to, to display it),
inspectable via devtools regardless of type="password" masking it
visually; the only way to not leak it is to never put it in a widget.

For API_URL: this page is public, so an editable sidebar field for it is
editable by ANY visitor, not just the owner. An earlier version allowed
it but checked the hostname against known tunnel-provider domain suffixes
(*.trycloudflare.com etc.) — that is NOT a security boundary, since those
are shared public domains anyone can get their own free subdomain under
in seconds (`cloudflared tunnel --url ...` against their own server).
A visitor could still point it at their own tunnel and the credentialed
server-side requests below (session-token mint, typed-chat /chat call)
would hand them the real API_KEY. Pinning it to st.secrets closes that:
whoever can edit this app's secrets is the only one who decides where
the key is sent, same trust level as the key itself.

The cost: since quick tunnels (cloudflared/ngrok without an account) get
a new random hostname every restart, API_URL in secrets needs updating
on Streamlit Cloud after every tunnel restart — there's no way to keep
both that convenience and this security property with a shared-domain
quick tunnel. A named Cloudflare Tunnel on a domain you own would give a
fixed hostname and remove this friction, but that's a bigger one-time
setup, not done here.
"""
from __future__ import annotations

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Jeanny", page_icon="🤖")


def _default(key: str, fallback: str = "") -> str:
    try:
        return str(st.secrets[key])
    except Exception:
        return fallback


def get_voice_session_token(api_url: str, api_key: str) -> str | None:
    """Mints (once) and caches a short-lived token scoped for the voice
    widget, re-minting only when api_url/api_key actually change — minting
    on every rerun would still be safe but would waste a round trip on
    every typed chat message too."""
    if not api_url.startswith("https://"):
        return None
    cache_key = (api_url, api_key)
    cached = st.session_state.get("_voice_token_cache")
    if cached and cached[0] == cache_key:
        return cached[1]  # type: ignore[no-any-return]

    try:
        resp = requests.post(
            f"{api_url.rstrip('/')}/session-token",
            headers={"X-API-Key": api_key},
            timeout=10,
            allow_redirects=False,  # never replay the header to a redirect target
        )
        resp.raise_for_status()
        token = resp.json()["token"]
    except requests.RequestException:
        return None

    st.session_state["_voice_token_cache"] = (cache_key, token)
    return token


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
    #talk-btn {
      border: none; border-radius: 999px; padding: 14px 36px;
      font-size: 15px; font-weight: 600; color: #05070c;
      background: linear-gradient(180deg, #5fd4ff, #2b8cff);
      box-shadow: 0 0 18px #2b8cff88, 0 2px 0 #1a4d8c inset;
      touch-action: none; -webkit-touch-callout: none; user-select: none;
      -webkit-user-select: none; cursor: pointer;
    }
    #talk-btn:active, #talk-btn.pressed {
      background: linear-gradient(180deg, #2b8cff, #1a6fd6);
      box-shadow: 0 0 28px #5fd4ffcc, 0 1px 0 #1a4d8c inset;
      transform: translateY(1px);
    }
    #timers {
      display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;
      min-height: 0; padding: 0 8px;
    }
    .timer-chip {
      border: 3px solid #2b8cff; border-radius: 12px; padding: 4px 14px;
      background: rgba(43,140,255,0.08); text-align: center;
      box-shadow: 0 0 10px #2b8cff55; transition: all 0.3s ease;
      opacity: 0.7; transform: scale(0.85);
    }
    .timer-chip.timer-foreground {
      opacity: 1; transform: scale(1.15);
      box-shadow: 0 0 22px #2b8cffcc; z-index: 5;
    }
    .timer-name {
      color: #9fe8c0; font-size: 11px; font-weight: 600;
      letter-spacing: 0.03em; text-transform: uppercase;
      margin-bottom: 2px;
    }
    .timer-time {
      color: #ff3b3b; font-weight: 700; font-family: "Courier New", monospace;
      font-size: 15px;
    }
    .timer-chip.timer-foreground .timer-time { font-size: 26px; }
    #play-fallback {
      border: 1px solid #2b8cff; border-radius: 999px; padding: 6px 18px;
      font-size: 12px; color: #9fe8c0; background: transparent; cursor: pointer;
    }
    #cooldown-choice { display: none; gap: 10px; }
    #cooldown-choice button, #cooldown-continue button {
      border: 1px solid #ffb020; border-radius: 999px; padding: 8px 20px;
      font-size: 13px; font-weight: 600; color: #ffb020; background: transparent;
      cursor: pointer;
    }
    #cooldown-continue { display: none; }
    /* Same pill size as #talk-btn (14px/36px padding, 15px font) — only
       the gradient colors differ, so each big button reads as equally
       "primary" rather than talk-btn looking like the only real action. */
    #cooldown-manual-btn, #games-btn {
      border: none; border-radius: 999px; padding: 14px 36px;
      font-size: 15px; font-weight: 600; color: #05070c; cursor: pointer;
    }
    #cooldown-manual-btn {
      background: linear-gradient(180deg, #ffd270, #ffb020);
      box-shadow: 0 0 18px #ffb02088, 0 2px 0 #a86f0c inset;
    }
    #games-btn {
      background: linear-gradient(180deg, #d9b3ff, #b388ff);
      box-shadow: 0 0 18px #b388ff88, 0 2px 0 #6b4fa8 inset;
    }
    #games-choice { display: none; gap: 10px; }
    #games-choice button {
      border: 1px solid #b388ff; border-radius: 999px; padding: 8px 20px;
      font-size: 13px; font-weight: 600; color: #b388ff; background: transparent;
      cursor: pointer;
    }
  </style>
  <div id="ring-wrap">
    <div id="ring-glow"></div>
    <div id="ring-core"></div>
  </div>
  <button id="talk-btn" type="button">🎤 Gedrückt halten zum Sprechen</button>
  <button id="cooldown-manual-btn" type="button">❄️ Cool Down</button>
  <button id="games-btn" type="button">🎮 Spiele</button>
  <div id="games-choice">
    <button id="game-tetris" type="button">Tetris</button>
    <button id="game-pacman" type="button">Pac-Man</button>
    <button id="game-aliens" type="button">Aliens</button>
  </div>
  <div id="timers"></div>
  <div id="status-line">Button oder Leertaste (Ring anklicken) halten zum Sprechen</div>
  <div id="hint-line"></div>
  <div id="cooldown-choice">
    <button id="choice-tecan" type="button">Timer Tecan (2:43:00)</button>
    <button id="choice-bioroboter" type="button">Timer Bioroboter (2:55:00)</button>
  </div>
  <div id="cooldown-continue">
    <button id="continue-btn" type="button">Weiter ▶</button>
  </div>
  <button id="play-fallback" type="button" style="display:none;">▶️ Antwort abspielen</button>
  <div id="transcript"></div>

  <script>
  (function () {
    const API_URL = "__API_URL__";
    const API_KEY = "__API_KEY__";
    const root = document.getElementById("jarvis-root");
    const statusLine = document.getElementById("status-line");
    const transcriptEl = document.getElementById("transcript");
    const talkBtn = document.getElementById("talk-btn");
    const cooldownManualBtn = document.getElementById("cooldown-manual-btn");
    const gamesBtn = document.getElementById("games-btn");
    const gamesChoiceEl = document.getElementById("games-choice");
    const gameTetrisBtn = document.getElementById("game-tetris");
    const gamePacmanBtn = document.getElementById("game-pacman");
    const gameAliensBtn = document.getElementById("game-aliens");
    const playFallback = document.getElementById("play-fallback");
    const timersEl = document.getElementById("timers");
    const hintLine = document.getElementById("hint-line");
    const choiceEl = document.getElementById("cooldown-choice");
    const choiceTecanBtn = document.getElementById("choice-tecan");
    const choiceBioroboterBtn = document.getElementById("choice-bioroboter");
    const continueEl = document.getElementById("cooldown-continue");
    const continueBtn = document.getElementById("continue-btn");

    // Real animal recordings (verified Wikimedia Commons .ogg files) mixed
    // with speech-synthesized meme alarm lines — using the browser's own
    // SpeechSynthesis API for the latter sidesteps hunting for sketchy,
    // unlicensed meme audio: it says exactly the requested line, with no
    // licensing/hosting risk at all.
    const ALARMS = [
      { type: "audio", url: "https://upload.wikimedia.org/wikipedia/commons/c/c5/Rooster_crowing.ogg" },
      { type: "audio", url: "https://upload.wikimedia.org/wikipedia/commons/5/58/Barking_of_a_dog_2.ogg" },
      { type: "audio", url: "https://upload.wikimedia.org/wikipedia/commons/0/0c/Meow_domestic_cat.ogg" },
      { type: "audio", url: "https://upload.wikimedia.org/wikipedia/commons/4/48/Mudchute_cow_1.ogg" },
      { type: "speech", text: "Achtung, Achtung! Zombie-Ausbruch gemeldet!", lang: "de-DE" },
      { type: "speech", text: "Warnung! Der Weltuntergang steht unmittelbar bevor!", lang: "de-DE" },
      { type: "speech", text: "Atomic missile will arrive in 3, 2, 1.", lang: "en-US" },
    ];

    function playRandomAlarm() {
      const alarm = ALARMS[Math.floor(Math.random() * ALARMS.length)];
      if (alarm.type === "audio") {
        new Audio(alarm.url).play().catch(() => {});
      } else if (window.speechSynthesis) {
        try {
          const utter = new SpeechSynthesisUtterance(alarm.text);
          utter.lang = alarm.lang;
          utter.rate = 0.95;
          window.speechSynthesis.speak(utter);
        } catch (e) { /* speech synthesis unavailable — skip silently */ }
      }
    }

    let timers = []; // {id, remaining, name}
    let timerIdSeq = 0;

    function formatMMSS(s) {
      const m = Math.floor(s / 60);
      const sec = s % 60;
      return String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");
    }

    function renderTimers() {
      timers.sort((a, b) => a.remaining - b.remaining);
      timersEl.innerHTML = "";
      timers.forEach((t, idx) => {
        const chip = document.createElement("div");
        chip.className = "timer-chip" + (idx === 0 ? " timer-foreground" : "");
        if (t.name) {
          const nameEl = document.createElement("div");
          nameEl.className = "timer-name";
          nameEl.textContent = t.name; // textContent, not innerHTML — name is user/STT-sourced text
          chip.appendChild(nameEl);
        }
        const timeEl = document.createElement("div");
        timeEl.className = "timer-time";
        timeEl.textContent = formatMMSS(t.remaining);
        chip.appendChild(timeEl);
        timersEl.appendChild(chip);
      });
    }

    function addTimer(totalSeconds, name, onComplete) {
      timers.push({ id: ++timerIdSeq, remaining: totalSeconds, name: name || null, onComplete: onComplete || null });
      renderTimers();
    }

    setInterval(() => {
      if (timers.length === 0) return;
      const fired = [];
      timers.forEach((t) => {
        t.remaining -= 1;
        if (t.remaining <= 0) fired.push(t);
      });
      if (fired.length > 0) {
        const firedIds = fired.map((t) => t.id);
        timers = timers.filter((t) => !firedIds.includes(t.id));
        // onComplete (cooldown steps) runs instead of the random alarm —
        // each step speaks its own cue or chains the next step itself.
        fired.forEach((t) => (t.onComplete ? t.onComplete() : playRandomAlarm()));
      }
      renderTimers();
    }, 1000);

    // --- "Cool down" lab-protocol macro -------------------------------
    // Entirely client-side, triggered only by matching the transcript
    // text of a /voice response — the server has no notion of this
    // workflow (see is_cooldown_trigger/is_continue_query in
    // remote_api.py, which exist only so "cool down"/"weiter" don't
    // waste an LLM call and get an irrelevant answer spoken aloud).
    // Fixed named sequence per the lab protocol: 30 min cooldown -> (x3
    // "Beadchips jetzt rausnehmen") -> [weiter] -> Chip Wash 1 (1 min) ->
    // Chip Bath 1 (1 min) -> [weiter] -> Chip Wash 2 (1 min) -> Chip Bath
    // 2 (1 min) -> choose Tecan (2:43:00) or Bioroboter (2:55:00) -> (x3
    // "Ready for Rumble").
    const COOLDOWN_TRIGGER_RE = /c+o{2,4}l\s*down/i;
    const CONTINUE_RE = /\bweiter\b/i;
    let cooldown = null; // { waitingForContinue: false | "wash1" | "wash2" }

    function cooldownHint(text) {
      hintLine.textContent = text || "";
    }

    function speakRepeated(text, lang, times, onDone) {
      if (!window.speechSynthesis) { if (onDone) onDone(); return; }
      let said = 0;
      function speakOnce() {
        try {
          const utter = new SpeechSynthesisUtterance(text);
          utter.lang = lang;
          utter.rate = 0.95;
          utter.onend = () => {
            said += 1;
            if (said < times) setTimeout(speakOnce, 500);
            else if (onDone) onDone();
          };
          window.speechSynthesis.speak(utter);
        } catch (e) {
          if (onDone) onDone(); // speech synthesis unavailable — don't get stuck
        }
      }
      speakOnce();
    }

    function startCooldown() {
      if (cooldown) return; // already running — ignore a repeated trigger
      cooldown = { waitingForContinue: false };
      cooldownHint("Cool-down läuft: 30 Minuten");
      addTimer(30 * 60, "Cool Down", () => {
        speakRepeated("Beadchips jetzt rausnehmen", "de-DE", 3, () => {
          cooldown.waitingForContinue = "wash1";
          cooldownHint("Sage 'weiter' oder klicke Weiter für Chip Wash 1");
          continueEl.style.display = "flex";
        });
      });
    }

    function startChipWash1() {
      cooldownHint("");
      continueEl.style.display = "none";
      addTimer(60, "Chip Wash 1", startChipBath1);
    }
    function startChipBath1() {
      addTimer(60, "Chip Bath 1", () => {
        cooldown.waitingForContinue = "wash2";
        cooldownHint("Sage 'weiter' oder klicke Weiter für Chip Wash 2");
        continueEl.style.display = "flex";
      });
    }
    function startChipWash2() {
      cooldownHint("");
      continueEl.style.display = "none";
      addTimer(60, "Chip Wash 2", startChipBath2);
    }
    function startChipBath2() {
      addTimer(60, "Chip Bath 2", () => {
        cooldownHint("Wähle: Tecan oder Bioroboter");
        choiceEl.style.display = "flex";
      });
    }

    function startInstrumentTimer(label, seconds) {
      choiceEl.style.display = "none";
      cooldownHint("");
      addTimer(seconds, label, () => {
        speakRepeated("Ready for Rumble", "en-US", 3, () => {
          cooldown = null; // workflow complete
        });
      });
    }
    choiceTecanBtn.addEventListener("click", () => startInstrumentTimer("Timer Tecan", 2 * 3600 + 43 * 60));
    choiceBioroboterBtn.addEventListener("click", () => startInstrumentTimer("Timer Bioroboter", 2 * 3600 + 55 * 60));
    continueBtn.addEventListener("click", () => handleCooldownContinue());
    cooldownManualBtn.addEventListener("click", () => startCooldown());
    gamesBtn.addEventListener("click", () => {
      gamesChoiceEl.style.display = gamesChoiceEl.style.display === "flex" ? "none" : "flex";
    });

    // Tetris/Pac-Man/Aliens: generated on request by the "large" coding
    // model (see games.py/remote_api.py) and served as a static page —
    // this just triggers generation, polls status, then opens the
    // result in a new tab once it's ready. API_KEY here is already the
    // short-lived session token (see render_voice_widget), safe to put
    // in a query string for the new-tab navigation (window.open can't
    // attach a custom header).
    const GAME_POLL_MS = 3000;
    const GAME_POLL_MAX_TRIES = 200; // ~10 min cap — a 14b-model full
                                      // game generation can genuinely
                                      // take minutes on CPU.

    async function playGame(gameId, label) {
      // Opened synchronously, inside the click handler itself — a
      // window.open() called later (after the awaits below) loses the
      // user-gesture context and most browsers silently block it as a
      // popup. Navigating this already-open tab's location afterwards
      // is not subject to that block.
      const gameWin = window.open("", "_blank");
      if (gameWin) {
        gameWin.document.write(
          "<body style='background:#05070c;color:#9fe8c0;font-family:sans-serif;" +
          "display:flex;align-items:center;justify-content:center;height:100vh;'>" +
          "<p>" + label + " wird programmiert …</p></body>"
        );
      }

      try {
        await fetch(API_URL.replace(/\/$/, "") + "/games/" + gameId + "/generate", {
          method: "POST",
          headers: { "X-API-Key": API_KEY },
        });
      } catch (e) {
        cooldownHint(label + ": Fehler beim Starten — " + e);
        return;
      }
      for (let i = 0; i < GAME_POLL_MAX_TRIES; i++) {
        let status;
        try {
          const resp = await fetch(
            API_URL.replace(/\/$/, "") + "/games/" + gameId + "/status",
            { headers: { "X-API-Key": API_KEY } }
          );
          status = await resp.json();
        } catch (e) {
          cooldownHint(label + ": Fehler beim Abfragen — " + e);
          return;
        }
        if (status.status === "done") {
          cooldownHint("");
          const gameUrl = API_URL.replace(/\/$/, "") + "/games/" + gameId + "?key=" + encodeURIComponent(API_KEY);
          if (gameWin && !gameWin.closed) gameWin.location.href = gameUrl;
          else window.open(gameUrl, "_blank"); // tab was blocked/closed — try again, may still be blocked
          return;
        }
        if (status.status === "error") {
          cooldownHint(label + ": Generierung fehlgeschlagen — " + (status.error || "unbekannt"));
          if (gameWin && !gameWin.closed) gameWin.close();
          return;
        }
        const tokens = status.tokens || 0;
        const maxTokens = status.max_tokens || 1;
        const pct = Math.min(Math.round((tokens / maxTokens) * 100), 98);
        cooldownHint(label + " wird programmiert … " + pct + "%");
        await new Promise((r) => setTimeout(r, GAME_POLL_MS));
      }
      cooldownHint(label + ": dauert ungewöhnlich lange, bitte später erneut versuchen.");
    }

    gameTetrisBtn.addEventListener("click", () => playGame("tetris", "Tetris"));
    gamePacmanBtn.addEventListener("click", () => playGame("pacman", "Pac-Man"));
    gameAliensBtn.addEventListener("click", () => playGame("aliens", "Aliens"));

    function handleCooldownContinue() {
      if (!cooldown || !cooldown.waitingForContinue) return;
      const next = cooldown.waitingForContinue;
      cooldown.waitingForContinue = false;
      if (next === "wash1") startChipWash1();
      else if (next === "wash2") startChipWash2();
    }

    let mediaStream = null;
    let recorder = null;
    let chunks = [];
    let spaceDown = false;
    let isActive = false; // guards against button + spacebar double-triggering

    function setState(state) {
      root.classList.remove("listening", "thinking", "speaking");
      if (state) root.classList.add(state);
      const labels = {
        idle: "Button oder Leertaste (Ring anklicken) halten zum Sprechen",
        listening: "Höre zu…",
        thinking: "Denke…",
        speaking: "Spreche…",
      };
      statusLine.textContent = labels[state] || labels.idle;
    }

    function addLine(cls, text) {
      const div = document.createElement("div");
      div.className = cls;
      div.textContent = (cls === "you" ? "Du: " : "Jeanny: ") + text;
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
      if (isActive) return; // already recording (button+space overlap) — ignore
      if (!API_URL || !API_KEY) {
        statusLine.textContent = "API-URL fehlt oder kein Sitzungs-Token (siehe Hinweis oben)";
        return;
      }
      try {
        await ensureStream();
      } catch (e) {
        statusLine.textContent = "Mikrofon-Zugriff verweigert: " + e;
        return;
      }
      isActive = true;
      talkBtn.classList.add("pressed");
      playFallback.style.display = "none";
      chunks = [];
      recorder = new MediaRecorder(mediaStream);
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.start();
      setState("listening");
    }

    function stopRecordingAndSend() {
      if (!isActive) return;
      isActive = false;
      talkBtn.classList.remove("pressed");
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
          if (body.timer_seconds) addTimer(body.timer_seconds, body.timer_name);
          if (COOLDOWN_TRIGGER_RE.test(body.transcript)) startCooldown();
          else if (CONTINUE_RE.test(body.transcript)) handleCooldownContinue();
          setState("speaking");

          if (body.audio_base64) {
            const audio = new Audio("data:audio/wav;base64," + body.audio_base64);
            audio.onended = () => setState("idle");
            audio.onerror = () => setState("idle");
            audio.play().catch(() => {
              // iOS can block autoplay here if too much time passed since
              // the last user gesture (the recording tap) — offer a
              // one-tap fallback, which is itself a fresh gesture.
              setState("idle");
              playFallback.style.display = "inline-block";
              playFallback.onclick = () => {
                playFallback.style.display = "none";
                setState("speaking");
                audio.play().catch(() => setState("idle"));
              };
            });
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

    // Button: works for mouse (PC) and touch (iPhone/mobile) via Pointer
    // Events. Additive to spacebar, not a replacement — isActive guards
    // against both firing at once.
    talkBtn.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      // Captures the pointer so pointerleave/pointerout don't fire while
      // held — without this, a finger/cursor drifting off the rounded
      // button's edge mid-press (very easy on the pill shape) fired
      // pointerleave and stopped recording early, jumping straight to
      // "thinking" while the user was still mid-sentence.
      try { talkBtn.setPointerCapture(e.pointerId); } catch (err) { /* unsupported — pointerleave fallback below still applies */ }
      startRecording();
    });
    talkBtn.addEventListener("pointerup", (e) => {
      e.preventDefault();
      stopRecordingAndSend();
    });
    talkBtn.addEventListener("pointercancel", (e) => {
      e.preventDefault();
      stopRecordingAndSend();
    });
    talkBtn.addEventListener("pointerleave", (e) => {
      // With setPointerCapture above, this should no longer fire while
      // actually pressed — kept as a defensive fallback for a browser
      // that ignores capture, guarded the same way it always was.
      if (isActive) stopRecordingAndSend();
    });
    // Prevent context menu / long-press callout from interrupting on mobile.
    talkBtn.addEventListener("contextmenu", (e) => e.preventDefault());

    setState("idle");
  })();
  </script>
</div>
"""


def render_voice_widget(api_url: str, voice_token: str) -> None:
    # voice_token is a short-lived /session-token value, never the real
    # JARVIS_REMOTE_API_KEY — see the module docstring for why.
    html = VOICE_WIDGET_HTML.replace("__API_URL__", api_url.replace('"', "")).replace(
        "__API_KEY__", voice_token.replace('"', "")
    )
    # 520, not 380 — Cool Down + Spiele buttons (plus the Tetris/Pac-Man/
    # Aliens reveal row) added below talk-btn pushed the flex column
    # taller; components.html doesn't auto-size or scroll, so anything
    # past the fixed height gets silently clipped.
    components.html(html, height=520)


# API_KEY and API_URL come ONLY from st.secrets — never from a widget.
# See the module docstring for why API_URL is pinned here too, not just
# the key: an editable sidebar field for it would let any visitor of this
# public page redirect where the real key gets sent.
api_key = _default("API_KEY")
api_url = _default("API_URL")

with st.sidebar:
    st.header("Verbindung")
    if api_url:
        st.caption(f"Tunnel: `{api_url}`")
    else:
        st.caption("Tunnel: nicht konfiguriert (API_URL fehlt in den Secrets)")
    st.caption(
        "API-URL und API-Key werden nur über die Streamlit-Secrets "
        "konfiguriert (Settings -> Secrets), nicht hier auf der Seite — "
        "sonst könnte jeder Besucher den echten Key auf einen eigenen "
        "Server umleiten."
    )

    if st.button("Text-Verlauf löschen"):
        st.session_state["messages"] = []
        st.rerun()

st.title("🤖 Jeanny")

if api_url and api_key:
    voice_token = get_voice_session_token(api_url, api_key)
    if voice_token:
        render_voice_widget(api_url, voice_token)
    else:
        st.warning(
            "Konnte kein Sitzungs-Token für das Sprach-Widget holen — "
            "Tunnel/API-Key prüfen. Getippter Chat unten geht trotzdem."
        )
elif not api_url:
    st.info("API_URL fehlt in den Secrets (Settings -> Secrets auf Streamlit Cloud).")
else:
    st.info("API_KEY fehlt in den Secrets (Settings -> Secrets auf Streamlit Cloud).")

st.caption(
    "Sprache läuft direkt im Widget oben (Leertaste halten). "
    "Getippte Nachrichten laufen über den Chat unten."
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("…oder Nachricht an Jeanny tippen")

if prompt:
    if not api_url or not api_key:
        st.error("API_URL und API_KEY müssen in den Secrets gesetzt sein.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Jeanny denkt…"):
            try:
                resp = requests.post(
                    f"{api_url.rstrip('/')}/chat",
                    json={
                        "message": prompt,
                        "history": st.session_state["messages"][:-1],
                    },
                    headers={"X-API-Key": api_key},
                    timeout=120,
                    allow_redirects=False,  # never replay the header to a redirect target
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
