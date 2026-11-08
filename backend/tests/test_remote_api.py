"""Remote chat/voice API tests. Uses Fake* backends injected onto
app.state — never hits real Ollama/cloud/Whisper/TTS."""
import base64
import re

import pytest

from jarvis.config import get_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JARVIS_REMOTE_API_KEY", "test-secret")
    monkeypatch.setenv("JARVIS_USE_CLOUD", "0")
    get_settings.cache_clear()

    from fastapi.testclient import TestClient
    from jarvis import remote_api
    from jarvis.llm import FakeLlm
    from jarvis.stt import FakeStt
    from jarvis.tts import FakeTts

    remote_api.app.state.llm = FakeLlm(["hallo ", "welt"])
    remote_api.app.state.stt = FakeStt(text="hallo jarvis")
    remote_api.app.state.tts = FakeTts()
    with TestClient(remote_api.app) as c:
        yield c

    get_settings.cache_clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_chat_requires_api_key(client):
    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 401


def test_chat_rejects_wrong_api_key(client):
    resp = client.post(
        "/chat", json={"message": "hi"}, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401


def test_chat_returns_reply_with_valid_key(client):
    resp = client.post(
        "/chat", json={"message": "hi"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "hallo welt"


def test_chat_rejects_empty_message(client):
    resp = client.post(
        "/chat", json={"message": "   "}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 400


def test_voice_options_preflight_is_allowed(client):
    # Streamlit Cloud serves a different origin than the tunnel URL, so the
    # browser sends a CORS preflight (OPTIONS) before the real POST /voice.
    # Without CORSMiddleware this 405s and the browser never sends the real
    # request — this reproduces that failure.
    resp = client.options(
        "/voice",
        headers={
            "Origin": "https://example.streamlit.app",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_voice_requires_api_key(client, fixtures_dir):
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post("/voice", files={"audio": ("rec.wav", wav_bytes, "audio/wav")})
    assert resp.status_code == 401


def test_voice_returns_transcript_reply_and_audio(client, fixtures_dir):
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "hallo jarvis"
    assert body["reply"] != ""
    audio = base64.b64decode(body["audio_base64"])
    assert len(audio) > 0
    assert body["sample_rate"] > 0


def test_voice_rejects_empty_transcript(client, fixtures_dir):
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="")
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "text",
    ["wie spät ist es", "Wie spät ist es?", "wieviel uhr ist es", "what time is it"],
)
def test_is_time_query_matches_time_questions(client, text):
    from jarvis.remote_api import is_time_query

    assert is_time_query(text) is True


@pytest.mark.parametrize("text", ["wie geht es dir", "baue einen taschenrechner", "hallo"])
def test_is_time_query_does_not_match_other_questions(client, text):
    from jarvis.remote_api import is_time_query

    assert is_time_query(text) is False


_TIME_PATTERN = re.compile(r"\d{2}:\d{2} Uhr")


def test_chat_answers_time_query_deterministically_without_llm(client):
    # A tiny local model can hallucinate on "wie spät ist es" (observed live:
    # answered with London weather instead of the time). Time queries must
    # bypass the LLM entirely rather than rely on it reading the injected
    # time context correctly.
    from jarvis.llm import FakeLlm

    class TrapLlm(FakeLlm):
        pass

    client.app.state.llm = TrapLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat", json={"message": "wie spät ist es"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "TRAP" not in reply
    assert _TIME_PATTERN.search(reply)


def test_voice_answers_time_query_deterministically_without_llm(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="wie spät ist es")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "TRAP" not in reply
    assert _TIME_PATTERN.search(reply)


@pytest.mark.parametrize(
    "text,expected_seconds",
    [
        ("timer auf 5 minuten", 300),
        ("stell einen timer auf 90 sekunden", 90),
        ("timer auf 1 stunde 30 minuten", 5400),
        ("timer auf zehn minuten", 600),
        ("timer auf fünf minuten", 300),
    ],
)
def test_parse_timer_seconds_parses_durations(client, text, expected_seconds):
    from jarvis.remote_api import parse_timer_seconds

    assert parse_timer_seconds(text) == expected_seconds


def test_parse_timer_seconds_returns_none_without_a_duration(client):
    from jarvis.remote_api import parse_timer_seconds

    assert parse_timer_seconds("timer") is None


@pytest.mark.parametrize("text", ["timer", "stell einen timer", "Timer auf 5 Minuten"])
def test_is_timer_query_matches_timer_mentions(client, text):
    from jarvis.remote_api import is_timer_query

    assert is_timer_query(text) is True


@pytest.mark.parametrize("text", ["wie spät ist es", "baue einen taschenrechner", "hallo"])
def test_is_timer_query_does_not_match_other_questions(client, text):
    from jarvis.remote_api import is_timer_query

    assert is_timer_query(text) is False


def test_chat_timer_with_duration_sets_timer_seconds_without_llm(client):
    from jarvis.llm import FakeLlm

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat",
        json={"message": "timer auf 5 minuten"},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["timer_seconds"] == 300


def test_chat_timer_without_duration_asks_for_one(client):
    from jarvis.llm import FakeLlm

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat", json={"message": "timer"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["timer_seconds"] is None


def test_voice_timer_with_duration_sets_timer_seconds_without_llm(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="timer auf 2 minuten")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["timer_seconds"] == 120


def test_chat_beadchip_exact_match_bypasses_llm(client):
    from jarvis.llm import FakeLlm
    from jarvis.qa_beadchip import BEADCHIP_QA
    from jarvis.qa_lookup import best_match

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat",
        json={"message": "was ist ein illumina beadchip chip"},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "TRAP" not in reply
    assert reply == best_match("was ist ein illumina beadchip chip", BEADCHIP_QA)


def test_chat_bare_reagent_abbreviation_bypasses_llm(client):
    from jarvis.llm import FakeLlm
    from jarvis.qa_beadchip import reagent_answer

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat", json={"message": "ra1"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "TRAP" not in reply
    assert reply == reagent_answer("ra1")


def test_chat_loose_beadchip_question_injects_context_for_llm(client):
    from jarvis.llm import LlmEvent

    captured_messages = []

    class CapturingLlm:
        async def chat(self, messages):
            captured_messages.append(messages)
            yield LlmEvent(kind="done", text="synthesized answer")

    client.app.state.llm = CapturingLlm()
    resp = client.post(
        "/chat",
        json={"message": "erzähl mir was über den beadchip workflow allgemein"},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "synthesized answer"
    assert len(captured_messages) == 1
    system_texts = [m["content"] for m in captured_messages[0] if m.get("role") == "system"]
    assert any("BeadChip" in t for t in system_texts)


def test_chat_unrelated_question_gets_no_beadchip_context(client):
    from jarvis.llm import LlmEvent

    captured_messages = []

    class CapturingLlm:
        async def chat(self, messages):
            captured_messages.append(messages)
            yield LlmEvent(kind="done", text="ok")

    client.app.state.llm = CapturingLlm()
    resp = client.post(
        "/chat", json={"message": "wie geht es dir"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    system_texts = [m["content"] for m in captured_messages[0] if m.get("role") == "system"]
    assert not any("BeadChip" in t for t in system_texts)


def test_voice_beadchip_exact_match_bypasses_llm(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.qa_beadchip import BEADCHIP_QA
    from jarvis.qa_lookup import best_match
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="was ist ein illumina beadchip chip")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["reply"] == best_match("was ist ein illumina beadchip chip", BEADCHIP_QA)


def test_voice_bare_reagent_abbreviation_bypasses_llm(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.qa_beadchip import reagent_answer
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="ra1")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["reply"] == reagent_answer("ra1")


@pytest.mark.parametrize(
    "text,expected_name",
    [
        ("timer auf 5 minuten und nenne ihn chip washing", "chip washing"),
        ("stell einen timer auf 5 minuten und benenn ihn chip washing", "chip washing"),
        ("timer auf 5 minuten und bennen ihn chip washing", "chip washing"),  # user's typo
        ("timer auf 10 minuten nenn ihn pause", "pause"),
        ("timer auf 5 minuten", None),
    ],
)
def test_parse_timer_name(client, text, expected_name):
    from jarvis.remote_api import parse_timer_name

    assert parse_timer_name(text) == expected_name


def test_chat_timer_with_name_includes_it_in_reply_and_response(client):
    from jarvis.llm import FakeLlm

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat",
        json={"message": "timer auf 5 minuten und nenne ihn chip washing"},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "TRAP" not in body["reply"]
    assert body["timer_seconds"] == 300
    assert body["timer_name"] == "chip washing"
    assert "chip washing" in body["reply"].lower()


def test_voice_timer_with_name_includes_it_in_response(client, fixtures_dir):
    from jarvis.llm import FakeLlm
    from jarvis.stt import FakeStt

    client.app.state.stt = FakeStt(text="timer auf 2 minuten und nenne ihn pause")
    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": "test-secret"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["timer_seconds"] == 120
    assert body["timer_name"] == "pause"


def test_session_token_requires_real_master_key(client):
    resp = client.post("/session-token", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_session_token_mint_returns_usable_token(client):
    resp = client.post("/session-token", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200
    token = resp.json()["token"]
    assert token

    chat_resp = client.post(
        "/chat", json={"message": "hi"}, headers={"X-API-Key": token}
    )
    assert chat_resp.status_code == 200


def test_session_token_cannot_mint_another_token(client):
    # A leaked session token must not be able to mint itself fresh ones
    # forever — that would defeat the whole point of it expiring.
    resp = client.post("/session-token", headers={"X-API-Key": "test-secret"})
    token = resp.json()["token"]

    resp2 = client.post("/session-token", headers={"X-API-Key": token})
    assert resp2.status_code == 401


def test_voice_works_with_session_token_instead_of_master_key(client, fixtures_dir):
    token = client.post(
        "/session-token", headers={"X-API-Key": "test-secret"}
    ).json()["token"]
    wav_bytes = (fixtures_dir / "sine_440.wav").read_bytes()
    resp = client.post(
        "/voice",
        files={"audio": ("rec.wav", wav_bytes, "audio/wav")},
        headers={"X-API-Key": token},
    )
    assert resp.status_code == 200


@pytest.mark.parametrize("text", ["cool down", "starte cool down", "starte coool down"])
def test_is_cooldown_trigger_matches(client, text):
    from jarvis.remote_api import is_cooldown_trigger

    assert is_cooldown_trigger(text) is True


@pytest.mark.parametrize("text", ["wie spät ist es", "chip bath", "cooler"])
def test_is_cooldown_trigger_does_not_match_other_text(client, text):
    from jarvis.remote_api import is_cooldown_trigger

    assert is_cooldown_trigger(text) is False


@pytest.mark.parametrize("text", ["weiter", "Weiter.", "  weiter  "])
def test_is_continue_query_matches(client, text):
    from jarvis.remote_api import is_continue_query

    assert is_continue_query(text) is True


@pytest.mark.parametrize("text", ["weiter so gehts nicht", "immer weiter", "hallo"])
def test_is_continue_query_does_not_match_other_text(client, text):
    from jarvis.remote_api import is_continue_query

    assert is_continue_query(text) is False


def test_chat_cooldown_trigger_bypasses_llm(client):
    from jarvis.llm import FakeLlm

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat", json={"message": "starte cool down"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    assert "TRAP" not in resp.json()["reply"]


def test_chat_continue_query_bypasses_llm(client):
    from jarvis.llm import FakeLlm

    client.app.state.llm = FakeLlm(["TRAP: should never be used"])
    resp = client.post(
        "/chat", json={"message": "weiter"}, headers={"X-API-Key": "test-secret"}
    )
    assert resp.status_code == 200
    assert "TRAP" not in resp.json()["reply"]


# --- Browser games (games.py) -----------------------------------------

_VALID_GAME_HTML = (
    "<html><body><canvas id='c'></canvas>"
    "<script>function tick(){requestAnimationFrame(tick);}tick();</script>"
    "</body></html>"
)


def test_generate_game_requires_api_key(client):
    resp = client.post("/games/tetris/generate")
    assert resp.status_code == 401


def test_generate_game_rejects_unknown_game(client):
    resp = client.post("/games/not-a-game/generate", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 404


def test_generate_game_returns_generating_immediately(client, monkeypatch, tmp_path):
    from jarvis import remote_api
    from jarvis.llm import FakeLlm

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    client.app.state.game_llm = FakeLlm(["```html\n" + _VALID_GAME_HTML + "\n```"])
    resp = client.post("/games/tetris/generate", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "generating"


async def test_generate_game_writes_html_on_success(monkeypatch, tmp_path):
    from jarvis import remote_api
    from jarvis.llm import FakeLlm

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    remote_api.app.state.game_llm = FakeLlm(["```html\n" + _VALID_GAME_HTML + "\n```"])
    await remote_api._generate_game("tetris")
    assert (tmp_path / "tetris.html").read_text(encoding="utf-8") == _VALID_GAME_HTML
    assert remote_api._read_progress("tetris")["status"] == "done"


async def test_generate_game_records_error_when_both_tiers_fail(monkeypatch, tmp_path):
    from jarvis import remote_api
    from jarvis.llm import FakeLlm

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    # Both the normal attempt AND the escalation fallback must be mocked
    # — otherwise a failed first attempt would escalate to the real
    # app.state.game_llm_xlarge (built against real Ollama at import
    # time) and this "unit" test would make an actual network call.
    remote_api.app.state.game_llm = FakeLlm(["kein code hier, nur text"])
    remote_api.app.state.game_llm_xlarge = FakeLlm(["kein code hier, nur text"])
    await remote_api._generate_game("tetris")
    assert not (tmp_path / "tetris.html").exists()
    assert remote_api._read_progress("tetris")["status"] == "error"
    # Fixed error code, never the raw model output/exception text — the
    # progress file is readable via /games/{game}/status by anyone
    # holding a (possibly leaked) session token.
    assert remote_api._read_progress("tetris")["error"] == "generation_failed_smoke_check"


async def test_generate_game_escalates_to_xlarge_after_smoke_check_failure(monkeypatch, tmp_path):
    from jarvis import remote_api
    from jarvis.llm import FakeLlm

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    # First (large) attempt crashes on load (same bug class as the real
    # one that shipped): canvas.width isn't evenly divisible by the
    # block size, so Array(10.667) throws immediately.
    broken_game = (
        "<html><body><canvas id='c' width='320' height='640'></canvas>"
        "<script>const g = Array(320 / 30);</script></body></html>"
    )
    remote_api.app.state.game_llm = FakeLlm(["```html\n" + broken_game + "\n```"])
    remote_api.app.state.game_llm_xlarge = FakeLlm(["```html\n" + _VALID_GAME_HTML + "\n```"])
    await remote_api._generate_game("tetris")
    assert (tmp_path / "tetris.html").read_text(encoding="utf-8") == _VALID_GAME_HTML
    assert remote_api._read_progress("tetris")["status"] == "done"


async def test_generate_game_clears_in_flight_flag_when_done(monkeypatch, tmp_path):
    from jarvis import remote_api
    from jarvis.llm import FakeLlm

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    remote_api.app.state.game_llm = FakeLlm(["```html\n" + _VALID_GAME_HTML + "\n```"])
    remote_api.app.state.game_in_flight.add("tetris")
    await remote_api._generate_game("tetris")
    assert "tetris" not in remote_api.app.state.game_in_flight


async def test_generate_game_clears_in_flight_flag_on_error(monkeypatch, tmp_path):
    from jarvis import remote_api
    from jarvis.llm import FakeLlm

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    remote_api.app.state.game_llm = FakeLlm(["kein code hier, nur text"])
    remote_api.app.state.game_llm_xlarge = FakeLlm(["kein code hier, nur text"])
    remote_api.app.state.game_in_flight.add("tetris")
    await remote_api._generate_game("tetris")
    assert "tetris" not in remote_api.app.state.game_in_flight


def test_generate_game_short_circuits_when_already_in_flight(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    remote_api.app.state.game_in_flight.add("tetris")
    try:
        resp = client.post("/games/tetris/generate", headers={"X-API-Key": "test-secret"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "generating"
    finally:
        remote_api.app.state.game_in_flight.discard("tetris")


def test_game_status_requires_api_key(client):
    resp = client.get("/games/tetris/status")
    assert resp.status_code == 401


def test_game_status_is_idle_when_never_generated(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    resp = client.get("/games/tetris/status", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"


def test_game_status_is_done_when_html_already_exists(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "pacman.html").write_text(_VALID_GAME_HTML, encoding="utf-8")
    resp = client.get("/games/pacman/status", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"


def test_mint_game_ticket_requires_api_key(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "aliens.html").write_text(_VALID_GAME_HTML, encoding="utf-8")
    resp = client.post("/games/aliens/ticket")
    assert resp.status_code == 401


def test_mint_game_ticket_404_when_not_generated_yet(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    resp = client.post("/games/aliens/ticket", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 404


def test_mint_game_ticket_returns_a_ticket_once_ready(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "aliens.html").write_text(_VALID_GAME_HTML, encoding="utf-8")
    resp = client.post("/games/aliens/ticket", headers={"X-API-Key": "test-secret"})
    assert resp.status_code == 200
    assert len(resp.json()["ticket"]) > 20


def test_serve_game_requires_a_ticket(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "aliens.html").write_text(_VALID_GAME_HTML, encoding="utf-8")
    assert client.get("/games/aliens").status_code == 401
    assert client.get("/games/aliens", params={"ticket": "garbage"}).status_code == 401


def test_serve_game_returns_html_with_valid_ticket(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "aliens.html").write_text(_VALID_GAME_HTML, encoding="utf-8")
    ticket = client.post(
        "/games/aliens/ticket", headers={"X-API-Key": "test-secret"}
    ).json()["ticket"]
    resp = client.get("/games/aliens", params={"ticket": ticket})
    assert resp.status_code == 200
    assert resp.text == _VALID_GAME_HTML
    assert resp.headers["content-security-policy"] == "sandbox allow-scripts"
    assert resp.headers["referrer-policy"] == "no-referrer"


def test_serve_game_ticket_is_single_use(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "aliens.html").write_text(_VALID_GAME_HTML, encoding="utf-8")
    ticket = client.post(
        "/games/aliens/ticket", headers={"X-API-Key": "test-secret"}
    ).json()["ticket"]
    first = client.get("/games/aliens", params={"ticket": ticket})
    second = client.get("/games/aliens", params={"ticket": ticket})
    assert first.status_code == 200
    assert second.status_code == 401


def test_serve_game_ticket_is_scoped_to_its_own_game(client, monkeypatch, tmp_path):
    from jarvis import remote_api

    monkeypatch.setattr(remote_api.settings, "games_dir", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "aliens.html").write_text(_VALID_GAME_HTML, encoding="utf-8")
    (tmp_path / "pacman.html").write_text(_VALID_GAME_HTML, encoding="utf-8")
    ticket = client.post(
        "/games/aliens/ticket", headers={"X-API-Key": "test-secret"}
    ).json()["ticket"]
    resp = client.get("/games/pacman", params={"ticket": ticket})
    assert resp.status_code == 401
