"""Tests for the browser-game generation prompts + extraction."""
import pytest

from jarvis.games import GAME_IDS, extract_html_game, game_prompt


def test_game_ids_are_the_three_requested_games() -> None:
    assert set(GAME_IDS) == {"tetris", "pacman", "aliens"}


def test_game_prompt_returns_nonempty_text_for_every_game() -> None:
    for game in GAME_IDS:
        assert len(game_prompt(game)) > 20


def test_game_prompt_raises_for_unknown_game() -> None:
    with pytest.raises(KeyError):
        game_prompt("not-a-game")


def test_tetris_prompt_mentions_20_levels() -> None:
    assert "20" in game_prompt("tetris")


def test_pacman_prompt_mentions_ghosts() -> None:
    assert "Geister" in game_prompt("pacman")


def test_aliens_prompt_mentions_spaceship_and_aliens() -> None:
    prompt = game_prompt("aliens")
    assert "Raumschiff" in prompt
    assert "Alien" in prompt


def _wrap(body: str, fence: str = "```html") -> str:
    return f"Hier ist das Spiel:\n\n{fence}\n{body}\n```\n\nViel Spass!"


_VALID_GAME = (
    "<html><head><style>body{margin:0}</style></head>"
    "<body><canvas id='c'></canvas>"
    "<script>function tick(){requestAnimationFrame(tick);}tick();</script>"
    "</body></html>"
)


def test_extract_html_game_from_fenced_block() -> None:
    assert extract_html_game(_wrap(_VALID_GAME)) == _VALID_GAME


def test_extract_html_game_from_bare_fence() -> None:
    assert extract_html_game(_wrap(_VALID_GAME, fence="```")) == _VALID_GAME


def test_extract_html_game_returns_none_when_no_fence() -> None:
    assert extract_html_game("Ich kann das nicht bauen.") is None


def test_extract_html_game_rejects_multiple_fenced_blocks() -> None:
    response = (
        "```html\n" + _VALID_GAME + "\n```\n\nOder:\n\n```html\n" + _VALID_GAME + "\n```"
    )
    assert extract_html_game(response) is None


def test_extract_html_game_rejects_missing_canvas() -> None:
    body = "<html><body><script>alert(1)</script></body></html>"
    assert extract_html_game(_wrap(body)) is None


def test_extract_html_game_rejects_missing_script() -> None:
    body = "<html><body><canvas id='c'></canvas></body></html>"
    assert extract_html_game(_wrap(body)) is None


def test_extract_html_game_rejects_unbalanced_braces_as_truncated() -> None:
    # Simulates generation cut off mid-script by hitting max_tokens —
    # the single cheap truncation signal available without a JS parser.
    truncated = (
        "<html><body><canvas id='c'></canvas>"
        "<script>function tick(){requestAnimationFrame(tick);"
        "</body></html>"
    )
    assert extract_html_game(_wrap(truncated)) is None


def test_extract_html_game_rejects_invalid_syntax_fence_tag() -> None:
    # Fence must be bare or exactly ```html, same guard as coding.py's
    # extract_python_code — rejects e.g. ```javascript slipping through.
    response = "```javascript\n" + _VALID_GAME + "\n```"
    assert extract_html_game(response) is None
