"""Tests for the browser-game generation prompts + extraction."""
import shutil
import subprocess

import pytest

from jarvis.games import (
    GAME_IDS,
    build_smoke_harness,
    extract_html_game,
    extract_script_body,
    game_prompt,
)

_HAS_NODE = shutil.which("node") is not None
_requires_node = pytest.mark.skipif(not _HAS_NODE, reason="node not on PATH")


def test_game_ids_are_the_four_requested_games() -> None:
    assert set(GAME_IDS) == {"tetris", "pacman", "aliens", "halo"}


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


def test_halo_prompt_scopes_down_to_2d_not_3d() -> None:
    prompt = game_prompt("halo")
    assert "2D" in prompt
    assert "KEIN 3D" in prompt


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


# --- Smoke-check harness (actually runs the generated JS, sandboxed) --


def test_extract_script_body_pulls_single_script_tag() -> None:
    html = "<html><body><canvas></canvas><script>let x = 1;</script></body></html>"
    assert extract_script_body(html) == "let x = 1;"


def test_extract_script_body_concatenates_multiple_script_tags() -> None:
    html = (
        "<html><body><canvas></canvas>"
        "<script>let a = 1;</script>"
        "<script>let b = 2;</script>"
        "</body></html>"
    )
    body = extract_script_body(html)
    assert body is not None
    assert "let a = 1;" in body
    assert "let b = 2;" in body


def test_extract_script_body_returns_none_without_any_script_tag() -> None:
    assert extract_script_body("<html><body><canvas></canvas></body></html>") is None


def test_build_smoke_harness_embeds_script_as_safe_json_string() -> None:
    harness = build_smoke_harness('alert("quotes \' and \\\\ backslashes");')
    assert '\\"quotes' in harness or "quotes" in harness  # survived json.dumps escaping
    assert "require(\"vm\")" in harness


@_requires_node
def test_smoke_check_passes_for_a_working_script(tmp_path) -> None:
    script = (
        "let ticks = 0;\n"
        "function tick() { ticks++; if (ticks < 5) setTimeout(tick, 1); }\n"
        "tick();\n"
    )
    harness_path = tmp_path / "harness.js"
    harness_path.write_text(build_smoke_harness(script), encoding="utf-8")
    result = subprocess.run(["node", str(harness_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0
    assert "SMOKE_OK" in result.stdout


@_requires_node
def test_smoke_check_fails_on_invalid_array_length(tmp_path) -> None:
    # The real bug that shipped: canvas dimensions not evenly divisible
    # by the block size, so `Array(10.667)` throws RangeError on load,
    # before the game ever does anything — exactly what this harness is
    # for catching automatically instead of a human spotting "nothing
    # happens" in a real browser.
    script = "const grid = Array(320 / 30);\n"
    harness_path = tmp_path / "harness.js"
    harness_path.write_text(build_smoke_harness(script), encoding="utf-8")
    result = subprocess.run(["node", str(harness_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "SMOKE_FAIL" in result.stderr


@_requires_node
def test_smoke_check_fails_on_crash_during_simulated_game_loop(tmp_path) -> None:
    # Catches a bug that only manifests once the loop actually runs a
    # few ticks (the harness drains queued setTimeout/rAF callbacks),
    # not just at parse/first-line time.
    script = (
        "let obj = null;\n"
        "function tick() { obj.missing(); }\n"
        "setTimeout(tick, 1);\n"
    )
    harness_path = tmp_path / "harness.js"
    harness_path.write_text(build_smoke_harness(script), encoding="utf-8")
    result = subprocess.run(["node", str(harness_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "SMOKE_FAIL" in result.stderr


@_requires_node
def test_smoke_check_sandbox_has_no_require_or_process_access(tmp_path) -> None:
    # The whole point of running this in a vm context instead of plain
    # node: the generated script itself must not be able to reach
    # require/process/filesystem, even though the harness around it
    # (outside the sandbox) does.
    script = "require('fs').readFileSync('/etc/hosts');"
    harness_path = tmp_path / "harness.js"
    harness_path.write_text(build_smoke_harness(script), encoding="utf-8")
    result = subprocess.run(["node", str(harness_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert "SMOKE_FAIL" in result.stderr
    assert "not defined" in result.stderr  # ReferenceError: require is not defined


@_requires_node
def test_smoke_check_sandbox_blocks_constructor_chain_escape(tmp_path) -> None:
    # The specific documented Node `vm` escape: a HOST object's
    # `.constructor.constructor` resolves to the host realm's Function
    # constructor, letting sandboxed code mint a function that runs
    # with host privileges (e.g. to reach `process`/fs/network). This
    # only works if a host-realm object crosses into the sandbox — the
    # fix (vm.createContext() with nothing injected, every stub built
    # from source text run inside the context) means even a plain
    # object literal created *inside* the sandbox has the *sandboxed*
    # realm's Function as its constructor, not this host process's.
    script = (
        "var hostProcess = ({}).constructor.constructor('return process')();\n"
        "hostProcess.exit(0);\n"  # if this ever runs, it kills THIS test's node process with code 0
    )
    harness_path = tmp_path / "harness.js"
    harness_path.write_text(build_smoke_harness(script), encoding="utf-8")
    result = subprocess.run(["node", str(harness_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0, (
        "escape succeeded — generated code reached the host `process` object"
    )
    assert "SMOKE_FAIL" in result.stderr


@_requires_node
def test_smoke_check_sandbox_blocks_the_exact_proven_console_escape(tmp_path) -> None:
    # This is the exploit confirmed to actually work against the
    # earlier, vulnerable version of this harness (which passed the
    # real host `console` object into the sandbox): run against that
    # old pattern directly, `console.constructor.constructor('return
    # process')().exit(42)` terminates the real host process with code
    # 42 — proven live, not theoretical. The fixed harness's `console`
    # is a fresh object built entirely from bootstrap source text run
    # inside the sandboxed context, so its constructor chain never
    # leaves that context.
    script = "console.constructor.constructor('return process')().exit(42);"
    harness_path = tmp_path / "harness.js"
    harness_path.write_text(build_smoke_harness(script), encoding="utf-8")
    result = subprocess.run(["node", str(harness_path)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 42, "escape succeeded — reached and called host process.exit"
    assert result.returncode != 0
    assert "SMOKE_FAIL" in result.stderr
