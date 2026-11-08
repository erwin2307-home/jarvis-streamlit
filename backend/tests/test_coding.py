"""Tests for the background code-generation + execution feature."""
import json

from jarvis.coding import (
    TIER_ORDER,
    decide_coding_action,
    extract_python_code,
    is_coding_request,
    is_coding_upgrade_request,
    new_id,
    next_tier,
    progress_path_for,
    safe_script_filename,
    save_launcher,
    save_script,
    write_progress,
)


def test_detects_direct_programming_request() -> None:
    assert is_coding_request("programmiere mir einen taschenrechner")


def test_detects_various_phrasings() -> None:
    for q in (
        "schreib mir ein python skript das würfelt",
        "erstelle ein programm mit einem fenster",
        "baue mir ein tool dafür",
        "kannst du mir code dafür schreiben",
    ):
        assert is_coding_request(q), f"not detected: {q!r}"


def test_unrelated_question_is_not_a_coding_request() -> None:
    assert not is_coding_request("wie spät ist es")
    assert not is_coding_request("was ist ein illumina beadchip")


def test_bare_generic_noun_without_verb_is_not_a_coding_request() -> None:
    # Real false-positive risk: "Programm"/"Skript" alone are common
    # German words unrelated to code generation. Must require an
    # imperative verb, not just the noun appearing anywhere.
    for q in (
        "was läuft heute im tv programm",
        "dieses programm kenne ich nicht",
        "das skript von dem film war gut",
    ):
        assert not is_coding_request(q), f"false positive on: {q!r}"


def test_extract_python_code_from_fenced_block() -> None:
    response = (
        "Klar, hier ist ein kurzes Skript:\n\n"
        "```python\n"
        "print('hallo')\n"
        "```\n\n"
        "Das gibt 'hallo' aus."
    )
    assert extract_python_code(response) == "print('hallo')"


def test_extract_python_code_from_bare_fence() -> None:
    response = "```\nprint(1 + 1)\n```"
    assert extract_python_code(response) == "print(1 + 1)"


def test_extract_python_code_returns_none_when_no_fence() -> None:
    assert extract_python_code("Ich kann das nicht programmieren.") is None


def test_extract_python_code_rejects_invalid_syntax() -> None:
    # Parser-differential guard: don't save/run something that merely
    # looks like it's in a code fence but isn't valid Python.
    response = "```python\nprint('unterminated\n```"
    assert extract_python_code(response) is None


def test_extract_python_code_rejects_multiple_fenced_blocks() -> None:
    # If the model's response contains more than one fenced block, don't
    # guess which one is "the" script — treat it as a generation failure.
    response = (
        "```python\nprint('one')\n```\n\n"
        "Oder alternativ:\n\n"
        "```python\nprint('two')\n```"
    )
    assert extract_python_code(response) is None


def test_extract_python_code_accepts_valid_multiline_script() -> None:
    response = "```python\ndef f(x):\n    return x + 1\n\nprint(f(2))\n```"
    assert extract_python_code(response) == "def f(x):\n    return x + 1\n\nprint(f(2))"


def test_safe_script_filename_has_py_extension_and_is_unique() -> None:
    a = safe_script_filename()
    b = safe_script_filename()
    assert a.endswith(".py")
    assert b.endswith(".py")
    assert a != b  # timestamp/uuid-based, must not collide


def test_safe_script_filename_has_no_path_separators() -> None:
    name = safe_script_filename()
    assert "/" not in name and "\\" not in name and ".." not in name


def test_save_script_writes_file_in_dedicated_dir(tmp_path) -> None:
    path = save_script("print('hi')", tmp_path / "scripts")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "print('hi')"
    assert path.parent == tmp_path / "scripts"


def test_save_script_creates_dir_if_missing(tmp_path) -> None:
    target = tmp_path / "does_not_exist_yet"
    path = save_script("x = 1", target)
    assert path.exists()
    assert target.is_dir()


def test_detects_upgrade_request_phrasings() -> None:
    for q in (
        "nimm das größere Modell",
        "der Code war schlecht",
        "mach den Code besser",
        "verbessere den Code bitte",
        "Modell wechseln",
    ):
        assert is_coding_upgrade_request(q), f"not detected: {q!r}"


def test_unrelated_question_is_not_an_upgrade_request() -> None:
    assert not is_coding_upgrade_request("wie spät ist es")
    assert not is_coding_upgrade_request("das war besser als gestern")


def test_next_tier_steps_up_and_clamps_at_large() -> None:
    assert next_tier("small") == "medium"
    assert next_tier("medium") == "large"
    assert next_tier("large") == "large"


def test_next_tier_defaults_unknown_to_small_then_steps_up() -> None:
    assert next_tier("not-a-tier") == TIER_ORDER[1]


def test_new_id_is_unique_and_has_no_path_separators() -> None:
    a, b = new_id(), new_id()
    assert a != b
    assert "/" not in a and "\\" not in a


def test_write_progress_then_read_back(tmp_path) -> None:
    path = progress_path_for(new_id(), tmp_path)
    write_progress(path, status="generating", tokens=5, max_tokens=2000, model="qwen2.5-coder:1.5b")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "status": "generating",
        "tokens": 5,
        "max_tokens": 2000,
        "model": "qwen2.5-coder:1.5b",
    }
    # No leftover .tmp file after the atomic replace.
    assert not path.with_suffix(".json.tmp").exists()


def test_save_launcher_embeds_progress_path_safely(tmp_path) -> None:
    script_id = new_id()
    progress_path = progress_path_for(script_id, tmp_path)
    launcher_path = save_launcher(progress_path, tmp_path, script_id)
    assert launcher_path.exists()
    source = launcher_path.read_text(encoding="utf-8")
    assert json.dumps(str(progress_path)) in source
    compile(source, str(launcher_path), "exec")  # must be valid Python


def test_decide_new_coding_request_resets_to_default_tier() -> None:
    decision = decide_coding_action(
        "programmiere mir einen taschenrechner",
        current_tier="large",  # leftover from a previous, unrelated request
        last_prompt="ein altes prompt",
        default_tier="small",
    )
    assert decision == ("programmiere mir einen taschenrechner", "small")


def test_decide_upgrade_request_bumps_tier_and_keeps_last_prompt() -> None:
    decision = decide_coding_action(
        "nimm das größere Modell",
        current_tier="small",
        last_prompt="programmiere mir einen taschenrechner",
        default_tier="small",
    )
    assert decision == ("programmiere mir einen taschenrechner", "medium")


def test_decide_upgrade_request_with_no_prior_prompt_is_a_noop() -> None:
    assert (
        decide_coding_action(
            "nimm das größere Modell",
            current_tier="small",
            last_prompt=None,
            default_tier="small",
        )
        is None
    )


def test_decide_upgrade_request_already_at_large_stays_clamped() -> None:
    decision = decide_coding_action(
        "der Code war schlecht",
        current_tier="large",
        last_prompt="baue mir ein tool dafür",
        default_tier="small",
    )
    assert decision == ("baue mir ein tool dafür", "large")


def test_decide_unrelated_text_is_a_noop() -> None:
    assert (
        decide_coding_action(
            "wie spät ist es",
            current_tier="small",
            last_prompt=None,
            default_tier="small",
        )
        is None
    )
