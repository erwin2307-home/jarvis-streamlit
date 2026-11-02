"""Tests for the background code-generation + execution feature."""
from jarvis.coding import (
    extract_python_code,
    is_coding_request,
    safe_script_filename,
    save_script,
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
