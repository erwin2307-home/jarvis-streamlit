"""Background code-generation + execution.

A dedicated small coding model (separate from the voice-response model)
writes a Python script on request; the script is saved and launched in
its own console window. Generation + execution run as a detached
background task (see orchestrator.py) so the normal voice turn completes
immediately and isn't blocked waiting on it.

Auto-run, no confirmation step — explicit user choice, accepted the
"Jarvis runs generated code on your real machine with your own user
privileges, no sandbox" trade-off in exchange for speed. The only safety
floor kept regardless: scripts land in one dedicated folder (not
scattered into arbitrary paths), filenames are generated (never taken
from model output or user speech), and nothing runs elevated.
"""
from __future__ import annotations

import ast
import re
import uuid
from pathlib import Path

# Each pattern requires an imperative VERB form, not just a bare noun —
# "Programm"/"Skript" alone are common German words unrelated to code
# generation ("dieses Programm kenne ich nicht", "das TV-Programm").
# This feeds straight into unsandboxed execution, so the trigger is
# precise rather than coarse: a false negative just means "say it more
# directly", a false positive means running whatever the model wrote.
_CODING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bprogrammier\w*\b",
        r"\bschreib\w*\s+(mir\s+)?(ein\w*\s+)?(python|skript|script|programm|code)\w*\b",
        r"\berstell\w*\s+(mir\s+)?(ein\w*\s+)?(python|skript|script|programm|tool)\w*\b",
        r"\bcod(?:e|ier)\w*\s+(?:mir|dafür|dafuer)\b",
        r"\bbau\w*\s+(mir\s+)?(ein\w*\s+)?tool\b",
    )
)

# Fence must start at a line boundary with an empty or exactly "python"
# tag (not e.g. ```bash or ```text), rejecting a parser differential
# where non-Python content gets extracted and executed anyway.
_CODE_FENCE_PATTERN = re.compile(r"(?m)^```(?:python)?[ \t]*\r?\n(.*?)^```", re.DOTALL)


def is_coding_request(text: str) -> bool:
    """Precise detection — this feeds a code-execution sink, so it's
    deliberately NOT the coarse substring-match style used for the
    (harmless, answer-only) topic detectors elsewhere in the codebase."""
    return any(p.search(text) for p in _CODING_PATTERNS)


def extract_python_code(llm_response: str) -> str | None:
    """Pull the code out of a single ```python ... ``` (or bare ```...```)
    fenced block. Returns None — treated as a generation failure, never
    run — when: there's no fence, there's more than one fenced block (no
    guessing which one is "the" script), or the extracted text doesn't
    parse as valid Python (catches non-code content slipping through a
    coincidental fence match)."""
    matches: list[str] = _CODE_FENCE_PATTERN.findall(llm_response)
    if len(matches) != 1:
        return None
    code: str = matches[0].strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    return code


def safe_script_filename() -> str:
    """Generated filename, never derived from model output or user
    speech — avoids any path-traversal or injection surface from
    untrusted text ending up in a filesystem path."""
    return f"jarvis_script_{uuid.uuid4().hex[:12]}.py"


def save_script(code: str, scripts_dir: Path) -> Path:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / safe_script_filename()
    path.write_text(code, encoding="utf-8")
    return path


def run_script_in_new_window(script_path: Path) -> int:
    """Launch the script in its own visible console window (so stdout/
    stderr and any GUI the script opens are visible), detached — doesn't
    wait for it to finish. Returns the child process PID.

    Lower-than-normal process priority so a CPU-heavy generated script
    doesn't starve the voice pipeline's own process for CPU on this
    machine's limited cores — asyncio-level "background" alone doesn't
    prevent OS-level CPU contention between separate processes."""
    import subprocess
    import sys

    creationflags = (
        subprocess.CREATE_NEW_CONSOLE | subprocess.BELOW_NORMAL_PRIORITY_CLASS
    )
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        creationflags=creationflags,
    )
    return proc.pid
