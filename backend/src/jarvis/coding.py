"""Background code-generation + execution.

A dedicated coding model (separate from the voice-response model) writes
a Python script on request; the script is saved and launched in its own
console window. Generation + execution run as a detached background task
(see orchestrator.py) so the normal voice turn completes immediately and
isn't blocked waiting on it.

Three model tiers (small/medium/large, see config.py's coding_model_*)
let a quick trigger ("größeres Modell", "code war schlecht" — see
is_coding_upgrade_request) re-run the last request one tier up instead
of generating anything new. The console window for a request opens
immediately (before generation starts) running a small fixed launcher
script that polls a per-request progress file (write_progress /
progress_path_for) and renders a text progress bar, then runs the
finished script in that same window once generation succeeds.

Auto-run, no confirmation step — explicit user choice, accepted the
"Jarvis runs generated code on your real machine with your own user
privileges, no sandbox" trade-off in exchange for speed. The only safety
floor kept regardless: scripts land in one dedicated folder (not
scattered into arbitrary paths), filenames are generated (never taken
from model output or user speech), and nothing runs elevated.
"""
from __future__ import annotations

import ast
import json
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


# Re-runs the LAST coding prompt one tier up (see TIER_ORDER) instead of
# generating anything new — deliberately requires "modell"/"code"/
# "skript" alongside the comparative word so idle chatter containing
# "besser" alone doesn't silently re-trigger a code-execution sink.
_UPGRADE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(gr(ö|oe)(ß|ss)er\w*|besser\w*)\s+modell\b",
        r"\bmodell\s+(wechseln|upgrade\w*|hochstufen)\b",
        r"\b(code|skript|script)\s+war\s+schlecht\b",
        r"\bmach\s+(das|den\s+code)\s+besser\b",
        r"\bverbesser\w*\s+den\s+code\b",
    )
)


def is_coding_upgrade_request(text: str) -> bool:
    """True for "nimm das größere Modell" / "code war schlecht" style
    requests to re-generate the last coding request with a bigger model."""
    return any(p.search(text) for p in _UPGRADE_PATTERNS)


TIER_ORDER: tuple[str, ...] = ("small", "medium", "large")


def next_tier(current: str) -> str:
    """One step up TIER_ORDER, clamped at "large" (including when
    `current` is already "large" or isn't a recognized tier name)."""
    idx = TIER_ORDER.index(current) if current in TIER_ORDER else 0
    return TIER_ORDER[min(idx + 1, len(TIER_ORDER) - 1)]


def decide_coding_action(
    text: str, current_tier: str, last_prompt: str | None, default_tier: str
) -> tuple[str, str] | None:
    """Pure decision for handle_coding_request: given the incoming
    transcript and the current coding_state, returns (prompt_to_run,
    tier_to_run) for the caller to dispatch, or None if `text` is neither
    a coding request nor an upgrade request (caller falls through to the
    normal QA/LLM path). Does not mutate anything — the caller is
    responsible for writing the returned tier/prompt back into its own
    state dict before dispatching, same as it already did inline."""
    if is_coding_upgrade_request(text):
        if last_prompt is None:
            return None  # nothing to re-run yet
        return last_prompt, next_tier(current_tier)
    if not is_coding_request(text):
        return None
    return text, default_tier


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


def new_id() -> str:
    """Generated id, never derived from model output or user speech —
    avoids any path-traversal or injection surface from untrusted text
    ending up in a filesystem path."""
    return uuid.uuid4().hex[:12]


def safe_script_filename() -> str:
    return f"jarvis_script_{new_id()}.py"


def save_script(code: str, scripts_dir: Path) -> Path:
    scripts_dir.mkdir(parents=True, exist_ok=True)
    path = scripts_dir / safe_script_filename()
    path.write_text(code, encoding="utf-8")
    return path


def progress_path_for(script_id: str, scripts_dir: Path) -> Path:
    return scripts_dir / "_progress" / f"{script_id}.json"


def write_progress(path: Path, **fields: object) -> None:
    """Status the launcher window polls while generation runs in the
    background task. Written via a temp-file-then-replace so the
    launcher never reads a half-written JSON file mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(fields), encoding="utf-8")
    tmp.replace(path)


# Fixed template, not model output — only the progress-file path is
# substituted, as a json.dumps()'d string (valid, safely-escaped Python
# string literal syntax too) so Windows backslashes survive intact.
# Opened immediately when a coding request is detected, before
# generation starts, so the visible console window appears right away
# instead of only once the script is ready; shows a text progress bar
# driven by the background task's token count against coding_max_tokens
# (approximate — Ollama's stream gives text chunks, not a token count,
# so this is chars-so-far/4 as a rough per-token estimate), then runs
# the finished script in the same window once generation succeeds.
_LAUNCHER_TEMPLATE = '''"""Auto-generated launcher: shows progress while Jeanny's background
coding task generates code, then runs the result in this console
window. Template is fixed; only the progress-file path is substituted.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PROGRESS_PATH = Path(__PROGRESS_PATH__)
BAR_WIDTH = 30


def render(pct, label):
    filled = int(BAR_WIDTH * pct)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    print("\\r[" + bar + "] " + str(int(pct * 100)).rjust(3) + "%  " + label, end="", flush=True)


print("Jeanny generiert Code ...")
while True:
    try:
        data = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        time.sleep(0.2)
        continue
    status = data.get("status")
    if status == "generating":
        tokens = data.get("tokens", 0)
        max_tokens = max(data.get("max_tokens", 1), 1)
        render(min(tokens / max_tokens, 0.98), str(tokens) + " Tokens (" + str(data.get("model", "?")) + ")")
    elif status == "done":
        render(1.0, "fertig")
        print()
        script_path = data.get("script_path")
        if script_path:
            # Resolve to absolute BEFORE handing it to subprocess's own
            # cwd= — a relative script_path re-joined against a cwd
            # that's itself inside that relative path doubles up the
            # directory segment instead of pointing at the file.
            resolved = Path(script_path).resolve()
            subprocess.run([sys.executable, str(resolved)], cwd=str(resolved.parent))
        break
    elif status == "error":
        print()
        print("Fehler:", data.get("error", "unbekannt"))
        break
    time.sleep(0.2)
input("\\n[Enter zum Schliessen]")
'''


def save_launcher(progress_path: Path, scripts_dir: Path, script_id: str) -> Path:
    path = scripts_dir / "_launchers" / f"jarvis_launcher_{script_id}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = _LAUNCHER_TEMPLATE.replace("__PROGRESS_PATH__", json.dumps(str(progress_path)))
    path.write_text(source, encoding="utf-8")
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
