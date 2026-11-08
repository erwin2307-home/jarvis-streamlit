"""Browser-playable arcade games, generated on request by the "large"
coding-model tier (see config.py's coding_model / coding.py's TIER_ORDER)
and served as a single self-contained HTML file.

Deliberately NOT the same pipeline as coding.py's Python-script fast-path
— that one auto-runs generated code with the user's own OS privileges on
their desktop (no sandbox, explicit trade-off, see coding.py's module
docstring). What actually PLAYS the game is plain client-side HTML/JS in
a browser tab, the same trust model as any website's JS — that part
never runs server-side, which is why serving a finished game is still
safe to expose on remote_api.py.

The smoke-check below (`build_smoke_harness`) is a narrow exception: it
DOES execute the model's generated JS once, server-side, before marking
a generation "done" — a cheap real-world catch for exactly the class of
bug that shipped in practice (a canvas-size/block-size mismatch that
threw on the very first line, so the "finished" game never did anything
at all; see remote_api.py's _run_smoke_check for how it's invoked and
escalated on failure).

Be clear about what this buys and doesn't: Node's `vm` module is
explicitly documented by Node itself as NOT a security mechanism, and
that's been proven true here across three rounds of security review,
not just a theoretical caveat — confirmed live: (1) an earlier version
passed real host objects (console, Math, ...) into the sandboxed
context, and `console.constructor.constructor("return process")()`
reached the real host `process` and exited the actual node process
with a chosen code; fixed by building every stub from source text run
*inside* a context created with nothing injected at all. (2) A second,
narrower vector survived that fix: a BARE top-level `this` in a
non-strict script run via `vm.runInContext` is still the context's own
global object, and `this.constructor.constructor("return process")()`
*still* reaches the host `process` — this is true even with
`vm.createContext()` given no argument object, and Node's own
documented stop-gap (`Object.setPrototypeOf(globalThis, null)`) was
tested here and does NOT block it; forcing strict mode on the
generated script doesn't help either (top-level `this` is the global
object in strict mode too, strict mode only changes `this` for
ordinary function calls without a receiver — verified, not assumed).

This residual gap is accepted, deliberately, for one concrete reason:
the text sent to the model (`game_prompt()` below) is a FIXED, server-
controlled constant — `/games/{game}/generate` takes no free-form
input from its caller, only a game id from a closed set of 4. A caller
cannot inject prompt content to coax the model toward writing escape
code; the only thing that varies is the model's own, non-adversarial
completion of a fixed "write me a canvas game" request. If this
pipeline is ever changed to accept caller-supplied prompt text, or to
run a model whose output isn't trusted this way, this smoke-check
would need real OS-level isolation (a container with no network and a
read-only filesystem, or `isolated-vm`'s separate V8 isolate) instead
of `vm` — do not treat the current hardening as sufficient for that.

Each game is requested as ONE complete, offline, self-contained HTML
document (inline <style>/<script>, no CDN/network calls — this project
is offline-first and a generated game shouldn't be the exception) with
canvas-based rendering and keyboard controls.
"""
from __future__ import annotations

import json
import re

GAME_IDS: tuple[str, ...] = ("tetris", "pacman", "aliens", "halo")

GAMES_SYSTEM_PROMPT = (
    "Du bist ein erfahrener JavaScript-Spieleentwickler. Antworte "
    "AUSSCHLIESSLICH mit einem einzigen vollständigen HTML-Dokument in "
    "einem einzigen ```html ... ``` Codeblock. Kein Fließtext davor oder "
    "danach. Das Dokument muss komplett eigenständig sein: <style> und "
    "<script> inline, kein CDN, kein externer Aufruf, kein fetch() ins "
    "Netz. Nutze ein <canvas> Element und steuere das Spiel per "
    "Tastatur (Pfeiltasten + ggf. Leertaste). Zeige Punktzahl und, falls "
    "zutreffend, Leben/Level sichtbar an. Bei Game Over: Text anzeigen "
    "und per Taste neu starten können, ohne die Seite neu zu laden."
)

_GAME_PROMPTS: dict[str, str] = {
    "tetris": (
        "Baue ein vollständiges Tetris. Standard-7 Tetromino-Formen, "
        "Rotation, Kollisionserkennung, Zeilen löschen und Punkte "
        "vergeben. 20 Level: die Fallgeschwindigkeit steigt mit jedem "
        "Level (z.B. alle 10 gelöschten Zeilen ein Level höher, bis "
        "Level 20, danach bleibt sie auf dem Level-20-Tempo). Zeige "
        "aktuelles Level, Punktzahl und eine Vorschau des nächsten "
        "Teils an. Steuerung: Pfeiltasten links/rechts/runter, Pfeil "
        "hoch zum Rotieren, Leertaste für Hard-Drop."
    ),
    "pacman": (
        "Baue einen Pac-Man-Klon mit einem festen Labyrinth (ein Level "
        "reicht), Pac-Man wird mit den Pfeiltasten gesteuert, sammelt "
        "Punkte (Pellets) im Labyrinth ein, 2 bis 4 Geister bewegen "
        "sich im Labyrinth (einfaches Verfolgungs- oder Zufalls-"
        "Verhalten reicht) und beenden das Spiel bei Kontakt mit "
        "Pac-Man (Leben abziehen, Neustart der Position; bei 0 Leben "
        "Game Over). Gewinn, wenn alle Pellets eingesammelt wurden. "
        "Punktzahl und verbleibende Leben anzeigen."
    ),
    "aliens": (
        "Baue einen klassischen Space-Invaders-Klon: ein Raumschiff am "
        "unteren Bildschirmrand wird mit den Pfeiltasten links/rechts "
        "bewegt und schießt mit der Leertaste nach oben. Ein Raster aus "
        "absteigenden Alien-Gegnern bewegt sich seitwärts und schießt "
        "gelegentlich zurück nach unten. Treffer auf einen Alien "
        "entfernen ihn und geben Punkte, ein Treffer auf das eigene "
        "Schiff zieht ein Leben ab. Gewinn, wenn alle Aliens zerstört "
        "sind; Game Over bei 0 Leben oder wenn die Aliens den unteren "
        "Rand erreichen. Punktzahl und Leben anzeigen."
    ),
    # 2D top-down (not a real 3D FPS) — deliberately scoped down from a
    # literal "Halo clone": no LLM can produce 3D assets/textures/sound,
    # and a one-shot 3D Three.js attempt would need far more code than
    # this pipeline reliably generates. Same achievable class as the
    # other three games instead, just Halo-themed.
    "halo": (
        "Baue einen 2D Top-Down-Shooter im Stil von Halo (Sci-Fi-"
        "Soldat gegen Alien-Invasoren, KEIN 3D, reine 2D-Draufsicht auf "
        "dem Canvas). Der Spieler wird mit den Pfeiltasten bewegt und "
        "zielt/schießt mit der Maus (Mausposition = Zielrichtung, "
        "Linksklick = Schuss) in alle Richtungen, nicht nur nach oben. "
        "Mehrere Gegnerwellen aus Alien-Gegnern bewegen sich auf den "
        "Spieler zu und greifen bei Kontakt an (Leben abziehen). Jede "
        "Welle wird stärker (mehr oder schnellere Gegner). Punktzahl, "
        "Leben und aktuelle Welle anzeigen. Game Over bei 0 Leben, "
        "Neustart per Taste ohne Seiten-Reload."
    ),
}


def game_prompt(game: str) -> str:
    """Raises KeyError for an unknown game id — callers are expected to
    validate `game in GAME_IDS` first, same as coding.py's tier dicts."""
    return _GAME_PROMPTS[game]


_HTML_FENCE_PATTERN = re.compile(r"(?m)^```(?:html)?[ \t]*\r?\n(.*?)^```", re.DOTALL)


def extract_html_game(llm_response: str) -> str | None:
    """Pull a single self-contained game out of a ```html ... ``` (or
    bare ```...```) fenced block. Returns None — treated as a generation
    failure, never served — when: there's no fence, more than one fenced
    block, the content is missing a <canvas>/<script> (not actually a
    playable game), or its brace/paren counts are unbalanced (the
    cheapest signal that generation was cut off mid-script by hitting
    max_tokens — unlike Python, JS has no ast.parse() to validate
    syntax, so this is a weak-but-free truncation check, not a real
    parser)."""
    matches: list[str] = _HTML_FENCE_PATTERN.findall(llm_response)
    if len(matches) != 1:
        return None
    code = matches[0].strip()
    lowered = code.lower()
    if "<canvas" not in lowered or "<script" not in lowered:
        return None
    if code.count("{") != code.count("}") or code.count("(") != code.count(")"):
        return None
    return code


_SCRIPT_TAG_PATTERN = re.compile(r"<script\b[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)


def extract_script_body(html: str) -> str | None:
    """Concatenate every inline <script>...</script> body in the document
    (a game may legitimately split logic across more than one tag), for
    the smoke-check harness below. None if there's no script at all —
    extract_html_game() already guarantees at least one exists for any
    html it accepted, so this only returns None for malformed input."""
    matches = _SCRIPT_TAG_PATTERN.findall(html)
    if not matches:
        return None
    return "\n".join(matches)


# Fixed harness, not model output — only the extracted script body is
# substituted, as a json.dumps()'d string (same safe-escaping approach
# as coding.py's save_launcher) so it can contain anything (quotes,
# backslashes, template literals) without breaking the harness around
# it.
#
# IMPORTANT — see the module docstring above for the full, verified
# history: Node's `vm` is NOT a security boundary (Node's own docs say
# so explicitly). Building every stub from source text run inside a
# context created with nothing injected (below) closes the escape via
# an injected host object's `.constructor.constructor` (proven exploitable
# before this fix, proven blocked after). It does NOT close every vm
# escape — a bare top-level `this` in the generated script still
# reaches the context's global object and from there the host process
# (proven, including against Node's own suggested stop-gap and against
# forcing strict mode on the generated script — neither helped). That
# residual gap is accepted deliberately because the text sent to the
# model is a fixed, server-controlled constant, never caller-supplied —
# see the module docstring before changing that assumption.
# `setTimeout`/`requestAnimationFrame` queue callbacks instead of
# actually scheduling them; a second, separate run drains a bounded
# number of them to exercise a few ticks of the game's own loop, which
# is what actually catches a bug that only shows up once gameplay
# starts (not just at parse time).
_BOOTSTRAP = """
function makeStubElement() {
  var ctx2d = {
    clearRect: function(){}, fillRect: function(){}, strokeRect: function(){},
    fillText: function(){}, strokeText: function(){}, beginPath: function(){},
    closePath: function(){}, arc: function(){}, moveTo: function(){},
    lineTo: function(){}, stroke: function(){}, fill: function(){},
    save: function(){}, restore: function(){}, translate: function(){},
    rotate: function(){}, scale: function(){}, drawImage: function(){},
    measureText: function(){ return { width: 0 }; },
    fillStyle: "", strokeStyle: "", font: "", textAlign: "", lineWidth: 1
  };
  return {
    style: {}, textContent: "", width: 800, height: 600, children: [],
    getContext: function(){ return ctx2d; },
    getBoundingClientRect: function(){ return { left: 0, top: 0, width: 800, height: 600 }; },
    addEventListener: function(){}, removeEventListener: function(){},
    appendChild: function(){}, focus: function(){}, setAttribute: function(){},
    remove: function(){}
  };
}
var _elements = {};
var _timers = [];
var document = {
  getElementById: function(id) { if (!_elements[id]) _elements[id] = makeStubElement(); return _elements[id]; },
  createElement: function() { return makeStubElement(); },
  addEventListener: function() {},
  removeEventListener: function() {},
  body: makeStubElement()
};
var navigator = { userAgent: "smoke-check" };
var console = { log: function(){}, warn: function(){}, error: function(){}, info: function(){} };
function alert() {}
function confirm() { return true; }
function prompt() { return ""; }
function setTimeout(fn) { _timers.push(fn); return _timers.length; }
function clearTimeout() {}
function setInterval(fn) { _timers.push(fn); return _timers.length; }
function clearInterval() {}
function requestAnimationFrame(fn) { _timers.push(fn); return _timers.length; }
function cancelAnimationFrame() {}
var window = globalThis;
function __drainTimers() {
  for (var i = 0; i < 300 && _timers.length; i++) {
    var fn = _timers.shift();
    fn();
  }
}
"""


def build_smoke_harness(script_body: str) -> str:
    bootstrap_json = json.dumps(_BOOTSTRAP)
    script_json = json.dumps(script_body)
    return (
        '"use strict";\n'
        'const vm = require("vm");\n'
        f"const BOOTSTRAP = {bootstrap_json};\n"
        f"const GAME_SCRIPT = {script_json};\n"
        "\n"
        "const context = vm.createContext();\n"
        "let exitCode = 0;\n"
        "try {\n"
        '  vm.runInContext(BOOTSTRAP, context, { timeout: 2000, filename: "smoke-bootstrap.js" });\n'
        '  vm.runInContext(GAME_SCRIPT, context, { timeout: 4000, filename: "generated-game.js" });\n'
        '  vm.runInContext("__drainTimers()", context, { timeout: 4000, filename: "smoke-drain.js" });\n'
        '  console.log("SMOKE_OK");\n'
        "} catch (e) {\n"
        '  console.error("SMOKE_FAIL: " + (e && e.message ? e.message : String(e)));\n'
        "  exitCode = 1;\n"
        "}\n"
        "process.exit(exitCode);\n"
    )
