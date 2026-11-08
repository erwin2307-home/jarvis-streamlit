"""Browser-playable arcade games, generated on request by the "large"
coding-model tier (see config.py's coding_model / coding.py's TIER_ORDER)
and served as a single self-contained HTML file.

Deliberately NOT the same pipeline as coding.py's Python-script fast-path
— that one auto-runs generated code with the user's own OS privileges on
their desktop (no sandbox, explicit trade-off, see coding.py's module
docstring). A generated game here is plain client-side HTML/JS served to
a browser, the same trust model as any website's JS running in its own
tab's sandbox — no server-side execution of model output at all, which
is exactly the risk remote_api.py's module docstring says it avoids by
not wiring in a coding_handler. That's why this module's generate/serve
endpoints are safe to expose on remote_api.py even though the Python
coding fast-path deliberately isn't.

Each game is requested as ONE complete, offline, self-contained HTML
document (inline <style>/<script>, no CDN/network calls — this project
is offline-first and a generated game shouldn't be the exception) with
canvas-based rendering and keyboard controls.
"""
from __future__ import annotations

import re

GAME_IDS: tuple[str, ...] = ("tetris", "pacman", "aliens")

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
