"""Short filler phrases played immediately after speech_end so the user
hears something while STT + LLM are still working. Max 2 sentences each."""
from __future__ import annotations

import random

# Categories: witty one-liners, Asimov's Three Laws, Shakespeare, science facts.
FILLER_PHRASES: tuple[str, ...] = (
    # Witty one-liners
    "Einen Moment, bitte.",
    "Lass mich kurz überlegen.",
    "Berechnung läuft.",
    "Sofort, mein Herr.",
    "Ich arbeite daran.",
    "Bitte warten.",
    "Ich wälze die Zahlen.",
    "Nur einen Augenblick.",
    # Asimov's Three Laws of Robotics
    "Ein Roboter darf keinen Menschen verletzen oder durch Untätigkeit zulassen, dass einem Menschen Schaden zugefügt wird.",
    "Ein Roboter muss den Befehlen von Menschen gehorchen, außer wenn diese Befehle dem ersten Gesetz widersprechen.",
    "Ein Roboter muss seine eigene Existenz schützen, solange dies nicht dem ersten oder zweiten Gesetz widerspricht.",
    # Shakespeare
    "Die ganze Welt ist eine Bühne, und alle Männer und Frauen bloß Schauspieler.",
    "Sein oder nicht sein — das ist hier die Frage.",
    "Kürze ist die Würze des Witzes.",
    "Wir wissen, was wir sind, doch nicht, was wir sein könnten.",
    "Der Lauf der wahren Liebe verläuft nie glatt.",
    # Science facts
    "Licht legt etwa dreihunderttausend Kilometer pro Sekunde zurück.",
    "Es gibt mehr Sterne im Universum als Sandkörner an allen Stränden der Erde.",
    "Wasser dehnt sich beim Gefrieren um etwa neun Prozent aus.",
    "Das menschliche Gehirn läuft mit etwa zwanzig Watt.",
    "Ein Teelöffel Neutronensternmaterie würde etwa sechs Milliarden Tonnen wiegen.",
    "Honig verdirbt nie — Archäologen haben dreitausend Jahre alte Proben gegessen.",
)


def random_filler(rng: random.Random | None = None) -> str:
    """Pick a random filler phrase. Optional rng for deterministic tests."""
    r = rng or random
    return r.choice(FILLER_PHRASES)
