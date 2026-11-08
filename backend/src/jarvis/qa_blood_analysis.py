"""Pre-researched Q&A pairs for blood-panel basics (CBC/Blutbild,
coagulation, common chemistry panel) and common pre-analytical error
sources.

First-pass seed set of well-established, standard adult reference facts
— not sourced from a dedicated reference folder like qa_beadchip.py's
Illumina docs, since none exists yet for this topic (see CLAUDE.md).
Deliberately states only general reference-range knowledge, not
diagnoses for a specific person — this is reference knowledge for a lab
assistant, not a clinical judgment tool; reference ranges also vary
somewhat by lab/method, so these are typical adult ranges, not a fixed
universal standard. Grows over time from backend/src/jarvis/unanswered.py's
logged questions: anything that matches _TOPIC_MARKERS below but misses
every entry here gets logged for review at the next content update.
Spoken answers are kept short — they're meant to be heard, not read.
"""
from __future__ import annotations

from jarvis.qa_lookup import QaEntry, make_entry

BLOOD_ANALYSIS_QA: tuple[QaEntry, ...] = (
    make_entry(
        "was ist ein blutbild cbc complete blood count",
        "Ein Blutbild misst die zellulären Bestandteile des Bluts: "
        "rote Blutkörperchen, weiße Blutkörperchen und Thrombozyten, "
        "dazu abgeleitete Werte wie Hämoglobin und Hämatokrit. Das "
        "kleine Blutbild zählt nur die Hauptgruppen, das große Blutbild "
        "schlüsselt die weißen Blutkörperchen zusätzlich nach Typ auf.",
    ),
    make_entry(
        "was ist hämoglobin hamoglobin referenzwert",
        "Hämoglobin transportiert Sauerstoff in den roten "
        "Blutkörperchen. Typische Referenzwerte für Erwachsene liegen "
        "bei etwa 13,5 bis 17,5 Gramm pro Deziliter für Männer und "
        "12 bis 16 Gramm pro Deziliter für Frauen.",
    ),
    make_entry(
        "was ist hämatokrit hamatokrit referenzwert",
        "Der Hämatokrit ist der Volumenanteil der roten Blutkörperchen "
        "am gesamten Blutvolumen, typisch etwa 40 bis 52 Prozent bei "
        "Männern und 36 bis 48 Prozent bei Frauen.",
    ),
    make_entry(
        "was sind leukozyten weiße blutkörperchen referenzwert",
        "Leukozyten, die weißen Blutkörperchen, sind Teil der "
        "Immunabwehr. Ein typischer Referenzbereich für Erwachsene "
        "liegt bei etwa 4 bis 10 Tausend Zellen pro Mikroliter.",
    ),
    make_entry(
        "was sind erythrozyten rote blutkörperchen referenzwert",
        "Erythrozyten, die roten Blutkörperchen, transportieren über "
        "Hämoglobin Sauerstoff. Typisch sind etwa 4,5 bis 5,9 Millionen "
        "Zellen pro Mikroliter bei Männern und 4,1 bis 5,1 Millionen "
        "bei Frauen.",
    ),
    make_entry(
        "was sind thrombozyten blutplättchen referenzwert",
        "Thrombozyten, die Blutplättchen, sind für die Blutgerinnung "
        "zuständig. Der typische Referenzbereich liegt bei etwa 150 "
        "bis 400 Tausend pro Mikroliter.",
    ),
    make_entry(
        "was ist mcv mch mchc blutbild erythrozytenindizes",
        "MCV ist das mittlere Volumen eines einzelnen roten "
        "Blutkörperchens, MCH der mittlere Hämoglobingehalt pro Zelle, "
        "MCHC die mittlere Hämoglobinkonzentration. Zusammen helfen "
        "diese Indizes, eine Anämie nach Zellgröße einzuordnen.",
    ),
    make_entry(
        "anämie mikrozytär normozytär makrozytär einteilung",
        "Anämien werden anhand des MCV eingeteilt: mikrozytär (MCV zu "
        "niedrig, häufig Eisenmangel), normozytär (MCV normal, z. B. "
        "bei akutem Blutverlust oder chronischer Erkrankung) und "
        "makrozytär (MCV zu hoch, häufig Vitamin-B12- oder "
        "Folsäuremangel).",
    ),
    make_entry(
        "differentialblutbild neutrophile lymphozyten monozyten eosinophile basophile",
        "Das Differentialblutbild schlüsselt die weißen Blutkörperchen "
        "nach Typ auf: neutrophile Granulozyten (akute Infektabwehr), "
        "Lymphozyten (virale Infekte, Immungedächtnis), Monozyten, "
        "eosinophile Granulozyten (Allergien, Parasiten) und basophile "
        "Granulozyten.",
    ),
    make_entry(
        "was ist quick wert inr aptt gerinnung",
        "Quick-Wert und INR messen den extrinsischen Gerinnungsweg und "
        "werden zur Überwachung von Vitamin-K-Antagonisten wie "
        "Warfarin genutzt — ein normaler INR liegt nahe 1,0, unter "
        "Therapie oft zwischen 2 und 3. Die aPTT misst den intrinsischen "
        "Gerinnungsweg und wird z. B. unter Heparin-Therapie überwacht.",
    ),
    make_entry(
        "kreatinin egfr nierenfunktion wert",
        "Kreatinin ist ein Abbauprodukt der Muskulatur, das über die "
        "Nieren ausgeschieden wird — ein erhöhter Wert kann auf "
        "eingeschränkte Nierenfunktion hindeuten. Die eGFR schätzt "
        "daraus die glomeruläre Filtrationsrate und ist aussagekräftiger "
        "als Kreatinin allein, besonders bei leichten Einschränkungen.",
    ),
    make_entry(
        "leberwerte alt ast transaminasen bedeutung",
        "ALT und AST sind Leberenzyme, die bei Leberzellschäden ins "
        "Blut freigesetzt werden. ALT gilt als leberspezifischer, AST "
        "kommt auch in Muskel- und Herzgewebe vor — deutlich erhöhte "
        "Werte deuten auf eine akute Leberzellschädigung hin.",
    ),
    make_entry(
        "elektrolyte natrium kalium blutwert",
        "Natrium und Kalium sind die wichtigsten Elektrolyte im "
        "Blutchemie-Panel — Natrium reguliert vor allem den "
        "Wasserhaushalt, Kalium ist entscheidend für die "
        "Erregbarkeit von Herz- und Muskelzellen; schon kleine "
        "Abweichungen bei Kalium können klinisch relevant sein.",
    ),
    make_entry(
        "hämolyse probe falsch verfälscht messfehler",
        "Hämolyse — wenn rote Blutkörperchen beim Abnehmen oder "
        "Transport der Probe platzen — gibt intrazelluläre Stoffe wie "
        "Kalium und LDH künstlich ins Serum ab und verfälscht diese "
        "Messwerte nach oben. Eine hämolytische Probe wird deshalb oft "
        "verworfen und neu abgenommen statt interpretiert.",
    ),
    make_entry(
        "serum plasma vollblut unterschied probenröhrchen",
        "Vollblut enthält alle Zellen plus Gerinnungsfaktoren. Plasma "
        "ist der flüssige Anteil nach Zentrifugation mit "
        "Gerinnungshemmer (z. B. EDTA, Blutbild) im Röhrchen. Serum ist "
        "der flüssige Anteil, nachdem das Blut ohne Gerinnungshemmer "
        "gerinnen durfte — die Gerinnungsfaktoren fehlen darin.",
    ),
)


_TOPIC_MARKERS: tuple[str, ...] = (
    "blutbild",
    "blutanalyse",
    "blutwert",
    "blutwerte",
    "hämoglobin",
    "hamoglobin",
    "hämatokrit",
    "hamatokrit",
    "leukozyten",
    "erythrozyten",
    "thrombozyten",
    "mcv",
    "mch",
    "mchc",
    "neutrophile",
    "lymphozyten",
    "monozyten",
    "eosinophile",
    "basophile",
    "anämie",
    "anaemie",
    "gerinnung",
    "quick-wert",
    "quick wert",
    "inr-wert",
    "inr wert",
    "aptt",
    "elektrolyte",
    "kreatinin",
    "egfr",
    "leberwerte",
    "transaminasen",
    "hämoly",  # catches hämolyse/hämolytisch/hämolytische(n)
    "hamoly",
    "serum",
    "vollblut",
)


def blood_analysis_topic_context(text: str) -> str | None:
    """Broad RAG-style context injection — any one marker word is
    enough, same coarse-matching reasoning as qa_beadchip.
    beadchip_topic_context."""
    lowered = text.lower()
    if any(marker in lowered for marker in _TOPIC_MARKERS):
        return _BLOOD_ANALYSIS_CONTEXT
    return None


_BLOOD_ANALYSIS_CONTEXT = (
    "Du hast folgendes Fachwissen über Blutanalyse (Blutbild, Gerinnung, "
    "Blutchemie). Nutze es, um die Frage des Nutzers präzise zu "
    "beantworten, erfinde nichts hinzu, und gib KEINE Diagnose für eine "
    "reale Person — nenne nur die allgemein bekannten Referenzbereiche "
    "und Zusammenhänge; sie können je nach Labor/Methode leicht "
    "abweichen.\n\n"
    "BLUTBILD: Hämoglobin (ca. 13,5-17,5 g/dl Männer, 12-16 g/dl "
    "Frauen), Hämatokrit (ca. 40-52% Männer, 36-48% Frauen), "
    "Leukozyten (ca. 4-10 Tsd/µl), Erythrozyten (ca. 4,5-5,9 Mio/µl "
    "Männer, 4,1-5,1 Mio/µl Frauen), Thrombozyten (ca. 150-400 Tsd/µl). "
    "MCV/MCH/MCHC ordnen eine Anämie nach Zellgröße ein: mikrozytär "
    "(oft Eisenmangel), normozytär, makrozytär (oft B12-/Folsäure-"
    "mangel). Differentialblutbild: neutrophile Granulozyten (akute "
    "Infekte), Lymphozyten (virale Infekte), Monozyten, eosinophile "
    "(Allergien/Parasiten), basophile Granulozyten.\n\n"
    "GERINNUNG: Quick-Wert/INR messen den extrinsischen Gerinnungsweg "
    "(Warfarin-Überwachung, Ziel-INR oft 2-3 unter Therapie); aPTT den "
    "intrinsischen Weg (z. B. Heparin-Überwachung).\n\n"
    "CHEMIE-PANEL: Kreatinin/eGFR für Nierenfunktion. ALT/AST "
    "(Transaminasen) für Leberzellschäden, ALT leberspezifischer. "
    "Natrium (Wasserhaushalt) und Kalium (Herz-/Muskelerregbarkeit) als "
    "wichtigste Elektrolyte.\n\n"
    "PRÄANALYTIK: Hämolyse (rote Blutkörperchen platzen beim Abnehmen/"
    "Transport) erhöht Kalium und LDH künstlich — hämolytische Proben "
    "werden meist verworfen statt interpretiert. Serum = geronnenes "
    "Blut ohne Gerinnungsfaktoren; Plasma = zentrifugiertes Blut MIT "
    "Gerinnungshemmer, Gerinnungsfaktoren bleiben enthalten."
)
