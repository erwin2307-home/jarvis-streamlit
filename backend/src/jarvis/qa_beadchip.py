"""Pre-researched Q&A pairs for the Illumina BeadChip reference topic.

Compiled from the reference material in
~/Desktop/Illumina_BeadChip_Referenz/ (Illumina support docs + Adler et
al. 2013, J Vis Exp, PMID 24300335). Spoken answers are kept short —
they're meant to be heard, not read.
"""
from __future__ import annotations

import re

from jarvis.qa_lookup import QaEntry, make_entry

_ALL_REAGENTS_ANSWER = (
    "Vor dem eigentlichen Assay wird die DNA mit Proteinase K aus der "
    "Probe extrahiert. Im Infinium-Workflow selbst kommen dann der "
    "Reihe nach zum Einsatz: DNA-Resuspensionspuffer, MA1, "
    "Natriumhydroxid, MA2 und MSM bei der Amplifikation; FMS bei der "
    "Fragmentierung; PM1 und Isopropanol bei der Präzipitation; RA1 bei "
    "der Resuspension; eine Formamid-EDTA-Mischung, XC4 und PB2 bei der "
    "Hybridisierung; und XC1, XC2, TEM, STM, ATM, PB1 sowie XC3 beim "
    "Staining und Waschen."
)

_FULL_WORKFLOW_24_ANSWER = (
    "Hier ist der komplette Ablauf für einen Chip mit 24 Proben, von "
    "der Speichelprobe bis zur Auswertung. "
    "Schritt null, Probenvorbereitung: Proteinase K baut die Proteine "
    "in der Speichelprobe ab und setzt die genomische DNA frei. "
    "Tag eins, Schritt eins, Vorbereitung: zweihundert Nanogramm DNA "
    "pro Well in eine 96er-Platte, zentrifugieren, über Nacht trocknen. "
    "Schritt zwei, Amplifikation: DNA-Resuspensionspuffer, dann MA1, "
    "dann Natriumhydroxid, dann MA2, dann MSM zugeben, danach zwanzig "
    "bis vierundzwanzig Stunden im Ofen bei siebenunddreißig Grad. "
    "Tag zwei, Schritt drei, Fragmentierung: FMS zugeben, eine Stunde "
    "im Heizblock. "
    "Schritt vier, Präzipitation: PM1, dann hundert Prozent Isopropanol "
    "zugeben, dreißig Minuten bei vier Grad, zwanzig Minuten "
    "zentrifugieren, eine Stunde trocknen. "
    "Schritt fünf, Resuspension: RA1 zugeben, eine Stunde im Ofen bei "
    "achtundvierzig Grad. "
    "Schritt sechs, Hybridisierung: die Formamid-EDTA-Mischung und XC4 "
    "vorbereiten — eine Flasche XC4 reicht für vierundzwanzig Chips — "
    "PB2 in die Feuchtigkeitsreservoirs, fünfzehn Mikroliter Probe pro "
    "Chip-Einlass, dann sechzehn bis vierundzwanzig Stunden "
    "hybridisieren, nicht mehr als vierundzwanzig Chips pro Tag. "
    "Tag drei, Schritt sieben, Staining-Vorbereitung: für vierundzwanzig "
    "Proben braucht man etwa dreißig Milliliter RA1, fünfundzwanzig "
    "Milliliter Formamid-EDTA-Mischung und hundertfünfzig Milliliter "
    "XC3, außerdem XC1, XC2, TEM, STM und ATM auftauen. "
    "Schritt acht, Staining und Extension bei vierundvierzig Grad: "
    "RA1, dann XC1, dann XC2, dann TEM, dann die Formamid-EDTA-Mischung, "
    "dann abwechselnd XC3 mit STM und ATM. "
    "Schritt neun, Waschen und Beschichten: PB1, dann XC4, mit Ethanol "
    "abwischen, fünfzig bis fünfundfünfzig Minuten im Vakuum-Exsikkator "
    "trocknen. "
    "Schritt zehn, Scannen: die Chips im iScan oder HiScan einlesen, "
    "Decode-Dateien laden, und am Ende die Genotyp-Calls auswerten — "
    "die durchschnittliche Call-Rate sollte bei etwa neunundneunzig "
    "Prozent liegen."
)

BEADCHIP_QA: tuple[QaEntry, ...] = (
    make_entry(
        "was ist ein illumina beadchip chip",
        "Ein Illumina BeadChip ist ein Microarray-Glasträger mit "
        "Millionen winziger Beads, jedes mit einer eigenen Sonden-"
        "Sequenz. Er wird für Genotypisierung, Methylierung oder "
        "Genexpression benutzt.",
    ),
    make_entry(
        "was ist infinium assay workflow funktioniert",
        "Der Infinium-Assay läuft über drei Tage: Amplifikation, "
        "Fragmentierung, Präzipitation, Resuspension, Hybridisierung, "
        "Extension, Staining und zum Schluss das Scannen der Chips.",
    ),
    make_entry(
        "wie lange dauert infinium beadchip workflow prozess tage",
        "Der komplette Ablauf dauert etwa drei Tage: Tag eins "
        "Amplifikation über Nacht, Tag zwei Hybridisierung über Nacht, "
        "Tag drei Staining und Scannen.",
    ),
    make_entry(
        "wie viel dna braucht man infinium menge probe",
        "Man braucht etwa zweihundert Nanogramm genomische DNA pro "
        "Probe, mit einem 260/280-Verhältnis zwischen 1,6 und 2,0.",
    ),
    make_entry(
        "welche chemikalien reagenzien werden beim beadchip infinium verwendet",
        _ALL_REAGENTS_ANSWER,
    ),
    make_entry(
        # Dedicated alias entry: STT can mishear "Proteinase K" as
        # "Proteinkinase" (a real, unrelated enzyme name). Kept as its own
        # short, tightly-scoped entry rather than folded into the general
        # reagents trigger phrase above — a short query against a long
        # trigger-keyword set scores too low (score is overlap / entry
        # keyword count), so broad and narrow phrasings need separate
        # entries pointing at the same answer.
        "proteinkinase proteinase k chemikalien reagenzien",
        _ALL_REAGENTS_ANSWER,
    ),
    make_entry(
        "was ist formamid wofür",
        "Formamid wird im Staining-Schritt benutzt, um die Proben-DNA "
        "nach der Extension wieder von der Sonde zu lösen. Es ist eine "
        "95-prozentige Formamid-Lösung mit etwas EDTA.",
    ),
    make_entry(
        "ist formamid gefährlich giftig sicherheit",
        "Ja. Formamid gilt als reproduktionstoxisch, Kategorie 1B, und "
        "kann die Haut und Augen reizen. Handschuhe, Augenschutz und "
        "Laborkittel sind Pflicht, und man muss es gekühlt und dicht "
        "verschlossen lagern.",
    ),
    make_entry(
        "was ist proteinase k wofür",
        "Proteinase K ist ein Enzym zum Abbau von Proteinen. Es wird "
        "benutzt, um DNA aus Gewebe oder Blut zu extrahieren, bevor "
        "die eigentliche BeadChip-Analyse überhaupt beginnt.",
    ),
    make_entry(
        "ist proteinase k gefährlich sicherheit",
        "Proteinase K gilt je nach Hersteller als unterschiedlich "
        "gefährlich — manche Datenblätter nennen Haut- und "
        "Augenreizung. Das Pulver sollte man wegen möglicher "
        "Atemwegsreizung nicht einatmen.",
    ),
    make_entry(
        "was ist ra1 reagenz",
        "RA1 ist ein Resuspensions- und Hybridisierungs-Reagenz im "
        "Infinium-Workflow. Es kann Kristalle bilden — vor Gebrauch "
        "muss man es prüfen und durch Invertieren wieder auflösen.",
    ),
    make_entry(
        "wie viele proben chips kann man pro woche verarbeiten durchsatz",
        "Mit vierundzwanzig Chips pro Tag über eine Fünf-Tage-Woche "
        "kommt man auf über tausend Proben pro Woche, je nach "
        "Chip-Typ.",
    ),
    make_entry(
        "wo sind die sicherheitsdatenblätter sds illumina",
        "Die offiziellen Illumina-Sicherheitsdatenblätter findest du "
        "auf support.illumina.com/sds.html.",
    ),
    make_entry(
        "zähle alle schritte auflisten komplett speichelprobe iscan proben 24",
        _FULL_WORKFLOW_24_ANSWER,
    ),
)


# ── Broad topic detection + RAG context injection ──────────────────────
#
# The exact-match QA_PAIRS above only fire on phrasings close to their
# hand-written triggers. Real speech varies too much for that to be the
# only safety net — a loosely-phrased question ("erzähl mir alles über
# den beadchip") would otherwise fall straight through to the local LLM,
# which knows nothing about Illumina BeadChips and hallucinates. Instead:
# detect the topic broadly (any one marker word is enough) and hand the
# LLM the full reference knowledge as context for that turn, so it
# synthesizes an answer from real facts instead of guessing.

_TOPIC_MARKERS: tuple[str, ...] = (
    "beadchip",
    "bead chip",
    # German Whisper, even with vocabulary biasing (see
    # whisper_initial_prompt in config.py), still mishears "BeadChip"
    # phonetically — observed live as these two spellings.
    "beatship",
    "beatschip",
    "illumina",
    "infinium",
    "iscan",
    "hiscan",
    "proteinase",
    "proteinkinase",
    "formamid",
    "chemikalien",
    "reagenz",
    "genotyp",
    "hybridisierung",
    "amplifikation",
    "fragmentierung",
    "präzipitation",
    "resuspension",
    "speichelprobe",
    "call rate",
    "gentrain",
    "qualitätskontrolle",
    "cluster",
)

# Bare reagent abbreviations (RA1, MA1, ...) need word-boundary matching,
# not plain substring containment — "pb1" as a substring could coincide
# inside an unrelated token, and these are too short/generic to trust
# otherwise. Checked separately from _TOPIC_MARKERS below.
_REAGENT_ABBREVIATIONS: tuple[str, ...] = (
    "ra1", "ma1", "ma2", "msm", "fms", "pm1",
    "lx1", "lx2", "pb1", "pb2",
    "xc1", "xc2", "xc3", "xc4",
    "tem", "stm", "atm",
)
_REAGENT_PATTERN = re.compile(
    r"\b(?:" + "|".join(_REAGENT_ABBREVIATIONS) + r")\b"
)

_BEADCHIP_CONTEXT = (
    "Du hast folgendes Fachwissen über den Illumina BeadChip (Infinium "
    "Assay) zur Genotypisierung. Nutze es, um die Frage des Nutzers "
    "präzise und konkret zu beantworten — nenne wo passend die "
    "genauen Reagenz-Namen und Schritte, erfinde nichts hinzu.\n\n"
    "WAS ES IST: Ein BeadChip ist ein Microarray-Glasträger mit "
    "Millionen Beads, jedes mit einer eigenen DNA-Sonde, für "
    "Genotypisierung, Methylierung oder Genexpression.\n\n"
    "PROBENVORBEREITUNG: Proteinase K baut Proteine in der Probe "
    "(z. B. Speichelprobe) ab und setzt die genomische DNA frei — das "
    "passiert vor dem eigentlichen Assay, nicht im Illumina-Kit selbst.\n\n"
    "WORKFLOW (3 Tage, bis zu 24 Chips/Tag): "
    "1) Vorbereitung — 200 ng DNA pro Well. "
    "2) Amplifikation — DNA-Resuspensionspuffer, MA1, Natriumhydroxid, "
    "MA2, MSM; über Nacht im Ofen bei 37 °C. "
    "3) Fragmentierung — FMS, 1 Std. Heizblock. "
    "4) Präzipitation — PM1, dann Isopropanol, Zentrifuge, trocknen. "
    "5) Resuspension — RA1 (kann kristallisieren, vor Gebrauch "
    "invertieren). "
    "6) Hybridisierung — Formamid/EDTA-Mischung, XC4, PB2; 16–24 Std. "
    "bei 48 °C, max. 24 Chips/Tag, eine XC4-Flasche reicht für 24 Chips. "
    "7–8) Staining und Extension bei 44 °C — RA1, XC1, XC2, TEM, "
    "Formamid/EDTA-Mischung, dann abwechselnd XC3 mit STM und ATM. "
    "9) Waschen und Beschichten — PB1, dann XC4, Vakuum-Trocknung "
    "50–55 Min. "
    "10) Scannen — iScan oder HiScan, Decode-Dateien, Ziel-Call-Rate "
    "~99%.\n\n"
    "SICHERHEIT: Formamid ist reproduktionstoxisch (GHS Kategorie 1B), "
    "reizt Haut und Augen — Handschuhe, Augenschutz, Laborkittel "
    "Pflicht, gekühlt und dicht verschlossen lagern. Proteinase K kann "
    "je nach Hersteller Haut-/Augenreizung verursachen, Pulverstaub "
    "nicht einatmen. Offizielle Sicherheitsdatenblätter: "
    "support.illumina.com/sds.html.\n\n"
    "QUALITÄTSKONTROLLE UND FEHLERBEHEBUNG (Zhao et al. 2017, "
    "Brief Bioinform, PMID 28334151): Die Call-Rate ist die wichtigste "
    "Qualitätskennzahl einer Probe — Standard ist 95–98%, Proben darunter "
    "werden verworfen. Erwartete Ausfallrate bei großen Studien: 1–2% der "
    "Proben. Der größte Call-Rate-Fix ist, ein bereits QC-geprüftes "
    "Cluster-File vom selben Array-Design zu laden statt neu zu clustern "
    "— das verbessert die Call-Rate im Schnitt um etwa 1,7% pro Probe. "
    "GenTrain (Illuminas Clustering-Algorithmus) clustert bis zu 5% aller "
    "SNPs fehlerhaft; die drei wichtigsten QC-Metriken sind GenTrain "
    "Score, Cluster Separation Score und Call Frequency (alle 0 bis 1, "
    "höher ist besser). Typische Fehlerbilder: AB- und BB-Cluster "
    "berühren sich (SNP entfernen), ein langer Schweif am AA-Cluster "
    "(Schweif-Proben entfernen), oder vier statt drei Cluster (SNP "
    "entfernen)."
)


def beadchip_topic_context(text: str) -> str | None:
    """Return the full BeadChip reference context if `text` looks like
    it's about this topic at all (any one marker word is enough), else
    None. Deliberately coarse — this is a RAG-style context injection,
    not an answer match, so false positives just mean the LLM gets
    unused context, not a wrong instant answer."""
    lowered = text.lower()
    if any(marker in lowered for marker in _TOPIC_MARKERS):
        return _BEADCHIP_CONTEXT
    if _REAGENT_PATTERN.search(lowered):
        return _BEADCHIP_CONTEXT
    return None
