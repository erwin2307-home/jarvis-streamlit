"""Pre-researched Q&A pairs for Next-Generation-Sequencing (NGS) basics
and troubleshooting.

First-pass seed set of well-established, stable facts (sequencing
chemistry fundamentals, standard file formats, common run-quality
failure modes) — not sourced from a dedicated reference folder like
qa_beadchip.py's Illumina docs, since none exists yet for this topic
(see CLAUDE.md). Grows over time from backend/src/jarvis/unanswered.py's
logged questions: anything that matches _TOPIC_MARKERS below but misses
every entry here gets logged for review at the next content update.
Spoken answers are kept short — they're meant to be heard, not read.
"""
from __future__ import annotations

from jarvis.qa_lookup import QaEntry, make_entry

NGS_QA: tuple[QaEntry, ...] = (
    make_entry(
        "was ist ngs next generation sequencing",
        "NGS, Next Generation Sequencing, bezeichnet Verfahren, die "
        "Millionen von DNA-Fragmenten gleichzeitig sequenzieren, statt "
        "wie klassisches Sanger-Sequencing ein Fragment nach dem "
        "anderen. Dadurch wird ein ganzes Genom oder Exom in wenigen "
        "Tagen statt Jahren lesbar.",
    ),
    make_entry(
        "wie funktioniert illumina sequenzierung sequencing by synthesis sbs",
        "Illuminas Sequencing-by-Synthesis liest die DNA Base für Base "
        "ab: fluoreszenzmarkierte Nukleotide werden eingebaut, ein "
        "Laser liest die Farbe aus, dann wird die Markierung entfernt "
        "und der nächste Zyklus beginnt — das passiert für Millionen "
        "Fragmente parallel auf der Flow Cell.",
    ),
    make_entry(
        "was ist eine flow cell flowcell",
        "Die Flow Cell ist die Glasoberfläche, auf der die DNA-"
        "Fragmente an Oligo-Primer binden, per Brücken-Amplifikation "
        "zu Clustern vervielfältigt werden und dann Zyklus für Zyklus "
        "sequenziert werden.",
    ),
    make_entry(
        "was ist bridge amplifikation cluster generation",
        "Bei der Brücken-Amplifikation binden einzelne DNA-Fragmente an "
        "Primer auf der Flow Cell und werden lokal vervielfältigt, bis "
        "ein Cluster aus tausenden identischen Kopien entsteht — jedes "
        "Cluster liefert später ein eigenes, klar lesbares Signal.",
    ),
    make_entry(
        "library prep bibliothek vorbereitung schritte ngs",
        "Library Prep läuft typischerweise so ab: DNA fragmentieren, "
        "Enden reparieren, Adapter anligieren, per PCR amplifizieren, "
        "dann nach Fragmentgröße aufreinigen — erst danach kommt die "
        "Bibliothek auf die Flow Cell.",
    ),
    make_entry(
        "paired end single end unterschied reads",
        "Single-End liest jedes Fragment nur von einer Seite, Paired-"
        "End von beiden Enden — Paired-End liefert mehr Information "
        "(bessere Alignment-Genauigkeit, erkennt Insertionen/Deletionen "
        "besser), kostet aber etwa doppelt so viele Reads.",
    ),
    make_entry(
        "was ist coverage sequenziertiefe depth",
        "Coverage, auch Sequenziertiefe, gibt an, wie oft jede Base im "
        "Durchschnitt gelesen wurde. 30x für ein menschliches Genom "
        "gilt als Standard für zuverlässige Keimbahn-Varianten-"
        "erkennung; für Tumorproben braucht man oft deutlich mehr.",
    ),
    make_entry(
        "was ist q30 phred score qualität basecall",
        "Der Phred-Score gibt die Wahrscheinlichkeit eines Lesefehlers "
        "pro Base an. Q30 bedeutet eine Fehlerwahrscheinlichkeit von "
        "1 zu 1000, also 99,9 Prozent Genauigkeit — das ist der "
        "übliche Mindestanspruch an einen guten Illumina-Lauf.",
    ),
    make_entry(
        "was ist fastq format dateiformat sequenzierung",
        "FASTQ speichert pro Read vier Zeilen: eine ID, die "
        "Nukleotidsequenz, eine Trennzeile und eine Qualitätszeile mit "
        "einem Phred-Score-Zeichen pro Base — das Standardformat, das "
        "ein Sequenzer ausgibt.",
    ),
    make_entry(
        "was ist vcf format variant calling varianten",
        "Variant Calling vergleicht die sequenzierten Reads nach dem "
        "Alignment gegen ein Referenzgenom und listet Abweichungen "
        "(SNPs, Insertionen, Deletionen) in einer VCF-Datei — Variant "
        "Call Format — mit Position, Referenz- und beobachteter Base "
        "sowie einem Qualitätswert pro Variante.",
    ),
    make_entry(
        "wgs wes unterschied whole genome exome sequencing",
        "WGS, Whole Genome Sequencing, liest das komplette Genom; WES, "
        "Whole Exome Sequencing, liest nur die proteincodierenden "
        "Exons — etwa 1 bis 2 Prozent des Genoms, aber dort, wo die "
        "meisten bekannten krankheitsrelevanten Varianten liegen. WES "
        "ist günstiger, übersieht aber nicht-codierende Varianten.",
    ),
    make_entry(
        "nanopore pacbio long read short read illumina unterschied",
        "Illumina liefert kurze, sehr genaue Reads (um die 150 "
        "Basen) in riesiger Zahl. Nanopore und PacBio liefern lange "
        "Reads (zehntausende Basen), die Strukturvarianten und "
        "repetitive Regionen besser auflösen, aber traditionell eine "
        "höhere Fehlerrate pro Base hatten.",
    ),
    make_entry(
        "was ist index hopping ngs problem",
        "Index Hopping passiert, wenn bei gepoolten Proben falsch "
        "zugeordnete Indizes dazu führen, dass Reads der falschen "
        "Probe zugeschrieben werden — häufiger bei Patterned-Flow-Cell-"
        "Chemien mit freien Adaptern im Pool. Unique Dual Indices "
        "reduzieren das Risiko deutlich.",
    ),
    make_entry(
        "adapter dimer problem ngs fehlersuche",
        "Adapter-Dimere entstehen, wenn sich zwei Adapter ohne "
        "dazwischenliegendes DNA-Fragment verbinden und mitamplifiziert "
        "werden — sie laufen bei der Größenselektion als sehr kurzes "
        "Fragment mit und verschwenden Lesekapazität, wenn sie nicht "
        "vorher abgetrennt werden.",
    ),
    make_entry(
        "niedrige clusterdichte cluster density zu niedrig hoch problem",
        "Zu niedrige Cluster-Dichte bedeutet zu wenig geladene DNA-"
        "Menge — weniger Daten, aber meist saubere Qualität. Zu hohe "
        "Dichte führt zu überlappenden Clustern, die der Sequenzer "
        "nicht mehr trennen kann — das drückt direkt auf die Q30-Rate. "
        "Beides wird über die Ladekonzentration der Bibliothek "
        "korrigiert.",
    ),
    make_entry(
        "was ist demultiplexing bcl2fastq basecalling",
        "Nach dem Lauf wandelt Basecalling die rohen Bildsignale der "
        "Flow Cell in Basen um (BCL-Dateien); Demultiplexing sortiert "
        "die Reads anhand ihrer Index-Sequenz den richtigen Proben "
        "zu und schreibt pro Probe eine eigene FASTQ-Datei.",
    ),
    make_entry(
        "was ist alignment referenzgenom bwa mapping",
        "Alignment ordnet jeden sequenzierten Read seiner "
        "wahrscheinlichsten Position im Referenzgenom zu — Tools wie "
        "BWA oder Bowtie machen das. Erst nach dem Alignment lässt "
        "sich erkennen, wo eine Probe vom Referenzgenom abweicht.",
    ),
    make_entry(
        "duplicate reads pcr duplikate problem ngs",
        "PCR-Duplikate sind Reads, die vom selben ursprünglichen "
        "Molekül stammen und nur durch die Amplifikation mehrfach "
        "vorkommen — sie werden vor dem Variant Calling markiert und "
        "ausgeschlossen, sonst verzerren sie die beobachtete "
        "Allelfrequenz.",
    ),
)


_TOPIC_MARKERS: tuple[str, ...] = (
    "ngs",
    "next generation sequencing",
    "next-generation sequencing",
    "sequenzierung",
    "sequenziertiefe",
    "fastq",
    "basecalling",
    "base calling",
    "bcl2fastq",
    "demultiplexing",
    "adapter dimer",
    "index hopping",
    "cluster density",
    "clusterdichte",
    "phasing",
    "prephasing",
    "paired-end",
    "paired end",
    "single-end",
    "phred",
    "q30",
    "variant calling",
    "vcf",
    "bwa",
    "gatk",
    "whole genome sequencing",
    "whole exome sequencing",
    "wgs",
    "wes",
    "nanopore",
    "pacbio",
    "long-read",
    "long read",
    "short-read",
    "bridge-amplifikation",
    "flow cell",
    "flowcell",
    "library prep",
)


def ngs_topic_context(text: str) -> str | None:
    """Broad RAG-style context injection — any one marker word is
    enough, same coarse-matching reasoning as qa_beadchip.
    beadchip_topic_context."""
    lowered = text.lower()
    if any(marker in lowered for marker in _TOPIC_MARKERS):
        return _NGS_CONTEXT
    return None


_NGS_CONTEXT = (
    "Du hast folgendes Fachwissen über Next-Generation-Sequencing (NGS). "
    "Nutze es, um die Frage des Nutzers präzise zu beantworten, erfinde "
    "nichts hinzu.\n\n"
    "GRUNDPRINZIP: NGS sequenziert Millionen DNA-Fragmente parallel. "
    "Illuminas Sequencing-by-Synthesis liest Base für Base per "
    "fluoreszenzmarkiertem Nukleotid-Einbau auf einer Flow Cell, nach "
    "Brücken-Amplifikation zu Clustern.\n\n"
    "WORKFLOW: Library Prep (Fragmentierung, Adapter-Ligation, PCR-"
    "Amplifikation, Größenselektion) -> Cluster-Generation auf der Flow "
    "Cell -> zyklisches Sequenzieren -> Basecalling (BCL) -> "
    "Demultiplexing nach Index zu FASTQ -> Alignment gegen ein "
    "Referenzgenom (z. B. BWA) -> Variant Calling (z. B. GATK) -> VCF.\n\n"
    "KENNZAHLEN: Coverage/Sequenziertiefe = wie oft jede Base gelesen "
    "wurde (30x Standard für Keimbahn-WGS). Q30/Phred-Score = "
    "Fehlerwahrscheinlichkeit pro Base (Q30 = 1 Fehler auf 1000 Basen).\n\n"
    "VARIANTEN: WGS liest das ganze Genom, WES nur die ~1-2% "
    "proteincodierenden Exons. Illumina liefert kurze, sehr genaue "
    "Reads in großer Zahl; Nanopore/PacBio liefern lange Reads, besser "
    "für Strukturvarianten, traditionell höhere Fehlerrate pro Base.\n\n"
    "HÄUFIGE FEHLERBILDER: Adapter-Dimere (Adapter ohne DNA-Insert, "
    "verschwenden Lesekapazität), Index Hopping (Reads landen bei "
    "gepoolten Proben bei der falschen Probe, Unique Dual Indices "
    "mindern das Risiko), zu hohe/niedrige Cluster-Dichte (überlappende "
    "Cluster senken Q30, zu wenig Ladung senkt die Datenmenge), PCR-"
    "Duplikate (verzerren Allelfrequenzen, werden vor Variant Calling "
    "markiert und ausgeschlossen)."
)
