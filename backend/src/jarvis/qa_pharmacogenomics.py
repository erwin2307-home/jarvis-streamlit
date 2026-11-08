"""Pre-researched Q&A pairs for pharmacogenomics (gene-drug interactions,
metabolizer phenotypes, CPIC/PharmGKB-style guidance).

First-pass seed set of well-established, widely-published gene-drug
pairs and terminology — not sourced from a dedicated reference folder
like qa_beadchip.py's Illumina docs, since none exists yet for this
topic (see CLAUDE.md). Deliberately states only broadly-established
facts (established CPIC/FDA-label-level associations) rather than
specific dosing numbers, which depend on genotype combinations, drug,
indication and local guidelines — this is reference knowledge for a
lab assistant, not a prescribing tool. Grows over time from
backend/src/jarvis/unanswered.py's logged questions: anything that matches
_TOPIC_MARKERS below but misses every entry here gets logged for review
at the next content update. Spoken answers are kept short — they're
meant to be heard, not read.
"""
from __future__ import annotations

from jarvis.qa_lookup import QaEntry, make_entry

PHARMACOGENOMICS_QA: tuple[QaEntry, ...] = (
    make_entry(
        "was ist pharmakogenomik pharmakogenetik",
        "Pharmakogenomik untersucht, wie genetische Varianten "
        "beeinflussen, wie ein Mensch ein Medikament verstoffwechselt, "
        "wie gut es wirkt und wie hoch das Risiko für Nebenwirkungen "
        "ist. Ziel ist, Dosis oder Wirkstoffwahl an das individuelle "
        "Genotyp-Profil anzupassen.",
    ),
    make_entry(
        "was sind metabolisierer typen poor intermediate normal ultra",
        "Die CPIC-Klassifikation kennt vier Metabolisierer-Typen: Poor "
        "Metabolizer (kaum Enzymaktivität), Intermediate (reduziert), "
        "Normal — früher 'Extensive' genannt — und Ultrarapid "
        "(deutlich erhöhte Aktivität). Welcher Typ vorliegt, hängt vom "
        "individuellen Allel-Paar eines Gens wie CYP2D6 ab.",
    ),
    make_entry(
        "was ist ein sternallel star allele nomenklatur",
        "Pharmakogene werden mit Sternallelen benannt, z. B. CYP2D6*4 "
        "— jedes Sternallel steht für eine bestimmte Kombination von "
        "Varianten in diesem Gen mit bekannter Funktionsänderung. Der "
        "Diplotyp, also beide Allele zusammen, bestimmt den "
        "Metabolisierer-Typ.",
    ),
    make_entry(
        "was ist cpic pharmgkb",
        "CPIC, das Clinical Pharmacogenetics Implementation Consortium, "
        "veröffentlicht evidenzbasierte Dosierungs-Leitlinien für "
        "Gen-Medikament-Paare. PharmGKB ist die zugehörige öffentliche "
        "Wissensdatenbank, die Varianten, Phänotypen und diese "
        "Leitlinien sammelt.",
    ),
    make_entry(
        "cyp2d6 codein tramadol metabolisierung",
        "CYP2D6 aktiviert Codein zu Morphin. Poor Metabolizer bekommen "
        "kaum Wirkung, Ultrarapid Metabolizer bilden gefährlich viel "
        "Morphin daraus — die FDA warnt deshalb ausdrücklich vor "
        "Codein bei bekannten ultraschnellen Metabolisierern, "
        "besonders bei Kindern nach Tonsillektomie.",
    ),
    make_entry(
        "cyp2c19 clopidogrel plavix wirkung",
        "CYP2C19 aktiviert Clopidogrel (Plavix) zur wirksamen Form. "
        "Poor Metabolizer aktivieren den Wirkstoff kaum und haben ein "
        "erhöhtes Risiko für Stent-Thrombosen und kardiovaskuläre "
        "Ereignisse — CPIC empfiehlt für sie einen alternativen "
        "Thrombozytenhemmer.",
    ),
    make_entry(
        "vkorc1 cyp2c9 warfarin marcumar dosierung",
        "Warfarin-Dosierung hängt stark von VKORC1- und CYP2C9-"
        "Varianten ab: VKORC1 bestimmt die Empfindlichkeit des "
        "Zielenzyms, CYP2C9 den Abbau des Wirkstoffs. Beide zusammen "
        "erklären einen großen Teil der individuellen Dosisunterschiede "
        "und fließen in pharmakogenetische Dosierungsalgorithmen ein.",
    ),
    make_entry(
        "tpmt nudt15 thiopurine azathioprin mercaptopurin",
        "TPMT und NUDT15 bauen Thiopurine wie Azathioprin oder "
        "Mercaptopurin ab. Bei verminderter Aktivität reichert sich der "
        "Wirkstoff an und das Risiko für schwere Knochenmarks-"
        "Suppression steigt stark — deshalb wird vor Therapiestart "
        "häufig genotypisiert, NUDT15-Varianten sind besonders in "
        "ostasiatischer Bevölkerung relevant.",
    ),
    make_entry(
        "dpyd 5fu fluorouracil capecitabin chemotherapie",
        "DPYD baut 5-Fluorouracil und Capecitabin ab. Menschen mit "
        "verminderter DPYD-Aktivität können bei Standarddosis schwere, "
        "teils lebensbedrohliche Toxizität entwickeln — CPIC empfiehlt "
        "Dosisreduktion oder Alternativtherapie je nach Aktivitätsgrad.",
    ),
    make_entry(
        "hla-b 5701 abacavir hypersensitivität",
        "HLA-B*57:01 ist stark mit schweren Überempfindlichkeits-"
        "reaktionen auf Abacavir assoziiert. Ein Test auf dieses Allel "
        "vor Therapiestart ist mittlerweile Standard und wird in "
        "Leitlinien ausdrücklich empfohlen.",
    ),
    make_entry(
        "hla-b 1502 carbamazepin stevens johnson syndrom",
        "HLA-B*15:02 ist mit schweren Hautreaktionen wie Stevens-"
        "Johnson-Syndrom unter Carbamazepin assoziiert, besonders in "
        "südostasiatischer Bevölkerung — dort wird vor Therapiestart "
        "oft gezielt darauf getestet.",
    ),
    make_entry(
        "slco1b1 statine muskelschmerzen myopathie",
        "SLCO1B1-Varianten beeinflussen den Transport von Statinen in "
        "die Leber. Reduzierte Transportfunktion erhöht die "
        "Statin-Konzentration im Blut und damit das Risiko für "
        "Muskelschmerzen und Myopathie, besonders bei Simvastatin in "
        "hoher Dosis.",
    ),
    make_entry(
        "ugt1a1 irinotecan toxizität",
        "UGT1A1 entgiftet den aktiven Metaboliten von Irinotecan. Bei "
        "reduzierter Aktivität (z. B. UGT1A1*28) steigt das Risiko für "
        "schwere Neutropenie und Diarrhö — eine niedrigere Startdosis "
        "wird bei bekannter Variante empfohlen.",
    ),
    make_entry(
        "cyp3a5 tacrolimus transplantation immunsuppression",
        "CYP3A5-Expresser bauen Tacrolimus schneller ab und brauchen "
        "oft eine höhere Startdosis, um nach einer Transplantation "
        "rechtzeitig einen wirksamen Blutspiegel zu erreichen — ohne "
        "Anpassung drohen zu niedrige Spiegel und Abstoßungsrisiko.",
    ),
    make_entry(
        "warum vor medikament genotypisieren pharmakogenetischer test",
        "Ein pharmakogenetischer Test vor Therapiestart erkennt, ob "
        "jemand ein Medikament ungewöhnlich langsam oder schnell "
        "verstoffwechselt — das erlaubt, Dosis oder Wirkstoff vorher "
        "anzupassen, statt es nach einer Nebenwirkung oder "
        "Wirkungslosigkeit zu bemerken.",
    ),
    make_entry(
        "wie wird pharmakogenomik getestet array ngs panel",
        "Pharmakogenetische Varianten werden entweder gezielt per "
        "Genotypisierungs-Array (ähnlich wie BeadChip-Panels für "
        "bekannte Sternallele) oder per NGS-Panel erfasst, das mehrere "
        "Pharmakogene gleichzeitig sequenziert — je nachdem, wie viele "
        "Gene und wie seltene Varianten erfasst werden sollen.",
    ),
)


_TOPIC_MARKERS: tuple[str, ...] = (
    "pharmakogenomik",
    "pharmakogenetik",
    "pharmacogenomics",
    "cyp2d6",
    "cyp2c19",
    "cyp2c9",
    "cyp3a5",
    "vkorc1",
    "tpmt",
    "dpyd",
    "nudt15",
    "ugt1a1",
    "slco1b1",
    "hla-b",
    "hla b",
    "metabolisierer",
    "metabolizer",
    "sternallel",
    "star allele",
    "cpic",
    "pharmgkb",
    "clopidogrel",
    "plavix",
    "warfarin",
    "marcumar",
    "codein",
    "tramadol",
    "azathioprin",
    "mercaptopurin",
    "thiopurin",
    "abacavir",
    "carbamazepin",
    "tacrolimus",
    "irinotecan",
    "fluorouracil",
    "capecitabin",
)


def pharmacogenomics_topic_context(text: str) -> str | None:
    """Broad RAG-style context injection — any one marker word is
    enough, same coarse-matching reasoning as qa_beadchip.
    beadchip_topic_context."""
    lowered = text.lower()
    if any(marker in lowered for marker in _TOPIC_MARKERS):
        return _PHARMACOGENOMICS_CONTEXT
    return None


_PHARMACOGENOMICS_CONTEXT = (
    "Du hast folgendes Fachwissen über Pharmakogenomik. Nutze es, um die "
    "Frage des Nutzers präzise zu beantworten, erfinde nichts hinzu und "
    "gib KEINE konkrete Dosierungsempfehlung für eine reale Person — "
    "nenne nur die allgemein bekannten Zusammenhänge.\n\n"
    "GRUNDPRINZIP: Genvarianten beeinflussen, wie ein Mensch ein "
    "Medikament verstoffwechselt. CPIC-Metabolisierer-Typen: Poor, "
    "Intermediate, Normal, Ultrarapid — bestimmt vom Sternallel-"
    "Diplotyp eines Gens. CPIC/PharmGKB veröffentlichen die "
    "evidenzbasierten Gen-Medikament-Leitlinien.\n\n"
    "WICHTIGE GEN-MEDIKAMENT-PAARE: CYP2D6/Codein (Ultrarapid = zu viel "
    "Morphin, FDA-Warnung). CYP2C19/Clopidogrel (Poor Metabolizer = "
    "Stent-Thrombose-Risiko). VKORC1+CYP2C9/Warfarin (Dosierungs-"
    "Algorithmen). TPMT+NUDT15/Thiopurine (Knochenmarks-Suppression, "
    "NUDT15 v. a. ostasiatisch relevant). DPYD/5-FU+Capecitabin "
    "(schwere Toxizität bei reduzierter Aktivität). HLA-B*57:01/"
    "Abacavir und HLA-B*15:02/Carbamazepin (schwere Hautreaktionen, "
    "Stevens-Johnson-Syndrom). SLCO1B1/Statine (Myopathie-Risiko). "
    "UGT1A1/Irinotecan (Neutropenie-Risiko). CYP3A5/Tacrolimus "
    "(Expresser brauchen höhere Dosis nach Transplantation).\n\n"
    "TESTMETHODEN: gezielte Genotypisierungs-Arrays für bekannte "
    "Sternallele, oder NGS-Panels für mehrere Pharmakogene gleichzeitig."
)
