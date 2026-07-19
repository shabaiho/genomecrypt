"""Plain-language descriptions of what a detected gene actually does.

The report has to be readable by someone treating a patient, not by someone who
trained the model. `genefam:blaKPC +1.664 toward_resistant curated=True` is
precise and useless at the bedside; "carbapenemase — destroys meropenem and
other carbapenems" is the same fact in a form a clinician can act on.

Only 54 features carry a non-zero weight in any of the five shipped models, so
the coverage needed here is small and finite. Patterns are ordered most specific
first; anything unmatched falls back to a neutral phrase rather than inventing
biology.
"""
from __future__ import annotations

import re

# (regex on the symbol, description) -- checked in order, first match wins.
GENE_DESCRIPTIONS: list[tuple[str, str]] = [
    # --- carbapenemases: the ones that matter most clinically ---
    (r"^blaKPC", "carbapenemase (KPC) — destroys carbapenems and most other beta-lactams"),
    (r"^blaNDM", "metallo-carbapenemase (NDM) — destroys carbapenems; not blocked by "
                 "the usual beta-lactamase inhibitors"),
    (r"^blaVIM|^blaIMP", "metallo-carbapenemase — destroys carbapenems"),
    (r"^blaOXA-48|^blaOXA-181|^blaOXA-232", "carbapenemase (OXA-48-like) — destroys carbapenems"),
    (r"^blaOXA", "beta-lactamase (OXA family) — breaks down penicillins and some "
                 "cephalosporins"),
    # --- extended-spectrum and other beta-lactamases ---
    (r"^blaCTX-M", "extended-spectrum beta-lactamase (CTX-M) — destroys cephalosporins "
                   "such as ceftriaxone"),
    (r"^blaSHV-12|^blaSHV-2|^blaSHV-5", "extended-spectrum beta-lactamase (SHV) — "
                                        "destroys cephalosporins"),
    (r"^blaSHV", "beta-lactamase (SHV) — present in almost every K. pneumoniae; on its "
                 "own it confers little resistance"),
    (r"^blaTEM", "beta-lactamase (TEM) — breaks down penicillins"),
    (r"^blaCMY|^blaDHA|^blaACT|^blaFOX", "AmpC beta-lactamase — destroys cephalosporins "
                                         "and resists clavulanate"),
    (r"^blaLAP|^blaEC|^bla", "beta-lactamase — breaks down beta-lactam antibiotics"),
    # --- aminoglycoside modifying enzymes ---
    (r"^rmt|^armA", "16S rRNA methyltransferase — blocks all aminoglycosides including "
                    "gentamicin and amikacin"),
    (r"^aac\(6'\)-Ib-cr", "modifies aminoglycosides and also reduces fluoroquinolone "
                          "activity"),
    (r"^aac\(3\)", "aminoglycoside acetyltransferase — inactivates gentamicin and "
                   "tobramycin"),
    (r"^aac\(6'\)", "aminoglycoside acetyltransferase — inactivates amikacin and "
                    "tobramycin"),
    (r"^ant\(|^aad", "aminoglycoside nucleotidyltransferase — inactivates streptomycin "
                     "and related drugs"),
    (r"^aph\(", "aminoglycoside phosphotransferase — inactivates kanamycin and "
                "related drugs"),
    # --- quinolone ---
    (r"^qnr", "plasmid-borne quinolone protection — reduces ciprofloxacin activity"),
    (r"^oqx", "efflux pump — pumps quinolones and other drugs out of the cell"),
    # --- folate pathway ---
    (r"^sul", "sulfonamide-resistant enzyme — bypasses the drug target of "
              "sulfamethoxazole"),
    (r"^dfr", "trimethoprim-resistant enzyme — bypasses the drug target of trimethoprim"),
    # --- other ---
    (r"^fosA", "fosfomycin-modifying enzyme — present in nearly every K. pneumoniae"),
    (r"^cat|^floR", "chloramphenicol resistance"),
    (r"^arr", "rifampicin-modifying enzyme"),
    (r"^tet\(", "tetracycline efflux or ribosomal protection"),
    (r"^mcr", "colistin resistance"),
    (r"^emr|^mdt|^acr", "efflux pump — present in nearly every K. pneumoniae; weak on "
                        "its own"),
]

MUTATION_DESCRIPTIONS: list[tuple[str, str]] = [
    (r"^gyrA", "mutation in gyrA, the target of fluoroquinolones — reduces "
               "ciprofloxacin binding"),
    (r"^parC", "mutation in parC, the second fluoroquinolone target — adds to "
               "ciprofloxacin resistance"),
    (r"^ompK35", "loss or damage of porin OmpK35 — fewer channels for beta-lactams to "
                 "enter the cell"),
    (r"^ompK36", "loss or damage of porin OmpK36 — a main route to carbapenem "
                 "resistance when a beta-lactamase is also present"),
    (r"^pmrB|^phoQ|^mgrB", "mutation linked to colistin resistance"),
    (r"^rpoB", "mutation in rpoB, the target of rifampicin"),
    (r"^ramR|^acrR|^marR", "regulator mutation — can increase efflux pump activity"),
]


def _match(symbol: str, table: list[tuple[str, str]]) -> str | None:
    for pattern, text in table:
        if re.match(pattern, symbol):
            return text
    return None


def describe_feature(feature: str) -> str:
    """One clause a clinician can read. Never invents a mechanism it does not know."""
    kind, _, name = feature.partition(":")

    if kind in ("gene", "genefam"):
        return _match(name, GENE_DESCRIPTIONS) or "resistance-associated gene"
    if kind in ("mut", "mutgene"):
        gene = name.split("_", 1)[0]
        return _match(gene, MUTATION_DESCRIPTIONS) or f"mutation in {gene}"
    if kind == "trunc":
        gene = name.split("_", 1)[0]
        base = _match(gene, MUTATION_DESCRIPTIONS)
        return base or f"{gene} disrupted — the protein is truncated"
    if kind == "class":
        return f"determinant of the {name.lower().replace('-', ' ')} class"
    return "resistance-associated feature"


def gene_label(feature: str) -> str:
    """Human-facing name: strip the internal prefix, keep the biology."""
    kind, _, name = feature.partition(":")
    if kind == "genefam":
        return f"{name} family"
    if kind == "trunc":
        return f"{name.split('_', 1)[0]} (disrupted)"
    if kind in ("mut", "mutgene"):
        return name.replace("_", " ")
    if kind == "class":
        return name.title()
    return name


def detected_mechanisms(supporting: list[dict]) -> list[str]:
    """Curated determinants that are an established mechanism FOR THIS DRUG.

    Reported independently of what the model concluded. Gene detection is a
    deterministic observation; the probability is a model opinion. When the two
    disagree the observation still has to reach the reader.
    """
    out = []
    for s in supporting:
        if (s.get("mechanistic_for_drug") and s.get("curated_determinant")
                and s.get("direction") == "toward_resistant"):
            out.append(f"**{gene_label(s['feature'])}** ({describe_feature(s['feature'])})")
    return out


def headline_evidence(supporting: list[dict], call: str) -> str:
    """The single most important sentence for this drug.

    Picks the strongest feature that argues for the reported call, prefers one
    that is an established mechanism for the drug in question, and renders it as
    a sentence. Falls back to an honest statement of absence.
    """
    if call == "no_call":
        # A no-call must never imply "nothing was found". The genome that
        # triggered this branch carried blaKPC and was laboratory-confirmed
        # resistant, while the report said no mechanism was detected.
        found = detected_mechanisms(supporting)
        if found:
            return ("The model is not confident, but a resistance mechanism WAS "
                    "detected: " + "; ".join(found) + ".")
        return "No resistance mechanism was detected, but the evidence is too weak to call."

    want = "toward_resistant" if call == "likely_to_fail" else "toward_susceptible"
    aligned = [s for s in supporting if s.get("direction") == want]
    if not aligned:
        if call == "likely_to_work":
            return "No known resistance mechanism for this drug was found in the genome."
        return "No single dominant marker; the call rests on the overall pattern."

    mechanistic = [s for s in aligned if s.get("mechanistic_for_drug")]
    pick = max(mechanistic or aligned, key=lambda s: abs(s.get("weight", 0)))

    label = gene_label(pick["feature"])
    what = describe_feature(pick["feature"])
    if call == "likely_to_fail":
        if pick.get("mechanistic_for_drug"):
            return f"**{label}** detected — {what}."
        return (f"**{label}** detected — {what}. Note: this is not a mechanism for "
                f"this particular drug; it travels with resistance genes on the same "
                f"plasmids.")
    return "No known resistance mechanism for this drug was found in the genome."


def supporting_sentences(supporting: list[dict], call: str, limit: int = 3) -> list[str]:
    """Remaining findings, one clause each, strongest first."""
    want = "toward_resistant" if call == "likely_to_fail" else "toward_susceptible"
    aligned = sorted((s for s in supporting if s.get("direction") == want),
                     key=lambda s: -abs(s.get("weight", 0)))
    return [f"{gene_label(s['feature'])} — {describe_feature(s['feature'])}"
            for s in aligned[1:limit + 1]]
