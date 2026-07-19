"""Plain-language descriptions of what a detected gene does.

Two rules, both deliberate.

OBJECTIVE ONLY. No "some", "weak", "strong", "nearly every". Those words move the
reader's judgement without telling them anything checkable. Where prevalence
matters it is stated as a measured number from the reference set, not as an
adjective. Where a gene's effect is partial, the drugs are named instead of
being called "some".

NO ADVICE. The report states what was found and what the model concluded. It does
not tell a clinician what to prescribe.

Only 54 features carry a non-zero weight in any shipped model, so coverage here is
small and finite. Patterns are ordered most specific first; anything unmatched
falls back to a neutral phrase rather than inventing biology.
"""
from __future__ import annotations

import re

# (regex on the symbol, description) -- checked in order, first match wins.
GENE_DESCRIPTIONS: list[tuple[str, str]] = [
    # --- carbapenemases ---
    (r"^blaKPC", "carbapenemase KPC — hydrolyses carbapenems, cephalosporins and penicillins"),
    (r"^blaNDM", "metallo-carbapenemase NDM — hydrolyses carbapenems; not inhibited by "
                 "clavulanate, tazobactam or avibactam"),
    (r"^blaVIM|^blaIMP", "metallo-carbapenemase — hydrolyses carbapenems"),
    (r"^blaOXA-48|^blaOXA-181|^blaOXA-232", "carbapenemase OXA-48-like — hydrolyses carbapenems"),
    (r"^blaOXA", "beta-lactamase OXA family — hydrolyses penicillins"),
    # --- extended-spectrum and other beta-lactamases ---
    (r"^blaCTX-M", "extended-spectrum beta-lactamase CTX-M — hydrolyses ceftriaxone, "
                   "cefotaxime and ceftazidime"),
    (r"^blaSHV-12|^blaSHV-2|^blaSHV-5", "extended-spectrum beta-lactamase SHV — "
                                        "hydrolyses ceftriaxone and ceftazidime"),
    (r"^blaSHV", "beta-lactamase SHV — hydrolyses ampicillin"),
    (r"^blaTEM", "beta-lactamase TEM — hydrolyses ampicillin"),
    (r"^blaCMY|^blaDHA|^blaACT|^blaFOX", "AmpC beta-lactamase — hydrolyses ceftriaxone "
                                         "and cefoxitin; not inhibited by clavulanate"),
    (r"^blaLAP|^blaEC|^bla", "beta-lactamase — hydrolyses beta-lactam antibiotics"),
    # --- aminoglycoside modifying enzymes ---
    (r"^rmt|^armA", "16S rRNA methyltransferase — blocks gentamicin, tobramycin and amikacin"),
    (r"^aac\(6'\)-Ib-cr", "acetyltransferase — inactivates amikacin and tobramycin, "
                          "and reduces ciprofloxacin activity"),
    (r"^aac\(3\)", "acetyltransferase — inactivates gentamicin and tobramycin"),
    (r"^aac\(6'\)", "acetyltransferase — inactivates amikacin and tobramycin"),
    (r"^ant\(|^aad", "nucleotidyltransferase — inactivates streptomycin and spectinomycin"),
    (r"^aph\(", "phosphotransferase — inactivates kanamycin and neomycin"),
    # --- quinolone ---
    (r"^qnr", "Qnr protein — shields DNA gyrase from ciprofloxacin"),
    (r"^oqx", "OqxAB efflux pump — exports ciprofloxacin and chloramphenicol"),
    # --- folate pathway ---
    (r"^sul", "sulfonamide-resistant dihydropteroate synthase — replaces the target of "
              "sulfamethoxazole"),
    (r"^dfr", "trimethoprim-resistant dihydrofolate reductase — replaces the target of "
              "trimethoprim"),
    # --- other ---
    (r"^fosA", "fosfomycin-modifying enzyme — inactivates fosfomycin"),
    (r"^cat|^floR", "chloramphenicol resistance enzyme — inactivates chloramphenicol"),
    (r"^arr", "ADP-ribosyltransferase — inactivates rifampicin"),
    (r"^tet\(", "tetracycline efflux pump — exports tetracycline"),
    (r"^mcr", "phosphoethanolamine transferase — reduces colistin binding"),
    (r"^emr|^mdt|^acr", "efflux pump — exports multiple drug classes"),
]

MUTATION_DESCRIPTIONS: list[tuple[str, str]] = [
    (r"^gyrA", "mutation in gyrA, the primary target of ciprofloxacin — reduces "
               "drug binding"),
    (r"^parC", "mutation in parC, the secondary target of ciprofloxacin — reduces "
               "drug binding"),
    (r"^ompK35", "porin OmpK35 altered or lost — reduces beta-lactam entry into the cell"),
    (r"^ompK36", "porin OmpK36 altered or lost — reduces carbapenem entry into the cell"),
    (r"^pmrB|^phoQ|^mgrB", "mutation in a colistin-resistance regulator"),
    (r"^rpoB", "mutation in rpoB, the target of rifampicin — reduces drug binding"),
    (r"^ramR|^acrR|^marR", "mutation in an efflux-pump regulator — increases pump expression"),
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
