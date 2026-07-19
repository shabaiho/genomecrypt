"""Which determinants are a MECHANISM for a drug, and which are merely linked.

Resistance genes travel together on plasmids, so a carbapenemase genuinely
predicts ciprofloxacin resistance in this dataset -- by linkage, not pharmacology.
MEASURED: restricting each drug to its own pharmacological class costs 0.025 AUROC
on average (scripts/exp_simplicity.py), so throwing the linked markers away makes
the model worse.

The honest answer is therefore not to remove them but to LABEL them. A feature can
be a curated AMR determinant and still be mechanistically irrelevant to the drug
in question; the report says which, so nobody reads a linkage marker as a cause.
"""
from __future__ import annotations

# AMRFinderPlus Class values relevant to each drug, plus chromosomal genes whose
# mutation is an established mechanism for it.
DRUG_MECHANISM = {
    "ciprofloxacin": ({"QUINOLONE", "PHENICOL/QUINOLONE"}, {"gyrA", "parC", "gyrB", "parE"}),
    "gentamicin": ({"AMINOGLYCOSIDE"}, {"rpsL"}),
    "meropenem": ({"BETA-LACTAM", "CARBAPENEM", "CEPHALOSPORIN"}, {"ompK35", "ompK36", "ftsI"}),
    "ceftriaxone": ({"BETA-LACTAM", "CEPHALOSPORIN"}, {"ompK35", "ompK36", "ftsI"}),
    "trimethoprim_sulfamethoxazole": ({"TRIMETHOPRIM", "SULFONAMIDE"}, {"folA", "folP"}),
}

# gene-symbol prefixes belonging to a class, for tokens carrying no class rollup
CLASS_PREFIXES = {
    "QUINOLONE": ("qnr", "oqx", "aac(6\')-Ib-cr"),
    "AMINOGLYCOSIDE": ("aac", "aad", "ant", "aph", "rmt", "arm", "str"),
    "BETA-LACTAM": ("bla", "amp", "ompK"),
    "CEPHALOSPORIN": ("bla",),
    "CARBAPENEM": ("bla",),
    "TRIMETHOPRIM": ("dfr",),
    "SULFONAMIDE": ("sul",),
}


def is_mechanistic(feature: str, drug_id: str) -> bool:
    """True if this feature is an established mechanism for THIS drug."""
    spec = DRUG_MECHANISM.get(drug_id)
    if spec is None:
        return False
    classes, genes = spec
    kind, _, name = feature.partition(":")
    if kind == "class":
        return name.upper() in classes
    if kind in ("mut", "mutgene", "trunc"):
        return name.split("_")[0] in genes
    if kind in ("gene", "genefam"):
        prefixes = tuple(p for c in classes for p in CLASS_PREFIXES.get(c, ()))
        return bool(prefixes) and name.startswith(prefixes)
    return False
