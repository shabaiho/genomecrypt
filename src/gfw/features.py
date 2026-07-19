"""Module 01b -- AMRFinderPlus TSV -> fixed-width binary feature vector.

FEATURE CONTRACT (feature_schema.json)
--------------------------------------
Features are binary presence/absence, one column per determinant:
    gene:<Element symbol>        e.g. gene:blaKPC-2, gene:aac(6')-Ib-cr
    mut:<Element symbol>         e.g. mut:gyrA_S83L        (Element type == POINT)
    class:<Class>                e.g. class:BETA-LACTAM    (rolled-up class hit count > 0)

The ordered `features` list in feature_schema.json is the ONLY thing the served
models depend on. Extraction at inference time must reproduce it exactly:
unknown symbols are dropped (and counted, for the OOD check), missing ones are 0.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical internal column names. AMRFinderPlus renamed several columns between
# v3 and v4 (verified against v4.2.7 / DB 2026-05-15.1), so we normalize on read
# and every downstream module sees one stable schema.
COL_SYMBOL = "Element symbol"
COL_TYPE = "Element type"
COL_SUBTYPE = "Element subtype"
COL_CLASS = "Class"
COL_COVERAGE = "pct_coverage"
COL_IDENTITY = "pct_identity"

# {name as written by some AMRFinderPlus version} -> canonical
COLUMN_ALIASES = {
    # v4.x
    "Type": COL_TYPE,
    "Subtype": COL_SUBTYPE,
    "% Coverage of reference": COL_COVERAGE,
    "% Identity to reference": COL_IDENTITY,
    # v3.x
    "Element type": COL_TYPE,
    "Element subtype": COL_SUBTYPE,
    "% Coverage of reference sequence": COL_COVERAGE,
    "% Identity to reference sequence": COL_IDENTITY,
    "Gene symbol": COL_SYMBOL,
    "Sequence name": "name",
}

# quality floor -- a 40%-coverage hit is noise, not a resistance gene
MIN_COVERAGE = 50.0
MIN_IDENTITY = 90.0


class WrongFileType(ValueError):
    """Raised when a file is not what the caller thought it was."""


def sniff_file_type(path: Path) -> str:
    """'fasta' | 'amrfinder_tsv' | 'unknown', decided by CONTENT not by extension.

    This exists because the failure it prevents is silent and dangerous: feeding a
    FASTA to the TSV reader yields a 68,000-row frame with no matching columns,
    hence ZERO determinants, hence a confident "likely to work" for every drug --
    on a genome that actually carries blaKPC. Never trust the user's mode toggle.
    """
    try:
        with Path(path).open("r", errors="replace") as fh:
            # skip leading blank lines -- some assemblers and every round-trip
            # through a text editor can leave them, and reading only line 1 then
            # reported a perfectly good genome as an unknown file type
            lines: list[str] = []
            for _ in range(64):
                ln = fh.readline()
                if not ln:
                    break
                if ln.strip() or lines:
                    lines.append(ln)
                if len(lines) >= 8:
                    break
    except OSError:
        return "unknown"

    if not lines:
        return "unknown"

    if lines[0].lstrip().startswith(">"):
        # nucleotide or protein? amrfinder is invoked in --nucleotide mode, so a
        # protein FASTA would be garbage in, garbage out
        seq = "".join(ln.strip() for ln in lines[1:] if not ln.startswith(">"))
        if seq:
            acgtn = sum(c in "ACGTUNacgtun" for c in seq)
            if acgtn / len(seq) < 0.85:
                return "protein_fasta"
        return "fasta"

    cols = {c.strip() for c in lines[0].split("\t")}
    # any AMRFinderPlus version identifies itself by these column names
    if cols & {"Element symbol", "Gene symbol", "Protein identifier", "Protein id"}:
        return "amrfinder_tsv"
    return "unknown"


def require_file_type(path: Path, expected: str) -> None:
    actual = sniff_file_type(path)
    if actual != expected:
        raise WrongFileType(
            f"expected {expected}, got {actual}. Parsing it anyway would produce "
            f"an empty feature vector and a confidently wrong prediction."
        )


def parse_amrfinder_tsv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    except pd.errors.EmptyDataError as e:
        # an empty file is not "a genome with no resistance genes"
        raise WrongFileType(f"{path.name} is empty -- no AMRFinderPlus output to read") from e
    df = df.rename(columns=COLUMN_ALIASES)
    for c in (COL_COVERAGE, COL_IDENTITY):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in (COL_SYMBOL, COL_TYPE, COL_SUBTYPE, COL_CLASS):
        if c not in df.columns:
            df[c] = ""
    return df


# A truncating change in a porin destroys the channel the drug enters through --
# biologically very different from a substitution, and the main non-carbapenemase
# route to carbapenem resistance in K. pneumoniae.
TRUNCATING = re.compile(r"(fs|Ter|STOP|del|ins|dup)", re.IGNORECASE)


# blaKPC-2 and blaKPC-3 differ by one residue and are the same carbapenemase
# clinically. As separate features L1 keeps whichever the training set favoured
# and zeroes the other -- which produced a real false-susceptible: the demo
# genome carries blaKPC-2, the model had learned blaKPC-3, and a
# carbapenemase-positive isolate was called "likely to work".
ALLELE_SUFFIX = re.compile(r"(-\d+(\.\d+)?|\.\d+)$")


# OXA is not one enzyme. blaOXA-48, -181, -204 and -232 are carbapenemases;
# blaOXA-1, -2, -9 and -10 are narrow-spectrum beta-lactamases. Rolling them into
# a single blaOXA family averaged two incompatible mechanisms into one
# coefficient, and the meropenem model was using it. The distinction cannot be
# read off the string, so it is stated.
OXA_CARBAPENEMASES = {"48", "162", "163", "181", "199", "204", "232", "244", "245",
                      "247", "370", "405", "436", "438", "484", "505", "517", "519",
                      "535", "538", "546", "547", "566", "567", "793"}


def gene_family(symbol: str) -> str:
    """blaKPC-2 -> blaKPC ; blaCTX-M-15 -> blaCTX-M ; blaOXA-48 -> blaOXA-48-like

    Families are derived from the symbol, which works because an allele number
    usually denotes a variant of one enzyme. Where that is not true the exception
    is written down rather than inferred.
    """
    if symbol.startswith("blaOXA-"):
        allele = symbol.split("-", 1)[1].split(".")[0]
        return "blaOXA-48-like" if allele in OXA_CARBAPENEMASES else "blaOXA"
    return ALLELE_SUFFIX.sub("", symbol)


def aggregate_mutations(tokens: set[str]) -> set[str]:
    """Add per-gene rollups for point mutations.

    MEASURED on the NCBI Klebsiella snapshot: 294 distinct point mutations, 73% of
    them seen fewer than 3 times, so min_prevalence drops almost all of them
    individually. But ompK35 is hit 746 times across 75 variants and ompK36 419
    times across 54 -- the porin-loss signal only exists once you aggregate.

        mut:ompK36_K231SfsTer16  ->  + mutgene:ompK36  + trunc:ompK36

    The exact-variant token is kept as well: gyrA_S83L is common enough to learn
    on its own and is a well-established quinolone determinant.
    """
    extra: set[str] = set()
    for t in tokens:
        if t.startswith("gene:"):
            fam = gene_family(t[5:])
            if fam and fam != t[5:]:
                extra.add(f"genefam:{fam}")
            continue
        if not t.startswith("mut:"):
            continue
        variant = t[4:]
        gene = variant.split("_", 1)[0]
        if not gene:
            continue
        extra.add(f"mutgene:{gene}")
        detail = variant[len(gene):]
        if TRUNCATING.search(detail):
            extra.add(f"trunc:{gene}")
    return tokens | extra


def _is_point(df: pd.DataFrame) -> pd.Series:
    """Point mutations in v4 are Type=AMR with Subtype=POINT / POINT_DISRUPT --
    NOT Type=POINT as in v3. Getting this wrong silently files every gyrA_S83L
    under `gene:` instead of `mut:`, which breaks the evidence categories."""
    return (df[COL_SUBTYPE].str.upper().str.startswith("POINT")
            | df[COL_TYPE].str.upper().eq("POINT"))


def determinants(df: pd.DataFrame) -> set[str]:
    """TSV rows -> set of feature tokens for ONE genome."""
    if df.empty:
        return set()
    point = _is_point(df)
    keep = df
    if COL_COVERAGE in df.columns and COL_IDENTITY in df.columns:
        qual = (df[COL_COVERAGE].fillna(0) >= MIN_COVERAGE) & (
            df[COL_IDENTITY].fillna(0) >= MIN_IDENTITY
        )
        # point mutations have no meaningful coverage -- never filter them out
        keep = df[qual | point]

    # AMR determinants only: --plus also emits VIRULENCE / STRESS rows we don't want
    keep = keep[keep[COL_TYPE].str.upper().isin({"AMR", "POINT"})]

    toks: set[str] = set()
    for i, r in keep.iterrows():
        sym = r[COL_SYMBOL].strip()
        if not sym:
            continue
        prefix = "mut" if point.loc[i] else "gene"
        toks.add(f"{prefix}:{sym}")
        for cls in str(r.get(COL_CLASS, "")).split("/"):
            cls = cls.strip().upper()
            if cls:
                toks.add(f"class:{cls}")
    return aggregate_mutations(toks)


def select_schema(counts: dict[str, int], n_samples: int,
                  min_prevalence: int = 3, max_prevalence: float = 0.95) -> list[str]:
    """Ordered feature list, dropping determinants that cannot discriminate.

    Two-sided filter, and the upper side matters more than it looks. A binary
    feature present in a fraction q of samples has variance q(1-q); at q=0.999
    that is 0.001, i.e. it is a constant. `gene:emrD` sits in 99.9% of these
    genomes, carries no information, and yet L1 handed it weight -1.18 in the
    meropenem model -- second largest by magnitude, acting as a second intercept
    while presenting itself as evidence of susceptibility. It offset blaKPC
    (+1.66) and flipped a carbapenemase-positive genome to "likely to work".
    """
    return sorted(t for t, n in counts.items()
                  if n >= min_prevalence and n / max(1, n_samples) <= max_prevalence)


def build_matrix(
    tsv_paths: dict[str, Path],
    min_prevalence: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """Many genomes -> (genome_id x feature) binary DataFrame + ordered schema.

    Features seen in fewer than `min_prevalence` genomes are dropped: a
    determinant present once cannot be learned from and only inflates variance.
    """
    per_genome = {gid: determinants(parse_amrfinder_tsv(p)) for gid, p in tsv_paths.items()}
    counts: dict[str, int] = {}
    for toks in per_genome.values():
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    schema = select_schema(counts, len(per_genome), min_prevalence)

    idx = {t: i for i, t in enumerate(schema)}
    X = np.zeros((len(per_genome), len(schema)), dtype=np.int8)
    ids = list(per_genome)
    for r, gid in enumerate(ids):
        for t in per_genome[gid]:
            j = idx.get(t)
            if j is not None:
                X[r, j] = 1
    return pd.DataFrame(X, index=ids, columns=schema), schema


def vectorize(tokens: set[str], schema: list[str]) -> tuple[np.ndarray, list[str]]:
    """Inference-time counterpart of build_matrix. Returns (vector, unknown_tokens).

    `unknown` drives the OOD no-call: a genome carrying determinants the model
    never saw is exactly the case where we should refuse to predict.
    """
    idx = {t: i for i, t in enumerate(schema)}
    v = np.zeros(len(schema), dtype=np.int8)
    unknown = []
    for t in tokens:
        j = idx.get(t)
        if j is None:
            if t.startswith(("gene:", "mut:")):  # class: rollups don't count as novel
                unknown.append(t)
        else:
            v[j] = 1
    return v.reshape(1, -1), sorted(unknown)
