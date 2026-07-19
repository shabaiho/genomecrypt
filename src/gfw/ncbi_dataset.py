"""FAST PATH -- a pre-decoded training set, no AMRFinderPlus run required.

NCBI Pathogen Detection publishes, for every isolate it has processed:
  * AMR_genotypes   -- AMRFinderPlus element symbols (the decoding, already done)
  * AST_phenotypes  -- the lab antibiogram (the labels)
  * PDS_acc         -- SNP cluster id (ready-made genetic-relatedness groups)

That is features + labels + grouping from two file downloads, replacing the ~5h
annotation step. The feature space is IDENTICAL to what gfw.annotate produces
locally, so a model trained here serves FASTA uploads at inference unchanged.

    python -m gfw.ncbi_dataset --organism Klebsiella

Verified for Klebsiella snapshot PDG000000012.2470:
    167,247 isolates -> 2,612 with both phenotype and genotype
    ciprofloxacin 1350 | gentamicin 1285 | meropenem 1345
    ceftriaxone 1599   | trim/sulfa 1316
    class balance 0.46-0.58 resistant -- far healthier than BV-BRC's 0.83 skew
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, Config
from .features import aggregate_mutations, select_schema

FTP = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results"

# AMRFinderPlus suffixes that qualify a hit rather than name a new determinant.
# `gyrA_S83I=POINT` -> mut:gyrA_S83I ; `blaOXA=MISTRANSLATION` -> dropped, since a
# mistranslated call is not evidence of a functional gene.
POINT_SUFFIXES = {"POINT"}
DROP_SUFFIXES = {"MISTRANSLATION", "PARTIAL", "PARTIAL_CONTIG_END", "INTERNAL_STOP", "HMM"}


def latest_snapshot(organism: str) -> str:
    with urllib.request.urlopen(f"{FTP}/{organism}/", timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    import re

    tags = sorted(set(re.findall(r"PDG\d+\.\d+", html)),
                  key=lambda s: [int(x) for x in s.replace("PDG", "").split(".")])
    if not tags:
        raise RuntimeError(f"no PDG snapshot found for {organism}")
    return tags[-1]


def download(organism: str, snapshot: str, raw: Path) -> tuple[Path, Path]:
    raw.mkdir(parents=True, exist_ok=True)
    amr = raw / f"{snapshot}.amr.metadata.tsv"
    clu = raw / f"{snapshot}.clusters.tsv"
    base = f"{FTP}/{organism}/{snapshot}"
    for url, dest, note in (
        (f"{base}/AMR/{snapshot}.amr.metadata.tsv", amr, "~175MB"),
        (f"{base}/Clusters/{snapshot}.reference_target.all_isolates.tsv", clu, "~6MB"),
    ):
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"cached {dest.name}")
            continue
        print(f"downloading {dest.name} ({note}) ...", flush=True)
        urllib.request.urlretrieve(url, dest)
    return amr, clu


def parse_genotypes(s: str) -> set[str]:
    """`aac(6')-Ib,blaKPC-2,gyrA_S83I=POINT` -> {gene:..., mut:...} tokens.

    Matches gfw.features exactly so training features and inference features
    live in the same space.
    """
    toks: set[str] = set()
    for raw in str(s).split(","):
        raw = raw.strip()
        if not raw:
            continue
        sym, _, qual = raw.partition("=")
        sym, qual = sym.strip(), qual.strip().upper()
        if not sym or qual in DROP_SUFFIXES:
            continue
        toks.add(f"{'mut' if qual in POINT_SUFFIXES else 'gene'}:{sym}")
    # identical rollups to gfw.features, so both paths share one feature space
    return aggregate_mutations(toks)


def parse_ast(s: str, label_map: dict[str, int]) -> dict[str, int]:
    """`ciprofloxacin=R,gentamicin=S,...` -> {drug: 0/1}. Intermediate is dropped."""
    out: dict[str, int] = {}
    for raw in str(s).split(","):
        drug, _, ph = raw.strip().partition("=")
        if not drug or not ph:
            continue
        # values look like R / S / I, sometimes with a trailing method qualifier
        code = ph.split("=")[0].strip()
        for key, val in label_map.items():
            if code.upper() == key.upper()[:1] or code.lower() == key.lower():
                out[drug.strip().lower()] = val
                break
    return out


def build(organism: str = "Klebsiella", min_prevalence: int = 3) -> dict:
    cfg = Config.load()
    raw = REPO_ROOT / "data" / "raw" / "ncbi"
    snapshot = latest_snapshot(organism)
    print(f"snapshot: {snapshot}")
    amr_path, clu_path = download(organism, snapshot, raw)

    amr = pd.read_csv(amr_path, sep="\t", dtype=str, low_memory=False,
                      usecols=["target_acc", "asm_acc", "scientific_name",
                               "AST_phenotypes", "AMR_genotypes"])
    amr = amr.dropna(subset=["AST_phenotypes", "AMR_genotypes"])
    clusters = pd.read_csv(clu_path, sep="\t", dtype=str, low_memory=False,
                           usecols=["target_acc", "PDS_acc"])
    df = amr.merge(clusters, on="target_acc", how="left")
    print(f"{len(df)} isolates with phenotype + genotype")

    # our drug ids use "_" where NCBI uses "-" / "/"
    want = {d.id: d.id.replace("_", "-") for d in cfg.drugs}
    alt = {d.id: d.id.replace("_", "/") for d in cfg.drugs}

    rows_lab, tok_by_iso = [], {}
    for _, r in df.iterrows():
        iso = r.target_acc
        tok_by_iso[iso] = parse_genotypes(r.AMR_genotypes)
        ast = parse_ast(r.AST_phenotypes, cfg.label_map)
        for did, ncbi_name in want.items():
            v = ast.get(ncbi_name, ast.get(alt[did], ast.get(did)))
            if v is not None:
                rows_lab.append({"genome_id": iso, "drug_id": did, "label": int(v)})

    labels = pd.DataFrame(rows_lab)

    # keep only isolates that ended up with at least one usable label
    keep = set(labels.genome_id)
    tok_by_iso = {k: v for k, v in tok_by_iso.items() if k in keep}

    counts: dict[str, int] = {}
    for toks in tok_by_iso.values():
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    schema = select_schema(counts, len(tok_by_iso), min_prevalence)

    idx = {t: i for i, t in enumerate(schema)}
    import numpy as np

    ids = sorted(tok_by_iso)
    X = np.zeros((len(ids), len(schema)), dtype=np.int8)
    for i, iso in enumerate(ids):
        for t in tok_by_iso[iso]:
            j = idx.get(t)
            if j is not None:
                X[i, j] = 1
    features = pd.DataFrame(X, index=ids, columns=schema)

    # groups: NCBI SNP clusters. Isolates with no cluster get their own singleton
    # group, which is the conservative choice -- never merge unknowns together.
    gmap = df.set_index("target_acc").PDS_acc.to_dict()
    groups = pd.DataFrame({
        "genome_id": ids,
        "group_id": [gmap.get(i) if isinstance(gmap.get(i), str) else f"_solo_{i}"
                     for i in ids],
    })
    groups["group_id"] = pd.factorize(groups.group_id)[0]

    # assembly accession per isolate -- used to fetch a demo FASTA later
    asm = df.set_index("target_acc").asm_acc.to_dict()

    return {"features": features, "labels": labels, "groups": groups,
            "asm": asm, "snapshot": snapshot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", default="Klebsiella")
    ap.add_argument("--min-prevalence", type=int, default=3)
    args = ap.parse_args()

    out = build(args.organism, args.min_prevalence)
    proc = REPO_ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    out["features"].to_parquet(proc / "features.parquet")
    out["labels"].to_csv(proc / "labels.csv", index=False)
    out["groups"].to_csv(proc / "groups.csv", index=False)
    pd.Series(out["asm"]).rename("asm_acc").to_csv(proc / "assembly_accessions.csv")

    f, lb, g = out["features"], out["labels"], out["groups"]
    print(f"\nfeatures {f.shape[0]} x {f.shape[1]} -> {proc / 'features.parquet'}")
    print(f"groups: {g.group_id.nunique()} clusters, largest "
          f"{g.group_id.value_counts().iloc[0]}")
    print(lb.groupby("drug_id").label.agg(n="size", resistant_frac="mean").round(3))


if __name__ == "__main__":
    main()
