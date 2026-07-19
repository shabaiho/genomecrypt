"""Add BV-BRC-labelled isolates to the NCBI dataset, without duplicates.

WHY. NCBI Pathogen Detection carries AMR_genotypes for 167k Klebsiella isolates
but an antibiogram for only ~2.6k -- the phenotype is the binding constraint, not
the genotype. BV-BRC publishes ~19k lab-measured phenotype records for the same
species. Where a BV-BRC genome maps to an NCBI assembly we already have a decoded
genotype for, we gain a labelled training row for zero annotation cost.

MEASURED YIELD: 587 genomes and 2,321 genome-drug labels on top of 1,992 (+29%).

DEDUPLICATION, three levels:
  1. exact isolate  -- join on assembly accession with the version stripped
     (GCA_000123456.1 and .2 are the same assembly), and drop anything already
     present in the NCBI AST set
  2. label conflict -- if the two sources disagree for one (genome, drug), drop
     the pair rather than pick a winner
  3. genetic        -- SNP cluster (PDS_acc) still drives the grouped split, so
     near-identical genomes cannot span train and test regardless of source

HONEST CAVEAT: on the 872 overlapping (assembly, drug) pairs the two sources agree
100%. That is not evidence of label quality -- it means BV-BRC and NCBI are not
independent and almost certainly share provenance. Merging buys volume, NOT a
second opinion, and the disagreement rate here cannot be used to estimate label
noise.
"""
from __future__ import annotations

import argparse
import json
import urllib.parse

import pandas as pd

from .config import REPO_ROOT, Config, write_json
from .download_data import LAB_METHODS, _get


def strip_version(acc: str) -> str:
    return acc.split(".")[0] if isinstance(acc, str) else ""


def fetch_bvbrc_labels(cfg: Config) -> pd.DataFrame:
    """Lab-measured phenotypes for our species and drug panel, with assembly ids."""
    drug_names = sorted({d.id.replace("_", "/") for d in cfg.drugs})
    in_drugs = "in(antibiotic,(%s))" % ",".join(
        urllib.parse.quote(f'"{d}"', safe="") for d in drug_names)
    in_methods = "in(laboratory_typing_method,(%s))" % ",".join(
        urllib.parse.quote(f'"{m}"') for m in LAB_METHODS)
    species = urllib.parse.quote(cfg.species)

    rows, offset = [], 0
    while True:
        q = (f"eq(genome_name,{species}*)&{in_drugs}&{in_methods}"
             f"&select(genome_id,antibiotic,resistant_phenotype)&limit(5000,{offset})")
        batch = json.loads(_get("genome_amr", q))
        rows.extend(batch)
        print(f"  {len(rows)} phenotype records", flush=True)
        if len(batch) < 5000:
            break
        offset += 5000

    df = pd.DataFrame(rows)
    df["drug_id"] = (df.antibiotic.fillna("").str.lower()
                     .str.replace("/", "_", regex=False)
                     .str.replace("-", "_", regex=False))
    df = df[df.drug_id.isin({d.id for d in cfg.drugs})]
    df["label"] = df.resistant_phenotype.map(cfg.label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df.label.astype(int)

    ids = sorted(df.genome_id.unique())
    amap: dict[str, str] = {}
    for i in range(0, len(ids), 200):
        q = ("in(genome_id,(%s))&select(genome_id,assembly_accession)&limit(1000)"
             % ",".join(ids[i:i + 200]))
        try:
            for r in json.loads(_get("genome", q)):
                a = r.get("assembly_accession")
                if isinstance(a, str) and a.startswith("GC"):
                    amap[r["genome_id"]] = strip_version(a)
        except Exception as e:
            print(f"  assembly batch {i} failed: {e}")
    df["asm_base"] = df.genome_id.map(amap)
    return df.dropna(subset=["asm_base"])


def merge(ncbi: dict, bv: pd.DataFrame, min_prevalence: int = 3) -> dict:
    """Fold BV-BRC labels into an already-built NCBI dataset dict."""
    from .features import select_schema  # shared with the NCBI path
    from .ncbi_dataset import parse_genotypes

    snapshot = ncbi["snapshot"]
    raw = pd.read_csv(REPO_ROOT / f"data/raw/ncbi/{snapshot}.amr.metadata.tsv",
                      sep="\t", dtype=str, low_memory=False,
                      usecols=["target_acc", "asm_acc", "AST_phenotypes", "AMR_genotypes"])
    clusters = pd.read_csv(REPO_ROOT / f"data/raw/ncbi/{snapshot}.clusters.tsv",
                           sep="\t", dtype=str, low_memory=False,
                           usecols=["target_acc", "PDS_acc"])
    raw = raw.merge(clusters, on="target_acc", how="left")
    raw["asm_base"] = raw.asm_acc.map(strip_version)

    have_geno = raw.dropna(subset=["AMR_genotypes"])
    have_geno = have_geno[have_geno.asm_base != ""]
    # level 1: anything already carrying an antibiogram is not "new"
    already = set(have_geno.dropna(subset=["AST_phenotypes"]).asm_base)

    geno_row = {}
    for _, r in have_geno.iterrows():
        geno_row.setdefault(r.asm_base, r)

    new_labels, new_tokens, new_groups = [], {}, {}
    for asm_base, g in bv.groupby("asm_base"):
        if asm_base in already or asm_base not in geno_row:
            continue
        row = geno_row[asm_base]
        iso = f"BV:{asm_base}"
        new_tokens[iso] = parse_genotypes(row.AMR_genotypes)
        pds = row.PDS_acc if isinstance(row.PDS_acc, str) else f"_solo_{iso}"
        new_groups[iso] = pds
        # level 2: a (genome, drug) pair whose sources disagree is dropped
        for drug_id, gg in g.groupby("drug_id"):
            if gg.label.nunique() != 1:
                continue
            new_labels.append({"genome_id": iso, "drug_id": drug_id,
                               "label": int(gg.label.iloc[0])})

    if not new_labels:
        print("no new rows found")
        return ncbi

    add_labels = pd.DataFrame(new_labels)
    print(f"adding {len(new_tokens)} genomes / {len(add_labels)} labels from BV-BRC")

    # Rebuild tokens for BOTH sources from the RAW genotypes, then filter once
    # over the union.
    #
    # BUG THIS FIXES: reading the base tokens back out of ncbi["features"] meant
    # they had already been through select_schema, while the BV-BRC genomes were
    # parsed fresh. Anything the base filter had dropped therefore appeared in the
    # BV-BRC rows only -- `gene:emrD`, present in 99.9% of genomes, came out at
    # prevalence 0.2276, exactly the BV-BRC share of the merged set. It had become
    # a marker for WHICH DATABASE the genome came from, and the meropenem model
    # duly gave it weight -0.96. Filtering must happen once, after the union.
    ncbi_geno = {}
    for _, r in have_geno.iterrows():
        ncbi_geno.setdefault(r.asm_base, r.AMR_genotypes)
    base_tokens: dict[str, set[str]] = {}
    id_to_asm = dict(zip(raw.target_acc, raw.asm_base))
    for gid in ncbi["features"].index:
        asm = id_to_asm.get(gid)
        g = ncbi_geno.get(asm)
        base_tokens[gid] = parse_genotypes(g) if isinstance(g, str) else set()
    base_tokens.update(new_tokens)

    counts: dict[str, int] = {}
    for toks in base_tokens.values():
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
    schema = select_schema(counts, len(base_tokens), min_prevalence)

    import numpy as np

    ids = sorted(base_tokens)
    idx = {t: i for i, t in enumerate(schema)}
    X = np.zeros((len(ids), len(schema)), dtype=np.int8)
    for i, gid in enumerate(ids):
        for t in base_tokens[gid]:
            j = idx.get(t)
            if j is not None:
                X[i, j] = 1
    features = pd.DataFrame(X, index=ids, columns=schema)

    labels = pd.concat([ncbi["labels"], add_labels], ignore_index=True)
    labels = labels.drop_duplicates(["genome_id", "drug_id"])

    gmap = dict(zip(ncbi["groups"].genome_id, ncbi["groups"].group_id.astype(str)))
    gmap.update(new_groups)
    groups = pd.DataFrame({"genome_id": ids,
                           "group_id": [gmap.get(i, f"_solo_{i}") for i in ids]})
    groups["group_id"] = pd.factorize(groups.group_id)[0]

    return {**ncbi, "features": features, "labels": labels, "groups": groups}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", default="Klebsiella")
    ap.add_argument("--min-prevalence", type=int, default=3)
    args = ap.parse_args()

    from .ncbi_dataset import build

    cfg = Config.load()
    print("== NCBI base ==")
    ncbi = build(args.organism, args.min_prevalence)
    print(f"base: {ncbi['features'].shape[0]} genomes, {len(ncbi['labels'])} labels")

    print("\n== BV-BRC ==")
    bv = fetch_bvbrc_labels(cfg)
    print(f"{bv.asm_base.nunique()} BV-BRC genomes carry an assembly accession")

    out = merge(ncbi, bv, args.min_prevalence)
    proc = REPO_ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    out["features"].to_parquet(proc / "features.parquet")
    out["labels"].to_csv(proc / "labels.csv", index=False)
    out["groups"].to_csv(proc / "groups.csv", index=False)
    write_json(proc / "sources.json", {
        "ncbi_snapshot": ncbi["snapshot"],
        "n_genomes": int(out["features"].shape[0]),
        "n_features": int(out["features"].shape[1]),
        "n_labels": int(len(out["labels"])),
        "n_groups": int(out["groups"].group_id.nunique()),
        "note": "BV-BRC and NCBI agree 100% on overlapping pairs -- shared provenance, "
                "not independent validation",
    })
    f, lb, g = out["features"], out["labels"], out["groups"]
    print(f"\nmerged: {f.shape[0]} genomes x {f.shape[1]} features, {len(lb)} labels, "
          f"{g.group_id.nunique()} groups")
    print(lb.groupby("drug_id").label.agg(n="size", resistant_frac="mean").round(3))


if __name__ == "__main__":
    main()
