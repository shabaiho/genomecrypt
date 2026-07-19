"""EXPERIMENT: can we get more labelled isolates without re-running annotation?

NCBI Pathogen Detection gave us 1,992 usable isolates -- everything that has BOTH
AMR_genotypes and AST_phenotypes. But the same file carries AMR_genotypes for
167,247 isolates; the binding constraint is the PHENOTYPE, not the genotype.

BV-BRC holds ~28k lab-measured phenotype records for K. pneumoniae on our drug
panel. If a BV-BRC genome can be matched to an NCBI isolate, we inherit NCBI's
genotype (already decoded) and BV-BRC's label -- new training rows for zero
annotation cost.

Join keys available on both sides: assembly accession (GCA_*) and BioSample
(SAMN*). This script measures the yield BEFORE we build the pipeline, and counts
how much of it is genuinely new rather than duplicate.

    uv run python scripts/exp_more_data.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import Config  # noqa: E402
from gfw.download_data import LAB_METHODS, _get  # noqa: E402

SNAPSHOT = "PDG000000012.2470"
PAGE = 5000


def bvbrc_phenotypes(cfg: Config) -> pd.DataFrame:
    drug_names = sorted({d.id.replace("_", "/") for d in cfg.drugs})
    in_drugs = "in(antibiotic,(%s))" % ",".join(
        urllib.parse.quote(f'"{d}"', safe="") for d in drug_names)
    in_methods = "in(laboratory_typing_method,(%s))" % ",".join(
        urllib.parse.quote(f'"{m}"') for m in LAB_METHODS)
    species = urllib.parse.quote(cfg.species)

    rows, offset = [], 0
    while True:
        q = (f"eq(genome_name,{species}*)&{in_drugs}&{in_methods}"
             f"&select(genome_id,antibiotic,resistant_phenotype)&limit({PAGE},{offset})")
        batch = json.loads(_get("genome_amr", q))
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
    return pd.DataFrame(rows)


def bvbrc_assembly_map(genome_ids: list[str]) -> dict[str, str]:
    """genome_id -> assembly_accession, batched."""
    out: dict[str, str] = {}
    for i in range(0, len(genome_ids), 200):
        chunk = genome_ids[i:i + 200]
        q = ("in(genome_id,(%s))&select(genome_id,assembly_accession)&limit(1000)"
             % ",".join(chunk))
        try:
            for r in json.loads(_get("genome", q)):
                acc = r.get("assembly_accession")
                if isinstance(acc, str) and acc.startswith("GC"):
                    out[r["genome_id"]] = acc
        except Exception as e:
            print(f"  batch {i} failed: {e}")
        if (i // 200) % 5 == 0:
            print(f"  mapped {len(out)} / {i + len(chunk)} genomes", flush=True)
    return out


def main() -> None:
    cfg = Config.load()

    print("== BV-BRC phenotypes ==")
    bv = bvbrc_phenotypes(cfg)
    bv["drug_id"] = (bv.antibiotic.str.lower().str.replace("/", "_", regex=False)
                     .str.replace("-", "_", regex=False))
    bv["label"] = bv.resistant_phenotype.map(cfg.label_map)
    bv = bv.dropna(subset=["label"])
    print(f"{len(bv)} phenotype records over {bv.genome_id.nunique()} genomes")

    print("\n== mapping BV-BRC genomes to assembly accessions ==")
    amap = bvbrc_assembly_map(sorted(bv.genome_id.unique()))
    print(f"{len(amap)} genomes have an assembly accession")

    print("\n== NCBI side ==")
    ncbi = pd.read_csv(ROOT / f"data/raw/ncbi/{SNAPSHOT}.amr.metadata.tsv",
                       sep="\t", dtype=str, low_memory=False,
                       usecols=["target_acc", "asm_acc", "AST_phenotypes", "AMR_genotypes"])
    has_geno = ncbi.dropna(subset=["AMR_genotypes"])
    already = set(ncbi.dropna(subset=["AST_phenotypes", "AMR_genotypes"]).asm_acc.dropna())
    print(f"{len(has_geno)} isolates with a genotype; {len(already)} already usable today")

    # version-insensitive join: GCA_000123456.1 and .2 are the same assembly
    def base(a: str) -> str:
        return a.split(".")[0] if isinstance(a, str) else ""

    geno_by_base = {base(a): a for a in has_geno.asm_acc.dropna()}
    already_base = {base(a) for a in already}

    matched, new_genomes = {}, set()
    for gid, acc in amap.items():
        b = base(acc)
        if b in geno_by_base:
            matched[gid] = geno_by_base[b]
            if b not in already_base:
                new_genomes.add(b)

    print(f"\nBV-BRC genomes matching an NCBI genotype : {len(matched)}")
    print(f"  of which NOT already usable (new)      : {len(new_genomes)}")

    new_labels = bv[bv.genome_id.isin(
        [g for g, a in matched.items() if base(a) in new_genomes])]
    print(f"new genome-drug labels available         : {len(new_labels)}")
    if len(new_labels):
        print(new_labels.groupby("drug_id").label.agg(n="size", resistant_frac="mean").round(3))

    print(f"\nverdict: {len(new_genomes)} new genomes on top of 1,992 "
          f"({100 * len(new_genomes) / 1992:.0f}% more)")


if __name__ == "__main__":
    main()
