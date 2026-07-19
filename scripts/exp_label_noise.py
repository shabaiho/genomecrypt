"""EXPERIMENT: how much do the two label sources disagree?

BV-BRC and NCBI Pathogen Detection both publish lab-measured R/S calls. Where the
same assembly appears in both, disagreement is pure label noise: same organism,
same genome, two different answers. That number is the CEILING on accuracy -- no
model can beat the labels it is graded against.

Worth measuring before merging the sources, because if they disagree a lot then
merging adds noise as fast as it adds data.

    uv run python scripts/exp_label_noise.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import Config  # noqa: E402
from gfw.download_data import LAB_METHODS, _get  # noqa: E402
from gfw.ncbi_dataset import parse_ast  # noqa: E402

SNAPSHOT = "PDG000000012.2470"


def main() -> None:
    cfg = Config.load()

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
        if len(batch) < 5000:
            break
        offset += 5000
    bv = pd.DataFrame(rows)
    bv["drug_id"] = (bv.antibiotic.str.lower().str.replace("/", "_", regex=False)
                     .str.replace("-", "_", regex=False))
    bv["label"] = bv.resistant_phenotype.map(cfg.label_map)
    bv = bv.dropna(subset=["label"])

    ids = sorted(bv.genome_id.unique())
    amap = {}
    for i in range(0, len(ids), 200):
        q = ("in(genome_id,(%s))&select(genome_id,assembly_accession)&limit(1000)"
             % ",".join(ids[i:i + 200]))
        try:
            for r in json.loads(_get("genome", q)):
                a = r.get("assembly_accession")
                if isinstance(a, str) and a.startswith("GC"):
                    amap[r["genome_id"]] = a.split(".")[0]
        except Exception:
            pass
    bv["asm_base"] = bv.genome_id.map(amap)
    bv = bv.dropna(subset=["asm_base"])

    ncbi = pd.read_csv(ROOT / f"data/raw/ncbi/{SNAPSHOT}.amr.metadata.tsv",
                       sep="\t", dtype=str, low_memory=False,
                       usecols=["asm_acc", "AST_phenotypes"]).dropna()
    nrows = []
    for _, r in ncbi.iterrows():
        base = r.asm_acc.split(".")[0]
        for drug, val in parse_ast(r.AST_phenotypes, cfg.label_map).items():
            did = drug.replace("-", "_").replace("/", "_")
            if did in {d.id for d in cfg.drugs}:
                nrows.append({"asm_base": base, "drug_id": did, "label_ncbi": int(val)})
    nc = pd.DataFrame(nrows).drop_duplicates(["asm_base", "drug_id"])

    bv2 = (bv.groupby(["asm_base", "drug_id"]).label
           .agg(lambda s: s.iloc[0] if s.nunique() == 1 else pd.NA)
           .dropna().astype(int).rename("label_bvbrc").reset_index())

    both = bv2.merge(nc, on=["asm_base", "drug_id"], how="inner")
    print(f"overlapping (assembly, drug) pairs: {len(both)}\n")
    if both.empty:
        print("no overlap -- cannot measure")
        return

    print(f"{'drug':32s} {'n':>6s} {'agree':>7s} {'BV=R,NCBI=S':>12s} {'BV=S,NCBI=R':>12s}")
    for d, g in both.groupby("drug_id"):
        agree = (g.label_bvbrc == g.label_ncbi).mean()
        rs = int(((g.label_bvbrc == 1) & (g.label_ncbi == 0)).sum())
        sr = int(((g.label_bvbrc == 0) & (g.label_ncbi == 1)).sum())
        print(f"{d[:32]:32s} {len(g):6d} {agree:7.3f} {rs:12d} {sr:12d}")

    overall = (both.label_bvbrc == both.label_ncbi).mean()
    print(f"\noverall agreement: {overall:.3f}")
    print(f"=> ~{(1 - overall) * 100:.1f}% of labels are contradicted by the other source.")
    print("   A model scoring above that agreement rate is fitting one lab's noise.")


if __name__ == "__main__":
    main()
