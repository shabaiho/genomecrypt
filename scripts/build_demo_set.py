"""Select a held-out demonstration set, before any model sees it.

A single demo genome is an anecdote: 3 calls, so 2/3 and 3/3 are indistinguishable
from luck. This picks a set large enough to mean something, and writes it out
BEFORE training so `gfw.train` can exclude every one of them -- along with their
whole SNP cluster, so a close relative cannot leak in either.

Selection is model-free on purpose. Criteria:
  * labelled for at least 3 of the 5 drugs
  * a mix of resistant and susceptible calls, so a model that answers the same
    thing every time is visibly wrong
  * small SNP cluster, so excluding it costs little training data

    uv run python scripts/build_demo_set.py --n 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import Config  # noqa: E402

OUT = ROOT / "data" / "demo" / "demo_set.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--min-drugs", type=int, default=3)
    args = ap.parse_args()

    cfg = Config.load()
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    L = pd.read_csv(ROOT / "data/processed/labels.csv")
    G = pd.read_csv(ROOT / "data/processed/groups.csv").set_index("genome_id").group_id
    sizes = G.value_counts()

    lab = L[L.genome_id.isin(X.index)]
    per = lab.groupby("genome_id").agg(n_drugs=("drug_id", "size"),
                                       n_res=("label", "sum"))
    per["cluster"] = G.reindex(per.index)
    per["cluster_size"] = per.cluster.map(sizes)
    per["mixed"] = (per.n_res > 0) & (per.n_res < per.n_drugs)

    # keep the existing single-FASTA demo genome in the set so the live upload
    # path and the batch numbers describe the same cohort
    truth = json.loads((ROOT / "data/demo/truth.json").read_text())
    anchor = truth["isolate"]

    cand = per[(per.n_drugs >= args.min_drugs) & per.mixed]
    cand = cand.sort_values(["cluster_size", "n_drugs"], ascending=[True, False])

    chosen: list[str] = [anchor] if anchor in per.index else []
    used_clusters = {per.at[anchor, "cluster"]} if chosen else set()
    for gid, row in cand.iterrows():
        if len(chosen) >= args.n:
            break
        if gid in chosen or row.cluster in used_clusters:
            continue
        chosen.append(gid)
        used_clusters.add(row.cluster)

    rows = []
    for gid in chosen:
        labs = lab[lab.genome_id == gid].set_index("drug_id").label.to_dict()
        rows.append({
            "genome_id": gid,
            "cluster": int(per.at[gid, "cluster"]),
            "phenotypes": {d: ("R" if v == 1 else "S") for d, v in labs.items()},
        })

    n_labels = sum(len(r["phenotypes"]) for r in rows)
    OUT.write_text(json.dumps({
        "species": cfg.species,
        "n_genomes": len(rows),
        "n_labels": n_labels,
        "note": ("Selected before training and excluded from train/calib/thresh "
                 "together with their whole SNP clusters. Phenotypes are the "
                 "lab antibiograms, never shown to the model."),
        "genomes": rows,
    }, indent=2))

    print(f"selected {len(rows)} genomes, {n_labels} lab-confirmed calls")
    print(f"clusters excluded: {sorted(used_clusters)}")
    res = sum(v == "R" for r in rows for v in r["phenotypes"].values())
    print(f"label balance: {res} resistant / {n_labels - res} susceptible")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
