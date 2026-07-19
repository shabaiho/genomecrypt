"""Glue: FASTAs -> AMRFinderPlus TSVs -> feature matrix + homology groups.

    python -m gfw.prepare --threads 2 --jobs 4

Slowest step in the project by far: ~1 genome/minute wall-clock on an 8-core box,
so ~5h for the default 300 genomes. It is resumable -- rerun it and it picks up
where it stopped, so start it early and let it run in the background.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .annotate import batch_annotate
from .config import REPO_ROOT, Config, write_json
from .dedup import DEFAULT_JACCARD, cluster, sketch_dir, summarize
from .features import build_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2, help="threads per amrfinder job")
    ap.add_argument("--jobs", type=int, default=None, help="concurrent genomes")
    ap.add_argument("--jaccard", type=float, default=DEFAULT_JACCARD)
    ap.add_argument("--min-prevalence", type=int, default=3)
    args = ap.parse_args()

    cfg = Config.load()
    fasta_dir = REPO_ROOT / "data" / "raw" / "fasta"
    tsv_dir = REPO_ROOT / "data" / "interim" / "amrfinder"
    proc = REPO_ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    print("== 1/3 annotate ==")
    batch_annotate(fasta_dir, tsv_dir, cfg.species_taxgroup, args.threads, args.jobs)

    print("== 2/3 features ==")
    tsvs = {p.stem: p for p in sorted(tsv_dir.glob("*.tsv"))}
    X, schema = build_matrix(tsvs, min_prevalence=args.min_prevalence)
    X.to_parquet(proc / "features.parquet")
    print(f"{X.shape[0]} genomes x {X.shape[1]} features")

    print("== 3/3 homology groups ==")
    sig = REPO_ROOT / "data" / "interim" / "sketches.sig.zip"
    if not sig.exists():
        sketch_dir(fasta_dir, sig)
    groups = cluster(sig, threshold=args.jaccard)
    pd.DataFrame({"genome_id": list(groups), "group_id": list(groups.values())}).to_csv(
        proc / "groups.csv", index=False)
    stats = summarize(groups)
    write_json(proc / "dedup_stats.json", {"jaccard_threshold": args.jaccard, **stats})
    print(stats)


if __name__ == "__main__":
    main()
