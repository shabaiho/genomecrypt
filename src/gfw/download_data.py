"""Fetch the fixed dataset from BV-BRC via the public Data API (stdlib only).

Two artifacts come out of this:
  data/processed/labels.csv    genome_id,drug_id,label  (lab-measured only)
  data/raw/fasta/<gid>.fna     one assembly per selected genome

Why the API and not ftp.bvbrc.org/RELEASE_NOTES/PATRIC_genomes_AMR.txt: the bulk
file is ~400MB of all species, and we need one. The API filters server-side.

IMPORTANT (brief, p.4): lab-measured results only. Verified counts for
K. pneumoniae at time of writing:
    Broth dilution 77,534 | MIC 36,739 | Disk diffusion 6,198 | Agar dilution 3,353
    Computational prediction 11,074   <-- EXCLUDED, these are model-generated
Training on the last group would launder another model's errors into ours.

    python -m gfw.download_data --limit 800
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, Config

API = "https://www.bv-brc.org/api"
PAGE = 5000          # API caps a single response; we page with limit(n,offset)
RETRIES = 3

# Values of `laboratory_typing_method` that mean "a human measured this in a lab".
LAB_METHODS = [
    "Broth dilution", "MIC", "Disk diffusion", "Agar dilution",
    "Etest", "Agar diffusion", "Microbroth dilution", "Disc diffusion",
]


def _get(endpoint: str, query: str, accept: str = "application/json") -> bytes:
    url = f"{API}/{endpoint}/?{query}"
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"Accept": accept})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            print(f"  retry {attempt + 1}/{RETRIES} ({e})")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_amr(cfg: Config) -> pd.DataFrame:
    """All lab-measured phenotypes for our species x our drug panel."""
    # NOTE: values containing "/" (trimethoprim/sulfamethoxazole -- the single
    # largest label set for K. pneumoniae) MUST be quoted inside in(), otherwise
    # RQL silently drops the term and you lose the drug without any error.
    drug_names = sorted({d.id.replace("_", "/") for d in cfg.drugs})
    in_drugs = "in(antibiotic,(%s))" % ",".join(
        urllib.parse.quote(f'"{d}"', safe="") for d in drug_names)
    in_methods = "in(laboratory_typing_method,(%s))" % ",".join(
        urllib.parse.quote(f'"{m}"') for m in LAB_METHODS)
    species = urllib.parse.quote(cfg.species)

    rows, offset = [], 0
    while True:
        q = (f"eq(genome_name,{species}*)&{in_drugs}&{in_methods}"
             f"&select(genome_id,antibiotic,resistant_phenotype,laboratory_typing_method)"
             f"&limit({PAGE},{offset})")
        batch = json.loads(_get("genome_amr", q))
        rows.extend(batch)
        print(f"  fetched {len(rows)} phenotype records", flush=True)
        if len(batch) < PAGE:
            break
        offset += PAGE
    return pd.DataFrame(rows)


def build_labels(df: pd.DataFrame, cfg: Config, limit: int) -> tuple[pd.DataFrame, list[str]]:
    if df.empty:
        raise SystemExit("no AMR records returned -- check species name in config/drugs.yaml")

    df["drug_id"] = (df.antibiotic.fillna("").str.lower()
                     .str.replace("/", "_", regex=False)
                     .str.replace("-", "_", regex=False))
    df = df[df.drug_id.isin({d.id for d in cfg.drugs})]

    df["label"] = df.resistant_phenotype.map(cfg.label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df.label.astype(int)

    # one final label per (genome, drug); drop pairs where sources disagree
    agree = df.groupby(["genome_id", "drug_id"]).label.nunique()
    keep = agree[agree == 1].index
    df = df.set_index(["genome_id", "drug_id"]).loc[keep].reset_index()
    df = df.drop_duplicates(["genome_id", "drug_id"])[["genome_id", "drug_id", "label"]]

    # prefer genomes with the most drugs measured -- denser labels per download
    ranked = df.groupby("genome_id").size().sort_values(ascending=False)
    gids = ranked.head(limit).index.tolist()
    return df[df.genome_id.isin(gids)], gids


def fetch_fastas(gids: list[str], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = []
    for i, gid in enumerate(gids, 1):
        dest = out_dir / f"{gid}.fna"
        if dest.exists() and dest.stat().st_size > 100_000:
            ok.append(gid)
            continue
        try:
            data = _get("genome_sequence",
                        f"eq(genome_id,{gid})&limit(10000)",
                        accept="application/dna+fasta")
            if len(data) < 100_000:          # truncated / empty assembly
                raise ValueError(f"only {len(data)} bytes")
            dest.write_bytes(data)
            ok.append(gid)
        except Exception as e:
            print(f"  miss {gid}: {e}")
            dest.unlink(missing_ok=True)
        if i % 25 == 0:
            print(f"  {i}/{len(gids)} genomes", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=800, help="number of genomes to download")
    ap.add_argument("--skip-fasta", action="store_true", help="labels only, no assemblies")
    args = ap.parse_args()

    cfg = Config.load()
    proc = REPO_ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    print(f"== phenotypes: {cfg.species} ==")
    labels, gids = build_labels(fetch_amr(cfg), cfg, args.limit)
    print(f"\n{len(gids)} genomes, {len(labels)} genome-drug labels")
    print(labels.groupby("drug_id").label.agg(n="size", resistant_frac="mean").round(3))

    if args.skip_fasta:
        labels.to_csv(proc / "labels.csv", index=False)
        return

    print(f"\n== assemblies (~5MB each, ~{len(gids) * 5 // 1000}GB total) ==")
    ok = set(fetch_fastas(gids, REPO_ROOT / "data" / "raw" / "fasta"))

    # only keep labels for genomes whose assembly actually downloaded
    labels = labels[labels.genome_id.isin(ok)]
    labels.to_csv(proc / "labels.csv", index=False)
    print(f"\n{len(ok)} assemblies, {len(labels)} usable labels -> {proc / 'labels.csv'}")


if __name__ == "__main__":
    main()
