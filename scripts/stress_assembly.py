"""STRESS TEST: does assembly quality change the answer?

The model was trained on genotypes from finished or near-finished assemblies. Real
submissions are draft assemblies: hundreds of contigs, genes cut across contig
boundaries, occasional missing regions. A resistance gene that falls on a break is
simply not detected -- and "not detected" is indistinguishable from "not there",
which turns into a confident "likely to work".

Two degradations applied to a genome whose answer we know:
  1. FRAGMENTATION  -- chop contigs to a fixed length. Genes longer than the piece
     or straddling a cut can be missed.
  2. DROPOUT        -- randomly delete a fraction of contigs, simulating incomplete
     assembly or low coverage.

We report, at every level, whether blaKPC is still seen and whether any call flips.

    uv run python scripts/stress_assembly.py     # needs `make tools`
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.annotate import amrfinder_available, run_amrfinder  # noqa: E402
from gfw.features import determinants, parse_amrfinder_tsv  # noqa: E402
from gfw.predict import Predictor  # noqa: E402

DEMO = ROOT / "data" / "demo"
FASTA = DEMO / "GCA_000417485.1.fna"
# The served bundle, not a pinned version. models/current is a symlink that
# retraining re-points; hardcoding "v19" here meant the tests would keep
# validating an older bundle than the app actually loads.
VERSION = "current"
ORGANISM = "Klebsiella_pneumoniae"


def read_contigs(path: Path) -> list[tuple[str, str]]:
    out, name, seq = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name:
                out.append((name, "".join(seq)))
            name, seq = line[1:], []
        else:
            seq.append(line.strip())
    if name:
        out.append((name, "".join(seq)))
    return out


def write_contigs(contigs: list[tuple[str, str]], path: Path) -> None:
    with path.open("w") as fh:
        for i, (n, s) in enumerate(contigs):
            fh.write(f">{n or f'c{i}'}\n")
            for j in range(0, len(s), 70):
                fh.write(s[j:j + 70] + "\n")


def fragment(contigs, piece: int):
    out = []
    for n, s in contigs:
        for i in range(0, len(s), piece):
            out.append((f"{n.split()[0]}_p{i // piece}", s[i:i + piece]))
    return out


def dropout(contigs, frac: float, seed: int = 0):
    rng = random.Random(seed)
    keep = [c for c in contigs if rng.random() > frac]
    return keep or contigs[:1]


def classify_flip(before: str, after: str) -> str:
    """Only one direction actually endangers a patient."""
    if before == after:
        return ""
    if before == "likely_to_fail" and after == "likely_to_work":
        return "DANGEROUS"          # resistance lost -> ineffective drug recommended
    if after == "no_call":
        return "safe"               # degraded into an honest refusal
    if before == "no_call":
        return "new call"
    return "changed"


def evaluate(pred: Predictor, fasta: Path, tmp: Path, tag: str) -> dict:
    tsv = tmp / f"{tag}.tsv"
    run_amrfinder(fasta, tsv, ORGANISM, threads=6)
    toks = determinants(parse_amrfinder_tsv(tsv))
    rep = pred.predict_from_tokens(toks, tag)
    return {
        "n_det": len(toks),
        "kpc": any("blaKPC" in t for t in toks),
        "calls": {r.drug_id: r.call for r in rep.results},
    }


def main() -> None:
    if not amrfinder_available():
        sys.exit("needs AMRFinderPlus on PATH -- run `make tools` and source .tools/env.sh")

    pred = Predictor(VERSION)
    contigs = read_contigs(FASTA)
    total = sum(len(s) for _, s in contigs)
    print(f"reference assembly: {len(contigs)} contigs, {total:,} bp\n")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        base = evaluate(pred, FASTA, tmp, "base")
        def row(label, n_contigs, r, base):
            flips = []
            for d, before in base["calls"].items():
                kind = classify_flip(before, r["calls"][d])
                if kind:
                    flips.append(f"{d[:9]}:{before[7:]}->{r['calls'][d][7:]}[{kind}]")
            return (f"{label:34s} {n_contigs:8d} {r['n_det']:8d} "
                    f"{'yes' if r['kpc'] else 'NO':>7s}  {'; '.join(flips) or '-'}")

        print(f"{'condition':34s} {'contigs':>8s} {'determ.':>8s} {'blaKPC':>7s}  flips")
        print(row("as submitted", len(contigs), base, base))

        for piece in (50000, 20000, 10000, 5000, 2000):
            frag = fragment(contigs, piece)
            p = tmp / f"frag{piece}.fna"
            write_contigs(frag, p)
            r = evaluate(pred, p, tmp, f"frag{piece}")
            print(row("fragmented to %d kb pieces" % (piece // 1000),
                      len(frag), r, base))

        for frac in (0.1, 0.25, 0.5):
            sub = dropout(contigs, frac)
            p = tmp / f"drop{frac}.fna"
            write_contigs(sub, p)
            r = evaluate(pred, p, tmp, f"drop{frac}")
            print(row(f"{int(frac * 100)}% of contigs dropped", len(sub), r, base))

    print("\nDANGEROUS = resistance was detected before degradation and is not after,")
    print("so an ineffective drug would now be recommended. `safe` means the call")
    print("degraded into a no-call, which is the correct behaviour under lost evidence.")


if __name__ == "__main__":
    main()
