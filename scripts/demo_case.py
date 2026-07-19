"""Demo: predict on a genome the model has never seen, compare to the lab result.

    uv run python scripts/demo_case.py

Shows the SAME model at two operating points, because the operating point is the
whole story on this genome: recall>=0.99 calls everything resistant (1/4 correct),
threshold 0.5 gets 3/4. Use this in the demo to explain the tradeoff honestly
rather than hiding it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import read_json  # noqa: E402
from gfw.features import determinants, parse_amrfinder_tsv, vectorize  # noqa: E402

DEMO = ROOT / "data" / "demo"
VERSION = "v16"


def main() -> None:
    truth = json.loads((DEMO / "truth.json").read_text())
    tsv = DEMO / f"{truth['sample_id']}.tsv"
    if not tsv.exists():
        sys.exit(f"missing {tsv} -- run: make demo-annotate")

    bundle_dir = ROOT / "models" / VERSION
    meta = read_json(bundle_dir / "metadata.json")
    tgt = meta.get("decision", {}).get("target_recall", "?")
    schema = read_json(bundle_dir / "feature_schema.json")["features"]
    tokens = determinants(parse_amrfinder_tsv(tsv))
    x, unknown = vectorize(tokens, schema)

    print(f"{truth['strain']}  ({truth['sample_id']})")
    print(f"{len(tokens)} determinants detected, {len(unknown)} unseen in training\n")

    lab = truth["lab_phenotypes"]
    hdr = f"{'drug':32s} {'P(R)':>6s} | {f'recall>={tgt}':>13s} | {'threshold 0.5':>14s} | lab"
    print(hdr)
    print("-" * len(hdr))

    hits = {"constrained": 0, "default": 0, "n": 0}
    for drug, want in lab.items():
        mp = bundle_dir / f"{drug}.joblib"
        if not mp.exists():
            continue
        b = joblib.load(mp)
        p = float(b["model"].predict_proba(x)[0, 1])
        c_thr = "R" if p >= b["threshold"] else "S"
        c_def = "R" if p >= 0.5 else "S"
        hits["n"] += 1
        hits["constrained"] += c_thr == want
        hits["default"] += c_def == want
        m1 = "OK " if c_thr == want else "BAD"
        m2 = "OK " if c_def == want else "BAD"
        print(f"{drug:32s} {p:6.3f} | {c_thr:>8s} {m1} | {c_def:>9s} {m2} | {want}")

    n = hits["n"]
    print(f"\naccuracy   recall>={tgt}: {hits['constrained']}/{n}"
          f"    threshold 0.5: {hits['default']}/{n}")
    print("\nNOTE: n=4 on one genome is an anecdote, not evidence. The held-out\n      metrics in models/<version>/eval/report.json are the evidence.")
    print("\nEvidence for the one true resistance (meropenem):")
    b = joblib.load(bundle_dir / "meropenem.joblib")
    coef = b["coef"]
    present = [(schema[i], float(coef[i])) for i in x[0].nonzero()[0]]
    for feat, w in sorted(present, key=lambda t: -abs(t[1]))[:4]:
        print(f"  {feat:28s} weight {w:+.3f}")
    print("\nRESEARCH PROTOTYPE - confirm every result with standard lab testing.")


if __name__ == "__main__":
    main()
