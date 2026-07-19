"""Evaluate the whole held-out demonstration cohort.

Replaces the single-genome demo, which had 3 lab-confirmed calls behind it -- too
few to distinguish a working model from a lucky one. This cohort was selected
model-free by scripts/build_demo_set.py and excluded from train/calib/thresh
along with its SNP clusters, so nothing here was ever seen.

The label balance is deliberately even, so "always resistant" scores 50%.

    uv run python scripts/demo_set_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.predict import CALL_FAIL, CALL_NONE, CALL_WORK, Predictor  # noqa: E402

# The served bundle, not a pinned version. models/current is a symlink that
# retraining re-points; hardcoding "v19" here meant the tests would keep
# validating an older bundle than the app actually loads.
VERSION = "current"
SET = ROOT / "data" / "demo" / "demo_set.json"


def main() -> None:
    spec = json.loads(SET.read_text())
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    pred = Predictor(VERSION)

    rows = []
    for g in spec["genomes"]:
        gid = g["genome_id"]
        if gid not in X.index:
            continue
        tokens = {c for c in X.columns if X.at[gid, c] == 1}
        rep = pred.predict_from_tokens(tokens, gid)
        for r in rep.results:
            want = g["phenotypes"].get(r.drug_id)
            if want is None:
                continue
            got = {CALL_FAIL: "R", CALL_WORK: "S", CALL_NONE: "?"}[r.call]
            rows.append({"genome": gid, "drug": r.drug_id, "lab": want,
                         "predicted": got, "p": r.confidence})

    df = pd.DataFrame(rows)
    called = df[df.predicted != "?"]
    correct = (called.lab == called.predicted).sum()

    print(f"HELD-OUT DEMONSTRATION COHORT -- model {VERSION}")
    print(f"{spec['n_genomes']} genomes never seen in training, "
          f"{len(df)} lab-confirmed calls\n")

    print(f"{'drug':32s} {'n':>4s} {'called':>7s} {'correct':>8s} {'accuracy':>9s}")
    for drug, g in df.groupby("drug"):
        c = g[g.predicted != "?"]
        acc = (c.lab == c.predicted).mean() if len(c) else float("nan")
        print(f"{drug[:32]:32s} {len(g):4d} {len(c):7d} "
              f"{int((c.lab == c.predicted).sum()):8d} {acc:9.3f}")

    n_call, n_all = len(called), len(df)
    print(f"\n{'TOTAL':32s} {n_all:4d} {n_call:7d} {correct:8d} "
          f"{correct / max(1, n_call):9.3f}")
    print(f"\nno-call rate {1 - n_call / n_all:.0%}  "
          f"({n_all - n_call} of {n_all} calls declined)")

    # the balance is even by construction, so this is the bar to clear
    res = (df.lab == "R").sum()
    print(f"label balance {res}R / {n_all - res}S -> "
          f"'always resistant' would score {res / n_all:.0%} overall")

    misses = called[called.lab != called.predicted]
    if len(misses):
        print(f"\n{len(misses)} incorrect calls:")
        for _, m in misses.iterrows():
            direction = ("FALSE SUSCEPTIBLE (dangerous)" if m.lab == "R"
                         else "false resistant (conservative)")
            print(f"  {m.genome[:18]:18s} {m.drug[:26]:26s} "
                  f"lab={m.lab} predicted={m.predicted}  {direction}")
        dangerous = (misses.lab == "R").sum()
        print(f"\n{dangerous} of {len(misses)} errors are in the dangerous direction "
              f"(resistance missed).")

    out = ROOT / "models" / VERSION / "eval" / "demo_cohort.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_genomes": spec["n_genomes"],
        "n_labels": n_all,
        "n_called": n_call,
        "n_correct": int(correct),
        "accuracy": round(correct / max(1, n_call), 4),
        "no_call_rate": round(1 - n_call / n_all, 4),
        "dangerous_errors": int((misses.lab == "R").sum()) if len(misses) else 0,
        "label_balance": {"R": int(res), "S": int(n_all - res)},
    }, indent=2))
    print(f"\nwrote {out}")
    print("RESEARCH PROTOTYPE - every result requires laboratory confirmation.")


if __name__ == "__main__":
    main()
