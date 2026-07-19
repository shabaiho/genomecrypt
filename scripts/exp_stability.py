"""EXPERIMENT: error bars. Every number reported so far came from ONE split.

seed=0 was used throughout, so each "X improved Y by 0.03" claim is a single draw
from a distribution we never measured. If the spread across splits is comparable
to the effect sizes, those claims are noise and the comparison history is void.

This repeats the whole grouped 4-way split over several seeds and reports
mean +/- std per drug. Nothing here changes the model; it tells us whether the
model comparisons were ever meaningful.

    uv run python scripts/exp_stability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import Config  # noqa: E402
from gfw.train import _freeze, _prefit_kwargs, grouped_four_way  # noqa: E402

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
GRID = (0.03, 0.1, 0.3, 1.0)


def run_one(Xa, ya, ga, seed):
    tr, ca, th, te = grouped_four_way(ga, seed)
    if any(len(np.unique(ya[s])) < 2 for s in (tr, ca, th, te)):
        return None
    best, best_auc = None, -1.0
    for C in GRID:
        m = LogisticRegression(penalty="l1", C=C, class_weight="balanced",
                               solver="liblinear", max_iter=5000)
        m.fit(Xa[tr], ya[tr])
        a = roc_auc_score(ya[ca], m.decision_function(Xa[ca]))
        if a > best_auc:
            best, best_auc = m, a
    cal = CalibratedClassifierCV(_freeze(best), method="sigmoid", **_prefit_kwargs())
    cal.fit(Xa[ca], ya[ca])
    p = cal.predict_proba(Xa[te])[:, 1]
    return {
        "bal_acc": balanced_accuracy_score(ya[te], (p >= 0.5).astype(int)),
        "auroc": roc_auc_score(ya[te], p),
        "brier": brier_score_loss(ya[te], p),
        "n_test": len(te),
        "nonzero": int((best.coef_[0] != 0).sum()),
    }


def main() -> None:
    cfg = Config.load()
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    L = pd.read_csv(ROOT / "data/processed/labels.csv")
    G = pd.read_csv(ROOT / "data/processed/groups.csv").set_index("genome_id").group_id

    rows = []
    for drug in cfg.drugs:
        sub = L[(L.drug_id == drug.id) & (L.genome_id.isin(X.index))]
        if sub.empty:
            continue
        Xa = X.loc[sub.genome_id].to_numpy(np.float32)
        ya = sub.label.to_numpy(int)
        ga = G.loc[sub.genome_id].to_numpy()
        for seed in SEEDS:
            r = run_one(Xa, ya, ga, seed)
            if r:
                rows.append({"drug": drug.id[:20], "seed": seed, **r})

    df = pd.DataFrame(rows)
    print(f"{'drug':22s} {'bal.acc':>16s} {'AUROC':>16s} {'Brier':>16s} {'genes':>10s}")
    for d, g in df.groupby("drug"):
        print(f"{d:22s} "
              f"{g.bal_acc.mean():.3f} +/- {g.bal_acc.std():.3f}  "
              f"{g.auroc.mean():.3f} +/- {g.auroc.std():.3f}  "
              f"{g.brier.mean():.3f} +/- {g.brier.std():.3f}  "
              f"{g.nonzero.mean():5.0f} +/- {g.nonzero.std():.0f}")

    print(f"\nacross all drugs and {len(SEEDS)} seeds:")
    for m in ("bal_acc", "auroc", "brier"):
        s = df.groupby("seed")[m].mean()
        print(f"  {m:8s} mean {s.mean():.4f}  std {s.std():.4f}  "
              f"range [{s.min():.4f}, {s.max():.4f}]")

    spread = df.groupby("seed").auroc.mean()
    print(f"\nseed-to-seed spread in mean AUROC: {spread.max() - spread.min():.4f}")
    print("Compare that against the effect sizes claimed earlier:")
    print("  extra BV-BRC data      +0.0296 AUROC")
    print("  Platt vs isotonic      +0.0138 AUROC")
    print("  class-restricted feats -0.0248 AUROC")
    print("\nAny effect smaller than the spread is not established by a single split.")


if __name__ == "__main__":
    main()
