"""EXPERIMENT: isotonic vs Platt (sigmoid) calibration.

Two questions, one of them mathematical and one empirical.

RESULT (5 drugs, held out): sigmoid wins on every axis.
    Brier 0.2032 vs 0.2144 | ECE 0.1174 vs 0.1211 | AUROC 0.7628 vs 0.7490
    largest plateau 10% vs 34% of samples

MATH. Platt scaling fits p = sigmoid(a*s + b) where s is the linear score
b0 + sum_i beta_i x_i. Composing two monotone maps of that form keeps the model
in the logistic family:

    logit(p) = a * (b0 + sum_i beta_i x_i) + b

so every feature still moves the log-odds by a FIXED amount, and exp(a*beta_i)
is the odds ratio for carrying that gene. Isotonic is a non-parametric step
function; after it, "this gene multiplies the odds by X" is no longer a true
statement about the output. Interpretability is a scored criterion here, so this
is not a cosmetic difference.

STATISTICS. Isotonic has O(n) effective parameters and the usual guidance is
~1000+ calibration points; Platt has exactly 2. Our calibration blocks are
114-171 rows. Isotonic should overfit, and the symptom would be a coarse step
function with big plateaus.

    uv run python scripts/exp_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import Config  # noqa: E402
from gfw.train import _freeze, _prefit_kwargs, grouped_four_way  # noqa: E402


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: mean |predicted - observed| weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


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
        tr, ca, th, te = grouped_four_way(ga, 0)

        base = LogisticRegression(penalty="l1", C=0.1, class_weight="balanced",
                                  solver="liblinear", max_iter=5000)
        base.fit(Xa[tr], ya[tr])

        for method in ("isotonic", "sigmoid"):
            cal = CalibratedClassifierCV(_freeze(base), method=method, **_prefit_kwargs())
            cal.fit(Xa[ca], ya[ca])
            p = cal.predict_proba(Xa[te])[:, 1]
            rows.append({
                "drug": drug.id[:18], "method": method,
                "n_calib": len(ca), "n_test": len(te),
                "brier": round(brier_score_loss(ya[te], p), 4),
                "ece": round(ece(ya[te], p), 4),
                "auroc": round(roc_auc_score(ya[te], p), 4),
                "distinct_p": len(np.unique(np.round(p, 4))),
                "largest_plateau_pct": round(
                    100 * max(np.unique(np.round(p, 4), return_counts=True)[1]) / len(p), 1),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n--- средние по 5 препаратам ---")
    agg = df.groupby("method")[["brier", "ece", "auroc", "distinct_p",
                                "largest_plateau_pct"]].mean().round(4)
    print(agg.to_string())

    iso, sig = agg.loc["isotonic"], agg.loc["sigmoid"]
    print(f"\nBrier: sigmoid {sig.brier:.4f} vs isotonic {iso.brier:.4f} "
          f"-> {'sigmoid wins' if sig.brier < iso.brier else 'isotonic wins'}")
    print(f"ECE:   sigmoid {sig.ece:.4f} vs isotonic {iso.ece:.4f} "
          f"-> {'sigmoid wins' if sig.ece < iso.ece else 'isotonic wins'}")
    # SELF-CORRECTION: the docstring claim that AUROC must be identical is wrong.
    # Both maps are monotone, but isotonic is only WEAKLY monotone -- it collapses
    # distinct scores onto shared plateaus, and tied samples score 0.5 credit in
    # AUROC instead of 1. So isotonic genuinely destroys ranking information:
    print(f"AUROC:  sigmoid {sig.auroc:.4f} vs isotonic {iso.auroc:.4f} -- NOT equal, "
          f"because isotonic ties {iso.largest_plateau_pct:.0f}% of samples onto one "
          f"value and ties lose AUROC credit")


if __name__ == "__main__":
    main()
