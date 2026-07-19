"""EXPERIMENT: is logistic regression leaving accuracy on the table?

Used as a DIAGNOSTIC, not as a product. Gradient boosting can represent gene-gene
interactions that an additive model cannot -- e.g. "porin loss matters only when a
beta-lactamase is also present", which is real carbapenem biology. If boosting
beats LR by a lot, the additive assumption is costing us and the fix is better
FEATURES (explicit interaction terms), not a black box. If it does not, we have a
measured argument for keeping the model the jury can read.

Paired across seeds -- same split, same features, only the learner changes.

    uv run python scripts/exp_ceiling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import Config  # noqa: E402
from gfw.train import _freeze, _prefit_kwargs, grouped_four_way  # noqa: E402

SEEDS = list(range(8))
GRID = (0.03, 0.1, 0.3, 1.0)


def make_lr(Xtr, ytr, Xca, yca):
    best, best_auc = None, -1.0
    for C in GRID:
        m = LogisticRegression(penalty="l1", C=C, class_weight="balanced",
                               solver="liblinear", max_iter=5000)
        m.fit(Xtr, ytr)
        a = roc_auc_score(yca, m.decision_function(Xca))
        if a > best_auc:
            best, best_auc = m, a
    return best


def score(model, Xca, yca, Xte, yte):
    cal = CalibratedClassifierCV(_freeze(model), method="sigmoid", **_prefit_kwargs())
    cal.fit(Xca, yca)
    p = cal.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), brier_score_loss(yte, p)


def main() -> None:
    cfg = Config.load()
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    L = pd.read_csv(ROOT / "data/processed/labels.csv")
    G = pd.read_csv(ROOT / "data/processed/groups.csv").set_index("genome_id").group_id

    per_seed = {k: [] for k in ("gbm-lr", "rf-lr", "gbm-lr-brier")}
    abs_scores = {k: [] for k in ("lr", "gbm", "rf")}

    for seed in SEEDS:
        d_gbm, d_rf, d_gbm_b = [], [], []
        a_lr, a_gbm, a_rf = [], [], []
        for drug in cfg.drugs:
            sub = L[(L.drug_id == drug.id) & (L.genome_id.isin(X.index))]
            if sub.empty:
                continue
            Xa = X.loc[sub.genome_id].to_numpy(np.float32)
            ya = sub.label.to_numpy(int)
            ga = G.loc[sub.genome_id].to_numpy()
            tr, ca, th, te = grouped_four_way(ga, seed)
            if any(len(np.unique(ya[s])) < 2 for s in (tr, ca, te)):
                continue

            lr = make_lr(Xa[tr], ya[tr], Xa[ca], ya[ca])
            gbm = HistGradientBoostingClassifier(
                max_depth=3, max_iter=300, learning_rate=0.05,
                l2_regularization=1.0, random_state=seed)
            gbm.fit(Xa[tr], ya[tr])
            rf = RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                n_jobs=-1, random_state=seed)
            rf.fit(Xa[tr], ya[tr])

            s_lr = score(lr, Xa[ca], ya[ca], Xa[te], ya[te])
            s_gbm = score(gbm, Xa[ca], ya[ca], Xa[te], ya[te])
            s_rf = score(rf, Xa[ca], ya[ca], Xa[te], ya[te])

            d_gbm.append(s_gbm[0] - s_lr[0])
            d_rf.append(s_rf[0] - s_lr[0])
            d_gbm_b.append(s_gbm[1] - s_lr[1])
            a_lr.append(s_lr[0])
            a_gbm.append(s_gbm[0])
            a_rf.append(s_rf[0])

        if d_gbm:
            per_seed["gbm-lr"].append(np.mean(d_gbm))
            per_seed["rf-lr"].append(np.mean(d_rf))
            per_seed["gbm-lr-brier"].append(np.mean(d_gbm_b))
            abs_scores["lr"].append(np.mean(a_lr))
            abs_scores["gbm"].append(np.mean(a_gbm))
            abs_scores["rf"].append(np.mean(a_rf))

    print("absolute mean AUROC over seeds:")
    for k, v in abs_scores.items():
        print(f"  {k:4s} {np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}")

    print(f"\npaired differences vs logistic regression ({len(SEEDS)} seeds):")
    for name, key in (("GBM - LR   (AUROC)", "gbm-lr"),
                      ("RF  - LR   (AUROC)", "rf-lr"),
                      ("GBM - LR   (Brier)", "gbm-lr-brier")):
        d = np.array(per_seed[key])
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else 0
        print(f"  {name:22s} {d.mean():+.4f} +/- {d.std(ddof=1):.4f} "
              f"(t={t:+.2f}) -> {'REAL' if abs(t) >= 2 else 'not established'}")

    gain = np.mean(per_seed["gbm-lr"])
    print(f"\nInterpretation: boosting buys {gain:+.4f} AUROC. If that is small, the "
          f"additive\nlog-odds model is not the bottleneck -- the data is -- and we keep "
          f"the model\nthat can be read off as one odds ratio per gene.")


if __name__ == "__main__":
    main()
