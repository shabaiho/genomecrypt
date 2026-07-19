"""EXPERIMENT: were the earlier claims real? Paired comparison over many seeds.

exp_stability.py showed the seed-to-seed spread in mean AUROC is 0.0465 -- larger
than every effect claimed earlier. That does NOT automatically invalidate them:
that spread is the variance of the ABSOLUTE score across different test sets. When
two methods are compared on the SAME split, the shared test set cancels and the
relevant quantity is the variance of the DIFFERENCE, which is much smaller.

So: for each seed, fit both arms on the identical split, take the difference, and
report mean +/- std of that difference. A paired effect is real if its mean is
several standard errors from zero.

    uv run python scripts/exp_paired.py
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

SEEDS = list(range(10))
GRID = (0.03, 0.1, 0.3, 1.0)
SCRATCH = Path("/tmp/claude-1000/-home-sabal-hackaton/"
               "b4c40662-f506-40a8-a4d9-3b024e7db077/scratchpad")


def fit_base(Xtr, ytr, Xca, yca):
    best, best_auc = None, -1.0
    for C in GRID:
        m = LogisticRegression(penalty="l1", C=C, class_weight="balanced",
                               solver="liblinear", max_iter=5000)
        m.fit(Xtr, ytr)
        a = roc_auc_score(yca, m.decision_function(Xca))
        if a > best_auc:
            best, best_auc = m, a
    return best


def report(name: str, diffs: list[float], higher_better: bool = True) -> None:
    d = np.array(diffs)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(len(d))
    t = mean / se if se > 0 else 0.0
    win = (d > 0).sum() if higher_better else (d < 0).sum()
    verdict = ("REAL" if abs(t) >= 2 else "not established")
    print(f"{name:34s} {mean:+.4f} +/- {sd:.4f}  (SE {se:.4f}, t={t:+.2f})  "
          f"wins {win}/{len(d)}  -> {verdict}")


def main() -> None:
    cfg = Config.load()
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    L = pd.read_csv(ROOT / "data/processed/labels.csv")
    G = pd.read_csv(ROOT / "data/processed/groups.csv").set_index("genome_id").group_id
    base_ids = set(pd.read_parquet(SCRATCH / "feat_base.parquet").index)

    auc_cal, brier_cal, auc_data = [], [], []

    for seed in SEEDS:
        pd_auc_cal, pd_brier_cal, pd_auc_data = [], [], []
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

            # --- A: calibration method, identical base model and split ---
            base = fit_base(Xa[tr], ya[tr], Xa[ca], ya[ca])
            got = {}
            for method in ("isotonic", "sigmoid"):
                cal = CalibratedClassifierCV(_freeze(base), method=method, **_prefit_kwargs())
                cal.fit(Xa[ca], ya[ca])
                p = cal.predict_proba(Xa[te])[:, 1]
                got[method] = (roc_auc_score(ya[te], p), brier_score_loss(ya[te], p))
            pd_auc_cal.append(got["sigmoid"][0] - got["isotonic"][0])
            pd_brier_cal.append(got["sigmoid"][1] - got["isotonic"][1])

            # --- B: extra data, identical test set (drawn from base genomes) ---
            nb = sub[sub.genome_id.isin(base_ids)]
            if len(nb) < 100:
                continue
            gb = G.loc[nb.genome_id].to_numpy()
            tr_b, ca_b, th_b, te_b = grouped_four_way(gb, seed)
            test_ids = set(nb.genome_id.to_numpy()[te_b])
            calib_ids = set(nb.genome_id.to_numpy()[ca_b])
            held = set(G.loc[list(test_ids | calib_ids)])
            if not test_ids or not calib_ids:
                continue

            def blk(ids):
                s = sub[sub.genome_id.isin(ids)]
                return X.loc[s.genome_id].to_numpy(np.float32), s.label.to_numpy(int)

            Xte, yte = blk(test_ids)
            Xca2, yca2 = blk(calib_ids)
            if len(np.unique(yte)) < 2 or len(np.unique(yca2)) < 2:
                continue

            scores = {}
            for arm, pool in (("base", base_ids), ("merged", set(X.index))):
                ids = [g for g in sub.genome_id if g in pool
                       and g not in test_ids | calib_ids and G.loc[g] not in held]
                Xtr2, ytr2 = blk(ids)
                if len(np.unique(ytr2)) < 2:
                    scores = {}
                    break
                m = fit_base(Xtr2, ytr2, Xca2, yca2)
                cal = CalibratedClassifierCV(_freeze(m), method="sigmoid", **_prefit_kwargs())
                cal.fit(Xca2, yca2)
                scores[arm] = roc_auc_score(yte, cal.predict_proba(Xte)[:, 1])
            if len(scores) == 2:
                pd_auc_data.append(scores["merged"] - scores["base"])

        if pd_auc_cal:
            auc_cal.append(np.mean(pd_auc_cal))
            brier_cal.append(np.mean(pd_brier_cal))
        if pd_auc_data:
            auc_data.append(np.mean(pd_auc_data))

    print(f"Paired differences over {len(SEEDS)} seeds "
          f"(same split, same test set in both arms)\n")
    print(f"{'comparison':34s} {'mean diff':>10s}")
    report("AUROC: sigmoid - isotonic", auc_cal)
    report("Brier: sigmoid - isotonic", brier_cal, higher_better=False)
    report("AUROC: merged data - base data", auc_data)
    print("\nRule used: |t| >= 2 (about 95% confidence) counts as established.")


if __name__ == "__main__":
    main()
