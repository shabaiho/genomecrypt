"""EXPERIMENT: does the extra BV-BRC data actually help, or did the split move?

The naive comparison (train on 1,992 vs 2,579 genomes, read the metrics) is
invalid: adding genomes changes the grouped split, so the two models are graded
on different test sets. Average balanced accuracy rose 0.730 -> 0.774 that way,
which could be entirely explained by an easier test set.

Controlled version: FREEZE one test set (drawn from the NCBI-only genomes), then
train twice -- with and without the BV-BRC additions -- and score both on it.
Only genomes whose SNP cluster is absent from the test set may be used for
training, so the comparison stays leak-free.

    uv run python scripts/exp_data_scaling.py
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

SCRATCH = Path("/tmp/claude-1000/-home-sabal-hackaton/"
               "b4c40662-f506-40a8-a4d9-3b024e7db077/scratchpad")


def fit_eval(Xtr, ytr, Xca, yca, Xte, yte, seed=0):
    best, best_auc = None, -1.0
    for C in (0.03, 0.1, 0.3, 1.0):
        m = LogisticRegression(penalty="l1", C=C, class_weight="balanced",
                               solver="liblinear", max_iter=5000)
        m.fit(Xtr, ytr)
        try:
            a = roc_auc_score(yca, m.predict_proba(Xca)[:, 1])
        except ValueError:
            continue
        if a > best_auc:
            best, best_auc = m, a
    if best is None:
        return None
    cal = CalibratedClassifierCV(_freeze(best), method="sigmoid", **_prefit_kwargs())
    cal.fit(Xca, yca)
    p = cal.predict_proba(Xte)[:, 1]
    return {
        "bal_acc": balanced_accuracy_score(yte, (p >= 0.5).astype(int)),
        "auroc": roc_auc_score(yte, p),
        "brier": brier_score_loss(yte, p),
        "nonzero": int((best.coef_[0] != 0).sum()),
    }


def main() -> None:
    cfg = Config.load()
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    L = pd.read_csv(ROOT / "data/processed/labels.csv")
    G = pd.read_csv(ROOT / "data/processed/groups.csv").set_index("genome_id").group_id

    base_ids = set(pd.read_parquet(SCRATCH / "feat_base.parquet").index)
    extra_ids = set(X.index) - base_ids
    print(f"NCBI-only genomes: {len(base_ids)} | BV-BRC additions: {len(extra_ids)}\n")

    rows = []
    for drug in cfg.drugs:
        sub = L[(L.drug_id == drug.id) & (L.genome_id.isin(X.index))]
        if sub.empty:
            continue
        # test + calib drawn ONLY from the original NCBI genomes, so both arms
        # face an identical, unchanged evaluation set
        nb = sub[sub.genome_id.isin(base_ids)]
        gb = G.loc[nb.genome_id].to_numpy()
        tr_b, ca_b, th_b, te_b = grouped_four_way(gb, 0)
        test_ids = nb.genome_id.to_numpy()[te_b]
        calib_ids = nb.genome_id.to_numpy()[ca_b]
        held_groups = set(G.loc[list(test_ids) + list(calib_ids)])

        def block(ids):
            s = sub[sub.genome_id.isin(ids)]
            return (X.loc[s.genome_id].to_numpy(np.float32), s.label.to_numpy(int))

        Xte, yte = block(test_ids)
        Xca, yca = block(calib_ids)
        if len(np.unique(yte)) < 2 or len(np.unique(yca)) < 2:
            continue

        for arm, pool in (("NCBI only", base_ids), ("NCBI + BV-BRC", base_ids | extra_ids)):
            train_ids = [g for g in sub.genome_id
                         if g in pool and g not in set(test_ids) | set(calib_ids)
                         and G.loc[g] not in held_groups]
            Xtr, ytr = block(train_ids)
            if len(np.unique(ytr)) < 2:
                continue
            r = fit_eval(Xtr, ytr, Xca, yca, Xte, yte)
            if r:
                rows.append({"drug": drug.id[:20], "arm": arm, "n_train": len(train_ids),
                             "n_test": len(yte), **{k: round(v, 4) for k, v in r.items()}})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\n--- средние (одинаковый тест в обеих ветках) ---")
    agg = df.groupby("arm")[["n_train", "bal_acc", "auroc", "brier"]].mean().round(4)
    print(agg.to_string())

    a, b = agg.loc["NCBI only"], agg.loc["NCBI + BV-BRC"]
    print(f"\nbal.acc {a.bal_acc:.4f} -> {b.bal_acc:.4f}  ({b.bal_acc - a.bal_acc:+.4f})")
    print(f"AUROC   {a.auroc:.4f} -> {b.auroc:.4f}  ({b.auroc - a.auroc:+.4f})")
    print(f"Brier   {a.brier:.4f} -> {b.brier:.4f}  ({b.brier - a.brier:+.4f}, lower is better)")
    verdict = "extra data helps" if b.auroc > a.auroc else "extra data does NOT help"
    print(f"\nverdict on the SAME test set: {verdict}")


if __name__ == "__main__":
    main()
