"""EXPERIMENT: how simple can the model be before it stops working?

Two questions the jury will actually ask.

(1) REGULARIZATION PATH. L1 has one knob, C. Small C = fewer non-zero
coefficients = an explanation a person can read. We trace AUROC against the
number of surviving genes and look for the knee, so "we used 8 genes" is a
measured choice rather than a taste.

(2) DOES DOMAIN RESTRICTION HELP? The model currently uses blaKPC (a
carbapenemase) as evidence about CIPROFLOXACIN, with a negative weight. That is a
spurious correlation: co-carriage on the same plasmids, not pharmacology.
Restricting each drug to determinants annotated for its own drug class is the
obvious fix -- but it is only worth doing if it does not cost accuracy. Testable,
so test it.

    uv run python scripts/exp_simplicity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import Config  # noqa: E402
from gfw.train import grouped_four_way  # noqa: E402

# AMRFinderPlus Class values that matter for each drug, plus the chromosomal
# genes whose mutation is an established mechanism for it.
DRUG_CLASSES = {
    "ciprofloxacin": ({"QUINOLONE", "PHENICOL/QUINOLONE"}, {"gyrA", "parC", "gyrB", "parE"}),
    "gentamicin": ({"AMINOGLYCOSIDE"}, {"rpsL"}),
    "meropenem": ({"BETA-LACTAM", "CARBAPENEM", "CEPHALOSPORIN"}, {"ompK35", "ompK36", "ftsI"}),
    "ceftriaxone": ({"BETA-LACTAM", "CEPHALOSPORIN"}, {"ompK35", "ompK36", "ftsI"}),
    "trimethoprim_sulfamethoxazole": ({"TRIMETHOPRIM", "SULFONAMIDE"}, {"folA", "folP"}),
}
# gene symbol prefixes that belong to a class, for tokens with no class rollup
CLASS_HINTS = {
    "QUINOLONE": ("qnr", "oqx", "aac(6')-Ib-cr"),
    "AMINOGLYCOSIDE": ("aac", "aad", "ant", "aph", "rmt", "arm", "str"),
    "BETA-LACTAM": ("bla", "amp", "ompK"),
    "CEPHALOSPORIN": ("bla",),
    "CARBAPENEM": ("bla",),
    "TRIMETHOPRIM": ("dfr",),
    "SULFONAMIDE": ("sul",),
}


def relevant_columns(schema: list[str], drug_id: str) -> list[int]:
    classes, genes = DRUG_CLASSES[drug_id]
    prefixes = tuple(p for c in classes for p in CLASS_HINTS.get(c, ()))
    keep = []
    for i, f in enumerate(schema):
        kind, _, name = f.partition(":")
        if kind == "class":
            if name.upper() in classes:
                keep.append(i)
        elif kind in ("mut", "mutgene", "trunc"):
            if name.split("_")[0] in genes:
                keep.append(i)
        elif kind in ("gene", "genefam"):
            if name.startswith(prefixes):
                keep.append(i)
    return keep


def main() -> None:
    cfg = Config.load()
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    L = pd.read_csv(ROOT / "data/processed/labels.csv")
    G = pd.read_csv(ROOT / "data/processed/groups.csv").set_index("genome_id").group_id
    schema = list(X.columns)

    grid = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    path_rows, restrict_rows = [], []

    for drug in cfg.drugs:
        sub = L[(L.drug_id == drug.id) & (L.genome_id.isin(X.index))]
        if sub.empty:
            continue
        Xa = X.loc[sub.genome_id].to_numpy(np.float32)
        ya = sub.label.to_numpy(int)
        ga = G.loc[sub.genome_id].to_numpy()
        tr, ca, th, te = grouped_four_way(ga, 0)

        for C in grid:
            m = LogisticRegression(penalty="l1", C=C, class_weight="balanced",
                                   solver="liblinear", max_iter=5000)
            m.fit(Xa[tr], ya[tr])
            path_rows.append({
                "drug": drug.id[:18], "C": C,
                "nonzero": int((m.coef_[0] != 0).sum()),
                "test_auroc": round(roc_auc_score(ya[te], m.decision_function(Xa[te])), 4),
            })

        cols = relevant_columns(schema, drug.id)
        for label, idx in (("all features", list(range(len(schema)))),
                           ("class-restricted", cols)):
            best_auc, best_C, best_nz = -1.0, None, None
            for C in grid:
                m = LogisticRegression(penalty="l1", C=C, class_weight="balanced",
                                       solver="liblinear", max_iter=5000)
                m.fit(Xa[tr][:, idx], ya[tr])
                a = roc_auc_score(ya[ca], m.decision_function(Xa[ca][:, idx]))
                if a > best_auc:
                    best_auc, best_C = a, C
                    best_nz = int((m.coef_[0] != 0).sum())
            m = LogisticRegression(penalty="l1", C=best_C, class_weight="balanced",
                                   solver="liblinear", max_iter=5000)
            m.fit(Xa[tr][:, idx], ya[tr])
            restrict_rows.append({
                "drug": drug.id[:18], "features": label, "n_avail": len(idx),
                "nonzero": best_nz, "C": best_C,
                "test_auroc": round(roc_auc_score(ya[te], m.decision_function(Xa[te][:, idx])), 4),
            })

    print("=== (1) регуляризационный путь: AUROC против числа генов ===")
    p = pd.DataFrame(path_rows)
    piv = p.pivot_table(index="C", values=["nonzero", "test_auroc"], aggfunc="mean").round(4)
    print(piv.to_string())
    knee = piv.test_auroc.idxmax()
    print(f"\nмаксимум AUROC при C={knee} ({piv.loc[knee, 'nonzero']:.0f} генов в среднем)")
    within = piv[piv.test_auroc >= piv.test_auroc.max() - 0.01]
    simplest = within.index.min()
    print(f"простейшая модель в пределах 0.01 AUROC: C={simplest} "
          f"({within.loc[simplest, 'nonzero']:.0f} генов, AUROC {within.loc[simplest, 'test_auroc']:.4f})")

    print("\n=== (2) ограничение признаков классом препарата ===")
    r = pd.DataFrame(restrict_rows)
    print(r.to_string(index=False))
    agg = r.groupby("features")[["n_avail", "nonzero", "test_auroc"]].mean().round(4)
    print("\n" + agg.to_string())
    d = agg.loc["class-restricted", "test_auroc"] - agg.loc["all features", "test_auroc"]
    print(f"\nAUROC при ограничении: {d:+.4f} "
          f"({'ограничение не вредит' if d > -0.01 else 'ограничение стоит точности'})")


if __name__ == "__main__":
    main()
