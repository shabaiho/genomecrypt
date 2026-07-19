"""Repeat the whole pipeline over several seeds -> mean +/- std per drug.

WHY. Every number this project reported for most of its life came from a single
grouped split with seed=0. Measured spread across splits is +/- 0.0465 AUROC,
larger than several of the effects that were claimed on the strength of one draw.
A model card showing one number per drug is not just incomplete, it is misleading.

Writes models/<version>/eval/stability.json, which the app's model card renders
next to the point estimates.

    python -m gfw.stability --version v18 --seeds 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, recall_score, roc_auc_score,
)

from .config import Config, bundle_path, write_json
from .policy import fit_abstain_band
from .train import fit_one_drug

METRICS = ("balanced_accuracy", "auroc", "pr_auc", "brier", "f1",
           "recall_resistant", "specificity", "no_call_rate",
           "balanced_accuracy_on_called")


def score(y: np.ndarray, p: np.ndarray, lo: float, hi: float) -> dict:
    yhat = (p >= 0.5).astype(int)
    called = (p <= lo) | (p >= hi)
    out = {
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, p),
        "f1": f1_score(y, yhat, zero_division=0),
        "recall_resistant": recall_score(y, yhat, pos_label=1, zero_division=0),
        "specificity": recall_score(y, yhat, pos_label=0, zero_division=0),
        "no_call_rate": float(1 - called.mean()),
    }
    out["balanced_accuracy_on_called"] = (
        balanced_accuracy_score(y[called], yhat[called])
        if called.sum() and len(np.unique(y[called])) > 1 else np.nan
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="current",
                    help="bundle to evaluate; defaults to the served one")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--matrix", type=Path, default=Path("data/processed/features.parquet"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/labels.csv"))
    ap.add_argument("--groups", type=Path, default=Path("data/processed/groups.csv"))
    args = ap.parse_args()

    cfg = Config.load()
    Xdf = pd.read_parquet(args.matrix)
    labels = pd.read_csv(args.labels)
    groups = pd.read_csv(args.groups).set_index("genome_id")["group_id"]

    per_drug: dict[str, dict[str, list[float]]] = {}
    for drug in cfg.drugs:
        sub = labels[(labels.drug_id == drug.id) & (labels.genome_id.isin(Xdf.index))]
        if sub.empty:
            continue
        X = Xdf.loc[sub.genome_id].to_numpy(np.float32)
        y = sub.label.to_numpy(int)
        g = groups.loc[sub.genome_id].to_numpy()

        runs: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in range(args.seeds):
            fitted = fit_one_drug(X, y, g, seed)
            if fitted is None:
                continue
            cal, _base, (_tr, _ca, th, te) = fitted
            band = fit_abstain_band(
                y[th], cal.predict_proba(X[th])[:, 1],
                target_accuracy=cfg.abstain.get("target_accuracy", 0.90),
                max_no_call_rate=cfg.abstain.get("max_no_call_rate", 0.30),
            )
            p = cal.predict_proba(X[te])[:, 1]
            if len(np.unique(y[te])) < 2:
                continue
            for k, v in score(y[te], p, band["low"], band["high"]).items():
                runs[k].append(float(v))
        per_drug[drug.id] = runs
        got = runs["auroc"]
        print(f"{drug.id[:26]:26s} {len(got)} seeds  "
              f"AUROC {np.nanmean(got):.3f} +/- {np.nanstd(got, ddof=1):.3f}")

    summary = {
        "seeds": args.seeds,
        "per_drug": {
            d: {m: {"mean": round(float(np.nanmean(v)), 4),
                    "std": round(float(np.nanstd(v, ddof=1)), 4),
                    "min": round(float(np.nanmin(v)), 4),
                    "max": round(float(np.nanmax(v)), 4),
                    "n": int(np.sum(~np.isnan(v)))}
                for m, v in runs.items() if len(v) and not np.all(np.isnan(v))}
            for d, runs in per_drug.items()
        },
    }
    overall = {m: [np.nanmean(r[m]) for r in per_drug.values() if len(r[m])]
               for m in METRICS}
    summary["overall"] = {m: {"mean": round(float(np.nanmean(v)), 4),
                              "std": round(float(np.nanstd(v, ddof=1)), 4)}
                          for m, v in overall.items() if v}

    out = bundle_path(args.version) / "eval" / "stability.json"
    write_json(out, summary)
    o = summary["overall"]
    print(f"\nacross drugs: balanced accuracy {o['balanced_accuracy']['mean']:.3f} "
          f"+/- {o['balanced_accuracy']['std']:.3f}, "
          f"AUROC {o['auroc']['mean']:.3f} +/- {o['auroc']['std']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
