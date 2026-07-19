"""Held-out evaluation -> models/<version>/eval/report.json, rendered in the app.

Reports exactly what the brief scores on -- no single headline accuracy number.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from .config import Config, bundle_path, read_json, write_json
from .policy import trivial_baseline


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            out.append({"bin": [round(lo, 2), round(hi, 2)], "n": int(m.sum()),
                        "mean_pred": round(float(p[m].mean()), 3),
                        "observed": round(float(y[m].mean()), 3)})
    return out


def score_drug(y: np.ndarray, p: np.ndarray, low: float, high: float,
               threshold: float | None = None) -> dict:
    called = (p <= low) | (p >= high)
    # score at the operating point we actually ship, not a hardcoded 0.5
    op = 0.5 if threshold is None else threshold
    yhat = (p >= op).astype(int)
    n_missed = int(((y == 1) & (yhat == 0)).sum())
    res = {
        "n": int(len(y)),
        "operating_threshold": round(float(op), 4),
        "prevalence_resistant": round(float(y.mean()), 3),
        "balanced_accuracy": round(balanced_accuracy_score(y, yhat), 3),
        # the constraint: recall on RESISTANT. Missing one of these is the
        # failure mode we are explicitly trading false alarms to avoid.
        "recall_resistant": round(recall_score(y, yhat, pos_label=1, zero_division=0), 3),
        "missed_resistant": n_missed,
        # what proves we beat "always resistant": that model scores 0.0 here
        "specificity": round(recall_score(y, yhat, pos_label=0, zero_division=0), 3),
        "recall_susceptible": round(recall_score(y, yhat, pos_label=0, zero_division=0), 3),
        "precision": round(precision_score(y, yhat, zero_division=0), 3),
        "f1": round(f1_score(y, yhat, zero_division=0), 3),
        "trivial_baseline": trivial_baseline(y),
        "auroc": round(roc_auc_score(y, p), 3) if len(np.unique(y)) > 1 else None,
        "pr_auc": round(average_precision_score(y, p), 3) if len(np.unique(y)) > 1 else None,
        "brier": round(brier_score_loss(y, p), 4),
        "no_call_rate": round(float(1 - called.mean()), 3),
        "reliability": reliability(y, p),
    }
    # the number that actually matters clinically: are the calls we DO make right?
    if called.sum():
        res["balanced_accuracy_on_called"] = round(
            balanced_accuracy_score(y[called], yhat[called]), 3)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    args = ap.parse_args()

    cfg = Config.load()
    out = bundle_path(args.version)
    Xdf = pd.read_parquet(args.matrix)
    labels = pd.read_csv(args.labels)
    groups = pd.read_csv(args.groups).set_index("genome_id")["group_id"]
    splits = read_json(out / "splits.json")
    ab = cfg.abstain

    report = {"version": args.version, "per_drug": {}, "by_group": {}}
    for drug_id, split in splits.items():
        mp = out / f"{drug_id}.joblib"
        if not mp.exists():
            continue
        _b = joblib.load(mp)
        model, thr = _b["model"], _b.get("threshold")
        band = _b.get("band") or {}
        ids = [g for g in split["test_genome_ids"] if g in Xdf.index]
        sub = labels[(labels.drug_id == drug_id) & (labels.genome_id.isin(ids))]
        if sub.empty:
            continue
        X = Xdf.loc[sub.genome_id].to_numpy(dtype=np.float32)
        y = sub.label.to_numpy(dtype=int)
        p = model.predict_proba(X)[:, 1]

        mode = cfg.decision.get("mode", "calibrated_abstain")
        if mode == "calibrated_abstain":
            # score at the shipped band: 0.5 decides the call, the band decides
            # whether we make one at all
            lo, hi, op = band.get("low", ab["low"]), band.get("high", ab["high"]), 0.5
        else:
            lo, hi, op = ab["low"], ab["high"], thr
        report["per_drug"][drug_id] = score_drug(y, p, lo, hi, op)
        report["per_drug"][drug_id]["mode"] = mode
        report["per_drug"][drug_id]["abstain_band"] = [lo, hi]

        # generalization: same metrics broken out by homology group (unseen lineages)
        g = groups.loc[sub.genome_id].to_numpy()
        per_group = {}
        for gid in np.unique(g):
            m = g == gid
            if m.sum() >= 10 and len(np.unique(y[m])) > 1:
                per_group[int(gid)] = {
                    "n": int(m.sum()),
                    "balanced_accuracy": round(balanced_accuracy_score(
                        y[m], (p[m] >= (thr if thr is not None else 0.5)).astype(int)), 3),
                }
        report["by_group"][drug_id] = per_group

    write_json(out / "eval" / "report.json", report)
    print(pd.DataFrame(report["per_drug"]).T.drop(columns=["reliability"], errors="ignore"))


if __name__ == "__main__":
    main()
