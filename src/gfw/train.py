"""Module 02c -- training. Writes a self-contained artifact bundle to models/<version>/.

Run:  python -m gfw.train --matrix data/processed/features.parquet \
                          --labels data/processed/labels.csv \
                          --groups data/processed/groups.csv \
                          --version v1

Split policy (three-way, ALL grouped by homology cluster -- no group spans splits):
    train  60%  -- fit the logistic regression
    calib  20%  -- fit the probability calibrator (isotonic). MUST be disjoint from
                   train, otherwise calibration is fit on in-sample scores and the
                   reliability plot lies.
    test   20%  -- held out; the only numbers we report.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .config import REPO_ROOT, Config, bundle_path, write_json
from .policy import fit_abstain_band, fit_threshold, trivial_baseline


def grouped_three_way(groups: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(groups))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    tr, rest = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    c_rel, t_rel = next(gss2.split(rest, groups=groups[rest]))
    return tr, rest[c_rel], rest[t_rel]


def grouped_four_way(groups: np.ndarray, seed: int = 0):
    """train / calib / thresh / test, no group spanning any two of them.

    The threshold MUST NOT be picked on the same rows the isotonic calibrator was
    fit on. Doing so selects an operating point against in-sample calibrator
    output: it reports recall 1.000 and then under-delivers on real data. Splitting
    the old calibration block in half costs a little calibrator precision and buys
    an honest recall estimate.
    """
    tr, cal_all, te = grouped_three_way(groups, seed)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    c_rel, t_rel = next(gss.split(cal_all, groups=groups[cal_all]))
    return tr, cal_all[c_rel], cal_all[t_rel], te


def fit_one_drug(X, y, groups, seed=0):
    """Returns (calibrated_model, base_model, split_indices) or None if unservable.

    Splits are train / calib / thresh / test -- see grouped_four_way for why the
    threshold gets its own block.
    """
    tr, ca, th, te = grouped_four_way(groups, seed)
    if any(len(np.unique(y[s])) < 2 for s in (tr, ca, th)):
        return None

    # L1 (Lasso) rather than L2: with ~300 mostly-zero binary features, L1 drives
    # nearly all coefficients to exactly 0 and leaves a handful of named
    # determinants per drug. That is the difference between an explanation a
    # clinician can read and a 300-term dot product.
    # C is selected on the CALIBRATION split -- never on train (in-sample optimism)
    # and never on test (that would leak the held-out set into model selection).
    best_score, grid, fits = -1.0, [], []
    for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        m = LogisticRegression(
            penalty="l1",
            C=C,
            class_weight="balanced",  # AMR datasets are imbalanced per drug
            solver="liblinear",
            max_iter=5000,
        )
        m.fit(X[tr], y[tr])
        # balanced accuracy, not accuracy: the classes are uneven per drug
        # AUROC, not balanced accuracy: threshold-free and far less jumpy on a
        # ~120-row calibration block. Selecting on balanced accuracy dropped
        # blaKPC-3 from the meropenem model, which then called a KPC-positive
        # genome "likely to work" -- a false-susceptible, the worst error here.
        score = roc_auc_score(y[ca], m.predict_proba(X[ca])[:, 1])
        grid.append({"C": C, "calib_auroc": round(float(score), 4),
                     "nonzero": int((m.coef_[0] != 0).sum())})
        fits.append((C, m, float(score)))
        if score > best_score:
            best_score = score

    # ONE-STANDARD-ERROR RULE. Picking the raw argmax on a ~150-row calibration
    # split is noise-chasing: it selected C=3 for gentamicin, giving 159 nonzero
    # coefficients topped by mcr-1.1 (a COLISTIN gene, weight -5.98) and qnrB6 (a
    # QUINOLONE gene) -- textbook spurious correlation, and exactly the failure the
    # brief warns about. Instead take the SPARSEST model whose score is within one
    # standard error of the best. Same accuracy within noise, far fewer features,
    # and the survivors are the ones with real signal.
    n_cal = max(len(ca), 2)
    # HALF a standard error, not a full one. The full 1-SE rule over-regularized
    # meropenem: it dropped blaKPC-3 -- the actual carbapenemase -- and left 4
    # features topped by an aminoglycoside gene, trading a causal determinant for
    # sparsity. Half-SE keeps the biology and still kills the mcr-1.1 nonsense.
    se = 0.5 * float(np.sqrt(best_score * (1 - best_score) / n_cal))
    ok = [(C, m, s) for C, m, s in fits if s >= best_score - se]
    C_sel, base, score_sel = min(ok, key=lambda t: t[0])  # smallest C == sparsest
    base.selection_grid_ = grid
    base.selection_rule_ = {
        "rule": "half_standard_error",
        "best_calib_auroc": round(best_score, 4),
        "standard_error": round(se, 4),
        "selected_C": C_sel,
        "selected_calib_auroc": round(score_sel, 4),
    }

    # isotonic calibration on the disjoint calibration split, fit against the
    # ALREADY-TRAINED base model (sklearn >=1.6 spells that FrozenEstimator;
    # <1.6 spells it cv="prefit").
    # PLATT (sigmoid), not isotonic. Two reasons, both measured -- see
    # scripts/exp_calibration.py:
    #   1. Statistics: isotonic is non-parametric and wants ~1000+ calibration
    #      points; our blocks are 114-171. It overfits into a coarse step function
    #      -- 15 distinct probabilities, a third of all samples tied on one value.
    #      Held out it lost on Brier (0.214 vs 0.203), ECE (0.121 vs 0.117) and even
    #      AUROC (0.749 vs 0.763), because ties forfeit ranking credit.
    #   2. Interpretability: sigmoid keeps the model in the logistic family,
    #      logit(p) = a*(b0 + sum_i beta_i x_i) + b, so each gene still shifts the
    #      log-odds by a fixed amount and exp(a*beta_i) is its odds ratio. After
    #      isotonic that statement is simply not true of the output.
    calibrated = CalibratedClassifierCV(_freeze(base), method="sigmoid", **_prefit_kwargs())
    calibrated.fit(X[ca], y[ca])
    return calibrated, base, (tr, ca, th, te)


def _freeze(est):
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        return FrozenEstimator(est)
    except ImportError:
        return est


def _prefit_kwargs() -> dict:
    try:
        import sklearn.frozen  # noqa: F401
        return {}
    except ImportError:
        return {"cv": "prefit"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True, help="genome x feature parquet")
    ap.add_argument("--labels", type=Path, required=True, help="genome_id,drug_id,label csv")
    ap.add_argument("--groups", type=Path, required=True, help="genome_id,group_id csv")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config.load()
    Xdf = pd.read_parquet(args.matrix)

    # The demo genome must never enter ANY split. Adding the BV-BRC source
    # reshuffled the grouped split and quietly pulled it into training for two
    # drugs, which would have made the "never seen in training" claim false while
    # the demo still looked fine. Excluding it by id (and by SNP cluster, so its
    # close relatives go too) keeps the claim true whatever the data does next.
    import json as _json

    demo_ids: set[str] = set()
    demo_path = REPO_ROOT / "data" / "demo" / "truth.json"
    if demo_path.exists():
        t = _json.loads(demo_path.read_text())
        demo_ids |= {t.get("isolate"), t.get("sample_id")} - {None}
    # the whole demonstration cohort, selected model-free by
    # scripts/build_demo_set.py before any training run
    set_path = REPO_ROOT / "data" / "demo" / "demo_set.json"
    if set_path.exists():
        demo_ids |= {g["genome_id"] for g in _json.loads(set_path.read_text())["genomes"]}
    labels = pd.read_csv(args.labels)
    groups = pd.read_csv(args.groups).set_index("genome_id")["group_id"]

    present = [g for g in demo_ids if g in groups.index]
    if present:
        demo_groups = set(groups.loc[present])
        blocked = set(groups[groups.isin(demo_groups)].index)
        labels = labels[~labels.genome_id.isin(blocked)]
        print(f"excluded {len(blocked)} genomes from the demo SNP cluster(s) "
              f"{sorted(demo_groups)} -- held out for demonstration")

    schema = list(Xdf.columns)
    out = bundle_path(args.version)
    out.mkdir(parents=True, exist_ok=True)

    served, skipped, splits = {}, {}, {}
    for drug in cfg.drugs:
        sub = labels[labels.drug_id == drug.id]
        sub = sub[sub.genome_id.isin(Xdf.index)]
        if sub.empty:
            skipped[drug.id] = "no labelled genomes"
            continue

        minority = int(min(sub.label.sum(), len(sub) - sub.label.sum()))
        if minority < cfg.abstain["min_train_support"]:
            skipped[drug.id] = f"minority class n={minority} below min_train_support"
            continue

        X = Xdf.loc[sub.genome_id].to_numpy(dtype=np.float32)
        y = sub.label.to_numpy(dtype=int)
        g = groups.loc[sub.genome_id].to_numpy()

        fitted = fit_one_drug(X, y, g, args.seed)
        if fitted is None:
            skipped[drug.id] = "a split ended up single-class after grouping"
            continue
        calibrated, base, (tr, ca, th, te) = fitted

        # Recall-constrained operating point, fitted on the CALIBRATION split.
        # Fitting it on train would pick a threshold that looks perfect in-sample
        # and misses resistant isolates in the field.
        dec = cfg.decision
        thr = fit_threshold(
            y[th], calibrated.predict_proba(X[th])[:, 1],
            target_recall=dec.get("target_recall", 0.99),
            safety_margin=dec.get("safety_margin", 0.005),
        )
        thr["trivial_baseline"] = trivial_baseline(y[th])

        # data-derived no-call band for calibrated_abstain mode
        band = fit_abstain_band(
            y[th], calibrated.predict_proba(X[th])[:, 1],
            target_accuracy=cfg.abstain.get("target_accuracy", 0.90),
            max_no_call_rate=cfg.abstain.get("max_no_call_rate", 0.30),
        )

        joblib.dump(
            {"model": calibrated, "coef": base.coef_[0], "schema": schema,
             "threshold": thr["threshold"], "policy": thr, "band": band},
            out / f"{drug.id}.joblib",
        )
        served[drug.id] = {
            "n_total": len(sub), "n_resistant": int(y.sum()),
            "n_train": len(tr), "n_calib": len(ca), "n_thresh": len(th), "n_test": len(te),
            "threshold": thr["threshold"],
            "nonzero_features": int((base.coef_[0] != 0).sum()),
            "C": float(base.C),
            "C_grid": getattr(base, "selection_grid_", []),
            "C_selection": getattr(base, "selection_rule_", {}),
            "abstain_band": [band["low"], band["high"]],
            "band_no_call_rate": band.get("no_call_rate"),
            "band_accuracy_on_called": band.get("accuracy_on_called"),
            "calib_recall": thr["calib_recall"],
            "calib_specificity": thr["calib_specificity"],
            "degenerate": thr["degenerate"],
        }
        if thr["degenerate"]:
            print(f"  WARNING {drug.id}: at recall>={dec.get('target_recall')} the "
                  f"threshold calls everything resistant (specificity 0) -- this drug "
                  f"adds nothing over the trivial baseline at this target")
        # persisted so evaluate.py scores the exact same held-out rows
        splits[drug.id] = {"test_genome_ids": sub.genome_id.to_numpy()[te].tolist()}
        print(f"trained {drug.id}: {served[drug.id]}")

    write_json(out / "feature_schema.json", {"features": schema, "n_features": len(schema)})
    write_json(out / "splits.json", splits)
    write_json(out / "metadata.json", {
        "version": args.version,
        "species": cfg.species,
        "drugs_served": sorted(served),
        "drugs_not_served": skipped,     # surfaced verbatim in the app's coverage panel
        "training_counts": served,
        "model": "LogisticRegression(l1, balanced, C by AUROC) + Platt calibration",
        "abstain": cfg.abstain,
        "decision": cfg.decision,
        "git_sha": _git_sha(),
        "seed": args.seed,
    })
    print(f"\nbundle -> {out}")
    if skipped:
        print("NOT SERVED (app will say so explicitly):")
        for k, v in skipped.items():
            print(f"  {k}: {v}")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
