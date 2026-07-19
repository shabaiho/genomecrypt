"""Decision policy: turn a calibrated probability into a call.

CLINICAL ASYMMETRY (what the user asked for)
--------------------------------------------
Missing a resistant isolate is far worse than a false alarm: a missed resistance
means the patient gets an antibiotic that will fail. So recall on the RESISTANT
class is a hard constraint, and everything else is optimized subject to it.

WHY WE DO NOT LITERALLY OPTIMIZE "recall == 1.0 with best F1"
-------------------------------------------------------------
Predicting "resistant" for everything scores recall = 1.000 by construction. On
the real BV-BRC label balance that trivial model also scores:

    ceftriaxone    F1 0.906   |  ciprofloxacin  F1 0.863
    trim/sulfa     F1 0.816   |  gentamicin     F1 0.598
    meropenem      F1 0.488   |  all: balanced accuracy 0.500

So a high F1 under a recall=1 constraint proves nothing -- it is exactly what a
model that learned nothing produces. The number that separates a real model from
the trivial one is SPECIFICITY (recall on the susceptible class), which is 0.0
for the trivial model by definition.

Policy implemented here:
    maximize specificity  subject to  recall_resistant >= target_recall
i.e. pick the LARGEST threshold that still catches the required share of
resistant isolates. Threshold is fitted on the calibration split, never on train.
F1 and specificity are then reported on the held-out split, next to the trivial
baseline, so the comparison is explicit.
"""
from __future__ import annotations

import numpy as np


def trivial_baseline(y: np.ndarray) -> dict:
    """Metrics of 'always predict resistant' -- the bar any recall-constrained
    model must clear. Report it beside every real result."""
    prev = float(np.mean(y))
    return {
        "strategy": "always_resistant",
        "recall_resistant": 1.0,
        "specificity": 0.0,
        "precision": round(prev, 3),
        "f1": round(2 * prev / (1 + prev), 3) if prev > 0 else 0.0,
        "balanced_accuracy": 0.5,
    }


def fit_threshold(
    y: np.ndarray,
    p: np.ndarray,
    target_recall: float = 0.99,
    safety_margin: float = 0.005,
) -> dict:
    """Largest threshold whose recall on the calibration split meets the target.

    `safety_margin` overshoots the target on calibration because held-out recall
    is systematically a little lower -- fitting exactly at the target reliably
    lands below it on test. Returns the threshold plus what it cost.
    """
    goal = min(1.0, target_recall + safety_margin)
    pos = y == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        return {"threshold": 0.5, "achievable": False, "reason": "no resistant examples"}

    # recall is non-increasing in the threshold, so scan candidates descending
    # and keep the largest one that still satisfies the goal.
    candidates = np.unique(np.concatenate([p, [0.0, 1.0]]))
    best_t, best_spec = 0.0, 0.0
    for t in candidates:
        pred = p >= t
        recall = pred[pos].mean()
        if recall < goal:
            continue
        spec = 1.0 - pred[~pos].mean() if (~pos).any() else 0.0
        if t >= best_t:
            best_t, best_spec = float(t), float(spec)

    achieved = float((p >= best_t)[pos].mean())
    return {
        "threshold": round(best_t, 4),
        "target_recall": target_recall,
        "calib_recall": round(achieved, 4),
        "calib_specificity": round(best_spec, 4),
        "achievable": achieved >= target_recall,
        # specificity 0 means the constraint forced "call everything resistant":
        # the model adds nothing at this recall target for this drug.
        "degenerate": best_spec <= 0.01,
    }


def fit_abstain_band(
    y: np.ndarray,
    p: np.ndarray,
    target_accuracy: float = 0.90,
    max_no_call_rate: float = 0.30,
) -> dict:
    """Widest-accuracy / narrowest-band tradeoff, derived from data.

    A hardcoded 0.40-0.60 band is arbitrary: it assumes the useful uncertainty
    always sits around 0.5, which is false once classes are uneven or the model
    is confident. Instead we widen the band outward from the median until the
    calls we DO make hit `target_accuracy`, and stop early if we would be
    abstaining on more than `max_no_call_rate` of samples -- an honest no-call is
    a strength, but a tool that abstains half the time is not decision support.

    Fitted on the threshold split (never train, never test).
    """
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"low": 0.35, "high": 0.65, "reason": "insufficient data, using default"}

    # A contiguous band is the right shape ONLY because we calibrate with Platt.
    # Under isotonic the probabilities collapsed onto ~15 plateaus (a third of all
    # samples tied on one value), so widening the band by 0.01 could swallow 25%
    # of the data at once and the fit stalled at 0.48-0.52. Platt gives ~120
    # distinct smooth probabilities, so scanning outward from 0.5 behaves.
    # This is the payoff for fixing the calibrator instead of patching the policy:
    # the plateau-refusal workaround that isotonic forced is simply not needed.
    best = {"low": 0.5, "high": 0.5, "accuracy_on_called": 0.0, "no_call_rate": 0.0}
    for half in np.arange(0.0, 0.45, 0.01):
        lo, hi = 0.5 - half, 0.5 + half
        called = (p <= lo) | (p >= hi)
        rate = 1.0 - called.mean()
        if rate > max_no_call_rate or called.sum() < 10:
            break
        acc = float(((p[called] >= 0.5).astype(int) == y[called]).mean())
        best = {"low": round(float(lo), 3), "high": round(float(hi), 3),
                "accuracy_on_called": round(acc, 4),
                "no_call_rate": round(float(rate), 4)}
        if acc >= target_accuracy:
            break
    best["target_accuracy"] = target_accuracy
    best["reason"] = ("reached target accuracy" if best["accuracy_on_called"] >= target_accuracy
                      else "stopped at max_no_call_rate before reaching target")
    return best


def apply(p: float, threshold: float) -> str:
    """Forced binary call -- no abstention. See config `decision.mode`."""
    from .predict import CALL_FAIL, CALL_WORK

    return CALL_FAIL if p >= threshold else CALL_WORK
