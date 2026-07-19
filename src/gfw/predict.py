"""Module 03 -- inference. Loads an artifact bundle and produces the decision report.

This module must NOT import training code, sourmash, or amrfinder. It is what
ships in the light app container. Everything it needs lives in models/<version>/.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .config import Config, bundle_path, read_json
from .features import determinants, parse_amrfinder_tsv, require_file_type, vectorize
from .gate import apply_gate
from .mechanism import is_mechanistic

CALL_FAIL = "likely_to_fail"
CALL_WORK = "likely_to_work"
CALL_NONE = "no_call"

# evidence categories, exactly as the brief enumerates them
EV_KNOWN = "known_determinant"        # (i)   a known resistance gene / point mutation was detected
EV_STATISTICAL = "statistical_only"   # (ii)  model leaned on features with no established causal link
EV_NONE = "no_signal"                 # (iii) no known resistance signal found

# Feature prefixes that trace back to a CURATED AMRFinderPlus determinant.
# genefam:/mutgene:/trunc: are our own rollups OF those curated calls, so they are
# just as curated as the exact allele -- omitting them labelled `genefam:blaKPC`,
# the single strongest carbapenemase signal, as "not an established cause".
# `class:` is deliberately excluded: it is a drug-class bucket, not a determinant.
CURATED_PREFIXES = ("gene:", "mut:", "genefam:", "mutgene:", "trunc:")


def is_curated(feature: str) -> bool:
    return feature.startswith(CURATED_PREFIXES)


def evidence_category(call: str, support: list[dict]) -> str:
    """Which of the brief's three evidence types backs THIS call.

    The old logic was `EV_KNOWN if support else EV_NONE`, which announced
    "Known resistance gene detected" for a *likely to work* call whose top
    features all pushed toward susceptibility. The category has to depend on
    whether a curated determinant supports the direction actually reported.
    """
    if not support:
        return EV_NONE
    want = "toward_resistant" if call == CALL_FAIL else "toward_susceptible"
    aligned = [s for s in support if s["direction"] == want]
    if any(s["curated_determinant"] for s in aligned):
        return EV_KNOWN
    return EV_STATISTICAL


@dataclass
class DrugReport:
    drug_id: str
    display: str
    call: str
    confidence: float | None
    evidence_category: str
    reason: str
    supporting: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class SampleReport:
    sample_id: str
    species: str
    model_version: str
    results: list[DrugReport]
    unknown_determinants: list[str]
    disclaimer: str = (
        "RESEARCH PROTOTYPE - NOT FOR CLINICAL USE. Every result must be confirmed "
        "by standard laboratory antimicrobial susceptibility testing before it "
        "informs any treatment decision."
    )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [asdict(r) if not isinstance(r, dict) else r for r in self.results]
        return d


class Predictor:
    def __init__(self, version: str = "current", config: Config | None = None):
        self.dir = bundle_path(version)
        if not self.dir.exists():
            raise FileNotFoundError(f"no model bundle at {self.dir}")
        self.cfg = config or Config.load()
        self.meta = read_json(self.dir / "metadata.json")
        self.schema: list[str] = read_json(self.dir / "feature_schema.json")["features"]
        self.models = {
            p.stem: joblib.load(p) for p in self.dir.glob("*.joblib")
        }

    @property
    def served_drugs(self) -> list[str]:
        return sorted(self.models)

    def predict_from_tsv(
        self,
        tsv_path: Path,
        sample_id: str,
        targets_found: set[str] | None = None,
        species_check: dict | None = None,
        assembly_qc: dict | None = None,
    ) -> SampleReport:
        # refuse a FASTA handed in as a TSV rather than silently returning
        # "no resistance found" for every drug
        require_file_type(tsv_path, "amrfinder_tsv")
        tokens = determinants(parse_amrfinder_tsv(tsv_path))
        return self.predict_from_tokens(tokens, sample_id, targets_found,
                                        species_check, assembly_qc)

    def predict_from_tokens(
        self,
        tokens: set[str],
        sample_id: str,
        targets_found: set[str] | None = None,
        species_check: dict | None = None,
        assembly_qc: dict | None = None,
    ) -> SampleReport:
        # Assembly too incomplete -> refuse everything. Missing sequence reads as
        # missing resistance, which is the dangerous direction.
        if assembly_qc is not None and not assembly_qc.get("ok", True):
            return SampleReport(
                sample_id=sample_id, species=self.cfg.species,
                model_version=self.meta.get("version", "unknown"),
                results=[DrugReport(d.id, d.display, CALL_NONE, None, EV_NONE,
                                    "assembly_qc_failed",
                                    notes=[assembly_qc.get("reason", "")])
                         for d in self.cfg.drugs],
                unknown_determinants=[],
            )

        # Wrong species -> refuse everything. No per-drug reasoning applies when
        # the organism is not the one the model was trained on.
        if species_check is not None and not species_check.get("ok", True):
            note = (f"Target genes match the {self.cfg.species} reference at only "
                    f"{species_check.get('identity', 0):.1f}% identity; this model "
                    f"only covers {self.cfg.species}. Confirm the species and use a "
                    f"model built for it.")
            return SampleReport(
                sample_id=sample_id, species=self.cfg.species,
                model_version=self.meta.get("version", "unknown"),
                results=[DrugReport(d.id, d.display, CALL_NONE, None, EV_NONE,
                                    "wrong_species", notes=[note]) for d in self.cfg.drugs],
                unknown_determinants=sorted(t for t in tokens if t not in set(self.schema)),
            )

        # ZERO determinants is not evidence of susceptibility. Every K. pneumoniae
        # carries some AMR determinant (fosA and a chromosomal blaSHV are close to
        # universal), so an empty set means the annotation failed or the input was
        # not what we think -- a header-only AMRFinderPlus TSV produced a confident
        # "likely to work" before this guard existed.
        if not tokens:
            note = ("No AMR determinants at all were detected. For this species that "
                    "indicates a failed or empty annotation, not a susceptible isolate. "
                    "Re-run the annotation before interpreting anything.")
            return SampleReport(
                sample_id=sample_id, species=self.cfg.species,
                model_version=self.meta.get("version", "unknown"),
                results=[DrugReport(d.id, d.display, CALL_NONE, None, EV_NONE,
                                    "no_determinants_detected", notes=[note])
                         for d in self.cfg.drugs],
                unknown_determinants=[],
            )

        x, unknown = vectorize(tokens, self.schema)
        ab = self.cfg.abstain
        known_frac_unknown = len(unknown) / max(1, len([t for t in tokens if t.startswith(("gene:", "mut:"))]))
        ood = known_frac_unknown > ab["ood_max_unknown_frac"]

        results: list[DrugReport] = []
        for drug in self.cfg.drugs:
            gate = apply_gate(drug, targets_found)
            if not gate.ok:
                call = CALL_FAIL if gate.reason == "intrinsic" else CALL_NONE
                results.append(DrugReport(
                    drug.id, drug.display, call,
                    1.0 if gate.reason == "intrinsic" else None,
                    EV_KNOWN if gate.reason == "intrinsic" else EV_NONE,
                    gate.reason, notes=[str(gate.detail)],
                ))
                continue

            bundle = self.models.get(drug.id)
            if bundle is None:
                results.append(DrugReport(
                    drug.id, drug.display, CALL_NONE, None, EV_NONE, "drug_not_covered",
                    notes=[self.meta.get("drugs_not_served", {}).get(drug.id, "not in this model bundle")],
                ))
                continue

            p = float(bundle["model"].predict_proba(x)[0, 1])
            support = self._supporting(bundle, x, drug.id)

            notes = []
            if gate.reason == "gate_skipped":
                notes.append("Target gate could not run; 'likely to work' is NOT target-verified.")

            # --- high-sensitivity policy: forced call, no abstention band ---
            if self.cfg.decision.get("mode") == "high_sensitivity":
                thr = bundle.get("threshold", 0.5)
                pol = bundle.get("policy", {})
                if ood and self.cfg.decision.get("keep_ood_no_call", True):
                    results.append(DrugReport(
                        drug.id, drug.display, CALL_NONE, round(p, 3), EV_STATISTICAL, "ood",
                        support, notes + [f"{len(unknown)} determinants unseen in training"],
                    ))
                    continue
                call = CALL_FAIL if p >= thr else CALL_WORK
                notes.append(
                    f"High-sensitivity mode: threshold {thr:.3f} tuned for "
                    f"recall>={pol.get('target_recall', '?')} on resistant "
                    f"(calibration recall {pol.get('calib_recall', '?')}, "
                    f"specificity {pol.get('calib_specificity', '?')}). "
                    "'Likely to work' is deliberately conservative; false alarms are "
                    "accepted to avoid missing resistance."
                )
                if pol.get("degenerate"):
                    notes.append("WARNING: at this recall target the model calls "
                                 "everything resistant -- no better than the trivial baseline.")
                # Report the raw P(resistant), NOT a "confidence in the call".
                # The threshold sits far below 0.5, so a FAIL call at p=0.14 is
                # correct policy but would read as a broken 14% confidence.
                results.append(DrugReport(
                    drug.id, drug.display, call, round(p, 3),
                    evidence_category(call, support),
                    "recall_constrained_threshold", support, notes,
                ))
                continue

            # per-drug band fitted on held-out data; falls back to the config
            # constants only for bundles trained before bands existed
            band = bundle.get("band") or {}
            lo = band.get("low", ab["low"])
            hi = band.get("high", ab["high"])
            in_band = lo < p < hi
            if band:
                notes.append(
                    f"No-call band {lo:.2f}-{hi:.2f} fitted on held-out data: abstains on "
                    f"{band.get('no_call_rate', 0):.0%} of samples, "
                    f"{band.get('accuracy_on_called', 0):.0%} accurate on the rest."
                )

            if ood:
                results.append(DrugReport(
                    drug.id, drug.display, CALL_NONE, round(p, 3), EV_STATISTICAL, "ood",
                    support, notes + [f"{len(unknown)} determinants unseen in training"],
                ))
            elif not in_band and p >= 0.5:
                results.append(DrugReport(
                    drug.id, drug.display, CALL_FAIL, round(p, 3),
                    evidence_category(CALL_FAIL, support), "above_threshold", support, notes,
                ))
            elif not in_band:
                results.append(DrugReport(
                    drug.id, drug.display, CALL_WORK, round(1 - p, 3),
                    evidence_category(CALL_WORK, support), "below_threshold", support, notes,
                ))
            else:
                results.append(DrugReport(
                    drug.id, drug.display, CALL_NONE, round(p, 3),
                    EV_STATISTICAL, "low_confidence", support, notes,
                ))

        return SampleReport(
            sample_id=sample_id,
            species=self.cfg.species,
            model_version=self.meta.get("version", "unknown"),
            results=results,
            unknown_determinants=unknown,
        )

    @staticmethod
    def _supporting(bundle: dict, x: np.ndarray, drug_id: str = "", top_k: int = 5) -> list[dict]:
        """Present features whose weight pushed this call, most influential first.

        NOTE for the writeup: a coefficient is a STATISTICAL association. We label
        evidence as `known_determinant` only when the contributing feature is a
        curated AMRFinderPlus AMR/POINT determinant -- the coefficient magnitude
        itself never proves biological causation.
        """
        coef, schema = bundle["coef"], bundle["schema"]
        present = np.nonzero(x[0])[0]
        # Drop zero-weight features. Under L1 most coefficients are exactly 0, and
        # a 0 contributes nothing to the decision -- listing it as "evidence"
        # (previously rendered as "toward_susceptible", because 0 is not > 0) is
        # simply false.
        contrib = sorted(((int(i), float(coef[i])) for i in present if coef[i] != 0),
                         key=lambda t: -abs(t[1]))[:top_k]
        return [
            {"feature": schema[i], "weight": round(w, 3),
             "direction": "toward_resistant" if w > 0 else "toward_susceptible",
             "curated_determinant": is_curated(schema[i]),
             # a curated determinant can still be pharmacologically unrelated to
             # THIS drug -- blaKPC predicting ciprofloxacin is plasmid linkage
             "mechanistic_for_drug": is_mechanistic(schema[i], drug_id)}
            for i, w in contrib
        ]
