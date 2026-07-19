"""Genome Firewall -- decision-report demo (Module 03).

Runs in the light container: no AMRFinderPlus needed if the user uploads a TSV.
Model weights are read from models/<version>/ mounted at runtime, so swapping a
model never means rebuilding the image.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gfw.annotate import amrfinder_available, run_amrfinder  # noqa: E402
from gfw.config import DEFAULT_MODEL_DIR, read_json  # noqa: E402
from gfw.features import determinants, parse_amrfinder_tsv, sniff_file_type  # noqa: E402
from gfw.predict import CALL_FAIL, CALL_NONE, CALL_WORK, Predictor  # noqa: E402
from gfw.gate import detect_targets, verify_species  # noqa: E402
from gfw.qc import check_assembly  # noqa: E402

st.set_page_config(page_title="Genome Firewall", page_icon="🧬", layout="wide")

CALL_STYLE = {
    CALL_FAIL: ("🔴", "Likely to FAIL"),
    CALL_WORK: ("🟢", "Likely to WORK"),
    CALL_NONE: ("⚪", "NO-CALL"),
}
EVIDENCE_LABEL = {
    "known_determinant": "Known resistance gene / DNA change detected",
    "statistical_only": "Statistical association only — not an established cause",
    "no_signal": "No known resistance signal found",
}


@st.cache_resource
def load_predictor(version: str) -> Predictor:
    return Predictor(version)


def bundles() -> list[str]:
    return sorted(p.name for p in DEFAULT_MODEL_DIR.glob("*") if (p / "metadata.json").exists())


st.title("🧬 Genome Firewall")
st.caption("Genome-based antibiotic-response prediction — research prototype")

st.error(
    "**RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.** Every result below must be "
    "confirmed by standard laboratory antimicrobial susceptibility testing before "
    "it informs any treatment decision. This tool is decision support for a trained "
    "healthcare or laboratory professional; it never makes a treatment decision.",
    icon="⚠️",
)

available = bundles()
if not available:
    st.warning(f"No model bundle found in `{DEFAULT_MODEL_DIR}`. Train one first: `make train`.")
    st.stop()

with st.sidebar:
    version = st.selectbox("Model bundle", available)
    pred = load_predictor(version)
    st.markdown(f"**Species:** {pred.cfg.species}")
    st.markdown(f"**Drugs served:** {', '.join(pred.served_drugs) or 'none'}")
    not_served = pred.meta.get("drugs_not_served", {})
    if not_served:
        st.markdown("**Not covered:**")
        for k, v in not_served.items():
            st.caption(f"· {k} — {v}")
    st.divider()
    st.caption(f"git {pred.meta.get('git_sha')} · {pred.meta.get('model')}")

tab_predict, tab_card = st.tabs(["Predict", "Model card"])

with tab_predict:
    st.caption("Upload an assembled genome (FASTA) or a precomputed AMRFinderPlus "
               "TSV — the file type is detected from its contents.")
    if not amrfinder_available():
        st.info("AMRFinderPlus is not on PATH, so FASTA input is unavailable. Run "
                "`make tools`, then `source .tools/env.sh` before starting the app — "
                "or upload a precomputed TSV.")

    up = st.file_uploader("Upload file", type=["tsv", "txt", "fna", "fasta", "fa"])

    if up is not None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / up.name
            src.write_bytes(up.getvalue())
            targets = None
            species_check = None
            assembly_qc = None

            # Detected from CONTENT. Trusting a mode toggle let a FASTA be parsed
            # as a TSV, which yields zero determinants and a confident
            # "likely to work" for every drug on a blaKPC-positive genome.
            kind = sniff_file_type(src)
            if kind == "protein_fasta":
                st.error("This looks like a PROTEIN FASTA. The pipeline annotates "
                         "nucleotide assemblies; upload the genome, not its proteome.")
                st.stop()
            if kind == "unknown":
                st.error("This file is neither a FASTA nor AMRFinderPlus output. "
                         "A FASTA starts with '>'; an AMRFinderPlus TSV has an "
                         "'Element symbol' (or 'Gene symbol') column.")
                st.stop()

            if kind == "fasta":
                if not amrfinder_available():
                    st.error("FASTA uploaded but AMRFinderPlus is not installed, so it "
                             "cannot be converted into features. Run `make tools`.")
                    st.stop()
                assembly_qc = check_assembly(src)
                if not assembly_qc["ok"]:
                    st.error(f"Assembly QC failed: {assembly_qc['reason']}", icon="🚫")
                for w in assembly_qc.get("warnings", []):
                    st.warning(w, icon="⚠️")
                st.success(
                    f"Detected: assembled FASTA — {assembly_qc['total_bp'] / 1e6:.2f} Mb "
                    f"in {assembly_qc['n_contigs']} contigs (N50 {assembly_qc['n50']:,} bp). "
                    f"Running the full pipeline, target gate included.")
                with st.spinner("Running AMRFinderPlus…"):
                    tsv = run_amrfinder(src, Path(td) / "amr.tsv", pred.cfg.species_taxgroup)
                try:
                    targets = detect_targets(src)
                    species_check = verify_species(src)
                    if not species_check.get("ok", True):
                        st.error(
                            f"Species check failed: chromosomal targets match the "
                            f"{pred.cfg.species} reference at only "
                            f"{species_check['identity']:.1f}% identity "
                            f"(expected >= 95%). This model covers {pred.cfg.species} "
                            f"only. Every drug is reported as no-call.", icon="🚫")
                except Exception as e:
                    st.warning(f"Target gate unavailable ({e}). Results are flagged accordingly.")
            else:
                tsv = src
                st.info("Detected: AMRFinderPlus TSV. The drug-target gate needs the "
                        "assembly, so 'likely to work' results are **not** "
                        "target-verified. Upload the FASTA for the full pipeline.", icon="ℹ️")

            report = pred.predict_from_tsv(tsv, sample_id=up.name, targets_found=targets,
                                           species_check=species_check,
                                           assembly_qc=assembly_qc)

            # A genome with no AMR determinants at all is possible but rare -- far
            # more often it means the input was misread. Say so instead of
            # reporting five confident "no resistance found" calls.
            n_det = len(determinants(parse_amrfinder_tsv(tsv)))
            if n_det == 0:
                st.warning("**No AMR determinants were detected in this file.** Every "
                           "prediction below is therefore made on an all-zero feature "
                           "vector. Check that the file is what you expect before "
                           "trusting any 'likely to work' result.", icon="⚠️")
            else:
                st.caption(f"{n_det} determinants detected.")

        st.subheader("Antibiotic-response report")
        for r in report.results:
            icon, label = CALL_STYLE[r.call]
            conf = f"{r.confidence:.0%}" if r.confidence is not None else "—"
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 2, 5])
                c1.markdown(f"### {icon} {r.display}")
                c1.markdown(f"**{label}**")
                if r.reason == "recall_constrained_threshold":
                    # p is a probability of resistance, not confidence in the call
                    c2.metric("P(resistant)", conf)
                    c2.caption("called FAIL above the recall-tuned threshold, "
                               "which sits well below 50%")
                else:
                    c2.metric("Confidence", conf)
                    c2.caption(f"reason: `{r.reason}`")
                c3.markdown(f"**Evidence:** {EVIDENCE_LABEL.get(r.evidence_category, r.evidence_category)}")
                if r.supporting:
                    sup = pd.DataFrame(r.supporting)
                    # UI columns unchanged for now; mechanistic_for_drug ships in the JSON
                    sup = sup[[c for c in sup.columns if c != "mechanistic_for_drug"]]
                    c3.dataframe(sup, hide_index=True, use_container_width=True)
                    if not sup.curated_determinant.all():
                        c3.caption("⚠️ Rows with `curated_determinant = False` are statistical "
                                   "associations only — a weight is not proof of a biological cause.")
                for n in r.notes:
                    c3.caption(f"· {n}")

        if report.unknown_determinants:
            st.warning(f"{len(report.unknown_determinants)} resistance determinants in this genome "
                       f"were never seen in training: `{', '.join(report.unknown_determinants[:10])}`"
                       " — predictions may be extrapolating.")

        st.download_button("Download report (JSON)",
                           json.dumps(report.to_dict(), indent=2),
                           file_name=f"{up.name}.genome-firewall.json")

with tab_card:
    ev_path = DEFAULT_MODEL_DIR / version / "eval" / "report.json"
    st.markdown("### Held-out performance (grouped split, unseen lineages)")
    if not ev_path.exists():
        st.info("No evaluation report yet — run `make eval`.")
    else:
        ev = read_json(ev_path)
        rows = {k: {m: v for m, v in d.items()
                    if m not in ("reliability", "trivial_baseline")}
                for k, d in ev["per_drug"].items()}
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

        # A recall-constrained model must be shown against "always predict
        # resistant", which scores recall 1.0 by construction. Specificity is
        # what separates them: the trivial model scores 0.0.
        st.markdown("**vs. trivial baseline** (always predict resistant)")
        cmp_rows = {}
        for k, d in ev["per_drug"].items():
            t = d.get("trivial_baseline", {})
            cmp_rows[k] = {
                "model F1": d.get("f1"), "trivial F1": t.get("f1"),
                "model specificity": d.get("specificity"), "trivial specificity": 0.0,
                "model recall(R)": d.get("recall_resistant"), "trivial recall(R)": 1.0,
                "missed resistant": d.get("missed_resistant"),
            }
        st.dataframe(pd.DataFrame(cmp_rows).T, use_container_width=True)
        st.caption("A high F1 at recall≈1 is not evidence of a good model — the "
                   "trivial baseline achieves it too. Specificity is the honest signal.")

        drug = st.selectbox("Reliability diagram", list(ev["per_drug"]))
        d = ev["per_drug"][drug]
        rel = pd.DataFrame(d["reliability"])
        if not rel.empty:
            import plotly.graph_objects as go

            fig = go.Figure()
            # the diagonal IS the point of the plot -- without it the curve is unreadable
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="perfect calibration",
                                     line=dict(dash="dash", color="gray")))
            fig.add_trace(go.Scatter(
                x=rel.mean_pred, y=rel.observed, mode="markers+lines",
                name="observed", marker=dict(size=rel.n, sizemode="area",
                                             sizeref=max(rel.n) / 400, sizemin=4),
                hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<br>n=%{marker.size}<extra></extra>",
            ))
            band = d.get("abstain_band")
            if band:
                fig.add_vrect(x0=band[0], x1=band[1], fillcolor="orange", opacity=0.12,
                              line_width=0, annotation_text="no-call band",
                              annotation_position="top left")
            fig.update_layout(
                xaxis_title="predicted P(resistant)", yaxis_title="observed fraction resistant",
                xaxis_range=[0, 1], yaxis_range=[0, 1], height=420,
                margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Brier score", f"{d.get('brier', 0):.3f}",
                      help="Mean squared error of the probabilities. Lower is better; "
                           "0.25 is what you get by always guessing 0.5.")
            c2.metric("No-call rate", f"{d.get('no_call_rate', 0):.0%}")
            c3.metric("Bal. accuracy on calls made",
                      f"{d.get('balanced_accuracy_on_called', 0):.3f}")
            st.caption("Marker area is the number of held-out samples in that bin — a "
                       "point far off the diagonal with n=4 behind it is noise, not "
                       "miscalibration. Perfect calibration follows the dashed line.")

        # L1 sparsity: what the model actually looks at, per drug
        counts = pred.meta.get("training_counts", {}).get(drug, {})
        if counts.get("nonzero_features") is not None:
            st.markdown(
                f"**Model for {drug}:** L1 logistic regression, "
                f"`C={counts.get('C')}`, **{counts['nonzero_features']} non-zero "
                f"coefficients** out of {len(pred.schema)} features. "
                f"Selection rule: {counts.get('C_selection', {}).get('rule', 'n/a')}."
            )
            st.caption("L1 zeroes out most features on purpose — a handful of named "
                       "determinants is an explanation a person can check, a 300-term "
                       "dot product is not.")

    st.markdown("### Scope & safety")
    st.markdown(
        f"- Covers **{pred.cfg.species}** only. Any other species is out of scope; "
        "results would be meaningless.\n"
        "- Predicts resistance that **already exists**. It never designs, modifies, "
        "or suggests changes to an organism.\n"
        "- Starts from an assembled, quality-checked genome — sample handling, "
        "sequencing, species ID and assembly are out of scope.\n"
        "- Returns **no-call** on weak, conflicting, or out-of-distribution evidence "
        "rather than forcing a yes/no answer."
    )
