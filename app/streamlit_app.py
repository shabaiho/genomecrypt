"""Genome Firewall -- antibiotic-response report (Module 03).

Two audiences, two views, and they need opposite things. A clinician needs a
short ranked answer and a reason in words. A reviewer needs calibration curves,
AUROC and coefficients. Serving both on one screen served neither, so the
default view is clinical and everything quantitative lives behind a tab.
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
from gfw.explain import (  # noqa: E402
    detected_mechanisms, headline_evidence, supporting_sentences,
)
from gfw.features import determinants, parse_amrfinder_tsv, sniff_file_type  # noqa: E402
from gfw.gate import detect_targets, verify_species  # noqa: E402
from gfw.predict import CALL_FAIL, CALL_NONE, CALL_WORK, Predictor  # noqa: E402
from gfw.qc import check_assembly  # noqa: E402

st.set_page_config(page_title="Genome Firewall", page_icon="🧬", layout="wide")

# Why a drug lands in a group, in words the report can print verbatim.
NO_ANSWER_REASON = {
    "low_confidence": "The evidence is mixed. The system will not guess.",
    "ood": "This genome carries resistance genes the system has never seen. "
           "Anything it said here would be extrapolation.",
    "target_absent": "The gene this drug acts on was not found in the genome.",
    "wrong_species": "This genome is not the species the system was built for.",
    "assembly_qc_failed": "The assembly is incomplete, so absent genes cannot be "
                          "told apart from missing sequence.",
    "no_determinants_detected": "No resistance markers were found at all, which "
                                "points to a failed annotation rather than a "
                                "susceptible isolate.",
    "drug_not_covered": "This drug is not covered by the current model.",
    "intrinsic": "This species is naturally resistant to this drug.",
}


@st.cache_resource
def load_predictor(version: str) -> Predictor:
    return Predictor(version)


def bundles() -> list[str]:
    return sorted(p.name for p in DEFAULT_MODEL_DIR.glob("*") if (p / "metadata.json").exists())


def track_record(version: str) -> str | None:
    """How often calls like these turned out right, read from measured files.

    Never hardcode the numbers here -- they go stale the moment the model is
    retrained, and a stale accuracy claim on a clinical screen is worse than none.
    """
    ev = DEFAULT_MODEL_DIR / version / "eval" / "stability.json"
    if not ev.exists():
        return None
    stab = read_json(ev)
    s = stab["overall"]
    line = (f"Across {stab['seeds']} independent evaluations on bacteria the system "
            f"had never seen, its balanced accuracy was "
            f"{s['balanced_accuracy']['mean']:.0%} "
            f"(± {s['balanced_accuracy']['std']:.1%}).")
    cohort = DEFAULT_MODEL_DIR / version / "eval" / "demo_cohort.json"
    if cohort.exists():
        c = read_json(cohort)
        line += (f" On {c['n_genomes']} held-out genomes it made {c['n_called']} calls, "
                 f"{c['n_correct']} of which matched the laboratory result "
                 f"({c['accuracy']:.0%}).")
    return line


st.title("🧬 Genome Firewall")
st.caption("Antibiotic-response prediction from a bacterial genome — research prototype")

available = bundles()
if not available:
    st.warning(f"No model bundle found in `{DEFAULT_MODEL_DIR}`. Train one first: `make train`.")
    st.stop()

with st.sidebar:
    version = st.selectbox("Model", available)
    pred = load_predictor(version)
    st.markdown(f"**Organism:** {pred.cfg.species}")
    st.markdown(f"**Antibiotics:** {len(pred.served_drugs)}")
    not_served = pred.meta.get("drugs_not_served", {})
    if not_served:
        st.caption("Not covered: " + ", ".join(not_served))
    st.divider()
    st.caption(f"build {pred.meta.get('git_sha')}")

tab_report, tab_tech = st.tabs(["Antibiotic report", "For reviewers"])

# ---------------------------------------------------------------- report ---
with tab_report:
    st.info(
        "**Decision support only.** Send the sample for standard susceptibility "
        "testing regardless of what this page says. Nothing here replaces a "
        "culture result or a clinician's judgement.",
        icon="⚠️",
    )

    up = st.file_uploader(
        "Upload an assembled genome (FASTA) or AMRFinderPlus output (TSV)",
        type=["tsv", "txt", "fna", "fasta", "fa"],
        help="The file type is detected from its contents, not its name.",
    )

    if up is not None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / up.name
            src.write_bytes(up.getvalue())
            targets = species_check = assembly_qc = None
            problems: list[str] = []

            # Detected from CONTENT. Trusting a mode toggle once let a FASTA be
            # parsed as a TSV, yielding zero determinants and a confident
            # "likely to work" for a blaKPC-positive genome.
            kind = sniff_file_type(src)
            if kind == "protein_fasta":
                st.error("This is a protein FASTA. Please upload the assembled "
                         "genome (nucleotide sequence).")
                st.stop()
            if kind == "unknown":
                st.error("This file is neither a genome FASTA nor AMRFinderPlus "
                         "output, so it cannot be analysed.")
                st.stop()

            if kind == "fasta":
                if not amrfinder_available():
                    st.error("Genome uploaded, but the annotation tool is not "
                             "installed on this machine. Run `make tools`.")
                    st.stop()
                assembly_qc = check_assembly(src)
                if not assembly_qc["ok"]:
                    problems.append(assembly_qc["reason"])
                for w in assembly_qc.get("warnings", []):
                    problems.append(w)
                with st.spinner("Reading the genome…"):
                    tsv = run_amrfinder(src, Path(td) / "amr.tsv", pred.cfg.species_taxgroup)
                try:
                    targets = detect_targets(src)
                    species_check = verify_species(src)
                    if not species_check.get("ok", True):
                        problems.append(
                            f"This genome does not match {pred.cfg.species} "
                            f"({species_check['identity']:.0f}% identity to the "
                            f"reference genes, {95}% required).")
                except Exception as e:
                    problems.append(f"Species and target checks could not run ({e}).")
            else:
                tsv = src
                problems.append(
                    "Annotation file uploaded instead of a genome, so the organism "
                    "and the drug targets could not be verified. Upload the genome "
                    "for the full set of checks.")

            report = pred.predict_from_tsv(tsv, sample_id=up.name, targets_found=targets,
                                           species_check=species_check,
                                           assembly_qc=assembly_qc)
            n_det = len(determinants(parse_amrfinder_tsv(tsv)))
            qc_line = ""
            if assembly_qc:
                qc_line = (f"{assembly_qc['total_bp'] / 1e6:.2f} Mb in "
                           f"{assembly_qc['n_contigs']} contigs · ")

        # ---- sample header ----
        st.markdown(f"#### {up.name}")
        st.caption(f"{pred.cfg.species} · {qc_line}{n_det} resistance markers found")

        for p in problems:
            st.warning(p, icon="⚠️")

        avoid = [r for r in report.results if r.call == CALL_FAIL]
        maybe = [r for r in report.results if r.call == CALL_WORK]
        unknown = [r for r in report.results if r.call == CALL_NONE]
        def _conf(r):
            return r.confidence if r.confidence is not None else 0.0

        avoid.sort(key=_conf, reverse=True)
        maybe.sort(key=_conf, reverse=True)

        def drug_row(r, icon: str, show_prob: bool = True) -> None:
            with st.container(border=True):
                left, right = st.columns([9, 1])
                left.markdown(f"**{icon}&nbsp; {r.display}**")
                if r.call == CALL_NONE:
                    left.markdown(NO_ANSWER_REASON.get(
                        r.reason, "The evidence is not strong enough to call."))
                    # A refusal must not bury a determinant we did detect.
                    found = detected_mechanisms(
                        [s if isinstance(s, dict) else s.__dict__ for s in r.supporting])
                    if found:
                        left.warning(
                            "A resistance mechanism was nonetheless detected: "
                            + "; ".join(found)
                            + ". The model is not confident enough to call the drug, "
                              "but this finding stands on its own.", icon="⚠️")
                else:
                    left.markdown(headline_evidence(
                        [s if isinstance(s, dict) else s.__dict__ for s in r.supporting],
                        r.call))
                    extra = supporting_sentences(
                        [s if isinstance(s, dict) else s.__dict__ for s in r.supporting],
                        r.call)
                    if extra:
                        with left.expander("Other findings for this drug"):
                            for e in extra:
                                st.markdown(f"- {e}")
                if show_prob and r.confidence is not None:
                    # small and grey on purpose: informative for a reviewer,
                    # safely ignorable at the bedside
                    right.caption(f"{r.confidence:.0%}")

        if avoid:
            st.markdown("### Avoid — resistance detected")
            for r in avoid:
                drug_row(r, "⛔")

        if maybe:
            st.markdown("### May work — no resistance markers found")
            for r in maybe:
                drug_row(r, "✅")

        if unknown:
            st.markdown("### No answer — insufficient evidence")
            st.caption("The system declines these rather than guessing. Treat them as "
                       "though this tool had not been run.")
            for r in unknown:
                drug_row(r, "❔", show_prob=False)

        if report.unknown_determinants:
            st.warning(
                f"This genome carries {len(report.unknown_determinants)} resistance "
                f"markers the system has not seen before "
                f"(`{', '.join(report.unknown_determinants[:5])}`). Treat every "
                f"result above with extra caution.", icon="⚠️")

        rec = track_record(version)
        if rec:
            st.caption(rec)

        with st.expander("What this tool does not do"):
            st.markdown(
                f"- It does not identify the organism. It assumes "
                f"{pred.cfg.species} and refuses genomes that do not match.\n"
                "- It does not process samples, sequence DNA, or assemble genomes. "
                "It starts from a finished assembly.\n"
                "- It does not separate mixed samples, choose a dose, or account for "
                "the site of infection.\n"
                "- It does not replace susceptibility testing. Every result above "
                "needs laboratory confirmation."
            )

        with st.expander("Technical detail for this sample"):
            rowsd = []
            for r in report.results:
                for s in r.supporting:
                    d = s if isinstance(s, dict) else s.__dict__
                    rowsd.append({"drug": r.drug_id, **d})
            if rowsd:
                st.dataframe(pd.DataFrame(rowsd), hide_index=True,
                             use_container_width=True)
                st.caption("`weight` is the logistic-regression coefficient: the gene "
                           "multiplies the odds of resistance by e^weight. "
                           "`mechanistic_for_drug = False` means the gene is a linked "
                           "marker, not a mechanism for this drug.")
            st.download_button("Download full report (JSON)",
                               json.dumps(report.to_dict(), indent=2),
                               file_name=f"{up.name}.genome-firewall.json")

# ------------------------------------------------------------ reviewers ---
with tab_tech:
    ev_path = DEFAULT_MODEL_DIR / version / "eval" / "report.json"
    if not ev_path.exists():
        st.info("No evaluation report yet — run `make eval`.")
    else:
        ev = read_json(ev_path)

        stab_path = DEFAULT_MODEL_DIR / version / "eval" / "stability.json"
        if stab_path.exists():
            stab = read_json(stab_path)
            st.markdown(f"### Repeated over {stab['seeds']} independent grouped splits")
            st.caption("Point estimates from a single split are unreliable here: the "
                       "measured spread is ±0.02 AUROC. Every split re-draws the "
                       "SNP-cluster grouping.")
            band = {
                drug: {m.replace("_", " "): f"{mm[m]['mean']:.3f} ± {mm[m]['std']:.3f}"
                       for m in ("balanced_accuracy", "auroc", "pr_auc", "brier",
                                 "recall_resistant", "specificity", "no_call_rate")
                       if m in mm}
                for drug, mm in stab["per_drug"].items()
            }
            st.dataframe(pd.DataFrame(band).T, use_container_width=True)
            o = stab["overall"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Balanced accuracy", f"{o['balanced_accuracy']['mean']:.3f}",
                      f"± {o['balanced_accuracy']['std']:.3f}", delta_color="off")
            c2.metric("AUROC", f"{o['auroc']['mean']:.3f}",
                      f"± {o['auroc']['std']:.3f}", delta_color="off")
            c3.metric("Brier", f"{o['brier']['mean']:.3f}",
                      f"± {o['brier']['std']:.3f}", delta_color="off")

        st.markdown("### Shipped split (the model being served)")
        rows = {k: {m: v for m, v in d.items()
                    if m not in ("reliability", "trivial_baseline")}
                for k, d in ev["per_drug"].items()}
        st.dataframe(pd.DataFrame(rows).T, use_container_width=True)

        st.markdown("### vs. trivial baseline (always predict resistant)")
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
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="perfect calibration",
                                     line=dict(dash="dash", color="gray")))
            fig.add_trace(go.Scatter(
                x=rel.mean_pred, y=rel.observed, mode="markers+lines", name="observed",
                marker=dict(size=rel.n, sizemode="area",
                            sizeref=max(rel.n) / 400, sizemin=4),
                hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<extra></extra>",
            ))
            bandv = d.get("abstain_band")
            if bandv:
                fig.add_vrect(x0=bandv[0], x1=bandv[1], fillcolor="orange", opacity=0.12,
                              line_width=0, annotation_text="no-call band",
                              annotation_position="top left")
            fig.update_layout(xaxis_title="predicted P(resistant)",
                              yaxis_title="observed fraction resistant",
                              xaxis_range=[0, 1], yaxis_range=[0, 1], height=420,
                              margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Marker area is the number of held-out samples in that bin — a "
                       "point far off the diagonal with n=4 is noise, not "
                       "miscalibration.")

        counts = pred.meta.get("training_counts", {}).get(drug, {})
        if counts.get("nonzero_features") is not None:
            st.markdown(
                f"**Model for {drug}:** L1 logistic regression, `C={counts.get('C')}`, "
                f"**{counts['nonzero_features']} non-zero coefficients** out of "
                f"{len(pred.schema)} features. Selection rule: "
                f"{counts.get('C_selection', {}).get('rule', 'n/a')}. "
                f"Calibrated with Platt scaling, which keeps the model in the logistic "
                f"family so exp(coefficient) remains an odds ratio."
            )

        st.markdown("### Scope & safety")
        st.markdown(
            f"- Covers **{pred.cfg.species}** only; other species are refused by a "
            "95% identity check against chromosomal target genes.\n"
            "- Predicts resistance that **already exists**. It never designs, "
            "modifies, or suggests changes to an organism.\n"
            "- Starts from an assembled genome — sample handling, sequencing, species "
            "identification and assembly are out of scope.\n"
            "- Returns **no answer** on weak, conflicting or out-of-distribution "
            "evidence rather than forcing a yes/no."
        )
