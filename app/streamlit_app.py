"""Genome Firewall -- antibiotic-response report (Module 03).

Two audiences with opposite needs. A clinician needs a verdict and a reason. A
reviewer needs calibration, AUROC and coefficients. One screen served neither, so
the default view is a clinical dashboard and everything quantitative sits behind
a second tab.

Wording rules enforced throughout: state findings, not judgements. No "some",
"weak", "strong", "nearly every" -- where prevalence matters it is a measured
number. No advice on what to prescribe; the mandatory laboratory-confirmation
notice is the one required exception.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gfw.annotate import amrfinder_available, run_amrfinder  # noqa: E402
from gfw.config import DEFAULT_MODEL_DIR, read_json  # noqa: E402
from gfw.explain import (  # noqa: E402
    describe_feature, detected_mechanisms, gene_label, headline_evidence,
    supporting_sentences,
)
from gfw.features import determinants, parse_amrfinder_tsv, sniff_file_type  # noqa: E402
from gfw.gate import detect_targets, verify_species  # noqa: E402
from gfw.predict import CALL_FAIL, CALL_NONE, CALL_WORK, Predictor  # noqa: E402
from gfw.qc import check_assembly  # noqa: E402
from gfw.storage import (  # noqa: E402
    already_annotated, history, load_sample, save_annotation, save_report, save_upload,
)

st.set_page_config(page_title="Genome Firewall", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1240px;}
  html, body, [class*="css"] {font-size: 17px;}

  .kpi {background: var(--secondary-background-color); border-radius: 12px;
        padding: 18px 22px; height: 100%;}
  .kpi .label {font-size: .85rem; letter-spacing: .08em; text-transform: uppercase;
               opacity: .6;}
  .kpi .value {font-size: 2rem; font-weight: 650; line-height: 1.2; margin-top: 4px;}
  .kpi .sub {font-size: 1rem; opacity: .7; margin-top: 2px;}

  .drug {border-radius: 12px; padding: 18px 22px; margin-bottom: 12px;
         background: var(--secondary-background-color); border-left: 6px solid #999;
         display: flex; gap: 16px; align-items: flex-start;}
  .drug.fail {border-left-color: #d64545;}
  .drug.work {border-left-color: #2f9e44;}
  .drug.none {border-left-color: #9aa0a6;}
  .drug .ico {flex: 0 0 auto; margin-top: 3px;}
  .drug.fail .ico {color: #d64545;}
  .drug.work .ico {color: #2f9e44;}
  .drug.none .ico {color: #9aa0a6;}
  .drug .body {flex: 1 1 auto;}
  .drug .name {font-size: 1.5rem; font-weight: 650; line-height: 1.2;}
  .drug .why {font-size: 1.12rem; opacity: .9; margin-top: 6px; line-height: 1.45;}
  .drug .pct {flex: 0 0 auto; font-size: 1rem; opacity: .45;
              font-variant-numeric: tabular-nums; margin-top: 6px;}

  .grouphead {font-size: 1.05rem; letter-spacing: .07em; text-transform: uppercase;
              font-weight: 600; opacity: .7; margin: 30px 0 12px;}

  .notice {display: flex; gap: 14px; align-items: flex-start; border-radius: 12px;
           padding: 16px 20px; margin: 12px 0; font-size: 1.08rem; line-height: 1.5;
           border: 1px solid transparent;}
  .notice .ico {flex: 0 0 auto; margin-top: 2px;}
  .notice.warn {background: rgba(214,145,45,.12); border-color: rgba(214,145,45,.35);
                color: inherit;}
  .notice.warn .ico {color: #d6912d;}
  .notice.stop {background: rgba(214,69,69,.12); border-color: rgba(214,69,69,.35);}
  .notice.stop .ico {color: #d64545;}
  .notice.note {background: rgba(90,120,180,.12); border-color: rgba(90,120,180,.32);}
  .notice.note .ico {color: #5a78b4;}
</style>
""", unsafe_allow_html=True)

# Inline SVG, sized in em so it scales with the surrounding text. No emoji anywhere
# in this interface: emoji render differently on every platform and read as
# decoration on a clinical screen.
SVG = {
    "warn": '<svg class="ico" width="22" height="22" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 '
            '0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/>'
            '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "stop": '<svg class="ico" width="22" height="22" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
            '<line x1="4.9" y1="4.9" x2="19.1" y2="19.1"/></svg>',
    "note": '<svg class="ico" width="22" height="22" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="16" x2="12" y2="12"/>'
            '<line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    "check": '<svg class="ico" width="26" height="26" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
             'stroke-linejoin="round"><path d="M22 11.1V12a10 10 0 1 1-5.9-9.1"/>'
             '<polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "ban": '<svg class="ico" width="26" height="26" viewBox="0 0 24 24" fill="none" '
           'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
           'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
           '<line x1="4.9" y1="4.9" x2="19.1" y2="19.1"/></svg>',
    "dash": '<svg class="ico" width="26" height="26" viewBox="0 0 24 24" fill="none" '
            'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
            'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/>'
            '<line x1="8" y1="12" x2="16" y2="12"/></svg>',
}


def notice(kind: str, text: str) -> None:
    """Alert box without Streamlit's built-in emoji icons."""
    st.markdown(f'<div class="notice {kind}">{SVG[kind]}<div>{text}</div></div>',
                unsafe_allow_html=True)

# Why a drug got no verdict, stated as fact.
NO_VERDICT_REASON = {
    "low_confidence": "The genome contains conflicting signals for this drug.",
    "ood": "This genome carries resistance genes absent from the reference set.",
    "target_absent": "The gene this drug binds was not found in the genome.",
    "wrong_species": "The genome does not match the organism this model covers.",
    "assembly_qc_failed": "The assembly is incomplete; missing sequence cannot be "
                          "distinguished from absent genes.",
    "no_determinants_detected": "No resistance markers were found in the file.",
    "drug_not_covered": "This drug is not in the current model.",
    "intrinsic": "This species carries intrinsic resistance to this drug.",
}


@st.cache_resource
def load_predictor(version: str) -> Predictor:
    return Predictor(version)


def bundles() -> list[str]:
    return sorted(p.name for p in DEFAULT_MODEL_DIR.glob("*") if (p / "metadata.json").exists())


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "demo" / "GCA_000417485.1.fna"

# AMRFinderPlus spawns blastn and hmmer, each peaking near a gigabyte. Two
# concurrent runs are what crashed this app: starting an upload while the example
# was still being analysed made Streamlit re-run the script, a second annotation
# began beside the first, and the process died on memory. One at a time.
ANNOTATE_LOCK = threading.Lock()


@st.cache_data(show_spinner=False, max_entries=4)
def read_pending(kind: str, ref: str) -> tuple[bytes, str]:
    """Resolve a queued sample to bytes, lazily.

    The reference is what lives in session state -- a path or a content digest,
    never the file itself. Holding 5.5 MB of genome in session state meant every
    interaction copied it.
    """
    if kind == "example":
        return EXAMPLE.read_bytes(), EXAMPLE.name
    loaded = load_sample(ref)
    if loaded is None:
        return b"", ref
    return loaded[0], loaded[1]


def browser_id() -> str:
    """A random id kept in the page URL.

    Deliberately not an account: no email, no password, nothing authenticated.
    It exists so two people trying the demo on the same server do not see each
    other's history. Clearing it is as easy as removing it from the address bar.
    """
    qp = st.query_params
    bid = qp.get("id")
    if not bid:
        bid = uuid.uuid4().hex[:12]
        qp["id"] = bid
    return bid


def as_dicts(supporting) -> list[dict]:
    return [s if isinstance(s, dict) else s.__dict__ for s in supporting]


st.title("Genome Firewall")
st.caption("Antibiotic-response prediction from a bacterial genome — research prototype")

available = bundles()
if not available:
    notice("stop", f"No model bundle found in <code>{DEFAULT_MODEL_DIR}</code>. "
                   f"Train one first: <code>make train</code>.")
    st.stop()

with st.sidebar:
    version = st.selectbox("Model", available)
    pred = load_predictor(version)
    st.markdown(f"**Organism:** {pred.cfg.species}")
    st.markdown(f"**Antibiotics:** {len(pred.served_drugs)}")
    not_served = pred.meta.get("drugs_not_served", {})
    if not_served:
        st.caption("Not in model: " + ", ".join(not_served))
    st.divider()
    st.caption(f"build {pred.meta.get('git_sha')}")

PREVALENCE = read_json(
    DEFAULT_MODEL_DIR / version / "feature_schema.json").get("prevalence", {})

tab_report, tab_tech = st.tabs(["Antibiotic report", "For reviewers"])

# ---------------------------------------------------------------- report ---
with tab_report:
    notice("note", "<b>Decision support only.</b> Every result requires confirmation "
                   "by standard laboratory susceptibility testing before it informs "
                   "treatment.")

    BID = browser_id()

    up = st.file_uploader(
        "Upload an assembled genome (FASTA) or AMRFinderPlus output (TSV)",
        type=["tsv", "txt", "fna", "fasta", "fa"],
        help="File type is detected from the contents, not the name.")

    # A visitor who has never seen a bacterial genome file has nothing to upload,
    # so the example is offered up front rather than hidden in a repository.
    if up is None and "pending" not in st.session_state and EXAMPLE.exists():
        with st.container(border=True):
            ex1, ex2 = st.columns([3, 1])
            ex1.markdown(
                "**Example loaded — Klebsiella pneumoniae DMC0799**  \n"
                "A 5.4 Mb assembly held out of training entirely. The laboratory "
                "found it resistant to meropenem and susceptible to ciprofloxacin, "
                "gentamicin and trimethoprim/sulfamethoxazole.")
            if ex2.button("Analyse example", type="primary", use_container_width=True):
                st.session_state["pending"] = ("example", "")
                st.rerun()

    # History for this browser. Clicking an entry re-opens the stored file.
    past = history(browser_id=BID, limit=12)
    if past:
        with st.expander(f"Previously analysed in this browser ({len(past)})"):
            for h in past:
                hc1, hc2, hc3 = st.columns([4, 3, 1])
                hc1.markdown(f"`{h['sample_id']}`")
                counts = {}
                for v in h.get("verdicts", {}).values():
                    counts[v] = counts.get(v, 0) + 1
                hc2.caption(
                    f"{h['timestamp'][:16].replace('T', ' ')} · model {h['model_version']}"
                    f" · {counts.get('likely_to_fail', 0)} avoid,"
                    f" {counts.get('likely_to_work', 0)} may work,"
                    f" {counts.get('no_call', 0)} no verdict")
                if hc3.button("Open", key=f"h_{h['digest']}", use_container_width=True):
                    st.session_state["pending"] = ("stored", h["digest"])
                    st.rerun()

    # a re-opened or example file behaves exactly like a fresh upload
    pending = st.session_state.pop("pending", None)
    if pending is not None:
        payload, payload_name = read_pending(*pending)
    elif up is not None:
        payload, payload_name = up.getvalue(), up.name
    else:
        payload = payload_name = None

    if payload is not None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / payload_name
            src.write_bytes(payload)
            targets = species_check = assembly_qc = None
            problems: list[str] = []

            # Detected from CONTENT. Trusting a mode toggle once let a FASTA be
            # parsed as a TSV, yielding zero determinants and a confident
            # "likely to work" for a blaKPC-positive genome.
            kind = sniff_file_type(src)
            # keep the raw file before anything can fail on it
            digest = save_upload(payload, payload_name, kind) if kind != "unknown" else None
            if kind == "protein_fasta":
                notice("stop", "This is a protein FASTA. Upload the assembled genome "
                               "(nucleotide sequence).")
                st.stop()
            if kind == "unknown":
                notice("stop", "This file is neither a genome FASTA nor "
                               "AMRFinderPlus output.")
                st.stop()

            if kind == "fasta":
                if not amrfinder_available():
                    notice("stop", "Genome uploaded, but the annotation tool is not "
                                   "installed on this machine. Run <code>make tools</code>.")
                    st.stop()
                assembly_qc = check_assembly(src)
                if not assembly_qc["ok"]:
                    problems.append(assembly_qc["reason"])
                problems.extend(assembly_qc.get("warnings", []))
                cached = already_annotated(digest) if digest else None
                if cached is not None:
                    tsv = cached
                else:
                    with st.spinner("Reading the genome…"):
                        # serialise: two annotations at once exhaust memory
                        with ANNOTATE_LOCK:
                            again = already_annotated(digest) if digest else None
                            if again is not None:
                                tsv = again
                            else:
                                tsv = run_amrfinder(src, Path(td) / "amr.tsv",
                                                    pred.cfg.species_taxgroup,
                                                    threads=2)
                                if digest:
                                    save_annotation(digest, tsv)
                try:
                    targets = detect_targets(src)
                    species_check = verify_species(src)
                    if not species_check.get("ok", True):
                        problems.append(
                            f"Chromosomal target genes match the {pred.cfg.species} "
                            f"reference at {species_check['identity']:.0f}% identity; "
                            f"95% is required.")
                except Exception as e:
                    problems.append(f"Species and target checks did not run ({e}).")
            else:
                tsv = src
                problems.append(
                    "An annotation file was uploaded instead of a genome, so the "
                    "organism and the drug targets were not verified.")

            report = pred.predict_from_tsv(tsv, sample_id=payload_name, targets_found=targets,
                                           species_check=species_check,
                                           assembly_qc=assembly_qc)
            tokens = determinants(parse_amrfinder_tsv(tsv))
            n_det = len(tokens)
            if digest:
                save_report(digest, pred.meta.get("version", version), report.to_dict(), BID)

        avoid = [r for r in report.results if r.call == CALL_FAIL]
        maybe = [r for r in report.results if r.call == CALL_WORK]
        unknown = [r for r in report.results if r.call == CALL_NONE]

        def _conf(r):
            return r.confidence if r.confidence is not None else 0.0

        avoid.sort(key=_conf, reverse=True)
        maybe.sort(key=_conf, reverse=True)

        # ---- dashboard header ----
        st.markdown(f"#### {payload_name}")
        c1, c2, c3, c4 = st.columns(4)

        short = " ".join(pred.cfg.species.split()[:2])
        if species_check is not None:
            org_sub = (f"confirmed · {species_check['identity']:.1f}% identity"
                       if species_check.get("ok")
                       else f"NOT confirmed · {species_check['identity']:.1f}% identity")
        else:
            org_sub = "assumed · not verified from this file"
        kpi(c1, "Organism", short, org_sub)

        if assembly_qc:
            kpi(c2, "Assembly", f"{assembly_qc['total_bp'] / 1e6:.2f} Mb",
                f"{assembly_qc['n_contigs']} contigs · N50 {assembly_qc['n50']:,} bp")
        else:
            kpi(c2, "Assembly", "—", "no genome provided")

        gene_n = len([t for t in tokens if t.startswith(("gene:", "genefam:"))])
        mut_n = len([t for t in tokens if t.startswith(("mut:", "mutgene:", "trunc:"))])
        kpi(c3, "Resistance markers", str(n_det), f"{gene_n} genes · {mut_n} mutations")

        kpi(c4, "Verdicts", f"{len(avoid)} · {len(maybe)} · {len(unknown)}",
            "avoid · may work · none")

        for p in problems:
            notice("warn", p)

        def drug_card(r, css: str, prob: bool = True) -> None:
            pct = (f'<div class="pct">{r.confidence:.0%}</div>'
                   if prob and r.confidence is not None else "")
            if r.call == CALL_NONE:
                why = NO_VERDICT_REASON.get(
                    r.reason, "The evidence does not support a verdict.")
            else:
                why = headline_evidence(as_dicts(r.supporting), r.call)
            ico = {"fail": SVG["ban"], "work": SVG["check"], "none": SVG["dash"]}[css]
            st.markdown(
                f'<div class="drug {css}">{ico}'
                f'<div class="body"><div class="name">{r.display}</div>'
                f'<div class="why">{why.replace("**", "")}</div></div>'
                f'{pct}</div>', unsafe_allow_html=True)

            if r.call == CALL_NONE:
                found = detected_mechanisms(as_dicts(r.supporting))
                if found:
                    notice("warn",
                           "A resistance mechanism was detected for this drug: "
                           + "; ".join(x.replace("**", "") for x in found)
                           + ". The model gave no verdict; this detection does not "
                             "depend on the model.")
            else:
                extra = supporting_sentences(as_dicts(r.supporting), r.call)
                if extra:
                    with st.expander(f"Other findings — {r.display}"):
                        for e in extra:
                            st.markdown(f"- {e}")

        if avoid:
            st.markdown('<div class="grouphead">Avoid — resistance detected</div>',
                        unsafe_allow_html=True)
            for r in avoid:
                drug_card(r, "fail")

        if maybe:
            st.markdown('<div class="grouphead">May work — no resistance marker found</div>',
                        unsafe_allow_html=True)
            for r in maybe:
                drug_card(r, "work")

        if unknown:
            st.markdown('<div class="grouphead">No verdict</div>', unsafe_allow_html=True)
            for r in unknown:
                drug_card(r, "none", prob=False)

        if report.unknown_determinants:
            notice("warn",
                   f"{len(report.unknown_determinants)} resistance markers in this "
                   f"genome are absent from the reference set: "
                   f"<code>{', '.join(report.unknown_determinants[:5])}</code>.")

        stab_path = DEFAULT_MODEL_DIR / version / "eval" / "stability.json"
        cohort_path = DEFAULT_MODEL_DIR / version / "eval" / "demo_cohort.json"
        if stab_path.exists():
            stab = read_json(stab_path)
            s = stab["overall"]
            line = (f"Measured on genomes absent from training: balanced accuracy "
                    f"{s['balanced_accuracy']['mean']:.0%} ± "
                    f"{s['balanced_accuracy']['std']:.1%} over {stab['seeds']} "
                    f"independent evaluations.")
            if cohort_path.exists():
                c = read_json(cohort_path)
                line += (f" On {c['n_genomes']} held-out genomes: {c['n_correct']} of "
                         f"{c['n_called']} verdicts matched the laboratory result.")
            st.caption(line)

        with st.expander("Coverage of this tool"):
            st.markdown(
                f"- Covers {pred.cfg.species} and {len(pred.served_drugs)} antibiotics: "
                f"{', '.join(d.replace('_', '/') for d in pred.served_drugs)}.\n"
                "- Requires an assembled genome. Sample handling, sequencing, species "
                "identification and assembly are outside the tool.\n"
                "- Does not separate mixed samples, select a dose, or account for the "
                "site of infection.\n"
                "- Reports resistance that is already present. It does not design or "
                "modify organisms."
            )

        with st.expander("All detected markers"):
            if tokens:
                det = pd.DataFrame([{
                    "marker": gene_label(t),
                    "effect": describe_feature(t),
                    "carried by": (f"{PREVALENCE[t]:.0%} of reference genomes"
                                   if t in PREVALENCE else "absent from reference set"),
                } for t in sorted(tokens) if not t.startswith("class:")])
                st.dataframe(det, hide_index=True, use_container_width=True)
            st.download_button("Download full report (JSON)",
                               json.dumps(report.to_dict(), indent=2),
                               file_name=f"{payload_name}.genome-firewall.json")

# ------------------------------------------------------------ reviewers ---
with tab_tech:
    ev_path = DEFAULT_MODEL_DIR / version / "eval" / "report.json"
    if not ev_path.exists():
        notice("note", "No evaluation report yet — run <code>make eval</code>.")
    else:
        ev = read_json(ev_path)

        stab_path = DEFAULT_MODEL_DIR / version / "eval" / "stability.json"
        if stab_path.exists():
            stab = read_json(stab_path)
            st.markdown(f"### Repeated over {stab['seeds']} independent grouped splits")
            st.caption("A single split is unreliable here: the measured spread is "
                       "±0.02 AUROC. Every split re-draws the SNP-cluster grouping.")
            band = {
                drug: {m.replace("_", " "): f"{mm[m]['mean']:.3f} ± {mm[m]['std']:.3f}"
                       for m in ("balanced_accuracy", "auroc", "pr_auc", "brier",
                                 "recall_resistant", "specificity", "no_call_rate")
                       if m in mm}
                for drug, mm in stab["per_drug"].items()
            }
            st.dataframe(pd.DataFrame(band).T, use_container_width=True)
            o = stab["overall"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Balanced accuracy", f"{o['balanced_accuracy']['mean']:.3f}",
                      f"± {o['balanced_accuracy']['std']:.3f}", delta_color="off")
            m2.metric("AUROC", f"{o['auroc']['mean']:.3f}",
                      f"± {o['auroc']['std']:.3f}", delta_color="off")
            m3.metric("Brier", f"{o['brier']['mean']:.3f}",
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
        st.caption("A high F1 at recall≈1 is not evidence of a good model — the trivial "
                   "baseline achieves it too. Specificity is the honest signal.")

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
                hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<extra></extra>"))
            bandv = d.get("abstain_band")
            if bandv:
                fig.add_vrect(x0=bandv[0], x1=bandv[1], fillcolor="orange", opacity=0.12,
                              line_width=0, annotation_text="no-verdict band",
                              annotation_position="top left")
            fig.update_layout(xaxis_title="predicted P(resistant)",
                              yaxis_title="observed fraction resistant",
                              xaxis_range=[0, 1], yaxis_range=[0, 1], height=420,
                              margin=dict(l=10, r=10, t=30, b=10),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Marker area is the number of held-out samples in that bin.")

        counts = pred.meta.get("training_counts", {}).get(drug, {})
        if counts.get("nonzero_features") is not None:
            st.markdown(
                f"**Model for {drug}:** L1 logistic regression, `C={counts.get('C')}`, "
                f"**{counts['nonzero_features']} non-zero coefficients** out of "
                f"{len(pred.schema)} features. Selection rule: "
                f"{counts.get('C_selection', {}).get('rule', 'n/a')}. Calibrated with "
                f"Platt scaling, which keeps the model in the logistic family so "
                f"exp(coefficient) is an odds ratio."
            )

        st.markdown("### Scope & safety")
        st.markdown(
            f"- Covers **{pred.cfg.species}** only; other species are refused by a 95% "
            "identity check against chromosomal target genes.\n"
            "- Predicts resistance that **already exists**. It never designs, modifies "
            "or suggests changes to an organism.\n"
            "- Starts from an assembled genome — sample handling, sequencing, species "
            "identification and assembly are out of scope.\n"
            "- Returns **no verdict** on weak, conflicting or out-of-distribution "
            "evidence rather than forcing a yes/no."
        )
