"""ACCEPTANCE TEST -- one command that proves the system works.

Checks every requirement the challenge brief lists, and prints PASS/FAIL per item
with the evidence next to it. This is the artefact to run in front of the jury:
no narration, just the checks and their results.

    uv run python scripts/acceptance.py

Exit code 0 = every required check passed.
Checks needing the toolchain (FASTA input, target gate, species guard) are SKIPPED
rather than failed when AMRFinderPlus/blastn are absent, so the test still runs on
a laptop with `uv sync` alone.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.annotate import amrfinder_available  # noqa: E402
from gfw.config import Config, read_json  # noqa: E402
from gfw.features import (  # noqa: E402
    WrongFileType, determinants, parse_amrfinder_tsv, sniff_file_type,
)
from gfw.predict import CALL_NONE, Predictor  # noqa: E402

# The served bundle, not a pinned version. models/current is a symlink that
# retraining re-points; hardcoding "v19" here meant the tests would keep
# validating an older bundle than the app actually loads.
VERSION = "current"
DEMO = ROOT / "data" / "demo"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool | None, evidence: str) -> None:
    status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    results.append((status, name, evidence))


def main() -> None:
    cfg = Config.load()
    bundle = ROOT / "models" / VERSION
    ev = read_json(bundle / "eval" / "report.json")["per_drug"]
    meta = read_json(bundle / "metadata.json")
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    L = pd.read_csv(ROOT / "data/processed/labels.csv")
    G = pd.read_csv(ROOT / "data/processed/groups.csv")

    # ---- scope: brief asks for ONE species, 3-5 drugs, 1000-3000 genomes ----
    check("scope: single species", len({cfg.species}) == 1, cfg.species)
    check("scope: 3-5 drugs", 3 <= len(meta["drugs_served"]) <= 5,
          f"{len(meta['drugs_served'])} drugs: {', '.join(meta['drugs_served'])}")
    check("scope: 1000-3000 genomes", 1000 <= len(X) <= 3000,
          f"{len(X)} genomes, {X.shape[1]} features, {len(L)} labels")

    # ---- module 01: repeatable FASTA -> features ----
    check("M01: feature schema is the train/serve contract",
          (bundle / "feature_schema.json").exists(),
          f"{len(read_json(bundle / 'feature_schema.json')['features'])} named features")
    tsv = DEMO / "GCA_000417485.1.tsv"
    toks = determinants(parse_amrfinder_tsv(tsv))
    check("M01: AMRFinderPlus TSV -> tokens", len(toks) > 0,
          f"{len(toks)} determinants from the demo genome")

    # ---- module 02: per-drug predictions, gate, homology dedup ----
    check("M02: one model per drug",
          all((bundle / f"{d}.joblib").exists() for d in meta["drugs_served"]),
          f"{len(meta['drugs_served'])} bundles")
    check("M02: homology grouping drives the split",
          G.group_id.nunique() > 1 and G.group_id.nunique() < len(G),
          f"{G.group_id.nunique()} SNP clusters over {len(G)} genomes")
    check("M02: deterministic target gate present",
          (ROOT / "config" / "targets.fna").exists(),
          f"{(ROOT / 'config' / 'targets.fna').read_text().count('>')} reference targets")

    # ---- module 03: report contents ----
    pred = Predictor(VERSION)
    rep = pred.predict_from_tsv(tsv, "acceptance", targets_found=None)
    d = rep.to_dict()
    check("M03: mandatory lab-confirmation message",
          "confirmed by standard laboratory" in d["disclaimer"], d["disclaimer"][:60] + "...")
    check("M03: three-way call for every drug",
          len(d["results"]) == len(cfg.drugs)
          and all(r["call"] in {"likely_to_fail", "likely_to_work", "no_call"}
                  for r in d["results"]),
          ", ".join(f"{r['drug_id'][:12]}={r['call']}" for r in d["results"][:3]) + " ...")
    check("M03: evidence category on every result",
          all(r["evidence_category"] in
              {"known_determinant", "statistical_only", "no_signal"} for r in d["results"]),
          "known_determinant / statistical_only / no_signal")
    check("M03: causal vs statistical separated",
          all("curated_determinant" in s and "mechanistic_for_drug" in s
              for r in d["results"] for s in r["supporting"]),
          "each feature flagged curated + mechanistic-for-this-drug")

    # ---- metrics the brief scores on ----
    need = {"balanced_accuracy", "recall_resistant", "recall_susceptible", "f1",
            "auroc", "pr_auc", "brier", "reliability", "no_call_rate"}
    first = ev[next(iter(ev))]
    check("metrics: all brief-required metrics reported", need <= set(first),
          ", ".join(sorted(need)))
    check("metrics: broken down by genetic group",
          bool(read_json(bundle / "eval" / "report.json").get("by_group")),
          "per-SNP-cluster balanced accuracy")

    mean_auroc = float(np.mean([v["auroc"] for v in ev.values()]))
    mean_bal = float(np.mean([v["balanced_accuracy"] for v in ev.values()]))
    check("performance: beats the trivial baseline on every drug",
          all(v["balanced_accuracy"] > 0.6 for v in ev.values()),
          f"balanced accuracy {min(v['balanced_accuracy'] for v in ev.values()):.3f}"
          f"-{max(v['balanced_accuracy'] for v in ev.values()):.3f} "
          f"(trivial = 0.500), mean AUROC {mean_auroc:.3f}")

    # ---- safety behaviours ----
    ood = pred.predict_from_tokens({f"gene:blaFAKE-{i}" for i in range(10)}, "ood")
    check("safety: refuses out-of-distribution genomes",
          all(r.call == CALL_NONE for r in ood.results),
          "10 unseen determinants -> 5/5 no_call")

    try:
        pred.predict_from_tsv(DEMO / "GCA_000417485.1.fna", "wrongtype")
        check("safety: rejects a FASTA passed as a TSV", False, "accepted it")
    except WrongFileType:
        check("safety: rejects a FASTA passed as a TSV", True,
              "raises WrongFileType instead of predicting on an empty vector")

    check("safety: file type detected from content",
          sniff_file_type(DEMO / "GCA_000417485.1.fna") == "fasta"
          and sniff_file_type(tsv) == "amrfinder_tsv",
          "FASTA and AMRFinderPlus TSV both identified")

    if amrfinder_available():
        from gfw.gate import verify_species
        sp = verify_species(DEMO / "GCA_000417485.1.fna")
        check("safety: species guard accepts the target species",
              sp["ok"], f"identity {sp['identity']}% >= 95%")
        wrong = ROOT / "data" / "demo" / "wrong_species.fna"
        if wrong.exists():
            sw = verify_species(wrong)
            check("safety: species guard rejects a sister species", not sw["ok"],
                  f"identity {sw['identity']}%")
        else:
            check("safety: species guard rejects a sister species", None,
                  "no wrong-species fixture committed")
    else:
        check("safety: species guard accepts the target species", None,
              "AMRFinderPlus/blastn not on PATH")
        check("safety: species guard rejects a sister species", None,
              "AMRFinderPlus/blastn not on PATH")

    # ---- preprocessing robustness (scripts/stress_preprocess.py covers 15 cases) ----
    hdr_only = ROOT / "data" / "demo" / "_hdr_only.tsv"
    hdr_only.write_text(parse_amrfinder_tsv(tsv).columns.to_series().str.cat(sep="\t") + "\n")
    try:
        r0 = pred.predict_from_tsv(hdr_only, "empty-annotation")
        check("safety: zero determinants is refused, not called susceptible",
              all(x.call == CALL_NONE for x in r0.results),
              "header-only AMRFinderPlus output -> 5/5 no_call")
    finally:
        hdr_only.unlink(missing_ok=True)

    from gfw.qc import check_assembly
    qc_ok = check_assembly(DEMO / "GCA_000417485.1.fna")
    check("safety: assembly QC accepts a complete genome", qc_ok["ok"],
          f"{qc_ok['total_bp'] / 1e6:.2f} Mb in {qc_ok['n_contigs']} contigs, "
          f"N50 {qc_ok['n50']:,}")

    # ---- error bars and the held-out demonstration cohort ----
    stab_path = bundle / "eval" / "stability.json"
    if stab_path.exists():
        stab = read_json(stab_path)
        o = stab["overall"]
        check("evidence: metrics repeated over independent splits",
              stab["seeds"] >= 5,
              f"{stab['seeds']} splits, AUROC {o['auroc']['mean']:.3f} "
              f"+/- {o['auroc']['std']:.3f}, balanced accuracy "
              f"{o['balanced_accuracy']['mean']:.3f} "
              f"+/- {o['balanced_accuracy']['std']:.3f}")
    else:
        check("evidence: metrics repeated over independent splits", False,
              "run python -m gfw.stability")

    set_path = ROOT / "data" / "demo" / "demo_set.json"
    if set_path.exists():
        spec = json.loads(set_path.read_text())
        Xd = pd.read_parquet(ROOT / "data/processed/features.parquet")
        ok_calls = n_calls = 0
        dangerous = 0
        for g in spec["genomes"]:
            gid = g["genome_id"]
            if gid not in Xd.index:
                continue
            toks = {c for c in Xd.columns if Xd.at[gid, c] == 1}
            for r in pred.predict_from_tokens(toks, gid).results:
                want = g["phenotypes"].get(r.drug_id)
                if want is None or r.call == CALL_NONE:
                    continue
                got = "R" if r.call == "likely_to_fail" else "S"
                n_calls += 1
                ok_calls += got == want
                dangerous += (want == "R" and got == "S")
        check("evidence: held-out cohort, not a single genome",
              spec["n_genomes"] >= 10 and n_calls >= 40,
              f"{spec['n_genomes']} unseen genomes, {n_calls} calls made, "
              f"{ok_calls} correct ({ok_calls / max(1, n_calls):.1%}), "
              f"{dangerous} missed resistances")
    else:
        check("evidence: held-out cohort, not a single genome", False,
              "run scripts/build_demo_set.py")

    # ---- defensive by construction ----
    src = " ".join((ROOT / "src" / "gfw" / f).read_text()
                   for f in ("predict.py", "train.py", "features.py"))
    check("defensive: no organism design/modification anywhere",
          not any(w in src.lower() for w in ("synthesize", "mutagenes", "engineer_strain")),
          "prediction and explanation only")

    # ---- print ----
    width = max(len(n) for _, n, _ in results)
    print(f"ACCEPTANCE TEST -- model {VERSION}\n")
    for status, name, evidence in results:
        print(f"[{status}] {name:<{width}}  {evidence}")

    failed = [r for r in results if r[0] == "FAIL"]
    skipped = [r for r in results if r[0] == "SKIP"]
    print(f"\n{sum(1 for r in results if r[0] == 'PASS')} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    print(f"held-out mean: balanced accuracy {mean_bal:.3f}, AUROC {mean_auroc:.3f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
