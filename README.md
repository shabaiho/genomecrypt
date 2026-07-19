# Genome Firewall

Assembled bacterial genome → per-antibiotic prediction (**likely to fail / likely to
work / no-call**) with a calibrated confidence score and the evidence behind it.

> **Research prototype — not for clinical use.** Every result must be confirmed by
> standard laboratory antimicrobial susceptibility testing. Strictly defensive: the
> system predicts and explains resistance that already exists. It never designs,
> modifies, or suggests changes to an organism.

## Train / serve separation

Training and serving share nothing but a folder:

```
models/
  v1/
    metadata.json         species, drugs served, drugs NOT served + why, git sha
    feature_schema.json    ordered feature names — the entire train↔serve contract
    ciprofloxacin.joblib   calibrated model + coefficients for explanations
    gentamicin.joblib
    ...
    splits.json            held-out genome ids, so eval scores the same rows
    eval/report.json       held-out metrics, rendered in the app's Model card tab
  current -> v1            symlink the app loads by default
```

`gfw.predict` imports no training code, no sourmash, no AMRFinderPlus. Retraining
means writing a new `models/<version>/` and re-pointing `current` — no image rebuild,
no code change. The app's bundle picker lets you A/B two versions side by side.

## Run it

Everything is driven by **uv** against a committed `uv.lock` — no Docker, no conda,
no venv to activate. Same locked environment on every machine.

```bash
uv sync --extra train        # python deps (or: make setup)
make test                    # end-to-end check on synthetic data — no network needed

# demo UI — needs only a model bundle in ./models
make app                     # -> http://localhost:8501
```

The three tools uv *cannot* install (AMRFinderPlus, BLAST+, HMMER) are native
binaries. One script puts them in `./.tools`, no root required:

```bash
make tools                   # ~350MB, ~5 min, then: source .tools/env.sh
make all                     # download -> annotate -> train -> evaluate -> promote
```

The UI accepts a **precomputed AMRFinderPlus TSV**, so a teammate or judge can run
the demo with `uv sync && make app` alone — no bioinformatics toolchain at all.
FASTA upload and the target gate need `make tools`.

### AMRFinderPlus version pitfall (already handled)

NCBI ships prebuilt binaries only. **v4.x needs GLIBCXX_3.4.32 (gcc 13+, Ubuntu
24.04+)**; on Ubuntu 22.04 it downloads fine and then dies at first invocation.
`install_tools.sh` tries newest-first, *verifies the binary actually runs*, and falls
back to v3.12.8. The two versions rename half the output columns and tag point
mutations differently — `features.py` normalizes both, verified to produce identical
token sets on the same genome.

**Pin one AMR database version for the whole project.** v3.12.8 gets DB 2024-07-22.1,
v4.2.7 gets 2026-05-15.1, and they disagree on real calls (on our test genome:
`pmrB_R256G` vs `ompK36_K231SfsTer16`). Training on one DB and serving on another
silently shifts the feature space.

## Pipeline

| Step | Module | In → Out |
|---|---|---|
| 1. Data | `gfw.download_data` | BV-BRC Data API → `labels.csv`, `fasta/` (~5MB per genome) |
| 2. Annotate | `gfw.annotate` | FASTA → AMRFinderPlus TSV (resumable, parallel) |
| 3. Features | `gfw.features` | TSV → binary matrix: `gene:` `mut:` `class:` + `genefam:` `mutgene:` `trunc:` rollups |
| 4. Dedup | `gfw.dedup` | sourmash MinHash → homology clusters → group ids |
| 5. Train | `gfw.train` | grouped 4-way split → L1 LogReg (C by AUROC) + isotonic → bundle |
| 6. Evaluate | `gfw.evaluate` | held-out → balanced acc, F1, AUROC, PR-AUC, Brier, reliability, by-group |
| 7. Serve | `gfw.predict` + `app/` | bundle + TSV → decision report JSON + Streamlit UI |

## The four things that decide the score

**Grouped split.** Outbreak clones in both train and test turn evaluation into a memory
test. We sketch every genome with sourmash (k=31, scaled=1000), single-linkage cluster at
Jaccard ≥ 0.90 (≈ 99.5% ANI), and split by cluster — train, calibration, and test never
share a group. Report metrics at 2–3 thresholds; a memorizing model falls off a cliff
as the threshold tightens, an honest one degrades gently.

**Honest calibration.** The isotonic calibrator is fit on a split disjoint from training.
Fitting it on in-sample scores produces a beautiful reliability plot that means nothing.

**No-call, three ways.** `low_confidence` (calibrated p in the abstain band),
`ood` (>30% of detected determinants unseen in training), `target_absent` (the
deterministic gate). Report no-call rate *and* accuracy on the calls actually made —
that pair is the honest summary, not headline accuracy.

**Evidence, separated.** Every result carries one of: known curated determinant detected /
statistical association only / no known signal. A model coefficient is never presented as
a biological cause; the UI flags non-curated features explicitly.

## Species guard (safety-critical)

The brief puts species identification out of scope. That means the tool must
REFUSE what it cannot confirm -- not quietly assume. Before this check existed it
served a *K. oxytoca* genome with all 7 targets found, 29% novel determinants
(just under the 30% OOD trigger) and returned **"gentamicin likely to WORK 66%"**
and **"trim/sulfa likely to WORK 84%"** -- confident, actionable, wrong organism.

`gfw.gate.verify_species` reuses the blastn run already done for the target gate
and takes mean identity of the chromosomal targets to the reference alleles:

| genome | mean target identity | verdict |
|---|---|---|
| *K. pneumoniae* | 99.8% | served |
| *K. oxytoca* | 89.0% | refused |
| *E. cloacae* | 86.9% | refused |
| *E. coli* | 85.4% | refused |

Threshold 95%. Below it, every drug returns `no_call` with reason `wrong_species`.
Note this needs the assembly -- **TSV-only input cannot be species-verified**, and
the app says so.

## Target gate

Absence of resistance markers alone must not produce "likely to work". Before the model
runs, `gfw.gate` blastn-screens the assembly for the drug's molecular target
(`gyrA`/`parC` for fluoroquinolones, `ftsI` for β-lactams, …). Target missing → `no_call`.
Species intrinsically resistant → `likely_to_fail` without consulting the model.

`config/targets.fna` is built by `uv run python scripts/build_targets.py` from the
K. pneumoniae HS11286 reference. Verified working: all 7 targets detected on the demo
genome, every drug passes the gate.

**Two traps that made the gate silently wrong, both fixed in the builder:**

1. **Plasmid genes are not drug targets.** HS11286 carries plasmids, and a plasmid
   "dihydrofolate reductase" is `dfrA` while a plasmid "dihydropteroate synthase" is
   `sul` — *acquired resistance genes*. Extracting those as folA/folP inverts the
   gate: it reports "target absent" for exactly the susceptible genomes. The builder
   now restricts to the chromosome `NC_016845.1`.
2. **Protein descriptions are substrings of each other.** "30S ribosomal protein S12"
   also matches "…S12 methylthiotransferase RimO" (1761 bp instead of 375 bp), and
   "penicillin-binding protein 2" matches a longer PBP2 fusion description. The
   builder requires an exact description match first.

`gfw.gate` also scores coverage with blastn's `qcovs` (summed over HSPs), not a single
HSP's `length/qlen` — the latter reported folA and folP absent because their alignments
fragment, which would have turned every trim/sulfa call into a spurious no-call.

## Fast path: pre-decoded training data (`make fast`)

Annotation costs hours. NCBI Pathogen Detection publishes it already done — for
every isolate it has processed it ships `AMR_genotypes` (AMRFinderPlus element
symbols), `AST_phenotypes` (the lab antibiogram) and `PDS_acc` (SNP cluster id).
That is **features + labels + grouping from two downloads**, and the feature space
is identical to what our local AMRFinderPlus produces, so the same model serves
FASTA uploads unchanged.

```bash
make fast     # ~3 min: download -> train -> evaluate -> promote
```

Verified on Klebsiella snapshot `PDG000000012.2470`:

| | |
|---|---|
| isolates in snapshot | 167,247 |
| with phenotype **and** genotype | 2,612 |
| usable for our 5-drug panel | **1,992 × 313 features** |
| SNP clusters (grouping) | 1,376, largest 35 |
| labels per drug | 1,285–1,599 |
| class balance | **0.46–0.58 resistant** |

That balance is the real win: BV-BRC's ceftriaxone labels are 83% resistant, where
the trivial "always resistant" model already scores F1 0.906. Here the trivial
baseline is much weaker, so the metrics mean something.

The slow path (`make all`, BV-BRC + local annotation) is still there and is what you
use if the organizers hand out their own dataset.

## Demo genome (`make demo`)

`data/demo/GCA_000417485.1.fna` — *K. pneumoniae* DMC0799, held out properly: its
SNP cluster appears in **no** drug's train or calibration split, so it is an unseen
lineage, not merely an unseen sample. Lab result: resistant to meropenem, susceptible
to ciprofloxacin, gentamicin and trimethoprim/sulfamethoxazole.

Same model, same genome, two operating points:

| drug | P(R) | recall≥0.99 | threshold 0.5 | lab |
|---|---|---|---|---|
| meropenem | 0.750 | R ✓ | R ✓ | R |
| ciprofloxacin | 0.360 | R ✗ | S ✓ | S |
| gentamicin | 0.588 | R ✗ | R ✗ | S |
| trim/sulfa | 0.282 | R ✗ | S ✓ | S |
| | | **1/4** | **3/4** | |

The meropenem call is right for the right reason — `blaKPC-2` (weight +1.675), a
carbapenemase, is the top contributing feature and a curated determinant.

## Model (`models/v8`)

Per-drug **L1 logistic regression**, `class_weight='balanced'`, isotonic calibration,
`C` chosen by AUROC on the calibration split under a half-standard-error rule
(sparsest model within noise of the best). Held out on unseen SNP clusters:

| drug | bal.acc | AUROC | PR-AUC | Brier | no-call | non-zero features |
|---|---|---|---|---|---|---|
| gentamicin | 0.770 | 0.794 | 0.641 | 0.192 | 27% | 103 |
| ceftriaxone | 0.762 | 0.869 | 0.872 | 0.150 | 19% | 47 |
| trim/sulfa | 0.702 | 0.768 | 0.699 | 0.204 | 16% | 37 |
| ciprofloxacin | 0.685 | 0.722 | 0.658 | 0.224 | 0% | 27 |
| meropenem | 0.674 | 0.726 | 0.659 | 0.236 | 13% | 17 |

Compared with the first real model (L2, no feature aggregation, recall-constrained):
gentamicin 0.583 → 0.770, meropenem 0.508 → 0.674, ciprofloxacin 0.610 → 0.685.

### Feature aggregation is where most of the gain came from

Raw AMRFinderPlus symbols are too fine-grained to learn on:

- **Point mutations.** 294 distinct ones in the snapshot, **73% seen fewer than 3
  times**, so `min_prevalence` deletes nearly all of them. But `ompK35` is hit 746
  times across 75 variants. Adding `mutgene:<gene>` and `trunc:<gene>` (frameshift /
  truncation) rollups recovers the porin-loss signal — the main non-carbapenemase
  route to carbapenem resistance. `trunc:ompK35` is now a live feature in the
  meropenem model.
- **Gene alleles.** `blaKPC-2` and `blaKPC-3` are the same carbapenemase clinically
  but were two features. L1 kept `blaKPC-3`, zeroed `blaKPC-2`, and the demo genome
  — which carries `blaKPC-2` — was called **"likely to work" despite being
  carbapenemase-positive**, a false-susceptible. `genefam:<family>` rollups fixed it;
  `genefam:blaKPC` is now the top meropenem feature and the demo genome is called
  correctly.

### Model selection: two traps we walked into

**Selecting on balanced accuracy over-fit the calibration split.** It picked `C=3`
for gentamicin: 159 non-zero coefficients topped by `mcr-1.1` (a *colistin* gene,
weight −5.98) and `qnrB6` (a *quinolone* gene) — textbook spurious correlation.
Switching the selection metric to AUROC (threshold-free, far less jumpy on ~120 rows)
fixed it.

**The full one-standard-error rule over-regularized.** It stripped meropenem to 4
features and dropped `blaKPC` entirely, trading a causal determinant for sparsity.
Half-SE keeps the biology.

### The split has four blocks, not three

`train / calib / thresh / test`, all grouped by SNP cluster. The threshold and the
no-call band are fitted on `thresh`, **not** on `calib` where the isotonic calibrator
was fit. Selecting an operating point against in-sample calibrator output reported
`calib_recall = 1.000` and then delivered 0.971 held-out; with a separate block the
number is honest.

### The no-call band is fitted, not hardcoded

A fixed 0.40–0.60 band assumes the useful uncertainty always sits around 0.5. Fitted
per drug on `thresh` (widen until accuracy-on-called reaches 90%, stop at 30% no-call
rate), the bands come out completely different: ceftriaxone needs only 0.49–0.51 to
reach 92% accuracy, ciprofloxacin needs 0.25–0.75 and still only reaches 79%.

## Alternative policy: high sensitivity (`decision.mode: high_sensitivity`)

Clinical asymmetry drives the operating point: a missed resistant isolate means the
patient gets a drug that will fail, so recall on the RESISTANT class is a hard
constraint and the abstention band is switched off.

**Recall = 1.0 cannot be promised, and chasing it exactly is a trap.** Predicting
"resistant" for everything scores recall 1.000 by construction. On the real BV-BRC
label balance that trivial model already scores:

| drug | prevalence | trivial recall | trivial F1 | trivial bal.acc |
|---|---|---|---|---|
| ceftriaxone | 0.828 | 1.000 | **0.906** | 0.500 |
| ciprofloxacin | 0.759 | 1.000 | 0.863 | 0.500 |
| trim/sulfa | 0.689 | 1.000 | 0.816 | 0.500 |
| gentamicin | 0.426 | 1.000 | 0.598 | 0.500 |
| meropenem | 0.323 | 1.000 | 0.488 | 0.500 |

So "recall 1.0 with F1 0.90" is exactly what a model that learned nothing produces,
and judges scoring balanced accuracy and PR-AUC will see it immediately. The policy
implemented instead is:

> **maximize specificity subject to recall_resistant ≥ `target_recall`**

Specificity is the number that separates a real model from the trivial one — the
trivial model scores 0.0 there by definition. Every report prints model vs. trivial
side by side, plus `missed_resistant` as an absolute count.

**Measured cost of the constraint — real data, held out on unseen SNP clusters**
(`models/v1`, 1,992 isolates):

| drug | thr | recall(R) | missed | specificity | F1 | trivial F1 | AUROC | bal.acc |
|---|---|---|---|---|---|---|---|---|
| ceftriaxone | 0.068 | 0.971 | 5 | 0.350 | 0.773 | 0.706 | 0.857 | 0.660 |
| ciprofloxacin | 0.087 | 0.992 | 1 | 0.137 | 0.683 | 0.654 | 0.802 | 0.565 |
| gentamicin | 0.125 | 1.000 | 0 | 0.122 | 0.596 | 0.564 | 0.801 | 0.561 |
| trim/sulfa | 0.118 | 0.992 | 1 | 0.076 | 0.687 | 0.674 | 0.755 | 0.534 |
| meropenem | 0.188 | 0.993 | 1 | 0.006 | 0.636 | **0.638** | 0.674 | 0.500 |

Read the last row carefully: at recall ≥ 0.99 meropenem scores **below** the trivial
baseline, with balanced accuracy 0.500. The models are not weak — AUROC is 0.674–0.857,
there is real signal. The recall constraint is what destroys them.

The same models at the default threshold:

| drug | bal.acc @ recall≥0.99 | bal.acc @ 0.5 | specificity @ recall≥0.99 | @ 0.5 |
|---|---|---|---|---|
| ceftriaxone | 0.660 | **0.786** | 0.350 | 0.699 |
| ciprofloxacin | 0.565 | **0.732** | 0.137 | 0.771 |
| gentamicin | 0.561 | **0.761** | 0.122 | 0.755 |
| meropenem | 0.500 | **0.671** | 0.006 | 0.591 |
| trim/sulfa | 0.534 | **0.695** | 0.076 | 0.714 |

**Pick your operating point from this sweep** (held-out, averaged over the 5 drugs):

| `target_recall` | actual recall | specificity | bal.acc | F1 | resistant isolates missed |
|---|---|---|---|---|---|
| 0.80 | 0.759 | 0.664 | **0.712** | 0.715 | 156 |
| 0.85 | 0.808 | 0.569 | 0.688 | 0.705 | 129 |
| 0.90 | 0.839 | 0.545 | 0.692 | 0.716 | 110 |
| 0.95 | 0.907 | 0.386 | 0.647 | 0.705 | **57** |
| 0.99 | 0.990 | 0.138 | 0.564 | 0.675 | **8** |

0.95 is the knee: it still catches 91% of resistant isolates and keeps specificity
0.386, while 0.99 buys 49 fewer misses at the price of nearly threefold worse
specificity. Change one line in `config/drugs.yaml` to move.

Three things worth saying out loud in the writeup:

1. **Recall 1.0 on calibration does not transfer.** Every drug hit 1.000 on the
   calibration split; held-out they landed at 0.971–1.000. A guarantee is not
   available at any threshold short of the trivial model.
2. **Where recall held, specificity collapsed** — meropenem at 0.006 is the trivial
   baseline wearing a model's clothes.
3. **It shows up in the live demo.** On the unseen demo genome the constrained model
   calls all five drugs "likely to fail" and gets 1/4; the same model at 0.5 gets 3/4.

Training prints a `degenerate` warning whenever the constraint has forced a drug into
calling everything resistant — it fired for ciprofloxacin and meropenem on this run.

**Scoring risk, stated plainly:** the brief lists "calibrated confidence and a
no-call option" as a scored requirement, and names "force every sample into a yes/no
answer and hide uncertainty" as a weak-submission marker. `high_sensitivity` drops
the abstention band, so it trades points on that criterion. Mitigations kept in
place: the OOD no-call still fires (`keep_ood_no_call: true`), calibration and
reliability plots are still produced, and `decision.mode: calibrated_abstain`
restores the brief's default without touching code.

## Budget your time: annotation dominates

**Measured on an 8-core box: ~1 genome/minute wall-clock** at `--jobs 4 --threads 2`
(v3.12.8). Annotation is ~95% of total pipeline runtime; everything else is minutes.

| Genomes | Annotation time | Disk (FASTA) |
|---|---|---|
| 150 | ~2.5 h | ~0.8 GB |
| 300 | ~5 h | ~1.6 GB |
| 800 | ~13 h | ~4.3 GB |

`LIMIT` defaults to **300** for this reason — start it early and run it in the
background while you build the UI. The step is resumable, so Ctrl-C is safe. Do not
plan on 2,000 genomes unless you have a machine that can run it overnight.

## Label policy

- Lab-measured phenotypes only (`laboratory_typing_method` in a broth/agar/disk/MIC
  allowlist). BV-BRC's general phenotype fields can contain model-generated predictions;
  training on those launders another model's errors into ours.
- `Intermediate` is dropped, not folded into either class — it is a genuinely different
  clinical category and forcing it in corrupts both.
- One final label per (genome, drug); pairs where sources disagree are dropped.

## Tunables worth defending in the writeup

| Knob | Where | Default | Why it matters |
|---|---|---|---|
| Jaccard threshold | `dedup.py` | 0.90 | The single biggest lever on reported score |
| Abstain band | `config/drugs.yaml` | 0.35 / 0.65 | Trades no-call rate against accuracy-on-called |
| `min_train_support` | `config/drugs.yaml` | 25 | Below this a drug is not served at all, and the app says so |
| OOD fraction | `config/drugs.yaml` | 0.30 | Novel determinants → refuse rather than extrapolate |
| Feature prevalence | `features.py` | 3 | Singleton determinants add variance, not signal |

## Stretch (only after the baseline is calibrated and evaluated)

- Genomic LM embeddings (HyenaDNA / DNABERT-2) over annotated regions, concatenated to
  the binary features — compare against baseline on the same grouped split, and report
  it honestly if it does not help.
- LLM-generated plain-language rationale over the *structured* report — never letting the
  model invent evidence, only phrasing what the pipeline already produced.
