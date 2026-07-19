# Bundle to write when training. Deliberately not a real version number: the
# default used to be v1, so `make train` silently overwrote the oldest bundle in
# the repository. Pass VERSION=v21 to name a keeper.
VERSION ?= dev
LIMIT   ?= 300
THREADS ?= 2
JOBS    ?= 4
SEEDS   ?= 8
UV      ?= uv

# Every python command goes through `uv run`, so the locked environment in
# uv.lock is the only environment that ever executes. No venv to activate.
RUN     = $(UV) run
RUN_TRAIN = $(UV) run --extra train

.PHONY: help setup tools fast-data data prepare train eval stability demo-set \
        promote app test units acceptance stress demo demo-annotate all fast \
        check serve clean

help:            ## list the targets
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk -F':.*?## ' '{printf "  %-14s %s\n", $$1, $$2}'

# FAST PATH (minutes): pre-decoded AMRFinderPlus genotypes + lab AST + SNP
# clusters straight from NCBI Pathogen Detection. No annotation run.
# Includes stability and the demonstration cohort: without them the model card
# has no error bars and the report has no track record to quote.
fast: fast-data train eval stability demo-set promote

# SLOW PATH (hours): our own assemblies, our own AMRFinderPlus run.
all: data prepare train eval stability demo-set promote

setup:           ## resolve + install python deps from uv.lock
	$(UV) sync --extra train

tools:           ## install AMRFinderPlus + BLAST+ + HMMER into ./.tools (no root)
	./scripts/install_tools.sh

fast-data:       ## NCBI + BV-BRC -> features + labels + groups, NO annotation
	$(RUN) python -m gfw.merge_sources --organism Klebsiella

data:            ## BV-BRC API -> data/raw/fasta + data/processed/labels.csv
	$(RUN) python -m gfw.download_data --limit $(LIMIT)

prepare:         ## FASTA -> AMRFinderPlus TSV -> features.parquet + groups.csv
	. .tools/env.sh && $(RUN_TRAIN) python -m gfw.prepare --threads $(THREADS) --jobs $(JOBS)

train:           ## fit + calibrate -> models/$(VERSION)/
	$(RUN) python -m gfw.train --version $(VERSION) \
	  --matrix data/processed/features.parquet \
	  --labels data/processed/labels.csv \
	  --groups data/processed/groups.csv

eval:            ## held-out metrics -> models/$(VERSION)/eval/report.json
	$(RUN) python -m gfw.evaluate --version $(VERSION) \
	  --matrix data/processed/features.parquet \
	  --labels data/processed/labels.csv \
	  --groups data/processed/groups.csv

stability:       ## repeat over $(SEEDS) splits -> eval/stability.json (error bars)
	$(RUN) python -m gfw.stability --version $(VERSION) --seeds $(SEEDS)

demo-set:        ## score the held-out cohort -> eval/demo_cohort.json
	$(RUN) python scripts/demo_set_eval.py

demo-set-build:  ## re-select the held-out cohort (run BEFORE training)
	$(RUN) python scripts/build_demo_set.py --n 15

promote:         ## point models/current at $(VERSION) -- what the app loads
	rm -f models/current && ln -s $(VERSION) models/current

targets:         ## rebuild config/targets.fna for the species gate
	$(RUN) python scripts/build_targets.py

app:             ## run the demo UI
	. .tools/env.sh 2>/dev/null; $(RUN) streamlit run app/streamlit_app.py

serve:           ## landing on 8600 and the supervised demo on 8501
	./scripts/serve.sh

demo:            ## predict on the single demo genome, vs the lab result
	$(RUN) python scripts/demo_case.py

demo-annotate:   ## re-run AMRFinderPlus on the demo genome (needs make tools)
	. .tools/env.sh && $(RUN_TRAIN) python -c "from pathlib import Path; \
	  from gfw.annotate import run_amrfinder; \
	  run_amrfinder(Path('data/demo/GCA_000417485.1.fna'), \
	                Path('data/demo/GCA_000417485.1.tsv'), 'Klebsiella_pneumoniae', 6)"

units:           ## unit tests over the pure functions
	$(RUN) pytest -q

acceptance:      ## 25 checks against the challenge brief
	$(RUN) python scripts/acceptance.py

stress:          ## malformed inputs and degraded assemblies
	$(RUN) python scripts/stress_preprocess.py

test:            ## end-to-end on synthetic data, no network, no amrfinder
	$(RUN) python tests/smoke_test.py

check: units test acceptance stress   ## everything that can run without a GPU

clean:
	rm -rf data/interim data/processed
