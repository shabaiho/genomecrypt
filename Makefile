VERSION ?= v1
LIMIT   ?= 300
THREADS ?= 2
JOBS    ?= 4
UV      ?= uv

# Every python command goes through `uv run`, so the locked environment in
# uv.lock is the only environment that ever executes. No venv to activate.
RUN     = $(UV) run
RUN_TRAIN = $(UV) run --extra train

.PHONY: setup tools fast-data data prepare train eval promote app test demo all fast clean

# FAST PATH (minutes): pre-decoded AMRFinderPlus genotypes + lab AST + SNP
# clusters straight from NCBI Pathogen Detection. No annotation run.
fast: fast-data train eval promote

# SLOW PATH (hours): our own assemblies, our own AMRFinderPlus run.
all: data prepare train eval promote

setup:           ## resolve + install python deps from uv.lock
	$(UV) sync --extra train

tools:           ## install AMRFinderPlus + BLAST+ + HMMER into ./.tools (no root)
	./scripts/install_tools.sh

fast-data:       ## NCBI Pathogen Detection -> features + labels + groups, NO annotation
	$(RUN) python -m gfw.ncbi_dataset --organism Klebsiella

demo:            ## predict on a genome never used in training, vs the lab result
	$(RUN) python scripts/demo_case.py

demo-annotate:   ## re-run AMRFinderPlus on the demo genome (needs make tools)
	. .tools/env.sh && $(RUN_TRAIN) python -c "from pathlib import Path; \
	  from gfw.annotate import run_amrfinder; \
	  run_amrfinder(Path('data/demo/GCA_000417485.1.fna'), \
	                Path('data/demo/GCA_000417485.1.tsv'), 'Klebsiella_pneumoniae', 6)"

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

promote:         ## point models/current at $(VERSION) -- what the app loads
	rm -f models/current && ln -s $(VERSION) models/current

app:             ## run the demo UI
	. .tools/env.sh 2>/dev/null; $(RUN) streamlit run app/streamlit_app.py

test:            ## end-to-end check on synthetic data, no network, no amrfinder
	$(RUN) python tests/smoke_test.py

clean:
	rm -rf data/interim data/processed
