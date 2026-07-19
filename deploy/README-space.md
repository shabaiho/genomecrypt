---
title: Genome Firewall
emoji: 🧬
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Antibiotic-response prediction from a Klebsiella pneumoniae genome
---

# Genome Firewall

Predicts, from one assembled *Klebsiella pneumoniae* genome, whether each of five
antibiotics is already defeated by resistance genes the organism carries.

**Research prototype. Not validated for clinical use. Every result requires
confirmation by standard laboratory susceptibility testing.**

Upload the example genome on arrival, or your own assembled FASTA. The file type
is read from the contents, not the name.

Four checks run before any verdict:

| Check | Refuses |
|---|---|
| File type | anything that is not a genome or AMRFinderPlus output |
| Assembly completeness | below 4.9 Mb — missing sequence reads as an absent gene |
| Species | below 95% identity to *K. pneumoniae* chromosomal targets |
| Familiarity | over 30% of determinants absent from the training set |

Held out on genomes from lineages absent from training, over 8 independent splits:
AUROC 0.893 ± 0.019, balanced accuracy 0.789 ± 0.014.

Source, methods and the full evaluation: https://github.com/shabaiho/genomecrypt
