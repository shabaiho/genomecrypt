"""Module 02b -- deterministic target gate.

The brief's requirement: never say "likely to work" purely because no resistance
marker was found. If the drug's molecular target is not in the assembly, the
model's opinion is not meaningful -> no_call.

The gate runs on a core-gene scan of the assembly, NOT on AMRFinderPlus output
(which reports resistance determinants, not drug targets). Cheapest honest
implementation for a hackathon: a small nucleotide BLAST/`makeblastdb` screen
against a curated FASTA of target genes for the species.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import Drug

TARGET_DB_FASTA = Path(__file__).resolve().parents[2] / "config" / "targets.fna"
MIN_IDENT = 80.0
MIN_COV = 70.0

# SPECIES GUARD. Identity of the chromosomal target genes to our K. pneumoniae
# reference alleles, MEASURED:
#     K. pneumoniae  99.8%   <- the species this model is for
#     K. oxytoca     89.0%   <- sister species, 7/7 targets still found
#     E. cloacae     86.9%
#     E. coli        85.4%
# Without this check the tool happily served K. oxytoca: all 7 targets passed the
# gate, only 29% of determinants looked novel (just under the 30% OOD trigger),
# and it returned "gentamicin likely to WORK 66%" and "trim/sulfa likely to WORK
# 84%". Confident, clinically actionable, and about the wrong organism.
# The brief puts species identification out of scope -- which means the tool must
# REFUSE anything it cannot confirm, not quietly assume.
SPECIES_MIN_IDENTITY = 95.0


class GateResult:
    def __init__(self, ok: bool, reason: str, detail: dict):
        self.ok, self.reason, self.detail = ok, reason, detail


def detect_targets(fasta: Path) -> set[str]:
    """Return the set of target gene names found in the assembly.

    Requires blastn (present in the `full` image). If unavailable, callers should
    fall back to `assume_present=True` and SAY SO in the report -- silently
    skipping the gate would be exactly the false confidence the brief warns about.
    """
    if shutil.which("blastn") is None or not TARGET_DB_FASTA.exists():
        raise RuntimeError("blastn or config/targets.fna missing -- cannot run target gate")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "hits.tsv"
        # qcovs = query coverage summed over all HSPs for that subject. Using a
        # single HSP's length/qlen instead silently misses divergent-but-present
        # genes whose alignment fragments: folA and folP -- essential genes every
        # K. pneumoniae carries -- were reported absent, which would have turned
        # every trimethoprim/sulfamethoxazole call into a spurious no-call.
        subprocess.run(
            ["blastn", "-query", str(TARGET_DB_FASTA), "-subject", str(fasta),
             "-outfmt", "6 qseqid pident qcovs", "-max_target_seqs", "5"],
            stdout=out.open("w"), stderr=subprocess.DEVNULL, check=True, timeout=300,
        )
        found = set()
        for line in out.read_text().splitlines():
            qid, pident, qcovs = line.split("\t")
            if float(pident) >= MIN_IDENT and float(qcovs) >= MIN_COV:
                found.add(qid.split("|")[0])  # header format: >gyrA|K.pneumoniae|...
        return found


def verify_species(fasta: Path) -> dict:
    """Mean identity of chromosomal target genes to the reference species.

    Returns {"ok": bool, "identity": float, "n_targets": int}. Requires blastn and
    config/targets.fna; raises otherwise, so callers can distinguish "wrong
    species" from "could not check".
    """
    if shutil.which("blastn") is None or not TARGET_DB_FASTA.exists():
        raise RuntimeError("blastn or config/targets.fna missing -- cannot verify species")

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sp.tsv"
        subprocess.run(
            ["blastn", "-query", str(TARGET_DB_FASTA), "-subject", str(fasta),
             "-outfmt", "6 qseqid pident qcovs", "-max_target_seqs", "5"],
            stdout=out.open("w"), stderr=subprocess.DEVNULL, check=True, timeout=300,
        )
        best: dict[str, float] = {}
        for line in out.read_text().splitlines():
            qid, pident, qcovs = line.split("\t")
            if float(qcovs) >= MIN_COV:
                gene = qid.split("|")[0]
                best[gene] = max(best.get(gene, 0.0), float(pident))
    if not best:
        return {"ok": False, "identity": 0.0, "n_targets": 0}
    mean_id = sum(best.values()) / len(best)
    return {"ok": mean_id >= SPECIES_MIN_IDENTITY,
            "identity": round(mean_id, 2), "n_targets": len(best)}


def apply_gate(drug: Drug, targets_found: set[str] | None) -> GateResult:
    if drug.intrinsic_resistance:
        return GateResult(False, "intrinsic", {"note": f"{drug.klass}: species is intrinsically resistant"})
    if targets_found is None:
        # gate could not run -- allow the model through but flag it loudly
        return GateResult(True, "gate_skipped", {"note": "target screen unavailable"})
    missing = [g for g in drug.target_genes if g not in targets_found]
    if missing:
        return GateResult(False, "target_absent", {"missing": missing})
    return GateResult(True, "target_present", {"found": drug.target_genes})
