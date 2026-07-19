"""Assembly QC -- refuse genomes too incomplete to answer from.

WHY THIS EXISTS (measured, scripts/stress_assembly.py). Degrading the demo genome
by deleting contigs made an aminoglycoside determinant disappear, and the report
flipped from "gentamicin likely to FAIL" to "gentamicin likely to WORK" -- an
ineffective drug recommended because the evidence for resistance was simply
missing from the file. Absence of a gene and absence of the sequence containing it
are indistinguishable downstream, so they have to be separated HERE.

    full assembly     5.43 Mb   no flips
    10% contigs lost  5.13 Mb   no flips
    25% contigs lost  4.67 Mb   DANGEROUS flip
    50% contigs lost  3.40 Mb   DANGEROUS flip

K. pneumoniae genomes run 5.0-5.9 Mb, so 4.9 Mb separates the safe cases from the
dangerous ones with room to spare.

Fragmentation alone is far less harmful: chopping the same genome into 5 kb pieces
(1153 contigs, worse than a typical draft) changed nothing at all, and at 2 kb the
calls degraded into no-calls rather than into wrong answers. So contig COUNT is a
warning; total LENGTH is a refusal.
"""
from __future__ import annotations

from pathlib import Path

# Klebsiella pneumoniae. Move these to config/drugs.yaml when a second species
# is added -- they are species constants, not global truths.
MIN_ASSEMBLY_BP = 4_900_000
MAX_ASSEMBLY_BP = 6_500_000
WARN_CONTIGS = 1_000


def assembly_stats(fasta: Path) -> dict:
    """Total length, contig count and N50 in one pass."""
    lengths: list[int] = []
    cur = 0
    with Path(fasta).open("r", errors="replace") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    lengths.append(cur)
                cur = 0
            else:
                cur += len(line.strip())
    if cur:
        lengths.append(cur)

    total = sum(lengths)
    n50 = 0
    if total:
        acc = 0
        for length in sorted(lengths, reverse=True):
            acc += length
            if acc >= total / 2:
                n50 = length
                break
    return {"total_bp": total, "n_contigs": len(lengths), "n50": n50}


def check_assembly(fasta: Path) -> dict:
    """{"ok": bool, "reason": str, "warnings": [str], **stats}"""
    st = assembly_stats(fasta)
    total, n = st["total_bp"], st["n_contigs"]

    if total < MIN_ASSEMBLY_BP:
        return {
            **st, "ok": False, "warnings": [],
            "reason": (
                f"assembly is {total / 1e6:.2f} Mb, below the {MIN_ASSEMBLY_BP / 1e6:.1f} Mb "
                f"minimum for this species. Missing sequence cannot be distinguished "
                f"from absent resistance genes, and this is exactly the case that "
                f"produced a false 'likely to work' in testing."
            ),
        }
    if total > MAX_ASSEMBLY_BP:
        return {
            **st, "ok": False, "warnings": [],
            "reason": (
                f"assembly is {total / 1e6:.2f} Mb, above the {MAX_ASSEMBLY_BP / 1e6:.1f} Mb "
                f"maximum. That usually means contamination or more than one genome in "
                f"the file; separating mixed samples is out of scope for this tool."
            ),
        }

    warnings = []
    if n > WARN_CONTIGS:
        warnings.append(
            f"{n} contigs (N50 {st['n50']:,} bp) -- heavily fragmented. Measured "
            f"effect: at this level calls degrade into no-calls rather than into "
            f"wrong answers, but confidence is reduced."
        )
    return {**st, "ok": True, "reason": "assembly within expected size", "warnings": warnings}
