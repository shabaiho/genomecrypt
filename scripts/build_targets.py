"""Build config/targets.fna -- reference nucleotide sequences of the drug targets.

The deterministic gate (gfw.gate) blastn-screens an assembly for the gene the drug
actually binds. Without this file the gate reports `gate_skipped` and every
"likely to work" is unverified.

Source: K. pneumoniae HS11286 (GCF_000240185.1 / NC_016845.1), the RefSeq
reference. Its CDS file predates the [gene=] tag, so genes are matched on the
[protein=] description instead.

    uv run python scripts/build_targets.py
"""
from __future__ import annotations

import gzip
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "config" / "targets.fna"
CHROMOSOME = "NC_016845.1"   # HS11286 chromosome; the rest of the assembly is plasmids
CDS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/240/185/"
    "GCF_000240185.1_ASM24018v2/GCF_000240185.1_ASM24018v2_cds_from_genomic.fna.gz"
)

# gene -> ordered list of [protein=] patterns, most specific first.
# Keep these in sync with target_genes in config/drugs.yaml.
TARGETS: dict[str, list[str]] = {
    "gyrA": [r"DNA gyrase subunit A"],
    "parC": [r"DNA topoisomerase IV subunit A", r"topoisomerase IV subunit A"],
    "ftsI": [r"peptidoglycan D,D-transpeptidase FtsI", r"penicillin-binding protein 3",
             r"cell division protein FtsI"],
    "mrdA": [r"penicillin-binding protein 2\b", r"peptidoglycan D,D-transpeptidase MrdA"],
    "folA": [r"dihydrofolate reductase"],
    "folP": [r"dihydropteroate synthase"],
    "rpsL": [r"30S ribosomal protein S12"],
}


def iter_fasta(text: str):
    header, seq = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if header:
                yield header, "".join(seq)
            header, seq = line[1:], []
        else:
            seq.append(line.strip())
    if header:
        yield header, "".join(seq)


def main() -> None:
    print("downloading reference CDS ...", flush=True)
    with urllib.request.urlopen(CDS_URL, timeout=300) as r:
        text = gzip.decompress(r.read()).decode("utf-8", "replace")

    records = list(iter_fasta(text))
    print(f"{len(records)} CDS in reference")

    found: dict[str, tuple[str, str]] = {}
    for gene, patterns in TARGETS.items():
        for pat in patterns:
            rx = re.compile(pat, re.IGNORECASE)
            hits = [(h, s) for h, s in records
                    if (m := re.search(r"\[protein=([^\]]+)\]", h)) and rx.search(m.group(1))
                    # CHROMOSOME ONLY. HS11286 carries plasmids, and a plasmid
                    # "dihydrofolate reductase" is dfrA and a plasmid
                    # "dihydropteroate synthase" is sul -- ACQUIRED RESISTANCE
                    # GENES, not drug targets. Putting those in targets.fna
                    # inverts the gate: it would report "target absent" for
                    # exactly the susceptible genomes.
                    and h.startswith(f"lcl|{CHROMOSOME}")]
            if hits:
                # Prefer an EXACT description match. "30S ribosomal protein S12"
                # is a substring of "...S12 methylthiotransferase RimO", and
                # "penicillin-binding protein 2" of "cell elongation ... PBP2";
                # picking longest or shortest among substring matches silently
                # grabs the wrong protein.
                exact = [(h, s) for h, s in hits
                         if (m := re.search(r"\[protein=([^\]]+)\]", h))
                         and m.group(1).strip().lower() == pat.replace(r"\b", "").lower()]
                h, s = (exact or hits)[0] if exact else max(hits, key=lambda t: len(t[1]))
                pid = re.search(r"\[protein_id=([^\]]+)\]", h)
                found[gene] = (pid.group(1) if pid else "unknown", s)
                break

    missing = [g for g in TARGETS if g not in found]
    if missing:
        print(f"WARNING: no reference sequence found for {missing}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for gene, (pid, seq) in sorted(found.items()):
            # header format expected by gfw.gate.detect_targets: <gene>|<species>|<acc>
            fh.write(f">{gene}|K.pneumoniae|{pid}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
            print(f"  {gene:6s} {len(seq):5d} bp  {pid}")

    print(f"\nwrote {len(found)} targets -> {OUT}")


if __name__ == "__main__":
    main()
