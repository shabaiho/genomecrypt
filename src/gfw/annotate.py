"""Module 01a -- assembled FASTA -> AMRFinderPlus TSV.

This is the only step that needs the heavy `ncbi/amr` image. Everything
downstream consumes the TSV, which is why the demo app can run without it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class AnnotatorUnavailable(RuntimeError):
    pass


def amrfinder_available() -> bool:
    return shutil.which("amrfinder") is not None


def run_amrfinder(
    fasta: Path,
    out_tsv: Path,
    organism: str,
    threads: int = 4,
    timeout_s: int = 900,
) -> Path:
    """Nucleotide-mode scan. --plus adds stress/virulence genes; --organism
    enables the point-mutation (gyrA/parC etc.) screen, which we need."""
    if not amrfinder_available():
        raise AnnotatorUnavailable(
            "amrfinder not on PATH. Use the `full` container target, or upload a "
            "precomputed AMRFinderPlus TSV instead of a FASTA."
        )
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "amrfinder",
        "--nucleotide", str(fasta),
        "--organism", organism,
        "--plus",
        "--threads", str(threads),
        "--output", str(out_tsv),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"amrfinder failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return out_tsv


def batch_annotate(
    fasta_dir: Path,
    out_dir: Path,
    organism: str,
    threads: int = 2,
    jobs: int | None = None,
) -> list[Path]:
    """Annotate every *.fna/*.fasta in a directory.

    MEASURED: ~76s per K. pneumoniae genome at --threads 4 (v4.2.7, 8-core box).
    Sequentially that is ~17h for 800 genomes, which does not fit a hackathon --
    so we run several genomes concurrently. amrfinder parallelizes poorly beyond
    ~2 threads, so many small jobs beat few big ones.

    Resumable: already-written TSVs are skipped, so Ctrl-C and rerun is safe.
    Memory is the real limit -- each job peaks around 1GB, so cap jobs by RAM too.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir.mkdir(parents=True, exist_ok=True)
    fastas = sorted([*fasta_dir.glob("*.fna"), *fasta_dir.glob("*.fasta")])
    todo, written = [], []
    for fa in fastas:
        dest = out_dir / f"{fa.stem}.tsv"
        if dest.exists() and dest.stat().st_size > 0:
            written.append(dest)
        else:
            todo.append((fa, dest))

    if not todo:
        print(f"all {len(written)} genomes already annotated")
        return written

    jobs = jobs or max(1, (os.cpu_count() or 4) // max(1, threads))
    print(f"annotating {len(todo)} genomes ({len(written)} cached) "
          f"with {jobs} jobs x {threads} threads", flush=True)

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_amrfinder, fa, dest, organism, threads): fa
                for fa, dest in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            fa = futs[fut]
            try:
                written.append(fut.result())
            except Exception as e:  # one bad assembly must not kill the whole run
                print(f"  SKIP {fa.name}: {e}", flush=True)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    return written
