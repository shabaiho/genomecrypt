"""Module 02a -- sequence-homology de-duplication -> group labels for the split.

Why: near-identical genomes (same outbreak clone) landing in both train and test
turn the evaluation into a memory test. The brief calls this out explicitly as
what separates a strong submission from a weak one.

Method: sourmash MinHash sketch per genome (k=31, scaled=1000) -> pairwise
containment/Jaccard -> single-linkage clustering at a distance threshold.
Threshold is a TUNABLE we must justify in the writeup:

    jaccard >= 0.95  ~  ANI >= 99.9%   very tight, keeps more groups
    jaccard >= 0.90  ~  ANI >= 99.5%   DEFAULT -- collapses outbreak clones
    jaccard >= 0.80  ~  ANI >= 99.0%   aggressive, may merge distinct lineages

Justify by reporting: #groups, largest group size, and how balanced accuracy
moves as the threshold changes (a model that only memorizes will fall off a cliff).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

KSIZE = 31
SCALED = 1000
DEFAULT_JACCARD = 0.90


def sketch_dir(fasta_dir: Path, out_sig: Path, ksize: int = KSIZE, scaled: int = SCALED) -> Path:
    import sourmash

    sigs = []
    for fa in sorted([*fasta_dir.glob("*.fna"), *fasta_dir.glob("*.fasta")]):
        mh = sourmash.MinHash(n=0, ksize=ksize, scaled=scaled)
        for _, seq in _iter_fasta(fa):
            mh.add_sequence(seq, force=True)
        sigs.append(sourmash.SourmashSignature(mh, name=fa.stem))
    out_sig.parent.mkdir(parents=True, exist_ok=True)
    with sourmash.sourmash_args.SaveSignaturesToLocation(str(out_sig)) as save:
        for s in sigs:
            save.add(s)
    return out_sig


def _iter_fasta(path: Path):
    import pyfastx

    for rec in pyfastx.Fasta(str(path), build_index=False):
        yield rec[0], rec[1]


def cluster(sig_path: Path, threshold: float = DEFAULT_JACCARD) -> dict[str, int]:
    """Single-linkage clustering via union-find. Returns {genome_id: group_id}."""
    import sourmash

    sigs = list(sourmash.load_file_as_signatures(str(sig_path)))
    names = [s.name for s in sigs]
    parent = list(range(len(sigs)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            if sigs[i].minhash.jaccard(sigs[j].minhash) >= threshold:
                union(i, j)

    roots = {find(i) for i in range(len(sigs))}
    remap = {r: g for g, r in enumerate(sorted(roots))}
    return {names[i]: remap[find(i)] for i in range(len(sigs))}


def summarize(groups: dict[str, int]) -> dict:
    sizes = np.bincount(list(groups.values()))
    return {
        "n_genomes": len(groups),
        "n_groups": int(len(sizes)),
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "singleton_groups": int((sizes == 1).sum()),
    }
