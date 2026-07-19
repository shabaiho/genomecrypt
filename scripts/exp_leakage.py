"""EXPERIMENT: does the SNP-cluster grouping actually prevent leakage?

Every metric in this project rests on the assumption that grouping by PDS_acc
keeps near-identical genomes out of both train and test. That assumption was
never measured -- it was inherited from NCBI's clustering and trusted.

Three checks:
  1. structural  -- can a cluster id appear on both sides? (must be impossible)
  2. genetic     -- how similar is each test genome to its nearest TRAINING
                    genome, versus its nearest same-split neighbour? If the
                    split works, test genomes should have no near-twin in train
  3. adversarial -- train a model to predict SPLIT MEMBERSHIP. If the split is
                    genetically arbitrary this is impossible (AUROC ~ 0.5)

Similarity is computed in the binary determinant space, which is a proxy for
genome relatedness -- coarse, but it is exactly the space the model sees, so a
near-twin here is a near-twin for leakage purposes.

    uv run python scripts/exp_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.config import Config  # noqa: E402
from gfw.train import grouped_four_way  # noqa: E402


def jaccard_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    inter = A @ B.T
    a = A.sum(1)[:, None]
    b = B.sum(1)[None, :]
    union = a + b - inter
    return np.divide(inter, np.maximum(union, 1))


def main() -> None:
    cfg = Config.load()
    X = pd.read_parquet(ROOT / "data/processed/features.parquet")
    L = pd.read_csv(ROOT / "data/processed/labels.csv")
    G = pd.read_csv(ROOT / "data/processed/groups.csv").set_index("genome_id").group_id

    print("=== cluster structure ===")
    sizes = G.value_counts()
    print(f"{G.nunique()} clusters over {len(G)} genomes")
    print(f"  singletons: {(sizes == 1).sum()}  ({100 * (sizes == 1).sum() / len(sizes):.0f}%)")
    print(f"  largest: {sizes.max()}   median size: {sizes.median():.0f}")
    # a split can only separate what the clustering actually groups; if almost
    # every genome is its own cluster, the grouping is doing nothing
    print(f"  genomes in a cluster of size 1: {sizes[sizes == 1].sum()} "
          f"({100 * sizes[sizes == 1].sum() / len(G):.0f}% of all genomes)")

    drug = cfg.drugs[0]
    sub = L[(L.drug_id == drug.id) & (L.genome_id.isin(X.index))]
    ids = sub.genome_id.to_numpy()
    ga = G.loc[sub.genome_id].to_numpy()
    tr, ca, th, te = grouped_four_way(ga, 0)

    print(f"\n=== structural check ({drug.id}) ===")
    overlap = set(ga[tr]) & set(ga[te])
    print(f"clusters appearing in BOTH train and test: {len(overlap)} "
          f"(must be 0) -> {'OK' if not overlap else 'LEAK'}")

    print("\n=== genetic proximity ===")
    A = X.loc[ids[te]].to_numpy(np.float32)
    B = X.loc[ids[tr]].to_numpy(np.float32)
    cross = jaccard_matrix(A, B)
    within = jaccard_matrix(A, A)
    np.fill_diagonal(within, 0.0)
    nn_train = cross.max(1)
    nn_test = within.max(1)
    print(f"test genome -> nearest TRAIN genome, mean Jaccard: {nn_train.mean():.3f}")
    print(f"test genome -> nearest TEST  genome, mean Jaccard: {nn_test.mean():.3f}")
    for thr in (0.95, 0.99, 1.0):
        n = int((nn_train >= thr).sum())
        print(f"  test genomes with a train twin at Jaccard >= {thr}: "
              f"{n} / {len(nn_train)} ({100 * n / len(nn_train):.0f}%)")

    print("\n=== adversarial check ===")
    idx = np.concatenate([tr, te])
    Xa = X.loc[ids[idx]].to_numpy(np.float32)
    ya = np.concatenate([np.zeros(len(tr)), np.ones(len(te))])
    auc = cross_val_score(LogisticRegression(max_iter=3000, class_weight="balanced"),
                          Xa, ya, cv=5, scoring="roc_auc")
    print(f"AUROC predicting split membership: {auc.mean():.3f} +/- {auc.std():.3f}")
    if auc.mean() > 0.7:
        print("  -> train and test differ systematically. Not leakage in the usual")
        print("     direction, but the test set is not exchangeable with training,")
        print("     so held-out numbers describe a shifted population.")
    else:
        print("  -> splits look exchangeable")


if __name__ == "__main__":
    main()
