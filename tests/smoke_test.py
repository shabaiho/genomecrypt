"""End-to-end smoke test on synthetic AMRFinderPlus output.

No network, no AMRFinderPlus, no FASTA -- proves the train->bundle->serve contract
holds so you can develop the UI before the real data finishes downloading.

    PYTHONPATH=src python tests/smoke_test.py
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gfw.config import Config, bundle_path  # noqa: E402
from gfw.features import build_matrix  # noqa: E402
from gfw.predict import Predictor  # noqa: E402

HEADER = ["Element symbol", "Element type", "Element subtype", "Class",
          "% Coverage of reference sequence", "% Identity to reference sequence"]

GENES = {
    "ciprofloxacin": [("gyrA_S83L", "POINT", "QUINOLONE"), ("qnrB1", "AMR", "QUINOLONE")],
    "gentamicin": [("aac(3)-IIa", "AMR", "AMINOGLYCOSIDE"), ("armA", "AMR", "AMINOGLYCOSIDE")],
    "meropenem": [("blaKPC-2", "AMR", "BETA-LACTAM"), ("blaNDM-1", "AMR", "BETA-LACTAM")],
    "ceftriaxone": [("blaCTX-M-15", "AMR", "BETA-LACTAM"), ("blaSHV-12", "AMR", "BETA-LACTAM")],
    "trimethoprim_sulfamethoxazole": [("dfrA14", "AMR", "TRIMETHOPRIM"), ("sul1", "AMR", "SULFONAMIDE")],
}
NOISE = [("tet(A)", "AMR", "TETRACYCLINE"), ("fosA", "AMR", "FOSFOMYCIN"),
         ("oqxA", "AMR", "PHENICOL/QUINOLONE"), ("catA1", "AMR", "PHENICOL")]


def write_tsv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    df = pd.DataFrame(
        [[s, t, "", c, 100.0, 99.5] for s, t, c in rows],
        columns=HEADER,
    )
    df.to_csv(path, sep="\t", index=False)


def synth(tmp: Path, n_genomes: int = 400, n_groups: int = 60, seed: int = 7):
    rng = random.Random(seed)
    cfg = Config.load()
    tsv_dir = tmp / "tsv"
    tsv_dir.mkdir()

    label_rows, group_rows, tsvs = [], [], {}
    for i in range(n_genomes):
        gid = f"g{i:04d}"
        group = i % n_groups          # clones share a group -> tests the grouped split
        rows = [rng.choice(NOISE) for _ in range(rng.randint(0, 3))]
        for drug_id, dgenes in GENES.items():
            resistant = rng.random() < 0.4
            if resistant:
                rows.append(rng.choice(dgenes))
            # 8% label noise, so metrics are not a suspicious 1.000
            label = int(resistant) if rng.random() > 0.08 else int(not resistant)
            label_rows.append({"genome_id": gid, "drug_id": drug_id, "label": label})
        p = tsv_dir / f"{gid}.tsv"
        write_tsv(p, rows)
        tsvs[gid] = p
        group_rows.append({"genome_id": gid, "group_id": group})

    X, schema = build_matrix(tsvs, min_prevalence=3)
    proc = tmp / "processed"
    proc.mkdir()
    X.to_parquet(proc / "features.parquet")
    pd.DataFrame(label_rows).to_csv(proc / "labels.csv", index=False)
    pd.DataFrame(group_rows).to_csv(proc / "groups.csv", index=False)
    return proc, tsvs, cfg


def main() -> None:
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        proc, tsvs, cfg = synth(tmp)
        print(f"synthetic: {pd.read_parquet(proc / 'features.parquet').shape} matrix")

        env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        import os
        env = {**os.environ, **env}

        for mod in ("gfw.train", "gfw.evaluate"):
            r = subprocess.run(
                [sys.executable, "-m", mod, "--version", "smoketest",
                 "--matrix", str(proc / "features.parquet"),
                 "--labels", str(proc / "labels.csv"),
                 "--groups", str(proc / "groups.csv")],
                env=env, capture_output=True, text=True,
            )
            print(f"--- {mod} ---\n{r.stdout}{r.stderr[-1500:] if r.returncode else ''}")
            assert r.returncode == 0, f"{mod} failed"

        pred = Predictor("smoketest", cfg)
        assert pred.served_drugs, "no drugs served"

        gid = next(iter(tsvs))
        report = pred.predict_from_tsv(tsvs[gid], sample_id=gid, targets_found=None)
        calls = {r.drug_id: (r.call, r.confidence) for r in report.results}
        print("\nsample report:", calls)
        assert len(report.results) == len(cfg.drugs)

        # a genome full of unknown determinants must be refused, not extrapolated on
        weird = {f"gene:blaFAKE-{i}" for i in range(10)}
        ood = pred.predict_from_tokens(weird, "weird", targets_found=None)
        assert all(r.call == "no_call" for r in ood.results if r.reason != "drug_not_covered"), \
            "OOD genome was not refused"
        print("OOD genome correctly refused ✓")

        print(f"\nbundle written to {bundle_path('smoketest')}")
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
