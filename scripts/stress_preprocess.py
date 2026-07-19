"""STRESS TEST: preprocessing robustness.

The pipeline is only as trustworthy as its front door. Everything here is an input
a real user could plausibly hand us, and for each one there are exactly two
acceptable outcomes: handle it correctly, or refuse loudly. Silently producing a
prediction from mangled input is the failure mode that matters, because the output
looks identical to a good one.

    uv run python scripts/stress_preprocess.py
"""
from __future__ import annotations

import gzip
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.features import (  # noqa: E402
    WrongFileType, determinants, parse_amrfinder_tsv, sniff_file_type,
)
from gfw.predict import Predictor  # noqa: E402

DEMO = ROOT / "data" / "demo"
FASTA = DEMO / "GCA_000417485.1.fna"
TSV = DEMO / "GCA_000417485.1.tsv"
VERSION = "v19"

rows: list[tuple[str, str, str]] = []


def record(case: str, outcome: str, note: str) -> None:
    rows.append((case, outcome, note))


def probe_fasta(case: str, path: Path, expect: str = "fasta") -> None:
    """Sniffing is the guard that decides which pipeline branch runs."""
    try:
        kind = sniff_file_type(path)
        ok = kind == expect
        record(case, "OK" if ok else "BUG", f"detected as {kind!r}, expected {expect!r}")
    except Exception as e:
        record(case, "BUG", f"sniff raised {type(e).__name__}: {e}")


def probe_tsv_predict(case: str, path: Path, pred: Predictor) -> None:
    """Feeding a TSV all the way to a prediction."""
    try:
        toks = determinants(parse_amrfinder_tsv(path))
        rep = pred.predict_from_tsv(path, case)
        calls = {r.call for r in rep.results}
        confident = [r for r in rep.results if r.call != "no_call"]
        if not toks and confident:
            record(case, "BUG",
                   f"0 determinants but {len(confident)} confident calls: {calls}")
        else:
            record(case, "OK", f"{len(toks)} determinants, {len(confident)} confident calls")
    except WrongFileType as e:
        record(case, "OK (refused)", str(e)[:60])
    except Exception as e:
        record(case, "CRASH", f"{type(e).__name__}: {e}"[:70])


def main() -> None:
    pred = Predictor(VERSION)
    raw = FASTA.read_text()

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)

        # ---------- FASTA variants a sequencer or user really produces ----------
        p = t / "lower.fna"
        p.write_text("".join(ln if ln.startswith(">") else ln.lower()
                             for ln in raw.splitlines(keepends=True)))
        probe_fasta("FASTA lowercase bases", p)

        p = t / "crlf.fna"
        p.write_bytes(raw.replace("\n", "\r\n").encode())
        probe_fasta("FASTA Windows CRLF line endings", p)

        p = t / "leadblank.fna"
        p.write_text("\n\n" + raw)
        probe_fasta("FASTA with leading blank lines", p)

        p = t / "ambig.fna"
        p.write_text("".join(ln if ln.startswith(">") else ln.replace("A", "N")
                             for ln in raw.splitlines(keepends=True)))
        probe_fasta("FASTA full of N ambiguity codes", p)

        p = t / "empty.fna"
        p.write_text("")
        probe_fasta("empty file", p, expect="unknown")

        p = t / "headeronly.fna"
        p.write_text(">contig1 no sequence follows\n")
        probe_fasta("FASTA header with no sequence", p)

        p = t / "protein.faa"
        p.write_text(">prot1\nMKVLATTLLLASGAWA\n")
        probe_fasta("PROTEIN fasta (wrong molecule)", p, expect="protein_fasta")

        p = t / "gz.fna"
        with gzip.open(p, "wt") as fh:
            fh.write(raw[:5000])
        probe_fasta("gzipped file with a .fna name", p, expect="unknown")

        # ---------- AMRFinderPlus TSV variants ----------
        tsv_raw = TSV.read_text()
        head = tsv_raw.splitlines()[0]

        p = t / "hdr.tsv"
        p.write_text(head + "\n")
        probe_tsv_predict("TSV header only, zero hits", p, pred)

        p = t / "trunc.tsv"
        lines = tsv_raw.splitlines()
        p.write_text("\n".join(lines[:3]) + "\n" + lines[3][:20])
        probe_tsv_predict("TSV truncated mid-row", p, pred)

        p = t / "extra.tsv"
        p.write_text(tsv_raw.replace(head, head + "\tExtraColumn"))
        probe_tsv_predict("TSV with an unexpected extra column", p, pred)

        p = t / "noheader.tsv"
        p.write_text("\n".join(tsv_raw.splitlines()[1:]))
        probe_tsv_predict("TSV with the header row missing", p, pred)

        p = t / "empty.tsv"
        p.write_text("")
        probe_tsv_predict("completely empty TSV", p, pred)

        probe_tsv_predict("FASTA renamed to .tsv", FASTA, pred)

        p = t / "csv.tsv"
        p.write_text(tsv_raw.replace("\t", ","))
        probe_tsv_predict("comma-separated instead of tab", p, pred)

    # ---------- report ----------
    width = max(len(c) for c, _, _ in rows)
    print("PREPROCESSING STRESS TEST\n")
    for case, outcome, note in rows:
        print(f"[{outcome:12s}] {case:<{width}}  {note}")
    bad = [r for r in rows if r[1] in ("BUG", "CRASH")]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} handled acceptably, {len(bad)} problems")
    for case, outcome, note in bad:
        print(f"  {outcome}: {case} -- {note}")


if __name__ == "__main__":
    main()
