"""Persist what was uploaded and what was answered.

Two reasons, and neither is convenience.

AUDIT. A tool that influences treatment has to be able to answer "what exactly did
it say about this sample, and on which model version" months later. The report
alone is not enough -- the input has to be kept too, because a prediction is only
interpretable against the file that produced it.

REUSE. The same isolate gets re-uploaded, and re-running AMRFinderPlus on a genome
already processed costs minutes for no new information. Keying the cache on the
file's content hash makes repeats instant and makes it impossible to confuse two
files with the same name.

Layout under data/store/:
    uploads/<sha256[:16]>/genome.fna        the file exactly as received
    uploads/<sha256[:16]>/amrfinder.tsv     the annotation, if we produced one
    uploads/<sha256[:16]>/meta.json         name, size, timestamp, file type
    reports/<sha256[:16]>__<model>.json     one report per (sample, model version)
    index.jsonl                             one line per submission, append-only
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .config import REPO_ROOT

STORE = REPO_ROOT / "data" / "store"


def content_hash(data: bytes) -> str:
    """Short content address. Two identical genomes land in one place regardless
    of what the user named them."""
    return hashlib.sha256(data).hexdigest()[:16]


def sample_dir(digest: str) -> Path:
    return STORE / "uploads" / digest


def already_annotated(digest: str) -> Path | None:
    """Return a cached AMRFinderPlus result for this exact file, if we have one."""
    p = sample_dir(digest) / "amrfinder.tsv"
    return p if p.exists() and p.stat().st_size > 0 else None


def save_upload(data: bytes, filename: str, kind: str) -> str:
    """Store the raw file. Returns its content digest."""
    digest = content_hash(data)
    d = sample_dir(digest)
    d.mkdir(parents=True, exist_ok=True)
    target = d / ("genome.fna" if kind == "fasta" else "amrfinder.tsv")
    if not target.exists():
        target.write_bytes(data)
    meta = {
        "digest": digest,
        "original_name": filename,
        "bytes": len(data),
        "file_type": kind,
        "first_seen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    mp = d / "meta.json"
    if mp.exists():
        meta["first_seen"] = json.loads(mp.read_text()).get("first_seen", meta["first_seen"])
    mp.write_text(json.dumps(meta, indent=2))
    return digest


def save_annotation(digest: str, tsv_path: Path) -> Path:
    """Keep the annotation so the same genome is never re-annotated."""
    d = sample_dir(digest)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / "amrfinder.tsv"
    if not dest.exists():
        shutil.copyfile(tsv_path, dest)
    return dest


def save_report(digest: str, model_version: str, report: dict) -> Path:
    """One report per sample per model version -- retraining does not overwrite
    what an earlier model said about the same genome."""
    d = STORE / "reports"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{digest}__{model_version}.json"
    dest.write_text(json.dumps(report, indent=2))

    line = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "digest": digest,
        "model_version": model_version,
        "sample_id": report.get("sample_id"),
        "verdicts": {r["drug_id"]: r["call"] for r in report.get("results", [])},
    }
    with (STORE / "index.jsonl").open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    return dest


def history(limit: int = 50) -> list[dict]:
    """Most recent submissions, newest first."""
    idx = STORE / "index.jsonl"
    if not idx.exists():
        return []
    rows = [json.loads(x) for x in idx.read_text().splitlines() if x.strip()]
    return rows[::-1][:limit]


def stats() -> dict:
    rows = history(limit=10 ** 9)
    return {
        "submissions": len(rows),
        "unique_samples": len({r["digest"] for r in rows}),
        "model_versions": sorted({r["model_version"] for r in rows}),
    }
