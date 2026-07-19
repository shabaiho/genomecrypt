"""Config loading + the artifact-bundle layout shared by train and serve."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "drugs.yaml"
DEFAULT_MODEL_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class Drug:
    id: str
    display: str
    klass: str
    target_genes: list[str]
    intrinsic_resistance: bool


@dataclass(frozen=True)
class Config:
    species: str
    species_taxgroup: str
    abstain: dict
    decision: dict
    drugs: list[Drug]
    label_map: dict[str, int]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            species=raw["species"],
            species_taxgroup=raw["species_taxgroup"],
            abstain=raw["abstain"],
            decision=raw.get("decision", {"mode": "calibrated_abstain"}),
            drugs=[Drug(**d) for d in raw["drugs"]],
            label_map={k: int(v) for k, v in raw["label_map"].items()},
        )

    def drug(self, drug_id: str) -> Drug:
        for d in self.drugs:
            if d.id == drug_id:
                return d
        raise KeyError(drug_id)


# --- artifact bundle -------------------------------------------------------
# models/<version>/
#   metadata.json        -- species, drugs served, training provenance, git sha
#   feature_schema.json  -- ordered feature names; the ONLY contract between
#                           feature extraction and the served models
#   <drug_id>.joblib     -- calibrated sklearn pipeline for that drug
#   eval/report.json     -- held-out metrics, shown in the app's "Model card" tab


def bundle_path(version: str = "current", root: Path = DEFAULT_MODEL_DIR) -> Path:
    return Path(root) / version


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def write_json(p: Path, obj: dict) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2, sort_keys=True))
