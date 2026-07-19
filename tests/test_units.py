"""Regression tests for the pure functions.

Every test here corresponds to a bug that actually happened during development.
These functions have no I/O and no model, so they are cheap to test and were
exactly where the silent mistakes lived: three consecutive bugs in the feature
rollups, one in file sniffing, one in evidence labelling.

    uv run pytest -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gfw.explain import describe_feature, detected_mechanisms, gene_label  # noqa: E402
from gfw.features import (  # noqa: E402
    WrongFileType, aggregate_mutations, gene_family, parse_amrfinder_tsv,
    require_file_type, select_schema, sniff_file_type,
)
from gfw.mechanism import is_mechanistic  # noqa: E402
from gfw.predict import (  # noqa: E402
    CALL_FAIL, CALL_WORK, EV_KNOWN, EV_NONE, EV_STATISTICAL,
    evidence_category, is_curated,
)
from gfw.qc import assembly_stats, check_assembly  # noqa: E402
from gfw.storage import content_hash  # noqa: E402


# --------------------------------------------------------------- features ---

class TestGeneFamily:
    """blaKPC-2 and blaKPC-3 are one enzyme. Treating them as two features let L1
    keep one and zero the other, so a blaKPC-2 genome was called 'likely to work'
    while carrying a carbapenemase."""

    @pytest.mark.parametrize("symbol,family", [
        ("blaKPC-2", "blaKPC"),
        ("blaKPC-3", "blaKPC"),
        ("blaCTX-M-15", "blaCTX-M"),
        ("blaSHV-1", "blaSHV"),
        ("sul1", "sul1"),          # no numeric allele suffix to strip
        ("fosA", "fosA"),
        # OXA is two mechanisms wearing one prefix
        ("blaOXA-48", "blaOXA-48-like"),
        ("blaOXA-232", "blaOXA-48-like"),
        ("blaOXA-1", "blaOXA"),
        ("blaOXA-9", "blaOXA"),
    ])
    def test_family(self, symbol, family):
        assert gene_family(symbol) == family

    def test_carbapenemases_do_not_share_a_family_with_narrow_enzymes(self):
        """blaOXA-48 hydrolyses carbapenems; blaOXA-1 does not. One coefficient
        cannot represent both."""
        assert gene_family("blaOXA-48") != gene_family("blaOXA-1")

    def test_two_alleles_share_a_family(self):
        a = aggregate_mutations({"gene:blaKPC-2"})
        b = aggregate_mutations({"gene:blaKPC-3"})
        assert "genefam:blaKPC" in a and "genefam:blaKPC" in b


class TestMutationRollups:
    """73% of point mutations occur fewer than 3 times and are dropped by the
    prevalence filter. Without per-gene rollups the porin-loss signal vanishes."""

    def test_truncation_emits_all_three_levels(self):
        out = aggregate_mutations({"mut:ompK36_K231SfsTer16"})
        assert out >= {"mut:ompK36_K231SfsTer16", "mutgene:ompK36", "trunc:ompK36"}

    def test_substitution_is_not_marked_as_truncation(self):
        out = aggregate_mutations({"mut:gyrA_S83L"})
        assert "mutgene:gyrA" in out
        assert "trunc:gyrA" not in out

    def test_plain_gene_gets_no_mutation_rollup(self):
        out = aggregate_mutations({"gene:sul1"})
        assert not any(t.startswith(("mutgene:", "trunc:")) for t in out)


class TestSelectSchema:
    """gene:emrD sits in 99.9% of genomes: variance q(1-q) is 0.001, so it is a
    constant. L1 still gave it weight -1.18 in the meropenem model, where it
    offset blaKPC and flipped the verdict."""

    def test_drops_near_constant_features(self):
        counts = {"gene:everywhere": 99, "gene:useful": 40}
        schema = select_schema(counts, n_samples=100, min_prevalence=3,
                               max_prevalence=0.95)
        assert "gene:everywhere" not in schema
        assert "gene:useful" in schema

    def test_drops_rare_features(self):
        counts = {"gene:rare": 2, "gene:common": 40}
        schema = select_schema(counts, n_samples=100, min_prevalence=3)
        assert "gene:rare" not in schema
        assert "gene:common" in schema

    def test_schema_is_ordered(self):
        counts = {"gene:b": 10, "gene:a": 10, "gene:c": 10}
        assert select_schema(counts, 100) == sorted(select_schema(counts, 100))


# ------------------------------------------------------------ file typing ---

class TestSniffFileType:
    """A FASTA read as a TSV produced 68,074 rows, zero features, and a confident
    'likely to work' for every drug on a blaKPC-positive genome."""

    def test_fasta(self, tmp_path):
        p = tmp_path / "g.fna"
        p.write_text(">contig1\nACGTACGTACGT\n")
        assert sniff_file_type(p) == "fasta"

    def test_fasta_with_leading_blank_lines(self, tmp_path):
        p = tmp_path / "g.fna"
        p.write_text("\n\n>contig1\nACGTACGT\n")
        assert sniff_file_type(p) == "fasta"

    def test_protein_fasta_is_distinguished(self, tmp_path):
        p = tmp_path / "p.faa"
        p.write_text(">prot\nMKVLATTLLLASGAWA\n")
        assert sniff_file_type(p) == "protein_fasta"

    def test_amrfinder_tsv(self, tmp_path):
        p = tmp_path / "a.tsv"
        p.write_text("Protein id\tElement symbol\tClass\nx\tblaKPC-2\tBETA-LACTAM\n")
        assert sniff_file_type(p) == "amrfinder_tsv"

    def test_empty_file(self, tmp_path):
        p = tmp_path / "e.tsv"
        p.write_text("")
        assert sniff_file_type(p) == "unknown"

    def test_require_rejects_a_mismatch(self, tmp_path):
        p = tmp_path / "g.fna"
        p.write_text(">contig1\nACGTACGT\n")
        with pytest.raises(WrongFileType):
            require_file_type(p, "amrfinder_tsv")

    def test_empty_tsv_refuses_instead_of_crashing(self, tmp_path):
        p = tmp_path / "e.tsv"
        p.write_text("")
        with pytest.raises(WrongFileType):
            parse_amrfinder_tsv(p)


# --------------------------------------------------------------- evidence ---

class TestEvidenceLabelling:
    """The report announced 'Known resistance gene detected' under a 'likely to
    work' verdict, because the category ignored which direction the feature
    pushed. And genefam: rollups were labelled uncurated, so blaKPC -- the
    strongest carbapenemase signal -- read as 'not an established cause'."""

    @pytest.mark.parametrize("feature,curated", [
        ("gene:blaKPC-2", True),
        ("genefam:blaKPC", True),
        ("mut:gyrA_S83L", True),
        ("mutgene:ompK36", True),
        ("trunc:ompK35", True),
        ("class:BETA-LACTAM", False),   # a drug-class bucket, not a determinant
    ])
    def test_curated_flag(self, feature, curated):
        assert is_curated(feature) is curated

    def test_category_follows_the_direction_of_the_verdict(self):
        support = [{"feature": "gene:blaKPC-2", "direction": "toward_susceptible",
                    "curated_determinant": True, "weight": -1.0}]
        # a curated feature exists, but it argues the other way
        assert evidence_category(CALL_FAIL, support) == EV_STATISTICAL
        assert evidence_category(CALL_WORK, support) == EV_KNOWN

    def test_no_support_means_no_signal(self):
        assert evidence_category(CALL_FAIL, []) == EV_NONE

    def test_uncurated_support_is_statistical_only(self):
        support = [{"feature": "class:BETA-LACTAM", "direction": "toward_resistant",
                    "curated_determinant": False, "weight": 1.0}]
        assert evidence_category(CALL_FAIL, support) == EV_STATISTICAL


class TestMechanisticRelevance:
    """A curated determinant can still be irrelevant to the drug in hand: blaKPC
    predicts ciprofloxacin resistance through plasmid linkage, not pharmacology."""

    def test_carbapenemase_is_mechanistic_for_meropenem(self):
        assert is_mechanistic("genefam:blaKPC", "meropenem") is True

    def test_carbapenemase_is_not_mechanistic_for_ciprofloxacin(self):
        assert is_mechanistic("genefam:blaKPC", "ciprofloxacin") is False

    def test_gyrase_mutation_is_mechanistic_for_ciprofloxacin(self):
        assert is_mechanistic("mut:gyrA_S83L", "ciprofloxacin") is True

    def test_unknown_drug_is_never_mechanistic(self):
        assert is_mechanistic("genefam:blaKPC", "not_a_drug") is False


class TestNoVerdictStillReportsFindings:
    """A no-verdict hid a detected carbapenemase on a genome the laboratory had
    confirmed resistant. Gene detection is an observation, not a model opinion."""

    def test_detected_mechanisms_survive_a_refusal(self):
        support = [{"feature": "genefam:blaKPC", "direction": "toward_resistant",
                    "curated_determinant": True, "mechanistic_for_drug": True,
                    "weight": 1.6}]
        found = detected_mechanisms(support)
        assert len(found) == 1 and "blaKPC" in found[0]

    def test_susceptible_direction_is_not_reported_as_a_mechanism(self):
        support = [{"feature": "genefam:blaKPC", "direction": "toward_susceptible",
                    "curated_determinant": True, "mechanistic_for_drug": True,
                    "weight": -1.0}]
        assert detected_mechanisms(support) == []


class TestDescriptionsAreObjective:
    """Clinical copy must not carry adjectives the reader cannot check."""

    BANNED = ("some", "weak", "strong", "nearly", "almost", "little", "often",
              "usually", "many", "few")

    @pytest.mark.parametrize("feature", [
        "genefam:blaKPC", "genefam:blaNDM", "genefam:blaCTX-M", "gene:emrD",
        "gene:fosA", "genefam:blaSHV", "mutgene:ompK36", "trunc:ompK35",
        "gene:sul1", "gene:qnrB1", "gene:armA", "mutgene:gyrA",
    ])
    def test_no_vague_adjectives(self, feature):
        words = describe_feature(feature).lower().replace("-", " ").split()
        assert not [w for w in words if w.strip(",.;") in self.BANNED]

    def test_unknown_feature_does_not_invent_biology(self):
        assert describe_feature("gene:totallyUnknownXYZ") == "resistance-associated gene"

    def test_label_strips_internal_prefixes(self):
        assert gene_label("genefam:blaKPC") == "blaKPC family"
        assert gene_label("trunc:ompK35") == "ompK35 (disrupted)"
        assert ":" not in gene_label("mut:gyrA_S83L")


# --------------------------------------------------------------------- qc ---

class TestAssemblyQC:
    """Deleting 25% of contigs made an aminoglycoside determinant disappear and
    flipped gentamicin from fail to work. 4.67 Mb was the dangerous case."""

    def _write(self, path: Path, total_bp: int, contigs: int = 10) -> Path:
        per = total_bp // contigs
        with path.open("w") as fh:
            for i in range(contigs):
                fh.write(f">c{i}\n")
                fh.write("ACGT" * (per // 4) + "\n")
        return path

    def test_complete_assembly_passes(self, tmp_path):
        p = self._write(tmp_path / "ok.fna", 5_400_000)
        assert check_assembly(p)["ok"] is True

    def test_incomplete_assembly_is_refused(self, tmp_path):
        p = self._write(tmp_path / "small.fna", 4_000_000)
        r = check_assembly(p)
        assert r["ok"] is False and "below" in r["reason"]

    def test_oversized_assembly_is_refused(self, tmp_path):
        p = self._write(tmp_path / "big.fna", 7_000_000)
        r = check_assembly(p)
        assert r["ok"] is False and "contamination" in r["reason"]

    def test_stats_count_bases_not_lines(self, tmp_path):
        p = tmp_path / "s.fna"
        p.write_text(">a\nACGT\nACGT\n>b\nAC\n")
        st = assembly_stats(p)
        assert st["total_bp"] == 10 and st["n_contigs"] == 2

    def test_fragmentation_warns_but_passes(self, tmp_path):
        p = self._write(tmp_path / "frag.fna", 5_400_000, contigs=2000)
        r = check_assembly(p)
        assert r["ok"] is True and r["warnings"]


# ---------------------------------------------------------------- storage ---

class TestContentAddressing:
    """Two files with the same name must not collide; the same genome uploaded
    twice must not be annotated twice."""

    def test_same_bytes_same_digest(self):
        assert content_hash(b">c\nACGT\n") == content_hash(b">c\nACGT\n")

    def test_different_bytes_differ(self):
        assert content_hash(b">c\nACGT\n") != content_hash(b">c\nACGA\n")


# --------------------------------------------------------- shipped bundle ---

class TestShippedBundle:
    """The bundle the app actually loads has to be self-describing."""

    BUNDLE = ROOT / "models" / "current"

    def test_bundle_exists(self):
        assert self.BUNDLE.exists(), "models/current is missing; run make promote"

    def test_schema_and_models_agree(self):
        import joblib

        schema = json.loads((self.BUNDLE / "feature_schema.json").read_text())["features"]
        for mp in self.BUNDLE.glob("*.joblib"):
            bundle = joblib.load(mp)
            assert bundle["schema"] == schema, f"{mp.name} was trained on another schema"
            assert len(bundle["coef"]) == len(schema)

    def test_metadata_records_provenance(self):
        meta = json.loads((self.BUNDLE / "metadata.json").read_text())
        assert meta["git_sha"] != "unknown", "bundle cannot be traced to a commit"
        assert meta["drugs_served"]

    def test_prevalence_ships_with_the_schema(self):
        schema = json.loads((self.BUNDLE / "feature_schema.json").read_text())
        assert schema.get("prevalence"), "the report needs measured prevalence"
        assert all(0.0 <= v <= 1.0 for v in schema["prevalence"].values())
