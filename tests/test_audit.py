from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bridgecheck.artifact import BridgeArtifact
from bridgecheck.audit import PairedSpectrum, audit_paired_spectra
from bridgecheck.predict import ContractError


def test_paired_audit_pass_requires_every_frozen_conjunct(
    artifact: BridgeArtifact, passing_samples: list[PairedSpectrum]
) -> None:
    report = audit_paired_spectra(artifact, passing_samples, bootstrap_repeats=500)

    assert report["status"] == "SUPPORTED_FOR_RECONSTRUCTION_RESEARCH"
    assert report["population"] == {"samples": 2, "biological_groups": 2}
    assert all(report["checks"].values())
    assert len(report["delta_bins"]) == 6
    assert all(row["present"] for row in report["delta_bins"].values())
    assert report["metrics"]["candidate"]["equal_group_mae"] == pytest.approx(0.0, abs=1e-15)
    assert report["metrics"]["context_shuffle"]["equal_group_mae"] > 0.0
    assert report["metrics"]["blank_context"]["equal_group_mae"] > 0.0
    assert report["paired_inference"]["all_context_mean"]["ci95"][0] > 0.0
    assert report["paired_inference"]["context_edge_mean"]["ci95"][0] > 0.0
    assert "no_downstream_or_measurement_equivalence_claim" in report["claim_ceiling"]
    assert any("does not establish downstream" in warning for warning in report["warnings"])


def test_audit_is_deterministic_with_frozen_seeds(
    artifact: BridgeArtifact, passing_samples: list[PairedSpectrum]
) -> None:
    first = audit_paired_spectra(artifact, passing_samples, bootstrap_repeats=250)
    second = audit_paired_spectra(artifact, passing_samples, bootstrap_repeats=250)

    assert first == second


def test_audit_fails_closed_when_comparator_is_better(
    artifact: BridgeArtifact, failing_samples: list[PairedSpectrum]
) -> None:
    report = audit_paired_spectra(artifact, failing_samples, bootstrap_repeats=300)

    assert report["status"] == "NOT_SUPPORTED"
    assert report["checks"]["candidate_beats_naive_ci95_lower_above_zero"] is False
    assert report["metrics"]["all_context_mean"]["equal_group_mae"] == pytest.approx(0.0)


def test_audit_rejects_duplicate_samples_and_nonindependent_population(
    artifact: BridgeArtifact, passing_samples: list[PairedSpectrum]
) -> None:
    duplicate = [passing_samples[0], replace(passing_samples[1], sample_id="sample-0")]
    one_group = [passing_samples[0], replace(passing_samples[1], group_id="plant-0")]

    with pytest.raises(ContractError, match="sample_id values must be unique"):
        audit_paired_spectra(artifact, duplicate, bootstrap_repeats=100)
    with pytest.raises(ContractError, match="at least two independent"):
        audit_paired_spectra(artifact, one_group, bootstrap_repeats=100)


def test_audit_rejects_blank_identity_fields(
    artifact: BridgeArtifact, passing_samples: list[PairedSpectrum]
) -> None:
    blank_sample = [replace(passing_samples[0], sample_id=""), passing_samples[1]]
    blank_group = [replace(passing_samples[0], group_id=""), passing_samples[1]]

    with pytest.raises(ContractError, match="sample_id.*non-empty|non-empty.*sample_id"):
        audit_paired_spectra(artifact, blank_sample, bootstrap_repeats=100)
    with pytest.raises(ContractError, match="group_id.*non-empty|non-empty.*group_id"):
        audit_paired_spectra(artifact, blank_group, bootstrap_repeats=100)


@pytest.mark.parametrize("invalid", [1.000001, -0.050001, np.nan])
def test_audit_rejects_invalid_measured_target_without_clipping(
    artifact: BridgeArtifact,
    passing_samples: list[PairedSpectrum],
    invalid: float,
) -> None:
    target = passing_samples[0].target_reflectance.copy()
    target[10] = invalid
    samples = [replace(passing_samples[0], target_reflectance=target), passing_samples[1]]

    with pytest.raises(ContractError, match="finite|without clipping"):
        audit_paired_spectra(artifact, samples, bootstrap_repeats=100)


def test_audit_rejects_target_wavelengths_outside_native_contract(
    artifact: BridgeArtifact, passing_samples: list[PairedSpectrum]
) -> None:
    invalid = replace(
        passing_samples[0],
        target_wavelength_nm=np.asarray([1048.0, 1052.0]),
        target_reflectance=np.asarray([0.4, 0.4]),
    )
    with pytest.raises(ContractError, match="1052.*2400"):
        audit_paired_spectra(artifact, [invalid, passing_samples[1]], bootstrap_repeats=100)


@pytest.mark.parametrize("repeats", [0, 99, 100_001])
def test_core_audit_enforces_bootstrap_bounds(
    artifact: BridgeArtifact, passing_samples: list[PairedSpectrum], repeats: int
) -> None:
    with pytest.raises(ContractError, match="bootstrap_repeats.*100.*100000"):
        audit_paired_spectra(artifact, passing_samples, bootstrap_repeats=repeats)
