from __future__ import annotations

import numpy as np
import pytest

from bridgecheck.artifact import BridgeArtifact
from bridgecheck.predict import ContractError, predict_spectrum, validate_context


def _valid_context(artifact: BridgeArtifact, state: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return (
        artifact.wavelengths_nm[artifact.context_mask].copy(),
        artifact.bank[state, artifact.context_mask].copy(),
    )


def test_exact_candidate_retrieval_preserves_origin_and_target_grid(
    artifact: BridgeArtifact,
) -> None:
    wavelength, context = _valid_context(artifact, state=2)
    result = predict_spectrum(artifact, wavelength, context, neighbors=3)

    assert result.nearest_candidate == 2
    assert result.context_fit_rmse == pytest.approx(0.0, abs=1e-15)
    assert result.support_tier == "WITHIN_REFERENCE_Q95"
    assert result.claim_status == "CANDIDATE_ONLY_UNVALIDATED"
    assert result.neighbor_count == 3
    np.testing.assert_array_equal(result.wavelengths_nm, artifact.wavelengths_nm[artifact.target_mask])
    np.testing.assert_array_equal(result.reflectance, artifact.bank[2, artifact.target_mask])

    public = result.to_dict()
    assert public["derived"]["origin"] == "model_derived"
    assert public["derived"]["observed_band_mask"] == [False] * 338
    assert public["retrieval"]["nearest_candidate"] == "state-0002"
    assert public["retrieval"]["calibrated_prediction_interval"] is None
    assert public["derived"]["neighbor_envelope"]["kind"] == (
        "descriptive_not_calibrated_uncertainty"
    )
    assert any("not measured SWIR" in warning for warning in public["warnings"])


def test_irregular_supported_grid_uses_linear_candidate_sampling(artifact: BridgeArtifact) -> None:
    wavelength = np.arange(400.0, 1000.0 + 3.0, 6.0, dtype=np.float64)
    full_wavelength, full_context = _valid_context(artifact, state=0)
    context = np.interp(wavelength, full_wavelength, full_context)

    result = predict_spectrum(artifact, wavelength, context)

    assert len(wavelength) == 101
    assert result.nearest_candidate == 0
    assert result.context_fit_rmse == pytest.approx(0.0, abs=1e-14)


def test_input_hash_is_deterministic_and_input_sensitive(artifact: BridgeArtifact) -> None:
    wavelength, context = _valid_context(artifact)
    first = predict_spectrum(artifact, wavelength, context)
    second = predict_spectrum(artifact, wavelength.copy(), context.copy())
    changed = predict_spectrum(artifact, wavelength, context + 1e-6)

    assert first.input_sha256 == second.input_sha256
    assert first.input_sha256 != changed.input_sha256


def test_support_tiers_are_descriptive_only(artifact: BridgeArtifact) -> None:
    wavelength, context = _valid_context(artifact)

    tail = predict_spectrum(artifact, wavelength, context + 0.01)
    outside = predict_spectrum(artifact, wavelength, np.zeros_like(context))

    assert tail.support_tier == "REFERENCE_TAIL_Q95_Q99"
    assert outside.support_tier == "OUTSIDE_REFERENCE_Q99"
    assert tail.to_dict()["retrieval"]["support_metric"] == "reference_distance_descriptive_only"
    assert any("paired measurement audit" in warning for warning in outside.warnings)


@pytest.mark.parametrize("neighbors", [0, 11])
def test_neighbor_count_contract_is_enforced(artifact: BridgeArtifact, neighbors: int) -> None:
    wavelength, context = _valid_context(artifact)
    with pytest.raises(ContractError, match="between 1 and 10"):
        predict_spectrum(artifact, wavelength, context, neighbors=neighbors)


def test_rejects_shape_and_nonfinite_values(artifact: BridgeArtifact) -> None:
    wavelength, context = _valid_context(artifact)
    with pytest.raises(ContractError, match="equal-length one-dimensional"):
        validate_context(wavelength[:, None], context, artifact.manifest["input_contract"])
    with pytest.raises(ContractError, match="finite"):
        validate_context(wavelength, np.where(wavelength == 600.0, np.nan, context), artifact.manifest["input_contract"])


def test_rejects_too_few_bands(artifact: BridgeArtifact) -> None:
    wavelength, context = _valid_context(artifact)
    with pytest.raises(ContractError, match="at least 100"):
        predict_spectrum(artifact, wavelength[::2], context[::2])


@pytest.mark.parametrize("mode", ["descending", "duplicate"])
def test_rejects_non_unique_or_unordered_wavelengths(artifact: BridgeArtifact, mode: str) -> None:
    wavelength, context = _valid_context(artifact)
    if mode == "descending":
        wavelength = wavelength[::-1]
        context = context[::-1]
    else:
        wavelength[50] = wavelength[49]
    with pytest.raises(ContractError, match="strictly increasing and unique"):
        predict_spectrum(artifact, wavelength, context)


@pytest.mark.parametrize("outside_wavelength", [396.0, 1004.0, 1052.0])
def test_rejects_bands_outside_measured_context(
    artifact: BridgeArtifact, outside_wavelength: float
) -> None:
    wavelength, context = _valid_context(artifact)
    if outside_wavelength < wavelength[0]:
        wavelength = np.insert(wavelength, 0, outside_wavelength)
        context = np.insert(context, 0, context[0])
    else:
        wavelength = np.append(wavelength, outside_wavelength)
        context = np.append(context, context[-1])
    with pytest.raises(ContractError, match="target/SWIR values are forbidden"):
        predict_spectrum(artifact, wavelength, context)


def test_rejects_late_start_early_end_and_large_gap(artifact: BridgeArtifact) -> None:
    wavelength, context = _valid_context(artifact)
    with pytest.raises(ContractError, match="starts too late"):
        predict_spectrum(artifact, wavelength[6:], context[6:])
    with pytest.raises(ContractError, match="ends too early"):
        predict_spectrum(artifact, wavelength[:-6], context[:-6])
    keep = np.ones(len(wavelength), dtype=bool)
    keep[70:72] = False
    with pytest.raises(ContractError, match="gap exceeds"):
        predict_spectrum(artifact, wavelength[keep], context[keep])


def test_rejects_clipping_candidates_and_percent_scaled_input(artifact: BridgeArtifact) -> None:
    wavelength, context = _valid_context(artifact)
    above = context.copy()
    above[20] = 1.01
    below = context.copy()
    below[20] = -0.051
    with pytest.raises(ContractError, match=r"inside \[-0.05, 1\] without clipping"):
        predict_spectrum(artifact, wavelength, above)
    with pytest.raises(ContractError, match=r"inside \[-0.05, 1\] without clipping"):
        predict_spectrum(artifact, wavelength, below)
    with pytest.raises(ContractError, match="percent-scaled"):
        predict_spectrum(artifact, wavelength, context * 100.0)
