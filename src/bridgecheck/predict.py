"""Context-only physics-state retrieval for BridgeCheck."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np

from .artifact import BridgeArtifact


class ContractError(ValueError):
    """Raised when an input is not a supported absolute-reflectance VNIR spectrum."""


def _input_hash(wavelengths_nm: np.ndarray, reflectance: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (wavelengths_nm, reflectance):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_context(
    wavelengths_nm: np.ndarray | list[float],
    reflectance: np.ndarray | list[float],
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelengths_nm, dtype=np.float64)
    values = np.asarray(reflectance, dtype=np.float64)
    if wavelength.ndim != 1 or values.ndim != 1 or wavelength.shape != values.shape:
        raise ContractError("wavelength_nm and reflectance must be equal-length one-dimensional arrays")
    if len(wavelength) < int(contract["minimum_bands"]):
        raise ContractError(f"at least {contract['minimum_bands']} VNIR bands are required")
    if not np.isfinite(wavelength).all() or not np.isfinite(values).all():
        raise ContractError("wavelengths and reflectance must all be finite")
    if not (np.diff(wavelength) > 0).all():
        raise ContractError("wavelengths must be strictly increasing and unique")
    context_lo, context_hi = (float(x) for x in contract["absolute_context_range_nm"])
    if float(wavelength[0]) < context_lo or float(wavelength[-1]) > context_hi:
        raise ContractError(
            f"predict accepts measured context only inside {context_lo:g}–{context_hi:g} nm; "
            "target/SWIR values are forbidden"
        )
    if float(wavelength[0]) > float(contract["start_at_or_below_nm"]):
        raise ContractError("VNIR coverage starts too late for the V1 support contract")
    if float(wavelength[-1]) < float(contract["end_at_or_above_nm"]):
        raise ContractError("VNIR coverage ends too early for the V1 support contract")
    if float(np.diff(wavelength).max()) > float(contract["maximum_gap_nm"]):
        raise ContractError("VNIR wavelength gap exceeds the V1 support contract")
    lo, hi = (float(x) for x in contract["reflectance_range"])
    if (values < lo).any() or (values > hi).any():
        if float(np.nanmax(values)) > 1.5:
            raise ContractError("reflectance appears to be percent-scaled; provide decimal fractions")
        raise ContractError(f"reflectance must remain inside [{lo:g}, {hi:g}] without clipping")
    return wavelength, values


def interpolate_candidate_context(
    artifact: BridgeArtifact, wavelengths_nm: np.ndarray
) -> np.ndarray:
    grid = artifact.wavelengths_nm
    if wavelengths_nm[0] < grid[0] or wavelengths_nm[-1] > grid[-1]:
        raise ContractError("candidate interpolation would require extrapolation")
    position = (wavelengths_nm - grid[0]) / (grid[1] - grid[0])
    left = np.floor(position).astype(np.int64)
    right = np.ceil(position).astype(np.int64)
    fraction = position - left
    return artifact.bank[:, left] * (1.0 - fraction)[None, :] + artifact.bank[:, right] * fraction[None, :]


@dataclass(frozen=True)
class BridgePrediction:
    model_id: str
    input_sha256: str
    wavelengths_nm: np.ndarray
    reflectance: np.ndarray
    nearest_candidate: int
    context_fit_rmse: float
    neighbor_min: np.ndarray
    neighbor_max: np.ndarray
    neighbor_count: int
    support_tier: str
    claim_status: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        count = len(self.wavelengths_nm)
        return {
            "model_id": self.model_id,
            "input_sha256": self.input_sha256,
            "derived": {
                "origin": "model_derived",
                "wavelength_nm": self.wavelengths_nm.tolist(),
                "reflectance": self.reflectance.tolist(),
                "observed_band_mask": [False] * count,
                "neighbor_envelope": {
                    "kind": "descriptive_not_calibrated_uncertainty",
                    "minimum": self.neighbor_min.tolist(),
                    "maximum": self.neighbor_max.tolist(),
                    "neighbors": self.neighbor_count,
                },
            },
            "retrieval": {
                "nearest_candidate": f"state-{self.nearest_candidate:04d}",
                "context_fit_rmse": self.context_fit_rmse,
                "support_tier": self.support_tier,
                "support_metric": "reference_distance_descriptive_only",
                "calibrated_prediction_interval": None,
            },
            "claim_status": self.claim_status,
            "warnings": list(self.warnings),
        }


def predict_spectrum(
    artifact: BridgeArtifact,
    wavelengths_nm: np.ndarray | list[float],
    reflectance: np.ndarray | list[float],
    *,
    neighbors: int = 5,
) -> BridgePrediction:
    wavelength, values = validate_context(
        wavelengths_nm, reflectance, artifact.manifest["input_contract"]
    )
    if not 1 <= neighbors <= 10:
        raise ContractError("neighbors must be between 1 and 10")
    candidates = interpolate_candidate_context(artifact, wavelength)
    residual = candidates - values[None, :]
    rmse = np.sqrt(np.mean(residual * residual, axis=1))
    order = np.argsort(rmse, kind="stable")[:neighbors]
    chosen = int(order[0])
    targets = artifact.bank[order][:, artifact.target_mask]
    prediction = targets[0].copy()
    neighbor_min = targets.min(axis=0)
    neighbor_max = targets.max(axis=0)
    thresholds = artifact.manifest["support_reference"]["context_rmse_quantiles"]
    if float(rmse[chosen]) <= float(thresholds["q95"]):
        support = "WITHIN_REFERENCE_Q95"
    elif float(rmse[chosen]) <= float(thresholds["q99"]):
        support = "REFERENCE_TAIL_Q95_Q99"
    else:
        support = "OUTSIDE_REFERENCE_Q99"
    warnings = [
        "Generated reflectance is model-derived, not measured SWIR.",
        "No calibrated prediction interval is available.",
        "Not validated for diagnosis, treatment decisions, or automatic model input.",
    ]
    if support != "WITHIN_REFERENCE_Q95":
        warnings.append("Input is outside the central reference-fit distribution; paired measurement audit is required.")
    return BridgePrediction(
        model_id=artifact.manifest["model_id"],
        input_sha256=_input_hash(wavelength, values),
        wavelengths_nm=artifact.wavelengths_nm[artifact.target_mask].copy(),
        reflectance=prediction,
        nearest_candidate=chosen,
        context_fit_rmse=float(rmse[chosen]),
        neighbor_min=neighbor_min,
        neighbor_max=neighbor_max,
        neighbor_count=len(order),
        support_tier=support,
        claim_status="CANDIDATE_ONLY_UNVALIDATED",
        warnings=tuple(warnings),
    )
