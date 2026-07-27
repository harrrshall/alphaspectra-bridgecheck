"""Fail-closed paired VNIR/SWIR feasibility audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from .artifact import BridgeArtifact
from .predict import ContractError, predict_spectrum, validate_context


DELTA_BINS = ((50.0, 100.0), (100.0, 200.0), (200.0, 400.0), (400.0, 700.0), (700.0, 1000.0), (1000.0, float("inf")))


@dataclass(frozen=True)
class PairedSpectrum:
    sample_id: str
    group_id: str
    context_wavelength_nm: np.ndarray
    context_reflectance: np.ndarray
    target_wavelength_nm: np.ndarray
    target_reflectance: np.ndarray


def _validate_target(sample: PairedSpectrum) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.asarray(sample.target_wavelength_nm, dtype=np.float64)
    values = np.asarray(sample.target_reflectance, dtype=np.float64)
    if wavelength.ndim != 1 or values.ndim != 1 or wavelength.shape != values.shape:
        raise ContractError(f"{sample.sample_id}: target arrays must be equal-length and one-dimensional")
    if len(wavelength) == 0 or not np.isfinite(wavelength).all() or not np.isfinite(values).all():
        raise ContractError(f"{sample.sample_id}: measured targets must be non-empty and finite")
    if not (np.diff(wavelength) > 0).all():
        raise ContractError(f"{sample.sample_id}: target wavelengths must be ordered and unique")
    if wavelength[0] < 1052.0 or wavelength[-1] > 2400.0:
        raise ContractError(f"{sample.sample_id}: target wavelengths must remain inside 1052–2400 nm")
    if (values < -0.05).any() or (values > 1.0).any():
        raise ContractError(f"{sample.sample_id}: measured targets violate [-0.05, 1] without clipping")
    return wavelength, values


def _group_means(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups.astype(str))
    return unique, np.asarray([values[groups == group].mean() for group in unique], dtype=np.float64)


def _paired_bootstrap(
    comparator_error: np.ndarray,
    candidate_error: np.ndarray,
    groups: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    unique, comparator = _group_means(comparator_error, groups)
    unique_b, candidate = _group_means(candidate_error, groups)
    if not np.array_equal(unique, unique_b):
        raise RuntimeError("paired group sets differ")
    difference = comparator - candidate
    rng = np.random.default_rng(seed)
    draws = rng.choice(difference, size=(repeats, len(difference)), replace=True).mean(axis=1)
    return {
        "groups": int(len(unique)),
        "mean_comparator_minus_candidate": float(difference.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "wins": int((difference > 0).sum()),
        "ties": int((difference == 0).sum()),
    }


def _metric_summary(values: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    _, grouped = _group_means(values, groups)
    return {"row_mean_mae": float(values.mean()), "equal_group_mae": float(grouped.mean())}


def _records_hash(samples: Iterable[PairedSpectrum]) -> str:
    records = [
        {
            "sample_id": row.sample_id,
            "group_id": row.group_id,
            "context_wavelength_nm": np.asarray(row.context_wavelength_nm, dtype=float).tolist(),
            "context_reflectance": np.asarray(row.context_reflectance, dtype=float).tolist(),
            "target_wavelength_nm": np.asarray(row.target_wavelength_nm, dtype=float).tolist(),
            "target_reflectance": np.asarray(row.target_reflectance, dtype=float).tolist(),
        }
        for row in samples
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def audit_paired_spectra(
    artifact: BridgeArtifact,
    samples: list[PairedSpectrum],
    *,
    bootstrap_repeats: int = 10_000,
    bootstrap_seed: int = 2026072703,
    control_seed: int = 2026072701,
) -> dict[str, Any]:
    if not 2 <= len(samples) <= 10_000:
        raise ContractError("audit requires between 2 and 10,000 paired samples")
    if (
        isinstance(bootstrap_repeats, bool)
        or not isinstance(bootstrap_repeats, (int, np.integer))
        or not 100 <= int(bootstrap_repeats) <= 100_000
    ):
        raise ContractError("bootstrap_repeats must be an integer between 100 and 100000")
    sample_ids = [str(row.sample_id) for row in samples]
    if any(not value.strip() for value in sample_ids):
        raise ContractError("sample_id values must be non-empty")
    if any(value != value.strip() or len(value) > 200 for value in sample_ids):
        raise ContractError("sample_id values must be trimmed and at most 200 characters")
    if len(set(sample_ids)) != len(sample_ids):
        raise ContractError("sample_id values must be unique")
    group_ids = [str(row.group_id) for row in samples]
    if any(not value.strip() for value in group_ids):
        raise ContractError("group_id values must be non-empty")
    if any(value != value.strip() or len(value) > 200 for value in group_ids):
        raise ContractError("group_id values must be trimmed and at most 200 characters")
    groups = np.asarray(group_ids, dtype=str)
    if len(np.unique(groups)) < 2:
        raise ContractError("audit requires at least two independent biological groups")

    candidate_error: list[float] = []
    naive_error: list[float] = []
    edge_error: list[float] = []
    shuffled_error: list[float] = []
    blank_error: list[float] = []
    bin_errors: dict[str, dict[str, list[float]]] = {}
    rng = np.random.default_rng(control_seed)

    for sample in samples:
        context_wl, context = validate_context(
            sample.context_wavelength_nm,
            sample.context_reflectance,
            artifact.manifest["input_contract"],
        )
        target_wl, target = _validate_target(sample)
        result = predict_spectrum(artifact, context_wl, context)
        predicted = np.interp(target_wl, result.wavelengths_nm, result.reflectance)
        naive = np.full_like(target, context.mean())
        edge_values = context[context_wl >= 900.0]
        if len(edge_values) == 0:
            raise ContractError(f"{sample.sample_id}: no context band at or above 900 nm")
        edge = np.full_like(target, edge_values.mean())

        permutation = rng.permutation(len(context))
        shuffled_result = predict_spectrum(artifact, context_wl, context[permutation])
        blank_result = predict_spectrum(artifact, context_wl, np.zeros_like(context))
        shuffled = np.interp(target_wl, shuffled_result.wavelengths_nm, shuffled_result.reflectance)
        blank = np.interp(target_wl, blank_result.wavelengths_nm, blank_result.reflectance)

        candidate_error.append(float(np.abs(predicted - target).mean()))
        naive_error.append(float(np.abs(naive - target).mean()))
        edge_error.append(float(np.abs(edge - target).mean()))
        shuffled_error.append(float(np.abs(shuffled - target).mean()))
        blank_error.append(float(np.abs(blank - target).mean()))

        delta = target_wl - 1000.0
        for lo, hi in DELTA_BINS:
            selected = (delta > lo) & (delta <= hi)
            label = f"({lo:g},{'inf' if np.isinf(hi) else f'{hi:g}'}]"
            if not selected.any():
                continue
            row = bin_errors.setdefault(label, {"candidate": [], "naive": [], "edge": [], "groups": []})
            row["candidate"].append(float(np.abs(predicted[selected] - target[selected]).mean()))
            row["naive"].append(float(np.abs(naive[selected] - target[selected]).mean()))
            row["edge"].append(float(np.abs(edge[selected] - target[selected]).mean()))
            row["groups"].append(str(sample.group_id))

    candidate_array = np.asarray(candidate_error)
    naive_array = np.asarray(naive_error)
    edge_array = np.asarray(edge_error)
    shuffled_array = np.asarray(shuffled_error)
    blank_array = np.asarray(blank_error)
    naive_bootstrap = _paired_bootstrap(
        naive_array, candidate_array, groups, repeats=bootstrap_repeats, seed=bootstrap_seed
    )
    edge_bootstrap = _paired_bootstrap(
        edge_array, candidate_array, groups, repeats=bootstrap_repeats, seed=bootstrap_seed + 1
    )

    bin_report: dict[str, Any] = {}
    required_labels = [f"({lo:g},{'inf' if np.isinf(hi) else f'{hi:g}'}]" for lo, hi in DELTA_BINS]
    for label in required_labels:
        if label not in bin_errors:
            bin_report[label] = {"present": False}
            continue
        values = bin_errors[label]
        bin_groups = np.asarray(values["groups"], dtype=str)
        _, cand = _group_means(np.asarray(values["candidate"]), bin_groups)
        _, naive = _group_means(np.asarray(values["naive"]), bin_groups)
        _, edge = _group_means(np.asarray(values["edge"]), bin_groups)
        bin_report[label] = {
            "present": True,
            "groups": int(len(cand)),
            "candidate_mae": float(cand.mean()),
            "naive_mae": float(naive.mean()),
            "edge_mae": float(edge.mean()),
            "naive_minus_candidate": float((naive - cand).mean()),
            "edge_minus_candidate": float((edge - cand).mean()),
        }

    checks = {
        "all_inputs_valid_without_clipping": True,
        "at_least_two_independent_groups": len(np.unique(groups)) >= 2,
        "candidate_beats_naive_ci95_lower_above_zero": naive_bootstrap["ci95"][0] > 0,
        "candidate_beats_edge_ci95_lower_above_zero": edge_bootstrap["ci95"][0] > 0,
        "all_six_delta_bins_present": all(row.get("present", False) for row in bin_report.values()),
        "all_delta_bins_positive_vs_both": all(
            row.get("present", False)
            and row.get("naive_minus_candidate", 0.0) > 0
            and row.get("edge_minus_candidate", 0.0) > 0
            for row in bin_report.values()
        ),
        "shuffled_context_worse": _metric_summary(shuffled_array, groups)["equal_group_mae"]
        > _metric_summary(candidate_array, groups)["equal_group_mae"],
        "blank_context_worse": _metric_summary(blank_array, groups)["equal_group_mae"]
        > _metric_summary(candidate_array, groups)["equal_group_mae"],
    }
    passed = all(checks.values())
    return {
        "schema_version": "1.0",
        "model_id": artifact.manifest["model_id"],
        "input_sha256": _records_hash(samples),
        "status": "SUPPORTED_FOR_RECONSTRUCTION_RESEARCH" if passed else "NOT_SUPPORTED",
        "claim_ceiling": "reconstruction_on_this_paired_dataset_only_no_downstream_or_measurement_equivalence_claim",
        "population": {"samples": len(samples), "biological_groups": int(len(np.unique(groups)))},
        "metrics": {
            "candidate": _metric_summary(candidate_array, groups),
            "all_context_mean": _metric_summary(naive_array, groups),
            "context_edge_mean": _metric_summary(edge_array, groups),
            "context_shuffle": _metric_summary(shuffled_array, groups),
            "blank_context": _metric_summary(blank_array, groups),
        },
        "paired_inference": {"all_context_mean": naive_bootstrap, "context_edge_mean": edge_bootstrap},
        "delta_bins": bin_report,
        "checks": checks,
        "warnings": [
            "A reconstruction pass does not establish downstream biological value.",
            "Measured SWIR must independently improve the held endpoint before sensor substitution.",
            "No generated value is a measurement or calibrated uncertainty interval.",
        ],
    }
