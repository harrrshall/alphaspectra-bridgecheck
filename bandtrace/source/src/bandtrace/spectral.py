"""Per-channel routed spectral-response support comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bundle import Band, Bundle
from .constants import S0, S1, S2, S3

_AVAILABLE_MASS_MIN = 0.99
_L1_MAX = 0.05
_ENDPOINT_PEAK_FRACTION_MAX = 1e-4
_MAX_GAUSSIAN_UNIQUE_KNOTS = 250_000
_MAX_GAUSSIAN_COMPONENT_EVALUATIONS = 50_000_000
_MAX_FULL_SRF_UNIQUE_KNOTS = 250_000
_MAX_FULL_SRF_COMPONENT_EVALUATIONS = 50_000_000
_GAUSSIAN_CONVERGENCE_TOLERANCE = 1e-6


class _NumericalUnresolved(RuntimeError):
    pass


@dataclass(frozen=True)
class SpectralResult:
    state: str
    channels: list[dict[str, Any]]
    faults: list[dict[str, Any]]
    method: str


@dataclass
class _GaussianBudget:
    component_evaluations: int = 0

    def charge(self, components: int, points: int) -> None:
        self.component_evaluations += int(components) * int(points)
        if self.component_evaluations > _MAX_GAUSSIAN_COMPONENT_EVALUATIONS:
            raise _NumericalUnresolved("Gaussian component-point evaluation budget exceeded")


@dataclass
class _FullSrfBudget:
    component_evaluations: int = 0

    def charge(self, components: int, points: int) -> None:
        self.component_evaluations += int(components) * int(points)
        if self.component_evaluations > _MAX_FULL_SRF_COMPONENT_EVALUATIONS:
            raise _NumericalUnresolved(
                "full-SRF interpolation component-point evaluation budget exceeded"
            )


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    function = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(function(values, grid))


def _fault(code: str, detail: str) -> dict[str, Any]:
    return {"code": code, "severity": 1, "axis": "spectral_support", "detail": detail}


def _piecewise_moment(grid: np.ndarray, response: np.ndarray) -> float:
    dx = np.diff(grid)
    x0, x1 = grid[:-1], grid[1:]
    y0, y1 = response[:-1], response[1:]
    return float(np.sum(dx / 6.0 * ((2.0 * x0 + x1) * y0 + (x0 + 2.0 * x1) * y1)))


def _half_max_component(grid: np.ndarray, response: np.ndarray) -> tuple[float, float] | None:
    half = 0.5 * float(np.max(response))
    shifted = response - half
    intervals: list[tuple[float, float]] = []
    for index in range(grid.size - 1):
        x0, x1 = float(grid[index]), float(grid[index + 1])
        y0, y1 = float(shifted[index]), float(shifted[index + 1])
        if y0 >= 0.0 and y1 >= 0.0:
            intervals.append((x0, x1))
        elif y0 >= 0.0 and y1 < 0.0:
            crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            intervals.append((x0, crossing))
        elif y0 < 0.0 and y1 >= 0.0:
            crossing = x0 - y0 * (x1 - x0) / (y1 - y0)
            intervals.append((crossing, x1))
    if not intervals:
        return None
    # Merge closed intervals, retaining point-only intervals. An isolated knot
    # exactly at half maximum is therefore a real singleton component rather
    # than being lost by midpoint sampling.
    components: list[tuple[float, float]] = []
    for start, end in intervals:
        if components and start <= components[-1][1]:
            previous_start, previous_end = components[-1]
            components[-1] = (previous_start, max(previous_end, end))
        else:
            components.append((start, end))
    if len(components) != 1 or components[0][1] <= components[0][0]:
        return None
    return components[0]


def _validate_srf_metadata(band: Band) -> tuple[bool, str, dict[str, float] | None]:
    wavelength = band.srf_wavelength_nm
    response = band.srf_response
    if wavelength is None or response is None:
        return False, "missing", None
    if wavelength.ndim != 1 or response.ndim != 1 or wavelength.size < 4:
        return False, "SRF requires at least four knots", None
    if wavelength.size != response.size or not np.isfinite(wavelength).all() or not np.isfinite(response).all():
        return False, "SRF arrays are mismatched or non-finite", None
    if np.any(np.diff(wavelength) <= 0) or np.any(response < 0):
        return False, "SRF wavelength must increase strictly and response must be non-negative", None
    peak = float(np.max(response))
    if peak <= 0:
        return False, "SRF peak must be positive", None
    if float(response[0]) > _ENDPOINT_PEAK_FRACTION_MAX * peak or float(response[-1]) > _ENDPOINT_PEAK_FRACTION_MAX * peak:
        return False, "SRF supplied tails do not reach the endpoint-response threshold", None
    area = _trapz(response, wavelength)
    if area <= 0:
        return False, "SRF area must be positive", None
    component = _half_max_component(wavelength, response)
    if component is None:
        return False, "half-maximum crossings do not form exactly one connected component", None
    derived_fwhm = component[1] - component[0]
    centroid = _piecewise_moment(wavelength, response) / area
    assert band.fwhm_nm is not None
    fwhm_tolerance = max(0.1, 0.10 * band.fwhm_nm)
    center_tolerance = max(0.25, 0.25 * band.fwhm_nm)
    if abs(derived_fwhm - band.fwhm_nm) > fwhm_tolerance:
        return False, "declared FWHM is inconsistent with the supplied SRF", None
    if abs(centroid - band.center_nm) > center_tolerance:
        return False, "declared center is inconsistent with the supplied SRF centroid", None
    return True, "valid", {"derived_fwhm_nm": derived_fwhm, "derived_center_nm": centroid}


def _unit_native(band: Band) -> tuple[np.ndarray, np.ndarray]:
    assert band.srf_wavelength_nm is not None and band.srf_response is not None
    area = _trapz(band.srf_response, band.srf_wavelength_nm)
    return band.srf_wavelength_nm, band.srf_response / area


def _interval_curve_values(
    native_grid: np.ndarray, native_response: np.ndarray, union_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate interval-side limits for a native curve that is zero outside."""

    left_grid = union_grid[:-1]
    right_grid = union_grid[1:]
    # Every native endpoint is inserted into ``union_grid`` by the callers, so
    # an interval is inside exactly when both of its endpoints are contained.
    # Avoid midpoint classification: for adjacent binary64 knots their mean
    # can round to an endpoint and misclassify a zero-outside interval.
    inside = (left_grid >= native_grid[0]) & (right_grid <= native_grid[-1])
    left = np.zeros_like(left_grid)
    right = np.zeros_like(right_grid)
    samples = np.interp(union_grid, native_grid, native_response)
    left[inside] = samples[:-1][inside]
    right[inside] = samples[1:][inside]
    return left, right


def _interval_area(
    grid: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    areas = np.diff(grid) * 0.5 * (left + right)
    return float(np.sum(areas if mask is None else areas[mask]))


def _interval_moment(
    grid: np.ndarray, left: np.ndarray, right: np.ndarray
) -> float:
    x0, x1 = grid[:-1], grid[1:]
    return float(
        np.sum(
            np.diff(grid)
            / 6.0
            * ((2.0 * x0 + x1) * left + (x0 + 2.0 * x1) * right)
        )
    )


def _interval_absolute_linear_difference(
    grid: np.ndarray,
    left_difference: np.ndarray,
    right_difference: np.ndarray,
) -> tuple[float, int]:
    """Integrate |linear difference| exactly and count inserted sign roots."""

    width = np.diff(grid)
    left_abs = np.abs(left_difference)
    right_abs = np.abs(right_difference)
    crosses = left_difference * right_difference < 0.0
    integral = width * 0.5 * (left_abs + right_abs)
    denominator = left_abs[crosses] + right_abs[crosses]
    integral[crosses] = (
        width[crosses]
        * (left_abs[crosses] ** 2 + right_abs[crosses] ** 2)
        / (2.0 * denominator)
    )
    return float(np.sum(integral)), int(np.sum(crosses))


def _full_srf_channel(
    bundle: Bundle,
    channel_index: int,
    derived_training_fwhm: float,
    budget: _FullSrfBudget,
) -> dict[str, Any]:
    channel = bundle.model.channels[channel_index]
    weights = bundle.route.canonical_matrix[channel_index]
    active = np.flatnonzero(weights > 0.0)
    assert channel.srf_wavelength_nm is not None
    grids = [channel.srf_wavelength_nm]
    normalized_targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for raw_index in active:
        index = int(raw_index)
        target = bundle.sensor.bands[index]
        native = _unit_native(target)
        normalized_targets[index] = native
        grids.append(native[0])
    grids.append(np.asarray(bundle.model.support_range_nm, dtype=np.float64))
    grid = np.unique(np.concatenate(grids)).astype(np.float64)
    if grid.size > _MAX_FULL_SRF_UNIQUE_KNOTS:
        raise _NumericalUnresolved("full-SRF unique-knot budget exceeded")
    evaluations_before = budget.component_evaluations
    budget.charge(1 + active.size, grid.size)
    train_wl, train_response = _unit_native(channel)
    training_left, training_right = _interval_curve_values(
        train_wl, train_response, grid
    )
    effective_left = np.zeros(grid.size - 1, dtype=np.float64)
    effective_right = np.zeros(grid.size - 1, dtype=np.float64)
    for raw_index in active:
        index = int(raw_index)
        wavelength, response = normalized_targets[index]
        target_left, target_right = _interval_curve_values(
            wavelength, response, grid
        )
        effective_left += float(weights[index]) * target_left
        effective_right += float(weights[index]) * target_right
    effective_total = _interval_area(grid, effective_left, effective_right)
    training_total = _interval_area(grid, training_left, training_right)
    if effective_total <= 0 or training_total <= 0:
        raise _NumericalUnresolved("non-positive full-SRF comparison area")
    support_mask = (grid[:-1] >= bundle.model.support_range_nm[0]) & (
        grid[1:] <= bundle.model.support_range_nm[1]
    )
    available_mass = _interval_area(
        grid, effective_left, effective_right, support_mask
    ) / effective_total
    normalized_training_left = training_left / training_total
    normalized_training_right = training_right / training_total
    normalized_effective_left = effective_left / effective_total
    normalized_effective_right = effective_right / effective_total
    l1, inserted_roots = _interval_absolute_linear_difference(
        grid,
        normalized_effective_left - normalized_training_left,
        normalized_effective_right - normalized_training_right,
    )
    root_grid_size = int(grid.size + inserted_roots)
    if root_grid_size > _MAX_FULL_SRF_UNIQUE_KNOTS:
        raise _NumericalUnresolved(
            "full-SRF unique-knot budget exceeded after difference-root insertion"
        )
    budget.charge(2, root_grid_size)
    training_center = _interval_moment(
        grid, normalized_training_left, normalized_training_right
    )
    effective_center = _interval_moment(
        grid, normalized_effective_left, normalized_effective_right
    )
    center_shift = abs(effective_center - training_center)
    center_tolerance = max(0.25, 0.25 * derived_training_fwhm)
    passed = (
        available_mass >= _AVAILABLE_MASS_MIN
        and l1 <= _L1_MAX
        and center_shift <= center_tolerance
    )
    return {
        "model_channel_id": channel.id,
        "paired_target_band_ids": [bundle.sensor.bands[int(index)].id for index in active],
        "method": "full_srf",
        "available_mass": available_mass,
        "normalized_l1": l1,
        "center_shift_nm": center_shift,
        "center_tolerance_nm": center_tolerance,
        "derived_training_fwhm_nm": derived_training_fwhm,
        "unique_union_knots": int(grid.size),
        "difference_root_grid_knots": root_grid_size,
        "difference_roots_inserted": inserted_roots,
        "interpolation_component_point_evaluations": (
            budget.component_evaluations - evaluations_before
        ),
        "pass": passed,
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _gaussian_density(points: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((points - center) / sigma) ** 2) / (
        sigma * math.sqrt(2.0 * math.pi)
    )


def _gaussian_difference(
    points: np.ndarray,
    training: tuple[float, float],
    targets: list[tuple[float, float, float]],
    budget: _GaussianBudget,
) -> np.ndarray:
    budget.charge(1 + len(targets), points.size)
    result = -_gaussian_density(points, training[0], training[1])
    for weight, center, sigma in targets:
        result += weight * _gaussian_density(points, center, sigma)
    return result


def _gaussian_l1(
    training: tuple[float, float],
    targets: list[tuple[float, float, float]],
    step: float,
    budget: _GaussianBudget,
) -> float:
    components = [training] + [(center, sigma) for _, center, sigma in targets]
    q = np.arange(-12.0, 12.0 + step * 0.5, step, dtype=np.float64)
    grid = np.unique(
        np.concatenate([center + sigma * q for center, sigma in components])
    )
    if grid.size > _MAX_GAUSSIAN_UNIQUE_KNOTS:
        raise _NumericalUnresolved("Gaussian unique-knot budget exceeded")
    difference = _gaussian_difference(grid, training, targets, budget)
    roots: list[float] = []
    brackets = np.flatnonzero(difference[:-1] * difference[1:] < 0.0)
    for raw_index in brackets:
        index = int(raw_index)
        left, right = float(grid[index]), float(grid[index + 1])
        left_value = float(difference[index])
        for _ in range(60):
            midpoint = 0.5 * (left + right)
            middle_value = float(
                _gaussian_difference(
                    np.asarray([midpoint], dtype=np.float64), training, targets, budget
                )[0]
            )
            if left_value * middle_value <= 0.0:
                right = midpoint
            else:
                left = midpoint
                left_value = middle_value
        roots.append(0.5 * (left + right))
    if roots:
        grid = np.unique(np.concatenate([grid, np.asarray(roots, dtype=np.float64)]))
        if grid.size > _MAX_GAUSSIAN_UNIQUE_KNOTS:
            raise _NumericalUnresolved("Gaussian unique-knot budget exceeded after root insertion")
        difference = _gaussian_difference(grid, training, targets, budget)
    return _trapz(np.abs(difference), grid)


def _gaussian_channel(
    bundle: Bundle,
    channel_index: int,
    budget: _GaussianBudget,
) -> dict[str, Any]:
    channel = bundle.model.channels[channel_index]
    assert channel.fwhm_nm is not None
    training_sigma = channel.fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    weights = bundle.route.canonical_matrix[channel_index]
    targets: list[tuple[float, float, float]] = []
    total_weight = 0.0
    for raw_index in np.flatnonzero(weights > 0.0):
        index = int(raw_index)
        target = bundle.sensor.bands[index]
        assert target.fwhm_nm is not None
        weight = float(weights[index])
        sigma = target.fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        targets.append((weight, target.center_nm, sigma))
        total_weight += weight
    if total_weight <= 0:
        raise _NumericalUnresolved("non-positive Gaussian routed mass")
    low, high = bundle.model.support_range_nm
    available_mass = sum(
        weight
        * (_normal_cdf((high - center) / sigma) - _normal_cdf((low - center) / sigma))
        for weight, center, sigma in targets
    ) / total_weight
    effective_center = sum(weight * center for weight, center, _ in targets) / total_weight
    center_shift = abs(effective_center - channel.center_nm)
    center_tolerance = max(0.25, 0.25 * channel.fwhm_nm)
    normalized_targets = [
        (weight / total_weight, center, sigma) for weight, center, sigma in targets
    ]
    training = (channel.center_nm, training_sigma)
    l1_coarse = _gaussian_l1(training, normalized_targets, 0.125, budget)
    l1_check = _gaussian_l1(training, normalized_targets, 0.0625, budget)
    if abs(l1_check - l1_coarse) <= _GAUSSIAN_CONVERGENCE_TOLERANCE:
        l1 = l1_check
    else:
        l1_final = _gaussian_l1(training, normalized_targets, 0.03125, budget)
        if abs(l1_final - l1_check) > _GAUSSIAN_CONVERGENCE_TOLERANCE:
            raise _NumericalUnresolved("Gaussian L1 refinement did not converge")
        l1 = l1_final
    passed = (
        available_mass >= _AVAILABLE_MASS_MIN
        and l1 <= _L1_MAX
        and center_shift <= center_tolerance
    )
    return {
        "model_channel_id": channel.id,
        "paired_target_band_ids": [
            bundle.sensor.bands[int(index)].id
            for index in np.flatnonzero(weights > 0.0)
        ],
        "method": "gaussian_fwhm",
        "available_mass": available_mass,
        "normalized_l1": l1,
        "center_shift_nm": center_shift,
        "center_tolerance_nm": center_tolerance,
        "pass": passed,
    }


def evaluate_spectral_support(bundle: Bundle, *, route_eligible: bool) -> SpectralResult:
    faults: list[dict[str, Any]] = []
    radiometric_mismatch = (
        bundle.model.radiometric_quantity != bundle.sensor.radiometric_quantity
    )
    if radiometric_mismatch:
        faults.append(
            _fault(
                "radiometric_quantity_mismatch",
                "model and target sensor radiometric quantities differ",
            )
        )
    support_missing = not bundle.model.support_assertion
    if support_missing:
        faults.append(_fault("support_declaration_missing", "supplier support assertion is missing"))

    all_bands = (*bundle.model.channels, *bundle.sensor.bands)
    model_metadata: dict[str, dict[str, float]] = {}
    target_metadata: dict[str, dict[str, float]] = {}
    invalid: list[str] = []
    inconsistent = False
    for role, bands in (("model", bundle.model.channels), ("target", bundle.sensor.bands)):
        for index, band in enumerate(bands):
            if band.srf_wavelength_nm is None and band.srf_response is None:
                continue
            valid, reason, derived = _validate_srf_metadata(band)
            if not valid:
                invalid.append(f"{band.id}: {reason}")
                inconsistent = inconsistent or "inconsistent" in reason
            elif derived is not None:
                destination = model_metadata if role == "model" else target_metadata
                destination[f"{index}:{band.id}"] = derived
    if invalid:
        code = "srf_metadata_inconsistent" if inconsistent else "invalid_present_srf"
        faults.append(_fault(code, "; ".join(invalid)))
        return SpectralResult(S0, [], faults, "invalid_present_srf")
    if radiometric_mismatch or support_missing:
        return SpectralResult(S0, [], faults, "unresolved")
    if not route_eligible:
        return SpectralResult(S0, [], faults, "unresolved_route")

    all_have_srf = all(
        band.srf_wavelength_nm is not None and band.srf_response is not None
        for band in all_bands
    )
    try:
        if all_have_srf:
            budget = _FullSrfBudget()
            channel_results = [
                _full_srf_channel(
                    bundle,
                    index,
                    model_metadata[f"{index}:{bundle.model.channels[index].id}"][
                        "derived_fwhm_nm"
                    ],
                    budget,
                )
                for index in range(len(bundle.model.channels))
            ]
            method = "full_srf"
            passing_state = S3
        else:
            budget = _GaussianBudget()
            channel_results = [
                _gaussian_channel(bundle, index, budget)
                for index in range(len(bundle.model.channels))
            ]
            method = "gaussian_fwhm"
            passing_state = S2
    except _NumericalUnresolved as error:
        faults.append(_fault("numerical_support_unresolved", str(error)))
        return SpectralResult(S0, [], faults, "numerical_unresolved")

    if all(bool(item["pass"]) for item in channel_results):
        return SpectralResult(passing_state, channel_results, faults, method)
    outside = [
        str(item["model_channel_id"])
        for item in channel_results
        if float(item["available_mass"]) < _AVAILABLE_MASS_MIN
    ]
    response_mismatch = [
        str(item["model_channel_id"])
        for item in channel_results
        if float(item["available_mass"]) >= _AVAILABLE_MASS_MIN
        and (
            float(item["normalized_l1"]) > _L1_MAX
            or float(item["center_shift_nm"]) > float(item["center_tolerance_nm"])
        )
    ]
    if outside:
        faults.append(
            _fault(
                "target_srf_outside_support",
                "routed effective response has insufficient mass inside support for "
                + ", ".join(outside),
            )
        )
    if response_mismatch:
        faults.append(
            _fault(
                "routed_response_mismatch",
                "in-support routed response fails L1/center agreement for "
                + ", ".join(response_mismatch),
            )
        )
    return SpectralResult(S1, channel_results, faults, method)
