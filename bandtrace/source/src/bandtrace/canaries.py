"""Frozen deterministic C0-C6 behavioral checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Iterator
from fractions import Fraction
from typing import Any

import numpy as np

from .adapters import Adapter, Invocation
from .bundle import Band, Bundle
from .canonical import (
    c3_rank_amplitudes,
    c4_permutation,
    c4_shift,
    derive_canary_seed,
    linear_quantile,
)
from .constants import (
    C1_CHUNK_MAX_FLOAT64_PROBE_BYTES,
    C1_CHUNK_MAX_ROWS,
    DEPENDENCE_ABSOLUTE_FLOOR,
    DEPENDENCE_REPLAY_MULTIPLIER,
    MINIMUM_PROBE_FRACTION,
    MUTATION_EXCITATION_FLOOR,
    NUMERIC_TOLERANCE,
    REPLAY_COUNT,
    REPLAY_JITTER_MAX,
    REQUIRED_ROUTE_WEIGHT_MIN,
    ROW_SUM_TOLERANCE,
    X0,
    X1,
    X2,
    X3,
)
from .errors import ExecutionError


@dataclass(frozen=True)
class CanaryResult:
    executable_state: str
    route_eligible: bool
    spectral_route_eligible: bool
    canaries: dict[str, dict[str, Any]]
    facts: dict[str, Any]
    faults: list[dict[str, Any]]
    arrays: dict[str, np.ndarray]


def _fault(code: str, axis: str, detail: str) -> dict[str, Any]:
    return {"code": code, "severity": 1, "axis": axis, "detail": detail}


def _metadata(bundle: Bundle) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    return (
        tuple(band.id for band in bundle.sensor.bands),
        np.asarray([band.center_nm for band in bundle.sensor.bands], dtype=np.float64),
        np.asarray([band.fwhm_nm for band in bundle.sensor.bands], dtype=np.float64),
    )


def _request(
    bundle: Bundle,
    probes: np.ndarray,
    *,
    ids: tuple[str, ...] | None = None,
    wavelength: np.ndarray | None = None,
    fwhm: np.ndarray | None = None,
) -> Invocation:
    default_ids, default_wavelength, default_fwhm = _metadata(bundle)
    return Invocation(
        probes=np.asarray(probes, dtype=np.float64),
        target_band_ids=default_ids if ids is None else ids,
        wavelength_nm=default_wavelength if wavelength is None else np.asarray(wavelength, dtype=np.float64),
        fwhm_nm=default_fwhm if fwhm is None else np.asarray(fwhm, dtype=np.float64),
    )


def expected_pre_core(bundle: Bundle, request: Invocation) -> np.ndarray:
    """Compute the declaration-derived tap without consulting the adapter."""

    sensor_ids = tuple(band.id for band in bundle.sensor.bands)
    positions = {identifier: index for index, identifier in enumerate(request.target_band_ids)}
    values = request.probes[:, [positions[identifier] for identifier in sensor_ids], ...]
    route = bundle.route.canonical_matrix
    offset = bundle.model.normalization_offset
    scale = bundle.model.normalization_scale
    if bundle.route.spatial_operation == "none":
        routed = np.einsum("nb,mb->nm", values, route, optimize=False)
        return np.asarray((routed - offset[None, :]) / scale[None, :], dtype=np.float64)
    # Mean is linear and the affine parameters are spatially constant. Reduce
    # raw target cells first so legal B=1/M=512 cubes cannot amplify into an
    # N*M*H*W intermediate.
    spatial_mean = np.mean(values, axis=(2, 3), dtype=np.float64)
    routed = np.einsum("nb,mb->nm", spatial_mean, route, optimize=False)
    return np.asarray(
        (routed - offset[None, :]) / scale[None, :], dtype=np.float64
    )


def _route_static(
    bundle: Bundle,
) -> tuple[bool, bool, list[dict[str, Any]], dict[str, Any]]:
    matrix = bundle.route.canonical_matrix
    faults: list[dict[str, Any]] = []
    eligible = True
    spectral_eligible = True
    row_sums = np.sum(matrix, axis=1)
    positive_counts = np.sum(matrix > 0.0, axis=1)
    if np.any(matrix < 0.0):
        eligible = False
        spectral_eligible = False
        faults.append(_fault("hidden_resampling_or_extrapolation", "route", "negative route weight"))
    if np.any(row_sums < 1.0 - ROW_SUM_TOLERANCE) or np.any(positive_counts == 0):
        eligible = False
        faults.append(_fault("dropped_band", "route", "route row lacks unit mass or provenance"))
    if np.any(positive_counts == 0):
        spectral_eligible = False
    if np.any(row_sums > 1.0 + ROW_SUM_TOLERANCE):
        eligible = False
        faults.append(
            _fault("hidden_resampling_or_extrapolation", "route", "route row exceeds unit mass")
        )
    if bundle.route.operation == "selection_or_permutation":
        positive = matrix[matrix > 0.0]
        if not (
            np.all(positive_counts == 1)
            and np.allclose(positive, 1.0, atol=ROW_SUM_TOLERANCE, rtol=0.0)
        ):
            eligible = False
            faults.append(
                _fault(
                    "hidden_resampling_or_extrapolation",
                    "route",
                    "selection route contains an undeclared mixture",
                )
            )
    if bundle.model.radiometric_quantity != bundle.sensor.radiometric_quantity:
        eligible = False
        spectral_eligible = False
        faults.append(
            _fault(
                "radiometric_quantity_mismatch",
                "static_contract",
                "model and target radiometric quantities differ",
            )
        )
    sensor_low = np.asarray(
        [band.valid_range[0] for band in bundle.sensor.bands], dtype=np.float64
    )
    sensor_high = np.asarray(
        [band.valid_range[1] for band in bundle.sensor.bands], dtype=np.float64
    )
    model_low, model_high = bundle.model.valid_range
    rational_matrix = [
        [Fraction.from_float(float(weight)) for weight in row] for row in matrix
    ]
    rational_sensor_low = [Fraction.from_float(float(value)) for value in sensor_low]
    rational_sensor_high = [Fraction.from_float(float(value)) for value in sensor_high]
    rational_model_low = Fraction.from_float(float(model_low))
    rational_model_high = Fraction.from_float(float(model_high))
    routed_low_exact = [
        sum(
            (weight * lower for weight, lower in zip(row, rational_sensor_low)),
            Fraction(0),
        )
        for row in rational_matrix
    ]
    routed_high_exact = [
        sum(
            (weight * upper for weight, upper in zip(row, rational_sensor_high)),
            Fraction(0),
        )
        for row in rational_matrix
    ]
    routed_low = np.asarray([float(value) for value in routed_low_exact], dtype=np.float64)
    routed_high = np.asarray([float(value) for value in routed_high_exact], dtype=np.float64)
    routed_domain_inside = all(
        lower >= rational_model_low and upper <= rational_model_high
        for lower, upper in zip(routed_low_exact, routed_high_exact)
    )
    maximum_violation_exact = max(
        [
            max(rational_model_low - lower, Fraction(0))
            for lower in routed_low_exact
        ]
        + [
            max(upper - rational_model_high, Fraction(0))
            for upper in routed_high_exact
        ]
    )
    maximum_violation = float(maximum_violation_exact)
    model_range_width = rational_model_high - rational_model_low
    if not routed_domain_inside:
        eligible = False
        faults.append(
            _fault(
                "routed_domain_outside_model_valid_range",
                "route",
                "the complete routed target raw interval is outside the model raw valid range",
            )
        )
    return eligible, spectral_eligible, faults, {
        "row_sums": row_sums.tolist(),
        "positive_weights_per_model_channel": positive_counts.tolist(),
        "operation": bundle.route.operation,
        "routed_raw_lower_by_model_channel": routed_low.tolist(),
        "routed_raw_upper_by_model_channel": routed_high.tolist(),
        "model_raw_valid_range": [model_low, model_high],
        "routed_raw_domain_inside_model_valid_range": routed_domain_inside,
        "routed_raw_domain_arithmetic": "EXACT_RATIONAL_OF_PARSED_BINARY64_INPUTS",
        "routed_raw_domain_absolute_tolerance": 0.0,
        "routed_raw_domain_inside_with_absolute_tolerance": routed_domain_inside,
        "routed_raw_domain_exactly_inside_model_valid_range": routed_domain_inside,
        "routed_raw_domain_maximum_absolute_violation": maximum_violation,
        "routed_raw_domain_maximum_violation_fraction_of_model_range": (
            float(maximum_violation_exact / model_range_width)
        ),
    }


def _neutral_values(bundle: Bundle) -> np.ndarray:
    neutral = np.asarray([band.neutral_value for band in bundle.sensor.bands], dtype=np.float64)
    if bundle.probes.values.ndim == 2:
        return np.broadcast_to(neutral[None, :], bundle.probes.values.shape).copy()
    return np.broadcast_to(neutral[None, :, None, None], bundle.probes.values.shape).copy()


def _basis_active_values(bundle: Bundle) -> np.ndarray:
    result = []
    for band in bundle.sensor.bands:
        assert band.valid_range is not None and band.neutral_value is not None
        low, high = band.valid_range
        low_distance = band.neutral_value - low
        high_distance = high - band.neutral_value
        result.append(high if high_distance > low_distance else low)
    return np.asarray(result, dtype=np.float64)


def _basis_chunks(bundle: Bundle) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    bands = len(bundle.sensor.bands)
    neutral = np.asarray([band.neutral_value for band in bundle.sensor.bands], dtype=np.float64)
    active = _basis_active_values(bundle)
    spatial_cells = 1 if bundle.probes.values.ndim == 2 else int(np.prod(bundle.probes.values.shape[2:]))
    bytes_per_row = bands * spatial_cells * np.dtype("float64").itemsize
    rows_per_chunk = min(C1_CHUNK_MAX_ROWS, C1_CHUNK_MAX_FLOAT64_PROBE_BYTES // bytes_per_row)
    if rows_per_chunk < 1:
        return
    for start in range(0, bands, rows_per_chunk):
        indices = np.arange(start, min(start + rows_per_chunk, bands), dtype=np.int64)
        if bundle.probes.values.ndim == 2:
            values = np.broadcast_to(neutral[None, :], (indices.size, bands)).copy()
            values[np.arange(indices.size), indices] = active[indices]
        else:
            height, width = bundle.probes.values.shape[2:]
            values = np.broadcast_to(
                neutral[None, :, None, None], (indices.size, bands, height, width)
            ).copy()
            for row, band_index in enumerate(indices):
                values[row, band_index, :, :] = active[band_index]
        yield indices, np.asarray(values, dtype=np.float64)


def _spatial_challenge(bundle: Bundle) -> np.ndarray | None:
    if bundle.probes.values.ndim != 4:
        return None
    bands = len(bundle.sensor.bands)
    height, width = bundle.probes.values.shape[2:]
    cells = height * width
    neutral = np.asarray([band.neutral_value for band in bundle.sensor.bands], dtype=np.float64)
    active = _basis_active_values(bundle)
    values = np.broadcast_to(neutral[None, :, None], (4, bands, cells)).copy()
    values[0, :, 0::2] = active[:, None]
    values[1, :, 1::2] = active[:, None]
    quarter = neutral + 0.25 * (active - neutral)
    levels = np.stack((neutral, quarter, active), axis=1)
    for cell in range(cells):
        values[2, :, cell] = levels[:, cell % 3]
    values[3] = values[2, :, ::-1]
    return np.asarray(values.reshape(4, bands, height, width), dtype=np.float64)


def _tap_mismatch_code(expected_columns: np.ndarray, observed_columns: np.ndarray) -> str:
    # A one-to-one permutation of model-channel derivative columns is a
    # reorder even though it necessarily also creates apparent support loss at
    # individual matrix positions. Classify that exact structure first.
    nearest: list[int] = []
    close = True
    for observed in observed_columns.T:
        errors = np.max(np.abs(expected_columns.T - observed[None, :]), axis=1)
        index = int(np.argmin(errors))
        nearest.append(index)
        close = close and float(errors[index]) <= NUMERIC_TOLERANCE
    if close and len(set(nearest)) == len(nearest) and nearest != list(range(len(nearest))):
        return "reordered_bands"
    expected_support = np.abs(expected_columns) > NUMERIC_TOLERANCE
    observed_support = np.abs(observed_columns) > NUMERIC_TOLERANCE
    if np.any(expected_support & ~observed_support):
        return "dropped_band"
    if np.array_equal(expected_support, observed_support):
        return "undeclared_normalization"
    return "hidden_resampling_or_extrapolation"


def _fraction(mask: np.ndarray) -> float:
    return float(np.mean(np.asarray(mask, dtype=np.float64))) if mask.size else 0.0


def _per_probe_change(original: np.ndarray, changed: np.ndarray) -> np.ndarray:
    difference = np.abs(original - changed)
    if difference.ndim == 1:
        return difference
    return np.max(difference.reshape(difference.shape[0], -1), axis=1)


def _select_probe_shift(values: np.ndarray, base_seed: bytes, canary_id: str) -> tuple[int, np.ndarray, float]:
    candidates: list[tuple[int, float]] = []
    for shift in range(1, values.shape[0]):
        rotated = np.roll(values, shift=shift, axis=0)
        fraction = _fraction(_per_probe_change(values, rotated) > MUTATION_EXCITATION_FLOOR)
        candidates.append((shift, fraction))
    maximum = max(candidate[1] for candidate in candidates)
    tied = [candidate for candidate in candidates if candidate[1] == maximum]
    seed = derive_canary_seed(base_seed, canary_id)
    selected_shift, selected_fraction = tied[int.from_bytes(seed[:8], "big") % len(tied)]
    return selected_shift, np.roll(values, shift=selected_shift, axis=0), selected_fraction


def _select_metadata_shift(values: np.ndarray, base_seed: bytes, canary_id: str) -> tuple[int, np.ndarray, float]:
    candidates: list[tuple[int, np.ndarray, float]] = []
    for shift in range(1, values.size):
        rotated = np.roll(values, shift=shift)
        fraction = _fraction(np.abs(rotated - values) > MUTATION_EXCITATION_FLOOR)
        candidates.append((shift, rotated, fraction))
    maximum = max(candidate[2] for candidate in candidates)
    tied = [candidate for candidate in candidates if candidate[2] == maximum]
    seed = derive_canary_seed(base_seed, canary_id)
    return tied[int.from_bytes(seed[:8], "big") % len(tied)]


def _raw_support_mass(
    band: Band, support: tuple[float, float]
) -> tuple[float, float, float] | None:
    low, high = support
    if band.srf_wavelength_nm is not None and band.srf_response is not None:
        # C6 may only interpret support for an SRF that passes the same complete
        # metadata validation used by the spectral axis. Otherwise its pairing
        # result is explicitly support-unresolved rather than a second, weaker
        # interpretation of malformed or inconsistent metadata.
        from .spectral import (
            _interval_area,
            _interval_curve_values,
            _validate_srf_metadata,
        )

        valid, _, _ = _validate_srf_metadata(band)
        if not valid:
            return None
        grid = np.unique(
            np.concatenate([band.srf_wavelength_nm, np.asarray([low, high], dtype=np.float64)])
        )
        left_response, right_response = _interval_curve_values(
            band.srf_wavelength_nm, band.srf_response, grid
        )
        total = _interval_area(grid, left_response, right_response)
        if not math.isfinite(total) or total <= 0.0:
            return None
        left_grid = grid[:-1]
        right_grid = grid[1:]
        inside = _interval_area(
            grid,
            left_response,
            right_response,
            (left_grid >= low) & (right_grid <= high),
        )
        below = _interval_area(
            grid, left_response, right_response, right_grid <= low
        )
        above = _interval_area(
            grid, left_response, right_response, left_grid >= high
        )
        masses = inside / total, below / total, above / total
        return masses if all(math.isfinite(value) for value in masses) else None
    assert band.fwhm_nm is not None
    sigma = band.fwhm_nm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    def cdf(value: float) -> float:
        return 0.5 * (
            1.0
            + math.erf(
                (value - band.center_nm) / (sigma * math.sqrt(2.0))
            )
        )
    below, above = cdf(low), 1.0 - cdf(high)
    return 1.0 - below - above, below, above


def run_canaries(bundle: Bundle, adapter: Adapter, base_seed: bytes) -> CanaryResult:
    ids, wavelength, fwhm = _metadata(bundle)
    faults: list[dict[str, Any]]
    route_eligible, spectral_route_eligible, faults, route_facts = _route_static(bundle)
    canaries: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    baseline_request = _request(bundle, bundle.probes.values)

    replays = [adapter.invoke(baseline_request) for _ in range(REPLAY_COUNT)]
    replay_outputs = np.stack([item.output for item in replays])
    replay_taps = np.stack([item.pre_core for item in replays])
    arrays["c0_replay_output"] = replay_outputs
    arrays["c0_replay_pre_core"] = replay_taps
    output_median = linear_quantile(replay_outputs[0], 0.5)
    scale = max(
        1.0,
        linear_quantile(np.abs(replay_outputs[0] - output_median), 0.99),
    )
    replay_range = np.max(replay_outputs, axis=0) - np.min(replay_outputs, axis=0)
    jitter = float(np.max(replay_range) / scale)
    tap_scale = max(1.0, linear_quantile(np.abs(replay_taps[0]), 0.99))
    tap_replay_range = np.max(replay_taps, axis=0) - np.min(replay_taps, axis=0)
    tap_jitter = float(np.max(tap_replay_range) / tap_scale)
    replay_stable = jitter <= REPLAY_JITTER_MAX and tap_jitter <= REPLAY_JITTER_MAX
    canaries["C0_replay"] = {
        "status": "PASS" if replay_stable else "FAIL",
        "replay_count": REPLAY_COUNT,
        "normalized_max_jitter": jitter,
        "normalized_max_output_jitter": jitter,
        "normalized_max_tap_jitter": tap_jitter,
        "output_scale": scale,
        "output_center_median": output_median,
        "tap_scale": tap_scale,
        "maximum_allowed": REPLAY_JITTER_MAX,
    }
    if not replay_stable:
        faults.append(_fault("stochastic_inference", "replay", "replay range exceeds limit"))
    tau = max(DEPENDENCE_ABSOLUTE_FLOOR, DEPENDENCE_REPLAY_MULTIPLIER * jitter)
    baseline = replays[0]

    neutral_values = _neutral_values(bundle)
    neutral_request = _request(bundle, neutral_values)
    neutral_response = adapter.invoke(neutral_request)
    arrays["c5_target_neutral_output"] = neutral_response.output

    active_values = _basis_active_values(bundle)
    expected_neutral = expected_pre_core(bundle, neutral_request)[0]
    expected_rows: list[np.ndarray] = []
    observed_rows: list[np.ndarray] = []
    basis_outputs: list[np.ndarray] = []
    basis_error = 0.0
    basis_chunk_count = 0
    for _, values in _basis_chunks(bundle):
        basis_chunk_count += 1
        request = _request(bundle, values)
        expected = expected_pre_core(bundle, request)
        response = adapter.invoke(request)
        expected_rows.append(expected)
        observed_rows.append(response.pre_core)
        basis_outputs.append(response.output)
        basis_error = max(basis_error, float(np.max(np.abs(expected - response.pre_core))))
    spatial_challenge_values = _spatial_challenge(bundle)
    spatial_challenge_rows = 0
    spatial_challenge_error = 0.0
    spatial_challenge_output = np.empty((0,), dtype=np.float64)
    spatial_challenge_pre_core = np.empty(
        (0, len(bundle.model.channels)), dtype=np.float64
    )
    if spatial_challenge_values is not None:
        spatial_challenge_rows = int(spatial_challenge_values.shape[0])
        spatial_request = _request(bundle, spatial_challenge_values)
        spatial_expected = expected_pre_core(bundle, spatial_request)
        spatial_response = adapter.invoke(spatial_request)
        spatial_challenge_output = spatial_response.output
        spatial_challenge_pre_core = spatial_response.pre_core
        spatial_challenge_error = float(
            np.max(np.abs(spatial_expected - spatial_response.pre_core))
        )
    arrays["c1_spatial_challenge_output"] = spatial_challenge_output
    arrays["c1_spatial_challenge_pre_core"] = spatial_challenge_pre_core
    expected_basis = np.concatenate(expected_rows) if expected_rows else np.empty((0, len(bundle.model.channels)))
    observed_basis = np.concatenate(observed_rows) if observed_rows else np.empty_like(expected_basis)
    arrays["c1_basis_output"] = np.concatenate(basis_outputs) if basis_outputs else np.empty((0,), dtype=np.float64)
    arrays["c1_basis_pre_core"] = observed_basis
    baseline_error = max(
        float(np.max(np.abs(expected_pre_core(bundle, baseline_request) - item.pre_core)))
        for item in replays
    )
    neutral_error = float(
        np.max(np.abs(neutral_response.pre_core - expected_neutral[None, :]))
    )
    active_delta = active_values - np.asarray(
        [band.neutral_value for band in bundle.sensor.bands], dtype=np.float64
    )
    if not np.isfinite(active_delta).all() or np.any(active_delta == 0.0):
        raise ExecutionError("C1 basis excitation contains a zero or non-finite active delta")
    expected_columns = (
        (expected_basis - expected_neutral[None, :]) / active_delta[:, None]
        if expected_basis.size
        else np.empty_like(expected_basis)
    )
    observed_columns = (
        (observed_basis - neutral_response.pre_core[0][None, :]) / active_delta[:, None]
        if observed_basis.size
        else np.empty_like(observed_basis)
    )
    if not np.isfinite(expected_columns).all() or not np.isfinite(observed_columns).all():
        raise ExecutionError("C1 basis derivative recovery produced a non-finite value")
    expected_recovered_route_columns = (
        expected_columns * bundle.model.normalization_scale[None, :]
    )
    reported_route_columns = (
        observed_columns * bundle.model.normalization_scale[None, :]
    )
    if (
        not np.isfinite(expected_recovered_route_columns).all()
        or not np.isfinite(reported_route_columns).all()
    ):
        raise ExecutionError("C1 raw route-column recovery produced a non-finite value")
    expected_route_columns = bundle.route.canonical_matrix.T
    recovered_route_error = float(
        np.max(
            np.abs(
                reported_route_columns - expected_route_columns
            )
        )
    )
    route_recovery_conditioning_error = float(
        np.max(
            np.abs(
                expected_recovered_route_columns - expected_route_columns
            )
        )
    )
    neutral_routed_raw = bundle.route.canonical_matrix @ np.asarray(
        [band.neutral_value for band in bundle.sensor.bands], dtype=np.float64
    )
    expected_recovered_offsets = (
        neutral_routed_raw[None, :]
        - expected_neutral[None, :]
        * bundle.model.normalization_scale[None, :]
    )
    reported_offsets = (
        neutral_routed_raw[None, :]
        - neutral_response.pre_core
        * bundle.model.normalization_scale[None, :]
    )
    if (
        not np.isfinite(expected_recovered_offsets).all()
        or not np.isfinite(reported_offsets).all()
    ):
        raise ExecutionError("C1 raw affine-offset recovery produced a non-finite value")
    recovered_offset_error = float(
        np.max(
            np.abs(
                reported_offsets
                - bundle.model.normalization_offset[None, :]
            )
        )
    )
    offset_recovery_conditioning_error = float(
        np.max(
            np.abs(
                expected_recovered_offsets
                - bundle.model.normalization_offset[None, :]
            )
        )
    )
    route_recovery_conditioned = (
        route_recovery_conditioning_error <= NUMERIC_TOLERANCE
    )
    offset_recovery_conditioned = (
        offset_recovery_conditioning_error <= NUMERIC_TOLERANCE
    )
    raw_recovery_conditioned = (
        route_recovery_conditioned and offset_recovery_conditioned
    )
    expected_span = (
        np.max(np.abs(expected_basis - expected_neutral[None, :]), axis=0)
        if expected_basis.size
        else np.zeros(len(bundle.model.channels), dtype=np.float64)
    )
    minimum_span = float(np.min(expected_span)) if expected_span.size else 0.0
    route_excited = bool(expected_span.size and np.all(expected_span > NUMERIC_TOLERANCE))
    adapter_spatial = getattr(adapter, "spatial_operation", "SUPPLIER_REPORTED_UNATTESTED")
    spatial_matches = adapter_spatial in {
        bundle.route.spatial_operation,
        "SUPPLIER_REPORTED_UNATTESTED",
    }
    normalized_route_matches = (
        baseline_error <= NUMERIC_TOLERANCE
        and basis_error <= NUMERIC_TOLERANCE
    )
    neutral_tap_matches = neutral_error <= NUMERIC_TOLERANCE
    recovered_route_matches = recovered_route_error <= NUMERIC_TOLERANCE
    recovered_offset_matches = recovered_offset_error <= NUMERIC_TOLERANCE
    declared_route_matches = (
        normalized_route_matches
        and raw_recovery_conditioned
        and recovered_offset_matches
        and recovered_route_matches
    )
    ordinary_tap_matches = declared_route_matches and neutral_tap_matches
    spatial_challenge_matches = spatial_challenge_error <= NUMERIC_TOLERANCE
    tap_matches = ordinary_tap_matches and spatial_challenge_matches
    if not route_excited:
        faults.append(
            _fault(
                "insufficient_route_excitation",
                "route",
                "basis challenge does not excite every expected model-channel tap",
            )
        )
        route_eligible = False
    if not spatial_matches:
        faults.append(
            _fault(
                "hidden_resampling_or_extrapolation",
                "route",
                "adapter spatial operation differs from the declaration",
            )
        )
        route_eligible = False
    route_mismatch_code: str | None = None
    if not raw_recovery_conditioned:
        faults.append(
            _fault(
                "ill_conditioned_raw_recovery",
                "route",
                "declaration-only raw route/offset recovery exceeds the numeric tolerance",
            )
        )
        route_eligible = False
    elif not declared_route_matches:
        if not recovered_offset_matches:
            route_mismatch_code = "undeclared_normalization"
        elif not recovered_route_matches:
            route_mismatch_code = _tap_mismatch_code(
                expected_route_columns, reported_route_columns
            )
        else:
            route_mismatch_code = _tap_mismatch_code(
                expected_columns, observed_columns
            )
        faults.append(
            _fault(
                route_mismatch_code,
                "route",
                "reported tap differs from declared transform",
            )
        )
        route_eligible = False
    if (
        not neutral_tap_matches
        and route_mismatch_code != "undeclared_normalization"
    ):
        faults.append(
            _fault(
                "undeclared_normalization",
                "route",
                "reported neutral tap differs from the declared affine transform",
            )
        )
        route_eligible = False
    if not spatial_challenge_matches:
        faults.append(
            _fault(
                "undeclared_spatial_reduction",
                "route",
                "the rank-4 spatial challenges disagree with exact mean reduction",
            )
        )
        route_eligible = False
    c1_pass = (
        raw_recovery_conditioned
        and route_excited
        and spatial_matches
        and tap_matches
    )
    tapped_route_recovery_usable = (
        raw_recovery_conditioned
        and route_excited
        and neutral_tap_matches
    )
    tapped_route_recovery_status = (
        "INCONCLUSIVE_ILL_CONDITIONED_RAW_RECOVERY"
        if not raw_recovery_conditioned
        else "INCONCLUSIVE_INSUFFICIENT_ROUTE_EXCITATION"
        if not route_excited
        else "INCONCLUSIVE_NEUTRAL_TAP_MISMATCH"
        if not neutral_tap_matches
        else "EVALUATED"
    )
    canaries["C1_declared_tap_agreement"] = {
        "status": (
            "PASS"
            if c1_pass
            else "INCONCLUSIVE_ILL_CONDITIONED_RAW_RECOVERY"
            if not raw_recovery_conditioned
            else "INCONCLUSIVE_INSUFFICIENT_ROUTE_EXCITATION"
            if not route_excited
            else "FAIL"
        ),
        "baseline_max_abs_error": baseline_error,
        "neutral_max_abs_error": neutral_error,
        "basis_max_abs_error": basis_error,
        "recovered_route_max_abs_error": recovered_route_error,
        "raw_route_recovery_conditioning_max_abs_error": (
            route_recovery_conditioning_error
        ),
        "raw_route_recovery_conditioned": route_recovery_conditioned,
        "raw_recovery_conditioned": raw_recovery_conditioned,
        "raw_route_recovery_usable_for_c6": tapped_route_recovery_usable,
        "recovered_offset_max_abs_error": recovered_offset_error,
        "raw_offset_recovery_conditioning_max_abs_error": (
            offset_recovery_conditioning_error
        ),
        "raw_offset_recovery_conditioned": offset_recovery_conditioned,
        "basis_active_rule": "FARTHEST_VALID_RANGE_ENDPOINT_LOWER_ON_TIE",
        "basis_target_band_order": list(ids),
        "basis_chunk_count": basis_chunk_count,
        "spatial_challenge_rows": spatial_challenge_rows,
        "spatial_challenge_max_abs_error": spatial_challenge_error,
        "spatial_challenge_status": (
            "NOT_APPLICABLE_RANK2"
            if spatial_challenge_values is None
            else "PASS"
            if spatial_challenge_matches
            else "FAIL"
        ),
        "single_spatial_cell_all_singleton_preserving_reducers_equivalent": bool(
            spatial_challenge_values is not None
            and int(np.prod(bundle.probes.values.shape[2:])) == 1
        ),
        "two_spatial_cell_mean_median_midrange_equivalence": bool(
            spatial_challenge_values is not None
            and int(np.prod(bundle.probes.values.shape[2:])) == 2
        ),
        "minimum_expected_channel_tap_span": minimum_span,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "route_assurance": adapter.assurance,
    }

    c2_outputs: list[np.ndarray] = []
    c2_deltas: dict[str, np.ndarray] = {}
    c2_bands: dict[str, dict[str, Any]] = {}
    c2_tap_error = 0.0
    required = set(bundle.model.required_dependence_target_band_ids)
    declared_abs_weight = np.sum(np.abs(bundle.route.canonical_matrix), axis=0)
    required_pass = True
    required_inconclusive = False
    for index, identifier in enumerate(ids):
        shift, rotated_band, excitation_fraction = _select_probe_shift(
            bundle.probes.values[:, index, ...], base_seed, f"C2_value_dependence:{identifier}"
        )
        mutated = np.array(bundle.probes.values, copy=True)
        mutated[:, index, ...] = rotated_band
        mutated_request = _request(bundle, mutated)
        response = adapter.invoke(mutated_request)
        tap_error = float(
            np.max(
                np.abs(response.pre_core - expected_pre_core(bundle, mutated_request))
            )
        )
        c2_tap_error = max(c2_tap_error, tap_error)
        c2_outputs.append(response.output)
        delta = (response.output - baseline.output) / scale
        c2_deltas[identifier] = delta
        output_fraction = _fraction(np.abs(delta) > tau)
        adequately_excited = excitation_fraction >= MINIMUM_PROBE_FRACTION
        dependent = bool(adequately_excited and output_fraction >= MINIMUM_PROBE_FRACTION)
        status = (
            "INCONCLUSIVE_INSUFFICIENT_EXCITATION"
            if not adequately_excited
            else "DEPENDENCE_OBSERVED"
            if dependent
            else "NO_DEPENDENCE_OBSERVED_ON_PROBES"
        )
        c2_bands[identifier] = {
            "status": status,
            "required": identifier in required,
            "selected_shift": shift,
            "input_excitation_fraction": excitation_fraction,
            "output_dependence_fraction": output_fraction,
            "dependent": dependent,
            "pre_core_max_abs_error": tap_error,
            "aggregate_absolute_route_weight": float(declared_abs_weight[index]),
        }
        if identifier in required and not dependent:
            required_pass = False
            required_inconclusive = required_inconclusive or not adequately_excited
        if dependent and bool(np.all(bundle.route.canonical_matrix[:, index] == 0.0)):
            faults.append(
                _fault(
                    "hidden_resampling_or_extrapolation",
                    "route",
                    f"output depends on exactly unrouted target {identifier}",
                )
            )
            route_eligible = False
    arrays["c2_value_dependence_output"] = np.stack(c2_outputs)
    if c2_tap_error > NUMERIC_TOLERANCE:
        faults.append(
            _fault(
                "context_dependent_undeclared_tap",
                "route",
                "a C2 returned tap differs from the declared transform for its mutated request",
            )
        )
        route_eligible = False
    if required_inconclusive:
        faults.append(
            _fault(
                "required_band_insufficient_excitation",
                "dependence",
                "a required target band cannot be adequately mutated on submitted probes",
            )
        )
    elif not required_pass:
        faults.append(
            _fault(
                "target_invariant_output_on_challenges",
                "dependence",
                "a required target band has no observed output dependence on submitted probes",
            )
        )
    c2_status = (
        "PASS"
        if required_pass
        else "INCONCLUSIVE_INSUFFICIENT_EXCITATION"
        if required_inconclusive
        else "FAIL"
    )
    canaries["C2_value_dependence"] = {
        "status": c2_status,
        "threshold": tau,
        "minimum_probe_fraction": MINIMUM_PROBE_FRACTION,
        "pre_core_max_abs_error": c2_tap_error,
        "bands": c2_bands,
    }

    metadata_conformance_pass = True
    metadata_dependence_observed: dict[str, bool] = {}
    metadata_excitation_adequate: dict[str, bool] = {}
    metadata_specs = (
        (
            "wavelength",
            wavelength,
            bundle.model.wavelength_conditioned,
            "C3_wavelength_dependence",
            "claimed_wavelength_input_ignored",
            "undeclared_wavelength_input_used",
        ),
        (
            "fwhm",
            fwhm,
            bundle.model.fwhm_conditioned,
            "C3_fwhm_dependence",
            "claimed_fwhm_input_ignored",
            "undeclared_fwhm_input_used",
        ),
    )
    expected_static_pre_core = expected_pre_core(bundle, baseline_request)
    for field, original, declared, canary_id, ignored_code, undeclared_code in metadata_specs:
        amplitudes = c3_rank_amplitudes(base_seed, canary_id, ids)
        if len(ids) < 2:
            selected_shift = 0
            rotated = np.array(original, copy=True)
            rotation_excitation = 0.0
        else:
            selected_shift, rotated, rotation_excitation = _select_metadata_shift(
                original, base_seed, canary_id
            )
        if field == "wavelength":
            increased = np.clip(original + amplitudes, 100.0, 100_000.0)
            decreased = np.clip(original - amplitudes, 100.0, 100_000.0)
        else:
            factors = 1.0 + 0.01 * amplitudes
            increased = np.clip(original * factors, 1.0, 50_000.0)
            decreased = np.clip(original / factors, 1.0, 50_000.0)
        mutation_inputs = (
            ("CYCLIC_ROTATION", rotated, rotation_excitation),
            (
                "NONUNIFORM_MAGNITUDE_INCREASE",
                increased,
                _fraction(
                    np.abs(increased - original) > MUTATION_EXCITATION_FLOOR
                ),
            ),
            (
                "NONUNIFORM_MAGNITUDE_DECREASE",
                decreased,
                _fraction(
                    np.abs(decreased - original) > MUTATION_EXCITATION_FLOOR
                ),
            ),
        )
        mutation_reports: list[dict[str, Any]] = []
        mutation_outputs: list[np.ndarray] = []
        adequately_excited = False
        observed = False
        c3_tap_error = 0.0
        for mutation_name, mutated_metadata, excitation_fraction in mutation_inputs:
            request = (
                _request(
                    bundle,
                    bundle.probes.values,
                    wavelength=mutated_metadata,
                    fwhm=fwhm,
                )
                if field == "wavelength"
                else _request(
                    bundle,
                    bundle.probes.values,
                    wavelength=wavelength,
                    fwhm=mutated_metadata,
                )
            )
            response = adapter.invoke(request)
            mutation_outputs.append(response.output)
            output_fraction = _fraction(
                np.abs(response.output - baseline.output) / scale > tau
            )
            mutation_adequate = (
                excitation_fraction >= MINIMUM_PROBE_FRACTION
            )
            mutation_observed = bool(
                mutation_adequate
                and output_fraction >= MINIMUM_PROBE_FRACTION
            )
            tap_error = float(
                np.max(np.abs(response.pre_core - expected_static_pre_core))
            )
            c3_tap_error = max(c3_tap_error, tap_error)
            adequately_excited = adequately_excited or mutation_adequate
            observed = observed or mutation_observed
            mutation_reports.append(
                {
                    "mutation": mutation_name,
                    "selected_shift": (
                        selected_shift if mutation_name == "CYCLIC_ROTATION" else None
                    ),
                    "submitted_metadata_by_target_band_id": {
                        identifier: float(mutated_metadata[index])
                        for index, identifier in enumerate(ids)
                    },
                    "metadata_excitation_fraction": excitation_fraction,
                    "output_dependence_fraction": output_fraction,
                    "adequately_excited": mutation_adequate,
                    "dependence_observed": mutation_observed,
                    "pre_core_max_abs_error": tap_error,
                }
            )
        if c3_tap_error > NUMERIC_TOLERANCE:
            faults.append(
                _fault(
                    "context_dependent_undeclared_tap",
                    "route",
                    f"a {field}-only C3 returned tap differs from the declared static transform",
                )
            )
            route_eligible = False
        field_pass = not declared
        if not adequately_excited:
            status = "INCONCLUSIVE_INSUFFICIENT_METADATA_EXCITATION"
        elif observed and declared:
            status = "PASS"
            field_pass = True
        elif observed:
            status = f"UNDECLARED_{field.upper()}_DEPENDENCE_OBSERVED_ON_PROBES"
            field_pass = False
            faults.append(
                _fault(
                    undeclared_code,
                    "dependence",
                    f"{field}-only output dependence contradicts the false declaration",
                )
            )
        else:
            status = f"NO_{field.upper()}_DEPENDENCE_OBSERVED_ON_PROBES"
            field_pass = not declared
            if declared:
                faults.append(
                    _fault(
                        ignored_code,
                        "dependence",
                        f"no {field}-only output dependence observed on submitted probes",
                    )
                )
        if declared and not adequately_excited:
            faults.append(
                _fault(
                    f"claimed_{field}_input_inconclusive",
                    "dependence",
                    f"the claimed {field} conditioning path could not be adequately excited",
                )
            )
        metadata_dependence_observed[field] = observed
        metadata_excitation_adequate[field] = adequately_excited
        metadata_conformance_pass = metadata_conformance_pass and field_pass
        canaries[canary_id] = {
            "status": status,
            "declared_conditioning_input": declared,
            "primary_mutation": (
                f"{field.upper()}_ONLY_ROTATION_AND_NONUNIFORM_MAGNITUDE"
            ),
            "canary_subseed_sha256_hex": derive_canary_seed(
                base_seed, canary_id
            ).hex(),
            "ranked_target_band_ids": [
                identifier
                for _, identifier in sorted(
                    zip(amplitudes.tolist(), ids), key=lambda item: item[0]
                )
            ],
            "rank_amplitudes_by_target_band_id": {
                identifier: float(amplitudes[index])
                for index, identifier in enumerate(ids)
            },
            "selected_shift": selected_shift,
            "metadata_excitation_fraction": max(
                row["metadata_excitation_fraction"] for row in mutation_reports
            ),
            "output_dependence_fraction": max(
                row["output_dependence_fraction"] for row in mutation_reports
            ),
            "pre_core_max_abs_error": c3_tap_error,
            "mutations": mutation_reports,
            "minimum_metadata_tuple_fraction": MINIMUM_PROBE_FRACTION,
            "threshold": tau,
        }
        stacked_outputs = np.stack(mutation_outputs)
        arrays[f"c3_{field}_dependence_output"] = stacked_outputs
        arrays[f"c3_{field}_rotation_output"] = mutation_outputs[0]
        arrays[f"c3_{field}_magnitude_increase_output"] = mutation_outputs[1]
        arrays[f"c3_{field}_magnitude_decrease_output"] = mutation_outputs[2]

    permutation = c4_permutation(base_seed, len(ids))
    shift = c4_shift(base_seed, len(ids))
    if len(ids) == 1:
        c4_pass = True
        c4_order_output = np.empty((0,), dtype=np.float64)
        c4_id_binding_output = np.empty((0,), dtype=np.float64)
        c4_report: dict[str, Any] = {
            "status": "NOT_APPLICABLE_SINGLE_BAND",
            "finding": "NOT_APPLICABLE_SINGLE_BAND",
            "permutation": [0],
            "shift": 0,
            "tied_tuple_order_invariance": {
                "status": "NOT_APPLICABLE_SINGLE_BAND"
            },
            "id_only_binding": {"status": "NOT_APPLICABLE_SINGLE_BAND"},
        }
    else:
        tied_request = _request(
            bundle,
            bundle.probes.values[:, permutation, ...],
            ids=tuple(ids[int(index)] for index in permutation),
            wavelength=wavelength[permutation],
            fwhm=fwhm[permutation],
        )
        tied_expected = expected_pre_core(bundle, tied_request)
        tied_response = adapter.invoke(tied_request)
        c4_order_output = tied_response.output
        tied_tap_error = float(
            np.max(np.abs(tied_response.pre_core - tied_expected))
        )
        tied_output_error = float(
            np.max(np.abs(tied_response.output - baseline.output))
        )
        tied_normalized_output_error = tied_output_error / scale
        tied_pass = (
            tied_tap_error <= NUMERIC_TOLERANCE
            and tied_normalized_output_error <= NUMERIC_TOLERANCE
        )

        # Hold values and numeric metadata at their submitted positions while
        # changing IDs alone. A declaration-conforming adapter must use those
        # IDs to recover sensor order at the pre-core tap. This distinct
        # challenge catches implementations that merely sort tied tuples by a
        # numeric metadata field and otherwise ignore target-band identity.
        id_only_request = _request(
            bundle,
            bundle.probes.values,
            ids=tuple(ids[int(index)] for index in permutation),
            wavelength=wavelength,
            fwhm=fwhm,
        )
        id_only_expected = expected_pre_core(bundle, id_only_request)
        id_only_expected_change = float(
            np.max(np.abs(id_only_expected - expected_static_pre_core))
        )
        id_only_excitation_adequate = (
            id_only_expected_change > NUMERIC_TOLERANCE
        )
        id_only_response = adapter.invoke(id_only_request)
        c4_id_binding_output = id_only_response.output
        id_only_tap_error = float(
            np.max(np.abs(id_only_response.pre_core - id_only_expected))
        )
        id_only_output_change = float(
            np.max(np.abs(id_only_response.output - baseline.output))
        )
        if not id_only_excitation_adequate:
            id_only_status = "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION"
            id_only_pass = False
        elif id_only_tap_error <= NUMERIC_TOLERANCE:
            id_only_status = "PASS"
            id_only_pass = True
        else:
            id_only_status = "FAIL"
            id_only_pass = False

        c4_pass = tied_pass and id_only_pass
        if not id_only_excitation_adequate:
            c4_status = "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION"
        else:
            c4_status = "PASS" if c4_pass else "FAIL"
        c4_report = {
            "status": c4_status,
            "finding": "ID_BOUND" if c4_pass else "NOT_ESTABLISHED",
            "permutation": permutation.tolist(),
            "shift": shift,
            "pre_core_max_abs_error": max(tied_tap_error, id_only_tap_error),
            "output_max_abs_error": tied_output_error,
            "normalized_output_max_abs_error": tied_normalized_output_error,
            "output_scale": scale,
            "tied_tuple_order_invariance": {
                "status": "PASS" if tied_pass else "FAIL",
                "pre_core_max_abs_error": tied_tap_error,
                "output_max_abs_error": tied_output_error,
                "normalized_output_max_abs_error": tied_normalized_output_error,
                "output_scale": scale,
            },
            "id_only_binding": {
                "status": id_only_status,
                "expected_pre_core_change_max_abs": id_only_expected_change,
                "minimum_exclusive_expected_pre_core_change": NUMERIC_TOLERANCE,
                "excitation_adequate": id_only_excitation_adequate,
                "pre_core_max_abs_error": id_only_tap_error,
                "output_change_from_baseline_max_abs": id_only_output_change,
            },
        }
        if not c4_pass:
            if not id_only_excitation_adequate:
                detail = (
                    "the declared route and submitted probes did not excite the "
                    "ID-only C4 binding challenge above the frozen threshold"
                )
            elif not tied_pass:
                detail = "tied-tuple C4 reorder changed the adapter result"
            else:
                detail = (
                    "the C4 ID-only permutation did not reproduce the "
                    "declaration-derived pre-core tap"
                )
            faults.append(_fault("reordered_bands", "route", detail))
            route_eligible = False
    arrays["c4_order_output"] = c4_order_output
    arrays["c4_id_binding_output"] = c4_id_binding_output
    canaries["C4_order"] = c4_report

    neutral_excitation = _fraction(
        _per_probe_change(bundle.probes.values, neutral_values) > MUTATION_EXCITATION_FLOOR
    )
    neutral_output_fraction = _fraction(
        np.abs(neutral_response.output - baseline.output) / scale > tau
    )
    if neutral_excitation < MINIMUM_PROBE_FRACTION:
        c5_status = "INCONCLUSIVE_INSUFFICIENT_EXCITATION"
    elif neutral_output_fraction < MINIMUM_PROBE_FRACTION:
        c5_status = "NO_JOINT_TARGET_NEUTRAL_EFFECT_OBSERVED"
    else:
        c5_status = "PASS"
    canaries["C5_target_neutral"] = {
        "status": c5_status,
        "input_excitation_fraction": neutral_excitation,
        "output_dependence_fraction": neutral_output_fraction,
        "pre_core_max_abs_error": neutral_error,
        "threshold": tau,
    }
    all_c2_adequate = all(
        row["status"] != "INCONCLUSIVE_INSUFFICIENT_EXCITATION" for row in c2_bands.values()
    )
    no_c2_dependence = all(not row["dependent"] for row in c2_bands.values())
    no_metadata_dependence = not any(metadata_dependence_observed.values())
    no_target_effect_diagnostic_applies = (
        all_c2_adequate
        and no_c2_dependence
        and all(metadata_excitation_adequate.values())
        and no_metadata_dependence
        and neutral_excitation >= MINIMUM_PROBE_FRACTION
        and neutral_output_fraction < MINIMUM_PROBE_FRACTION
    )
    bounded_target_effect_diagnostic = (
        "NO_TARGET_EFFECT_OBSERVED_ABOVE_FROZEN_THRESHOLD_ON_CHALLENGES"
        if no_target_effect_diagnostic_applies
        else "NOT_ESTABLISHED"
    )
    canaries["C5_target_neutral"]["bounded_target_effect_diagnostic"] = (
        bounded_target_effect_diagnostic
    )

    support_masses = {
        band.id: _raw_support_mass(band, bundle.model.support_range_nm)
        for band in bundle.sensor.bands
    }
    invalid_mass_ids = [identifier for identifier in ids if support_masses[identifier] is None]
    outside_ids = [
        identifier
        for identifier in ids
        if support_masses[identifier] is not None
        and support_masses[identifier][0] < 0.99
    ]
    supported_ids = [
        identifier
        for identifier in ids
        if support_masses[identifier] is not None
        and support_masses[identifier][0] >= 0.99
    ]
    id_index = {identifier: index for index, identifier in enumerate(ids)}
    pairs: list[dict[str, Any]] = []
    tapped_fault = False
    for outside_id in outside_ids:
        mass = support_masses[outside_id]
        assert mass is not None
        inside_mass, below_mass, above_mass = mass
        endpoint = (
            bundle.model.support_range_nm[1]
            if above_mass > below_mass
            else bundle.model.support_range_nm[0]
        )
        if not supported_ids:
            pairs.append(
                {
                    "outside_target_band_id": outside_id,
                    "edge_target_band_id": None,
                    "finding": "INCONCLUSIVE_NO_SUPPORTED_PARTNER",
                    "declared_columns_equal": False,
                    "reported_tap_columns_equal": False,
                    "reported_tap_comparison_status": "NOT_EVALUATED",
                }
            )
            continue
        distances = {
            identifier: abs(bundle.sensor.bands[id_index[identifier]].center_nm - endpoint)
            for identifier in supported_ids
        }
        minimum_distance = min(distances.values())
        edge_id = sorted(
            identifier for identifier, distance in distances.items() if distance == minimum_distance
        )[0]
        outside_index, edge_index = id_index[outside_id], id_index[edge_id]
        declared_out = bundle.route.canonical_matrix[:, outside_index]
        declared_edge = bundle.route.canonical_matrix[:, edge_index]
        declared_equal = bool(
            np.sum(np.abs(declared_out)) >= REQUIRED_ROUTE_WEIGHT_MIN
            and np.sum(np.abs(declared_edge)) >= REQUIRED_ROUTE_WEIGHT_MIN
            and np.max(np.abs(declared_out - declared_edge)) <= NUMERIC_TOLERANCE
        )
        tapped_out = (
            reported_route_columns[outside_index]
            if reported_route_columns.size
            else np.zeros(0)
        )
        tapped_edge = (
            reported_route_columns[edge_index]
            if reported_route_columns.size
            else np.zeros(0)
        )
        tapped_equal = bool(
            tapped_route_recovery_usable
            and tapped_out.size
            and np.sum(np.abs(tapped_out)) >= REQUIRED_ROUTE_WEIGHT_MIN
            and np.sum(np.abs(tapped_edge)) >= REQUIRED_ROUTE_WEIGHT_MIN
            and np.max(np.abs(tapped_out - tapped_edge)) <= NUMERIC_TOLERANCE
        )
        outside_row, edge_row = c2_bands[outside_id], c2_bands[edge_id]
        if (
            outside_row["status"] == "INCONCLUSIVE_INSUFFICIENT_EXCITATION"
            or edge_row["status"] == "INCONCLUSIVE_INSUFFICIENT_EXCITATION"
        ):
            finding = "INCONCLUSIVE_INSUFFICIENT_EXCITATION"
        elif declared_equal:
            finding = "CLAMP_ALIAS_CONFIRMED"
        elif not tapped_route_recovery_usable:
            finding = "INCONCLUSIVE_ROUTE_RECOVERY"
        elif tapped_equal:
            finding = "CLAMP_ALIAS_CONFIRMED"
        elif np.max(np.abs(c2_deltas[outside_id] - c2_deltas[edge_id])) <= tau:
            finding = "ALIAS_SUSPECTED"
        else:
            finding = "NO_ALIAS_OBSERVED_ON_PROBES"
        pairs.append(
            {
                "outside_target_band_id": outside_id,
                "edge_target_band_id": edge_id,
                "finding": finding,
                "declared_columns_equal": declared_equal,
                "reported_tap_columns_equal": tapped_equal,
                "reported_tap_comparison_status": (
                    tapped_route_recovery_status
                ),
            }
        )
        if tapped_equal and not declared_equal:
            tapped_fault = True
    if tapped_fault:
        faults.append(_fault("edge_clamp", "route", "reported tap has an undeclared edge alias"))
        route_eligible = False
    findings = [row["finding"] for row in pairs]
    if invalid_mass_ids:
        aggregate_finding = "INCONCLUSIVE_SUPPORT_UNRESOLVED"
    elif "CLAMP_ALIAS_CONFIRMED" in findings:
        aggregate_finding = "CLAMP_ALIAS_CONFIRMED"
    elif "ALIAS_SUSPECTED" in findings:
        aggregate_finding = "ALIAS_SUSPECTED"
    elif any(str(item).startswith("INCONCLUSIVE") for item in findings):
        aggregate_finding = "INCONCLUSIVE"
    elif pairs:
        aggregate_finding = "PASS"
    else:
        aggregate_finding = "NOT_APPLICABLE"
    canaries["C6_edge_alias"] = {
        "status": "FAIL" if tapped_fault else aggregate_finding,
        "finding": aggregate_finding,
        "unresolved_target_band_ids": invalid_mass_ids,
        "reported_tap_route_recovery_conditioned": raw_recovery_conditioned,
        "reported_tap_route_recovery_usable": tapped_route_recovery_usable,
        "reported_tap_route_recovery_status": tapped_route_recovery_status,
        "pairs": pairs,
    }

    if not replay_stable:
        executable_state = X0
    elif not c1_pass or not c4_pass or not route_eligible:
        executable_state = X1
    elif not required_pass or not metadata_conformance_pass:
        executable_state = X2
    else:
        executable_state = X3
    executed_probe_value_bytes = int(
        getattr(
            adapter,
            "cumulative_probe_value_bytes",
            bundle.adapter_work_plan.cumulative_request_probe_value_bytes,
        )
    )
    if (
        executed_probe_value_bytes
        != bundle.adapter_work_plan.cumulative_request_probe_value_bytes
    ):
        raise ExecutionError(
            "executed canary request bytes differ from the frozen work plan"
        )
    facts = {
        "replay_stable_on_probes": replay_stable,
        "declared_tap_matches_on_challenges": c1_pass,
        "all_required_output_dependence_observed": required_pass,
        "wavelength_dependence_observed_if_required": (
            not bundle.model.wavelength_conditioned
            or metadata_dependence_observed["wavelength"]
        ),
        "fwhm_dependence_observed_if_required": (
            not bundle.model.fwhm_conditioned
            or metadata_dependence_observed["fwhm"]
        ),
        "metadata_conditioning_conformance": metadata_conformance_pass,
        "bounded_target_effect_diagnostic": bounded_target_effect_diagnostic,
        "route_static": route_facts,
        "output_scale": scale,
        "dependence_threshold": tau,
        "adapter_invocation_count": adapter.invocations,
        "adapter_invocation_limit": 2 * len(ids) + 12,
        "adapter_cumulative_request_probe_value_bytes": (
            executed_probe_value_bytes
        ),
        "declared_spatial_operation": bundle.route.spatial_operation,
        "adapter_spatial_operation": adapter_spatial,
    }
    return CanaryResult(
        executable_state,
        route_eligible,
        spectral_route_eligible,
        canaries,
        facts,
        faults,
        arrays,
    )
