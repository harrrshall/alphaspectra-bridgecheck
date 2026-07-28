from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import BundleCase, BundleFactory, run_and_read, varied_probes


X2 = "X2_DECLARED_TAP_MATCHES_ROUTE_ON_CHALLENGES"
X3 = "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES"
S3 = "S3_SRF_WITHIN_DECLARED_SUPPORT"


def _exit_code(result: object, report: dict[str, Any]) -> int:
    value = getattr(result, "exit_code", report.get("exit_code"))
    assert isinstance(value, int)
    return value


def _fault_codes(report: dict[str, Any]) -> set[str]:
    return {str(fault["code"]) for fault in report["faults"]}


def _load(bundle: BundleCase):
    from bandtrace.bundle import load_bundle

    return load_bundle(bundle.root)


def _rewrite_artifact_numeric(bundle: BundleCase, key: str, array: np.ndarray) -> None:
    artifact_path = bundle.file_path("artifact")
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays[key] = np.asarray(array)
    np.savez(artifact_path, **arrays)
    bundle.refresh_hash("artifact")


WAVELENGTH_LOCATIONS = (
    "model_center",
    "sensor_center",
    "model_support_endpoint",
    "model_srf_knot",
    "sensor_srf_knot",
)


def _wavelength_location_mutators(
    location: str,
    *,
    raw_value: float,
    boundary_nm: float,
    unit: str,
) -> tuple[Any, Any]:
    scale = 1.0 if unit == "nm" else 1000.0

    def model_mutator(model: dict[str, object]) -> None:
        if location == "model_center":
            band = model["model_channels"][0]
            band["center_wavelength"] = raw_value
            band["wavelength_unit"] = unit
        elif location == "model_support_endpoint":
            support = model["declared_validated_support"]
            wavelengths_nm = list(support["wavelength_range"])
            endpoint = 0 if boundary_nm == 100.0 else 1
            wavelengths_nm[endpoint] = raw_value * scale
            support["wavelength_range"] = [value / scale for value in wavelengths_nm]
            support["wavelength_unit"] = unit
        elif location == "model_srf_knot":
            srf = model["model_channels"][0]["srf"]
            wavelengths_nm = list(srf["wavelengths"])
            endpoint = 0 if boundary_nm == 100.0 else -1
            wavelengths_nm[endpoint] = raw_value * scale
            srf["wavelengths"] = [value / scale for value in wavelengths_nm]
            srf["wavelength_unit"] = unit

    def sensor_mutator(sensor: dict[str, object]) -> None:
        if location == "sensor_center":
            band = sensor["target_bands"][0]
            band["center_wavelength"] = raw_value
            band["wavelength_unit"] = unit
        elif location == "sensor_srf_knot":
            srf = sensor["target_bands"][0]["srf"]
            wavelengths_nm = list(srf["wavelengths"])
            endpoint = 0 if boundary_nm == 100.0 else -1
            wavelengths_nm[endpoint] = raw_value * scale
            srf["wavelengths"] = [value / scale for value in wavelengths_nm]
            srf["wavelength_unit"] = unit

    return model_mutator, sensor_mutator


@pytest.mark.parametrize("location", WAVELENGTH_LOCATIONS)
@pytest.mark.parametrize(("unit", "scale"), [("nm", 1.0), ("um", 1000.0)])
@pytest.mark.parametrize("boundary_nm", [100.0, 100_000.0])
def test_wavelength_domain_includes_exact_boundaries_after_unit_conversion(
    bundle_factory: BundleFactory,
    location: str,
    unit: str,
    scale: float,
    boundary_nm: float,
) -> None:
    raw_value = boundary_nm / scale
    model_mutator, sensor_mutator = _wavelength_location_mutators(
        location,
        raw_value=raw_value,
        boundary_nm=boundary_nm,
        unit=unit,
    )

    _load(
        bundle_factory(
            model_mutator=model_mutator,
            sensor_mutator=sensor_mutator,
        )
    )


@pytest.mark.parametrize("location", WAVELENGTH_LOCATIONS)
@pytest.mark.parametrize(("unit", "scale"), [("nm", 1.0), ("um", 1000.0)])
@pytest.mark.parametrize("boundary_nm", [100.0, 100_000.0])
def test_wavelength_domain_rejects_next_float_outside_after_unit_conversion(
    bundle_factory: BundleFactory,
    location: str,
    unit: str,
    scale: float,
    boundary_nm: float,
) -> None:
    raw_boundary = np.float64(boundary_nm / scale)
    direction = np.float64(0.0 if boundary_nm == 100.0 else np.inf)
    raw_value = np.nextafter(raw_boundary, direction).item()
    converted = raw_value * scale
    if boundary_nm == 100.0:
        assert converted < 100.0
    else:
        assert converted > 100_000.0
    model_mutator, sensor_mutator = _wavelength_location_mutators(
        location,
        raw_value=raw_value,
        boundary_nm=boundary_nm,
        unit=unit,
    )

    from bandtrace.errors import BundleError

    with pytest.raises(
        BundleError,
        match="wavelength_nm_micron_mismatch|outside.*domain|support range.*unit-mismatched",
    ):
        _load(
            bundle_factory(
                model_mutator=model_mutator,
                sensor_mutator=sensor_mutator,
            )
        )


@pytest.mark.parametrize(
    "quantity",
    [
        "foo",
        "reflectance_fraction",
        "radiance_w_m2_sr_nm",
        "spectral_radiance_w_m2_sr_nm",
    ],
)
def test_equal_but_out_of_vocabulary_radiometric_labels_are_invalid(
    bundle_factory: BundleFactory,
    quantity: str,
) -> None:
    def model_quantity(model: dict[str, object]) -> None:
        model["radiometric_quantity"] = quantity

    def sensor_quantity(sensor: dict[str, object]) -> None:
        sensor["radiometric_quantity"] = quantity

    bundle = bundle_factory(model_mutator=model_quantity, sensor_mutator=sensor_quantity)
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="radiometric|quantity|vocabulary"):
        _load(bundle)


def test_routed_target_raw_domain_outside_model_range_blocks_x_only(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def narrow_model(model: dict[str, object]) -> None:
        model["valid_range"] = [0.1, 0.9]

    result, report = run_and_read(
        bundle_factory(
            model_mutator=narrow_model,
            normalization_offset=np.full(4, 0.5, dtype=np.float64),
        ),
        tmp_path / "routed-domain",
    )

    assert _exit_code(result, report) == 4
    assert "routed_domain_outside_model_valid_range" in _fault_codes(report)
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == S3


def test_selection_operation_counts_every_strictly_positive_weight(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def almost_one_hot(route: dict[str, object]) -> None:
        route["matrix"][3] = [0.0, 0.0, 0.0, 0.9999995, 0.0000005]

    result, report = run_and_read(
        bundle_factory(route_mutator=almost_one_hot),
        tmp_path / "not-one-hot",
    )

    assert _exit_code(result, report) == 4
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == S3
    assert "hidden_resampling_or_extrapolation" in _fault_codes(report)


def test_c2_shift_work_formula_accepts_exact_cap_and_rejects_one_below(
    clean_numpy_bundle: BundleCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.bundle as bundle_module
    from bandtrace.errors import BundleError

    probes = varied_probes()
    exact_work = (probes.shape[0] - 1) * probes.size
    assert exact_work == 1900
    monkeypatch.setattr(bundle_module, "MAX_C2_SHIFT_CELL_COMPARISONS", exact_work)
    _load(clean_numpy_bundle)

    monkeypatch.setattr(bundle_module, "MAX_C2_SHIFT_CELL_COMPARISONS", exact_work - 1)
    with pytest.raises(BundleError, match="C2|shift|comparison|work|1900"):
        _load(clean_numpy_bundle)


def test_declared_numeric_magnitude_over_1e12_is_invalid(
    bundle_factory: BundleFactory,
) -> None:
    def extreme_offset(model: dict[str, object]) -> None:
        model["normalization"]["offset"][0] = np.nextafter(1e12, np.inf).item()

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="1e12|1000000000000|magnitude|bound"):
        _load(bundle_factory(model_mutator=extreme_offset))


def test_declared_numeric_magnitude_bound_does_not_override_raw_field_domain(
    bundle_factory: BundleFactory,
) -> None:
    def boundary_offset(model: dict[str, object]) -> None:
        model["normalization"]["offset"][0] = 1e12

    from bandtrace.errors import BundleError

    with pytest.raises(
        BundleError,
        match="normalization.offset.*inside.*raw valid range",
    ):
        _load(bundle_factory(model_mutator=boundary_offset))


def test_numpy_artifact_numeric_magnitude_over_1e12_is_invalid(
    bundle_factory: BundleFactory,
) -> None:
    bundle = bundle_factory()
    weights = np.asarray([np.nextafter(1e12, np.inf), -0.5, 0.35, 0.9], dtype=np.float64)
    _rewrite_artifact_numeric(bundle, "output_weights", weights)
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="artifact|1e12|1000000000000|magnitude|bound"):
        _load(bundle)


def test_probe_numeric_magnitude_over_1e12_is_invalid_before_range_interpretation(
    bundle_factory: BundleFactory,
) -> None:
    probes = varied_probes()
    probes[0, 0] = np.nextafter(1e12, np.inf)
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="probe|1e12|1000000000000|magnitude|bound"):
        _load(bundle_factory(probes=probes))


def test_normalization_scale_exact_minimum_is_valid_and_subminimum_is_invalid(
    bundle_factory: BundleFactory,
) -> None:
    exact = np.asarray([0.01, 1.0, 1.0, 1.0], dtype=np.float64)
    _load(bundle_factory(normalization_scale=exact))

    below = exact.copy()
    below[0] = np.nextafter(np.float64(0.01), np.float64(0.0))
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match=r"scale.*\[0\.01, 2\]"):
        _load(bundle_factory(normalization_scale=below))


@pytest.mark.parametrize("side", ["model", "sensor"])
def test_fwhm_below_one_nm_is_invalid_bundle(
    bundle_factory: BundleFactory,
    side: str,
) -> None:
    below = np.nextafter(np.float64(1.0), np.float64(0.0)).item()

    def model_fwhm(model: dict[str, object]) -> None:
        model["model_channels"][0]["fwhm"] = below

    def sensor_fwhm(sensor: dict[str, object]) -> None:
        sensor["target_bands"][0]["fwhm"] = below

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="fwhm.*outside.*domain"):
        _load(
            bundle_factory(
                model_mutator=model_fwhm if side == "model" else None,
                sensor_mutator=sensor_fwhm if side == "sensor" else None,
            )
        )


@pytest.mark.parametrize("side", ["model", "sensor"])
def test_fwhm_exactly_one_nm_is_schema_valid(
    bundle_factory: BundleFactory,
    side: str,
) -> None:
    def model_fwhm(model: dict[str, object]) -> None:
        model["model_channels"][0]["fwhm"] = 1.0

    def sensor_fwhm(sensor: dict[str, object]) -> None:
        sensor["target_bands"][0]["fwhm"] = 1.0

    _load(
        bundle_factory(
            model_mutator=model_fwhm if side == "model" else None,
            sensor_mutator=sensor_fwhm if side == "sensor" else None,
        )
    )


@pytest.mark.parametrize("side", ["model", "sensor"])
def test_valid_range_narrower_than_minimum_width_is_invalid_before_c1(
    bundle_factory: BundleFactory,
    side: str,
) -> None:
    narrow = [0.5, np.nextafter(np.float64(0.6), np.float64(0.5)).item()]

    def model_range(model: dict[str, object]) -> None:
        model["valid_range"] = narrow

    def sensor_range(sensor: dict[str, object]) -> None:
        sensor["valid_range"] = narrow

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="valid_range.*width.*0.1"):
        _load(
            bundle_factory(
                model_mutator=model_range if side == "model" else None,
                sensor_mutator=sensor_range if side == "sensor" else None,
            )
        )


@pytest.mark.parametrize("side", ["model", "sensor"])
def test_valid_range_exactly_minimum_width_is_schema_valid(
    bundle_factory: BundleFactory,
    side: str,
) -> None:
    def model_range(model: dict[str, object]) -> None:
        model["valid_range"] = [0.0, 0.1]

    def sensor_range(sensor: dict[str, object]) -> None:
        sensor["valid_range"] = [0.0, 0.1]
        for band in sensor["target_bands"]:
            band["neutral_value"] = 0.0

    probes = np.zeros((20, 5), dtype=np.float64) if side == "sensor" else None
    _load(
        bundle_factory(
            probes=probes,
            model_mutator=model_range if side == "model" else None,
            sensor_mutator=sensor_range if side == "sensor" else None,
        )
    )


def test_positive_route_weight_below_1e_minus_12_is_invalid_bundle(
    bundle_factory: BundleFactory,
) -> None:
    tiny = np.nextafter(np.float64(1e-12), np.float64(0.0)).item()

    def subminimum_weight(route: dict[str, object]) -> None:
        route["operation"] = "nonnegative_row_normalized_linear_resampling"
        route["matrix"][3] = [0.0, 0.0, 0.0, 1.0 - tiny, tiny]

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="weight|1e-12|minimum|small"):
        _load(bundle_factory(route_mutator=subminimum_weight))


def test_exact_1e_minus_12_route_weight_retains_hex_and_categorical_evidence(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    tiny = 1e-12

    def minimum_weight(route: dict[str, object]) -> None:
        route["operation"] = "nonnegative_row_normalized_linear_resampling"
        # Keep the exact-rational routed raw upper bound strictly below 1.0;
        # 1.0 - tiny plus tiny can exceed 1.0 after the two binary64 values
        # are interpreted exactly.
        route["matrix"][3] = [0.0, 0.0, 0.0, 1.0 - 2.0 * tiny, tiny]

    output_dir = tmp_path / "minimum-positive-weight"
    result, report = run_and_read(bundle_factory(route_mutator=minimum_weight), output_dir)

    assert _exit_code(result, report) == 0
    assert report["states"]["spectral"] == S3
    support = report["facts"]["spectral_support_by_model_channel"]["m750"]
    assert support["paired_target_band_ids"] == ["t750", "t950"]

    route_audit = report["facts"]["route_audit"]
    model_index = route_audit["model_channel_ids"].index("m750")
    target_index = route_audit["target_band_ids"].index("t950")
    assert route_audit["declared_weight_float64_hex"][model_index][target_index] == (
        float(tiny).hex()
    )
    assert route_audit["declared_weight_is_strictly_positive"][model_index][
        target_index
    ] is True
    assert route_audit["declared_target_column_is_exactly_zero"][target_index] is False

    rows = list(csv.DictReader((output_dir / "route.csv").open(newline="", encoding="utf-8")))
    csv_row = next(
        row
        for row in rows
        if row["model_channel_id"] == "m750" and row["target_band_id"] == "t950"
    )
    assert csv_row["declared_weight_float64_hex"] == float(tiny).hex()
    assert csv_row["declared_weight_is_strictly_positive"].lower() == "true"
    assert csv_row["declared_target_column_is_exactly_zero"].lower() == "false"


def test_clean_exact_zero_target_column_is_auditable_after_rounding(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exact-zero-column"
    _, report = run_and_read(clean_numpy_bundle, output_dir)
    route_audit = report["facts"]["route_audit"]
    target_index = route_audit["target_band_ids"].index("t950")

    assert route_audit["declared_target_column_is_exactly_zero"][target_index] is True
    assert all(
        row[target_index] == float(0.0).hex()
        for row in route_audit["declared_weight_float64_hex"]
    )
    assert all(
        row[target_index] is False
        for row in route_audit["declared_weight_is_strictly_positive"]
    )

    csv_rows = list(
        csv.DictReader((output_dir / "route.csv").open(newline="", encoding="utf-8"))
    )
    t950_rows = [row for row in csv_rows if row["target_band_id"] == "t950"]
    assert t950_rows
    assert all(
        row["declared_weight_float64_hex"] == float(0.0).hex()
        for row in t950_rows
    )
    assert all(
        row["declared_weight_is_strictly_positive"].lower() == "false"
        for row in t950_rows
    )
    assert all(
        row["declared_target_column_is_exactly_zero"].lower() == "true"
        for row in t950_rows
    )
