from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import BundleFactory, run_and_read


S0 = "S0_SUPPORT_UNRESOLVED"
S1 = "S1_OUTSIDE_DECLARED_SUPPORT"
S2 = "S2_APPROX_WITHIN_SUPPORT"


def _evaluate(bundle):
    from bandtrace.bundle import load_bundle
    from bandtrace.spectral import evaluate_spectral_support

    return evaluate_spectral_support(load_bundle(bundle.root), route_eligible=True)


def _channel(result, identifier: str) -> dict[str, object]:
    return next(row for row in result.channels if row["model_channel_id"] == identifier)


def _zero_outside_interval_metrics(
    training_wavelength: np.ndarray,
    training_response: np.ndarray,
    target_wavelength: np.ndarray,
    target_response: np.ndarray,
    support: tuple[float, float],
) -> tuple[float, float, float]:
    """Integrate native piecewise-linear SRFs without bridging endpoint jumps."""

    def native_area_and_moment(
        wavelength: np.ndarray, response: np.ndarray
    ) -> tuple[float, float]:
        dx = np.diff(wavelength)
        area = float(np.sum(dx * (response[:-1] + response[1:]) / 2.0))
        moment = float(
            np.sum(
                dx
                / 6.0
                * (
                    (2.0 * wavelength[:-1] + wavelength[1:]) * response[:-1]
                    + (wavelength[:-1] + 2.0 * wavelength[1:]) * response[1:]
                )
            )
        )
        return area, moment

    training_area, training_moment = native_area_and_moment(
        training_wavelength, training_response
    )
    target_area, target_moment = native_area_and_moment(
        target_wavelength, target_response
    )

    grid = np.unique(
        np.concatenate(
            [
                training_wavelength,
                target_wavelength,
                np.asarray(support, dtype=np.float64),
            ]
        )
    )
    l1 = 0.0
    support_target_area = 0.0
    for left, right in zip(grid[:-1], grid[1:]):
        midpoint = 0.5 * (float(left) + float(right))

        def interval_values(
            wavelength: np.ndarray,
            response: np.ndarray,
            area: float,
        ) -> tuple[float, float]:
            if midpoint < float(wavelength[0]) or midpoint > float(wavelength[-1]):
                return 0.0, 0.0
            values = np.interp(
                np.asarray([left, right], dtype=np.float64), wavelength, response
            )
            return float(values[0] / area), float(values[1] / area)

        train_left, train_right = interval_values(
            training_wavelength, training_response, training_area
        )
        target_left, target_right = interval_values(
            target_wavelength, target_response, target_area
        )
        difference_left = target_left - train_left
        difference_right = target_right - train_right
        width = float(right - left)
        if difference_left * difference_right < 0.0:
            crossing_fraction = abs(difference_left) / (
                abs(difference_left) + abs(difference_right)
            )
            l1 += 0.5 * width * (
                abs(difference_left) * crossing_fraction
                + abs(difference_right) * (1.0 - crossing_fraction)
            )
        else:
            l1 += 0.5 * width * (
                abs(difference_left) + abs(difference_right)
            )

        clipped_left = max(float(left), support[0])
        clipped_right = min(float(right), support[1])
        if clipped_left < clipped_right and (
            float(target_wavelength[0]) <= midpoint <= float(target_wavelength[-1])
        ):
            clipped = np.interp(
                np.asarray([clipped_left, clipped_right]),
                target_wavelength,
                target_response,
            )
            support_target_area += (
                0.5 * (clipped_right - clipped_left) * float(clipped[0] + clipped[1])
            )

    available_mass = support_target_area / target_area
    center_shift = abs(target_moment / target_area - training_moment / training_area)
    return available_mass, l1, center_shift


@pytest.mark.parametrize("value", ["false", "true"])
def test_support_assertion_boolean_lookalike_strings_are_rejected(
    bundle_factory: BundleFactory,
    value: str,
) -> None:
    def string_value(model: dict[str, object]) -> None:
        model["declared_validated_support"]["supplier_assertion"] = value

    from bandtrace.bundle import load_bundle
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="supplier_assertion|literal boolean"):
        load_bundle(bundle_factory(model_mutator=string_value).root)


@pytest.mark.parametrize("assertion", ["missing", False])
def test_missing_or_literal_false_support_assertion_is_s0(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    assertion: str | bool,
) -> None:
    def absent_or_false(model: dict[str, object]) -> None:
        support = model["declared_validated_support"]
        if assertion == "missing":
            support.pop("supplier_assertion")
        else:
            support["supplier_assertion"] = False

    result, report = run_and_read(
        bundle_factory(model_mutator=absent_or_false),
        tmp_path / f"support-assertion-{assertion}",
    )

    assert result.exit_code == 4
    assert report["states"]["spectral"] == S0
    assert report["states"]["executable"] == "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES"
    assert report["facts"]["declared_support_assertion"] is False
    assert "support_declaration_missing" in {
        fault["code"] for fault in report["faults"]
    }


def test_literal_true_support_assertion_is_eligible_for_s3(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    result, report = run_and_read(
        bundle_factory(),
        tmp_path / "support-assertion-true",
    )

    assert result.exit_code == 0
    assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"
    assert report["facts"]["declared_support_assertion"] is True


def test_available_mass_clips_at_support_boundary_without_half_segment_leakage(
    bundle_factory: BundleFactory,
) -> None:
    def clip_inside_first_srf_segment(model: dict[str, object]) -> None:
        model["declared_validated_support"]["wavelength_range"] = [435.0, 780.0]

    result = _evaluate(bundle_factory(model_mutator=clip_inside_first_srf_segment))
    first = _channel(result, "m450")

    # t450 is a unit-area triangle on 430,440,450,460,470.  The exact excluded
    # 430..435 triangle has area 0.625 of total area 20, hence 1 - 0.625/20.
    assert first["available_mass"] == pytest.approx(0.96875, rel=0.0, abs=1e-12)
    assert first["normalized_l1"] == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert result.state == S1
    assert "target_srf_outside_support" in {fault["code"] for fault in result.faults}


def test_nonzero_srf_endpoints_do_not_create_far_support_tails_or_false_s1(
    bundle_factory: BundleFactory,
) -> None:
    endpoint = 0.0001
    offsets = np.asarray([-20.0, -10.0, 0.0, 10.0, 20.0])
    training_response = np.asarray(
        [endpoint, 0.5, 1.0, 0.5, endpoint], dtype=np.float64
    )
    target_response = np.asarray([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float64)
    support = (100.0, 100_000.0)

    def nonzero_training_endpoints(model: dict[str, object]) -> None:
        model["declared_validated_support"]["wavelength_range"] = list(support)
        for band in model["model_channels"]:
            center = float(band["center_wavelength"])
            band["srf"] = {
                "wavelengths": (center + offsets).tolist(),
                "responses": training_response.tolist(),
                "wavelength_unit": "nm",
            }

    def zero_target_endpoints(sensor: dict[str, object]) -> None:
        for band in sensor["target_bands"]:
            center = float(band["center_wavelength"])
            band["srf"] = {
                "wavelengths": (center + offsets).tolist(),
                "responses": target_response.tolist(),
                "wavelength_unit": "nm",
            }

    expected_mass, expected_l1, expected_center_shift = (
        _zero_outside_interval_metrics(
            450.0 + offsets,
            training_response,
            450.0 + offsets,
            target_response,
            support,
        )
    )
    assert expected_mass == pytest.approx(1.0, rel=0.0, abs=1e-15)
    assert expected_l1 == pytest.approx(
        0.00007999600019991403, rel=0.0, abs=1e-15
    )
    assert expected_center_shift == pytest.approx(0.0, rel=0.0, abs=1e-12)

    result = _evaluate(
        bundle_factory(
            model_mutator=nonzero_training_endpoints,
            sensor_mutator=zero_target_endpoints,
        )
    )
    row = _channel(result, "m450")

    assert result.state == "S3_SRF_WITHIN_DECLARED_SUPPORT"
    assert row["available_mass"] == pytest.approx(expected_mass, rel=0.0, abs=1e-12)
    assert row["normalized_l1"] == pytest.approx(expected_l1, rel=0.0, abs=1e-12)
    assert row["center_shift_nm"] == pytest.approx(
        expected_center_shift, rel=0.0, abs=1e-12
    )
    assert row["pass"] is True


def test_nonzero_srf_endpoints_do_not_dilute_native_l1_into_false_s3(
    bundle_factory: BundleFactory,
) -> None:
    endpoint = 0.0001
    offsets = np.asarray([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    training_response = np.asarray(
        [endpoint, 0.0, 0.2, 0.5, 1.0, 0.5, 0.2, 0.0, endpoint],
        dtype=np.float64,
    )
    target_response = np.asarray(
        [endpoint, 0.2, 0.0, 0.5, 1.0, 0.5, 0.0, 0.2, endpoint],
        dtype=np.float64,
    )
    support = (100.0, 100_000.0)

    def redistributed_srf(
        bands: list[dict[str, object]], response: np.ndarray
    ) -> None:
        for band in bands:
            center = float(band["center_wavelength"])
            band["fwhm"] = 1.0
            band["srf"] = {
                "wavelengths": (center + offsets).tolist(),
                "responses": response.tolist(),
                "wavelength_unit": "nm",
            }

    def redistributed_training(model: dict[str, object]) -> None:
        model["declared_validated_support"]["wavelength_range"] = list(support)
        redistributed_srf(model["model_channels"], training_response)

    def redistributed_target(sensor: dict[str, object]) -> None:
        redistributed_srf(sensor["target_bands"], target_response)

    expected_mass, expected_l1, expected_center_shift = (
        _zero_outside_interval_metrics(
            450.0 + offsets,
            training_response,
            450.0 + offsets,
            target_response,
            support,
        )
    )
    assert expected_mass == pytest.approx(1.0, rel=0.0, abs=1e-15)
    assert expected_l1 == pytest.approx(
        0.24998958376734304, rel=0.0, abs=1e-15
    )
    assert expected_l1 > 0.05
    assert expected_center_shift == pytest.approx(0.0, rel=0.0, abs=1e-12)

    result = _evaluate(
        bundle_factory(
            model_mutator=redistributed_training,
            sensor_mutator=redistributed_target,
        )
    )
    row = _channel(result, "m450")

    assert result.state == S1
    assert "routed_response_mismatch" in {fault["code"] for fault in result.faults}
    assert row["available_mass"] == pytest.approx(expected_mass, rel=0.0, abs=1e-12)
    assert row["normalized_l1"] == pytest.approx(expected_l1, rel=0.0, abs=1e-12)
    assert row["center_shift_nm"] == pytest.approx(
        expected_center_shift, rel=0.0, abs=1e-12
    )
    assert row["pass"] is False


def test_malformed_present_partial_srf_cannot_hide_behind_gaussian_fallback(
    bundle_factory: BundleFactory,
) -> None:
    def partial_model(model: dict[str, object]) -> None:
        for band in model["model_channels"]:
            band.pop("srf")

    def partial_sensor(sensor: dict[str, object]) -> None:
        for band in sensor["target_bands"]:
            band.pop("srf")
        sensor["target_bands"][0]["srf"] = {
            "wavelengths": [440.0, 450.0, 460.0],
            "responses": [0.0, 1.0, 0.0],
            "wavelength_unit": "nm",
        }

    result = _evaluate(
        bundle_factory(model_mutator=partial_model, sensor_mutator=partial_sensor)
    )

    assert result.state == S0
    assert result.method == "invalid_present_srf"
    assert "invalid_present_srf" in {fault["code"] for fault in result.faults}


@pytest.mark.parametrize("inconsistency", ["fwhm", "center"])
def test_srf_declared_fwhm_or_center_inconsistency_is_s0(
    bundle_factory: BundleFactory,
    inconsistency: str,
) -> None:
    def inconsistent_sensor(sensor: dict[str, object]) -> None:
        band = sensor["target_bands"][0]
        if inconsistency == "fwhm":
            band["srf"] = {
                "wavelengths": [410.0, 430.0, 450.0, 470.0, 490.0],
                "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
                "wavelength_unit": "nm",
            }
        else:
            band["srf"] = {
                "wavelengths": [440.0, 450.0, 460.0, 470.0, 480.0],
                "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
                "wavelength_unit": "nm",
            }

    result = _evaluate(bundle_factory(sensor_mutator=inconsistent_sensor))

    assert result.state == S0
    assert "srf_metadata_inconsistent" in {fault["code"] for fault in result.faults}


def test_narrow_gaussian_components_converge_over_very_broad_support(
    bundle_factory: BundleFactory,
) -> None:
    def narrow_model(model: dict[str, object]) -> None:
        model["declared_validated_support"]["wavelength_range"] = [100.0, 100_000.0]
        for band in model["model_channels"]:
            band.pop("srf")
            band["fwhm"] = 1.0

    def narrow_sensor(sensor: dict[str, object]) -> None:
        for band in sensor["target_bands"]:
            band.pop("srf")
            band["fwhm"] = 1.0

    result = _evaluate(
        bundle_factory(model_mutator=narrow_model, sensor_mutator=narrow_sensor)
    )

    assert result.state == S2
    assert result.method == "gaussian_fwhm"
    for row in result.channels:
        assert row["available_mass"] == pytest.approx(1.0, rel=0.0, abs=1e-10)
        assert row["normalized_l1"] == pytest.approx(0.0, rel=0.0, abs=1e-10)
        assert row["pass"] is True


def test_gaussian_numerical_work_cap_fails_closed_to_s0(
    bundle_factory: BundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fwhm_only_model(model: dict[str, object]) -> None:
        for band in model["model_channels"]:
            band.pop("srf")

    def fwhm_only_sensor(sensor: dict[str, object]) -> None:
        for band in sensor["target_bands"]:
            band.pop("srf")

    import bandtrace.spectral as spectral

    monkeypatch.setattr(spectral, "_MAX_GAUSSIAN_UNIQUE_KNOTS", 8, raising=False)
    monkeypatch.setattr(spectral, "_MAX_GAUSSIAN_COMPONENT_EVALUATIONS", 8, raising=False)
    result = _evaluate(
        bundle_factory(
            model_mutator=fwhm_only_model,
            sensor_mutator=fwhm_only_sensor,
        )
    )

    assert result.state == S0
    assert "numerical_support_unresolved" in {fault["code"] for fault in result.faults}


def test_isolated_knots_exactly_at_half_peak_are_zero_width_extra_components(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def add_isolated_half_peak_components(sensor: dict[str, object]) -> None:
        sensor["target_bands"][0]["srf"] = {
            "wavelengths": [
                390.0,
                400.0,
                410.0,
                420.0,
                430.0,
                440.0,
                450.0,
                460.0,
                470.0,
                480.0,
                490.0,
                500.0,
                510.0,
            ],
            "responses": [
                0.0,
                0.0,
                0.5,
                0.0,
                0.0,
                0.5,
                1.0,
                0.5,
                0.0,
                0.0,
                0.5,
                0.0,
                0.0,
            ],
            "wavelength_unit": "nm",
        }

    bundle = bundle_factory(sensor_mutator=add_isolated_half_peak_components)
    result, report = run_and_read(bundle, tmp_path / "isolated-half-peak")

    assert getattr(result, "exit_code", report.get("exit_code")) == 4
    assert report["states"]["spectral"] == S0
    assert "invalid_present_srf" in {fault["code"] for fault in report["faults"]}


def test_full_srf_unique_knot_limit_accepts_exact_boundary_and_fails_one_below(
    bundle_factory: BundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.spectral as spectral

    bundle = bundle_factory()
    baseline = _evaluate(bundle)
    exact_required_knots = max(
        max(int(row["unique_union_knots"]), int(row["difference_root_grid_knots"]))
        for row in baseline.channels
    )
    assert spectral._MAX_FULL_SRF_UNIQUE_KNOTS == 250_000

    monkeypatch.setattr(spectral, "_MAX_FULL_SRF_UNIQUE_KNOTS", exact_required_knots)
    at_boundary = _evaluate(bundle)
    assert at_boundary.state != S0
    assert "numerical_support_unresolved" not in {
        fault["code"] for fault in at_boundary.faults
    }

    monkeypatch.setattr(
        spectral, "_MAX_FULL_SRF_UNIQUE_KNOTS", exact_required_knots - 1
    )
    over_boundary = _evaluate(bundle)
    assert over_boundary.state == S0
    assert "numerical_support_unresolved" in {
        fault["code"] for fault in over_boundary.faults
    }


def test_full_srf_budget_uses_exact_union_plus_two_root_grid_accounting(
    bundle_factory: BundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.spectral as spectral

    def two_positive_targets(route: dict[str, object]) -> None:
        route["operation"] = "nonnegative_row_normalized_linear_resampling"
        route["matrix"][0] = [0.5, 0.5, 0.0, 0.0, 0.0]

    bundle = bundle_factory(route_mutator=two_positive_targets)
    baseline = _evaluate(bundle)
    assert baseline.method == "full_srf"
    exact_total = 0
    saw_multi_target_channel = False
    for row in baseline.channels:
        positive_targets = len(row["paired_target_band_ids"])
        union_knots = int(row["unique_union_knots"])
        root_knots = int(row["difference_root_grid_knots"])
        expected_charge = (1 + positive_targets) * union_knots + 2 * root_knots
        assert row["interpolation_component_point_evaluations"] == expected_charge
        exact_total += expected_charge
        saw_multi_target_channel = saw_multi_target_channel or positive_targets > 1
    assert saw_multi_target_channel is True
    assert spectral._MAX_FULL_SRF_COMPONENT_EVALUATIONS == 50_000_000

    monkeypatch.setattr(
        spectral, "_MAX_FULL_SRF_COMPONENT_EVALUATIONS", exact_total
    )
    at_boundary = _evaluate(bundle)
    assert at_boundary.state != S0
    assert "numerical_support_unresolved" not in {
        fault["code"] for fault in at_boundary.faults
    }

    monkeypatch.setattr(
        spectral, "_MAX_FULL_SRF_COMPONENT_EVALUATIONS", exact_total - 1
    )
    over_boundary = _evaluate(bundle)
    assert over_boundary.state == S0
    assert "numerical_support_unresolved" in {
        fault["code"] for fault in over_boundary.faults
    }


def test_full_srf_budget_object_accepts_exact_50m_and_rejects_50m_plus_one() -> None:
    import bandtrace.spectral as spectral

    budget = spectral._FullSrfBudget()
    budget.charge(1, 50_000_000)
    assert budget.component_evaluations == 50_000_000
    with pytest.raises(spectral._NumericalUnresolved, match="budget exceeded"):
        budget.charge(1, 1)
