from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np
import pytest

from conftest import BundleCase, BundleFactory, SOURCE_ROOT


X0 = "X0_NO_EXECUTABLE_OBSERVATION"
X2 = "X2_DECLARED_TAP_MATCHES_ROUTE_ON_CHALLENGES"
X3 = "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES"
S0 = "S0_SUPPORT_UNRESOLVED"
S1 = "S1_OUTSIDE_DECLARED_SUPPORT"
S2 = "S2_APPROX_WITHIN_SUPPORT"
S3 = "S3_SRF_WITHIN_DECLARED_SUPPORT"
T0 = "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED"


def _cli(bundle: BundleCase, output_dir: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "bandtrace",
            "audit",
            str(bundle.root),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )


def _report(output_dir: Path) -> dict[str, object]:
    return json.loads((output_dir / "report.json").read_text(encoding="utf-8"))


def _fault_codes(report: dict[str, object]) -> set[str]:
    return {fault["code"] for fault in report["faults"]}


def _micron_mismatch(sensor: dict[str, object]) -> None:
    for band in sensor["target_bands"]:
        band["wavelength_unit"] = "um"
        band["fwhm_unit"] = "um"
        band["srf"]["wavelength_unit"] = "um"


def _missing_spectral_width(sensor: dict[str, object]) -> None:
    band = sensor["target_bands"][0]
    for key in ("fwhm", "fwhm_unit", "srf"):
        band.pop(key)


def _outside_srf(sensor: dict[str, object]) -> None:
    band = sensor["target_bands"][0]
    band["center_wavelength"] = 390.0
    band["srf"] = {
        "wavelengths": [370.0, 380.0, 390.0, 400.0, 410.0],
        "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
        "wavelength_unit": "nm",
    }


def _invalid_present_srf(sensor: dict[str, object]) -> None:
    sensor["target_bands"][0]["srf"]["responses"] = [0.0] * 5


def _routed_response_mismatch(route: dict[str, object]) -> None:
    route["matrix"] = [
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]


def _radiometric_mismatch(sensor: dict[str, object]) -> None:
    sensor["radiometric_quantity"] = "spectral_radiance_w_m2_sr_nm"


def _duplicate_band_id(sensor: dict[str, object]) -> None:
    sensor["target_bands"][4]["id"] = sensor["target_bands"][3]["id"]


def _declare_spatial_mean(route: dict[str, object]) -> None:
    route["spatial_operation"] = "mean"


def _rank4_asymmetric_probes() -> np.ndarray:
    base = np.asarray(
        [
            [
                0.08 + 0.84 * (((row * (band * 2 + 3) + band * 7) % 23) / 22.0)
                for band in range(5)
            ]
            for row in range(20)
        ],
        dtype=np.float64,
    )
    offsets = np.asarray([[-0.03, 0.0, 0.02], [0.01, -0.01, 0.03]], dtype=np.float64)
    probes = base[:, :, None, None] + offsets[None, None, :, :]
    assert np.all((probes >= 0.0) & (probes <= 1.0))
    return probes


def _vary_fwhm_metadata(sensor: dict[str, object]) -> None:
    widths = [19.5, 19.75, 20.0, 20.25, 20.5]
    for band, width in zip(sensor["target_bands"], widths):
        center = float(band["center_wavelength"])
        band["fwhm"] = width
        band["srf"] = {
            "wavelengths": [
                center - width,
                center - width / 2.0,
                center,
                center + width / 2.0,
                center + width,
            ],
            "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
            "wavelength_unit": "nm",
        }


def _narrow_model_raw_domain(model: dict[str, object]) -> None:
    model["valid_range"] = [-0.05, 0.9]


def _zero_numpy_output_weights(artifact_path: Path) -> None:
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["output_weights"] = np.zeros_like(arrays["output_weights"], dtype=np.float64)
    np.savez(artifact_path, **arrays)


def _reorder_numpy_artifact_route(artifact_path: Path) -> None:
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    matrix = np.asarray(arrays["route_matrix"], dtype=np.float64).copy()
    matrix[[0, 1]] = matrix[[1, 0]]
    arrays["route_matrix"] = matrix
    np.savez(artifact_path, **arrays)


def _mismatch_numpy_spatial_operation(artifact_path: Path) -> None:
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    arrays["spatial_operation"] = np.asarray("mean")
    np.savez(artifact_path, **arrays)


def _make_fault(bundle_factory: BundleFactory, fault: str) -> BundleCase:
    if fault == "dropped_band":
        return bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="dropped_band")
    if fault == "edge_clamp":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="edge_clamp",
        )
    if fault == "reordered_bands":
        return bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="reordered_bands")
    if fault == "id_ignored_metadata_sorted":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="id_ignored_metadata_sorted",
        )
    if fault == "wavelength_nm_micron_mismatch":
        return bundle_factory(adapter="numpy-linear-v1", sensor_mutator=_micron_mismatch)
    if fault == "missing_mandatory_fwhm":
        return bundle_factory(adapter="numpy-linear-v1", sensor_mutator=_missing_spectral_width)
    if fault == "invalid_present_srf":
        return bundle_factory(adapter="numpy-linear-v1", sensor_mutator=_invalid_present_srf)
    if fault == "routed_response_mismatch":
        return bundle_factory(
            adapter="numpy-linear-v1", route_mutator=_routed_response_mismatch
        )
    if fault == "target_srf_outside_support":
        return bundle_factory(adapter="numpy-linear-v1", sensor_mutator=_outside_srf)
    if fault == "claimed_wavelength_input_ignored":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="clean",
            wavelength_conditioned=True,
        )
    if fault == "undeclared_wavelength_input_used":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="wavelength_aware",
            wavelength_conditioned=False,
        )
    if fault == "claimed_fwhm_input_ignored":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="clean",
            fwhm_conditioned=True,
            sensor_mutator=_vary_fwhm_metadata,
        )
    if fault == "undeclared_fwhm_input_used":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="fwhm_aware",
            fwhm_conditioned=False,
            sensor_mutator=_vary_fwhm_metadata,
        )
    if fault == "routed_domain_outside_model_valid_range":
        return bundle_factory(model_mutator=_narrow_model_raw_domain)
    if fault == "undeclared_spatial_reduction":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="first_pixel_reduction",
            probes=_rank4_asymmetric_probes(),
            route_mutator=_declare_spatial_mean,
        )
    if fault == "target_invariant_output_on_challenges":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="prior_only",
            sensor_mutator=_vary_fwhm_metadata,
        )
    if fault == "stochastic_inference":
        return bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="stochastic")
    if fault == "radiometric_quantity_mismatch":
        return bundle_factory(adapter="numpy-linear-v1", sensor_mutator=_radiometric_mismatch)
    if fault == "undeclared_normalization":
        return bundle_factory(
            adapter="subprocess-npz-v1", subprocess_mode="wrong_normalization"
        )
    if fault == "context_dependent_undeclared_tap":
        return bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="c2_context_dependent_tap",
        )
    if fault == "hidden_resampling_or_extrapolation":
        return bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="hidden_resampling")
    if fault == "duplicate_band_ids":
        return bundle_factory(adapter="numpy-linear-v1", sensor_mutator=_duplicate_band_id)
    raise AssertionError(f"unhandled planted fault {fault}")


FAULT_EXPECTATIONS: dict[str, dict[str, object]] = {
    "dropped_band": {
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "edge_clamp": {
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "reordered_bands": {
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "id_ignored_metadata_sorted": {
        "code": "reordered_bands",
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "wavelength_nm_micron_mismatch": {"exit": 2, "axis": "bundle"},
    "missing_mandatory_fwhm": {"exit": 2, "axis": "bundle"},
    "invalid_present_srf": {"exit": 4, "axis": "spectral_support", "spectral": S0},
    "routed_response_mismatch": {
        "exit": 4,
        "axis": "spectral_support",
        "spectral": S1,
        "executable": X3,
    },
    "target_srf_outside_support": {
        "exit": 4,
        "axis": "spectral_support",
        "spectral": S1,
        "executable": X3,
    },
    "claimed_wavelength_input_ignored": {
        "exit": 4,
        "axis": "dependence",
        "forbidden_x": {X3},
        "spectral": S3,
    },
    "undeclared_wavelength_input_used": {
        "exit": 4,
        "axis": "dependence",
        "forbidden_x": {X3},
        "spectral": S3,
    },
    "claimed_fwhm_input_ignored": {
        "exit": 4,
        "axis": "dependence",
        "forbidden_x": {X3},
        "spectral": S3,
    },
    "undeclared_fwhm_input_used": {
        "exit": 4,
        "axis": "dependence",
        "forbidden_x": {X3},
        "spectral": S3,
    },
    "routed_domain_outside_model_valid_range": {
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "target_invariant_output_on_challenges": {
        "exit": 4,
        "axis": "dependence",
        "forbidden_x": {X3},
        "spectral": S3,
    },
    "stochastic_inference": {
        "exit": 4,
        "axis": "replay",
        "executable": X0,
        "spectral": S3,
    },
    "radiometric_quantity_mismatch": {"exit": 2, "axis": "bundle"},
    "undeclared_normalization": {
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "undeclared_spatial_reduction": {
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "context_dependent_undeclared_tap": {
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "hidden_resampling_or_extrapolation": {
        "exit": 4,
        "axis": "route",
        "forbidden_x": {X2, X3},
        "spectral": S3,
    },
    "duplicate_band_ids": {"exit": 2, "axis": "bundle"},
}


@pytest.mark.parametrize("fault", list(FAULT_EXPECTATIONS))
def test_every_rev29_planted_fault_is_detected_with_exact_axis_and_exit_semantics(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    fault: str,
) -> None:
    bundle = _make_fault(bundle_factory, fault)
    output_dir = tmp_path / f"out-{fault}"
    completed = _cli(bundle, output_dir)
    expected = FAULT_EXPECTATIONS[fault]

    assert completed.returncode == expected["exit"], (
        fault,
        completed.stdout,
        completed.stderr,
    )
    if completed.returncode == 2:
        assert not output_dir.exists()
        assert fault in completed.stderr
        return

    report = _report(output_dir)
    emitted_code = str(expected.get("code", fault))
    assert emitted_code in _fault_codes(report)
    assert any(
        planted["code"] == emitted_code and planted["axis"] == expected["axis"]
        for planted in report["faults"]
    )
    assert report["facts"]["overall_conformance_fault"] is True
    assert report["states"]["biological"] == T0
    if "forbidden_x" in expected:
        assert report["states"]["executable"] not in expected["forbidden_x"]
    if "executable" in expected:
        assert report["states"]["executable"] == expected["executable"]
    if "spectral" in expected:
        assert report["states"]["spectral"] == expected["spectral"]
    if "spectral_in" in expected:
        assert report["states"]["spectral"] in expected["spectral_in"]


def test_fault_matrix_covers_exactly_the_22_rev29_planted_release_faults() -> None:
    assert set(FAULT_EXPECTATIONS) == {
        "dropped_band",
        "edge_clamp",
        "reordered_bands",
        "id_ignored_metadata_sorted",
        "wavelength_nm_micron_mismatch",
        "missing_mandatory_fwhm",
        "invalid_present_srf",
        "routed_response_mismatch",
        "target_srf_outside_support",
        "claimed_wavelength_input_ignored",
        "undeclared_wavelength_input_used",
        "claimed_fwhm_input_ignored",
        "undeclared_fwhm_input_used",
        "routed_domain_outside_model_valid_range",
        "target_invariant_output_on_challenges",
        "stochastic_inference",
        "radiometric_quantity_mismatch",
        "undeclared_normalization",
        "undeclared_spatial_reduction",
        "context_dependent_undeclared_tap",
        "hidden_resampling_or_extrapolation",
        "duplicate_band_ids",
    }


def test_edge_clamp_is_an_undeclared_executable_alias_not_a_declared_bad_route(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="edge_clamp")
    declared_route = bundle.read_json("route")
    assert declared_route["matrix"] == [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]
    output_dir = tmp_path / "edge-clamp-secret-mix"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4
    assert "edge_clamp" in _fault_codes(report)
    assert report["states"]["spectral"] == S3
    assert report["states"]["executable"] not in {X2, X3}
    assert report["canaries"]["C1_declared_tap_agreement"]["status"] == "FAIL"
    assert report["canaries"]["C6_edge_alias"]["finding"] == "CLAMP_ALIAS_CONFIRMED"


@pytest.mark.parametrize("adapter", ["numpy-linear-v1", "subprocess-npz-v1"])
@pytest.mark.parametrize(
    ("fault", "sensor_mutator", "expected_exit"),
    [
        ("wavelength_nm_micron_mismatch", _micron_mismatch, 2),
        ("missing_mandatory_fwhm", _missing_spectral_width, 2),
        ("invalid_present_srf", _invalid_present_srf, 4),
        ("target_srf_outside_support", _outside_srf, 4),
        ("radiometric_quantity_mismatch", _radiometric_mismatch, 2),
        ("duplicate_band_ids", _duplicate_band_id, 2),
    ],
)
def test_adapter_independent_contract_faults_are_detected_for_both_adapters(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    adapter: str,
    fault: str,
    sensor_mutator: Callable[[dict[str, object]], None],
    expected_exit: int,
) -> None:
    bundle = bundle_factory(adapter=adapter, sensor_mutator=sensor_mutator)
    output_dir = tmp_path / f"cross-{adapter}-{fault}"
    completed = _cli(bundle, output_dir)

    assert completed.returncode == expected_exit, (completed.stdout, completed.stderr)
    if expected_exit == 2:
        assert fault in completed.stderr
    else:
        assert fault in _fault_codes(_report(output_dir))


def test_no_effect_challenges_emit_only_the_bounded_nonfault_diagnostic(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        artifact_mutator=_zero_numpy_output_weights,
        sensor_mutator=_vary_fwhm_metadata,
    )
    output_dir = tmp_path / "numpy-prior-only"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4
    assert "prior_only_executable" not in _fault_codes(report)
    diagnostic = "NO_TARGET_EFFECT_OBSERVED_ABOVE_FROZEN_THRESHOLD_ON_CHALLENGES"
    assert report["canaries"]["C5_target_neutral"]["bounded_target_effect_diagnostic"] == diagnostic
    assert report["facts"]["bounded_target_effect_diagnostic"] == diagnostic
    assert report["states"]["executable"] == X2
    assert report["states"]["spectral"] == S3


def test_negative_declared_route_weight_is_bundle_error_without_report(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def negative_route(route: dict[str, object]) -> None:
        route["matrix"][0][0] = -0.1

    output_dir = tmp_path / "negative-declared-route"
    completed = _cli(bundle_factory(route_mutator=negative_route), output_dir)

    assert completed.returncode == 2
    assert "route.json.matrix" in completed.stderr
    assert "exactly zero or at least" in completed.stderr
    assert not output_dir.exists()


def test_all_zero_declared_route_row_completes_as_route_fault_and_s0(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def zero_row_with_required_provenance_elsewhere(route: dict[str, object]) -> None:
        route["operation"] = "nonnegative_row_normalized_linear_resampling"
        route["matrix"] = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]

    output_dir = tmp_path / "all-zero-declared-route-row"
    completed = _cli(
        bundle_factory(route_mutator=zero_row_with_required_provenance_elsewhere),
        output_dir,
    )
    report = _report(output_dir)

    assert completed.returncode == 4
    assert "dropped_band" in _fault_codes(report)
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == S0


def test_numpy_reference_detects_a_declared_but_absent_wavelength_path(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(wavelength_conditioned=True)
    output_dir = tmp_path / "numpy-wavelength-ignored"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4
    assert "claimed_wavelength_input_ignored" in _fault_codes(report)
    assert report["states"]["executable"] != X3
    assert report["states"]["spectral"] == S3


def test_numpy_reference_c1_is_not_tautologically_derived_from_declared_route(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(artifact_mutator=_reorder_numpy_artifact_route)
    output_dir = tmp_path / "numpy-artifact-route-mismatch"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4
    assert "reordered_bands" in _fault_codes(report)
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == S3
    assert report["canaries"]["C1_declared_tap_agreement"]["status"] == "FAIL"


def test_subprocess_fixture_owns_normalization_and_exposes_undeclared_mismatch(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="wrong_normalization",
        normalization_offset=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        normalization_scale=np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float64),
    )
    output_dir = tmp_path / "subprocess-wrong-normalization"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4
    assert "undeclared_normalization" in _fault_codes(report)
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == S3
    assert report["canaries"]["C1_declared_tap_agreement"]["status"] == "FAIL"


def test_radiometric_mismatch_is_invalid_bundle_without_completed_axes(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(sensor_mutator=_radiometric_mismatch)
    output_dir = tmp_path / "radiometric-static-contract"
    completed = _cli(bundle, output_dir)

    assert completed.returncode == 2
    assert "radiometric_quantity_mismatch" in completed.stderr
    assert not output_dir.exists()


def test_numpy_artifact_spatial_operation_mismatch_is_c1_fault_not_fallback(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(artifact_mutator=_mismatch_numpy_spatial_operation)
    output_dir = tmp_path / "numpy-spatial-mismatch"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4, (completed.stdout, completed.stderr)
    assert report["canaries"]["C1_declared_tap_agreement"]["status"] == "FAIL"
    assert report["states"]["executable"] not in {X2, X3}
    assert "hidden_resampling_or_extrapolation" in _fault_codes(report)
    assert report["facts"]["declared_spatial_operation"] == "none"
    assert report["facts"]["adapter_spatial_operation"] == "mean"


def test_decoy_supplier_tap_cannot_be_misreported_as_independent_attestation(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="decoy_hidden_use")
    output_dir = tmp_path / "decoy"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4
    assert report["facts"]["route_assurance"] == "SUPPLIER_REPORTED_TAP"
    assert report["canaries"]["C1_declared_tap_agreement"]["status"] == "PASS"
    assert report["canaries"]["C2_value_dependence"]["bands"]["t950"]["dependent"] is True
    assert "hidden_resampling_or_extrapolation" in _fault_codes(report)
    limitations = " ".join(report["limitations"]).lower()
    assert "decoy" in limitations or "cannot attest" in limitations


def test_wrong_route_cannot_retain_s3_by_matching_raw_target_srf_set(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def wrong_pairing(route: dict[str, object]) -> None:
        route["matrix"] = [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]

    bundle = bundle_factory(route_mutator=wrong_pairing)
    output_dir = tmp_path / "effective-srf"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4
    assert report["states"]["spectral"] == S1
    assert report["states"]["spectral"] != S3
    support_rows = report["facts"]["spectral_support_by_model_channel"]
    assert support_rows["m450"]["paired_target_band_ids"] == ["t550"]
    assert support_rows["m450"]["normalized_l1"] > 0.05


@pytest.mark.parametrize(
    ("mode", "fault", "canary"),
    [
        ("dropped_band", "dropped_band", "C1_declared_tap_agreement"),
        ("decoy_hidden_use", "hidden_resampling_or_extrapolation", "C2_value_dependence"),
        ("positional_only", "reordered_bands", "C4_order"),
        ("edge_clamp", "edge_clamp", "C6_edge_alias"),
    ],
)
def test_executable_canary_faults_do_not_erase_clean_declared_spectral_s3(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    mode: str,
    fault: str,
    canary: str,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode=mode)
    output_dir = tmp_path / f"orthogonal-{canary}"
    completed = _cli(bundle, output_dir)
    report = _report(output_dir)

    assert completed.returncode == 4, (completed.stdout, completed.stderr)
    assert fault in _fault_codes(report)
    assert report["states"]["spectral"] == S3
    assert report["states"]["biological"] == T0
