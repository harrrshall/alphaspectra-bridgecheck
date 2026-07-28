from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import BundleCase, BundleFactory, run_and_read


EXPECTED_STATES = {
    "executable": "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES",
    "spectral": "S3_SRF_WITHIN_DECLARED_SUPPORT",
    "biological": "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED",
}
EXPECTED_CANARIES = {
    "C0_replay",
    "C1_declared_tap_agreement",
    "C2_value_dependence",
    "C3_wavelength_dependence",
    "C3_fwhm_dependence",
    "C4_order",
    "C5_target_neutral",
    "C6_edge_alias",
}
REQUIRED_OUTPUTS = {
    "report.json",
    "report.html",
    "route.csv",
    "canary_outputs.npz",
    "manifest.sha256",
}


def _exit_code(result: object, report: dict[str, object]) -> int:
    value = getattr(result, "exit_code", report.get("exit_code"))
    assert isinstance(value, int)
    return value


@pytest.mark.parametrize(
    ("adapter", "assurance"),
    [
        ("numpy-linear-v1", "INSTRUMENT_CONTROLLED_REFERENCE"),
        ("subprocess-npz-v1", "SUPPLIER_REPORTED_TAP"),
    ],
)
def test_clean_bundle_reaches_exact_orthogonal_states_and_all_canaries(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    adapter: str,
    assurance: str,
) -> None:
    bundle = bundle_factory(adapter=adapter)
    result, report = run_and_read(bundle, tmp_path / f"out-{adapter}")

    assert _exit_code(result, report) == 0
    assert report["states"] == EXPECTED_STATES
    assert set(report["canaries"]) == EXPECTED_CANARIES
    assert report["faults"] == []
    assert report["facts"]["route_assurance"] == assurance
    assert report["facts"]["overall_conformance_fault"] is False
    assert report["states"]["biological"] == "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED"

    limitations = " ".join(report["limitations"])
    assert "not a certificate" in limitations.lower()
    assert "biological" in limitations.lower()
    if adapter == "subprocess-npz-v1":
        assert "SUPPLIER_REPORTED_TAP" in json.dumps(report)
        assert report["facts"]["subprocess_dependency_state"] == (
            "SUBPROCESS_DEPENDENCIES_UNATTESTED"
        )
        assert "SUBPROCESS_DEPENDENCIES_UNATTESTED" in json.dumps(report)
        assert "decoy" in limitations.lower() or "cannot attest" in limitations.lower()


def test_three_runs_are_byte_identical_for_every_required_output(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(adapter="numpy-linear-v1")
    output_dirs = [tmp_path / f"replay-{index}" for index in range(3)]
    for output_dir in output_dirs:
        result, report = run_and_read(bundle, output_dir)
        assert _exit_code(result, report) == 0

    assert {path.name for path in output_dirs[0].iterdir()} == REQUIRED_OUTPUTS
    for filename in sorted(REQUIRED_OUTPUTS):
        payloads = [(root / filename).read_bytes() for root in output_dirs]
        assert payloads[0] == payloads[1] == payloads[2], filename


def test_output_manifest_pins_exact_output_bytes(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    run_and_read(clean_numpy_bundle, output_dir)

    listed: dict[str, str] = {}
    for line in (output_dir / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        listed[filename.lstrip(" *")] = digest
    assert set(listed) == REQUIRED_OUTPUTS - {"manifest.sha256"}
    for filename, expected in listed.items():
        actual = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()
        assert actual == expected


def test_c1_chunks_and_unselected_c2_shift_candidates_are_not_retained(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.canaries as canaries

    monkeypatch.setattr(canaries, "C1_CHUNK_MAX_ROWS", 2, raising=False)
    monkeypatch.setattr(canaries, "C1_CHUNK_MAX_FLOAT64_PROBE_BYTES", 80, raising=False)
    output_dir = tmp_path / "streamed-canary-artifacts"
    result, report = run_and_read(clean_numpy_bundle, output_dir)

    assert _exit_code(result, report) == 0
    with np.load(output_dir / "canary_outputs.npz", allow_pickle=False) as archive:
        keys = set(archive.files)
        assert not {key for key in keys if "candidate" in key or "chunk" in key}
        c2 = np.asarray(archive["c2_value_dependence_output"])
    assert c2.shape == (5, 20)


def test_route_csv_is_machine_readable_and_formula_safe(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    run_and_read(clean_numpy_bundle, output_dir)

    rows = list(csv.DictReader((output_dir / "route.csv").open(newline="", encoding="utf-8")))
    assert rows
    assert {row["model_channel_id"] for row in rows} == {"m450", "m550", "m650", "m750"}
    assert all(
        not value.startswith(("=", "+", "-", "@", "\t", "\r"))
        for row in rows
        for value in row.values()
        if value is not None
    )


def test_report_json_rejects_nonstandard_nan_during_independent_parse(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    run_and_read(clean_numpy_bundle, output_dir)

    def reject_constant(token: str) -> None:
        raise AssertionError(f"non-finite JSON token {token}")

    parsed = json.loads(
        (output_dir / "report.json").read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )
    assert parsed["states"] == EXPECTED_STATES


def test_full_srf_amplitude_scaling_is_invariant(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    scales = [0.01, 0.3, 7.0, 91.0, 1_000.0]

    def scale_sensor_srf(sensor: dict[str, object]) -> None:
        for band, scale in zip(sensor["target_bands"], scales):
            band["srf"]["responses"] = [scale * value for value in band["srf"]["responses"]]

    bundle = bundle_factory(sensor_mutator=scale_sensor_srf)
    result, report = run_and_read(bundle, tmp_path / "scaled-srf")

    assert _exit_code(result, report) == 0
    assert report["states"] == EXPECTED_STATES


def test_subthreshold_route_weight_is_not_silently_promoted_to_required_dependence(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def duplicate_edge_srf(sensor: dict[str, object]) -> None:
        band = sensor["target_bands"][4]
        band["center_wavelength"] = 750.0
        band["fwhm"] = 20.0
        band["srf"] = {
            "wavelengths": [730.0, 740.0, 750.0, 760.0, 770.0],
            "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
            "wavelength_unit": "nm",
        }

    def add_tiny_weight(route: dict[str, object]) -> None:
        route["operation"] = "nonnegative_row_normalized_linear_resampling"
        route["matrix"][3] = [
            0.0,
            0.0,
            0.0,
            0.99993896484375,
            0.00006103515625,
        ]

    bundle = bundle_factory(sensor_mutator=duplicate_edge_srf, route_mutator=add_tiny_weight)
    result, report = run_and_read(bundle, tmp_path / "tiny-weight")

    assert _exit_code(result, report) == 0
    assert report["states"]["executable"] == "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES"
    assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"
    required = report["facts"]["required_dependence_target_band_ids"]
    assert required == ["t450", "t550", "t650", "t750"]
    assert "t950" not in required


@pytest.mark.parametrize("adapter", ["numpy-linear-v1", "subprocess-npz-v1"])
def test_nonidentity_affine_pre_core_path_is_verified_exactly_for_both_adapters(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    adapter: str,
) -> None:
    offset = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    scale = np.asarray([0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    bundle = bundle_factory(
        adapter=adapter,
        normalization_offset=offset,
        normalization_scale=scale,
    )
    result, report = run_and_read(bundle, tmp_path / f"affine-{adapter}")

    assert _exit_code(result, report) == 0
    assert report["states"] == EXPECTED_STATES
    c1 = report["canaries"]["C1_declared_tap_agreement"]
    assert c1["baseline_max_abs_error"] <= 0.000001
    assert c1["basis_max_abs_error"] <= 0.000001


@pytest.mark.parametrize("adapter", ["numpy-linear-v1", "subprocess-npz-v1"])
def test_rank_four_mean_is_the_only_declared_spatial_reduction(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    adapter: str,
) -> None:
    base = np.asarray(
        [[0.08 + 0.84 * (((row * (band * 2 + 3) + band * 7) % 23) / 22.0) for band in range(5)] for row in range(20)],
        dtype=np.float64,
    )
    spatial_offsets = np.asarray([[-0.02, 0.0], [0.01, 0.02]], dtype=np.float64)
    probes = base[:, :, None, None] + spatial_offsets[None, None, :, :]

    def mean_route(route: dict[str, object]) -> None:
        route["spatial_operation"] = "mean"

    bundle = bundle_factory(adapter=adapter, probes=probes, route_mutator=mean_route)
    result, report = run_and_read(bundle, tmp_path / f"mean-{adapter}")

    assert _exit_code(result, report) == 0
    assert report["states"] == EXPECTED_STATES


def test_positive_route_mass_below_and_above_one_block_x_symmetrically_but_preserve_s3(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    reports: dict[float, dict[str, object]] = {}
    for factor in (0.9, 1.1):
        def scaled_route(route: dict[str, object], factor: float = factor) -> None:
            route["operation"] = "nonnegative_row_normalized_linear_resampling"
            route["matrix"] = (
                factor * np.asarray(route["matrix"], dtype=np.float64)
            ).tolist()

        def widened_model_domain(model: dict[str, object]) -> None:
            model["valid_range"] = [0.0, 2.0]

        result, report = run_and_read(
            bundle_factory(
                route_mutator=scaled_route,
                model_mutator=widened_model_domain,
            ),
            tmp_path / f"positive-route-mass-{factor}",
        )
        reports[factor] = report

        assert _exit_code(result, report) == 4
        assert report["states"]["executable"] == "X1_REPLAY_STABLE_ON_PROBES"
        assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"
        assert report["states"]["biological"] == "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED"
        c1 = report["canaries"]["C1_declared_tap_agreement"]
        assert c1["status"] == "PASS"
        assert c1["recovered_route_max_abs_error"] == 0.0
        assert c1["raw_route_recovery_conditioned"] is True
        assert report["facts"]["overall_conformance_fault"] is True

    assert reports[0.9]["facts"]["spectral_support_by_model_channel"] == (
        reports[1.1]["facts"]["spectral_support_by_model_channel"]
    )


def test_512_by_512_route_audit_is_compact_in_json_and_streamed_per_cell_in_csv(
    bundle_factory: BundleFactory,
) -> None:
    band_count = 512
    model_ids = [f"m{index:03d}" for index in range(band_count)]
    target_ids = [f"t{index:03d}" for index in range(band_count)]

    def resize_sensor(sensor: dict[str, object]) -> None:
        template = copy.deepcopy(sensor["target_bands"][0])
        sensor["target_bands"] = [
            {**copy.deepcopy(template), "id": identifier}
            for identifier in target_ids
        ]

    def resize_model(model: dict[str, object]) -> None:
        template = copy.deepcopy(model["model_channels"][0])
        model["model_channels"] = [
            {**copy.deepcopy(template), "id": identifier}
            for identifier in model_ids
        ]
        model["normalization"]["offset"] = [0.0] * band_count
        model["normalization"]["scale"] = [1.0] * band_count
        model["required_dependence_target_band_ids"] = [target_ids[0]]

    def resize_route(route: dict[str, object]) -> None:
        route["model_channel_ids"] = model_ids
        route["target_band_ids"] = target_ids
        route["matrix"] = [
            [int(model_index == target_index) for target_index in range(band_count)]
            for model_index in range(band_count)
        ]

    def resize_artifact(artifact_path: Path) -> None:
        with np.load(artifact_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        arrays["output_weights"] = np.zeros(band_count, dtype=np.float64)
        np.savez(artifact_path, **arrays)

    probes = np.broadcast_to(
        np.linspace(0.1, 0.9, 20, dtype=np.float64)[:, None],
        (20, band_count),
    ).copy()
    bundle_case = bundle_factory(
        probes=probes,
        sensor_mutator=resize_sensor,
        model_mutator=resize_model,
        route_mutator=resize_route,
        artifact_mutator=resize_artifact,
        normalization_offset=np.zeros(band_count, dtype=np.float64),
        normalization_scale=np.ones(band_count, dtype=np.float64),
    )
    route_path = bundle_case.file_path("route")
    route_path.write_text(
        json.dumps(bundle_case.read_json("route"), separators=(",", ":")),
        encoding="utf-8",
    )
    bundle_case.refresh_hash("route")

    from bandtrace.bundle import load_bundle
    from bandtrace.canonical import canonical_json_bytes
    from bandtrace.report import compact_route_audit, render_route_csv, route_audit_rows

    bundle = load_bundle(bundle_case.root)
    route_audit = compact_route_audit(bundle)
    assert set(route_audit) == {
        "model_channel_ids",
        "target_band_ids",
        "declared_weight",
        "declared_weight_float64_hex",
        "declared_weight_is_strictly_positive",
        "declared_target_column_is_exactly_zero",
    }
    assert len(route_audit["model_channel_ids"]) == band_count
    assert len(route_audit["target_band_ids"]) == band_count
    for key in (
        "declared_weight",
        "declared_weight_float64_hex",
        "declared_weight_is_strictly_positive",
    ):
        matrix = route_audit[key]
        assert len(matrix) == band_count
        assert all(len(row) == band_count for row in matrix)
        assert all(not isinstance(cell, dict) for row in matrix for cell in row)
    assert len(route_audit["declared_target_column_is_exactly_zero"]) == band_count
    assert "model_channel_index" not in canonical_json_bytes(route_audit).decode("utf-8")

    rows = route_audit_rows(bundle)
    assert not isinstance(rows, list)
    route_csv = render_route_csv(rows)
    assert route_csv.count(b"\n") == band_count * band_count + 1
    parsed = csv.DictReader(route_csv.decode("utf-8").splitlines())
    first = next(parsed)
    last = first
    row_count = 1
    for last in parsed:
        row_count += 1
    assert row_count == band_count * band_count
    assert first["model_channel_id"] == "m000"
    assert first["target_band_id"] == "t000"
    assert first["declared_weight_float64_hex"] == float(1.0).hex()
    assert first["declared_weight_is_strictly_positive"] == "true"
    assert last["model_channel_id"] == "m511"
    assert last["target_band_id"] == "t511"
    assert last["declared_target_column_is_exactly_zero"] == "false"


def test_html_is_escaped_bounded_summary_with_linkage_hashes_but_no_route_matrix(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    _, report = run_and_read(bundle_factory(), tmp_path / "html-source")
    sentinel = "FULL_ROUTE_MATRIX_SENTINEL_<script>&"
    report["facts"]["route_audit"]["model_channel_ids"][0] = sentinel
    report["limitations"].append("<script>alert('escaped')</script>&")
    linkage = {
        "report.json": "1" * 64,
        "route.csv": "2" * 64,
        "canary_outputs.npz": "3" * 64,
    }

    from bandtrace.report import render_html

    rendered = render_html(report, linkage).decode("utf-8")
    assert sentinel not in rendered
    assert '"route_audit"' not in rendered
    assert '"declared_weight"' not in rendered
    assert "<script>alert('escaped')</script>" not in rendered
    assert "&lt;script&gt;alert(&#x27;escaped&#x27;)&lt;/script&gt;&amp;" in rendered
    for filename, digest in linkage.items():
        assert filename in rendered
        assert digest in rendered


def test_center_fwhm_only_is_capped_at_approximate_support_s2(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def remove_full_model_srfs(model: dict[str, object]) -> None:
        for band in model["model_channels"]:
            band.pop("srf")

    def remove_full_target_srfs(sensor: dict[str, object]) -> None:
        for band in sensor["target_bands"]:
            band.pop("srf")

    bundle = bundle_factory(
        model_mutator=remove_full_model_srfs,
        sensor_mutator=remove_full_target_srfs,
    )
    result, report = run_and_read(bundle, tmp_path / "fwhm-only")

    assert _exit_code(result, report) == 0
    assert report["states"]["executable"] == "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES"
    assert report["states"]["spectral"] == "S2_APPROX_WITHIN_SUPPORT"
    assert report["states"]["spectral"] != "S3_SRF_WITHIN_DECLARED_SUPPORT"


@pytest.mark.parametrize("side", ["model", "sensor"])
def test_full_srf_never_waives_mandatory_fwhm(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    side: str,
) -> None:
    def remove_model_fwhm(model: dict[str, object]) -> None:
        model["model_channels"][0].pop("fwhm")
        model["model_channels"][0].pop("fwhm_unit")

    def remove_sensor_fwhm(sensor: dict[str, object]) -> None:
        sensor["target_bands"][0].pop("fwhm")
        sensor["target_bands"][0].pop("fwhm_unit")

    bundle = bundle_factory(
        model_mutator=remove_model_fwhm if side == "model" else None,
        sensor_mutator=remove_sensor_fwhm if side == "sensor" else None,
    )
    from bandtrace.bundle import load_bundle
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="FWHM|fwhm|missing_mandatory_fwhm"):
        load_bundle(bundle.root)


def test_present_but_invalid_full_srf_is_s0_not_s2_or_s3(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def invalidate_srf(sensor: dict[str, object]) -> None:
        sensor["target_bands"][0]["srf"] = {
            "wavelengths": [440.0, 450.0, 460.0],
            "responses": [0.5, 1.0, 0.5],
            "wavelength_unit": "nm",
        }

    bundle = bundle_factory(sensor_mutator=invalidate_srf)
    result, report = run_and_read(bundle, tmp_path / "invalid-srf")

    assert _exit_code(result, report) == 4
    assert report["states"]["spectral"] == "S0_SUPPORT_UNRESOLVED"
    assert "invalid_present_srf" in {fault["code"] for fault in report["faults"]}


def test_explicit_matched_outside_route_preserves_x_evidence_but_is_s1(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def route_outside(route: dict[str, object]) -> None:
        route["matrix"][3] = [0.0, 0.0, 0.0, 0.0, 1.0]

    def require_routed_targets(model: dict[str, object]) -> None:
        model["required_dependence_target_band_ids"] = [
            "t450",
            "t550",
            "t650",
            "t950",
        ]

    bundle = bundle_factory(route_mutator=route_outside, model_mutator=require_routed_targets)
    result, report = run_and_read(bundle, tmp_path / "declared-outside-route")

    assert _exit_code(result, report) == 4
    assert report["states"] == {
        "executable": "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES",
        "spectral": "S1_OUTSIDE_DECLARED_SUPPORT",
        "biological": "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED",
    }
    assert report["canaries"]["C1_declared_tap_agreement"]["status"] == "PASS"
    assert "target_srf_outside_support" in {fault["code"] for fault in report["faults"]}
