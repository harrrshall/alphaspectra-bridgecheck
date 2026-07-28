from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import (
    BundleCase,
    BundleFactory,
    TARGET_BAND_IDS,
    varied_probes,
    run_and_read,
)


X1 = "X1_REPLAY_STABLE_ON_PROBES"
X2 = "X2_DECLARED_TAP_MATCHES_ROUTE_ON_CHALLENGES"
X3 = "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES"
INCONCLUSIVE = "INCONCLUSIVE_INSUFFICIENT_EXCITATION"
INCONCLUSIVE_METADATA = "INCONCLUSIVE_INSUFFICIENT_METADATA_EXCITATION"


def _exit_code(result: object, report: dict[str, Any]) -> int:
    value = getattr(result, "exit_code", report.get("exit_code"))
    assert isinstance(value, int)
    return value


def _fault_codes(report: dict[str, Any]) -> set[str]:
    return {str(fault["code"]) for fault in report["faults"]}


def _load(bundle: BundleCase):
    from bandtrace.bundle import load_bundle

    return load_bundle(bundle.root)


class _RecordingNumpyAdapter:
    """Record canary requests while delegating responses to the pinned NumPy artifact."""

    def __init__(self, bundle: Any) -> None:
        from bandtrace.adapters import NumpyLinearAdapter

        self._inner = NumpyLinearAdapter(bundle)
        self.requests: list[Any] = []

    @property
    def assurance(self) -> str:
        return self._inner.assurance

    @property
    def trust_state(self) -> str:
        return self._inner.trust_state

    @property
    def invocations(self) -> int:
        return self._inner.invocations

    @property
    def wall_seconds(self) -> float:
        return self._inner.wall_seconds

    def invoke(self, request: Any):
        from bandtrace.adapters import Invocation

        copied = Invocation(
            probes=np.array(request.probes, copy=True),
            target_band_ids=tuple(request.target_band_ids),
            wavelength_nm=np.array(request.wavelength_nm, copy=True),
            fwhm_nm=np.array(request.fwhm_nm, copy=True),
        )
        self.requests.append(copied)
        return self._inner.invoke(request)

    def close(self) -> None:
        self._inner.close()


def _run_recorded(bundle_case: BundleCase, *, seed: bytes = bytes(range(32))):
    from bandtrace.canaries import run_canaries

    bundle = _load(bundle_case)
    adapter = _RecordingNumpyAdapter(bundle)
    result = run_canaries(bundle, adapter, seed)
    return result, adapter


def _basis_requests(adapter: _RecordingNumpyAdapter) -> list[Any]:
    # Bundle probes always contain at least sixteen rows. C1 is the only canary whose
    # synthetic requests contain one row per target band and may therefore be chunked below N=16.
    candidates = [request for request in adapter.requests if request.probes.shape[0] < 16]
    result: list[Any] = []
    for request in candidates:
        probes = request.probes
        if probes.ndim == 4:
            first_cell = np.broadcast_to(probes[:, :, :1, :1], probes.shape)
            if not np.array_equal(probes, first_cell):
                continue
        result.append(request)
    return result


def _spatial_challenge_requests(adapter: _RecordingNumpyAdapter) -> list[Any]:
    result: list[Any] = []
    for request in adapter.requests:
        probes = request.probes
        if probes.shape[0] != 4 or probes.ndim != 4:
            continue
        first_cell = np.broadcast_to(probes[:, :, :1, :1], probes.shape)
        if not np.array_equal(probes, first_cell):
            result.append(request)
    return result


def _declare_spatial_mean(route: dict[str, object]) -> None:
    route["spatial_operation"] = "mean"


def _rank4_probes(*, height: int = 2, width: int = 3) -> np.ndarray:
    base = varied_probes()[:, :, None, None]
    offsets = np.linspace(-0.03, 0.03, height * width, dtype=np.float64).reshape(
        height, width
    )
    probes = base + offsets[None, None, :, :]
    assert np.all((probes >= 0.0) & (probes <= 1.0))
    return np.asarray(probes, dtype=np.float64)


def _independent_canary_seed(base_seed: bytes, canary_id: str) -> bytes:
    encoded = canary_id.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"bandtrace-canary-v1\x00")
    digest.update(base_seed)
    digest.update(len(encoded).to_bytes(2, "big"))
    digest.update(encoded)
    return digest.digest()


def _two_block_probes() -> np.ndarray:
    column = np.asarray([0.2] * 10 + [0.8] * 10, dtype=np.float64)
    return np.column_stack([column] * len(TARGET_BAND_IDS))


def _unique_tie_probes() -> np.ndarray:
    column = np.linspace(0.1, 0.9, 20, dtype=np.float64)
    return np.column_stack([np.roll(column, index) for index in range(len(TARGET_BAND_IDS))])


def _joint_cancellation_probes() -> np.ndarray:
    base = np.linspace(-0.08, 0.08, 20, dtype=np.float64)
    d0 = base
    d1 = np.roll(base, 3)
    d2 = np.roll(base, 7)
    d3 = -(0.7 * d0 - 0.5 * d1 + 0.35 * d2) / 0.9
    deviations = np.column_stack([d0, d1, d2, d3, np.roll(base, 11)])
    probes = 0.5 + deviations
    assert np.all((probes >= 0.0) & (probes <= 1.0))
    assert np.max(np.abs(deviations[:, :4] @ np.asarray([0.7, -0.5, 0.35, 0.9]))) < 1e-14
    return probes


def _make_t450_raw_mass_outside(sensor: dict[str, object]) -> None:
    band = sensor["target_bands"][0]
    band["fwhm"] = 100.0
    band["srf"] = {
        "wavelengths": [350.0, 400.0, 450.0, 500.0, 550.0],
        "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
        "wavelength_unit": "nm",
    }


def _vary_target_fwhm(sensor: dict[str, object]) -> None:
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


def _half_numpy_route_with_maximal_output_weights(artifact_path: Path) -> None:
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["route_matrix"] = 0.5 * np.asarray(arrays["route_matrix"], dtype=np.float64)
    arrays["output_weights"] = np.full(4, 1e12, dtype=np.float64)
    np.savez(artifact_path, **arrays)


def _scale_numpy_route_just_beyond_raw_tolerance(artifact_path: Path) -> None:
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["route_matrix"] = (
        np.asarray(arrays["route_matrix"], dtype=np.float64) * 1.0000011
    )
    np.savez(artifact_path, **arrays)


def _shrink_numpy_bundle_to_one_band(bundle: BundleCase) -> None:
    sensor = bundle.read_json("sensor")
    sensor["target_bands"] = sensor["target_bands"][:1]
    bundle.rewrite_json("sensor", sensor)

    route = bundle.read_json("route")
    route["model_channel_ids"] = route["model_channel_ids"][:1]
    route["target_band_ids"] = route["target_band_ids"][:1]
    route["matrix"] = [[1.0]]
    bundle.rewrite_json("route", route)

    artifact_path = bundle.file_path("artifact")
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["route_matrix"] = np.asarray([[1.0]], dtype=np.float64)
    arrays["target_band_ids"] = arrays["target_band_ids"][:1]
    arrays["normalization_offset"] = arrays["normalization_offset"][:1]
    arrays["normalization_scale"] = arrays["normalization_scale"][:1]
    arrays["output_weights"] = arrays["output_weights"][:1]
    np.savez(artifact_path, **arrays)
    bundle.refresh_hash("artifact")

    model = bundle.read_json("model")
    model["model_channels"] = model["model_channels"][:1]
    model["normalization"]["offset"] = model["normalization"]["offset"][:1]
    model["normalization"]["scale"] = model["normalization"]["scale"][:1]
    model["required_dependence_target_band_ids"] = model[
        "required_dependence_target_band_ids"
    ][:1]
    bundle.rewrite_json("model", model)

    probe_path = bundle.file_path("probes")
    with np.load(probe_path, allow_pickle=False) as archive:
        probes = np.asarray(archive["probes"], dtype=np.float64)[:, :1]
        ids = np.asarray(archive["target_band_ids"])[:1]
    np.savez(probe_path, probes=probes, target_band_ids=ids)
    bundle.refresh_hash("probes")


def test_c0_uses_complete_max_minus_min_replay_range(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="replay_range_straddles_first",
    )
    result, report = run_and_read(bundle, tmp_path / "replay-range")

    assert _exit_code(result, report) == 4
    assert report["states"]["executable"] == "X0_NO_EXECUTABLE_OBSERVATION"
    assert report["canaries"]["C0_replay"]["normalized_max_jitter"] > 0.0000001
    assert "stochastic_inference" in _fault_codes(report)


def test_c0_applies_max_minus_min_replay_guard_to_reported_tap_too(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="tap_replay_range_straddles_first",
    )
    result, report = run_and_read(bundle, tmp_path / "tap-replay-range")

    assert _exit_code(result, report) == 4
    assert report["states"]["executable"] == "X0_NO_EXECUTABLE_OBSERVATION"
    c0 = report["canaries"]["C0_replay"]
    assert c0["normalized_max_jitter"] == 0.0
    assert c0["normalized_max_tap_jitter"] > 0.0000001
    assert "stochastic_inference" in _fault_codes(report)


def test_c1_baseline_error_aggregates_all_three_replay_taps(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="third_replay_tap_mismatch",
        normalization_offset=np.full(4, 1.0, dtype=np.float64),
        normalization_scale=np.full(4, 0.01, dtype=np.float64),
    )
    result, report = run_and_read(bundle, tmp_path / "third-replay-tap")

    assert _exit_code(result, report) == 4
    assert report["canaries"]["C0_replay"]["status"] == "PASS"
    c1 = report["canaries"]["C1_declared_tap_agreement"]
    assert c1["status"] == "FAIL"
    assert c1["baseline_max_abs_error"] > 0.000001
    assert report["states"]["executable"] not in {X2, X3}


def test_c1_rank2_basis_uses_exact_farthest_endpoint_with_lower_tie(
    bundle_factory: BundleFactory,
) -> None:
    result, adapter = _run_recorded(bundle_factory())
    requests = _basis_requests(adapter)
    observed = np.concatenate([request.probes for request in requests], axis=0)
    expected = np.full((5, 5), 0.5, dtype=np.float64)
    expected[np.arange(5), np.arange(5)] = 0.0

    assert np.array_equal(observed, expected)
    assert result.canaries["C1_declared_tap_agreement"]["status"] == "PASS"


def test_c1_rank4_basis_is_constant_over_the_original_spatial_grid(
    bundle_factory: BundleFactory,
) -> None:
    probes = np.broadcast_to(varied_probes()[:, :, None, None], (20, 5, 3, 4)).copy()

    def declare_mean(route: dict[str, object]) -> None:
        route["spatial_operation"] = "mean"

    result, adapter = _run_recorded(
        bundle_factory(probes=probes, route_mutator=declare_mean)
    )
    requests = _basis_requests(adapter)
    assert requests
    assert all(request.probes.shape[2:] == (3, 4) for request in requests)
    observed = np.concatenate([request.probes for request in requests], axis=0)
    expected = np.full((5, 5, 3, 4), 0.5, dtype=np.float64)
    for index in range(5):
        expected[index, index, :, :] = 0.0

    assert np.array_equal(observed, expected)
    assert result.canaries["C1_declared_tap_agreement"]["status"] == "PASS"


def test_numpy_rank4_mean_routes_after_spatial_reduction_without_nmhw_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import bandtrace.adapters as adapters
    from bandtrace.adapters import Invocation, NumpyLinearAdapter

    probes = np.linspace(0.01, 0.99, 3 * 1 * 4 * 5, dtype=np.float64).reshape(
        3, 1, 4, 5
    )
    channels = 512
    route = np.linspace(0.25, 1.0, channels, dtype=np.float64)[:, None]
    offset = np.linspace(0.0, 0.5, channels, dtype=np.float64)
    scale = np.linspace(0.5, 2.0, channels, dtype=np.float64)
    output_weights = np.linspace(-0.1, 0.1, channels, dtype=np.float64)
    output_bias = np.float64(0.125)
    expected_spatial_mean = np.mean(probes, axis=(2, 3), dtype=np.float64)
    expected_routed = np.einsum(
        "nb,mb->nm", expected_spatial_mean, route, optimize=False
    )
    expected_pre_core = (expected_routed - offset[None, :]) / scale[None, :]
    expected_output = expected_pre_core @ output_weights + output_bias

    bundle = SimpleNamespace(
        numpy_artifact={
            "route_matrix": route,
            "target_band_ids": np.asarray(["t0"]),
            "normalization_offset": offset,
            "normalization_scale": scale,
            "output_weights": output_weights,
            "output_bias": np.asarray(output_bias, dtype=np.float64),
            "spatial_operation": np.asarray("mean"),
        },
        model=SimpleNamespace(channels=tuple(object() for _ in range(channels))),
        sensor=SimpleNamespace(bands=(SimpleNamespace(id="t0"),)),
    )

    original_einsum = np.einsum
    observed_signatures: list[str] = []

    def reject_channel_expanded_spatial_einsum(
        signature: str, *operands: np.ndarray, **kwargs: object
    ) -> np.ndarray:
        observed_signatures.append(signature)
        if signature == "nbhw,mb->nmhw":
            raise AssertionError("rank-4 reference path allocated N*M*H*W")
        return original_einsum(signature, *operands, **kwargs)

    monkeypatch.setattr(adapters.np, "einsum", reject_channel_expanded_spatial_einsum)
    adapter = NumpyLinearAdapter(bundle)
    response = adapter.invoke(
        Invocation(
            probes=probes,
            target_band_ids=("t0",),
            wavelength_nm=np.asarray([450.0], dtype=np.float64),
            fwhm_nm=np.asarray([20.0], dtype=np.float64),
        )
    )

    assert "nbhw,mb->nmhw" not in observed_signatures
    assert response.pre_core.shape == (3, channels)
    assert response.pre_core == pytest.approx(
        expected_pre_core, rel=0.0, abs=0.000001
    )
    assert response.output == pytest.approx(
        expected_output, rel=0.0, abs=0.000001
    )


def test_c1_rank4_mean_uses_exact_complementary_and_asymmetric_spatial_challenges(
    bundle_factory: BundleFactory,
) -> None:
    result, adapter = _run_recorded(
        bundle_factory(
            probes=_rank4_probes(),
            route_mutator=_declare_spatial_mean,
        )
    )
    requests = _spatial_challenge_requests(adapter)

    assert len(requests) == 1
    observed = requests[0].probes
    expected = np.full((4, 5, 2, 3), 0.5, dtype=np.float64)
    flat = expected.reshape(4, 5, -1)
    flat[0, :, 0::2] = 0.0
    flat[1, :, 1::2] = 0.0
    levels = np.asarray([0.5, 0.375, 0.0], dtype=np.float64)
    for cell in range(flat.shape[2]):
        flat[2, :, cell] = levels[cell % levels.size]
        flat[3, :, cell] = levels[::-1][cell % levels.size]
    assert np.array_equal(observed, expected)

    c1 = result.canaries["C1_declared_tap_agreement"]
    assert c1["status"] == "PASS"
    assert c1["spatial_challenge_rows"] == 4
    assert c1["spatial_challenge_status"] == "PASS"
    assert c1["spatial_challenge_max_abs_error"] <= 0.000001
    assert c1["single_spatial_cell_all_singleton_preserving_reducers_equivalent"] is False
    assert c1["two_spatial_cell_mean_median_midrange_equivalence"] is False
    expected_pre_core = np.mean(expected[:, :4], axis=(2, 3), dtype=np.float64)
    np.testing.assert_allclose(
        result.arrays["c1_spatial_challenge_pre_core"],
        expected_pre_core,
        rtol=0.0,
        atol=1e-15,
    )
    coefficients = np.asarray([0.7, -0.5, 0.35, 0.9], dtype=np.float64)
    np.testing.assert_allclose(
        result.arrays["c1_spatial_challenge_output"],
        expected_pre_core @ coefficients + 0.125,
        rtol=0.0,
        atol=1e-15,
    )


def test_c1_one_spatial_cell_reports_singleton_reducer_equivalence(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    probes = np.asarray(varied_probes()[:, :, None, None], dtype=np.float64)
    result, report = run_and_read(
        bundle_factory(probes=probes, route_mutator=_declare_spatial_mean),
        tmp_path / "one-spatial-cell-equivalence",
    )

    assert _exit_code(result, report) == 0
    c1 = report["canaries"]["C1_declared_tap_agreement"]
    assert c1["status"] == "PASS"
    assert c1["spatial_challenge_rows"] == 4
    assert c1["spatial_challenge_status"] == "PASS"
    assert c1["spatial_challenge_max_abs_error"] <= 0.000001
    assert c1["single_spatial_cell_all_singleton_preserving_reducers_equivalent"] is True
    assert c1["two_spatial_cell_mean_median_midrange_equivalence"] is False
    assert (
        "With one spatial cell, all singleton-preserving spatial reducers are "
        "observationally equivalent on every possible challenge."
    ) in report["limitations"]


def test_c1_two_spatial_cells_report_mean_median_midrange_equivalence(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    result, report = run_and_read(
        bundle_factory(
            probes=_rank4_probes(height=1, width=2),
            route_mutator=_declare_spatial_mean,
        ),
        tmp_path / "two-spatial-cell-equivalence",
    )

    assert _exit_code(result, report) == 0
    c1 = report["canaries"]["C1_declared_tap_agreement"]
    assert c1["status"] == "PASS"
    assert c1["single_spatial_cell_all_singleton_preserving_reducers_equivalent"] is False
    assert c1["two_spatial_cell_mean_median_midrange_equivalence"] is True
    assert (
        "With exactly two spatial cells, arithmetic mean, median defined as the "
        "average of the middle pair, and midrange are mathematically identical "
        "and cannot be distinguished by BandTrace."
    ) in report["limitations"]


def test_docs_and_source_do_not_claim_two_cell_median_distinguishability() -> None:
    product_document = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "bandtrace"
        / "normative"
        / "BANDTRACE_PRODUCT.md"
    ).read_text(encoding="utf-8")
    canary_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "bandtrace"
        / "canaries.py"
    ).read_text(encoding="utf-8")
    audit_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "bandtrace"
        / "audit.py"
    ).read_text(encoding="utf-8")

    assert (
        "two cells the usual even-sample median, midrange and mean are identical "
        "for every input"
    ) in product_document
    assert "rather than claimed distinguishable" in product_document
    assert "two_spatial_cell_mean_median_midrange_equivalence" in canary_source
    assert "two_spatial_cell_mean_midrange_equivalence" not in canary_source
    assert (
        "arithmetic mean, median defined as the average of the middle pair, and "
        "midrange are mathematically identical and cannot be distinguished"
    ) in audit_source


@pytest.mark.parametrize(
    "mode",
    [
        "first_pixel_reduction",
        "max_reduction",
        "midrange_reduction",
        "median_reduction",
        "cropped_mean_reduction",
    ],
)
def test_c1_rank4_wrong_reduction_is_explicit_route_fault_without_erasing_s(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    mode: str,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode=mode,
        probes=_rank4_probes(),
        route_mutator=_declare_spatial_mean,
    )
    result, report = run_and_read(bundle, tmp_path / mode)

    assert _exit_code(result, report) == 4
    assert "undeclared_spatial_reduction" in _fault_codes(report)
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"
    c1 = report["canaries"]["C1_declared_tap_agreement"]
    assert c1["status"] == "FAIL"
    assert c1["spatial_challenge_rows"] == 4
    assert c1["spatial_challenge_status"] == "FAIL"
    assert c1["spatial_challenge_max_abs_error"] > 0.000001
    assert c1["single_spatial_cell_all_singleton_preserving_reducers_equivalent"] is False
    assert c1["two_spatial_cell_mean_median_midrange_equivalence"] is False


def test_c1_basis_is_chunked_by_both_row_and_float64_byte_caps(
    bundle_factory: BundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.canaries as canaries

    monkeypatch.setattr(canaries, "C1_CHUNK_MAX_ROWS", 2, raising=False)
    monkeypatch.setattr(canaries, "C1_CHUNK_MAX_FLOAT64_PROBE_BYTES", 80, raising=False)
    result, adapter = _run_recorded(bundle_factory())
    requests = _basis_requests(adapter)

    assert [request.probes.shape[0] for request in requests] == [2, 2, 1]
    assert all(request.probes.nbytes <= 80 for request in requests)
    assert sum(request.probes.shape[0] for request in requests) == 5
    assert adapter.invocations <= 2 * len(TARGET_BAND_IDS) + 12
    assert result.canaries["C1_declared_tap_agreement"]["status"] == "PASS"


def test_c1_scale_that_previously_hid_tap_span_is_rejected_at_bundle_boundary(
    bundle_factory: BundleFactory,
) -> None:
    from bandtrace.bundle import load_bundle
    from bandtrace.errors import BundleError

    bundle = bundle_factory(normalization_scale=np.full(4, 1_000_000.0, dtype=np.float64))
    with pytest.raises(BundleError, match=r"normalization.scale|\[0.01, 2\]"):
        load_bundle(bundle.root)


def test_c1_raw_route_recovery_rejects_half_route_large_scale_evasion(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        normalization_scale=np.full(4, 2.0, dtype=np.float64),
        artifact_mutator=_half_numpy_route_with_maximal_output_weights,
    )
    result, report = run_and_read(bundle, tmp_path / "half-route-scale-evasion")

    assert _exit_code(result, report) == 4
    c1 = report["canaries"]["C1_declared_tap_agreement"]
    assert c1["baseline_max_abs_error"] > 0.000001
    assert c1["neutral_max_abs_error"] > 0.000001
    assert c1["basis_max_abs_error"] > 0.000001
    assert c1["raw_route_recovery_conditioned"] is True
    assert c1["raw_offset_recovery_conditioned"] is True
    assert c1["recovered_route_max_abs_error"] == pytest.approx(
        0.5, rel=0.0, abs=1e-12
    )
    assert c1["status"] == "FAIL"
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"
    assert "undeclared_normalization" in _fault_codes(report)


def test_c1_large_offset_and_scale_ill_conditioning_witness_is_rejected_at_boundary(
    bundle_factory: BundleFactory,
) -> None:
    from bandtrace.bundle import load_bundle
    from bandtrace.errors import BundleError

    bundle = bundle_factory(
        normalization_offset=np.full(4, 1e12, dtype=np.float64),
        normalization_scale=np.full(4, 100.0, dtype=np.float64),
    )
    with pytest.raises(BundleError, match=r"normalization.scale|\[0.01, 2\]"):
        load_bundle(bundle.root)


def test_c1_extreme_range_ill_conditioning_witness_is_rejected_at_boundary(
    bundle_factory: BundleFactory,
) -> None:
    low = np.float64(-1e12)
    high = np.nextafter(np.nextafter(low, np.float64(np.inf)), np.float64(np.inf))
    probes = np.full((20, 5), low, dtype=np.float64)
    probes[1::2, :] = high

    def extreme_model(model: dict[str, object]) -> None:
        model["valid_range"] = [float(low), float(high)]

    def extreme_sensor(sensor: dict[str, object]) -> None:
        sensor["valid_range"] = [float(low), float(high)]
        for band in sensor["target_bands"]:
            band["neutral_value"] = float(low)

    bundle = bundle_factory(
        probes=probes,
        model_mutator=extreme_model,
        sensor_mutator=extreme_sensor,
        normalization_offset=np.full(4, 1e12, dtype=np.float64),
        normalization_scale=np.full(4, 100.0, dtype=np.float64),
    )
    from bandtrace.bundle import load_bundle
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match=r"reflectance raw domain|\[-0.1, 2\]"):
        load_bundle(bundle.root)


def test_c1_compares_observed_raw_route_directly_to_declaration_without_two_tolerances(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        normalization_offset=np.full(4, 0.5, dtype=np.float64),
        normalization_scale=np.full(4, 2.0, dtype=np.float64),
        artifact_mutator=_scale_numpy_route_just_beyond_raw_tolerance,
    )
    raw_result, _ = _run_recorded(bundle)
    result, report = run_and_read(bundle, tmp_path / "direct-raw-route-comparison")

    assert _exit_code(result, report) == 4
    c1 = report["canaries"]["C1_declared_tap_agreement"]
    assert c1["raw_recovery_conditioned"] is True
    assert c1["raw_route_recovery_conditioning_max_abs_error"] <= 0.000001
    assert c1["raw_offset_recovery_conditioning_max_abs_error"] <= 0.000001
    raw_c1 = raw_result.canaries["C1_declared_tap_agreement"]
    assert raw_c1["recovered_route_max_abs_error"] > 0.000001
    assert c1["status"] == "FAIL"
    assert report["states"]["executable"] not in {X2, X3}
    assert "undeclared_normalization" in _fault_codes(report)


def test_c2_selects_the_unique_maximally_exciting_shift(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(probes=_two_block_probes())
    result, report = run_and_read(bundle, tmp_path / "c2-max-shift")

    assert _exit_code(result, report) == 4
    bands = report["canaries"]["C2_value_dependence"]["bands"]
    for identifier in TARGET_BAND_IDS:
        assert bands[identifier]["selected_shift"] == 10
        assert bands[identifier]["input_excitation_fraction"] == 1.0
    c4 = report["canaries"]["C4_order"]
    assert c4["status"] == "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION"
    assert c4["id_only_binding"]["excitation_adequate"] is False
    assert report["states"]["executable"] == X1
    assert "reordered_bands" in _fault_codes(report)


def test_c2_seed_selects_among_sorted_maximal_shift_ties(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(probes=_unique_tie_probes())
    manifest = bundle.manifest()
    from bandtrace.canonical import derive_seed

    base_seed = derive_seed(
        model_hash=manifest["files"]["model"]["sha256"],
        sensor_hash=manifest["files"]["sensor"]["sha256"],
        probe_hash=manifest["files"]["probes"]["sha256"],
        route_hash=manifest["files"]["route"]["sha256"],
        policy_id="bandtrace-0.1-r29",
    )
    result, report = run_and_read(bundle, tmp_path / "c2-seeded-ties")

    assert _exit_code(result, report) == 0
    bands = report["canaries"]["C2_value_dependence"]["bands"]
    for identifier in TARGET_BAND_IDS:
        subseed = _independent_canary_seed(base_seed, f"C2_value_dependence:{identifier}")
        expected_shift = 1 + int.from_bytes(subseed[:8], "big") % 19
        assert bands[identifier]["selected_shift"] == expected_shift
        assert bands[identifier]["input_excitation_fraction"] == 1.0


def test_required_c2_under_excitation_is_exit_four_with_explicit_fault(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(probes=np.full((20, 5), 0.4, dtype=np.float64))
    result, report = run_and_read(bundle, tmp_path / "required-under-excited")

    assert _exit_code(result, report) == 4
    assert report["states"]["executable"] == X1
    assert report["canaries"]["C2_value_dependence"]["status"] == INCONCLUSIVE
    c4 = report["canaries"]["C4_order"]
    assert c4["status"] == "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION"
    assert c4["id_only_binding"]["excitation_adequate"] is False
    assert "required_band_insufficient_excitation" in _fault_codes(report)
    assert any(
        fault["code"] == "required_band_insufficient_excitation"
        and fault["severity"] == 1
        and fault["axis"] == "dependence"
        for fault in report["faults"]
    )


def test_positive_tiny_nonrequired_route_weight_is_diagnostic_not_hidden(
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
    result, report = run_and_read(bundle, tmp_path / "positive-tiny-route")

    assert _exit_code(result, report) == 0
    row = report["canaries"]["C2_value_dependence"]["bands"]["t950"]
    assert row["required"] is False
    assert row["dependent"] is True
    assert row["aggregate_absolute_route_weight"] == 0.00006104
    assert "hidden_resampling_or_extrapolation" not in _fault_codes(report)


def test_exactly_zero_route_with_observed_dependence_is_hidden_fault(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="decoy_hidden_use")
    result, report = run_and_read(bundle, tmp_path / "zero-route-hidden-use")

    assert _exit_code(result, report) == 4
    row = report["canaries"]["C2_value_dependence"]["bands"]["t950"]
    assert row["aggregate_absolute_route_weight"] == 0.0
    assert row["dependent"] is True
    assert "hidden_resampling_or_extrapolation" in _fault_codes(report)


@pytest.mark.parametrize(
    ("mode", "canary_id"),
    [
        ("c2_context_dependent_tap", "C2_value_dependence"),
        ("c3_context_dependent_tap", "C3_wavelength_dependence"),
    ],
)
def test_every_c2_c3_returned_tap_must_match_the_declared_mutated_transform(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    mode: str,
    canary_id: str,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode=mode)
    result, report = run_and_read(bundle, tmp_path / mode)

    assert _exit_code(result, report) == 4
    assert "context_dependent_undeclared_tap" in _fault_codes(report)
    assert report["canaries"][canary_id]["pre_core_max_abs_error"] > 0.000001
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"


def test_c3_wavelength_primary_holds_fwhm_fixed_and_fwhm_only_cannot_pass(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="fwhm_aware",
        wavelength_conditioned=True,
        fwhm_conditioned=True,
        sensor_mutator=_vary_target_fwhm,
    )
    result, report = run_and_read(bundle, tmp_path / "fwhm-only-c3")

    assert _exit_code(result, report) == 4
    assert report["states"]["executable"] != X3
    wavelength = report["canaries"]["C3_wavelength_dependence"]
    fwhm = report["canaries"]["C3_fwhm_dependence"]
    assert wavelength["status"] == "NO_WAVELENGTH_DEPENDENCE_OBSERVED_ON_PROBES"
    assert fwhm["output_dependence_fraction"] >= 0.20
    assert fwhm["status"] == "PASS"
    assert "claimed_wavelength_input_ignored" in _fault_codes(report)


def test_c3_nonuniform_magnitude_vectors_are_exact_and_clip_to_frozen_domains(
    bundle_factory: BundleFactory,
) -> None:
    centers = np.asarray(
        [100.0, 200.0, 500.0, 99_999.0, 100_000.0], dtype=np.float64
    )

    def boundary_metadata(sensor: dict[str, object]) -> None:
        for band, center in zip(sensor["target_bands"], centers):
            band["center_wavelength"] = float(center)
            band["wavelength_unit"] = "nm"
            band["fwhm"] = 20.0
            band["fwhm_unit"] = "nm"
            band.pop("srf", None)

    base_seed = bytes(range(32))
    probes = varied_probes()
    result, adapter = _run_recorded(
        bundle_factory(sensor_mutator=boundary_metadata),
        seed=base_seed,
    )
    original_fwhm = np.full(len(TARGET_BAND_IDS), 20.0, dtype=np.float64)
    metadata_requests = [
        request
        for request in adapter.requests
        if request.target_band_ids == tuple(TARGET_BAND_IDS)
        and np.array_equal(request.probes, probes)
    ]
    wavelength_requests = [
        request
        for request in metadata_requests
        if np.array_equal(request.fwhm_nm, original_fwhm)
        and not np.array_equal(request.wavelength_nm, centers)
    ]
    fwhm_requests = [
        request
        for request in metadata_requests
        if np.array_equal(request.wavelength_nm, centers)
        and not np.array_equal(request.fwhm_nm, original_fwhm)
    ]

    from bandtrace.canonical import c3_rank_amplitudes

    wavelength_amplitudes = c3_rank_amplitudes(
        base_seed,
        "C3_wavelength_dependence",
        tuple(TARGET_BAND_IDS),
    )
    expected_wavelength_plus = np.clip(
        centers + wavelength_amplitudes, 100.0, 100_000.0
    )
    expected_wavelength_minus = np.clip(
        centers - wavelength_amplitudes, 100.0, 100_000.0
    )
    fwhm_amplitudes = c3_rank_amplitudes(
        base_seed,
        "C3_fwhm_dependence",
        tuple(TARGET_BAND_IDS),
    )
    factors = 1.0 + 0.01 * fwhm_amplitudes
    expected_fwhm_plus = np.clip(original_fwhm * factors, 1.0, 50_000.0)
    expected_fwhm_minus = np.clip(original_fwhm / factors, 1.0, 50_000.0)

    assert len(wavelength_requests) == 3
    # The FWHM rotation is still invoked and reported, but all submitted FWHM
    # values are equal so its request bytes are indistinguishable from baseline.
    assert len(fwhm_requests) == 2
    assert any(
        np.array_equal(request.wavelength_nm, expected_wavelength_plus)
        for request in wavelength_requests
    )
    assert any(
        np.array_equal(request.wavelength_nm, expected_wavelength_minus)
        for request in wavelength_requests
    )
    assert any(
        np.array_equal(request.fwhm_nm, expected_fwhm_plus)
        for request in fwhm_requests
    )
    assert any(
        np.array_equal(request.fwhm_nm, expected_fwhm_minus)
        for request in fwhm_requests
    )
    assert expected_wavelength_plus[-1] == 100_000.0
    assert expected_wavelength_minus[0] == 100.0
    assert all(
        np.all((request.wavelength_nm >= 100.0) & (request.wavelength_nm <= 100_000.0))
        for request in wavelength_requests
    )
    assert all(
        np.all((request.fwhm_nm >= 1.0) & (request.fwhm_nm <= 50_000.0))
        for request in fwhm_requests
    )
    wavelength_report = result.canaries["C3_wavelength_dependence"]
    fwhm_report = result.canaries["C3_fwhm_dependence"]
    assert wavelength_report["status"] == (
        "NO_WAVELENGTH_DEPENDENCE_OBSERVED_ON_PROBES"
    )
    expected_mutation_names = [
        "CYCLIC_ROTATION",
        "NONUNIFORM_MAGNITUDE_INCREASE",
        "NONUNIFORM_MAGNITUDE_DECREASE",
    ]
    assert [row["mutation"] for row in wavelength_report["mutations"]] == (
        expected_mutation_names
    )
    assert [row["mutation"] for row in fwhm_report["mutations"]] == (
        expected_mutation_names
    )
    wavelength_by_mutation = {
        row["mutation"]: row["submitted_metadata_by_target_band_id"]
        for row in wavelength_report["mutations"]
    }
    fwhm_by_mutation = {
        row["mutation"]: row["submitted_metadata_by_target_band_id"]
        for row in fwhm_report["mutations"]
    }
    assert wavelength_by_mutation["NONUNIFORM_MAGNITUDE_INCREASE"] == dict(
        zip(TARGET_BAND_IDS, expected_wavelength_plus.tolist())
    )
    assert wavelength_by_mutation["NONUNIFORM_MAGNITUDE_DECREASE"] == dict(
        zip(TARGET_BAND_IDS, expected_wavelength_minus.tolist())
    )
    assert fwhm_by_mutation["NONUNIFORM_MAGNITUDE_INCREASE"] == dict(
        zip(TARGET_BAND_IDS, expected_fwhm_plus.tolist())
    )
    assert fwhm_by_mutation["NONUNIFORM_MAGNITUDE_DECREASE"] == dict(
        zip(TARGET_BAND_IDS, expected_fwhm_minus.tolist())
    )
    assert wavelength_report["rank_amplitudes_by_target_band_id"] == dict(
        zip(TARGET_BAND_IDS, wavelength_amplitudes.tolist())
    )
    assert fwhm_report["rank_amplitudes_by_target_band_id"] == dict(
        zip(TARGET_BAND_IDS, fwhm_amplitudes.tolist())
    )
    assert wavelength_report["ranked_target_band_ids"] == [
        identifier
        for _, identifier in sorted(zip(wavelength_amplitudes, TARGET_BAND_IDS))
    ]
    assert fwhm_report["ranked_target_band_ids"] == [
        identifier for _, identifier in sorted(zip(fwhm_amplitudes, TARGET_BAND_IDS))
    ]
    assert set(result.arrays) >= {
        "c3_wavelength_dependence_output",
        "c3_wavelength_rotation_output",
        "c3_wavelength_magnitude_increase_output",
        "c3_wavelength_magnitude_decrease_output",
        "c3_fwhm_dependence_output",
        "c3_fwhm_rotation_output",
        "c3_fwhm_magnitude_increase_output",
        "c3_fwhm_magnitude_decrease_output",
    }
    assert result.arrays["c3_wavelength_dependence_output"].shape == (3, 20)
    assert result.arrays["c3_fwhm_dependence_output"].shape == (3, 20)


def test_claimed_wavelength_conditioning_with_one_band_uses_magnitude_excitation(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(wavelength_conditioned=True)
    _shrink_numpy_bundle_to_one_band(bundle)
    result, report = run_and_read(bundle, tmp_path / "one-band-metadata")

    assert _exit_code(result, report) == 4
    c3 = report["canaries"]["C3_wavelength_dependence"]
    assert c3["status"] == "NO_WAVELENGTH_DEPENDENCE_OBSERVED_ON_PROBES"
    assert c3["metadata_excitation_fraction"] >= 0.20
    assert "claimed_wavelength_input_ignored" in _fault_codes(report)
    assert "claimed_wavelength_input_inconclusive" not in _fault_codes(report)
    assert report["states"]["executable"] == X2
    assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"
    assert report["canaries"]["C4_order"]["status"] == "NOT_APPLICABLE_SINGLE_BAND"


def test_nonuniform_c3_supports_bounded_no_target_effect_diagnostic(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="prior_only")
    result, report = run_and_read(bundle, tmp_path / "prior-only-c3-inconclusive")

    assert _exit_code(result, report) == 4
    assert report["canaries"]["C3_fwhm_dependence"]["status"] == (
        "NO_FWHM_DEPENDENCE_OBSERVED_ON_PROBES"
    )
    assert report["states"]["executable"] == X2
    assert "prior_only_executable" not in _fault_codes(report)
    assert "target_invariant_output_on_challenges" in _fault_codes(report)
    diagnostic = "NO_TARGET_EFFECT_OBSERVED_ABOVE_FROZEN_THRESHOLD_ON_CHALLENGES"
    assert report["canaries"]["C5_target_neutral"]["bounded_target_effect_diagnostic"] == diagnostic
    assert report["facts"]["bounded_target_effect_diagnostic"] == diagnostic


def test_c5_joint_neutral_cancellation_does_not_lower_valid_x_or_prove_prior_only(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(probes=_joint_cancellation_probes())
    result, report = run_and_read(bundle, tmp_path / "neutral-cancellation")

    assert _exit_code(result, report) == 0
    assert report["states"]["executable"] == X3
    c2_bands = report["canaries"]["C2_value_dependence"]["bands"]
    assert all(c2_bands[identifier]["dependent"] for identifier in TARGET_BAND_IDS[:4])
    assert report["canaries"]["C5_target_neutral"]["status"] == (
        "NO_JOINT_TARGET_NEUTRAL_EFFECT_OBSERVED"
    )
    assert "prior_only_executable" not in _fault_codes(report)
    assert report["canaries"]["C5_target_neutral"]["bounded_target_effect_diagnostic"] == (
        "NOT_ESTABLISHED"
    )
    assert report["facts"]["bounded_target_effect_diagnostic"] == "NOT_ESTABLISHED"


def test_c6_reports_every_raw_mass_outside_pair_even_when_center_is_inside(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(sensor_mutator=_make_t450_raw_mass_outside)
    result, report = run_and_read(bundle, tmp_path / "c6-every-pair")

    assert _exit_code(result, report) == 4
    assert report["states"]["executable"] == X3
    assert report["states"]["spectral"] == "S1_OUTSIDE_DECLARED_SUPPORT"
    pairs = report["canaries"]["C6_edge_alias"]["pairs"]
    by_outside = {pair["outside_target_band_id"]: pair for pair in pairs}
    assert set(by_outside) == {"t450", "t950"}
    assert by_outside["t450"]["edge_target_band_id"] == "t550"
    assert by_outside["t950"]["edge_target_band_id"] == "t750"
    assert all(
        set(pair) == {
            "outside_target_band_id",
            "edge_target_band_id",
            "finding",
            "declared_columns_equal",
            "reported_tap_columns_equal",
            "reported_tap_comparison_status",
        }
        for pair in pairs
    )


def test_c6_finds_an_alias_on_the_second_of_multiple_outside_pairs(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="edge_clamp",
        sensor_mutator=_make_t450_raw_mass_outside,
    )
    result, report = run_and_read(bundle, tmp_path / "c6-second-pair-alias")

    assert _exit_code(result, report) == 4
    pairs = report["canaries"]["C6_edge_alias"]["pairs"]
    by_outside = {pair["outside_target_band_id"]: pair for pair in pairs}
    assert set(by_outside) == {"t450", "t950"}
    assert by_outside["t950"]["finding"] == "CLAMP_ALIAS_CONFIRMED"
    assert "edge_clamp" in _fault_codes(report)


def test_exactly_declared_c6_collision_is_diagnostic_and_preserves_x(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def declare_collision(route: dict[str, object]) -> None:
        route["operation"] = "nonnegative_row_normalized_linear_resampling"
        route["matrix"][3] = [0.0, 0.0, 0.0, 0.5, 0.5]

    bundle = bundle_factory(route_mutator=declare_collision)
    result, report = run_and_read(bundle, tmp_path / "declared-c6-collision")

    assert _exit_code(result, report) == 4  # spectral S1, not an executable-axis failure
    assert report["states"]["executable"] == X3
    assert report["states"]["spectral"] == "S1_OUTSIDE_DECLARED_SUPPORT"
    assert "edge_clamp" not in _fault_codes(report)
    pair = report["canaries"]["C6_edge_alias"]["pairs"][0]
    assert pair["outside_target_band_id"] == "t950"
    assert pair["edge_target_band_id"] == "t750"
    assert pair["finding"] == "CLAMP_ALIAS_CONFIRMED"
    assert pair["declared_columns_equal"] is True
    assert pair["reported_tap_columns_equal"] is True


def test_c6_reported_collision_is_invariant_to_declared_normalization_scale(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    scales = np.asarray([0.2, 2.0, 0.05, 1.5], dtype=np.float64)
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="edge_clamp",
        normalization_scale=scales,
    )
    result, report = run_and_read(bundle, tmp_path / "c6-scale-invariant")

    assert _exit_code(result, report) == 4
    pair = next(
        row
        for row in report["canaries"]["C6_edge_alias"]["pairs"]
        if row["outside_target_band_id"] == "t950"
    )
    assert pair["finding"] == "CLAMP_ALIAS_CONFIRMED"
    assert pair["reported_tap_columns_equal"] is True
    assert "edge_clamp" in _fault_codes(report)


def test_c6_does_not_confuse_proportional_but_unequal_columns_with_a_collision(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def unequal_columns(route: dict[str, object]) -> None:
        route["operation"] = "nonnegative_row_normalized_linear_resampling"
        route["matrix"][3] = [0.0, 0.0, 0.0, 0.75, 0.25]

    result, report = run_and_read(
        bundle_factory(route_mutator=unequal_columns),
        tmp_path / "c6-proportional-not-equal",
    )

    assert _exit_code(result, report) == 4  # S1 only
    assert report["states"]["executable"] == X3
    assert "edge_clamp" not in _fault_codes(report)
    pair = report["canaries"]["C6_edge_alias"]["pairs"][0]
    assert pair["declared_columns_equal"] is False
    assert pair["reported_tap_columns_equal"] is False
    assert pair["finding"] != "CLAMP_ALIAS_CONFIRMED"


def test_wrong_c5_neutral_tap_is_a_route_fault_not_dependence_evidence(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="wrong_neutral_tap",
    )
    result, report = run_and_read(bundle, tmp_path / "wrong-neutral-tap")

    assert _exit_code(result, report) == 4
    c5 = report["canaries"]["C5_target_neutral"]
    assert c5["pre_core_max_abs_error"] > 0.000001
    assert report["states"]["executable"] not in {X2, X3}
    assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"
    assert "undeclared_normalization" in _fault_codes(report)


def test_all_zero_present_srf_completes_s0_and_makes_c6_inconclusive(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def zero_srf(sensor: dict[str, object]) -> None:
        sensor["target_bands"][4]["srf"]["responses"] = [0.0] * 5

    result, report = run_and_read(
        bundle_factory(sensor_mutator=zero_srf),
        tmp_path / "zero-present-srf",
    )

    assert _exit_code(result, report) == 4
    assert report["states"]["spectral"] == "S0_SUPPORT_UNRESOLVED"
    assert "invalid_present_srf" in _fault_codes(report)
    c6 = report["canaries"]["C6_edge_alias"]
    assert c6["finding"] == "INCONCLUSIVE_SUPPORT_UNRESOLVED"
    assert c6["pairs"] == []


def test_c6_partial_invalid_srf_keeps_valid_outside_pair_visible(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def invalidate_one_supported_target(sensor: dict[str, object]) -> None:
        sensor["target_bands"][2]["srf"]["responses"] = [0.0] * 5

    result, report = run_and_read(
        bundle_factory(sensor_mutator=invalidate_one_supported_target),
        tmp_path / "c6-partial-invalid-srf",
    )

    assert _exit_code(result, report) == 4
    assert report["states"]["spectral"] == "S0_SUPPORT_UNRESOLVED"
    assert "invalid_present_srf" in _fault_codes(report)
    c6 = report["canaries"]["C6_edge_alias"]
    assert c6["finding"] == "INCONCLUSIVE_SUPPORT_UNRESOLVED"
    assert c6["unresolved_target_band_ids"] == ["t650"]
    pairs = {
        row["outside_target_band_id"]: row
        for row in c6["pairs"]
    }
    assert set(pairs) == {"t950"}
    assert pairs["t950"]["edge_target_band_id"] == "t750"
    assert pairs["t950"]["finding"] == "NO_ALIAS_OBSERVED_ON_PROBES"


def test_subprocess_runtime_invocation_cap_is_twice_bands_plus_twelve(
    bundle_factory: BundleFactory,
) -> None:
    from bandtrace.adapters import build_adapter

    bundle = _load(bundle_factory(adapter="subprocess-npz-v1"))
    adapter = build_adapter(bundle)
    try:
        assert getattr(adapter, "_max_invocations") == 2 * len(TARGET_BAND_IDS) + 12
    finally:
        adapter.close()


def test_same_text_id_is_legal_across_model_and_target_roles_with_distinct_srfs(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    shared_id = "shared450"

    def mutate_model(model: dict[str, object]) -> None:
        model["model_channels"][0]["id"] = shared_id
        model["required_dependence_target_band_ids"][0] = shared_id

    def mutate_sensor(sensor: dict[str, object]) -> None:
        band = sensor["target_bands"][0]
        band["id"] = shared_id
        band["fwhm"] = 100.0
        band["srf"] = {
            "wavelengths": [350.0, 400.0, 450.0, 500.0, 550.0],
            "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
            "wavelength_unit": "nm",
        }

    def mutate_route(route: dict[str, object]) -> None:
        route["model_channel_ids"][0] = shared_id
        route["target_band_ids"][0] = shared_id

    bundle = bundle_factory(
        model_mutator=mutate_model,
        sensor_mutator=mutate_sensor,
        route_mutator=mutate_route,
    )
    result, report = run_and_read(bundle, tmp_path / "same-id-different-role")

    assert _exit_code(result, report) == 4
    assert report["states"]["executable"] == X3
    assert report["states"]["spectral"] == "S1_OUTSIDE_DECLARED_SUPPORT"
    assert "duplicate_band_ids" not in _fault_codes(report)
    row = report["facts"]["spectral_support_by_model_channel"][shared_id]
    assert row["paired_target_band_ids"] == [shared_id]
    assert row["normalized_l1"] > 0.05


def test_every_positive_route_weight_enters_effective_srf_even_below_dependence_threshold(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def add_tiny_outside_weight(route: dict[str, object]) -> None:
        route["operation"] = "nonnegative_row_normalized_linear_resampling"
        route["matrix"][3] = [
            0.0,
            0.0,
            0.0,
            0.99993896484375,
            0.00006103515625,
        ]

    bundle = bundle_factory(route_mutator=add_tiny_outside_weight)
    result, report = run_and_read(bundle, tmp_path / "tiny-effective-srf")

    assert _exit_code(result, report) == 0
    assert report["states"]["spectral"] == "S3_SRF_WITHIN_DECLARED_SUPPORT"
    row = report["facts"]["spectral_support_by_model_channel"]["m750"]
    assert row["paired_target_band_ids"] == ["t750", "t950"]
    assert 0.0 < row["normalized_l1"] <= 0.05
    assert 0.99 <= row["available_mass"] < 1.0
    assert "hidden_resampling_or_extrapolation" not in _fault_codes(report)


def test_arbitrary_complete_id_tied_route_axes_and_probe_columns_canonicalize(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    model_permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    target_permutation = np.asarray([4, 2, 0, 3, 1], dtype=np.int64)

    def permute_route_axes(route: dict[str, object]) -> None:
        matrix = np.asarray(route["matrix"], dtype=np.float64)
        model_ids = list(route["model_channel_ids"])
        target_ids = list(route["target_band_ids"])
        route["model_channel_ids"] = [model_ids[index] for index in model_permutation]
        route["target_band_ids"] = [target_ids[index] for index in target_permutation]
        route["matrix"] = matrix[np.ix_(model_permutation, target_permutation)].tolist()

    def keep_artifact_rows_in_model_contract_order(artifact_path: Path) -> None:
        with np.load(artifact_path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
        arrays["route_matrix"] = arrays["route_matrix"][np.argsort(model_permutation), :]
        np.savez(artifact_path, **arrays)

    bundle = bundle_factory(
        route_mutator=permute_route_axes,
        artifact_mutator=keep_artifact_rows_in_model_contract_order,
    )
    with np.load(bundle.file_path("probes"), allow_pickle=False) as archive:
        probe_values = np.array(archive["probes"], copy=True)
        probe_ids = np.array(archive["target_band_ids"], copy=True)
    np.savez(
        bundle.file_path("probes"),
        probes=probe_values[:, target_permutation],
        target_band_ids=probe_ids[target_permutation],
    )
    bundle.refresh_hash("probes")

    result, report = run_and_read(bundle, tmp_path / "arbitrary-id-tied-axes")

    assert _exit_code(result, report) == 0
    assert report["states"]["executable"] == X3
    assert report["canaries"]["C1_declared_tap_agreement"]["status"] == "PASS"
    assert report["canaries"]["C4_order"]["status"] == "PASS"
    assert "reordered_bands" not in _fault_codes(report)


def test_required_artifact_report_contains_counts_and_budgets_but_no_observed_timing(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory()
    result, report = run_and_read(bundle, tmp_path / "no-wall-time")

    assert _exit_code(result, report) == 0
    assert isinstance(report["facts"]["adapter_invocation_count"], int)
    assert report["facts"]["adapter_invocation_count"] <= 2 * len(TARGET_BAND_IDS) + 12
    assert report["facts"]["configured_time_budgets"]

    def keys(value: object) -> list[str]:
        if isinstance(value, dict):
            return [str(key) for key in value] + [item for child in value.values() for item in keys(child)]
        if isinstance(value, list):
            return [item for child in value for item in keys(child)]
        return []

    forbidden_fragments = ("wall_seconds", "wall_clock", "elapsed", "duration")
    assert not [
        key
        for key in keys(report)
        if any(fragment in key.lower() for fragment in forbidden_fragments)
    ]
