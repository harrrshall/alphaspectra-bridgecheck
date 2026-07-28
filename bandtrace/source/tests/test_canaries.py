from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import BundleFactory, run_and_read


X1 = "X1_REPLAY_STABLE_ON_PROBES"
X2 = "X2_DECLARED_TAP_MATCHES_ROUTE_ON_CHALLENGES"
X3 = "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES"
S3 = "S3_SRF_WITHIN_DECLARED_SUPPORT"
INCONCLUSIVE = "INCONCLUSIVE_INSUFFICIENT_EXCITATION"
INCONCLUSIVE_METADATA = "INCONCLUSIVE_INSUFFICIENT_METADATA_EXCITATION"


def test_wavelength_conditioned_clean_adapter_must_show_probe_local_metadata_dependence(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="wavelength_aware",
        wavelength_conditioned=True,
    )
    result, report = run_and_read(bundle, tmp_path / "wavelength-aware")

    assert result.exit_code == 0
    assert report["states"]["executable"] == X3
    assert report["states"]["spectral"] == S3
    assert report["canaries"]["C3_wavelength_dependence"]["status"] == "PASS"


def test_fwhm_conditioned_clean_adapter_must_show_field_only_metadata_dependence(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def vary_fwhm(sensor: dict[str, object]) -> None:
        for band, width in zip(sensor["target_bands"], [19.5, 19.75, 20.0, 20.25, 20.5]):
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

    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="fwhm_aware",
        fwhm_conditioned=True,
        sensor_mutator=vary_fwhm,
    )
    result, report = run_and_read(bundle, tmp_path / "fwhm-aware")

    assert result.exit_code == 0
    assert report["states"]["executable"] == X3
    assert report["states"]["spectral"] == S3
    assert report["canaries"]["C3_fwhm_dependence"]["status"] == "PASS"


def test_both_unclaimed_metadata_challenges_still_run_without_proving_global_nonuse(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory()
    result, report = run_and_read(bundle, tmp_path / "unclaimed-metadata")

    assert result.exit_code == 0
    assert report["states"]["executable"] == X3
    assert report["canaries"]["C3_wavelength_dependence"]["status"] == (
        "NO_WAVELENGTH_DEPENDENCE_OBSERVED_ON_PROBES"
    )
    assert report["canaries"]["C3_fwhm_dependence"]["status"] == (
        "NO_FWHM_DEPENDENCE_OBSERVED_ON_PROBES"
    )
    fault_codes = {fault["code"] for fault in report["faults"]}
    assert not fault_codes.intersection(
        {
            "undeclared_wavelength_input_used",
            "undeclared_fwhm_input_used",
            "claimed_wavelength_input_ignored",
            "claimed_fwhm_input_ignored",
        }
    )
    limitations = " ".join(report["limitations"]).lower()
    assert "global" in limitations and ("metadata" in limitations or "non-use" in limitations)


@pytest.mark.parametrize(
    ("mode", "field", "fault_code"),
    [
        (
            "wavelength_range_aware",
            "wavelength",
            "undeclared_wavelength_input_used",
        ),
        ("fwhm_ratio_aware", "fwhm", "undeclared_fwhm_input_used"),
    ],
)
def test_nonuniform_c3_magnitudes_expose_rotation_invariant_undeclared_metadata_use(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    mode: str,
    field: str,
    fault_code: str,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode=mode)
    result, report = run_and_read(bundle, tmp_path / mode)

    assert result.exit_code == 4
    canary = report["canaries"][f"C3_{field}_dependence"]
    assert canary["status"] == (
        f"UNDECLARED_{field.upper()}_DEPENDENCE_OBSERVED_ON_PROBES"
    )
    assert fault_code in {fault["code"] for fault in report["faults"]}
    assert report["states"]["executable"] != X3
    limitations = " ".join(report["limitations"]).lower()
    assert "finite challenge" in limitations or "global non-use" in limitations


def test_constant_probes_are_inconclusive_not_evidence_of_global_nonuse(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    probes = np.full((20, 5), 0.4, dtype=np.float64)
    bundle = bundle_factory(probes=probes)
    result, report = run_and_read(bundle, tmp_path / "constant")

    assert result.exit_code == 4
    assert report["states"]["executable"] == X1
    assert report["states"]["executable"] != X3
    c2 = report["canaries"]["C2_value_dependence"]
    assert c2["status"] == INCONCLUSIVE
    assert {row["status"] for row in c2["bands"].values()} == {INCONCLUSIVE}
    serialized = str(c2)
    assert "NO_SPECTRAL_DEPENDENCE_OBSERVED" not in serialized
    assert "globally unused" not in serialized.lower()
    c4 = report["canaries"]["C4_order"]
    assert c4["status"] == "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION"
    assert c4["id_only_binding"] == {
        "status": "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION",
        "expected_pre_core_change_max_abs": 0.0,
        "minimum_exclusive_expected_pre_core_change": 0.000001,
        "excitation_adequate": False,
        "pre_core_max_abs_error": 0.0,
        "output_change_from_baseline_max_abs": 0.0,
    }
    assert "reordered_bands" in {fault["code"] for fault in report["faults"]}


def test_all_neutral_probes_make_both_rotation_and_target_neutral_canaries_inconclusive(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    probes = np.full((20, 5), 0.5, dtype=np.float64)
    bundle = bundle_factory(probes=probes)
    result, report = run_and_read(bundle, tmp_path / "all-neutral")

    assert result.exit_code == 4
    assert report["states"]["executable"] == X1
    assert report["canaries"]["C2_value_dependence"]["status"] == INCONCLUSIVE
    assert report["canaries"]["C5_target_neutral"]["status"] == INCONCLUSIVE
    c4 = report["canaries"]["C4_order"]
    assert c4["status"] == "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION"
    assert c4["id_only_binding"]["excitation_adequate"] is False
    assert c4["id_only_binding"]["expected_pre_core_change_max_abs"] == 0.0
    assert "NO_SPECTRAL_DEPENDENCE_OBSERVED" not in str(report["canaries"])


def test_target_neutral_values_must_be_finite_and_inside_target_raw_ranges(
    bundle_factory: BundleFactory,
) -> None:
    def invalid_neutral(sensor: dict[str, object]) -> None:
        sensor["target_bands"][2]["neutral_value"] = 1.5

    bundle = bundle_factory(sensor_mutator=invalid_neutral)

    from bandtrace.bundle import load_bundle
    from bandtrace.errors import BundleError

    try:
        load_bundle(bundle.root)
    except BundleError as exc:
        message = str(exc).lower()
        assert "neutral" in message
        assert "range" in message
    else:
        raise AssertionError("out-of-range target neutral value was admitted")


def test_c0_through_c6_retain_raw_threshold_and_mutation_evidence(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory()
    result, report = run_and_read(bundle, tmp_path / "canary-evidence")

    assert result.exit_code == 0
    assert report["facts"]["raw_thresholds"]["dependence_absolute_floor"] == 0.000001
    assert report["facts"]["raw_thresholds"]["minimum_probe_fraction"] == 0.20
    assert report["canaries"]["C0_replay"]["replay_count"] == 3
    assert report["canaries"]["C0_replay"]["normalized_max_jitter"] <= 0.0000001
    assert report["canaries"]["C1_declared_tap_agreement"]["baseline_max_abs_error"] <= 0.000001
    assert report["canaries"]["C1_declared_tap_agreement"]["basis_max_abs_error"] <= 0.000001
    assert report["canaries"]["C2_value_dependence"]["bands"]
    assert report["canaries"]["C3_wavelength_dependence"]["status"] == (
        "NO_WAVELENGTH_DEPENDENCE_OBSERVED_ON_PROBES"
    )
    assert report["canaries"]["C3_fwhm_dependence"]
    assert report["canaries"]["C4_order"]["permutation"]
    assert report["canaries"]["C5_target_neutral"]["status"] == "PASS"
    assert report["canaries"]["C6_edge_alias"]["status"] in {"PASS", "NOT_APPLICABLE"}


def test_wavelength_conditioned_uniform_metadata_is_excited_by_nonuniform_challenge(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    def invariant_metadata(sensor: dict[str, object]) -> None:
        for band in sensor["target_bands"]:
            band["center_wavelength"] = 600.0
            band["fwhm"] = 20.0
            band["srf"] = {
                "wavelengths": [580.0, 590.0, 600.0, 610.0, 620.0],
                "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
                "wavelength_unit": "nm",
            }

    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="clean",
        wavelength_conditioned=True,
        sensor_mutator=invariant_metadata,
    )
    result, report = run_and_read(bundle, tmp_path / "metadata-inconclusive")

    assert result.exit_code == 4
    assert report["canaries"]["C3_wavelength_dependence"]["status"] == (
        "NO_WAVELENGTH_DEPENDENCE_OBSERVED_ON_PROBES"
    )
    assert report["states"]["executable"] != X3
    fault_codes = {fault["code"] for fault in report["faults"]}
    assert "claimed_wavelength_input_ignored" in fault_codes
    assert "claimed_wavelength_input_inconclusive" not in fault_codes
    assert "globally" not in str(report["canaries"]["C3_wavelength_dependence"]).lower()


def test_fwhm_conditioned_uniform_metadata_is_excited_by_nonuniform_challenge(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(fwhm_conditioned=True)
    result, report = run_and_read(bundle, tmp_path / "fwhm-metadata-inconclusive")

    assert result.exit_code == 4
    assert report["canaries"]["C3_fwhm_dependence"]["status"] == (
        "NO_FWHM_DEPENDENCE_OBSERVED_ON_PROBES"
    )
    assert report["states"]["executable"] != X3
    fault_codes = {fault["code"] for fault in report["faults"]}
    assert "claimed_fwhm_input_ignored" in fault_codes
    assert "claimed_fwhm_input_inconclusive" not in fault_codes


def test_c4_id_aware_adapter_passes_but_positional_adapter_fails(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    aware = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="clean")
    positional = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="positional_only")
    aware_result, aware_report = run_and_read(aware, tmp_path / "id-aware")
    positional_result, positional_report = run_and_read(positional, tmp_path / "positional")

    assert aware_result.exit_code == 0
    aware_c4 = aware_report["canaries"]["C4_order"]
    assert aware_c4["status"] == "PASS"
    assert aware_c4["finding"] == "ID_BOUND"
    assert aware_c4["output_max_abs_error"] <= 0.000001
    assert aware_c4["pre_core_max_abs_error"] <= 0.000001
    assert aware_c4["tied_tuple_order_invariance"]["status"] == "PASS"
    assert aware_c4["id_only_binding"]["status"] == "PASS"
    assert aware_c4["id_only_binding"]["excitation_adequate"] is True
    assert aware_c4["id_only_binding"]["expected_pre_core_change_max_abs"] > 0.000001
    assert aware_c4["id_only_binding"][
        "minimum_exclusive_expected_pre_core_change"
    ] == 0.000001
    assert aware_c4["id_only_binding"]["pre_core_max_abs_error"] <= 0.000001
    with np.load(tmp_path / "id-aware" / "canary_outputs.npz", allow_pickle=False) as archive:
        assert "c4_order_output" in archive.files
        assert "c4_id_binding_output" in archive.files

    assert positional_result.exit_code == 4
    positional_c4 = positional_report["canaries"]["C4_order"]
    assert positional_c4["status"] == "FAIL"
    assert positional_c4["finding"] == "NOT_ESTABLISHED"
    assert positional_c4["tied_tuple_order_invariance"]["status"] == "FAIL"
    assert positional_c4["id_only_binding"]["status"] == "FAIL"
    assert positional_report["states"]["executable"] not in {X2, X3}
    assert "reordered_bands" in {fault["code"] for fault in positional_report["faults"]}


def test_c4_id_only_challenge_rejects_metadata_sorted_id_blind_adapter(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="id_ignored_metadata_sorted",
    )
    result, report = run_and_read(bundle, tmp_path / "metadata-sorted-id-blind")

    assert result.exit_code == 4
    c4 = report["canaries"]["C4_order"]
    assert c4["status"] == "FAIL"
    assert c4["finding"] == "NOT_ESTABLISHED"
    assert c4["tied_tuple_order_invariance"]["status"] == "PASS"
    assert c4["id_only_binding"]["excitation_adequate"] is True
    assert c4["id_only_binding"]["expected_pre_core_change_max_abs"] > 0.000001
    assert c4["id_only_binding"]["status"] == "FAIL"
    assert c4["id_only_binding"]["pre_core_max_abs_error"] > 0.000001
    assert report["states"]["executable"] not in {X2, X3}
    assert "reordered_bands" in {fault["code"] for fault in report["faults"]}


def test_c4_id_only_probe_symmetry_is_conformance_fault_inconclusive(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(probes=np.column_stack([np.linspace(0.1, 0.9, 20)] * 5))
    result, report = run_and_read(bundle, tmp_path / "id-binding-probe-symmetry")

    assert result.exit_code == 4
    c4 = report["canaries"]["C4_order"]
    assert c4["status"] == "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION"
    assert c4["finding"] == "NOT_ESTABLISHED"
    assert c4["tied_tuple_order_invariance"]["status"] == "PASS"
    assert c4["id_only_binding"] == {
        "status": "INCONCLUSIVE_INSUFFICIENT_ID_BINDING_EXCITATION",
        "expected_pre_core_change_max_abs": 0.0,
        "minimum_exclusive_expected_pre_core_change": 0.000001,
        "excitation_adequate": False,
        "pre_core_max_abs_error": 0.0,
        "output_change_from_baseline_max_abs": 0.0,
    }
    assert report["states"]["executable"] not in {X2, X3}
    assert "reordered_bands" in {fault["code"] for fault in report["faults"]}


def test_frozen_base_seed_and_c4_permutation_known_answer() -> None:
    from bandtrace.canonical import (
        c3_rank_amplitudes,
        c4_permutation,
        derive_canary_seed,
        derive_seed,
    )

    base_seed = derive_seed(
        model_hash="0" * 64,
        sensor_hash="1" * 64,
        probe_hash="2" * 64,
        route_hash="3" * 64,
        policy_id="bandtrace-0.1-r29",
    )

    assert base_seed.hex() == "d9d2f0dc8aefdc56e2b3d91ca7112b698b7463f4f0a38747f7bd1b6c5e3257bf"
    canary_seed = derive_canary_seed(base_seed, "C4_order")
    shift = 1 + int.from_bytes(canary_seed[:8], "big") % 4
    permutation = c4_permutation(base_seed, 5)
    assert shift == 1
    assert permutation.tolist() == [1, 2, 3, 4, 0]

    ids = ("b0", "b1", "b2")
    assert derive_canary_seed(base_seed, "C3_wavelength_dependence").hex() == (
        "378a64babd2f4e1452b4b136a3578160c7ab195d0671e9fed8d0a86ff72c27c8"
    )
    assert derive_canary_seed(base_seed, "C3_fwhm_dependence").hex() == (
        "1b0f69a72d1def6ff200738011f2cb8d2cd1fd3acf44e56cbe12ad5a9c10ee6f"
    )
    np.testing.assert_array_equal(
        c3_rank_amplitudes(base_seed, "C3_wavelength_dependence", ids),
        np.asarray([0.5, 0.25, 0.75], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        c3_rank_amplitudes(base_seed, "C3_fwhm_dependence", ids),
        np.asarray([0.75, 0.25, 0.5], dtype=np.float64),
    )


def test_c3_rank_amplitudes_are_deterministic_and_bound_to_target_ids() -> None:
    from bandtrace.canonical import c3_rank_amplitudes, derive_seed

    base_seed = derive_seed(
        model_hash="0" * 64,
        sensor_hash="1" * 64,
        probe_hash="2" * 64,
        route_hash="3" * 64,
        policy_id="bandtrace-0.1-r29",
    )
    canonical_ids = ("b0", "b1", "b2")
    reordered_ids = ("b2", "b0", "b1")
    first = c3_rank_amplitudes(
        base_seed, "C3_wavelength_dependence", canonical_ids
    )
    second = c3_rank_amplitudes(
        base_seed, "C3_wavelength_dependence", canonical_ids
    )
    reordered = c3_rank_amplitudes(
        base_seed, "C3_wavelength_dependence", reordered_ids
    )

    np.testing.assert_array_equal(first, second)
    canonical_by_id = dict(zip(canonical_ids, first.tolist()))
    reordered_by_id = dict(zip(reordered_ids, reordered.tolist()))
    assert reordered_by_id == canonical_by_id


def test_frozen_linear_quantile_flattens_sorts_and_interpolates_explicitly() -> None:
    from bandtrace.canonical import linear_quantile

    values = np.asarray([[30.0, 0.0], [20.0, 10.0]], dtype=np.float64)

    assert linear_quantile(values, 0.5) == 15.0
    assert linear_quantile(values, 0.99).hex() == "0x1.db33333333332p+4"
    assert np.percentile(values, 99, method="nearest") == 30.0
    assert np.percentile(values, 99, method="lower") == 20.0


def test_frozen_linear_quantile_rejects_empty_nonfinite_and_out_of_range_inputs() -> None:
    from bandtrace.canonical import linear_quantile
    from bandtrace.errors import BandTraceError

    for values, q in (
        (np.asarray([], dtype=np.float64), 0.5),
        (np.asarray([0.0, np.nan], dtype=np.float64), 0.5),
        (np.asarray([0.0, np.inf], dtype=np.float64), 0.5),
        (np.asarray([0.0], dtype=np.float64), -0.01),
        (np.asarray([0.0], dtype=np.float64), 1.01),
    ):
        with pytest.raises(BandTraceError):
            linear_quantile(values, q)


def test_canary_quantiles_do_not_use_numpy_default_percentile_semantics(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_default(*args: object, **kwargs: object) -> float:
        raise AssertionError("NumPy percentile defaults are not part of the frozen contract")

    monkeypatch.setattr(np, "percentile", forbidden_default)
    monkeypatch.setattr(np, "quantile", forbidden_default)

    result, report = run_and_read(
        bundle_factory(adapter="numpy-linear-v1"),
        tmp_path / "explicit-linear-quantiles",
    )
    assert result.exit_code == 0
    assert report["canaries"]["C0_replay"]["status"] == "PASS"
