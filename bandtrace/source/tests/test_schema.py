from __future__ import annotations

import sys

import numpy as np
import pytest

from conftest import BundleFactory


def _assert_bundle_rejected(bundle) -> None:
    from bandtrace.bundle import load_bundle
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError):
        load_bundle(bundle.root)


@pytest.mark.parametrize("field", ["wavelength_conditioned", "fwhm_conditioned"])
def test_both_metadata_conditioning_declarations_are_explicit_and_required(
    bundle_factory: BundleFactory,
    field: str,
) -> None:
    def remove_declaration(model: dict[str, object]) -> None:
        model.pop(field)

    from bandtrace.bundle import load_bundle
    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match=field):
        load_bundle(bundle_factory(model_mutator=remove_declaration).root)


def test_stale_route_owned_required_dependence_alias_is_rejected(
    bundle_factory: BundleFactory,
) -> None:
    def stale_alias(route: dict[str, object]) -> None:
        route["required_target_band_ids"] = ["t450", "t550", "t650", "t750"]

    _assert_bundle_rejected(bundle_factory(route_mutator=stale_alias))


def test_stale_model_required_target_alias_is_not_silently_authoritative(
    bundle_factory: BundleFactory,
) -> None:
    def stale_alias(model: dict[str, object]) -> None:
        model.pop("required_dependence_target_band_ids")
        model["required_target_band_ids"] = ["t450", "t550", "t650", "t750"]

    _assert_bundle_rejected(bundle_factory(model_mutator=stale_alias))


@pytest.mark.parametrize(
    "required",
    [
        [],
        ["t450", "t450"],
        ["does-not-exist"],
        ["t950"],
    ],
)
def test_required_dependence_ids_are_nonempty_unique_known_and_materially_routed(
    bundle_factory: BundleFactory,
    required: list[str],
) -> None:
    def invalid_required(model: dict[str, object]) -> None:
        model["required_dependence_target_band_ids"] = required

    _assert_bundle_rejected(bundle_factory(model_mutator=invalid_required))


def test_exact_top_level_manifest_keys_are_enforced(bundle_factory: BundleFactory) -> None:
    bundle = bundle_factory()
    manifest = bundle.manifest()
    manifest["helpful_but_unpinned"] = "must not be ignored"
    bundle.rewrite_manifest(manifest)

    _assert_bundle_rejected(bundle)


def test_exact_probe_member_names_are_enforced_by_security_suite_contract(
    bundle_factory: BundleFactory,
) -> None:
    bundle = bundle_factory()
    probes = bundle.file_path("probes")
    probes.rename(bundle.root / "renamed-probes.npz")

    _assert_bundle_rejected(bundle)


def test_subprocess_inline_code_without_pinned_artifact_is_rejected(
    bundle_factory: BundleFactory,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1")
    manifest = bundle.manifest()
    manifest["adapter"]["argv"] = [
        sys.executable,
        "-c",
        "raise SystemExit('inline code must not execute')",
        "{input_npz}",
        "{output_npz}",
    ]
    bundle.rewrite_manifest(manifest)

    _assert_bundle_rejected(bundle)


@pytest.mark.parametrize(
    ("location", "mutator"),
    [
        ("model", lambda payload: payload.__setitem__("unknown", True)),
        ("sensor", lambda payload: payload.__setitem__("unknown", True)),
        ("route", lambda payload: payload.__setitem__("unknown", True)),
        ("model_band", lambda payload: payload["model_channels"][0].__setitem__("unknown", True)),
        ("sensor_band", lambda payload: payload["target_bands"][0].__setitem__("unknown", True)),
        (
            "model_srf",
            lambda payload: payload["model_channels"][0]["srf"].__setitem__("unknown", True),
        ),
        (
            "sensor_srf",
            lambda payload: payload["target_bands"][0]["srf"].__setitem__("unknown", True),
        ),
        (
            "normalization",
            lambda payload: payload["normalization"].__setitem__("unknown", True),
        ),
        (
            "support",
            lambda payload: payload["declared_validated_support"].__setitem__("unknown", True),
        ),
        (
            "output",
            lambda payload: payload["pre_decision_output"].__setitem__("unknown", True),
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_unknown_nested_contract_fields_are_never_silently_ignored(
    bundle_factory: BundleFactory,
    location: str,
    mutator,
) -> None:
    if location.startswith("sensor"):
        bundle = bundle_factory(sensor_mutator=mutator)
    elif location == "route":
        bundle = bundle_factory(route_mutator=mutator)
    else:
        bundle = bundle_factory(model_mutator=mutator)
    _assert_bundle_rejected(bundle)


@pytest.mark.parametrize(
    "missing_key",
    [
        "route_matrix",
        "target_band_ids",
        "normalization_offset",
        "normalization_scale",
        "output_weights",
        "spatial_operation",
    ],
)
def test_numpy_reference_artifact_missing_required_key_fails_at_bundle_load(
    bundle_factory: BundleFactory,
    missing_key: str,
) -> None:
    bundle = bundle_factory(adapter="numpy-linear-v1")
    artifact_path = bundle.file_path("artifact")
    with np.load(artifact_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files if key != missing_key}
    np.savez(artifact_path, **arrays)
    bundle.refresh_hash("artifact")

    _assert_bundle_rejected(bundle)


@pytest.mark.parametrize(
    ("spatial_operation", "rank_four"),
    [("none", True), ("mean", False)],
)
def test_spatial_operation_and_probe_rank_must_match_exactly(
    bundle_factory: BundleFactory,
    spatial_operation: str,
    rank_four: bool,
) -> None:
    probes = (
        np.full((20, 5, 2, 2), 0.4, dtype=np.float64)
        if rank_four
        else np.full((20, 5), 0.4, dtype=np.float64)
    )

    def mutate_route(route: dict[str, object]) -> None:
        route["spatial_operation"] = spatial_operation

    _assert_bundle_rejected(bundle_factory(probes=probes, route_mutator=mutate_route))


@pytest.mark.parametrize("scale", [[1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 1.0, 1.0]])
def test_normalization_scale_must_be_strictly_positive(
    bundle_factory: BundleFactory,
    scale: list[float],
) -> None:
    _assert_bundle_rejected(bundle_factory(normalization_scale=np.asarray(scale, dtype=np.float64)))
