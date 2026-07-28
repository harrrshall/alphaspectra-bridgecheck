from __future__ import annotations

import errno
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest

from conftest import BundleCase, BundleFactory, SOURCE_ROOT, TARGET_BAND_IDS, _band, varied_probes


def _load_bundle(bundle: BundleCase):
    from bandtrace.bundle import load_bundle

    return load_bundle(bundle.root)


def _baseline_invocation(bundle: object):
    from bandtrace.adapters import Invocation

    return Invocation(
        probes=np.asarray(bundle.probes.values, dtype=np.float64),
        target_band_ids=tuple(band.id for band in bundle.sensor.bands),
        wavelength_nm=np.asarray([band.center_nm for band in bundle.sensor.bands]),
        fwhm_nm=np.asarray([band.fwhm_nm for band in bundle.sensor.bands]),
    )


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


def _replace_probes(bundle: BundleCase, **arrays: np.ndarray) -> None:
    np.savez(bundle.file_path("probes"), **arrays)
    bundle.refresh_hash("probes")


def _write_npz_with_compression(
    path: Path,
    arrays: dict[str, np.ndarray],
    compression: int,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.save(member, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", member.getvalue())


def test_every_referenced_file_hash_is_checked_before_use(clean_numpy_bundle: BundleCase) -> None:
    probes_path = clean_numpy_bundle.file_path("probes")
    probes_path.write_bytes(probes_path.read_bytes() + b"tamper")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="SHA-256|sha256|hash"):
        _load_bundle(clean_numpy_bundle)


def test_validated_probe_and_numpy_artifact_raw_bytes_are_released_after_parse(
    clean_numpy_bundle: BundleCase,
) -> None:
    expected_counts = {
        key: clean_numpy_bundle.file_path(key).stat().st_size
        for key in clean_numpy_bundle.manifest()["files"]
    }

    loaded = _load_bundle(clean_numpy_bundle)

    assert loaded.files["probes"].data is None
    assert loaded.files["artifact"].data is None
    assert loaded.files["probes"].byte_count == expected_counts["probes"]
    assert loaded.files["artifact"].byte_count == expected_counts["artifact"]
    assert loaded.probes.values.size > 0
    assert loaded.numpy_artifact is not None
    for key in ("model", "sensor", "route"):
        assert isinstance(loaded.files[key].data, bytes)
        assert loaded.files[key].byte_count == expected_counts[key]

    from bandtrace.audit import _input_facts

    reported_counts = {
        record["key"]: record["bytes"] for record in _input_facts(loaded)
    }
    assert reported_counts == expected_counts


def test_cumulative_adapter_request_charge_accepts_exact_cap_and_rejects_one_below(
    bundle_factory: BundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.bundle as bundle_module

    reference = _load_bundle(bundle_factory())
    charge = reference.adapter_work_plan.cumulative_request_probe_value_bytes
    assert charge > 1

    monkeypatch.setattr(
        bundle_module,
        "MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES",
        charge,
    )
    exact = _load_bundle(bundle_factory())
    assert exact.adapter_work_plan.cumulative_request_probe_value_bytes == charge

    monkeypatch.setattr(
        bundle_module,
        "MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES",
        charge - 1,
    )
    from bandtrace.errors import BundleError

    with pytest.raises(
        BundleError,
        match="planned cumulative adapter request probe-value bytes exceed",
    ):
        _load_bundle(bundle_factory())


def test_reported_adapter_work_charge_matches_independent_frozen_schedule_formula(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    from bandtrace.audit import run_audit
    from bandtrace.constants import (
        C1_CHUNK_MAX_FLOAT64_PROBE_BYTES,
        C1_CHUNK_MAX_ROWS,
        MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES,
    )

    values = varied_probes()
    bands = len(TARGET_BAND_IDS)
    spatial_cells = 1
    baseline_bytes = int(values.nbytes)
    bytes_per_basis_row = bands * spatial_cells * np.dtype("float64").itemsize
    rows_per_basis_request = min(
        C1_CHUNK_MAX_ROWS,
        C1_CHUNK_MAX_FLOAT64_PROBE_BYTES // bytes_per_basis_row,
    )
    basis_request_count = (bands + rows_per_basis_request - 1) // rows_per_basis_request
    basis_bytes = bands * bytes_per_basis_row
    full_size_request_count = 3 + 1 + bands + 6 + 2
    spatial_request_count = 0
    spatial_bytes = 0
    total_invocation_count = (
        full_size_request_count + basis_request_count + spatial_request_count
    )
    cumulative = full_size_request_count * baseline_bytes + basis_bytes + spatial_bytes

    result = run_audit(clean_numpy_bundle.root, tmp_path / "planned-work-report")
    planned = result.report["facts"]["planned_adapter_work"]

    assert planned == {
        "baseline_probe_value_bytes": baseline_bytes,
        "full_size_request_count": full_size_request_count,
        "basis_request_count": basis_request_count,
        "basis_probe_value_bytes": basis_bytes,
        "spatial_request_count": spatial_request_count,
        "spatial_probe_value_bytes": spatial_bytes,
        "total_invocation_count": total_invocation_count,
        "cumulative_request_probe_value_bytes": cumulative,
        "maximum_cumulative_request_probe_value_bytes": (
            MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES
        ),
        "charge_formula": (
            "full_size_request_count*P + B*B*spatial_cells*8 + "
            "(rank4 ? 4*B*spatial_cells*8 : 0)"
        ),
        "schedule": (
            "C0_THREE_REPLAYS+C5_ONE_NEUTRAL+C1_BASIS_CHUNKS+"
            "C1_OPTIONAL_SPATIAL+C2_ONE_PER_BAND+C3_SIX_MUTATIONS+"
            "C4_TWO_IF_MULTIBAND"
        ),
    }
    assert result.report["facts"]["configured_byte_budgets"][
        "maximum_cumulative_adapter_request_probe_value_bytes"
    ] == MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES
    assert result.report["facts"]["adapter_invocation_count"] == total_invocation_count
    assert total_invocation_count <= 2 * bands + 12


def test_pathological_b512_rank4_work_plan_rejects_before_adapter_construction(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bandtrace.audit as audit_module
    import bandtrace.bundle as bundle_module
    from bandtrace.cli import main as cli_main
    from bandtrace.constants import MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES

    extra_ids = [f"tx{index:03d}" for index in range(507)]
    all_ids = TARGET_BAND_IDS + extra_ids

    def expand_sensor(sensor: dict[str, object]) -> None:
        sensor["target_bands"].extend(
            _band(identifier, 800.0 + 0.1 * index, target=True)
            for index, identifier in enumerate(extra_ids)
        )

    def expand_route(route: dict[str, object]) -> None:
        route["target_band_ids"] = all_ids
        route["matrix"] = np.pad(
            np.asarray(route["matrix"], dtype=np.float64),
            ((0, 0), (0, 507)),
        ).tolist()

    bundle = bundle_factory(
        probes=np.full((16, 512), 0.4, dtype=np.float64),
        sensor_mutator=expand_sensor,
        route_mutator=expand_route,
    )
    logical_values = np.broadcast_to(
        np.zeros((), dtype=np.float64),
        (16, 512, 1, 4095),
    )
    logical_probe_set = bundle_module.ProbeSet(
        values=logical_values,
        target_band_ids=tuple(all_ids),
        order_matches_sensor=True,
    )
    plan = bundle_module._plan_adapter_work(logical_probe_set, 512)
    assert logical_values.nbytes == 268_369_920
    assert plan.cumulative_request_probe_value_bytes > (
        MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES
    )
    assert plan.total_invocation_count <= 2 * 512 + 12

    real_parse_probes = bundle_module._parse_probes

    def substitute_logical_pathological_shape(data: bytes, sensor: object) -> object:
        parsed = real_parse_probes(data, sensor)
        assert parsed.target_band_ids == tuple(all_ids)
        return logical_probe_set

    build_calls: list[object] = []

    def record_build_adapter(loaded: object) -> object:
        build_calls.append(loaded)
        raise AssertionError("adapter constructed after cumulative work-plan rejection")

    monkeypatch.setattr(
        bundle_module,
        "_parse_probes",
        substitute_logical_pathological_shape,
    )
    monkeypatch.setattr(audit_module, "build_adapter", record_build_adapter)
    destination = tmp_path / "pathological-b512"
    exit_code = cli_main(
        ["audit", str(bundle.root), "--output-dir", str(destination)]
    )
    stderr = capsys.readouterr().err

    assert exit_code == 2
    assert "planned cumulative adapter request probe-value bytes exceed" in stderr
    assert build_calls == []
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


@pytest.mark.parametrize("path_kind", ["parent", "absolute"])
def test_manifest_paths_cannot_escape_bundle(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    path_kind: str,
) -> None:
    manifest = clean_numpy_bundle.manifest()
    source = clean_numpy_bundle.file_path("probes")
    outside = tmp_path / "outside-probes.npz"
    shutil.copyfile(source, outside)
    manifest["files"]["probes"]["path"] = (
        "../outside-probes.npz" if path_kind == "parent" else str(outside.resolve())
    )
    clean_numpy_bundle.rewrite_manifest(manifest)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="relative|traversal|bundle|path"):
        _load_bundle(clean_numpy_bundle)


def test_symlinked_referenced_file_is_rejected_even_when_target_hash_matches(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    probes_path = clean_numpy_bundle.file_path("probes")
    outside = tmp_path / "outside.npz"
    shutil.copyfile(probes_path, outside)
    probes_path.unlink()
    probes_path.symlink_to(outside)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="symlink|regular file"):
        _load_bundle(clean_numpy_bundle)


def test_ordinary_nested_bundle_relative_regular_file_loads(
    clean_numpy_bundle: BundleCase,
) -> None:
    manifest = clean_numpy_bundle.manifest()
    original = clean_numpy_bundle.file_path("model")
    nested = clean_numpy_bundle.root / "contracts" / "model" / original.name
    nested.parent.mkdir(parents=True)
    original.rename(nested)
    manifest["files"]["model"]["path"] = "contracts/model/model.json"
    clean_numpy_bundle.rewrite_manifest(manifest)

    loaded = _load_bundle(clean_numpy_bundle)

    assert loaded.files["model"].relative_path == "contracts/model/model.json"
    assert loaded.files["model"].path == nested
    assert loaded.files["model"].data == nested.read_bytes()


def test_intermediate_directory_swapped_to_outside_symlink_before_read_fails_closed(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.bundle as bundle_module

    manifest = clean_numpy_bundle.manifest()
    original_model = clean_numpy_bundle.file_path("model")
    contracts = clean_numpy_bundle.root / "contracts"
    contracts.mkdir()
    nested_model = contracts / "model.json"
    original_model.rename(nested_model)
    manifest["files"]["model"]["path"] = "contracts/model.json"
    clean_numpy_bundle.rewrite_manifest(manifest)

    outside = tmp_path / "outside-contracts"
    outside.mkdir()
    outside_model = outside / "model.json"
    shutil.copyfile(nested_model, outside_model)
    assert outside_model.read_bytes() == nested_model.read_bytes()

    captured_contracts = clean_numpy_bundle.root / "captured-contracts"
    original_read = bundle_module._read_pinned_regular_file
    attack_triggered = False

    def swap_intermediate_before_read(
        root: Path,
        root_fd: int,
        pinned: object,
        *,
        max_bytes: int,
        source: str,
    ) -> bytes:
        nonlocal attack_triggered
        if source == "bandtrace.yaml.files.model" and not attack_triggered:
            attack_triggered = True
            contracts.rename(captured_contracts)
            contracts.symlink_to(outside, target_is_directory=True)
        return original_read(
            root,
            root_fd,
            pinned,
            max_bytes=max_bytes,
            source=source,
        )

    monkeypatch.setattr(
        bundle_module,
        "_read_pinned_regular_file",
        swap_intermediate_before_read,
    )

    from bandtrace.errors import BundleError

    with pytest.raises(
        BundleError,
        match="intermediate|revalidate|identity|symlink|pinned",
    ):
        _load_bundle(clean_numpy_bundle)

    assert attack_triggered
    assert contracts.is_symlink()
    assert outside_model.read_bytes() == (captured_contracts / "model.json").read_bytes()


@pytest.mark.parametrize("failure_loop", ["intermediate_open", "revalidation"])
def test_repeated_intermediate_directory_fstat_failures_do_not_leak_descriptors(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_loop: str,
) -> None:
    import bandtrace.bundle as bundle_module
    from bandtrace.cli import main as cli_main
    from bandtrace.errors import BundleError

    manifest = clean_numpy_bundle.manifest()
    original_model = clean_numpy_bundle.file_path("model")
    contracts = clean_numpy_bundle.root / "contracts"
    contracts.mkdir()
    original_model.rename(contracts / "model.json")
    manifest["files"]["model"]["path"] = "contracts/model.json"
    clean_numpy_bundle.rewrite_manifest(manifest)

    real_open = bundle_module.os.open
    real_fstat = bundle_module.os.fstat
    pending_intermediate_fstat: set[int] = set()
    intermediate_fstat_count = 0

    def track_intermediate_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "contracts" and flags & getattr(os, "O_DIRECTORY", 0):
            pending_intermediate_fstat.add(descriptor)
        return descriptor

    def inject_intermediate_fstat_failure(descriptor: int) -> os.stat_result:
        nonlocal intermediate_fstat_count
        if descriptor in pending_intermediate_fstat:
            pending_intermediate_fstat.remove(descriptor)
            intermediate_fstat_count += 1
            should_fail = failure_loop == "intermediate_open" or (
                intermediate_fstat_count % 2 == 0
            )
            if should_fail:
                raise OSError(
                    errno.EIO,
                    f"injected {failure_loop} intermediate fstat failure",
                )
        return real_fstat(descriptor)

    monkeypatch.setattr(bundle_module.os, "open", track_intermediate_open)
    monkeypatch.setattr(
        bundle_module.os,
        "fstat",
        inject_intermediate_fstat_failure,
    )

    descriptor_directory = Path("/proc/self/fd")
    assert descriptor_directory.is_dir()
    before = len(list(descriptor_directory.iterdir()))
    for _ in range(20):
        with pytest.raises(BundleError, match="cannot open|cannot revalidate|fstat failure"):
            _load_bundle(clean_numpy_bundle)
    after_repeated_loads = len(list(descriptor_directory.iterdir()))
    assert after_repeated_loads == before
    assert pending_intermediate_fstat == set()

    destination = tmp_path / f"fd-loop-{failure_loop}"
    exit_code = cli_main(
        ["audit", str(clean_numpy_bundle.root), "--output-dir", str(destination)]
    )
    stderr = capsys.readouterr().err
    after_cli = len(list(descriptor_directory.iterdir()))

    assert exit_code == 2
    assert "BandTrace invalid bundle" in stderr
    assert "Traceback" not in stderr
    assert after_cli == before
    assert pending_intermediate_fstat == set()
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


def test_bundle_root_renamed_and_replaced_during_manifest_load_fails_closed(
    clean_numpy_bundle: BundleCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.bundle as bundle_module

    root = clean_numpy_bundle.root
    moved_root = root.with_name(f"{root.name}-captured")
    original_identity = (root.stat().st_dev, root.stat().st_ino)
    original_read = bundle_module._read_pinned_regular_file
    attack_triggered = False
    replacement_identity: tuple[int, int] | None = None

    def replace_root_after_manifest_read(
        pinned_root: Path,
        root_fd: int,
        pinned: object,
        *,
        max_bytes: int,
        source: str,
    ) -> bytes:
        nonlocal attack_triggered, replacement_identity
        data = original_read(
            pinned_root,
            root_fd,
            pinned,
            max_bytes=max_bytes,
            source=source,
        )
        if source == "bandtrace.yaml" and not attack_triggered:
            attack_triggered = True
            root.rename(moved_root)
            shutil.copytree(moved_root, root)
            replacement = root.stat()
            replacement_identity = (replacement.st_dev, replacement.st_ino)
        return data

    monkeypatch.setattr(
        bundle_module,
        "_read_pinned_regular_file",
        replace_root_after_manifest_read,
    )

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="bundle root.*identity changed|root.*available"):
        _load_bundle(clean_numpy_bundle)

    assert attack_triggered
    assert replacement_identity is not None
    assert replacement_identity != original_identity
    moved_identity = (moved_root.stat().st_dev, moved_root.stat().st_ino)
    assert moved_identity == original_identity


def test_manifest_name_replaced_after_parse_before_final_return_fails_closed(
    clean_numpy_bundle: BundleCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.bundle as bundle_module

    manifest_path = clean_numpy_bundle.manifest_path
    captured_manifest = clean_numpy_bundle.root / "captured-bandtrace.yaml"
    expected_bytes = manifest_path.read_bytes()
    original_identity = (manifest_path.stat().st_dev, manifest_path.stat().st_ino)
    real_verify = bundle_module._verify_pinned_regular_file_unchanged
    manifest_verification_calls = 0
    attack_triggered = False

    def replace_manifest_before_final_validation(
        root: Path,
        root_fd: int,
        pinned: object,
        *,
        source: str,
    ) -> None:
        nonlocal manifest_verification_calls, attack_triggered
        if source == "bandtrace.yaml":
            manifest_verification_calls += 1
            if manifest_verification_calls == 2:
                attack_triggered = True
                manifest_path.rename(captured_manifest)
                manifest_path.write_bytes(expected_bytes)
        real_verify(root, root_fd, pinned, source=source)

    monkeypatch.setattr(
        bundle_module,
        "_verify_pinned_regular_file_unchanged",
        replace_manifest_before_final_validation,
    )

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="pinned regular-file path identity changed"):
        _load_bundle(clean_numpy_bundle)

    assert attack_triggered
    assert manifest_verification_calls == 2
    assert manifest_path.read_bytes() == expected_bytes
    assert captured_manifest.read_bytes() == expected_bytes
    assert (captured_manifest.stat().st_dev, captured_manifest.stat().st_ino) == original_identity
    assert (manifest_path.stat().st_dev, manifest_path.stat().st_ino) != original_identity


def test_top_manifest_duplicate_key_is_rejected(clean_numpy_bundle: BundleCase) -> None:
    text = clean_numpy_bundle.manifest_path.read_text(encoding="utf-8")
    text = text.replace(
        '"policy_id": "bandtrace-0.1-r29",',
        '"policy_id": "bandtrace-0.1-r29",\n  "policy_id": "bandtrace-0.1-r29",',
        1,
    )
    assert text.count('"policy_id": "bandtrace-0.1-r29"') == 2
    clean_numpy_bundle.manifest_path.write_text(text, encoding="utf-8")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="duplicate"):
        _load_bundle(clean_numpy_bundle)


def test_nested_json_duplicate_key_is_rejected(clean_numpy_bundle: BundleCase) -> None:
    model_path = clean_numpy_bundle.file_path("model")
    text = model_path.read_text(encoding="utf-8").replace(
        '"model_id": "synthetic-linear-model",',
        '"model_id": "synthetic-linear-model",\n  "model_id": "decoy",',
        1,
    )
    model_path.write_text(text, encoding="utf-8")
    clean_numpy_bundle.refresh_hash("model")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="duplicate"):
        _load_bundle(clean_numpy_bundle)


def test_yaml_aliases_are_rejected_before_schema_interpretation(clean_numpy_bundle: BundleCase) -> None:
    clean_numpy_bundle.manifest_path.write_text(
        'schema_version: &version "1.0"\npolicy_id: *version\nfiles: {}\nadapter: {}\n',
        encoding="utf-8",
    )

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="alias|anchor"):
        _load_bundle(clean_numpy_bundle)


def test_yaml_non_json_scalar_type_is_rejected_before_policy_interpretation(
    clean_numpy_bundle: BundleCase,
) -> None:
    text = clean_numpy_bundle.manifest_path.read_text(encoding="utf-8").replace(
        '"policy_id": "bandtrace-0.1-r29"',
        '"policy_id": 2026-07-27',
        1,
    )
    clean_numpy_bundle.manifest_path.write_text(text, encoding="utf-8")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="JSON|json|scalar|string|data model"):
        _load_bundle(clean_numpy_bundle)


def test_yaml_mapping_keys_must_be_strings(
    clean_numpy_bundle: BundleCase,
) -> None:
    text = clean_numpy_bundle.manifest_path.read_text(encoding="utf-8").replace(
        "{",
        '{\n  1: "non-string-key",',
        1,
    )
    clean_numpy_bundle.manifest_path.write_text(text, encoding="utf-8")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="key|string|JSON|json|data model"):
        _load_bundle(clean_numpy_bundle)


def test_structured_document_depth_above_32_is_invalid_bundle_not_recursion_crash(
    clean_numpy_bundle: BundleCase,
) -> None:
    manifest = clean_numpy_bundle.manifest()
    nested: object = "leaf"
    for _ in range(33):
        nested = [nested]
    manifest["too_deep"] = nested
    clean_numpy_bundle.rewrite_manifest(manifest)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="depth|nest|32|recursion"):
        _load_bundle(clean_numpy_bundle)


def test_manifest_size_is_bounded_before_parse(clean_numpy_bundle: BundleCase) -> None:
    with clean_numpy_bundle.manifest_path.open("wb") as stream:
        stream.truncate(1_048_577)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="manifest|size|large|exceeds"):
        _load_bundle(clean_numpy_bundle)


def test_manifest_declaring_more_than_32_files_rejects_before_any_payload_read_or_hash(
    clean_numpy_bundle: BundleCase,
) -> None:
    manifest = clean_numpy_bundle.manifest()
    clean_numpy_bundle.file_path("model").write_bytes(b"tampered-before-count-preflight")
    for index in range(28):
        manifest["files"][f"extra{index:02d}"] = {
            "path": f"missing-{index:02d}.bin",
            "sha256": "0" * 64,
        }
    assert len(manifest["files"]) == 33
    clean_numpy_bundle.rewrite_manifest(manifest)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="32|declared files|file count|many"):
        _load_bundle(clean_numpy_bundle)


def test_manifest_aggregate_stat_over_512_mib_rejects_before_payload_read_or_hash(
    clean_numpy_bundle: BundleCase,
) -> None:
    manifest = clean_numpy_bundle.manifest()
    clean_numpy_bundle.file_path("model").write_bytes(b"tampered-before-stat-preflight")
    for index in range(5):
        sparse = clean_numpy_bundle.root / f"sparse-{index}.bin"
        with sparse.open("wb") as stream:
            stream.truncate(220 * 1024 * 1024)
        manifest["files"][f"sparse{index}"] = {
            "path": sparse.name,
            "sha256": "0" * 64,
        }
    declared_stat_bytes = sum(
        (clean_numpy_bundle.root / record["path"]).stat().st_size
        for record in manifest["files"].values()
    )
    assert declared_stat_bytes > 536_870_912
    clean_numpy_bundle.rewrite_manifest(manifest)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="aggregate|total|536870912|512 MiB|stat bytes"):
        _load_bundle(clean_numpy_bundle)


def test_manifest_aggregate_stat_exactly_512_mib_passes_complete_stat_preflight(
    clean_numpy_bundle: BundleCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.bundle as bundle_module

    manifest = clean_numpy_bundle.manifest()
    existing_bytes = sum(
        (clean_numpy_bundle.root / record["path"]).stat().st_size
        for record in manifest["files"].values()
    )
    exact = clean_numpy_bundle.root / "exact-aggregate-cap.bin"
    with exact.open("wb") as stream:
        stream.truncate(536_870_912 - existing_bytes)
    manifest["files"]["exact_cap"] = {"path": exact.name, "sha256": "0" * 64}
    clean_numpy_bundle.rewrite_manifest(manifest)

    original_read = bundle_module._read_pinned_regular_file

    def stop_before_large_payload(
        root: Path,
        root_fd: int,
        pinned: object,
        *,
        max_bytes: int,
        source: str,
    ) -> object:
        if pinned.relative_path == exact.name:
            raise RuntimeError("EXACT_CAP_REACHED_PAYLOAD_PHASE")
        return original_read(
            root,
            root_fd,
            pinned,
            max_bytes=max_bytes,
            source=source,
        )

    monkeypatch.setattr(
        bundle_module,
        "_read_pinned_regular_file",
        stop_before_large_payload,
    )
    with pytest.raises(RuntimeError, match="EXACT_CAP_REACHED_PAYLOAD_PHASE"):
        _load_bundle(clean_numpy_bundle)


def test_output_byte_caps_accept_exact_boundaries_and_fail_closed_without_destination(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bandtrace.audit as audit_module
    from bandtrace.cli import main as cli_main

    assert audit_module.MAX_SINGLE_OUTPUT_BYTES == 268_435_456
    assert audit_module.MAX_TOTAL_OUTPUT_BYTES == 536_870_912
    monkeypatch.setattr(
        audit_module,
        "_build_report",
        lambda bundle, **kwargs: ({"exit_code": 0}, {}),
    )
    monkeypatch.setattr(audit_module, "canonical_json_bytes", lambda report: b"r" * 400)
    monkeypatch.setattr(audit_module, "render_route_csv", lambda rows: b"c" * 400)
    monkeypatch.setattr(
        audit_module, "deterministic_npz_bytes", lambda arrays: b"n" * 400
    )
    monkeypatch.setattr(
        audit_module, "render_html", lambda report, hashes: b"h" * 400
    )

    discovery = tmp_path / "output-budget-discovery"
    audit_module.run_audit(clean_numpy_bundle.root, discovery)
    sizes = {path.name: path.stat().st_size for path in discovery.iterdir()}
    exact_single = max(sizes.values())
    exact_total = sum(sizes.values())
    assert exact_single == 400

    monkeypatch.setattr(audit_module, "MAX_SINGLE_OUTPUT_BYTES", exact_single)
    monkeypatch.setattr(audit_module, "MAX_TOTAL_OUTPUT_BYTES", exact_total)
    exact_destination = tmp_path / "output-budget-exact"
    audit_module.run_audit(clean_numpy_bundle.root, exact_destination)
    assert exact_destination.is_dir()

    monkeypatch.setattr(audit_module, "MAX_SINGLE_OUTPUT_BYTES", exact_single - 1)
    per_file_destination = tmp_path / "output-budget-per-file-over"
    per_file_exit = cli_main(
        [
            "audit",
            str(clean_numpy_bundle.root),
            "--output-dir",
            str(per_file_destination),
        ]
    )
    per_file_stderr = capsys.readouterr().err
    assert per_file_exit == 3
    assert "per-file byte limit" in per_file_stderr
    assert not per_file_destination.exists()
    assert not list(tmp_path.glob(f".{per_file_destination.name}.bandtrace-*"))

    monkeypatch.setattr(audit_module, "MAX_SINGLE_OUTPUT_BYTES", exact_single)
    monkeypatch.setattr(audit_module, "MAX_TOTAL_OUTPUT_BYTES", exact_total - 1)
    aggregate_destination = tmp_path / "output-budget-aggregate-over"
    aggregate_exit = cli_main(
        [
            "audit",
            str(clean_numpy_bundle.root),
            "--output-dir",
            str(aggregate_destination),
        ]
    )
    aggregate_stderr = capsys.readouterr().err
    assert aggregate_exit == 3
    assert "aggregate byte limit" in aggregate_stderr
    assert not aggregate_destination.exists()
    assert not list(tmp_path.glob(f".{aggregate_destination.name}.bandtrace-*"))


def test_audit_serializer_memory_error_is_stable_exit_3_without_publication_or_traceback(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bandtrace.audit as audit_module
    from bandtrace.cli import main as cli_main

    serialization_calls = 0

    def exhaust_memory(_arrays: object) -> bytes:
        nonlocal serialization_calls
        serialization_calls += 1
        raise MemoryError("injected canary serializer allocation failure")

    monkeypatch.setattr(audit_module, "deterministic_npz_bytes", exhaust_memory)
    destination = tmp_path / "serializer-memory-error"
    exit_code = cli_main(
        [
            "audit",
            str(clean_numpy_bundle.root),
            "--output-dir",
            str(destination),
        ]
    )
    stderr = capsys.readouterr().err

    assert exit_code == 3
    assert serialization_calls == 1
    assert (
        "audit execution or serialization exceeded available memory within frozen byte limits"
        in stderr
    )
    assert "Traceback" not in stderr
    assert "injected canary serializer" not in stderr
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


def test_initial_staging_descriptor_fstat_failure_cleans_stage_and_closes_fd(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bandtrace.audit as audit_module
    from bandtrace.cli import main as cli_main

    real_create_staging = audit_module._create_staging_directory
    real_fstat = audit_module.os.fstat
    created: list[tuple[str, int]] = []
    staging_fstat_calls = 0

    def track_created_stage(parent_fd: int) -> tuple[str, int]:
        staging_name, staging_fd = real_create_staging(parent_fd)
        created.append((staging_name, staging_fd))
        return staging_name, staging_fd

    def fail_first_staging_fstat(descriptor: int) -> os.stat_result:
        nonlocal staging_fstat_calls
        if created and descriptor == created[0][1]:
            staging_fstat_calls += 1
            if staging_fstat_calls == 1:
                raise OSError(
                    errno.EIO,
                    "injected staging fstat acquisition failure",
                )
        return real_fstat(descriptor)

    monkeypatch.setattr(
        audit_module,
        "_create_staging_directory",
        track_created_stage,
    )
    monkeypatch.setattr(audit_module.os, "fstat", fail_first_staging_fstat)
    destination = tmp_path / "staging-fstat-failure"

    exit_code = cli_main(
        [
            "audit",
            str(clean_numpy_bundle.root),
            "--output-dir",
            str(destination),
        ]
    )
    stderr = capsys.readouterr().err

    assert exit_code == 3
    assert "injected staging fstat acquisition failure" in stderr
    assert "Traceback" not in stderr
    assert len(created) == 1
    assert staging_fstat_calls >= 2
    staging_name, staging_fd = created[0]
    assert not (tmp_path / staging_name).exists()
    with pytest.raises(OSError) as closed:
        real_fstat(staging_fd)
    assert closed.value.errno == errno.EBADF
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


def test_atomic_noreplace_publication_preserves_destination_injected_at_race_seam(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.audit as audit_module

    destination = tmp_path / "race-destination"
    real_noreplace = audit_module._rename_directory_noreplace
    injected_inode: list[int] = []

    def inject_empty_destination_then_publish(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        os.mkdir(destination_name, mode=0o711, dir_fd=destination_directory_fd)
        injected_inode.append(
            os.stat(
                destination_name,
                dir_fd=destination_directory_fd,
                follow_symlinks=False,
            ).st_ino
        )
        real_noreplace(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        )

    monkeypatch.setattr(
        audit_module,
        "_rename_directory_noreplace",
        inject_empty_destination_then_publish,
    )

    with pytest.raises(OSError) as caught:
        audit_module.run_audit(clean_numpy_bundle.root, destination)

    assert caught.value.errno == errno.EEXIST
    assert injected_inode
    assert destination.is_dir()
    assert destination.stat().st_ino == injected_inode[0]
    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


def test_staging_source_substitution_is_intercepted_before_atomic_publication(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.audit as audit_module

    destination = tmp_path / "precheck-destination"
    captured_name = ".captured-legitimate-bandtrace-stage"
    captured = tmp_path / captured_name
    replacement_sentinel = b"same-uid replacement; never trust or delete\n"
    real_verify = audit_module._verify_staging_source_identity
    precheck_errors: list[str] = []
    staging_names: list[str] = []
    rename_calls: list[tuple[object, ...]] = []

    def substitute_then_delegate_to_real_precheck(
        parent_fd: int,
        staging_name: str,
        staging_fd: int,
    ) -> None:
        staging_names.append(staging_name)
        os.rename(
            staging_name,
            captured_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        (tmp_path / staging_name / "replacement-sentinel").write_bytes(
            replacement_sentinel
        )
        try:
            real_verify(parent_fd, staging_name, staging_fd)
        except OSError as error:
            precheck_errors.append(str(error))
            raise

    def record_forbidden_rename(*args: object) -> None:
        rename_calls.append(args)
        raise AssertionError("atomic rename reached after failed staging-source precheck")

    monkeypatch.setattr(
        audit_module,
        "_verify_staging_source_identity",
        substitute_then_delegate_to_real_precheck,
    )
    monkeypatch.setattr(
        audit_module,
        "_rename_directory_noreplace",
        record_forbidden_rename,
    )

    with pytest.raises(OSError, match="cannot clean failed BandTrace output staging directory"):
        audit_module.run_audit(clean_numpy_bundle.root, destination)

    assert len(staging_names) == 1
    assert precheck_errors == [
        "staging source identity changed before atomic publication; refusing publication"
    ]
    assert rename_calls == []
    assert not destination.exists()

    replacement = tmp_path / staging_names[0]
    assert replacement.is_dir()
    assert (replacement / "replacement-sentinel").read_bytes() == replacement_sentinel
    assert captured.is_dir()
    assert {path.name for path in captured.iterdir()} == {
        "canary_outputs.npz",
        "manifest.sha256",
        "report.html",
        "report.json",
        "route.csv",
    }


def test_cleanup_source_substitution_is_intercepted_immediately_before_rmdir(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.audit as audit_module

    destination = tmp_path / "cleanup-pre-rmdir-destination"
    captured_name = ".captured-empty-legitimate-stage"
    captured = tmp_path / captured_name
    replacement_sentinel = b"replacement at cleanup rmdir seam; preserve\n"
    real_verify = audit_module._verify_staging_source_identity
    verification_calls = 0
    cleanup_identity_errors: list[str] = []
    staging_names: list[str] = []
    rename_calls: list[tuple[object, ...]] = []

    def substitute_on_cleanup_pre_rmdir_check(
        parent_fd: int,
        staging_name: str,
        staging_fd: int,
    ) -> None:
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 1:
            real_verify(parent_fd, staging_name, staging_fd)
            return
        assert verification_calls == 2
        staging_names.append(staging_name)
        os.rename(
            staging_name,
            captured_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_fd)
        (tmp_path / staging_name / "replacement-sentinel").write_bytes(
            replacement_sentinel
        )
        try:
            real_verify(parent_fd, staging_name, staging_fd)
        except OSError as error:
            cleanup_identity_errors.append(str(error))
            raise

    def fail_before_publication(*args: object) -> None:
        rename_calls.append(args)
        raise OSError(errno.EIO, "injected failure before atomic publication")

    monkeypatch.setattr(
        audit_module,
        "_verify_staging_source_identity",
        substitute_on_cleanup_pre_rmdir_check,
    )
    monkeypatch.setattr(
        audit_module,
        "_rename_directory_noreplace",
        fail_before_publication,
    )

    with pytest.raises(OSError, match="cannot clean failed BandTrace output staging directory"):
        audit_module.run_audit(clean_numpy_bundle.root, destination)

    assert verification_calls == 2
    assert len(rename_calls) == 1
    assert len(staging_names) == 1
    assert cleanup_identity_errors == [
        "staging source identity changed before atomic publication; refusing publication"
    ]
    assert not destination.exists()

    replacement = tmp_path / staging_names[0]
    assert replacement.is_dir()
    assert (replacement / "replacement-sentinel").read_bytes() == replacement_sentinel
    assert captured.is_dir()
    assert list(captured.iterdir()) == []


def test_report_discloses_exact_publication_and_rounded_boundary_contracts(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    from bandtrace.audit import run_audit

    destination = tmp_path / "publication-contract-report"
    run_audit(clean_numpy_bundle.root, destination)
    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))

    assert report["facts"]["output_publication_contract"] == {
        "platform": "Linux",
        "mechanism": "renameat2(RENAME_NOREPLACE)",
        "fresh_destination_required": True,
        "private_parent_required": True,
        "source_identity_prechecked": True,
        "source_inode_conditioned_rename": False,
        "destination_identity_postchecked": True,
        "prepublication_cleanup_recursive": False,
        "prepublication_cleanup_source_identity_rechecked_before_rmdir": True,
        "same_uid_parent_name_race_fully_prevented": False,
        "postpublication_rollback": False,
    }
    assert report["facts"]["installed_source_digest_scope"] == (
        "REGULAR_PY_FILES_ONLY"
    )
    assert report["facts"]["execution_environment_attested"] is False
    assert report["facts"]["native_dependency_bytes_hashed"] is False
    assert report["facts"]["runtime_fingerprint_non_exhaustive"] is True
    assert report["facts"]["packaged_hash_gate_is_external_authentication"] is False
    assert report["facts"]["configured_time_budgets"] == {
        "subprocess_timeout_seconds": 120,
        "adapter_total_measured_seconds_failure_threshold": 600.0,
        "adapter_total_hard_deadline": False,
        "synchronous_parent_and_cleanup_preemptible": False,
        "subprocess_child_active_wall_poll": True,
        "current_adapter_uses_active_child_wall_poll": False,
        "external_hard_deadline_supervisor_required": True,
    }
    assert (
        "Categorical decisions use unrounded binary64 values; an eight-decimal "
        "metric and threshold can display as equal at a boundary, so status/pass "
        "fields are authoritative."
    ) in report["limitations"]
    assert (
        "Output publication and prepublication cleanup require a private, trusted "
        "parent directory; Linux renameat2(RENAME_NOREPLACE) and rmdir cannot bind "
        "the source name to the staged inode against a same-UID actor."
    ) in report["limitations"]
    assert (
        "The installed source-tree digest covers regular BandTrace .py files only; "
        "it is Python-source provenance, not execution attestation, and excludes "
        "bytecode, interpreter, native/dependency bytes, environment state, and "
        "in-memory mutation."
    ) in report["limitations"]
    assert (
        "The 600-second cumulative adapter wall threshold is measured and enforced "
        "at checkpoints; synchronous parent work and cleanup are not preemptible, "
        "so a hard end-to-end deadline requires an external supervisor."
    ) in report["limitations"]
    assert (
        "The packaged normative hash gate checks build-internal byte consistency; "
        "coordinated replacement of code, embedded hashes, and resources is not "
        "detected, so authenticity requires an independently trusted distribution "
        "digest or signature."
    ) in report["limitations"]


def test_parent_swap_after_publication_fails_without_touching_replacement_parent(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.audit as audit_module

    live_parent = tmp_path / "live-parent"
    moved_parent = tmp_path / "moved-original-parent"
    live_parent.mkdir(mode=0o700)
    destination = live_parent / "report"
    real_noreplace = audit_module._rename_directory_noreplace
    replacement_inode: list[int] = []

    def publish_then_swap_parent(
        source_directory_fd: int,
        source_name: str,
        destination_directory_fd: int,
        destination_name: str,
    ) -> None:
        real_noreplace(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        )
        os.rename(live_parent, moved_parent)
        live_parent.mkdir(mode=0o711)
        (live_parent / "replacement-sentinel").write_bytes(b"replacement-owned\n")
        replacement_inode.append(live_parent.stat().st_ino)

    monkeypatch.setattr(
        audit_module,
        "_rename_directory_noreplace",
        publish_then_swap_parent,
    )

    with pytest.raises(
        OSError, match="output parent directory identity changed during audit"
    ):
        audit_module.run_audit(clean_numpy_bundle.root, destination)

    assert replacement_inode
    assert live_parent.stat().st_ino == replacement_inode[0]
    assert (live_parent / "replacement-sentinel").read_bytes() == b"replacement-owned\n"
    assert not destination.exists()
    assert {path.name for path in live_parent.iterdir()} == {"replacement-sentinel"}

    published = moved_parent / destination.name
    expected_outputs = {
        "canary_outputs.npz",
        "manifest.sha256",
        "report.html",
        "report.json",
        "route.csv",
    }
    assert published.is_dir()
    assert {path.name for path in published.iterdir()} == expected_outputs
    manifest_names = {
        line.split(maxsplit=1)[1].lstrip(" *")
        for line in (published / "manifest.sha256").read_text(encoding="ascii").splitlines()
    }
    assert manifest_names == expected_outputs - {"manifest.sha256"}
    assert not list(moved_parent.glob(".bandtrace-stage-*"))


@pytest.mark.parametrize("resource_failure", ["missing", "tampered"])
def test_normative_resource_failure_aborts_before_bundle_load_or_publication(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    resource_failure: str,
) -> None:
    import bandtrace.audit as audit_module
    import bandtrace.authority as authority_module
    from bandtrace.cli import main as cli_main

    load_calls: list[Path] = []
    real_load_bundle = audit_module.load_bundle

    def track_load_bundle(path: Path) -> object:
        load_calls.append(path)
        return real_load_bundle(path)

    monkeypatch.setattr(audit_module, "load_bundle", track_load_bundle)
    if resource_failure == "missing":
        def missing_package(_package: str) -> object:
            raise ModuleNotFoundError("injected missing normative package")

        monkeypatch.setattr(
            authority_module.importlib.resources,
            "files",
            missing_package,
        )
    else:
        real_read = authority_module._read_normative_resource

        def tampered_document(filename: str) -> bytes:
            payload = real_read(filename)
            return payload + b"\ninjected tamper\n" if filename.endswith(".md") else payload

        monkeypatch.setattr(
            authority_module,
            "_read_normative_resource",
            tampered_document,
        )

    destination = tmp_path / f"normative-{resource_failure}"
    exit_code = cli_main(
        [
            "audit",
            str(clean_numpy_bundle.root),
            "--output-dir",
            str(destination),
        ]
    )
    stderr = capsys.readouterr().err

    assert exit_code == 3
    assert "packaged normative authority" in stderr
    assert "unavailable" in stderr or "hash mismatch" in stderr
    assert load_calls == []
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


def test_missing_installed_source_attestation_aborts_before_bundle_or_adapter_execution(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bandtrace.audit as audit_module
    from bandtrace.cli import main as cli_main
    from bandtrace.errors import ExecutionError

    load_calls: list[Path] = []
    adapter_calls: list[object] = []

    def unavailable_source_tree() -> str:
        raise ExecutionError(
            "installed BandTrace source tree contains no regular Python members"
        )

    def record_load(path: Path) -> object:
        load_calls.append(path)
        raise AssertionError("bundle load reached after source attestation failure")

    def record_adapter(bundle: object) -> object:
        adapter_calls.append(bundle)
        raise AssertionError("adapter construction reached after source attestation failure")

    monkeypatch.setattr(
        audit_module,
        "installed_source_tree_sha256",
        unavailable_source_tree,
    )
    monkeypatch.setattr(audit_module, "load_bundle", record_load)
    monkeypatch.setattr(audit_module, "build_adapter", record_adapter)
    destination = tmp_path / "missing-source-attestation"

    exit_code = cli_main(
        [
            "audit",
            str(clean_numpy_bundle.root),
            "--output-dir",
            str(destination),
        ]
    )
    stderr = capsys.readouterr().err

    assert exit_code == 3
    assert "installed BandTrace source tree contains no regular Python members" in stderr
    assert "Traceback" not in stderr
    assert load_calls == []
    assert adapter_calls == []
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


@pytest.mark.parametrize(
    ("metadata_failure", "expected_message"),
    [
        (
            "invalid_utf8",
            "cannot read installed BandTrace distribution metadata",
        ),
        (
            "none",
            "installed BandTrace distribution version is missing or empty",
        ),
        (
            "empty",
            "installed BandTrace distribution version is missing or empty",
        ),
    ],
)
def test_invalid_installed_distribution_metadata_aborts_before_bundle_or_adapter(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    metadata_failure: str,
    expected_message: str,
) -> None:
    import bandtrace.audit as audit_module
    import bandtrace.canonical as canonical_module
    from bandtrace.cli import main as cli_main

    load_calls: list[Path] = []
    adapter_calls: list[object] = []

    def invalid_metadata(_distribution: str) -> object:
        if metadata_failure == "invalid_utf8":
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid metadata")
        return None if metadata_failure == "none" else ""

    def record_load(path: Path) -> object:
        load_calls.append(path)
        raise AssertionError("bundle load reached after distribution metadata failure")

    def record_adapter(bundle: object) -> object:
        adapter_calls.append(bundle)
        raise AssertionError("adapter construction reached after distribution metadata failure")

    monkeypatch.setattr(
        canonical_module.importlib.metadata,
        "version",
        invalid_metadata,
    )
    monkeypatch.setattr(audit_module, "load_bundle", record_load)
    monkeypatch.setattr(audit_module, "build_adapter", record_adapter)
    destination = tmp_path / f"distribution-metadata-{metadata_failure}"

    exit_code = cli_main(
        [
            "audit",
            str(clean_numpy_bundle.root),
            "--output-dir",
            str(destination),
        ]
    )
    stderr = capsys.readouterr().err

    assert exit_code == 3
    assert expected_message in stderr
    assert "Traceback" not in stderr
    assert "invalid metadata" not in stderr
    assert load_calls == []
    assert adapter_calls == []
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


def test_renameat2_unavailable_fails_closed_without_os_rename_fallback(
    clean_numpy_bundle: BundleCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.audit as audit_module

    class LibcWithoutRenameAt2:
        pass

    fallback_calls: list[tuple[object, ...]] = []
    load_calls: list[Path] = []
    real_load_bundle = audit_module.load_bundle

    def forbidden_rename(*args: object) -> None:
        fallback_calls.append(args)
        raise AssertionError("unsafe os.rename fallback was called")

    def track_load_bundle(path: Path) -> object:
        load_calls.append(path)
        return real_load_bundle(path)

    monkeypatch.setattr(audit_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        audit_module.ctypes,
        "CDLL",
        lambda *args, **kwargs: LibcWithoutRenameAt2(),
    )
    monkeypatch.setattr(audit_module.os, "rename", forbidden_rename)
    monkeypatch.setattr(audit_module, "load_bundle", track_load_bundle)
    destination = tmp_path / "renameat2-unavailable"

    with pytest.raises(OSError) as caught:
        audit_module.run_audit(clean_numpy_bundle.root, destination)

    assert caught.value.errno == errno.ENOTSUP
    assert "no unsafe fallback" in str(caught.value).lower() or "refusing unsafe" in str(
        caught.value
    ).lower()
    assert fallback_calls == []
    assert load_calls == []
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.bandtrace-*"))


def test_probe_file_size_is_checked_before_hash_or_deserialisation(
    clean_numpy_bundle: BundleCase,
) -> None:
    with clean_numpy_bundle.file_path("probes").open("wb") as stream:
        stream.truncate(268_435_457)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="probe|size|large"):
        _load_bundle(clean_numpy_bundle)


def test_expanded_float64_probe_budget_is_checked_before_dtype_conversion(
    clean_numpy_bundle: BundleCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The stored int8 payload is tiny, but its canonical 100-element float64 expansion is 800 B.
    # Lowering only the expansion cap proves this guard is distinct from archive/file-size limits.
    _replace_probes(
        clean_numpy_bundle,
        probes=np.zeros((20, 5), dtype=np.int8),
        target_band_ids=np.asarray(TARGET_BAND_IDS),
    )
    import bandtrace.bundle as bundle_module
    from bandtrace.errors import BundleError

    monkeypatch.setattr(bundle_module, "MAX_EXPANDED_FLOAT64_PROBE_BYTES", 799, raising=False)
    with pytest.raises(BundleError, match="float64|expansion|expanded|byte budget"):
        _load_bundle(clean_numpy_bundle)


def test_real_probe_shape_over_256_mib_float64_expansion_rejects_before_conversion(
    bundle_factory: BundleFactory,
) -> None:
    bundle = bundle_factory()
    sensor = bundle.read_json("sensor")
    extra_ids = [f"tx{index:03d}" for index in range(507)]
    extra_bands = [_band(identifier, 600.0, target=True) for identifier in extra_ids]
    sensor["target_bands"].extend(extra_bands)
    bundle.rewrite_json("sensor", sensor)

    target_ids = TARGET_BAND_IDS + extra_ids
    matrix = np.zeros((4, 512), dtype=np.float64)
    matrix[np.arange(4), np.arange(4)] = 1.0
    route = bundle.read_json("route")
    route["target_band_ids"] = target_ids
    route["matrix"] = matrix.tolist()
    bundle.rewrite_json("route", route)

    artifact_path = bundle.file_path("artifact")
    with np.load(artifact_path, allow_pickle=False) as archive:
        artifact = {key: np.array(archive[key], copy=True) for key in archive.files}
    artifact["route_matrix"] = matrix
    artifact["target_band_ids"] = np.asarray(target_ids)
    np.savez(artifact_path, **artifact)
    bundle.refresh_hash("artifact")

    shape = (16, 512, 1, 4097)
    expanded_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype("float64").itemsize
    assert expanded_bytes > 268_435_456
    probes = np.zeros(shape, dtype=np.int8)
    np.savez(
        bundle.file_path("probes"),
        probes=probes,
        target_band_ids=np.asarray(target_ids),
    )
    del probes
    assert bundle.file_path("probes").stat().st_size < 268_435_456
    bundle.refresh_hash("probes")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="float64|expansion|expanded|268435456|256 MiB"):
        _load_bundle(bundle)


def test_object_dtype_npz_is_rejected_without_pickle(clean_numpy_bundle: BundleCase) -> None:
    _replace_probes(
        clean_numpy_bundle,
        probes=np.asarray([[{"payload": "forbidden"}]] * 20, dtype=object),
        target_band_ids=np.asarray(TARGET_BAND_IDS),
    )

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="object|pickle|dtype"):
        _load_bundle(clean_numpy_bundle)


def test_npz_duplicate_archive_member_is_rejected(clean_numpy_bundle: BundleCase) -> None:
    probes_path = clean_numpy_bundle.file_path("probes")
    with zipfile.ZipFile(probes_path, "r") as archive:
        member = archive.read("probes.npy")
    with zipfile.ZipFile(probes_path, "a") as archive:
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("probes.npy", member)
    clean_numpy_bundle.refresh_hash("probes")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="duplicate|member"):
        _load_bundle(clean_numpy_bundle)


def test_npz_member_count_is_bounded(clean_numpy_bundle: BundleCase) -> None:
    probes_path = clean_numpy_bundle.file_path("probes")
    buffer = io.BytesIO()
    np.save(buffer, np.asarray([1.0], dtype=np.float64), allow_pickle=False)
    with zipfile.ZipFile(probes_path, "a") as archive:
        for index in range(31):
            archive.writestr(f"extra_{index}.npy", buffer.getvalue())
    clean_numpy_bundle.refresh_hash("probes")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="member|32|many"):
        _load_bundle(clean_numpy_bundle)


def test_npz_compression_bomb_is_rejected_before_array_materialisation(
    clean_numpy_bundle: BundleCase,
) -> None:
    np.savez_compressed(
        clean_numpy_bundle.file_path("probes"),
        probes=np.zeros((20, 5, 128, 128), dtype=np.float64),
        target_band_ids=np.asarray(TARGET_BAND_IDS),
    )
    clean_numpy_bundle.refresh_hash("probes")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="compression|ratio|bomb"):
        _load_bundle(clean_numpy_bundle)


@pytest.mark.parametrize(
    ("compression", "name"),
    [(zipfile.ZIP_BZIP2, "BZIP2"), (zipfile.ZIP_LZMA, "LZMA")],
)
def test_npz_bzip2_and_lzma_members_are_rejected_by_the_frozen_codec_allowlist(
    clean_numpy_bundle: BundleCase,
    compression: int,
    name: str,
) -> None:
    probe_path = clean_numpy_bundle.file_path("probes")
    with np.load(probe_path, allow_pickle=False) as archive:
        arrays = {member: np.array(archive[member], copy=True) for member in archive.files}
    _write_npz_with_compression(probe_path, arrays, compression)
    clean_numpy_bundle.refresh_hash("probes")

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="compression|method|STORED|DEFLATED") as error:
        _load_bundle(clean_numpy_bundle)
    assert name.lower() in str(error.value).lower() or "unsupported" in str(error.value).lower()


def test_extra_endpoint_or_label_arrays_are_forbidden(clean_numpy_bundle: BundleCase) -> None:
    _replace_probes(
        clean_numpy_bundle,
        probes=varied_probes(),
        target_band_ids=np.asarray(TARGET_BAND_IDS),
        labels=np.arange(20, dtype=np.int64),
    )

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="labels|unexpected|member|keys|exactly"):
        _load_bundle(clean_numpy_bundle)


def test_nonfinite_probes_fail_closed(clean_numpy_bundle: BundleCase) -> None:
    probes = varied_probes()
    probes[3, 2] = np.nan
    _replace_probes(
        clean_numpy_bundle,
        probes=probes,
        target_band_ids=np.asarray(TARGET_BAND_IDS),
    )

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="finite|NaN|non-finite"):
        _load_bundle(clean_numpy_bundle)


def test_subprocess_argv_missing_required_artifact_placeholder_is_rejected(
    clean_subprocess_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    rogue = tmp_path / "rogue.py"
    rogue.write_text("raise SystemExit(0)\n", encoding="utf-8")
    manifest = clean_subprocess_bundle.manifest()
    manifest["adapter"]["argv"][1] = str(rogue)
    clean_subprocess_bundle.rewrite_manifest(manifest)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="artifact|placeholder|pinned|argv"):
        _load_bundle(clean_subprocess_bundle)


def test_ordinary_path_looking_subprocess_argv_tokens_are_load_only_verbatim_and_unattested(
    clean_subprocess_bundle: BundleCase,
    tmp_path: Path,
) -> None:
    manifest = clean_subprocess_bundle.manifest()
    ordinary_tokens = [
        "relative/tools/helper.py",
        "../operator-owned/config.json",
        str(tmp_path / "absolute-unmanaged-resource.dat"),
    ]
    manifest["adapter"]["argv"].extend(
        ["--ordinary-unmanaged-token", *ordinary_tokens]
    )
    clean_subprocess_bundle.rewrite_manifest(manifest)

    loaded = _load_bundle(clean_subprocess_bundle)

    assert loaded.adapter["argv"][-4:] == (
        "--ordinary-unmanaged-token",
        *ordinary_tokens,
    )
    assert loaded.adapter["argv"].count("{artifact}") == 1
    assert loaded.adapter["argv"].count("{input_npz}") == 1
    assert loaded.adapter["argv"].count("{output_npz}") == 1
    pinned_relative_paths = {
        record.relative_path for record in loaded.files.values()
    }
    assert set(ordinary_tokens).isdisjoint(pinned_relative_paths)
    assert loaded.adapter["asset_keys"] == ()


def test_subprocess_staging_exposes_no_contracts_or_original_probe_archive(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="scan_isolation")
    output_dir = tmp_path / "isolated-staging"
    completed = _cli(bundle, output_dir)

    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["facts"]["subprocess_staging_contract"] == (
        "ARTIFACT_AND_EXPLICIT_PINNED_ASSETS_ONLY"
    )
    assert report["facts"]["subprocess_original_probes_staged"] is False


def test_direct_subprocess_artifact_remains_executable_after_staging(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="assert_staged_modes",
        subprocess_direct=True,
    )
    completed = _cli(bundle, tmp_path / "direct-executable")

    assert completed.returncode == 0, (completed.stdout, completed.stderr)


def test_exact_pinned_bundle_relative_argv0_runner_is_staged_mode_0700(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="assert_staged_modes",
        subprocess_runner=True,
    )
    manifest = bundle.manifest()
    assert manifest["adapter"]["argv"][0] == "{asset:runner}"
    assert manifest["files"]["runner"]["path"] == "runner.py"
    completed = _cli(bundle, tmp_path / "argv0-runner-mode")

    assert completed.returncode == 0, (completed.stdout, completed.stderr)


def test_non_runner_staged_asset_is_0600_even_when_source_is_executable(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="assert_staged_modes",
        subprocess_extra_executable_asset=True,
    )
    assert bundle.file_path("asset").stat().st_mode & 0o777 == 0o700
    completed = _cli(bundle, tmp_path / "non-runner-asset-mode")

    assert completed.returncode == 0, (completed.stdout, completed.stderr)


@pytest.mark.parametrize(
    "mode",
    [
        "wrong_output_shape",
        "wrong_tap_shape",
        "wrong_dtype",
        "extra_output_member",
        "malformed_object_output",
        "nonfinite_output",
        "extreme_output_response",
        "extreme_tap_response",
        "bzip2_output",
        "lzma_output",
    ],
)
def test_subprocess_output_contract_rejects_shape_dtype_member_pickle_and_finite_faults(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    mode: str,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode=mode)
    completed = _cli(bundle, tmp_path / f"out-{mode}")

    assert completed.returncode == 3, (mode, completed.stdout, completed.stderr)
    assert not (tmp_path / f"out-{mode}" / "report.json").exists()


@pytest.mark.parametrize(
    "mode",
    ["execution_failure", "oversized_output", "stdout_flood", "stdout_infinite"],
)
def test_subprocess_resource_and_nonzero_failures_use_execution_exit_three(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    mode: str,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode=mode)
    completed = _cli(bundle, tmp_path / f"out-{mode}")

    assert completed.returncode == 3, (mode, completed.stdout, completed.stderr)
    assert not (tmp_path / f"out-{mode}" / "report.json").exists()


def test_bounded_report_html_omits_unselected_supplier_free_text(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    payload = '<script>alert("bandtrace")</script>&'

    def inject_sensor_text(sensor: dict[str, object]) -> None:
        sensor["sensor_model"] = payload
        sensor["calibration_state"] = payload

    bundle = bundle_factory(sensor_mutator=inject_sensor_text)
    output_dir = tmp_path / "escaped"
    completed = _cli(bundle, output_dir)

    assert completed.returncode == 0, completed.stderr
    html = (output_dir / "report.html").read_text(encoding="utf-8")
    assert payload not in html
    assert "sensor_model" not in html
    assert "sensor_calibration_state" not in html


@pytest.mark.parametrize("rerun_kind", ["clean", "invalid", "execution"])
def test_existing_output_directory_is_refused_before_any_rerun_can_reuse_stale_reports(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    rerun_kind: str,
) -> None:
    stale_dir = tmp_path / f"stale-{rerun_kind}"
    first = _cli(bundle_factory(), stale_dir)
    assert first.returncode == 0, first.stderr
    stale_bytes = {
        path.name: path.read_bytes()
        for path in sorted(stale_dir.iterdir())
        if path.is_file()
    }

    if rerun_kind == "execution":
        rerun_bundle = bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="execution_failure",
        )
    else:
        rerun_bundle = bundle_factory()
    if rerun_kind == "invalid":
        manifest = rerun_bundle.manifest()
        manifest["policy_id"] = "not-bandtrace-0.1"
        rerun_bundle.rewrite_manifest(manifest)

    completed = _cli(rerun_bundle, stale_dir)

    assert completed.returncode == 3
    assert "output failure" in completed.stderr.lower()
    assert "fresh destination" in completed.stderr.lower()
    assert "audit completed" not in completed.stdout.lower()
    assert {
        path.name: path.read_bytes()
        for path in sorted(stale_dir.iterdir())
        if path.is_file()
    } == stale_bytes


@pytest.mark.parametrize(
    ("failure_kind", "expected_exit"),
    [("invalid", 2), ("execution", 3)],
)
def test_invalid_or_execution_failure_never_creates_a_fresh_output_directory(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    failure_kind: str,
    expected_exit: int,
) -> None:
    if failure_kind == "execution":
        bundle = bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="execution_failure",
        )
    else:
        bundle = bundle_factory()
        manifest = bundle.manifest()
        manifest["policy_id"] = "not-bandtrace-0.1"
        bundle.rewrite_manifest(manifest)
    output_dir = tmp_path / f"fresh-failure-{failure_kind}"

    completed = _cli(bundle, output_dir)

    assert completed.returncode == expected_exit
    assert not output_dir.exists()


def test_process_group_cleanup_disclosures_are_scoped_to_subprocess_reports(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    from bandtrace.audit import run_audit

    expected_subprocess_facts = {
        "subprocess_group_signal_attempted_before_leader_reap": True,
        "subprocess_group_cleanup_guaranteed_on_os_signal_failure": False,
        "subprocess_group_signal_failure_can_leave_members_running": True,
        "subprocess_group_signal_failure_state": "execution_failure",
        (
            "subprocess_direct_leader_kill_and_reap_fallback_configured_"
            "on_group_signal_error"
        ): True,
    }
    cleanup_limitation = (
        "Subprocess cleanup attempts one same-group SIGKILL before leader reap. "
        "If the OS rejects that group signal, BandTrace fails the run and still "
        "attempts to kill and reap the leader, but cannot guarantee cleanup of "
        "same-group descendants."
    )

    numpy_result = run_audit(
        bundle_factory().root,
        tmp_path / "numpy-cleanup-scope",
    )
    subprocess_result = run_audit(
        bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="clean",
        ).root,
        tmp_path / "subprocess-cleanup-scope",
    )

    numpy_facts = numpy_result.report["facts"]
    assert expected_subprocess_facts.keys().isdisjoint(numpy_facts)
    assert cleanup_limitation not in numpy_result.report["limitations"]

    subprocess_facts = subprocess_result.report["facts"]
    assert {
        key: subprocess_facts[key] for key in expected_subprocess_facts
    } == expected_subprocess_facts
    assert cleanup_limitation in subprocess_result.report["limitations"]


@pytest.mark.parametrize(
    "mode",
    ["timeout_with_child", "parent_exits_child_holds_pipes"],
)
def test_timeout_kills_the_entire_subprocess_group_including_children(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode=mode)
    marker = tmp_path / f"{mode}-child-survived.marker"
    monkeypatch.setenv("BANDTRACE_TEST_MARKER", str(marker))

    import bandtrace.adapters as adapters
    from bandtrace.audit import run_audit
    from bandtrace.errors import ExecutionError

    monkeypatch.setattr(adapters, "SUBPROCESS_TIMEOUT_SECONDS", 0.25)
    with pytest.raises(ExecutionError, match="exceeded|timeout"):
        run_audit(bundle.root, tmp_path / "timeout-output")

    time.sleep(2.25)
    assert not marker.exists(), "timed-out adapter child escaped its process group"


def test_successful_adapter_parent_kills_remaining_members_of_its_created_group(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="success_with_child")
    marker = tmp_path / "success-child-survived.marker"
    monkeypatch.setenv("BANDTRACE_TEST_MARKER", str(marker))

    from bandtrace.audit import run_audit

    result = run_audit(bundle.root, tmp_path / "success-with-child")
    assert result.exit_code == 0
    time.sleep(2.25)
    assert not marker.exists(), "successful adapter child escaped same-group cleanup"


def test_subprocess_group_is_signalled_once_while_leader_remains_unreaped(
    bundle_factory: BundleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.adapters as adapters_module
    from bandtrace.adapters import build_adapter

    bundle = _load_bundle(bundle_factory(adapter="subprocess-npz-v1"))
    adapter = build_adapter(bundle)
    request = _baseline_invocation(bundle)
    real_kill_group = getattr(adapter, "_kill_group")
    real_killpg = adapters_module.os.killpg
    active_process: list[object] = []
    helper_entries: list[tuple[int, object]] = []
    signal_observations: list[tuple[int, object]] = []
    helper_returns: list[tuple[int, object, int]] = []

    def inspect_killpg(process_group_id: int, signal_number: object) -> None:
        assert len(active_process) == 1
        process = active_process[0]
        signal_observations.append(
            (process_group_id, getattr(process, "returncode"))
        )
        real_killpg(process_group_id, signal_number)

    def inspect_kill_group(process: object) -> int:
        helper_entries.append((getattr(process, "pid"), getattr(process, "returncode")))
        active_process.append(process)
        try:
            return_code = real_kill_group(process)
        finally:
            active_process.pop()
        helper_returns.append(
            (getattr(process, "pid"), getattr(process, "returncode"), return_code)
        )
        return return_code

    monkeypatch.setattr(adapters_module.os, "killpg", inspect_killpg)
    monkeypatch.setattr(adapter, "_kill_group", inspect_kill_group)
    temporary_root = Path(getattr(adapter, "_root"))
    try:
        for _ in range(2):
            response = adapter.invoke(request)
            assert response.output.shape == (request.probes.shape[0],)
    finally:
        adapter.close()

    assert len(helper_entries) == 2
    assert len(signal_observations) == 2
    assert len(helper_returns) == 2
    assert all(returncode is None for _, returncode in helper_entries)
    assert all(returncode is None for _, returncode in signal_observations)
    assert all(process_returncode == returned for _, process_returncode, returned in helper_returns)
    assert not temporary_root.exists()


@pytest.mark.parametrize(
    ("group_signal_error", "error_number"),
    [("unexpected_oserror", errno.EIO), ("permission_error", errno.EACCES)],
)
def test_group_signal_oserror_reaps_leader_but_does_not_claim_descendant_cleanup(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    group_signal_error: str,
    error_number: int,
) -> None:
    import bandtrace.adapters as adapters_module
    import bandtrace.audit as audit_module
    from bandtrace.cli import main as cli_main

    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="success_with_child",
    )
    marker = tmp_path / f"{group_signal_error}-descendant-survived.marker"
    monkeypatch.setenv("BANDTRACE_TEST_MARKER", str(marker))

    real_build_adapter = audit_module.build_adapter
    adapter_roots: list[Path] = []

    def track_adapter_root(loaded: object) -> object:
        adapter = real_build_adapter(loaded)
        adapter_roots.append(Path(getattr(adapter, "_root")))
        return adapter

    real_kill_group = adapters_module.SubprocessNpzAdapter._kill_group
    real_kill = adapters_module.os.kill
    processes: list[object] = []
    kill_group_entries: list[tuple[int, object]] = []
    kill_group_exits: list[tuple[int, object]] = []
    group_signal_attempts: list[tuple[int, object, object]] = []
    direct_signal_attempts: list[tuple[int, object, object]] = []

    def fail_group_signal(process_group_id: int, signal_number: object) -> None:
        assert len(processes) == 1
        process = processes[0]
        group_signal_attempts.append(
            (process_group_id, signal_number, getattr(process, "returncode"))
        )
        message = f"injected {group_signal_error}"
        if group_signal_error == "permission_error":
            raise PermissionError(error_number, message)
        raise OSError(error_number, message)

    def inspect_direct_signal(process_id: int, signal_number: object) -> None:
        assert len(processes) == 1
        process = processes[0]
        direct_signal_attempts.append(
            (process_id, signal_number, getattr(process, "returncode"))
        )
        real_kill(process_id, signal_number)

    def inspect_kill_group(self: object, process: object) -> int:
        kill_group_entries.append(
            (getattr(process, "pid"), getattr(process, "returncode"))
        )
        processes.append(process)
        try:
            return real_kill_group(self, process)
        finally:
            processes.pop()
            kill_group_exits.append(
                (getattr(process, "pid"), getattr(process, "returncode"))
            )

    monkeypatch.setattr(audit_module, "build_adapter", track_adapter_root)
    monkeypatch.setattr(adapters_module.os, "killpg", fail_group_signal)
    monkeypatch.setattr(adapters_module.os, "kill", inspect_direct_signal)
    monkeypatch.setattr(
        adapters_module.SubprocessNpzAdapter,
        "_kill_group",
        inspect_kill_group,
    )
    destination = tmp_path / f"{group_signal_error}-output"

    exit_code = cli_main(
        ["audit", str(bundle.root), "--output-dir", str(destination)]
    )
    captured = capsys.readouterr()

    assert exit_code == 3
    assert captured.out == ""
    assert captured.err == (
        "BandTrace execution failure: cannot signal trusted subprocess process group; "
        "descendant cleanup is not established\n"
    )
    assert "Traceback" not in captured.err
    assert len(kill_group_entries) == 1
    assert len(kill_group_exits) == 1
    leader_pid, entry_returncode = kill_group_entries[0]
    assert entry_returncode is None
    assert kill_group_exits[0][0] == leader_pid
    assert kill_group_exits[0][1] is not None
    assert group_signal_attempts == [
        (leader_pid, adapters_module.signal.SIGKILL, None)
    ]
    assert direct_signal_attempts == [
        (leader_pid, adapters_module.signal.SIGKILL, None)
    ]
    with pytest.raises(ChildProcessError):
        os.waitpid(leader_pid, os.WNOHANG)
    assert not destination.exists()
    assert not (destination / "report.json").exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))
    assert len(adapter_roots) == 1
    assert not adapter_roots[0].exists()

    # The injected group-signal failure deliberately defeats descendant cleanup.
    # Let the fixture descendant finish, and prove this failure is not reported as
    # a successful group cleanup.
    time.sleep(2.25)
    assert marker.read_text(encoding="utf-8") == "alive"


@pytest.mark.parametrize(
    "selector_failure",
    ["constructor", "second_register"],
)
def test_selector_setup_failure_still_signals_and_reaps_started_process_group_once(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selector_failure: str,
) -> None:
    import bandtrace.adapters as adapters_module
    import bandtrace.audit as audit_module
    from bandtrace.cli import main as cli_main

    bundle = bundle_factory(
        adapter="subprocess-npz-v1",
        subprocess_mode="timeout_with_child",
    )
    marker = tmp_path / f"selector-{selector_failure}-child-survived.marker"
    monkeypatch.setenv("BANDTRACE_TEST_MARKER", str(marker))
    real_selector_factory = adapters_module.selectors.DefaultSelector

    if selector_failure == "constructor":
        def fail_selector_construction() -> object:
            time.sleep(0.75)
            raise OSError("injected selector constructor failure")

        monkeypatch.setattr(
            adapters_module.selectors,
            "DefaultSelector",
            fail_selector_construction,
        )
    else:
        class FailSecondRegister:
            def __init__(self) -> None:
                self._inner = real_selector_factory()
                self._register_calls = 0

            def register(
                self,
                fileobj: object,
                events: object,
                data: object = None,
            ) -> object:
                self._register_calls += 1
                if self._register_calls == 2:
                    time.sleep(0.75)
                    raise OSError("injected selector second-register failure")
                return self._inner.register(fileobj, events, data)

            def close(self) -> None:
                self._inner.close()

        monkeypatch.setattr(
            adapters_module.selectors,
            "DefaultSelector",
            FailSecondRegister,
        )

    real_build_adapter = audit_module.build_adapter
    adapter_roots: list[Path] = []

    def track_adapter_root(loaded: object) -> object:
        adapter = real_build_adapter(loaded)
        adapter_roots.append(Path(getattr(adapter, "_root")))
        return adapter

    real_kill_group = adapters_module.SubprocessNpzAdapter._kill_group
    real_killpg = adapters_module.os.killpg
    active_process: list[object] = []
    kill_helper_entries: list[tuple[int, object]] = []
    signal_observations: list[tuple[int, object]] = []
    reap_observations: list[tuple[int, object, int]] = []

    def inspect_killpg(process_group_id: int, signal_number: object) -> None:
        assert len(active_process) == 1
        process = active_process[0]
        signal_observations.append(
            (process_group_id, getattr(process, "returncode"))
        )
        real_killpg(process_group_id, signal_number)

    def inspect_kill_group(self: object, process: object) -> int:
        kill_helper_entries.append(
            (getattr(process, "pid"), getattr(process, "returncode"))
        )
        active_process.append(process)
        try:
            return_code = real_kill_group(self, process)
        finally:
            active_process.pop()
        reap_observations.append(
            (getattr(process, "pid"), getattr(process, "returncode"), return_code)
        )
        return return_code

    monkeypatch.setattr(audit_module, "build_adapter", track_adapter_root)
    monkeypatch.setattr(adapters_module.os, "killpg", inspect_killpg)
    monkeypatch.setattr(
        adapters_module.SubprocessNpzAdapter,
        "_kill_group",
        inspect_kill_group,
    )
    destination = tmp_path / f"selector-{selector_failure}-output"

    exit_code = cli_main(
        ["audit", str(bundle.root), "--output-dir", str(destination)]
    )
    stderr = capsys.readouterr().err

    assert exit_code == 3
    assert "cannot initialize trusted subprocess output selector" in stderr
    assert "Traceback" not in stderr
    assert len(kill_helper_entries) == 1
    assert len(signal_observations) == 1
    assert len(reap_observations) == 1
    assert kill_helper_entries[0][1] is None
    assert signal_observations[0][1] is None
    assert reap_observations[0][1] == reap_observations[0][2]
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))
    assert len(adapter_roots) == 1
    assert not adapter_roots[0].exists()

    time.sleep(2.25)
    assert not marker.exists(), "selector setup failure leaked a same-group child"


def test_nonzero_adapter_parent_kills_its_process_group_in_finally(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = bundle_factory(adapter="subprocess-npz-v1", subprocess_mode="nonzero_with_child")
    marker = tmp_path / "nonzero-child-survived.marker"
    monkeypatch.setenv("BANDTRACE_TEST_MARKER", str(marker))

    from bandtrace.audit import run_audit
    from bandtrace.errors import ExecutionError

    with pytest.raises(ExecutionError, match="23|status|nonzero|exited"):
        run_audit(bundle.root, tmp_path / "nonzero-with-child")

    time.sleep(2.25)
    assert not marker.exists(), "nonzero adapter child escaped cleanup in the finally path"


def test_conforming_subprocess_adapter_removes_its_mode_0700_temporary_tree_on_close(
    bundle_factory: BundleFactory,
) -> None:
    from bandtrace.adapters import build_adapter

    bundle = _load_bundle(bundle_factory(adapter="subprocess-npz-v1"))
    adapter = build_adapter(bundle)
    temporary_root = Path(getattr(adapter, "_root"))
    assert temporary_root.is_dir()
    assert temporary_root.stat().st_mode & 0o777 == 0o700
    request = _baseline_invocation(bundle)
    try:
        adapter.invoke(request)
        assert any(temporary_root.iterdir())
    finally:
        adapter.close()
    assert not temporary_root.exists()


def test_repeated_successful_invocations_delete_per_call_files_but_keep_arrays_usable(
    bundle_factory: BundleFactory,
) -> None:
    from bandtrace.adapters import build_adapter

    bundle = _load_bundle(bundle_factory(adapter="subprocess-npz-v1"))
    adapter = build_adapter(bundle)
    temporary_root = Path(getattr(adapter, "_root"))
    request = _baseline_invocation(bundle)
    expected_tap = np.asarray(bundle.probes.values[:, :4], dtype=np.float64)
    try:
        assert {path.name for path in temporary_root.iterdir()} == {"pinned"}
        for _ in range(3):
            response = adapter.invoke(request)
            assert {path.name for path in temporary_root.iterdir()} == {"pinned"}
            assert not list(temporary_root.glob("invoke-*"))
            np.testing.assert_array_equal(response.pre_core, expected_tap)
            np.testing.assert_allclose(
                response.output,
                expected_tap @ np.asarray([0.7, -0.5, 0.35, 0.9]) + 0.125,
                rtol=0.0,
                atol=1e-15,
            )
    finally:
        adapter.close()
    assert not temporary_root.exists()


def test_failed_invocation_deletes_per_call_files_before_adapter_close(
    bundle_factory: BundleFactory,
) -> None:
    from bandtrace.adapters import build_adapter
    from bandtrace.errors import ExecutionError

    bundle = _load_bundle(
        bundle_factory(
            adapter="subprocess-npz-v1",
            subprocess_mode="execution_failure",
        )
    )
    adapter = build_adapter(bundle)
    temporary_root = Path(getattr(adapter, "_root"))
    try:
        with pytest.raises(ExecutionError, match="status 17|nonzero|exited"):
            adapter.invoke(_baseline_invocation(bundle))
        assert {path.name for path in temporary_root.iterdir()} == {"pinned"}
        assert not list(temporary_root.glob("invoke-*"))
    finally:
        adapter.close()
    assert not temporary_root.exists()


def test_subprocess_input_writer_memory_error_is_stable_exit_3_without_traceback(
    bundle_factory: BundleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import bandtrace.adapters as adapters_module
    from bandtrace.cli import main as cli_main

    bundle = bundle_factory(adapter="subprocess-npz-v1")
    writer_calls = 0

    def exhaust_memory(_path: Path, _arrays: object) -> None:
        nonlocal writer_calls
        writer_calls += 1
        raise MemoryError("injected subprocess serializer allocation failure")

    monkeypatch.setattr(
        adapters_module,
        "write_deterministic_npz",
        exhaust_memory,
    )
    destination = tmp_path / "subprocess-writer-memory-error"
    exit_code = cli_main(
        [
            "audit",
            str(bundle.root),
            "--output-dir",
            str(destination),
        ]
    )
    stderr = capsys.readouterr().err

    assert exit_code == 3
    assert writer_calls == 1
    assert "subprocess input serialization exceeded available memory" in stderr
    assert "Traceback" not in stderr
    assert "injected subprocess serializer" not in stderr
    assert not destination.exists()
    assert not list(tmp_path.glob(".bandtrace-stage-*"))


@pytest.mark.parametrize(
    "charged_phase",
    ["parent_request_serialization", "response_decode_and_validation", "invocation_cleanup"],
)
def test_subprocess_outer_wall_ledger_includes_parent_decode_and_cleanup_phases(
    bundle_factory: BundleFactory,
    monkeypatch: pytest.MonkeyPatch,
    charged_phase: str,
) -> None:
    import bandtrace.adapters as adapters_module
    from bandtrace.adapters import build_adapter
    from bandtrace.canonical import write_deterministic_npz as write_output_npz
    from bandtrace.errors import ExecutionError

    class FakeClock:
        def __init__(self) -> None:
            self.now = 1000.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    bundle = _load_bundle(bundle_factory(adapter="subprocess-npz-v1"))
    adapter = build_adapter(bundle)
    temporary_root = Path(getattr(adapter, "_root"))
    request = _baseline_invocation(bundle)
    expected_pre_core = np.asarray(request.probes[:, :4], dtype=np.float64)
    expected_output = (
        expected_pre_core @ np.asarray([0.7, -0.5, 0.35, 0.9]) + 0.125
    )
    clock = FakeClock()
    run_calls = 0

    def fake_run(
        _command: list[str],
        output_path: Path,
        _invocation_started: float,
    ) -> tuple[bytes, bytes]:
        nonlocal run_calls
        run_calls += 1
        write_output_npz(
            output_path,
            {
                "output": np.asarray(expected_output, dtype=np.float64),
                "pre_core": np.asarray(expected_pre_core, dtype=np.float64),
            },
        )
        return b"", b""

    monkeypatch.setattr(adapters_module.time, "monotonic", clock)
    monkeypatch.setattr(adapters_module, "MAX_TOTAL_ADAPTER_WALL_SECONDS", 5.0)
    monkeypatch.setattr(adapter, "_run", fake_run)

    if charged_phase == "parent_request_serialization":
        real_writer = adapters_module.write_deterministic_npz

        def advancing_writer(path: Path, arrays: object) -> None:
            real_writer(path, arrays)
            clock.advance(6.0)

        monkeypatch.setattr(
            adapters_module,
            "write_deterministic_npz",
            advancing_writer,
        )
    elif charged_phase == "response_decode_and_validation":
        real_decode = adapters_module.load_npz_bytes

        def advancing_decode(*args: object, **kwargs: object) -> object:
            decoded = real_decode(*args, **kwargs)
            clock.advance(6.0)
            return decoded

        monkeypatch.setattr(adapters_module, "load_npz_bytes", advancing_decode)
    else:
        real_rmtree = adapters_module.shutil.rmtree

        def advancing_cleanup(path: object, *args: object, **kwargs: object) -> object:
            result = real_rmtree(path, *args, **kwargs)
            if Path(path).name.startswith("invoke-"):
                clock.advance(6.0)
            return result

        monkeypatch.setattr(adapters_module.shutil, "rmtree", advancing_cleanup)

    try:
        with pytest.raises(ExecutionError, match="total adapter wall-time budget exceeded"):
            adapter.invoke(request)
        assert adapter.wall_seconds == pytest.approx(6.0, rel=0.0, abs=0.0)
        assert adapter.invocations == 1
        assert {path.name for path in temporary_root.iterdir()} == {"pinned"}
        assert not list(temporary_root.glob("invoke-*"))
        assert run_calls == (
            0 if charged_phase == "parent_request_serialization" else 1
        )
    finally:
        adapter.close()
    assert not temporary_root.exists()


def test_subprocess_report_calls_group_reaping_cleanup_not_sandbox_containment(
    bundle_factory: BundleFactory,
    tmp_path: Path,
) -> None:
    completed = _cli(
        bundle_factory(adapter="subprocess-npz-v1"),
        tmp_path / "not-a-sandbox",
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "not-a-sandbox" / "report.json").read_text())
    assert report["facts"]["configured_time_budgets"][
        "subprocess_child_active_wall_poll"
    ] is True
    assert report["facts"]["configured_time_budgets"][
        "current_adapter_uses_active_child_wall_poll"
    ] is True
    assert report["facts"]["configured_time_budgets"][
        "adapter_total_hard_deadline"
    ] is False
    limitations = " ".join(report["limitations"]).lower()
    assert "not sandboxed" in limitations
    assert "independently network-disabled boundary" in limitations
    assert "hard end-to-end deadline" in limitations
    assert "external supervisor" in limitations


@pytest.mark.parametrize("bad_id", ["=2+3", "+SUM(A1:A2)", "@cmd", "<script>"])
def test_formula_or_html_active_band_ids_are_rejected_at_bundle_boundary(
    bundle_factory: BundleFactory,
    bad_id: str,
) -> None:
    def inject_id(sensor: dict[str, object]) -> None:
        sensor["target_bands"][0]["id"] = bad_id

    bundle = bundle_factory(sensor_mutator=inject_id)

    from bandtrace.errors import BundleError

    with pytest.raises(BundleError, match="id|identifier|safe"):
        _load_bundle(bundle)
