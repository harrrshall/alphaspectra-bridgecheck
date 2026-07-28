"""Public BandTrace audit API."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .adapters import build_adapter
from .authority import verify_packaged_normative_authority
from .bundle import Bundle, load_bundle
from .canonical import (
    canonical_json_bytes,
    derive_seed,
    deterministic_npz_bytes,
    installed_distribution_version,
    installed_source_tree_sha256,
    sha256_bytes,
)
from .canaries import run_canaries
from .constants import (
    DEPENDENCE_ABSOLUTE_FLOOR,
    DEPENDENCE_REPLAY_MULTIPLIER,
    MANDATORY_LIMITATIONS,
    MAX_ADAPTER_OUTPUT_BYTES,
    MAX_C2_SHIFT_SELECTION_FLOAT_COMPARISONS,
    MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES,
    MAX_SINGLE_OUTPUT_BYTES,
    MAX_SUBPROCESS_LOG_BYTES,
    MAX_TOTAL_OUTPUT_BYTES,
    MAX_TOTAL_ADAPTER_WALL_SECONDS,
    MINIMUM_PROBE_FRACTION,
    MUTATION_EXCITATION_FLOOR,
    NUMERIC_TOLERANCE,
    POLICY_ID,
    PRODUCT_VERSION,
    REPLAY_JITTER_MAX,
    REQUIRED_ROUTE_WEIGHT_MIN,
    ROW_SUM_TOLERANCE,
    SCHEMA_VERSION,
    SUBPROCESS_TIMEOUT_SECONDS,
    T0,
)
from .errors import ExecutionError
from .report import compact_route_audit, render_html, render_route_csv, route_audit_rows
from .spectral import evaluate_spectral_support


@dataclass(frozen=True)
class AuditResult:
    exit_code: int
    report: dict[str, Any]
    paths: dict[str, Path]


def _runtime_fingerprint() -> dict[str, Any]:
    return {
        "scope": "NON_EXHAUSTIVE_PROCESS_RUNTIME",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyyaml_version": getattr(yaml, "__version__", "unknown"),
        "operating_system": platform.system(),
        "machine": platform.machine(),
    }


def _input_facts(bundle: Bundle) -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            "path": record.relative_path,
            "sha256": record.sha256,
            "bytes": record.byte_count,
        }
        for key, record in sorted(bundle.files.items())
    ]


def _spectral_rows(channels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["model_channel_id"]): dict(row)
        for row in channels
    }


def _limitations(bundle: Bundle, canaries: dict[str, dict[str, Any]]) -> list[str]:
    limitations = list(MANDATORY_LIMITATIONS)
    limitations.extend(
        [
            "Declared radiometric quantity and support are supplier assertions; calibration traceability and acquisition geometry are not evaluated.",
            "A non-observation on submitted probes is not evidence of global non-use.",
            "Declared-tap agreement is limited to the frozen C1-C4 challenges; an undeclared transform that is numerically equivalent on those inputs is not observable.",
            "C3 rotations and keyed non-uniform magnitude challenges are challenge-local; no finite challenge set proves global metadata non-use.",
            "An executable effect already rounded away in the adapter's float64 output is unobservable to BandTrace.",
            "Raw-domain containment is exact for the parsed binary64 declarations, but it does not validate an executable outside the returned tap challenges.",
            "Categorical decisions use unrounded binary64 values; an eight-decimal metric and threshold can display as equal at a boundary, so status/pass fields are authoritative.",
            "Output publication and prepublication cleanup require a private, trusted parent directory; Linux renameat2(RENAME_NOREPLACE) and rmdir cannot bind the source name to the staged inode against a same-UID actor.",
            "The installed source-tree digest covers regular BandTrace .py files only; it is Python-source provenance, not execution attestation, and excludes bytecode, interpreter, native/dependency bytes, environment state, and in-memory mutation.",
            "The 600-second cumulative adapter wall threshold is measured and enforced at checkpoints; synchronous parent work and cleanup are not preemptible, so a hard end-to-end deadline requires an external supervisor.",
            "The packaged normative hash gate checks build-internal byte consistency; coordinated replacement of code, embedded hashes, and resources is not detected, so authenticity requires an independently trusted distribution digest or signature.",
        ]
    )
    c1 = canaries["C1_declared_tap_agreement"]
    if bundle.probes.values.ndim == 4:
        spatial_cells = int(np.prod(bundle.probes.values.shape[2:]))
        if spatial_cells == 1:
            limitations.append(
                "With one spatial cell, all singleton-preserving spatial reducers are observationally equivalent on every possible challenge."
            )
        elif spatial_cells == 2:
            limitations.append(
                "With exactly two spatial cells, arithmetic mean, median defined as the average of the middle pair, and midrange are mathematically identical and cannot be distinguished by BandTrace."
            )
    if not c1.get("raw_route_recovery_conditioned", False) or not c1.get(
        "raw_offset_recovery_conditioned", False
    ):
        limitations.append(
            "Raw route/offset inversion is ill-conditioned for this numeric envelope; it does not strengthen C1, and C6 cannot interpret recovered tapped-column equality when route recovery is affected."
        )
    for field, canary_id in (
        ("wavelength", "C3_wavelength_dependence"),
        ("FWHM", "C3_fwhm_dependence"),
    ):
        if str(canaries[canary_id]["status"]).startswith("INCONCLUSIVE"):
            limitations.append(
                f"The {field} metadata challenges were insufficiently exciting; metadata non-use was not established."
            )
    if bundle.adapter["type"] == "subprocess-npz-v1":
        limitations.extend(
            [
                "SUBPROCESS_DEPENDENCIES_UNATTESTED: manifest hashes do not cover every runtime dependency or resource.",
                "Ordinary subprocess argv tokens are passed verbatim and may name ambient paths; only exact BandTrace placeholders are interpreted, staged, and hash-pinned.",
                "A SUPPLIER_REPORTED_TAP can be a decoy; BandTrace cannot attest that it feeds the selected output.",
                "The trusted subprocess adapter is not sandboxed; run it inside an independently network-disabled boundary.",
                "Subprocess cleanup attempts one same-group SIGKILL before leader reap. If the OS rejects that group signal, BandTrace fails the run and still attempts to kill and reap the leader, but cannot guarantee cleanup of same-group descendants.",
            ]
        )
    return limitations


def _build_report(
    bundle: Bundle,
    *,
    installed_source_sha256: str,
    installed_version: str,
    normative_product_document_sha256: str,
    normative_machine_config_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    base_seed = derive_seed(
        model_hash=bundle.files["model"].sha256,
        sensor_hash=bundle.files["sensor"].sha256,
        probe_hash=bundle.files["probes"].sha256,
        route_hash=bundle.files["route"].sha256,
        policy_id=POLICY_ID,
    )
    adapter = build_adapter(bundle)
    try:
        canary_result = run_canaries(bundle, adapter, base_seed)
    finally:
        adapter.close()

    spectral = evaluate_spectral_support(
        bundle, route_eligible=canary_result.spectral_route_eligible
    )
    faults = [*canary_result.faults, *spectral.faults]
    exit_code = 4 if faults else 0
    facts: dict[str, Any] = {
        "overall_conformance_fault": bool(faults),
        "model_id": bundle.model.model_id,
        "model_version": bundle.model.model_version,
        "sensor_id": bundle.sensor.sensor_id,
        "sensor_model": bundle.sensor.sensor_model,
        "sensor_serial": bundle.sensor.sensor_serial,
        "sensor_calibration_state": bundle.sensor.calibration_state,
        "sensor_preprocessing_version": bundle.sensor.preprocessing_version,
        "radiometric_quantity_model": bundle.model.radiometric_quantity,
        "radiometric_quantity_sensor": bundle.sensor.radiometric_quantity,
        "adapter_type": bundle.adapter["type"],
        "adapter_configuration": {
            "type": bundle.adapter["type"],
            **(
                {"argv": list(bundle.adapter["argv"])}
                if bundle.adapter["type"] == "subprocess-npz-v1"
                else {}
            ),
        },
        "adapter_trust_state": adapter.trust_state,
        "route_assurance": adapter.assurance,
        "required_dependence_target_band_ids": list(
            bundle.model.required_dependence_target_band_ids
        ),
        "wavelength_conditioned": bundle.model.wavelength_conditioned,
        "fwhm_conditioned": bundle.model.fwhm_conditioned,
        "declared_support_assertion": bundle.model.support_assertion,
        "declared_support_range_nm": list(bundle.model.support_range_nm),
        "route_submitted_order_matches_contracts": bundle.route.order_matches_contracts,
        "probe_submitted_order_matches_sensor": bundle.probes.order_matches_sensor,
        "route_audit": compact_route_audit(bundle),
        "spectral_support_method": spectral.method,
        "spectral_support_by_model_channel": _spectral_rows(spectral.channels),
        "input_files": _input_facts(bundle),
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "mutation_seed_sha256_hex": base_seed.hex(),
        "installed_source_tree_sha256": installed_source_sha256,
        "installed_distribution_version": installed_version,
        "installed_source_digest_scope": "REGULAR_PY_FILES_ONLY",
        "execution_environment_attested": False,
        "native_dependency_bytes_hashed": False,
        "runtime_fingerprint_non_exhaustive": True,
        "normative_product_document_sha256": normative_product_document_sha256,
        "normative_machine_config_sha256": normative_machine_config_sha256,
        "packaged_hash_gate_is_external_authentication": False,
        "runtime_fingerprint": _runtime_fingerprint(),
        "raw_thresholds": {
            "row_sum_tolerance": ROW_SUM_TOLERANCE,
            "numeric_tolerance": NUMERIC_TOLERANCE,
            "replay_jitter_max": REPLAY_JITTER_MAX,
            "dependence_absolute_floor": DEPENDENCE_ABSOLUTE_FLOOR,
            "dependence_replay_multiplier": DEPENDENCE_REPLAY_MULTIPLIER,
            "minimum_probe_fraction": MINIMUM_PROBE_FRACTION,
            "mutation_excitation_floor": MUTATION_EXCITATION_FLOOR,
            "required_route_weight_minimum": REQUIRED_ROUTE_WEIGHT_MIN,
            "maximum_c2_shift_cell_comparisons": MAX_C2_SHIFT_SELECTION_FLOAT_COMPARISONS,
        },
        "configured_time_budgets": {
            "subprocess_timeout_seconds": SUBPROCESS_TIMEOUT_SECONDS,
            "adapter_total_measured_seconds_failure_threshold": (
                MAX_TOTAL_ADAPTER_WALL_SECONDS
            ),
            "adapter_total_hard_deadline": False,
            "synchronous_parent_and_cleanup_preemptible": False,
            "subprocess_child_active_wall_poll": True,
            "current_adapter_uses_active_child_wall_poll": (
                bundle.adapter["type"] == "subprocess-npz-v1"
            ),
            "external_hard_deadline_supervisor_required": True,
        },
        "configured_byte_budgets": {
            "adapter_output_bytes": MAX_ADAPTER_OUTPUT_BYTES,
            "subprocess_combined_log_bytes": MAX_SUBPROCESS_LOG_BYTES,
            "maximum_single_output_bytes": MAX_SINGLE_OUTPUT_BYTES,
            "maximum_aggregate_output_bytes": MAX_TOTAL_OUTPUT_BYTES,
            "maximum_cumulative_adapter_request_probe_value_bytes": (
                MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES
            ),
        },
        "planned_adapter_work": {
            "baseline_probe_value_bytes": (
                bundle.adapter_work_plan.baseline_probe_value_bytes
            ),
            "full_size_request_count": (
                bundle.adapter_work_plan.full_size_request_count
            ),
            "basis_request_count": bundle.adapter_work_plan.basis_request_count,
            "basis_probe_value_bytes": (
                bundle.adapter_work_plan.basis_probe_value_bytes
            ),
            "spatial_request_count": (
                bundle.adapter_work_plan.spatial_request_count
            ),
            "spatial_probe_value_bytes": (
                bundle.adapter_work_plan.spatial_probe_value_bytes
            ),
            "total_invocation_count": (
                bundle.adapter_work_plan.total_invocation_count
            ),
            "cumulative_request_probe_value_bytes": (
                bundle.adapter_work_plan.cumulative_request_probe_value_bytes
            ),
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
        },
        "output_publication_contract": {
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
        },
        **canary_result.facts,
    }
    if bundle.adapter["type"] == "subprocess-npz-v1":
        facts.update(
            {
                "subprocess_dependency_state": "SUBPROCESS_DEPENDENCIES_UNATTESTED",
                "subprocess_staging_contract": "ARTIFACT_AND_EXPLICIT_PINNED_ASSETS_ONLY",
                "subprocess_interpreted_placeholder_tokens": [
                    "{artifact}",
                    "{input_npz}",
                    "{output_npz}",
                    "{asset:<manifest_extra_key>}",
                ],
                "subprocess_ordinary_argv_tokens_passed_verbatim": True,
                "subprocess_ordinary_argv_ambient_paths": "UNPINNED_UNATTESTED",
                "subprocess_original_probes_staged": False,
                "subprocess_supported_platform": "POSIX",
                "subprocess_release_security_target": "Linux",
                "subprocess_group_signal_attempted_before_leader_reap": True,
                "subprocess_group_cleanup_guaranteed_on_os_signal_failure": False,
                "subprocess_group_signal_failure_can_leave_members_running": True,
                "subprocess_group_signal_failure_state": "execution_failure",
                (
                    "subprocess_direct_leader_kill_and_reap_fallback_configured_"
                    "on_group_signal_error"
                ): True,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": POLICY_ID,
        "product": "BandTrace",
        "product_version": PRODUCT_VERSION,
        "exit_code": exit_code,
        "states": {
            "executable": canary_result.executable_state,
            "spectral": spectral.state,
            "biological": T0,
        },
        "canaries": canary_result.canaries,
        "faults": faults,
        "facts": facts,
        "limitations": _limitations(bundle, canary_result.canaries),
    }
    return report, canary_result.arrays


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _verify_parent_identity(parent: Path, parent_fd: int) -> None:
    try:
        path_info = os.stat(parent, follow_symlinks=True)
    except OSError as error:
        raise OSError("output parent path is no longer available") from error
    descriptor_info = os.fstat(parent_fd)
    if not stat.S_ISDIR(descriptor_info.st_mode) or not _same_file_identity(
        path_info, descriptor_info
    ):
        raise OSError("output parent directory identity changed during audit")


def _entry_exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _create_staging_directory(parent_fd: int) -> tuple[str, int]:
    for _ in range(128):
        name = f".bandtrace-stage-{secrets.token_hex(12)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except BaseException:
            try:
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                pass
            raise
        return name, descriptor
    raise OSError(errno.EEXIST, "cannot allocate a fresh BandTrace staging directory")


def _write_fsynced_file(directory_fd: int, name: str, payload: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _cleanup_staging_directory(parent_fd: int, name: str, staging_fd: int) -> None:
    try:
        named_info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        # A missing staging name may mean publication already occurred before
        # an injected wrapper raised. Never mutate through the still-open fd.
        return
    if not _same_file_identity(named_info, os.fstat(staging_fd)):
        raise OSError("staging directory identity changed; refusing cleanup")
    try:
        entries = os.listdir(staging_fd)
    except FileNotFoundError:
        return
    for entry in entries:
        info = os.stat(entry, dir_fd=staging_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            raise OSError(
                "refusing recursive cleanup of unexpected staging subdirectory"
            )
        os.unlink(entry, dir_fd=staging_fd)
    # Shrink the remaining name-based rmdir race. POSIX has no
    # inode-conditional rmdir, so the private-parent requirement still applies
    # to a same-UID actor after this final check.
    _verify_staging_source_identity(parent_fd, name, staging_fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _verify_staging_source_identity(
    parent_fd: int, staging_name: str, staging_fd: int
) -> None:
    """Require the publish source name to still identify the opened stage."""

    try:
        named_info = os.stat(
            staging_name, dir_fd=parent_fd, follow_symlinks=False
        )
        descriptor_info = os.fstat(staging_fd)
    except OSError as error:
        raise OSError(
            "staging source is no longer available before atomic publication"
        ) from error
    if (
        not stat.S_ISDIR(named_info.st_mode)
        or not stat.S_ISDIR(descriptor_info.st_mode)
        or not _same_file_identity(named_info, descriptor_info)
    ):
        raise OSError(
            "staging source identity changed before atomic publication; "
            "refusing publication"
        )


def _require_atomic_publication_support() -> Any:
    """Resolve Linux renameat2 before bundle loading or adapter execution."""

    if platform.system() != "Linux":
        raise OSError(
            errno.ENOTSUP,
            "BandTrace atomic publication requires Linux renameat2(RENAME_NOREPLACE)",
        )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as error:
        raise OSError(
            errno.ENOTSUP,
            "Linux libc is unavailable; refusing output publication",
        ) from error
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOTSUP,
            "Linux libc does not expose renameat2; refusing unsafe publication fallback",
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    return renameat2


def _rename_directory_noreplace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically publish a directory without replacing any existing entry."""

    renameat2 = _require_atomic_publication_support()
    rename_noreplace = 1
    ctypes.set_errno(0)
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(
            error_number,
            "atomic no-replace output publication failed",
            destination_name,
        )


def _publish_outputs(
    *,
    parent: Path,
    parent_fd: int,
    destination: Path,
    outputs: dict[str, bytes],
    manifest_bytes: bytes,
) -> None:
    _verify_parent_identity(parent, parent_fd)
    if _entry_exists(parent_fd, destination.name):
        raise OSError("output destination appeared before staging")
    staging_name, staging_fd = _create_staging_directory(parent_fd)
    published = False
    try:
        staging_identity = os.fstat(staging_fd)
        for filename, payload in sorted(outputs.items()):
            _write_fsynced_file(staging_fd, filename, payload)
        _write_fsynced_file(staging_fd, "manifest.sha256", manifest_bytes)
        os.fsync(staging_fd)
        _verify_parent_identity(parent, parent_fd)
        if _entry_exists(parent_fd, destination.name):
            raise OSError("output destination appeared before atomic publication")
        # This closes ordinary substitutions before the syscall. Linux
        # renameat2 still cannot condition the source name on this inode, so a
        # same-UID actor with write access to the parent can race the remaining
        # syscall window; the post-rename identity check below fails closed.
        _verify_staging_source_identity(parent_fd, staging_name, staging_fd)
        _rename_directory_noreplace(
            parent_fd,
            staging_name,
            parent_fd,
            destination.name,
        )
        published = True
        published_info = os.stat(
            destination.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not _same_file_identity(staging_identity, published_info):
            raise OSError(
                "published output was substituted during atomic publication; "
                "the destination is untrusted and is not a BandTrace report"
            )
        os.fsync(parent_fd)
        _verify_parent_identity(parent, parent_fd)
        final_relative = os.stat(
            destination.name, dir_fd=parent_fd, follow_symlinks=False
        )
        try:
            final_path = os.stat(destination, follow_symlinks=False)
        except OSError as error:
            raise OSError("published output path is no longer reachable") from error
        if not _same_file_identity(staging_identity, final_relative) or not _same_file_identity(
            staging_identity, final_path
        ):
            raise OSError("published output path identity changed during audit")
    except BaseException:
        if not published:
            try:
                _cleanup_staging_directory(parent_fd, staging_name, staging_fd)
            except OSError as cleanup_error:
                raise OSError(
                    "cannot clean failed BandTrace output staging directory"
                ) from cleanup_error
        raise
    finally:
        os.close(staging_fd)


def _run_audit_impl(bundle_dir: Path, output_dir: Path) -> AuditResult:
    """Audit one frozen bundle and write the five deterministic artifacts."""

    destination = Path(output_dir)
    if destination.is_symlink() or destination.exists():
        raise OSError(
            "output directory must not already exist; choose a fresh destination "
            "so stale reports cannot survive a failed audit"
        )
    parent = destination.parent
    if not parent.is_dir():
        raise OSError("output directory parent must already exist as a directory")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        _verify_parent_identity(parent, parent_fd)
        if _entry_exists(parent_fd, destination.name):
            raise OSError(
                "output directory must not already exist; choose a fresh destination"
            )
        _require_atomic_publication_support()
        normative_authority = verify_packaged_normative_authority()
        installed_source_sha256 = installed_source_tree_sha256()
        installed_version = installed_distribution_version()
        bundle = load_bundle(Path(bundle_dir))
        report, canary_arrays = _build_report(
            bundle,
            installed_source_sha256=installed_source_sha256,
            installed_version=installed_version,
            normative_product_document_sha256=(
                normative_authority.product_document_sha256
            ),
            normative_machine_config_sha256=(
                normative_authority.machine_config_sha256
            ),
        )
        report_bytes = canonical_json_bytes(report)
        route_bytes = render_route_csv(route_audit_rows(bundle))
        canary_bytes = deterministic_npz_bytes(canary_arrays)
        html_bytes = render_html(
            report,
            {
                "report.json": sha256_bytes(report_bytes),
                "route.csv": sha256_bytes(route_bytes),
                "canary_outputs.npz": sha256_bytes(canary_bytes),
            },
        )
        outputs = {
            "canary_outputs.npz": canary_bytes,
            "report.html": html_bytes,
            "report.json": report_bytes,
            "route.csv": route_bytes,
        }
        manifest_bytes = "".join(
            f"{sha256_bytes(payload)}  {filename}\n"
            for filename, payload in sorted(outputs.items())
        ).encode("ascii")
        completed_outputs = {**outputs, "manifest.sha256": manifest_bytes}
        oversized = [
            filename
            for filename, payload in completed_outputs.items()
            if len(payload) > MAX_SINGLE_OUTPUT_BYTES
        ]
        if oversized:
            raise OSError(
                "BandTrace output exceeds the per-file byte limit: "
                + ", ".join(sorted(oversized))
            )
        aggregate_output_bytes = sum(
            len(payload) for payload in completed_outputs.values()
        )
        if aggregate_output_bytes > MAX_TOTAL_OUTPUT_BYTES:
            raise OSError(
                "BandTrace outputs exceed the aggregate byte limit "
                f"{MAX_TOTAL_OUTPUT_BYTES}"
            )
        _publish_outputs(
            parent=parent,
            parent_fd=parent_fd,
            destination=destination,
            outputs=outputs,
            manifest_bytes=manifest_bytes,
        )
    finally:
        os.close(parent_fd)
    paths = {
        filename: destination / filename
        for filename in (*sorted(outputs), "manifest.sha256")
    }
    return AuditResult(int(report["exit_code"]), report, paths)


def run_audit(bundle_dir: Path, output_dir: Path) -> AuditResult:
    """Translate catchable allocation failure into a stable execution error."""

    try:
        return _run_audit_impl(bundle_dir, output_dir)
    except MemoryError as error:
        raise ExecutionError(
            "audit execution or serialization exceeded available memory "
            "within frozen byte limits"
        ) from error
