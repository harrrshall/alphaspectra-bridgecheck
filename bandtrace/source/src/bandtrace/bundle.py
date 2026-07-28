"""Strict, hash-pinned BandTrace bundle loading."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import yaml
from yaml.constructor import ConstructorError
from yaml.tokens import AliasToken, AnchorToken

from .canonical import sha256_bytes
from .constants import (
    C1_CHUNK_MAX_FLOAT64_PROBE_BYTES,
    C1_CHUNK_MAX_ROWS,
    MAX_ADAPTER_OUTPUT_BYTES,
    MAX_ABS_DECLARED_NUMERIC_VALUE,
    MAX_ABS_NUMPY_ARTIFACT_NUMERIC_VALUE,
    MAX_BANDS,
    MAX_C2_SHIFT_SELECTION_FLOAT_COMPARISONS,
    MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_MANIFEST_DECLARED_FILES,
    MAX_MANIFEST_DECLARED_TOTAL_STAT_BYTES,
    MAX_PROBE_FILE_BYTES,
    MAX_PROBES,
    MAX_SPATIAL_CELLS,
    MAX_STRUCTURED_DOCUMENT_NESTING_DEPTH,
    MAX_FWHM_NM,
    MAX_NORMALIZATION_SCALE,
    MAX_REFLECTANCE_RAW_VALUE,
    MAX_SPECTRAL_WAVELENGTH_NM,
    MIN_PROBES,
    MIN_REFLECTANCE_RAW_VALUE,
    MIN_NORMALIZATION_SCALE,
    MIN_FWHM_NM,
    MIN_POSITIVE_ROUTE_WEIGHT,
    MIN_SPECTRAL_WAVELENGTH_NM,
    POLICY_ID,
    REQUIRED_FILES,
    REPLAY_COUNT,
    REQUIRED_ROUTE_WEIGHT_MIN,
    SCHEMA_VERSION,
    MIN_VALID_RANGE_WIDTH,
    SUPPORTED_ADAPTERS,
    SUPPORTED_RADIOMETRIC_QUANTITIES,
)
from .errors import BundleError
from .npzio import load_npz_bytes

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_ASSET_PLACEHOLDER = re.compile(
    r"\{asset:([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})\}\Z"
)
_ARTIFACT_KEYS = {
    "route_matrix",
    "target_band_ids",
    "normalization_offset",
    "normalization_scale",
    "output_weights",
    "output_bias",
    "wavelength_weights",
    "fwhm_weights",
    "spatial_operation",
}
MAX_EXPANDED_FLOAT64_PROBE_BYTES = 268_435_456
MAX_CANARY_INVOCATION_PROBE_BYTES = 268_435_456
MAX_C2_SHIFT_CELL_COMPARISONS = MAX_C2_SHIFT_SELECTION_FLOAT_COMPARISONS


class _StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ConstructorError("while constructing a mapping", node.start_mark, "unhashable key", key_node.start_mark) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _validate_structured_document(value: Any, where: str) -> None:
    """Enforce the bounded JSON data model without recursive traversal."""

    stack: list[tuple[Any, str, int]] = [(value, where, 1)]
    while stack:
        current, location, depth = stack.pop()
        if isinstance(current, dict):
            if depth > MAX_STRUCTURED_DOCUMENT_NESTING_DEPTH:
                raise BundleError(
                    f"{where}: structured document nesting exceeds "
                    f"{MAX_STRUCTURED_DOCUMENT_NESTING_DEPTH} levels"
                )
            for key, child in current.items():
                if not isinstance(key, str):
                    raise BundleError(
                        f"{location}: mappings must use JSON-compatible string keys"
                    )
                stack.append((child, f"{location}.{key}", depth + 1))
        elif isinstance(current, list):
            if depth > MAX_STRUCTURED_DOCUMENT_NESTING_DEPTH:
                raise BundleError(
                    f"{where}: structured document nesting exceeds "
                    f"{MAX_STRUCTURED_DOCUMENT_NESTING_DEPTH} levels"
                )
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], f"{location}[{index}]", depth + 1))
        elif current is None or isinstance(current, (str, bool, int)):
            continue
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise BundleError(f"{location}: non-finite numbers are forbidden")
        else:
            raise BundleError(
                f"{location}: value type {type(current).__name__} is outside the JSON data model"
            )


def _strict_yaml(data: bytes, source: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        for token in yaml.scan(text):
            if isinstance(token, (AliasToken, AnchorToken)):
                raise BundleError(f"{source}: YAML anchors and aliases are forbidden")
        payload = yaml.load(text, Loader=_StrictSafeLoader)
    except BundleError:
        raise
    except (UnicodeDecodeError, yaml.YAMLError, ValueError, OverflowError, RecursionError) as error:
        raise BundleError(f"{source}: invalid strict YAML: {error}") from error
    if not isinstance(payload, dict):
        raise BundleError(f"{source}: top-level value must be a mapping")
    try:
        _validate_structured_document(payload, source)
    except RecursionError as error:
        raise BundleError(f"{source}: structured document nesting is invalid") from error
    return payload


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_json(data: bytes, source: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise BundleError(f"{source}: non-finite JSON token {value!r} is forbidden")

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=reject_constant,
        )
    except BundleError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as error:
        raise BundleError(f"{source}: invalid strict JSON: {error}") from error
    if not isinstance(payload, dict):
        raise BundleError(f"{source}: top-level JSON value must be an object")
    try:
        _validate_structured_document(payload, source)
    except RecursionError as error:
        raise BundleError(f"{source}: structured document nesting is invalid") from error
    return payload


def _require_schema_version(payload: Mapping[str, Any], where: str) -> None:
    value = payload.get("schema_version")
    if not isinstance(value, str) or value != SCHEMA_VERSION:
        raise BundleError(f"{where}.schema_version must be the exact string {SCHEMA_VERSION!r}")


def _expect_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise BundleError(f"{where} must be an object")
    return value


def _reject_unknown(item: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(item) - allowed
    if unknown:
        raise BundleError(f"{where} has unknown or undeclared fields: {sorted(unknown)}")


def _expect_string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise BundleError(f"{where} must be a non-empty string")
    return value


def _identifier(value: Any, where: str) -> str:
    identifier = _expect_string(value, where)
    if not _SAFE_ID.fullmatch(identifier):
        raise BundleError(f"{where} is not a safe identifier")
    return identifier


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise BundleError(f"{where} must be a bounded finite number") from error
    if not math.isfinite(result):
        raise BundleError(f"{where} must be a finite number")
    if abs(result) > MAX_ABS_DECLARED_NUMERIC_VALUE:
        raise BundleError(
            f"{where} exceeds the absolute numeric bound {MAX_ABS_DECLARED_NUMERIC_VALUE:g}"
        )
    return result


def _radiometric_quantity(value: Any, where: str) -> str:
    quantity = _expect_string(value, where)
    if quantity not in SUPPORTED_RADIOMETRIC_QUANTITIES:
        raise BundleError(
            "radiometric_quantity_mismatch: "
            f"{where} must be one of {SUPPORTED_RADIOMETRIC_QUANTITIES}"
        )
    return quantity


def _number_vector(value: Any, where: str, length: int | None = None) -> np.ndarray:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        suffix = f" with length {length}" if length is not None else ""
        raise BundleError(f"{where} must be an array{suffix}")
    result = np.asarray([_number(item, f"{where}[{index}]") for index, item in enumerate(value)], dtype=np.float64)
    return result


def _unit_scale(unit: Any, where: str) -> float:
    normalized = _expect_string(unit, where).strip().lower().replace("μ", "µ")
    if normalized in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        return 1.0
    if normalized in {"um", "µm", "micrometer", "micrometers", "micrometre", "micrometres"}:
        return 1000.0
    raise BundleError(f"{where}: wavelength unit must be explicitly nm or um")


def _wavelength_nm(value: Any, unit: Any, where: str) -> float:
    converted = _number(value, where) * _unit_scale(unit, f"{where}_unit")
    # The v0.1 HSI boundary deliberately rejects values that are physically
    # characteristic of a mislabeled nm/um declaration.
    if (
        converted < MIN_SPECTRAL_WAVELENGTH_NM
        or converted > MAX_SPECTRAL_WAVELENGTH_NM
    ):
        raise BundleError(
            "wavelength_nm_micron_mismatch: "
            f"{where}: converted wavelength {converted:g} nm is outside the v0.1 HSI domain"
        )
    return converted


@dataclass(frozen=True)
class FileRecord:
    key: str
    relative_path: str
    path: Path
    sha256: str
    data: bytes | None
    byte_count: int


@dataclass(frozen=True)
class Band:
    id: str
    center_nm: float
    fwhm_nm: float | None
    srf_wavelength_nm: np.ndarray | None
    srf_response: np.ndarray | None
    neutral_value: float | None
    valid_range: tuple[float, float] | None


@dataclass(frozen=True)
class ModelContract:
    model_id: str
    model_version: str
    artifact_sha256: str
    channels: tuple[Band, ...]
    radiometric_quantity: str
    valid_range: tuple[float, float]
    normalization_offset: np.ndarray
    normalization_scale: np.ndarray
    support_assertion: bool
    support_range_nm: tuple[float, float]
    output_name: str
    wavelength_conditioned: bool
    fwhm_conditioned: bool
    required_dependence_target_band_ids: tuple[str, ...]


@dataclass(frozen=True)
class SensorContract:
    sensor_id: str
    sensor_model: str
    sensor_serial: str
    bands: tuple[Band, ...]
    radiometric_quantity: str
    valid_range: tuple[float, float]
    calibration_state: str
    preprocessing_version: str


@dataclass(frozen=True)
class RouteContract:
    model_channel_ids: tuple[str, ...]
    target_band_ids: tuple[str, ...]
    matrix: np.ndarray
    canonical_matrix: np.ndarray
    operation: str
    spatial_operation: str
    order_matches_contracts: bool


@dataclass(frozen=True)
class ProbeSet:
    values: np.ndarray
    target_band_ids: tuple[str, ...]
    order_matches_sensor: bool


@dataclass(frozen=True)
class AdapterWorkPlan:
    baseline_probe_value_bytes: int
    full_size_request_count: int
    basis_request_count: int
    basis_probe_value_bytes: int
    spatial_request_count: int
    spatial_probe_value_bytes: int
    total_invocation_count: int
    cumulative_request_probe_value_bytes: int


@dataclass(frozen=True)
class Bundle:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    files: dict[str, FileRecord]
    adapter: dict[str, Any]
    model: ModelContract
    sensor: SensorContract
    route: RouteContract
    probes: ProbeSet
    adapter_work_plan: AdapterWorkPlan
    numpy_artifact: dict[str, np.ndarray] | None


@dataclass(frozen=True)
class _PinnedRegularFile:
    relative_path: str
    path: Path
    descriptor: int
    initial_stat: os.stat_result
    directory_identities: tuple[tuple[str, int, int], ...]
    final_name: str


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _verify_bundle_root_identity(root: Path, root_fd: int) -> None:
    try:
        path_info = os.stat(root, follow_symlinks=False)
        descriptor_info = os.fstat(root_fd)
    except OSError as error:
        raise BundleError("bundle root path is no longer available") from error
    if (
        not stat.S_ISDIR(path_info.st_mode)
        or not stat.S_ISDIR(descriptor_info.st_mode)
        or not _same_file_identity(path_info, descriptor_info)
    ):
        raise BundleError("bundle root directory identity changed during loading")


def _open_bundle_root(bundle_dir: Path) -> tuple[Path, int]:
    root = Path(os.path.abspath(os.fspath(bundle_dir)))
    try:
        descriptor = os.open(
            root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except (OSError, ValueError) as error:
        raise BundleError(
            "bundle root must be a real directory, not a symlink"
        ) from error
    try:
        _verify_bundle_root_identity(root, descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return root, descriptor


def _relative_file_parts(relative: str, *, source: str) -> tuple[str, ...]:
    if "\\" in relative:
        raise BundleError(f"{source}: paths must use bundle-relative POSIX syntax")
    if any(ord(character) < 32 or ord(character) == 127 for character in relative):
        raise BundleError(f"{source}: paths cannot contain NUL or control characters")
    raw_parts = relative.split("/")
    if not relative or any(part in {"", ".", ".."} for part in raw_parts):
        raise BundleError(f"{source}: absolute, empty, and traversal paths are forbidden")
    try:
        pure = PurePosixPath(relative)
    except (TypeError, ValueError) as error:
        raise BundleError(f"{source}: invalid bundle-relative path") from error
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise BundleError(f"{source}: absolute, empty, and traversal paths are forbidden")
    return tuple(pure.parts)


def _open_pinned_regular_file(
    root: Path,
    root_fd: int,
    relative: str,
    *,
    source: str,
) -> _PinnedRegularFile:
    """Open a bundle file through no-follow dirfds and retain its exact fd."""

    parts = _relative_file_parts(relative, source=source)
    current_fd = root_fd
    owned_directory_fd: int | None = None
    final_fd: int | None = None
    directory_identities: list[tuple[str, int, int]] = []
    try:
        for part in parts[:-1]:
            next_fd: int | None = None
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=current_fd,
                )
                info = os.fstat(next_fd)
                if not stat.S_ISDIR(info.st_mode):
                    raise BundleError(
                        f"{source}: intermediate path component is not a real directory"
                    )
                directory_identities.append(
                    (part, int(info.st_dev), int(info.st_ino))
                )
                if owned_directory_fd is not None:
                    os.close(owned_directory_fd)
                owned_directory_fd = next_fd
                current_fd = next_fd
                next_fd = None
            finally:
                if next_fd is not None:
                    os.close(next_fd)
        final_fd = os.open(
            parts[-1],
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current_fd,
        )
        info = os.fstat(final_fd)
        if not stat.S_ISREG(info.st_mode):
            raise BundleError(f"{source}: referenced path is not a regular file")
        pinned = _PinnedRegularFile(
            relative_path=relative,
            path=root.joinpath(*parts),
            descriptor=final_fd,
            initial_stat=info,
            directory_identities=tuple(directory_identities),
            final_name=parts[-1],
        )
        final_fd = None
        return pinned
    except BundleError:
        raise
    except (OSError, ValueError) as error:
        raise BundleError(
            f"{source}: cannot open pinned bundle-relative regular file: {error}"
        ) from error
    finally:
        if final_fd is not None:
            os.close(final_fd)
        if owned_directory_fd is not None:
            os.close(owned_directory_fd)


def _verify_pinned_regular_file_path(
    root_fd: int, pinned: _PinnedRegularFile, *, source: str
) -> None:
    """Re-walk names safely and require the initially opened identity chain."""

    current_fd = root_fd
    owned_directory_fd: int | None = None
    try:
        for part, expected_device, expected_inode in pinned.directory_identities:
            next_fd: int | None = None
            try:
                next_fd = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=current_fd,
                )
                info = os.fstat(next_fd)
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or (info.st_dev, info.st_ino)
                    != (expected_device, expected_inode)
                ):
                    raise BundleError(
                        f"{source}: intermediate directory identity changed during loading"
                    )
                if owned_directory_fd is not None:
                    os.close(owned_directory_fd)
                owned_directory_fd = next_fd
                current_fd = next_fd
                next_fd = None
            finally:
                if next_fd is not None:
                    os.close(next_fd)
        current = os.stat(
            pinned.final_name,
            dir_fd=current_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(current.st_mode)
            or not _same_file_identity(current, pinned.initial_stat)
        ):
            raise BundleError(
                f"{source}: pinned regular-file path identity changed during loading"
            )
    except BundleError:
        raise
    except (OSError, ValueError) as error:
        raise BundleError(
            f"{source}: cannot revalidate pinned bundle-relative path: {error}"
        ) from error
    finally:
        if owned_directory_fd is not None:
            os.close(owned_directory_fd)


def _regular_file_snapshot(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _verify_pinned_regular_file_unchanged(
    root: Path,
    root_fd: int,
    pinned: _PinnedRegularFile,
    *,
    source: str,
) -> None:
    _verify_bundle_root_identity(root, root_fd)
    _verify_pinned_regular_file_path(root_fd, pinned, source=source)
    try:
        current = os.fstat(pinned.descriptor)
    except OSError as error:
        raise BundleError(
            f"{source}: cannot inspect pinned regular-file descriptor"
        ) from error
    if (
        not stat.S_ISREG(current.st_mode)
        or _regular_file_snapshot(current)
        != _regular_file_snapshot(pinned.initial_stat)
    ):
        raise BundleError(f"{source}: pinned file changed during bundle loading")


def _read_pinned_regular_file(
    root: Path,
    root_fd: int,
    pinned: _PinnedRegularFile,
    *,
    max_bytes: int,
    source: str,
) -> bytes:
    _verify_pinned_regular_file_unchanged(
        root, root_fd, pinned, source=source
    )
    initial_snapshot = _regular_file_snapshot(pinned.initial_stat)
    try:
        info = os.fstat(pinned.descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise BundleError(f"{source}: referenced path is not a regular file")
        if _regular_file_snapshot(info) != initial_snapshot:
            raise BundleError(f"{source}: file changed after descriptor preflight")
        if info.st_size > max_bytes:
            raise BundleError(f"{source}: file size exceeds {max_bytes} bytes")
        os.lseek(pinned.descriptor, 0, os.SEEK_SET)
        with os.fdopen(pinned.descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise BundleError(f"{source}: file size exceeds {max_bytes} bytes")
        after = os.fstat(pinned.descriptor)
        if _regular_file_snapshot(after) != initial_snapshot:
            raise BundleError(f"{source}: file changed while it was being read")
    except BundleError:
        raise
    except (OSError, ValueError) as error:
        raise BundleError(f"{source}: cannot read pinned regular file: {error}") from error
    _verify_pinned_regular_file_path(root_fd, pinned, source=source)
    _verify_bundle_root_identity(root, root_fd)
    return data


def _parse_range(value: Any, where: str) -> tuple[float, float]:
    vector = _number_vector(value, where, 2)
    if (
        vector[0] < MIN_REFLECTANCE_RAW_VALUE
        or vector[1] > MAX_REFLECTANCE_RAW_VALUE
    ):
        raise BundleError(
            f"{where} must lie inside the v0.1 reflectance raw domain "
            f"[{MIN_REFLECTANCE_RAW_VALUE:g}, {MAX_REFLECTANCE_RAW_VALUE:g}]"
        )
    if vector[1] - vector[0] < MIN_VALID_RANGE_WIDTH:
        raise BundleError(
            f"{where} width must be at least {MIN_VALID_RANGE_WIDTH:g}"
        )
    return float(vector[0]), float(vector[1])


def _parse_band(value: Any, where: str, *, sensor: bool, global_range: tuple[float, float] | None) -> Band:
    item = _expect_mapping(value, where)
    allowed = {"id", "center_wavelength", "wavelength_unit", "fwhm", "fwhm_unit", "srf"}
    if sensor:
        allowed |= {"neutral_value", "valid_range"}
    _reject_unknown(item, allowed, where)
    identifier = _identifier(item.get("id"), f"{where}.id")
    center = _wavelength_nm(item.get("center_wavelength"), item.get("wavelength_unit"), f"{where}.center_wavelength")
    if item.get("fwhm") is None:
        raise BundleError(
            "missing_mandatory_fwhm: "
            f"{where}.fwhm is mandatory in BandTrace v0.1"
        )
    fwhm = _number(item["fwhm"], f"{where}.fwhm") * _unit_scale(
        item.get("fwhm_unit"), f"{where}.fwhm_unit"
    )
    if fwhm < MIN_FWHM_NM or fwhm > MAX_FWHM_NM:
        raise BundleError(f"{where}.fwhm is outside the v0.1 domain")
    srf_wavelength = None
    srf_response = None
    if item.get("srf") is not None:
        srf = _expect_mapping(item["srf"], f"{where}.srf")
        _reject_unknown(srf, {"wavelengths", "responses", "wavelength_unit"}, f"{where}.srf")
        wavelengths = _number_vector(srf.get("wavelengths"), f"{where}.srf.wavelengths")
        responses = _number_vector(srf.get("responses"), f"{where}.srf.responses")
        if wavelengths.size != responses.size:
            raise BundleError(f"{where}.srf wavelength/response lengths differ")
        scale = _unit_scale(srf.get("wavelength_unit"), f"{where}.srf.wavelength_unit")
        srf_wavelength = wavelengths * scale
        if np.any(srf_wavelength < MIN_SPECTRAL_WAVELENGTH_NM) or np.any(
            srf_wavelength > MAX_SPECTRAL_WAVELENGTH_NM
        ):
            raise BundleError(
                "wavelength_nm_micron_mismatch: "
                f"{where}.srf.wavelengths contain a converted coordinate outside "
                f"[{MIN_SPECTRAL_WAVELENGTH_NM:g}, "
                f"{MAX_SPECTRAL_WAVELENGTH_NM:g}] nm"
            )
        srf_response = responses
    valid_range = global_range
    neutral = None
    if sensor:
        if "valid_range" in item:
            valid_range = _parse_range(item["valid_range"], f"{where}.valid_range")
        neutral = _number(item.get("neutral_value"), f"{where}.neutral_value")
        assert valid_range is not None
        if not valid_range[0] <= neutral <= valid_range[1]:
            raise BundleError(f"{where}.neutral_value is outside its valid raw range")
    return Band(identifier, center, fwhm, srf_wavelength, srf_response, neutral, valid_range)


def _parse_model(payload: dict[str, Any], artifact_hash: str) -> ModelContract:
    where = "model.json"
    _reject_unknown(
        payload,
        {
            "schema_version",
            "model_id",
            "model_version",
            "artifact_sha256",
            "model_channels",
            "radiometric_quantity",
            "valid_range",
            "normalization",
            "declared_validated_support",
            "pre_decision_output",
            "wavelength_conditioned",
            "fwhm_conditioned",
            "required_dependence_target_band_ids",
        },
        where,
    )
    _require_schema_version(payload, where)
    model_id = _identifier(payload.get("model_id"), f"{where}.model_id")
    version = _expect_string(payload.get("model_version"), f"{where}.model_version")
    declared_artifact = _expect_string(payload.get("artifact_sha256"), f"{where}.artifact_sha256")
    if declared_artifact != artifact_hash:
        raise BundleError(f"{where}: artifact_sha256 does not match pinned artifact")
    valid_range = _parse_range(payload.get("valid_range"), f"{where}.valid_range")
    raw_channels = payload.get("model_channels")
    if not isinstance(raw_channels, list) or not 1 <= len(raw_channels) <= MAX_BANDS:
        raise BundleError(f"{where}.model_channels must contain 1..{MAX_BANDS} channels")
    channels = tuple(
        _parse_band(item, f"{where}.model_channels[{index}]", sensor=False, global_range=valid_range)
        for index, item in enumerate(raw_channels)
    )
    ids = [band.id for band in channels]
    if len(ids) != len(set(ids)):
        raise BundleError("duplicate_band_ids: model channel IDs must be unique")
    normalization = _expect_mapping(payload.get("normalization"), f"{where}.normalization")
    _reject_unknown(normalization, {"type", "offset", "scale"}, f"{where}.normalization")
    if normalization.get("type") != "affine":
        raise BundleError(f"{where}.normalization.type must be affine")
    offset = _number_vector(normalization.get("offset"), f"{where}.normalization.offset", len(channels))
    scale = _number_vector(normalization.get("scale"), f"{where}.normalization.scale", len(channels))
    if np.any(scale < MIN_NORMALIZATION_SCALE) or np.any(
        scale > MAX_NORMALIZATION_SCALE
    ):
        raise BundleError(
            f"{where}.normalization.scale must lie in "
            f"[{MIN_NORMALIZATION_SCALE:g}, {MAX_NORMALIZATION_SCALE:g}]"
        )
    if np.any(offset < valid_range[0]) or np.any(offset > valid_range[1]):
        raise BundleError(
            f"{where}.normalization.offset must lie inside the model raw valid range"
        )
    support = _expect_mapping(payload.get("declared_validated_support"), f"{where}.declared_validated_support")
    _reject_unknown(
        support,
        {"supplier_assertion", "wavelength_range", "wavelength_unit"},
        f"{where}.declared_validated_support",
    )
    if "supplier_assertion" not in support:
        assertion = False
    else:
        assertion_value = support["supplier_assertion"]
        if not isinstance(assertion_value, bool):
            raise BundleError(
                f"{where}.declared_validated_support.supplier_assertion "
                "must be a literal boolean"
            )
        assertion = assertion_value
    raw_support_range = _number_vector(support.get("wavelength_range"), f"{where}.declared_validated_support.wavelength_range", 2)
    support_scale = _unit_scale(support.get("wavelength_unit"), f"{where}.declared_validated_support.wavelength_unit")
    support_range = (float(raw_support_range[0] * support_scale), float(raw_support_range[1] * support_scale))
    if (
        support_range[0] < MIN_SPECTRAL_WAVELENGTH_NM
        or support_range[1] > MAX_SPECTRAL_WAVELENGTH_NM
        or support_range[0] >= support_range[1]
    ):
        raise BundleError(f"{where}: declared support range is invalid or unit-mismatched")
    output = _expect_mapping(payload.get("pre_decision_output"), f"{where}.pre_decision_output")
    _reject_unknown(output, {"name"}, f"{where}.pre_decision_output")
    output_name = _identifier(output.get("name"), f"{where}.pre_decision_output.name")
    if "wavelength_conditioned" not in payload:
        raise BundleError(f"{where}.wavelength_conditioned must be explicitly declared")
    wavelength_conditioned = payload["wavelength_conditioned"]
    if not isinstance(wavelength_conditioned, bool):
        raise BundleError(f"{where}.wavelength_conditioned must be boolean")
    if "fwhm_conditioned" not in payload:
        raise BundleError(f"{where}.fwhm_conditioned must be explicitly declared")
    fwhm_conditioned = payload["fwhm_conditioned"]
    if not isinstance(fwhm_conditioned, bool):
        raise BundleError(f"{where}.fwhm_conditioned must be boolean")
    required_raw = payload.get("required_dependence_target_band_ids")
    if not isinstance(required_raw, list) or not required_raw:
        raise BundleError(
            f"{where}.required_dependence_target_band_ids must be a non-empty explicit array"
        )
    required = tuple(
        _identifier(value, f"{where}.required_dependence_target_band_ids")
        for value in required_raw
    )
    if len(required) != len(set(required)):
        raise BundleError(f"{where}.required_dependence_target_band_ids contains duplicates")
    return ModelContract(
        model_id=model_id,
        model_version=version,
        artifact_sha256=artifact_hash,
        channels=channels,
        radiometric_quantity=_radiometric_quantity(
            payload.get("radiometric_quantity"), f"{where}.radiometric_quantity"
        ),
        valid_range=valid_range,
        normalization_offset=offset,
        normalization_scale=scale,
        support_assertion=assertion,
        support_range_nm=support_range,
        output_name=output_name,
        wavelength_conditioned=wavelength_conditioned,
        fwhm_conditioned=fwhm_conditioned,
        required_dependence_target_band_ids=required,
    )


def _parse_sensor(payload: dict[str, Any]) -> SensorContract:
    where = "sensor.json"
    _reject_unknown(
        payload,
        {
            "schema_version",
            "sensor_id",
            "sensor_model",
            "sensor_serial",
            "target_bands",
            "radiometric_quantity",
            "valid_range",
            "calibration_state",
            "preprocessing_version",
        },
        where,
    )
    _require_schema_version(payload, where)
    valid_range = _parse_range(payload.get("valid_range"), f"{where}.valid_range")
    raw_bands = payload.get("target_bands")
    if not isinstance(raw_bands, list) or not 1 <= len(raw_bands) <= MAX_BANDS:
        raise BundleError(f"{where}.target_bands must contain 1..{MAX_BANDS} bands")
    bands = tuple(
        _parse_band(item, f"{where}.target_bands[{index}]", sensor=True, global_range=valid_range)
        for index, item in enumerate(raw_bands)
    )
    ids = [band.id for band in bands]
    if len(ids) != len(set(ids)):
        raise BundleError("duplicate_band_ids: target band IDs must be unique")
    return SensorContract(
        sensor_id=_identifier(payload.get("sensor_id"), f"{where}.sensor_id"),
        sensor_model=_expect_string(payload.get("sensor_model"), f"{where}.sensor_model"),
        sensor_serial=_expect_string(payload.get("sensor_serial"), f"{where}.sensor_serial"),
        bands=bands,
        radiometric_quantity=_radiometric_quantity(
            payload.get("radiometric_quantity"), f"{where}.radiometric_quantity"
        ),
        valid_range=valid_range,
        calibration_state=_expect_string(payload.get("calibration_state"), f"{where}.calibration_state"),
        preprocessing_version=_expect_string(payload.get("preprocessing_version"), f"{where}.preprocessing_version"),
    )


def _parse_route(payload: dict[str, Any], model: ModelContract, sensor: SensorContract) -> RouteContract:
    where = "route.json"
    _reject_unknown(
        payload,
        {
            "schema_version",
            "model_channel_ids",
            "target_band_ids",
            "matrix",
            "operation",
            "spatial_operation",
        },
        where,
    )
    _require_schema_version(payload, where)
    model_ids_raw = payload.get("model_channel_ids")
    target_ids_raw = payload.get("target_band_ids")
    if not isinstance(model_ids_raw, list) or not isinstance(target_ids_raw, list):
        raise BundleError(f"{where}: ID orders must be arrays")
    model_ids = tuple(_identifier(value, f"{where}.model_channel_ids") for value in model_ids_raw)
    target_ids = tuple(_identifier(value, f"{where}.target_band_ids") for value in target_ids_raw)
    if len(model_ids) != len(set(model_ids)) or len(target_ids) != len(set(target_ids)):
        raise BundleError("duplicate_band_ids: route IDs must be unique")
    if set(model_ids) != {band.id for band in model.channels} or set(target_ids) != {band.id for band in sensor.bands}:
        raise BundleError(f"{where}: route IDs must cover exactly the model and target contracts")
    raw_matrix = payload.get("matrix")
    if not isinstance(raw_matrix, list) or len(raw_matrix) != len(model_ids):
        raise BundleError(f"{where}.matrix has the wrong row count")
    rows = [_number_vector(row, f"{where}.matrix[{index}]", len(target_ids)) for index, row in enumerate(raw_matrix)]
    matrix = np.asarray(rows, dtype=np.float64)
    if np.any((matrix != 0.0) & (matrix < MIN_POSITIVE_ROUTE_WEIGHT)):
        raise BundleError(
            "route.json.matrix weights must be exactly zero or at least "
            f"{MIN_POSITIVE_ROUTE_WEIGHT:g}"
        )
    row_index = {identifier: index for index, identifier in enumerate(model_ids)}
    col_index = {identifier: index for index, identifier in enumerate(target_ids)}
    canonical = np.asarray(
        [
            [matrix[row_index[channel.id], col_index[band.id]] for band in sensor.bands]
            for channel in model.channels
        ],
        dtype=np.float64,
    )
    required = model.required_dependence_target_band_ids
    if not set(required).issubset({band.id for band in sensor.bands}):
        raise BundleError(
            "model.json.required_dependence_target_band_ids contains unknown target IDs"
        )
    sensor_index = {band.id: index for index, band in enumerate(sensor.bands)}
    for identifier in required:
        weight = float(np.sum(np.abs(canonical[:, sensor_index[identifier]])))
        if weight < REQUIRED_ROUTE_WEIGHT_MIN:
            raise BundleError(
                "model.json: required dependence target "
                f"{identifier!r} has aggregate route weight below {REQUIRED_ROUTE_WEIGHT_MIN}"
            )
    operation = _expect_string(payload.get("operation"), f"{where}.operation")
    if operation not in {"selection_or_permutation", "nonnegative_row_normalized_linear_resampling"}:
        raise BundleError(f"{where}.operation is outside v0.1")
    spatial = _expect_string(payload.get("spatial_operation"), f"{where}.spatial_operation")
    if spatial not in {"mean", "none"}:
        raise BundleError(f"{where}.spatial_operation must be mean or none")
    return RouteContract(
        model_channel_ids=model_ids,
        target_band_ids=target_ids,
        matrix=matrix,
        canonical_matrix=canonical,
        operation=operation,
        spatial_operation=spatial,
        order_matches_contracts=(
            model_ids == tuple(band.id for band in model.channels)
            and target_ids == tuple(band.id for band in sensor.bands)
        ),
    )


def _npz_ids(array: np.ndarray, where: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in "SU":
        raise BundleError(f"{where} must be a one-dimensional string array")
    if array.size > MAX_BANDS:
        raise BundleError(f"{where} cannot contain more than {MAX_BANDS} IDs")
    maximum_item_bytes = 128 * (4 if array.dtype.kind == "U" else 1)
    if array.dtype.itemsize > maximum_item_bytes:
        raise BundleError(f"{where} string dtype exceeds the bounded identifier width")
    values: list[str] = []
    for index, value in enumerate(array.tolist()):
        if isinstance(value, bytes):
            try:
                value = value.decode("ascii")
            except UnicodeDecodeError as error:
                raise BundleError(f"{where}[{index}] is not ASCII") from error
        values.append(_identifier(value, f"{where}[{index}]"))
    if len(values) != len(set(values)):
        raise BundleError("duplicate_band_ids: NPZ target IDs must be unique")
    return tuple(values)


def _validate_numpy_artifact(
    artifact: dict[str, np.ndarray], model: ModelContract, sensor: SensorContract
) -> None:
    channels = len(model.channels)
    bands = len(sensor.bands)

    def exact_float(key: str, shape: tuple[int, ...], *, optional: bool = False) -> None:
        if optional and key not in artifact:
            return
        array = artifact[key]
        if array.dtype != np.dtype("float64") or array.shape != shape:
            raise BundleError(
                f"numpy-linear-v1 artifact {key!r} must be float64 with exact shape {shape}"
            )
        if not np.isfinite(array).all():
            raise BundleError(f"numpy-linear-v1 artifact {key!r} must be finite")
        if array.size and float(np.max(np.abs(array))) > MAX_ABS_NUMPY_ARTIFACT_NUMERIC_VALUE:
            raise BundleError(
                f"numpy-linear-v1 artifact {key!r} exceeds the absolute numeric bound"
            )

    exact_float("route_matrix", (channels, bands))
    exact_float("normalization_offset", (channels,))
    exact_float("normalization_scale", (channels,))
    exact_float("output_weights", (channels,))
    exact_float("wavelength_weights", (channels,), optional=True)
    exact_float("fwhm_weights", (channels,), optional=True)
    if "output_bias" in artifact:
        bias = artifact["output_bias"]
        if bias.shape not in {(), (1,)}:
            raise BundleError("numpy-linear-v1 artifact 'output_bias' must be a scalar")
        exact_float("output_bias", bias.shape)
    if np.any(artifact["normalization_scale"] < MIN_NORMALIZATION_SCALE) or np.any(
        artifact["normalization_scale"] > MAX_NORMALIZATION_SCALE
    ):
        raise BundleError(
            "numpy-linear-v1 artifact normalization_scale is outside the permitted range"
        )

    target_ids = _npz_ids(artifact["target_band_ids"], "artifact.target_band_ids")
    if len(target_ids) != bands or set(target_ids) != {band.id for band in sensor.bands}:
        raise BundleError(
            "numpy-linear-v1 artifact target_band_ids must cover the sensor exactly"
        )
    spatial = artifact["spatial_operation"]
    if spatial.shape not in {(), (1,)} or spatial.dtype.kind not in "SU":
        raise BundleError(
            "numpy-linear-v1 artifact spatial_operation must be a scalar string"
        )
    if spatial.dtype.itemsize > 64:
        raise BundleError(
            "numpy-linear-v1 artifact spatial_operation string exceeds its bound"
        )
    raw_spatial = spatial.reshape(-1)[0]
    if isinstance(raw_spatial, bytes):
        try:
            spatial_value = raw_spatial.decode("ascii")
        except UnicodeDecodeError as error:
            raise BundleError(
                "numpy-linear-v1 artifact spatial_operation must be ASCII"
            ) from error
    else:
        spatial_value = str(raw_spatial)
    if spatial_value not in {"none", "mean"}:
        raise BundleError(
            "numpy-linear-v1 artifact spatial_operation must be none or mean"
        )


def _parse_probes(data: bytes, sensor: SensorContract) -> ProbeSet:
    arrays = load_npz_bytes(data, source="probes.npz", exact_keys={"probes", "target_band_ids"})
    raw = arrays["probes"]
    if raw.dtype.kind not in "fiu" or raw.ndim not in {2, 4}:
        raise BundleError("probes.npz: probes must be numeric [N,B] or [N,B,H,W]")
    if not MIN_PROBES <= raw.shape[0] <= MAX_PROBES:
        raise BundleError(f"probes.npz: N must be in {MIN_PROBES}..{MAX_PROBES}")
    if raw.shape[1] != len(sensor.bands):
        raise BundleError("probes.npz: band axis does not match target sensor")
    spatial = 1 if raw.ndim == 2 else int(raw.shape[2] * raw.shape[3])
    if spatial < 1 or spatial > MAX_SPATIAL_CELLS:
        raise BundleError(f"probes.npz: spatial cells per probe exceed {MAX_SPATIAL_CELLS}")
    if raw.size > MAX_EXPANDED_FLOAT64_PROBE_BYTES // np.dtype("float64").itemsize:
        raise BundleError("probes.npz: float64 expansion exceeds decompressed byte budget")
    if (raw.shape[0] - 1) * raw.size > MAX_C2_SHIFT_CELL_COMPARISONS:
        raise BundleError(
            "probes.npz: exact C2 shift-selection work exceeds "
            f"{MAX_C2_SHIFT_CELL_COMPARISONS} float-cell comparisons"
        )
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all():
        raise BundleError("probes.npz: probes must be finite")
    if np.any(np.abs(values) > MAX_ABS_DECLARED_NUMERIC_VALUE):
        raise BundleError("probes.npz: probe values exceed the absolute numeric bound")
    if values.nbytes > MAX_CANARY_INVOCATION_PROBE_BYTES:
        raise BundleError("probes.npz: baseline exceeds the per-invocation float64 byte cap")
    target_ids = _npz_ids(arrays["target_band_ids"], "probes.npz.target_band_ids")
    sensor_ids = tuple(band.id for band in sensor.bands)
    if set(target_ids) != set(sensor_ids) or len(target_ids) != len(sensor_ids):
        raise BundleError("probes.npz: target_band_ids do not cover the sensor bands")
    # Canonicalize the input tensor to sensor-contract order while retaining the
    # submitted-order fact for the report.
    position = {identifier: index for index, identifier in enumerate(target_ids)}
    values = values[:, [position[identifier] for identifier in sensor_ids], ...]
    for index, band in enumerate(sensor.bands):
        assert band.valid_range is not None
        column = values[:, index, ...]
        if np.any(column < band.valid_range[0]) or np.any(column > band.valid_range[1]):
            raise BundleError(f"probes.npz: values for {band.id!r} are outside the declared target range")
    return ProbeSet(values=values, target_band_ids=target_ids, order_matches_sensor=target_ids == sensor_ids)


def _plan_adapter_work(probes: ProbeSet, bands: int) -> AdapterWorkPlan:
    values = probes.values
    rank4 = values.ndim == 4
    spatial_cells = 1 if not rank4 else int(values.shape[2] * values.shape[3])
    bytes_per_basis_row = bands * spatial_cells * np.dtype("float64").itemsize
    rows_per_basis_request = min(
        C1_CHUNK_MAX_ROWS,
        C1_CHUNK_MAX_FLOAT64_PROBE_BYTES // bytes_per_basis_row,
    )
    if rows_per_basis_request < 1:
        raise BundleError(
            "planned C1 basis request cannot fit the per-invocation probe-value cap"
        )
    basis_request_count = (
        bands + rows_per_basis_request - 1
    ) // rows_per_basis_request
    full_size_request_count = (
        REPLAY_COUNT
        + 1  # target-neutral request
        + bands  # C2, one request per target band
        + 6  # C3, three mutations for each of two metadata fields
        + (2 if bands > 1 else 0)  # tied-tuple and ID-only C4 requests
    )
    baseline_bytes = int(values.nbytes)
    basis_bytes = int(bands * bytes_per_basis_row)
    spatial_request_count = 1 if rank4 else 0
    spatial_bytes = int(4 * bytes_per_basis_row) if rank4 else 0
    cumulative = (
        full_size_request_count * baseline_bytes
        + basis_bytes
        + spatial_bytes
    )
    return AdapterWorkPlan(
        baseline_probe_value_bytes=baseline_bytes,
        full_size_request_count=full_size_request_count,
        basis_request_count=basis_request_count,
        basis_probe_value_bytes=basis_bytes,
        spatial_request_count=spatial_request_count,
        spatial_probe_value_bytes=spatial_bytes,
        total_invocation_count=(
            full_size_request_count
            + basis_request_count
            + spatial_request_count
        ),
        cumulative_request_probe_value_bytes=cumulative,
    )


def _validate_adapter(
    manifest: dict[str, Any], files: dict[str, FileRecord], root: Path
) -> dict[str, Any]:
    adapter = _expect_mapping(manifest.get("adapter"), "bandtrace.yaml.adapter")
    adapter_type = _expect_string(adapter.get("type"), "bandtrace.yaml.adapter.type")
    if adapter_type not in SUPPORTED_ADAPTERS:
        raise BundleError(f"bandtrace.yaml.adapter.type must be one of {SUPPORTED_ADAPTERS}")
    allowed = {"type"} if adapter_type == "numpy-linear-v1" else {"type", "argv"}
    unknown = set(adapter) - allowed
    if unknown:
        raise BundleError(f"bandtrace.yaml.adapter has unknown fields: {sorted(unknown)}")
    if adapter_type == "numpy-linear-v1":
        return {"type": adapter_type}
    argv = adapter.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        raise BundleError("bandtrace.yaml.adapter.argv must be a non-empty string array")
    if any(
        any(ord(character) < 32 or ord(character) == 127 for character in token)
        for token in argv
    ):
        raise BundleError("subprocess argv cannot contain NUL or control characters")
    allowed_placeholders = {"{input_npz}", "{output_npz}", "{artifact}"}
    if "{input_npz}" not in argv or "{output_npz}" not in argv:
        raise BundleError("subprocess argv must contain exact {input_npz} and {output_npz} tokens")
    if "{artifact}" not in argv:
        raise BundleError("subprocess argv must contain exact {artifact} token")
    if any(token in {"-c", "-m"} for token in argv):
        raise BundleError("subprocess inline code/module switches are forbidden; pin a runner file")
    asset_keys: list[str] = []
    for token in argv:
        if "{" in token or "}" in token:
            if token in allowed_placeholders:
                continue
            match = _ASSET_PLACEHOLDER.fullmatch(token)
            if match is None:
                raise BundleError(
                    f"subprocess argv token {token!r} contains an unsupported placeholder"
                )
            key = match.group(1)
            if key in REQUIRED_FILES:
                raise BundleError(
                    f"subprocess asset placeholder {token!r} must reference a manifest extra"
                )
            if key not in files:
                raise BundleError(
                    f"subprocess asset placeholder {token!r} is not a pinned manifest file"
                )
            asset_keys.append(key)
    return {
        "type": adapter_type,
        "argv": tuple(argv),
        "asset_keys": tuple(dict.fromkeys(asset_keys)),
    }


def _load_bundle_from_pinned_root_impl(
    root: Path, root_fd: int, held_descriptors: list[int]
) -> Bundle:
    manifest_file = _open_pinned_regular_file(
        root, root_fd, "bandtrace.yaml", source="bandtrace.yaml"
    )
    held_descriptors.append(manifest_file.descriptor)
    manifest_bytes = _read_pinned_regular_file(
        root,
        root_fd,
        manifest_file,
        max_bytes=MAX_MANIFEST_BYTES,
        source="bandtrace.yaml",
    )
    manifest = _strict_yaml(manifest_bytes, "bandtrace.yaml")
    _require_schema_version(manifest, "bandtrace.yaml")
    if manifest.get("policy_id") != POLICY_ID:
        raise BundleError(f"bandtrace.yaml.policy_id must be {POLICY_ID}")
    top_unknown = set(manifest) - {"schema_version", "policy_id", "files", "adapter"}
    if top_unknown:
        raise BundleError(f"bandtrace.yaml has unknown fields: {sorted(top_unknown)}")
    raw_files = _expect_mapping(manifest.get("files"), "bandtrace.yaml.files")
    if len(raw_files) > MAX_MANIFEST_DECLARED_FILES:
        raise BundleError(
            "bandtrace.yaml.files declares more than "
            f"{MAX_MANIFEST_DECLARED_FILES} files"
        )
    missing = set(REQUIRED_FILES) - set(raw_files)
    if missing:
        raise BundleError(f"bandtrace.yaml.files is missing {sorted(missing)}")
    records: dict[str, FileRecord] = {}
    specifications: dict[
        str, tuple[_PinnedRegularFile, str, int]
    ] = {}
    seen_paths: set[str] = set()
    aggregate_stat_bytes = 0
    try:
        for key, raw_ref in raw_files.items():
            _identifier(key, f"bandtrace.yaml.files key {key!r}")
            ref = _expect_mapping(raw_ref, f"bandtrace.yaml.files.{key}")
            if set(ref) != {"path", "sha256"}:
                raise BundleError(f"bandtrace.yaml.files.{key} must contain exactly path and sha256")
            relative = _expect_string(ref["path"], f"bandtrace.yaml.files.{key}.path")
            expected_hash = _expect_string(ref["sha256"], f"bandtrace.yaml.files.{key}.sha256")
            if not _SHA256.fullmatch(expected_hash):
                raise BundleError(f"bandtrace.yaml.files.{key}.sha256 must be lowercase SHA-256")
            if relative in seen_paths:
                raise BundleError("bandtrace.yaml.files cannot alias one path through multiple keys")
            seen_paths.add(relative)
            max_bytes = (
                MAX_PROBE_FILE_BYTES
                if key == "probes"
                else MAX_ADAPTER_OUTPUT_BYTES
                if key == "artifact"
                else MAX_MANIFEST_BYTES
                if key in {"model", "sensor", "route"}
                else MAX_MANIFEST_DECLARED_TOTAL_STAT_BYTES
            )
            source = f"bandtrace.yaml.files.{key}"
            pinned = _open_pinned_regular_file(
                root, root_fd, relative, source=source
            )
            specifications[key] = (pinned, expected_hash, max_bytes)
            stat_bytes = int(pinned.initial_stat.st_size)
            if stat_bytes > max_bytes:
                raise BundleError(
                    f"{source}: file size exceeds {max_bytes} bytes"
                )
            aggregate_stat_bytes += stat_bytes
            if aggregate_stat_bytes > MAX_MANIFEST_DECLARED_TOTAL_STAT_BYTES:
                raise BundleError(
                    "bandtrace.yaml.files aggregate stat bytes exceed "
                    f"{MAX_MANIFEST_DECLARED_TOTAL_STAT_BYTES}"
                )
            _verify_bundle_root_identity(root, root_fd)

        # Every declared file descriptor is held across the complete aggregate
        # stat gate. Payload reads use those same descriptors; safe path
        # re-walks only verify that no component identity was substituted.
        aggregate_payload_bytes = 0
        for key, (pinned, expected_hash, max_bytes) in specifications.items():
            source = f"bandtrace.yaml.files.{key}"
            data = _read_pinned_regular_file(
                root,
                root_fd,
                pinned,
                max_bytes=max_bytes,
                source=source,
            )
            aggregate_payload_bytes += len(data)
            if aggregate_payload_bytes > MAX_MANIFEST_DECLARED_TOTAL_STAT_BYTES:
                raise BundleError(
                    "bandtrace.yaml.files aggregate payload bytes changed beyond the "
                    f"{MAX_MANIFEST_DECLARED_TOTAL_STAT_BYTES} preflight cap"
                )
            observed_hash = sha256_bytes(data)
            if observed_hash != expected_hash:
                raise BundleError(f"{source}: SHA-256 mismatch")
            records[key] = FileRecord(
                key,
                pinned.relative_path,
                pinned.path,
                observed_hash,
                data,
                len(data),
            )
    finally:
        for pinned, _, _ in specifications.values():
            held_descriptors.append(pinned.descriptor)

    model_payload = _strict_json(records["model"].data, records["model"].relative_path)
    sensor_payload = _strict_json(records["sensor"].data, records["sensor"].relative_path)
    route_payload = _strict_json(records["route"].data, records["route"].relative_path)
    model = _parse_model(model_payload, records["artifact"].sha256)
    sensor = _parse_sensor(sensor_payload)
    route = _parse_route(route_payload, model, sensor)
    probes = _parse_probes(records["probes"].data, sensor)
    adapter_work_plan = _plan_adapter_work(probes, len(sensor.bands))
    if (
        adapter_work_plan.cumulative_request_probe_value_bytes
        > MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES
    ):
        raise BundleError(
            "planned cumulative adapter request probe-value bytes exceed "
            f"{MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES}"
        )
    probe_record = records["probes"]
    records["probes"] = FileRecord(
        key=probe_record.key,
        relative_path=probe_record.relative_path,
        path=probe_record.path,
        sha256=probe_record.sha256,
        data=None,
        byte_count=probe_record.byte_count,
    )
    adapter = _validate_adapter(manifest, records, root)
    numpy_artifact = None
    if adapter["type"] == "numpy-linear-v1":
        numpy_artifact = load_npz_bytes(
            records["artifact"].data,
            source=records["artifact"].relative_path,
            allowed_keys=_ARTIFACT_KEYS,
        )
        required_artifact = {
            "route_matrix",
            "target_band_ids",
            "normalization_offset",
            "normalization_scale",
            "output_weights",
            "spatial_operation",
        }
        missing_artifact = required_artifact - set(numpy_artifact)
        if missing_artifact:
            raise BundleError(
                f"numpy-linear-v1 artifact is missing keys: {sorted(missing_artifact)}"
            )
        _validate_numpy_artifact(numpy_artifact, model, sensor)
        artifact_record = records["artifact"]
        records["artifact"] = FileRecord(
            key=artifact_record.key,
            relative_path=artifact_record.relative_path,
            path=artifact_record.path,
            sha256=artifact_record.sha256,
            data=None,
            byte_count=artifact_record.byte_count,
        )
    if (route.spatial_operation == "none" and probes.values.ndim != 2) or (
        route.spatial_operation == "mean" and probes.values.ndim != 4
    ):
        raise BundleError(
            "route spatial_operation must be none for [N,B] or mean for [N,B,H,W] exactly"
        )
    result = Bundle(
        root=root,
        manifest=manifest,
        manifest_sha256=sha256_bytes(manifest_bytes),
        files=records,
        adapter=adapter,
        model=model,
        sensor=sensor,
        route=route,
        probes=probes,
        adapter_work_plan=adapter_work_plan,
        numpy_artifact=numpy_artifact,
    )
    _verify_pinned_regular_file_unchanged(
        root,
        root_fd,
        manifest_file,
        source="bandtrace.yaml",
    )
    for key, (pinned, _, _) in specifications.items():
        _verify_pinned_regular_file_unchanged(
            root,
            root_fd,
            pinned,
            source=f"bandtrace.yaml.files.{key}",
        )
    _verify_bundle_root_identity(root, root_fd)
    return result


def _load_bundle_from_pinned_root(root: Path, root_fd: int) -> Bundle:
    held_descriptors: list[int] = []
    try:
        return _load_bundle_from_pinned_root_impl(
            root, root_fd, held_descriptors
        )
    finally:
        for descriptor in reversed(held_descriptors):
            os.close(descriptor)


def load_bundle(bundle_dir: Path) -> Bundle:
    """Load and validate a hash-pinned BandTrace v0.1 bundle."""

    root, root_fd = _open_bundle_root(Path(bundle_dir))
    try:
        try:
            return _load_bundle_from_pinned_root(root, root_fd)
        except MemoryError as error:
            raise BundleError(
                "bundle loading exceeded available memory within frozen byte limits"
            ) from error
    finally:
        os.close(root_fd)
