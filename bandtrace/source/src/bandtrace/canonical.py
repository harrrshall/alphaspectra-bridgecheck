"""Canonical serialization and content hashing."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import CANONICAL_FLOAT_PLACES
from .errors import BandTraceError, ExecutionError


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_seed(
    *,
    model_hash: str,
    sensor_hash: str,
    probe_hash: str,
    route_hash: str,
    policy_id: str,
) -> bytes:
    """Length-frame the frozen seed materials to avoid concatenation ambiguity."""

    digest = hashlib.sha256()
    digest.update(b"bandtrace-mutation-seed-v1\x00")
    for label, value in (
        (b"model", model_hash.encode("ascii")),
        (b"sensor", sensor_hash.encode("ascii")),
        (b"probe", probe_hash.encode("ascii")),
        (b"route", route_hash.encode("ascii")),
        (b"policy", policy_id.encode("utf-8")),
    ):
        digest.update(len(label).to_bytes(2, "big"))
        digest.update(label)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.digest()


def derive_canary_seed(base_seed: bytes, canary_id: str) -> bytes:
    if len(base_seed) != 32:
        raise BandTraceError("base seed must contain exactly 32 bytes")
    identifier = canary_id.encode("utf-8")
    if len(identifier) > 65535:
        raise BandTraceError("canary identifier is too long")
    digest = hashlib.sha256()
    digest.update(b"bandtrace-canary-v1\x00")
    digest.update(base_seed)
    digest.update(len(identifier).to_bytes(2, "big"))
    digest.update(identifier)
    return digest.digest()


def c3_rank_amplitudes(
    base_seed: bytes, canary_id: str, target_band_ids: tuple[str, ...]
) -> np.ndarray:
    """Return the frozen C3 non-uniform amplitudes in submitted ID order."""

    if not target_band_ids or len(target_band_ids) != len(set(target_band_ids)):
        raise BandTraceError("C3 target band IDs must be non-empty and unique")
    subseed = derive_canary_seed(base_seed, canary_id)
    ranked: list[tuple[bytes, str, int]] = []
    for index, identifier in enumerate(target_band_ids):
        encoded = identifier.encode("utf-8")
        if len(encoded) > 65_535:
            raise BandTraceError("C3 target band identifier is too long")
        digest = hashlib.sha256()
        digest.update(subseed)
        digest.update(b"\x00rank\x00")
        digest.update(len(encoded).to_bytes(2, "big"))
        digest.update(encoded)
        ranked.append((digest.digest(), identifier, index))
    ranked.sort(key=lambda item: (item[0], item[1]))
    amplitudes = np.empty(len(target_band_ids), dtype=np.float64)
    denominator = len(target_band_ids) + 1
    for rank, (_, _, input_index) in enumerate(ranked):
        amplitudes[input_index] = (rank + 1) / denominator
    return amplitudes


def linear_quantile(values: np.ndarray, q: float) -> float:
    """Frozen finite flattened linear quantile, independent of NumPy defaults."""

    if not 0.0 <= q <= 1.0:
        raise BandTraceError("quantile must be between zero and one")
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0 or not np.isfinite(flattened).all():
        raise BandTraceError("quantile input must be non-empty and finite")
    ordered = np.sort(flattened)
    h = (ordered.size - 1) * q
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    fraction = h - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def installed_distribution_version() -> str:
    """Return the installed distribution version, with a source-tree fallback."""

    try:
        version = importlib.metadata.version("alphaspectra-bandtrace")
    except importlib.metadata.PackageNotFoundError:
        from .constants import PRODUCT_VERSION

        version = PRODUCT_VERSION
    except Exception as error:
        raise ExecutionError(
            "cannot read installed BandTrace distribution metadata"
        ) from error
    if not isinstance(version, str) or not version.strip():
        raise ExecutionError(
            "installed BandTrace distribution version is missing or empty"
        )
    return version


def installed_source_tree_sha256(package_directory: Path | None = None) -> str:
    """Hash regular Python sources under the installed ``bandtrace`` package."""

    root = Path(__file__).resolve().parent if package_directory is None else Path(package_directory)
    if root.is_symlink() or not root.is_dir():
        raise ExecutionError(
            "installed BandTrace source tree is unavailable as a real directory"
        )
    records: list[tuple[bytes, bytes]] = []
    try:
        for candidate in root.rglob("*.py"):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            relative = candidate.relative_to(root).as_posix().encode("utf-8")
            records.append((relative, candidate.read_bytes()))
    except OSError as error:
        raise ExecutionError(
            "cannot read the installed BandTrace Python source tree"
        ) from error
    if not records:
        raise ExecutionError(
            "installed BandTrace source tree contains no regular Python members"
        )
    records.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    digest.update(b"bandtrace-source-tree-v1\x00")
    for relative, data in records:
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def c4_shift(base_seed: bytes, n_bands: int) -> int:
    if n_bands < 1:
        raise BandTraceError("band count must be positive")
    if n_bands == 1:
        return 0
    seed = derive_canary_seed(base_seed, "C4_order")
    return 1 + int.from_bytes(seed[:8], "big") % (n_bands - 1)


def c4_permutation(base_seed: bytes, n_bands: int) -> np.ndarray:
    """Return the frozen non-identity cyclic C4 permutation."""

    shift = c4_shift(base_seed, n_bands)
    # p maps each submitted position to its source position.
    return (np.arange(n_bands, dtype=np.int64) + shift) % n_bands


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BandTraceError("canonical outputs cannot contain non-finite numbers")
        rounded = round(value, CANONICAL_FLOAT_PLACES)
        return 0.0 if rounded == 0 else rounded
    raise BandTraceError(f"unsupported canonical output type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _canonical(value),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _write_deterministic_npz_archive(
    output: Any, arrays: Mapping[str, np.ndarray]
) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for key in sorted(arrays):
            if not key or any(
                ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                for ch in key
            ):
                raise BandTraceError(f"unsafe NPZ member key: {key!r}")
            array = np.asarray(arrays[key])
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise BandTraceError(f"NPZ output {key!r} has a forbidden dtype")

            # ZipFile can stream a stored member without changing the legacy
            # deterministic bytes when its exact size is known before open.
            # The NPY v1 header is small and deterministic; array payloads are
            # then emitted by NumPy's bounded external-loop iterator instead
            # of first materializing a second member-sized bytes object.
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header, np.lib.format.header_data_from_array_1_0(array)
            )
            info = zipfile.ZipInfo(
                f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.file_size = len(header.getbuffer()) + int(array.nbytes)
            with archive.open(info, "w") as member:
                np.lib.format.write_array(
                    member, array, version=(1, 0), allow_pickle=False
                )


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create a byte-stable, non-pickled, uncompressed NPZ archive."""

    output = io.BytesIO()
    _write_deterministic_npz_archive(output, arrays)
    return output.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Stream the deterministic NPZ representation to a fresh regular path."""

    destination = Path(path)
    with destination.open("xb") as output:
        _write_deterministic_npz_archive(output, arrays)
