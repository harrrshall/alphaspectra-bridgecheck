"""Bounded, non-pickled NPZ input handling."""

from __future__ import annotations

import io
import math
import re
import zipfile
from typing import Iterable

import numpy as np

from .constants import (
    MAX_NPZ_COMPRESSION_RATIO,
    MAX_NPZ_MEMBERS,
    MAX_NPZ_UNCOMPRESSED_BYTES,
)
from .errors import BundleError, ExecutionError

_MEMBER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\.npy\Z")
_SUPPORTED_COMPRESSION_METHODS = {
    zipfile.ZIP_STORED,
    zipfile.ZIP_DEFLATED,
}


def load_npz_bytes(
    data: bytes,
    *,
    source: str,
    allowed_keys: Iterable[str] | None = None,
    exact_keys: Iterable[str] | None = None,
    execution_output: bool = False,
) -> dict[str, np.ndarray]:
    """Inspect ZIP metadata before loading arrays with ``allow_pickle=False``."""

    error_type = ExecutionError if execution_output else BundleError
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_NPZ_MEMBERS:
                raise error_type(f"{source}: NPZ member count is outside 1..{MAX_NPZ_MEMBERS}")
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or len({name.casefold() for name in names}) != len(names):
                raise error_type(f"{source}: duplicate or case-aliasing NPZ member")
            total = 0
            for member in members:
                if not _MEMBER.fullmatch(member.filename):
                    raise error_type(f"{source}: unsafe NPZ member name {member.filename!r}")
                if member.flag_bits & 0x1:
                    raise error_type(f"{source}: encrypted NPZ members are forbidden")
                if member.compress_type not in _SUPPORTED_COMPRESSION_METHODS:
                    raise error_type(
                        f"{source}: unsupported NPZ compression method {member.compress_type}"
                    )
                total += member.file_size
                if total > MAX_NPZ_UNCOMPRESSED_BYTES:
                    raise error_type(f"{source}: NPZ uncompressed byte budget exceeded")
                if member.file_size:
                    if member.compress_size == 0:
                        raise error_type(f"{source}: invalid zero compressed size")
                    if member.file_size / member.compress_size > MAX_NPZ_COMPRESSION_RATIO:
                        raise error_type(f"{source}: NPZ compression ratio exceeds policy")
            keys = [name[:-4] for name in names]
            allowed = set(allowed_keys) if allowed_keys is not None else None
            exact = set(exact_keys) if exact_keys is not None else None
            if allowed is not None and not set(keys).issubset(allowed):
                unknown = sorted(set(keys) - allowed)
                raise error_type(f"{source}: unexpected NPZ keys: {unknown}")
            if exact is not None and set(keys) != exact:
                raise error_type(f"{source}: NPZ keys must be exactly {sorted(exact)}")
            result: dict[str, np.ndarray] = {}
            for member in members:
                member_data = archive.read(member)
                member_stream = io.BytesIO(member_data)
                try:
                    version = np.lib.format.read_magic(member_stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(member_stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(member_stream)
                    else:
                        raise error_type(
                            f"{source}: unsupported NPY format version {version}"
                        )
                except error_type:
                    raise
                except (EOFError, ValueError) as error:
                    raise error_type(f"{source}: invalid bounded NPY header: {error}") from error
                dtype = np.dtype(dtype)
                if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufSU":
                    raise error_type(
                        f"{source}: object, structured, or unsupported dtype forbidden"
                    )
                element_count = math.prod(shape)
                expected_payload = element_count * dtype.itemsize
                remaining_payload = len(member_data) - member_stream.tell()
                if expected_payload > MAX_NPZ_UNCOMPRESSED_BYTES:
                    raise error_type(f"{source}: declared NPY array exceeds byte budget")
                if expected_payload != remaining_payload:
                    raise error_type(
                        f"{source}: NPY header shape/dtype does not match member payload bytes"
                    )
                array = np.load(io.BytesIO(member_data), allow_pickle=False)
                if not isinstance(array, np.ndarray):
                    raise error_type(f"{source}: member is not a NumPy array")
                # Individual .npy loading returns an owned ndarray. Release the
                # member bytes before reading the next member.
                result[member.filename[:-4]] = array
    except (BundleError, ExecutionError):
        raise
    except (
        OSError,
        ValueError,
        OverflowError,
        MemoryError,
        EOFError,
        RuntimeError,
        NotImplementedError,
        RecursionError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as error:
        raise error_type(f"{source}: invalid safe NPZ: {error}") from error

    for key, array in result.items():
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise error_type(f"{source}: object or structured dtype forbidden for {key!r}")
        if array.dtype.kind not in "biufSU":
            raise error_type(f"{source}: unsupported dtype for {key!r}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise error_type(f"{source}: non-finite values forbidden for {key!r}")
    return result
