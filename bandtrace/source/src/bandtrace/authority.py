"""Build-internal verification of frozen normative resources shipped in the wheel."""

from __future__ import annotations

import hashlib
import importlib.resources
import re
from dataclasses import dataclass

from .constants import (
    NORMATIVE_MACHINE_CONFIG_SHA256,
    NORMATIVE_PRODUCT_DOCUMENT_SHA256,
)
from .errors import ExecutionError

_NORMATIVE_PACKAGE = "bandtrace.normative"
_PRODUCT_DOCUMENT = "BANDTRACE_PRODUCT.md"
_MACHINE_CONFIG = "bandtrace_v1.yaml"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class NormativeAuthority:
    """Build-verified hashes of the exact normative resources used here."""

    product_document_sha256: str
    machine_config_sha256: str


def _read_normative_resource(filename: str) -> bytes:
    try:
        return (
            importlib.resources.files(_NORMATIVE_PACKAGE)
            .joinpath(filename)
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError) as error:
        raise ExecutionError(
            f"packaged normative authority resource is unavailable: {filename}"
        ) from error


def _verify_digest(filename: str, payload: bytes, expected: str) -> str:
    if _SHA256_PATTERN.fullmatch(expected) is None:
        raise ExecutionError(
            f"frozen normative authority digest is invalid for {filename}"
        )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ExecutionError(
            f"packaged normative authority hash mismatch for {filename}"
        )
    return observed


def verify_packaged_normative_authority() -> NormativeAuthority:
    """Check co-packaged resource bytes against this build's embedded hashes."""

    product_document = _read_normative_resource(_PRODUCT_DOCUMENT)
    machine_config = _read_normative_resource(_MACHINE_CONFIG)
    return NormativeAuthority(
        product_document_sha256=_verify_digest(
            _PRODUCT_DOCUMENT,
            product_document,
            NORMATIVE_PRODUCT_DOCUMENT_SHA256,
        ),
        machine_config_sha256=_verify_digest(
            _MACHINE_CONFIG,
            machine_config,
            NORMATIVE_MACHINE_CONFIG_SHA256,
        ),
    )
