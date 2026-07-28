"""Deterministic BandTrace report rendering."""

from __future__ import annotations

import csv
import html
import io
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np

from .bundle import Bundle
from .canonical import canonical_json_bytes


def compact_route_audit(bundle: Bundle) -> dict[str, Any]:
    """Return axis-keyed parallel arrays without per-cell mapping overhead."""

    matrix = bundle.route.canonical_matrix
    return {
        "model_channel_ids": [channel.id for channel in bundle.model.channels],
        "target_band_ids": [band.id for band in bundle.sensor.bands],
        "declared_weight": matrix.tolist(),
        "declared_weight_float64_hex": [
            [float(weight).hex() for weight in row] for row in matrix
        ],
        "declared_weight_is_strictly_positive": (matrix > 0.0).tolist(),
        "declared_target_column_is_exactly_zero": np.all(
            matrix == 0.0, axis=0
        ).tolist(),
    }


def route_audit_rows(bundle: Bundle) -> Iterator[dict[str, Any]]:
    """Yield per-cell rows for the separate route.csv artifact."""

    matrix = bundle.route.canonical_matrix
    column_is_zero = np.all(matrix == 0.0, axis=0)
    for model_index, channel in enumerate(bundle.model.channels):
        for target_index, band in enumerate(bundle.sensor.bands):
            weight = float(matrix[model_index, target_index])
            yield {
                "model_channel_index": model_index,
                "model_channel_id": channel.id,
                "target_band_index": target_index,
                "target_band_id": band.id,
                "declared_weight": weight,
                "declared_weight_float64_hex": weight.hex(),
                "declared_weight_is_strictly_positive": weight > 0.0,
                "declared_target_column_is_exactly_zero": bool(
                    column_is_zero[target_index]
                ),
            }


def _csv_safe(value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, float):
        text = format(value, ".8f")
    else:
        text = str(value)
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def render_route_csv(rows: Iterable[dict[str, Any]]) -> bytes:
    fields = [
        "model_channel_index",
        "model_channel_id",
        "target_band_index",
        "target_band_id",
        "declared_weight",
        "declared_weight_float64_hex",
        "declared_weight_is_strictly_positive",
        "declared_target_column_is_exactly_zero",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_safe(row[field]) for field in fields})
    return stream.getvalue().encode("utf-8")


def render_html(
    report: dict[str, Any], artifact_sha256: Mapping[str, str]
) -> bytes:
    """Render a bounded human summary without duplicating route matrices."""

    facts = report["facts"]
    provenance_keys = (
        "bundle_manifest_sha256",
        "input_files",
        "mutation_seed_sha256_hex",
        "installed_source_tree_sha256",
        "installed_distribution_version",
        "installed_source_digest_scope",
        "execution_environment_attested",
        "native_dependency_bytes_hashed",
        "runtime_fingerprint_non_exhaustive",
        "normative_product_document_sha256",
        "normative_machine_config_sha256",
        "packaged_hash_gate_is_external_authentication",
        "adapter_type",
        "adapter_trust_state",
        "route_assurance",
        "runtime_fingerprint",
    )
    summary = {
        "product": report["product"],
        "product_version": report["product_version"],
        "policy_id": report["policy_id"],
        "exit_code": report["exit_code"],
        "states": report["states"],
        "faults": report["faults"],
        "limitations": report["limitations"],
        "canary_statuses": {
            canary_id: canary["status"]
            for canary_id, canary in sorted(report["canaries"].items())
        },
        "provenance": {key: facts[key] for key in provenance_keys},
        "artifact_sha256": dict(artifact_sha256),
    }
    serialized = canonical_json_bytes(summary).decode("utf-8")
    document = (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        "<title>BandTrace v0.1 report</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:90rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:1rem}"
        "</style></head><body>"
        "<h1>BandTrace v0.1 model–sensor conformance preflight</h1>"
        "<p>This is a software-conformance report, not a certificate or deployment approval.</p>"
        f"<pre>{html.escape(serialized, quote=True)}</pre>"
        "</body></html>\n"
    )
    return document.encode("utf-8")
