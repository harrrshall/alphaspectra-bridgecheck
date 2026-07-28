"""Create the installed BandTrace reference bundle used by the quickstart."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import deterministic_npz_bytes
from .constants import POLICY_ID

MODEL_CENTERS_NM = np.asarray([450.0, 550.0, 650.0, 750.0], dtype=np.float64)
TARGET_CENTERS_NM = np.asarray(
    [450.0, 550.0, 650.0, 750.0, 950.0], dtype=np.float64
)
MODEL_CHANNEL_IDS = [f"m{int(center)}" for center in MODEL_CENTERS_NM]
TARGET_BAND_IDS = [f"t{int(center)}" for center in TARGET_CENTERS_NM]
ROUTE_MATRIX = np.pad(np.eye(4, dtype=np.float64), ((0, 0), (0, 1)))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _srf(center_nm: float) -> dict[str, Any]:
    return {
        "wavelengths": [
            center_nm - 20.0,
            center_nm - 10.0,
            center_nm,
            center_nm + 10.0,
            center_nm + 20.0,
        ],
        "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
        "wavelength_unit": "nm",
    }


def _band(identifier: str, center_nm: float, *, target: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": identifier,
        "center_wavelength": center_nm,
        "wavelength_unit": "nm",
        "fwhm": 20.0,
        "fwhm_unit": "nm",
        "srf": _srf(center_nm),
    }
    if target:
        result["neutral_value"] = 0.5
    return result


def _probes(count: int = 20) -> np.ndarray:
    rows = np.arange(count, dtype=np.int64)[:, None]
    bands = np.arange(len(TARGET_BAND_IDS), dtype=np.int64)[None, :]
    values = 0.08 + 0.84 * (((rows * (bands * 2 + 3) + bands * 7) % 23) / 22.0)
    return np.ascontiguousarray(values, dtype=np.float64)


def make_reference_bundle(root: Path) -> None:
    """Write a small clean ``numpy-linear-v1`` bundle to a fresh path."""

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {root}")
    root.mkdir(parents=True, mode=0o700)

    model_path = root / "model.json"
    sensor_path = root / "sensor.json"
    route_path = root / "route.json"
    probes_path = root / "probes.npz"
    artifact_path = root / "reference_model.npz"

    route = {
        "schema_version": "1.0",
        "model_channel_ids": MODEL_CHANNEL_IDS,
        "target_band_ids": TARGET_BAND_IDS,
        "matrix": ROUTE_MATRIX.tolist(),
        "operation": "selection_or_permutation",
        "spatial_operation": "none",
    }
    _write_json(route_path, route)

    sensor = {
        "schema_version": "1.0",
        "sensor_id": "bandtrace-reference-sensor",
        "sensor_model": "Synthetic Five Band",
        "sensor_serial": "REFERENCE-0001",
        "target_bands": [
            _band(identifier, float(center), target=True)
            for identifier, center in zip(
                TARGET_BAND_IDS, TARGET_CENTERS_NM, strict=True
            )
        ],
        "radiometric_quantity": "unitless_reflectance_factor",
        "valid_range": [0.0, 1.0],
        "calibration_state": "synthetic_exact",
        "preprocessing_version": "identity-v1",
    }
    _write_json(sensor_path, sensor)

    probes_path.write_bytes(
        deterministic_npz_bytes(
            {
                "probes": _probes(),
                "target_band_ids": np.asarray(TARGET_BAND_IDS),
            }
        )
    )
    artifact_path.write_bytes(
        deterministic_npz_bytes(
            {
                "route_matrix": ROUTE_MATRIX,
                "target_band_ids": np.asarray(TARGET_BAND_IDS),
                "normalization_offset": np.zeros(4, dtype=np.float64),
                "normalization_scale": np.ones(4, dtype=np.float64),
                "output_weights": np.asarray(
                    [0.7, -0.5, 0.35, 0.9], dtype=np.float64
                ),
                "output_bias": np.asarray(0.125, dtype=np.float64),
                "spatial_operation": np.asarray("none"),
            }
        )
    )

    model = {
        "schema_version": "1.0",
        "model_id": "bandtrace-reference-linear-model",
        "model_version": "1.0",
        "artifact_sha256": _sha256(artifact_path),
        "model_channels": [
            _band(identifier, float(center), target=False)
            for identifier, center in zip(
                MODEL_CHANNEL_IDS, MODEL_CENTERS_NM, strict=True
            )
        ],
        "radiometric_quantity": "unitless_reflectance_factor",
        "valid_range": [0.0, 1.0],
        "normalization": {
            "type": "affine",
            "offset": [0.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0, 1.0],
        },
        "declared_validated_support": {
            "supplier_assertion": True,
            "wavelength_range": [420.0, 780.0],
            "wavelength_unit": "nm",
        },
        "pre_decision_output": {"name": "score"},
        "wavelength_conditioned": False,
        "fwhm_conditioned": False,
        "required_dependence_target_band_ids": TARGET_BAND_IDS[:4],
    }
    _write_json(model_path, model)

    paths = {
        "artifact": artifact_path,
        "model": model_path,
        "probes": probes_path,
        "route": route_path,
        "sensor": sensor_path,
    }
    _write_json(
        root / "bandtrace.yaml",
        {
            "schema_version": "1.0",
            "policy_id": POLICY_ID,
            "files": {
                key: {"path": path.name, "sha256": _sha256(path)}
                for key, path in paths.items()
            },
            "adapter": {"type": "numpy-linear-v1"},
        },
    )
