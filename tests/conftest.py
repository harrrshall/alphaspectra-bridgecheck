from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pytest


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PRODUCT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bridgecheck.artifact import BridgeArtifact, MODEL_ID, array_sha256
from bridgecheck.audit import PairedSpectrum


def _candidate_bank() -> np.ndarray:
    wavelength = np.arange(400.0, 2400.0 + 2.0, 4.0, dtype=np.float64)
    context_mask = (wavelength >= 400.0) & (wavelength <= 1000.0)
    target_mask = wavelength > 1050.0
    context_x = np.linspace(0.0, 1.0, int(context_mask.sum()), dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, int(target_mask.sum()), dtype=np.float64)

    context_0 = 0.08 + 0.24 * context_x + 0.025 * np.sin(5.0 * np.pi * context_x)
    context_2 = 0.28 + 0.13 * context_x + 0.020 * np.cos(4.0 * np.pi * context_x)

    # The audit's fixed control seed sees these exact first and second permutations.
    # This makes the shuffle control deterministic and observably worse than intact retrieval.
    rng = np.random.default_rng(2026072701)
    context_1 = context_0[rng.permutation(len(context_0))]
    context_3 = context_2[rng.permutation(len(context_2))]

    targets = np.asarray(
        [
            0.43 + 0.035 * np.sin(4.0 * np.pi * target_x),
            0.73 + 0.025 * np.cos(3.0 * np.pi * target_x),
            0.59 + 0.030 * np.cos(5.0 * np.pi * target_x),
            0.82 + 0.020 * np.sin(2.0 * np.pi * target_x),
        ],
        dtype=np.float64,
    )
    contexts = np.asarray([context_0, context_1, context_2, context_3], dtype=np.float64)

    bank = np.empty((4, len(wavelength)), dtype=np.float64)
    for index in range(len(bank)):
        # Fill the unexposed 1004--1048 nm bridge without relying on it in retrieval or scoring.
        bank[index] = np.interp(
            wavelength,
            [400.0, 1000.0, 1052.0, 2400.0],
            [contexts[index, 0], contexts[index, -1], targets[index, 0], targets[index, -1]],
        )
        bank[index, context_mask] = contexts[index]
        bank[index, target_mask] = targets[index]
    return np.ascontiguousarray(bank, dtype=np.float64)


def _manifest_for(bank: np.ndarray) -> dict[str, Any]:
    bank_bytes = np.ascontiguousarray(bank, dtype="<f8").tobytes(order="C")
    return {
        "schema_version": "1.0",
        "model_id": MODEL_ID,
        "version": "0.1.0-test",
        "artifact": {
            "filename": "bridge_v1.f64",
            "dtype": "<f8",
            "order": "C",
            "shape": list(bank.shape),
            "file_sha256": hashlib.sha256(bank_bytes).hexdigest(),
            "array_sha256": array_sha256(np.asarray(bank, dtype="<f8")),
        },
        "spectral_grid": {
            "start_nm": 400.0,
            "end_nm": 2400.0,
            "step_nm": 4.0,
            "count": 501,
            "context_range_nm": [400.0, 1000.0],
            "context_count": 151,
            "target_above_nm": 1050.0,
            "target_count": 338,
        },
        "input_contract": {
            "reflectance_unit": "fraction",
            "absolute_context_range_nm": [400.0, 1000.0],
            "minimum_bands": 100,
            "start_at_or_below_nm": 420.0,
            "end_at_or_above_nm": 980.0,
            "maximum_gap_nm": 10.0,
            "reflectance_range": [-0.05, 1.0],
        },
        "support_reference": {
            "metric": "context_rmse_descriptive_only",
            "context_rmse_quantiles": {"q50": 0.0, "q90": 0.0, "q95": 0.005, "q99": 0.02},
        },
        "claim_ceiling": "candidate_only_without_paired_measurement_audit",
        "evidence": {"test_fixture": True},
        "licenses": {"code": "Apache-2.0", "fixture": "synthetic_test_only"},
    }


@pytest.fixture
def synthetic_bank() -> np.ndarray:
    return _candidate_bank()


@pytest.fixture
def model_factory(tmp_path: Path) -> Callable[..., Path]:
    counter = 0

    def write_model(
        bank: np.ndarray | None = None,
        *,
        mutate_manifest: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        nonlocal counter
        counter += 1
        root = tmp_path / f"model-{counter}"
        root.mkdir()
        candidate = _candidate_bank() if bank is None else np.ascontiguousarray(bank)
        manifest = _manifest_for(candidate)
        if mutate_manifest is not None:
            mutate_manifest(manifest)
        artifact_name = manifest.get("artifact", {}).get("filename", "bridge_v1.f64")
        (root / artifact_name).write_bytes(
            np.ascontiguousarray(candidate, dtype="<f8").tobytes(order="C")
        )
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return root

    return write_model


@pytest.fixture
def model_dir(model_factory: Callable[..., Path]) -> Path:
    return model_factory()


@pytest.fixture
def artifact(model_dir: Path) -> BridgeArtifact:
    return BridgeArtifact.load(model_dir, verify_official=False)


@pytest.fixture
def passing_samples(artifact: BridgeArtifact) -> list[PairedSpectrum]:
    context_wavelength = artifact.wavelengths_nm[artifact.context_mask]
    target_wavelength = artifact.wavelengths_nm[artifact.target_mask]
    return [
        PairedSpectrum(
            sample_id="sample-0",
            group_id="plant-0",
            context_wavelength_nm=context_wavelength.copy(),
            context_reflectance=artifact.bank[0, artifact.context_mask].copy(),
            target_wavelength_nm=target_wavelength.copy(),
            target_reflectance=artifact.bank[0, artifact.target_mask].copy(),
        ),
        PairedSpectrum(
            sample_id="sample-2",
            group_id="plant-2",
            context_wavelength_nm=context_wavelength.copy(),
            context_reflectance=artifact.bank[2, artifact.context_mask].copy(),
            target_wavelength_nm=target_wavelength.copy(),
            target_reflectance=artifact.bank[2, artifact.target_mask].copy(),
        ),
    ]


@pytest.fixture
def failing_samples(passing_samples: list[PairedSpectrum]) -> list[PairedSpectrum]:
    result: list[PairedSpectrum] = []
    for sample in passing_samples:
        result.append(
            PairedSpectrum(
                sample_id=sample.sample_id,
                group_id=sample.group_id,
                context_wavelength_nm=sample.context_wavelength_nm.copy(),
                context_reflectance=sample.context_reflectance.copy(),
                target_wavelength_nm=sample.target_wavelength_nm.copy(),
                target_reflectance=np.full_like(
                    sample.target_reflectance, float(sample.context_reflectance.mean())
                ),
            )
        )
    return result


def sample_to_api(sample: PairedSpectrum) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "context_wavelength_nm": sample.context_wavelength_nm.tolist(),
        "context_reflectance": sample.context_reflectance.tolist(),
        "target_wavelength_nm": sample.target_wavelength_nm.tolist(),
        "target_reflectance": sample.target_reflectance.tolist(),
    }
