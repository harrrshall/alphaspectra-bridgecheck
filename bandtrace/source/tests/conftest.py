from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import numpy as np
import pytest


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PRODUCT_ROOT / "src"
FIXTURE_ADAPTER = Path(__file__).with_name("fixture_adapter.py")

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


MODEL_CENTERS_NM = np.asarray([450.0, 550.0, 650.0, 750.0], dtype=np.float64)
TARGET_CENTERS_NM = np.asarray([450.0, 550.0, 650.0, 750.0, 950.0], dtype=np.float64)
MODEL_CHANNEL_IDS = [f"m{int(value)}" for value in MODEL_CENTERS_NM]
TARGET_BAND_IDS = [f"t{int(value)}" for value in TARGET_CENTERS_NM]
ROUTE_MATRIX = np.pad(np.eye(4, dtype=np.float64), ((0, 0), (0, 1)))
OUTPUT_WEIGHTS = np.asarray([0.7, -0.5, 0.35, 0.9], dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _srf(center_nm: float) -> dict[str, Any]:
    return {
        "wavelengths": [center_nm - 20.0, center_nm - 10.0, center_nm, center_nm + 10.0, center_nm + 20.0],
        "responses": [0.0, 0.5, 1.0, 0.5, 0.0],
        "wavelength_unit": "nm",
    }


def _band(identifier: str, center_nm: float, *, target: bool) -> dict[str, Any]:
    band: dict[str, Any] = {
        "id": identifier,
        "center_wavelength": center_nm,
        "wavelength_unit": "nm",
        "fwhm": 20.0,
        "fwhm_unit": "nm",
        "srf": _srf(center_nm),
    }
    if target:
        band["neutral_value"] = 0.5
    return band


def varied_probes(n_probes: int = 20) -> np.ndarray:
    """Return bounded probes whose every band is changed by cyclic rotation."""
    rows = np.arange(n_probes, dtype=np.int64)[:, None]
    bands = np.arange(len(TARGET_BAND_IDS), dtype=np.int64)[None, :]
    values = 0.08 + 0.84 * (((rows * (bands * 2 + 3) + bands * 7) % 23) / 22.0)
    return np.ascontiguousarray(values, dtype=np.float64)


@dataclass(frozen=True)
class BundleCase:
    root: Path
    adapter_type: str

    @property
    def manifest_path(self) -> Path:
        return self.root / "bandtrace.yaml"

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def file_path(self, key: str) -> Path:
        return self.root / self.manifest()["files"][key]["path"]

    def read_json(self, key: str) -> dict[str, Any]:
        return json.loads(self.file_path(key).read_text(encoding="utf-8"))

    def rewrite_json(self, key: str, payload: dict[str, Any]) -> None:
        write_json(self.file_path(key), payload)
        self.refresh_hash(key)

    def refresh_hash(self, key: str) -> None:
        manifest = self.manifest()
        manifest["files"][key]["sha256"] = sha256_file(self.file_path(key))
        if key == "artifact":
            model = self.read_json("model")
            model["artifact_sha256"] = manifest["files"][key]["sha256"]
            write_json(self.file_path("model"), model)
            manifest["files"]["model"]["sha256"] = sha256_file(self.file_path("model"))
        write_json(self.manifest_path, manifest)

    def rewrite_manifest(self, payload: dict[str, Any]) -> None:
        write_json(self.manifest_path, payload)


class BundleFactory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.counter = 0

    def __call__(
        self,
        *,
        adapter: str = "numpy-linear-v1",
        subprocess_mode: str = "clean",
        subprocess_direct: bool = False,
        subprocess_runner: bool = False,
        subprocess_extra_executable_asset: bool = False,
        probes: np.ndarray | None = None,
        wavelength_conditioned: bool = False,
        fwhm_conditioned: bool = False,
        model_mutator: Callable[[dict[str, Any]], None] | None = None,
        sensor_mutator: Callable[[dict[str, Any]], None] | None = None,
        route_mutator: Callable[[dict[str, Any]], None] | None = None,
        artifact_mutator: Callable[[Path], None] | None = None,
        normalization_offset: np.ndarray | None = None,
        normalization_scale: np.ndarray | None = None,
    ) -> BundleCase:
        self.counter += 1
        bundle_root = self.root / f"bundle-{self.counter}"
        bundle_root.mkdir(mode=0o700)

        model_path = bundle_root / "model.json"
        sensor_path = bundle_root / "sensor.json"
        route_path = bundle_root / "route.json"
        probes_path = bundle_root / "probes.npz"
        artifact_path = bundle_root / (
            "reference_model.npz" if adapter == "numpy-linear-v1" else "adapter.py"
        )
        runner_path = bundle_root / "runner.py"
        asset_path = bundle_root / "executable-asset.dat"

        route = {
            "schema_version": "1.0",
            "model_channel_ids": MODEL_CHANNEL_IDS.copy(),
            "target_band_ids": TARGET_BAND_IDS.copy(),
            "matrix": ROUTE_MATRIX.tolist(),
            "operation": "selection_or_permutation",
            "spatial_operation": "none",
        }
        if route_mutator is not None:
            route_mutator(route)
        write_json(route_path, route)

        sensor = {
            "schema_version": "1.0",
            "sensor_id": "synthetic-sensor-v1",
            "sensor_model": "BandTrace Test Five",
            "sensor_serial": "TEST-0001",
            "target_bands": [
                _band(identifier, center, target=True)
                for identifier, center in zip(TARGET_BAND_IDS, TARGET_CENTERS_NM)
            ],
            "radiometric_quantity": "unitless_reflectance_factor",
            "valid_range": [0.0, 1.0],
            "calibration_state": "synthetic_exact",
            "preprocessing_version": "identity-v1",
        }
        if sensor_mutator is not None:
            sensor_mutator(sensor)
        write_json(sensor_path, sensor)

        probe_values = varied_probes() if probes is None else np.asarray(probes)
        probe_target_ids = [str(band["id"]) for band in sensor["target_bands"]]
        np.savez(
            probes_path,
            probes=probe_values,
            target_band_ids=np.asarray(probe_target_ids),
        )

        if adapter == "numpy-linear-v1":
            offset = (
                np.zeros(len(MODEL_CHANNEL_IDS), dtype=np.float64)
                if normalization_offset is None
                else np.asarray(normalization_offset, dtype=np.float64)
            )
            scale = (
                np.ones(len(MODEL_CHANNEL_IDS), dtype=np.float64)
                if normalization_scale is None
                else np.asarray(normalization_scale, dtype=np.float64)
            )
            np.savez(
                artifact_path,
                route_matrix=np.asarray(route["matrix"], dtype=np.float64),
                target_band_ids=np.asarray(route["target_band_ids"]),
                normalization_offset=offset,
                normalization_scale=scale,
                output_weights=OUTPUT_WEIGHTS,
                output_bias=np.asarray(0.125, dtype=np.float64),
                spatial_operation=np.asarray(route["spatial_operation"]),
            )
        elif adapter == "subprocess-npz-v1":
            if subprocess_direct and subprocess_runner:
                raise ValueError("direct artifact and argv0 runner modes are mutually exclusive")
            shutil.copyfile(FIXTURE_ADAPTER, artifact_path)
            if subprocess_direct:
                artifact_path.write_bytes(
                    b"#!/usr/bin/env python3\n" + artifact_path.read_bytes()
                )
                artifact_path.chmod(0o700)
            if subprocess_runner:
                runner_path.write_text(
                    "#!/usr/bin/env python3\n"
                    "import os, sys\n"
                    "os.environ['BANDTRACE_TEST_RUNNER_PATH'] = __file__\n"
                    "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n",
                    encoding="utf-8",
                )
                runner_path.chmod(0o700)
            if subprocess_extra_executable_asset:
                asset_path.write_bytes(b"fixture auxiliary bytes\n")
                asset_path.chmod(0o700)
        else:
            raise ValueError(f"unknown test adapter {adapter!r}")
        if artifact_mutator is not None:
            artifact_mutator(artifact_path)

        artifact_sha256 = sha256_file(artifact_path)
        model_offset = (
            np.zeros(len(MODEL_CHANNEL_IDS), dtype=np.float64)
            if normalization_offset is None
            else np.asarray(normalization_offset, dtype=np.float64)
        )
        model_scale = (
            np.ones(len(MODEL_CHANNEL_IDS), dtype=np.float64)
            if normalization_scale is None
            else np.asarray(normalization_scale, dtype=np.float64)
        )
        model = {
            "schema_version": "1.0",
            "model_id": "synthetic-linear-model",
            "model_version": "1.0-test",
            "artifact_sha256": artifact_sha256,
            "model_channels": [
                _band(identifier, center, target=False)
                for identifier, center in zip(MODEL_CHANNEL_IDS, MODEL_CENTERS_NM)
            ],
            "radiometric_quantity": "unitless_reflectance_factor",
            "valid_range": [0.0, 1.0],
            "normalization": {
                "type": "affine",
                "offset": model_offset.tolist(),
                "scale": model_scale.tolist(),
            },
            "declared_validated_support": {
                "supplier_assertion": True,
                "wavelength_range": [420.0, 780.0],
                "wavelength_unit": "nm",
            },
            "pre_decision_output": {"name": "score"},
            "wavelength_conditioned": wavelength_conditioned,
            "fwhm_conditioned": fwhm_conditioned,
            "required_dependence_target_band_ids": TARGET_BAND_IDS[:4],
        }
        if model_mutator is not None:
            model_mutator(model)
        write_json(model_path, model)

        file_paths = {
            "model": model_path,
            "sensor": sensor_path,
            "probes": probes_path,
            "route": route_path,
            "artifact": artifact_path,
        }
        if subprocess_runner:
            file_paths["runner"] = runner_path
        if subprocess_extra_executable_asset:
            file_paths["asset"] = asset_path
        files = {
            key: {"path": path.name, "sha256": sha256_file(path)}
            for key, path in file_paths.items()
        }
        adapter_manifest: dict[str, Any] = {"type": adapter}
        if adapter == "subprocess-npz-v1":
            executable_prefix = (
                ["{artifact}"]
                if subprocess_direct
                else ["{asset:runner}", "{artifact}"]
                if subprocess_runner
                else [sys.executable, "{artifact}"]
            )
            adapter_manifest["argv"] = [
                *executable_prefix,
                "--mode",
                subprocess_mode,
                "--input",
                "{input_npz}",
                "--output",
                "{output_npz}",
                "--offset",
                ",".join(format(value, ".17g") for value in model_offset),
                "--scale",
                ",".join(format(value, ".17g") for value in model_scale),
            ]
            if subprocess_extra_executable_asset:
                adapter_manifest["argv"].extend(["--asset", "{asset:asset}"])
        write_json(
            bundle_root / "bandtrace.yaml",
            {
                "schema_version": "1.0",
                "policy_id": "bandtrace-0.1-r29",
                "files": files,
                "adapter": adapter_manifest,
            },
        )
        return BundleCase(bundle_root, adapter)


@pytest.fixture
def bundle_factory(tmp_path: Path) -> BundleFactory:
    return BundleFactory(tmp_path)


@pytest.fixture
def clean_numpy_bundle(bundle_factory: BundleFactory) -> BundleCase:
    return bundle_factory(adapter="numpy-linear-v1")


@pytest.fixture
def clean_subprocess_bundle(bundle_factory: BundleFactory) -> BundleCase:
    return bundle_factory(adapter="subprocess-npz-v1")


def run_and_read(bundle: BundleCase, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    from bandtrace.audit import run_audit

    result = run_audit(bundle.root, output_dir)
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    return result, report
