"""Immutable BridgeCheck model-artifact loading and verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np


MODEL_ID = "alphaspectra-bridge-p1-20260727"
OFFICIAL_ARRAY_SHA256 = "1d64d6a0ec1dec48c2fa0c9c33c4cfcef832e714ecf88aaee55941a0d5fc2ac0"
OFFICIAL_FILE_SHA256 = "aa9700558836fb7730ab650dd0eaaf921038dce65a25aba9d9efc8d451d0d83f"
OFFICIAL_MANIFEST_SHA256 = "da3d63a6f535219a618f6b2a118693f59f2aa001a75480372a163a480ab4d2bd"


def array_sha256(value: np.ndarray) -> str:
    """Hash dtype, shape and bytes using the frozen AlphaSpectra convention."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BridgeArtifact:
    """Verified, immutable candidate bank and its public manifest."""

    manifest: dict[str, Any]
    bank: np.ndarray

    @property
    def wavelengths_nm(self) -> np.ndarray:
        spectral = self.manifest["spectral_grid"]
        return np.arange(
            float(spectral["start_nm"]),
            float(spectral["end_nm"]) + float(spectral["step_nm"]) / 2.0,
            float(spectral["step_nm"]),
            dtype=np.float64,
        )

    @property
    def context_mask(self) -> np.ndarray:
        lo, hi = self.manifest["spectral_grid"]["context_range_nm"]
        wavelength = self.wavelengths_nm
        return (wavelength >= float(lo)) & (wavelength <= float(hi))

    @property
    def target_mask(self) -> np.ndarray:
        spectral = self.manifest["spectral_grid"]
        wavelength = self.wavelengths_nm
        return (wavelength > float(spectral["target_above_nm"])) & (
            wavelength <= float(spectral["end_nm"])
        )

    @classmethod
    def load(
        cls,
        model_dir: Path | str | None = None,
        *,
        verify_official: bool = True,
    ) -> "BridgeArtifact":
        if model_dir is None:
            model_root = resources.files("bridgecheck").joinpath("model")
            manifest_bytes = model_root.joinpath("manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            bank_bytes = model_root.joinpath(manifest["artifact"]["filename"]).read_bytes()
        else:
            root = Path(model_dir)
            if root.is_file() and root.name == "manifest.json":
                manifest_path = root
                root = root.parent
            else:
                manifest_path = root / "manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            bank_bytes = (root / manifest["artifact"]["filename"]).read_bytes()

        if manifest.get("schema_version") != "1.0" or manifest.get("model_id") != MODEL_ID:
            raise ValueError("unsupported or unexpected BridgeCheck model manifest")
        if verify_official and hashlib.sha256(manifest_bytes).hexdigest() != OFFICIAL_MANIFEST_SHA256:
            raise ValueError("model manifest does not match the independently pinned official manifest")
        artifact = manifest["artifact"]
        if verify_official and (
            artifact.get("array_sha256") != OFFICIAL_ARRAY_SHA256
            or artifact.get("file_sha256") != OFFICIAL_FILE_SHA256
        ):
            raise ValueError("model bundle does not match the independently pinned official BridgeCheck artifact")
        if artifact.get("dtype") != "<f8" or artifact.get("order") != "C":
            raise ValueError("BridgeCheck V1 requires little-endian float64 C-order bank")
        shape = artifact.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError("candidate bank must be a two-dimensional state-by-band matrix")
        spectral = manifest["spectral_grid"]
        if shape[1] != int(spectral["count"]):
            raise ValueError("candidate bank width does not match the manifest spectral grid")
        expected_bytes = int(np.prod(shape)) * np.dtype("<f8").itemsize
        if "bytes" in artifact and int(artifact["bytes"]) != expected_bytes:
            raise ValueError("candidate bank manifest byte count does not match its shape")
        if len(bank_bytes) != expected_bytes:
            raise ValueError(
                f"candidate bank byte count mismatch: expected {expected_bytes}, got {len(bank_bytes)}"
            )
        observed_file_hash = hashlib.sha256(bank_bytes).hexdigest()
        if observed_file_hash != artifact["file_sha256"]:
            raise ValueError("candidate bank file SHA-256 mismatch")
        bank = np.frombuffer(bank_bytes, dtype="<f8").reshape(tuple(shape), order="C")
        if array_sha256(bank) != artifact["array_sha256"]:
            raise ValueError("candidate bank array SHA-256 mismatch")
        bank.setflags(write=False)
        instance = cls(manifest=manifest, bank=bank)
        if (
            len(instance.wavelengths_nm) != int(spectral["count"])
            or int(instance.context_mask.sum()) != int(spectral["context_count"])
            or int(instance.target_mask.sum()) != int(spectral["target_count"])
        ):
            raise ValueError("manifest spectral counts do not reproduce")
        return instance

    def public_info(self) -> dict[str, Any]:
        """Return only public, non-bank metadata suitable for API responses."""
        return {
            "model_id": self.manifest["model_id"],
            "version": self.manifest["version"],
            "artifact_array_sha256": self.manifest["artifact"]["array_sha256"],
            "candidate_states": int(self.bank.shape[0]),
            "spectral_grid": self.manifest["spectral_grid"],
            "input_contract": self.manifest["input_contract"],
            "claim_ceiling": self.manifest["claim_ceiling"],
            "evidence": self.manifest["evidence"],
            "licenses": self.manifest["licenses"],
        }
