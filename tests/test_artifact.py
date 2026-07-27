from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from bridgecheck.artifact import BridgeArtifact, MODEL_ID, array_sha256


FROZEN_ARRAY_SHA256 = "1d64d6a0ec1dec48c2fa0c9c33c4cfcef832e714ecf88aaee55941a0d5fc2ac0"


def test_load_verifies_bank_and_makes_it_immutable(
    model_dir: Path, synthetic_bank: np.ndarray
) -> None:
    artifact = BridgeArtifact.load(model_dir, verify_official=False)

    np.testing.assert_array_equal(artifact.bank, synthetic_bank)
    assert artifact.bank.flags.c_contiguous
    assert not artifact.bank.flags.writeable
    assert artifact.wavelengths_nm.shape == (501,)
    assert int(artifact.context_mask.sum()) == 151
    assert int(artifact.target_mask.sum()) == 338
    with pytest.raises(ValueError):
        artifact.bank[0, 0] = 0.0


def test_public_info_exposes_metadata_but_not_candidate_values(artifact: BridgeArtifact) -> None:
    info = artifact.public_info()

    assert info["model_id"] == MODEL_ID
    assert info["candidate_states"] == 4
    assert info["spectral_grid"]["target_count"] == 338
    serialized = json.dumps(info)
    assert "bridge_v1.f64" not in serialized
    assert "test_fixture" in serialized
    assert "bank" not in info


def test_bank_byte_tamper_is_rejected(model_dir: Path) -> None:
    bank_path = model_dir / "bridge_v1.f64"
    payload = bytearray(bank_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    bank_path.write_bytes(payload)

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        BridgeArtifact.load(model_dir, verify_official=False)


def test_truncated_bank_is_rejected(model_dir: Path) -> None:
    bank_path = model_dir / "bridge_v1.f64"
    bank_path.write_bytes(bank_path.read_bytes()[:-8])

    with pytest.raises(ValueError, match="byte count mismatch"):
        BridgeArtifact.load(model_dir, verify_official=False)


def test_array_hash_tamper_is_rejected(model_dir: Path) -> None:
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"]["array_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="array SHA-256 mismatch"):
        BridgeArtifact.load(model_dir, verify_official=False)


@pytest.mark.parametrize(
    ("key", "value"),
    [("schema_version", "2.0"), ("model_id", "another-model")],
)
def test_unexpected_manifest_identity_is_rejected(model_dir: Path, key: str, value: str) -> None:
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported or unexpected"):
        BridgeArtifact.load(model_dir, verify_official=False)


def test_spectral_count_tamper_is_rejected(model_dir: Path) -> None:
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["spectral_grid"]["target_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="spectral counts"):
        BridgeArtifact.load(model_dir, verify_official=False)


def test_bank_width_must_equal_spectral_grid_count(model_factory, synthetic_bank: np.ndarray) -> None:
    malformed = synthetic_bank[:, :-1]
    malformed_dir = model_factory(bank=malformed)

    with pytest.raises(ValueError, match="bank.*spectral|spectral.*bank"):
        BridgeArtifact.load(malformed_dir, verify_official=False)


def test_bank_must_be_a_two_dimensional_state_by_band_matrix(
    model_factory, synthetic_bank: np.ndarray
) -> None:
    malformed = synthetic_bank[np.newaxis, ...]
    malformed_dir = model_factory(bank=malformed)

    with pytest.raises(ValueError, match="two-dimensional|rank"):
        BridgeArtifact.load(malformed_dir, verify_official=False)


def test_self_consistent_nonofficial_bundle_is_rejected_by_default(model_dir: Path) -> None:
    with pytest.raises(ValueError, match="independently pinned official"):
        BridgeArtifact.load(model_dir)


def test_official_manifest_metadata_tamper_is_rejected(tmp_path: Path) -> None:
    bundled = Path(__file__).resolve().parents[1] / "src" / "bridgecheck" / "model"
    copied = tmp_path / "official-copy"
    shutil.copytree(bundled, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["claim_ceiling"]["output_status"] = "MEASURED"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="pinned official manifest"):
        BridgeArtifact.load(copied)


def test_bundled_release_artifact_matches_frozen_contract() -> None:
    model_dir = Path(__file__).resolve().parents[1] / "src" / "bridgecheck" / "model"
    assert (model_dir / "manifest.json").is_file(), "the release model artifact has not been exported"

    artifact = BridgeArtifact.load(model_dir)
    assert artifact.bank.shape == (1213, 501)
    assert artifact.bank.dtype == np.dtype("float64")
    assert array_sha256(artifact.bank) == FROZEN_ARRAY_SHA256
    assert artifact.manifest["artifact"]["array_sha256"] == FROZEN_ARRAY_SHA256


def test_default_package_resource_loads_the_frozen_release_artifact() -> None:
    artifact = BridgeArtifact.load()

    assert artifact.bank.shape == (1213, 501)
    assert artifact.manifest["artifact"]["array_sha256"] == FROZEN_ARRAY_SHA256
