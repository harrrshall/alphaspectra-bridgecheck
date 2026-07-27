from __future__ import annotations

import importlib
import sys

import pytest

from bridgecheck.artifact import BridgeArtifact
from bridgecheck.audit import PairedSpectrum

from conftest import sample_to_api


@pytest.fixture
def client(artifact: BridgeArtifact, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        BridgeArtifact,
        "load",
        classmethod(lambda cls, model_dir=None: artifact),
    )
    sys.modules.pop("bridgecheck.api", None)
    api = importlib.import_module("bridgecheck.api")
    return TestClient(api.create_app(artifact), raise_server_exceptions=False)


def test_health_and_public_model_metadata(client, artifact: BridgeArtifact) -> None:
    health = client.get("/healthz")
    model = client.get("/v1/model")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["artifact_array_sha256"] == artifact.manifest["artifact"]["array_sha256"]
    assert model.status_code == 200
    assert model.json()["candidate_states"] == 4
    assert "bank" not in model.json()


def test_predict_api_keeps_measured_and_model_derived_arrays_separate(
    client, artifact: BridgeArtifact
) -> None:
    wavelength = artifact.wavelengths_nm[artifact.context_mask]
    context = artifact.bank[2, artifact.context_mask]
    response = client.post(
        "/v1/predict",
        json={
            "wavelength_nm": wavelength.tolist(),
            "reflectance": context.tolist(),
            "reflectance_unit": "fraction",
            "neighbors": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["claim_status"] == "CANDIDATE_ONLY_UNVALIDATED"
    assert payload["observed"]["origin"] == "measured"
    assert payload["observed"]["observed_band_mask"] == [True] * 151
    assert payload["derived"]["origin"] == "model_derived"
    assert payload["derived"]["observed_band_mask"] == [False] * 338
    assert payload["observed"]["wavelength_nm"][-1] == 1000.0
    assert payload["derived"]["wavelength_nm"][0] == 1052.0


def test_predict_api_rejects_unit_contract_and_extra_fields(client, artifact: BridgeArtifact) -> None:
    wavelength = artifact.wavelengths_nm[artifact.context_mask].tolist()
    context = artifact.bank[0, artifact.context_mask].tolist()
    request = {"wavelength_nm": wavelength, "reflectance": context}

    wrong_unit = client.post("/v1/predict", json={**request, "reflectance_unit": "percent"})
    extra = client.post("/v1/predict", json={**request, "target_reflectance": [0.1]})
    bad_value = context.copy()
    bad_value[0] = 1.1
    out_of_range = client.post(
        "/v1/predict", json={"wavelength_nm": wavelength, "reflectance": bad_value}
    )

    assert wrong_unit.status_code == 422
    assert wrong_unit.json()["detail"] == "reflectance_unit must be 'fraction'"
    assert extra.status_code == 422
    assert out_of_range.status_code == 422
    assert "without clipping" in out_of_range.json()["detail"]


def test_audit_api_smoke_passes_frozen_checks(
    client, passing_samples: list[PairedSpectrum]
) -> None:
    response = client.post(
        "/v1/audit",
        json={
            "samples": [sample_to_api(sample) for sample in passing_samples],
            "bootstrap_repeats": 100,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "SUPPORTED_FOR_RECONSTRUCTION_RESEARCH"
    assert all(response.json()["checks"].values())


def test_api_body_limit_returns_413(client) -> None:
    response = client.post(
        "/v1/predict",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": "8000001"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "request body exceeds 8 MB"


@pytest.mark.parametrize("content_length", ["not-a-number", "-1"])
def test_api_rejects_malformed_content_length(client, content_length: str) -> None:
    response = client.post(
        "/v1/predict",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": content_length},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid Content-Length"
