from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from bridgecheck.artifact import BridgeArtifact
from bridgecheck.predict import predict_spectrum


PRODUCT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser parity")
def test_browser_and_python_predictions_match() -> None:
    request = json.loads((PRODUCT_ROOT / "examples/predict_request.json").read_text())
    artifact = BridgeArtifact.load(PRODUCT_ROOT / "src/bridgecheck/model")
    python_result = predict_spectrum(
        artifact,
        request["wavelength_nm"],
        request["reflectance"],
        neighbors=request["neighbors"],
    ).to_dict()
    completed = subprocess.run(
        ["node", str(PRODUCT_ROOT / "tests/browser_parity.mjs")],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    browser_result = json.loads(completed.stdout)

    assert browser_result["model_id"] == python_result["model_id"]
    assert browser_result["input_sha256"] == python_result["input_sha256"]
    assert browser_result["claim_status"] == python_result["claim_status"]
    assert browser_result["retrieval"]["nearest_candidate"] == python_result["retrieval"]["nearest_candidate"]
    assert browser_result["retrieval"]["support_tier"] == python_result["retrieval"]["support_tier"]
    assert browser_result["retrieval"]["context_fit_rmse"] == pytest.approx(
        python_result["retrieval"]["context_fit_rmse"], rel=0.0, abs=1e-15
    )
    for key in ("reflectance",):
        np.testing.assert_allclose(
            browser_result["derived"][key], python_result["derived"][key], rtol=0.0, atol=1e-15
        )
    for key in ("minimum", "maximum"):
        np.testing.assert_allclose(
            browser_result["derived"]["neighbor_envelope"][key],
            python_result["derived"]["neighbor_envelope"][key],
            rtol=0.0,
            atol=1e-15,
        )
    assert browser_result["derived"]["origin"] == "model_derived"
    assert not any(browser_result["derived"]["observed_band_mask"])
