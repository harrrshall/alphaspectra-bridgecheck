from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


PRODUCT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for browser examples")
def test_browser_examples_are_frozen_honest_and_network_free() -> None:
    completed = subprocess.run(
        ["node", str(PRODUCT_ROOT / "tests/browser_examples.mjs")],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    payload = json.loads(completed.stdout)
    assert payload["artifact_requests"] == 2
    assert payload["post_artifact_requests"] == 0

    expected = {
        "measured-cabo": (
            "state-0478",
            0.029233001099817284,
            "WITHIN_REFERENCE_Q95",
            "4e18ee7d0af3f8049ab3579acb5d30d319bb0e7a7ef4cb01a24d9b3a624171e4",
            3,
        ),
        "lower-reference": (
            "state-1191",
            0.0,
            "WITHIN_REFERENCE_Q95",
            "09884975e703773882ff75820867409cb762540be468f5a110aac6f4b18a1881",
            3,
        ),
        "median-reference": (
            "state-0083",
            0.0,
            "WITHIN_REFERENCE_Q95",
            "2798317892f02dfe4051ff923d1c3c7a2c0557943c03591a288bfbcad3c9cad1",
            3,
        ),
        "higher-reference": (
            "state-0470",
            0.0,
            "WITHIN_REFERENCE_Q95",
            "e500a344e413ea89bc68947aff6fe8933a525d0d545c1ba21d6b89f885fd0638",
            3,
        ),
        "support-warning": (
            "state-0387",
            0.03787556968078218,
            "REFERENCE_TAIL_Q95_Q99",
            "7c2ddfdec91bfe3ed7ad57eb43147716ca2c263634e1aba832fe6769728a6805",
            4,
        ),
    }
    expected_origins = {
        "measured-cabo": "measured_training_example",
        "lower-reference": "generated_bank_example_not_measured",
        "median-reference": "generated_bank_example_not_measured",
        "higher-reference": "generated_bank_example_not_measured",
        "support-warning": "constructed_support_test_not_measured",
    }
    assert {row["id"] for row in payload["results"]} == set(expected)
    for row in payload["results"]:
        nearest, rmse, tier, input_hash, warning_count = expected[row["id"]]
        assert row["wavelengths"] == 151
        assert row["observed_bands"] == 151
        assert row["derived_bands"] == 338
        assert row["derived_csv_rows"] == 338
        assert row["nearest_candidate"] == nearest
        assert row["context_fit_rmse"] == pytest.approx(rmse, rel=0.0, abs=1e-15)
        assert row["support_tier"] == tier
        assert row["input_sha256"] == input_hash
        assert row["input_origin"] == expected_origins[row["id"]]
        assert row["claim_status"] == "CANDIDATE_ONLY_UNVALIDATED"
        assert row["warning_count"] == warning_count
        assert "diagnos" not in row["provenance"].lower()


def test_examples_keep_generated_and_measured_provenance_explicit() -> None:
    source = (PRODUCT_ROOT / "src/bridgecheck/static/examples.js").read_text()
    assert "not independent validation" in source
    assert source.count("not measured") >= 3
    assert "not a biological or sensor example" in source
    for forbidden in ("healthy", "diseased", "drought", "wet leaf", "dry leaf"):
        assert forbidden not in source.lower()
