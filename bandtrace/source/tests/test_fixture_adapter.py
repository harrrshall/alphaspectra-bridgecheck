from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np


ADAPTER = Path(__file__).with_name("fixture_adapter.py")


def _run_fixture_adapter(
    tmp_path: Path,
    mode: str,
    *,
    probes: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    input_path = tmp_path / f"{mode}-input.npz"
    output_path = tmp_path / f"{mode}-output.npz"
    values = (
        np.asarray(probes, dtype=np.float64)
        if probes is not None
        else np.linspace(0.05, 0.95, 100, dtype=np.float64).reshape(20, 5)
    )
    np.savez(
        input_path,
        probes=values,
        target_band_ids=np.asarray(["b0", "b1", "b2", "b3", "b4"]),
        wavelength_nm=np.asarray([450.0, 550.0, 650.0, 750.0, 950.0]),
        fwhm_nm=np.full(5, 20.0),
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(ADAPTER),
            "--mode",
            mode,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output_path, allow_pickle=False) as archive:
        return np.asarray(archive["output"]), np.asarray(archive["pre_core"])


def test_clean_subprocess_fixture_has_an_independent_exact_tap(tmp_path: Path) -> None:
    probes = np.linspace(0.05, 0.95, 100, dtype=np.float64).reshape(20, 5)
    output, tap = _run_fixture_adapter(tmp_path, "clean", probes=probes)

    np.testing.assert_array_equal(tap, probes[:, :4])
    np.testing.assert_allclose(
        output,
        probes[:, :4] @ np.asarray([0.7, -0.5, 0.35, 0.9]) + 0.125,
        rtol=0.0,
        atol=1e-15,
    )


def test_decoy_fixture_proves_supplier_tap_limit(tmp_path: Path) -> None:
    probes = np.linspace(0.05, 0.95, 100, dtype=np.float64).reshape(20, 5)
    clean_output, clean_tap = _run_fixture_adapter(tmp_path, "clean", probes=probes)
    decoy_output, decoy_tap = _run_fixture_adapter(tmp_path, "decoy_hidden_use", probes=probes)

    np.testing.assert_array_equal(decoy_tap, clean_tap)
    assert np.max(np.abs(decoy_output - clean_output)) > 0.1
    np.testing.assert_allclose(
        decoy_output - clean_output,
        0.8 * probes[:, 4],
        rtol=0.0,
        atol=1e-15,
    )
