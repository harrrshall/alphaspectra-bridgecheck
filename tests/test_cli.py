from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bridgecheck.artifact import BridgeArtifact
from bridgecheck.audit import PairedSpectrum
from bridgecheck.cli import main


@pytest.fixture(autouse=True)
def use_synthetic_cli_artifact(monkeypatch, artifact: BridgeArtifact) -> None:
    """Exercise CLI semantics with the synthetic bank without weakening runtime authenticity."""
    monkeypatch.setattr(
        BridgeArtifact,
        "load",
        classmethod(lambda cls, model_dir=None, **kwargs: artifact),
    )


def _write_predict_csv(path: Path, artifact: BridgeArtifact, *, percent: bool = False) -> None:
    scale = 100.0 if percent else 1.0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wavelength_nm", "reflectance"])
        for wavelength, value in zip(
            artifact.wavelengths_nm[artifact.context_mask],
            artifact.bank[2, artifact.context_mask],
        ):
            writer.writerow([wavelength, value * scale])


def _write_audit_csv(path: Path, samples: list[PairedSpectrum]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "group_id", "band_origin", "wavelength_nm", "reflectance"])
        for sample in samples:
            for wavelength, value in zip(
                sample.context_wavelength_nm, sample.context_reflectance
            ):
                writer.writerow(
                    [sample.sample_id, sample.group_id, "measured_context", wavelength, value]
                )
            for wavelength, value in zip(sample.target_wavelength_nm, sample.target_reflectance):
                writer.writerow(
                    [sample.sample_id, sample.group_id, "measured_target", wavelength, value]
                )


def test_cli_predict_writes_typed_spectrum_and_report(
    artifact: BridgeArtifact, model_dir: Path, tmp_path: Path, capsys
) -> None:
    source = tmp_path / "context.csv"
    output = tmp_path / "prediction.csv"
    report_path = tmp_path / "prediction.json"
    _write_predict_csv(source, artifact)

    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--model-dir",
                str(model_dir),
                "predict",
                str(source),
                "--output",
                str(output),
                "--report",
                str(report_path),
                "--neighbors",
                "3",
            ]
        )

    assert exit_info.value.code == 0
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(rows) == 338
    assert {row["origin"] for row in rows} == {"model_derived"}
    assert {row["observed_band"] for row in rows} == {"false"}
    assert report["observed"]["origin"] == "measured"
    assert report["derived"]["origin"] == "model_derived"
    assert report["claim_status"] == "CANDIDATE_ONLY_UNVALIDATED"
    assert json.loads(capsys.readouterr().out)["status"] == "CANDIDATE_ONLY_UNVALIDATED"


def test_cli_audit_uses_nonzero_exit_for_fail_closed_result(
    model_dir: Path,
    tmp_path: Path,
    passing_samples: list[PairedSpectrum],
    failing_samples: list[PairedSpectrum],
) -> None:
    passing_csv = tmp_path / "passing.csv"
    failing_csv = tmp_path / "failing.csv"
    passing_report = tmp_path / "passing.json"
    failing_report = tmp_path / "failing.json"
    _write_audit_csv(passing_csv, passing_samples)
    _write_audit_csv(failing_csv, failing_samples)

    with pytest.raises(SystemExit) as pass_exit:
        main(
            [
                "--model-dir",
                str(model_dir),
                "audit",
                str(passing_csv),
                "--report",
                str(passing_report),
                "--bootstrap-repeats",
                "100",
            ]
        )
    with pytest.raises(SystemExit) as fail_exit:
        main(
            [
                "--model-dir",
                str(model_dir),
                "audit",
                str(failing_csv),
                "--report",
                str(failing_report),
                "--bootstrap-repeats",
                "100",
            ]
        )

    assert pass_exit.value.code == 0
    assert fail_exit.value.code == 2
    assert json.loads(passing_report.read_text(encoding="utf-8"))["status"] == (
        "SUPPORTED_FOR_RECONSTRUCTION_RESEARCH"
    )
    assert json.loads(failing_report.read_text(encoding="utf-8"))["status"] == "NOT_SUPPORTED"


def test_cli_rejects_percent_scaled_input(
    artifact: BridgeArtifact, model_dir: Path, tmp_path: Path, capsys
) -> None:
    source = tmp_path / "percent.csv"
    _write_predict_csv(source, artifact, percent=True)

    with pytest.raises(SystemExit) as exit_info:
        main(["--model-dir", str(model_dir), "predict", str(source)])

    assert exit_info.value.code == 2
    assert "percent-scaled" in capsys.readouterr().err


def test_cli_info_smoke(model_dir: Path, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--model-dir", str(model_dir), "info"])

    assert exit_info.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_states"] == 4
    assert "bank" not in payload
