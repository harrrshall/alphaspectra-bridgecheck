"""Command-line interface for BridgeCheck."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import BridgeArtifact
from .audit import PairedSpectrum, audit_paired_spectra
from .predict import ContractError, predict_spectrum


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_predict_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"wavelength_nm", "reflectance"}.issubset(rows[0]):
        raise ContractError("prediction CSV requires wavelength_nm and reflectance columns")
    return (
        np.asarray([float(row["wavelength_nm"]) for row in rows]),
        np.asarray([float(row["reflectance"]) for row in rows]),
    )


def _read_audit_csv(path: Path) -> list[PairedSpectrum]:
    required = {"sample_id", "group_id", "band_origin", "wavelength_nm", "reflectance"}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ContractError(f"audit CSV requires columns: {', '.join(sorted(required))}")
        grouped: dict[str, dict[str, Any]] = {}
        for row in reader:
            sample_id = row["sample_id"].strip()
            group_id = row["group_id"].strip()
            origin = row["band_origin"].strip()
            if origin not in {"measured_context", "measured_target"}:
                raise ContractError("band_origin must be measured_context or measured_target")
            record = grouped.setdefault(
                sample_id,
                {"group_id": group_id, "measured_context": [], "measured_target": []},
            )
            if record["group_id"] != group_id:
                raise ContractError(f"sample {sample_id} maps to multiple group_id values")
            record[origin].append((float(row["wavelength_nm"]), float(row["reflectance"])))
    samples = []
    for sample_id in sorted(grouped):
        record = grouped[sample_id]
        context = sorted(record["measured_context"])
        target = sorted(record["measured_target"])
        samples.append(
            PairedSpectrum(
                sample_id=sample_id,
                group_id=record["group_id"],
                context_wavelength_nm=np.asarray([row[0] for row in context]),
                context_reflectance=np.asarray([row[1] for row in context]),
                target_wavelength_nm=np.asarray([row[0] for row in target]),
                target_reflectance=np.asarray([row[1] for row in target]),
            )
        )
    return samples


def command_predict(args: argparse.Namespace) -> int:
    artifact = BridgeArtifact.load(args.model_dir)
    wavelength, reflectance = _read_predict_csv(args.input)
    result = predict_spectrum(artifact, wavelength, reflectance, neighbors=args.neighbors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wavelength_nm", "reflectance", "origin", "observed_band"])
        for wl, value in zip(result.wavelengths_nm, result.reflectance):
            writer.writerow([f"{wl:.10g}", f"{value:.17g}", "model_derived", "false"])
    report = result.to_dict()
    report["observed"] = {
        "origin": "measured",
        "wavelength_nm": wavelength.tolist(),
        "reflectance": reflectance.tolist(),
        "observed_band_mask": [True] * len(wavelength),
    }
    _write_json(args.report, report)
    print(json.dumps({"status": report["claim_status"], "output": str(args.output), "report": str(args.report)}))
    return 0


def command_audit(args: argparse.Namespace) -> int:
    artifact = BridgeArtifact.load(args.model_dir)
    report = audit_paired_spectra(
        artifact,
        _read_audit_csv(args.input),
        bootstrap_repeats=args.bootstrap_repeats,
    )
    _write_json(args.report, report)
    print(json.dumps({"status": report["status"], "report": str(args.report)}))
    return 0 if report["status"] == "SUPPORTED_FOR_RECONSTRUCTION_RESEARCH" else 2


def command_info(args: argparse.Namespace) -> int:
    print(json.dumps(BridgeArtifact.load(args.model_dir).public_info(), indent=2, sort_keys=True))
    return 0


def command_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("install the 'api' extra to run the server") from error
    uvicorn.run("bridgecheck.api:app", host=args.host, port=args.port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bridgecheck",
        description="Physics-grounded candidate SWIR and paired-data feasibility auditing.",
    )
    parser.add_argument("--model-dir", type=Path, default=None, help="verified model directory")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="generate candidate SWIR from a VNIR CSV")
    predict.add_argument("input", type=Path)
    predict.add_argument("--output", type=Path, default=Path("prediction.csv"))
    predict.add_argument("--report", type=Path, default=Path("prediction_report.json"))
    predict.add_argument("--neighbors", type=int, default=5)
    predict.set_defaults(func=command_predict)

    audit = sub.add_parser("audit", help="audit paired same-sample VNIR/SWIR long-form CSV")
    audit.add_argument("input", type=Path)
    audit.add_argument("--report", type=Path, default=Path("audit_report.json"))
    audit.add_argument("--bootstrap-repeats", type=int, default=10_000)
    audit.set_defaults(func=command_audit)

    info = sub.add_parser("info", help="show verified public model metadata")
    info.set_defaults(func=command_info)

    serve = sub.add_parser("serve", help="run the optional FastAPI service and browser UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.set_defaults(func=command_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        raise SystemExit(args.func(args))
    except ContractError as error:
        parser.error(str(error))
