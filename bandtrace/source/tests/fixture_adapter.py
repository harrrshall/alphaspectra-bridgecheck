"""Deliberately tiny subprocess adapter used only by the planted-fault tests.

The executable is intentionally outside the BandTrace package.  That keeps the release
instrument honest: subprocess protocol tests exercise user code rather than importing a private
shortcut from the implementation under test.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import zipfile

import numpy as np


def _required_array(archive: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in archive.files:
            return np.asarray(archive[name])
    raise KeyError(f"none of the required arrays are present: {names!r}")


def _spectral_values(archive: np.lib.npyio.NpzFile, mode: str) -> np.ndarray:
    values = _required_array(archive, "values", "probes", "target_values", "spectra")
    if values.ndim == 4:
        if mode == "first_pixel_reduction":
            values = values[:, :, 0, 0]
        elif mode == "max_reduction":
            values = np.max(values, axis=(2, 3))
        elif mode == "midrange_reduction":
            values = 0.5 * (
                np.min(values, axis=(2, 3)) + np.max(values, axis=(2, 3))
            )
        elif mode == "median_reduction":
            values = np.median(values, axis=(2, 3))
        elif mode == "cropped_mean_reduction":
            crop_height = max(1, values.shape[2] - 1)
            crop_width = max(1, values.shape[3] - 1)
            values = np.mean(
                values[:, :, :crop_height, :crop_width],
                axis=(2, 3),
                dtype=np.float64,
            )
        else:
            values = np.mean(values, axis=(2, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError(f"expected [N,5] or [N,5,H,W] values, got {values.shape}")
    return np.asarray(values, dtype=np.float64)


def _wavelengths(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    values = _required_array(
        archive,
        "wavelength_nm",
        "wavelengths_nm",
        "target_wavelength_nm",
        "target_wavelengths_nm",
    )
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _fwhm(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    values = _required_array(archive, "fwhm_nm")
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _default_probe_values() -> np.ndarray:
    rows = np.arange(20, dtype=np.int64)[:, None]
    bands = np.arange(5, dtype=np.int64)[None, :]
    return np.asarray(
        0.08 + 0.84 * (((rows * (bands * 2 + 3) + bands * 7) % 23) / 22.0),
        dtype=np.float64,
    )


def _write_npz(path: Path, *, output: np.ndarray, pre_core: np.ndarray) -> None:
    # np.savez itself is adequate here.  BandTrace, not this untrusted adapter, is responsible for
    # canonicalising the accepted result into deterministic release outputs.
    np.savez(path, output=output, pre_core=pre_core)


def _write_npz_with_compression(
    path: Path,
    *,
    output: np.ndarray,
    pre_core: np.ndarray,
    compression: int,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, array in (("output", output), ("pre_core", pre_core)):
            member = io.BytesIO()
            np.save(member, np.asarray(array, dtype=np.float64), allow_pickle=False)
            archive.writestr(f"{name}.npy", member.getvalue())


def _next_invocation_index(label: str) -> int:
    counter_path = Path.cwd() / f".bandtrace-{label}-count"
    try:
        invocation_index = int(counter_path.read_text(encoding="ascii"))
    except FileNotFoundError:
        invocation_index = 0
    counter_path.write_text(str(invocation_index + 1), encoding="ascii")
    return invocation_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offset", default="0,0,0,0")
    parser.add_argument("--scale", default="1,1,1,1")
    parser.add_argument("--asset", type=Path)
    args = parser.parse_args(argv)
    offset = np.asarray([float(value) for value in args.offset.split(",")], dtype=np.float64)
    scale = np.asarray([float(value) for value in args.scale.split(",")], dtype=np.float64)
    if offset.shape != (4,) or scale.shape != (4,):
        raise ValueError("fixture adapter normalization must contain exactly four channels")

    if args.mode == "execution_failure":
        return 17
    if args.mode == "nonzero_with_child":
        marker = Path(
            os.environ.get(
                "BANDTRACE_TEST_MARKER",
                str(args.output.with_suffix(".child-survived")),
            )
        )
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import pathlib,time,sys; time.sleep(2); pathlib.Path(sys.argv[1]).write_text('alive')",
                str(marker),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return 23
    if args.mode == "success_with_child":
        marker = Path(
            os.environ.get(
                "BANDTRACE_TEST_MARKER",
                str(args.output.with_suffix(".child-survived")),
            )
        )
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import pathlib,time,sys; time.sleep(2); pathlib.Path(sys.argv[1]).write_text('alive')",
                str(marker),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if args.mode == "stdout_flood":
        os.write(sys.stdout.fileno(), b"x" * 1_100_000)
        return 19
    if args.mode == "stdout_infinite":
        block = b"x" * 65_536
        while True:
            os.write(sys.stdout.fileno(), block)
    if args.mode == "timeout_with_child":
        marker = Path(
            os.environ.get(
                "BANDTRACE_TEST_MARKER",
                str(args.output.with_suffix(".child-survived")),
            )
        )
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import pathlib,time,sys; time.sleep(2); pathlib.Path(sys.argv[1]).write_text('alive')",
                str(marker),
            ]
        )
        time.sleep(10)
        return 0
    if args.mode == "parent_exits_child_holds_pipes":
        marker = Path(
            os.environ.get(
                "BANDTRACE_TEST_MARKER",
                str(args.output.with_suffix(".child-survived")),
            )
        )
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import pathlib,time,sys; time.sleep(2); pathlib.Path(sys.argv[1]).write_text('alive')",
                str(marker),
            ]
        )
        # The child inherits stdout/stderr, so the adapter parent exits while BandTrace still has
        # open pipe writers in the same process group.
        return 0
    if args.mode == "oversized_output":
        # A sparse file exercises the stat-before-deserialisation size guard without consuming
        # hundreds of MiB of disk or memory.
        with args.output.open("wb") as stream:
            stream.truncate(268_435_457)
        return 0
    if args.mode == "assert_staged_modes":
        artifact_mode = Path(__file__).stat().st_mode & 0o777
        if artifact_mode != 0o700:
            raise ValueError(f"staged artifact mode is {artifact_mode:#o}, expected 0o700")
        runner = os.environ.get("BANDTRACE_TEST_RUNNER_PATH")
        if runner is not None:
            runner_mode = Path(runner).stat().st_mode & 0o777
            if runner_mode != 0o700:
                raise ValueError(f"staged argv0 runner mode is {runner_mode:#o}, expected 0o700")
        if args.asset is not None:
            asset_mode = args.asset.stat().st_mode & 0o777
            if asset_mode != 0o600:
                raise ValueError(f"staged non-runner asset mode is {asset_mode:#o}, expected 0o600")
    if args.mode == "scan_isolation":
        allowed = {
            args.input.resolve(),
            args.output.resolve(),
            Path(__file__).resolve(),
        }
        leaked: list[str] = []
        for candidate in Path.cwd().rglob("*"):
            if not candidate.is_file() or candidate.suffix == ".pyc":
                continue
            relative = candidate.relative_to(Path.cwd())
            if relative.parts and relative.parts[0].startswith("invoke-"):
                continue
            if candidate.resolve() not in allowed:
                leaked.append(str(relative))
        if leaked:
            raise ValueError(
                "subprocess staging exposed undeclared contract/probe files: "
                f"{sorted(leaked)}"
            )

    with np.load(args.input, allow_pickle=False) as archive:
        exact_input_members = {"probes", "target_band_ids", "wavelength_nm", "fwhm_nm"}
        if set(archive.files) != exact_input_members:
            raise ValueError(
                "adapter input members are not the frozen exact set: "
                f"{sorted(archive.files)}"
            )
        forbidden = {
            "route_matrix",
            "normalization_offset",
            "normalization_scale",
            "expected_pre_core",
            "expected_tap",
        }
        leaked = forbidden.intersection(archive.files)
        if leaked:
            raise ValueError(f"BandTrace leaked declared answers to adapter: {sorted(leaked)}")
        values = _spectral_values(archive, args.mode)
        wavelength_nm = _wavelengths(archive)
        fwhm_nm = _fwhm(archive)
        raw_ids = _required_array(archive, "target_band_ids").reshape(-1).tolist()
        target_ids = [
            value.decode("ascii") if isinstance(value, bytes) else str(value)
            for value in raw_ids
        ]
    canonical_ids = ["t450", "t550", "t650", "t750", "t950"]
    if args.mode == "id_ignored_metadata_sorted":
        positions = np.argsort(wavelength_nm, kind="stable").tolist()
        values = values[:, positions]
        wavelength_nm = wavelength_nm[positions]
        fwhm_nm = fwhm_nm[positions]
    elif args.mode != "positional_only" and set(target_ids) == set(canonical_ids):
        positions = [target_ids.index(identifier) for identifier in canonical_ids]
        values = values[:, positions]
        wavelength_nm = wavelength_nm[positions]
        fwhm_nm = fwhm_nm[positions]

    # The clean executable selects the first four target bands in order.  This is independent of
    # the declaration in the bundle and is therefore suitable for a basis-challenge route test.
    pre_core = values[:, :4].copy()
    if args.mode == "dropped_band":
        # Preserve the declared neutral tap while removing the channel derivative.
        pre_core[:, 2] = 0.5
    elif args.mode == "reordered_bands":
        pre_core = pre_core[:, [1, 0, 2, 3]]
    elif args.mode == "hidden_resampling":
        pre_core[:, 2] = 0.5 * values[:, 2] + 0.5 * values[:, 3]
    elif args.mode == "edge_clamp":
        pre_core[:, 3] = 0.5 * values[:, 3] + 0.5 * values[:, 4]
    elif args.mode == "wrong_normalization":
        offset = offset + 0.1
    pre_core = (pre_core - offset[None, :]) / scale[None, :]
    if (
        args.mode == "c2_context_dependent_tap"
        and values.shape == (20, 5)
        and not np.array_equal(values, _default_probe_values())
        and not np.all(values == 0.5)
    ):
        pre_core[:, 0] += 0.01
    if args.mode == "c3_context_dependent_tap" and (
        not np.array_equal(
            wavelength_nm,
            np.asarray([450.0, 550.0, 650.0, 750.0, 950.0], dtype=np.float64),
        )
        or not np.array_equal(fwhm_nm, np.full(5, 20.0, dtype=np.float64))
    ):
        pre_core[:, 0] += 0.01
    if args.mode == "wrong_neutral_tap" and np.array_equal(
        values, np.full(values.shape, 0.5, dtype=np.float64)
    ):
        pre_core[:, 0] += 0.1

    coefficients = np.asarray([0.7, -0.5, 0.35, 0.9], dtype=np.float64)
    output = pre_core @ coefficients + 0.125

    if args.mode == "tap_replay_range_straddles_first":
        invocation_index = _next_invocation_index("tap-replay")
        if invocation_index in {1, 2}:
            tap_scale = max(1.0, float(np.percentile(np.abs(pre_core), 99)))
            direction = 1.0 if invocation_index == 1 else -1.0
            pre_core = pre_core + direction * 0.75e-7 * tap_scale
    elif args.mode == "third_replay_tap_mismatch":
        if _next_invocation_index("third-tap") == 2:
            pre_core = pre_core + 5e-6
    elif args.mode == "decoy_hidden_use":
        # A trusted subprocess can lie about its tap.  The declared/tapped route is pristine while
        # the selected output secretly depends on the fifth (outside-support) target band.  The
        # instrument should detect undeclared probe-local dependence, but must not claim it has
        # independently proven where the tap sits inside user code.
        clean_pre_core = (values[:, :4] - offset[None, :]) / scale[None, :]
        output = clean_pre_core @ coefficients + 0.8 * values[:, 4] + 0.125
    elif args.mode == "prior_only":
        output = np.full(values.shape[0], 0.125, dtype=np.float64)
    elif args.mode == "stochastic":
        entropy = secrets.randbits(53) / float(1 << 53)
        output = output + entropy * 1e-2
    elif args.mode == "wavelength_aware":
        # Metadata dependence is intentionally large relative to the 1e-6 absolute floor while
        # remaining a smooth numeric pre-decision output.
        output = output + (values[:, :4] @ (wavelength_nm[:4] / 1_000_000.0))
    elif args.mode == "fwhm_aware":
        # This is a deliberate C3 confound: the executable consumes FWHM but ignores wavelength.
        # A wavelength-only primary mutation must fail even when the separate FWHM diagnostic moves.
        output = output + (values[:, :4] @ (fwhm_nm[:4] / 10_000.0))
    elif args.mode == "wavelength_range_aware":
        # Rotation preserves this statistic. The keyed non-uniform magnitude vectors must expose it.
        output = output + (float(np.max(wavelength_nm)) - float(np.min(wavelength_nm))) / 1000.0
    elif args.mode == "fwhm_ratio_aware":
        # Rotation and uniform scaling preserve this statistic; keyed per-ID factors do not.
        output = output + float(np.max(fwhm_nm)) / float(np.min(fwhm_nm))
    elif args.mode == "replay_range_straddles_first":
        # C0's frozen statistic is max(replay)-min(replay), not maximum distance from replay zero.
        # Make each side individually sub-threshold while their complete range is supra-threshold.
        invocation_index = _next_invocation_index("output-replay")
        if invocation_index in {1, 2}:
            scale_for_jitter = max(1.0, float(np.percentile(np.abs(output), 99)))
            direction = 1.0 if invocation_index == 1 else -1.0
            output = output + direction * 0.75e-7 * scale_for_jitter
    elif args.mode == "malformed_object_output":
        np.savez(
            args.output,
            output=np.asarray([{"forbidden": "pickle"}], dtype=object),
            pre_core=pre_core,
        )
        return 0
    elif args.mode == "wrong_output_shape":
        output = output[:, None]
    elif args.mode == "wrong_tap_shape":
        pre_core = pre_core[:, :, None]
    elif args.mode == "wrong_dtype":
        np.savez(
            args.output,
            output=np.asarray(output, dtype=np.float32),
            pre_core=np.asarray(pre_core, dtype=np.float32),
        )
        return 0
    elif args.mode == "extra_output_member":
        np.savez(
            args.output,
            output=np.asarray(output, dtype=np.float64),
            pre_core=np.asarray(pre_core, dtype=np.float64),
            surprise=np.asarray([1.0], dtype=np.float64),
        )
        return 0
    elif args.mode == "nonfinite_output":
        output[0] = np.nan
    elif args.mode == "extreme_output_response":
        output[0] = np.nextafter(np.float64(1e150), np.float64(np.inf))
    elif args.mode == "extreme_tap_response":
        pre_core[0, 0] = np.nextafter(np.float64(1e150), np.float64(np.inf))
    elif args.mode in {"bzip2_output", "lzma_output"}:
        _write_npz_with_compression(
            args.output,
            output=output,
            pre_core=pre_core,
            compression=(
                zipfile.ZIP_BZIP2 if args.mode == "bzip2_output" else zipfile.ZIP_LZMA
            ),
        )
        return 0

    _write_npz(
        args.output,
        output=np.asarray(output, dtype=np.float64),
        pre_core=np.asarray(pre_core, dtype=np.float64),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
