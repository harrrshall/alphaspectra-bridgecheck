"""Independently configured reference and trusted subprocess adapters."""

from __future__ import annotations

import os
import selectors
import signal
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .bundle import Bundle, _npz_ids
from .canonical import write_deterministic_npz
from .constants import (
    MAX_ABS_ADAPTER_RESPONSE_VALUE,
    MAX_ADAPTER_OUTPUT_BYTES,
    MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES,
    MAX_NORMALIZATION_SCALE,
    MAX_SUBPROCESS_LOG_BYTES,
    MAX_TOTAL_ADAPTER_WALL_SECONDS,
    MIN_NORMALIZATION_SCALE,
    SUBPROCESS_TIMEOUT_SECONDS,
)
from .errors import BandTraceError, BundleError, ExecutionError
from .npzio import load_npz_bytes


@dataclass(frozen=True)
class Invocation:
    probes: np.ndarray
    target_band_ids: tuple[str, ...]
    wavelength_nm: np.ndarray
    fwhm_nm: np.ndarray


@dataclass(frozen=True)
class AdapterResponse:
    output: np.ndarray
    pre_core: np.ndarray


class Adapter(Protocol):
    assurance: str
    trust_state: str
    invocations: int
    wall_seconds: float
    cumulative_probe_value_bytes: int

    def invoke(self, request: Invocation) -> AdapterResponse: ...

    def close(self) -> None: ...


def _text_scalar(array: np.ndarray, where: str) -> str:
    if array.shape not in {(), (1,)} or array.dtype.kind not in "SU":
        raise BundleError(f"{where} must be a scalar string array")
    value = array.reshape(-1)[0]
    if isinstance(value, bytes):
        try:
            return value.decode("ascii")
        except UnicodeDecodeError as error:
            raise BundleError(f"{where} must be ASCII") from error
    return str(value)


def _artifact_float64(
    artifact: dict[str, np.ndarray], key: str, shape: tuple[int, ...], *, optional: bool = False
) -> np.ndarray | None:
    if key not in artifact:
        if optional:
            return None
        raise BundleError(f"numpy-linear-v1 artifact is missing {key!r}")
    array = artifact[key]
    if array.dtype != np.dtype("float64") or array.shape != shape or not np.isfinite(array).all():
        raise BundleError(
            f"numpy-linear-v1 artifact {key!r} must be finite float64 with exact shape {shape}"
        )
    return np.asarray(array, dtype=np.float64)


def validate_response(response: AdapterResponse, *, n: int, channels: int) -> AdapterResponse:
    output = np.asarray(response.output)
    pre_core = np.asarray(response.pre_core)
    if output.dtype != np.dtype("float64") or output.shape != (n,) or not np.isfinite(output).all():
        raise ExecutionError(f"adapter output must be finite float64 with exact shape ({n},)")
    if (
        pre_core.dtype != np.dtype("float64")
        or pre_core.shape != (n, channels)
        or not np.isfinite(pre_core).all()
    ):
        raise ExecutionError(
            f"adapter pre_core must be finite float64 with exact shape ({n}, {channels})"
        )
    if np.any(np.abs(output) > MAX_ABS_ADAPTER_RESPONSE_VALUE) or np.any(
        np.abs(pre_core) > MAX_ABS_ADAPTER_RESPONSE_VALUE
    ):
        raise ExecutionError(
            f"adapter response magnitude exceeds {MAX_ABS_ADAPTER_RESPONSE_VALUE:g}"
        )
    return AdapterResponse(np.array(output, copy=True), np.array(pre_core, copy=True))


class NumpyLinearAdapter:
    """Safe reference instrument whose transform is owned by the pinned artifact."""

    assurance = "INSTRUMENT_CONTROLLED_REFERENCE"
    trust_state = "LOCAL_SAFE_NUMPY_ARTIFACT"

    def __init__(self, bundle: Bundle):
        if bundle.numpy_artifact is None:
            raise BundleError("numpy-linear-v1 artifact was not loaded")
        artifact = bundle.numpy_artifact
        channels = len(bundle.model.channels)
        bands = len(bundle.sensor.bands)
        self._route = _artifact_float64(artifact, "route_matrix", (channels, bands))
        self._offset = _artifact_float64(artifact, "normalization_offset", (channels,))
        self._scale = _artifact_float64(artifact, "normalization_scale", (channels,))
        self._output_weights = _artifact_float64(artifact, "output_weights", (channels,))
        assert self._route is not None and self._offset is not None
        assert self._scale is not None and self._output_weights is not None
        if np.any(self._scale < MIN_NORMALIZATION_SCALE) or np.any(
            self._scale > MAX_NORMALIZATION_SCALE
        ):
            raise BundleError(
                "numpy-linear-v1 normalization_scale is outside the permitted range"
            )
        self._output_bias = 0.0
        if "output_bias" in artifact:
            bias = artifact["output_bias"]
            if bias.dtype != np.dtype("float64") or bias.shape not in {(), (1,)} or not np.isfinite(bias).all():
                raise BundleError("numpy-linear-v1 output_bias must be a finite float64 scalar")
            self._output_bias = float(bias.reshape(-1)[0])
        self._wavelength_weights = _artifact_float64(
            artifact, "wavelength_weights", (channels,), optional=True
        )
        self._fwhm_weights = _artifact_float64(
            artifact, "fwhm_weights", (channels,), optional=True
        )
        self._target_ids = _npz_ids(artifact["target_band_ids"], "artifact.target_band_ids")
        if len(self._target_ids) != bands or set(self._target_ids) != {
            band.id for band in bundle.sensor.bands
        }:
            raise BundleError("numpy-linear-v1 target_band_ids must cover the sensor exactly")
        self._spatial = _text_scalar(artifact["spatial_operation"], "artifact.spatial_operation")
        if self._spatial not in {"none", "mean"}:
            raise BundleError("numpy-linear-v1 spatial_operation must be none or mean")
        self.spatial_operation = self._spatial
        self._channels = channels
        self.invocations = 0
        self.wall_seconds = 0.0
        self.cumulative_probe_value_bytes = 0

    def invoke(self, request: Invocation) -> AdapterResponse:
        started = time.monotonic()
        self.invocations += 1
        request_bytes = int(
            np.asarray(request.probes).size * np.dtype("float64").itemsize
        )
        if (
            self.cumulative_probe_value_bytes + request_bytes
            > MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES
        ):
            raise ExecutionError(
                "cumulative adapter request probe-value byte budget exceeded"
            )
        self.cumulative_probe_value_bytes += request_bytes
        if len(request.target_band_ids) != len(set(request.target_band_ids)):
            raise ExecutionError("adapter invocation target IDs are not unique")
        positions = {identifier: index for index, identifier in enumerate(request.target_band_ids)}
        if set(positions) != set(self._target_ids):
            raise ExecutionError("adapter invocation target IDs do not match the artifact")
        order = [positions[identifier] for identifier in self._target_ids]
        values = np.asarray(request.probes, dtype=np.float64)[:, order, ...]
        wavelength = np.asarray(request.wavelength_nm, dtype=np.float64)[order]
        fwhm = np.asarray(request.fwhm_nm, dtype=np.float64)[order]
        if values.ndim == 2:
            if self._spatial not in {"none", "mean"}:
                raise ExecutionError("numpy-linear-v1 none spatial operation requires rank-2 probes")
            routed = np.einsum("nb,mb->nm", values, self._route, optimize=False)
            pre_core = (routed - self._offset[None, :]) / self._scale[None, :]
        else:
            if values.ndim != 4 or self._spatial != "mean":
                raise ExecutionError("numpy-linear-v1 mean spatial operation requires rank-4 probes")
            spatial_mean = np.mean(values, axis=(2, 3), dtype=np.float64)
            routed = np.einsum(
                "nb,mb->nm", spatial_mean, self._route, optimize=False
            )
            pre_core = (routed - self._offset[None, :]) / self._scale[None, :]
        output = pre_core @ self._output_weights + self._output_bias
        routed_wavelength = self._route @ wavelength
        routed_fwhm = self._route @ fwhm
        if self._wavelength_weights is not None:
            output = output + float(routed_wavelength @ self._wavelength_weights)
        if self._fwhm_weights is not None:
            output = output + float(routed_fwhm @ self._fwhm_weights)
        validated = validate_response(
            AdapterResponse(np.asarray(output, dtype=np.float64), np.asarray(pre_core, dtype=np.float64)),
            n=values.shape[0],
            channels=self._channels,
        )
        self.wall_seconds += time.monotonic() - started
        if self.wall_seconds > MAX_TOTAL_ADAPTER_WALL_SECONDS:
            raise ExecutionError("total adapter wall-time budget exceeded")
        return validated

    def close(self) -> None:
        return None


class SubprocessNpzAdapter:
    """Bounded protocol for explicitly trusted user code (not a sandbox)."""

    assurance = "SUPPLIER_REPORTED_TAP"
    trust_state = "USER_CODE_TRUSTED"
    spatial_operation = "SUPPLIER_REPORTED_UNATTESTED"

    def __init__(self, bundle: Bundle):
        self._bundle = bundle
        self._channels = len(bundle.model.channels)
        self._max_invocations = 2 * len(bundle.sensor.bands) + 12
        self.invocations = 0
        self.wall_seconds = 0.0
        self.cumulative_probe_value_bytes = 0
        self._temporary = tempfile.TemporaryDirectory(prefix="bandtrace-")
        self._root = Path(self._temporary.name)
        os.chmod(self._root, 0o700)
        self._pinned = self._root / "pinned"
        self._pinned.mkdir(mode=0o700)
        self._assets: dict[str, Path] = {}
        staged_records = [
            ("artifact", bundle.files["artifact"]),
            *[
                (key, bundle.files[key])
                for key in sorted(bundle.adapter.get("asset_keys", ()))
            ],
        ]
        for stage_index, (key, record) in enumerate(staged_records):
            if record.data is None:
                raise BundleError(
                    f"pinned subprocess payload is unavailable for {key!r}"
                )
            suffix = Path(record.relative_path).suffix
            target = self._pinned / f"{stage_index:02d}-{key}{suffix}"
            target.write_bytes(record.data)
            executable = key == "artifact" or bundle.adapter["argv"][0] == (
                f"{{asset:{key}}}"
            )
            os.chmod(target, 0o700 if executable else 0o600)
            if key == "artifact":
                self._artifact = target
            else:
                self._assets[key] = target

    def _leader_exited_without_reap(
        self, process: subprocess.Popen[bytes]
    ) -> bool:
        try:
            status = os.waitid(
                os.P_PID,
                process.pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except ChildProcessError as error:
            raise ExecutionError(
                "trusted subprocess leader was reaped outside BandTrace"
            ) from error
        return status is not None

    def _kill_group(self, process: subprocess.Popen[bytes]) -> int:
        """Attempt one group signal, then bounded leader termination/reap."""

        group_signal_error: OSError | None = None
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            group_signal_error = error
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                # Reaping below still runs. If the leader remains live, the
                # bounded wait/retry path fails with a stable ExecutionError.
                pass
        try:
            return_code = int(process.wait(timeout=5))
        except subprocess.TimeoutExpired:
            try:
                os.kill(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                pass
            try:
                return_code = int(process.wait(timeout=5))
            except subprocess.TimeoutExpired as error:
                raise ExecutionError(
                    "cannot reap trusted subprocess leader after termination attempts"
                ) from error
        if group_signal_error is not None:
            raise ExecutionError(
                "cannot signal trusted subprocess process group; "
                "descendant cleanup is not established"
            ) from group_signal_error
        return return_code

    def _run(
        self,
        command: list[str],
        output_path: Path,
        invocation_started: float,
    ) -> tuple[bytes, bytes]:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        started = time.monotonic()
        selector: selectors.BaseSelector | None = None
        logs = {"stdout": bytearray(), "stderr": bytearray()}
        failure: str | None = None
        return_code: int | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=self._root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
            )
        except (OSError, ValueError) as error:
            raise ExecutionError(f"cannot start trusted subprocess: {error}") from error
        try:
            if process.stdout is None or process.stderr is None:
                raise ExecutionError(
                    "trusted subprocess output pipes were not created"
                )
            try:
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            except Exception as error:
                raise ExecutionError(
                    f"cannot initialize trusted subprocess output selector: {error}"
                ) from error
            while True:
                leader_exited = self._leader_exited_without_reap(process)
                if not selector.get_map() and leader_exited:
                    break
                elapsed = time.monotonic() - started
                if elapsed > float(SUBPROCESS_TIMEOUT_SECONDS):
                    failure = f"trusted subprocess exceeded {SUBPROCESS_TIMEOUT_SECONDS} seconds"
                    break
                if (
                    self.wall_seconds
                    + (time.monotonic() - invocation_started)
                    > MAX_TOTAL_ADAPTER_WALL_SECONDS
                ):
                    failure = "total adapter wall-time budget exceeded"
                    break
                try:
                    if output_path.is_symlink():
                        failure = "trusted subprocess output path became a symlink"
                        break
                    if output_path.exists() and output_path.stat().st_size > MAX_ADAPTER_OUTPUT_BYTES:
                        failure = "trusted subprocess output exceeded byte limit"
                        break
                except OSError:
                    failure = "cannot inspect trusted subprocess output"
                    break
                for key, _ in selector.select(timeout=0.02):
                    chunk = os.read(key.fileobj.fileno(), 65_536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    logs[key.data].extend(chunk)
                    if len(logs["stdout"]) + len(logs["stderr"]) > MAX_SUBPROCESS_LOG_BYTES:
                        failure = "trusted subprocess stdout/stderr exceeded 1 MiB"
                        break
                if failure:
                    break
        finally:
            try:
                # waitid(WNOWAIT) leaves the leader PID reserved until the one
                # group-signal attempt has completed; only then do we
                # reap it. OS rejection remains an explicit execution failure.
                return_code = self._kill_group(process)
            finally:
                if selector is not None:
                    selector.close()
        if failure:
            raise ExecutionError(failure)
        assert return_code is not None
        if return_code != 0:
            excerpt = bytes(logs["stderr"][-4096:]).decode(
                "utf-8", errors="replace"
            )
            raise ExecutionError(
                f"trusted subprocess exited with status {return_code}: {excerpt}"
            )
        return bytes(logs["stdout"]), bytes(logs["stderr"])

    def _invoke_in_directory(
        self,
        request: Invocation,
        invocation: Path,
        invocation_started: float,
    ) -> AdapterResponse:
        input_path = invocation / "input.npz"
        output_path = invocation / "output.npz"
        try:
            write_deterministic_npz(
                input_path,
                {
                    "fwhm_nm": np.asarray(request.fwhm_nm, dtype=np.float64),
                    "probes": np.asarray(request.probes, dtype=np.float64),
                    "target_band_ids": np.asarray(
                        request.target_band_ids, dtype="U128"
                    ),
                    "wavelength_nm": np.asarray(
                        request.wavelength_nm, dtype=np.float64
                    ),
                },
            )
        except MemoryError as error:
            raise ExecutionError(
                "subprocess input serialization exceeded available memory"
            ) from error
        except (BandTraceError, OSError, OverflowError, ValueError) as error:
            raise ExecutionError(
                f"cannot serialize bounded subprocess input NPZ: {error}"
            ) from error
        if (
            self.wall_seconds + (time.monotonic() - invocation_started)
            > MAX_TOTAL_ADAPTER_WALL_SECONDS
        ):
            raise ExecutionError("total adapter wall-time budget exceeded")
        argv: list[str] = []
        for token in self._bundle.adapter["argv"]:
            if token == "{input_npz}":
                argv.append(str(input_path))
            elif token == "{output_npz}":
                argv.append(str(output_path))
            elif token == "{artifact}":
                argv.append(str(self._artifact))
            elif token.startswith("{asset:") and token.endswith("}"):
                key = token[len("{asset:") : -1]
                try:
                    argv.append(str(self._assets[key]))
                except KeyError as error:
                    raise ExecutionError(
                        f"subprocess asset placeholder {token!r} was not staged"
                    ) from error
            else:
                argv.append(token)
        self._run(argv, output_path, invocation_started)
        try:
            info = output_path.lstat()
        except FileNotFoundError as error:
            raise ExecutionError("trusted subprocess did not create output NPZ") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ExecutionError("trusted subprocess output must be a regular non-symlink file")
        if info.st_size > MAX_ADAPTER_OUTPUT_BYTES:
            raise ExecutionError("trusted subprocess output exceeded byte limit")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                output_path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ExecutionError(
                    "trusted subprocess output must remain a regular file when opened"
                )
            if opened.st_size > MAX_ADAPTER_OUTPUT_BYTES:
                raise ExecutionError("trusted subprocess output exceeded byte limit")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(MAX_ADAPTER_OUTPUT_BYTES + 1)
        except ExecutionError:
            raise
        except OSError as error:
            raise ExecutionError(f"cannot read trusted subprocess output: {error}") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if len(data) > MAX_ADAPTER_OUTPUT_BYTES:
            raise ExecutionError("trusted subprocess output exceeded byte limit")
        arrays = load_npz_bytes(
            data,
            source="subprocess output",
            exact_keys={"output", "pre_core"},
            execution_output=True,
        )
        return validate_response(
            AdapterResponse(arrays["output"], arrays["pre_core"]),
            n=request.probes.shape[0],
            channels=self._channels,
        )

    def invoke(self, request: Invocation) -> AdapterResponse:
        if self.invocations >= self._max_invocations:
            raise ExecutionError("adapter invocation budget exceeded")
        request_bytes = int(
            np.asarray(request.probes).size * np.dtype("float64").itemsize
        )
        if (
            self.cumulative_probe_value_bytes + request_bytes
            > MAX_CUMULATIVE_ADAPTER_PROBE_VALUE_BYTES
        ):
            raise ExecutionError(
                "cumulative adapter request probe-value byte budget exceeded"
            )
        started = time.monotonic()
        try:
            self.invocations += 1
            self.cumulative_probe_value_bytes += request_bytes
            invocation = self._root / f"invoke-{self.invocations:04d}"
            invocation.mkdir(mode=0o700)
            try:
                # validate_response returns owned copies, so successful
                # response arrays remain usable after transient files go.
                return self._invoke_in_directory(
                    request, invocation, started
                )
            finally:
                try:
                    shutil.rmtree(invocation)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    raise ExecutionError(
                        "cannot clean transient subprocess invocation "
                        f"directory: {error}"
                    ) from error
        finally:
            self.wall_seconds += time.monotonic() - started
            if self.wall_seconds > MAX_TOTAL_ADAPTER_WALL_SECONDS:
                raise ExecutionError("total adapter wall-time budget exceeded")

    def close(self) -> None:
        self._temporary.cleanup()


def build_adapter(bundle: Bundle) -> Adapter:
    if bundle.adapter["type"] == "numpy-linear-v1":
        return NumpyLinearAdapter(bundle)
    return SubprocessNpzAdapter(bundle)
