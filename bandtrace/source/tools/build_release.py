#!/usr/bin/env python3
"""Build the canonical reproducible BandTrace source and wheel artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
from email.parser import BytesParser
from email.policy import compat32
import gzip
import hashlib
import io
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile


PINNED_BUILD_REQUIREMENTS = (
    "setuptools==83.0.0",
    "wheel==0.47.0",
)
PINNED_BUILD_VERSIONS = {
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
}
PINNED_BUILD_WHEEL_HASHES = {
    "setuptools==83.0.0": (
        "29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3"
    ),
    "wheel==0.47.0": (
        "212281cab4dff978f6cedd499cd893e1f620791ca6ff7107cf270781e587eced"
    ),
}
PYPI_SIMPLE_INDEX = "https://pypi.org/simple"
EXPECTED_SDIST_NAME = "alphaspectra_bandtrace-0.1.0.tar.gz"
EXPECTED_SDIST_ROOT = "alphaspectra_bandtrace-0.1.0"
EXPECTED_WHEEL_NAME = "alphaspectra_bandtrace-0.1.0-py3-none-any.whl"
EXPECTED_DIST_INFO = "alphaspectra_bandtrace-0.1.0.dist-info"
MIN_SOURCE_DATE_EPOCH = 315_532_800  # 1980-01-01, the ZIP epoch.
MAX_SOURCE_DATE_EPOCH = (1 << 32) - 1  # gzip's unsigned 32-bit field.
_IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}


class ReleaseBuildError(RuntimeError):
    """The canonical release artifacts could not be built safely."""


def _source_date_epoch() -> int:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is None:
        raise ReleaseBuildError(
            "SOURCE_DATE_EPOCH must be set to a decimal Unix timestamp"
        )
    try:
        epoch = int(raw, 10)
    except ValueError as error:
        raise ReleaseBuildError(
            "SOURCE_DATE_EPOCH must be a decimal Unix timestamp"
        ) from error
    if str(epoch) != raw or not (
        MIN_SOURCE_DATE_EPOCH <= epoch <= MAX_SOURCE_DATE_EPOCH
    ):
        raise ReleaseBuildError(
            "SOURCE_DATE_EPOCH must be a canonical decimal integer in "
            f"[{MIN_SOURCE_DATE_EPOCH}, {MAX_SOURCE_DATE_EPOCH}]"
        )
    return epoch


def _ignore_clean_export(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in _IGNORED_DIRECTORY_NAMES:
            ignored.add(name)
        elif name.endswith(
            (".egg-info", ".pyc", ".pyo", ".tar.gz", ".whl")
        ):
            ignored.add(name)
    return ignored


def _normalize_export_tree(root: Path, epoch: int) -> None:
    directories = [root]
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ReleaseBuildError(
                f"clean release export contains a symbolic link: {path}"
            )
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            os.chmod(path, 0o644)
            os.utime(path, (epoch, epoch), follow_symlinks=False)
        else:
            raise ReleaseBuildError(
                f"clean release export contains a non-file entry: {path}"
            )
    for directory in sorted(
        directories,
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        os.chmod(directory, 0o755)
        os.utime(directory, (epoch, epoch), follow_symlinks=False)


def _copy_clean_export(project_root: Path, destination: Path, epoch: int) -> None:
    if not (project_root / "pyproject.toml").is_file():
        raise ReleaseBuildError("project root does not contain pyproject.toml")
    shutil.copytree(
        project_root,
        destination,
        symlinks=True,
        ignore=_ignore_clean_export,
    )
    _normalize_export_tree(destination, epoch)


def _minimal_environment(
    root: Path,
    *,
    python_directory: Path,
    epoch: int | None = None,
) -> dict[str, str]:
    home = root / "home"
    xdg_config = root / "xdg-config"
    xdg_cache = root / "xdg-cache"
    temporary = root / "tmp"
    for directory in (home, xdg_config, xdg_cache, temporary):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    environment = {
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": str(python_directory) + os.pathsep + os.defpath,
        "PIP_CONFIG_FILE": os.devnull,
        "PYTHONHASHSEED": "0",
        "TMPDIR": str(temporary),
        "TZ": "UTC",
        "XDG_CACHE_HOME": str(xdg_cache),
        "XDG_CONFIG_HOME": str(xdg_config),
    }
    if epoch is not None:
        environment["SOURCE_DATE_EPOCH"] = str(epoch)
    return environment


def _create_isolated_toolchain(root: Path) -> Path:
    toolchain = root / "toolchain"
    bootstrap_environment = _minimal_environment(
        root / "bootstrap-environment",
        python_directory=Path(sys.executable).resolve().parent,
    )
    created = subprocess.run(
        [sys.executable, "-I", "-m", "venv", str(toolchain)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=bootstrap_environment,
    )
    if created.returncode != 0:
        detail = (created.stderr or created.stdout).strip()[-4000:]
        raise ReleaseBuildError(
            f"cannot create isolated release environment: {detail}"
        )
    python = toolchain / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = _minimal_environment(
        root / "toolchain-environment",
        python_directory=python.parent,
    )
    requirements = root / "release-build-requirements.txt"
    requirements.write_text(
        "".join(
            f"{requirement} --hash=sha256:{PINNED_BUILD_WHEEL_HASHES[requirement]}\n"
            for requirement in PINNED_BUILD_REQUIREMENTS
        ),
        encoding="ascii",
    )
    installed = subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-deps",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--require-hashes",
            "--index-url",
            PYPI_SIMPLE_INDEX,
            "--requirement",
            str(requirements),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    if installed.returncode != 0:
        detail = (installed.stderr or installed.stdout).strip()[-4000:]
        raise ReleaseBuildError(
            f"cannot create pinned isolated release toolchain: {detail}"
        )
    probe = (
        "import importlib.metadata,json; "
        "print(json.dumps({name: importlib.metadata.version(name) "
        "for name in ('setuptools','wheel')}, sort_keys=True))"
    )
    checked = subprocess.run(
        [str(python), "-I", "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    try:
        observed = json.loads(checked.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise ReleaseBuildError(
            "cannot inspect the pinned isolated release toolchain"
        ) from error
    if checked.returncode != 0 or observed != PINNED_BUILD_VERSIONS:
        raise ReleaseBuildError(
            "isolated release toolchain version mismatch: "
            f"expected {PINNED_BUILD_VERSIONS}, observed {observed}"
        )
    return python


def _backend_environment(
    root: Path, python: Path, epoch: int
) -> dict[str, str]:
    return _minimal_environment(
        root,
        python_directory=python.parent,
        epoch=epoch,
    )


def _run_backend_hook(
    python: Path,
    *,
    hook: str,
    source_root: Path,
    output_directory: Path,
    epoch: int,
) -> Path:
    output_directory.mkdir(mode=0o755)
    code = (
        "import setuptools.build_meta as backend,sys; "
        f"backend.{hook}(sys.argv[1])"
    )
    completed = subprocess.run(
        [str(python), "-I", "-c", code, str(output_directory)],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=_backend_environment(
            output_directory.parent / f"{output_directory.name}-environment",
            python,
            epoch,
        ),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise ReleaseBuildError(f"release backend {hook} failed: {detail}")
    expected_suffix = ".tar.gz" if hook == "build_sdist" else ".whl"
    artifacts = sorted(
        path
        for path in output_directory.iterdir()
        if path.is_file() and path.name.endswith(expected_suffix)
    )
    if len(artifacts) != 1 or len(list(output_directory.iterdir())) != 1:
        raise ReleaseBuildError(
            f"release backend {hook} did not produce exactly one artifact"
        )
    expected_name = (
        EXPECTED_SDIST_NAME if hook == "build_sdist" else EXPECTED_WHEEL_NAME
    )
    if artifacts[0].name != expected_name:
        raise ReleaseBuildError(
            f"release backend {hook} produced noncanonical artifact name "
            f"{artifacts[0].name!r}"
        )
    return artifacts[0]


def _safe_archive_name(name: str) -> tuple[str, ...]:
    if (
        not name
        or not name.isascii()
        or not name.isprintable()
        or "\\" in name
        or ":" in name
    ):
        raise ReleaseBuildError(f"unsafe backend sdist member path: {name!r}")
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != name
    ):
        raise ReleaseBuildError(f"unsafe backend sdist member path: {name!r}")
    return pure.parts


def _validated_sdist_members(
    archive: tarfile.TarFile,
) -> tuple[list[tarfile.TarInfo], str]:
    members = archive.getmembers()
    if not members:
        raise ReleaseBuildError("backend sdist is empty")
    names: set[str] = set()
    roots: set[str] = set()
    for member in members:
        parts = _safe_archive_name(member.name)
        roots.add(parts[0])
        if member.name in names:
            raise ReleaseBuildError(
                f"backend sdist contains duplicate member {member.name!r}"
            )
        names.add(member.name)
        if not (member.isdir() or member.isfile()):
            raise ReleaseBuildError(
                "backend sdist contains a non-regular member: "
                f"{member.name!r}"
            )
    if roots != {EXPECTED_SDIST_ROOT}:
        raise ReleaseBuildError(
            "backend sdist must contain the canonical top-level directory "
            f"{EXPECTED_SDIST_ROOT!r}"
        )
    return members, roots.pop()


def _normalize_sdist(source: Path, destination: Path, epoch: int) -> None:
    with tarfile.open(source, mode="r:gz") as input_archive:
        members, _ = _validated_sdist_members(input_archive)

        with destination.open("xb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as output_archive:
                    for member in sorted(members, key=lambda item: item.name):
                        normalized = tarfile.TarInfo(member.name)
                        normalized.type = member.type
                        normalized.size = member.size if member.isfile() else 0
                        normalized.mode = 0o644 if member.isfile() else 0o755
                        normalized.mtime = epoch
                        normalized.uid = 0
                        normalized.gid = 0
                        normalized.uname = ""
                        normalized.gname = ""
                        normalized.pax_headers = {}
                        if member.isfile():
                            stream = input_archive.extractfile(member)
                            if stream is None:
                                raise ReleaseBuildError(
                                    "cannot read backend sdist member "
                                    f"{member.name!r}"
                                )
                            with stream:
                                output_archive.addfile(normalized, stream)
                        else:
                            output_archive.addfile(normalized)


def _extract_normalized_sdist(source: Path, destination: Path) -> Path:
    destination.mkdir(mode=0o755)
    with tarfile.open(source, mode="r:gz") as archive:
        _, archive_root = _validated_sdist_members(archive)
        extractall_parameters = inspect.signature(archive.extractall).parameters
        if "filter" in extractall_parameters:
            archive.extractall(destination, filter="fully_trusted")
        else:  # Python 3.10 releases before the extraction-filter backport.
            archive.extractall(destination)
    extracted_root = destination / archive_root
    if not extracted_root.is_dir():
        raise ReleaseBuildError("normalized sdist top-level directory is missing")
    return extracted_root


def _selected_release_source_payload(root: Path) -> dict[str, bytes]:
    selected: dict[str, bytes] = {}
    for filename in (
        "CHANGELOG.md",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
    ):
        path = root / filename
        if path.is_file():
            selected[filename] = path.read_bytes()
    for relative_directory in (
        Path("examples"),
        Path("src/bandtrace"),
        Path("tools"),
    ):
        directory = root / relative_directory
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file():
                selected[path.relative_to(root).as_posix()] = path.read_bytes()
    tests = root / "tests"
    if tests.is_dir():
        for path in sorted(tests.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file() and path.suffix in {".py", ".ini"}:
                selected[path.relative_to(root).as_posix()] = path.read_bytes()
    return selected


def _verify_sdist_source_payload(
    extracted_root: Path, clean_export: Path
) -> None:
    expected = _selected_release_source_payload(clean_export)
    observed = _selected_release_source_payload(extracted_root)
    if not expected:
        raise ReleaseBuildError("clean release source payload is empty")
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(observed)
        if expected[name] != observed[name]
    )
    if missing or unexpected or mismatched:
        raise ReleaseBuildError(
            "normalized sdist does not match the clean release source: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )


def _safe_wheel_name(name: str) -> tuple[str, ...]:
    if (
        not name
        or not name.isascii()
        or not name.isprintable()
        or "\\" in name
        or ":" in name
    ):
        raise ReleaseBuildError(f"unsafe wheel member path: {name!r}")
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != name
    ):
        raise ReleaseBuildError(f"unsafe wheel member path: {name!r}")
    return pure.parts


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _verify_wheel_payload(wheel: Path, source_root: Path) -> None:
    if wheel.name != EXPECTED_WHEEL_NAME:
        raise ReleaseBuildError(
            f"unexpected canonical wheel filename: {wheel.name!r}"
        )
    package_root = source_root / "src" / "bandtrace"
    expected: dict[str, bytes] = {}
    for path in sorted(
        package_root.rglob("*"), key=lambda item: item.as_posix()
    ):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        if path.suffix == ".py" or (
            relative.startswith("normative/")
            and path.suffix in {".md", ".yaml"}
        ):
            expected[f"bandtrace/{relative}"] = path.read_bytes()
    if not expected:
        raise ReleaseBuildError("release source package payload is empty")
    with zipfile.ZipFile(wheel, mode="r") as archive:
        members = archive.infolist()
        raw_names = [member.filename for member in members]
        if len(raw_names) != len(set(raw_names)):
            raise ReleaseBuildError("wheel contains duplicate member names")
        member_parts: dict[str, tuple[str, ...]] = {}
        for member in members:
            if member.is_dir():
                raise ReleaseBuildError(
                    f"wheel contains an unexpected directory member: {member.filename!r}"
                )
            member_parts[member.filename] = _safe_wheel_name(member.filename)
        names = set(raw_names)
        dist_info_roots = {
            parts[0]
            for parts in member_parts.values()
            if parts[0].endswith(".dist-info")
        }
        if dist_info_roots != {EXPECTED_DIST_INFO}:
            raise ReleaseBuildError(
                "wheel must contain exactly one canonical dist-info directory"
            )

        missing = sorted(set(expected) - names)
        unexpected_package_files = sorted(
            name
            for name in names
            if name.startswith("bandtrace/")
            and not name.endswith("/")
            and name not in expected
        )
        mismatched = sorted(
            name
            for name, payload in expected.items()
            if name in names and archive.read(name) != payload
        )
        required_dist_info = {
            f"{EXPECTED_DIST_INFO}/METADATA",
            f"{EXPECTED_DIST_INFO}/WHEEL",
            f"{EXPECTED_DIST_INFO}/entry_points.txt",
            f"{EXPECTED_DIST_INFO}/top_level.txt",
            f"{EXPECTED_DIST_INFO}/RECORD",
        }
        missing_dist_info = sorted(required_dist_info - names)
        legal_names = {
            f"{EXPECTED_DIST_INFO}/licenses/{filename}"
            for filename in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
        }
        allowed_names = set(expected) | required_dist_info | legal_names
        unexpected_members = sorted(names - allowed_names)
        entrypoint_name = f"{EXPECTED_DIST_INFO}/entry_points.txt"
        expected_entrypoint = b"[console_scripts]\nbandtrace = bandtrace.cli:main\n"
        if (
            entrypoint_name in names
            and archive.read(entrypoint_name) != expected_entrypoint
        ):
            raise ReleaseBuildError("wheel console entry point is not canonical")
        wheel_metadata_name = f"{EXPECTED_DIST_INFO}/WHEEL"
        expected_wheel_metadata = (
            b"Wheel-Version: 1.0\n"
            b"Generator: setuptools (83.0.0)\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n\n"
        )
        if (
            wheel_metadata_name in names
            and archive.read(wheel_metadata_name) != expected_wheel_metadata
        ):
            raise ReleaseBuildError("wheel WHEEL metadata is not canonical")
        top_level_name = f"{EXPECTED_DIST_INFO}/top_level.txt"
        if top_level_name in names and archive.read(top_level_name) != b"bandtrace\n":
            raise ReleaseBuildError("wheel top_level.txt is not canonical")
        metadata_name = f"{EXPECTED_DIST_INFO}/METADATA"
        metadata_errors: list[str] = []
        if metadata_name in names:
            metadata_payload = archive.read(metadata_name)
            metadata = BytesParser(policy=compat32).parsebytes(
                metadata_payload
            )
            for header, expected_value in (
                ("Metadata-Version", "2.4"),
                ("Name", "alphaspectra-bandtrace"),
                ("Version", "0.1.0"),
                ("Requires-Python", ">=3.10"),
                ("License-Expression", "Apache-2.0"),
            ):
                observed_values = [str(value) for value in metadata.get_all(header, [])]
                if observed_values != [expected_value]:
                    metadata_errors.append(
                        f"{header}={observed_values!r}, expected {expected_value!r}"
                    )
            expected_repeated_headers = {
                "License-File": [
                    "LICENSE",
                    "NOTICE",
                    "THIRD_PARTY_NOTICES.md",
                ],
                "Provides-Extra": ["test", "build"],
                "Requires-Dist": [
                    "numpy<3,>=1.26",
                    "PyYAML<7,>=6.0",
                    'pytest<10,>=8; extra == "test"',
                    'setuptools==83.0.0; extra == "build"',
                    'wheel==0.47.0; extra == "build"',
                ],
            }
            for header, expected_values in expected_repeated_headers.items():
                observed_values = [
                    str(value) for value in metadata.get_all(header, [])
                ]
                if observed_values != expected_values:
                    metadata_errors.append(
                        f"{header}={observed_values!r}, expected {expected_values!r}"
                    )
            pkg_info = source_root / "PKG-INFO"
            if not pkg_info.is_file() or pkg_info.read_bytes() != metadata_payload:
                metadata_errors.append("METADATA differs from normalized sdist PKG-INFO")

        legal_mismatches: list[str] = []
        for filename in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
            wheel_name = f"{EXPECTED_DIST_INFO}/licenses/{filename}"
            source_name = source_root / filename
            if (
                wheel_name not in names
                or not source_name.is_file()
                or archive.read(wheel_name) != source_name.read_bytes()
            ):
                legal_mismatches.append(filename)

        record_name = f"{EXPECTED_DIST_INFO}/RECORD"
        record_errors: list[str] = []
        if record_name in names:
            try:
                record_text = archive.read(record_name).decode("utf-8")
                record_rows = list(csv.reader(io.StringIO(record_text, newline="")))
            except (UnicodeDecodeError, csv.Error) as error:
                raise ReleaseBuildError("wheel RECORD is not valid UTF-8 CSV") from error
            record_map: dict[str, tuple[str, str]] = {}
            for row in record_rows:
                if len(row) != 3 or row[0] in record_map:
                    record_errors.append("duplicate or malformed RECORD row")
                    continue
                try:
                    _safe_wheel_name(row[0])
                except ReleaseBuildError:
                    record_errors.append(f"unsafe RECORD path {row[0]!r}")
                    continue
                record_map[row[0]] = (row[1], row[2])
            if set(record_map) != names:
                record_errors.append("RECORD member coverage differs from wheel")
            for name in sorted(names & set(record_map)):
                digest, size = record_map[name]
                if name == record_name:
                    if digest or size:
                        record_errors.append("RECORD self row must have empty hash and size")
                    continue
                payload = archive.read(name)
                if digest != _record_digest(payload) or size != str(len(payload)):
                    record_errors.append(f"RECORD hash/size mismatch for {name}")
        else:
            record_errors.append("RECORD is missing")
    if missing or unexpected_package_files or mismatched:
        raise ReleaseBuildError(
            "wheel payload does not exactly match normalized sdist source: "
            f"missing={missing}, unexpected={unexpected_package_files}, "
            f"mismatched={mismatched}"
        )
    if (
        missing_dist_info
        or unexpected_members
        or metadata_errors
        or legal_mismatches
        or record_errors
    ):
        raise ReleaseBuildError(
            "wheel dist-info validation failed: "
            f"missing={missing_dist_info}, unexpected={unexpected_members}, "
            f"metadata={metadata_errors}, legal={legal_mismatches}, "
            f"record={record_errors}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_release(
    project_root: Path,
    output_directory: Path,
    *,
    epoch: int,
) -> tuple[Path, Path]:
    project_root = project_root.resolve(strict=True)
    output_directory = output_directory.absolute()
    if output_directory.exists():
        raise ReleaseBuildError("output directory must be a fresh path")
    if not output_directory.parent.is_dir():
        raise ReleaseBuildError("output directory parent must already exist")
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    for requirement in PINNED_BUILD_REQUIREMENTS:
        if f'"{requirement}"' not in pyproject:
            raise ReleaseBuildError(
                f"pyproject.toml does not pin release tool {requirement}"
            )

    previous_umask = os.umask(0o022)
    try:
        with tempfile.TemporaryDirectory(prefix="bandtrace-release-") as temporary:
            workspace = Path(temporary)
            clean_export = workspace / "clean-export"
            _copy_clean_export(project_root, clean_export, epoch)
            toolchain_python = _create_isolated_toolchain(workspace)

            raw_sdist = _run_backend_hook(
                toolchain_python,
                hook="build_sdist",
                source_root=clean_export,
                output_directory=workspace / "raw-sdist",
                epoch=epoch,
            )
            normalized_sdist = workspace / raw_sdist.name
            _normalize_sdist(raw_sdist, normalized_sdist, epoch)

            extracted_root = _extract_normalized_sdist(
                normalized_sdist, workspace / "sdist-source"
            )
            _verify_sdist_source_payload(extracted_root, clean_export)
            wheel = _run_backend_hook(
                toolchain_python,
                hook="build_wheel",
                source_root=extracted_root,
                output_directory=workspace / "wheel",
                epoch=epoch,
            )
            _verify_wheel_payload(wheel, extracted_root)

            output_directory.mkdir(mode=0o755)
            published: list[Path] = []
            try:
                for source in sorted(
                    (normalized_sdist, wheel), key=lambda path: path.name
                ):
                    target = output_directory / source.name
                    with source.open("rb") as input_stream:
                        with target.open("xb") as output_stream:
                            published.append(target)
                            shutil.copyfileobj(input_stream, output_stream)
                    os.chmod(target, 0o644)
            except BaseException:
                for target in published:
                    try:
                        target.unlink()
                    except FileNotFoundError:
                        pass
                try:
                    output_directory.rmdir()
                except OSError:
                    pass
                raise
    finally:
        os.umask(previous_umask)

    artifacts = sorted(output_directory.iterdir(), key=lambda path: path.name)
    if len(artifacts) != 2 or any(not path.is_file() for path in artifacts):
        raise ReleaseBuildError("release publication did not produce exactly two files")
    return artifacts[0], artifacts[1]


def main(argv: list[str] | None = None) -> int:
    canonical_cli = argv is None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        if canonical_cli and not sys.flags.isolated:
            raise ReleaseBuildError(
                "canonical release builds require Python isolated mode; "
                "invoke `python -I tools/build_release.py`"
            )
        epoch = _source_date_epoch()
        artifacts = build_release(
            Path(__file__).resolve().parents[1],
            arguments.output_dir,
            epoch=epoch,
        )
    except (
        OSError,
        ReleaseBuildError,
        subprocess.SubprocessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        print(f"BandTrace release build failed: {error}", file=sys.stderr)
        return 2
    for artifact in sorted(artifacts, key=lambda path: path.name):
        print(f"{_sha256(artifact)}  {artifact.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
