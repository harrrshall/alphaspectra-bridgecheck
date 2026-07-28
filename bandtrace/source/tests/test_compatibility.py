from __future__ import annotations

import base64
import configparser
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from types import ModuleType
import zipfile

import numpy as np
import pytest


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
UMBRELLA_REPOSITORY_ROOT = PRODUCT_ROOT.parents[1]
SOURCE_ROOT = PRODUCT_ROOT / "src"
NORMATIVE_ROOT = SOURCE_ROOT / "bandtrace" / "normative"
NORMATIVE_PRODUCT_DOCUMENT = NORMATIVE_ROOT / "BANDTRACE_PRODUCT.md"
NORMATIVE_MACHINE_CONFIG = NORMATIVE_ROOT / "bandtrace_v1.yaml"
PRODUCT_COPY_IGNORE = shutil.ignore_patterns(
    "build",
    "dist",
    "*.egg-info",
    ".tox",
    "__pycache__",
    ".pytest_cache",
    "*.pyc",
)


def _copy_clean_product_tree(destination: Path) -> None:
    shutil.copytree(PRODUCT_ROOT, destination, ignore=PRODUCT_COPY_IGNORE)


def _load_release_builder(path: Path | None = None) -> ModuleType:
    if path is None:
        path = PRODUCT_ROOT / "tools" / "build_release.py"
    spec = importlib.util.spec_from_file_location("bandtrace_test_release_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_canonical_wheel_boundary(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        assert names
        assert len(names) == len(set(names))
        for name in names:
            assert name
            assert "\x00" not in name
            assert "\\" not in name
            assert not name.startswith("/")
            parts = name.split("/")
            assert all(part not in {"", ".", ".."} for part in parts)
            assert not parts[0].endswith(":")
            assert PurePosixPath(*parts).as_posix() == name

        dist_info_roots = {
            name.split("/", 1)[0]
            for name in names
            if name.split("/", 1)[0].endswith(".dist-info")
        }
        assert dist_info_roots == {"alphaspectra_bandtrace-0.1.0.dist-info"}
        dist_info = next(iter(dist_info_roots))

        record_name = f"{dist_info}/RECORD"
        record_rows = list(
            csv.reader(
                io.StringIO(
                    archive.read(record_name).decode("utf-8"),
                    newline="",
                )
            )
        )
        assert all(len(row) == 3 for row in record_rows)
        record_paths = [row[0] for row in record_rows]
        assert len(record_paths) == len(set(record_paths))
        assert set(record_paths) == set(names)
        for path, recorded_digest, recorded_size in record_rows:
            if path == record_name:
                assert recorded_digest == ""
                assert recorded_size == ""
                continue
            payload = archive.read(path)
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(payload).digest()
            ).rstrip(b"=").decode("ascii")
            assert recorded_digest == f"sha256={digest}"
            assert recorded_size == str(len(payload))

        assert archive.read(f"{dist_info}/entry_points.txt") == (
            b"[console_scripts]\n"
            b"bandtrace = bandtrace.cli:main\n"
        )
        for filename in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
            packaged = [
                name
                for name in names
                if name.startswith(f"{dist_info}/licenses/")
                and PurePosixPath(name).name == filename
            ]
            assert len(packaged) == 1
            assert archive.read(packaged[0]) == (PRODUCT_ROOT / filename).read_bytes()


@pytest.fixture
def legacy_extracted_sdist_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the retired environment switch has no whole-test skip authority."""

    monkeypatch.setenv("BANDTRACE_EXTRACTED_SDIST_TEST", "1")


def test_packaging_declares_python_310_plus_and_numpy_126_through_2x() -> None:
    metadata = (PRODUCT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in metadata
    assert '"numpy>=1.26,<3"' in metadata
    assert '"Programming Language :: Python :: 3.10"' in metadata
    assert '"Programming Language :: Python :: 3.11"' in metadata
    assert '"Programming Language :: Python :: 3.12"' in metadata


def test_tox_matrix_covers_python_310_through_312_and_numpy_126_2x() -> None:
    parser = configparser.ConfigParser()
    parser.read(PRODUCT_ROOT / "tox.ini", encoding="utf-8")
    environments = set(parser["tox"]["env_list"].split())

    assert environments == {
        "py310-np126",
        "py310-np2",
        "py311-np126",
        "py311-np2",
        "py312-np126",
        "py312-np2",
    }
    deps = parser["testenv"]["deps"]
    assert parser["testenv"]["package"] == "wheel"
    assert "np126: numpy>=1.26,<2" in deps
    assert "np2: numpy>=2,<3" in deps


def test_current_environment_import_and_cli_help_smoke() -> None:
    major, minor = (int(part) for part in np.__version__.split(".")[:2])
    assert (major, minor) >= (1, 26)
    assert major < 3

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PRODUCT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "bandtrace", "--help"],
        cwd=PRODUCT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert "audit" in completed.stdout.lower()


def test_installed_source_tree_digest_known_answer_is_path_sorted_and_exact(
    tmp_path: Path,
) -> None:
    from bandtrace.canonical import installed_source_tree_sha256

    package = tmp_path / "package"
    nested = package / "z"
    nested.mkdir(parents=True)
    (nested / "b.py").write_bytes(bytes.fromhex("790a"))
    (package / "ignored.txt").write_bytes(b"not part of the Python source digest\n")
    (package / "a.py").write_bytes(bytes.fromhex("780a"))
    (package / "linked.py").symlink_to(package / "a.py")

    assert installed_source_tree_sha256(package) == (
        "b46a98a1db6ef4189c0ebfe002d2b1dc4169e3e7367b1ebf60cd2c642012a7e7"
    )


@pytest.mark.parametrize("invalid_version", [None, "", "  \t\n"])
def test_installed_distribution_version_rejects_missing_or_empty_metadata(
    monkeypatch: pytest.MonkeyPatch,
    invalid_version: object,
) -> None:
    import bandtrace.canonical as canonical_module
    from bandtrace.errors import ExecutionError

    monkeypatch.setattr(
        canonical_module.importlib.metadata,
        "version",
        lambda _distribution: invalid_version,
    )

    with pytest.raises(
        ExecutionError,
        match="installed BandTrace distribution version is missing or empty",
    ):
        canonical_module.installed_distribution_version()


def test_installed_distribution_version_preserves_source_tree_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.canonical as canonical_module
    from bandtrace.constants import PRODUCT_VERSION

    def missing_distribution(_distribution: str) -> str:
        raise canonical_module.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(
        canonical_module.importlib.metadata,
        "version",
        missing_distribution,
    )

    assert canonical_module.installed_distribution_version() == PRODUCT_VERSION


@pytest.mark.parametrize("source_kind", ["missing", "regular_file", "symlink"])
def test_installed_source_tree_digest_rejects_non_directory_sources(
    tmp_path: Path,
    source_kind: str,
) -> None:
    from bandtrace.canonical import installed_source_tree_sha256
    from bandtrace.errors import ExecutionError

    source = tmp_path / source_kind
    if source_kind == "regular_file":
        source.write_bytes(b"not a package directory\n")
    elif source_kind == "symlink":
        target = tmp_path / "real-package"
        target.mkdir()
        (target / "__init__.py").write_bytes(b"# real but reached by symlink\n")
        source.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        ExecutionError,
        match="installed BandTrace source tree is unavailable as a real directory",
    ):
        installed_source_tree_sha256(source)


def test_installed_source_tree_digest_rejects_empty_python_tree(
    tmp_path: Path,
) -> None:
    from bandtrace.canonical import installed_source_tree_sha256
    from bandtrace.errors import ExecutionError

    empty = tmp_path / "empty-package"
    empty.mkdir()
    (empty / "data.txt").write_bytes(b"non-Python resources do not attest source\n")

    with pytest.raises(
        ExecutionError,
        match="installed BandTrace source tree contains no regular Python members",
    ):
        installed_source_tree_sha256(empty)


def test_default_installed_source_digest_rejects_nonfilesystem_zipimport_pseudopath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bandtrace.canonical as canonical_module
    from bandtrace.errors import ExecutionError

    pseudo_file = tmp_path / "bandtrace.whl" / "bandtrace" / "canonical.py"
    monkeypatch.setattr(canonical_module, "__file__", str(pseudo_file))

    with pytest.raises(
        ExecutionError,
        match="installed BandTrace source tree is unavailable as a real directory",
    ):
        canonical_module.installed_source_tree_sha256()


def test_streaming_deterministic_npz_is_legacy_byte_exact_and_requires_fresh_path(
    tmp_path: Path,
) -> None:
    from bandtrace.canonical import deterministic_npz_bytes, write_deterministic_npz

    arrays = {
        "z_c_order": np.arange(12, dtype=np.float64).reshape(3, 4),
        "a_f_order": np.asfortranarray(
            np.arange(15, dtype=np.int16).reshape(3, 5)
        ),
        "unicode_ids": np.asarray(["alpha", "βeta", "病害", ""], dtype="U8"),
        "empty": np.empty((0, 3), dtype=np.float32),
    }

    legacy_output = io.BytesIO()
    with zipfile.ZipFile(
        legacy_output,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for key in sorted(arrays):
            member = io.BytesIO()
            np.save(member, arrays[key], allow_pickle=False)
            info = zipfile.ZipInfo(
                f"{key}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, member.getvalue())
    expected = legacy_output.getvalue()

    streamed = tmp_path / "streamed.npz"
    write_deterministic_npz(streamed, arrays)
    assert streamed.read_bytes() == expected
    assert deterministic_npz_bytes(arrays) == expected

    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"preserve-existing-bytes\n")
    with pytest.raises(FileExistsError):
        write_deterministic_npz(existing, arrays)
    assert existing.read_bytes() == b"preserve-existing-bytes\n"


def test_independently_generated_reference_bundles_are_byte_identical(
    tmp_path: Path,
) -> None:
    generator = PRODUCT_ROOT / "examples" / "make_reference_bundle.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    destinations = [tmp_path / "reference-a", tmp_path / "reference-b"]
    for destination in destinations:
        completed = subprocess.run(
            [sys.executable, str(generator), str(destination)],
            cwd=PRODUCT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
        assert completed.returncode == 0, completed.stderr

    first_files = {
        path.relative_to(destinations[0]).as_posix(): path.read_bytes()
        for path in destinations[0].rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(destinations[1]).as_posix(): path.read_bytes()
        for path in destinations[1].rglob("*")
        if path.is_file()
    }
    assert first_files
    assert first_files == second_files


def test_reference_bundle_public_api_and_cli_are_byte_identical_and_audit_clean(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bandtrace.audit import run_audit
    from bandtrace.cli import main as cli_main
    from bandtrace.reference import make_reference_bundle

    api_bundle = tmp_path / "api-reference"
    cli_bundle = tmp_path / "cli-reference"
    make_reference_bundle(api_bundle)
    exit_code = cli_main(["make-reference-bundle", str(cli_bundle)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == f"{cli_bundle}\n"
    assert captured.err == ""
    api_files = {
        path.relative_to(api_bundle).as_posix(): path.read_bytes()
        for path in api_bundle.rglob("*")
        if path.is_file()
    }
    cli_files = {
        path.relative_to(cli_bundle).as_posix(): path.read_bytes()
        for path in cli_bundle.rglob("*")
        if path.is_file()
    }
    assert api_files
    assert api_files == cli_files

    for index, bundle in enumerate((api_bundle, cli_bundle)):
        result = run_audit(bundle, tmp_path / f"reference-report-{index}")
        assert result.exit_code == 0
        assert result.report["states"] == {
            "executable": "X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES",
            "spectral": "S3_SRF_WITHIN_DECLARED_SUPPORT",
            "biological": "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED",
        }


def test_reference_bundle_repeat_destination_is_stable_exit_three_without_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bandtrace.cli import main as cli_main
    from bandtrace.reference import make_reference_bundle

    destination = tmp_path / "existing-reference"
    make_reference_bundle(destination)
    original = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }

    exit_code = cli_main(["make-reference-bundle", str(destination)])
    captured = capsys.readouterr()

    assert exit_code == 3
    assert captured.out == ""
    assert captured.err == (
        f"BandTrace output failure: refusing to overwrite existing path: {destination}\n"
    )
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == original
    with pytest.raises(
        FileExistsError,
        match="refusing to overwrite existing path",
    ):
        make_reference_bundle(destination)


@pytest.mark.release
def test_built_wheel_contains_and_verifies_exact_frozen_normative_authorities(
    tmp_path: Path,
) -> None:
    product_copy = tmp_path / "product"
    _copy_clean_product_tree(product_copy)
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            ".",
        ],
        cwd=product_copy,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert built.returncode == 0, built.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    _assert_canonical_wheel_boundary(wheel)

    normative_document = NORMATIVE_PRODUCT_DOCUMENT.read_bytes()
    normative_config = NORMATIVE_MACHINE_CONFIG.read_bytes()
    expected_document_hash = hashlib.sha256(normative_document).hexdigest()
    expected_config_hash = hashlib.sha256(normative_config).hexdigest()
    expected_python_hashes = {
        path.relative_to(SOURCE_ROOT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((SOURCE_ROOT / "bandtrace").rglob("*.py"))
        if path.is_file()
    }
    with zipfile.ZipFile(wheel) as archive:
        wheel_python_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist()
            if name.startswith("bandtrace/") and name.endswith(".py")
        }
        assert wheel_python_hashes == expected_python_hashes
        assert (
            archive.read("bandtrace/normative/BANDTRACE_PRODUCT.md")
            == normative_document
        )
        assert archive.read("bandtrace/normative/bandtrace_v1.yaml") == normative_config

    probe = """
import hashlib
import importlib.resources
import json
import bandtrace
from bandtrace.canonical import installed_source_tree_sha256
from bandtrace.constants import (
    NORMATIVE_MACHINE_CONFIG_SHA256,
    NORMATIVE_PRODUCT_DOCUMENT_SHA256,
)
from bandtrace.errors import ExecutionError

authority = bandtrace.verify_packaged_normative_authority()
package = importlib.resources.files("bandtrace.normative")
document = package.joinpath("BANDTRACE_PRODUCT.md").read_bytes()
config = package.joinpath("bandtrace_v1.yaml").read_bytes()
try:
    installed_source_digest = installed_source_tree_sha256()
    installed_source_error = None
except ExecutionError as error:
    installed_source_digest = None
    installed_source_error = str(error)
print(json.dumps({
    "module_file": bandtrace.__file__,
    "document_resource_sha256": hashlib.sha256(document).hexdigest(),
    "config_resource_sha256": hashlib.sha256(config).hexdigest(),
    "document_constant": NORMATIVE_PRODUCT_DOCUMENT_SHA256,
    "config_constant": NORMATIVE_MACHINE_CONFIG_SHA256,
    "verified_document": authority.product_document_sha256,
    "verified_config": authority.machine_config_sha256,
    "installed_source_digest": installed_source_digest,
    "installed_source_error": installed_source_error,
}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(wheel)
    checked = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert checked.returncode == 0, checked.stderr
    observed = json.loads(checked.stdout)
    assert wheel.name in observed["module_file"]
    assert {
        observed["document_resource_sha256"],
        observed["document_constant"],
        observed["verified_document"],
    } == {expected_document_hash}
    assert {
        observed["config_resource_sha256"],
        observed["config_constant"],
        observed["verified_config"],
    } == {expected_config_hash}
    assert observed["installed_source_digest"] is None
    assert "unavailable as a real directory" in observed["installed_source_error"]


def test_umbrella_repository_authorities_match_vendored_standalone_authorities() -> None:
    umbrella_document = UMBRELLA_REPOSITORY_ROOT / "docs" / "BANDTRACE_PRODUCT.md"
    umbrella_config = (
        UMBRELLA_REPOSITORY_ROOT / "configs" / "product" / "bandtrace_v1.yaml"
    )
    missing = [
        path
        for path in (umbrella_document, umbrella_config)
        if not path.is_file()
    ]
    if missing:
        pytest.skip(
            "umbrella repository authorities are not present in this standalone tree"
        )

    assert umbrella_document.read_bytes() == NORMATIVE_PRODUCT_DOCUMENT.read_bytes()
    assert umbrella_config.read_bytes() == NORMATIVE_MACHINE_CONFIG.read_bytes()


def test_release_builder_cli_requires_python_isolated_mode(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = "1700000000"
    destination = tmp_path / "nonisolated-release"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/build_release.py",
            "--output-dir",
            str(destination),
        ],
        cwd=PRODUCT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == (
        "BandTrace release build failed: canonical release builds require Python "
        "isolated mode; invoke `python -I tools/build_release.py`\n"
    )
    assert not destination.exists()


def test_release_builder_rejects_external_source_symlink_before_backend_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "symlink-source"
    _copy_clean_product_tree(source)
    builder = _load_release_builder(source / "tools" / "build_release.py")
    sentinel_bytes = b"external-sentinel-must-not-be-packaged\n"
    sentinel = tmp_path / "external-sentinel.txt"
    sentinel.write_bytes(sentinel_bytes)
    (source / "src" / "external-sentinel-link.txt").symlink_to(sentinel)
    backend_calls: list[str] = []

    def unexpected_toolchain(_root: Path) -> Path:
        backend_calls.append("toolchain")
        raise AssertionError("backend setup ran before source symlink rejection")

    monkeypatch.setattr(builder, "_create_isolated_toolchain", unexpected_toolchain)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    destination = tmp_path / "symlink-release"

    exit_code = builder.main(["--output-dir", str(destination)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("BandTrace release build failed: ")
    assert "symbolic link" in captured.err
    assert "Traceback" not in captured.err
    assert backend_calls == []
    assert sentinel.read_bytes() == sentinel_bytes
    assert not destination.exists()


@pytest.mark.parametrize(
    "member_name",
    [
        "",
        "/absolute",
        "../escape",
        "root/../escape",
        "root//alias",
        "root/./alias",
        "root/trailing/",
        r"root\windows",
        "C:/drive-prefix",
        "root/\x00nul",
        "root/\ncontrol",
        "root/nonascii-é",
    ],
)
def test_release_builder_rejects_noncanonical_archive_member_names(
    member_name: str,
) -> None:
    builder = _load_release_builder()

    with pytest.raises(builder.ReleaseBuildError, match="unsafe|non-canonical"):
        builder._safe_archive_name(member_name)


def test_release_builder_accepts_only_canonical_relative_posix_member_names() -> None:
    builder = _load_release_builder()

    assert builder._safe_archive_name("root") == ("root",)
    assert builder._safe_archive_name("root/nested/file.txt") == (
        "root",
        "nested",
        "file.txt",
    )


@pytest.mark.parametrize("stage", ["normalize", "extract"])
@pytest.mark.parametrize(
    "member_name",
    ["root/../escape.txt", "root//alias.txt", "root/nonascii-é.txt"],
)
def test_release_builder_rejects_malicious_tar_names_before_write_or_extract(
    tmp_path: Path,
    stage: str,
    member_name: str,
) -> None:
    builder = _load_release_builder()
    source = tmp_path / f"malicious-{stage}.tar.gz"
    payload = b"malicious-member-must-not-be-written\n"
    with tarfile.open(source, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        root = tarfile.TarInfo("root")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    destination = tmp_path / f"malicious-{stage}-destination"
    with pytest.raises(builder.ReleaseBuildError, match="unsafe|non-canonical"):
        if stage == "normalize":
            builder._normalize_sdist(source, destination, 1_700_000_000)
        else:
            builder._extract_normalized_sdist(source, destination)

    assert not (tmp_path / "escape.txt").exists()
    assert not destination.is_file()
    if destination.is_dir():
        assert list(destination.iterdir()) == []


@pytest.mark.parametrize("failed_publication", [1, 2])
def test_release_builder_partial_publication_copy_failure_removes_all_owned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_publication: int,
) -> None:
    builder = _load_release_builder()

    def fake_clean_export(_source: Path, destination: Path, _epoch: int) -> None:
        destination.mkdir()

    def fake_backend(
        _python: Path,
        *,
        hook: str,
        source_root: Path,
        output_directory: Path,
        epoch: int,
    ) -> Path:
        assert source_root.is_dir()
        assert epoch == 1_700_000_000
        output_directory.mkdir()
        name = (
            "alphaspectra_bandtrace-0.1.0.tar.gz"
            if hook == "build_sdist"
            else "alphaspectra_bandtrace-0.1.0-py3-none-any.whl"
        )
        artifact = output_directory / name
        artifact.write_bytes(hook.encode("ascii"))
        return artifact

    def fake_normalize(_source: Path, destination: Path, _epoch: int) -> None:
        destination.write_bytes(b"normalized-sdist")

    def fake_extract(_source: Path, destination: Path) -> Path:
        extracted = destination / "alphaspectra_bandtrace-0.1.0"
        extracted.mkdir(parents=True)
        return extracted

    monkeypatch.setattr(builder, "_copy_clean_export", fake_clean_export)
    monkeypatch.setattr(
        builder,
        "_create_isolated_toolchain",
        lambda root: root / "fake-python",
    )
    monkeypatch.setattr(builder, "_run_backend_hook", fake_backend)
    monkeypatch.setattr(builder, "_normalize_sdist", fake_normalize)
    monkeypatch.setattr(builder, "_extract_normalized_sdist", fake_extract)
    monkeypatch.setattr(builder, "_verify_sdist_source_payload", lambda *_args: None)
    monkeypatch.setattr(builder, "_verify_wheel_payload", lambda *_args: None)

    real_copyfileobj = builder.shutil.copyfileobj
    copy_calls = 0

    def fail_selected_copy(
        source_stream: object,
        destination_stream: object,
        length: int = 0,
    ) -> None:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == failed_publication:
            chunk = source_stream.read(1)
            destination_stream.write(chunk)
            destination_stream.flush()
            raise OSError(f"injected publication copy failure {failed_publication}")
        real_copyfileobj(source_stream, destination_stream, length)

    monkeypatch.setattr(builder.shutil, "copyfileobj", fail_selected_copy)
    destination = tmp_path / f"partial-publication-{failed_publication}"

    with pytest.raises(
        OSError,
        match=f"injected publication copy failure {failed_publication}",
    ):
        builder.build_release(
            PRODUCT_ROOT,
            destination,
            epoch=1_700_000_000,
        )

    assert copy_calls == failed_publication
    assert not destination.exists()


@pytest.mark.release
def test_release_builder_is_reproducible_and_sdist_runs_standalone_tests(
    tmp_path: Path,
    legacy_extracted_sdist_environment: None,
) -> None:
    assert legacy_extracted_sdist_environment is None
    assert os.environ["BANDTRACE_EXTRACTED_SDIST_TEST"] == "1"

    source = tmp_path / "release-source"
    _copy_clean_product_tree(source)
    stale_payload = b"must-not-enter-release-artifacts\n"
    stale_paths = (
        source / "build" / "stale_payload.py",
        source / "dist" / "stale_payload.whl",
        source / "src" / "stale.egg-info" / "stale_payload.txt",
        source / "tests" / "__pycache__" / "stale_payload.pyc",
        source / ".pytest_cache" / "stale_payload.txt",
    )
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stale_payload)

    epoch = 1_700_000_000
    hostile_python = tmp_path / "hostile-python"
    hostile_python.mkdir()
    hostile_witness = tmp_path / "hostile-sitecustomize-ran"
    (hostile_python / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(hostile_witness)!r}).write_bytes(b'hostile import ran')\n"
        "raise RuntimeError('hostile sitecustomize must be ignored')\n",
        encoding="utf-8",
    )
    hostile_find_links = tmp_path / "hostile-find-links"
    hostile_find_links.mkdir()
    expected_names = {
        "alphaspectra_bandtrace-0.1.0-py3-none-any.whl",
        "alphaspectra_bandtrace-0.1.0.tar.gz",
    }

    def build_once(output_dir: Path) -> dict[str, Path]:
        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = str(epoch)
        environment["PYTHONPATH"] = str(hostile_python)
        environment["PIP_FIND_LINKS"] = str(hostile_find_links)
        environment["PIP_INDEX_URL"] = "https://127.0.0.1:1/untrusted"
        environment["PIP_NO_INDEX"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "tools/build_release.py",
                "--output-dir",
                str(output_dir),
            ],
            cwd=source,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env=environment,
        )
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        artifacts = {
            path.name: path
            for path in output_dir.iterdir()
            if path.is_file()
        }
        assert set(artifacts) == expected_names
        expected_lines = [
            f"{hashlib.sha256(artifacts[name].read_bytes()).hexdigest()}  {name}"
            for name in sorted(artifacts)
        ]
        assert completed.stdout.splitlines() == expected_lines
        return artifacts

    first = build_once(tmp_path / "release-a")
    second = build_once(tmp_path / "release-b")
    assert {
        name: path.read_bytes() for name, path in first.items()
    } == {
        name: path.read_bytes() for name, path in second.items()
    }
    assert not hostile_witness.exists()

    wheel = first["alphaspectra_bandtrace-0.1.0-py3-none-any.whl"]
    _assert_canonical_wheel_boundary(wheel)

    sdist = first["alphaspectra_bandtrace-0.1.0.tar.gz"]
    compressed = sdist.read_bytes()
    assert int.from_bytes(compressed[4:8], "little") == epoch
    tar_bytes = gzip.decompress(compressed)
    offset = 0
    while tar_bytes[offset : offset + 512] != bytes(512):
        header = tar_bytes[offset : offset + 512]
        assert len(header) == 512
        assert header[257:263] == b"ustar\x00"
        size_field = header[124:136].strip(b" \x00") or b"0"
        size = int(size_field, 8)
        offset += 512 + ((size + 511) // 512) * 512
    assert tar_bytes[offset : offset + 1024] == bytes(1024)

    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        member_names = [member.name for member in members]
        assert member_names == sorted(member_names)
        assert members
        roots = {name.split("/", 1)[0] for name in member_names}
        assert len(roots) == 1
        archive_root = roots.pop()
        for member in members:
            assert member.uid == 0
            assert member.gid == 0
            assert member.uname == ""
            assert member.gname == ""
            assert member.mtime == epoch
            assert member.pax_headers == {}
            if member.isdir():
                assert member.mode & 0o777 == 0o755
            else:
                assert member.isreg()
                assert member.mode & 0o777 == 0o644

        relative_names = {
            name.removeprefix(f"{archive_root}/") for name in member_names
        }
        expected_test_python = {
            path.relative_to(PRODUCT_ROOT).as_posix()
            for path in (PRODUCT_ROOT / "tests").rglob("*.py")
            if path.is_file()
            and not any(
                part in {".tox", "__pycache__", ".pytest_cache"}
                for part in path.relative_to(PRODUCT_ROOT / "tests").parts
            )
        }
        archived_test_python = {
            name
            for name in relative_names
            if name.startswith("tests/") and name.endswith(".py")
        }
        assert archived_test_python == expected_test_python
        assert "tox.ini" in relative_names
        assert "examples/make_reference_bundle.py" in relative_names
        assert "tools/build_release.py" in relative_names
        assert all("stale_payload" not in name for name in relative_names)

        extracted_parent = tmp_path / "extracted"
        extracted_parent.mkdir()
        if sys.version_info >= (3, 12):
            archive.extractall(extracted_parent, filter="data")
        else:
            archive.extractall(extracted_parent)

    extracted = extracted_parent / archive_root
    nested_environment = os.environ.copy()
    nested_environment["PYTHONPATH"] = str(extracted / "src")
    nested_environment.pop("PYTEST_ADDOPTS", None)
    nested_environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    nested = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests",
            "--deselect=tests/test_compatibility.py::"
            "test_release_builder_is_reproducible_and_sdist_runs_standalone_tests",
        ],
        cwd=extracted,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=nested_environment,
    )
    assert nested.returncode == 0, (nested.stdout, nested.stderr)
    assert "1 deselected" in nested.stdout

    installed = tmp_path / "installed"
    installation = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert installation.returncode == 0, installation.stderr
    installed_environment = os.environ.copy()
    installed_environment["PYTHONPATH"] = str(installed)
    reference_bundle = tmp_path / "installed-reference-bundle"
    quickstart = subprocess.run(
        [
            sys.executable,
            "-m",
            "bandtrace",
            "make-reference-bundle",
            str(reference_bundle),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=installed_environment,
    )
    assert quickstart.returncode == 0, quickstart.stderr
    installed_report = tmp_path / "installed-reference-report"
    audited = subprocess.run(
        [
            sys.executable,
            "-m",
            "bandtrace",
            "audit",
            str(reference_bundle),
            "--output-dir",
            str(installed_report),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=installed_environment,
    )
    assert audited.returncode == 0, (audited.stdout, audited.stderr)
    assert (installed_report / "report.json").is_file()
