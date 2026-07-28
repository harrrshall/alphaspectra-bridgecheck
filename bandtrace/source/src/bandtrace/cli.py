"""BandTrace command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import run_audit
from .errors import BandTraceError, BundleError, ExecutionError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bandtrace",
        description="Deterministic local model–sensor spectral conformance preflight.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    audit = subcommands.add_parser("audit", help="audit a hash-pinned BandTrace bundle")
    audit.add_argument("bundle", type=Path, help="bundle directory containing bandtrace.yaml")
    audit.add_argument("--output-dir", required=True, type=Path, help="artifact output directory")
    reference = subcommands.add_parser(
        "make-reference-bundle",
        help="create the installed clean reference bundle",
    )
    reference.add_argument("destination", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "make-reference-bundle":
        from .reference import make_reference_bundle

        try:
            make_reference_bundle(arguments.destination)
        except OSError as error:
            print(f"BandTrace output failure: {error}", file=sys.stderr)
            return 3
        print(arguments.destination)
        return 0
    if arguments.command != "audit":
        return 2
    try:
        result = run_audit(arguments.bundle, arguments.output_dir)
    except BundleError as error:
        print(f"BandTrace invalid bundle: {error}", file=sys.stderr)
        return 2
    except ExecutionError as error:
        print(f"BandTrace execution failure: {error}", file=sys.stderr)
        return 3
    except BandTraceError as error:
        print(f"BandTrace execution failure: {error}", file=sys.stderr)
        return 3
    except OSError as error:
        print(f"BandTrace output failure: {error}", file=sys.stderr)
        return 3
    print(
        "BandTrace audit completed: "
        f"X={result.report['states']['executable']} "
        f"S={result.report['states']['spectral']} "
        f"T={result.report['states']['biological']}"
    )
    return result.exit_code
