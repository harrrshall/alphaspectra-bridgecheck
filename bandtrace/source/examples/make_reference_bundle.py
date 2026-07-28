#!/usr/bin/env python3
"""Create a small clean BandTrace numpy-linear-v1 reference bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from bandtrace.reference import make_reference_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    make_reference_bundle(arguments.destination)
    print(arguments.destination)


if __name__ == "__main__":
    main()
