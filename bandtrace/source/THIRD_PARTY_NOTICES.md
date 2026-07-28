# Third-party notices

BandTrace depends on the following separately licensed Python packages. They are dependencies, not
vendored source in this distribution.

| Package | Required range | Upstream license |
|---|---:|---|
| NumPy | `>=1.26,<3` | BSD-3-Clause |
| PyYAML | `>=6.0,<7` | MIT |

Testing and build environments may additionally install pytest, build, setuptools, wheel, tox,
ruff, and packaging/audit utilities under their respective upstream licenses. Consult the exact
environment lock or installed distribution metadata for the transitive dependency set.
