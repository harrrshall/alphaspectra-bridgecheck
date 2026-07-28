# Changelog

All notable changes to BandTrace are recorded here.

## 0.1.0 — 2026-07-28

- Added deterministic X0–X3 executable-route and probe-dependence checks.
- Added S0–S3 per-channel target-SRF support comparison with exact interval-side full-SRF
  integration and bounded Gaussian fallback.
- Added the mandatory `T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED` claim boundary.
- Added safe NumPy and explicitly trusted subprocess NPZ adapters.
- Added hash-pinned, symlink-resistant bundle loading and atomic no-clobber Linux report
  publication.
- Added deterministic JSON, HTML, CSV, NPZ, and SHA-256 evidence artifacts.
- Added 22 planted release-fault scenarios plus parser, resource, process-cleanup, path-race, and
  numerical-boundary tests.
- Added an installed `make-reference-bundle` command so the wheel quickstart is self-contained.
- Added pre-audit verification of the exact frozen normative contract and policy packaged in the
  wheel.
- Added a hash-pinned, isolated release builder that produces an anonymous, normalized,
  self-testable source archive and verifies wheel layout, legal files, entry point, and RECORD.

Version 0.1 is a software-conformance preflight. It is not a certificate, biological validation,
deployment approval, calibration finding, safety finding, or regulatory assessment.
The public source and artifacts are hosted under the product-scoped `bandtrace-v0.1.0` tag in the
AlphaSpectra BridgeCheck repository. BridgeCheck and BandTrace remain independent products with
separate executables, evidence, licenses, receipts and claim ceilings.
