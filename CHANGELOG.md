# Changelog

All notable changes to BridgeCheck are recorded here. The format follows Keep a Changelog and the
project uses semantic versioning for public artifacts.

## [Unreleased]

### Changed

- Added a separately namespaced BandTrace source/release subtree and static companion page. The
  Linux CLI does not execute in the BridgeCheck browser, receives no BridgeCheck inputs, and does
  not alter BridgeCheck's model, evidence, output typing or claim ceiling.
- Added a repository-level product/licensing matrix and product-scoped release links so BridgeCheck's
  transformed numerical bank terms are never presented as BandTrace's Apache-2.0-only terms.
- Rebuilt the browser application as a compact research notebook with publication-style typography,
  a dominant scientific plot, tabular evidence and a reduced visual palette.
- Added five one-click local example spectra with explicit provenance, expected support
  behavior, keyboard state and automatic inference.
- Added dynamic result announcements, chart summaries, stale-result clearing and responsive research
  layouts for desktop and mobile.
- Simplified section and evidence copy, removed the product/model footer, and placed a one-click
  measured CABO test spectrum directly inside the VNIR input panel.
- Reduced the header mark and favicon to a minimal three-line spectral symbol.

### Tests

- Added static product-boundary, artifact-link, hash, accessibility and no-browser-execution checks
  for the BandTrace companion surface.
- Added frozen expected-state/hash/support checks for every browser example and static accessibility,
  local-dependency, provenance and responsive-design invariants.

## [0.1.0] - 2026-07-27

### Added

- Frozen P1 physics-state bank with startup hash verification.
- Strict VNIR input contract and explicitly `model_derived` SWIR candidate output.
- Group-aware paired VNIR/SWIR reconstruction audit with comparators, distance bins, bootstrap
  intervals, controls and fail-closed no-clipping checks.
- Python library, command-line interface and optional FastAPI service.
- Browser-only application with no spectrum upload backend or CDN dependency.
- Apache-2.0 code license, source-data attribution, model card, third-party notices, privacy policy
  and security policy.
- Wheel/sdist build configuration, non-root OCI image, continuous integration and GitHub Pages
  deployment.

### Evidence boundary

- Records the positive point-spectrum reconstruction results from EXP-0123 and EXP-0124.
- Preserves the EXP-0125 camera/radiometric and downstream-utility failure.
- Does not authorize generated bands as AlphaSpectra training input or make diagnostic,
  measurement-equivalence or calibrated-uncertainty claims.
