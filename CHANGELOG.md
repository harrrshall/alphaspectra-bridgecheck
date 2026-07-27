# Changelog

All notable changes to BridgeCheck are recorded here. The format follows Keep a Changelog and the
project uses semantic versioning for public artifacts.

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
