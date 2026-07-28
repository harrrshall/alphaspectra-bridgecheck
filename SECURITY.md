# Security policy

## Supported version

The latest `0.1.x` release receives security fixes.

## Reporting

Private vulnerability reporting is enabled for this repository. Use
<https://github.com/harrrshall/alphaspectra-bridgecheck/security/advisories/new> for BridgeCheck or
BandTrace security reports. Do not attach confidential spectra, proprietary model bundles,
credentials or other sensitive inputs to a public issue.

## Data handling

The static app has no upload backend. The API rejects oversized requests, nonfinite values,
duplicate or unsorted wavelengths and out-of-contract radiometry. It never evaluates caller code,
loads pickle files or clips values into range.

BandTrace is a separate Linux CLI. Its optional subprocess adapter executes explicitly trusted
user-supplied code and is not sandboxed; see `bandtrace/source/SECURITY.md` before use.
