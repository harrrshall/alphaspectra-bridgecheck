# Security policy

## Supported version

The latest `0.1.x` release receives security fixes.

## Reporting

Open a private security advisory in the GitHub repository. Do not attach confidential spectra to a
public issue.

## Data handling

The static app has no upload backend. The API rejects oversized requests, nonfinite values,
duplicate or unsorted wavelengths and out-of-contract radiometry. It never evaluates caller code,
loads pickle files or clips values into range.
