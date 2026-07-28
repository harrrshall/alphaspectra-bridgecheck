# Security policy

## Supported version

Security fixes are made against the current `0.1.x` line.

## Trust boundary

BandTrace parses hash-pinned JSON, YAML, and non-pickled NPZ inputs. It does not deserialize eager
model formats. The `numpy-linear-v1` adapter is the instrument-controlled reference. A
`subprocess-npz-v1` adapter executes user-supplied code and is explicitly trusted, not sandboxed;
run it under a dedicated Unix account inside an independently network-disabled containment
boundary.

Output publication requires Linux `renameat2(RENAME_NOREPLACE)`. The filesystem threat model
trusts all processes running under the BandTrace Unix UID. Use a dedicated account and private
input/output parents if unrelated same-UID processes cannot be trusted. Any destination left after
exit code `3` is untrusted and is not a completed BandTrace report.

## Reporting a vulnerability

Use the private vulnerability-reporting form at
<https://github.com/harrrshall/alphaspectra-bridgecheck/security/advisories/new>. Do not open a
public issue for a suspected vulnerability and do not attach proprietary model artifacts, raw
cubes, credentials, or sensitive bundle contents. A minimal synthetic reproducer, affected
version, platform, expected behavior, and observed behavior are sufficient for triage.
