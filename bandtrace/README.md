# AlphaSpectra BandTrace 0.1.0

BandTrace is a separate Apache-2.0 Linux CLI co-hosted in this repository. It does not run inside
BridgeCheck, consume BridgeCheck candidate bands or change BridgeCheck's evidence and licensing.

Public entry points:

- Static release page: <https://harrrshall.github.io/alphaspectra-bridgecheck/bandtrace/>
- Product-scoped release: <https://github.com/harrrshall/alphaspectra-bridgecheck/releases/tag/bandtrace-v0.1.0>
- Immutable source commit: <https://github.com/harrrshall/alphaspectra-bridgecheck/tree/42dbc6248daf91fab5c4a6cf3630ef5441cf66f6/bandtrace/source>
- Private security reporting: <https://github.com/harrrshall/alphaspectra-bridgecheck/security/advisories/new>
- Postpublication receipt: <https://harrrshall.github.io/alphaspectra-bridgecheck/docs/PUBLICATION_VERIFICATION.json>

## Repository layout

- `source/` — the v0.1.0 source-archive tree, tests, legal files and frozen normative resources.
- `dist/` — the v0.1.0 wheel/source archive, prepublication verification receipt and SHA-256 manifest.
- `PUBLICATION_VERIFICATION.json` — immutable commit identity, workflow runs, release-asset IDs and
  hashes, public fetch-back checks, live Pages hashes, security state and claim boundary.

The public wheel SHA-256 is
`e6800aec7e8a8411940a1f53ed9ae56273bacc0c8c22ecccc72e0c9de9938e7f`; the source archive SHA-256
is `6ed50ec69baf2031ef3025bf6dc639c7f15777ae78b9fcb712a8351dd0725cb1`.

## Claim boundary

The instrument-controlled reference bundle reaches
`X3_OUTPUT_DEPENDENCE_OBSERVED_ON_PROBES + S3_SRF_WITHIN_DECLARED_SUPPORT +
T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED`. That is software-conformance evidence for the exact frozen
bundle, not a certificate or a promise that another bundle will pass. BandTrace does not establish
accuracy, global causal band use, calibration, camera equivalence, biological transport, disease,
safety, regulatory conformity or deployment approval. The subprocess adapter is explicitly trusted
and not sandboxed.

See `../PRODUCTS.md` for the repository-level product and licensing matrix.
