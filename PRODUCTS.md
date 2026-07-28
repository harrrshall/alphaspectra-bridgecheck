# AlphaSpectra public tools in this repository

This repository hosts two independent products. Co-location does not combine their evidence,
runtime, licensing or claim ceilings.

| Product | Runtime and purpose | License boundary | Maximum claim |
|---|---|---|---|
| **BridgeCheck 0.1.x** | Browser/Python candidate SWIR generation plus paired-spectrum reconstruction audit | Code is Apache-2.0; the transformed numerical bank retains the separate attribution terms in `DATA_ATTRIBUTION.md` | Output is `model_derived`; no measurement equivalence, camera transfer, diagnosis or downstream utility |
| **BandTrace 0.1.x** | Separate local Linux CLI for executable-route, probe-dependence and declared-SRF-support preflight | Apache-2.0; no BridgeCheck model bank is bundled or consumed | X and S software evidence only; biological transport is always `T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED` |

BridgeCheck does not execute BandTrace or send it spectra. BandTrace does not validate BridgeCheck,
and BridgeCheck-generated values are not BandTrace ground truth. A result from either product does
not certify, calibrate, approve or establish the biological performance of the other.

## Public entry points

- BridgeCheck browser application: <https://harrrshall.github.io/alphaspectra-bridgecheck/>
- BandTrace static release page: <https://harrrshall.github.io/alphaspectra-bridgecheck/bandtrace/>
- BandTrace publication receipt: <https://harrrshall.github.io/alphaspectra-bridgecheck/docs/PUBLICATION_VERIFICATION.json>
- BridgeCheck release tag: `v0.1.0`
- BandTrace release tag: `bandtrace-v0.1.0`
- Private vulnerability reporting:
  <https://github.com/harrrshall/alphaspectra-bridgecheck/security/advisories/new>

The root `LICENSE`, combined Python-package metadata and `DATA_ATTRIBUTION.md` govern BridgeCheck.
The legal files inside `bandtrace/source/` govern BandTrace. Do not summarize the entire repository
as Apache-2.0-only because the BridgeCheck numerical artifact carries separate source terms.
