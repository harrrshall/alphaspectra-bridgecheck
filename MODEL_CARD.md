# BridgeCheck P1 model card

## Intended use

Generate a physics-grounded **candidate** SWIR reflectance curve from an absolute-reflectance VNIR
leaf point spectrum, and audit the unchanged candidate generator against paired same-unit measured
SWIR. Intended users are spectroscopy laboratories, plant-phenotyping teams and sensor integrators.

## Model

- ID: `alphaspectra-bridge-p1-20260727`
- Method: nearest-state retrieval from a frozen, PROSPECT-D-rendered state bank.
- Bank: 1,213 states × 501 wavelengths, 400–2400 nm at 4 nm, float64.
- Context: 400–1000 nm.
- Output: 1052–2400 nm, always typed `model_derived`.
- Frozen parameters: `N=1.6`, intercept `0.013675502624398372`, slope
  `0.965228931274191`.
- Calibrated uncertainty: none. Neighbor spread and reference distance are descriptive only.

## Evidence

| Evaluation | Unit | Naive MAE | P1 MAE | Relative reduction | Result |
|---|---|---:|---:|---:|---|
| EXP-0123 CABO | 468 biological groups | 0.108635 | 0.041303 | 61.98% | development mechanism pass |
| EXP-0124 NASA FFT | 9 sites | 0.115014 | 0.051438 | 55.28% | external-source development pass |
| EXP-0125 potato HySpex cubes | 46 plants | 0.147156 | 0.073832 | 49.83% | strict camera/utility gate fail |

EXP-0125 failed because the nearest target bin lost to the strong VNIR-edge comparator, 13/230
cubes failed the no-clipping target-validity requirement, and reconstructed plus actual measured
SWIR both worsened held-plant drought Brier. This failure is part of the product evidence, not a
footnote.

## Limitations

- Point-spectrum evidence is stronger than camera/cube evidence.
- No disease, pathogen, treatment, diagnostic, safety or deployment endpoint was validated.
- The state bank imports a measured physiological distribution; this is not VNIR-only learning.
- Unknown camera response functions, geometry, calibration and species/campaign shifts can break
  the mapping.
- A low context-fit error is not a calibrated confidence score.
- Predicted SWIR may not add downstream value even when reconstruction MAE improves.

## Required use pattern

Retain measured/model-derived origin, observed-band masks, source calibration metadata and all
warnings. Run `bridgecheck audit` on paired same-unit data and then independently test whether
actual measured SWIR improves the held biological/campaign endpoint. Never use this artifact to
fill an observed cube silently.
