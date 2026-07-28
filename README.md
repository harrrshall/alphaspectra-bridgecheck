# AlphaSpectra BridgeCheck

**See beyond your VNIR sensor—without pretending a prediction is a measurement.**

BridgeCheck is a commercially usable, open-source tool for leaf spectroscopy. It generates a
physics-grounded, explicitly `model_derived` SWIR candidate from measured VNIR reflectance and can
audit the unchanged decoder on paired customer measurements before anyone trusts it.

Try the public, browser-only app: **https://harrrshall.github.io/alphaspectra-bridgecheck/**

## What it does

- `predict`: measured 400–1000 nm leaf reflectance → candidate 1052–2400 nm reflectance.
- `audit`: paired same-sample VNIR+SWIR → grouped comparison against two baselines, spectral-distance
  bins, bootstrap intervals, context controls and strict no-clipping checks.
- Runs locally on CPU or entirely in a browser. The static app uploads nothing.
- Five one-click browser examples cover a measured training-domain spectrum, lower/median/higher
  generated reference geometry and a deliberate support-tail warning. Their provenance is displayed
  in the interface; generated cases are not measurements or accuracy evidence.

## What it does not do

BridgeCheck does **not** measure SWIR, diagnose disease, identify pathogens, infer drought or health,
replace a calibrated sensor, or provide calibrated uncertainty. Generated bands must never be
silently concatenated with measurements. A dataset-specific reconstruction pass still requires a
separate downstream measurement-value study.

## Companion tool: BandTrace

[BandTrace](https://harrrshall.github.io/alphaspectra-bridgecheck/bandtrace/) is a separate local
Linux CLI for checking whether a declared sensor-to-model spectral route agrees with an executable-
reported tap, whether required bands affect one selected numeric output on adequate probes, and
whether routed sensor response functions remain within supplier-declared training support. It
reports executable (`X`), spectral-support (`S`) and biological-transport (`T`) states separately.

The verified reference release reaches `X3 + S3 + T0`; `T0` means biological transport was **not
evaluated**. That reference result is not evidence that a customer model/camera pair will pass.
BandTrace does not run in the BridgeCheck browser app, BridgeCheck sends it no inputs, and neither
product validates the other. See the [product and licensing matrix](PRODUCTS.md) and the independent
[BandTrace source and release record](bandtrace/README.md).

## Quick start

```bash
python -m pip install .
bridgecheck predict examples/vnir_example.csv --output prediction.csv --report report.json
bridgecheck info
```

Run the local API and browser UI:

```bash
python -m pip install ".[api]"
bridgecheck serve --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080` or call:

```bash
curl -X POST http://127.0.0.1:8080/v1/predict \
  -H 'content-type: application/json' \
  --data @examples/predict_request.json
```

Container:

```bash
docker build -t alphaspectra-bridgecheck:0.1.0 .
docker run --rm -p 8080:8080 alphaspectra-bridgecheck:0.1.0
```

## Input contract

CSV prediction input has two columns:

```csv
wavelength_nm,reflectance
400,0.052
404,0.053
...
1000,0.441
```

Reflectance must be an absolute decimal fraction, finite, ordered and unclipped. Supported V1 input
has at least 100 bands, starts by 420 nm, reaches 980 nm and has no gap larger than 10 nm. Radiance,
absorbance, SNV, percentages and normalized features are rejected.

Audit input is long-form CSV with columns `sample_id`, `group_id`, `band_origin`, `wavelength_nm`,
`reflectance`; `band_origin` is either `measured_context` or `measured_target`.

## Evidence

The frozen decoder reduced held-SWIR MAE by 61.98% on CABO and 55.28% on a separate NASA FFT source.
On 230 proximal HySpex cubes it still reduced aggregate MAE by 49.83%, but failed the strict
camera/radiometric gate and neither reconstructed nor actual measured SWIR improved held-plant
drought prediction. BridgeCheck exposes that boundary instead of hiding it. See
[MODEL_CARD.md](MODEL_CARD.md) and the machine-readable
[RELEASE_VERIFICATION.json](RELEASE_VERIFICATION.json).

## Licensing

Code is Apache-2.0. The transformed candidate bank carries the source-data attribution and terms in
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). EcoSIS labels the source generic `cc-by` without a
version; this is disclosed rather than silently upgraded to CC-BY-4.0. Commercial use is permitted
subject to those attribution terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Privacy and security

The hosted static browser app processes spectra locally and has no upload backend. The optional API
is self-hosted. See [PRIVACY.md](PRIVACY.md) and [SECURITY.md](SECURITY.md).
