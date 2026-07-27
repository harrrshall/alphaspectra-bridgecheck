# Deployment

BridgeCheck has two independent deployment targets. GitHub Pages is the preferred public product:
it runs prediction entirely in the visitor's browser and has no upload service. The optional
container exposes the same frozen artifact through a self-hosted API.

Neither deployment changes the model's claim ceiling. Output remains `model_derived`, not measured
SWIR, and is not a disease, pathogen, health, drought, calibrated-uncertainty or sensor-equivalence
claim.

## Release checks

From this standalone repository root:

```bash
python -m pip install ".[api,test,build]"
python -m pytest
python -m build
bridgecheck info
docker build -t alphaspectra-bridgecheck:0.1.0 .
docker run --rm -p 8080:8080 alphaspectra-bridgecheck:0.1.0
```

In a second terminal, check `http://127.0.0.1:8080/healthz`, open
`http://127.0.0.1:8080`, and exercise one valid and one rejected input. The artifact loader verifies
the byte and array hashes before serving a request.

## Static browser application

`.github/workflows/pages.yml` publishes an intentionally narrow site artifact:

- the contents of `src/bridgecheck/static/` at the site root, with a small `/assets/` mirror that
  preserves the optional FastAPI server's asset paths;
- the contents of `src/bridgecheck/model/` under `/model/`;
- public model, attribution, privacy, security and third-party notices.

No research data, experiment output, secret, Python server or customer spectrum is part of the Pages
artifact. Browser scripts and styles must be local; a workflow check rejects remote executable or
stylesheet dependencies. The app may link to public documentation, but it must not load code from a
CDN.

Enable **Settings → Pages → Source: GitHub Actions** if Pages was previously disabled. A push to
`main` then deploys the site. The workflow also requests safe first-time enablement through the
official Pages action.

GitHub Pages does not provide `/v1/predict` or `/v1/audit`; those routes exist only in the optional
API container. The static application must use relative `./model/...` URLs so project Pages and a
local server resolve the same immutable files.

## API container

The image runs as UID/GID `10001`, listens on port `8080`, writes no application data, and includes a
health check. Only files explicitly admitted by `.dockerignore` enter the build context. Deploy it
to any OCI-compatible CPU service that provides HTTPS in front of the container.

Operational requirements:

- keep the image and model manifest immutable for a tagged release;
- terminate TLS at the platform or reverse proxy;
- set request/time limits and retain the built-in 8 MB request cap;
- do not log request bodies or customer spectra;
- restrict CORS before exposing a customer-specific API;
- publish `PRIVACY.md`, `SECURITY.md`, `DATA_ATTRIBUTION.md` and `THIRD_PARTY_NOTICES.md` beside it;
- do not market API availability as evidence of camera transfer or downstream utility.

For a different container port, override the command explicitly; the shipped health check and
default command both assume `8080`.

## Release contents and secrets

The standalone repository may contain source, tests, examples, workflows, the transformed frozen
bank, and public documentation. It must not contain AlphaSpectra's resident `data/`, research
`outputs/`, `.secrets/`, environment files, private keys, API tokens or user spectra. CI scans the
tracked file list for these forbidden paths and credential-like file names.

If a model byte changes, assign a new version, regenerate its manifest and evidence-verification
report, rerun Python/browser parity and integration tests, and update the model card and changelog.
Do not replace the bytes under an existing release tag.
