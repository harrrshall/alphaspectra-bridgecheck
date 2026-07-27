# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY pyproject.toml MANIFEST.in README.md LICENSE NOTICE DATA_ATTRIBUTION.md THIRD_PARTY_NOTICES.md MODEL_CARD.md PRIVACY.md SECURITY.md ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheels ".[api]"

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="AlphaSpectra BridgeCheck" \
      org.opencontainers.image.description="Physics-grounded VNIR-to-SWIR candidate generation and paired-data auditing" \
      org.opencontainers.image.licenses="Apache-2.0 AND LicenseRef-EcoSIS-CC-BY" \
      org.opencontainers.image.source="https://github.com/harrrshall/alphaspectra-bridgecheck"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 bridgecheck \
    && useradd --uid 10001 --gid bridgecheck --no-create-home --shell /usr/sbin/nologin bridgecheck

COPY --from=build /wheels /wheels
RUN python -m pip install /wheels/* \
    && rm -rf /wheels

USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).read()"]

CMD ["uvicorn", "bridgecheck.api:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
