from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path


PRODUCT_ROOT = Path(__file__).resolve().parents[1]


class InterfaceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.example_buttons: list[dict[str, str | None]] = []
        self.remote_runtime_dependencies: list[str] = []
        self.hrefs: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "button" and values.get("data-example"):
            self.example_buttons.append(values)
        if tag == "a" and values.get("href"):
            self.hrefs.append(values["href"])
        candidate = None
        if tag == "script" and values.get("src"):
            candidate = values["src"]
        if tag == "link" and "stylesheet" in values.get("rel", ""):
            candidate = values.get("href")
        if candidate and candidate.startswith(("http://", "https://", "//")):
            self.remote_runtime_dependencies.append(candidate)


def test_research_interface_has_five_accessible_playable_cases() -> None:
    html = (PRODUCT_ROOT / "src/bridgecheck/static/index.html").read_text()
    parser = InterfaceParser()
    parser.feed(html)

    assert len(parser.example_buttons) == 5
    assert {button["data-example"] for button in parser.example_buttons} == {
        "measured-cabo",
        "lower-reference",
        "median-reference",
        "higher-reference",
        "support-warning",
    }
    assert all(button.get("aria-pressed") == "false" for button in parser.example_buttons)
    assert all("disabled" in button for button in parser.example_buttons)
    assert not parser.remote_runtime_dependencies
    assert {
        "workspace",
        "examples",
        "spectrum-input",
        "run-prediction",
        "prediction-result",
        "result-announcement",
        "spectrum-chart",
        "evidence",
        "protocol",
        "companion",
        "load-sample-dataset",
    } <= parser.ids
    assert "Labels describe spectral geometry—not biology" in html
    assert "zero context error is not evidence of real-world accuracy" in html
    assert "Input / measured" not in html
    assert "— measured bands" not in html
    assert "Reference support is descriptive distance—not confidence or validation" in html
    assert "Built-in test data" in html
    assert "CABO measured VNIR sample" in html
    assert "The useful result and the failed gate" in html
    assert "The useful result—and the failed gate" not in html
    assert "Relative MAE reductions are reconstruction results, not gains" in html
    assert "Challenge the declared spectral route before external validation" in html
    assert "BandTrace does not run on this page" in html
    assert "X3 + S3 + T0 is the instrument-controlled reference result" in html
    assert "Biology not evaluated" in html
    assert "No transport, accuracy, diagnosis, certificate or deployment approval" in html
    assert "https://harrrshall.github.io/alphaspectra-bridgecheck/bandtrace/" in parser.hrefs
    assert (
        "https://github.com/harrrshall/alphaspectra-bridgecheck/tree/"
        "42dbc6248daf91fab5c4a6cf3630ef5441cf66f6/bandtrace/source"
    ) in parser.hrefs
    for removed in (
        "Notebook / 01",
        "Evidence / 02",
        "Protocol / 03",
        "AlphaSpectra / BridgeCheck",
        "Open source. Commercially usable with required source attribution.",
        "Source + audit CLI",
        "Model card",
        "Release verification",
        "Frozen model",
        "Not a measurement · Not calibrated uncertainty · Not a diagnosis",
    ):
        assert removed not in html


def test_interface_uses_local_research_visual_system_and_responsive_rules() -> None:
    css = (PRODUCT_ROOT / "src/bridgecheck/static/styles.css").read_text()
    app = (PRODUCT_ROOT / "src/bridgecheck/static/app.js").read_text()

    assert '"./examples.js"' in app
    assert "ui-serif" in css
    assert "--measured: #176b55" in css
    assert "--derived: #6758c9" in css
    assert "linear-gradient" not in css
    assert "radial-gradient" not in css
    assert "@media (max-width: 600px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "min-height: 44px" in css
    assert "aria-label" in app
    assert "Input changed. The previous candidate was cleared." in app
    assert "browser_example" in app
    assert "The previous candidate was cleared" in app
    assert 'loadExample("measured-cabo")' in app
    assert "bandtrace" not in app.lower()
    assert "bandtrace" not in (PRODUCT_ROOT / "src/bridgecheck/static/bridge-core.js").read_text().lower()
    assert "footerModelId" not in app
    assert "footerHash" not in app


def test_bandtrace_release_page_is_static_bounded_and_hash_specific() -> None:
    html = (PRODUCT_ROOT / "src/bridgecheck/static/bandtrace/index.html").read_text()
    parser = InterfaceParser()
    parser.feed(html)

    assert not parser.remote_runtime_dependencies
    assert {"meaning", "install", "release-evidence", "claim-boundary"} <= parser.ids
    assert "This page does not execute BandTrace" in html
    assert "Challenge the declared spectral route" in html
    assert "Fails closed inside its declared boundary" in html
    assert "Published release / verified hosted bytes" in html
    assert "Byte-identical hosted artifacts" in html
    assert "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED" in html
    assert "not a compatibility certificate" in html
    for removed in ("Separate product", "Apache-2.0", "Local Linux CLI", "Linux CLI · v0.1.0"):
        assert removed not in html
    assert "e6800aec7e8a8411940a1f53ed9ae56273bacc0c8c22ecccc72e0c9de9938e7f" in html
    assert "6ed50ec69baf2031ef3025bf6dc639c7f15777ae78b9fcb712a8351dd0725cb1" in html
    assert "../downloads/bandtrace/alphaspectra_bandtrace-0.1.0-py3-none-any.whl" in parser.hrefs
    assert "../downloads/bandtrace/alphaspectra_bandtrace-0.1.0.tar.gz" in parser.hrefs
    assert "../downloads/bandtrace/PREPUBLICATION_VERIFICATION.json" in parser.hrefs
    assert "../downloads/bandtrace/SHA256SUMS" in parser.hrefs
    assert "../docs/PUBLICATION_VERIFICATION.json" in parser.hrefs
    assert (
        "https://github.com/harrrshall/alphaspectra-bridgecheck/tree/"
        "42dbc6248daf91fab5c4a6cf3630ef5441cf66f6/bandtrace/source"
    ) in parser.hrefs


def test_bandtrace_publication_receipt_preserves_identity_and_claim_ceiling() -> None:
    receipt = json.loads((PRODUCT_ROOT / "bandtrace/PUBLICATION_VERIFICATION.json").read_text())

    assert receipt["status"] == "PASS_PUBLIC_RELEASE_AND_HOSTED_BYTE_VERIFICATION"
    assert receipt["source_identity"]["release_source_commit"] == (
        "42dbc6248daf91fab5c4a6cf3630ef5441cf66f6"
    )
    assert receipt["source_identity"]["github_release_immutable"] is False
    assert receipt["source_identity"]["exact_source_authority"] == "full_git_commit_sha"
    assert receipt["continuous_integration"]["conclusion"] == "success"
    assert receipt["initial_pages_deployment"]["conclusion"] == "success"
    assert receipt["release"]["all_platform_digests_match_public_fetches"] is True
    assert receipt["release"]["all_public_fetches_match_local_candidates"] is True
    assert receipt["claim_ceiling"]["biological_transport_state"] == (
        "T0_BIOLOGICAL_TRANSPORT_NOT_EVALUATED"
    )
    assert receipt["claim_ceiling"]["certificate"] is False
    assert receipt["claim_ceiling"]["biological_validation"] is False
    assert receipt["claim_ceiling"]["deployment_approval"] is False
