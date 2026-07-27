import {
  BridgeArtifactError,
  BridgeContractError,
  loadBridgeArtifact,
  parseSpectrumCsv,
  predictSpectrum,
  predictionToCsv,
} from "./bridge-core.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const state = {
  artifact: null,
  prediction: null,
  parsedInput: null,
  busy: false,
};

const elements = {
  modelStatus: document.querySelector("#model-status"),
  modelStatusText: document.querySelector("#model-status-text"),
  input: document.querySelector("#spectrum-input"),
  inputSummary: document.querySelector("#input-summary"),
  inputError: document.querySelector("#input-error"),
  fileInput: document.querySelector("#file-input"),
  chooseFile: document.querySelector("#choose-file"),
  dropZone: document.querySelector("#drop-zone"),
  clearInput: document.querySelector("#clear-input"),
  run: document.querySelector("#run-prediction"),
  demo: document.querySelector("#load-demo"),
  emptyResult: document.querySelector("#empty-result"),
  result: document.querySelector("#prediction-result"),
  claimChip: document.querySelector("#claim-chip"),
  metricState: document.querySelector("#metric-state"),
  metricRmse: document.querySelector("#metric-rmse"),
  metricSupport: document.querySelector("#metric-support"),
  chart: document.querySelector("#spectrum-chart"),
  observedCount: document.querySelector("#observed-count"),
  derivedCount: document.querySelector("#derived-count"),
  warnings: document.querySelector("#prediction-warnings"),
  downloadCsv: document.querySelector("#download-csv"),
  downloadJson: document.querySelector("#download-json"),
  copyJson: document.querySelector("#copy-json"),
  footerModelId: document.querySelector("#footer-model-id"),
  footerHash: document.querySelector("#footer-hash"),
};

function setModelStatus(kind, message) {
  elements.modelStatus.className = `model-status is-${kind}`;
  elements.modelStatusText.textContent = message;
}

function showError(message) {
  elements.inputError.querySelector("p").textContent = message;
  elements.inputError.hidden = false;
}

function hideError() {
  elements.inputError.hidden = true;
  elements.inputError.querySelector("p").textContent = "";
}

function summarizeInput(parsed, filename = null) {
  if (!parsed) {
    elements.inputSummary.textContent = "No spectrum loaded";
    return;
  }
  const first = parsed.wavelengthNm[0];
  const last = parsed.wavelengthNm[parsed.wavelengthNm.length - 1];
  const prefix = filename ? `${filename} · ` : "";
  elements.inputSummary.textContent = `${prefix}${parsed.wavelengthNm.length} rows · ${first}–${last} nm`;
}

function parseCurrentInput({ report = false, filename = null } = {}) {
  try {
    const parsed = parseSpectrumCsv(elements.input.value);
    state.parsedInput = parsed;
    summarizeInput(parsed, filename);
    if (report) {
      hideError();
    }
    return parsed;
  } catch (error) {
    state.parsedInput = null;
    summarizeInput(null);
    if (report) {
      showError(error.message);
    }
    return null;
  }
}

function updateRunState() {
  elements.run.disabled = !state.artifact || state.busy || !elements.input.value.trim();
  elements.demo.disabled = !state.artifact || state.busy;
}

function safeFilenamePart(value) {
  return String(value).replace(/[^A-Za-z0-9._-]/g, "-");
}

function downloadText(text, filename, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function svgElement(name, attributes = {}, text = null) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  if (text !== null) {
    element.textContent = text;
  }
  return element;
}

function linePath(wavelengths, values, x, y) {
  return wavelengths
    .map((wavelength, index) => `${index === 0 ? "M" : "L"}${x(wavelength).toFixed(2)},${y(values[index]).toFixed(2)}`)
    .join(" ");
}

function renderChart(prediction) {
  const width = 960;
  const height = 410;
  const margin = { top: 34, right: 28, bottom: 50, left: 62 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xLow = 400;
  const xHigh = 2400;
  const allValues = [
    ...prediction.observed.reflectance,
    ...prediction.derived.reflectance,
    ...prediction.derived.neighbor_envelope.minimum,
    ...prediction.derived.neighbor_envelope.maximum,
  ];
  let yLow = Math.min(...allValues);
  let yHigh = Math.max(...allValues);
  const ySpan = Math.max(yHigh - yLow, 0.05);
  yLow = Math.max(-0.08, yLow - ySpan * 0.1);
  yHigh = Math.min(1.05, yHigh + ySpan * 0.1);
  if (yHigh - yLow < 0.05) {
    yHigh = yLow + 0.05;
  }
  const x = (value) => margin.left + ((value - xLow) / (xHigh - xLow)) * plotWidth;
  const y = (value) => margin.top + (1 - (value - yLow) / (yHigh - yLow)) * plotHeight;

  const svg = svgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "presentation",
    "aria-hidden": "true",
  });
  svg.append(
    svgElement("rect", {
      x: x(400), y: margin.top, width: x(1000) - x(400), height: plotHeight,
      class: "chart-context-region",
    }),
    svgElement("rect", {
      x: x(1000), y: margin.top, width: x(1052) - x(1000), height: plotHeight,
      class: "chart-gap-region",
    }),
    svgElement("rect", {
      x: x(1052), y: margin.top, width: x(2400) - x(1052), height: plotHeight,
      class: "chart-target-region",
    }),
  );

  const xTicks = [400, 700, 1000, 1400, 1800, 2200, 2400];
  xTicks.forEach((tick) => {
    svg.append(
      svgElement("line", {
        x1: x(tick), x2: x(tick), y1: margin.top, y2: margin.top + plotHeight,
        class: "chart-grid",
      }),
      svgElement("text", {
        x: x(tick), y: height - 21, "text-anchor": "middle", class: "chart-axis-label",
      }, String(tick)),
    );
  });
  for (let index = 0; index <= 4; index += 1) {
    const tick = yLow + ((yHigh - yLow) * index) / 4;
    svg.append(
      svgElement("line", {
        x1: margin.left, x2: margin.left + plotWidth, y1: y(tick), y2: y(tick),
        class: "chart-grid",
      }),
      svgElement("text", {
        x: margin.left - 10, y: y(tick) + 3, "text-anchor": "end", class: "chart-axis-label",
      }, tick.toFixed(2)),
    );
  }
  svg.append(
    svgElement("text", {
      x: x(700), y: 20, "text-anchor": "middle", class: "chart-region-label",
    }, "MEASURED CONTEXT"),
    svgElement("text", {
      x: x(1726), y: 20, "text-anchor": "middle", class: "chart-region-label",
    }, "MODEL-DERIVED TARGET"),
    svgElement("text", {
      x: margin.left + plotWidth / 2, y: height - 4, "text-anchor": "middle", class: "chart-axis-label",
    }, "WAVELENGTH (NM)"),
    svgElement("text", {
      x: 14, y: margin.top + plotHeight / 2, transform: `rotate(-90 14 ${margin.top + plotHeight / 2})`,
      "text-anchor": "middle", class: "chart-axis-label",
    }, "REFLECTANCE (FRACTION)"),
  );

  const derivedWavelength = prediction.derived.wavelength_nm;
  const minimum = prediction.derived.neighbor_envelope.minimum;
  const maximum = prediction.derived.neighbor_envelope.maximum;
  const upper = derivedWavelength.map((value, index) => `${x(value).toFixed(2)},${y(maximum[index]).toFixed(2)}`);
  const lower = Array.from(derivedWavelength)
    .reverse()
    .map((value, reverseIndex) => {
      const index = derivedWavelength.length - 1 - reverseIndex;
      return `${x(value).toFixed(2)},${y(minimum[index]).toFixed(2)}`;
    });
  svg.append(
    svgElement("path", { d: `M${upper.join(" L")} L${lower.join(" L")} Z`, class: "chart-envelope" }),
    svgElement("path", {
      d: linePath(prediction.observed.wavelength_nm, prediction.observed.reflectance, x, y),
      class: "chart-observed",
    }),
    svgElement("path", {
      d: linePath(derivedWavelength, prediction.derived.reflectance, x, y),
      class: "chart-derived",
    }),
  );

  elements.chart.replaceChildren(svg);
}

function displaySupport(tier) {
  const labels = {
    WITHIN_REFERENCE_Q95: "Within reference Q95",
    REFERENCE_TAIL_Q95_Q99: "Reference tail Q95–Q99",
    OUTSIDE_REFERENCE_Q99: "Outside reference Q99",
  };
  const classes = {
    WITHIN_REFERENCE_Q95: "support-central",
    REFERENCE_TAIL_Q95_Q99: "support-tail",
    OUTSIDE_REFERENCE_Q99: "support-outside",
  };
  elements.metricSupport.textContent = labels[tier] ?? tier;
  elements.metricSupport.className = classes[tier] ?? "support-outside";
}

function renderPrediction(prediction) {
  elements.emptyResult.hidden = true;
  elements.result.hidden = false;
  elements.claimChip.hidden = false;
  elements.metricState.textContent = prediction.retrieval.nearest_candidate;
  elements.metricRmse.textContent = prediction.retrieval.context_fit_rmse.toPrecision(6);
  displaySupport(prediction.retrieval.support_tier);
  elements.observedCount.textContent = `${prediction.observed.wavelength_nm.length} measured bands`;
  elements.derivedCount.textContent = `${prediction.derived.wavelength_nm.length} derived bands`;
  elements.warnings.replaceChildren(
    ...prediction.warnings.map((warning) => {
      const item = document.createElement("li");
      item.textContent = warning;
      return item;
    }),
  );
  renderChart(prediction);
}

async function runPrediction() {
  hideError();
  const parsed = parseCurrentInput({ report: true });
  if (!parsed || !state.artifact) {
    return;
  }
  state.busy = true;
  updateRunState();
  const original = elements.run.querySelector("span").textContent;
  elements.run.querySelector("span").textContent = "Retrieving state…";
  try {
    const prediction = await predictSpectrum(
      state.artifact,
      parsed.wavelengthNm,
      parsed.reflectance,
      { neighbors: 5 },
    );
    state.prediction = prediction;
    renderPrediction(prediction);
  } catch (error) {
    state.prediction = null;
    if (error instanceof BridgeContractError || error instanceof BridgeArtifactError) {
      showError(error.message);
    } else {
      showError(`Prediction failed closed: ${error.message}`);
    }
  } finally {
    state.busy = false;
    elements.run.querySelector("span").textContent = original;
    updateRunState();
  }
}

function loadDemo() {
  if (!state.artifact) {
    return;
  }
  const { contextIndices, wavelengthsNm, bank, shape } = state.artifact;
  const lines = ["wavelength_nm,reflectance"];
  contextIndices.forEach((band) => {
    lines.push(`${wavelengthsNm[band]},${bank[band].toPrecision(17)}`);
  });
  elements.input.value = `${lines.join("\n")}\n`;
  hideError();
  const parsed = parseCurrentInput({ filename: "frozen-bank-demo.csv" });
  state.parsedInput = parsed;
  state.prediction = null;
  elements.emptyResult.hidden = false;
  elements.result.hidden = true;
  elements.claimChip.hidden = true;
  updateRunState();
  elements.input.scrollTop = 0;
  elements.input.focus();
  // shape is touched deliberately: this demo always uses state 0 in row-major order.
  void shape;
}

async function loadLocalFile(file) {
  if (!file) {
    return;
  }
  if (file.size > 2_000_000) {
    showError("Local CSV exceeds the 2 MB browser interface limit");
    return;
  }
  try {
    const text = await file.text();
    elements.input.value = text;
    hideError();
    parseCurrentInput({ report: true, filename: file.name });
    updateRunState();
  } catch (error) {
    showError(`Could not read the local file: ${error.message}`);
  }
}

function clearInput() {
  elements.input.value = "";
  elements.fileInput.value = "";
  state.parsedInput = null;
  state.prediction = null;
  hideError();
  summarizeInput(null);
  elements.emptyResult.hidden = false;
  elements.result.hidden = true;
  elements.claimChip.hidden = true;
  updateRunState();
  elements.input.focus();
}

function currentJson() {
  if (!state.prediction) {
    return null;
  }
  return `${JSON.stringify(state.prediction, null, 2)}\n`;
}

function installEvents() {
  let parseTimer = null;
  elements.input.addEventListener("input", () => {
    hideError();
    updateRunState();
    clearTimeout(parseTimer);
    parseTimer = setTimeout(() => parseCurrentInput(), 250);
  });
  elements.input.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      runPrediction();
    }
  });
  elements.chooseFile.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", () => loadLocalFile(elements.fileInput.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("is-over");
    });
  });
  elements.dropZone.addEventListener("drop", (event) => loadLocalFile(event.dataTransfer.files[0]));
  elements.clearInput.addEventListener("click", clearInput);
  elements.run.addEventListener("click", runPrediction);
  elements.demo.addEventListener("click", loadDemo);
  elements.downloadCsv.addEventListener("click", () => {
    if (!state.prediction) return;
    downloadText(
      predictionToCsv(state.prediction),
      `bridgecheck-${safeFilenamePart(state.prediction.input_sha256.slice(0, 12))}-derived.csv`,
      "text/csv;charset=utf-8",
    );
  });
  elements.downloadJson.addEventListener("click", () => {
    const json = currentJson();
    if (!json) return;
    downloadText(
      json,
      `bridgecheck-${safeFilenamePart(state.prediction.input_sha256.slice(0, 12))}-report.json`,
      "application/json;charset=utf-8",
    );
  });
  elements.copyJson.addEventListener("click", async () => {
    const json = currentJson();
    if (!json) return;
    try {
      await navigator.clipboard.writeText(json);
      const original = elements.copyJson.textContent;
      elements.copyJson.textContent = "Copied";
      setTimeout(() => { elements.copyJson.textContent = original; }, 1400);
    } catch (error) {
      showError(`Clipboard permission was denied: ${error.message}`);
    }
  });
}

async function initialize() {
  installEvents();
  updateRunState();
  try {
    const manifestUrl = new URL("./model/manifest.json", document.baseURI);
    state.artifact = await loadBridgeArtifact(manifestUrl);
    const states = state.artifact.shape[0].toLocaleString();
    setModelStatus("ready", `Verified · ${states} frozen states`);
    elements.footerModelId.textContent = state.artifact.manifest.model_id;
    elements.footerHash.textContent = `SHA-256 ${state.artifact.verifiedArraySha256}`;
  } catch (error) {
    state.artifact = null;
    setModelStatus("error", "Artifact verification failed");
    elements.footerHash.textContent = `Prediction disabled: ${error.message}`;
    showError(`Model unavailable; inference is disabled. ${error.message}`);
  } finally {
    updateRunState();
  }
}

initialize();
