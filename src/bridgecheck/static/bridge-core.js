/*
 * AlphaSpectra BridgeCheck browser inference core.
 *
 * This file deliberately has no runtime dependencies. It mirrors the frozen
 * Python predictor in bridgecheck/predict.py and keeps provided input separate
 * from model-derived output. The optional input-origin label exists only so
 * built-in generated examples cannot be misrepresented as measurements.
 */

export const MODEL_ID = "alphaspectra-bridge-p1-20260727";
export const BANK_FILE_SHA256 = "aa9700558836fb7730ab650dd0eaaf921038dce65a25aba9d9efc8d451d0d83f";
export const BANK_ARRAY_SHA256 = "1d64d6a0ec1dec48c2fa0c9c33c4cfcef832e714ecf88aaee55941a0d5fc2ac0";
export const MANIFEST_FILE_SHA256 = "da3d63a6f535219a618f6b2a118693f59f2aa001a75480372a163a480ab4d2bd";

const CLAIM_STATUS = "CANDIDATE_ONLY_UNVALIDATED";
const DEFAULT_NEIGHBORS = 5;
const INPUT_ORIGINS = new Set([
  "measured",
  "measured_training_example",
  "generated_bank_example_not_measured",
  "constructed_support_test_not_measured",
]);

export class BridgeContractError extends Error {
  constructor(message) {
    super(message);
    this.name = "BridgeContractError";
  }
}

export class BridgeArtifactError extends Error {
  constructor(message) {
    super(message);
    this.name = "BridgeArtifactError";
  }
}

function requireFiniteNumber(value, label) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new BridgeArtifactError(`${label} must be a finite number`);
  }
  return number;
}

function bytesToHex(bytes) {
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

export async function sha256Hex(value) {
  if (!globalThis.crypto?.subtle) {
    throw new BridgeArtifactError("WebCrypto SHA-256 is unavailable in this browser");
  }
  let bytes;
  if (value instanceof ArrayBuffer) {
    bytes = value;
  } else if (ArrayBuffer.isView(value)) {
    bytes = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  } else {
    bytes = new TextEncoder().encode(String(value));
  }
  return bytesToHex(new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes)));
}

function concatenateBytes(parts) {
  const length = parts.reduce((total, part) => total + part.byteLength, 0);
  const merged = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    const bytes = part instanceof Uint8Array
      ? part
      : new Uint8Array(part.buffer ?? part, part.byteOffset ?? 0, part.byteLength);
    merged.set(bytes, offset);
    offset += bytes.byteLength;
  }
  return merged;
}

function int64ShapeBytes(shape) {
  const bytes = new Uint8Array(shape.length * 8);
  const view = new DataView(bytes.buffer);
  shape.forEach((value, index) => view.setBigInt64(index * 8, BigInt(value), true));
  return bytes;
}

async function arraySha256(rawBytes, shape) {
  // Mirrors hashlib updates over str(np.dtype("<f8")), int64 shape, and C-order bytes.
  const payload = concatenateBytes([
    new TextEncoder().encode("float64"),
    int64ShapeBytes(shape),
    new Uint8Array(rawBytes),
  ]);
  return sha256Hex(payload);
}

function platformIsLittleEndian() {
  const probe = new Uint16Array([0x0102]);
  return new Uint8Array(probe.buffer)[0] === 0x02;
}

function decodeLittleEndianFloat64(buffer) {
  if (platformIsLittleEndian()) {
    return new Float64Array(buffer);
  }
  const source = new DataView(buffer);
  const decoded = new Float64Array(buffer.byteLength / 8);
  for (let index = 0; index < decoded.length; index += 1) {
    decoded[index] = source.getFloat64(index * 8, true);
  }
  return decoded;
}

function validateManifest(manifest) {
  if (!manifest || typeof manifest !== "object") {
    throw new BridgeArtifactError("model manifest is not a JSON object");
  }
  if (manifest.schema_version !== "1.0" || manifest.model_id !== MODEL_ID) {
    throw new BridgeArtifactError("unsupported or unexpected BridgeCheck model manifest");
  }
  const artifact = manifest.artifact;
  if (!artifact || artifact.dtype !== "<f8" || artifact.order !== "C") {
    throw new BridgeArtifactError("BridgeCheck V1 requires a little-endian float64 C-order bank");
  }
  if (!Array.isArray(artifact.shape) || artifact.shape.length !== 2) {
    throw new BridgeArtifactError("candidate bank shape must have two dimensions");
  }
  const shape = artifact.shape.map((value) => Number(value));
  if (shape.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
    throw new BridgeArtifactError("candidate bank shape is invalid");
  }
  if (!/^[a-f0-9]{64}$/i.test(String(artifact.file_sha256))) {
    throw new BridgeArtifactError("candidate bank file SHA-256 is missing or malformed");
  }
  if (!/^[a-f0-9]{64}$/i.test(String(artifact.array_sha256))) {
    throw new BridgeArtifactError("candidate bank array SHA-256 is missing or malformed");
  }
  if (
    String(artifact.file_sha256).toLowerCase() !== BANK_FILE_SHA256
    || String(artifact.array_sha256).toLowerCase() !== BANK_ARRAY_SHA256
  ) {
    throw new BridgeArtifactError("manifest does not identify the hard-pinned BridgeCheck V1 bank");
  }
  if (!/^[A-Za-z0-9._-]+$/.test(String(artifact.filename))) {
    throw new BridgeArtifactError("candidate bank filename is unsafe");
  }
  if (!manifest.spectral_grid || !manifest.input_contract || !manifest.support_reference) {
    throw new BridgeArtifactError("model manifest is missing a required public contract");
  }
  return shape;
}

export async function loadBridgeArtifact(manifestUrl = "./model/manifest.json") {
  let manifestResponse;
  try {
    manifestResponse = await fetch(manifestUrl, { cache: "no-store", credentials: "same-origin" });
  } catch (error) {
    throw new BridgeArtifactError(`could not load the local model manifest: ${error.message}`);
  }
  if (!manifestResponse.ok) {
    throw new BridgeArtifactError(`model manifest request failed (${manifestResponse.status})`);
  }

  let manifest;
  try {
    const manifestBuffer = await manifestResponse.arrayBuffer();
    const manifestHash = await sha256Hex(manifestBuffer);
    if (manifestHash !== MANIFEST_FILE_SHA256) {
      throw new BridgeArtifactError("model manifest SHA-256 mismatch; prediction is disabled");
    }
    manifest = JSON.parse(new TextDecoder().decode(manifestBuffer));
  } catch (error) {
    if (error instanceof BridgeArtifactError) {
      throw error;
    }
    throw new BridgeArtifactError(`model manifest is not valid JSON: ${error.message}`);
  }
  const shape = validateManifest(manifest);

  const bankUrl = new URL(manifest.artifact.filename, manifestResponse.url);
  let bankResponse;
  try {
    bankResponse = await fetch(bankUrl, { cache: "no-store", credentials: "same-origin" });
  } catch (error) {
    throw new BridgeArtifactError(`could not load the local candidate bank: ${error.message}`);
  }
  if (!bankResponse.ok) {
    throw new BridgeArtifactError(`candidate bank request failed (${bankResponse.status})`);
  }
  const bankBuffer = await bankResponse.arrayBuffer();
  const expectedBytes = shape[0] * shape[1] * 8;
  if (bankBuffer.byteLength !== expectedBytes) {
    throw new BridgeArtifactError(
      `candidate bank byte count mismatch: expected ${expectedBytes}, got ${bankBuffer.byteLength}`,
    );
  }

  const fileHash = await sha256Hex(bankBuffer);
  if (fileHash !== BANK_FILE_SHA256) {
    throw new BridgeArtifactError("candidate bank file SHA-256 mismatch; prediction is disabled");
  }
  const semanticHash = await arraySha256(bankBuffer, shape);
  if (semanticHash !== BANK_ARRAY_SHA256) {
    throw new BridgeArtifactError("candidate bank array SHA-256 mismatch; prediction is disabled");
  }

  const spectral = manifest.spectral_grid;
  const start = requireFiniteNumber(spectral.start_nm, "spectral_grid.start_nm");
  const end = requireFiniteNumber(spectral.end_nm, "spectral_grid.end_nm");
  const step = requireFiniteNumber(spectral.step_nm, "spectral_grid.step_nm");
  const count = Number(spectral.count);
  if (!Number.isSafeInteger(count) || count <= 0 || step <= 0 || shape[1] !== count) {
    throw new BridgeArtifactError("manifest spectral grid does not match the candidate bank");
  }
  const wavelengthsNm = Float64Array.from({ length: count }, (_, index) => start + index * step);
  if (Math.abs(wavelengthsNm[count - 1] - end) > 1e-9) {
    throw new BridgeArtifactError("manifest spectral endpoint does not reproduce");
  }
  const contextRange = spectral.context_range_nm;
  if (!Array.isArray(contextRange) || contextRange.length !== 2) {
    throw new BridgeArtifactError("manifest context range is invalid");
  }
  const targetAbove = requireFiniteNumber(spectral.target_above_nm, "spectral_grid.target_above_nm");
  const contextIndices = [];
  const targetIndices = [];
  wavelengthsNm.forEach((wavelength, index) => {
    if (wavelength >= Number(contextRange[0]) && wavelength <= Number(contextRange[1])) {
      contextIndices.push(index);
    }
    if (wavelength > targetAbove && wavelength <= end) {
      targetIndices.push(index);
    }
  });
  if (
    contextIndices.length !== Number(spectral.context_count)
    || targetIndices.length !== Number(spectral.target_count)
  ) {
    throw new BridgeArtifactError("manifest spectral counts do not reproduce");
  }

  return {
    manifest,
    bank: decodeLittleEndianFloat64(bankBuffer),
    shape,
    wavelengthsNm,
    contextIndices: Uint32Array.from(contextIndices),
    targetIndices: Uint32Array.from(targetIndices),
    verifiedFileSha256: fileHash,
    verifiedArraySha256: semanticHash,
  };
}

export function validateContext(wavelengthsNm, reflectance, contract) {
  if (!Array.isArray(wavelengthsNm) && !ArrayBuffer.isView(wavelengthsNm)) {
    throw new BridgeContractError("wavelength_nm must be a one-dimensional numeric array");
  }
  if (!Array.isArray(reflectance) && !ArrayBuffer.isView(reflectance)) {
    throw new BridgeContractError("reflectance must be a one-dimensional numeric array");
  }
  if (wavelengthsNm.length !== reflectance.length) {
    throw new BridgeContractError(
      "wavelength_nm and reflectance must be equal-length one-dimensional arrays",
    );
  }

  const wavelength = Float64Array.from(wavelengthsNm, Number);
  const values = Float64Array.from(reflectance, Number);
  if (wavelength.length < Number(contract.minimum_bands)) {
    throw new BridgeContractError(`at least ${contract.minimum_bands} VNIR bands are required`);
  }
  for (let index = 0; index < wavelength.length; index += 1) {
    if (!Number.isFinite(wavelength[index]) || !Number.isFinite(values[index])) {
      throw new BridgeContractError("wavelengths and reflectance must all be finite");
    }
    if (index > 0 && !(wavelength[index] > wavelength[index - 1])) {
      throw new BridgeContractError("wavelengths must be strictly increasing and unique");
    }
  }

  const [contextLow, contextHigh] = contract.absolute_context_range_nm.map(Number);
  if (wavelength[0] < contextLow || wavelength[wavelength.length - 1] > contextHigh) {
    throw new BridgeContractError(
      `predict accepts measured context only inside ${contextLow}–${contextHigh} nm; target/SWIR values are forbidden`,
    );
  }
  if (wavelength[0] > Number(contract.start_at_or_below_nm)) {
    throw new BridgeContractError("VNIR coverage starts too late for the V1 support contract");
  }
  if (wavelength[wavelength.length - 1] < Number(contract.end_at_or_above_nm)) {
    throw new BridgeContractError("VNIR coverage ends too early for the V1 support contract");
  }
  let maximumGap = 0;
  for (let index = 1; index < wavelength.length; index += 1) {
    maximumGap = Math.max(maximumGap, wavelength[index] - wavelength[index - 1]);
  }
  if (maximumGap > Number(contract.maximum_gap_nm)) {
    throw new BridgeContractError("VNIR wavelength gap exceeds the V1 support contract");
  }

  const [reflectanceLow, reflectanceHigh] = contract.reflectance_range.map(Number);
  let maximumReflectance = -Infinity;
  for (const value of values) {
    maximumReflectance = Math.max(maximumReflectance, value);
    if (value < reflectanceLow || value > reflectanceHigh) {
      if (maximumReflectance > 1.5 || value > 1.5) {
        throw new BridgeContractError(
          "reflectance appears to be percent-scaled; provide decimal fractions",
        );
      }
      throw new BridgeContractError(
        `reflectance must remain inside [${reflectanceLow}, ${reflectanceHigh}] without clipping`,
      );
    }
  }
  return { wavelength, values };
}

function retrievalPositions(artifact, wavelength) {
  const spectral = artifact.manifest.spectral_grid;
  const start = Number(spectral.start_nm);
  const step = Number(spectral.step_nm);
  const end = Number(spectral.end_nm);
  if (wavelength[0] < start || wavelength[wavelength.length - 1] > end) {
    throw new BridgeContractError("candidate interpolation would require extrapolation");
  }
  const left = new Uint32Array(wavelength.length);
  const right = new Uint32Array(wavelength.length);
  const fraction = new Float64Array(wavelength.length);
  for (let index = 0; index < wavelength.length; index += 1) {
    const position = (wavelength[index] - start) / step;
    left[index] = Math.floor(position);
    right[index] = Math.ceil(position);
    fraction[index] = position - left[index];
  }
  return { left, right, fraction };
}

function formatState(index) {
  return `state-${String(index).padStart(4, "0")}`;
}

async function inputSha256(wavelength, values) {
  const parts = [];
  for (const array of [wavelength, values]) {
    parts.push(int64ShapeBytes([array.length]));
    const bytes = new Uint8Array(array.length * 8);
    const view = new DataView(bytes.buffer);
    array.forEach((value, index) => view.setFloat64(index * 8, value, true));
    parts.push(bytes);
  }
  return sha256Hex(concatenateBytes(parts));
}

export async function predictSpectrum(
  artifact,
  wavelengthsNm,
  reflectance,
  { neighbors = DEFAULT_NEIGHBORS, inputOrigin = "measured" } = {},
) {
  if (!artifact?.manifest || !artifact?.bank) {
    throw new BridgeArtifactError("a verified BridgeCheck artifact is required");
  }
  const { wavelength, values } = validateContext(
    wavelengthsNm,
    reflectance,
    artifact.manifest.input_contract,
  );
  if (!Number.isInteger(neighbors) || neighbors < 1 || neighbors > 10) {
    throw new BridgeContractError("neighbors must be between 1 and 10");
  }
  if (!INPUT_ORIGINS.has(inputOrigin)) {
    throw new BridgeContractError("input origin is unsupported");
  }

  const [stateCount, bandCount] = artifact.shape;
  const { left, right, fraction } = retrievalPositions(artifact, wavelength);
  const ranked = new Array(stateCount);
  for (let state = 0; state < stateCount; state += 1) {
    const base = state * bandCount;
    let squaredError = 0;
    for (let index = 0; index < wavelength.length; index += 1) {
      const predicted = artifact.bank[base + left[index]] * (1 - fraction[index])
        + artifact.bank[base + right[index]] * fraction[index];
      const residual = predicted - values[index];
      squaredError += residual * residual;
    }
    ranked[state] = { state, rmse: Math.sqrt(squaredError / wavelength.length) };
  }
  ranked.sort((a, b) => a.rmse - b.rmse || a.state - b.state);
  const selected = ranked.slice(0, neighbors);
  const nearest = selected[0];

  const targetCount = artifact.targetIndices.length;
  const targetWavelengths = new Array(targetCount);
  const prediction = new Array(targetCount);
  const neighborMin = new Array(targetCount);
  const neighborMax = new Array(targetCount);
  for (let index = 0; index < targetCount; index += 1) {
    const band = artifact.targetIndices[index];
    targetWavelengths[index] = artifact.wavelengthsNm[band];
    prediction[index] = artifact.bank[nearest.state * bandCount + band];
    let minimum = Infinity;
    let maximum = -Infinity;
    for (const neighbor of selected) {
      const value = artifact.bank[neighbor.state * bandCount + band];
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    neighborMin[index] = minimum;
    neighborMax[index] = maximum;
  }

  const thresholds = artifact.manifest.support_reference.context_rmse_quantiles;
  let supportTier = "OUTSIDE_REFERENCE_Q99";
  if (nearest.rmse <= Number(thresholds.q95)) {
    supportTier = "WITHIN_REFERENCE_Q95";
  } else if (nearest.rmse <= Number(thresholds.q99)) {
    supportTier = "REFERENCE_TAIL_Q95_Q99";
  }
  const warnings = [
    "Generated reflectance is model-derived, not measured SWIR.",
    "No calibrated prediction interval is available.",
    "Not validated for diagnosis, treatment decisions, or automatic model input.",
  ];
  if (supportTier !== "WITHIN_REFERENCE_Q95") {
    warnings.push(
      "Input is outside the central reference-fit distribution; paired measurement audit is required.",
    );
  }

  return {
    model_id: artifact.manifest.model_id,
    input_sha256: await inputSha256(wavelength, values),
    observed: {
      origin: inputOrigin,
      wavelength_nm: Array.from(wavelength),
      reflectance: Array.from(values),
      observed_band_mask: Array.from({ length: wavelength.length }, () => true),
    },
    derived: {
      origin: "model_derived",
      wavelength_nm: targetWavelengths,
      reflectance: prediction,
      observed_band_mask: Array.from({ length: targetCount }, () => false),
      neighbor_envelope: {
        kind: "descriptive_not_calibrated_uncertainty",
        minimum: neighborMin,
        maximum: neighborMax,
        neighbors: selected.length,
      },
    },
    retrieval: {
      nearest_candidate: formatState(nearest.state),
      context_fit_rmse: nearest.rmse,
      support_tier: supportTier,
      support_metric: "reference_distance_descriptive_only",
      calibrated_prediction_interval: null,
    },
    claim_status: CLAIM_STATUS,
    warnings,
  };
}

function parseDelimitedRows(text, delimiter) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        cell += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === delimiter) {
      row.push(cell);
      cell = "";
    } else if (character === "\n") {
      row.push(cell.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += character;
    }
  }
  if (quoted) {
    throw new BridgeContractError("CSV contains an unterminated quoted field");
  }
  row.push(cell.replace(/\r$/, ""));
  rows.push(row);
  return rows.filter((values) => values.some((value) => value.trim() !== ""));
}

function delimiterForHeader(headerLine) {
  const candidates = [",", "\t", ";"];
  let selected = ",";
  let count = -1;
  for (const candidate of candidates) {
    const candidateCount = headerLine.split(candidate).length - 1;
    if (candidateCount > count) {
      selected = candidate;
      count = candidateCount;
    }
  }
  return selected;
}

export function parseSpectrumCsv(text) {
  const normalized = String(text ?? "").replace(/^\uFEFF/, "").trim();
  if (!normalized) {
    throw new BridgeContractError("paste a CSV or choose a local CSV file first");
  }
  const delimiter = delimiterForHeader(normalized.split(/\n/, 1)[0]);
  const rows = parseDelimitedRows(normalized, delimiter);
  if (rows.length < 2) {
    throw new BridgeContractError("CSV must contain a header and at least one spectrum row");
  }
  const headers = rows[0].map((value) => value.trim().toLowerCase());
  if (new Set(headers).size !== headers.length) {
    throw new BridgeContractError("CSV header names must be unique");
  }
  const wavelengthColumn = headers.indexOf("wavelength_nm");
  const reflectanceColumn = headers.indexOf("reflectance");
  if (wavelengthColumn < 0 || reflectanceColumn < 0) {
    throw new BridgeContractError("CSV requires wavelength_nm and reflectance columns");
  }
  if (rows.length - 1 > 5000) {
    throw new BridgeContractError("CSV exceeds the 5,000-row browser limit");
  }

  const wavelengthNm = [];
  const reflectance = [];
  rows.slice(1).forEach((row, rowIndex) => {
    const rawWavelength = row[wavelengthColumn]?.trim() ?? "";
    const rawReflectance = row[reflectanceColumn]?.trim() ?? "";
    if (rawWavelength === "" || rawReflectance === "") {
      throw new BridgeContractError(`CSV row ${rowIndex + 2} has a missing required value`);
    }
    const wavelength = Number(rawWavelength);
    const value = Number(rawReflectance);
    if (!Number.isFinite(wavelength) || !Number.isFinite(value)) {
      throw new BridgeContractError(`CSV row ${rowIndex + 2} contains a non-finite number`);
    }
    wavelengthNm.push(wavelength);
    reflectance.push(value);
  });
  return { wavelengthNm, reflectance };
}

function csvNumber(value) {
  return Number(value).toPrecision(17);
}

export function predictionToCsv(prediction) {
  const lines = [
    "origin,wavelength_nm,reflectance,observed_band,neighbor_min,neighbor_max",
  ];
  const envelope = prediction.derived.neighbor_envelope;
  prediction.derived.wavelength_nm.forEach((wavelength, index) => {
    lines.push([
      "model_derived",
      csvNumber(wavelength),
      csvNumber(prediction.derived.reflectance[index]),
      "false",
      csvNumber(envelope.minimum[index]),
      csvNumber(envelope.maximum[index]),
    ].join(","));
  });
  return `${lines.join("\n")}\n`;
}
