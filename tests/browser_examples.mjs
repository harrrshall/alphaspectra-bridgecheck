import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  loadBridgeArtifact,
  predictSpectrum,
  predictionToCsv,
} from "../src/bridgecheck/static/bridge-core.js";
import {
  buildExample,
  EXAMPLE_DEFINITIONS,
} from "../src/bridgecheck/static/examples.js";


const testRoot = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(testRoot, "..");
const modelRoot = path.join(packageRoot, "src", "bridgecheck", "model");
let requestCount = 0;

function safeModelPath(requestUrl) {
  const url = new URL(requestUrl, "http://127.0.0.1");
  const relative = decodeURIComponent(url.pathname).replace(/^\/model\//, "");
  if (!/^[A-Za-z0-9._-]+$/.test(relative)) return null;
  return path.join(modelRoot, relative);
}

const server = createServer(async (request, response) => {
  requestCount += 1;
  const filename = safeModelPath(request.url ?? "");
  if (!filename) {
    response.writeHead(404).end();
    return;
  }
  try {
    const metadata = await stat(filename);
    response.writeHead(200, {
      "content-length": metadata.size,
      "content-type": filename.endsWith(".json") ? "application/json" : "application/octet-stream",
    });
    createReadStream(filename).pipe(response);
  } catch {
    response.writeHead(404).end();
  }
});

try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const artifact = await loadBridgeArtifact(
    `http://127.0.0.1:${address.port}/model/manifest.json`,
  );
  const requestsAfterArtifact = requestCount;
  const results = [];
  for (const definition of EXAMPLE_DEFINITIONS) {
    const example = buildExample(artifact, definition.id);
    const prediction = await predictSpectrum(
      artifact,
      example.wavelengthNm,
      example.reflectance,
      { neighbors: 5, inputOrigin: definition.inputOrigin },
    );
    results.push({
      id: definition.id,
      provenance: definition.provenance,
      wavelengths: example.wavelengthNm.length,
      observed_bands: prediction.observed.wavelength_nm.length,
      derived_bands: prediction.derived.wavelength_nm.length,
      nearest_candidate: prediction.retrieval.nearest_candidate,
      context_fit_rmse: prediction.retrieval.context_fit_rmse,
      support_tier: prediction.retrieval.support_tier,
      input_sha256: prediction.input_sha256,
      input_origin: prediction.observed.origin,
      claim_status: prediction.claim_status,
      warning_count: prediction.warnings.length,
      derived_csv_rows: predictionToCsv(prediction).trim().split(/\r?\n/).length - 1,
    });
  }
  process.stdout.write(`${JSON.stringify({
    artifact_requests: requestsAfterArtifact,
    post_artifact_requests: requestCount - requestsAfterArtifact,
    results,
  })}\n`);
} finally {
  await new Promise((resolve) => server.close(resolve));
}
