import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  loadBridgeArtifact,
  predictSpectrum,
} from "../src/bridgecheck/static/bridge-core.js";


const testRoot = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(testRoot, "..");
const modelRoot = path.join(packageRoot, "src", "bridgecheck", "model");

function safeModelPath(requestUrl) {
  const url = new URL(requestUrl, "http://127.0.0.1");
  const relative = decodeURIComponent(url.pathname).replace(/^\/model\//, "");
  if (!/^[A-Za-z0-9._-]+$/.test(relative)) {
    return null;
  }
  return path.join(modelRoot, relative);
}

const server = createServer(async (request, response) => {
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

let requestText = "";
for await (const chunk of process.stdin) {
  requestText += chunk;
}

try {
  const request = JSON.parse(requestText);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const artifact = await loadBridgeArtifact(
    `http://127.0.0.1:${address.port}/model/manifest.json`,
  );
  const prediction = await predictSpectrum(
    artifact,
    request.wavelength_nm,
    request.reflectance,
    { neighbors: request.neighbors ?? 5 },
  );
  process.stdout.write(`${JSON.stringify(prediction)}\n`);
} finally {
  await new Promise((resolve) => server.close(resolve));
}
