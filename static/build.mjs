// Build step for the static frontend (run by Vercel via static/vercel.json).
//
// The page talks to the backend at the URL in <meta name="api-base">. That URL differs
// per deployment, so it is NOT hardcoded in index.html: set the API_BASE_URL environment
// variable in Vercel (Project -> Settings -> Environment Variables) and this script writes
// it into dist/index.html.
//
//   API_BASE_URL=https://your-backend.up.railway.app node build.mjs
//
// A production build without API_BASE_URL fails on purpose: it would ship a page that
// calls itself and 404s on every request. Preview builds may omit it (same-origin).

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const META = '<meta name="api-base" content="" />';

function fail(message) {
  console.error(`\nbuild.mjs: ${message}\n`);
  process.exit(1);
}

let apiBase = (process.env.API_BASE_URL || "").trim().replace(/\/+$/, "");

if (!apiBase) {
  if (process.env.VERCEL_ENV === "production") {
    fail("API_BASE_URL is not set. Set it to your backend's public https URL (no trailing path).");
  }
  console.warn("build.mjs: API_BASE_URL not set; the page will call its own origin.");
} else {
  let url;
  try {
    url = new URL(apiBase);
  } catch {
    fail(`API_BASE_URL is not a valid URL: ${apiBase}`);
  }
  if (url.protocol !== "https:" && url.hostname !== "localhost" && url.hostname !== "127.0.0.1") {
    fail(`API_BASE_URL must be https (browsers block http calls from an https page): ${apiBase}`);
  }
  apiBase = url.origin; // scheme + host + port only
}

const html = readFileSync(join(here, "index.html"), "utf8");
if (!html.includes(META)) fail("index.html no longer contains the api-base meta tag this script replaces.");

const escaped = apiBase.replace(/&/g, "&amp;").replace(/"/g, "&quot;");
const out = join(here, "dist");
mkdirSync(out, { recursive: true });
writeFileSync(join(out, "index.html"), html.replace(META, `<meta name="api-base" content="${escaped}" />`));

console.log(`build.mjs: wrote dist/index.html (api-base: ${apiBase || "same origin"})`);
