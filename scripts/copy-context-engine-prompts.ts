#!/usr/bin/env tsx
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");

const srcFile = path.join(
  projectRoot,
  "src",
  "context-engine",
  "per-message-summary",
  "summary-worker-prompt.md",
);
const distFile = path.join(
  projectRoot,
  "dist",
  "context-engine",
  "per-message-summary",
  "summary-worker-prompt.md",
);

if (!fs.existsSync(srcFile)) {
  console.warn("[copy-context-engine-prompts] Source prompt file not found:", srcFile);
  process.exit(0);
}

fs.mkdirSync(path.dirname(distFile), { recursive: true });
fs.copyFileSync(srcFile, distFile);
console.log("[copy-context-engine-prompts] Copied 1 context-engine prompt asset.");
