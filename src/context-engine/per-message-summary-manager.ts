import { createHash, randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { resolveOpenClawAgentDir } from "../agents/agent-paths.js";
import { resolveAgentDir } from "../agents/agent-scope.js";
import { estimateMessagesTokens } from "../agents/compaction.js";
import { runEmbeddedPiAgent } from "../agents/pi-embedded.js";
import type { OpenClawConfig } from "../config/config.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { requireNodeSqlite } from "../memory/sqlite.js";
import { resolveAgentIdFromSessionKey } from "../routing/session-key.js";
import type { CompactResult, ContextEngineRuntimeContext } from "./types.js";

const log = createSubsystemLogger("context-engine/per-message-summary");

export const PER_MESSAGE_SUMMARY_ENGINE_ID = "per-message-summary";

const DEFAULT_COMPACTED_CONTEXT_SIZE = 12_000;
const DEFAULT_COMPACTION_TRIGGER_THRESHOLD = 18_000;
const DEFAULT_RETAIN_SUMMARY_COUNT = 200;
const RETRY_BASE_DELAY_MS = 15_000;
const RETRY_MAX_DELAY_MS = 5 * 60_000;
const MAX_SUMMARY_ATTEMPTS = 5;
const RUNNING_JOB_STALE_MS = 5 * 60_000;
const MAX_WORKER_JOBS_PER_KICK = 3;
const MAX_SUMMARY_CHARS = 4_000;
const SUMMARY_TIMEOUT_MS = 90_000;
const DEFAULT_PROMPT_FILE_NAME = "per-message-summary.worker.prompt.md";
const DEFAULT_SUMMARY_PROMPT_FALLBACK = [
  "You are a context compression assistant.",
  "Summarize exactly one chat message for future context reconstruction.",
  'Return STRICT JSON only with this schema: {"summary":"..."}.',
  "Do not include markdown fences.",
  "Preserve identifiers exactly as-is (IDs, URLs, hashes, filenames, commands).",
  "Keep the summary concise and factual.",
].join("\n");

const activeSessionWorkers = new Set<string>();

export type SummaryStatus = "pending" | "processing" | "done" | "failed";
export type JobStatus = "queued" | "running" | "done" | "failed";

export type PerMessageSummarySettings = {
  provider?: string;
  model?: string;
  summaryPromptFile?: string;
  compactedContextSize: number;
  compactionTriggerThreshold: number;
  retainSummaryCount: number;
  configErrors: string[];
};

export type PerMessageSummaryRuntimeHint = {
  config?: OpenClawConfig;
  workspaceDir?: string;
  agentDir?: string;
  provider?: string;
  model?: string;
};

type StorageTarget = {
  agentDir: string;
  dbPath: string;
  workerSessionFile: string;
  workspaceDir: string;
};

export type PerMessageSummaryMessageRow = {
  message_key: string;
  message_index: number;
  role: string;
  summary_content: string | null;
  summary_status: SummaryStatus;
  raw_token_estimate: number;
  summary_token_estimate: number | null;
  replaced_in_context: number;
  pruned_in_context: number;
};

type JobRow = {
  job_id: string;
  message_key: string;
  attempt_count: number;
};

export type PerMessageSummaryDequeuedTask = {
  jobId: string;
  messageKey: string;
  attemptCount: number;
  role: string;
  rawContent: string;
};

export type PerMessageSummarySearchHit = {
  messageKey: string;
  messageIndex: number;
  role: string;
  rawContent: string;
  summaryContent?: string;
  summaryStatus: SummaryStatus;
};

function normalizePositiveInt(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  const normalized = Math.floor(value);
  return normalized > 0 ? normalized : fallback;
}

function normalizeNonNegativeInt(value: unknown, fallback: number): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fallback;
  }
  const normalized = Math.floor(value);
  return normalized >= 0 ? normalized : fallback;
}

function normalizeNonEmptyString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

export function resolvePerMessageSummarySettings(
  config?: OpenClawConfig,
): PerMessageSummarySettings {
  const rawEntry = config?.plugins?.entries?.[PER_MESSAGE_SUMMARY_ENGINE_ID]?.config;
  const entry = rawEntry && typeof rawEntry === "object" ? rawEntry : {};

  const provider = normalizeNonEmptyString(entry.provider);
  const model = normalizeNonEmptyString(entry.model);
  const summaryPromptFile = normalizeNonEmptyString(entry.summaryPromptFile);
  const compactedContextSize = normalizePositiveInt(
    entry.compactedContextSize,
    DEFAULT_COMPACTED_CONTEXT_SIZE,
  );
  let compactionTriggerThreshold = normalizePositiveInt(
    entry.compactionTriggerThreshold,
    DEFAULT_COMPACTION_TRIGGER_THRESHOLD,
  );
  const retainSummaryCount = normalizeNonNegativeInt(
    entry.retainSummaryCount,
    DEFAULT_RETAIN_SUMMARY_COUNT,
  );

  const configErrors: string[] = [];
  if (!provider) {
    configErrors.push("provider is required");
  }
  if (!model) {
    configErrors.push("model is required");
  }
  if (compactionTriggerThreshold <= compactedContextSize) {
    configErrors.push("compactionTriggerThreshold must be greater than compactedContextSize");
    compactionTriggerThreshold = compactedContextSize + 1;
  }

  return {
    provider,
    model,
    summaryPromptFile,
    compactedContextSize,
    compactionTriggerThreshold,
    retainSummaryCount,
    configErrors,
  };
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }
  const rec = value as Record<string, unknown>;
  const keys = Object.keys(rec).toSorted();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${stableStringify(rec[key])}`).join(",")}}`;
}

function stableStringifyFallback(value: unknown): string {
  try {
    return stableStringify(value);
  } catch {
    if (value === null || value === undefined) {
      return `${value}`;
    }
    if (typeof value === "string") {
      return value;
    }
    if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") {
      return `${value}`;
    }
    return Object.prototype.toString.call(value);
  }
}

function messageSignature(message: AgentMessage): string {
  return stableStringifyFallback({
    role: message.role,
    content: (message as { content?: unknown }).content,
    timestamp: (message as { timestamp?: unknown }).timestamp,
  });
}

export function buildPerMessageSummaryMessageKey(sessionId: string, message: AgentMessage): string {
  const digest = createHash("sha256").update(messageSignature(message)).digest("hex");
  return `pms:${sessionId}:${digest}`;
}

function extractMessageText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return stableStringifyFallback(content);
  }

  const lines: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const record = block as { type?: unknown; text?: unknown };
    if (typeof record.text === "string") {
      lines.push(record.text);
      continue;
    }
    if (typeof record.type === "string") {
      lines.push(`[${record.type}]`);
    }
  }
  return lines.join("\n");
}

function renderMessageForStorage(message: AgentMessage): string {
  const text = extractMessageText((message as { content?: unknown }).content).trim();
  if (text.length > 0) {
    return text;
  }
  return "(empty message)";
}

function estimateMessageTokens(message: AgentMessage): number {
  const estimate = estimateMessagesTokens([message]);
  if (!Number.isFinite(estimate) || estimate <= 0) {
    return 1;
  }
  return Math.max(1, Math.floor(estimate));
}

export function buildPerMessageSummaryReplacementContent(role: string, summary: string): string {
  return `[summary of earlier ${role} message]\n${summary}`;
}

function normalizeSummaryText(raw: string): string {
  const squashed = raw.replace(/\r\n/g, "\n").trim();
  if (squashed.length <= MAX_SUMMARY_CHARS) {
    return squashed;
  }
  return `${squashed.slice(0, MAX_SUMMARY_CHARS).trim()}...`;
}

function parseSummaryFromJsonCandidate(text: string): string | undefined {
  try {
    const parsed = JSON.parse(text) as { summary?: unknown };
    const summary = typeof parsed.summary === "string" ? normalizeSummaryText(parsed.summary) : "";
    return summary.length > 0 ? summary : undefined;
  } catch {
    return undefined;
  }
}

function extractSummaryFromModelOutput(output: string): string | undefined {
  const trimmed = output.trim();
  if (!trimmed) {
    return undefined;
  }

  const direct = parseSummaryFromJsonCandidate(trimmed);
  if (direct) {
    return direct;
  }

  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) {
    const fromFence = parseSummaryFromJsonCandidate(fenced[1]);
    if (fromFence) {
      return fromFence;
    }
  }

  const objectStart = trimmed.indexOf("{");
  const objectEnd = trimmed.lastIndexOf("}");
  if (objectStart >= 0 && objectEnd > objectStart) {
    const fromSlice = parseSummaryFromJsonCandidate(trimmed.slice(objectStart, objectEnd + 1));
    if (fromSlice) {
      return fromSlice;
    }
  }

  return undefined;
}

function resolveBackoffDelayMs(attemptCount: number): number {
  const exponent = Math.max(0, attemptCount - 1);
  const delay = RETRY_BASE_DELAY_MS * 2 ** exponent;
  return Math.min(RETRY_MAX_DELAY_MS, delay);
}

function toLegacyContextEngineConfig(config?: OpenClawConfig): OpenClawConfig | undefined {
  if (!config) {
    return config;
  }
  return {
    ...config,
    plugins: {
      ...config.plugins,
      slots: {
        ...config.plugins?.slots,
        contextEngine: "legacy",
      },
    },
  };
}

function nowTs(): number {
  return Date.now();
}

function resolvePromptPath(promptFile: string, workspaceDir?: string): string {
  if (path.isAbsolute(promptFile)) {
    return promptFile;
  }
  const baseDir = workspaceDir?.trim() || (typeof process.cwd === "function" ? process.cwd() : ".");
  return path.resolve(baseDir, promptFile);
}

function readPromptFileMaybe(filePath: string): string | undefined {
  try {
    const prompt = fs.readFileSync(filePath, "utf8").trim();
    return prompt.length > 0 ? prompt : undefined;
  } catch {
    return undefined;
  }
}

function seedSessionPromptFile(params: {
  sessionPromptPath: string;
  workspaceDir: string;
}): string | undefined {
  const templateCandidates = [
    path.resolve(
      params.workspaceDir,
      "dist",
      "context-engine",
      "per-message-summary",
      "summary-worker-prompt.md",
    ),
    path.resolve(
      params.workspaceDir,
      "src",
      "context-engine",
      "per-message-summary",
      "summary-worker-prompt.md",
    ),
  ];

  for (const candidate of templateCandidates) {
    const template = readPromptFileMaybe(candidate);
    if (!template) {
      continue;
    }
    fs.mkdirSync(path.dirname(params.sessionPromptPath), { recursive: true });
    try {
      fs.writeFileSync(params.sessionPromptPath, `${template}\n`, { flag: "wx" });
    } catch {
      // Another process may create this file first.
    }
    return readPromptFileMaybe(params.sessionPromptPath) ?? template;
  }

  return undefined;
}

function buildSummaryPromptFromTemplate(params: {
  template: string;
  role: string;
  rawContent: string;
}): string {
  const template = params.template.trim();
  if (template.includes("{{role}}") || template.includes("{{message}}")) {
    return template
      .replaceAll("{{role}}", params.role)
      .replaceAll("{{message}}", params.rawContent);
  }

  return [
    template,
    "",
    `Role: ${params.role}`,
    "Message:",
    "<message>",
    params.rawContent,
    "</message>",
  ].join("\n");
}

export function normalizePerMessageSummaryRuntimeHint(
  runtimeContext?: ContextEngineRuntimeContext,
): PerMessageSummaryRuntimeHint {
  if (!runtimeContext || typeof runtimeContext !== "object") {
    return {};
  }
  const rec = runtimeContext as Record<string, unknown>;
  return {
    config: rec.config as OpenClawConfig | undefined,
    workspaceDir: typeof rec.workspaceDir === "string" ? rec.workspaceDir : undefined,
    agentDir: typeof rec.agentDir === "string" ? rec.agentDir : undefined,
    provider: typeof rec.provider === "string" ? rec.provider : undefined,
    model: typeof rec.modelId === "string" ? rec.modelId : undefined,
  };
}

export function normalizePerMessageSummaryTokenCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : undefined;
}

function resolveStorageTarget(params: {
  sessionKey?: string;
  runtimeHint?: PerMessageSummaryRuntimeHint;
  engineConfig?: OpenClawConfig;
}): StorageTarget {
  const workspaceDir =
    params.runtimeHint?.workspaceDir?.trim() ||
    (typeof process.cwd === "function" ? process.cwd() : ".");

  const runtimeAgentDir = params.runtimeHint?.agentDir?.trim();
  let agentDir = runtimeAgentDir;
  if (!agentDir && params.sessionKey && params.engineConfig) {
    const agentId = resolveAgentIdFromSessionKey(params.sessionKey);
    agentDir = resolveAgentDir(params.engineConfig, agentId);
  }
  if (!agentDir) {
    agentDir = resolveOpenClawAgentDir();
  }

  const storeDir = path.join(agentDir, "context-engines");
  const dbPath = path.join(storeDir, "per-message-summary.sqlite");
  const workerSessionFile = path.join(storeDir, "per-message-summary.worker.sessions.jsonl");

  return {
    agentDir,
    dbPath,
    workerSessionFile,
    workspaceDir,
  };
}

function ensureSchema(db: import("node:sqlite").DatabaseSync): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS pms_messages (
      session_id TEXT NOT NULL,
      message_key TEXT PRIMARY KEY,
      message_index INTEGER NOT NULL,
      role TEXT NOT NULL,
      raw_content TEXT NOT NULL,
      raw_token_estimate INTEGER NOT NULL,
      summary_content TEXT,
      summary_token_estimate INTEGER,
      summary_status TEXT NOT NULL,
      replaced_in_context INTEGER NOT NULL DEFAULT 0,
      pruned_in_context INTEGER NOT NULL DEFAULT 0,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS pms_summary_jobs (
      job_id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      message_key TEXT NOT NULL,
      status TEXT NOT NULL,
      attempt_count INTEGER NOT NULL DEFAULT 0,
      next_run_at INTEGER NOT NULL,
      last_error TEXT,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL,
      UNIQUE(session_id, message_key)
    );
  `);

  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_pms_messages_session_index ON pms_messages(session_id, message_index);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_pms_messages_session_status_index ON pms_messages(session_id, summary_status, message_index);",
  );
  db.exec(
    "CREATE INDEX IF NOT EXISTS idx_pms_jobs_session_status_next_run ON pms_summary_jobs(session_id, status, next_run_at);",
  );

  // Best effort FTS table creation; continue if unavailable.
  try {
    db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS pms_summary_fts USING fts5(
        session_id,
        message_key UNINDEXED,
        summary_content,
        raw_content
      );
    `);
  } catch (err) {
    log.warn(`per-message-summary: FTS5 unavailable; search index disabled: ${String(err)}`);
  }
}

function tryUpdateFts(params: {
  db: import("node:sqlite").DatabaseSync;
  sessionId: string;
  messageKey: string;
  summaryContent: string;
  rawContent: string;
}): void {
  try {
    params.db.prepare("DELETE FROM pms_summary_fts WHERE message_key = ?").run(params.messageKey);
    params.db
      .prepare(
        "INSERT INTO pms_summary_fts (session_id, message_key, summary_content, raw_content) VALUES (?, ?, ?, ?)",
      )
      .run(params.sessionId, params.messageKey, params.summaryContent, params.rawContent);
  } catch {
    // FTS can be unavailable on some runtimes; silently skip search index updates.
  }
}

function estimateCurrentContextTokens(
  db: import("node:sqlite").DatabaseSync,
  sessionId: string,
): number {
  const row = db
    .prepare(
      `SELECT
         COALESCE(SUM(
           CASE
             WHEN pruned_in_context = 1 THEN 0
             WHEN replaced_in_context = 1 AND summary_status = 'done' THEN COALESCE(summary_token_estimate, raw_token_estimate)
             ELSE raw_token_estimate
           END
         ), 0) AS total
       FROM pms_messages
       WHERE session_id = ?`,
    )
    .get(sessionId) as { total?: number } | undefined;

  const total = row?.total;
  if (typeof total !== "number" || !Number.isFinite(total) || total < 0) {
    return 0;
  }
  return Math.floor(total);
}

async function summarizeMessageWithEmbeddedRun(params: {
  provider: string;
  model: string;
  promptTemplate: string;
  role: string;
  rawContent: string;
  storage: StorageTarget;
  config?: OpenClawConfig;
}): Promise<string> {
  const prompt = buildSummaryPromptFromTemplate({
    template: params.promptTemplate,
    role: params.role,
    rawContent: params.rawContent,
  });

  const result = await runEmbeddedPiAgent({
    sessionId: `pms-worker-${randomUUID()}`,
    runId: `pms-${randomUUID()}`,
    sessionFile: params.storage.workerSessionFile,
    workspaceDir: params.storage.workspaceDir,
    agentDir: params.storage.agentDir,
    prompt,
    provider: params.provider,
    model: params.model,
    config: toLegacyContextEngineConfig(params.config),
    timeoutMs: SUMMARY_TIMEOUT_MS,
    disableTools: true,
    thinkLevel: "off",
    reasoningLevel: "off",
    trigger: "memory",
  });

  const payloadText = (result.payloads ?? [])
    .filter((payload) => payload.isError !== true)
    .map((payload) => payload.text ?? "")
    .join("\n")
    .trim();

  if (!payloadText) {
    const errorText = result.meta?.error?.message?.trim();
    throw new Error(
      errorText
        ? `summary run returned empty output: ${errorText}`
        : "summary run returned empty output",
    );
  }

  const summary = extractSummaryFromModelOutput(payloadText);
  if (!summary) {
    throw new Error("summary run did not return valid JSON summary");
  }

  return summary;
}

function buildFtsQuery(raw: string): string | undefined {
  const terms = raw
    .trim()
    .split(/\s+/)
    .map((term) => term.replace(/["']/g, ""))
    .filter((term) => term.length > 1)
    .slice(0, 8);
  if (terms.length === 0) {
    return undefined;
  }
  return terms.map((term) => `"${term}"`).join(" AND ");
}

export class PerMessageSummaryManager {
  private readonly settings: PerMessageSummarySettings;

  constructor(private readonly runtimeConfig?: OpenClawConfig) {
    this.settings = resolvePerMessageSummarySettings(runtimeConfig);
  }

  getSettings(): PerMessageSummarySettings {
    return this.settings;
  }

  private withDb<T>(
    params: { sessionKey?: string; runtimeHint?: PerMessageSummaryRuntimeHint },
    fn: (db: import("node:sqlite").DatabaseSync, storage: StorageTarget) => Promise<T> | T,
  ): Promise<T> {
    const storage = resolveStorageTarget({
      sessionKey: params.sessionKey,
      runtimeHint: params.runtimeHint,
      engineConfig: this.runtimeConfig,
    });

    fs.mkdirSync(path.dirname(storage.dbPath), { recursive: true });
    fs.mkdirSync(path.dirname(storage.workerSessionFile), { recursive: true });

    const { DatabaseSync } = requireNodeSqlite();
    const db = new DatabaseSync(storage.dbPath);
    db.exec("PRAGMA journal_mode=WAL;");
    db.exec("PRAGMA synchronous=NORMAL;");
    db.exec("PRAGMA busy_timeout=5000;");

    ensureSchema(db);

    const run = async () => await fn(db, storage);
    return run().finally(() => {
      db.close();
    });
  }

  private resolveWorkerModel(runtimeHint?: PerMessageSummaryRuntimeHint): {
    provider?: string;
    model?: string;
  } {
    const provider = this.settings.provider ?? normalizeNonEmptyString(runtimeHint?.provider);
    const model = this.settings.model ?? normalizeNonEmptyString(runtimeHint?.model);
    return { provider, model };
  }

  private resolveSummaryPromptTemplate(params: { storage: StorageTarget }): string {
    const configuredPromptFile = this.settings.summaryPromptFile;
    if (configuredPromptFile) {
      const configuredPath = resolvePromptPath(configuredPromptFile, params.storage.workspaceDir);
      const configuredPrompt = readPromptFileMaybe(configuredPath);
      if (configuredPrompt) {
        return configuredPrompt;
      }
      log.warn(
        `per-message-summary: configured summaryPromptFile is unreadable: ${configuredPath}; using fallback prompt source`,
      );
    }

    const sessionPromptPath = path.join(
      path.dirname(params.storage.dbPath),
      DEFAULT_PROMPT_FILE_NAME,
    );
    const sessionPrompt =
      readPromptFileMaybe(sessionPromptPath) ??
      seedSessionPromptFile({
        sessionPromptPath,
        workspaceDir: params.storage.workspaceDir,
      });
    if (sessionPrompt) {
      return sessionPrompt;
    }

    log.warn("per-message-summary: prompt markdown not found, falling back to built-in template");
    return DEFAULT_SUMMARY_PROMPT_FALLBACK;
  }

  async ensureSession(params: {
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
  }): Promise<void> {
    await this.withDb(params, async () => undefined);
  }

  kickWorker(params: {
    sessionId: string;
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
  }): void {
    const workerKey = `${params.sessionId}:${params.sessionKey ?? ""}`;
    if (activeSessionWorkers.has(workerKey)) {
      return;
    }
    activeSessionWorkers.add(workerKey);

    queueMicrotask(() => {
      void this.processQueuedJobs({
        sessionId: params.sessionId,
        sessionKey: params.sessionKey,
        runtimeHint: params.runtimeHint,
        maxJobs: MAX_WORKER_JOBS_PER_KICK,
      }).finally(() => {
        activeSessionWorkers.delete(workerKey);
      });
    });
  }

  async processQueuedJobs(params: {
    sessionId: string;
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
    maxJobs?: number;
  }): Promise<void> {
    const maxJobs = Math.max(1, params.maxJobs ?? MAX_WORKER_JOBS_PER_KICK);
    const { provider, model } = this.resolveWorkerModel(params.runtimeHint);
    if (!provider || !model) {
      if (this.settings.configErrors.length > 0) {
        log.warn(
          `per-message-summary: worker paused due to config errors: ${this.settings.configErrors.join(", ")}`,
        );
      }
      return;
    }

    const storage = resolveStorageTarget({
      sessionKey: params.sessionKey,
      runtimeHint: params.runtimeHint,
      engineConfig: this.runtimeConfig,
    });
    const promptTemplate = this.resolveSummaryPromptTemplate({
      storage,
    });

    for (let i = 0; i < maxJobs; i += 1) {
      const claimed = await this.dequeueTask({
        sessionId: params.sessionId,
        sessionKey: params.sessionKey,
        runtimeHint: params.runtimeHint,
      });
      if (!claimed) {
        return;
      }

      try {
        const summary = await this.withDb(
          { sessionKey: params.sessionKey, runtimeHint: params.runtimeHint },
          async (_db, storage) => {
            return await summarizeMessageWithEmbeddedRun({
              provider,
              model,
              promptTemplate,
              role: claimed.role,
              rawContent: claimed.rawContent,
              storage,
              config: params.runtimeHint?.config ?? this.runtimeConfig,
            });
          },
        );

        const summaryTokenEstimate = estimateMessageTokens({
          role: claimed.role,
          content: summary,
          timestamp: nowTs(),
        } as AgentMessage);

        await this.withDb(
          { sessionKey: params.sessionKey, runtimeHint: params.runtimeHint },
          async (db) => {
            const ts = nowTs();
            db.prepare(
              `UPDATE pms_messages
               SET summary_content = ?,
                   summary_token_estimate = ?,
                   summary_status = 'done',
                   updated_at = ?
               WHERE session_id = ? AND message_key = ?`,
            ).run(summary, summaryTokenEstimate, ts, params.sessionId, claimed.messageKey);

            db.prepare(
              `UPDATE pms_summary_jobs
               SET status = 'done',
                   next_run_at = ?,
                   last_error = NULL,
                   updated_at = ?
               WHERE job_id = ?`,
            ).run(ts, ts, claimed.jobId);

            tryUpdateFts({
              db,
              sessionId: params.sessionId,
              messageKey: claimed.messageKey,
              summaryContent: summary,
              rawContent: claimed.rawContent,
            });
          },
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        const delay = resolveBackoffDelayMs(claimed.attemptCount);
        const nextRunAt = nowTs() + delay;

        await this.withDb(
          { sessionKey: params.sessionKey, runtimeHint: params.runtimeHint },
          async (db) => {
            const ts = nowTs();
            db.prepare(
              `UPDATE pms_summary_jobs
               SET status = 'failed',
                   next_run_at = ?,
                   last_error = ?,
                   updated_at = ?
               WHERE job_id = ?`,
            ).run(nextRunAt, message.slice(0, 1000), ts, claimed.jobId);

            db.prepare(
              `UPDATE pms_messages
               SET summary_status = 'failed',
                   updated_at = ?
               WHERE session_id = ? AND message_key = ?`,
            ).run(ts, params.sessionId, claimed.messageKey);
          },
        );

        log.warn(
          `per-message-summary: worker failed for message ${claimed.messageKey.slice(0, 16)} (attempt ${claimed.attemptCount}/${MAX_SUMMARY_ATTEMPTS}): ${message}`,
        );
      }
    }
  }

  async dequeueTask(params: {
    sessionId: string;
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
  }): Promise<PerMessageSummaryDequeuedTask | null> {
    const now = nowTs();

    return await this.withDb(
      { sessionKey: params.sessionKey, runtimeHint: params.runtimeHint },
      async (db) => {
        const staleCutoff = now - RUNNING_JOB_STALE_MS;
        db.prepare(
          "UPDATE pms_summary_jobs SET status = 'queued', updated_at = ? WHERE session_id = ? AND status = 'running' AND updated_at < ?",
        ).run(now, params.sessionId, staleCutoff);

        const row = db
          .prepare(
            `SELECT job_id, message_key, attempt_count
             FROM pms_summary_jobs
             WHERE session_id = ?
               AND status IN ('queued', 'failed')
               AND attempt_count < ?
               AND next_run_at <= ?
             ORDER BY next_run_at ASC, created_at ASC
             LIMIT 1`,
          )
          .get(params.sessionId, MAX_SUMMARY_ATTEMPTS, now) as JobRow | undefined;

        if (!row) {
          return null;
        }

        const runResult = db
          .prepare(
            "UPDATE pms_summary_jobs SET status = 'running', attempt_count = attempt_count + 1, updated_at = ? WHERE job_id = ? AND status IN ('queued', 'failed')",
          )
          .run(now, row.job_id) as { changes?: number };

        if ((runResult.changes ?? 0) < 1) {
          return null;
        }

        const nextAttempt = (row.attempt_count ?? 0) + 1;
        db.prepare(
          "UPDATE pms_messages SET summary_status = 'processing', updated_at = ? WHERE session_id = ? AND message_key = ? AND summary_status != 'done'",
        ).run(now, params.sessionId, row.message_key);

        const message = db
          .prepare(
            "SELECT role, raw_content FROM pms_messages WHERE session_id = ? AND message_key = ? LIMIT 1",
          )
          .get(params.sessionId, row.message_key) as
          | { role?: string; raw_content?: string }
          | undefined;

        if (
          !message ||
          typeof message.raw_content !== "string" ||
          typeof message.role !== "string"
        ) {
          db.prepare(
            "UPDATE pms_summary_jobs SET status = 'failed', next_run_at = ?, last_error = ?, updated_at = ? WHERE job_id = ?",
          ).run(now + RETRY_MAX_DELAY_MS, "message row missing", now, row.job_id);
          db.prepare(
            "UPDATE pms_messages SET summary_status = 'failed', updated_at = ? WHERE session_id = ? AND message_key = ?",
          ).run(now, params.sessionId, row.message_key);
          return null;
        }

        return {
          jobId: row.job_id,
          messageKey: row.message_key,
          attemptCount: nextAttempt,
          role: message.role,
          rawContent: message.raw_content,
        };
      },
    );
  }

  async enqueueTask(params: {
    sessionId: string;
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
    message: AgentMessage;
    messageIndex: number;
  }): Promise<boolean> {
    const inserted = await this.enqueueMessages({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      runtimeHint: params.runtimeHint,
      messages: [params.message],
      startIndex: params.messageIndex,
    });
    return inserted > 0;
  }

  async enqueueMessages(params: {
    sessionId: string;
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
    messages: AgentMessage[];
    startIndex: number;
  }): Promise<number> {
    if (params.messages.length === 0) {
      return 0;
    }

    return await this.withDb(
      { sessionKey: params.sessionKey, runtimeHint: params.runtimeHint },
      async (db) => {
        const ts = nowTs();
        let ingestedCount = 0;

        const insertMessage = db.prepare(
          `INSERT INTO pms_messages (
             session_id,
             message_key,
             message_index,
             role,
             raw_content,
             raw_token_estimate,
             summary_status,
             created_at,
             updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
           ON CONFLICT(message_key) DO UPDATE SET
             message_index = MIN(message_index, excluded.message_index),
             role = excluded.role,
             raw_content = excluded.raw_content,
             raw_token_estimate = excluded.raw_token_estimate,
             updated_at = excluded.updated_at`,
        );

        const insertJob = db.prepare(
          `INSERT INTO pms_summary_jobs (
             job_id,
             session_id,
             message_key,
             status,
             attempt_count,
             next_run_at,
             last_error,
             created_at,
             updated_at
           ) VALUES (?, ?, ?, 'queued', 0, ?, NULL, ?, ?)
           ON CONFLICT(session_id, message_key) DO NOTHING`,
        );

        for (let i = 0; i < params.messages.length; i += 1) {
          const message = params.messages[i];
          const messageKey = buildPerMessageSummaryMessageKey(params.sessionId, message);
          const messageIndex = params.startIndex + i;
          const role = typeof message.role === "string" ? message.role : "unknown";
          const rawContent = renderMessageForStorage(message);
          const rawTokenEstimate = estimateMessageTokens(message);
          const jobId = createHash("sha256")
            .update(`${params.sessionId}:${messageKey}`)
            .digest("hex")
            .slice(0, 24);

          insertMessage.run(
            params.sessionId,
            messageKey,
            messageIndex,
            role,
            rawContent,
            rawTokenEstimate,
            ts,
            ts,
          );
          const jobResult = insertJob.run(jobId, params.sessionId, messageKey, ts, ts, ts) as {
            changes?: number;
          };
          if ((jobResult.changes ?? 0) > 0) {
            ingestedCount += 1;
          }
        }

        return ingestedCount;
      },
    );
  }

  async getMessageRowsByKeys(params: {
    sessionId: string;
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
    messageKeys: string[];
  }): Promise<Map<string, PerMessageSummaryMessageRow>> {
    if (params.messageKeys.length === 0) {
      return new Map();
    }

    return await this.withDb(
      { sessionKey: params.sessionKey, runtimeHint: params.runtimeHint },
      async (db) => {
        const placeholders = params.messageKeys.map(() => "?").join(",");
        const rows = db
          .prepare(
            `SELECT
               message_key,
               message_index,
               role,
               summary_content,
               summary_status,
               raw_token_estimate,
               summary_token_estimate,
               replaced_in_context,
               pruned_in_context
             FROM pms_messages
             WHERE session_id = ? AND message_key IN (${placeholders})`,
          )
          .all(params.sessionId, ...params.messageKeys) as PerMessageSummaryMessageRow[];

        const byKey = new Map<string, PerMessageSummaryMessageRow>();
        for (const row of rows) {
          byKey.set(row.message_key, row);
        }
        return byKey;
      },
    );
  }

  private async applyRetainSummaryLimit(params: {
    db: import("node:sqlite").DatabaseSync;
    sessionId: string;
    retainCount: number;
  }): Promise<{ pruneChanged: number }> {
    const rows = params.db
      .prepare(
        `SELECT message_key, pruned_in_context
         FROM pms_messages
         WHERE session_id = ?
           AND summary_status = 'done'
           AND replaced_in_context = 1
         ORDER BY message_index DESC`,
      )
      .all(params.sessionId) as Array<{ message_key: string; pruned_in_context: number }>;

    let pruneChanged = 0;
    const keepSet = new Set<string>();
    for (let i = 0; i < rows.length && i < params.retainCount; i += 1) {
      keepSet.add(rows[i].message_key);
    }

    const updatePruned = params.db.prepare(
      "UPDATE pms_messages SET pruned_in_context = ?, updated_at = ? WHERE session_id = ? AND message_key = ?",
    );

    const ts = nowTs();
    for (const row of rows) {
      const nextPruned = keepSet.has(row.message_key) ? 0 : 1;
      if (nextPruned === row.pruned_in_context) {
        continue;
      }
      updatePruned.run(nextPruned, ts, params.sessionId, row.message_key);
      pruneChanged += 1;
    }

    return { pruneChanged };
  }

  async compactSession(params: {
    sessionId: string;
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
    force?: boolean;
    currentTokenCount?: number;
  }): Promise<CompactResult> {
    const tokenCountHint = normalizePerMessageSummaryTokenCount(params.currentTokenCount);

    return await this.withDb(
      { sessionKey: params.sessionKey, runtimeHint: params.runtimeHint },
      async (db) => {
        const beforeTokens = tokenCountHint ?? estimateCurrentContextTokens(db, params.sessionId);
        const triggerThreshold = this.settings.compactionTriggerThreshold;
        const targetSize = this.settings.compactedContextSize;
        const retainCount = this.settings.retainSummaryCount;

        const shouldCompact = params.force === true || beforeTokens >= triggerThreshold;
        if (!shouldCompact) {
          return {
            ok: true,
            compacted: false,
            reason: `below threshold (${beforeTokens}/${triggerThreshold})`,
            result: {
              tokensBefore: beforeTokens,
              tokensAfter: beforeTokens,
            },
          } satisfies CompactResult;
        }

        let currentTokens = beforeTokens;
        let replacedCount = 0;

        const pickNext = db.prepare(
          `SELECT message_key, raw_token_estimate, COALESCE(summary_token_estimate, raw_token_estimate) AS summary_tokens
           FROM pms_messages
           WHERE session_id = ?
             AND summary_status = 'done'
             AND replaced_in_context = 0
             AND pruned_in_context = 0
           ORDER BY message_index ASC
           LIMIT 1`,
        );
        const markReplaced = db.prepare(
          `UPDATE pms_messages
           SET replaced_in_context = 1,
               pruned_in_context = 0,
               updated_at = ?
           WHERE session_id = ? AND message_key = ?`,
        );

        while (currentTokens > targetSize) {
          const row = pickNext.get(params.sessionId) as
            | { message_key?: string; raw_token_estimate?: number; summary_tokens?: number }
            | undefined;
          if (!row?.message_key) {
            break;
          }

          const rawTokens =
            typeof row.raw_token_estimate === "number" && Number.isFinite(row.raw_token_estimate)
              ? Math.max(1, Math.floor(row.raw_token_estimate))
              : 1;
          const summaryTokens =
            typeof row.summary_tokens === "number" && Number.isFinite(row.summary_tokens)
              ? Math.max(1, Math.floor(row.summary_tokens))
              : rawTokens;

          markReplaced.run(nowTs(), params.sessionId, row.message_key);
          currentTokens = Math.max(0, currentTokens - rawTokens + summaryTokens);
          replacedCount += 1;
        }

        const { pruneChanged } = await this.applyRetainSummaryLimit({
          db,
          sessionId: params.sessionId,
          retainCount,
        });

        const afterTokens = estimateCurrentContextTokens(db, params.sessionId);
        const compacted = replacedCount > 0 || pruneChanged > 0;

        return {
          ok: true,
          compacted,
          ...(compacted ? {} : { reason: "no eligible summaries to replace or prune" }),
          result: {
            tokensBefore: beforeTokens,
            tokensAfter: afterTokens,
            details: {
              replacedCount,
              pruneChanged,
              targetSize,
              triggerThreshold,
              retainSummaryCount: retainCount,
            },
          },
        } satisfies CompactResult;
      },
    );
  }

  async searchSummary(params: {
    sessionId: string;
    sessionKey?: string;
    runtimeHint?: PerMessageSummaryRuntimeHint;
    query: string;
    limit?: number;
  }): Promise<PerMessageSummarySearchHit[]> {
    const query = params.query.trim();
    if (!query) {
      return [];
    }

    const limit = Math.max(1, Math.min(100, Math.floor(params.limit ?? 20)));
    return await this.withDb(
      { sessionKey: params.sessionKey, runtimeHint: params.runtimeHint },
      async (db) => {
        const ftsQuery = buildFtsQuery(query);
        if (ftsQuery) {
          try {
            const ftsRows = db
              .prepare(
                `SELECT
                   m.message_key,
                   m.message_index,
                   m.role,
                   m.raw_content,
                   m.summary_content,
                   m.summary_status
                 FROM pms_summary_fts f
                 JOIN pms_messages m ON m.message_key = f.message_key
                 WHERE f.session_id = ?
                   AND pms_summary_fts MATCH ?
                 ORDER BY bm25(pms_summary_fts), m.message_index DESC
                 LIMIT ?`,
              )
              .all(params.sessionId, ftsQuery, limit) as Array<{
              message_key: string;
              message_index: number;
              role: string;
              raw_content: string;
              summary_content: string | null;
              summary_status: SummaryStatus;
            }>;

            return ftsRows.map((row) => ({
              messageKey: row.message_key,
              messageIndex: row.message_index,
              role: row.role,
              rawContent: row.raw_content,
              ...(typeof row.summary_content === "string"
                ? { summaryContent: row.summary_content }
                : {}),
              summaryStatus: row.summary_status,
            }));
          } catch {
            // FTS can be unavailable; fallback to LIKE search below.
          }
        }

        const likeRows = db
          .prepare(
            `SELECT
               message_key,
               message_index,
               role,
               raw_content,
               summary_content,
               summary_status
             FROM pms_messages
             WHERE session_id = ?
               AND raw_content LIKE ?
             ORDER BY message_index DESC
             LIMIT ?`,
          )
          .all(params.sessionId, `%${query}%`, limit) as Array<{
          message_key: string;
          message_index: number;
          role: string;
          raw_content: string;
          summary_content: string | null;
          summary_status: SummaryStatus;
        }>;

        return likeRows.map((row) => ({
          messageKey: row.message_key,
          messageIndex: row.message_index,
          role: row.role,
          rawContent: row.raw_content,
          ...(typeof row.summary_content === "string"
            ? { summaryContent: row.summary_content }
            : {}),
          summaryStatus: row.summary_status,
        }));
      },
    );
  }

  async dispose(): Promise<void> {
    // No persistent handles to dispose. DB connections are opened per operation.
  }
}
