import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { estimateMessagesTokens } from "../agents/compaction.js";
import type { OpenClawConfig } from "../config/config.js";
import {
  buildPerMessageSummaryMessageKey,
  buildPerMessageSummaryReplacementContent,
  normalizePerMessageSummaryRuntimeHint,
  normalizePerMessageSummaryTokenCount,
  PER_MESSAGE_SUMMARY_ENGINE_ID,
  type PerMessageSummarySearchHit,
  PerMessageSummaryManager,
} from "./per-message-summary-manager.js";
import { registerContextEngineForOwner } from "./registry.js";
import type {
  AssembleResult,
  BootstrapResult,
  CompactResult,
  ContextEngine,
  ContextEngineInfo,
  ContextEngineMaintenanceResult,
  ContextEngineRuntimeContext,
  IngestBatchResult,
  IngestResult,
} from "./types.js";

export { PER_MESSAGE_SUMMARY_ENGINE_ID };

export class PerMessageSummaryContextEngine implements ContextEngine {
  readonly info: ContextEngineInfo = {
    id: PER_MESSAGE_SUMMARY_ENGINE_ID,
    name: "Per-Message Summary Context Engine",
    version: "1.0.0",
    ownsCompaction: true,
  };

  private readonly manager: PerMessageSummaryManager;

  constructor(runtimeConfig?: OpenClawConfig) {
    this.manager = new PerMessageSummaryManager(runtimeConfig);
  }

  async bootstrap(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
  }): Promise<BootstrapResult> {
    await this.manager.ensureSession({ sessionKey: params.sessionKey });

    return {
      bootstrapped: true,
      importedMessages: 0,
    };
  }

  async maintain(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    runtimeContext?: ContextEngineRuntimeContext;
  }): Promise<ContextEngineMaintenanceResult> {
    const runtimeHint = normalizePerMessageSummaryRuntimeHint(params.runtimeContext);
    this.manager.kickWorker({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      runtimeHint,
    });

    return {
      changed: false,
      bytesFreed: 0,
      rewrittenEntries: 0,
    };
  }

  async ingest(params: {
    sessionId: string;
    sessionKey?: string;
    message: AgentMessage;
    isHeartbeat?: boolean;
  }): Promise<IngestResult> {
    const ingested = await this.manager.enqueueTask({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      message: params.message,
      messageIndex: 0,
    });

    this.manager.kickWorker({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
    });

    return { ingested };
  }

  async ingestBatch(params: {
    sessionId: string;
    sessionKey?: string;
    messages: AgentMessage[];
    isHeartbeat?: boolean;
  }): Promise<IngestBatchResult> {
    const ingestedCount = await this.manager.enqueueMessages({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      messages: params.messages,
      startIndex: 0,
    });

    this.manager.kickWorker({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
    });

    return { ingestedCount };
  }

  async afterTurn(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    messages: AgentMessage[];
    prePromptMessageCount: number;
    autoCompactionSummary?: string;
    isHeartbeat?: boolean;
    tokenBudget?: number;
    runtimeContext?: ContextEngineRuntimeContext;
  }): Promise<void> {
    const runtimeHint = normalizePerMessageSummaryRuntimeHint(params.runtimeContext);
    const startIndex = Math.max(0, params.prePromptMessageCount);
    const newMessages = params.messages.slice(startIndex);

    await this.manager.enqueueMessages({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      runtimeHint,
      messages: newMessages,
      startIndex,
    });

    this.manager.kickWorker({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      runtimeHint,
    });
  }

  async assemble(params: {
    sessionId: string;
    sessionKey?: string;
    messages: AgentMessage[];
    tokenBudget?: number;
    model?: string;
    prompt?: string;
  }): Promise<AssembleResult> {
    const messageKeys = params.messages.map((message) =>
      buildPerMessageSummaryMessageKey(params.sessionId, message),
    );

    const rowsByKey = await this.manager.getMessageRowsByKeys({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      messageKeys,
    });

    const output: AgentMessage[] = [];
    for (let i = 0; i < params.messages.length; i += 1) {
      const message = params.messages[i];
      const key = messageKeys[i];
      const row = rowsByKey.get(key);
      if (!row) {
        output.push(message);
        continue;
      }

      if (row.pruned_in_context === 1) {
        continue;
      }

      if (
        row.replaced_in_context === 1 &&
        row.summary_status === "done" &&
        typeof row.summary_content === "string" &&
        row.summary_content.trim().length > 0
      ) {
        output.push({
          ...(message as unknown as Record<string, unknown>),
          content: buildPerMessageSummaryReplacementContent(
            row.role || message.role,
            row.summary_content,
          ),
        } as AgentMessage);
        continue;
      }

      output.push(message);
    }

    this.manager.kickWorker({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
    });

    return {
      messages: output,
      estimatedTokens: estimateMessagesTokens(output),
    };
  }

  async compact(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    currentTokenCount?: number;
    compactionTarget?: "budget" | "threshold";
    customInstructions?: string;
    runtimeContext?: ContextEngineRuntimeContext;
  }): Promise<CompactResult> {
    const runtimeHint = normalizePerMessageSummaryRuntimeHint(params.runtimeContext);
    const currentTokenCount =
      normalizePerMessageSummaryTokenCount(params.currentTokenCount) ??
      normalizePerMessageSummaryTokenCount(
        (params.runtimeContext as { currentTokenCount?: unknown } | undefined)?.currentTokenCount,
      );

    const result = await this.manager.compactSession({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      runtimeHint,
      force: params.force,
      currentTokenCount,
    });

    this.manager.kickWorker({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      runtimeHint,
    });

    return result;
  }

  async searchSummary(params: {
    sessionId: string;
    sessionKey?: string;
    query: string;
    limit?: number;
    runtimeContext?: ContextEngineRuntimeContext;
  }): Promise<PerMessageSummarySearchHit[]> {
    const runtimeHint = normalizePerMessageSummaryRuntimeHint(params.runtimeContext);
    return await this.manager.searchSummary({
      sessionId: params.sessionId,
      sessionKey: params.sessionKey,
      runtimeHint,
      query: params.query,
      limit: params.limit,
    });
  }

  async dispose(): Promise<void> {
    await this.manager.dispose();
  }
}

export function registerPerMessageSummaryContextEngine(): void {
  registerContextEngineForOwner(
    PER_MESSAGE_SUMMARY_ENGINE_ID,
    (config) => new PerMessageSummaryContextEngine(config),
    "core",
    {
      allowSameOwnerRefresh: true,
    },
  );
}
