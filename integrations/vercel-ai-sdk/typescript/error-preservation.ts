/**
 * Error Preservation -- Vercel AI SDK Integration
 *
 * Implements structured error capture and retention that integrates with
 * the Vercel AI SDK's error handling and retry mechanisms. Errors are
 * treated as high-priority, protected context that resists compression.
 *
 * Pattern: patterns/optimization/error-preservation.md
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ErrorSeverity = "low" | "medium" | "high" | "critical";
type ResolutionStatus = "unresolved" | "resolved" | "workaround";

interface CapturedError {
  readonly errorId: string;
  readonly timestamp: string;
  readonly errorType: string;
  readonly message: string;
  readonly stackTrace: string;
  readonly failedInput: string;
  readonly attemptNumber: number;
  readonly severity: ErrorSeverity;
  readonly resolution: ResolutionStatus;
  readonly context: Record<string, unknown>;
}

interface CoreMessage {
  readonly role: "system" | "user" | "assistant" | "tool";
  readonly content: string;
}

interface RetentionPolicy {
  readonly maxUnresolvedErrors: number;
  readonly maxResolvedErrors: number;
  readonly compactResolvedAfterTurns: number;
}

const DEFAULT_RETENTION: RetentionPolicy = {
  maxUnresolvedErrors: 10,
  maxResolvedErrors: 5,
  compactResolvedAfterTurns: 20,
};

// ---------------------------------------------------------------------------
// ErrorPreserver
// ---------------------------------------------------------------------------

/**
 * Captures, structures, and preserves error context for AI agent retry loops.
 *
 * Integrates with the Vercel AI SDK's error handling:
 *
 *   import { generateText } from 'ai';
 *   import { openai } from '@ai-sdk/openai';
 *
 *   const preserver = createErrorPreserver();
 *
 *   try {
 *     const result = await generateText({
 *       model: openai('gpt-4o'),
 *       messages,
 *     });
 *   } catch (error) {
 *     // Capture the full error context
 *     preserver.capture({
 *       error,
 *       failedInput: messages[messages.length - 1].content,
 *       severity: 'high',
 *     });
 *
 *     // Build retry messages with preserved error context
 *     const retryMessages = preserver.buildRetryMessages(messages);
 *     const retryResult = await generateText({
 *       model: openai('gpt-4o'),
 *       messages: retryMessages,
 *     });
 *   }
 */
class ErrorPreserver {
  private errors: CapturedError[] = [];
  private attemptCounter: Map<string, number> = new Map();
  private readonly retention: RetentionPolicy;
  private turnsSinceCapture: number = 0;

  constructor(retention: Partial<RetentionPolicy> = {}) {
    this.retention = { ...DEFAULT_RETENTION, ...retention };
  }

  /**
   * Capture an error with full context.
   *
   * The error is structured with all available debugging information
   * and stored as protected context that resists compaction.
   */
  capture(params: {
    error: unknown;
    failedInput?: string;
    severity?: ErrorSeverity;
    context?: Record<string, unknown>;
  }): CapturedError {
    const errorObj =
      params.error instanceof Error
        ? params.error
        : new Error(String(params.error));

    const errorType = errorObj.constructor.name;

    // Track attempt number for this error type
    const currentAttempt = (this.attemptCounter.get(errorType) ?? 0) + 1;
    this.attemptCounter.set(errorType, currentAttempt);

    const captured: CapturedError = {
      errorId: `err_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      errorType,
      message: errorObj.message,
      stackTrace: errorObj.stack ?? "",
      failedInput: params.failedInput ?? "",
      attemptNumber: currentAttempt,
      severity: params.severity ?? "medium",
      resolution: "unresolved",
      context: params.context ?? {},
    };

    this.errors = [...this.errors, captured];
    this.turnsSinceCapture = 0;
    this.enforceRetention();

    return captured;
  }

  /**
   * Mark an error as resolved. Resolved errors are compacted more
   * aggressively than unresolved ones.
   */
  resolve(errorId: string, resolution: ResolutionStatus = "resolved"): void {
    this.errors = this.errors.map((e) =>
      e.errorId === errorId ? { ...e, resolution } : e
    );
  }

  /**
   * Build the error context block for injection into the prompt.
   *
   * Unresolved errors are preserved at full fidelity. Resolved errors
   * are compacted to a brief summary after the retention period.
   */
  buildErrorContext(): string {
    if (this.errors.length === 0) return "";

    const unresolved = this.errors.filter(
      (e) => e.resolution === "unresolved"
    );
    const resolved = this.errors.filter(
      (e) => e.resolution !== "unresolved"
    );

    const sections: string[] = [];

    if (unresolved.length > 0) {
      sections.push("## Active Errors (UNRESOLVED -- do not repeat these)");
      for (const err of unresolved) {
        sections.push(this.formatErrorFull(err));
      }
    }

    if (resolved.length > 0) {
      sections.push("## Resolved Errors (for reference)");
      for (const err of resolved) {
        if (this.turnsSinceCapture > this.retention.compactResolvedAfterTurns) {
          sections.push(this.formatErrorBrief(err));
        } else {
          sections.push(this.formatErrorFull(err));
        }
      }
    }

    return `<error_context>\n${sections.join("\n\n")}\n</error_context>`;
  }

  /**
   * Build retry messages that include the full error context.
   *
   * This is designed to be called after an error to give the model
   * the information it needs to avoid repeating the same mistake.
   */
  buildRetryMessages(
    originalMessages: readonly CoreMessage[]
  ): CoreMessage[] {
    const errorContext = this.buildErrorContext();
    if (!errorContext) return [...originalMessages];

    // Inject error context as a system message right before the last user message
    const result: CoreMessage[] = [];
    const lastUserIdx = originalMessages.reduce(
      (acc, msg, idx) => (msg.role === "user" ? idx : acc),
      -1
    );

    for (let i = 0; i < originalMessages.length; i++) {
      if (i === lastUserIdx) {
        result.push({
          role: "system",
          content: errorContext,
        });
      }
      result.push(originalMessages[i]);
    }

    return result;
  }

  /**
   * Notify the preserver that a turn has passed. Used for aging
   * resolved errors toward compaction.
   */
  advanceTurn(): void {
    this.turnsSinceCapture++;
  }

  get unresolvedCount(): number {
    return this.errors.filter((e) => e.resolution === "unresolved").length;
  }

  get totalCount(): number {
    return this.errors.length;
  }

  // -- Internal ------------------------------------------------------------

  private formatErrorFull(err: CapturedError): string {
    const lines = [
      `### ${err.errorType} (attempt #${err.attemptNumber})`,
      `**Severity**: ${err.severity}`,
      `**Message**: ${err.message}`,
    ];

    if (err.stackTrace) {
      // Include first 5 lines of stack trace
      const stackLines = err.stackTrace.split("\n").slice(0, 5).join("\n");
      lines.push(`**Stack**:\n\`\`\`\n${stackLines}\n\`\`\``);
    }

    if (err.failedInput) {
      lines.push(
        `**Failed Input**: ${err.failedInput.slice(0, 200)}${err.failedInput.length > 200 ? "..." : ""}`
      );
    }

    if (Object.keys(err.context).length > 0) {
      lines.push(`**Context**: ${JSON.stringify(err.context)}`);
    }

    return lines.join("\n");
  }

  private formatErrorBrief(err: CapturedError): string {
    return `- [${err.resolution}] ${err.errorType}: ${err.message} (attempt #${err.attemptNumber})`;
  }

  private enforceRetention(): void {
    const unresolved = this.errors.filter(
      (e) => e.resolution === "unresolved"
    );
    const resolved = this.errors.filter(
      (e) => e.resolution !== "unresolved"
    );

    // Keep only the most recent unresolved errors
    const keptUnresolved = unresolved.slice(
      -this.retention.maxUnresolvedErrors
    );

    // Keep only the most recent resolved errors
    const keptResolved = resolved.slice(-this.retention.maxResolvedErrors);

    this.errors = [...keptResolved, ...keptUnresolved];
  }
}

/** Factory function. */
function createErrorPreserver(
  retention: Partial<RetentionPolicy> = {}
): ErrorPreserver {
  return new ErrorPreserver(retention);
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function main(): void {
  const preserver = createErrorPreserver({
    maxUnresolvedErrors: 5,
    maxResolvedErrors: 3,
  });

  // Simulate a sequence of errors during an agent loop

  // Error 1: Database connection failure
  const err1 = preserver.capture({
    error: new TypeError(
      "Cannot read properties of undefined (reading 'connect')"
    ),
    failedInput: "await db.client.connect()",
    severity: "critical",
    context: { file: "src/db/pool.ts", line: 42 },
  });

  console.log(`Captured error: ${err1.errorId}`);
  console.log(`Unresolved: ${preserver.unresolvedCount}`);
  console.log();

  // Error 2: API timeout
  preserver.capture({
    error: new Error("Request timeout after 30000ms"),
    failedInput: 'fetch("https://api.example.com/users")',
    severity: "high",
    context: { endpoint: "/users", timeout: 30000 },
  });

  // Show the error context block
  console.log("=== Error Context (for prompt injection) ===");
  console.log(preserver.buildErrorContext());
  console.log();

  // Resolve the first error
  preserver.resolve(err1.errorId);
  console.log(`After resolving ${err1.errorId}:`);
  console.log(`Unresolved: ${preserver.unresolvedCount}`);
  console.log(`Total: ${preserver.totalCount}`);
  console.log();

  // Show retry messages
  const originalMessages: CoreMessage[] = [
    { role: "system", content: "You are a coding assistant." },
    { role: "user", content: "Fix the database connection issue" },
    {
      role: "assistant",
      content: "I'll try connecting to the database.",
    },
    { role: "user", content: "It's still failing. Try a different approach." },
  ];

  const retryMessages = preserver.buildRetryMessages(originalMessages);
  console.log("=== Retry Messages ===");
  for (const msg of retryMessages) {
    const preview = msg.content.slice(0, 120).replace(/\n/g, " ");
    console.log(`[${msg.role}] ${preview}${msg.content.length > 120 ? "..." : ""}`);
  }
  console.log();

  // Show that error context is injected before the last user message
  const errorContextMsg = retryMessages.find(
    (m) => m.role === "system" && m.content.includes("<error_context>")
  );
  console.log(
    `Error context injected: ${errorContextMsg ? "yes" : "no"}`
  );
  console.log(
    `Position: before last user message (index ${retryMessages.indexOf(errorContextMsg!)})`
  );
}

main();
