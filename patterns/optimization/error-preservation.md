# Error Preservation

> Preserve full error context (stack traces, failed inputs, environment state) in the conversation rather than summarizing or discarding it -- errors are the highest-signal tokens for self-correction.

## Problem

When agents encounter errors, the natural instinct in context management is to compress or summarize them. After all, a 50-line stack trace consumes precious tokens. But this is a false economy:

- **Summarized errors lose critical details**: "Database connection failed" tells the agent nothing about *which* database, *which* credentials, or *which* network path failed.
- **Agents repeat the same mistakes**: Without the full error, the agent cannot distinguish between similar-but-different failure modes. It retries the same broken approach.
- **Debugging loops lengthen**: Each failed attempt without full error context requires the agent to re-discover the failure mode from scratch.
- **Context compaction destroys error signal**: Standard summarization treats errors as low-priority content, but they are actually the highest-value tokens in a debugging session.

## Solution

Treat error context as **high-priority, protected content** that resists compression and summarization. Implement middleware that captures the full error (stack trace, failed inputs, environment state, timing) in a structured format. Apply retention policies that keep recent errors at full fidelity while gradually compressing older ones. Never discard error context until the error has been resolved.

The key insight: **one detailed error message is worth more than ten summarized ones**. The tokens spent preserving a full stack trace are recovered many times over by avoiding repeated failures.

## How It Works

```
Agent attempts action
        |
        v
    Action fails
        |
        v
+-------------------------+
| Error Capture           |
| - Full stack trace      |
| - Failed input/args     |
| - Environment snapshot  |
| - Timestamp             |
| - Attempt number        |
+-------------------------+
        |
        v
+-------------------------+
| Structure & Tag         |
| - Error category        |
| - Severity              |
| - Related past errors   |
| - Resolution status     |
+-------------------------+
        |
        v
+-------------------------+
| Retention Policy        |
| - Recent errors: FULL   |
| - Older resolved: BRIEF |
| - Older unresolved: FULL|
+-------------------------+
        |
        v
Context window with protected error blocks

Standard context compaction:
  [System][Tools][History...][Errors...][Recent Messages]
                   ^                      ^
                   compressed             kept

Error-preserving compaction:
  [System][Tools][History...][Errors...][Recent Messages]
                   ^            ^          ^
                   compressed   PROTECTED  kept
```

## Implementation

### Python

```python
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResolutionStatus(Enum):
    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    WORKAROUND = "workaround"


@dataclass(frozen=True)
class CapturedError:
    """A structured record of an error occurrence."""
    error_id: str
    timestamp: str
    error_type: str
    message: str
    stack_trace: str
    failed_input: str
    environment: dict
    attempt_number: int
    severity: ErrorSeverity
    category: str
    resolution: ResolutionStatus = ResolutionStatus.UNRESOLVED
    resolution_note: str = ""

    def to_full_context(self) -> str:
        """Format the error with full detail for context injection."""
        env_str = "\n".join(
            f"  {k}: {v}" for k, v in self.environment.items()
        )
        return (
            f"### Error [{self.error_id}] (Attempt #{self.attempt_number})\n"
            f"**Type**: {self.error_type}\n"
            f"**Message**: {self.message}\n"
            f"**Severity**: {self.severity.value}\n"
            f"**Status**: {self.resolution.value}\n"
            f"**When**: {self.timestamp}\n\n"
            f"**Stack Trace**:\n```\n{self.stack_trace}\n```\n\n"
            f"**Failed Input**:\n```\n{self.failed_input}\n```\n\n"
            f"**Environment**:\n{env_str}\n"
        )

    def to_brief_context(self) -> str:
        """Compressed format for older, resolved errors."""
        return (
            f"- [{self.error_id}] {self.error_type}: {self.message} "
            f"(attempt #{self.attempt_number}, {self.resolution.value}"
            f"{': ' + self.resolution_note if self.resolution_note else ''})"
        )

    def with_resolution(
        self,
        status: ResolutionStatus,
        note: str = "",
    ) -> "CapturedError":
        """Return a new CapturedError with updated resolution status."""
        return CapturedError(
            error_id=self.error_id,
            timestamp=self.timestamp,
            error_type=self.error_type,
            message=self.message,
            stack_trace=self.stack_trace,
            failed_input=self.failed_input,
            environment=self.environment,
            attempt_number=self.attempt_number,
            severity=self.severity,
            category=self.category,
            resolution=status,
            resolution_note=note,
        )


class ErrorPreservationMiddleware:
    """
    Captures, structures, and preserves error context with configurable
    retention policies. Designed to sit between an agent's execution
    layer and its context management layer.
    """

    def __init__(
        self,
        max_full_errors: int = 5,
        max_brief_errors: int = 20,
    ):
        self._errors: list[CapturedError] = []
        self._max_full = max_full_errors
        self._max_brief = max_brief_errors
        self._attempt_counter: dict[str, int] = {}

    def capture(
        self,
        exception: Exception,
        failed_input: str = "",
        environment: dict | None = None,
        category: str = "general",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ) -> CapturedError:
        """Capture an exception with full context."""
        error_type = type(exception).__name__
        # Track attempts per error category
        attempt_key = f"{category}:{error_type}"
        self._attempt_counter[attempt_key] = (
            self._attempt_counter.get(attempt_key, 0) + 1
        )

        error_id = f"err-{len(self._errors):04d}"
        captured = CapturedError(
            error_id=error_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            error_type=error_type,
            message=str(exception),
            stack_trace=traceback.format_exc(),
            failed_input=failed_input[:2000],  # Cap input size
            environment=environment or {},
            attempt_number=self._attempt_counter[attempt_key],
            severity=severity,
            category=category,
        )

        self._errors = [*self._errors, captured]
        return captured

    def resolve(
        self,
        error_id: str,
        status: ResolutionStatus = ResolutionStatus.RESOLVED,
        note: str = "",
    ) -> None:
        """Mark an error as resolved."""
        self._errors = [
            e.with_resolution(status, note) if e.error_id == error_id else e
            for e in self._errors
        ]

    def build_context_block(self) -> str:
        """
        Build an error context block with retention policy applied.

        Recent and unresolved errors get full detail.
        Older resolved errors get brief summaries.
        """
        if not self._errors:
            return ""

        # Separate unresolved and resolved
        unresolved = [
            e for e in self._errors
            if e.resolution == ResolutionStatus.UNRESOLVED
        ]
        resolved = [
            e for e in self._errors
            if e.resolution != ResolutionStatus.UNRESOLVED
        ]

        sections = ["# Error Context (Protected)\n"]

        # All unresolved errors get full context (most recent first)
        if unresolved:
            sections.append("## Unresolved Errors\n")
            for error in reversed(unresolved[-self._max_full:]):
                sections.append(error.to_full_context())

        # Recent resolved errors get brief context
        if resolved:
            sections.append("## Resolved Errors (Reference)\n")
            for error in reversed(resolved[-self._max_brief:]):
                sections.append(error.to_brief_context())

        # Add pattern detection
        patterns = self._detect_patterns()
        if patterns:
            sections.append("\n## Error Patterns Detected\n")
            for pattern in patterns:
                sections.append(f"- {pattern}")

        return "\n".join(sections)

    def _detect_patterns(self) -> list[str]:
        """Detect recurring error patterns to surface to the agent."""
        patterns = []

        # Check for repeated failures in same category
        for key, count in self._attempt_counter.items():
            if count >= 3:
                category, error_type = key.split(":", 1)
                patterns.append(
                    f"REPEATED FAILURE: {error_type} in '{category}' "
                    f"has failed {count} times. Consider a different approach."
                )

        # Check for unresolved error accumulation
        unresolved_count = sum(
            1 for e in self._errors
            if e.resolution == ResolutionStatus.UNRESOLVED
        )
        if unresolved_count >= 3:
            patterns.append(
                f"ACCUMULATION: {unresolved_count} unresolved errors. "
                f"Address existing errors before attempting new actions."
            )

        return patterns

    @property
    def unresolved_count(self) -> int:
        return sum(
            1 for e in self._errors
            if e.resolution == ResolutionStatus.UNRESOLVED
        )

    @property
    def error_count(self) -> int:
        return len(self._errors)
```

### TypeScript

```typescript
type ErrorSeverity = "low" | "medium" | "high" | "critical";
type ResolutionStatus = "unresolved" | "resolved" | "workaround";

interface CapturedError {
  readonly errorId: string;
  readonly timestamp: string;
  readonly errorType: string;
  readonly message: string;
  readonly stackTrace: string;
  readonly failedInput: string;
  readonly environment: Readonly<Record<string, string>>;
  readonly attemptNumber: number;
  readonly severity: ErrorSeverity;
  readonly category: string;
  readonly resolution: ResolutionStatus;
  readonly resolutionNote: string;
}

function errorToFullContext(error: CapturedError): string {
  const envStr = Object.entries(error.environment)
    .map(([k, v]) => `  ${k}: ${v}`)
    .join("\n");

  return [
    `### Error [${error.errorId}] (Attempt #${error.attemptNumber})`,
    `**Type**: ${error.errorType}`,
    `**Message**: ${error.message}`,
    `**Severity**: ${error.severity}`,
    `**Status**: ${error.resolution}`,
    `**When**: ${error.timestamp}`,
    "",
    `**Stack Trace**:\n\`\`\`\n${error.stackTrace}\n\`\`\``,
    "",
    `**Failed Input**:\n\`\`\`\n${error.failedInput}\n\`\`\``,
    "",
    `**Environment**:\n${envStr}`,
  ].join("\n");
}

function errorToBriefContext(error: CapturedError): string {
  const resolution = error.resolutionNote
    ? `${error.resolution}: ${error.resolutionNote}`
    : error.resolution;
  return `- [${error.errorId}] ${error.errorType}: ${error.message} (attempt #${error.attemptNumber}, ${resolution})`;
}

function withResolution(
  error: CapturedError,
  status: ResolutionStatus,
  note: string = ""
): CapturedError {
  return { ...error, resolution: status, resolutionNote: note };
}

class ErrorPreservationMiddleware {
  private errors: readonly CapturedError[] = [];
  private readonly maxFullErrors: number;
  private readonly maxBriefErrors: number;
  private attemptCounter: ReadonlyMap<string, number> = new Map();

  constructor(maxFullErrors = 5, maxBriefErrors = 20) {
    this.maxFullErrors = maxFullErrors;
    this.maxBriefErrors = maxBriefErrors;
  }

  capture(params: {
    error: Error;
    failedInput?: string;
    environment?: Record<string, string>;
    category?: string;
    severity?: ErrorSeverity;
  }): CapturedError {
    const category = params.category ?? "general";
    const attemptKey = `${category}:${params.error.name}`;

    const currentCount = this.attemptCounter.get(attemptKey) ?? 0;
    const newCount = currentCount + 1;
    this.attemptCounter = new Map([
      ...this.attemptCounter,
      [attemptKey, newCount],
    ]);

    const captured: CapturedError = {
      errorId: `err-${String(this.errors.length).padStart(4, "0")}`,
      timestamp: new Date().toISOString(),
      errorType: params.error.name,
      message: params.error.message,
      stackTrace: params.error.stack ?? "No stack trace available",
      failedInput: (params.failedInput ?? "").slice(0, 2000),
      environment: Object.freeze({ ...(params.environment ?? {}) }),
      attemptNumber: newCount,
      severity: params.severity ?? "medium",
      category,
      resolution: "unresolved",
      resolutionNote: "",
    };

    this.errors = Object.freeze([...this.errors, captured]);
    return captured;
  }

  resolve(
    errorId: string,
    status: ResolutionStatus = "resolved",
    note: string = ""
  ): void {
    this.errors = Object.freeze(
      this.errors.map((e) =>
        e.errorId === errorId ? withResolution(e, status, note) : e
      )
    );
  }

  buildContextBlock(): string {
    if (this.errors.length === 0) return "";

    const unresolved = this.errors.filter((e) => e.resolution === "unresolved");
    const resolved = this.errors.filter((e) => e.resolution !== "unresolved");

    const sections: string[] = ["# Error Context (Protected)\n"];

    if (unresolved.length > 0) {
      sections.push("## Unresolved Errors\n");
      const recent = unresolved.slice(-this.maxFullErrors).reverse();
      for (const error of recent) {
        sections.push(errorToFullContext(error));
      }
    }

    if (resolved.length > 0) {
      sections.push("## Resolved Errors (Reference)\n");
      const recent = resolved.slice(-this.maxBriefErrors).reverse();
      for (const error of recent) {
        sections.push(errorToBriefContext(error));
      }
    }

    const patterns = this.detectPatterns();
    if (patterns.length > 0) {
      sections.push("\n## Error Patterns Detected\n");
      for (const pattern of patterns) {
        sections.push(`- ${pattern}`);
      }
    }

    return sections.join("\n");
  }

  private detectPatterns(): string[] {
    const patterns: string[] = [];

    for (const [key, count] of this.attemptCounter) {
      if (count >= 3) {
        const [category, errorType] = key.split(":", 2);
        patterns.push(
          `REPEATED FAILURE: ${errorType} in '${category}' has failed ${count} times. Consider a different approach.`
        );
      }
    }

    const unresolvedCount = this.errors.filter(
      (e) => e.resolution === "unresolved"
    ).length;
    if (unresolvedCount >= 3) {
      patterns.push(
        `ACCUMULATION: ${unresolvedCount} unresolved errors. Address existing errors before attempting new actions.`
      );
    }

    return patterns;
  }

  get unresolvedCount(): number {
    return this.errors.filter((e) => e.resolution === "unresolved").length;
  }

  get errorCount(): number {
    return this.errors.length;
  }
}

export {
  ErrorPreservationMiddleware,
  CapturedError,
  ErrorSeverity,
  ResolutionStatus,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Dramatically reduces repeated failures and debugging loops | Consumes more tokens than summarized errors |
| Enables pattern detection across multiple error occurrences | Full stack traces can be noisy if the error is in a deep dependency |
| Structured format makes errors parseable by the agent | Retention policies add complexity to context management |
| Resolution tracking prevents re-investigating solved problems | Error accumulation can crowd out other useful context |
| Human-debuggable -- the context block is readable by developers | Requires integration with the agent's execution layer |

## When to Use

- Agentic coding systems where errors drive the debugging loop
- Any agent that makes multiple attempts to accomplish a task
- Long-running sessions where error history informs future decisions
- Systems where the cost of a repeated mistake exceeds the cost of extra tokens
- Debugging-heavy workflows (test-driven development, deployment pipelines)

## When NOT to Use

- Simple, stateless Q&A systems with no retry logic
- When errors are trivial and self-explanatory (e.g., "file not found")
- Extremely token-constrained environments where every token matters
- When errors contain sensitive information that should not persist in context (redact first)

## Related Patterns

- **KV-Cache Optimization**: Error context belongs in the dynamic suffix, never in the stable prefix. This preserves cache validity while retaining error detail.
- **Context Rot Detection** (Evaluation): Error accumulation is a form of context rot. Use rot detection to identify when error context is crowding out productive context.
- **Filesystem-as-Memory** (Persistence): For errors that span sessions, write structured error logs to disk so future sessions can reference past failure modes.

## Real-World Examples

- **Manus**: Documents error preservation as a key architectural principle. Compressing error details was identified as a primary cause of agents repeating mistakes in loops.
- **Claude Code**: Keeps full terminal output (including errors) in the conversation context rather than summarizing. This allows the agent to reference exact error messages when debugging.
- **Devin**: Maintains error context across its planning and execution loops, using past errors to inform revised plans rather than starting from scratch.
- **SWE-bench Agents**: Top-performing agents on SWE-bench consistently preserve full test output and error messages, using them to iteratively refine patches until tests pass.
