# Context Rot Detection

> Monitor context quality over long-running sessions, detecting when instructions are forgotten, contradictions accumulate, or stale data persists -- and trigger remediation before output quality degrades.

## Problem

As conversations grow, context degrades silently:

- **Instruction drift**: Early system instructions get "pushed out" of the model's effective attention window. The model stops following rules it was given at the start.
- **Contradiction accumulation**: As context grows, conflicting statements appear. The model is told to "use tabs" in one place and "use spaces" in another. It picks one arbitrarily or oscillates.
- **Stale data persistence**: Information that was accurate 50 messages ago is now outdated, but it still sits in context influencing behavior.
- **Signal dilution**: Important instructions are buried under thousands of tokens of conversation, tool output, and intermediate results.

The danger is that context rot is **invisible**. The model does not flag that it has forgotten your instructions. It simply produces worse output, and you blame the model rather than the context.

## Solution

Implement periodic health checks that evaluate context quality along multiple dimensions: instruction adherence, contradiction detection, and staleness measurement. When a check fails, trigger remediation -- re-inject key instructions, compact stale sections, or reset context with a fresh summary.

Context rot detection is the **observability layer** for context engineering. Just as you monitor application health with metrics and alerts, you monitor context health with adherence tests and quality scores.

## How It Works

```
Long-running session
        |
        v
+-------------------+
| Periodic Check    |  <-- Every N messages or N minutes
| Trigger           |
+-------------------+
        |
        v
+---------------------------+
| Health Check Suite        |
|                           |
| 1. Instruction Adherence  |  <-- Does the model still follow key rules?
| 2. Contradiction Scan     |  <-- Are there conflicting statements?
| 3. Staleness Check        |  <-- Is information outdated?
| 4. Attention Test         |  <-- Can the model recall early context?
+---------------------------+
        |
        v
+-------------------+
| Health Score      |  <-- 0-100 composite score
| 85/100            |
+-------------------+
        |
   Score > threshold?
   /            \
  YES            NO
  |               |
  v               v
Continue     +-------------------+
             | Remediation       |
             | - Re-inject rules |
             | - Compact history |
             | - Reset context   |
             +-------------------+


Health score over time (typical long session):

100 |****
    |    ****
    |        ***
    |           **              ** <-- remediation applied
    |             **          **
    |               **      **
    |                 **  **
    |                   **
  0 +----------------------------> messages
    0    50   100  150  200  250
```

## Implementation

### Python

```python
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class HealthLevel(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class RemediationAction(Enum):
    NONE = "none"
    REINJECT_INSTRUCTIONS = "reinject_instructions"
    COMPACT_HISTORY = "compact_history"
    FULL_RESET = "full_reset"


@dataclass(frozen=True)
class HealthCheckResult:
    """Result of a single health check dimension."""
    dimension: str
    score: float  # 0.0 to 1.0
    details: str
    failed_checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextHealthReport:
    """Aggregate health report across all dimensions."""
    timestamp: str
    overall_score: float
    level: HealthLevel
    checks: tuple[HealthCheckResult, ...]
    recommended_action: RemediationAction
    message_count: int
    context_tokens_estimate: int

    def to_context_block(self) -> str:
        """Format the report for injection into context."""
        lines = [
            f"## Context Health Report",
            f"**Score**: {self.overall_score:.0%} ({self.level.value})",
            f"**Messages**: {self.message_count}",
            f"**Est. Tokens**: {self.context_tokens_estimate:,}",
            f"**Action**: {self.recommended_action.value}",
            "",
        ]
        for check in self.checks:
            status = "PASS" if check.score >= 0.7 else "FAIL"
            lines.append(f"- [{status}] {check.dimension}: {check.score:.0%}")
            if check.failed_checks:
                for fc in check.failed_checks:
                    lines.append(f"  - {fc}")
        return "\n".join(lines)


class InstructionAdherenceChecker:
    """
    Tests whether the model still follows key instructions by
    checking recent outputs against expected behaviors.
    """

    def __init__(self, key_instructions: list[dict]):
        """
        Args:
            key_instructions: List of dicts with 'rule' (str) and
                'test_fn' (callable that takes recent messages and
                returns bool).
        """
        self._instructions = key_instructions

    def check(self, recent_messages: list[dict]) -> HealthCheckResult:
        """Test instruction adherence against recent messages."""
        if not self._instructions:
            return HealthCheckResult(
                dimension="instruction_adherence",
                score=1.0,
                details="No instructions to check.",
            )

        passed = 0
        failed = []

        for instruction in self._instructions:
            rule = instruction["rule"]
            test_fn = instruction["test_fn"]
            try:
                if test_fn(recent_messages):
                    passed += 1
                else:
                    failed.append(f"Rule not followed: {rule}")
            except Exception as e:
                failed.append(f"Check error for '{rule}': {e}")

        score = passed / len(self._instructions) if self._instructions else 1.0

        return HealthCheckResult(
            dimension="instruction_adherence",
            score=score,
            details=f"{passed}/{len(self._instructions)} rules followed.",
            failed_checks=tuple(failed),
        )


class ContradictionScanner:
    """
    Scans context for contradictory statements by comparing
    key-value assertions across the conversation.
    """

    def __init__(self, extract_assertions_fn=None):
        """
        Args:
            extract_assertions_fn: Optional callable that takes a message
                and returns a list of (key, value) tuples representing
                assertions. If None, uses simple heuristic extraction.
        """
        self._extract_fn = extract_assertions_fn or self._default_extract

    def check(self, messages: list[dict]) -> HealthCheckResult:
        """Scan for contradictions across all messages."""
        assertions: dict[str, list[tuple[str, int]]] = {}
        contradictions = []

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            extracted = self._extract_fn(content)
            for key, value in extracted:
                if key in assertions:
                    for prev_value, prev_idx in assertions[key]:
                        if prev_value != value:
                            contradictions.append(
                                f"'{key}' is '{prev_value}' (msg {prev_idx}) "
                                f"vs '{value}' (msg {i})"
                            )
                    assertions[key].append((value, i))
                else:
                    assertions[key] = [(value, i)]

        # Score: fewer contradictions = higher score
        max_contradictions = 5  # Normalize against this
        score = max(0.0, 1.0 - len(contradictions) / max_contradictions)

        return HealthCheckResult(
            dimension="contradiction_scan",
            score=score,
            details=f"{len(contradictions)} contradictions found.",
            failed_checks=tuple(contradictions[:10]),
        )

    @staticmethod
    def _default_extract(content: str) -> list[tuple[str, str]]:
        """Simple heuristic: extract 'X is Y' and 'use X' patterns."""
        assertions = []
        for line in content.split("\n"):
            line = line.strip().lower()
            # "use X" pattern
            if line.startswith("use ") and len(line.split()) <= 4:
                category = "tool_preference"
                assertions.append((category, line))
        return assertions


class StalenessChecker:
    """
    Checks whether information in context is outdated based on
    timestamps, message age, and explicit staleness markers.
    """

    def __init__(self, max_age_messages: int = 100):
        """
        Args:
            max_age_messages: Messages older than this are considered
                potentially stale.
        """
        self._max_age = max_age_messages

    def check(self, messages: list[dict]) -> HealthCheckResult:
        """Check for stale content in the message history."""
        total = len(messages)
        if total <= self._max_age:
            return HealthCheckResult(
                dimension="staleness",
                score=1.0,
                details="Context is within freshness window.",
            )

        stale_count = total - self._max_age
        # Score degrades as the stale portion grows
        stale_ratio = stale_count / total
        score = max(0.0, 1.0 - stale_ratio)

        issues = []
        if stale_ratio > 0.5:
            issues.append(
                f"{stale_count} messages ({stale_ratio:.0%}) are beyond "
                f"the freshness window of {self._max_age} messages."
            )

        return HealthCheckResult(
            dimension="staleness",
            score=score,
            details=f"{stale_count}/{total} messages potentially stale.",
            failed_checks=tuple(issues),
        )


class ContextRotDetector:
    """
    Orchestrates periodic context health checks and triggers
    remediation when quality degrades below thresholds.
    """

    def __init__(
        self,
        instruction_checker: InstructionAdherenceChecker,
        contradiction_scanner: ContradictionScanner,
        staleness_checker: StalenessChecker,
        check_interval: int = 20,
        degraded_threshold: float = 0.7,
        critical_threshold: float = 0.4,
    ):
        self._checkers = {
            "adherence": instruction_checker,
            "contradictions": contradiction_scanner,
            "staleness": staleness_checker,
        }
        self._check_interval = check_interval
        self._degraded_threshold = degraded_threshold
        self._critical_threshold = critical_threshold
        self._messages_since_check = 0
        self._history: list[ContextHealthReport] = []

    def on_message(self, messages: list[dict]) -> ContextHealthReport | None:
        """
        Call after each message. Returns a health report if a check
        is triggered, otherwise None.
        """
        self._messages_since_check += 1

        if self._messages_since_check < self._check_interval:
            return None

        self._messages_since_check = 0
        return self.run_check(messages)

    def run_check(self, messages: list[dict]) -> ContextHealthReport:
        """Run all health checks and produce a report."""
        # Run each checker
        recent = messages[-50:] if len(messages) > 50 else messages
        results = []

        adherence_result = self._checkers["adherence"].check(recent)
        contradiction_result = self._checkers["contradictions"].check(messages)
        staleness_result = self._checkers["staleness"].check(messages)

        results = [adherence_result, contradiction_result, staleness_result]

        # Weighted average (instruction adherence matters most)
        weights = {"instruction_adherence": 0.5, "contradiction_scan": 0.3, "staleness": 0.2}
        total_weight = sum(weights.get(r.dimension, 0.2) for r in results)
        overall = sum(
            r.score * weights.get(r.dimension, 0.2) for r in results
        ) / total_weight

        # Determine health level and action
        if overall >= self._degraded_threshold:
            level = HealthLevel.HEALTHY
            action = RemediationAction.NONE
        elif overall >= self._critical_threshold:
            level = HealthLevel.DEGRADED
            action = RemediationAction.REINJECT_INSTRUCTIONS
        else:
            level = HealthLevel.CRITICAL
            action = RemediationAction.FULL_RESET

        # Estimate tokens (rough: 4 chars per token)
        token_estimate = sum(
            len(m.get("content", "")) // 4
            for m in messages
            if isinstance(m.get("content"), str)
        )

        report = ContextHealthReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=overall,
            level=level,
            checks=tuple(results),
            recommended_action=action,
            message_count=len(messages),
            context_tokens_estimate=token_estimate,
        )

        self._history = [*self._history, report]
        return report

    @property
    def health_trend(self) -> list[float]:
        """Return the trend of overall scores for analysis."""
        return [r.overall_score for r in self._history]

    @property
    def is_degrading(self) -> bool:
        """Check if health scores show a downward trend."""
        scores = self.health_trend
        if len(scores) < 3:
            return False
        recent = scores[-3:]
        return all(recent[i] < recent[i - 1] for i in range(1, len(recent)))
```

### TypeScript

```typescript
type HealthLevel = "healthy" | "degraded" | "critical";
type RemediationAction =
  | "none"
  | "reinject_instructions"
  | "compact_history"
  | "full_reset";

interface HealthCheckResult {
  readonly dimension: string;
  readonly score: number;
  readonly details: string;
  readonly failedChecks: readonly string[];
}

interface ContextHealthReport {
  readonly timestamp: string;
  readonly overallScore: number;
  readonly level: HealthLevel;
  readonly checks: readonly HealthCheckResult[];
  readonly recommendedAction: RemediationAction;
  readonly messageCount: number;
  readonly contextTokensEstimate: number;
}

interface Message {
  role: string;
  content: string;
}

interface InstructionRule {
  rule: string;
  testFn: (recentMessages: Message[]) => boolean;
}

function reportToContextBlock(report: ContextHealthReport): string {
  const lines = [
    `## Context Health Report`,
    `**Score**: ${Math.round(report.overallScore * 100)}% (${report.level})`,
    `**Messages**: ${report.messageCount}`,
    `**Est. Tokens**: ${report.contextTokensEstimate.toLocaleString()}`,
    `**Action**: ${report.recommendedAction}`,
    "",
  ];

  for (const check of report.checks) {
    const status = check.score >= 0.7 ? "PASS" : "FAIL";
    lines.push(
      `- [${status}] ${check.dimension}: ${Math.round(check.score * 100)}%`
    );
    for (const fc of check.failedChecks) {
      lines.push(`  - ${fc}`);
    }
  }

  return lines.join("\n");
}

class InstructionAdherenceChecker {
  private readonly rules: readonly InstructionRule[];

  constructor(rules: InstructionRule[]) {
    this.rules = Object.freeze([...rules]);
  }

  check(recentMessages: Message[]): HealthCheckResult {
    if (this.rules.length === 0) {
      return {
        dimension: "instruction_adherence",
        score: 1.0,
        details: "No instructions to check.",
        failedChecks: [],
      };
    }

    let passed = 0;
    const failed: string[] = [];

    for (const rule of this.rules) {
      try {
        if (rule.testFn(recentMessages)) {
          passed++;
        } else {
          failed.push(`Rule not followed: ${rule.rule}`);
        }
      } catch (e) {
        failed.push(`Check error for '${rule.rule}': ${e}`);
      }
    }

    return {
      dimension: "instruction_adherence",
      score: passed / this.rules.length,
      details: `${passed}/${this.rules.length} rules followed.`,
      failedChecks: Object.freeze(failed),
    };
  }
}

class ContradictionScanner {
  check(messages: Message[]): HealthCheckResult {
    const assertions: Map<string, Array<{ value: string; index: number }>> =
      new Map();
    const contradictions: string[] = [];

    for (let i = 0; i < messages.length; i++) {
      const content = messages[i].content;
      if (!content) continue;

      for (const line of content.split("\n")) {
        const trimmed = line.trim().toLowerCase();
        if (trimmed.startsWith("use ") && trimmed.split(/\s+/).length <= 4) {
          const key = "tool_preference";
          const existing = assertions.get(key) ?? [];
          for (const prev of existing) {
            if (prev.value !== trimmed) {
              contradictions.push(
                `'${key}' is '${prev.value}' (msg ${prev.index}) vs '${trimmed}' (msg ${i})`
              );
            }
          }
          existing.push({ value: trimmed, index: i });
          assertions.set(key, existing);
        }
      }
    }

    const maxContradictions = 5;
    const score = Math.max(0, 1.0 - contradictions.length / maxContradictions);

    return {
      dimension: "contradiction_scan",
      score,
      details: `${contradictions.length} contradictions found.`,
      failedChecks: Object.freeze(contradictions.slice(0, 10)),
    };
  }
}

class StalenessChecker {
  private readonly maxAge: number;

  constructor(maxAgeMessages = 100) {
    this.maxAge = maxAgeMessages;
  }

  check(messages: Message[]): HealthCheckResult {
    const total = messages.length;
    if (total <= this.maxAge) {
      return {
        dimension: "staleness",
        score: 1.0,
        details: "Context is within freshness window.",
        failedChecks: [],
      };
    }

    const staleCount = total - this.maxAge;
    const staleRatio = staleCount / total;
    const score = Math.max(0, 1.0 - staleRatio);

    const issues: string[] = [];
    if (staleRatio > 0.5) {
      issues.push(
        `${staleCount} messages (${Math.round(staleRatio * 100)}%) are beyond the freshness window of ${this.maxAge} messages.`
      );
    }

    return {
      dimension: "staleness",
      score,
      details: `${staleCount}/${total} messages potentially stale.`,
      failedChecks: Object.freeze(issues),
    };
  }
}

class ContextRotDetector {
  private readonly adherenceChecker: InstructionAdherenceChecker;
  private readonly contradictionScanner: ContradictionScanner;
  private readonly stalenessChecker: StalenessChecker;
  private readonly checkInterval: number;
  private readonly degradedThreshold: number;
  private readonly criticalThreshold: number;
  private messagesSinceCheck = 0;
  private reportHistory: readonly ContextHealthReport[] = [];

  constructor(params: {
    adherenceChecker: InstructionAdherenceChecker;
    contradictionScanner: ContradictionScanner;
    stalenessChecker: StalenessChecker;
    checkInterval?: number;
    degradedThreshold?: number;
    criticalThreshold?: number;
  }) {
    this.adherenceChecker = params.adherenceChecker;
    this.contradictionScanner = params.contradictionScanner;
    this.stalenessChecker = params.stalenessChecker;
    this.checkInterval = params.checkInterval ?? 20;
    this.degradedThreshold = params.degradedThreshold ?? 0.7;
    this.criticalThreshold = params.criticalThreshold ?? 0.4;
  }

  onMessage(messages: Message[]): ContextHealthReport | null {
    this.messagesSinceCheck += 1;
    if (this.messagesSinceCheck < this.checkInterval) return null;
    this.messagesSinceCheck = 0;
    return this.runCheck(messages);
  }

  runCheck(messages: Message[]): ContextHealthReport {
    const recent = messages.length > 50 ? messages.slice(-50) : messages;

    const adherenceResult = this.adherenceChecker.check(recent);
    const contradictionResult = this.contradictionScanner.check(messages);
    const stalenessResult = this.stalenessChecker.check(messages);

    const checks = [adherenceResult, contradictionResult, stalenessResult];

    const weights: Record<string, number> = {
      instruction_adherence: 0.5,
      contradiction_scan: 0.3,
      staleness: 0.2,
    };

    const totalWeight = checks.reduce(
      (sum, c) => sum + (weights[c.dimension] ?? 0.2),
      0
    );
    const overall =
      checks.reduce(
        (sum, c) => sum + c.score * (weights[c.dimension] ?? 0.2),
        0
      ) / totalWeight;

    let level: HealthLevel;
    let action: RemediationAction;

    if (overall >= this.degradedThreshold) {
      level = "healthy";
      action = "none";
    } else if (overall >= this.criticalThreshold) {
      level = "degraded";
      action = "reinject_instructions";
    } else {
      level = "critical";
      action = "full_reset";
    }

    const tokenEstimate = messages.reduce(
      (sum, m) => sum + Math.floor((m.content?.length ?? 0) / 4),
      0
    );

    const report: ContextHealthReport = {
      timestamp: new Date().toISOString(),
      overallScore: overall,
      level,
      checks: Object.freeze(checks),
      recommendedAction: action,
      messageCount: messages.length,
      contextTokensEstimate: tokenEstimate,
    };

    this.reportHistory = Object.freeze([...this.reportHistory, report]);
    return report;
  }

  get healthTrend(): readonly number[] {
    return this.reportHistory.map((r) => r.overallScore);
  }

  get isDegrading(): boolean {
    const scores = this.healthTrend;
    if (scores.length < 3) return false;
    const recent = scores.slice(-3);
    return recent.every((s, i) => i === 0 || s < recent[i - 1]);
  }
}

export {
  ContextRotDetector,
  InstructionAdherenceChecker,
  ContradictionScanner,
  StalenessChecker,
  ContextHealthReport,
  HealthCheckResult,
  reportToContextBlock,
};
```

## Trade-offs

| Pros | Cons |
|------|------|
| Detects silent degradation before it impacts output quality | Health checks consume tokens and latency (the check itself is a cost) |
| Enables automated remediation (re-inject, compact, reset) | Instruction adherence tests must be hand-written per application |
| Provides observability into context quality over time | False positives can trigger unnecessary remediation |
| Trend analysis catches gradual degradation | Contradiction detection is imperfect without an LLM-in-the-loop |
| Forces explicit definition of "what matters" in context | Adds complexity to the agent orchestration layer |

## When to Use

- Long-running agent sessions (30+ minutes, 100+ messages)
- Production agents that run autonomously without human oversight
- Systems where instruction adherence is critical (compliance, safety)
- Multi-turn workflows where context grows continuously
- Any system where you have observed behavioral drift over time

## When NOT to Use

- Short conversations (under 20 messages) where rot cannot develop
- Systems with aggressive context compaction that already manages context size
- When the overhead of health checks is not justified by session length
- Single-turn, stateless applications

## Related Patterns

- **Error Preservation** (Optimization): Error accumulation is a specific form of context rot. Error preservation policies and rot detection work together -- preservation keeps errors available, rot detection flags when they are crowding out productive context.
- **Progressive Disclosure** (Construction): Remediation via re-injection of key instructions is essentially progressive disclosure applied retroactively -- re-revealing context that has been "forgotten."
- **Filesystem-as-Memory** (Persistence): When a context reset is triggered, filesystem memory provides the durable ground truth that can be re-loaded into the fresh context.

## Real-World Examples

- **Anthropic Research**: Research on long-context behavior demonstrates that models lose adherence to early instructions as context grows. The "lost in the middle" phenomenon is a well-documented form of context rot.
- **Production Coding Agents**: Agents working on multi-hour tasks (large refactors, multi-file features) exhibit measurable degradation in instruction following. Teams report that agents "forget" coding standards after 200+ messages.
- **Customer Support Bots**: Long customer interactions where the bot gradually stops following tone guidelines or escalation rules. Context rot causes the bot to become less helpful over time.
- **Claude Code Session Management**: The recommendation to run `/clear` between unrelated tasks is a manual form of context rot remediation -- resetting context before accumulated staleness causes problems.
