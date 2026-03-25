/**
 * Context Rot Detection -- LlamaIndex TypeScript Integration
 *
 * Monitors context quality over long-running sessions, detecting instruction
 * drift, contradictions, and staleness. Produces structured health reports
 * compatible with LlamaIndex's evaluation patterns.
 *
 * Pattern: patterns/evaluation/context-rot-detection.md
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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

interface ChatMessage {
  readonly role: "system" | "user" | "assistant" | "tool";
  readonly content: string;
}

interface InstructionRule {
  readonly rule: string;
  readonly testFn: (messages: ChatMessage[]) => boolean;
}

// ---------------------------------------------------------------------------
// Report formatter
// ---------------------------------------------------------------------------

function reportToContextBlock(report: ContextHealthReport): string {
  const lines = [
    "## Context Health Report",
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

// ---------------------------------------------------------------------------
// Evaluators
// ---------------------------------------------------------------------------

class InstructionAdherenceEvaluator {
  private readonly rules: readonly InstructionRule[];

  constructor(rules: InstructionRule[]) {
    this.rules = Object.freeze([...rules]);
  }

  evaluate(messages: ChatMessage[]): HealthCheckResult {
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
        if (rule.testFn(messages)) {
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

class ContradictionEvaluator {
  evaluate(messages: ChatMessage[]): HealthCheckResult {
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

    const score = Math.max(0, 1.0 - contradictions.length / 5);
    return {
      dimension: "contradiction_scan",
      score,
      details: `${contradictions.length} contradictions found.`,
      failedChecks: Object.freeze(contradictions.slice(0, 10)),
    };
  }
}

class StalenessEvaluator {
  private readonly maxAge: number;

  constructor(maxAge: number = 100) {
    this.maxAge = maxAge;
  }

  evaluate(messages: ChatMessage[]): HealthCheckResult {
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
        `${staleCount} messages (${Math.round(staleRatio * 100)}%) beyond freshness window of ${this.maxAge}.`
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

// ---------------------------------------------------------------------------
// ContextRotDetector
// ---------------------------------------------------------------------------

class ContextRotDetector {
  private readonly adherence: InstructionAdherenceEvaluator;
  private readonly contradictions: ContradictionEvaluator;
  private readonly staleness: StalenessEvaluator;
  private readonly checkInterval: number;
  private readonly degradedThreshold: number;
  private readonly criticalThreshold: number;
  private messagesSinceCheck: number = 0;
  private reportHistory: ContextHealthReport[] = [];

  constructor(params: {
    rules?: InstructionRule[];
    checkInterval?: number;
    degradedThreshold?: number;
    criticalThreshold?: number;
    maxStalenessAge?: number;
  }) {
    this.adherence = new InstructionAdherenceEvaluator(params.rules ?? []);
    this.contradictions = new ContradictionEvaluator();
    this.staleness = new StalenessEvaluator(params.maxStalenessAge ?? 100);
    this.checkInterval = params.checkInterval ?? 20;
    this.degradedThreshold = params.degradedThreshold ?? 0.7;
    this.criticalThreshold = params.criticalThreshold ?? 0.4;
  }

  onMessage(messages: ChatMessage[]): ContextHealthReport | null {
    this.messagesSinceCheck++;
    if (this.messagesSinceCheck < this.checkInterval) return null;
    this.messagesSinceCheck = 0;
    return this.runCheck(messages);
  }

  runCheck(messages: ChatMessage[]): ContextHealthReport {
    const recent = messages.length > 50 ? messages.slice(-50) : messages;

    const adherenceResult = this.adherence.evaluate(recent);
    const contradictionResult = this.contradictions.evaluate(messages);
    const stalenessResult = this.staleness.evaluate(messages);

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

    this.reportHistory = [...this.reportHistory, report];
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

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function main(): void {
  const checkMarkdown = (messages: ChatMessage[]): boolean => {
    const assistantMsgs = messages.filter((m) => m.role === "assistant");
    if (assistantMsgs.length === 0) return true;
    const last = assistantMsgs[assistantMsgs.length - 1].content;
    return last.includes("#") || last.includes("```") || last.includes("**");
  };

  const checkConciseness = (messages: ChatMessage[]): boolean => {
    const assistantMsgs = messages.filter((m) => m.role === "assistant");
    if (assistantMsgs.length === 0) return true;
    return assistantMsgs[assistantMsgs.length - 1].content.length < 2000;
  };

  const detector = new ContextRotDetector({
    rules: [
      { rule: "Use markdown formatting", testFn: checkMarkdown },
      {
        rule: "Keep responses concise (<2000 chars)",
        testFn: checkConciseness,
      },
    ],
    checkInterval: 5,
  });

  const messages: ChatMessage[] = [
    { role: "system", content: "You are a code reviewer. Use markdown." },
  ];

  for (let i = 0; i < 15; i++) {
    messages.push({ role: "user", content: `Review change #${i + 1}` });

    if (i < 8) {
      messages.push({
        role: "assistant",
        content: `## Review #${i + 1}\n\n**Status**: Approved\n\n\`\`\`python\n# looks good\n\`\`\``,
      });
    } else {
      messages.push({
        role: "assistant",
        content: `Change ${i + 1} looks fine. No issues found.`,
      });
    }

    const report = detector.onMessage(messages);
    if (report) {
      console.log(`--- Health check at turn ${i + 1} ---`);
      console.log(reportToContextBlock(report));
      console.log();
    }
  }

  console.log(
    `Health trend: ${detector.healthTrend.map((s) => `${Math.round(s * 100)}%`)}`
  );
  console.log(`Is degrading: ${detector.isDegrading}`);
}

main();
