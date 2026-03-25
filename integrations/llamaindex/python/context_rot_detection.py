"""
Context Rot Detection -- LlamaIndex Integration

Implements context health monitoring using LlamaIndex's evaluation framework.
Custom evaluator classes check for instruction drift, contradictions, and
staleness in conversation context, producing structured health reports.

Pattern: https://github.com/context-engineering-handbook/patterns/evaluation/context-rot-detection.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from llama_index.core.llms import ChatMessage, MessageRole


# ---------------------------------------------------------------------------
# Health report types
# ---------------------------------------------------------------------------


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
            "## Context Health Report",
            f"**Score**: {self.overall_score:.0%} ({self.level.value})",
            f"**Messages**: {self.message_count}",
            f"**Est. Tokens**: {self.context_tokens_estimate:,}",
            f"**Action**: {self.recommended_action.value}",
            "",
        ]
        for check in self.checks:
            status = "PASS" if check.score >= 0.7 else "FAIL"
            lines.append(f"- [{status}] {check.dimension}: {check.score:.0%}")
            for fc in check.failed_checks:
                lines.append(f"  - {fc}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Health check evaluators (modeled after LlamaIndex's evaluator pattern)
# ---------------------------------------------------------------------------


class InstructionAdherenceEvaluator:
    """Tests whether the model still follows key instructions.

    Modeled after LlamaIndex's ResponseEvaluator pattern -- each evaluator
    has an `evaluate` method that returns a structured result.

    In production, you could use an LLM to judge adherence:
        from llama_index.core.evaluation import FaithfulnessEvaluator
    """

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        """
        Args:
            rules: List of dicts with 'rule' (str) and 'test_fn'
                   (callable that takes list[ChatMessage] and returns bool).
        """
        self._rules = rules

    def evaluate(self, messages: list[ChatMessage]) -> HealthCheckResult:
        """Check instruction adherence against recent messages."""
        if not self._rules:
            return HealthCheckResult(
                dimension="instruction_adherence",
                score=1.0,
                details="No instructions to check.",
            )

        passed = 0
        failed: list[str] = []

        for rule in self._rules:
            rule_name = rule["rule"]
            test_fn: Callable = rule["test_fn"]
            try:
                if test_fn(messages):
                    passed += 1
                else:
                    failed.append(f"Rule not followed: {rule_name}")
            except Exception as e:
                failed.append(f"Check error for '{rule_name}': {e}")

        score = passed / len(self._rules)
        return HealthCheckResult(
            dimension="instruction_adherence",
            score=score,
            details=f"{passed}/{len(self._rules)} rules followed.",
            failed_checks=tuple(failed),
        )


class ContradictionEvaluator:
    """Scans context for contradictory statements."""

    def evaluate(self, messages: list[ChatMessage]) -> HealthCheckResult:
        assertions: dict[str, list[tuple[str, int]]] = {}
        contradictions: list[str] = []

        for i, msg in enumerate(messages):
            content = msg.content or ""
            for line in content.split("\n"):
                trimmed = line.strip().lower()
                if trimmed.startswith("use ") and len(trimmed.split()) <= 4:
                    key = "tool_preference"
                    if key in assertions:
                        for prev_value, prev_idx in assertions[key]:
                            if prev_value != trimmed:
                                contradictions.append(
                                    f"'{key}' is '{prev_value}' (msg {prev_idx}) "
                                    f"vs '{trimmed}' (msg {i})"
                                )
                        assertions[key].append((trimmed, i))
                    else:
                        assertions[key] = [(trimmed, i)]

        max_contradictions = 5
        score = max(0.0, 1.0 - len(contradictions) / max_contradictions)

        return HealthCheckResult(
            dimension="contradiction_scan",
            score=score,
            details=f"{len(contradictions)} contradictions found.",
            failed_checks=tuple(contradictions[:10]),
        )


class StalenessEvaluator:
    """Checks whether information in context is outdated."""

    def __init__(self, max_age_messages: int = 100) -> None:
        self._max_age = max_age_messages

    def evaluate(self, messages: list[ChatMessage]) -> HealthCheckResult:
        total = len(messages)
        if total <= self._max_age:
            return HealthCheckResult(
                dimension="staleness",
                score=1.0,
                details="Context is within freshness window.",
            )

        stale_count = total - self._max_age
        stale_ratio = stale_count / total
        score = max(0.0, 1.0 - stale_ratio)

        issues: list[str] = []
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


# ---------------------------------------------------------------------------
# Context Rot Detector (orchestrator)
# ---------------------------------------------------------------------------


class ContextRotDetector:
    """Orchestrates periodic context health checks.

    Integrates with LlamaIndex's chat engine by accepting ChatMessage lists.
    Call `on_message()` after each turn to trigger periodic checks.

    Usage with a LlamaIndex chat engine:
        from llama_index.core.chat_engine import SimpleChatEngine

        detector = ContextRotDetector(
            rules=[
                {"rule": "Use markdown", "test_fn": check_markdown_usage},
                {"rule": "Include citations", "test_fn": check_citations},
            ],
            check_interval=10,
        )

        # In your chat loop:
        response = chat_engine.chat(user_input)
        report = detector.on_message(chat_engine.chat_history)
        if report and report.recommended_action != RemediationAction.NONE:
            # Trigger remediation
            ...
    """

    def __init__(
        self,
        rules: list[dict[str, Any]] | None = None,
        check_interval: int = 20,
        degraded_threshold: float = 0.7,
        critical_threshold: float = 0.4,
        max_staleness_age: int = 100,
    ) -> None:
        self._adherence = InstructionAdherenceEvaluator(rules or [])
        self._contradictions = ContradictionEvaluator()
        self._staleness = StalenessEvaluator(max_staleness_age)
        self._check_interval = check_interval
        self._degraded_threshold = degraded_threshold
        self._critical_threshold = critical_threshold
        self._messages_since_check = 0
        self._history: list[ContextHealthReport] = []

    def on_message(
        self, messages: list[ChatMessage]
    ) -> ContextHealthReport | None:
        """Call after each message. Returns a report if a check is triggered."""
        self._messages_since_check += 1
        if self._messages_since_check < self._check_interval:
            return None
        self._messages_since_check = 0
        return self.run_check(messages)

    def run_check(self, messages: list[ChatMessage]) -> ContextHealthReport:
        """Run all health checks and produce a report."""
        recent = messages[-50:] if len(messages) > 50 else messages

        adherence_result = self._adherence.evaluate(recent)
        contradiction_result = self._contradictions.evaluate(messages)
        staleness_result = self._staleness.evaluate(messages)

        checks = [adherence_result, contradiction_result, staleness_result]

        # Weighted average
        weights = {
            "instruction_adherence": 0.5,
            "contradiction_scan": 0.3,
            "staleness": 0.2,
        }
        total_weight = sum(weights.get(c.dimension, 0.2) for c in checks)
        overall = (
            sum(c.score * weights.get(c.dimension, 0.2) for c in checks)
            / total_weight
        )

        if overall >= self._degraded_threshold:
            level = HealthLevel.HEALTHY
            action = RemediationAction.NONE
        elif overall >= self._critical_threshold:
            level = HealthLevel.DEGRADED
            action = RemediationAction.REINJECT_INSTRUCTIONS
        else:
            level = HealthLevel.CRITICAL
            action = RemediationAction.FULL_RESET

        token_estimate = sum(
            len(m.content or "") // 4 for m in messages
        )

        report = ContextHealthReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=overall,
            level=level,
            checks=tuple(checks),
            recommended_action=action,
            message_count=len(messages),
            context_tokens_estimate=token_estimate,
        )

        self._history = [*self._history, report]
        return report

    @property
    def health_trend(self) -> list[float]:
        return [r.overall_score for r in self._history]

    @property
    def is_degrading(self) -> bool:
        scores = self.health_trend
        if len(scores) < 3:
            return False
        recent = scores[-3:]
        return all(recent[i] < recent[i - 1] for i in range(1, len(recent)))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate Context Rot Detection with LlamaIndex ChatMessages."""

    # Define instruction rules to monitor
    def check_markdown(messages: list[ChatMessage]) -> bool:
        """Check if assistant responses use markdown formatting."""
        assistant_msgs = [m for m in messages if m.role == MessageRole.ASSISTANT]
        if not assistant_msgs:
            return True
        last_msg = assistant_msgs[-1].content or ""
        return "#" in last_msg or "```" in last_msg or "**" in last_msg

    def check_conciseness(messages: list[ChatMessage]) -> bool:
        """Check if assistant responses are reasonably concise."""
        assistant_msgs = [m for m in messages if m.role == MessageRole.ASSISTANT]
        if not assistant_msgs:
            return True
        last_msg = assistant_msgs[-1].content or ""
        return len(last_msg) < 2000

    rules = [
        {"rule": "Use markdown formatting", "test_fn": check_markdown},
        {"rule": "Keep responses concise (<2000 chars)", "test_fn": check_conciseness},
    ]

    detector = ContextRotDetector(
        rules=rules,
        check_interval=5,  # Low interval for demo
    )

    # Simulate a conversation
    messages: list[ChatMessage] = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a code reviewer. Use markdown."),
    ]

    # Add some turns
    for i in range(15):
        messages.append(
            ChatMessage(role=MessageRole.USER, content=f"Review change #{i + 1}")
        )
        # Simulate good responses early, degrading later
        if i < 8:
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"## Review #{i + 1}\n\n**Status**: Approved\n\n```python\n# looks good\n```",
                )
            )
        else:
            # Later responses drop markdown (simulating instruction drift)
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"Change {i + 1} looks fine. No issues found.",
                )
            )

        report = detector.on_message(messages)
        if report is not None:
            print(f"--- Health check at turn {i + 1} ---")
            print(report.to_context_block())
            print()

    # Show trend
    print(f"Health trend: {[f'{s:.0%}' for s in detector.health_trend]}")
    print(f"Is degrading: {detector.is_degrading}")


if __name__ == "__main__":
    main()
