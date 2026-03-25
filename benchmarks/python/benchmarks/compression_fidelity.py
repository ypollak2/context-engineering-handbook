"""Compression Fidelity Benchmark.

Takes a conversation with known facts and decisions, compresses it using
different strategies, then tests if the compressed version preserves key
information.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from benchmarks.utils.llm_client import LLMClient, Message
from benchmarks.utils.metrics import count_tokens, compression_ratio


class CompressionStrategy(Enum):
    LLM_SUMMARY = "llm_summary"
    TRUNCATION_HEAD = "truncation_head"
    TRUNCATION_TAIL = "truncation_tail"
    KEY_POINTS_EXTRACTION = "key_points_extraction"


@dataclass(frozen=True)
class ConversationScenario:
    """A conversation scenario with verifiable facts and decisions."""

    name: str
    messages: tuple[tuple[str, str], ...]  # (role, content) pairs
    facts: tuple[tuple[str, str], ...]  # (question, expected_fragment) pairs
    decisions: tuple[tuple[str, str], ...]  # (question, expected_fragment) pairs


# Self-contained test scenarios
SCENARIOS: tuple[ConversationScenario, ...] = (
    ConversationScenario(
        name="architecture_review",
        messages=(
            ("user", "We need to redesign the payment system. Current throughput is 500 TPS but we need 5000 TPS by Q3."),
            ("assistant", "I recommend a microservices architecture with event sourcing. We should use Kafka for the event bus and PostgreSQL for the command side, with a separate read model in Redis."),
            ("user", "What about the existing Oracle database? We have 3 years left on the license."),
            ("assistant", "We can keep Oracle as the system of record during the transition. The new services will use PostgreSQL, and we'll implement a change data capture pipeline from Oracle using Debezium to keep systems in sync."),
            ("user", "The team agreed to use Go for the new services instead of Java. Sarah will lead the migration team of 5 engineers starting February 1st."),
            ("assistant", "Good choice. Go's concurrency model is well-suited for high-throughput payment processing. I suggest starting with the authorization service since it's the bottleneck. Target timeline: authorization service by March, settlement by April, reconciliation by May."),
            ("user", "Budget approved: $2.3M for the migration project. We also decided to use Stripe as the fallback processor instead of Adyen."),
            ("assistant", "Understood. With the $2.3M budget, I recommend allocating roughly 60% to engineering ($1.38M), 25% to infrastructure ($575K), and 15% to testing and contingency ($345K). Stripe fallback makes sense given our existing integration."),
        ),
        facts=(
            ("What is the current payment throughput?", "500 TPS"),
            ("What is the target throughput?", "5000 TPS"),
            ("What database is used for the command side?", "PostgreSQL"),
            ("What is the migration budget?", "$2.3M"),
            ("What tool is used for change data capture?", "Debezium"),
            ("What is the read model stored in?", "Redis"),
        ),
        decisions=(
            ("What programming language was chosen for new services?", "Go"),
            ("Who leads the migration team?", "Sarah"),
            ("What is the fallback payment processor?", "Stripe"),
            ("Which service is built first?", "authorization"),
            ("When does the migration team start?", "February 1st"),
        ),
    ),
    ConversationScenario(
        name="incident_response",
        messages=(
            ("user", "URGENT: Production is down. Error rate spiked to 45% at 14:32 UTC. All API endpoints returning 503."),
            ("assistant", "Checking the dashboards. The root cause appears to be a connection pool exhaustion in the user-service. Current active connections: 500/500 (max). The spike correlates with a deployment of version 2.7.1 at 14:28 UTC."),
            ("user", "Rolling back to 2.7.0 now. But we also noticed memory usage on db-primary-01 is at 97%."),
            ("assistant", "The memory issue on db-primary-01 is likely caused by the new query in 2.7.1 that does a full table scan on the sessions table (450M rows). Immediate actions: 1) Complete the rollback to 2.7.0. 2) Restart the connection pools. 3) Run VACUUM on the sessions table during the maintenance window at 02:00 UTC."),
            ("user", "Rollback complete at 14:47 UTC. Error rate dropping. Team decided: Maria owns the post-mortem, due by Friday. We're adding connection pool monitoring alerts with a threshold at 80%."),
            ("assistant", "Good. Recovery confirmed. Final incident timeline: Detection 14:32, Root cause identified 14:35, Rollback initiated 14:40, Rollback complete 14:47, Full recovery 14:52. Total downtime: 20 minutes. I recommend adding a pre-deployment check for query plans on tables over 100M rows."),
        ),
        facts=(
            ("What was the peak error rate?", "45%"),
            ("What time did the incident start?", "14:32"),
            ("What version caused the issue?", "2.7.1"),
            ("How many rows in the sessions table?", "450M"),
            ("What was the total downtime?", "20 minutes"),
            ("What is the connection pool alert threshold?", "80%"),
        ),
        decisions=(
            ("Who owns the post-mortem?", "Maria"),
            ("When is the post-mortem due?", "Friday"),
            ("What version was rolled back to?", "2.7.0"),
            ("When is the maintenance window for VACUUM?", "02:00 UTC"),
        ),
    ),
)


def _compress_llm_summary(conversation: str, client: LLMClient) -> str:
    """Compress via LLM summarization."""
    response = client.complete(
        messages=[
            Message(
                role="user",
                content=(
                    "Summarize the following conversation concisely, preserving all "
                    "specific facts, numbers, names, dates, and decisions:\n\n"
                    f"{conversation}"
                ),
            ),
        ],
        system="You are a precise summarizer. Preserve all specific details, numbers, and decisions.",
    )
    return response.content


def _compress_truncation_head(conversation: str, ratio: float = 0.5) -> str:
    """Keep only the first portion of the conversation."""
    lines = conversation.split("\n")
    keep = max(1, int(len(lines) * ratio))
    return "\n".join(lines[:keep])


def _compress_truncation_tail(conversation: str, ratio: float = 0.5) -> str:
    """Keep only the last portion of the conversation."""
    lines = conversation.split("\n")
    keep = max(1, int(len(lines) * ratio))
    return "\n".join(lines[-keep:])


def _compress_key_points(conversation: str, client: LLMClient) -> str:
    """Extract only key points and decisions."""
    response = client.complete(
        messages=[
            Message(
                role="user",
                content=(
                    "Extract ONLY the key facts, decisions, and action items from "
                    "this conversation as a bullet-point list. Include all specific "
                    "numbers, names, and dates:\n\n"
                    f"{conversation}"
                ),
            ),
        ],
        system="Extract key points as concise bullet points. Preserve all specifics.",
    )
    return response.content


@dataclass(frozen=True)
class CompressionTrialResult:
    """Result of testing one compression strategy on one scenario."""

    scenario_name: str
    strategy: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    facts_retained: int
    facts_total: int
    fact_retention_rate: float
    decisions_retained: int
    decisions_total: int
    decision_retention_rate: float


@dataclass(frozen=True)
class CompressionFidelityResult:
    """Aggregate results for the compression fidelity benchmark."""

    trials: tuple[CompressionTrialResult, ...]
    by_strategy: dict[str, dict[str, float]]
    model: str

    def to_dict(self) -> dict:
        return {
            "benchmark": "compression_fidelity",
            "model": self.model,
            "by_strategy": {
                k: {sk: round(sv, 4) for sk, sv in v.items()}
                for k, v in self.by_strategy.items()
            },
            "trials": [
                {
                    "scenario": t.scenario_name,
                    "strategy": t.strategy,
                    "compression_ratio": round(t.compression_ratio, 4),
                    "fact_retention_rate": round(t.fact_retention_rate, 4),
                    "decision_retention_rate": round(t.decision_retention_rate, 4),
                    "original_tokens": t.original_tokens,
                    "compressed_tokens": t.compressed_tokens,
                }
                for t in self.trials
            ],
        }


def _format_conversation(messages: tuple[tuple[str, str], ...]) -> str:
    """Format conversation messages into a readable string."""
    return "\n".join(f"{role.upper()}: {content}" for role, content in messages)


def _test_information_preservation(
    compressed: str,
    questions: tuple[tuple[str, str], ...],
    client: LLMClient,
) -> int:
    """Test how many facts/decisions are preserved in the compressed context."""
    retained = 0
    for question, expected_fragment in questions:
        response = client.complete(
            messages=[Message(role="user", content=question)],
            system=(
                "Answer the question based ONLY on the following context. "
                "Be specific and include exact values. If the information is not "
                "available in the context, say 'NOT FOUND'.\n\n"
                f"Context:\n{compressed}"
            ),
        )
        if expected_fragment.lower() in response.content.lower():
            retained += 1
    return retained


@dataclass(frozen=True)
class CompressionFidelityBenchmark:
    """Benchmark: How well do compression strategies preserve information?"""

    scenarios: tuple[ConversationScenario, ...] = SCENARIOS
    strategies: tuple[CompressionStrategy, ...] = (
        CompressionStrategy.LLM_SUMMARY,
        CompressionStrategy.TRUNCATION_HEAD,
        CompressionStrategy.TRUNCATION_TAIL,
        CompressionStrategy.KEY_POINTS_EXTRACTION,
    )

    def run(self, client: LLMClient) -> CompressionFidelityResult:
        """Run the compression fidelity benchmark."""
        trials: list[CompressionTrialResult] = []

        for scenario in self.scenarios:
            original = _format_conversation(scenario.messages)
            original_tokens = count_tokens(original, client.model)

            for strategy in self.strategies:
                compressed = self._apply_compression(original, strategy, client)
                compressed_tokens = count_tokens(compressed, client.model)

                facts_retained = _test_information_preservation(
                    compressed, scenario.facts, client
                )
                decisions_retained = _test_information_preservation(
                    compressed, scenario.decisions, client
                )

                trial = CompressionTrialResult(
                    scenario_name=scenario.name,
                    strategy=strategy.value,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    compression_ratio=compression_ratio(original_tokens, compressed_tokens),
                    facts_retained=facts_retained,
                    facts_total=len(scenario.facts),
                    fact_retention_rate=facts_retained / len(scenario.facts) if scenario.facts else 0.0,
                    decisions_retained=decisions_retained,
                    decisions_total=len(scenario.decisions),
                    decision_retention_rate=decisions_retained / len(scenario.decisions) if scenario.decisions else 0.0,
                )
                trials.append(trial)

        frozen_trials = tuple(trials)

        by_strategy: dict[str, dict[str, float]] = {}
        for strategy in self.strategies:
            strat_trials = [t for t in frozen_trials if t.strategy == strategy.value]
            if strat_trials:
                by_strategy[strategy.value] = {
                    "avg_compression_ratio": sum(t.compression_ratio for t in strat_trials) / len(strat_trials),
                    "avg_fact_retention": sum(t.fact_retention_rate for t in strat_trials) / len(strat_trials),
                    "avg_decision_retention": sum(t.decision_retention_rate for t in strat_trials) / len(strat_trials),
                }

        return CompressionFidelityResult(
            trials=frozen_trials,
            by_strategy=by_strategy,
            model=client.model,
        )

    def _apply_compression(
        self,
        original: str,
        strategy: CompressionStrategy,
        client: LLMClient,
    ) -> str:
        """Apply a compression strategy to the conversation."""
        match strategy:
            case CompressionStrategy.LLM_SUMMARY:
                return _compress_llm_summary(original, client)
            case CompressionStrategy.TRUNCATION_HEAD:
                return _compress_truncation_head(original)
            case CompressionStrategy.TRUNCATION_TAIL:
                return _compress_truncation_tail(original)
            case CompressionStrategy.KEY_POINTS_EXTRACTION:
                return _compress_key_points(original, client)
