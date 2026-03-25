"""Token Efficiency Benchmark.

Measures the signal-to-noise ratio of a context window. Given a task and its
context, calculates what percentage of tokens actually contributed to the
correct answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.utils.llm_client import LLMClient, Message
from benchmarks.utils.metrics import count_tokens


@dataclass(frozen=True)
class ContextSection:
    """A labeled section of context with known relevance."""

    label: str
    content: str
    is_signal: bool  # True if this section is needed for the correct answer


@dataclass(frozen=True)
class EfficiencyScenario:
    """A task with context sections of known relevance."""

    name: str
    task: str
    expected_answer_fragment: str
    sections: tuple[ContextSection, ...]


SCENARIOS: tuple[EfficiencyScenario, ...] = (
    EfficiencyScenario(
        name="api_debugging",
        task="Why is the /users endpoint returning 500 errors since the last deployment?",
        expected_answer_fragment="connection string",
        sections=(
            ContextSection(
                "error_log",
                "2024-01-15 14:32:01 ERROR [user-service] Failed to connect to database: connection string 'postgresql://prod-db:5432/users' is unreachable. Timeout after 30s. Previous connection string was 'postgresql://prod-db-v2:5432/users'.",
                True,
            ),
            ContextSection(
                "deployment_diff",
                "--- config/database.yml\n+++ config/database.yml\n@@ -3,7 +3,7 @@\n production:\n-  host: prod-db-v2\n+  host: prod-db\n   port: 5432\n   database: users",
                True,
            ),
            ContextSection(
                "unrelated_metrics",
                "System Metrics Dashboard:\nCPU Usage: 23% (normal)\nMemory: 4.2GB/16GB (26%)\nDisk I/O: 120 IOPS (low)\nNetwork: 45Mbps in, 12Mbps out\nGC Pauses: avg 2.3ms (healthy)\nThread Count: 142 (normal range)\nOpen File Descriptors: 234/65536",
                False,
            ),
            ContextSection(
                "team_standup_notes",
                "Standup Notes 2024-01-15:\n- Alice: Working on search feature, PR #456 ready for review\n- Bob: Updated documentation for API v3\n- Carol: Fixed CSS alignment on dashboard\n- Dave: Set up new monitoring alerts\n- Eve: Onboarding meeting with new hires",
                False,
            ),
            ContextSection(
                "infrastructure_docs",
                "Infrastructure Overview:\nLoad Balancer: AWS ALB in us-east-1\nCompute: EKS cluster with 12 nodes (m5.xlarge)\nDatabase: RDS PostgreSQL 15.4, Multi-AZ\nCache: ElastiCache Redis 7.0, 3 node cluster\nCDN: CloudFront with 24 edge locations",
                False,
            ),
            ContextSection(
                "recent_prs",
                "Recent Pull Requests:\nPR #457: Update database host configuration (merged 14:28)\nPR #456: Add full-text search to /products endpoint\nPR #455: Bump lodash from 4.17.20 to 4.17.21\nPR #454: Add dark mode toggle to settings page",
                True,
            ),
        ),
    ),
    EfficiencyScenario(
        name="feature_planning",
        task="What technical approach should we use for the real-time notifications feature?",
        expected_answer_fragment="WebSocket",
        sections=(
            ContextSection(
                "requirements",
                "Feature Requirements - Real-time Notifications:\n- Users should see notifications within 2 seconds of event\n- Support 50,000 concurrent connections\n- Notifications include: order updates, messages, system alerts\n- Must work on mobile and web\n- Offline users receive notifications on reconnect",
                True,
            ),
            ContextSection(
                "tech_evaluation",
                "Technology Evaluation:\n1. WebSocket (via Socket.IO): Low latency (<100ms), bidirectional, good browser support. Server memory ~50KB per connection = 2.5GB for 50K connections.\n2. Server-Sent Events: Simpler, unidirectional only. Not suitable for our bidirectional needs.\n3. Long Polling: Higher latency (1-30s), more server load. Fallback option only.\nRecommendation: WebSocket with long-polling fallback.",
                True,
            ),
            ContextSection(
                "company_history",
                "Company History:\nFounded in 2018 by two Stanford graduates. Series A in 2019 ($5M). Series B in 2021 ($25M). Currently 150 employees across San Francisco and London offices. Named in Forbes 30 Under 30 list in 2020.",
                False,
            ),
            ContextSection(
                "hr_policies",
                "Remote Work Policy:\nAll employees may work remotely up to 3 days per week. Core hours are 10am-3pm local time. Team leads may approve full remote for specific roles. Equipment stipend of $1,500 for home office setup.",
                False,
            ),
            ContextSection(
                "current_architecture",
                "Current Architecture:\n- API: Node.js with Express, deployed on Kubernetes\n- Database: PostgreSQL 15 (primary), Redis (cache/sessions)\n- Message Queue: RabbitMQ for async processing\n- Frontend: React 18 with Next.js\n- Mobile: React Native\n- Existing WebSocket infrastructure: None (greenfield)",
                True,
            ),
            ContextSection(
                "marketing_plan",
                "Q1 Marketing Plan:\n- Launch email campaign for new dashboard features\n- Attend 3 industry conferences\n- Partner with 5 tech influencers\n- Increase blog output to 4 posts/month\n- Run A/B test on pricing page",
                False,
            ),
            ContextSection(
                "old_meeting_notes",
                "Engineering All-Hands (2023-06-15):\nTopics discussed: Migrating from Heroku to AWS (complete), Adopting TypeScript (in progress), New CI/CD pipeline with GitHub Actions (complete), Hiring plan for Q3 (5 engineers).",
                False,
            ),
        ),
    ),
    EfficiencyScenario(
        name="performance_diagnosis",
        task="Why did the p99 latency spike to 8 seconds on the /search endpoint yesterday?",
        expected_answer_fragment="missing index",
        sections=(
            ContextSection(
                "slow_query_log",
                "Slow Query Log (2024-01-14):\nQuery: SELECT * FROM products WHERE category_id = $1 AND status = 'active' ORDER BY updated_at DESC LIMIT 20\nDuration: 6.8s (normally 45ms)\nRows scanned: 2,400,000\nNote: Index on (category_id, status, updated_at) was dropped during migration #342. Missing index causes full table scan.",
                True,
            ),
            ContextSection(
                "apm_dashboard",
                "APM Dashboard (2024-01-14):\n/search p99: 8.2s (baseline: 200ms)\n/search p50: 3.1s (baseline: 45ms)\n/search error_rate: 2.3% (timeout errors)\nSpike began at 09:15 UTC, correlates with migration #342 deployment at 09:12 UTC.",
                True,
            ),
            ContextSection(
                "unrelated_service_logs",
                "payment-service logs (2024-01-14):\n09:00 INFO Processing batch of 450 transactions\n09:15 INFO Batch complete, all successful\n09:30 INFO Stripe webhook received for subscription renewal\n10:00 INFO Daily reconciliation started\n10:05 INFO Reconciliation complete, 0 discrepancies",
                False,
            ),
            ContextSection(
                "team_vacation_calendar",
                "Team Availability (January 2024):\n- Alice: OOO Jan 8-12 (vacation)\n- Bob: OOO Jan 15 (doctor appointment)\n- Carol: Available all month\n- Dave: OOO Jan 22-26 (conference)\n- Eve: Half day Jan 10 (school event)",
                False,
            ),
            ContextSection(
                "migration_342",
                "Migration #342 - Clean up deprecated columns:\n- Removed columns: products.legacy_category, products.old_sku\n- Dropped indexes: idx_products_category_status_updated (accidentally included)\n- Added columns: products.variant_group_id\n- Author: Bob (ran on 2024-01-14 at 09:12 UTC)",
                True,
            ),
            ContextSection(
                "kubernetes_events",
                "Kubernetes Events (2024-01-14):\n08:00 Normal Scheduled pod/search-service-7f8d9-abc to node-3\n08:00 Normal Pulled container image 'search-service:2.14.0'\n08:01 Normal Started container search-service\n12:00 Normal Scaled deployment/search-service from 3 to 5 replicas (HPA)\n12:30 Normal Scaled deployment/search-service from 5 to 3 replicas (HPA)",
                False,
            ),
        ),
    ),
)


@dataclass(frozen=True)
class EfficiencyTrialResult:
    """Result of a single token efficiency trial."""

    scenario_name: str
    total_tokens: int
    signal_tokens: int
    noise_tokens: int
    effective_ratio: float
    answer_correct: bool
    answer_correct_signal_only: bool
    response_full: str
    response_signal_only: str


@dataclass(frozen=True)
class TokenEfficiencyResult:
    """Aggregate results for the token efficiency benchmark."""

    trials: tuple[EfficiencyTrialResult, ...]
    avg_effective_ratio: float
    avg_accuracy_full: float
    avg_accuracy_signal_only: float
    total_wasted_tokens: int
    model: str

    def to_dict(self) -> dict:
        return {
            "benchmark": "token_efficiency",
            "model": self.model,
            "avg_effective_ratio": round(self.avg_effective_ratio, 4),
            "avg_accuracy_full_context": round(self.avg_accuracy_full, 4),
            "avg_accuracy_signal_only": round(self.avg_accuracy_signal_only, 4),
            "total_wasted_tokens": self.total_wasted_tokens,
            "trials": [
                {
                    "scenario": t.scenario_name,
                    "total_tokens": t.total_tokens,
                    "signal_tokens": t.signal_tokens,
                    "noise_tokens": t.noise_tokens,
                    "effective_ratio": round(t.effective_ratio, 4),
                    "answer_correct_full": t.answer_correct,
                    "answer_correct_signal_only": t.answer_correct_signal_only,
                }
                for t in self.trials
            ],
        }


@dataclass(frozen=True)
class TokenEfficiencyBenchmark:
    """Benchmark: What fraction of context tokens contribute to correct answers?"""

    scenarios: tuple[EfficiencyScenario, ...] = SCENARIOS

    def run(self, client: LLMClient) -> TokenEfficiencyResult:
        """Run the token efficiency benchmark."""
        trials: list[EfficiencyTrialResult] = []

        for scenario in self.scenarios:
            # Build full context (signal + noise)
            full_context = "\n\n".join(
                f"[{s.label}]\n{s.content}" for s in scenario.sections
            )
            signal_context = "\n\n".join(
                f"[{s.label}]\n{s.content}" for s in scenario.sections if s.is_signal
            )

            total_tokens = count_tokens(full_context, client.model)
            signal_tokens = count_tokens(signal_context, client.model)
            noise_tokens = total_tokens - signal_tokens

            # Test with full context
            response_full = client.complete(
                messages=[Message(role="user", content=scenario.task)],
                system=(
                    "Answer the question based on the provided context. "
                    "Be specific and cite evidence from the context.\n\n"
                    f"Context:\n{full_context}"
                ),
            )

            # Test with signal-only context
            response_signal = client.complete(
                messages=[Message(role="user", content=scenario.task)],
                system=(
                    "Answer the question based on the provided context. "
                    "Be specific and cite evidence from the context.\n\n"
                    f"Context:\n{signal_context}"
                ),
            )

            answer_correct_full = scenario.expected_answer_fragment.lower() in response_full.content.lower()
            answer_correct_signal = scenario.expected_answer_fragment.lower() in response_signal.content.lower()

            effective_ratio = signal_tokens / total_tokens if total_tokens > 0 else 0.0

            trials.append(
                EfficiencyTrialResult(
                    scenario_name=scenario.name,
                    total_tokens=total_tokens,
                    signal_tokens=signal_tokens,
                    noise_tokens=noise_tokens,
                    effective_ratio=effective_ratio,
                    answer_correct=answer_correct_full,
                    answer_correct_signal_only=answer_correct_signal,
                    response_full=response_full.content,
                    response_signal_only=response_signal.content,
                )
            )

        frozen_trials = tuple(trials)
        n = len(frozen_trials)

        return TokenEfficiencyResult(
            trials=frozen_trials,
            avg_effective_ratio=sum(t.effective_ratio for t in frozen_trials) / n if n > 0 else 0.0,
            avg_accuracy_full=sum(1 for t in frozen_trials if t.answer_correct) / n if n > 0 else 0.0,
            avg_accuracy_signal_only=sum(1 for t in frozen_trials if t.answer_correct_signal_only) / n if n > 0 else 0.0,
            total_wasted_tokens=sum(t.noise_tokens for t in frozen_trials),
            model=client.model,
        )
