"""Retrieval Relevance Benchmark.

Given a question and a set of retrieved chunks (some relevant, some not),
measures whether the LLM uses the relevant chunks and ignores irrelevant ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from benchmarks.utils.llm_client import LLMClient, Message
from benchmarks.utils.metrics import count_tokens


@dataclass(frozen=True)
class Chunk:
    """A retrieved context chunk."""

    id: str
    content: str
    is_relevant: bool


@dataclass(frozen=True)
class RetrievalScenario:
    """A question with relevant and irrelevant chunks."""

    question: str
    expected_answer_fragment: str
    chunks: tuple[Chunk, ...]


SCENARIOS: tuple[RetrievalScenario, ...] = (
    RetrievalScenario(
        question="What is the maximum file upload size in our API?",
        expected_answer_fragment="50MB",
        chunks=(
            Chunk("doc-1", "API File Upload Limits: The maximum file upload size is 50MB per request. Files larger than 50MB must use the multipart chunked upload endpoint. Supported formats include PNG, JPG, PDF, and DOCX.", True),
            Chunk("doc-2", "Authentication: All API requests require a Bearer token in the Authorization header. Tokens expire after 24 hours and must be refreshed using the /auth/refresh endpoint.", False),
            Chunk("doc-3", "Rate Limiting: Free tier users are limited to 100 requests per minute. Premium users get 1000 requests per minute. Enterprise users have custom limits.", False),
            Chunk("doc-4", "File Upload Best Practices: For optimal performance, compress images before uploading. Use the /upload/presigned endpoint to get a pre-signed URL for direct uploads to S3.", True),
            Chunk("doc-5", "Webhook Configuration: Webhooks can be configured in the dashboard under Settings > Integrations. Each webhook must have a valid HTTPS URL and a secret key for signature verification.", False),
        ),
    ),
    RetrievalScenario(
        question="How do I configure database connection pooling in our framework?",
        expected_answer_fragment="pool_size",
        chunks=(
            Chunk("doc-1", "Database Connection Pooling: Set pool_size in your config.yaml to control the connection pool. Default is 10. For production, recommend pool_size: 25 with max_overflow: 10. Use pool_recycle: 3600 to prevent stale connections.", True),
            Chunk("doc-2", "Logging Configuration: Set log_level to DEBUG, INFO, WARNING, or ERROR in config.yaml. Structured logging uses JSON format by default. Log rotation is configured with max_size: 100MB and keep: 5.", False),
            Chunk("doc-3", "Database Migrations: Run 'framework migrate up' to apply pending migrations. Use 'framework migrate create <name>' to generate a new migration file. Migrations are stored in the db/migrations directory.", False),
            Chunk("doc-4", "Connection Health Checks: The framework performs automatic health checks on pooled connections. Configure health_check_interval: 30 in the database section. Failed connections are automatically removed from the pool.", True),
            Chunk("doc-5", "Caching Layer: Redis caching is configured with cache_ttl: 300 in config.yaml. The framework supports both Redis and Memcached backends. Use the @cache decorator on repository methods.", False),
            Chunk("doc-6", "Environment Variables: Database credentials should be set via DB_HOST, DB_PORT, DB_USER, and DB_PASS environment variables. Never hardcode credentials in config files.", False),
        ),
    ),
    RetrievalScenario(
        question="What is the retry policy for failed message processing?",
        expected_answer_fragment="exponential backoff",
        chunks=(
            Chunk("doc-1", "Message Queue Retry Policy: Failed messages are retried with exponential backoff. Initial delay is 1 second, doubling with each retry up to a maximum of 5 retries. After 5 failures, messages are moved to the dead letter queue (DLQ).", True),
            Chunk("doc-2", "Message Serialization: All messages must be serialized as JSON. The maximum message size is 256KB. Binary payloads should be stored in S3 with a reference URL in the message body.", False),
            Chunk("doc-3", "Queue Monitoring: Use the /admin/queues endpoint to view queue depths and processing rates. Alert when queue depth exceeds 10,000 messages or processing latency exceeds 30 seconds.", False),
            Chunk("doc-4", "Dead Letter Queue: Messages in the DLQ can be inspected via the admin dashboard. Use the 'reprocess' action to move messages back to the main queue. DLQ messages are retained for 14 days before automatic deletion.", True),
            Chunk("doc-5", "Topic Subscriptions: Consumers subscribe to topics using pattern matching. Wildcards are supported: 'orders.*' matches 'orders.created' and 'orders.updated'. Each consumer group maintains its own offset.", False),
        ),
    ),
    RetrievalScenario(
        question="How do I enable two-factor authentication for admin users?",
        expected_answer_fragment="TOTP",
        chunks=(
            Chunk("doc-1", "Two-Factor Authentication: Admin users can enable 2FA via Settings > Security. The system supports TOTP (Time-based One-Time Password) using apps like Google Authenticator. Backup codes are generated on setup - store them securely.", True),
            Chunk("doc-2", "Role-Based Access Control: Roles are defined in the admin panel under Users > Roles. Each role has granular permissions for read, write, and delete operations. Custom roles can be created for specific use cases.", False),
            Chunk("doc-3", "Session Management: Admin sessions expire after 30 minutes of inactivity. The session_timeout can be configured in security.yaml. Concurrent session limits can be set per user or per role.", False),
            Chunk("doc-4", "Password Policy: Admin passwords must be at least 12 characters with uppercase, lowercase, numbers, and special characters. Passwords expire every 90 days. The last 10 passwords cannot be reused.", False),
            Chunk("doc-5", "2FA Recovery: If a user loses access to their 2FA device, an admin with 'security_admin' role can reset their 2FA. The user will need to set up 2FA again on next login. Recovery requires identity verification via email.", True),
        ),
    ),
)


def _format_chunks_for_prompt(chunks: tuple[Chunk, ...]) -> str:
    """Format chunks into a context string with chunk IDs."""
    sections = []
    for chunk in chunks:
        sections.append(f"[{chunk.id}]\n{chunk.content}")
    return "\n\n---\n\n".join(sections)


def _check_answer_accuracy(response: str, expected_fragment: str) -> bool:
    """Check if the response contains the expected answer."""
    return expected_fragment.lower() in response.lower()


def _check_chunk_usage(response: str, chunks: tuple[Chunk, ...]) -> tuple[set[str], set[str]]:
    """Determine which chunks were likely used based on response content.

    Returns (relevant_used, irrelevant_used) as sets of chunk IDs.
    """
    relevant_used: set[str] = set()
    irrelevant_used: set[str] = set()

    for chunk in chunks:
        # Check if distinctive content from this chunk appears in the response
        # Use key phrases (3+ word sequences) from the chunk
        words = chunk.content.split()
        key_phrases = []
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i + 3]).lower()
            # Skip very common phrases
            if len(phrase) > 15:
                key_phrases.append(phrase)

        chunk_referenced = any(
            phrase in response.lower() for phrase in key_phrases[:5]
        )

        # Also check if chunk ID is explicitly referenced
        if chunk.id.lower() in response.lower():
            chunk_referenced = True

        if chunk_referenced:
            if chunk.is_relevant:
                relevant_used.add(chunk.id)
            else:
                irrelevant_used.add(chunk.id)

    return relevant_used, irrelevant_used


@dataclass(frozen=True)
class ScenarioResult:
    """Result for a single retrieval scenario."""

    question: str
    answer_correct: bool
    relevant_chunks_used: tuple[str, ...]
    relevant_chunks_total: int
    irrelevant_chunks_used: tuple[str, ...]
    irrelevant_chunks_total: int
    utilization_rate: float
    contamination_rate: float
    response: str


@dataclass(frozen=True)
class RetrievalRelevanceResult:
    """Aggregate results for the retrieval relevance benchmark."""

    scenario_results: tuple[ScenarioResult, ...]
    avg_answer_accuracy: float
    avg_utilization_rate: float
    avg_contamination_rate: float
    model: str

    def to_dict(self) -> dict:
        return {
            "benchmark": "retrieval_relevance",
            "model": self.model,
            "avg_answer_accuracy": round(self.avg_answer_accuracy, 4),
            "avg_utilization_rate": round(self.avg_utilization_rate, 4),
            "avg_contamination_rate": round(self.avg_contamination_rate, 4),
            "scenarios": [
                {
                    "question": s.question,
                    "answer_correct": s.answer_correct,
                    "relevant_used": list(s.relevant_chunks_used),
                    "relevant_total": s.relevant_chunks_total,
                    "irrelevant_used": list(s.irrelevant_chunks_used),
                    "irrelevant_total": s.irrelevant_chunks_total,
                    "utilization_rate": round(s.utilization_rate, 4),
                    "contamination_rate": round(s.contamination_rate, 4),
                }
                for s in self.scenario_results
            ],
        }


@dataclass(frozen=True)
class RetrievalRelevanceBenchmark:
    """Benchmark: Does the LLM use relevant chunks and ignore irrelevant ones?"""

    scenarios: tuple[RetrievalScenario, ...] = SCENARIOS

    def run(self, client: LLMClient) -> RetrievalRelevanceResult:
        """Run the retrieval relevance benchmark."""
        results: list[ScenarioResult] = []

        for scenario in self.scenarios:
            context = _format_chunks_for_prompt(scenario.chunks)

            response = client.complete(
                messages=[Message(role="user", content=scenario.question)],
                system=(
                    "Answer the question using ONLY the provided context chunks. "
                    "Reference the chunk IDs (e.g., [doc-1]) that you used in your answer. "
                    "Be specific and include exact values from the relevant chunks.\n\n"
                    f"Context:\n{context}"
                ),
            )

            answer_correct = _check_answer_accuracy(
                response.content, scenario.expected_answer_fragment
            )

            relevant_used, irrelevant_used = _check_chunk_usage(
                response.content, scenario.chunks
            )

            relevant_total = sum(1 for c in scenario.chunks if c.is_relevant)
            irrelevant_total = sum(1 for c in scenario.chunks if not c.is_relevant)

            utilization = len(relevant_used) / relevant_total if relevant_total > 0 else 0.0
            contamination = len(irrelevant_used) / irrelevant_total if irrelevant_total > 0 else 0.0

            results.append(
                ScenarioResult(
                    question=scenario.question,
                    answer_correct=answer_correct,
                    relevant_chunks_used=tuple(sorted(relevant_used)),
                    relevant_chunks_total=relevant_total,
                    irrelevant_chunks_used=tuple(sorted(irrelevant_used)),
                    irrelevant_chunks_total=irrelevant_total,
                    utilization_rate=utilization,
                    contamination_rate=contamination,
                    response=response.content,
                )
            )

        frozen_results = tuple(results)
        n = len(frozen_results)

        return RetrievalRelevanceResult(
            scenario_results=frozen_results,
            avg_answer_accuracy=sum(1 for r in frozen_results if r.answer_correct) / n if n > 0 else 0.0,
            avg_utilization_rate=sum(r.utilization_rate for r in frozen_results) / n if n > 0 else 0.0,
            avg_contamination_rate=sum(r.contamination_rate for r in frozen_results) / n if n > 0 else 0.0,
            model=client.model,
        )
