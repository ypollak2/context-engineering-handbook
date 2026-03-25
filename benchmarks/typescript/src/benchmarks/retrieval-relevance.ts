/**
 * Retrieval Relevance Benchmark.
 *
 * Given a question and a set of retrieved chunks (some relevant, some not),
 * measures whether the LLM uses the relevant chunks and ignores irrelevant ones.
 */

import type { LLMClient } from "../utils/llm-client.js";

interface Chunk {
  readonly id: string;
  readonly content: string;
  readonly isRelevant: boolean;
}

interface RetrievalScenario {
  readonly question: string;
  readonly expectedAnswerFragment: string;
  readonly chunks: readonly Chunk[];
}

const SCENARIOS: readonly RetrievalScenario[] = [
  {
    question: "What is the maximum file upload size in our API?",
    expectedAnswerFragment: "50MB",
    chunks: [
      { id: "doc-1", content: "API File Upload Limits: The maximum file upload size is 50MB per request. Files larger than 50MB must use the multipart chunked upload endpoint. Supported formats include PNG, JPG, PDF, and DOCX.", isRelevant: true },
      { id: "doc-2", content: "Authentication: All API requests require a Bearer token in the Authorization header. Tokens expire after 24 hours and must be refreshed using the /auth/refresh endpoint.", isRelevant: false },
      { id: "doc-3", content: "Rate Limiting: Free tier users are limited to 100 requests per minute. Premium users get 1000 requests per minute. Enterprise users have custom limits.", isRelevant: false },
      { id: "doc-4", content: "File Upload Best Practices: For optimal performance, compress images before uploading. Use the /upload/presigned endpoint to get a pre-signed URL for direct uploads to S3.", isRelevant: true },
      { id: "doc-5", content: "Webhook Configuration: Webhooks can be configured in the dashboard under Settings > Integrations. Each webhook must have a valid HTTPS URL and a secret key for signature verification.", isRelevant: false },
    ],
  },
  {
    question: "How do I configure database connection pooling in our framework?",
    expectedAnswerFragment: "pool_size",
    chunks: [
      { id: "doc-1", content: "Database Connection Pooling: Set pool_size in your config.yaml to control the connection pool. Default is 10. For production, recommend pool_size: 25 with max_overflow: 10. Use pool_recycle: 3600 to prevent stale connections.", isRelevant: true },
      { id: "doc-2", content: "Logging Configuration: Set log_level to DEBUG, INFO, WARNING, or ERROR in config.yaml. Structured logging uses JSON format by default. Log rotation is configured with max_size: 100MB and keep: 5.", isRelevant: false },
      { id: "doc-3", content: "Database Migrations: Run 'framework migrate up' to apply pending migrations. Use 'framework migrate create <name>' to generate a new migration file. Migrations are stored in the db/migrations directory.", isRelevant: false },
      { id: "doc-4", content: "Connection Health Checks: The framework performs automatic health checks on pooled connections. Configure health_check_interval: 30 in the database section. Failed connections are automatically removed from the pool.", isRelevant: true },
      { id: "doc-5", content: "Caching Layer: Redis caching is configured with cache_ttl: 300 in config.yaml. The framework supports both Redis and Memcached backends. Use the @cache decorator on repository methods.", isRelevant: false },
      { id: "doc-6", content: "Environment Variables: Database credentials should be set via DB_HOST, DB_PORT, DB_USER, and DB_PASS environment variables. Never hardcode credentials in config files.", isRelevant: false },
    ],
  },
  {
    question: "What is the retry policy for failed message processing?",
    expectedAnswerFragment: "exponential backoff",
    chunks: [
      { id: "doc-1", content: "Message Queue Retry Policy: Failed messages are retried with exponential backoff. Initial delay is 1 second, doubling with each retry up to a maximum of 5 retries. After 5 failures, messages are moved to the dead letter queue (DLQ).", isRelevant: true },
      { id: "doc-2", content: "Message Serialization: All messages must be serialized as JSON. The maximum message size is 256KB. Binary payloads should be stored in S3 with a reference URL in the message body.", isRelevant: false },
      { id: "doc-3", content: "Queue Monitoring: Use the /admin/queues endpoint to view queue depths and processing rates. Alert when queue depth exceeds 10,000 messages or processing latency exceeds 30 seconds.", isRelevant: false },
      { id: "doc-4", content: "Dead Letter Queue: Messages in the DLQ can be inspected via the admin dashboard. Use the 'reprocess' action to move messages back to the main queue. DLQ messages are retained for 14 days before automatic deletion.", isRelevant: true },
      { id: "doc-5", content: "Topic Subscriptions: Consumers subscribe to topics using pattern matching. Wildcards are supported: 'orders.*' matches 'orders.created' and 'orders.updated'. Each consumer group maintains its own offset.", isRelevant: false },
    ],
  },
  {
    question: "How do I enable two-factor authentication for admin users?",
    expectedAnswerFragment: "TOTP",
    chunks: [
      { id: "doc-1", content: "Two-Factor Authentication: Admin users can enable 2FA via Settings > Security. The system supports TOTP (Time-based One-Time Password) using apps like Google Authenticator. Backup codes are generated on setup - store them securely.", isRelevant: true },
      { id: "doc-2", content: "Role-Based Access Control: Roles are defined in the admin panel under Users > Roles. Each role has granular permissions for read, write, and delete operations. Custom roles can be created for specific use cases.", isRelevant: false },
      { id: "doc-3", content: "Session Management: Admin sessions expire after 30 minutes of inactivity. The session_timeout can be configured in security.yaml. Concurrent session limits can be set per user or per role.", isRelevant: false },
      { id: "doc-4", content: "Password Policy: Admin passwords must be at least 12 characters with uppercase, lowercase, numbers, and special characters. Passwords expire every 90 days. The last 10 passwords cannot be reused.", isRelevant: false },
      { id: "doc-5", content: "2FA Recovery: If a user loses access to their 2FA device, an admin with 'security_admin' role can reset their 2FA. The user will need to set up 2FA again on next login. Recovery requires identity verification via email.", isRelevant: true },
    ],
  },
];

function formatChunks(chunks: readonly Chunk[]): string {
  return chunks
    .map((c) => `[${c.id}]\n${c.content}`)
    .join("\n\n---\n\n");
}

function checkChunkUsage(
  response: string,
  chunks: readonly Chunk[]
): { relevantUsed: Set<string>; irrelevantUsed: Set<string> } {
  const relevantUsed = new Set<string>();
  const irrelevantUsed = new Set<string>();

  for (const chunk of chunks) {
    const words = chunk.content.split(/\s+/);
    const keyPhrases: string[] = [];
    for (let i = 0; i < words.length - 2; i++) {
      const phrase = words.slice(i, i + 3).join(" ").toLowerCase();
      if (phrase.length > 15) {
        keyPhrases.push(phrase);
      }
    }

    let referenced =
      response.toLowerCase().includes(chunk.id.toLowerCase()) ||
      keyPhrases.slice(0, 5).some((p) => response.toLowerCase().includes(p));

    if (referenced) {
      if (chunk.isRelevant) {
        relevantUsed.add(chunk.id);
      } else {
        irrelevantUsed.add(chunk.id);
      }
    }
  }

  return { relevantUsed, irrelevantUsed };
}

interface ScenarioResult {
  readonly question: string;
  readonly answerCorrect: boolean;
  readonly relevantUsed: readonly string[];
  readonly relevantTotal: number;
  readonly irrelevantUsed: readonly string[];
  readonly irrelevantTotal: number;
  readonly utilizationRate: number;
  readonly contaminationRate: number;
}

export interface RetrievalRelevanceResult {
  readonly benchmark: "retrieval_relevance";
  readonly model: string;
  readonly avgAnswerAccuracy: number;
  readonly avgUtilizationRate: number;
  readonly avgContaminationRate: number;
  readonly scenarios: readonly ScenarioResult[];
}

export async function runRetrievalRelevance(
  client: LLMClient
): Promise<RetrievalRelevanceResult> {
  const results: ScenarioResult[] = [];

  for (const scenario of SCENARIOS) {
    const context = formatChunks(scenario.chunks);

    const response = await client.complete(
      [{ role: "user", content: scenario.question }],
      `Answer the question using ONLY the provided context chunks. Reference the chunk IDs (e.g., [doc-1]) that you used in your answer. Be specific and include exact values from the relevant chunks.\n\nContext:\n${context}`
    );

    const answerCorrect = response.content
      .toLowerCase()
      .includes(scenario.expectedAnswerFragment.toLowerCase());

    const { relevantUsed, irrelevantUsed } = checkChunkUsage(
      response.content,
      scenario.chunks
    );

    const relevantTotal = scenario.chunks.filter((c) => c.isRelevant).length;
    const irrelevantTotal = scenario.chunks.filter((c) => !c.isRelevant).length;

    results.push({
      question: scenario.question,
      answerCorrect,
      relevantUsed: [...relevantUsed].sort(),
      relevantTotal,
      irrelevantUsed: [...irrelevantUsed].sort(),
      irrelevantTotal,
      utilizationRate:
        relevantTotal > 0
          ? Math.round((relevantUsed.size / relevantTotal) * 10000) / 10000
          : 0,
      contaminationRate:
        irrelevantTotal > 0
          ? Math.round((irrelevantUsed.size / irrelevantTotal) * 10000) / 10000
          : 0,
    });
  }

  const n = results.length;

  return {
    benchmark: "retrieval_relevance",
    model: client.model,
    avgAnswerAccuracy:
      n > 0
        ? Math.round(
            (results.filter((r) => r.answerCorrect).length / n) * 10000
          ) / 10000
        : 0,
    avgUtilizationRate:
      n > 0
        ? Math.round(
            (results.reduce((s, r) => s + r.utilizationRate, 0) / n) * 10000
          ) / 10000
        : 0,
    avgContaminationRate:
      n > 0
        ? Math.round(
            (results.reduce((s, r) => s + r.contaminationRate, 0) / n) * 10000
          ) / 10000
        : 0,
    scenarios: results,
  };
}
