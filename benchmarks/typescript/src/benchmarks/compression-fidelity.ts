/**
 * Compression Fidelity Benchmark.
 *
 * Takes a conversation with known facts and decisions, compresses it using
 * different strategies, then tests if the compressed version preserves key
 * information.
 */

import type { LLMClient } from "../utils/llm-client.js";
import { countTokens, compressionRatio } from "../utils/metrics.js";

type CompressionStrategy =
  | "llm_summary"
  | "truncation_head"
  | "truncation_tail"
  | "key_points_extraction";

interface ConversationMessage {
  readonly role: string;
  readonly content: string;
}

interface VerificationPair {
  readonly question: string;
  readonly expectedFragment: string;
}

interface ConversationScenario {
  readonly name: string;
  readonly messages: readonly ConversationMessage[];
  readonly facts: readonly VerificationPair[];
  readonly decisions: readonly VerificationPair[];
}

const SCENARIOS: readonly ConversationScenario[] = [
  {
    name: "architecture_review",
    messages: [
      { role: "user", content: "We need to redesign the payment system. Current throughput is 500 TPS but we need 5000 TPS by Q3." },
      { role: "assistant", content: "I recommend a microservices architecture with event sourcing. We should use Kafka for the event bus and PostgreSQL for the command side, with a separate read model in Redis." },
      { role: "user", content: "What about the existing Oracle database? We have 3 years left on the license." },
      { role: "assistant", content: "We can keep Oracle as the system of record during the transition. The new services will use PostgreSQL, and we'll implement a change data capture pipeline from Oracle using Debezium to keep systems in sync." },
      { role: "user", content: "The team agreed to use Go for the new services instead of Java. Sarah will lead the migration team of 5 engineers starting February 1st." },
      { role: "assistant", content: "Good choice. Go's concurrency model is well-suited for high-throughput payment processing. I suggest starting with the authorization service since it's the bottleneck. Target timeline: authorization service by March, settlement by April, reconciliation by May." },
      { role: "user", content: "Budget approved: $2.3M for the migration project. We also decided to use Stripe as the fallback processor instead of Adyen." },
      { role: "assistant", content: "Understood. With the $2.3M budget, I recommend allocating roughly 60% to engineering ($1.38M), 25% to infrastructure ($575K), and 15% to testing and contingency ($345K). Stripe fallback makes sense given our existing integration." },
    ],
    facts: [
      { question: "What is the current payment throughput?", expectedFragment: "500 TPS" },
      { question: "What is the target throughput?", expectedFragment: "5000 TPS" },
      { question: "What database is used for the command side?", expectedFragment: "PostgreSQL" },
      { question: "What is the migration budget?", expectedFragment: "$2.3M" },
      { question: "What tool is used for change data capture?", expectedFragment: "Debezium" },
      { question: "What is the read model stored in?", expectedFragment: "Redis" },
    ],
    decisions: [
      { question: "What programming language was chosen for new services?", expectedFragment: "Go" },
      { question: "Who leads the migration team?", expectedFragment: "Sarah" },
      { question: "What is the fallback payment processor?", expectedFragment: "Stripe" },
      { question: "Which service is built first?", expectedFragment: "authorization" },
      { question: "When does the migration team start?", expectedFragment: "February 1st" },
    ],
  },
  {
    name: "incident_response",
    messages: [
      { role: "user", content: "URGENT: Production is down. Error rate spiked to 45% at 14:32 UTC. All API endpoints returning 503." },
      { role: "assistant", content: "Checking the dashboards. The root cause appears to be a connection pool exhaustion in the user-service. Current active connections: 500/500 (max). The spike correlates with a deployment of version 2.7.1 at 14:28 UTC." },
      { role: "user", content: "Rolling back to 2.7.0 now. But we also noticed memory usage on db-primary-01 is at 97%." },
      { role: "assistant", content: "The memory issue on db-primary-01 is likely caused by the new query in 2.7.1 that does a full table scan on the sessions table (450M rows). Immediate actions: 1) Complete the rollback to 2.7.0. 2) Restart the connection pools. 3) Run VACUUM on the sessions table during the maintenance window at 02:00 UTC." },
      { role: "user", content: "Rollback complete at 14:47 UTC. Error rate dropping. Team decided: Maria owns the post-mortem, due by Friday. We're adding connection pool monitoring alerts with a threshold at 80%." },
      { role: "assistant", content: "Good. Recovery confirmed. Final incident timeline: Detection 14:32, Root cause identified 14:35, Rollback initiated 14:40, Rollback complete 14:47, Full recovery 14:52. Total downtime: 20 minutes. I recommend adding a pre-deployment check for query plans on tables over 100M rows." },
    ],
    facts: [
      { question: "What was the peak error rate?", expectedFragment: "45%" },
      { question: "What time did the incident start?", expectedFragment: "14:32" },
      { question: "What version caused the issue?", expectedFragment: "2.7.1" },
      { question: "How many rows in the sessions table?", expectedFragment: "450M" },
      { question: "What was the total downtime?", expectedFragment: "20 minutes" },
      { question: "What is the connection pool alert threshold?", expectedFragment: "80%" },
    ],
    decisions: [
      { question: "Who owns the post-mortem?", expectedFragment: "Maria" },
      { question: "When is the post-mortem due?", expectedFragment: "Friday" },
      { question: "What version was rolled back to?", expectedFragment: "2.7.0" },
      { question: "When is the maintenance window for VACUUM?", expectedFragment: "02:00 UTC" },
    ],
  },
];

function formatConversation(messages: readonly ConversationMessage[]): string {
  return messages
    .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
    .join("\n");
}

async function compressLLMSummary(
  conversation: string,
  client: LLMClient
): Promise<string> {
  const response = await client.complete(
    [
      {
        role: "user",
        content: `Summarize the following conversation concisely, preserving all specific facts, numbers, names, dates, and decisions:\n\n${conversation}`,
      },
    ],
    "You are a precise summarizer. Preserve all specific details, numbers, and decisions."
  );
  return response.content;
}

function compressTruncationHead(
  conversation: string,
  ratio: number = 0.5
): string {
  const lines = conversation.split("\n");
  const keep = Math.max(1, Math.floor(lines.length * ratio));
  return lines.slice(0, keep).join("\n");
}

function compressTruncationTail(
  conversation: string,
  ratio: number = 0.5
): string {
  const lines = conversation.split("\n");
  const keep = Math.max(1, Math.floor(lines.length * ratio));
  return lines.slice(-keep).join("\n");
}

async function compressKeyPoints(
  conversation: string,
  client: LLMClient
): Promise<string> {
  const response = await client.complete(
    [
      {
        role: "user",
        content: `Extract ONLY the key facts, decisions, and action items from this conversation as a bullet-point list. Include all specific numbers, names, and dates:\n\n${conversation}`,
      },
    ],
    "Extract key points as concise bullet points. Preserve all specifics."
  );
  return response.content;
}

async function testPreservation(
  compressed: string,
  pairs: readonly VerificationPair[],
  client: LLMClient
): Promise<number> {
  let retained = 0;
  for (const pair of pairs) {
    const response = await client.complete(
      [{ role: "user", content: pair.question }],
      `Answer the question based ONLY on the following context. Be specific and include exact values. If the information is not available in the context, say 'NOT FOUND'.\n\nContext:\n${compressed}`
    );
    if (
      response.content.toLowerCase().includes(pair.expectedFragment.toLowerCase())
    ) {
      retained++;
    }
  }
  return retained;
}

interface CompressionTrialResult {
  readonly scenario: string;
  readonly strategy: string;
  readonly compressionRatio: number;
  readonly factRetentionRate: number;
  readonly decisionRetentionRate: number;
  readonly originalTokens: number;
  readonly compressedTokens: number;
}

export interface CompressionFidelityResult {
  readonly benchmark: "compression_fidelity";
  readonly model: string;
  readonly byStrategy: Record<
    string,
    {
      readonly avgCompressionRatio: number;
      readonly avgFactRetention: number;
      readonly avgDecisionRetention: number;
    }
  >;
  readonly trials: readonly CompressionTrialResult[];
}

const STRATEGIES: readonly CompressionStrategy[] = [
  "llm_summary",
  "truncation_head",
  "truncation_tail",
  "key_points_extraction",
];

async function applyCompression(
  original: string,
  strategy: CompressionStrategy,
  client: LLMClient
): Promise<string> {
  switch (strategy) {
    case "llm_summary":
      return compressLLMSummary(original, client);
    case "truncation_head":
      return compressTruncationHead(original);
    case "truncation_tail":
      return compressTruncationTail(original);
    case "key_points_extraction":
      return compressKeyPoints(original, client);
  }
}

export async function runCompressionFidelity(
  client: LLMClient
): Promise<CompressionFidelityResult> {
  const trials: CompressionTrialResult[] = [];

  for (const scenario of SCENARIOS) {
    const original = formatConversation(scenario.messages);
    const originalTokens = countTokens(original);

    for (const strategy of STRATEGIES) {
      const compressed = await applyCompression(original, strategy, client);
      const compressedTokens = countTokens(compressed);

      const factsRetained = await testPreservation(
        compressed,
        scenario.facts,
        client
      );
      const decisionsRetained = await testPreservation(
        compressed,
        scenario.decisions,
        client
      );

      trials.push({
        scenario: scenario.name,
        strategy,
        compressionRatio:
          Math.round(
            compressionRatio(originalTokens, compressedTokens) * 10000
          ) / 10000,
        factRetentionRate:
          Math.round(
            (factsRetained / scenario.facts.length) * 10000
          ) / 10000,
        decisionRetentionRate:
          Math.round(
            (decisionsRetained / scenario.decisions.length) * 10000
          ) / 10000,
        originalTokens,
        compressedTokens,
      });
    }
  }

  const byStrategy: Record<string, any> = {};
  for (const strategy of STRATEGIES) {
    const stratTrials = trials.filter((t) => t.strategy === strategy);
    if (stratTrials.length > 0) {
      byStrategy[strategy] = {
        avgCompressionRatio:
          Math.round(
            (stratTrials.reduce((s, t) => s + t.compressionRatio, 0) /
              stratTrials.length) *
              10000
          ) / 10000,
        avgFactRetention:
          Math.round(
            (stratTrials.reduce((s, t) => s + t.factRetentionRate, 0) /
              stratTrials.length) *
              10000
          ) / 10000,
        avgDecisionRetention:
          Math.round(
            (stratTrials.reduce((s, t) => s + t.decisionRetentionRate, 0) /
              stratTrials.length) *
              10000
          ) / 10000,
      };
    }
  }

  return {
    benchmark: "compression_fidelity",
    model: client.model,
    byStrategy,
    trials,
  };
}
