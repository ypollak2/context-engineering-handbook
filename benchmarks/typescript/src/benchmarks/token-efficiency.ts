/**
 * Token Efficiency Benchmark.
 *
 * Measures the signal-to-noise ratio of a context window. Given a task and
 * its context, calculates what percentage of tokens actually contributed to
 * the correct answer.
 */

import type { LLMClient } from "../utils/llm-client.js";
import { countTokens } from "../utils/metrics.js";

interface ContextSection {
  readonly label: string;
  readonly content: string;
  readonly isSignal: boolean;
}

interface EfficiencyScenario {
  readonly name: string;
  readonly task: string;
  readonly expectedAnswerFragment: string;
  readonly sections: readonly ContextSection[];
}

const SCENARIOS: readonly EfficiencyScenario[] = [
  {
    name: "api_debugging",
    task: "Why is the /users endpoint returning 500 errors since the last deployment?",
    expectedAnswerFragment: "connection string",
    sections: [
      {
        label: "error_log",
        content: "2024-01-15 14:32:01 ERROR [user-service] Failed to connect to database: connection string 'postgresql://prod-db:5432/users' is unreachable. Timeout after 30s. Previous connection string was 'postgresql://prod-db-v2:5432/users'.",
        isSignal: true,
      },
      {
        label: "deployment_diff",
        content: "--- config/database.yml\n+++ config/database.yml\n@@ -3,7 +3,7 @@\n production:\n-  host: prod-db-v2\n+  host: prod-db\n   port: 5432\n   database: users",
        isSignal: true,
      },
      {
        label: "unrelated_metrics",
        content: "System Metrics Dashboard:\nCPU Usage: 23% (normal)\nMemory: 4.2GB/16GB (26%)\nDisk I/O: 120 IOPS (low)\nNetwork: 45Mbps in, 12Mbps out\nGC Pauses: avg 2.3ms (healthy)\nThread Count: 142 (normal range)\nOpen File Descriptors: 234/65536",
        isSignal: false,
      },
      {
        label: "team_standup_notes",
        content: "Standup Notes 2024-01-15:\n- Alice: Working on search feature, PR #456 ready for review\n- Bob: Updated documentation for API v3\n- Carol: Fixed CSS alignment on dashboard\n- Dave: Set up new monitoring alerts\n- Eve: Onboarding meeting with new hires",
        isSignal: false,
      },
      {
        label: "infrastructure_docs",
        content: "Infrastructure Overview:\nLoad Balancer: AWS ALB in us-east-1\nCompute: EKS cluster with 12 nodes (m5.xlarge)\nDatabase: RDS PostgreSQL 15.4, Multi-AZ\nCache: ElastiCache Redis 7.0, 3 node cluster\nCDN: CloudFront with 24 edge locations",
        isSignal: false,
      },
      {
        label: "recent_prs",
        content: "Recent Pull Requests:\nPR #457: Update database host configuration (merged 14:28)\nPR #456: Add full-text search to /products endpoint\nPR #455: Bump lodash from 4.17.20 to 4.17.21\nPR #454: Add dark mode toggle to settings page",
        isSignal: true,
      },
    ],
  },
  {
    name: "feature_planning",
    task: "What technical approach should we use for the real-time notifications feature?",
    expectedAnswerFragment: "WebSocket",
    sections: [
      {
        label: "requirements",
        content: "Feature Requirements - Real-time Notifications:\n- Users should see notifications within 2 seconds of event\n- Support 50,000 concurrent connections\n- Notifications include: order updates, messages, system alerts\n- Must work on mobile and web\n- Offline users receive notifications on reconnect",
        isSignal: true,
      },
      {
        label: "tech_evaluation",
        content: "Technology Evaluation:\n1. WebSocket (via Socket.IO): Low latency (<100ms), bidirectional, good browser support. Server memory ~50KB per connection = 2.5GB for 50K connections.\n2. Server-Sent Events: Simpler, unidirectional only. Not suitable for our bidirectional needs.\n3. Long Polling: Higher latency (1-30s), more server load. Fallback option only.\nRecommendation: WebSocket with long-polling fallback.",
        isSignal: true,
      },
      {
        label: "company_history",
        content: "Company History:\nFounded in 2018 by two Stanford graduates. Series A in 2019 ($5M). Series B in 2021 ($25M). Currently 150 employees across San Francisco and London offices. Named in Forbes 30 Under 30 list in 2020.",
        isSignal: false,
      },
      {
        label: "hr_policies",
        content: "Remote Work Policy:\nAll employees may work remotely up to 3 days per week. Core hours are 10am-3pm local time. Team leads may approve full remote for specific roles. Equipment stipend of $1,500 for home office setup.",
        isSignal: false,
      },
      {
        label: "current_architecture",
        content: "Current Architecture:\n- API: Node.js with Express, deployed on Kubernetes\n- Database: PostgreSQL 15 (primary), Redis (cache/sessions)\n- Message Queue: RabbitMQ for async processing\n- Frontend: React 18 with Next.js\n- Mobile: React Native\n- Existing WebSocket infrastructure: None (greenfield)",
        isSignal: true,
      },
      {
        label: "marketing_plan",
        content: "Q1 Marketing Plan:\n- Launch email campaign for new dashboard features\n- Attend 3 industry conferences\n- Partner with 5 tech influencers\n- Increase blog output to 4 posts/month\n- Run A/B test on pricing page",
        isSignal: false,
      },
      {
        label: "old_meeting_notes",
        content: "Engineering All-Hands (2023-06-15):\nTopics discussed: Migrating from Heroku to AWS (complete), Adopting TypeScript (in progress), New CI/CD pipeline with GitHub Actions (complete), Hiring plan for Q3 (5 engineers).",
        isSignal: false,
      },
    ],
  },
  {
    name: "performance_diagnosis",
    task: "Why did the p99 latency spike to 8 seconds on the /search endpoint yesterday?",
    expectedAnswerFragment: "missing index",
    sections: [
      {
        label: "slow_query_log",
        content: "Slow Query Log (2024-01-14):\nQuery: SELECT * FROM products WHERE category_id = $1 AND status = 'active' ORDER BY updated_at DESC LIMIT 20\nDuration: 6.8s (normally 45ms)\nRows scanned: 2,400,000\nNote: Index on (category_id, status, updated_at) was dropped during migration #342. Missing index causes full table scan.",
        isSignal: true,
      },
      {
        label: "apm_dashboard",
        content: "APM Dashboard (2024-01-14):\n/search p99: 8.2s (baseline: 200ms)\n/search p50: 3.1s (baseline: 45ms)\n/search error_rate: 2.3% (timeout errors)\nSpike began at 09:15 UTC, correlates with migration #342 deployment at 09:12 UTC.",
        isSignal: true,
      },
      {
        label: "unrelated_service_logs",
        content: "payment-service logs (2024-01-14):\n09:00 INFO Processing batch of 450 transactions\n09:15 INFO Batch complete, all successful\n09:30 INFO Stripe webhook received for subscription renewal\n10:00 INFO Daily reconciliation started\n10:05 INFO Reconciliation complete, 0 discrepancies",
        isSignal: false,
      },
      {
        label: "team_vacation_calendar",
        content: "Team Availability (January 2024):\n- Alice: OOO Jan 8-12 (vacation)\n- Bob: OOO Jan 15 (doctor appointment)\n- Carol: Available all month\n- Dave: OOO Jan 22-26 (conference)\n- Eve: Half day Jan 10 (school event)",
        isSignal: false,
      },
      {
        label: "migration_342",
        content: "Migration #342 - Clean up deprecated columns:\n- Removed columns: products.legacy_category, products.old_sku\n- Dropped indexes: idx_products_category_status_updated (accidentally included)\n- Added columns: products.variant_group_id\n- Author: Bob (ran on 2024-01-14 at 09:12 UTC)",
        isSignal: true,
      },
      {
        label: "kubernetes_events",
        content: "Kubernetes Events (2024-01-14):\n08:00 Normal Scheduled pod/search-service-7f8d9-abc to node-3\n08:00 Normal Pulled container image 'search-service:2.14.0'\n08:01 Normal Started container search-service\n12:00 Normal Scaled deployment/search-service from 3 to 5 replicas (HPA)\n12:30 Normal Scaled deployment/search-service from 5 to 3 replicas (HPA)",
        isSignal: false,
      },
    ],
  },
];

interface EfficiencyTrialResult {
  readonly scenario: string;
  readonly totalTokens: number;
  readonly signalTokens: number;
  readonly noiseTokens: number;
  readonly effectiveRatio: number;
  readonly answerCorrectFull: boolean;
  readonly answerCorrectSignalOnly: boolean;
}

export interface TokenEfficiencyResult {
  readonly benchmark: "token_efficiency";
  readonly model: string;
  readonly avgEffectiveRatio: number;
  readonly avgAccuracyFullContext: number;
  readonly avgAccuracySignalOnly: number;
  readonly totalWastedTokens: number;
  readonly trials: readonly EfficiencyTrialResult[];
}

export async function runTokenEfficiency(
  client: LLMClient
): Promise<TokenEfficiencyResult> {
  const trials: EfficiencyTrialResult[] = [];

  for (const scenario of SCENARIOS) {
    const fullContext = scenario.sections
      .map((s) => `[${s.label}]\n${s.content}`)
      .join("\n\n");
    const signalContext = scenario.sections
      .filter((s) => s.isSignal)
      .map((s) => `[${s.label}]\n${s.content}`)
      .join("\n\n");

    const totalTokens = countTokens(fullContext);
    const signalTokens = countTokens(signalContext);
    const noiseTokens = totalTokens - signalTokens;

    const systemBase =
      "Answer the question based on the provided context. Be specific and cite evidence from the context.\n\nContext:\n";

    const responseFull = await client.complete(
      [{ role: "user", content: scenario.task }],
      systemBase + fullContext
    );

    const responseSignal = await client.complete(
      [{ role: "user", content: scenario.task }],
      systemBase + signalContext
    );

    trials.push({
      scenario: scenario.name,
      totalTokens,
      signalTokens,
      noiseTokens,
      effectiveRatio:
        totalTokens > 0
          ? Math.round((signalTokens / totalTokens) * 10000) / 10000
          : 0,
      answerCorrectFull: responseFull.content
        .toLowerCase()
        .includes(scenario.expectedAnswerFragment.toLowerCase()),
      answerCorrectSignalOnly: responseSignal.content
        .toLowerCase()
        .includes(scenario.expectedAnswerFragment.toLowerCase()),
    });
  }

  const n = trials.length;

  return {
    benchmark: "token_efficiency",
    model: client.model,
    avgEffectiveRatio:
      n > 0
        ? Math.round(
            (trials.reduce((s, t) => s + t.effectiveRatio, 0) / n) * 10000
          ) / 10000
        : 0,
    avgAccuracyFullContext:
      n > 0
        ? Math.round(
            (trials.filter((t) => t.answerCorrectFull).length / n) * 10000
          ) / 10000
        : 0,
    avgAccuracySignalOnly:
      n > 0
        ? Math.round(
            (trials.filter((t) => t.answerCorrectSignalOnly).length / n) *
              10000
          ) / 10000
        : 0,
    totalWastedTokens: trials.reduce((s, t) => s + t.noiseTokens, 0),
    trials,
  };
}
