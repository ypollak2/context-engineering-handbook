/**
 * Episodic Memory -- LlamaIndex TypeScript Integration
 *
 * Structured episode capture and retrieval using LlamaIndex-compatible
 * node storage. Episodes are stored as text nodes with rich metadata
 * for semantic recall across sessions.
 *
 * Pattern: patterns/persistence/episodic-memory.md
 */

import { createHash } from "crypto";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Episode {
  readonly episodeId: string;
  readonly timestamp: string;
  readonly goal: string;
  readonly contextSummary: string;
  readonly decisions: readonly string[];
  readonly toolsUsed: readonly string[];
  readonly outcome: string;
  readonly outcomeSuccess: boolean;
  readonly tags: readonly string[];
  readonly metadata: Record<string, unknown>;
}

interface EpisodeNode {
  readonly id: string;
  readonly text: string;
  readonly metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Episode helpers
// ---------------------------------------------------------------------------

function episodeToSearchText(ep: Episode): string {
  return `${ep.goal}\n${ep.contextSummary}\n${ep.outcome}`;
}

function episodeToContextBlock(ep: Episode): string {
  const decisions = ep.decisions.map((d) => `  - ${d}`).join("\n");
  return [
    `## Past Episode: ${ep.goal}`,
    `**When**: ${ep.timestamp}`,
    `**Outcome**: ${ep.outcomeSuccess ? "Success" : "Failure"} -- ${ep.outcome}`,
    `**Key Decisions**:\n${decisions}`,
    `**Tools Used**: ${ep.toolsUsed.join(", ")}`,
  ].join("\n");
}

function episodeToNode(ep: Episode): EpisodeNode {
  return {
    id: ep.episodeId,
    text: episodeToSearchText(ep),
    metadata: {
      episodeId: ep.episodeId,
      timestamp: ep.timestamp,
      goal: ep.goal,
      outcomeSuccess: ep.outcomeSuccess,
      tags: JSON.stringify([...ep.tags]),
      decisions: JSON.stringify([...ep.decisions]),
      toolsUsed: JSON.stringify([...ep.toolsUsed]),
      outcome: ep.outcome,
      contextSummary: ep.contextSummary,
    },
  };
}

function nodeToEpisode(node: EpisodeNode): Episode {
  const m = node.metadata;
  return {
    episodeId: m.episodeId as string,
    timestamp: m.timestamp as string,
    goal: m.goal as string,
    contextSummary: (m.contextSummary as string) ?? "",
    decisions: Object.freeze(JSON.parse((m.decisions as string) ?? "[]")),
    toolsUsed: Object.freeze(JSON.parse((m.toolsUsed as string) ?? "[]")),
    outcome: m.outcome as string,
    outcomeSuccess: m.outcomeSuccess as boolean,
    tags: Object.freeze(JSON.parse((m.tags as string) ?? "[]")),
    metadata: {},
  };
}

// ---------------------------------------------------------------------------
// EpisodicMemoryStore
// ---------------------------------------------------------------------------

class EpisodicMemoryStore {
  private readonly episodes: Map<string, Episode> = new Map();
  private readonly nodes: EpisodeNode[] = [];

  capture(params: {
    goal: string;
    contextSummary: string;
    decisions: string[];
    toolsUsed: string[];
    outcome: string;
    outcomeSuccess: boolean;
    tags?: string[];
    metadata?: Record<string, unknown>;
  }): Episode {
    const timestamp = new Date().toISOString();
    const episodeId = createHash("sha256")
      .update(`${params.goal}:${timestamp}`)
      .digest("hex")
      .slice(0, 16);

    const episode: Episode = {
      episodeId,
      timestamp,
      goal: params.goal,
      contextSummary: params.contextSummary,
      decisions: Object.freeze([...params.decisions]),
      toolsUsed: Object.freeze([...params.toolsUsed]),
      outcome: params.outcome,
      outcomeSuccess: params.outcomeSuccess,
      tags: Object.freeze([...(params.tags ?? [])]),
      metadata: { ...(params.metadata ?? {}) },
    };

    this.episodes.set(episodeId, episode);
    this.nodes.push(episodeToNode(episode));
    return episode;
  }

  /**
   * Recall relevant episodes. In production, use LlamaIndex's
   * VectorStoreIndex.asRetriever() for semantic search. This demo
   * uses simple keyword matching.
   */
  recall(
    query: string,
    topK: number = 3,
    successOnly: boolean = false
  ): Episode[] {
    const queryTerms = new Set(query.toLowerCase().split(/\s+/));
    const scored: Array<{ episode: Episode; score: number }> = [];

    for (const episode of this.episodes.values()) {
      if (successOnly && !episode.outcomeSuccess) continue;

      const searchText = episodeToSearchText(episode).toLowerCase();
      const searchTerms = new Set(searchText.split(/\s+/));
      const overlap = [...queryTerms].filter((t) => searchTerms.has(t)).length;
      const score = overlap / Math.max(queryTerms.size, 1);

      scored.push({ episode, score });
    }

    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((s) => s.episode);
  }

  buildContextBlock(
    query: string,
    topK: number = 3,
    maxTokens: number = 2000
  ): string {
    const episodes = this.recall(query, topK);
    if (episodes.length === 0) return "";

    const header = "# Relevant Past Episodes\n\n";
    const blocks: string[] = [];
    let totalChars = header.length;

    for (const ep of episodes) {
      const block = episodeToContextBlock(ep);
      if (totalChars + block.length > maxTokens * 4) break;
      blocks.push(block);
      totalChars += block.length;
    }

    if (blocks.length === 0) return "";
    return header + blocks.join("\n---\n\n");
  }

  get episodeCount(): number {
    return this.episodes.size;
  }
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

function main(): void {
  const store = new EpisodicMemoryStore();

  const ep1 = store.capture({
    goal: "Fix authentication bug in login.py",
    contextSummary:
      "JWT token validation was missing. Added PyJWT-based validation.",
    decisions: [
      "Use PyJWT library",
      "Add 24h token expiry",
      "Store refresh tokens in Redis",
    ],
    toolsUsed: ["read_file", "edit_file", "run_tests"],
    outcome: "Bug fixed. All 42 auth tests passing. Deployed to staging.",
    outcomeSuccess: true,
    tags: ["auth", "bug-fix", "jwt"],
  });

  const ep2 = store.capture({
    goal: "Add rate limiting to API endpoints",
    contextSummary: "Implemented sliding window rate limiting using Redis.",
    decisions: [
      "Use sliding window algorithm",
      "5 req/min for auth, 100 req/min for API",
    ],
    toolsUsed: ["read_file", "edit_file", "run_tests", "run_command"],
    outcome: "Rate limiting deployed. Load test confirms correct behavior.",
    outcomeSuccess: true,
    tags: ["api", "rate-limiting", "redis"],
  });

  const ep3 = store.capture({
    goal: "Migrate database to PostgreSQL 16",
    contextSummary:
      "Attempted migration but hit incompatibility with pg_trgm extension.",
    decisions: ["Use pg_dump for migration", "Keep pg_trgm extension"],
    toolsUsed: ["run_command", "read_file"],
    outcome:
      "Migration failed. pg_trgm extension not available in target version.",
    outcomeSuccess: false,
    tags: ["database", "migration", "postgresql"],
  });

  console.log(`Captured ${store.episodeCount} episodes`);
  console.log();

  // Recall episodes by query
  const recalled = store.recall("authentication token issue", 2, true);
  console.log(
    `Recalled ${recalled.length} episodes for "authentication token issue":`
  );
  for (const ep of recalled) {
    console.log(`  - ${ep.goal} (${ep.outcomeSuccess ? "success" : "failed"})`);
  }
  console.log();

  // Build context block
  const contextBlock = store.buildContextBlock("auth token validation bug");
  console.log("=== Context Block ===");
  console.log(contextBlock);
}

main();
