# Episodic Memory

> Store and retrieve complete interaction episodes -- full context snapshots of past sessions -- to inform current behavior through semantic recall.

## Problem

Standard chat history gives you a flat list of messages. But when an agent needs to recall *how it solved a similar problem last week*, a message list is insufficient. You need to know: what was the goal? What tools were used? What decisions were made and why? What was the outcome? Without episodic memory, agents re-discover solutions, repeat mistakes, and cannot learn from experience across sessions.

The core challenges:
- Chat history is too granular (individual messages) and too flat (no structure)
- Simple key-value memory stores lose the *relationships* between facts
- Vector search over raw messages returns fragments, not coherent experiences
- Sessions disappear entirely when conversation history is cleared

## Solution

Capture each significant interaction as a structured **episode** -- a self-contained record that includes the goal, context, actions taken, decisions made, and outcome. Index episodes with embeddings for semantic retrieval. When starting a new session, retrieve relevant past episodes and inject them into context as reference material.

An episode is not a transcript. It is a *distilled narrative* of what happened, structured for machine retrieval and human readability.

## How It Works

```
Session in progress
        |
        v
+-------------------+
| Episode Capture   |  <-- End of session or significant milestone
| - Goal            |
| - Context summary |
| - Key decisions   |
| - Tools used      |
| - Outcome         |
| - Tags/metadata   |
+-------------------+
        |
        v
+-------------------+
| Embedding Index   |  <-- Generate embedding from episode summary
| (Vector Store)    |
+-------------------+
        |
        v
    Stored Episode

        ...later...

New session starts
        |
        v
+-------------------+
| Query Formation   |  <-- Current goal/context -> search query
+-------------------+
        |
        v
+-------------------+
| Semantic Search   |  <-- Find similar past episodes
| (Top-K retrieval) |
+-------------------+
        |
        v
+-------------------+
| Context Injection |  <-- Format and inject relevant episodes
+-------------------+
        |
        v
Agent proceeds with historical context
```

## Implementation

### Python

```python
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Episode:
    """A complete record of a significant interaction or session."""
    episode_id: str
    timestamp: str
    goal: str
    context_summary: str
    decisions: tuple[str, ...]
    tools_used: tuple[str, ...]
    outcome: str
    outcome_success: bool
    tags: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)

    def to_search_text(self) -> str:
        """Combine fields into a single string for embedding."""
        return f"{self.goal}\n{self.context_summary}\n{self.outcome}"

    def to_context_block(self) -> str:
        """Format episode for injection into LLM context."""
        decisions_str = "\n".join(f"  - {d}" for d in self.decisions)
        return (
            f"## Past Episode: {self.goal}\n"
            f"**When**: {self.timestamp}\n"
            f"**Outcome**: {'Success' if self.outcome_success else 'Failure'} -- {self.outcome}\n"
            f"**Key Decisions**:\n{decisions_str}\n"
            f"**Tools Used**: {', '.join(self.tools_used)}\n"
        )


class EpisodicMemoryStore:
    """
    Stores and retrieves interaction episodes using semantic search.

    Requires an embedding function and a vector store. This implementation
    uses simple interfaces that can be backed by OpenAI embeddings + Chroma,
    or any other embedding/vector store combination.
    """

    def __init__(self, embed_fn, vector_store):
        """
        Args:
            embed_fn: Callable that takes a string and returns a list of floats.
            vector_store: Object with add(id, embedding, metadata) and
                          query(embedding, top_k) -> list[dict] methods.
        """
        self._embed_fn = embed_fn
        self._vector_store = vector_store
        self._episodes: dict[str, Episode] = {}

    def capture_episode(
        self,
        goal: str,
        context_summary: str,
        decisions: list[str],
        tools_used: list[str],
        outcome: str,
        outcome_success: bool,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Episode:
        """Capture a new episode and index it for retrieval."""
        timestamp = datetime.utcnow().isoformat()
        episode_id = hashlib.sha256(
            f"{goal}:{timestamp}".encode()
        ).hexdigest()[:16]

        episode = Episode(
            episode_id=episode_id,
            timestamp=timestamp,
            goal=goal,
            context_summary=context_summary,
            decisions=tuple(decisions),
            tools_used=tuple(tools_used),
            outcome=outcome,
            outcome_success=outcome_success,
            tags=tuple(tags or []),
            metadata=metadata or {},
        )

        # Generate embedding from the episode's searchable text
        embedding = self._embed_fn(episode.to_search_text())

        # Store in vector store with metadata for filtering
        self._vector_store.add(
            id=episode_id,
            embedding=embedding,
            metadata={
                "episode_id": episode_id,
                "timestamp": timestamp,
                "success": outcome_success,
                "tags": json.dumps(tags or []),
            },
        )

        self._episodes[episode_id] = episode
        return episode

    def recall(
        self,
        query: str,
        top_k: int = 3,
        success_only: bool = False,
    ) -> list[Episode]:
        """Retrieve the most relevant past episodes for a query."""
        query_embedding = self._embed_fn(query)
        results = self._vector_store.query(
            embedding=query_embedding, top_k=top_k * 2
        )

        episodes = []
        for result in results:
            episode_id = result["metadata"]["episode_id"]
            episode = self._episodes.get(episode_id)
            if episode is None:
                continue
            if success_only and not episode.outcome_success:
                continue
            episodes.append(episode)
            if len(episodes) >= top_k:
                break

        return episodes

    def build_context_block(
        self,
        query: str,
        top_k: int = 3,
        max_tokens: int = 2000,
    ) -> str:
        """
        Build a formatted context block of relevant past episodes,
        ready for injection into an LLM prompt.
        """
        episodes = self.recall(query, top_k=top_k)
        if not episodes:
            return ""

        header = "# Relevant Past Episodes\n\n"
        blocks = []
        total_chars = len(header)

        for episode in episodes:
            block = episode.to_context_block()
            # Rough token estimate: 1 token ~= 4 chars
            if total_chars + len(block) > max_tokens * 4:
                break
            blocks.append(block)
            total_chars += len(block)

        if not blocks:
            return ""

        return header + "\n---\n\n".join(blocks)
```

### TypeScript

```typescript
import { createHash } from "crypto";

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

interface VectorStore {
  add(id: string, embedding: number[], metadata: Record<string, unknown>): void;
  query(
    embedding: number[],
    topK: number
  ): Array<{ metadata: Record<string, unknown> }>;
}

type EmbedFn = (text: string) => Promise<number[]>;

function episodeToSearchText(episode: Episode): string {
  return `${episode.goal}\n${episode.contextSummary}\n${episode.outcome}`;
}

function episodeToContextBlock(episode: Episode): string {
  const decisions = episode.decisions.map((d) => `  - ${d}`).join("\n");
  return [
    `## Past Episode: ${episode.goal}`,
    `**When**: ${episode.timestamp}`,
    `**Outcome**: ${episode.outcomeSuccess ? "Success" : "Failure"} -- ${episode.outcome}`,
    `**Key Decisions**:\n${decisions}`,
    `**Tools Used**: ${episode.toolsUsed.join(", ")}`,
  ].join("\n");
}

class EpisodicMemoryStore {
  private readonly embedFn: EmbedFn;
  private readonly vectorStore: VectorStore;
  private readonly episodes: Map<string, Episode> = new Map();

  constructor(embedFn: EmbedFn, vectorStore: VectorStore) {
    this.embedFn = embedFn;
    this.vectorStore = vectorStore;
  }

  async captureEpisode(params: {
    goal: string;
    contextSummary: string;
    decisions: string[];
    toolsUsed: string[];
    outcome: string;
    outcomeSuccess: boolean;
    tags?: string[];
    metadata?: Record<string, unknown>;
  }): Promise<Episode> {
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

    const embedding = await this.embedFn(episodeToSearchText(episode));

    this.vectorStore.add(episodeId, embedding, {
      episodeId,
      timestamp,
      success: params.outcomeSuccess,
      tags: JSON.stringify(params.tags ?? []),
    });

    this.episodes.set(episodeId, episode);
    return episode;
  }

  async recall(
    query: string,
    topK: number = 3,
    successOnly: boolean = false
  ): Promise<Episode[]> {
    const queryEmbedding = await this.embedFn(query);
    const results = this.vectorStore.query(queryEmbedding, topK * 2);

    const episodes: Episode[] = [];
    for (const result of results) {
      const episodeId = result.metadata.episodeId as string;
      const episode = this.episodes.get(episodeId);
      if (!episode) continue;
      if (successOnly && !episode.outcomeSuccess) continue;
      episodes.push(episode);
      if (episodes.length >= topK) break;
    }

    return episodes;
  }

  async buildContextBlock(
    query: string,
    topK: number = 3,
    maxTokens: number = 2000
  ): Promise<string> {
    const episodes = await this.recall(query, topK);
    if (episodes.length === 0) return "";

    const header = "# Relevant Past Episodes\n\n";
    const blocks: string[] = [];
    let totalChars = header.length;

    for (const episode of episodes) {
      const block = episodeToContextBlock(episode);
      if (totalChars + block.length > maxTokens * 4) break;
      blocks.push(block);
      totalChars += block.length;
    }

    if (blocks.length === 0) return "";
    return header + blocks.join("\n---\n\n");
  }
}

export { EpisodicMemoryStore, Episode, EmbedFn, VectorStore };
```

## Trade-offs

| Pros | Cons |
|------|------|
| Rich contextual recall beyond simple key-value memory | Requires embedding infrastructure (API or local model) |
| Semantic search finds relevant episodes even with different wording | Episode capture logic must be tuned -- too aggressive creates noise, too conservative misses important context |
| Episodes are self-contained and debuggable | Storage grows over time; needs retention/archival policy |
| Can distinguish successful from failed approaches | Embedding quality directly impacts retrieval quality |
| Human-readable episode format aids debugging | Additional latency from embedding generation and vector search |

## When to Use

- Agents that run across multiple sessions and should learn from past interactions
- Customer support systems that need to recall prior conversations with the same user
- Development tools that should remember how a codebase was previously modified
- Any system where "have I seen something like this before?" would improve behavior

## When NOT to Use

- Single-session, stateless interactions (chatbots with no continuity)
- When simple key-value memory (user preferences, settings) is sufficient
- Low-volume systems where a human can manually curate relevant context
- When you cannot afford the latency of embedding + vector search on each request

## Related Patterns

- **Filesystem-as-Memory**: Simpler persistence that does not require vector infrastructure. Consider starting here and adding episodic memory when file browsing becomes impractical.
- **RAG Context Assembly** (Retrieval): Episodes are a specialized form of retrieved context. RAG assembly patterns apply when formatting episodes for injection.
- **Progressive Disclosure** (Construction): Retrieved episodes should be injected selectively, not dumped wholesale. Use progressive disclosure to control when episodes appear.

## Real-World Examples

- **ChatGPT Memory**: Captures user preferences and facts across sessions, resurfaces them when relevant. Simplified episodic memory with entity extraction.
- **Claude Projects Knowledge**: Project-level persistent context that informs all conversations within a project. A filesystem-flavored approach to episodic knowledge.
- **Customer Support Platforms (Intercom, Zendesk)**: Store past ticket histories and surface relevant past interactions when a user opens a new ticket.
- **Voyager (Minecraft Agent)**: Stores successful action sequences as "skills" that can be retrieved and reused in similar situations -- episodic memory applied to embodied agents.
