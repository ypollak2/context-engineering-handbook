"""
Episodic Memory -- LlamaIndex Integration

Implements structured episode capture and semantic retrieval using LlamaIndex's
storage and embedding infrastructure. Episodes are stored as nodes in a
VectorStoreIndex for semantic recall across sessions.

Pattern: https://github.com/context-engineering-handbook/patterns/persistence/episodic-memory.md
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from llama_index.core.schema import TextNode
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.storage.chat_store import BaseChatStore
from llama_index.core.llms import ChatMessage, MessageRole


# ---------------------------------------------------------------------------
# Episode data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Episode:
    """A structured record of a significant interaction or session."""

    episode_id: str
    timestamp: str
    goal: str
    context_summary: str
    decisions: tuple[str, ...]
    tools_used: tuple[str, ...]
    outcome: str
    outcome_success: bool
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

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
            f"**Tools Used**: {', '.join(self.tools_used)}"
        )

    def to_node(self) -> TextNode:
        """Convert to a LlamaIndex TextNode for indexing."""
        return TextNode(
            text=self.to_search_text(),
            metadata={
                "episode_id": self.episode_id,
                "timestamp": self.timestamp,
                "goal": self.goal,
                "outcome_success": self.outcome_success,
                "tags": json.dumps(list(self.tags)),
                "decisions": json.dumps(list(self.decisions)),
                "tools_used": json.dumps(list(self.tools_used)),
                "outcome": self.outcome,
                "context_summary": self.context_summary,
            },
            id_=self.episode_id,
        )

    @classmethod
    def from_node(cls, node: TextNode) -> Episode:
        """Reconstruct an Episode from a LlamaIndex TextNode."""
        meta = node.metadata
        return cls(
            episode_id=meta["episode_id"],
            timestamp=meta["timestamp"],
            goal=meta["goal"],
            context_summary=meta.get("context_summary", ""),
            decisions=tuple(json.loads(meta.get("decisions", "[]"))),
            tools_used=tuple(json.loads(meta.get("tools_used", "[]"))),
            outcome=meta["outcome"],
            outcome_success=meta["outcome_success"],
            tags=tuple(json.loads(meta.get("tags", "[]"))),
        )


# ---------------------------------------------------------------------------
# EpisodicMemoryStore -- LlamaIndex-backed episodic memory
# ---------------------------------------------------------------------------


class EpisodicMemoryStore:
    """Stores and retrieves interaction episodes using LlamaIndex.

    Uses a VectorStoreIndex for semantic episode retrieval. Episodes are
    stored as TextNodes, enabling LlamaIndex's full retrieval pipeline
    (embedding, indexing, postprocessing) to work on episodic data.

    Usage:
        from llama_index.embeddings.openai import OpenAIEmbedding

        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        store = EpisodicMemoryStore(embed_model=embed_model)

        # Capture an episode after a session
        episode = store.capture(
            goal="Fix authentication bug",
            context_summary="JWT token validation was missing in login.py",
            decisions=["Use PyJWT library", "Add token expiry check"],
            tools_used=["read_file", "edit_file", "run_tests"],
            outcome="Bug fixed, all tests passing",
            outcome_success=True,
            tags=["auth", "bug-fix"],
        )

        # Recall relevant episodes in a new session
        episodes = store.recall("authentication issue with tokens")
    """

    def __init__(self, embed_model: Any = None) -> None:
        self._episodes: dict[str, Episode] = {}
        self._nodes: list[TextNode] = []
        self._embed_model = embed_model
        self._index: VectorStoreIndex | None = None

    def capture(
        self,
        goal: str,
        context_summary: str,
        decisions: list[str],
        tools_used: list[str],
        outcome: str,
        outcome_success: bool,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Capture a new episode and index it for retrieval."""
        timestamp = datetime.now(timezone.utc).isoformat()
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

        # Store and index
        self._episodes[episode_id] = episode
        node = episode.to_node()
        self._nodes.append(node)

        # Rebuild the index with the new node
        self._rebuild_index()

        return episode

    def recall(
        self,
        query: str,
        top_k: int = 3,
        success_only: bool = False,
    ) -> list[Episode]:
        """Retrieve the most relevant past episodes for a query."""
        if self._index is None or not self._nodes:
            return []

        retriever = self._index.as_retriever(similarity_top_k=top_k * 2)
        results = retriever.retrieve(query)

        episodes: list[Episode] = []
        for node_with_score in results:
            episode_id = node_with_score.node.metadata.get("episode_id", "")
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
        """Build a formatted context block of relevant past episodes."""
        episodes = self.recall(query, top_k=top_k)
        if not episodes:
            return ""

        header = "# Relevant Past Episodes\n\n"
        blocks: list[str] = []
        total_chars = len(header)

        for episode in episodes:
            block = episode.to_context_block()
            if total_chars + len(block) > max_tokens * 4:
                break
            blocks.append(block)
            total_chars += len(block)

        if not blocks:
            return ""
        return header + "\n---\n\n".join(blocks)

    def _rebuild_index(self) -> None:
        """Rebuild the vector index from all stored nodes."""
        if self._embed_model is not None:
            self._index = VectorStoreIndex(
                nodes=self._nodes,
                embed_model=self._embed_model,
            )
        # If no embed model, index remains None (demo mode)

    @property
    def episode_count(self) -> int:
        return len(self._episodes)


# ---------------------------------------------------------------------------
# Chat history to episode converter
# ---------------------------------------------------------------------------


def extract_episode_from_chat(
    messages: list[ChatMessage],
    goal: str = "",
) -> dict[str, Any]:
    """Extract episode components from a LlamaIndex chat history.

    Analyzes the conversation to identify decisions, tools used, and outcome.
    In production, use an LLM to extract these more accurately.
    """
    decisions: list[str] = []
    tools: set[str] = []
    last_assistant_msg = ""

    for msg in messages:
        content = msg.content or ""
        lower = content.lower()

        if msg.role == MessageRole.ASSISTANT:
            last_assistant_msg = content
            # Heuristic: lines with "decision" or "decided"
            for line in content.split("\n"):
                if "decided" in line.lower() or "decision:" in line.lower():
                    decisions.append(line.strip())

        # Detect tool usage from message metadata or content
        if msg.role == MessageRole.TOOL:
            tool_name = msg.additional_kwargs.get("tool_name", "unknown_tool")
            tools.add(tool_name)

    return {
        "goal": goal or "Extracted from chat session",
        "context_summary": last_assistant_msg[:200] if last_assistant_msg else "",
        "decisions": decisions,
        "tools_used": list(tools),
        "outcome": last_assistant_msg[:150] if last_assistant_msg else "No outcome recorded",
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate Episodic Memory without requiring API keys."""

    # Create store without embedding model (demo mode)
    store = EpisodicMemoryStore(embed_model=None)

    # Capture some episodes
    ep1 = store.capture(
        goal="Fix authentication bug in login.py",
        context_summary="JWT token validation was missing. Added PyJWT-based validation.",
        decisions=["Use PyJWT library", "Add 24h token expiry", "Store refresh tokens in Redis"],
        tools_used=["read_file", "edit_file", "run_tests"],
        outcome="Bug fixed. All 42 auth tests passing. Deployed to staging.",
        outcome_success=True,
        tags=["auth", "bug-fix", "jwt"],
    )

    ep2 = store.capture(
        goal="Add rate limiting to API endpoints",
        context_summary="Implemented sliding window rate limiting using Redis.",
        decisions=["Use sliding window algorithm", "5 req/min for auth, 100 req/min for API"],
        tools_used=["read_file", "edit_file", "run_tests", "run_command"],
        outcome="Rate limiting deployed. Load test confirms correct behavior.",
        outcome_success=True,
        tags=["api", "rate-limiting", "redis"],
    )

    ep3 = store.capture(
        goal="Migrate database to PostgreSQL 16",
        context_summary="Attempted migration but hit incompatibility with pg_trgm extension.",
        decisions=["Use pg_dump for migration", "Keep pg_trgm extension"],
        tools_used=["run_command", "read_file"],
        outcome="Migration failed. pg_trgm extension not available in target version.",
        outcome_success=False,
        tags=["database", "migration", "postgresql"],
    )

    print(f"Captured {store.episode_count} episodes")
    print()

    # Display episodes (in production, these would be recalled via semantic search)
    print("=== Captured Episodes ===")
    for ep in [ep1, ep2, ep3]:
        print(ep.to_context_block())
        print("---")
        print()

    # Show what a context block would look like
    print("=== Example Context Block (for injection into prompt) ===")
    header = "# Relevant Past Episodes\n\n"
    blocks = [ep1.to_context_block(), ep2.to_context_block()]
    print(header + "\n---\n\n".join(blocks))


if __name__ == "__main__":
    main()
