"""
Semantic Tool Selection -- LangChain Integration

Implements a dynamic tool filter that selects relevant tools by embedding
similarity before passing them to a LangChain agent. Instead of binding all
tools to the agent, this filters to the top-k most relevant tools per turn.

Pattern: https://github.com/context-engineering-handbook/patterns/retrieval/semantic-tool-selection.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from langchain_core.tools import BaseTool, StructuredTool, tool


# ---------------------------------------------------------------------------
# Embedding interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredTool:
    """A tool with its relevance score for the current query."""

    tool: BaseTool
    score: float


class ToolEmbedder:
    """Generates and caches embeddings for tool descriptions.

    In production, replace _embed with a real embedding model:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector = embeddings.embed_query(text)
    """

    def __init__(self, embedding_dim: int = 384) -> None:
        self._embedding_dim = embedding_dim
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        """Embed text into a vector. Cached for repeated calls."""
        if text in self._cache:
            return self._cache[text]

        # Deterministic pseudo-embedding for demonstration
        vec = self._pseudo_embed(text)
        self._cache[text] = vec
        return vec

    def _pseudo_embed(self, text: str) -> list[float]:
        """Hash-based pseudo-embedding (replace with real model in production)."""
        seed = 0
        for ch in text:
            seed = ((seed << 5) - seed + ord(ch)) & 0x7FFFFFFF

        vec: list[float] = []
        for _ in range(self._embedding_dim):
            seed = (seed * 1664525 + 1013904223) & 0x7FFFFFFF
            vec.append((seed / 0x7FFFFFFF) * 2 - 1)

        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# SemanticToolSelector
# ---------------------------------------------------------------------------


class SemanticToolSelector:
    """Selects relevant LangChain tools for each turn using embedding similarity.

    Usage with a LangChain agent:

        from langchain_openai import ChatOpenAI
        from langchain.agents import create_tool_calling_agent, AgentExecutor

        selector = SemanticToolSelector(top_k=5, min_score=0.3)
        selector.register_tools(all_tools)

        # Per-turn: select relevant tools
        selected = selector.select("I need a refund for order #12345")
        tools = [s.tool for s in selected]

        # Create agent with only the relevant tools
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
    """

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.3,
        embedder: ToolEmbedder | None = None,
        pinned_tools: list[str] | None = None,
    ) -> None:
        self._top_k = top_k
        self._min_score = min_score
        self._embedder = embedder or ToolEmbedder()
        self._pinned_names: set[str] = set(pinned_tools or [])
        self._tools: list[BaseTool] = []
        self._tool_embeddings: list[list[float]] = []

    def register_tools(self, tools: list[BaseTool]) -> None:
        """Register tools and pre-compute their description embeddings."""
        self._tools = list(tools)
        self._tool_embeddings = [
            self._embedder.embed(f"{t.name}: {t.description}")
            for t in tools
        ]

    def select(self, query: str) -> list[ScoredTool]:
        """Select the most relevant tools for the given query."""
        if not self._tools:
            return []

        query_embedding = self._embedder.embed(query)

        pinned: list[ScoredTool] = []
        scored: list[ScoredTool] = []

        for i, t in enumerate(self._tools):
            similarity = _cosine_similarity(
                query_embedding, self._tool_embeddings[i]
            )
            st = ScoredTool(tool=t, score=similarity)

            if t.name in self._pinned_names:
                pinned.append(st)
            else:
                scored.append(st)

        # Sort by score, filter, and take top_k
        selected = sorted(scored, key=lambda s: s.score, reverse=True)
        selected = [s for s in selected if s.score >= self._min_score]
        selected = selected[: self._top_k]

        return pinned + selected

    def select_tools(self, query: str) -> list[BaseTool]:
        """Convenience method: returns just the BaseTool objects."""
        return [s.tool for s in self.select(query)]

    def format_selection_report(self, scored_tools: list[ScoredTool]) -> str:
        """Format the selection for logging/debugging."""
        lines = [
            f"Selected {len(scored_tools)} tools "
            f"(from {len(self._tools)} registered):"
        ]
        for st in scored_tools:
            pin_label = " [PINNED]" if st.tool.name in self._pinned_names else ""
            lines.append(
                f"  {st.tool.name} (score: {st.score:.3f}){pin_label}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo tools
# ---------------------------------------------------------------------------


@tool
def search_knowledge_base(query: str, category: str = "") -> str:
    """Search the company knowledge base for articles, FAQs, and documentation."""
    return f"Results for '{query}' in category '{category}'"


@tool
def create_ticket(title: str, description: str, priority: str = "medium") -> str:
    """Create a support ticket for issues that need human agent follow-up."""
    return f"Created ticket: {title}"


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer order by order ID."""
    return f"Order {order_id}: shipped"


@tool
def process_refund(order_id: str, reason: str, amount: float = 0.0) -> str:
    """Initiate a refund for a specific order or transaction."""
    return f"Refund initiated for {order_id}"


@tool
def update_account_email(customer_id: str, new_email: str) -> str:
    """Update the email address on a customer account."""
    return f"Email updated for {customer_id}"


@tool
def schedule_callback(phone: str, preferred_time: str) -> str:
    """Schedule a phone callback from a human agent at a specific time."""
    return f"Callback scheduled for {phone} at {preferred_time}"


@tool
def send_invoice(invoice_id: str) -> str:
    """Send or resend an invoice to the customer's billing email."""
    return f"Invoice {invoice_id} sent"


@tool
def final_answer(answer: str) -> str:
    """Provide the final response to the user's question."""
    return answer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate Semantic Tool Selection with LangChain tools."""
    all_tools: list[BaseTool] = [
        search_knowledge_base,
        create_ticket,
        get_order_status,
        process_refund,
        update_account_email,
        schedule_callback,
        send_invoice,
        final_answer,
    ]

    selector = SemanticToolSelector(
        top_k=3,
        min_score=0.1,
        pinned_tools=["final_answer"],
    )
    selector.register_tools(all_tools)

    # Test with different queries
    queries = [
        "I need a refund for order #12345",
        "What's the status of my order?",
        "Can you update my email address?",
        "I need to speak with a human agent",
    ]

    for query in queries:
        scored = selector.select(query)
        print(f"Query: {query}")
        print(selector.format_selection_report(scored))
        print()


if __name__ == "__main__":
    main()
