"""
Semantic Tool Selection -- Semantic Kernel Integration

Implements a dynamic plugin filter that selects relevant Semantic Kernel
functions by embedding similarity before passing them to the planner.
Instead of exposing all plugin functions to the model, this pre-filters
to the most relevant subset per turn.

Pattern: https://github.com/context-engineering-handbook/patterns/retrieval/semantic-tool-selection.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Types compatible with Semantic Kernel's function model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PluginFunction:
    """Represents a Semantic Kernel plugin function.

    In production, use actual KernelFunction objects:
        from semantic_kernel.functions import KernelFunction

    This dataclass provides the same interface for demonstration.
    """

    plugin_name: str
    function_name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    pinned: bool = False

    @property
    def fully_qualified_name(self) -> str:
        return f"{self.plugin_name}.{self.function_name}"


@dataclass(frozen=True)
class ScoredFunction:
    """A plugin function with its relevance score for the current query."""

    function: PluginFunction
    score: float


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


class FunctionEmbedder:
    """Generates embeddings for plugin function descriptions.

    In production, use Semantic Kernel's embedding service:
        from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

        embedding_service = AzureTextEmbedding(
            deployment_name="text-embedding-3-small",
            endpoint=endpoint,
            api_key=api_key,
        )
        vectors = await embedding_service.generate_embeddings([text])
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        vec = self._pseudo_embed(text)
        self._cache[text] = vec
        return vec

    def _pseudo_embed(self, text: str) -> list[float]:
        """Hash-based pseudo-embedding (replace with real model in production)."""
        seed = 0
        for ch in text:
            seed = ((seed << 5) - seed + ord(ch)) & 0x7FFFFFFF
        vec: list[float] = []
        for _ in range(self._dim):
            seed = (seed * 1664525 + 1013904223) & 0x7FFFFFFF
            vec.append((seed / 0x7FFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_val = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_val / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# SemanticPluginSelector
# ---------------------------------------------------------------------------


class SemanticPluginSelector:
    """Selects relevant SK plugin functions using embedding similarity.

    Designed to sit between the user query and SK's planner/agent. It
    filters the available functions so the planner only sees the most
    relevant subset, reducing token cost and hallucinated function calls.

    Usage with Semantic Kernel:
        from semantic_kernel import Kernel

        kernel = Kernel()
        kernel.add_plugin(billing_plugin, "billing")
        kernel.add_plugin(support_plugin, "support")

        # Extract all functions from registered plugins
        all_functions = extract_plugin_functions(kernel)

        selector = SemanticPluginSelector(top_k=5)
        selector.register_functions(all_functions)

        # Per-turn: select relevant functions
        selected = selector.select("I need a refund")
        # Create a filtered kernel or pass selected functions to the planner
    """

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.3,
        embedder: FunctionEmbedder | None = None,
    ) -> None:
        self._top_k = top_k
        self._min_score = min_score
        self._embedder = embedder or FunctionEmbedder()
        self._functions: list[PluginFunction] = []
        self._embeddings: list[list[float]] = []

    def register_functions(self, functions: list[PluginFunction]) -> None:
        """Register plugin functions and pre-compute their embeddings."""
        self._functions = list(functions)
        self._embeddings = [
            self._embedder.embed(
                f"{f.fully_qualified_name}: {f.description}"
            )
            for f in functions
        ]

    def select(self, query: str) -> list[ScoredFunction]:
        """Select the most relevant functions for the given query."""
        if not self._functions:
            return []

        query_embedding = self._embedder.embed(query)

        pinned: list[ScoredFunction] = []
        scored: list[ScoredFunction] = []

        for i, func in enumerate(self._functions):
            similarity = _cosine_similarity(
                query_embedding, self._embeddings[i]
            )
            sf = ScoredFunction(function=func, score=similarity)

            if func.pinned:
                pinned.append(sf)
            else:
                scored.append(sf)

        selected = sorted(scored, key=lambda s: s.score, reverse=True)
        selected = [s for s in selected if s.score >= self._min_score]
        selected = selected[: self._top_k]

        return pinned + selected

    def select_functions(self, query: str) -> list[PluginFunction]:
        """Convenience: returns just the PluginFunction objects."""
        return [s.function for s in self.select(query)]

    def format_for_planner(self, scored: list[ScoredFunction]) -> str:
        """Format selected functions for planner/agent prompt injection.

        Produces the function description block that would be included
        in the system prompt for the SK planner.
        """
        lines: list[str] = []
        for sf in scored:
            f = sf.function
            params = ", ".join(
                f"{name}: {info.get('type', 'any')}"
                + (" (required)" if info.get("required") else "")
                for name, info in f.parameters.items()
            )
            pin = " [ALWAYS AVAILABLE]" if f.pinned else ""
            lines.append(
                f"- {f.fully_qualified_name}: {f.description}{pin}\n"
                f"  Parameters: {params or 'none'}"
            )
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Demo plugin functions
# ---------------------------------------------------------------------------


DEMO_FUNCTIONS = [
    PluginFunction(
        plugin_name="billing",
        function_name="process_refund",
        description="Initiate a refund for a specific order or transaction",
        parameters={
            "order_id": {"type": "string", "required": True},
            "reason": {"type": "string", "required": True},
            "amount": {"type": "number"},
        },
    ),
    PluginFunction(
        plugin_name="billing",
        function_name="send_invoice",
        description="Send or resend an invoice to the customer's billing email",
        parameters={"invoice_id": {"type": "string", "required": True}},
    ),
    PluginFunction(
        plugin_name="orders",
        function_name="get_status",
        description="Look up the current status of a customer order by order ID",
        parameters={"order_id": {"type": "string", "required": True}},
    ),
    PluginFunction(
        plugin_name="support",
        function_name="create_ticket",
        description="Create a support ticket for issues that need human follow-up",
        parameters={
            "title": {"type": "string", "required": True},
            "description": {"type": "string", "required": True},
            "priority": {"type": "string"},
        },
    ),
    PluginFunction(
        plugin_name="support",
        function_name="schedule_callback",
        description="Schedule a phone callback from a human agent",
        parameters={
            "phone": {"type": "string", "required": True},
            "preferred_time": {"type": "string", "required": True},
        },
    ),
    PluginFunction(
        plugin_name="account",
        function_name="update_email",
        description="Update the email address on a customer account",
        parameters={
            "customer_id": {"type": "string", "required": True},
            "new_email": {"type": "string", "required": True},
        },
    ),
    PluginFunction(
        plugin_name="knowledge",
        function_name="search",
        description="Search the company knowledge base for articles and documentation",
        parameters={
            "query": {"type": "string", "required": True},
            "category": {"type": "string"},
        },
    ),
    PluginFunction(
        plugin_name="core",
        function_name="final_response",
        description="Provide the final response to the user's question",
        parameters={"answer": {"type": "string", "required": True}},
        pinned=True,
    ),
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate Semantic Plugin Selection."""
    selector = SemanticPluginSelector(top_k=3, min_score=0.1)
    selector.register_functions(DEMO_FUNCTIONS)

    queries = [
        "I need a refund for order #12345",
        "What's the status of my order?",
        "Can you update my email address?",
        "I need to speak with a human agent",
    ]

    for query in queries:
        scored = selector.select(query)
        print(f"Query: {query}")
        print(f"Selected {len(scored)} functions (from {len(DEMO_FUNCTIONS)} total):")
        for sf in scored:
            pin = " [PINNED]" if sf.function.pinned else ""
            print(f"  {sf.function.fully_qualified_name} (score: {sf.score:.3f}){pin}")
        print()

    # Show planner-ready format
    print("=== Planner Function Block ===")
    scored = selector.select("I need a refund for order #12345")
    print(selector.format_for_planner(scored))


if __name__ == "__main__":
    main()
