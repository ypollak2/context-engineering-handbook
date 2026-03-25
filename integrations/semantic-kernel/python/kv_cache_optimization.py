"""
KV-Cache Optimization -- Semantic Kernel Integration

Implements prompt template management with stable prefixes for maximum
KV-cache hit rates. Separates the frozen prefix (system prompt, tool
definitions, static examples) from the dynamic suffix (user messages,
retrieved context) to enable cache reuse across calls.

Pattern: https://github.com/context-engineering-handbook/patterns/optimization/kv-cache-optimization.md
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Message types compatible with Semantic Kernel's ChatHistory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheableMessage:
    """An immutable message with cache control metadata.

    Maps to Semantic Kernel's ChatMessageContent with additional cache hints.
    In production, use SK's actual message types:
        from semantic_kernel.contents import ChatMessageContent
    """

    role: str  # "system", "user", "assistant"
    content: str
    cache_control: dict[str, Any] = field(default_factory=dict)

    @property
    def is_cacheable(self) -> bool:
        """Messages in the frozen prefix are cacheable."""
        return self.cache_control.get("type") == "ephemeral"


# ---------------------------------------------------------------------------
# StablePrefixManager
# ---------------------------------------------------------------------------


class StablePrefixManager:
    """Manages the frozen prefix portion of the context for KV-cache optimization.

    Ensures that the stable content (system prompt, tool definitions, static
    examples) always appears first and in the same order. Variable content
    is appended at the end, preserving the cache for the prefix.

    Usage with Semantic Kernel:
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

        manager = StablePrefixManager()
        manager = manager.with_system_prompt("You are a code reviewer.")
        manager = manager.with_tool_definitions([...])
        manager = manager.with_static_examples([...])

        # Per-turn: append dynamic content and build messages
        messages = manager.build_messages(
            user_message="Review this code",
            retrieved_context="<context>...</context>",
        )

        # The frozen prefix is identical across calls -> KV-cache hit
        chat = kernel.get_service(type=AzureChatCompletion)
        response = await chat.get_chat_message_content(messages, settings)
    """

    def __init__(self) -> None:
        self._system_prompt: str = ""
        self._tool_definitions: list[dict[str, Any]] = []
        self._static_examples: list[CacheableMessage] = []
        self._prefix_hash: str = ""

    def with_system_prompt(self, prompt: str) -> StablePrefixManager:
        """Set the system prompt. Returns a new instance (immutable pattern)."""
        new = StablePrefixManager()
        new._system_prompt = prompt
        new._tool_definitions = list(self._tool_definitions)
        new._static_examples = list(self._static_examples)
        new._recompute_hash()
        return new

    def with_tool_definitions(
        self, tools: list[dict[str, Any]]
    ) -> StablePrefixManager:
        """Set tool definitions. Returns a new instance."""
        new = StablePrefixManager()
        new._system_prompt = self._system_prompt
        new._tool_definitions = list(tools)
        new._static_examples = list(self._static_examples)
        new._recompute_hash()
        return new

    def with_static_examples(
        self, examples: list[CacheableMessage]
    ) -> StablePrefixManager:
        """Set static few-shot examples. Returns a new instance."""
        new = StablePrefixManager()
        new._system_prompt = self._system_prompt
        new._tool_definitions = list(self._tool_definitions)
        new._static_examples = list(examples)
        new._recompute_hash()
        return new

    def build_messages(
        self,
        user_message: str,
        retrieved_context: str = "",
        conversation_history: list[CacheableMessage] | None = None,
    ) -> list[CacheableMessage]:
        """Build the full message list with frozen prefix + dynamic suffix.

        The frozen prefix (system prompt + tools + examples) is always
        identical, maximizing KV-cache hit rates. Dynamic content is
        appended at the end.

        Message order:
        1. [FROZEN] System prompt (with tool definitions embedded)
        2. [FROZEN] Static few-shot examples
        3. [DYNAMIC] Conversation history
        4. [DYNAMIC] Retrieved context (as a system message)
        5. [DYNAMIC] Current user message
        """
        messages: list[CacheableMessage] = []

        # 1. Frozen: System prompt with tool definitions
        system_content = self._system_prompt
        if self._tool_definitions:
            tools_block = self._format_tools()
            system_content += f"\n\n{tools_block}"

        messages.append(
            CacheableMessage(
                role="system",
                content=system_content,
                cache_control={"type": "ephemeral"},
            )
        )

        # 2. Frozen: Static examples
        for example in self._static_examples:
            messages.append(example)

        # --- Cache boundary: everything above should be identical across calls ---

        # 3. Dynamic: Conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # 4. Dynamic: Retrieved context
        if retrieved_context:
            messages.append(
                CacheableMessage(
                    role="system",
                    content=f"[Retrieved Context]\n{retrieved_context}",
                )
            )

        # 5. Dynamic: Current user message
        messages.append(CacheableMessage(role="user", content=user_message))

        return messages

    @property
    def prefix_hash(self) -> str:
        """Hash of the frozen prefix for cache tracking."""
        return self._prefix_hash

    @property
    def prefix_token_estimate(self) -> int:
        """Estimated tokens in the frozen prefix."""
        system_tokens = len(self._system_prompt) // 4
        tool_tokens = (
            sum(len(json.dumps(t)) // 4 for t in self._tool_definitions)
            if self._tool_definitions
            else 0
        )
        example_tokens = sum(
            len(e.content) // 4 for e in self._static_examples
        )
        return system_tokens + tool_tokens + example_tokens

    def _format_tools(self) -> str:
        """Format tool definitions for inclusion in the system prompt."""
        sections: list[str] = []
        for tool in self._tool_definitions:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            param_str = ", ".join(
                f"{k}: {v.get('type', 'any')}"
                for k, v in params.items()
            )
            sections.append(f"- **{name}**: {desc}\n  Parameters: {param_str}")
        return "<tools>\n" + "\n\n".join(sections) + "\n</tools>"

    def _recompute_hash(self) -> None:
        """Recompute the prefix hash when content changes."""
        content = (
            self._system_prompt
            + json.dumps(self._tool_definitions, sort_keys=True)
            + "".join(e.content for e in self._static_examples)
        )
        self._prefix_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache analytics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheAnalytics:
    """Tracks KV-cache hit rates across calls."""

    prefix_tokens: int
    total_tokens: int
    cache_hit_ratio: float
    estimated_savings_pct: float


def analyze_cache_efficiency(
    prefix_tokens: int,
    dynamic_tokens_per_call: list[int],
) -> CacheAnalytics:
    """Estimate cache efficiency for a series of calls.

    With KV-cache optimization, the prefix is computed once and reused.
    Without it, the entire prompt is recomputed each call.
    """
    num_calls = len(dynamic_tokens_per_call)
    if num_calls == 0:
        return CacheAnalytics(
            prefix_tokens=prefix_tokens,
            total_tokens=0,
            cache_hit_ratio=0.0,
            estimated_savings_pct=0.0,
        )

    total_dynamic = sum(dynamic_tokens_per_call)

    # Without cache: every call computes prefix + dynamic
    without_cache = num_calls * prefix_tokens + total_dynamic

    # With cache: first call computes prefix, subsequent calls reuse it
    with_cache = prefix_tokens + total_dynamic

    savings_pct = (
        (without_cache - with_cache) / without_cache * 100
        if without_cache > 0
        else 0
    )

    cache_hit_ratio = (
        prefix_tokens * (num_calls - 1) / without_cache
        if without_cache > 0
        else 0
    )

    return CacheAnalytics(
        prefix_tokens=prefix_tokens,
        total_tokens=with_cache,
        cache_hit_ratio=cache_hit_ratio,
        estimated_savings_pct=savings_pct,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate KV-Cache Optimization with Semantic Kernel patterns."""

    # Build a stable prefix manager
    manager = StablePrefixManager()
    manager = manager.with_system_prompt(
        "You are a customer support agent for Acme Corp. "
        "Be helpful, concise, and professional. "
        "Always verify the customer's identity before making account changes."
    )

    manager = manager.with_tool_definitions(
        [
            {
                "name": "get_order",
                "description": "Look up order details by ID",
                "parameters": {"order_id": {"type": "string"}},
            },
            {
                "name": "process_refund",
                "description": "Process a refund for an order",
                "parameters": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
            {
                "name": "search_knowledge_base",
                "description": "Search for articles in the help center",
                "parameters": {"query": {"type": "string"}},
            },
        ]
    )

    manager = manager.with_static_examples(
        [
            CacheableMessage(
                role="user",
                content="Where's my order #1234?",
                cache_control={"type": "ephemeral"},
            ),
            CacheableMessage(
                role="assistant",
                content="Let me look that up for you. [calls get_order]",
                cache_control={"type": "ephemeral"},
            ),
        ]
    )

    print(f"Prefix hash: {manager.prefix_hash}")
    print(f"Prefix token estimate: {manager.prefix_token_estimate}")
    print()

    # Simulate multiple calls with the same prefix but different user messages
    print("=== Call 1 ===")
    messages_1 = manager.build_messages(
        user_message="I need a refund for order #5678",
        retrieved_context="Order #5678: Delivered 3 days ago, $49.99",
    )
    for msg in messages_1:
        cacheable = "[CACHED] " if msg.cache_control.get("type") == "ephemeral" else "[DYNAMIC] "
        content_preview = msg.content[:80].replace("\n", " ")
        print(f"  {cacheable}[{msg.role}] {content_preview}...")
    print()

    print("=== Call 2 (same prefix = cache hit) ===")
    messages_2 = manager.build_messages(
        user_message="What's the status of order #9999?",
    )
    for msg in messages_2:
        cacheable = "[CACHED] " if msg.cache_control.get("type") == "ephemeral" else "[DYNAMIC] "
        content_preview = msg.content[:80].replace("\n", " ")
        print(f"  {cacheable}[{msg.role}] {content_preview}...")
    print()

    # Cache analytics
    prefix_tokens = manager.prefix_token_estimate
    dynamic_per_call = [50, 30, 45, 60, 35]  # Simulated dynamic tokens per call

    analytics = analyze_cache_efficiency(prefix_tokens, dynamic_per_call)
    print("=== Cache Analytics ===")
    print(f"Prefix tokens: {analytics.prefix_tokens}")
    print(f"Total tokens (with cache): {analytics.total_tokens}")
    print(f"Cache hit ratio: {analytics.cache_hit_ratio:.1%}")
    print(f"Estimated savings: {analytics.estimated_savings_pct:.1f}%")
    print()

    # Show the cost comparison
    num_calls = len(dynamic_per_call)
    without_cache = num_calls * prefix_tokens + sum(dynamic_per_call)
    with_cache = analytics.total_tokens
    print(f"Without cache: {without_cache} tokens across {num_calls} calls")
    print(f"With cache:    {with_cache} tokens across {num_calls} calls")
    print(f"Savings:       {without_cache - with_cache} tokens ({analytics.estimated_savings_pct:.0f}%)")


if __name__ == "__main__":
    main()
