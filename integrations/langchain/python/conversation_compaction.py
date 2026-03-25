"""
Conversation Compaction -- LangChain Integration

Replaces LangChain's deprecated ConversationSummaryBufferMemory with a
structured fact extraction approach. Uses an LLM chain to extract decisions,
facts, current state, and user preferences from older messages rather than
producing a generic summary.

Pattern: https://github.com/context-engineering-handbook/patterns/compression/conversation-compaction.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionResult:
    """Immutable record of a compaction operation."""

    summary: str
    preserved_count: int
    removed_count: int
    tokens_before: int
    tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


# ---------------------------------------------------------------------------
# Extraction prompt -- the core of structured compaction
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a context compaction engine. Extract structured facts "
                "from conversation turns. Be precise and factual. Include file "
                "paths, variable names, and specific values -- not vague "
                "descriptions.\n\n"
                "Output ONLY the structured summary in the format below."
            ),
        ),
        (
            "human",
            (
                "Analyze these conversation turns and extract a structured summary.\n\n"
                "CONVERSATION TURNS:\n{turns}\n\n"
                "Extract the following categories:\n\n"
                "## Decisions Made\n"
                '- List each decision as: "<what> -- <why, if stated>"\n\n'
                "## Facts Established\n"
                "- Concrete information discovered (error messages, file contents, API behavior)\n\n"
                "## Current State\n"
                "- Files modified and how\n"
                "- Current step in the overall task\n"
                "- What remains to be done\n\n"
                "## User Preferences\n"
                "- Expressed constraints, style preferences, or requirements\n\n"
                "## Key Context\n"
                "- Any other information needed to continue without re-reading the original turns\n\n"
                "Omit reasoning chains, exploratory tangents, and superseded information."
            ),
        ),
    ]
)


# ---------------------------------------------------------------------------
# CompactionChain -- a composable LCEL chain for conversation compaction
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return len(text) // 4


def _estimate_message_tokens(msg: BaseMessage) -> int:
    """Estimate token count for a LangChain message."""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return _estimate_tokens(content)


class ConversationCompactor:
    """Compacts older conversation turns using structured LLM extraction.

    Unlike LangChain's deprecated ConversationSummaryBufferMemory, this
    extracts structured facts (decisions, state, preferences) rather than
    producing a generic paragraph summary. This preserves the highest-signal
    information for the model to use.

    Usage:
        compactor = ConversationCompactor(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            max_context_tokens=100_000,
        )

        # In your agent loop:
        if compactor.should_compact(messages):
            messages, result = await compactor.compact(messages)
            print(f"Saved {result.tokens_saved} tokens")
    """

    def __init__(
        self,
        llm: Runnable | None = None,
        max_context_tokens: int = 100_000,
        compaction_threshold: float = 0.75,
        preserve_recent_turns: int = 10,
    ) -> None:
        self._llm = llm
        self._max_context_tokens = max_context_tokens
        self._compaction_threshold = compaction_threshold
        self._preserve_recent_turns = preserve_recent_turns
        # Build the extraction chain using LCEL
        if llm is not None:
            self._extraction_chain = EXTRACTION_PROMPT | llm
        else:
            self._extraction_chain = None

    def should_compact(self, messages: list[BaseMessage]) -> bool:
        """Check if the conversation has crossed the compaction threshold."""
        total_tokens = sum(_estimate_message_tokens(m) for m in messages)
        threshold = int(self._max_context_tokens * self._compaction_threshold)
        return total_tokens > threshold

    def _split_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """Split messages into (to_compact, to_preserve).

        System messages are always preserved. Recent turns are always preserved.
        """
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(non_system) <= self._preserve_recent_turns:
            return [], messages

        split_point = len(non_system) - self._preserve_recent_turns
        to_compact = non_system[:split_point]
        to_preserve = system_messages + non_system[split_point:]

        return to_compact, to_preserve

    def _format_turns(self, messages: list[BaseMessage]) -> str:
        """Format messages into a string for the extraction prompt."""
        lines: list[str] = []
        for msg in messages:
            role = msg.type  # "human", "ai", "system"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def compact(
        self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], CompactionResult]:
        """Compact older turns into a structured summary.

        Returns (new_messages, compaction_result). Does not mutate the input.
        """
        tokens_before = sum(_estimate_message_tokens(m) for m in messages)

        if not self.should_compact(messages):
            return list(messages), CompactionResult(
                summary="",
                preserved_count=len(messages),
                removed_count=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        to_compact, to_preserve = self._split_messages(messages)

        if not to_compact:
            return list(messages), CompactionResult(
                summary="",
                preserved_count=len(messages),
                removed_count=0,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        # Run the structured extraction chain
        formatted_turns = self._format_turns(to_compact)

        if self._extraction_chain is not None:
            response = await self._extraction_chain.ainvoke(
                {"turns": formatted_turns}
            )
            summary = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
        else:
            # Fallback: simple concatenation when no LLM is provided (for testing)
            summary = self._fallback_extract(to_compact)

        # Build the compacted message
        summary_message = SystemMessage(
            content=(
                f"[COMPACTED CONTEXT from {len(to_compact)} earlier turns]\n\n"
                f"{summary}"
            )
        )

        new_messages = [summary_message, *to_preserve]
        tokens_after = sum(_estimate_message_tokens(m) for m in new_messages)

        return new_messages, CompactionResult(
            summary=summary,
            preserved_count=len(to_preserve),
            removed_count=len(to_compact),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def _fallback_extract(self, messages: list[BaseMessage]) -> str:
        """Simple extraction without an LLM (for testing/demo purposes)."""
        decisions: list[str] = []
        facts: list[str] = []

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Heuristic: lines starting with "Decision:" or containing "decided"
            for line in content.split("\n"):
                lower = line.strip().lower()
                if "decided" in lower or "decision:" in lower:
                    decisions.append(f"- {line.strip()}")
                elif any(
                    kw in lower
                    for kw in ["error:", "file:", "path:", "version:", "using"]
                ):
                    facts.append(f"- {line.strip()}")

        parts = ["## Decisions Made"]
        parts.extend(decisions or ["- No explicit decisions captured"])
        parts.append("\n## Facts Established")
        parts.extend(facts or ["- No explicit facts captured"])
        parts.append(
            f"\n## Current State\n- Compacted {len(messages)} messages"
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate Conversation Compaction without requiring an API key."""
    compactor = ConversationCompactor(
        llm=None,  # Use fallback extraction for demo
        max_context_tokens=500,  # Low threshold to trigger compaction
        compaction_threshold=0.5,
        preserve_recent_turns=4,
    )

    # Simulate a long conversation
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a coding assistant."),
        HumanMessage(content="Fix the auth bug in login.py"),
        AIMessage(
            content="I see the bug. The token validation is missing. Decision: use JWT tokens."
        ),
        HumanMessage(content="Yes, use JWT. Also, the file path is src/auth/login.py"),
        AIMessage(
            content="Updated src/auth/login.py with JWT validation. Using PyJWT version 2.8."
        ),
        HumanMessage(content="Now add rate limiting to the login endpoint"),
        AIMessage(
            content="I'll add rate limiting. Decision: decided to use Redis for rate limit storage."
        ),
        HumanMessage(content="Use a 5-request-per-minute limit"),
        AIMessage(
            content="Added rate limiting middleware with 5 req/min limit using Redis."
        ),
        HumanMessage(content="Now let's add logging"),
        AIMessage(content="I'll add structured logging to the auth module."),
    ]

    print(f"Messages: {len(messages)}")
    print(f"Should compact: {compactor.should_compact(messages)}")
    print()

    if compactor.should_compact(messages):
        import asyncio

        new_messages, result = asyncio.run(compactor.compact(messages))
        print(f"Compaction result:")
        print(f"  Removed: {result.removed_count} turns")
        print(f"  Preserved: {result.preserved_count} turns")
        print(f"  Tokens saved: {result.tokens_saved}")
        print(f"  New message count: {len(new_messages)}")
        print()
        print("=== Compacted context ===")
        for msg in new_messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"[{msg.type}]: {content[:200]}...")
            print()


if __name__ == "__main__":
    main()
