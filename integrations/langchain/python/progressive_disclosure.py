"""
Progressive Disclosure -- LangChain Integration

Maps the Progressive Disclosure pattern to a custom LangChain Runnable that
stages context injection based on conversation state. The Runnable sits in an
LCEL chain and dynamically assembles the system prompt before each LLM call.

Pattern: https://github.com/context-engineering-handbook/patterns/construction/progressive-disclosure.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig


# ---------------------------------------------------------------------------
# Domain types (from the handbook pattern)
# ---------------------------------------------------------------------------


class DisclosureStage(Enum):
    BASELINE = auto()
    TASK_SCOPED = auto()
    DEEP_CONTEXT = auto()
    RESOLUTION = auto()


@dataclass(frozen=True)
class ContextBlock:
    """A unit of context with metadata about when it should be disclosed."""

    name: str
    content: str
    stage: DisclosureStage
    token_estimate: int
    ttl_turns: int | None = None
    trigger: Callable[[ConversationState], bool] | None = None


@dataclass
class ConversationState:
    """Tracks conversation progress to drive disclosure decisions."""

    current_turn: int = 0
    task_type: str | None = None
    files_referenced: set[str] = field(default_factory=set)
    tools_used: set[str] = field(default_factory=set)
    user_messages: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LangChain Runnable: ProgressiveDisclosureRunnable
#
# This Runnable wraps the handbook's ProgressiveContext logic and exposes it
# as a composable LCEL step. It accepts a list of BaseMessage and returns
# a modified list with the appropriate system context prepended.
# ---------------------------------------------------------------------------


class ProgressiveDisclosureRunnable(Runnable[list[BaseMessage], list[BaseMessage]]):
    """LangChain Runnable that injects staged context into a message list.

    Usage in an LCEL chain:

        from langchain_openai import ChatOpenAI

        disclosure = ProgressiveDisclosureRunnable(max_tokens=8000)
        disclosure.register(ContextBlock(...))

        chain = disclosure | ChatOpenAI(model="gpt-4o")
        result = chain.invoke([HumanMessage(content="Review this PR")])
    """

    def __init__(self, max_tokens: int = 16_000) -> None:
        self._blocks: list[ContextBlock] = []
        self._active_blocks: list[tuple[ContextBlock, int]] = []
        self._max_tokens = max_tokens
        self._state = ConversationState()

    # -- Registration API --------------------------------------------------

    def register(self, block: ContextBlock) -> None:
        """Register a context block for potential disclosure."""
        self._blocks.append(block)

    def set_task(self, task_type: str) -> None:
        """Transition from BASELINE to TASK_SCOPED stage."""
        self._state.task_type = task_type
        self._promote_stage(DisclosureStage.TASK_SCOPED)

    def add_deep_context(
        self, name: str, content: str, token_estimate: int, ttl_turns: int = 5
    ) -> None:
        """Inject deep context on demand (e.g., from a tool call)."""
        block = ContextBlock(
            name=name,
            content=content,
            stage=DisclosureStage.DEEP_CONTEXT,
            token_estimate=token_estimate,
            ttl_turns=ttl_turns,
        )
        self._active_blocks.append((block, self._state.current_turn))

    def summarize_and_prune(self, summary: str) -> None:
        """Replace deep context with a summary to free token budget."""
        self._active_blocks = [
            (block, turn)
            for block, turn in self._active_blocks
            if block.stage != DisclosureStage.DEEP_CONTEXT
        ]
        summary_block = ContextBlock(
            name="context_summary",
            content=summary,
            stage=DisclosureStage.RESOLUTION,
            token_estimate=len(summary) // 4,
        )
        self._active_blocks.append((summary_block, self._state.current_turn))

    # -- Runnable interface ------------------------------------------------

    def invoke(
        self,
        input: list[BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[BaseMessage]:
        """Process messages: advance state, build context, prepend system message."""
        # Extract user messages for state tracking
        for msg in input:
            if isinstance(msg, HumanMessage):
                self._state.current_turn += 1
                self._state.user_messages.append(str(msg.content))

        self._evaluate_triggers()
        self._expire_stale_blocks()

        # Build the assembled context
        context_str = self._build_context()

        # Prepend as a SystemMessage (LangChain convention)
        system_msg = SystemMessage(content=context_str)

        # Remove any existing system messages to avoid duplication
        non_system = [m for m in input if not isinstance(m, SystemMessage)]
        return [system_msg, *non_system]

    # -- Internal context assembly -----------------------------------------

    def _build_context(self) -> str:
        """Assemble the current context string within token budget."""
        baseline = [b for b in self._blocks if b.stage == DisclosureStage.BASELINE]
        all_blocks = [(b, 0) for b in baseline] + self._active_blocks
        all_blocks.sort(key=lambda pair: pair[0].stage.value)

        result_parts: list[str] = []
        token_total = 0
        for block, _ in all_blocks:
            if token_total + block.token_estimate > self._max_tokens:
                continue
            result_parts.append(f"<!-- {block.name} -->\n{block.content}")
            token_total += block.token_estimate

        return "\n\n".join(result_parts)

    def _promote_stage(self, stage: DisclosureStage) -> None:
        for block in self._blocks:
            if block.stage == stage:
                already_active = any(
                    b.name == block.name for b, _ in self._active_blocks
                )
                if not already_active:
                    self._active_blocks.append(
                        (block, self._state.current_turn)
                    )

    def _evaluate_triggers(self) -> None:
        for block in self._blocks:
            if block.trigger and block.trigger(self._state):
                already_active = any(
                    b.name == block.name for b, _ in self._active_blocks
                )
                if not already_active:
                    self._active_blocks.append(
                        (block, self._state.current_turn)
                    )

    def _expire_stale_blocks(self) -> None:
        self._active_blocks = [
            (block, added_turn)
            for block, added_turn in self._active_blocks
            if block.ttl_turns is None
            or (self._state.current_turn - added_turn) < block.ttl_turns
        ]

    @property
    def token_usage(self) -> int:
        baseline_tokens = sum(
            b.token_estimate
            for b in self._blocks
            if b.stage == DisclosureStage.BASELINE
        )
        active_tokens = sum(b.token_estimate for b, _ in self._active_blocks)
        return baseline_tokens + active_tokens


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate Progressive Disclosure as a LangChain Runnable."""

    # Create the runnable
    disclosure = ProgressiveDisclosureRunnable(max_tokens=8000)

    # Stage 0: Baseline -- always present
    disclosure.register(
        ContextBlock(
            name="role",
            content="You are a senior code reviewer. Be concise and precise.",
            stage=DisclosureStage.BASELINE,
            token_estimate=20,
        )
    )
    disclosure.register(
        ContextBlock(
            name="output_format",
            content="Respond in markdown. Use code fences with language tags.",
            stage=DisclosureStage.BASELINE,
            token_estimate=15,
        )
    )

    # Stage 1: Activated when task is classified as "code_review"
    disclosure.register(
        ContextBlock(
            name="review_guidelines",
            content=(
                "When reviewing code:\n"
                "1. Check for correctness first\n"
                "2. Then readability\n"
                "3. Then performance\n"
                "Flag security issues immediately."
            ),
            stage=DisclosureStage.TASK_SCOPED,
            token_estimate=40,
        )
    )

    # Trigger-based: activate when user mentions "database"
    disclosure.register(
        ContextBlock(
            name="db_schema",
            content="Schema: users(id, email, name, created_at), orders(id, user_id, total, status)",
            stage=DisclosureStage.TASK_SCOPED,
            token_estimate=30,
            trigger=lambda state: any(
                "database" in msg.lower() for msg in state.user_messages
            ),
        )
    )

    # -- Simulate a conversation using the Runnable --

    print("=== Turn 1: Initial request ===")
    messages_t1 = [HumanMessage(content="Please review this pull request")]
    result_t1 = disclosure.invoke(messages_t1)
    disclosure.set_task("code_review")
    print(f"Token usage: {disclosure.token_usage}")
    print(f"Messages out: {len(result_t1)}")
    print(f"System prompt preview: {result_t1[0].content[:120]}...")
    print()

    print("=== Turn 2: Mention database (triggers db_schema) ===")
    messages_t2 = [HumanMessage(content="Here's the diff for the database migration")]
    result_t2 = disclosure.invoke(messages_t2)
    print(f"Token usage: {disclosure.token_usage}")
    print()

    print("=== Turn 2+: Add deep context from a tool call ===")
    disclosure.add_deep_context(
        name="migration_file",
        content="ALTER TABLE users ADD COLUMN last_login TIMESTAMP;",
        token_estimate=20,
    )
    print(f"Token usage after deep context: {disclosure.token_usage}")
    print()

    print("=== Prune and summarize ===")
    disclosure.summarize_and_prune(
        "Previously reviewed a DB migration adding last_login to users table."
    )
    print(f"Token usage after prune: {disclosure.token_usage}")
    print()

    # Show the final assembled context
    messages_t3 = [HumanMessage(content="What's the final verdict?")]
    result_t3 = disclosure.invoke(messages_t3)
    print("=== Final assembled context ===")
    print(result_t3[0].content)


if __name__ == "__main__":
    main()
