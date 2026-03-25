"""
Sub-Agent Delegation -- LangChain Integration

Implements parent-child agent composition using LangChain's AgentExecutor
with isolated contexts per sub-agent. The parent spawns child agents for
discrete sub-tasks, receives concise results, and keeps its own context clean.

Pattern: https://github.com/context-engineering-handbook/patterns/isolation/sub-agent-delegation.md
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig


# ---------------------------------------------------------------------------
# Domain types (from the handbook pattern)
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SubTask:
    """A discrete sub-task to delegate to a child agent."""

    task_id: str
    description: str
    relevant_files: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    parent_decisions: tuple[str, ...] = ()
    output_format: str = "Provide a concise summary of findings (max 200 words)."
    max_tokens: int = 50_000
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class SubTaskResult:
    """The concise result returned from a child agent to the parent."""

    task_id: str
    status: TaskStatus
    result: str
    token_count: int
    error: str | None = None


# ---------------------------------------------------------------------------
# Delegation context builder
# ---------------------------------------------------------------------------


def build_delegation_prompt(task: SubTask) -> str:
    """Build the system prompt for a child agent.

    The child receives only the information it needs -- no parent conversation
    history, no unrelated context. This is the key isolation mechanism.
    """
    sections = [f"## Task\n{task.description}"]

    if task.relevant_files:
        files_list = "\n".join(f"- {f}" for f in task.relevant_files)
        sections.append(f"## Relevant Files\n{files_list}")

    if task.constraints:
        constraints_list = "\n".join(f"- {c}" for c in task.constraints)
        sections.append(f"## Constraints\n{constraints_list}")

    if task.parent_decisions:
        decisions_list = "\n".join(f"- {d}" for d in task.parent_decisions)
        sections.append(
            f"## Prior Decisions (do not revisit)\n{decisions_list}"
        )

    sections.append(f"## Expected Output Format\n{task.output_format}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# SubAgentOrchestrator
# ---------------------------------------------------------------------------


class SubAgentOrchestrator:
    """Orchestrates sub-agent delegation with isolated contexts.

    Each child agent runs in its own LLM call with a fresh context window.
    The child's intermediate reasoning, tool outputs, and exploratory work
    never enter the parent's context -- only the concise result is returned.

    Usage with LangChain:

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o")
        orchestrator = SubAgentOrchestrator(llm=llm)

        results = await orchestrator.delegate_parallel([
            SubTask(task_id="research", description="Research JWT vs sessions"),
            SubTask(task_id="coverage", description="Check test coverage"),
        ])

        # Parent receives only the concise summaries
        summary = orchestrator.format_results_for_parent(results)
    """

    def __init__(
        self,
        llm: Runnable | None = None,
        default_max_tokens: int = 50_000,
    ) -> None:
        self._llm = llm
        self._default_max_tokens = default_max_tokens
        self._results: dict[str, SubTaskResult] = {}

    async def delegate(self, task: SubTask) -> SubTaskResult:
        """Spawn an isolated child agent for a sub-task.

        The child gets a fresh context with only the provided information.
        Its intermediate work never enters the parent's context.
        """
        system_prompt = build_delegation_prompt(task)

        try:
            if self._llm is not None:
                # Use the LLM with an isolated message list
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content="Execute the task described above."
                    ),
                ]
                response = await self._llm.ainvoke(messages)
                result_text = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
            else:
                # Fallback for demo without an LLM
                result_text = self._simulate_child(task)

            result = SubTaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result_text,
                token_count=len(result_text.split()) * 2,
            )

        except TimeoutError:
            result = SubTaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                result="",
                token_count=0,
                error=f"Child agent timed out after {task.timeout_seconds}s",
            )
        except Exception as e:
            result = SubTaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                result="",
                token_count=0,
                error=str(e),
            )

        self._results[task.task_id] = result
        return result

    async def delegate_parallel(
        self, tasks: list[SubTask]
    ) -> list[SubTaskResult]:
        """Run multiple sub-tasks in parallel with isolated contexts.

        Each child agent runs concurrently in its own context window.
        """
        return list(
            await asyncio.gather(*(self.delegate(task) for task in tasks))
        )

    def format_results_for_parent(
        self, results: list[SubTaskResult]
    ) -> str:
        """Format child results for injection into the parent context.

        Only the concise results are returned -- no intermediate reasoning,
        no raw file contents, no tool output chains.
        """
        sections: list[str] = []
        for r in results:
            if r.status == TaskStatus.COMPLETED:
                sections.append(
                    f"### Sub-task: {r.task_id}\n"
                    f"**Status**: completed\n"
                    f"**Result**: {r.result}"
                )
            else:
                sections.append(
                    f"### Sub-task: {r.task_id}\n"
                    f"**Status**: {r.status.value}\n"
                    f"**Error**: {r.error}"
                )
        return "\n\n".join(sections)

    @property
    def total_child_tokens(self) -> int:
        """Total tokens consumed across all child agents."""
        return sum(r.token_count for r in self._results.values())

    def _simulate_child(self, task: SubTask) -> str:
        """Simulate a child agent response for demo purposes."""
        return (
            f"[Simulated child result for '{task.task_id}']\n"
            f"Analyzed: {task.description}\n"
            f"Files checked: {', '.join(task.relevant_files) or 'none specified'}\n"
            f"Recommendation: Based on analysis, proceed with the standard approach.\n"
            f"Confidence: high"
        )


# ---------------------------------------------------------------------------
# LangChain agent composition helper
# ---------------------------------------------------------------------------


def create_child_agent_prompt(task: SubTask) -> ChatPromptTemplate:
    """Create a LangChain prompt template for a child agent.

    This can be used with create_tool_calling_agent to give the child
    access to tools while keeping its context isolated.

    Usage:
        from langchain.agents import create_tool_calling_agent, AgentExecutor

        prompt = create_child_agent_prompt(task)
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        result = await executor.ainvoke({"input": "Execute the task."})
    """
    system_prompt = build_delegation_prompt(task)

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate Sub-Agent Delegation without requiring an API key."""
    orchestrator = SubAgentOrchestrator(llm=None)  # Use simulation

    # Define sub-tasks for a feature implementation
    research_tasks = [
        SubTask(
            task_id="research-approach",
            description="Research implementation approaches for JWT authentication",
            relevant_files=("src/auth/", "docs/architecture.md"),
            constraints=(
                "Do not modify any files",
                "Focus on existing codebase patterns",
            ),
            parent_decisions=("Using Python/FastAPI stack",),
            output_format="Recommend one approach with 3 bullet points of justification.",
        ),
        SubTask(
            task_id="research-tests",
            description="Analyze existing test patterns and coverage for the auth module",
            relevant_files=("tests/auth/",),
            constraints=("Read-only analysis",),
            output_format="List existing test patterns and coverage gaps (max 150 words).",
        ),
        SubTask(
            task_id="research-security",
            description="Review security implications of JWT implementation",
            relevant_files=("src/auth/", "src/middleware/"),
            constraints=(
                "Check for OWASP Top 10 relevance",
                "Do not implement fixes",
            ),
            output_format="List security considerations as bullet points.",
        ),
    ]

    # Run in parallel with isolated contexts
    results = asyncio.run(orchestrator.delegate_parallel(research_tasks))

    # Parent receives only the concise summaries
    summary = orchestrator.format_results_for_parent(results)

    print("=== Parent Agent Receives ===")
    print(f"Total child tokens used: {orchestrator.total_child_tokens}")
    print(f"Results count: {len(results)}")
    print()
    print(summary)
    print()

    # Show the isolation: the parent's context contains only the summaries,
    # not the hundreds of lines of code, tool outputs, and intermediate
    # reasoning that the children processed.
    parent_context_tokens = len(summary) // 4
    estimated_child_work = sum(r.token_count for r in results) * 5
    print(f"Parent context cost: ~{parent_context_tokens} tokens")
    print(f"Estimated undelgated cost: ~{estimated_child_work} tokens")
    print(
        f"Savings: ~{estimated_child_work - parent_context_tokens} tokens "
        f"kept out of parent context"
    )


if __name__ == "__main__":
    main()
