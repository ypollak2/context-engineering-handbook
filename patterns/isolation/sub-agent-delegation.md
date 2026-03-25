# Sub-Agent Delegation

> Spawn child agents with isolated context windows for specific sub-tasks, keeping the parent agent's context clean and enabling parallel execution.

## Problem

Complex agent tasks require exploratory work: reading multiple files to understand a codebase, searching the web for solutions, trying different approaches. If all of this happens in a single context window, the parent agent's context fills up with intermediate observations, dead-end reasoning, and tangential information that is irrelevant to the final decision.

Without sub-agent delegation:
- Exploratory research pollutes the main context with information only needed temporarily.
- The parent agent "forgets" its high-level plan because low-level details crowd it out.
- Sequential processing prevents parallelizing independent sub-tasks.
- A single failed sub-task can corrupt the context for the entire session.
- Context window limits restrict the total scope of work an agent can handle.

## Solution

When the parent agent identifies a discrete sub-task, it spawns a child agent with a fresh, isolated context window. The parent provides only the information the child needs (task description, relevant file paths, constraints) and receives back only the result (answer, generated code, summary of findings). The child's intermediate reasoning, tool outputs, and exploratory work never enter the parent's context.

## How It Works

```
Parent Agent Context (stays clean)
+--------------------------------------------------+
| System prompt + user request                      |
| Plan: 1. Research auth options  2. Implement      |
|                                                   |
| [Delegating: "Research JWT vs session auth"]      |
|   --> Child A (isolated context)                  |
|       +----------------------------------+        |
|       | Task: Compare JWT vs sessions    |        |
|       | read_file("docs/requirements.md")|        |  These never enter
|       | search("JWT vs session 2025")    |        |  the parent context
|       | read_file("auth/current.py")     |        |
|       | [500 lines of research]          |        |
|       +----------------------------------+        |
|   <-- Result: "JWT recommended because..."        |
|                                                   |
| [Delegating: "Check test coverage"]               |
|   --> Child B (isolated context, runs in parallel)|
|       +----------------------------------+        |
|       | Task: Analyze test coverage      |        |
|       | run_command("pytest --cov")      |        |
|       | [coverage report output]         |        |
|       +----------------------------------+        |
|   <-- Result: "Auth module: 34% coverage"         |
|                                                   |
| Decision: Use JWT, need tests for auth module     |
+--------------------------------------------------+
Parent used ~2,000 tokens. Without delegation: ~15,000.
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SubTask:
    task_id: str
    description: str
    context: dict[str, Any]         # Only what the child needs
    constraints: list[str] = field(default_factory=list)
    max_tokens: int = 50_000        # Child's context budget
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class SubTaskResult:
    task_id: str
    status: TaskStatus
    result: str                     # Concise result for the parent
    token_count: int                # Tokens used by the child
    error: str | None = None


@dataclass(frozen=True)
class DelegationContext:
    """The minimal context package sent to a child agent."""
    task_description: str
    relevant_files: list[str]
    constraints: list[str]
    parent_decisions: list[str]     # Decisions already made that the child should respect
    output_format: str              # What the parent expects back

    def to_system_prompt(self) -> str:
        sections = [
            f"## Task\n{self.task_description}",
            f"## Relevant Files\n" + "\n".join(f"- {f}" for f in self.relevant_files),
        ]
        if self.constraints:
            sections.append(
                f"## Constraints\n" + "\n".join(f"- {c}" for c in self.constraints)
            )
        if self.parent_decisions:
            sections.append(
                f"## Prior Decisions (do not revisit)\n"
                + "\n".join(f"- {d}" for d in self.parent_decisions)
            )
        sections.append(f"## Expected Output Format\n{self.output_format}")
        return "\n\n".join(sections)


class SubAgentOrchestrator:
    """Orchestrates sub-agent delegation with isolated contexts.

    Note: This class manages mutable state (running tasks) by design,
    as it represents a stateful orchestration process.
    """

    def __init__(self, llm_client, default_max_tokens: int = 50_000):
        self._llm_client = llm_client
        self._default_max_tokens = default_max_tokens
        self._results: dict[str, SubTaskResult] = {}

    async def delegate(self, sub_task: SubTask) -> SubTaskResult:
        """Spawn an isolated child agent for a sub-task.

        The child gets a fresh context with only the provided information.
        Its intermediate work never enters the parent's context.
        """
        delegation_ctx = DelegationContext(
            task_description=sub_task.description,
            relevant_files=sub_task.context.get("files", []),
            constraints=sub_task.constraints,
            parent_decisions=sub_task.context.get("decisions", []),
            output_format=sub_task.context.get(
                "output_format",
                "Provide a concise summary of findings (max 200 words)."
            ),
        )

        system_prompt = delegation_ctx.to_system_prompt()

        try:
            # The child agent runs in its own context window
            child_response = await self._llm_client.complete(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": "Execute the task described above."}],
                max_tokens=sub_task.max_tokens,
                timeout=sub_task.timeout_seconds,
                # Tools are provided but their outputs stay in the child's context
                tools=sub_task.context.get("tools", []),
            )

            result = SubTaskResult(
                task_id=sub_task.task_id,
                status=TaskStatus.COMPLETED,
                result=child_response,
                token_count=len(child_response.split()) * 2,
            )
        except TimeoutError:
            result = SubTaskResult(
                task_id=sub_task.task_id,
                status=TaskStatus.FAILED,
                result="",
                token_count=0,
                error=f"Child agent timed out after {sub_task.timeout_seconds}s",
            )
        except Exception as e:
            result = SubTaskResult(
                task_id=sub_task.task_id,
                status=TaskStatus.FAILED,
                result="",
                token_count=0,
                error=str(e),
            )

        self._results[sub_task.task_id] = result
        return result

    async def delegate_parallel(
        self, sub_tasks: list[SubTask]
    ) -> list[SubTaskResult]:
        """Run multiple sub-tasks in parallel with isolated contexts."""
        import asyncio
        return await asyncio.gather(
            *(self.delegate(task) for task in sub_tasks)
        )

    def format_results_for_parent(
        self, results: list[SubTaskResult]
    ) -> str:
        """Format child results for injection into the parent context.

        Only the concise results are returned -- no intermediate reasoning.
        """
        sections = []
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


# --- Usage Example ---

async def implement_feature(llm_client, feature_request: str):
    """Parent agent that delegates research and implementation sub-tasks."""
    orchestrator = SubAgentOrchestrator(llm_client)

    # Phase 1: Parallel research (child contexts are isolated)
    research_tasks = [
        SubTask(
            task_id="research-approach",
            description=f"Research implementation approaches for: {feature_request}",
            context={
                "files": ["src/", "docs/architecture.md"],
                "output_format": "Recommend one approach with 3 bullet points of justification.",
            },
            constraints=["Do not modify any files", "Focus on the existing codebase patterns"],
        ),
        SubTask(
            task_id="research-tests",
            description=f"Analyze existing test patterns and coverage for the module related to: {feature_request}",
            context={
                "files": ["tests/"],
                "output_format": "List existing test patterns and coverage gaps (max 150 words).",
            },
            constraints=["Read-only analysis"],
        ),
    ]

    # Both run in parallel, each with isolated context
    results = await orchestrator.delegate_parallel(research_tasks)

    # Parent receives only the concise results
    research_summary = orchestrator.format_results_for_parent(results)
    print(f"Research complete. Summary added to parent context:\n{research_summary}")

    # Parent makes the decision based on concise summaries,
    # not the 1000s of tokens of raw file contents the children read
```

### TypeScript

```typescript
interface SubTask {
  readonly taskId: string;
  readonly description: string;
  readonly context: Readonly<Record<string, unknown>>;
  readonly constraints: readonly string[];
  readonly maxTokens: number;
  readonly timeoutMs: number;
}

interface SubTaskResult {
  readonly taskId: string;
  readonly status: "completed" | "failed";
  readonly result: string;
  readonly tokenCount: number;
  readonly error?: string;
}

interface DelegationContext {
  readonly taskDescription: string;
  readonly relevantFiles: readonly string[];
  readonly constraints: readonly string[];
  readonly parentDecisions: readonly string[];
  readonly outputFormat: string;
}

interface LLMClient {
  complete(params: {
    systemPrompt: string;
    messages: readonly { role: string; content: string }[];
    maxTokens: number;
    timeoutMs: number;
    tools?: readonly unknown[];
  }): Promise<string>;
}

function buildDelegationPrompt(ctx: DelegationContext): string {
  const sections = [
    `## Task\n${ctx.taskDescription}`,
    `## Relevant Files\n${ctx.relevantFiles.map((f) => `- ${f}`).join("\n")}`,
  ];

  if (ctx.constraints.length > 0) {
    sections.push(
      `## Constraints\n${ctx.constraints.map((c) => `- ${c}`).join("\n")}`
    );
  }

  if (ctx.parentDecisions.length > 0) {
    sections.push(
      `## Prior Decisions (do not revisit)\n${ctx.parentDecisions.map((d) => `- ${d}`).join("\n")}`
    );
  }

  sections.push(`## Expected Output Format\n${ctx.outputFormat}`);
  return sections.join("\n\n");
}

function createSubTask(
  taskId: string,
  description: string,
  context: Record<string, unknown> = {},
  constraints: string[] = [],
  maxTokens: number = 50_000,
  timeoutMs: number = 120_000
): SubTask {
  return { taskId, description, context, constraints, maxTokens, timeoutMs };
}

async function delegateSubTask(
  client: LLMClient,
  subTask: SubTask
): Promise<SubTaskResult> {
  const delegationCtx: DelegationContext = {
    taskDescription: subTask.description,
    relevantFiles: (subTask.context.files as string[]) ?? [],
    constraints: [...subTask.constraints],
    parentDecisions: (subTask.context.decisions as string[]) ?? [],
    outputFormat:
      (subTask.context.outputFormat as string) ??
      "Provide a concise summary of findings (max 200 words).",
  };

  const systemPrompt = buildDelegationPrompt(delegationCtx);

  try {
    const response = await client.complete({
      systemPrompt,
      messages: [{ role: "user", content: "Execute the task described above." }],
      maxTokens: subTask.maxTokens,
      timeoutMs: subTask.timeoutMs,
      tools: (subTask.context.tools as unknown[]) ?? [],
    });

    return {
      taskId: subTask.taskId,
      status: "completed",
      result: response,
      tokenCount: Math.ceil(response.split(/\s+/).length * 1.5),
    };
  } catch (err) {
    return {
      taskId: subTask.taskId,
      status: "failed",
      result: "",
      tokenCount: 0,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function delegateParallel(
  client: LLMClient,
  subTasks: readonly SubTask[]
): Promise<readonly SubTaskResult[]> {
  return Promise.all(subTasks.map((task) => delegateSubTask(client, task)));
}

function formatResultsForParent(
  results: readonly SubTaskResult[]
): string {
  return results
    .map((r) => {
      if (r.status === "completed") {
        return `### Sub-task: ${r.taskId}\n**Status**: completed\n**Result**: ${r.result}`;
      }
      return `### Sub-task: ${r.taskId}\n**Status**: failed\n**Error**: ${r.error}`;
    })
    .join("\n\n");
}

// --- Usage Example ---

async function implementFeature(
  client: LLMClient,
  featureRequest: string
): Promise<void> {
  const researchTasks = [
    createSubTask(
      "research-approach",
      `Research implementation approaches for: ${featureRequest}`,
      {
        files: ["src/", "docs/architecture.md"],
        outputFormat: "Recommend one approach with 3 bullet points of justification.",
      },
      ["Do not modify any files", "Focus on existing codebase patterns"]
    ),
    createSubTask(
      "research-tests",
      `Analyze existing test patterns and coverage for: ${featureRequest}`,
      {
        files: ["tests/"],
        outputFormat: "List existing test patterns and coverage gaps (max 150 words).",
      },
      ["Read-only analysis"]
    ),
  ];

  // Both run in parallel with isolated contexts
  const results = await delegateParallel(client, researchTasks);

  // Parent receives only concise summaries
  const summary = formatResultsForParent(results);
  console.log(`Research complete. Summary:\n${summary}`);
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Parent context stays clean and focused | Additional LLM calls for each child agent (cost) |
| Enables parallel execution of independent tasks | Child agents lack the parent's full conversational context |
| Failed sub-tasks do not corrupt the parent context | Results must be summarized, which can lose nuance |
| Each child can use its full context budget for its task | Orchestration logic adds system complexity |
| Natural isolation boundary for different tool sets | Cold start: children must re-read files the parent already read |

## When to Use

- Complex tasks that involve distinct research and action phases.
- When exploratory work (searching, reading many files, trying approaches) would pollute the main context.
- When sub-tasks are independent and can benefit from parallel execution.
- When different sub-tasks need different tool sets or permissions.
- When you want to limit blast radius: a failing child agent does not take down the parent.

## When NOT to Use

- Simple, linear tasks where context pollution is not a concern.
- When the sub-task heavily depends on the nuance of the parent conversation (the context hand-off is lossy).
- When latency is critical and the overhead of spawning a child agent is unacceptable.
- When the total cost of multiple LLM sessions outweighs the benefit of isolation.
- When the sub-task requires continuous back-and-forth with the user (the child cannot interact with the user).

## Related Patterns

- **[Multi-Agent Context Orchestration](multi-agent-context-orchestration.md)** -- For peer-to-peer collaboration between multiple specialized agents, rather than parent-child delegation.
- **[Observation Masking](../compression/observation-masking.md)** -- An alternative when full isolation is overkill: mask stale tool outputs in the parent's context instead of delegating to a child.
- **[Conversation Compaction](../compression/conversation-compaction.md)** -- If you cannot delegate, compaction is the fallback for keeping a single context manageable.

## Real-World Examples

1. **Claude Code's Agent tool** -- Claude Code can spawn sub-agents for tasks like "investigate why this test is failing" or "search the codebase for all usages of this function." The sub-agent reads files, runs commands, and reports back a concise answer. The parent's context never sees the raw file contents.

2. **Devin's sub-agent architecture** -- Cognition's Devin uses sub-agents for research, planning, and implementation. The research agent explores documentation and code, then hands a summary to the planning agent, which hands a plan to the implementation agent.

3. **ChatGPT's tool sandboxing** -- When ChatGPT invokes the code interpreter or browser tool, the tool execution happens in an isolated context. The model receives a summary of the output, not the full execution trace.

4. **Custom coding agents** -- Any agent that needs to "look around" a codebase before making changes benefits from delegating the exploration to a sub-agent, so the main context stays focused on the actual implementation plan.
