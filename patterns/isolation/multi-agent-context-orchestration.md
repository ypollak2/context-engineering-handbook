# Multi-Agent Context Orchestration

> Manage what context flows between collaborating agents using explicit contracts, shared pools, and routing rules to prevent context explosion in multi-agent systems.

## Problem

When multiple specialized agents collaborate on a task -- a planner, a coder, a reviewer, a tester -- each agent generates context that could be relevant to others. Without orchestration, you face two bad options:

1. **Share everything**: Every agent sees every other agent's full context. Context windows explode, agents get confused by irrelevant information, and costs multiply.
2. **Share nothing**: Agents work in complete isolation and miss critical information from other agents, leading to inconsistent or contradictory outputs.

Without orchestration:
- Agent A's detailed reasoning distracts Agent B from its own task.
- Shared context grows as O(n^2) with the number of agents.
- No clear contract for what each agent receives and produces.
- Debugging multi-agent failures becomes impossible because context flow is implicit.
- Agents repeat work because they do not know what other agents have already established.

## Solution

Define explicit **context contracts** for each agent: what it receives (input schema) and what it produces (output schema). Route context between agents through an orchestration layer that transforms, filters, and budgets context before delivery. Optionally maintain a **shared context pool** for facts that all agents need, separate from agent-specific working context.

## How It Works

```
Shared Context Pool (accessible to all agents, read-only)
+-----------------------------------------------+
| Project: e-commerce checkout rewrite           |
| Stack: Python 3.12, FastAPI, PostgreSQL        |
| Constraints: Must maintain backward compat     |
| Decisions log: [JWT auth, Redis sessions, ...] |
+-----------------------------------------------+
         |              |              |
         v              v              v
+----------------+ +----------------+ +----------------+
| Planner Agent  | | Coder Agent    | | Reviewer Agent |
| IN:  user req, | | IN:  plan,     | | IN:  code diff,|
|   shared pool  | |   shared pool, | |   shared pool, |
|                | |   file paths   | |   plan summary |
| OUT: plan doc, | | OUT: code diff,| | OUT: review    |
|   task list    | |   test results | |   comments,    |
|                | |                | |   approval     |
+----------------+ +----------------+ +----------------+
     |                    |                    |
     | plan summary       | code diff          | review
     | (transformed)      | (as-is)            | (filtered)
     v                    v                    v
+-----------------------------------------------+
| Orchestrator: routes, transforms, budgets     |
| - Planner -> Coder: plan summary (not full    |
|   reasoning), task list, relevant file paths  |
| - Coder -> Reviewer: diff only (not all files |
|   read), test results summary                 |
| - Reviewer -> Coder: actionable comments only |
|   (not style nitpicks if config says so)      |
+-----------------------------------------------+
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"


@dataclass(frozen=True)
class ContextContract:
    """Defines what an agent receives and produces."""
    agent_role: AgentRole
    required_inputs: tuple[str, ...]    # Keys this agent must receive
    optional_inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()       # Keys this agent produces
    max_input_tokens: int = 50_000      # Budget for this agent's input context
    receives_shared_pool: bool = True


@dataclass(frozen=True)
class ContextMessage:
    """A piece of context flowing between agents."""
    key: str
    content: str
    source_agent: AgentRole
    token_count: int
    priority: int = 1  # Higher = more important when budgeting


@dataclass(frozen=True)
class SharedContextPool:
    """Immutable shared context available to all agents."""
    entries: tuple[ContextMessage, ...] = ()

    def add(self, message: ContextMessage) -> "SharedContextPool":
        return SharedContextPool(entries=self.entries + (message,))

    def get(self, key: str) -> ContextMessage | None:
        for entry in self.entries:
            if entry.key == key:
                return entry
        return None

    def to_prompt_section(self) -> str:
        if not self.entries:
            return ""
        lines = ["## Shared Context (established facts, do not re-derive)"]
        for entry in self.entries:
            lines.append(f"### {entry.key}\n{entry.content}")
        return "\n\n".join(lines)

    @property
    def total_tokens(self) -> int:
        return sum(e.token_count for e in self.entries)


@dataclass(frozen=True)
class TransformRule:
    """Rule for transforming context when routing between agents."""
    source_role: AgentRole
    target_role: AgentRole
    source_key: str
    target_key: str
    transform_fn: str  # Name of the transform function to apply
    # "identity" = pass as-is, "summarize" = LLM summarize, "truncate" = first N lines


class MultiAgentContextOrchestrator:
    """Orchestrates context flow between collaborating agents.

    Manages mutable routing state by design, as orchestration is
    inherently a stateful coordination process.
    """

    def __init__(
        self,
        contracts: list[ContextContract],
        transform_rules: list[TransformRule],
        llm_client=None,
    ):
        self._contracts = {c.agent_role: c for c in contracts}
        self._transform_rules = transform_rules
        self._llm_client = llm_client
        self._shared_pool = SharedContextPool()
        self._agent_outputs: dict[AgentRole, dict[str, ContextMessage]] = {}

    @property
    def shared_pool(self) -> SharedContextPool:
        return self._shared_pool

    def publish_to_shared_pool(self, message: ContextMessage) -> None:
        """Add a fact to the shared context pool (available to all agents)."""
        self._shared_pool = self._shared_pool.add(message)

    def record_agent_output(
        self, agent_role: AgentRole, key: str, content: str
    ) -> None:
        """Record an output produced by an agent."""
        message = ContextMessage(
            key=key,
            content=content,
            source_agent=agent_role,
            token_count=len(content.split()) * 2,
        )
        if agent_role not in self._agent_outputs:
            self._agent_outputs[agent_role] = {}
        self._agent_outputs[agent_role][key] = message

    async def _apply_transform(
        self, content: str, transform_fn: str
    ) -> str:
        """Apply a transformation to context content."""
        if transform_fn == "identity":
            return content

        if transform_fn == "truncate":
            lines = content.split("\n")
            if len(lines) > 50:
                return "\n".join(lines[:50]) + f"\n... [{len(lines) - 50} lines truncated]"
            return content

        if transform_fn == "summarize":
            if self._llm_client is None:
                raise ValueError("LLM client required for summarize transform")
            prompt = (
                "Summarize the following content concisely, preserving all "
                "actionable information and specific details (file paths, "
                "function names, error messages). Remove reasoning chains "
                "and exploratory discussion.\n\n"
                f"{content}"
            )
            return await self._llm_client.complete(prompt)

        raise ValueError(f"Unknown transform function: {transform_fn}")

    async def build_agent_context(
        self, target_role: AgentRole
    ) -> str:
        """Build the complete input context for a specific agent.

        Gathers required inputs from other agents' outputs, applies
        transforms, and assembles within the token budget.
        """
        contract = self._contracts.get(target_role)
        if contract is None:
            raise ValueError(f"No contract defined for {target_role}")

        sections: list[str] = []
        token_budget = contract.max_input_tokens

        # Add shared pool if the agent receives it
        if contract.receives_shared_pool:
            pool_section = self._shared_pool.to_prompt_section()
            if pool_section:
                sections.append(pool_section)
                token_budget -= self._shared_pool.total_tokens

        # Route context from other agents via transform rules
        for rule in self._transform_rules:
            if rule.target_role != target_role:
                continue

            source_outputs = self._agent_outputs.get(rule.source_role, {})
            source_msg = source_outputs.get(rule.source_key)

            if source_msg is None:
                if rule.source_key in contract.required_inputs:
                    raise ValueError(
                        f"Required input '{rule.source_key}' from "
                        f"{rule.source_role} not available for {target_role}"
                    )
                continue

            transformed = await self._apply_transform(
                source_msg.content, rule.transform_fn
            )

            transformed_tokens = len(transformed.split()) * 2
            if transformed_tokens > token_budget:
                # Force truncation if over budget
                transformed = await self._apply_transform(
                    transformed, "truncate"
                )
                transformed_tokens = len(transformed.split()) * 2

            sections.append(f"## {rule.target_key}\n{transformed}")
            token_budget -= transformed_tokens

        return "\n\n".join(sections)


# --- Usage Example ---

async def orchestrated_feature_development(llm_client, feature_request: str):
    """Multi-agent pipeline with orchestrated context flow."""

    contracts = [
        ContextContract(
            agent_role=AgentRole.PLANNER,
            required_inputs=("feature_request",),
            outputs=("plan", "task_list"),
            max_input_tokens=30_000,
        ),
        ContextContract(
            agent_role=AgentRole.CODER,
            required_inputs=("plan_summary", "task_list"),
            optional_inputs=("relevant_code",),
            outputs=("code_diff", "test_results"),
            max_input_tokens=80_000,
        ),
        ContextContract(
            agent_role=AgentRole.REVIEWER,
            required_inputs=("code_diff",),
            optional_inputs=("plan_summary", "test_results_summary"),
            outputs=("review_comments", "approval"),
            max_input_tokens=40_000,
        ),
    ]

    transform_rules = [
        # Planner -> Coder: summarize the plan (not full reasoning)
        TransformRule(AgentRole.PLANNER, AgentRole.CODER, "plan", "plan_summary", "summarize"),
        TransformRule(AgentRole.PLANNER, AgentRole.CODER, "task_list", "task_list", "identity"),
        # Coder -> Reviewer: pass diff as-is, summarize test results
        TransformRule(AgentRole.CODER, AgentRole.REVIEWER, "code_diff", "code_diff", "identity"),
        TransformRule(AgentRole.CODER, AgentRole.REVIEWER, "test_results", "test_results_summary", "summarize"),
        # Reviewer -> Coder: pass review comments for iteration
        TransformRule(AgentRole.REVIEWER, AgentRole.CODER, "review_comments", "review_feedback", "identity"),
    ]

    orchestrator = MultiAgentContextOrchestrator(contracts, transform_rules, llm_client)

    # Shared facts available to all agents
    orchestrator.publish_to_shared_pool(ContextMessage(
        key="project_stack", content="Python 3.12, FastAPI, PostgreSQL, Redis",
        source_agent=AgentRole.PLANNER, token_count=20,
    ))
    orchestrator.publish_to_shared_pool(ContextMessage(
        key="feature_request", content=feature_request,
        source_agent=AgentRole.PLANNER, token_count=len(feature_request.split()) * 2,
    ))

    # Step 1: Planner
    planner_context = await orchestrator.build_agent_context(AgentRole.PLANNER)
    plan = await llm_client.complete(system_prompt=planner_context, messages=[])
    orchestrator.record_agent_output(AgentRole.PLANNER, "plan", plan)
    orchestrator.record_agent_output(AgentRole.PLANNER, "task_list", extract_task_list(plan))

    # Step 2: Coder (receives summarized plan, not full reasoning)
    coder_context = await orchestrator.build_agent_context(AgentRole.CODER)
    code = await llm_client.complete(system_prompt=coder_context, messages=[])
    orchestrator.record_agent_output(AgentRole.CODER, "code_diff", code)
    orchestrator.record_agent_output(AgentRole.CODER, "test_results", run_tests())

    # Step 3: Reviewer (receives diff + summarized test results, not full coder context)
    reviewer_context = await orchestrator.build_agent_context(AgentRole.REVIEWER)
    review = await llm_client.complete(system_prompt=reviewer_context, messages=[])
    orchestrator.record_agent_output(AgentRole.REVIEWER, "review_comments", review)
```

### TypeScript

```typescript
type AgentRole = "planner" | "coder" | "reviewer" | "tester";

interface ContextContract {
  readonly agentRole: AgentRole;
  readonly requiredInputs: readonly string[];
  readonly optionalInputs: readonly string[];
  readonly outputs: readonly string[];
  readonly maxInputTokens: number;
  readonly receivesSharedPool: boolean;
}

interface ContextMessage {
  readonly key: string;
  readonly content: string;
  readonly sourceAgent: AgentRole;
  readonly tokenCount: number;
  readonly priority: number;
}

interface TransformRule {
  readonly sourceRole: AgentRole;
  readonly targetRole: AgentRole;
  readonly sourceKey: string;
  readonly targetKey: string;
  readonly transformFn: "identity" | "summarize" | "truncate";
}

interface LLMClient {
  complete(params: {
    systemPrompt: string;
    messages: readonly { role: string; content: string }[];
  }): Promise<string>;
}

// Immutable shared pool
interface SharedContextPool {
  readonly entries: readonly ContextMessage[];
}

function createSharedPool(): SharedContextPool {
  return { entries: [] };
}

function addToPool(
  pool: SharedContextPool,
  message: ContextMessage
): SharedContextPool {
  return { entries: [...pool.entries, message] };
}

function poolToPromptSection(pool: SharedContextPool): string {
  if (pool.entries.length === 0) return "";
  const lines = ["## Shared Context (established facts, do not re-derive)"];
  for (const entry of pool.entries) {
    lines.push(`### ${entry.key}\n${entry.content}`);
  }
  return lines.join("\n\n");
}

function poolTotalTokens(pool: SharedContextPool): number {
  return pool.entries.reduce((sum, e) => sum + e.tokenCount, 0);
}

function createMessage(
  key: string,
  content: string,
  sourceAgent: AgentRole,
  priority: number = 1
): ContextMessage {
  return {
    key,
    content,
    sourceAgent,
    tokenCount: Math.ceil(content.split(/\s+/).length * 1.5),
    priority,
  };
}

async function applyTransform(
  content: string,
  transformFn: TransformRule["transformFn"],
  client?: LLMClient
): Promise<string> {
  if (transformFn === "identity") return content;

  if (transformFn === "truncate") {
    const lines = content.split("\n");
    if (lines.length > 50) {
      return (
        lines.slice(0, 50).join("\n") +
        `\n... [${lines.length - 50} lines truncated]`
      );
    }
    return content;
  }

  if (transformFn === "summarize") {
    if (!client) throw new Error("LLM client required for summarize transform");
    return client.complete({
      systemPrompt:
        "Summarize concisely, preserving actionable information and specifics " +
        "(file paths, function names, error messages). Remove reasoning chains.",
      messages: [{ role: "user", content }],
    });
  }

  throw new Error(`Unknown transform: ${transformFn}`);
}

interface OrchestratorState {
  readonly sharedPool: SharedContextPool;
  readonly agentOutputs: Readonly<
    Record<string, Readonly<Record<string, ContextMessage>>>
  >;
}

function createOrchestratorState(): OrchestratorState {
  return { sharedPool: createSharedPool(), agentOutputs: {} };
}

function publishToSharedPool(
  state: OrchestratorState,
  message: ContextMessage
): OrchestratorState {
  return {
    ...state,
    sharedPool: addToPool(state.sharedPool, message),
  };
}

function recordAgentOutput(
  state: OrchestratorState,
  agentRole: AgentRole,
  key: string,
  content: string
): OrchestratorState {
  const message = createMessage(key, content, agentRole);
  const existingOutputs = state.agentOutputs[agentRole] ?? {};
  return {
    ...state,
    agentOutputs: {
      ...state.agentOutputs,
      [agentRole]: { ...existingOutputs, [key]: message },
    },
  };
}

async function buildAgentContext(
  state: OrchestratorState,
  targetRole: AgentRole,
  contract: ContextContract,
  transformRules: readonly TransformRule[],
  client?: LLMClient
): Promise<string> {
  const sections: string[] = [];
  let tokenBudget = contract.maxInputTokens;

  // Add shared pool
  if (contract.receivesSharedPool) {
    const poolSection = poolToPromptSection(state.sharedPool);
    if (poolSection) {
      sections.push(poolSection);
      tokenBudget -= poolTotalTokens(state.sharedPool);
    }
  }

  // Route context via transform rules
  for (const rule of transformRules) {
    if (rule.targetRole !== targetRole) continue;

    const sourceOutputs = state.agentOutputs[rule.sourceRole] ?? {};
    const sourceMsg = sourceOutputs[rule.sourceKey];

    if (!sourceMsg) {
      if (contract.requiredInputs.includes(rule.sourceKey)) {
        throw new Error(
          `Required input '${rule.sourceKey}' from ${rule.sourceRole} ` +
          `not available for ${targetRole}`
        );
      }
      continue;
    }

    let transformed = await applyTransform(
      sourceMsg.content,
      rule.transformFn,
      client
    );
    let transformedTokens = Math.ceil(transformed.split(/\s+/).length * 1.5);

    if (transformedTokens > tokenBudget) {
      transformed = await applyTransform(transformed, "truncate");
      transformedTokens = Math.ceil(transformed.split(/\s+/).length * 1.5);
    }

    sections.push(`## ${rule.targetKey}\n${transformed}`);
    tokenBudget -= transformedTokens;
  }

  return sections.join("\n\n");
}

// --- Usage Example ---

async function orchestratedPipeline(
  client: LLMClient,
  featureRequest: string
): Promise<void> {
  const contracts: ContextContract[] = [
    {
      agentRole: "planner",
      requiredInputs: ["feature_request"],
      optionalInputs: [],
      outputs: ["plan", "task_list"],
      maxInputTokens: 30_000,
      receivesSharedPool: true,
    },
    {
      agentRole: "coder",
      requiredInputs: ["plan_summary", "task_list"],
      optionalInputs: ["relevant_code"],
      outputs: ["code_diff", "test_results"],
      maxInputTokens: 80_000,
      receivesSharedPool: true,
    },
    {
      agentRole: "reviewer",
      requiredInputs: ["code_diff"],
      optionalInputs: ["plan_summary", "test_results_summary"],
      outputs: ["review_comments", "approval"],
      maxInputTokens: 40_000,
      receivesSharedPool: true,
    },
  ];

  const rules: TransformRule[] = [
    { sourceRole: "planner", targetRole: "coder", sourceKey: "plan", targetKey: "plan_summary", transformFn: "summarize" },
    { sourceRole: "planner", targetRole: "coder", sourceKey: "task_list", targetKey: "task_list", transformFn: "identity" },
    { sourceRole: "coder", targetRole: "reviewer", sourceKey: "code_diff", targetKey: "code_diff", transformFn: "identity" },
    { sourceRole: "coder", targetRole: "reviewer", sourceKey: "test_results", targetKey: "test_results_summary", transformFn: "summarize" },
    { sourceRole: "reviewer", targetRole: "coder", sourceKey: "review_comments", targetKey: "review_feedback", transformFn: "identity" },
  ];

  let state = createOrchestratorState();

  // Publish shared facts
  state = publishToSharedPool(state, createMessage("project_stack", "Python 3.12, FastAPI, PostgreSQL, Redis", "planner"));
  state = publishToSharedPool(state, createMessage("feature_request", featureRequest, "planner"));

  // Planner phase
  const plannerContract = contracts.find((c) => c.agentRole === "planner")!;
  const plannerCtx = await buildAgentContext(state, "planner", plannerContract, rules, client);
  const plan = await client.complete({ systemPrompt: plannerCtx, messages: [] });
  state = recordAgentOutput(state, "planner", "plan", plan);
  state = recordAgentOutput(state, "planner", "task_list", extractTaskList(plan));

  // Coder phase (receives summarized plan, not full reasoning)
  const coderContract = contracts.find((c) => c.agentRole === "coder")!;
  const coderCtx = await buildAgentContext(state, "coder", coderContract, rules, client);
  const code = await client.complete({ systemPrompt: coderCtx, messages: [] });
  state = recordAgentOutput(state, "coder", "code_diff", code);

  // Reviewer phase (receives diff + summarized test results)
  const reviewerContract = contracts.find((c) => c.agentRole === "reviewer")!;
  const reviewerCtx = await buildAgentContext(state, "reviewer", reviewerContract, rules, client);
  const review = await client.complete({ systemPrompt: reviewerCtx, messages: [] });
  console.log("Review:", review);
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| Explicit contracts make context flow debuggable and auditable | Significant upfront design effort for contracts and routing rules |
| Each agent operates within a controlled token budget | Transforms (especially LLM summarization) add latency and cost |
| Shared pool prevents agents from re-deriving established facts | Rigid contracts can be brittle when task requirements change |
| Scales to many agents without O(n^2) context explosion | Over-filtering can deprive agents of context they actually need |
| Natural boundary for different permission/tool sets per agent | Requires infrastructure for routing, transformation, and state management |

## When to Use

- Multi-agent systems with 3+ specialized agents collaborating on a task.
- Pipelines where one agent's output feeds the next (planner -> coder -> reviewer).
- When debugging multi-agent failures and you need to trace what context each agent received.
- When different agents have different token budgets or cost constraints.
- When you need to enforce information boundaries (e.g., a security reviewer should not see API keys in raw code).

## When NOT to Use

- Simple two-agent parent-child relationships -- use Sub-Agent Delegation instead.
- When agents are fully independent and do not need to share context.
- Prototyping phase where the overhead of defining contracts slows iteration.
- When all agents need the same complete context (contracts add complexity without benefit).

## Related Patterns

- **[Sub-Agent Delegation](sub-agent-delegation.md)** -- Simpler parent-child isolation for when you do not need peer-to-peer orchestration. Use delegation for 1-to-many; use orchestration for many-to-many.
- **[Conversation Compaction](../compression/conversation-compaction.md)** -- Use within individual agents to manage their own context window, complementary to the inter-agent context management this pattern provides.
- **[Observation Masking](../compression/observation-masking.md)** -- Can be applied within the orchestrator's transform rules as a cheap alternative to LLM summarization.

## Real-World Examples

1. **CrewAI's agent communication** -- CrewAI defines agents with specific roles, goals, and backstories. The framework manages what context flows between agents during task execution, including a shared memory that agents can read from.

2. **AutoGen's group chat** -- Microsoft's AutoGen framework implements a group chat pattern where a "manager" agent routes messages between specialist agents, deciding who speaks next and what context they receive.

3. **Manus's multi-agent pipeline** -- Manus uses a pipeline of agents where each stage transforms and filters context before passing it to the next. The planner's full deliberation is summarized before reaching the executor.

4. **LangGraph's state machines** -- LangGraph models multi-agent systems as state machines where edges define context transformations. Each node (agent) reads from and writes to a typed state object, enforcing implicit context contracts.

5. **Enterprise AI pipelines** -- Companies building AI-powered code review, document processing, or customer support systems use orchestration to route context between specialized models (e.g., a classification model routes to a domain-specific generation model, which routes to a quality-checking model).
