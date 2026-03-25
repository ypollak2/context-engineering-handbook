# Progressive Disclosure

> Start with minimal context and reveal more as the task demands it, preserving tokens and model focus.

## Problem

Front-loading the entire context window with everything the model *might* need is wasteful and counterproductive. A model given 50,000 tokens of context to answer a question that requires 2,000 tokens of relevant information will perform worse -- the signal is diluted by noise, attention is spread thin, and you burn tokens (and money) on every request. Worse, in multi-turn conversations, the accumulated context grows monotonically unless actively managed, eventually hitting the window limit and forcing lossy truncation.

## Solution

Structure context delivery in stages. Start with the minimum viable context -- enough for the model to understand its role and begin reasoning. As the conversation or task evolves, add context that is specifically relevant to the current step. This mirrors how humans work: you do not read every file in a codebase before fixing a bug; you start with the stack trace and pull in files as needed.

The key insight is that context has a *relevance lifecycle*. Instructions about output formatting are always relevant. But the contents of `utils/parser.ts` are only relevant when the model is reasoning about parsing. Progressive disclosure makes this lifecycle explicit by defining stages or triggers that control when context enters the window.

## How It Works

```
Stage 0: Baseline (always present)
+------------------------------------------+
| Role + constraints + output format       |
| (minimal, ~200 tokens)                   |
+------------------------------------------+

Stage 1: Task-scoped (added when task is known)
+------------------------------------------+
| Baseline context                         |
|------------------------------------------|
| Task description + relevant schema       |
| (~500 tokens)                            |
+------------------------------------------+

Stage 2: Deep context (added on demand)
+------------------------------------------+
| Baseline context                         |
|------------------------------------------|
| Task context                             |
|------------------------------------------|
| File contents, API docs, examples        |
| (variable, pulled as needed)             |
+------------------------------------------+

Stage 3: Resolution (context can be pruned)
+------------------------------------------+
| Baseline context                         |
|------------------------------------------|
| Summary of prior stages                  |
| Current step context only                |
+------------------------------------------+

Token budget over time:
Tokens
  ^
  |          ___________
  |         /           \         <-- Deep context peak
  |    ____/             \____
  |   /                       \__ <-- Pruned/summarized
  |__/
  +------------------------------> Conversation turns
```

## Implementation

### Python

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


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
    ttl_turns: int | None = None  # Auto-expire after N turns, None = permanent
    trigger: Callable[["ConversationState"], bool] | None = None


@dataclass
class ConversationState:
    """Tracks conversation progress to drive disclosure decisions."""
    current_turn: int = 0
    task_type: str | None = None
    files_referenced: set[str] = field(default_factory=set)
    tools_used: set[str] = field(default_factory=set)
    user_messages: list[str] = field(default_factory=list)


class ProgressiveContext:
    """Manages staged context disclosure across a conversation.

    Context blocks are added with stage annotations and optional triggers.
    At each turn, the manager assembles only the blocks that are currently
    relevant, respecting token budgets and TTL expiration.
    """

    def __init__(self, max_tokens: int = 16000) -> None:
        self._blocks: list[ContextBlock] = []
        self._active_blocks: list[tuple[ContextBlock, int]] = []  # (block, added_at_turn)
        self._max_tokens = max_tokens
        self._state = ConversationState()

    def register(self, block: ContextBlock) -> None:
        """Register a context block for potential disclosure."""
        self._blocks.append(block)

    def advance_turn(self, user_message: str) -> None:
        """Process a new user turn, updating state."""
        self._state.current_turn += 1
        self._state.user_messages.append(user_message)
        self._evaluate_triggers()
        self._expire_stale_blocks()

    def set_task(self, task_type: str) -> None:
        """Transition from BASELINE to TASK_SCOPED stage."""
        self._state.task_type = task_type
        self._promote_stage(DisclosureStage.TASK_SCOPED)

    def add_deep_context(self, name: str, content: str, token_estimate: int) -> None:
        """Inject deep context on demand (e.g., file contents from a tool call)."""
        block = ContextBlock(
            name=name,
            content=content,
            stage=DisclosureStage.DEEP_CONTEXT,
            token_estimate=token_estimate,
            ttl_turns=5,  # Deep context expires after 5 turns by default
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

    def build_context(self) -> str:
        """Assemble the current context string within token budget."""
        # Always include baseline blocks
        baseline = [b for b in self._blocks if b.stage == DisclosureStage.BASELINE]

        # Combine with active blocks
        all_blocks = [(b, 0) for b in baseline] + self._active_blocks
        all_blocks.sort(key=lambda pair: pair[0].stage.value)

        # Trim to budget
        result_parts: list[str] = []
        token_total = 0
        for block, _ in all_blocks:
            if token_total + block.token_estimate > self._max_tokens:
                continue
            result_parts.append(f"<!-- {block.name} -->\n{block.content}")
            token_total += block.token_estimate

        return "\n\n".join(result_parts)

    @property
    def token_usage(self) -> int:
        baseline_tokens = sum(
            b.token_estimate for b in self._blocks
            if b.stage == DisclosureStage.BASELINE
        )
        active_tokens = sum(b.token_estimate for b, _ in self._active_blocks)
        return baseline_tokens + active_tokens

    def _promote_stage(self, stage: DisclosureStage) -> None:
        """Activate all registered blocks for the given stage."""
        for block in self._blocks:
            if block.stage == stage:
                already_active = any(b.name == block.name for b, _ in self._active_blocks)
                if not already_active:
                    self._active_blocks.append((block, self._state.current_turn))

    def _evaluate_triggers(self) -> None:
        """Check trigger predicates on registered blocks."""
        for block in self._blocks:
            if block.trigger and block.trigger(self._state):
                already_active = any(b.name == block.name for b, _ in self._active_blocks)
                if not already_active:
                    self._active_blocks.append((block, self._state.current_turn))

    def _expire_stale_blocks(self) -> None:
        """Remove blocks that have exceeded their TTL."""
        self._active_blocks = [
            (block, added_turn)
            for block, added_turn in self._active_blocks
            if block.ttl_turns is None
            or (self._state.current_turn - added_turn) < block.ttl_turns
        ]


# --- Usage ---

ctx = ProgressiveContext(max_tokens=8000)

# Stage 0: Always present
ctx.register(ContextBlock(
    name="role",
    content="You are a coding assistant. Be concise and precise.",
    stage=DisclosureStage.BASELINE,
    token_estimate=20,
))

ctx.register(ContextBlock(
    name="output_format",
    content="Respond in markdown. Use code fences with language tags.",
    stage=DisclosureStage.BASELINE,
    token_estimate=15,
))

# Stage 1: Activated when task is classified as "code_review"
ctx.register(ContextBlock(
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
))

# Trigger-based: activate when user mentions "database"
ctx.register(ContextBlock(
    name="db_schema",
    content="Schema: users(id, email, name, created_at), orders(id, user_id, total, status)",
    stage=DisclosureStage.TASK_SCOPED,
    token_estimate=30,
    trigger=lambda state: any("database" in msg.lower() for msg in state.user_messages),
))

# Simulate a conversation
print(f"Turn 0 tokens: {ctx.token_usage}")  # Baseline only

ctx.advance_turn("Please review this pull request")
ctx.set_task("code_review")
print(f"Turn 1 tokens: {ctx.token_usage}")  # + review guidelines

ctx.advance_turn("Here's the diff for the database migration")
# Trigger fires because "database" appeared
print(f"Turn 2 tokens: {ctx.token_usage}")  # + db schema

# Deep context injected by a tool call
ctx.add_deep_context(
    name="migration_file",
    content="ALTER TABLE users ADD COLUMN last_login TIMESTAMP;",
    token_estimate=20,
)
print(f"Turn 2 tokens (with file): {ctx.token_usage}")

# Later: prune and summarize
ctx.summarize_and_prune("Previously reviewed a DB migration adding last_login to users table.")
print(f"After prune tokens: {ctx.token_usage}")

print("\n--- Current Context ---")
print(ctx.build_context())
```

### TypeScript

```typescript
enum DisclosureStage {
  BASELINE = 0,
  TASK_SCOPED = 1,
  DEEP_CONTEXT = 2,
  RESOLUTION = 3,
}

interface ContextBlock {
  readonly name: string;
  readonly content: string;
  readonly stage: DisclosureStage;
  readonly tokenEstimate: number;
  readonly ttlTurns?: number;
  readonly trigger?: (state: ConversationState) => boolean;
}

interface ConversationState {
  currentTurn: number;
  taskType: string | null;
  filesReferenced: Set<string>;
  toolsUsed: Set<string>;
  userMessages: string[];
}

function createConversationState(): ConversationState {
  return {
    currentTurn: 0,
    taskType: null,
    filesReferenced: new Set(),
    toolsUsed: new Set(),
    userMessages: [],
  };
}

class ProgressiveContext {
  private readonly registered: ContextBlock[] = [];
  private activeBlocks: Array<{ block: ContextBlock; addedAtTurn: number }> =
    [];
  private state: ConversationState = createConversationState();

  constructor(private readonly maxTokens: number = 16000) {}

  register(block: ContextBlock): void {
    this.registered.push(block);
  }

  advanceTurn(userMessage: string): void {
    this.state.currentTurn++;
    this.state.userMessages.push(userMessage);
    this.evaluateTriggers();
    this.expireStaleBlocks();
  }

  setTask(taskType: string): void {
    this.state.taskType = taskType;
    this.promoteStage(DisclosureStage.TASK_SCOPED);
  }

  addDeepContext(
    name: string,
    content: string,
    tokenEstimate: number
  ): void {
    const block: ContextBlock = {
      name,
      content,
      stage: DisclosureStage.DEEP_CONTEXT,
      tokenEstimate,
      ttlTurns: 5,
    };
    this.activeBlocks.push({
      block,
      addedAtTurn: this.state.currentTurn,
    });
  }

  summarizeAndPrune(summary: string): void {
    this.activeBlocks = this.activeBlocks.filter(
      ({ block }) => block.stage !== DisclosureStage.DEEP_CONTEXT
    );

    const summaryBlock: ContextBlock = {
      name: "context_summary",
      content: summary,
      stage: DisclosureStage.RESOLUTION,
      tokenEstimate: Math.ceil(summary.length / 4),
    };
    this.activeBlocks.push({
      block: summaryBlock,
      addedAtTurn: this.state.currentTurn,
    });
  }

  buildContext(): string {
    const baseline = this.registered.filter(
      (b) => b.stage === DisclosureStage.BASELINE
    );

    const allBlocks: Array<{ block: ContextBlock; addedAtTurn: number }> = [
      ...baseline.map((block) => ({ block, addedAtTurn: 0 })),
      ...this.activeBlocks,
    ];

    allBlocks.sort((a, b) => a.block.stage - b.block.stage);

    const parts: string[] = [];
    let tokenTotal = 0;

    for (const { block } of allBlocks) {
      if (tokenTotal + block.tokenEstimate > this.maxTokens) continue;
      parts.push(`<!-- ${block.name} -->\n${block.content}`);
      tokenTotal += block.tokenEstimate;
    }

    return parts.join("\n\n");
  }

  get tokenUsage(): number {
    const baselineTokens = this.registered
      .filter((b) => b.stage === DisclosureStage.BASELINE)
      .reduce((sum, b) => sum + b.tokenEstimate, 0);

    const activeTokens = this.activeBlocks.reduce(
      (sum, { block }) => sum + block.tokenEstimate,
      0
    );

    return baselineTokens + activeTokens;
  }

  private promoteStage(stage: DisclosureStage): void {
    for (const block of this.registered) {
      if (block.stage !== stage) continue;
      const alreadyActive = this.activeBlocks.some(
        ({ block: b }) => b.name === block.name
      );
      if (!alreadyActive) {
        this.activeBlocks.push({
          block,
          addedAtTurn: this.state.currentTurn,
        });
      }
    }
  }

  private evaluateTriggers(): void {
    for (const block of this.registered) {
      if (!block.trigger || !block.trigger(this.state)) continue;
      const alreadyActive = this.activeBlocks.some(
        ({ block: b }) => b.name === block.name
      );
      if (!alreadyActive) {
        this.activeBlocks.push({
          block,
          addedAtTurn: this.state.currentTurn,
        });
      }
    }
  }

  private expireStaleBlocks(): void {
    this.activeBlocks = this.activeBlocks.filter(({ block, addedAtTurn }) => {
      if (block.ttlTurns === undefined) return true;
      return this.state.currentTurn - addedAtTurn < block.ttlTurns;
    });
  }
}

// --- Usage ---

const ctx = new ProgressiveContext(8000);

ctx.register({
  name: "role",
  content: "You are a coding assistant. Be concise and precise.",
  stage: DisclosureStage.BASELINE,
  tokenEstimate: 20,
});

ctx.register({
  name: "review_guidelines",
  content: [
    "When reviewing code:",
    "1. Check for correctness first",
    "2. Then readability",
    "3. Then performance",
    "Flag security issues immediately.",
  ].join("\n"),
  stage: DisclosureStage.TASK_SCOPED,
  tokenEstimate: 40,
});

ctx.register({
  name: "db_schema",
  content:
    "Schema: users(id, email, name, created_at), orders(id, user_id, total, status)",
  stage: DisclosureStage.TASK_SCOPED,
  tokenEstimate: 30,
  trigger: (state) =>
    state.userMessages.some((msg) => msg.toLowerCase().includes("database")),
});

// Simulate conversation
console.log(`Turn 0 tokens: ${ctx.tokenUsage}`);

ctx.advanceTurn("Please review this pull request");
ctx.setTask("code_review");
console.log(`Turn 1 tokens: ${ctx.tokenUsage}`);

ctx.advanceTurn("Here's the diff for the database migration");
console.log(`Turn 2 tokens: ${ctx.tokenUsage}`);

ctx.addDeepContext(
  "migration_file",
  "ALTER TABLE users ADD COLUMN last_login TIMESTAMP;",
  20
);
console.log(`Turn 2 tokens (with file): ${ctx.tokenUsage}`);

ctx.summarizeAndPrune(
  "Previously reviewed a DB migration adding last_login to users table."
);
console.log(`After prune tokens: ${ctx.tokenUsage}`);

console.log("\n--- Current Context ---");
console.log(ctx.buildContext());
```

## Trade-offs

| Pros | Cons |
|------|------|
| Dramatically reduces token usage in multi-turn conversations | Adds state management complexity |
| Keeps model attention focused on currently relevant information | Risk of withholding context the model actually needs |
| Graceful degradation as context grows (summarize instead of truncate) | TTL-based expiration is a heuristic -- may expire too early or late |
| Natural fit for tool-using agents that pull context on demand | Trigger predicates need careful design to avoid false negatives |
| Enables longer effective conversations within fixed context windows | Harder to debug prompt behavior when context changes per turn |

## When to Use

- Multi-turn agent conversations where context accumulates over many steps
- Tool-using systems where the model retrieves information dynamically (RAG, code assistants)
- Applications with strict token budgets or cost sensitivity
- Long-running tasks (code review of a large PR, multi-file refactoring) where different phases need different context
- Systems serving users with varying levels of complexity in their requests

## When NOT to Use

- Single-turn request/response APIs where all context is known upfront
- Tasks where the entire context easily fits within the window with room to spare
- Latency-critical applications where the overhead of context management adds unacceptable delay
- Situations where you cannot reliably predict what context will be needed (better to include everything)

## Related Patterns

- [System Prompt Architecture](system-prompt-architecture.md) -- defines the sections that Progressive Disclosure selectively includes
- [Few-Shot Curation](few-shot-curation.md) -- another form of selective context inclusion, focused on examples

## Real-World Examples

- **Claude Code** is the canonical example: it starts with the contents of `CLAUDE.md` and project structure, then reads specific files only when the task requires them. File contents are injected as deep context and effectively expire as the conversation moves on to different files.
- **Cursor and GitHub Copilot** progressively expand context based on what the user is currently editing. The active file gets full context; related files get partial context; distant files are excluded entirely until referenced.
- **ChatGPT with browsing** starts with the user's question, decides whether to search, then injects search results as deep context. The search results are not present in the initial system prompt -- they are disclosed when the model determines they are needed.
- **Devin (Cognition)** manages an explicit context window for its coding agent, summarizing completed steps and only keeping active file contents in the window as it works through multi-step plans.
