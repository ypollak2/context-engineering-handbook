# Observation Masking

> Selectively hide or truncate tool outputs and observations that have been consumed, achieving compression comparable to LLM summarization at a fraction of the cost.

## Problem

Agent systems generate enormous amounts of intermediate output: file contents read from disk, API responses, command-line output, search results, database query results. These observations are essential when they first appear -- the model needs them to make decisions -- but they become dead weight once the model has acted on them.

Without observation masking:
- Tool outputs accumulate and dominate the context window (often 60-80% of total tokens).
- The model wastes attention on stale observations that have already informed decisions.
- Context window exhaustion forces premature session termination.
- LLM-powered summarization of tool outputs is expensive and slow, adding latency to every agent loop iteration.

JetBrains' research (NeurIPS 2025) demonstrated that observation masking -- a deterministic, non-LLM technique -- achieves comparable task completion rates to full LLM summarization while cutting context management costs by approximately 50%.

## Solution

Track each tool output (observation) in the conversation. Score its continued relevance based on recency, whether it has been "consumed" (acted upon), and explicit relevance signals. Apply masking strategies -- truncation, hash replacement, or full removal -- to observations that score below a relevance threshold. No LLM call is required.

## How It Works

```
Before masking (agent at step 5):
+-------------------------------------------------------------+
| Step 1: read_file("auth.py") -> [842 lines of Python]       |  <- consumed
| Step 2: search("JWT library") -> [10 search results]        |  <- consumed
| Step 3: read_file("requirements.txt") -> [45 lines]         |  <- consumed
| Step 4: run_command("pip install") -> [23 lines of output]   |  <- consumed
| Step 5: edit_file("auth.py") -> [diff output]                |  <- ACTIVE
+-------------------------------------------------------------+
Total: ~12,000 tokens

After masking:
+-------------------------------------------------------------+
| Step 1: read_file("auth.py") -> [MASKED: 842 lines, sha256: a3f2...] |
| Step 2: search("JWT library") -> [MASKED: 10 results]                |
| Step 3: read_file("requirements.txt") -> [TRUNCATED: first 5 lines]  |
| Step 4: run_command("pip install") -> [MASKED: exit code 0]          |
| Step 5: edit_file("auth.py") -> [diff output]                        |
+-------------------------------------------------------------+
Total: ~800 tokens (93% reduction)
```

## Implementation

### Python

```python
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from time import time


class MaskingStrategy(str, Enum):
    NONE = "none"           # Keep full content
    TRUNCATE = "truncate"   # Keep first N lines
    HASH = "hash"           # Replace with content hash + metadata
    REMOVE = "remove"       # Remove content entirely, keep tool call record


@dataclass(frozen=True)
class Observation:
    tool_name: str
    content: str
    token_count: int
    step_index: int
    timestamp: float = field(default_factory=time)
    consumed: bool = False

    def mark_consumed(self) -> "Observation":
        return replace(self, consumed=True)


@dataclass(frozen=True)
class MaskingConfig:
    max_observation_tokens: int = 500     # Max tokens per masked observation
    truncate_lines: int = 5               # Lines to keep in truncate mode
    recency_window: int = 3               # Steps within this window are never masked
    relevance_decay: float = 0.3          # Relevance drops by this per step of distance
    consumed_penalty: float = 0.5         # Relevance penalty for consumed observations
    relevance_threshold: float = 0.3      # Below this, apply masking

    # Tools whose output should always be preserved (e.g., errors, diffs)
    preserve_tools: tuple[str, ...] = ("edit_file", "apply_diff", "error")

    # Tools whose output can be aggressively masked
    aggressive_mask_tools: tuple[str, ...] = ("read_file", "search", "run_command", "list_directory")


@dataclass(frozen=True)
class MaskedObservation:
    tool_name: str
    original_token_count: int
    masked_content: str
    masked_token_count: int
    strategy_applied: MaskingStrategy

    @property
    def tokens_saved(self) -> int:
        return self.original_token_count - self.masked_token_count


def score_relevance(
    observation: Observation,
    current_step: int,
    config: MaskingConfig,
) -> float:
    """Score an observation's continued relevance (0.0 to 1.0)."""
    # Observations within the recency window are always relevant
    distance = current_step - observation.step_index
    if distance <= config.recency_window:
        return 1.0

    # Preserved tools are always relevant
    if observation.tool_name in config.preserve_tools:
        return 1.0

    # Base relevance decays with distance
    relevance = max(0.0, 1.0 - (distance - config.recency_window) * config.relevance_decay)

    # Consumed observations get a penalty
    if observation.consumed:
        relevance = max(0.0, relevance - config.consumed_penalty)

    return relevance


def choose_strategy(
    observation: Observation,
    relevance: float,
    config: MaskingConfig,
) -> MaskingStrategy:
    """Choose the masking strategy based on relevance score and tool type."""
    if relevance >= config.relevance_threshold:
        return MaskingStrategy.NONE

    if observation.tool_name in config.aggressive_mask_tools:
        if relevance < 0.1:
            return MaskingStrategy.HASH
        return MaskingStrategy.TRUNCATE

    # Default: truncate for moderate irrelevance, hash for low relevance
    if relevance < 0.1:
        return MaskingStrategy.HASH
    return MaskingStrategy.TRUNCATE


def apply_masking(
    observation: Observation,
    strategy: MaskingStrategy,
    config: MaskingConfig,
) -> MaskedObservation:
    """Apply the chosen masking strategy to an observation."""
    if strategy == MaskingStrategy.NONE:
        return MaskedObservation(
            tool_name=observation.tool_name,
            original_token_count=observation.token_count,
            masked_content=observation.content,
            masked_token_count=observation.token_count,
            strategy_applied=strategy,
        )

    if strategy == MaskingStrategy.TRUNCATE:
        lines = observation.content.split("\n")
        truncated = "\n".join(lines[: config.truncate_lines])
        suffix = f"\n... [{len(lines) - config.truncate_lines} more lines truncated]"
        masked_content = truncated + suffix
        return MaskedObservation(
            tool_name=observation.tool_name,
            original_token_count=observation.token_count,
            masked_content=masked_content,
            masked_token_count=len(masked_content.split()) * 2,  # Rough estimate
            strategy_applied=strategy,
        )

    if strategy == MaskingStrategy.HASH:
        content_hash = sha256(observation.content.encode()).hexdigest()[:12]
        line_count = observation.content.count("\n") + 1
        masked_content = (
            f"[MASKED: {observation.tool_name} output, "
            f"{line_count} lines, sha256: {content_hash}]"
        )
        return MaskedObservation(
            tool_name=observation.tool_name,
            original_token_count=observation.token_count,
            masked_content=masked_content,
            masked_token_count=20,  # Fixed small size
            strategy_applied=strategy,
        )

    # MaskingStrategy.REMOVE
    masked_content = f"[REMOVED: {observation.tool_name} output]"
    return MaskedObservation(
        tool_name=observation.tool_name,
        original_token_count=observation.token_count,
        masked_content=masked_content,
        masked_token_count=8,
        strategy_applied=strategy,
    )


def mask_observations(
    observations: list[Observation],
    current_step: int,
    config: MaskingConfig = MaskingConfig(),
) -> list[MaskedObservation]:
    """Apply masking to a list of observations.

    Returns a new list; does not mutate the input.
    """
    results = []
    for obs in observations:
        relevance = score_relevance(obs, current_step, config)
        strategy = choose_strategy(obs, relevance, config)
        masked = apply_masking(obs, strategy, config)
        results.append(masked)
    return results


# --- Usage Example ---

def example():
    config = MaskingConfig(
        recency_window=2,
        relevance_decay=0.3,
        consumed_penalty=0.5,
    )

    observations = [
        Observation(tool_name="read_file", content="def auth():\n" * 100, token_count=800, step_index=1, consumed=True),
        Observation(tool_name="search", content="Result 1\nResult 2\n" * 10, token_count=400, step_index=2, consumed=True),
        Observation(tool_name="run_command", content="Installing...\nSuccess", token_count=200, step_index=3, consumed=True),
        Observation(tool_name="edit_file", content="- old line\n+ new line", token_count=100, step_index=4),
        Observation(tool_name="run_command", content="All tests passed", token_count=50, step_index=5),
    ]

    masked = mask_observations(observations, current_step=5, config=config)

    total_before = sum(o.token_count for o in observations)
    total_after = sum(m.masked_token_count for m in masked)
    print(f"Before: {total_before} tokens")
    print(f"After:  {total_after} tokens")
    print(f"Saved:  {total_before - total_after} tokens ({(1 - total_after/total_before)*100:.0f}%)")

    for m in masked:
        print(f"  {m.tool_name}: {m.strategy_applied.value} ({m.tokens_saved} tokens saved)")
```

### TypeScript

```typescript
type MaskingStrategy = "none" | "truncate" | "hash" | "remove";

interface Observation {
  readonly toolName: string;
  readonly content: string;
  readonly tokenCount: number;
  readonly stepIndex: number;
  readonly timestamp: number;
  readonly consumed: boolean;
}

interface MaskedObservation {
  readonly toolName: string;
  readonly originalTokenCount: number;
  readonly maskedContent: string;
  readonly maskedTokenCount: number;
  readonly strategyApplied: MaskingStrategy;
  readonly tokensSaved: number;
}

interface MaskingConfig {
  readonly maxObservationTokens: number;
  readonly truncateLines: number;
  readonly recencyWindow: number;
  readonly relevanceDecay: number;
  readonly consumedPenalty: number;
  readonly relevanceThreshold: number;
  readonly preserveTools: readonly string[];
  readonly aggressiveMaskTools: readonly string[];
}

const DEFAULT_MASKING_CONFIG: MaskingConfig = {
  maxObservationTokens: 500,
  truncateLines: 5,
  recencyWindow: 3,
  relevanceDecay: 0.3,
  consumedPenalty: 0.5,
  relevanceThreshold: 0.3,
  preserveTools: ["edit_file", "apply_diff", "error"],
  aggressiveMaskTools: ["read_file", "search", "run_command", "list_directory"],
};

function createObservation(
  toolName: string,
  content: string,
  tokenCount: number,
  stepIndex: number
): Observation {
  return {
    toolName,
    content,
    tokenCount,
    stepIndex,
    timestamp: Date.now(),
    consumed: false,
  };
}

function markConsumed(observation: Observation): Observation {
  return { ...observation, consumed: true };
}

function scoreRelevance(
  observation: Observation,
  currentStep: number,
  config: MaskingConfig
): number {
  const distance = currentStep - observation.stepIndex;

  if (distance <= config.recencyWindow) return 1.0;
  if (config.preserveTools.includes(observation.toolName)) return 1.0;

  let relevance = Math.max(
    0.0,
    1.0 - (distance - config.recencyWindow) * config.relevanceDecay
  );

  if (observation.consumed) {
    relevance = Math.max(0.0, relevance - config.consumedPenalty);
  }

  return relevance;
}

function chooseStrategy(
  observation: Observation,
  relevance: number,
  config: MaskingConfig
): MaskingStrategy {
  if (relevance >= config.relevanceThreshold) return "none";

  if (config.aggressiveMaskTools.includes(observation.toolName)) {
    return relevance < 0.1 ? "hash" : "truncate";
  }

  return relevance < 0.1 ? "hash" : "truncate";
}

async function hashContent(content: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(content);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 12);
}

function applyMaskingSync(
  observation: Observation,
  strategy: MaskingStrategy,
  config: MaskingConfig,
  contentHash?: string
): MaskedObservation {
  if (strategy === "none") {
    return {
      toolName: observation.toolName,
      originalTokenCount: observation.tokenCount,
      maskedContent: observation.content,
      maskedTokenCount: observation.tokenCount,
      strategyApplied: strategy,
      tokensSaved: 0,
    };
  }

  if (strategy === "truncate") {
    const lines = observation.content.split("\n");
    const truncated = lines.slice(0, config.truncateLines).join("\n");
    const remaining = lines.length - config.truncateLines;
    const maskedContent = `${truncated}\n... [${remaining} more lines truncated]`;
    const maskedTokenCount = Math.ceil(maskedContent.split(/\s+/).length * 1.5);
    return {
      toolName: observation.toolName,
      originalTokenCount: observation.tokenCount,
      maskedContent,
      maskedTokenCount,
      strategyApplied: strategy,
      tokensSaved: observation.tokenCount - maskedTokenCount,
    };
  }

  if (strategy === "hash") {
    const lineCount = observation.content.split("\n").length;
    const hash = contentHash ?? "precomputed";
    const maskedContent = `[MASKED: ${observation.toolName} output, ${lineCount} lines, sha256: ${hash}]`;
    return {
      toolName: observation.toolName,
      originalTokenCount: observation.tokenCount,
      maskedContent,
      maskedTokenCount: 20,
      strategyApplied: strategy,
      tokensSaved: observation.tokenCount - 20,
    };
  }

  // "remove"
  const maskedContent = `[REMOVED: ${observation.toolName} output]`;
  return {
    toolName: observation.toolName,
    originalTokenCount: observation.tokenCount,
    maskedContent,
    maskedTokenCount: 8,
    strategyApplied: strategy,
    tokensSaved: observation.tokenCount - 8,
  };
}

async function maskObservations(
  observations: readonly Observation[],
  currentStep: number,
  config: MaskingConfig = DEFAULT_MASKING_CONFIG
): Promise<readonly MaskedObservation[]> {
  const results = await Promise.all(
    observations.map(async (obs) => {
      const relevance = scoreRelevance(obs, currentStep, config);
      const strategy = chooseStrategy(obs, relevance, config);

      let contentHash: string | undefined;
      if (strategy === "hash") {
        contentHash = await hashContent(obs.content);
      }

      return applyMaskingSync(obs, strategy, config, contentHash);
    })
  );

  return results;
}

// --- Usage Example ---

async function example(): Promise<void> {
  const observations: Observation[] = [
    createObservation("read_file", "def auth():\n".repeat(100), 800, 1),
    createObservation("search", "Result 1\nResult 2\n".repeat(10), 400, 2),
    createObservation("run_command", "Installing...\nSuccess", 200, 3),
    createObservation("edit_file", "- old line\n+ new line", 100, 4),
    createObservation("run_command", "All tests passed", 50, 5),
  ].map((obs, i) => (i < 3 ? markConsumed(obs) : obs));

  const masked = await maskObservations(observations, 5);

  const totalBefore = observations.reduce((s, o) => s + o.tokenCount, 0);
  const totalAfter = masked.reduce((s, m) => s + m.maskedTokenCount, 0);

  console.log(`Before: ${totalBefore} tokens`);
  console.log(`After:  ${totalAfter} tokens`);
  console.log(`Saved:  ${totalBefore - totalAfter} tokens (${Math.round((1 - totalAfter / totalBefore) * 100)}%)`);
}
```

## Trade-offs

| Pros | Cons |
|------|------|
| No LLM call required -- purely deterministic | Cannot extract semantic meaning from masked content |
| ~50% cost savings compared to LLM summarization | Aggressive masking can remove content the model later needs |
| Near-zero latency overhead | Requires tuning relevance thresholds per use case |
| Predictable and debuggable behavior | Hash-replaced content cannot be recovered without re-fetching |
| Composable with other compression patterns | "Consumed" tracking requires integration with the agent loop |

## When to Use

- Agent loops with heavy tool usage (file reading, search, command execution).
- Cost-sensitive deployments where LLM summarization overhead is unacceptable.
- Systems where tool outputs are the primary context bloat (not conversation turns).
- When you need a first-pass compression before applying LLM-powered compaction.
- Real-time agents where latency from summarization calls is unacceptable.

## When NOT to Use

- When tool outputs contain ongoing reference material that the model consults repeatedly (e.g., a schema definition used throughout the session).
- When the conversation itself (not tool outputs) is the primary source of context bloat -- use Conversation Compaction instead.
- When you need semantic understanding of what was in the masked content (the model cannot reason about hashed or removed observations).
- Single-turn interactions where context window pressure does not exist.

## Related Patterns

- **[Conversation Compaction](conversation-compaction.md)** -- Complementary pattern that compresses conversation turns using LLM summarization. Use observation masking for tool outputs, conversation compaction for dialogue.
- **[Sub-Agent Delegation](../isolation/sub-agent-delegation.md)** -- If a sub-task generates many observations, delegate it to a child agent so those observations never enter the parent's context at all.
- **Progressive Disclosure** (construction) -- Controls what enters the context initially; observation masking cleans up what has already entered.

## Real-World Examples

1. **JetBrains' agent research (NeurIPS 2025)** -- Their study compared observation masking against LLM-powered summarization across coding benchmarks. Masking achieved comparable task completion rates while eliminating the cost and latency of summarization LLM calls.

2. **Claude Code's tool result clearing** -- After the model has processed a tool's output and made a decision, Claude Code can clear the raw output from context, retaining only the tool call record and the model's interpretation of the result.

3. **LangChain's message trimming** -- The `trim_messages` utility allows developers to cap message history by token count, effectively masking older messages. While simpler than relevance-scored masking, it follows the same principle.

4. **Custom agent frameworks** -- Any agent that reads files, runs commands, or calls APIs in a loop benefits from masking observations that have already been acted upon, especially in resource-constrained environments.
